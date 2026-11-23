"""Benchmark and train SmolVLA on the validated 25 Hz insertion dataset."""

import argparse
import json
import math
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from .smolvla25_data import CAMERAS, CORNER_INDICES, FPS, REPO_ID, SmolVLAInsertion


TOTAL_UPDATES = 15_000
SAVE_EVERY = 5_000
BENCHMARK_UPDATES = 20
CAMERA_MAP = {key: f"observation.images.camera{i}" for i, key in enumerate(CAMERAS, 1)}


class BenchmarkComplete(Exception):
    pass


def evaluate(policy, preprocessor, postprocessor, splits, destination, accelerator):
    import cv2
    import torch

    destination.mkdir(exist_ok=False)
    env = SmolVLAInsertion()
    summary = {}
    try:
        with torch.inference_mode(), accelerator.autocast(), (destination / "episodes.jsonl").open("x", buffering=1) as log:
            groups = {"train": splits["eval_train_indices"], "validation": range(len(splits["validation"]))}
            for group, indices in groups.items():
                results = []
                for index in indices:
                    case = splits[group][index]
                    env.start(case)
                    policy.reset()
                    writer = None
                    try:
                        done = timeout = False
                        while not (done or timeout):
                            obs = env.observation()
                            batch = {key: torch.from_numpy(obs[key]).permute(2, 0, 1).float() / 255 for key in CAMERAS}
                            batch.update(task=obs["task"], **{"observation.state": torch.from_numpy(obs["observation.state"])})
                            actions = policy.predict_action_chunk(preprocessor(batch))[:, :policy.config.n_action_steps]
                            actions = postprocessor(actions.float())[0].detach().cpu().numpy()
                            for action in actions:
                                if group == "train" and index in CORNER_INDICES:
                                    rgb = np.concatenate([obs[key] for key in CAMERAS], axis=1)
                                    if writer is None:
                                        writer = cv2.VideoWriter(str(destination / f"corner-{index:02d}.mp4"),
                                                                 cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                                                 (rgb.shape[1], rgb.shape[0]))
                                        if not writer.isOpened():
                                            raise RuntimeError("Could not open evaluation video writer")
                                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                                _, done, timeout, info = env.step(action)
                                if done or timeout:
                                    break
                                if writer is not None:
                                    obs = env.observation()
                    finally:
                        if writer is not None:
                            writer.release()
                    results.append(info)
                    print(json.dumps(dict(group=group, index=index, steps=env.steps, **info)), file=log)
                keys = ("pickup", "arrival", "success", "opened_early", "slipped")
                summary[group] = dict(episodes=len(results), **{key: sum(row[key] for row in results) for key in keys})
    finally:
        env.close()
    return summary


def run(args):
    import torch
    from lerobot.scripts import lerobot_train as trainer

    root = args.data.resolve()
    data = json.loads((root / "dataset-summary.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    if (data["episodes"], data["fps"], data["warnings"]) != (100, FPS, 0):
        raise RuntimeError("Expected a complete, warning-free 100-episode dataset at 25 Hz")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required for this run")
    if args.batch_size % args.micro_batch_size:
        raise ValueError("Microbatch size must divide batch size")
    accumulation = args.batch_size // args.micro_batch_size
    output = root / (f"benchmark-b{args.batch_size}-m{args.micro_batch_size}" if args.benchmark else "training")
    steps = TOTAL_UPDATES * accumulation
    original_update = trainer.update_policy
    original_save = trainer.save_checkpoint
    original_last = trainer.update_last_checkpoint
    original_make_policy = trainer.make_policy
    updates = 0
    times = []
    last_update = None
    peak_memory = 0.
    losses = deque(maxlen=50 * accumulation)
    current = {}
    previous = None

    def make_policy(*policy_args, **policy_kwargs):
        policy = original_make_policy(*policy_args, **policy_kwargs)
        trainable = [p for p in policy.parameters() if p.requires_grad]
        if not trainable or any(p.dtype != torch.float32 for p in trainable):
            raise RuntimeError("SmolVLA trainable parameters must use FP32 master weights")
        return policy

    def update(*update_args, **update_kwargs):
        nonlocal updates, last_update, peak_memory
        accelerator = update_kwargs["accelerator"]
        if accelerator.mixed_precision != "bf16" or accelerator.num_processes != 1:
            raise RuntimeError("Use one GPU with BF16 mixed precision")
        tracker, metrics = original_update(*update_args, **update_kwargs)
        loss = float(tracker.loss.val)
        if not math.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss after {updates} optimizer updates")
        losses.append(loss)
        peak_memory = max(peak_memory, torch.cuda.max_memory_reserved() / 1024**3)
        if accelerator.sync_gradients:
            if accelerator.optimizer_step_was_skipped:
                raise RuntimeError("Optimizer update was skipped")
            updates += 1
            if args.benchmark:
                torch.cuda.synchronize()
                now = time.perf_counter()
                if last_update is not None:
                    times.append(now - last_update)
                last_update = now
                if updates == BENCHMARK_UPDATES:
                    seconds = float(np.mean(times[4:]))
                    result = dict(updates=updates, train_loss=float(np.mean(losses)),
                                  seconds_per_update=seconds, estimated_train_hours=seconds * TOTAL_UPDATES / 3600,
                                  peak_reserved_gib=peak_memory, batch_size=args.batch_size,
                                  micro_batch_size=args.micro_batch_size, video_backend=args.video_backend)
                    (output / "timing.json").write_text(json.dumps(result, indent=2))
                    print(json.dumps(result), flush=True)
                    raise BenchmarkComplete
        return tracker, metrics

    def save(**kwargs):
        original_save(**kwargs)
        current.update(kwargs)

    def mark_latest(checkpoint_dir):
        nonlocal previous
        original_last(checkpoint_dir)
        accelerator = current["accelerator"]
        policy = accelerator.unwrap_model(current["policy"])
        destination = output / f"eval-{updates:05d}"
        result = evaluate(policy, current["preprocessor"], current["postprocessor"], splits, destination, accelerator)
        summary = dict(step=updates, train_loss=float(np.mean(losses)), **result)
        (destination / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if previous is not None:
            old = previous.resolve(strict=True)
            if old.parent != checkpoint_dir.resolve().parent or old != previous.absolute() or old == checkpoint_dir.resolve():
                raise RuntimeError(f"Refusing to prune checkpoint outside this run: {previous}")
            shutil.rmtree(old)
        previous = checkpoint_dir

    trainer.make_policy = make_policy
    trainer.update_policy = update
    trainer.save_checkpoint = save
    trainer.update_last_checkpoint = mark_latest
    sys.argv = [sys.argv[0], f"--dataset.repo_id={REPO_ID}", f"--dataset.root={root / 'dataset'}",
                f"--dataset.video_backend={args.video_backend}", "--policy.path=lerobot/smolvla_base",
                "--policy.device=cuda", "--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false",
                "--policy.load_vlm_weights=false", "--policy.chunk_size=25", "--policy.n_action_steps=5",
                "--policy.empty_cameras=1", "--policy.use_amp=false", "--accelerator.mixed_precision=bf16",
                "--policy.optimizer_lr=2.5e-5", f"--policy.scheduler_decay_steps={TOTAL_UPDATES}",
                f"--batch_size={args.micro_batch_size}", f"--accelerator.gradient_accumulation.steps={accumulation}",
                f"--steps={steps}", f"--save_freq={SAVE_EVERY * accumulation}", f"--log_freq={50 * accumulation}",
                f"--num_workers={args.workers}", f"--output_dir={output}",
                f"--rename_map={json.dumps(CAMERA_MAP)}", "--ema.enable=false", "--policy.push_to_hub=false",
                "--wandb.enable=false", "--env_eval_freq=0", f"--save_checkpoint={str(not args.benchmark).lower()}"]
    try:
        trainer.main()
    except BenchmarkComplete:
        if not args.benchmark:
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--micro-batch-size", type=int, choices=(4, 8, 16, 32, 64), default=32)
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"), default="torchcodec")
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
