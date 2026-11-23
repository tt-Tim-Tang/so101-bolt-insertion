"""Benchmark and train SmolVLA on the validated 25 Hz insertion dataset."""

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import deque
from copy import copy
from itertools import islice
from pathlib import Path

import numpy as np

from .smolvla25_data import CAMERAS, CORNER_INDICES, FPS, REPO_ID, VALIDATION_REPO_ID, SmolVLAInsertion


TOTAL_UPDATES = 15_000
SAVE_EVERY = 2_500
EVAL_EVERY = 5_000
BENCHMARK_UPDATES = 20
CAMERA_MAP = {key: f"observation.images.camera{i}" for i, key in enumerate(CAMERAS, 1)}
VALIDATION_SEED = 271828


def validation_loader(root, policy_config, backend, batch_size, workers):
    from torch.utils.data import DataLoader
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    path = root / "validation-dataset"
    meta = LeRobotDatasetMetadata(VALIDATION_REPO_ID, root=path)
    summary = json.loads((root / "validation-summary.json").read_text(encoding="utf-8"))
    if (meta.total_episodes, meta.fps, meta.total_frames) != (10, FPS, summary["frames"]):
        raise RuntimeError("Validation dataset is incomplete or has an incompatible frame rate")
    dataset = LeRobotDataset(VALIDATION_REPO_ID, root=path, return_uint8=True, video_backend=backend,
                             delta_timestamps=resolve_delta_timestamps(policy_config, meta, CAMERA_MAP))
    return DataLoader(dataset, batch_size=batch_size, num_workers=min(workers, 2), shuffle=False, pin_memory=True)


def validation_loss(policy, preprocessor, loader, *, limit_batches=None):
    import torch
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    was_training = policy.training
    policy.eval()
    frames = valid_action_steps = 0
    loss_sum = 0.0
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad(), \
                torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(VALIDATION_SEED)
            batches = loader if limit_batches is None else islice(loader, limit_batches)
            for raw in batches:
                batch = _preprocess_dataset_batch(raw, loader.dataset.meta.camera_keys, CAMERA_MAP, preprocessor)
                loss, _ = policy(batch)
                batch_frames, horizon = batch["action"].shape[:2]
                pad = batch.get("action_is_pad")
                count = batch_frames * horizon if pad is None else int((~pad).sum().item())
                value = loss.item()
                if not math.isfinite(value) or count == 0:
                    raise RuntimeError(f"Invalid held-out loss: {value}, valid action steps: {count}")
                loss_sum += value * count
                valid_action_steps += count
                frames += batch_frames
    finally:
        policy.train(was_training)
    if not frames or (limit_batches is None and frames != len(loader.dataset)):
        raise RuntimeError(f"Validation scored {frames} of {len(loader.dataset)} frames")
    return dict(val_loss=loss_sum / valid_action_steps, validation_frames=frames,
                validation_action_steps=valid_action_steps, validation_seed=VALIDATION_SEED)


def append_json(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def save_curves(output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    train = [json.loads(line) for line in (output / "train-metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    val = {row["step"]: row for row in map(json.loads, (output / "validation-metrics.jsonl")
           .read_text(encoding="utf-8").splitlines())}
    csv_temp = output / "curves.tmp.csv"
    fields = (*train[0].keys(), "val_loss")
    with csv_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in train:
            writer.writerow(dict(**row, val_loss=val.get(row["step"], {}).get("val_loss")))
    csv_temp.replace(output / "curves.csv")

    steps = [row["step"] for row in train]
    losses = np.array([row["train_loss"] for row in train])
    window = min(50, len(losses))
    smoothed = np.convolve(losses, np.full(window, 1 / window), mode="valid")
    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), layout="constrained")
    ax_loss.plot(steps, losses, color="#2867a0", alpha=.19, linewidth=.8)
    ax_loss.plot(steps[window - 1:], smoothed, color="#2867a0", label=f"Train loss ({window}-step mean)")
    ax_loss.plot(list(val), [row["val_loss"] for row in val.values()], "o-", color="#bd592c", label="Validation loss")
    ax_loss.set_ylabel("Flow matching loss")
    ax_loss.legend()
    ax_lr.plot(steps, [row["lr"] for row in train], color="#337d74")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_xlabel("Optimizer step")
    for axis in (ax_loss, ax_lr):
        axis.grid(alpha=.2)
    figure_temp = output / "curves.tmp.png"
    fig.savefig(figure_temp, dpi=150)
    plt.close(fig)
    figure_temp.replace(output / "curves.png")


def evaluate(policy, preprocessor, postprocessor, splits, destination, *, smoke=False):
    import cv2
    import torch

    destination.mkdir(exist_ok=False)
    env = SmolVLAInsertion()
    summary = {}
    groups = {"train": [0]} if smoke else {"train": splits["eval_train_indices"],
                                          "validation": range(len(splits["validation"]))}
    policy.eval()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), \
                (destination / "episodes.jsonl").open("x", buffering=1) as log:
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
                                if smoke or (group == "train" and index in CORNER_INDICES):
                                    rgb = np.concatenate([obs[key] for key in CAMERAS], axis=1)
                                    if writer is None:
                                        video_name = "smoke.mp4" if smoke else f"corner-{index:02d}.mp4"
                                        writer = cv2.VideoWriter(str(destination / video_name),
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
    if (data["episodes"], splits["fps"]) != (100, FPS):
        raise RuntimeError("Expected a complete 100-episode dataset at 25 Hz")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required for this run")
    if args.batch_size % args.micro_batch_size:
        raise ValueError("Microbatch size must divide batch size")
    if args.speed_test is not None and args.speed_test < 12:
        raise ValueError("Speed tests need at least 12 optimizer updates")
    accumulation = args.batch_size // args.micro_batch_size
    speed_test = args.speed_test is not None
    measure_timing = args.benchmark or speed_test
    formal = not measure_timing
    if formal and not (root / "validation-summary.json").is_file():
        raise RuntimeError("Collect the ten held-out expert trajectories before formal training")
    run_name = "speed-test" if speed_test else "smoke" if args.benchmark else "training"
    output = args.output.resolve() if args.output else root / run_name
    target_updates = args.speed_test if speed_test else BENCHMARK_UPDATES if args.benchmark else TOTAL_UPDATES
    steps = target_updates * accumulation
    original_update = trainer.update_policy
    original_save = trainer.save_checkpoint
    original_last = trainer.update_last_checkpoint
    original_make_policy = trainer.make_policy
    original_make_optimizer = trainer.make_optimizer_and_scheduler
    updates = 0
    times = []
    first_update_elapsed = None
    last_update = None
    peak_reserved = peak_allocated = 0.
    losses = deque(maxlen=50 * accumulation)
    window_losses = []
    window_data = []
    data_times = []
    update_losses = []
    weight_info = {}
    current = {}
    previous = None
    began = time.perf_counter()

    def make_policy(*policy_args, **policy_kwargs):
        policy = original_make_policy(*policy_args, **policy_kwargs)
        # SmolVLA freezes some structurally unused layers even with both freeze flags off.
        # Enable gradients on every parameter for this full-finetuning run.
        policy.float().requires_grad_(True)
        params = list(policy.parameters())
        if not params or any(p.dtype != torch.float32 for p in params):
            raise RuntimeError("SmolVLA parameters must use FP32 master weights")
        weight_info.update(trainable_parameters=sum(p.numel() for p in params if p.requires_grad),
                           frozen_parameters=sum(p.numel() for p in params if not p.requires_grad),
                           master_weight_dtype="float32", ema_enabled=False,
                           chunk_size=policy.config.chunk_size, n_action_steps=policy.config.n_action_steps,
                           empty_cameras=policy.config.empty_cameras, rename_map=CAMERA_MAP,
                           image_resize=policy.config.resize_imgs_with_padding, compiled=policy.config.compile_model)
        if formal:
            current["validation_loader"] = validation_loader(root, policy.config, args.video_backend,
                                                             args.micro_batch_size, args.workers)
        print("SMOL_WEIGHT_CONFIG", json.dumps(weight_info), flush=True)
        return policy

    def make_optimizer_and_scheduler(cfg, policy):
        schedule_cfg = copy(cfg)
        schedule_cfg.steps = TOTAL_UPDATES * accumulation
        return original_make_optimizer(schedule_cfg, policy)

    def update(*update_args, **update_kwargs):
        nonlocal updates, last_update, first_update_elapsed, peak_reserved, peak_allocated
        accelerator = update_kwargs["accelerator"]
        scheduler = update_kwargs.pop("lr_scheduler")
        if accelerator.mixed_precision != "bf16" or accelerator.num_processes != 1:
            raise RuntimeError("Use one GPU with BF16 mixed precision")
        if last_update is None:
            last_update = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
        tracker, metrics = original_update(*update_args, **update_kwargs)
        loss = float(tracker.loss.val)
        if not math.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss after {updates} optimizer updates")
        losses.append(loss)
        window_losses.append(loss)
        data_meter = getattr(tracker, "dataloading_s", None)
        if data_meter is not None:
            window_data.append(float(data_meter.val))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 1024**3)
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 1024**3)
        if accelerator.sync_gradients:
            if accelerator.optimizer_step_was_skipped:
                raise RuntimeError("Optimizer update was skipped")
            if scheduler is not None:
                scheduler.step()
            updates += 1
            if measure_timing:
                torch.cuda.synchronize()
                now = time.perf_counter()
                if first_update_elapsed is None:
                    first_update_elapsed = now - began
                times.append(now - last_update)
                last_update = now
                update_losses.append(float(np.mean(window_losses)))
                data_times.append(float(np.sum(window_data)))
                print("SMOL_UPDATE", updates, "loss", update_losses[-1], "seconds", round(times[-1], 3), flush=True)
            else:
                now = time.perf_counter()
                optimizer = update_args[3] if len(update_args) > 3 else update_kwargs["optimizer"]
                row = dict(step=updates, train_loss=float(np.mean(window_losses)), lr=optimizer.param_groups[0]["lr"],
                           seconds=now - last_update, peak_reserved_gib=peak_reserved)
                output.mkdir(parents=True, exist_ok=True)
                append_json(output / "train-metrics.jsonl", row)
                last_update = now
                if updates == 1 or updates % 50 == 0:
                    print("SMOL_FORMAL_UPDATE", json.dumps(row), flush=True)
            window_losses.clear()
            window_data.clear()
        return tracker, metrics

    def save(**kwargs):
        original_save(**kwargs)
        current["checkpoint_dir"] = kwargs["checkpoint_dir"]
        if not args.benchmark:
            current.update(kwargs)

    def mark_latest(checkpoint_dir):
        nonlocal previous, last_update
        original_last(checkpoint_dir)
        if args.benchmark:
            return
        accelerator = current["accelerator"]
        policy = accelerator.unwrap_model(current["policy"])
        held_out = validation_loss(policy, current["preprocessor"], current["validation_loader"])
        validation_row = dict(step=updates, train_loss=float(np.mean(losses)), **held_out)
        append_json(output / "validation-metrics.jsonl", validation_row)
        save_curves(output)
        print("SMOL_VALIDATION_LOSS", json.dumps(validation_row), flush=True)
        if updates % EVAL_EVERY == 0:
            destination = output / f"eval-{updates:05d}"
            was_training = policy.training
            try:
                result = evaluate(policy, current["preprocessor"], current["postprocessor"], splits, destination)
            finally:
                policy.train(was_training)
            summary = dict(**validation_row, **result)
            (destination / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
        if previous is not None:
            old = previous.resolve(strict=True)
            if old.parent != checkpoint_dir.resolve().parent or old != previous.absolute() or old == checkpoint_dir.resolve():
                raise RuntimeError(f"Refusing to prune checkpoint outside this run: {previous}")
            shutil.rmtree(old)
        previous = checkpoint_dir
        last_update = time.perf_counter()

    trainer.make_policy = make_policy
    trainer.make_optimizer_and_scheduler = make_optimizer_and_scheduler
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
                f"--steps={steps}", f"--save_freq={min(SAVE_EVERY, target_updates) * accumulation}",
                f"--log_freq={min(50, target_updates) * accumulation}",
                f"--num_workers={args.workers}", f"--output_dir={output}",
                f"--rename_map={json.dumps(CAMERA_MAP)}", "--ema.enable=false", "--policy.push_to_hub=false",
                "--wandb.enable=false", "--env_eval_freq=0", f"--save_checkpoint={str(not speed_test).lower()}"]
    if args.compile_mode is not None:
        sys.argv.extend(["--policy.compile_model=true", f"--policy.compile_mode={args.compile_mode}"])
    if args.image_size is not None:
        sys.argv.append(f"--policy.resize_imgs_with_padding=[{args.image_size},{args.image_size}]")
    trainer.main()
    if updates != target_updates:
        raise RuntimeError(f"Expected {target_updates} optimizer updates, got {updates}")
    if speed_test:
        steady = np.asarray(times[len(times) // 2:])
        seconds = float(steady.mean())
        result = dict(
            **weight_info, updates=updates, effective_batch=args.batch_size, micro_batch=args.micro_batch_size,
            accumulation=accumulation, compile_mode=args.compile_mode, video_backend=args.video_backend,
            seconds_per_update=seconds, median_seconds_per_update=float(np.median(steady)),
            steady_min_seconds=float(steady.min()), steady_max_seconds=float(steady.max()),
            samples_per_second=args.batch_size / seconds, estimated_15000_hours=seconds * TOTAL_UPDATES / 3600,
            first_update_elapsed_seconds=first_update_elapsed, total_elapsed_seconds=time.perf_counter() - began,
            peak_reserved_gib=peak_reserved, peak_allocated_gib=peak_allocated, update_seconds=times,
            first_loss=update_losses[0], last_loss=update_losses[-1],
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "speed.json").write_text(json.dumps(result, indent=2))
        print("SMOL_SPEED_RESULT", json.dumps(result), flush=True)
    if args.benchmark:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        checkpoint = current["checkpoint_dir"]
        model_path = checkpoint / "pretrained_model"
        if (checkpoint / "pretrained_model_ema").exists() or (checkpoint / "training_state" / "ema_state.pt").exists():
            raise RuntimeError("EMA files appeared in a run with EMA disabled")
        if not (model_path / "model.safetensors").is_file():
            raise RuntimeError("The twentieth optimizer update did not save a policy checkpoint")
        scheduler_state = json.loads((checkpoint / "training_state" / "scheduler_state.json").read_text())
        if scheduler_state["last_epoch"] != BENCHMARK_UPDATES:
            raise RuntimeError(f"Scheduler advanced {scheduler_state['last_epoch']} times during the smoke test")
        seconds = float(np.mean(times[4:]))
        timing = dict(**weight_info, updates=updates, train_loss_step1=update_losses[0],
                      train_loss_step20=update_losses[-1], seconds_per_update=seconds,
                      dataloading_seconds_per_update=float(np.mean(data_times[4:])),
                      estimated_train_hours=seconds * TOTAL_UPDATES / 3600,
                      peak_reserved_gib=peak_reserved, peak_allocated_gib=peak_allocated,
                      batch_size=args.batch_size, micro_batch_size=args.micro_batch_size,
                      gradient_accumulation_steps=accumulation, video_backend=args.video_backend,
                      scheduler_updates=scheduler_state["last_epoch"], scheduler_decay_steps=TOTAL_UPDATES,
                      checkpoint=str(checkpoint), seconds_through_save=time.perf_counter() - began)
        (output / "timing.json").write_text(json.dumps(timing, indent=2))
        print("SMOL_TIMING", json.dumps(timing), flush=True)
        torch.cuda.empty_cache()
        policy = SmolVLAPolicy.from_pretrained(model_path, strict=True).float()
        if (policy.config.n_action_steps, policy.config.chunk_size) != (5, 25):
            raise RuntimeError("Reloaded checkpoint has a different action cadence")
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(model_path))
        destination = output / "reload-eval"
        result = evaluate(policy, preprocessor, postprocessor, splits, destination, smoke=True)
        summary = dict(**timing, reload_eval=result, total_seconds=time.perf_counter() - began)
        (output / "smoke-summary.json").write_text(json.dumps(summary, indent=2))
        print("SMOL_SMOKE_PASS", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--benchmark", action="store_true")
    run_mode.add_argument("--speed-test", type=int, metavar="UPDATES")
    parser.add_argument("--compile-mode", choices=("default", "max-autotune", "max-autotune-no-cudagraphs"))
    parser.add_argument("--image-size", type=int, choices=(384, 512))
    parser.add_argument("--batch-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--micro-batch-size", type=int, choices=(4, 8, 16, 32, 64), default=8)
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"), default="torchcodec")
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
