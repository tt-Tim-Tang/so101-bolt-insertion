"""Benchmark 20 optimizer updates, then run the bounded 500/1000-update check."""
import json
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from . import upstream
from .fast16 import FastMetrics
from .online import Policy
from .wide_eval import evaluate


def main():
    from diffusers.training_utils import EMAModel
    from lerobot.scripts import lerobot_train as trainer

    root = Path(os.environ["FAST"])
    cases = json.loads((root/"splits.json").read_text())
    settings = json.loads((root/"environment.json").read_text())
    collection = [json.loads(x) for x in (root/"collection.jsonl").read_text().splitlines()]
    assert len(collection) == 16 and all(r["success"] for r in collection)
    physical = int(os.environ.get("FAST_MICROBATCH", "8"))
    assert 32 % physical == 0
    accumulation = 32//physical
    native_update, native_processors, native_ema_step = trainer.update_policy, trainer.make_pre_post_processors, EMAModel.step
    completed = 0
    raw = None
    processors = None
    last_time = None
    durations = []
    last_loss = None
    last_lr = None

    def capture_processors(*args, **kwargs):
        nonlocal processors
        processors = native_processors(*args, **kwargs)
        return processors

    def update(tracker, policy, batch, optimizer, *args, **kwargs):
        nonlocal completed, raw, last_time, last_loss, last_lr
        if last_time is None:
            last_time = time.perf_counter()-tracker.dataloading_s.val-tracker.preprocessing_s.val
        scheduler = kwargs.pop("lr_scheduler")
        result = native_update(tracker, policy, batch, optimizer, *args, **kwargs)
        accelerator = kwargs["accelerator"]
        raw = accelerator.unwrap_model(policy)
        if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
            scheduler.step()
            completed += 1
        last_loss, last_lr = float(tracker.loss.val), optimizer.param_groups[0]["lr"]
        return result

    def evaluate_ema(ema):
        wrapper = Policy.__new__(Policy)
        wrapper.policy = raw
        wrapper.preprocess, wrapper.postprocess = processors
        wrapper.execution_steps, wrapper.encoder, wrapper.token_dim = 10, None, 0
        env_type = partial(FastMetrics, target_bore_radius=.005+settings["clearance_mm"]/1000)
        summary = {}
        with torch.random.fork_rng(devices=[0]), trainer._ema_weights(ema, raw):
            raw.eval()
            try:
                for group in ("train", "validation"):
                    path = root/f"{group}-{completed}.jsonl"
                    evaluate(wrapper, cases[group], path, env_type=env_type, batch_size=2)
                    rows = [json.loads(x) for x in path.read_text().splitlines()]
                    assert sorted(r["index"] for r in rows) == list(range(len(cases[group])))
                    errors = [r["closure_xy_mm"] for r in rows if r["closure_xy_mm"] is not None]
                    summary[group] = dict(n=len(rows), **{k: sum(r[k] for r in rows)
                        for k in ("pickup", "arrival", "success")},
                        closure_xy_median_mm=float(np.median(errors)) if errors else None,
                        closure_observed=len(errors))
                dest = root/"ema_weights"/str(completed)
                raw.save_pretrained(dest)
                for processor in processors:
                    processor.save_pretrained(dest)
            finally:
                raw.train()
        (root/f"metrics-{completed}.json").write_text(json.dumps(summary, indent=2))
        # Retain only the latest complete inference weights; keep both evaluations.
        old = root/"ema_weights"/"500"
        if completed == 1000 and old.is_dir() and not old.is_symlink():
            assert old.resolve().parent == (root/"ema_weights").resolve()
            shutil.rmtree(old)
        print("EVALUATED_EMA", completed, summary, flush=True)

    def ema_step(ema, parameters):
        nonlocal last_time
        native_ema_step(ema, parameters)
        assert ema.optimization_step == completed
        torch.cuda.synchronize()
        now = time.perf_counter()
        durations.append(now-last_time)
        row = dict(updates=completed, samples=completed*32, seconds=durations[-1],
                   loss=last_loss, lr=last_lr, ema_decay=ema.cur_decay_value)
        with (root/"training-progress.jsonl").open("a") as stream:
            print(json.dumps(row), file=stream)
        if completed == 20:
            seconds = float(np.mean(durations[2:]))
            forecast = dict(optimizer_updates=20, effective_batch_size=32,
                            physical_batch_size=physical, accumulation=accumulation,
                            seconds_per_update=seconds, projected_1000_seconds=1000*seconds,
                            excludes="model loading, checkpoint writes and online evaluation",
                            continue_training=1000*seconds <= 5400)
            (root/"benchmark.json").write_text(json.dumps(forecast, indent=2))
            print("BENCHMARK", forecast, flush=True)
            if not forecast["continue_training"]:
                print("STOPPED_AT_USER_RUNTIME_LIMIT", flush=True)
                raise SystemExit(0)
        if completed in (500, 1000):
            evaluate_ema(ema)
        if completed % 25 == 0:
            print("UPDATE", row, flush=True)
        last_time = time.perf_counter()

    trainer.make_pre_post_processors = capture_processors
    trainer.update_policy = update
    EMAModel.step = ema_step
    sys.argv = [sys.argv[0], "finetune", "--dataset.repo_id=local/so101_fast16",
        f"--dataset.root={root/'dataset'}", "--dataset.video_backend=pyav",
        "--policy.type=pi05", "--policy.pretrained_path=/root/so101-pi05-base",
        "--policy.dtype=bfloat16", "--policy.device=cuda", "--policy.gradient_checkpointing=true",
        "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=false",
        "--policy.empty_cameras=1", "--policy.use_relative_actions=false", "--policy.push_to_hub=false",
        f"--policy.text_tokenizer_name={Path(os.environ['SRC'])/'tokenizer'}",
        '--rename_map={"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}',
        "--ema.enable=true", "--ema.decay=0.99", "--policy.optimizer_lr=5e-5",
        "--policy.scheduler_warmup_steps=100", "--policy.scheduler_decay_steps=1000",
        "--policy.scheduler_decay_lr=5e-5", f"--batch_size={physical}",
        f"--accelerator.gradient_accumulation.steps={accumulation}", "--num_workers=0",
        "--seed=1000", "--wandb.enable=false", f"--steps={1000*accumulation}",
        f"--log_freq={25*accumulation}", "--save_checkpoint=false", f"--output_dir={root/'train'}"]
    (root/"training-command.json").write_text(json.dumps(sys.argv, indent=2))
    upstream.main()


if __name__ == "__main__":
    main()
