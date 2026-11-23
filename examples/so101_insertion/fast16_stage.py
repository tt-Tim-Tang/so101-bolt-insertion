"""Train fast16 to a fixed optimizer update, and evaluate the saved EMA on its frozen split."""
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

from . import upstream
from .fast16 import FastMetrics
from .fast16_train import evaluate
from .online import Policy


def train(root, run, scratch, target, resume=None):
    from lerobot.scripts import lerobot_train as trainer

    args = json.loads((root / "training-command.json").read_text())
    accum = 4
    completed = 0
    if resume is None:
        args = [arg for arg in args if not arg.startswith(("--steps=", "--save_checkpoint=", "--output_dir="))]
        args.append(f"--output_dir={run / 'train'}")
    else:
        from diffusers.training_utils import EMAModel

        load_ema = EMAModel.load_state_dict

        @torch.no_grad()
        def restore_ema(ema, state):
            # diffusers deep-copies the entire state, which needs a third L40-sized
            # set of EMA weights. Reuse the allocated shadow for the large tensors.
            metadata = state.copy()
            saved = metadata.pop("shadow_params")
            load_ema(ema, metadata)
            for destination, source in zip(ema.shadow_params, saved, strict=True):
                if destination.shape != source.shape or destination.dtype != source.dtype:
                    raise ValueError("EMA checkpoint tensor mismatch")
                destination.copy_(source)
                if not torch.equal(destination, source):
                    raise ValueError("EMA checkpoint restore mismatch")
            assert ema.optimization_step == completed

        EMAModel.load_state_dict = restore_ema
        metadata = json.loads((resume / "training_state/training_step.json").read_text())
        assert metadata["grad_accum_steps"] == accum and metadata["batch_size"] == 8
        completed = metadata["step"] // accum
        args = [args[0], "finetune", "--resume=true",
                f"--config_path={resume / 'pretrained_model/train_config.json'}"]
    assert completed <= target
    args.extend([f"--steps={target * accum}", f"--save_freq={target * accum}", "--save_checkpoint=true"])
    original_update = trainer.update_policy
    losses = []

    def update(tracker, policy, batch, optimizer, *params, **kwargs):
        nonlocal completed
        scheduler = kwargs.pop("lr_scheduler")
        result = original_update(tracker, policy, batch, optimizer, *params, **kwargs)
        losses.append((float(tracker.loss.val), len(batch["action"])))
        engine = kwargs["accelerator"]
        if engine.sync_gradients:
            if engine.optimizer_step_was_skipped:
                raise RuntimeError("Optimizer update skipped")
            assert len(losses) == accum
            scheduler.step()
            completed += 1
            samples = sum(n for _, n in losses)
            row = dict(update=completed, samples=samples, train_loss=sum(v*n for v,n in losses)/samples,
                       last_microbatch_loss=losses[-1][0], lr=optimizer.param_groups[0]["lr"])
            with (run / "updates.jsonl").open("a") as log:
                print(json.dumps(row), file=log)
            losses.clear()
        return result

    def checkpoint_dir(output_dir, total_steps, step):
        assert step == total_steps == target * accum and completed == target
        return scratch / f"update-{target}"

    trainer.update_policy = update
    trainer.get_step_checkpoint_dir = checkpoint_dir
    sys.argv = args
    (run / f"command-{'resume' if resume else 'train'}-{target}.json").write_text(json.dumps(args, indent=2))
    upstream.main()
    assert completed == target and not losses


def evaluate_checkpoint(root, run, checkpoint, target):
    cases = json.loads((root / "splits.json").read_text())
    clearance = json.loads((root / "environment.json").read_text())["clearance_mm"]
    policy = Policy(checkpoint / "pretrained_model_ema", None, 0, "cuda",
                    str(Path(os.environ["SRC"]) / "tokenizer"), execution_steps=10)
    metrics = {}
    for group in ("train", "validation"):
        path = run / f"{group}-{target}.jsonl"
        evaluate(policy.policy, (policy.preprocess, policy.postprocess), cases[group], path,
                 partial(FastMetrics, target_bore_radius=.005 + clearance / 1000))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["case"] for r in sorted(rows, key=lambda r: r["index"])] == cases[group]
        errors = [r["closure_xy_mm"] for r in rows if r["closure_xy_mm"] is not None]
        metrics[group] = dict(n=len(rows), **{k:sum(r[k] for r in rows) for k in ("pickup","arrival","success")},
                              closure_observed=len(errors), closure_xy_median_mm=float(np.median(errors)) if errors else None)
    last = json.loads((run / "updates.jsonl").read_text().splitlines()[-1])
    assert last["update"] == target
    (run / f"metrics-{target}.json").write_text(json.dumps(dict(**last, **metrics), indent=2))


if __name__ == "__main__":
    root = Path(os.environ["FAST"])
    run = Path(os.environ["FAST_RUN"])
    scratch = Path(os.environ["FAST_SCRATCH"])
    mode, target = sys.argv[1], int(sys.argv[2])
    if mode == "eval":
        evaluate_checkpoint(root, run, scratch / f"update-{target}", target)
    elif mode in ("train", "resume"):
        train(root, run, scratch, target, Path(sys.argv[3]) if mode == "resume" else None)
    else:
        raise ValueError(mode)
