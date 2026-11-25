"""Export pre-contact RLT snapshots and label them with frozen SmolVLA rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .smolvla_padding_e2e30 import EndToEndTelemetry, quantiles
from .smolvla_rlt_stage2 import FrozenSmolVLA
from .smolvla_rlt_v3 import episode_result, handoff, read_cases

TRAIN_SEED = 714290
VALIDATION_SEED = 724290


class EpisodeTelemetry(EndToEndTelemetry):
    def start(self, case):
        super().start(case)
        self.last_step = None

    def step(self, action):
        self.last_step = super().step(action)
        return self.last_step


def save_json(path, value):
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fixed_cases(args):
    held, train = read_cases(args.episodes, args.baseline, args.splits)
    validation = json.loads(args.splits.read_text())["validation"]
    if any(c["source"] != [0, 0] or c["joints"] != [0] * 5 for c in train + validation):
        raise ValueError("Training and validation must start from the fixed pickup setup")
    if len(np.unique([c["target"] for c in train], axis=0)) != 150:
        raise ValueError("The split does not have 150 distinct training holes")
    return held, train, validation


@torch.inference_mode()
def finish_reference(env, features, *, frame=None):
    while not any(env.last_step[1:3]):
        raw = env.observation()
        commands = features.predict_reference(features.preprocess_observation(raw)).cpu().numpy()
        for command in commands:
            if frame is not None:
                frame(raw)
            _, done, timeout, _ = env.step(command)
            if done or timeout:
                break
            raw = env.observation()
    if frame is not None:
        frame(env.observation())
    _, done, timeout, info = env.last_step
    return dict(episode_result(info, timeout, done), steps=env.steps, stable_s=info["stable_s"])


def run(args):
    _, train, _ = fixed_cases(args)
    args.output.mkdir(parents=True, exist_ok=True)
    features = FrozenSmolVLA(args.checkpoint)
    env = EpisodeTelemetry(penetration_limit_mm=3.)
    rows = []
    try:
        for index in range(args.start, args.stop):
            case = train[index]
            stem = args.output / f"{index:03d}"
            metadata = stem.with_suffix(".json")
            if metadata.is_file():
                row = json.loads(metadata.read_text())
                if row["case"] != case or (row["snapshot"] and not stem.with_suffix(".npz").is_file()):
                    raise ValueError(f"Existing snapshot metadata does not match training hole {index}")
            else:
                attempts = []
                # A retry is still on the same training hole, with an independently recorded inference seed.
                for retry in range(3):
                    seed = TRAIN_SEED + index + retry * 100_000
                    with torch.inference_mode():
                        snapshot, rejection = handoff(env, features, case, seed)
                    xy = float(np.linalg.norm(env.data.body(env.bolt_id).xpos[:2]
                                              - env.model.body(env.target_id).pos[:2]) * 1000)
                    result = finish_reference(env, features)
                    attempts.append(dict(seed=seed, rejection=rejection, **result))
                    if snapshot is not None:
                        break
                row = dict(index=index, case=case, seed=seed, attempts=attempts, reference=result,
                           group="base_success" if result["success"] else "base_fail", snapshot=None)
                if snapshot is not None:
                    handoff_row, physics, cpu, cuda = snapshot
                    np.savez_compressed(stem.with_suffix(".npz"), physics=physics, cpu_rng=cpu.numpy(),
                                        cuda_rng=cuda.numpy())
                    row.update(snapshot=handoff_row, handoff_xy_mm=xy)
                save_json(metadata, row)
            rows.append(row)
            print("EXPORT", index + 1, len(train), row["group"], row["snapshot"] is not None,
                  row["seed"], row["reference"]["steps"], flush=True)
    finally:
        env.close()
    if (args.start, args.stop) != (0, len(train)):
        return
    summarize(args.output, rows)


def summarize(output, rows):
    starts = [r for r in rows if r["snapshot"] is not None]
    result = dict(train_holes=len(rows), validation_holes=10, heldout_holes=30, overlap_holes=0,
                  valid_handoffs=len(starts), rejected_indices=[r["index"] for r in rows if r["snapshot"] is None],
                  reseeded_indices=[r["index"] for r in rows if len(r["attempts"]) > 1],
                  base_success=sum(r["group"] == "base_success" for r in rows),
                  base_fail=sum(r["group"] == "base_fail" for r in rows),
                  crossing_xy_mm=quantiles([r["snapshot"]["crossing"]["xy_mm"] for r in starts]),
                  handoff_xy_mm=quantiles([r["handoff_xy_mm"] for r in starts]), rows=rows)
    save_json(output / "manifest.json", result)
    print("EXPORT_COMPLETE", {k: v for k, v in result.items() if k != "rows"}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "episodes", "baseline", "splits", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=150)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        _, train, _ = fixed_cases(args)
        rows = [json.loads((args.output / f"{i:03d}.json").read_text()) for i in range(len(train))]
        if [(r["index"], r["case"]) for r in rows] != list(enumerate(train)):
            raise ValueError("Exported rows differ from the training split")
        summarize(args.output, rows)
    else:
        if not 0 <= args.start < args.stop <= 150:
            parser.error("Expected a nonempty range inside the 150 training holes")
        run(args)


if __name__ == "__main__":
    main()
