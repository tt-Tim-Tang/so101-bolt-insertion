"""Collect random-hole SO-101 warmup, train RLT-TD3 offline, and evaluate once."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .smolvla25_data import TARGET_RANGE_MM, SmolVLAInsertion, make_case
from .smolvla_rlt_stage2 import ActionMap, FrozenSmolVLA, collect_rollouts, collect_start, rollout, save_json
from .smolvla_rlt_td3 import update_td3


WARMUP_SEED = 20261123
SIGMA = .002
MAX_STEPS = 200
UPDATES = 3000


def demonstration_limits(root):
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if (info["total_episodes"], info["fps"]) != (150, 25):
        raise ValueError("Expected the merged 150-episode, 25 Hz demonstration dataset")
    rows = [np.stack(pq.read_table(path, columns=["action"])["action"].to_pylist())
            for path in sorted((root / "data").glob("*/*.parquet"))]
    arm = np.concatenate(rows)[:, :5]
    if len(arm) != info["total_frames"] or not np.isfinite(arm).all():
        raise ValueError("Missing or non-finite demonstration actions")
    return arm.min(axis=0) - .05, arm.max(axis=0) + .05


def evaluation_starts(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["episodes"]) != 10 or len({tuple(row["case"]["target"]) for row in manifest["episodes"]}) != 10:
        raise ValueError("Expected ten distinct fixed evaluation holes")
    starts = []
    for row in manifest["episodes"]:
        with np.load(folder / f'{row["index"]:03}.npz') as state:
            starts.append((row, state["physics"], None))
    return starts


def warmup(env, reset_env, features, mapping, heldout, output):
    reserved = np.array([row["case"]["target"] for row, _, _ in heldout])
    rng = np.random.default_rng(WARMUP_SEED)
    seen, replay, rows, rejected = set(), [], [], []
    started = time.monotonic()
    with (output / "warmup-episodes.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for attempt in range(500):
            xy = rng.uniform(-TARGET_RANGE_MM, TARGET_RANGE_MM, size=2)
            distance = float(np.linalg.norm(reserved - xy, axis=1).min())
            if distance <= 1e-6 or tuple(xy) in seen:
                continue
            seen.add(tuple(xy))
            case = make_case(xy.tolist(), rng)
            start = collect_start(reset_env, attempt, case, int(rng.integers(2**31)))
            start[0]["nearest_eval_hole_mm"] = distance
            if start[1] is None:
                rejected.append(start[0])
                print(json.dumps(start[0]), file=log)
                continue
            before = len(replay)
            row = {**start[0], **rollout(env, features, mapping, start, replay)}
            row["replay_chunks"] = len(replay) - before
            rows.append(row)
            line = json.dumps(dict(episode=len(rows), total_chunks=len(replay), **row))
            print(line, file=log)
            print(line, flush=True)
            if len(rows) >= 30 and len(replay) >= 1000:
                break
        else:
            raise RuntimeError("Warmup did not reach both collection minima within 500 reset attempts")
    positives = sum(bool(row["rewards"].any()) for row in replay)
    steps = sum(row["steps"] for row in rows)
    save_json(output / "warmup.json", dict(
        episodes=len(rows), replay_chunks=len(replay), positive_chunks=positives,
        positive_fraction=positives / len(replay), insertions=sum(row["success"] for row in rows),
        closed_gripper_slips=sum(row["slipped"] for row in rows), control_steps=steps,
        steps_per_s=steps / (time.monotonic() - started), rejected_resets=len(rejected),
        nearest_eval_hole_mm=min(row["nearest_eval_hole_mm"] for row in rows),
        target_xy_min_mm=np.min([row["case"]["target"] for row in rows], axis=0).tolist(),
        target_xy_max_mm=np.max([row["case"]["target"] for row in rows], axis=0).tolist(),
    ))
    return replay, rng.bit_generator.state


def run(args):
    args.output.mkdir()
    heldout = evaluation_starts(args.eval_starts)
    env = SmolVLAInsertion(penetration_limit_mm=3.)
    reset_env = SmolVLAInsertion(penetration_limit_mm=3.)
    env.max_steps = MAX_STEPS
    try:
        actions = ActionMap(env, demonstration_limits(args.dataset), SIGMA)
        config = dict(
            joints=actions.report(), sigma=SIGMA, gripper_rad=-.055, max_steps=MAX_STEPS,
            warmup_seed=WARMUP_SEED, warmup_target_range_mm=TARGET_RANGE_MM,
            control_fps=env.control_fps, physics_dt=env.model.opt.timestep,
            smolvla=str(args.smolvla.resolve()), token=str(args.token.resolve()),
            token_source=str(args.token_source.resolve()), eval_starts=str(args.eval_starts.resolve()),
            training_dataset=str(args.dataset.resolve()), rlinf=str(args.rlinf.resolve()),
        )
        save_json(args.output / "run-config.json", config)
        features = FrozenSmolVLA(args.smolvla)
        features.load_token(args.token, args.token_source, args.smolvla)
        replay, warmup_rng = warmup(env, reset_env, features, actions, heldout, args.output)
        policy = update_td3(args.rlinf, actions, replay, args.output, updates=UPDATES,
                            offline_run={**config, "warmup_rng": warmup_rng})
        del replay
        baseline = collect_rollouts(env, features, actions, heldout)
        save_json(args.output / "baseline.json", baseline)
        eval_delta = []

        @torch.no_grad()
        def action(obs):
            batched = {key: value.unsqueeze(0) for key, value in obs.items()}
            predicted, _, _ = policy.sac_forward(batched, deterministic=True, apply_action_noise=False)
            predicted = predicted.reshape_as(obs["ref_chunk"])
            eval_delta.append(((predicted - obs["ref_chunk"]).abs()[..., :5] * actions.half[:5]).cpu())
            return predicted

        actor = collect_rollouts(env, features, actions, heldout, actor=action)
        delta = torch.stack(eval_delta)
        save_json(args.output / "actor-eval.json", dict(
            **actor, arm_mean_abs_rad=delta.mean().item(), arm_max_abs_rad=delta.max().item(),
        ))
    finally:
        reset_env.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("smolvla", "token", "token-source", "dataset", "eval-starts", "rlinf", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
