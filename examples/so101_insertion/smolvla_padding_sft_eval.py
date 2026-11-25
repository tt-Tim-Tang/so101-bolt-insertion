"""Evaluate the padding-velocity change on the original held-out SFT initial states."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .smolvla25_data import CAMERAS, FPS, SmolVLAInsertion
from .smolvla_motion_audit import columns, write_csv, write_json
from .smolvla_padding_ab import padding_velocity
from .smolvla_rlt_stage2 import FrozenSmolVLA


def replay(env, features, case, seed, corrected, video=None):
    env.start(case)
    policy = features.policy
    noises, ratios, wrist_steps = [], [], []
    writer = None
    try:
        if video is not None:
            obs = env.observation()
            height, width = obs[CAMERAS[0]].shape[:2]
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width * len(CAMERAS), height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video: {video}")

        def write_frame(obs):
            frame = np.concatenate([obs[key] for key in CAMERAS], axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode(), \
                torch.autocast("cuda", dtype=torch.bfloat16), padding_velocity(policy, corrected, noises, ratios):
            torch.manual_seed(seed)
            policy.reset()
            done = timeout = False
            while not (done or timeout):
                obs = env.observation()
                batch = {key: torch.from_numpy(obs[key]).permute(2, 0, 1).float() / 255 for key in CAMERAS}
                batch.update(task=obs["task"], **{"observation.state": torch.from_numpy(obs["observation.state"])})
                chunk = policy.predict_action_chunk(features.pre(batch))[:, :policy.config.n_action_steps]
                for action in features.post(chunk.float())[0].cpu().numpy():
                    if writer is not None:
                        write_frame(obs)
                    previous = env.data.site(env.grasp_id).xpos.copy()
                    _, done, timeout, info = env.step(action)
                    wrist_steps.append(float(np.linalg.norm(env.data.site(env.grasp_id).xpos - previous) * 1000))
                    if done or timeout:
                        break
                    if writer is not None:
                        obs = env.observation()
            if writer is not None:
                write_frame(env.observation())
    finally:
        if writer is not None:
            writer.release()
    if corrected and not all(0 <= ratio < .01 for ratio in ratios):
        raise RuntimeError("Corrected padding residual exceeds 1%")
    end = dict(info, steps=env.steps, timeout=bool(timeout and not done),
               wrist_p95_mm=float(np.quantile(wrist_steps, .95)), wrist_max_mm=max(wrist_steps),
               median_terminal_padding_ratio=float(np.median(ratios)))
    return end, noises


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("original_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--videos-only", action="store_true", help="Render videos from a completed pairs.jsonl")
    args = parser.parse_args()
    cases = json.loads(args.splits.read_text())["validation"]
    historical = json.loads(args.original_summary.read_text())
    seed_base = historical["rollout_seed"] + 10_000
    penetration = historical["penetration_limit_mm"]
    if not args.videos_only:
        args.output.mkdir()
    features = FrozenSmolVLA(args.checkpoint)
    env = SmolVLAInsertion(penetration_limit_mm=penetration)
    rows = []
    try:
        pairs_path = args.output / "pairs.jsonl"
        if args.videos_only:
            rows = [json.loads(line) for line in pairs_path.read_text().splitlines()]
            expected = [(i, case["target"], seed_base + i) for i, case in enumerate(cases)]
            if [(r["index"], r["target"], r["seed"]) for r in rows] != expected:
                raise ValueError("Saved evaluation pairs are incomplete or do not match the split")
        else:
            with pairs_path.open("x", buffering=1) as log:
                for i, case in enumerate(cases):
                    seed = seed_base + i
                    native, native_noise = replay(env, features, case, seed, False)
                    corrected, corrected_noise = replay(env, features, case, seed, True)
                    common = min(len(native_noise), len(corrected_noise))
                    np.testing.assert_array_equal(native_noise[:common], corrected_noise[:common])
                    row = dict(index=i, target=case["target"], seed=seed, matched_noise_chunks=common,
                               native=native, corrected=corrected)
                    rows.append(row)
                    print(json.dumps(row), file=log)
                    print(f"VALIDATION {i}: insertion {int(native['success'])} -> {int(corrected['success'])}, "
                          f"wrist P95 {native['wrist_p95_mm']:.2f} -> {corrected['wrist_p95_mm']:.2f} mm", flush=True)

        def video_rank(row):
            before, after = row["native"], row["corrected"]
            return (after["success"] and not before["success"], after["success"],
                    before["wrist_p95_mm"] - after["wrist_p95_mm"])

        selected = max(rows, key=video_rank)
        video_outcomes = {}
        video_noises = []
        for mode in ("native", "corrected"):
            outcome, noise = replay(env, features, cases[selected["index"]], selected["seed"], mode == "corrected",
                                    args.output / f"{mode}.mp4")
            video_outcomes[mode] = outcome
            video_noises.append(noise)
            print("VIDEO", mode, json.dumps(outcome), flush=True)
        common = min(map(len, video_noises))
        np.testing.assert_array_equal(video_noises[0][:common], video_noises[1][:common])

        metrics = ("pickup", "arrival", "success", "slipped", "opened_early", "penetration_failure", "timeout")
        summary = dict(checkpoint=str(args.checkpoint), original=historical["validation"],
                       episodes=len(cases), penetration_limit_mm=penetration, control_hz=FPS, seed_base=seed_base,
                       video_index=selected["index"], video_selection="Prioritize a gained insertion, then wrist P95 reduction",
                       videos=dict(outcomes=video_outcomes, matched_noise_chunks=common), pairs=rows)
        for mode in ("native", "corrected"):
            group = [row[mode] for row in rows]
            summary[mode] = {key: sum(r[key] for r in group) for key in metrics}
            for key in ("wrist_p95_mm", "wrist_max_mm"):
                summary[mode]["median_" + key] = float(np.median([r[key] for r in group]))
        flat = [dict(index=row["index"], seed=row["seed"], mode=mode, **{
            key: row[mode][key] for key in (*metrics, "steps", "wrist_p95_mm", "wrist_max_mm")
        }) for row in rows for mode in ("native", "corrected")]
        write_csv(args.output / "episodes.csv", columns(flat))
        write_json(args.output / "summary.json", summary)
        print("EVAL_COMPLETE", summary["native"], summary["corrected"], "video index", selected["index"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
