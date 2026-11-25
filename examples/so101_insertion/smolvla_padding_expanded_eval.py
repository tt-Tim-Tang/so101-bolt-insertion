"""Compare padding inference on the archived 30 held-out insertion starts."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .smolvla25_data import CAMERAS, FPS, SmolVLAInsertion
from .smolvla_motion_audit import columns, write_csv, write_json
from .smolvla_padding_ab import padding_velocity
from .smolvla_rlt_delta_train import expanded_starts
from .smolvla_rlt_offline import demonstration_limits
from .smolvla_rlt_stage2 import ActionMap, FrozenSmolVLA, rollout


class WristReplay(SmolVLAInsertion):
    def __init__(self):
        super().__init__(penetration_limit_mm=3.)
        self.wrist_steps = []
        self.video_path = None
        self.writer = None

    def frame(self):
        obs = self.observation()
        rgb = np.concatenate([obs[key] for key in CAMERAS], axis=1)
        if self.writer is None:
            shape = (rgb.shape[1], rgb.shape[0])
            self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, shape)
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open {self.video_path}")
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def step(self, action):
        if self.video_path is not None:
            self.frame()
        prev = self.data.site(self.grasp_id).xpos.copy()
        result = super().step(action)
        self.wrist_steps.append(float(np.linalg.norm(self.data.site(self.grasp_id).xpos - prev) * 1000))
        return result


def evaluate_pair(env, features, mapping, start, video_folder=None):
    row = start[0]
    pair = dict(index=row["index"], seed=row["seed"], target_xy_mm=row["case"]["target"],
                initial_tip_offset_mm=row["tip_offset_mm"])
    sequences = []
    for mode in ("native", "corrected"):
        corrected = mode == "corrected"
        noises, ratios = [], []
        env.wrist_steps.clear()
        env.video_path = video_folder / f"{mode}.mp4" if video_folder is not None else None
        try:
            with padding_velocity(features.policy, corrected, noises, ratios):
                outcome = rollout(env, features, mapping, start)
            if video_folder is not None:
                env.frame()
        finally:
            if env.writer is not None:
                env.writer.release()
                env.writer = None
            env.video_path = None
        if corrected and not all(0 <= ratio < .01 for ratio in ratios):
            raise RuntimeError("Corrected padding residual exceeds 1%")
        outcome["timeout"] = (outcome["steps"] == env.max_steps and not any(
            outcome[key] for key in ("success", "slipped", "penetration_failure")
        ))
        pair[mode] = dict(**outcome, wrist_p95_mm=float(np.quantile(env.wrist_steps, .95)),
                          wrist_max_mm=max(env.wrist_steps),
                          median_terminal_padding_ratio=float(np.median(ratios)))
        sequences.append(noises)
    common = min(map(len, sequences))
    np.testing.assert_array_equal(sequences[0][:common], sequences[1][:common])
    pair["matched_noise_chunks"] = common
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round1", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.round1 / "launch.json").read_text())["config"]
    starts = expanded_starts(args.round1 / "expanded-states")
    references = {mode: json.loads((args.round1 / filename).read_text()) for mode, filename in (
        ("smolvla", "expanded-smolvla.json"), ("round1_actor", "expanded-best.json")
    )}
    indices = [start[0]["index"] for start in starts]
    if any([r["index"] for r in ref["rows"]] != indices for ref in references.values()):
        raise ValueError("The archived references and restored starts differ")
    args.output.mkdir()
    env = WristReplay()
    env.max_steps = config["max_steps"]
    try:
        mapping = ActionMap(env, demonstration_limits(Path(config["training_dataset"])), config["sigma"])
        if mapping.report() != config["joints"] or env.model.opt.timestep != config["physics_dt"]:
            raise ValueError("The live environment differs from the archived evaluation")
        features = FrozenSmolVLA(Path(config["smolvla"]))
        rows = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), \
                (args.output / "pairs.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for start in starts:
                pair = evaluate_pair(env, features, mapping, start)
                rows.append(pair)
                print(json.dumps(pair), file=log)
                before, after = pair["native"], pair["corrected"]
                print(f"EVAL {pair['index']:02d}: insertion {int(before['success'])} -> {int(after['success'])}, "
                      f"slip {before['slipped']} -> {after['slipped']}", flush=True)

        def rank(pair):
            before, after = pair["native"], pair["corrected"]
            return (after["success"] and not before["success"], after["success"],
                    before["wrist_p95_mm"] - after["wrist_p95_mm"])

        selected = max(rows, key=rank)
        keys = ("success", "slipped", "penetration_failure", "timeout")
        summary = dict(checkpoint=config["smolvla"], episodes=len(rows), control_hz=FPS,
                       max_steps=env.max_steps, penetration_limit_mm=3., historical=references, pairs=rows)
        for mode in ("native", "corrected"):
            outcomes = [pair[mode] for pair in rows]
            summary[mode] = {key: sum(row[key] for row in outcomes) for key in keys}
            summary[mode]["mean_steps"] = float(np.mean([row["steps"] for row in outcomes]))
            summary[mode]["median_wrist_p95_mm"] = float(np.median([row["wrist_p95_mm"] for row in outcomes]))
        write_json(args.output / "summary.json", summary)
        csv_rows = [dict(index=pair["index"], mode=mode, seed=pair["seed"], **{
            key: pair[mode][key] for key in (*keys, "steps", "wrist_p95_mm", "wrist_max_mm")
        }) for pair in rows for mode in ("native", "corrected")]
        write_csv(args.output / "episodes.csv", columns(csv_rows))

        start = starts[indices.index(selected["index"])]
        summary["video_replay"] = evaluate_pair(env, features, mapping, start, args.output)
        write_json(args.output / "summary.json", summary)
        print("EVAL_COMPLETE", summary["native"], summary["corrected"], "video index", selected["index"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
