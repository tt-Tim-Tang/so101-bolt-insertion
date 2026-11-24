"""Record bolt contact and seating telemetry for the original expanded evaluation."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .scene import MOUTH_Z, SHAFT_RADIUS
from .smolvla25_data import DEPTH_TOLERANCE_M, MAX_FLOOR_OVERRUN_M, MIN_AXIS_Z, SmolVLAInsertion
from .smolvla_rlt_delta_train import expanded_starts
from .smolvla_rlt_offline import MAX_STEPS, demonstration_limits
from .smolvla_rlt_stage2 import ActionMap, FrozenSmolVLA, rollout
from .smolvla_rlt_td3 import load_rlinf


class TelemetryInsertion(SmolVLAInsertion):
    def start(self, case):
        super().start(case)
        self.initial_pose = None
        self.first_contact = None
        self.first_floor_contact = None
        self.min_held_xy = math.inf
        self.max_held_depth = -math.inf
        self.min_held_xy_after_contact = math.inf
        self.slip_seated = None
        self.final_pose = None

    def bolt_pose(self):
        bolt = self.data.body(self.bolt_id)
        axis = bolt.xmat.reshape(3, 3)[:, 2]
        xy = bolt.xpos[:2] - self.model.body(self.target_id).pos[:2]
        depth = MOUTH_Z - bolt.xpos[2]
        tilt = math.degrees(math.acos(np.clip(axis[2], -1, 1)))
        return bolt.xpos, axis, xy, float(depth), tilt

    def capture(self, pose):
        _, _, xy, depth, tilt = pose
        return dict(
            elapsed_s=float(self.data.time - self.start_time),
            step=self.steps + 1,
            dx_mm=float(xy[0] * 1000),
            dy_mm=float(xy[1] * 1000),
            xy_mm=float(np.linalg.norm(xy) * 1000),
            tilt_deg=tilt,
            depth_mm=depth * 1000,
        )

    def record_held(self, pose):
        tip, axis, xy, depth, _ = pose
        head = tip + .045 * axis
        if self.pick and np.linalg.norm(self.data.site(self.grasp_id).xpos - head) < .025:
            error = float(np.linalg.norm(xy) * 1000)
            self.min_held_xy = min(self.min_held_xy, error)
            self.max_held_depth = max(self.max_held_depth, depth * 1000)
            if self.first_contact is not None:
                self.min_held_xy_after_contact = min(self.min_held_xy_after_contact, error)

    def sample_seating(self):
        super().sample_seating()
        pose = self.bolt_pose()
        if self.first_floor_contact is None:
            contacts = self.data.contact
            if len(contacts):
                bodies = self.model.geom_bodyid[contacts.geom]
                touching = (contacts.dist <= 0) & (bodies == self.bolt_id).any(axis=1)
                touching &= (bodies == self.target_id).any(axis=1)
                if touching.any():
                    floor = bool((contacts.geom[touching] == self.floor).any())
                    if self.first_contact is None:
                        self.first_contact = dict(**self.capture(pose), floor_contact=floor)
                    if floor:
                        self.first_floor_contact = self.capture(pose)
        self.record_held(pose)

    def spatially_seated(self, pose):
        tip, axis, xy, depth, _ = pose
        if depth <= 0 or axis[2] < MIN_AXIS_Z:
            return False
        gap = tip[2] - SHAFT_RADIUS * np.linalg.norm(axis[:2]) - self.floor_z
        if not -MAX_FLOOR_OVERRUN_M <= gap <= DEPTH_TOLERANCE_M:
            return False
        mouth = xy + depth * axis[:2] / axis[2]
        return bool(np.all(np.maximum(self.normals @ xy, self.normals @ mouth) <= self.faces))

    def step(self, action):
        if self.steps == 0:
            pose = self.bolt_pose()
            self.initial_pose = self.capture(pose)
            self.initial_pose["step"] = 0
            self.record_held(pose)
        outcome = super().step(action)
        pose = self.bolt_pose()
        self.final_pose = self.capture(pose)
        self.final_pose["step"] = self.steps
        self.record_held(pose)
        if outcome[-1]["slipped"]:
            self.slip_seated = self.spatially_seated(pose)
        return outcome

    def report(self):
        if not math.isfinite(self.min_held_xy) or not math.isfinite(self.max_held_depth):
            raise RuntimeError("A restored expanded state did not have any valid held sample")
        return dict(
            initial=self.initial_pose,
            first_base_contact=self.first_contact,
            first_floor_contact=self.first_floor_contact,
            min_held_xy_mm=self.min_held_xy,
            min_held_xy_from_first_contact_mm=(
                self.min_held_xy_after_contact if math.isfinite(self.min_held_xy_after_contact) else None
            ),
            max_held_depth_mm=self.max_held_depth,
            max_seated_seconds=float(self.max_streak * self.model.opt.timestep),
            slip_spatially_seated=self.slip_seated,
            final_pose=self.final_pose,
        )


def result(row):
    if row["success"]:
        return "success"
    if row["slipped"]:
        return "slip"
    if row["penetration_failure"]:
        return "penetration"
    if row["steps"] == MAX_STEPS:
        return "timeout"
    raise RuntimeError(f"Unexpected evaluation terminal state: {row}")


def bucket(row):
    if row["success"]:
        return None
    if row["max_seated_seconds"] > 0 or row["slip_spatially_seated"]:
        return "D"
    if row["max_held_depth_mm"] >= 15:
        return "C"
    return "B" if row["min_held_xy_mm"] <= 2 else "A"


def model_summary(rows):
    measures = Counter(row["result"] for row in rows)
    counts = Counter(row["failure_bucket"] for row in rows if row["failure_bucket"])
    failures = len(rows) - measures["success"]
    assert sum(counts.values()) == failures
    return dict(
        episodes=len(rows),
        results=dict(measures),
        failures=failures,
        failure_buckets={k: dict(count=counts[k], fraction=counts[k] / failures if failures else 0.) for k in "ABCD"},
        outcome_matches_original=sum(row["result"] == row["original_result"] for row in rows),
        exact_matches_original=sum(row["matches_original"] for row in rows),
    )


def write_csv(path, episodes):
    columns = ("policy", "index", "target_x_mm", "target_y_mm", "result", "steps", "original_result",
               "matches_original", "failure_bucket", "first_base_contact_xy_mm", "first_base_contact_dx_mm",
               "first_base_contact_dy_mm", "first_base_contact_tilt_deg", "first_base_contact_depth_mm",
               "first_base_contact_elapsed_s", "first_base_contact_floor", "first_floor_contact_xy_mm",
               "first_floor_contact_tilt_deg", "min_held_xy_mm", "min_held_xy_from_first_contact_mm",
               "max_held_depth_mm", "max_seated_seconds", "slip_spatially_seated", "initial_xy_mm")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in episodes:
            flat = {name: row.get(name) for name in columns}
            flat["target_x_mm"], flat["target_y_mm"] = row["target_xy_mm"]
            flat["initial_xy_mm"] = row["initial"]["xy_mm"]
            for name in ("first_base_contact", "first_floor_contact"):
                for key, value in (row[name] or {}).items():
                    col = f"{name}_{'floor' if key == 'floor_contact' else key}"
                    if col in flat:
                        flat[col] = value
            writer.writerow(flat)


def run(source, output):
    output.mkdir()
    starts = expanded_starts(source / "expanded-states")
    saved = torch.load(source / "best.pt", weights_only=True, map_location="cpu")
    if saved["update_step"] != 9000:
        raise ValueError("The selected Round 1 checkpoint is not the accepted 9,000-step Actor")
    config = saved["run"]
    env = TelemetryInsertion(penetration_limit_mm=3.)
    env.max_steps = MAX_STEPS
    try:
        features = FrozenSmolVLA(Path(config["smolvla"]))
        features.load_token(Path(config["token"]), Path(config["token_source"]), Path(config["smolvla"]))
        action_map = ActionMap(env, demonstration_limits(Path(config["training_dataset"])), config["sigma"])
        if action_map.report() != config["joints"] or env.model.opt.timestep != config["physics_dt"]:
            raise ValueError("The loaded evaluation configuration differs from the original")
        model_class, _ = load_rlinf(Path(config["rlinf"]))
        actor = model_class(**saved["model_config"]).cuda().eval().requires_grad_(False)
        actor.load_state_dict(saved["model"], strict=True)

        @torch.no_grad()
        def deterministic(obs):
            predicted, _, _ = actor.sac_forward(
                {k: v.unsqueeze(0) for k, v in obs.items()}, deterministic=True, apply_action_noise=False
            )
            return predicted.reshape_as(obs["ref_chunk"])

        data = {}
        with (output / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for name, filename, command in (
                ("smolvla", "expanded-smolvla.json", None), ("round1_actor", "expanded-best.json", deterministic)
            ):
                original = json.loads((source / filename).read_text(encoding="utf-8"))["rows"]
                if [r["index"] for r in original] != [start[0]["index"] for start in starts]:
                    raise ValueError("The expansion order differs from the original replay")
                rows = []
                for start, previous in zip(starts, original, strict=True):
                    observed = rollout(env, features, action_map, start, actor=command)
                    row = dict(**observed, **env.report(), policy=name, seed=start[0]["seed"],
                               target_xy_mm=start[0]["case"]["target"], original_result=result(previous),
                               original_steps=previous["steps"])
                    row["result"] = result(observed)
                    row["failure_bucket"] = bucket(row)
                    row["matches_original"] = row["result"] == row["original_result"] and row["steps"] == row["original_steps"]
                    rows.append(row)
                    print(json.dumps(row), file=log)
                    print("EVAL", name, row["index"], row["result"], row["failure_bucket"],
                          row["matches_original"], flush=True)
                data[name] = rows

        pairs = []
        for base, rl in zip(data["smolvla"], data["round1_actor"], strict=True):
            assert base["index"] == rl["index"]
            pairs.append(dict(
                index=base["index"], smolvla=base["result"], round1_actor=rl["result"],
                smolvla_failure_bucket=base["failure_bucket"], round1_failure_bucket=rl["failure_bucket"],
                original_smolvla=base["original_result"], original_round1_actor=rl["original_result"],
            ))
        final = dict(
            models={name: model_summary(rows) for name, rows in data.items()},
            replay_gains=[p for p in pairs if p["smolvla"] != "success" and p["round1_actor"] == "success"],
            original_gains=[p for p in pairs if p["original_smolvla"] != "success" and p["original_round1_actor"] == "success"],
            pairs=pairs,
        )
        write_csv(output / "episodes.csv", data["smolvla"] + data["round1_actor"])
        (output / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print("EVAL_COMPLETE", final["models"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
