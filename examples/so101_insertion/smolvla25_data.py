"""Collect and verify 25 Hz SO-101 insertion demonstrations."""

import argparse
import json
from itertools import product
from math import ceil, cos, radians
from pathlib import Path

import mujoco
import numpy as np

from .scene import IMAGE_SIZE, JOINTS, MOUTH_Z, SHAFT_RADIUS
from .wide_env import WideInsertion, expert


FPS = 25
BORE_RADIUS = .0057
TARGET_RANGE_MM = 20
MAX_PENETRATION_M = .001
DEPTH_TOLERANCE_M = .002
MIN_AXIS_Z = cos(radians(12))
REPO_ID = "local/so101_smolvla25"
CAMERAS = ("observation.images.top", "observation.images.wrist")
EVAL_TRAIN_INDICES = [10 * x + y for x, y in product((0, 3, 6, 9), repeat=2)]
CORNER_INDICES = {0, 9, 90, 99}


class SmolVLAInsertion(WideInsertion):
    def __init__(self):
        super().__init__(target_bore_radius=BORE_RADIUS, control_fps=FPS)
        self.success_steps = ceil(.2 / self.model.opt.timestep)
        self.bolt_id = self.model.body("bolt").id
        self.target_id = self.model.body("target").id
        if not np.isclose(self.substeps * self.model.opt.timestep, 1 / FPS):
            raise RuntimeError("Control interval does not match the dataset FPS")

    def start(self, case):
        super().start(case)
        self.start_time = self.data.time
        self.opened_early = self.slipped = False
        self.open_count = self.slip_count = 0
        self.was_open = False
        self.drop_ticks = 0
        self.highest_lift = 0.

    def sample_seating(self):
        bolt = self.data.body(self.bolt_id)
        tip = bolt.xpos
        axis = bolt.xmat.reshape(3, 3)[:, 2]
        xy = tip[:2] - self.model.body(self.target_id).pos[:2]
        inside = False
        if tip[2] < MOUTH_Z and axis[2] >= MIN_AXIS_Z:
            gap = tip[2] - SHAFT_RADIUS * np.linalg.norm(axis[:2]) - self.floor_z
            if -MAX_PENETRATION_M <= gap <= DEPTH_TOLERANCE_M:
                at_mouth = xy + (MOUTH_Z - tip[2]) * axis[:2] / axis[2]
                inside = np.all(np.maximum(self.normals @ xy, self.normals @ at_mouth) <= self.faces)
        self.seated_ticks = self.seated_ticks + 1 if inside else 0
        self.max_streak = max(self.max_streak, self.seated_ticks)

        contacts = self.data.contact
        if len(contacts):
            bodies = self.model.geom_bodyid[contacts.geom]
            mask = (bodies == self.bolt_id).any(axis=1) & (bodies == self.target_id).any(axis=1)
            if mask.any():
                self.max_penetration = max(self.max_penetration, float(-contacts.dist[mask].min()))

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(JOINTS),) or not np.isfinite(action).all():
            raise ValueError("Expected six finite absolute joint targets in radians")
        self.data.ctrl[:] = np.clip(action, self.low, self.high)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            # mj_step exposes the evaluated contact and pose of this physics interval.
            self.sample_seating()
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        info = self.info()
        lift, _, _, jaws = self.pickup_state()
        self.held_ticks = self.held_ticks + 1 if lift >= .025 and jaws else 0
        self.pick |= self.held_ticks >= ceil(.4 * FPS)
        approach = self.pick and jaws and 0 <= -info["insertion_depth_m"] <= .010 and info["tilt_rad"] <= .15
        self.arrive |= approach and info["radial_error_m"] <= .005
        penetration = self.max_penetration > MAX_PENETRATION_M
        success = self.max_streak >= self.success_steps and not penetration
        self.highest_lift = max(self.highest_lift, lift)
        opened = bool(self.pick and not success and self.data.ctrl[-1] > .15)
        self.open_count += int(opened and not self.was_open)
        self.was_open = opened
        losing_grasp = self.pick and not success and not opened and not jaws and self.highest_lift - lift > .01
        self.drop_ticks = self.drop_ticks + 1 if losing_grasp else 0
        self.slip_count += int(self.drop_ticks == ceil(.2 * FPS))
        info.update(success=bool(success), pickup=bool(self.pick), arrival=bool(self.arrive), phase=self.phase,
                    opened_early=self.open_count, slipped=self.slip_count,
                    penetration_failure=penetration, max_penetration_mm=self.max_penetration * 1000,
                    stable_s=self.max_streak * self.model.opt.timestep,
                    reason="penetration" if penetration else None)
        return float(success), success or penetration, self.steps >= self.max_steps, info


def make_case(target, rng):
    return dict(target=list(target), source=[0, 0], joints=[0] * 5, noise=int(rng.integers(2**31)))


def trial(env, case, record=None):
    try:
        result = expert(env, case, record=record)
    except RuntimeError as exc:
        if not str(exc).startswith(("Held-bolt IK did not converge", "Unreachable grasp waypoint")):
            raise
        result = dict(success=False, reason=str(exc), phase=env.phase)
    warnings = {i: int(w.number) for i, w in enumerate(env.data.warning) if w.number}
    elapsed = env.data.time - env.start_time
    finite = bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all())
    clock_ok = bool(np.isclose(elapsed, env.steps / FPS, atol=1e-9))
    return dict(case=case, steps=env.steps, warnings=warnings, finite=finite, clock_ok=clock_ok, **result)


def gate(root):
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(519307)
    env = SmolVLAInsertion()
    rows = []
    try:
        corners = product((-TARGET_RANGE_MM, TARGET_RANGE_MM), repeat=2)
        for target in ((0, 0), *corners):
            actions = []
            case = make_case(target, rng)

            def record(action):
                actions.append(np.clip(action, env.low, env.high))

            live = trial(env, case, record)
            env.start(case)
            replay = dict(success=False, reason="empty recording")
            for action in actions:
                _, done, timeout, replay = env.step(action)
                if done or timeout:
                    break
            rows.append(dict(target_mm=target, sampled_frames=len(actions), live=live, replay=replay))
    finally:
        env.close()
    live_success = sum(bool(r["live"]["success"] and r["live"]["clock_ok"] and r["live"]["finite"]) for r in rows)
    replay_success = sum(bool(r["replay"]["success"]) for r in rows)
    result = dict(live_success=live_success, replay_success=replay_success, required=4, episodes=len(rows),
                  fps=FPS, physics_dt=env.model.opt.timestep, success_physics_steps=env.success_steps, rows=rows)
    (root / "gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)
    return live_success >= 4 and replay_success >= 4


def collect(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    accepted = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if min(accepted["live_success"], accepted["replay_success"]) < accepted["required"]:
        raise RuntimeError("25 Hz expert gate failed")
    rng = np.random.default_rng(519307)
    grid = np.linspace(-TARGET_RANGE_MM, TARGET_RANGE_MM, 10)
    edges = np.r_[grid[0], (grid[1:] + grid[:-1]) / 2, grid[-1]]
    features = {key: dict(dtype="float32", shape=(6,), names=list(JOINTS))
                for key in ("observation.state", "action")}
    features.update({key: dict(dtype="video", shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                               names=["height", "width", "channel"]) for key in CAMERAS})
    dataset = LeRobotDataset.create(repo_id=REPO_ID, root=root / "dataset", fps=FPS,
                                    robot_type="so101_mujoco", features=features, use_videos=True,
                                    vcodec="libx264", video_backend="pyav")
    env = SmolVLAInsertion()
    cases = []
    lengths = []
    static_frames = comparisons = 0
    try:
        with (root / "collection.jsonl").open("x", buffering=1) as log:
            for x, y in product(range(10), repeat=2):
                lower, upper = edges[[x, y]], edges[[x + 1, y + 1]]
                for target in ([grid[x], grid[y]], *rng.uniform(lower, upper, size=(49, 2))):
                    frames = []
                    case = make_case(target, rng)

                    def record(action):
                        frames.append(dict(**env.observation(), action=np.clip(action, env.low, env.high).astype(np.float32)))

                    result = trial(env, case, record)
                    print(json.dumps(dict(cell=[x, y], frames=len(frames), **result)), file=log)
                    if not (result["success"] and result["clock_ok"] and result["finite"]):
                        continue
                    states = np.array([frame["observation.state"] for frame in frames])
                    static_frames += int(np.count_nonzero(np.max(np.abs(np.diff(states, axis=0)), axis=1) < .001))
                    comparisons += len(states) - 1
                    for frame in frames:
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    cases.append(case)
                    lengths.append(len(frames))
                    break
                else:
                    raise RuntimeError(f"No successful 25 Hz demonstration in grid cell {(x, y)} after 50 attempts")
    finally:
        dataset.finalize()
        env.close()
    validation = [make_case(xy, rng) for xy in rng.uniform(-TARGET_RANGE_MM, TARGET_RANGE_MM, size=(10, 2))]
    splits = dict(train=cases, validation=validation, eval_train_indices=EVAL_TRAIN_INDICES,
                  fps=FPS, physics_dt=env.model.opt.timestep, clearance_mm=.7, target_range_mm=TARGET_RANGE_MM)
    (root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = dict(episodes=len(lengths), mean_frames=sum(lengths) / len(lengths), static_fraction=static_frames / comparisons)
    (root / "dataset-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("gate", "collect"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.stage == "gate":
        if not gate(args.out):
            raise SystemExit(1)
    else:
        collect(args.out)


if __name__ == "__main__":
    main()
