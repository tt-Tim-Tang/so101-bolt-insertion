"""Collect and verify 25 Hz SO-101 insertion demonstrations."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from .scene import IMAGE_SIZE, JOINTS
from .wide_env import WideInsertion, expert


FPS = 25
BORE_RADIUS = .0057
TARGET_RANGE_MM = 18
REPO_ID = "local/so101_smolvla25"
CAMERAS = ("observation.images.top", "observation.images.wrist")
EVAL_TRAIN_INDICES = [10 * x + y for x, y in product((0, 3, 6, 9), repeat=2)]
CORNER_INDICES = {0, 9, 90, 99}


class SmolVLAInsertion(WideInsertion):
    def __init__(self):
        super().__init__(target_bore_radius=BORE_RADIUS, control_fps=FPS)

    def start(self, case):
        super().start(case)
        self.opened_early = self.slipped = False
        self.drop_ticks = 0
        self.highest_lift = 0.

    def step(self, action):
        reward, done, timeout, info = super().step(action)
        lift, _, _, jaws = self.pickup_state()
        self.highest_lift = max(self.highest_lift, lift)
        self.opened_early |= self.pick and not done and self.state()[-1] > .15
        losing_grasp = self.pick and not done and not jaws and self.highest_lift - lift > .01
        self.drop_ticks = self.drop_ticks + 1 if losing_grasp else 0
        self.slipped |= self.drop_ticks >= 5
        info.update(opened_early=bool(self.opened_early), slipped=bool(self.slipped))
        return reward, done, timeout, info


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
    elapsed = env.data.time - .05  # start() settles for 25 physics steps of 2 ms.
    finite = bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all())
    clock_ok = bool(np.isclose(elapsed, env.steps / FPS, atol=1e-9))
    return dict(case=case, steps=env.steps, warnings=warnings, finite=finite, clock_ok=clock_ok, **result)


def freeze_cases(root):
    rng = np.random.default_rng(519307)
    edges = np.linspace(-TARGET_RANGE_MM, TARGET_RANGE_MM, 11)
    env = SmolVLAInsertion()
    if not np.isclose(env.model.opt.timestep * env.substeps, 1 / FPS):
        raise RuntimeError("Control interval does not match the dataset FPS")
    cases = []
    try:
        with (root / "expert-check.jsonl").open("x", buffering=1) as log:
            for x, y in product(range(10), repeat=2):
                lower = edges[[x, y]]
                upper = edges[[x + 1, y + 1]]
                positions = [(lower + upper) / 2, *rng.uniform(lower, upper, (12, 2))]
                for target in positions:
                    case = make_case(target, rng)
                    result = trial(env, case)
                    print(json.dumps(dict(cell=[x, y], **result)), file=log)
                    if result["success"] and result["finite"] and result["clock_ok"] and not result["warnings"]:
                        cases.append(case)
                        break
                else:
                    raise RuntimeError(f"No clean insertion in target cell {(x, y)}")
    finally:
        env.close()
    validation = [make_case(xy, rng) for xy in rng.uniform(-TARGET_RANGE_MM, TARGET_RANGE_MM, (10, 2))]
    doc = dict(train=cases, validation=validation, eval_train_indices=EVAL_TRAIN_INDICES,
               fps=FPS, physics_dt=.002, clearance_mm=.7, target_range_mm=TARGET_RANGE_MM)
    (root / "splits.json").write_text(json.dumps(doc, indent=2))
    return doc


def collect(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root.mkdir(parents=True, exist_ok=False)
    splits = freeze_cases(root)
    features = {key: dict(dtype="float32", shape=(6,), names=list(JOINTS))
                for key in ("observation.state", "action")}
    features.update({key: dict(dtype="video", shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                               names=["height", "width", "channel"]) for key in CAMERAS})
    dataset = LeRobotDataset.create(repo_id=REPO_ID, root=root / "dataset", fps=FPS,
                                    robot_type="so101_mujoco", features=features, use_videos=True,
                                    vcodec="libx264", video_backend="pyav")
    env = SmolVLAInsertion()
    lengths = []
    try:
        with (root / "collection.jsonl").open("x", buffering=1) as log:
            for index, case in enumerate(splits["train"]):
                frames = []

                def record(action):
                    frames.append(dict(**env.observation(), action=action.copy()))

                result = trial(env, case, record)
                print(json.dumps(dict(index=index, frames=len(frames), **result)), file=log)
                if (not result["success"] or not result["clock_ok"] or not result["finite"]
                        or result["warnings"] or not 0 < len(frames) <= env.max_steps):
                    raise RuntimeError(f"Demonstration {index} failed the recorded replay check")
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                lengths.append(len(frames))
                print(f"Collected {index + 1}/100: {len(frames)} frames", flush=True)
    finally:
        dataset.finalize()
        env.close()
    summary = dict(episodes=len(lengths), frames=sum(lengths), fps=FPS, warnings=0,
                   min_frames=min(lengths), median_frames=float(np.median(lengths)), max_frames=max(lengths))
    (root / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    collect(parser.parse_args().out)


if __name__ == "__main__":
    main()
