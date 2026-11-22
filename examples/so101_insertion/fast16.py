"""Small new-camera verification with a frozen 16/10 split and continuous insertion."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from .wide_env import WideInsertion, expert
from .scene import IMAGE_SIZE, JOINTS


class FastMetrics(WideInsertion):
    def start(self, case):
        super().start(case)
        self.closure_xy_mm = None

    def step(self, action):
        reward, done, timeout, info = super().step(action)
        if self.closure_xy_mm is None and self.state()[-1] < .15:
            bolt = self.data.body("bolt")
            head = bolt.xpos + .045 * bolt.xmat.reshape(3, 3)[:, 2]
            self.closure_xy_mm = float(1000*np.linalg.norm(self.data.site("grasp").xpos[:2]-head[:2]))
        info["closure_xy_mm"] = self.closure_xy_mm
        return reward, done, timeout, info


def collect(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cases = json.loads((root/"splits.json").read_text())
    settings = json.loads((root/"environment.json").read_text())
    features = {k: dict(dtype="float32", shape=(6,), names=list(JOINTS))
                for k in ("observation.state", "action")}
    for camera in ("top", "wrist"):
        features["observation.images."+camera] = dict(dtype="image", shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                                      names=["height","width","channel"])
    ds = LeRobotDataset.create(repo_id="local/so101_fast16", fps=50, root=root/"dataset", features=features,
                              robot_type="so101_mujoco", use_videos=False, video_backend="pyav")
    env = FastMetrics(target_bore_radius=.005+settings["clearance_mm"]/1000)
    first = None
    rows = []
    try:
        with (root/"collection.jsonl").open("x", buffering=1) as log:
            for i, case in enumerate(cases["train"]):
                frames, counts = [], {}
                def record(action):
                    frames.append(dict(**env.observation(), action=action.copy()))
                    counts[env.phase] = counts.get(env.phase, 0)+1
                result = expert(env, case, record=record)
                row = dict(index=i, case=case, frames=len(frames), phase_frames=counts, **result)
                print(json.dumps(row), file=log)
                if not result["success"]:
                    raise RuntimeError(f"Expert failed on fixed training case {i}; dataset incomplete")
                for frame in frames:
                    ds.add_frame(frame.copy())
                ds.save_episode()
                if first is None:
                    first = [f["action"] for f in frames]
                rows.append(row)
                print("COLLECTED", i, len(frames), flush=True)
        with (root/"fixed-validation.jsonl").open("x", buffering=1) as log:
            for i, case in enumerate(cases["validation"]):
                env.start(case)
                while True:
                    _, done, timeout, info = env.step(first[min(env.steps, len(first)-1)])
                    if done or timeout:
                        break
                print(json.dumps(dict(index=i, case=case, steps=env.steps, **info)), file=log)
        total = sum(r["frames"] for r in rows)
        near = sum(sum(n for phase,n in r["phase_frames"].items()
                       if phase in ("air_align","insert","hold")) for r in rows)
        summary = dict(episodes=len(rows), frames=total, mean_frames=total/len(rows), hole_phase_fraction=near/total)
        (root/"dataset-summary.json").write_text(json.dumps(summary, indent=2))
        print("DATASET", summary, flush=True)
    finally:
        ds.finalize()
        env.close()


def make_cases():
    rng = np.random.default_rng(916182)
    def case(target):
        return dict(source=[0, 0], target=list(target), joints=[0]*5, noise=int(rng.integers(2**31)))
    return dict(train=[case(xy) for xy in product((-13.5, -4.5, 4.5, 13.5), repeat=2)],
                validation=[case(xy) for xy in rng.uniform(-18, 18, size=(10, 2))])


def preflight(root, clearance):
    root.mkdir(parents=True, exist_ok=False)
    cases = make_cases()
    (root/"splits.json").write_text(json.dumps(cases, indent=2))
    env = WideInsertion(target_bore_radius=.005 + clearance/1000)
    (root/"environment.json").write_text(json.dumps(dict(clearance_mm=clearance,
        source_jitter_mm=0,target_range_mm=18,control_hz=50,stable_ticks=25,
        insertion_tick_limit=100,descent_ticks=65,expert="held_bolt_pose_ik_continuous_xy"),indent=2))
    try:
        with (root/"expert-check.jsonl").open("x", buffering=1) as log:
            for group, group_cases in cases.items():
                for i, case in enumerate(group_cases):
                    try:
                        counts = {}
                        def record(action):
                            counts[env.phase] = counts.get(env.phase, 0) + 1
                        result = expert(env, case, record=record)
                    except RuntimeError as exc:
                        result = dict(success=False, pickup=env.pick, arrival=env.arrive, error=str(exc))
                    print(json.dumps(dict(group=group,index=i,steps=env.steps,case=case,phase_frames=counts,**result)),file=log)
                    print(group, i, env.steps, result.get("pickup"), result.get("arrival"), result["success"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clearance",type=float,default=.7)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.out,args.clearance)
    else:
        collect(args.out)
