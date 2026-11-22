"""Full-task clearance sweep with paired hole positions and five blind controls."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch

from .demonstrate import demonstrate
from .hole_shift_eval import move_target
from .online import Policy, diffusion_noise
from .scene import BORE_RADIUS, HOME, MOUTH_Z, SHAFT_RADIUS, SOURCE, TARGET, Insertion


def make_cases(start_seed, pairs):
    cases = []
    for seed in range(start_seed, start_seed + pairs):
        rng = np.random.default_rng(seed)
        source, hole = rng.uniform(-1, 1, size=(2, 2))
        noise = int(rng.integers(1, 2**31))
        for side in (1, -1):
            cases.append(dict(seed=seed, side=side, source_mm=source.tolist(),
                              hole_mm=(hole * side).tolist(), noise=noise))
    return cases


class PairedInsertion(Insertion):
    def reset(self, seed=0, jitter=.001):
        super().reset(self.case["seed"], jitter=.001)
        np.testing.assert_allclose(self.model.body("source").pos[:2],
                                   np.array(SOURCE) + np.array(self.case["source_mm"]) / 1000)
        move_target(self, np.array(self.case["hole_mm"]) / 1000)
        return self.info()


def check_geometry(env, clearance):
    for name, expected in (("source", BORE_RADIUS), ("target", SHAFT_RADIUS + clearance / 1000)):
        ids = np.flatnonzero(env.model.geom_bodyid == env.model.body(name).id)[1:-1]
        assert len(ids) == 16
        radii = np.linalg.norm(env.model.geom_pos[ids, :2], axis=1) - env.model.geom_size[ids, 0]
        np.testing.assert_allclose(radii, expected, atol=1e-10)
    env.reset()
    i = env.bolt_address
    env.data.qpos[i:i+7] = [*env.model.body("target").pos[:2], MOUTH_Z+.0005, 1, 0, 0, 0]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(50):
        if env.step(HOME)[1]:
            return
    raise AssertionError("Centered vertical bolt failed insertion")


def read_demos(dataset):
    table = pq.read_table(sorted((dataset / "data").rglob("*.parquet")), columns=["episode_index", "frame_index", "action"],
                          filters=[("episode_index", "in", list(range(5)))]).sort_by(
                              [("episode_index", "ascending"), ("frame_index", "ascending")])
    demos = []
    for episode in range(5):
        rows = table.filter(np.array(table["episode_index"]) == episode)
        assert len(rows) and rows["frame_index"].to_pylist() == list(range(len(rows)))
        demos.append(np.repeat(rows["action"].to_pylist(), 2, axis=0))
    return demos


def run(env, case, *, policy=None, actions=None):
    env.case = case
    env.reset()
    issued = []
    if policy:
        policy.policy.reset()
        gen = torch.Generator(device="cuda").manual_seed(case["noise"])
    while True:
        if policy:
            noise = diffusion_noise(policy.policy.config, gen)
            decision = policy.decide(env.observation(), False, noise=noise)
            chunk = policy.joint_targets(decision.reference)
        else:
            chunk = [actions[min(env.steps, len(actions)-1)]]
        for action in chunk:
            if policy:
                issued.append(action.copy())
            _, success, timeout, _ = env.step(action)
            if success or timeout:
                return dict(success=success, steps=env.steps), issued


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manifest", "sft", "blind", "expert"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--clearances", nargs="+", type=float, default=[1.5, 1., .5])
    parser.add_argument("--split", choices=("validation", "holdout"), default="validation")
    args = parser.parse_args()
    manifest = args.out / "cases.json"
    if args.mode == "manifest":
        args.out.mkdir(exist_ok=True, parents=True)
        with manifest.open("x") as stream:
            json.dump(dict(validation=make_cases(879031, 15), holdout=make_cases(925731, 25)), stream, indent=2)
        return
    cases = json.loads(manifest.read_text())[args.split]
    policy = Policy(args.checkpoint, None, 0, "cuda", args.tokenizer) if args.mode == "sft" else None
    demos = read_demos(args.dataset) if args.mode == "blind" else None
    for clearance in args.clearances:
        hz = 25 if args.mode == "expert" else 50
        env = PairedInsertion(max_steps=12*hz, control_fps=hz,
                              target_bore_radius=round(SHAFT_RADIUS + clearance / 1000, 7))
        try:
            env.case = cases[0]
            check_geometry(env, clearance)
            with (args.out / f"{args.split}-{args.mode}-{clearance:g}.jsonl").open("x", buffering=1) as log:
                for i, case in enumerate(cases):
                    env.case = case
                    if args.mode == "expert":
                        info = demonstrate(env, case["seed"], jitter=.001)
                        results = dict(success=info["success"], steps=env.steps)
                    elif args.mode == "blind":
                        results = dict(replays=[run(env, case, actions=a)[0] for a in demos])
                    else:
                        results, actions = run(env, case, policy=policy)
                        # Same source and robot, opposite hole; verify exact self-replay first.
                        direct, _ = run(env, case, actions=actions)
                        if direct != results:
                            raise AssertionError(f"Fixed command replay diverged: {direct} != {results}")
                        mirror = cases[i ^ 1]
                        assert case["seed"] == mirror["seed"] and case["side"] == -mirror["side"]
                        results["opposite_hole"], _ = run(env, mirror, actions=actions)
                    print(json.dumps(dict(**case, clearance_mm=clearance, **results)), file=log)
                print(args.mode, clearance, len(cases), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
