"""Collect physical insertion starts from the frozen SFT policy and trace its control path."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .online import CHUNK, Policy
from .scene import Insertion


STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def physics(env):
    values = np.empty(mujoco.mj_stateSize(env.model, STATE))
    mujoco.mj_getState(env.model, env.data, values, STATE)
    return values


def restore(env, start):
    env.reset(int(start["seed"]), jitter=.001)
    mujoco.mj_setState(env.model, env.data, start["physics"], STATE)
    mujoco.mj_forward(env.model, env.data)
    env.steps = 0


def snapshot(env, seed, action):
    start = dict(seed=seed, step=env.steps, physics=physics(env),
                 policy_rng=torch.cuda.get_rng_state().numpy())
    env.step(action)
    expected = physics(env)
    restore(env, start)
    env.step(action)
    np.testing.assert_allclose(physics(env), expected, atol=1e-10, rtol=0)
    return start


@torch.no_grad()
def collect(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    policy = Policy(args.pi05, None, 0, "cuda", args.tokenizer, CHUNK)
    env = Insertion(max_steps=1000, control_fps=50)
    try:
        trace = []
        draw = policy.policy.model.sample_noise
        def record_noise(shape, device):
            noise = draw(shape, device)
            trace.append({"noise": noise.detach().cpu()})
            return noise
        policy.policy.model.sample_noise = record_noise
        torch.manual_seed(10000)
        env.reset(10000)
        while env.steps < env.max_steps:
            command = policy.decide(env.observation(), False)
            targets = policy.joint_targets(command.reference)
            trace[-1].update(targets=torch.from_numpy(targets.copy()), state=torch.from_numpy(env.state()))
            for target in targets:
                _, win, timeout, _ = env.step(target)
                if win or timeout:
                    break
            if win or timeout:
                break
        torch.save(dict(trace=trace, steps=env.steps, success=win), out/"reference.pt")
        print("reference", 10000, env.steps, win, flush=True)
        policy.policy.model.sample_noise = draw

        manifest = {}
        for split, first, count in [("train", 3000, args.train), ("test", 6000, args.test)]:
            folder = out/split
            folder.mkdir()
            manifest[split] = {}
            for seed in range(first, first+count):
                torch.manual_seed(seed)
                env.reset(seed)
                while env.steps < env.max_steps:
                    if env.info()["critical"] and env.pickup_state()[-1]:
                        start = snapshot(env, seed, env.data.ctrl.copy())
                        np.savez(folder/f"{seed}.npz", **start)
                        manifest[split][seed] = int(start["step"])
                        print(split, seed, int(start["step"]), flush=True)
                        break
                    decision = policy.decide(env.observation(), False)
                    for target in policy.joint_targets(decision.reference):
                        _, win, timeout, _ = env.step(target)
                        if win or timeout:
                            break
                    if win or timeout:
                        manifest[split][seed] = None
                        print(split, seed, "no held critical start", flush=True)
                        break
        (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--test", type=int, default=30)
    collect(parser.parse_args())
