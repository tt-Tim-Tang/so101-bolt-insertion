"""Paired target-only displacement test for the two π0.5 checkpoints and open-loop replay."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .online import Policy, diffusion_noise
from .scene import Insertion, TARGET


def make_cases():
    rng = np.random.default_rng(726905)
    cases = []
    for radius in (2, 5, 10, 15, 20):
        phase = rng.uniform(0, np.pi / 5)
        for pair in range(5):
            angle = phase + pair * np.pi / 5
            seed = int(rng.integers(1, 2**31))
            for side in (1, -1):
                dx, dy = radius * side * np.array([np.cos(angle), np.sin(angle)])
                cases.append(dict(radius=radius, pair=pair, side=side, dx_mm=dx, dy_mm=dy, seed=seed))
    for pair in range(5):
        cases.append(dict(radius=0, pair=pair, side=0, dx_mm=0, dy_mm=0, seed=int(rng.integers(1, 2**31))))
    return cases


@torch.no_grad()
def load_base(policy, base):
    from safetensors import safe_open

    params = policy.policy.state_dict()
    with safe_open(base / "model.safetensors", framework="pt") as weights:
        key_map = {"model." + key: key for key in weights.keys()}
        # The upstream π0.5 importer also initializes the token embeddings from the LM head.
        key_map["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = (
            "paligemma_with_expert.paligemma.lm_head.weight"
        )
        if params.keys() != key_map.keys():
            raise ValueError(f"Base tensor mismatch: {set(params) ^ set(key_map)}")
        for name, tensor in params.items():
            raw = weights.get_tensor(key_map[name])
            if tensor.shape != raw.shape:
                raise ValueError(f"Base shape mismatch at {name}")
            tensor.copy_(raw)
    print(f"Restored {len(params)} base tensors", flush=True)


def episode_zero(dataset):
    import pyarrow.parquet as pq

    frames = pq.read_table(sorted((dataset / "data").rglob("*.parquet")),
                           columns=["frame_index", "action"], filters=[("episode_index", "=", 0)])
    frames = frames.sort_by("frame_index")
    assert frames["frame_index"].to_pylist() == list(range(len(frames)))
    return np.array(frames["action"].to_pylist())


def move_target(env, offset):
    body = env.model.body("target")
    fixture = np.where(env.model.geom_bodyid == body.id)[0]
    old_positions = env.data.geom_xpos[fixture].copy()
    new_xy = np.array(TARGET) + offset
    delta = new_xy - body.pos[:2]
    body.pos[:2] = new_xy
    mujoco.mj_forward(env.model, env.data)
    np.testing.assert_allclose(env.data.geom_xpos[fixture, :2], old_positions[:, :2] + delta, atol=1e-9)


def run_case(env, case, policy, demonstration):
    env.reset(0, jitter=0)
    move_target(env, np.array([case["dx_mm"], case["dy_mm"]]) / 1000)
    body = env.model.body("target")
    if policy:
        policy.policy.reset()
        noise = torch.Generator(device="cuda").manual_seed(case["seed"])
    held_ticks = 0
    pickup = False
    approach = False
    contact = None
    trace = []
    while env.steps < env.max_steps:
        if policy:
            point = policy.decide(env.observation(), False, noise=diffusion_noise(policy.policy.config, noise))
            targets = policy.joint_targets(point.reference)
        else:
            targets = [demonstration[min(env.steps // 2, len(demonstration) - 1)]]
        for target in targets:
            _, success, timeout, info = env.step(target)
            lift, _, transferred, at_jaws = env.pickup_state()
            held_ticks = held_ticks + 1 if lift >= .025 and at_jaws else 0
            pickup |= held_ticks >= 20
            approach |= pickup and at_jaws and transferred and info["critical"]
            if contact is None:
                geoms = env.data.contact.geom
                if np.any(env.model.geom_bodyid[geoms] == body.id):
                    contact = env.steps
                else:
                    xy = env.data.body("bolt").xpos[:2] * 1000
                    trace.append([*np.round(xy, 4), bool(pickup and transferred and at_jaws)])
            if success or timeout:
                return dict(**case, success=success, pickup=bool(pickup), approach=bool(approach),
                            steps=env.steps, contact=contact, precontact_xy_mm=trace)


def check_controls(cases, dataset):
    from .demonstrate import demonstrate

    class TargetOnly(Insertion):
        def reset(self, seed, jitter):
            super().reset(0, jitter=0)
            move_target(self, self.offset)
            return self.info()

    # The expert reads simulator poses and runs at its original demonstration clock.
    expert = TargetOnly(max_steps=300, control_fps=25)
    replay = Insertion(max_steps=600, control_fps=50)
    demo = episode_zero(dataset)
    result = {"expert": [], "historical_replay": []}
    try:
        for case in cases:
            expert.offset = np.array([case["dx_mm"], case["dy_mm"]]) / 1000
            info = demonstrate(expert, 0, jitter=0)
            result["expert"].append(dict(**case, success=info["success"]))
        for seed in range(10000, 10030):
            replay.reset(seed, jitter=.001)
            while True:
                _, success, timeout, _ = replay.step(demo[min(replay.steps // 2, len(demo) - 1)])
                if success or timeout:
                    break
            result["historical_replay"].append(dict(seed=seed, success=success, steps=replay.steps))
    finally:
        expert.close()
        replay.close()
    return result


def visual_probe(policy, cases, dataset, output):
    """Vary only the rendered socket; apply each chunk to identical nominal physics."""
    from PIL import Image, ImageDraw

    env = Insertion(max_steps=600, control_fps=50)
    demo = episode_zero(dataset)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(env.model, spec))
    result = []
    try:
        for tick in (230, 260):
            env.reset(0, jitter=0)
            for step in range(tick):
                env.step(demo[step // 2])
            mujoco.mj_getState(env.model, env.data, state, spec)
            initial = env.data.body("bolt").xpos.copy()
            target_body = env.model.body("target").id
            assert env.pickup_state()[-1] and not np.any(env.model.geom_bodyid[env.data.contact.geom] == target_body)
            def restore():
                env.model.body("target").pos[:2] = TARGET
                mujoco.mj_setState(env.model, env.data, state, spec)
                mujoco.mj_forward(env.model, env.data)
                env.steps = tick

            views = {}
            for case in cases:
                restore()
                move_target(env, np.array([case["dx_mm"], case["dy_mm"]]) / 1000)
                obs = env.observation()
                if tick == 260 and case["pair"] == 0 and case["radius"] in (0, 20):
                    views[case["side"]] = obs
                restore()
                gen = torch.Generator(device="cuda").manual_seed(case["seed"])
                policy.policy.reset()
                point = policy.decide(obs, False, noise=diffusion_noise(policy.policy.config, gen))
                actions = policy.joint_targets(point.reference)
                for action in actions:
                    env.step(action)
                result.append(dict(**case, tick=tick, initial_mm=(initial * 1000).tolist(),
                                   endpoint_mm=(env.data.body("bolt").xpos * 1000).tolist(),
                                   actions=actions.tolist()))
            if views:
                panel = Image.new("RGB", (672, 470), "white")
                labels = ImageDraw.Draw(panel)
                for col, side in enumerate((-1, 0, 1)):
                    labels.text((224*col+8, 4), f"{20*side:+d} mm", fill="black")
                    for row, name in enumerate(("top", "wrist")):
                        key = f"observation.images.{name}"
                        panel.paste(Image.fromarray(views[side][key]), (224*col, 22+224*row))
                assert np.any(views[-1]["observation.images.top"] != views[1]["observation.images.top"])
                panel.save(output / "camera-check.png")
    finally:
        env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manifest", "base", "sft", "replay", "controls", "visual"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    manifest = args.out / "cases.json"
    if args.mode == "manifest":
        args.out.mkdir(parents=True, exist_ok=True)
        with manifest.open("x") as stream:
            json.dump(make_cases(), stream, indent=2)
        return
    cases = json.loads(manifest.read_text())
    if args.mode == "controls":
        with (args.out / "controls.json").open("x") as stream:
            json.dump(check_controls(cases, args.dataset), stream)
        return
    demo = policy = None
    if args.mode == "replay":
        demo = episode_zero(args.dataset)
    else:
        policy = Policy(args.checkpoint, None, 0, "cuda", args.tokenizer)
        if args.mode == "base":
            load_base(policy, args.base)
        if args.mode == "visual":
            with (args.out / "visual.json").open("x") as stream:
                json.dump(visual_probe(policy, cases, args.dataset, args.out), stream)
            return
    env = Insertion(max_steps=600, control_fps=50)
    try:
        with (args.out / f"{args.mode}.jsonl").open("x") as stream:
            for case in cases:
                row = run_case(env, case, policy, demo)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(case["radius"], case["pair"], case["side"], int(row["success"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
