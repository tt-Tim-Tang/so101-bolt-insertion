"""Zero-update exploration comparison on the two archived v3 training failures."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .smolvla25_data import SmolVLAInsertion
from .smolvla_motion_audit import write_json
from .smolvla_rlt_stage2 import CHUNK, ActionMap, FrozenSmolVLA
from .smolvla_rlt_td3 import load_rlinf
from .smolvla_rlt_v3 import EPS_RAD, HANDOFF_LIMIT, SIGMA, physical_commands, read_cases, read_snapshot, restore_handoff


MODES = ("iid", "block")
INDICES = (60, 66)
TRIALS = 30


def bounded_action(reference, standard_noise, eps, sigma):
    return (reference + (standard_noise * sigma).clamp(-eps, eps)).clamp(-1., 1.)


def batched_reference(features, envs, flow):
    batches = [features.preprocess_observation(env.observation()) for env in envs]
    batch = {key: torch.cat([b[key] for b in batches]) for key in batches[0] if torch.is_tensor(batches[0][key])}
    policy = features.policy
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    noise = torch.cat([torch.randn(shape, device="cuda", generator=g, dtype=torch.float32) for g in flow])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actions = policy.predict_action_chunk(batch, noise=noise, correct_padding_velocity=True)[:, :CHUNK]
    return features.post(actions.float())


def calibrate(env, features, mapping, actor, starts, eps):
    baseline = {}

    def tips(snapshot, physical, action, reference):
        restore_handoff(env, features, snapshot)
        env.max_steps = env.steps + HANDOFF_LIMIT
        points = []
        for command in physical_commands(env, mapping, physical, action, reference):
            env.step(command)
            points.append(env.data.body(env.bolt_id).xpos[:2].copy())
        return np.stack(points)

    for index, snapshot in starts.items():
        restore_handoff(env, features, snapshot)
        obs = features.observe(env, mapping, with_token=False, include_physical_reference=True)
        ref = obs["ref_chunk"]
        raw = obs["reference_rad"].cpu().numpy().astype(np.float64)
        z = torch.zeros(1, actor.z_dim, device="cuda")
        proprio = torch.zeros(1, actor.proprio_dim, device="cuda")
        check = actor.sac_forward(dict(z_rl=z, proprio=proprio, ref_chunk=ref.unsqueeze(0)),
                                  deterministic=True, apply_action_noise=False)[0].reshape_as(ref)
        if not torch.equal(ref, check):
            raise RuntimeError("Exploration requires the zero Actor")
        baseline[index] = raw, ref, tips(snapshot, raw, ref, ref)

    generator = torch.Generator(device="cuda").manual_seed(20261201)
    draws = {mode: torch.randn((64, CHUNK if mode == "iid" else 1, len(EPS_RAD)), device="cuda", generator=generator)
             for mode in MODES}

    def measure(mode, sigma):
        result = {}
        for index, snapshot in starts.items():
            raw, ref, no_noise = baseline[index]
            noisy = bounded_action(ref, draws[mode], eps, sigma)
            result[str(index)] = [float(np.linalg.norm(tips(snapshot, raw, action, ref) - no_noise, axis=1).max() * 1000)
                                  for action in noisy]
        values = [value for trial in result.values() for value in trial]
        return dict(sigma=sigma, median_mm=float(np.median(values)), p95_mm=float(np.quantile(values, .95)), by_hole=result)

    iid = measure("iid", SIGMA)
    block = measure("block", SIGMA)
    if not .6 <= block["median_mm"] <= 1.:
        lo, hi = 0., .004
        for _ in range(7):
            block = measure("block", (lo + hi) / 2)
            if .72 <= block["median_mm"] <= .88:
                break
            if block["median_mm"] < .8:
                lo = block["sigma"]
            else:
                hi = block["sigma"]
    if not .6 <= block["median_mm"] <= 1.:
        raise RuntimeError("The block exploration did not meet the measured lateral amplitude band")
    return dict(iid=iid, block=block)


def wave(envs, features, mapping, starts, tasks, eps, scales):
    active = []
    for env, (index, trial, mode) in zip(envs, tasks, strict=True):
        snapshot = starts[index]
        restore_handoff(env, features, snapshot)
        env.max_steps = env.steps + HANDOFF_LIMIT
        flow = torch.Generator(device="cuda").set_state(snapshot[3])
        noise_seed = 719473 + trial + 100 * index
        explore = torch.Generator(device="cuda").manual_seed(noise_seed)
        active.append((env, index, trial, mode, flow, explore, env.steps))
    results = []
    while active:
        raw = batched_reference(features, [r[0] for r in active], [r[4] for r in active])
        reference = mapping.normalize(raw)
        next_group = []
        for i, entry in enumerate(active):
            env, index, trial, mode, flow, explore, start = entry
            n = CHUNK if mode == "iid" else 1
            draw = torch.randn((n, len(EPS_RAD)), device="cuda", generator=explore)
            action = bounded_action(reference[i], draw, eps, scales[mode]["sigma"])
            commands = physical_commands(env, mapping, raw[i].cpu().numpy().astype(np.float64), action, reference[i])
            for command in commands:
                _, stopped, timeout, info = env.step(command)
                if stopped or timeout or info["slipped"]:
                    results.append(dict(index=index, trial=trial, mode=mode, success=bool(info["success"]),
                                        slip=int(info["slipped"]), timeout=bool(timeout and not stopped and not info["slipped"]),
                                        penetration=bool(info["penetration_failure"]), steps=env.steps-start))
                    break
            else:
                next_group.append(entry)
        active = next_group
    return results


@torch.inference_mode()
def run(args):
    _, train = read_cases(args.episodes, args.baseline, args.splits)
    starts = {}
    for index in INDICES:
        path = args.snapshots / f"{index:03d}"
        row = json.loads(path.with_suffix(".json").read_text())["row"]
        if row["case"]["target"] != train[index]["target"]:
            raise ValueError("A selected start does not match the disjoint training split")
        starts[index] = read_snapshot(path.with_suffix(".npz"), row)
    args.output.mkdir()
    cfg = json.loads(args.gate.read_text())["actor_config"]
    model, _ = load_rlinf(args.rlinf)
    actor = model(**cfg).cuda().eval().requires_grad_(False)
    if any(torch.count_nonzero(p) for p in actor.actor.mlp.net[-1].parameters()):
        raise RuntimeError("Expected a zero-initialized, frozen Actor")
    features = FrozenSmolVLA(args.checkpoint)
    if features.policy.config.n_obs_steps != 1:
        raise ValueError("The batch replay expects a one-observation policy")
    envs = [SmolVLAInsertion(penetration_limit_mm=3.) for _ in range(8)]
    try:
        mapping = ActionMap(envs[0], sigma=SIGMA)
        eps = (torch.tensor(EPS_RAD, device="cuda") / mapping.half).reshape(1, -1)
        scales = calibrate(envs[0], features, mapping, actor, starts, eps)
        write_json(args.output / "noise-scale.json", scales)
        print("SCALE", {k: {x: v for x, v in z.items() if x != "by_hole"} for k, z in scales.items()}, flush=True)
        rows = []
        with (args.output / "episodes.jsonl").open("x", buffering=1) as stream:
            for trial in range(0, TRIALS, 2):
                tasks = [(index, t, mode) for t in (trial, trial+1) for index in INDICES for mode in MODES]
                result = wave(envs, features, mapping, starts, tasks, eps, scales)
                for row in result:
                    print(json.dumps(row), file=stream)
                rows.extend(result)
                print("WAVE", trial+2, len(rows), dict(Counter(r["mode"] for r in rows if r["success"])), flush=True)
        outcomes = {mode: {} for mode in MODES}
        for mode in MODES:
            for index in INDICES:
                subset = [r for r in rows if r["mode"] == mode and r["index"] == index]
                if len(subset) != TRIALS or len({r["trial"] for r in subset}) != TRIALS:
                    raise RuntimeError("Missing or duplicate exploration episodes")
                outcomes[mode][str(index)] = {key: sum(r[key] for r in subset)
                                              for key in ("success", "slip", "timeout", "penetration")}
        write_json(args.output / "summary.json", dict(updates=0, trials_per_cell=TRIALS, outcomes=outcomes,
                   sigmas={mode: item["sigma"] for mode, item in scales.items()}))
        print("EXPLORE_COMPLETE", outcomes, flush=True)
    finally:
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "rlinf", "episodes", "baseline", "splits", "output", "gate", "snapshots"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
