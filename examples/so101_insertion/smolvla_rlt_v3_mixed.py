"""Run the second v3 short test after the zero-update exploration gate."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .smolvla_motion_audit import write_json
from .smolvla_padding_e2e30 import EndToEndTelemetry
from .smolvla_rlt_online import Replay, write_checkpoint
from .smolvla_rlt_stage2 import CHUNK, ActionMap, FrozenSmolVLA
from .smolvla_rlt_td3 import batch_replay, learner, load_rlinf, take_batch, train_step
from .smolvla_rlt_v3 import SIGMA, handoff, read_cases, read_snapshot
from .smolvla_rlt_v3_short import ASSESSMENT_SEEDS, NoiseStream, assessments, rollout


ONLINE_UPDATES = 1200
FAILURES = (60, 66)


class Exploration:
    def __init__(self, mode, sigma):
        self.mode, self.sigma = mode, sigma
        self.rng = torch.Generator(device="cuda").manual_seed(20261203)

    def __call__(self, _actor_command, policy, obs, *, noisy):
        ref = obs["ref_chunk"]
        _, _, unit = policy.sac_forward({k: v.unsqueeze(0) for k, v in obs.items()}, return_residual_u=True,
                                       deterministic=True, apply_action_noise=False)
        eps = policy.actor.residual_eps_norm.reshape_as(ref)
        shape = (1 if self.mode == "block" else CHUNK, ref.shape[-1])
        noise = torch.randn(shape, generator=self.rng, device=ref.device) * self.sigma if noisy else 0.
        return (ref + (eps * unit.reshape_as(ref) + noise).clamp(-eps, eps)).clamp(-1., 1.)


def save_start(folder, index, snap, baseline):
    row, physics, cpu, cuda = snap
    np.savez_compressed(folder / f"{index:03d}.npz", physics=physics, cpu_rng=cpu.numpy(), cuda_rng=cuda.numpy())
    write_json(folder / f"{index:03d}.json", dict(index=index, row=row, baseline=baseline))


def choose_starts(args, env, features, mapping, policy, train):
    failures, successes = [], []
    for index in FAILURES:
        path = args.snapshots / f"{index:03d}"
        row = json.loads(path.with_suffix(".json").read_text())["row"]
        if row["case"]["target"] != train[index]["target"]:
            raise ValueError("A failure snapshot is outside the verified training split")
        failures.append((index, read_snapshot(path.with_suffix(".npz"), row)))
    folder = args.output / "success-starts"
    folder.mkdir()
    for index in (20, 0, 3, 6, 30, 33, 36, 39, 63, 69):
        with torch.inference_mode():
            snap, reason = handoff(env, features, train[index], 714290 + index)
        if snap is None:
            print("SUCCESS_SCREEN_REJECT", index, reason, flush=True)
            continue
        samples = assessments(env, features, mapping, policy, [(index, snap)])
        if all(row["success"] and not row["slip"] for row in samples):
            save_start(folder, index, snap, samples)
            successes.append((index, snap))
        print("SUCCESS_SCREEN", index, sum(r["success"] for r in samples), flush=True)
        if len(successes) == 2:
            break
    if len(successes) != 2:
        raise RuntimeError("Could not find two reproducibly successful training snapshots")
    return failures, successes


@torch.no_grad()
def q_by_outcome(policy, replay, labels):
    known = [i for i, label in enumerate(labels) if label is not None]
    batch = take_batch(replay.data, known)
    obs = batch["curr_obs"]
    action, _, unit = policy.sac_forward(obs, deterministic=True, apply_action_noise=False, return_residual_u=True)
    data_q = policy.sac_q_forward(obs, batch["actions"]).min(dim=-1).values
    actor_q = policy.sac_q_forward(obs, action).min(dim=-1).values
    mask = torch.tensor([labels[i] for i in known], dtype=torch.bool, device=actor_q.device)
    positive = (batch["rewards"] > 0).any(dim=-1)
    if not (bool(mask.any()) and bool((~mask).any()) and bool(positive.any())):
        raise ValueError("Warmup must contain positive and negative trajectories and positive rewards")

    def stats(q, selected):
        values = q[selected]
        return dict(n=len(values), mean=float(values.mean()), below_0=float((values < 0).float().mean()),
                    below_minus_03=float((values < -.03).float().mean()), above_1=float((values > 1).float().mean()))

    return dict(positive_trajectory={"replay_q": stats(data_q, mask), "actor_q": stats(actor_q, mask)},
                negative_trajectory={"replay_q": stats(data_q, ~mask), "actor_q": stats(actor_q, ~mask)},
                positive_reward_block_q=stats(data_q, positive), mean_abs_actor_q=float(actor_q.abs().mean()),
                unit_bc=float(unit.square().mean()))


def run(args):
    exploration = json.loads(args.exploration.read_text())
    totals = {mode: sum(r["success"] for r in by_hole.values()) for mode, by_hole in exploration["outcomes"].items()}
    mode = max(totals, key=totals.get)
    if totals[mode] < 2 or exploration["updates"] != 0 or exploration["trials_per_cell"] != 30:
        raise ValueError("No reproducible zero-update exploration result")
    sigma = exploration["sigmas"][mode]
    explore = Exploration(mode, sigma)
    _, train = read_cases(args.episodes, args.baseline, args.splits)
    args.output.mkdir()
    cfg = json.loads(args.gate.read_text())["actor_config"]
    defaults = yaml.safe_load((args.rlinf / "examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml").read_text())
    actor_cfg, alg = defaults["actor"], defaults["algorithm"]
    q_weight = alg["actor_weight_schedule"]["online_q_weight"]
    alg.update(actor_weight_schedule={"enable": False}, reference_dropout_prob=0.,
               residual_unit_bc=True, bc_weight=0., q_weight=q_weight)
    group_size = alg["rlt_schedule"]["train_every_transitions"]
    rng = random.Random(20261204)
    update_rng = NoiseStream(20261205)
    features = FrozenSmolVLA(args.checkpoint)
    features.load_token(args.token, args.rlinf / "rlinf/models/embodiment/modules/rlt_token_transformer.py", args.checkpoint)
    env = EndToEndTelemetry(penetration_limit_mm=3.)
    try:
        mapping = ActionMap(env, sigma=SIGMA)
        model, mixin = load_rlinf(args.rlinf)
        torch.manual_seed(actor_cfg["seed"])
        loss, optimizer = learner(model, mixin, mapping, cfg, alg, actor_cfg, "cuda")
        policy = loss.model
        negatives, positives = choose_starts(args, env, features, mapping, policy, train)
        starts = [x for pair in zip(negatives, positives, strict=True) for x in pair]
        positive_index = {i for i, _ in positives}
        warm, warm_log, labels = [], [], []
        while len(warm) < 400:
            for index, snap in starts:
                # The successful half contributes reference-policy trajectories and terminal positive rewards.
                is_positive_start = index in positive_index
                flow = ASSESSMENT_SEEDS[len(warm_log) // 4 % len(ASSESSMENT_SEEDS)] if is_positive_start else None
                n = len(warm)
                row = rollout(env, features, mapping, policy, snap, flow_seed=flow,
                              explore=None if is_positive_start else explore, transition=warm.append)
                labels.extend([row["success"]] * (len(warm)-n))
                warm_log.append(dict(index=index, positive_start=is_positive_start, **row))
        if len(warm) > 600:
            raise RuntimeError("Warmup exceeded its block budget")
        reward_count = sum(bool(row["rewards"].any()) for row in warm)
        write_json(args.output / "warmup.json", dict(rows=warm_log, blocks=len(warm), positive_blocks=reward_count))
        print("WARMUP", len(warm_log), len(warm), reward_count, flush=True)
        if reward_count == 0:
            raise RuntimeError("No positive reward in mixed Warmup")
        replay = Replay(batch_replay(warm, "cuda"), len(warm) + ONLINE_UPDATES, "cuda")

        def sample():
            return take_batch(replay.data, rng.choices(range(replay.count), k=actor_cfg["global_batch_size"]))

        policy.actor.requires_grad_(False)
        gates = []
        with (args.output / "critic-only.jsonl").open("x", buffering=1) as log:
            for step in range(800):
                row = update_rng(train_step, loss, optimizer, actor_cfg, sample(), step, update_actor=False)
                print(json.dumps(row), file=log)
                if step+1 in (400, 600, 800):
                    q = q_by_outcome(policy, replay, labels)
                    pos = q["positive_trajectory"]["replay_q"]
                    neg = q["negative_trajectory"]["replay_q"]
                    passed = (pos["mean"] > neg["mean"] and q["positive_reward_block_q"]["mean"] > neg["mean"]
                              and pos["below_minus_03"] <= .1 and neg["below_minus_03"] <= .1)
                    gates.append(dict(updates=step+1, passed=passed, **q))
                    write_json(args.output / "critic-gate.json", gates)
                    print("CRITIC", gates[-1], flush=True)
                    if passed:
                        break
        if not gates[-1]["passed"]:
            raise RuntimeError("Critic-only warmup did not separate outcomes; Actor stayed frozen")
        policy.actor.requires_grad_(True)
        # Match the warm Critic scale at a unit residual RMS of 0.25 rather than at the identically-zero Actor.
        bc_weight = q_weight * gates[-1]["mean_abs_actor_q"] / (.25**2)
        loss.cfg.algorithm["bc_weight"] = bc_weight
        print("BC", bc_weight, q_weight, flush=True)
        online = []
        checks = []
        completed = 0

        def assess(step):
            groups = {}
            for name, subset in (("failures", negatives), ("successes", positives)):
                rows = assessments(env, features, mapping, policy, subset)
                groups[name] = dict(rows=rows, trials=len(rows), insertions=sum(r["success"] for r in rows),
                                    slips=sum(r["slip"] for r in rows),
                                    mean_residual_rad=float(np.mean([r["mean_residual_rad"] for r in rows])))
            checks.append(dict(step=step, groups=groups, q=q_by_outcome(policy, replay, labels)))
            write_json(args.output / "evaluations.json", checks)
            print("ASSESS", step, {k: (v["insertions"], v["slips"]) for k, v in groups.items()}, flush=True)

        assess(0)
        with (args.output / "updates.jsonl").open("x", buffering=1) as log, \
                (args.output / "online.jsonl").open("x", buffering=1) as episode_log:
            def add(row):
                nonlocal completed
                replay.add(row)
                labels.append(None)
                if (replay.count-len(warm)) % group_size == 0:
                    for _ in range(group_size):
                        metric = update_rng(train_step, loss, optimizer, actor_cfg, sample(), completed)
                        print(json.dumps(metric), file=log)
                        completed += 1
                        if completed % 100 == 0:
                            print("UPDATE", completed, metric["q_data"], flush=True)
                return completed == ONLINE_UPDATES

            while completed < ONLINE_UPDATES:
                index, snap = starts[len(online) % len(starts)]
                first = replay.count
                row = rollout(env, features, mapping, policy, snap, explore=explore, transition=add)
                if not row["budget_truncated"]:
                    labels[first:replay.count] = [row["success"]] * (replay.count-first)
                online.append(dict(index=index, positive_start=index in positive_index, step=completed, **row))
                print(json.dumps(online[-1]), file=episode_log)
                if (len(checks) == 1 and completed >= ONLINE_UPDATES//2) or completed == ONLINE_UPDATES:
                    assess(completed)

        final = dict(exploration=dict(mode=mode, sigma=sigma, gate_counts=totals),
                     failure_indices=list(FAILURES), success_indices=sorted(positive_index),
                     warmup=dict(blocks=len(warm), episodes=len(warm_log), positive_blocks=reward_count),
                     critic_only_updates=gates[-1]["updates"], online_updates=completed,
                     total_positive_blocks=int((replay.data["rewards"][:replay.count] > 0).any(dim=-1).sum()),
                     config=dict(bc_weight=bc_weight, q_weight=q_weight), evaluations=checks)
        write_checkpoint(args.output / "mixed-short-actor.pt", dict(model=policy.state_dict(), model_config=cfg,
                                                                     update_step=completed))
        write_json(args.output / "summary.json", final)
        print("MIXED_COMPLETE", completed, final["total_positive_blocks"], flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "token", "rlinf", "episodes", "baseline", "splits", "output", "gate", "snapshots", "exploration"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
