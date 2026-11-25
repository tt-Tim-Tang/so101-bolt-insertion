"""Probe v3 residual exploration and run a small training-only memorization test."""

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
from .smolvla_rlt_stage2 import CHUNK, GRIP, ActionMap, FrozenSmolVLA
from .smolvla_rlt_td3 import batch_replay, learner, load_rlinf, take_batch, train_step
from .smolvla_rlt_v3 import HANDOFF_LIMIT, SIGMA, handoff, physical_commands, read_cases, read_snapshot, restore_handoff


ASSESSMENT_SEEDS = (None, 541702, 541703)


def actor_command(actor, obs, *, noisy=False):
    action, _, _ = actor.sac_forward({key: value.unsqueeze(0) for key, value in obs.items()},
                                     deterministic=not noisy, apply_action_noise=noisy)
    return action.reshape_as(obs["ref_chunk"])


class NoiseStream:
    """Keep exploration and learner randomness separate from the frozen flow sampler."""

    def __init__(self, seed):
        self.state = torch.Generator(device="cuda").manual_seed(seed).get_state()

    def __call__(self, fn, *args, **kwargs):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.set_rng_state(self.state)
            output = fn(*args, **kwargs)
            self.state = torch.cuda.get_rng_state()
        return output


@torch.no_grad()
def scale_probe(env, features, mapping, actor, snapshot):
    restore_handoff(env, features, snapshot)
    data = features.observe(env, mapping, with_token=True, include_physical_reference=True)
    physical = data.pop("reference_rad").cpu().numpy().astype(np.float64)
    reference = data["ref_chunk"]
    tip = env.data.body(env.bolt_id)

    def tips(action):
        restore_handoff(env, features, snapshot)
        env.max_steps = env.steps + HANDOFF_LIMIT
        points = []
        for cmd in physical_commands(env, mapping, physical, action, reference):
            env.step(cmd)
            points.append(tip.xpos[:2].copy())
        return np.stack(points)

    pure = actor_command(actor, data)
    if not torch.equal(pure, reference):
        raise RuntimeError("Noise probe requires the original zero Actor")
    baseline = tips(pure)
    stream = NoiseStream(874201)
    distances = []
    for _ in range(128):
        noisy = stream(actor_command, actor, data, noisy=True)
        distances.append(float(np.linalg.norm(tips(noisy) - baseline, axis=1).max() * 1000))
    return dict(sigma=SIGMA, trials=len(distances), median_mm=float(np.median(distances)),
                p95_mm=float(np.quantile(distances, .95)), max_mm=max(distances), samples_mm=distances)


def rollout(env, features, mapping, actor, snapshot, *, flow_seed=None, explore=None, transition=None):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        restore_handoff(env, features, snapshot)
        if flow_seed is not None:
            torch.cuda.manual_seed(flow_seed)
        start_step = env.steps
        env.max_steps = start_step + HANDOFF_LIMIT

        def observe():
            data = features.observe(env, mapping, with_token=True, include_physical_reference=True)
            return data, data.pop("reference_rad").cpu().numpy().astype(np.float64)

        obs, raw = observe()
        unit_actions = getattr(actor, "critic_in_residual_space", False)
        if unit_actions:
            eps_rad = (actor.actor.residual_eps_norm.reshape_as(obs["ref_chunk"]) * mapping.half).cpu().numpy()
        budget = done = False
        residuals = []
        while not done and not budget:
            with torch.no_grad():
                ref = obs["ref_chunk"]
                action = (explore(actor_command, actor, obs, noisy=True) if explore is not None
                          else actor_command(actor, obs))
                if unit_actions:
                    anchor = np.clip(raw, env.low, env.high)
                    delta = action.cpu().numpy() * eps_rad
                    residuals.append(float(np.abs(delta[:, :5]).mean()))
                    commands = anchor + delta
                    commands[:, -1] = GRIP
                else:
                    residuals.append(float(((action - ref).abs()[..., :5] * mapping.half[:5]).mean()))
                    commands = physical_commands(env, mapping, raw, action, ref)
            rewards = np.zeros(CHUNK, dtype=np.float32)
            executed = []
            for i, command in enumerate(commands):
                rewards[i], stopped, timeout, info = env.step(command)
                executed.append(env.data.ctrl.copy())
                done = bool(stopped or timeout or info["slipped"])
                if done:
                    break
            next_obs, next_raw = (obs, raw) if done else observe()
            if transition is not None:
                if unit_actions:
                    count = len(executed)
                    units = np.zeros((CHUNK, ref.shape[-1]), dtype=np.float32)
                    units[:count, :5] = np.clip((np.array(executed)[:, :5] - anchor[:count, :5]) / eps_rad[:count, :5], -1, 1)
                    # Repeat the last realized residual in unused terminal positions.
                    units[count:] = units[count-1]
                    measured = torch.as_tensor(units, device=ref.device)
                else:
                    commands = executed + [executed[-1]] * (CHUNK - len(executed))
                    measured = mapping.normalize(torch.as_tensor(np.array(commands), device=ref.device))
                budget = bool(transition(dict(curr_obs=obs, next_obs=next_obs, actions=measured, rewards=rewards,
                                              dones=done, terminations=done, intervene_flags=np.zeros(CHUNK, bool))))
            obs, raw = next_obs, next_raw
        return dict(success=bool(info["success"]), slip=int(info["slipped"]), timeout=bool(timeout and not stopped),
                    penetration=bool(info["penetration_failure"]), steps=env.steps-start_step, chunks=len(residuals),
                    deepest_held_mm=env.max_near_depth_mm, stable_s=info["stable_s"],
                    mean_residual_rad=float(np.mean(residuals)), budget_truncated=bool(budget and not done))


def assessments(env, features, mapping, actor, starts):
    return [dict(index=index, flow_seed=seed, **rollout(env, features, mapping, actor, snapshot, flow_seed=seed))
            for index, snapshot in starts for seed in ASSESSMENT_SEEDS]


def select_starts(env, features, mapping, actor, train, output):
    # Obtain candidates from the training split only. Start with the existing training-eval grid.
    order = [20, 0, 9, 90, 99, 3, 6, 30, 33, 36, 39, 60, 63, 66, 69, 93, 96]
    starts = []
    folder = output / "selected-starts"
    folder.mkdir()
    with (output / "screening.jsonl").open("x", buffering=1) as stream:
        for index in order + [i for i in range(len(train)) if i not in order]:
            with torch.inference_mode():
                snapshot, reason = handoff(env, features, train[index], 714290 + index)
            tests = []
            if snapshot is not None:
                for seed in ASSESSMENT_SEEDS:
                    result = rollout(env, features, mapping, actor, snapshot, flow_seed=seed)
                    tests.append(dict(flow_seed=seed, **result))
                    if result["success"] or not result["timeout"] or result["deepest_held_mm"] < 15.:
                        break
                if len(tests) == 3 and all(t["timeout"] and not t["success"] and t["deepest_held_mm"] >= 15 for t in tests):
                    row, state, cpu, cuda = snapshot
                    np.savez_compressed(folder / f"{index:03d}.npz", physics=state, cpu_rng=cpu.numpy(), cuda_rng=cuda.numpy())
                    write_json(folder / f"{index:03d}.json", dict(index=index, row=row, baseline=tests))
                    starts.append((index, snapshot))
            accepted = bool(starts and starts[-1][0] == index)
            print(json.dumps(dict(index=index, target=train[index]["target"], accepted=accepted,
                                  rejection=reason, tests=tests)), file=stream)
            print("SCREEN", index, accepted, [t["success"] for t in tests], flush=True)
            if len(starts) == 2:
                return starts
    raise RuntimeError("Could not identify two stable training-only timeout snapshots")


@torch.no_grad()
def diagnostics(policy, batch, mapping):
    obs = batch["curr_obs"]
    action = policy.sac_forward(obs, deterministic=True, apply_action_noise=False)[0]
    data_q = policy.sac_q_forward(obs, batch["actions"])
    policy_q = policy.sac_q_forward(obs, action).min(dim=-1).values
    residual = (action.reshape_as(obs["ref_chunk"]) - obs["ref_chunk"])[..., :5] * mapping.half[:5]

    def q_stats(q):
        return dict(mean=float(q.mean()), minimum=float(q.min()), maximum=float(q.max()),
                    below_0=float((q < 0).float().mean()), above_1=float((q > 1).float().mean()))

    return dict(replay_q=q_stats(data_q), actor_q=q_stats(policy_q),
                mean_abs_residual_rad=float(residual.abs().mean()), signed_residual_rad=residual.mean(dim=(0, 1)).tolist(),
                max_abs_residual_rad=float(residual.abs().max()))


def run(args):
    _, train = read_cases(args.episodes, args.baseline, args.splits)
    args.output.mkdir()
    gate = json.loads((args.gate / "summary.json").read_text())
    cfg = gate["actor_config"]
    defaults = yaml.safe_load((args.rlinf / "examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml").read_text())
    actor_cfg, alg = defaults["actor"], defaults["algorithm"]
    online = alg["actor_weight_schedule"]
    # Warmup here is collection only. Use the original schedule's online endpoint during the short online test.
    alg.update(actor_weight_schedule={"enable": False}, reference_dropout_prob=0.,
               bc_weight=online["online_bc_weight"], q_weight=online["online_q_weight"])
    group = alg["rlt_schedule"]["train_every_transitions"]
    if args.updates < 1000 or args.updates > 2000 or args.updates % group:
        raise ValueError("Expected 1,000–2,000 updates divisible by the RLinf update group size")
    features = FrozenSmolVLA(args.checkpoint)
    features.load_token(args.token, args.rlinf / "rlinf/models/embodiment/modules/rlt_token_transformer.py", args.checkpoint)
    env = EndToEndTelemetry(penetration_limit_mm=3.)
    try:
        mapping = ActionMap(env, sigma=SIGMA)
        model, mixin = load_rlinf(args.rlinf)
        torch.manual_seed(actor_cfg["seed"])
        loss, optimizer = learner(model, mixin, mapping, cfg, alg, actor_cfg, "cuda")
        policy = loss.model
        audit = next(r for r in gate["training_snapshots"] if r.get("file"))
        if audit["row"]["case"]["target"] != train[audit["index"]]["target"]:
            raise ValueError("Calibration snapshot is outside the verified training set")
        scale = scale_probe(env, features, mapping, policy, read_snapshot(args.gate / "train-starts" / audit["file"], audit["row"]))
        write_json(args.output / "noise-scale.json", dict(training_index=audit["index"], **scale))
        print("SCALE", {k: v for k, v in scale.items() if k != "samples_mm"}, flush=True)

        starts = select_starts(env, features, mapping, policy, train, args.output)
        scales = [dict(training_index=audit["index"], **scale)]
        scales.extend(dict(training_index=index, **scale_probe(env, features, mapping, policy, snapshot))
                      for index, snapshot in starts if index != audit["index"])
        draws = [value for probe in scales for value in probe["samples_mm"]]
        write_json(args.output / "noise-scale.json", dict(sigma=SIGMA, probes=scales, draws=len(draws),
                   median_mm=float(np.median(draws)), p95_mm=float(np.quantile(draws, .95))))
        warm = []
        warm_log = []
        rollout_rng = NoiseStream(20261129)
        update_rng = NoiseStream(20261130)
        rng = random.Random(20261131)
        for repeat in range(3):
            for index, start in starts:
                row = rollout(env, features, mapping, policy, start, explore=rollout_rng, transition=warm.append)
                warm_log.append(dict(index=index, repeat=repeat, **row))
        replay = Replay(batch_replay(warm, "cuda"), len(warm) + args.updates, "cuda")
        positives = sum(bool(row["rewards"].any()) for row in warm)
        write_json(args.output / "warmup.json", dict(episodes=warm_log, blocks=len(warm), positive_blocks=positives))
        print("WARMUP", len(warm_log), len(warm), positives, flush=True)
        evaluations = []
        step = online_blocks = 0
        online_eps = []

        def evaluate():
            rows = assessments(env, features, mapping, policy, starts)
            probe = take_batch(replay.data, rng.choices(range(replay.count), k=256))
            record = dict(step=step, eval=rows, insertions=sum(row["success"] for row in rows),
                          **diagnostics(policy, probe, mapping))
            evaluations.append(record)
            print("EVAL", json.dumps(record), flush=True)
            write_json(args.output / "evaluations.json", evaluations)

        # Screening already provides the exact noise-free step-0 evaluation on these starts.
        selected = [json.loads((args.output / "selected-starts" / f"{index:03d}.json").read_text()) for index, _ in starts]
        evaluations.append(dict(step=0, eval=[dict(index=r["index"], **row) for r in selected for row in r["baseline"]],
                                insertions=0, **diagnostics(policy, take_batch(replay.data, list(range(replay.count))), mapping)))
        with (args.output / "updates.jsonl").open("x", buffering=1) as updates, \
                (args.output / "online.jsonl").open("x", buffering=1) as outputs:
            def add(row):
                nonlocal step, online_blocks
                replay.add(row)
                online_blocks += 1
                if online_blocks % group == 0:
                    for _ in range(group):
                        batch = take_batch(replay.data, rng.choices(range(replay.count), k=actor_cfg["global_batch_size"]))
                        metric = update_rng(train_step, loss, optimizer, actor_cfg, batch, step)
                        print(json.dumps(metric), file=updates)
                        step += 1
                        if step % 100 == 0:
                            print("UPDATE", step, metric["q_data"], flush=True)
                return step == args.updates

            while step < args.updates:
                index, start = starts[len(online_eps) % len(starts)]
                row = rollout(env, features, mapping, policy, start, explore=rollout_rng, transition=add)
                online_eps.append(dict(index=index, step=step, **row))
                print(json.dumps(online_eps[-1]), file=outputs)
                if step == args.updates or (len(evaluations) == 1 and step >= args.updates // 2):
                    evaluate()

        write_checkpoint(args.output / "short-actor.pt", dict(model=policy.state_dict(), model_config=cfg, update_step=step))
        write_json(args.output / "summary.json", dict(training_indices=[index for index, _ in starts],
                   updates=step, warmup=dict(episodes=len(warm_log), blocks=len(warm), positive_blocks=positives),
                   online=dict(episodes=len(online_eps), blocks=online_blocks, insertions=sum(r["success"] for r in online_eps),
                               positive_blocks=int((replay.data["rewards"][len(warm):replay.count] > 0).any(dim=-1).sum())),
                   config=dict(bc_weight=alg["bc_weight"], q_weight=alg["q_weight"], batch=actor_cfg["global_batch_size"],
                               group=group, sigma=SIGMA), evaluations=evaluations))
        print("SHORT_COMPLETE", step, evaluations[-1]["insertions"], flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "token", "rlinf", "episodes", "baseline", "splits", "output", "gate"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
