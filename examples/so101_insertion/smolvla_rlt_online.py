"""Resume SO-101 RLT-TD3 online and evaluate on disjoint, paired starts."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .smolvla25_data import TARGET_RANGE_MM, SmolVLAInsertion, make_case
from .smolvla_rlt_offline import MAX_STEPS, SIGMA, demonstration_limits, evaluation_starts
from .smolvla_rlt_stage2 import CHUNK, GRIP, ActionMap, FrozenSmolVLA, collect_rollouts, collect_start, rollout, save_json
from .smolvla_rlt_td3 import batch_replay, learner, load_rlinf, plot, take_batch, train_step


END_STEP = 9000
EVAL_EVERY = 1200
EXPANSION_SEED = 20261124


def log_row(stream, row):
    print(json.dumps(row), file=stream)


def write_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class Replay:
    def __init__(self, warmup, capacity, device):
        self.count = len(warmup["actions"])

        def allocate(value):
            if isinstance(value, dict):
                return {k: allocate(v) for k, v in value.items()}
            result = torch.empty((capacity, *value.shape[1:]), dtype=value.dtype, device=device)
            result[:self.count].copy_(value)
            return result

        self.data = allocate(warmup)

    def add(self, row):
        batch = batch_replay([row], self.data["actions"].device)

        def write(target, source):
            if isinstance(target, dict):
                for key in target:
                    write(target[key], source[key])
            else:
                target[self.count].copy_(source[0])

        write(self.data, batch)
        self.count += 1

    def snapshot(self):
        def copy_prefix(value):
            if isinstance(value, dict):
                return {key: copy_prefix(tensor) for key, tensor in value.items()}
            return value[:self.count].to(device="cpu", copy=True)

        return copy_prefix(self.data)


def new_start(env, rng, reserved, seen, index):
    target = rng.uniform(-TARGET_RANGE_MM, TARGET_RANGE_MM, size=2)
    if tuple(target) in seen or np.any(np.linalg.norm(reserved - target, axis=1) <= 1e-6):
        return None
    seen.add(tuple(target))
    return collect_start(env, index, make_case(target.tolist(), rng), int(rng.integers(2**31)))


def deterministic_eval(env, features, actions, starts, policy=None):
    delta = []

    @torch.no_grad()
    def command(obs):
        batched = {key: value.unsqueeze(0) for key, value in obs.items()}
        predicted, _, _ = policy.sac_forward(batched, deterministic=True, apply_action_noise=False)
        predicted = predicted.reshape_as(obs["ref_chunk"])
        delta.append(((predicted - obs["ref_chunk"]).abs()[..., :5] * actions.half[:5]).cpu())
        return predicted

    queues = features.policy._queues
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            stats = collect_rollouts(env, features, actions, starts, actor=command if policy is not None else None)
    finally:
        features.policy._queues = queues
    if delta:
        errors = torch.stack(delta)
        stats.update(arm_mean_abs_rad=errors.mean().item(), arm_max_abs_rad=errors.max().item())
    return stats


def plot_eval(output, evaluations, baseline, episodes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    x = [row["step"] for row in evaluations]
    for axis, (key, label) in zip(axes.flat, (
        ("insertions", "Insertions out of 10"), ("closed_gripper_slips", "Closed-gripper slips"),
        ("mean_steps", "Mean control steps"), ("arm_mean_abs_rad", "Mean reference error (rad)"),
        ("arm_max_abs_rad", "Max reference error (rad)"), ("train", "Online insertion rate (last 20 episodes)"),
    ), strict=True):
        if key == "train":
            rate = [np.mean([r["success"] for r in episodes[max(0, i - 19):i + 1]])
                    for i in range(len(episodes))]
            axis.plot([r["step"] for r in episodes], rate)
        else:
            axis.plot(x, [row[key] for row in evaluations], marker="o", label="RLT Actor")
            if key in baseline:
                axis.axhline(baseline[key], linestyle="--", color="gray", label="Frozen SmolVLA")
            axis.legend()
        axis.grid(alpha=.25)
        axis.set_ylabel(label)
        axis.set_xlabel("Total Critic updates")
    fig.tight_layout()
    fig.savefig(output / "evaluation-curves.png")
    plt.close(fig)


def run(source, output):
    output.mkdir()
    device = torch.device("cuda")
    saved = torch.load(source / "td3-offline.pt", weights_only=True, map_location="cpu")
    config, args, alg = saved["run"], saved["model_config"], saved["algorithm"]
    initial_step = saved["update_step"]
    schedule = alg["actor_weight_schedule"]
    states = evaluation_starts(Path(config["eval_starts"]))
    fixed_xy = np.array([row["case"]["target"] for row, _, _ in states])
    if (initial_step, args["actor_noise_sigma"], config["max_steps"], config["gripper_rad"],
        config["control_fps"], args["num_action_chunks"], config["warmup_target_range_mm"]) != (
        3000, SIGMA, MAX_STEPS, GRIP, 25, CHUNK, TARGET_RANGE_MM
    ):
        raise ValueError("Offline checkpoint does not match the accepted control settings")
    if (schedule["warmup_updates"], schedule["ramp_updates"], schedule["online_bc_weight"],
        schedule["online_q_weight"]) != (2000, 5000, 2.8, .50):
        raise ValueError("Unexpected saved BC/Q schedule")
    chunks_per_group = alg["rlt_schedule"]["train_every_transitions"]
    updates_per_group = alg["update_epoch"]
    if (chunks_per_group, updates_per_group, alg["rlt_schedule"]["train_every_episodes"]) != (5, 5, 0):
        raise ValueError("The saved run is not using the official 5:5 online update schedule")
    required = (END_STEP - initial_step) // updates_per_group * chunks_per_group
    rlinf = Path(config["rlinf"])
    template = yaml.safe_load(
        (rlinf / "examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml").read_text(encoding="utf-8")
    )
    actor_cfg = template["actor"]
    warmup = torch.load(source / saved["replay_path"], weights_only=True, map_location="cpu")
    warm_count = len(warmup["actions"])
    warm_rows = [json.loads(line) for line in (source / "warmup-episodes.jsonl").read_text().splitlines()]
    warm_xy = {tuple(row["case"]["target"]) for row in warm_rows}
    if warm_xy.intersection(map(tuple, fixed_xy)) or sum(row["replay_chunks"] for row in warm_rows) != warm_count:
        raise ValueError("Offline replay and the disjoint warmup manifest do not match")
    if warm_count + required > alg["replay_buffer"]["max_num_samples"]:
        raise ValueError("Run exceeds the upstream replay capacity")
    baseline = json.loads((source / "baseline.json").read_text(encoding="utf-8"))
    evaluations = [dict(step=initial_step, **json.loads((source / "actor-eval.json").read_text(encoding="utf-8")))]
    save_json(output / "launch.json", dict(
        start_step=initial_step, end_step=END_STEP, required_new_action_chunks=required,
        estimated_online_episodes=round(required / (warm_count / len(warm_rows))),
        estimate_basis="warmup action chunks per episode", chunks_per_update_group=chunks_per_group,
        critic_updates_per_group=updates_per_group, eval_every_critic_updates=EVAL_EVERY,
        fixed_eval_episodes=len(states), expansion_episodes=30, source=str(source),
        best_order=["most insertions", "fewest closed-gripper slips", "lowest mean control steps"],
        best_ties="earliest online checkpoint", config=config,
    ))

    env = SmolVLAInsertion(penetration_limit_mm=3.)
    eval_env = SmolVLAInsertion(penetration_limit_mm=3.)
    reset_env = SmolVLAInsertion(penetration_limit_mm=3.)
    env.max_steps = eval_env.max_steps = MAX_STEPS
    try:
        actions = ActionMap(env, demonstration_limits(Path(config["training_dataset"])), SIGMA)
        if actions.report() != config["joints"] or env.model.opt.timestep != config["physics_dt"]:
            raise ValueError("The live environment or the action mapping differs from the offline run")
        model_class, mixin = load_rlinf(rlinf)
        loss, actor_optim = learner(model_class, mixin, actions, args, alg, actor_cfg, device)
        policy = loss.model
        policy.load_state_dict(saved["model"], strict=True)
        loss.target_model.load_state_dict(saved["target"], strict=True)
        loss.qf_optimizer.load_state_dict(saved["critic_optimizer"])
        actor_optim.load_state_dict(saved["actor_optimizer"])
        if {int(v["step"].item()) for v in loss.qf_optimizer.state.values()} != {initial_step}:
            raise ValueError("Critic optimizer step did not resume at 3,000")
        if {int(v["step"].item()) for v in actor_optim.state.values()} != {initial_step // 2}:
            raise ValueError("Actor optimizer step did not resume at 1,500")
        replay = Replay(warmup, warm_count + required, device)
        rng = np.random.default_rng()
        rng.bit_generator.state = config["warmup_rng"]
        replay_rng = random.Random()
        replay_rng.setstate(saved["replay_rng"])
        cpu_rng, cuda_rng = saved["torch_rng"], saved["cuda_rng"]
        features = FrozenSmolVLA(Path(config["smolvla"]))
        features.load_token(Path(config["token"]), Path(config["token_source"]), Path(config["smolvla"]))
        train_xy = set(warm_xy)
        step = initial_step
        best_score, best_step = None, None
        episodes = []
        positive = 0
        started = time.monotonic()

        @torch.no_grad()
        def exploratory_action(obs):
            batched = {key: value.unsqueeze(0) for key, value in obs.items()}
            predicted, _, _ = policy.sac_forward(batched, deterministic=False, apply_action_noise=True)
            return predicted.reshape_as(obs["ref_chunk"])

        with ((output / "online-updates.jsonl").open("x", buffering=1, encoding="utf-8") as updates,
              (output / "online-episodes.jsonl").open("x", buffering=1, encoding="utf-8") as episode_log,
              (output / "fixed-evaluations.jsonl").open("x", buffering=1, encoding="utf-8") as eval_log):
            log_row(eval_log, evaluations[0])

            def on_transition(row):
                nonlocal step, cpu_rng, cuda_rng, best_score, best_step, positive
                replay.add(row)
                positive += int(row["rewards"].any())
                online_count = replay.count - warm_count
                if online_count % chunks_per_group:
                    return False
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng)
                    for _ in range(updates_per_group):
                        indices = replay_rng.choices(range(replay.count), k=actor_cfg["global_batch_size"])
                        stat = train_step(loss, actor_optim, actor_cfg, take_batch(replay.data, indices), step)
                        step += 1
                        log_row(updates, dict(**stat, online_chunks=online_count, replay_chunks=replay.count,
                                              online_positive_chunks=positive, elapsed_s=time.monotonic() - started))
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = torch.cuda.get_rng_state()
                if (step - initial_step) % EVAL_EVERY == 0:
                    result = deterministic_eval(eval_env, features, actions, states, policy)
                    evaluation = dict(step=step, online_chunks=online_count, **result)
                    evaluations.append(evaluation)
                    log_row(eval_log, evaluation)
                    save_json(output / f"fixed-{step}.json", evaluation)
                    score = (result["insertions"], -result["closed_gripper_slips"], -result["mean_steps"])
                    checkpoint = dict(
                        model=policy.state_dict(), target=loss.target_model.state_dict(), model_config=args,
                        algorithm=alg, critic_optimizer=loss.qf_optimizer.state_dict(),
                        actor_optimizer=actor_optim.state_dict(), update_step=step, run=config,
                        replay=replay.snapshot(), replay_rng=replay_rng.getstate(),
                        torch_rng=cpu_rng, cuda_rng=cuda_rng, training_hole_rng=rng.bit_generator.state,
                        training_holes=[list(map(float, xy)) for xy in sorted(train_xy)],
                        fixed_evaluation=result, online_chunks=online_count,
                        online_positive_chunks=positive,
                    )
                    write_checkpoint(output / ("final.pt" if step == END_STEP else "latest.pt"), checkpoint)
                    if best_score is None or score > best_score:
                        best_score, best_step = score, step
                        write_checkpoint(output / "best.pt", checkpoint)
                return step >= END_STEP

            for index in range(required * 2):
                start = new_start(reset_env, rng, fixed_xy, train_xy, index)
                if start is None:
                    continue
                if start[1] is None:
                    log_row(episode_log, start[0])
                    continue
                before = replay.count
                row = rollout(env, features, actions, start, actor=exploratory_action, on_transition=on_transition)
                episode = dict(
                    **start[0], **{k: v for k, v in row.items() if k != "index"}, episode=len(episodes) + 1,
                    step=step, action_chunks=replay.count - before, total_online_chunks=replay.count - warm_count,
                    elapsed_s=time.monotonic() - started,
                )
                episodes.append(episode)
                log_row(episode_log, episode)
                if len(episodes) % 10 == 0 or step == END_STEP:
                    print(json.dumps(episode), flush=True)
                if step >= END_STEP:
                    break
        if step != END_STEP or replay.count - warm_count != required:
            raise RuntimeError("The completed run does not match the agreed UTD and update budgets")
        if best_step is None:
            raise RuntimeError("No qualifying online checkpoint was evaluated")
        best = torch.load(output / "best.pt", weights_only=True, map_location="cpu")
        policy.load_state_dict(best["model"], strict=True)
        if best["update_step"] != best_step:
            raise RuntimeError("Best checkpoint update count differs from the ranking")

        # Draw once after training. Keep both policies on the same saved physics
        # states and exclude every known SFT, warmup, online, and fixed hole.
        data = Path(config["training_dataset"]).parent
        sft = json.loads((data / "splits.json").read_text(encoding="utf-8"))
        seen = train_xy | {tuple(row["target"]) for row in sft["train"]}
        for line in (data / "supplement-collection.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "case" in row:
                seen.add(tuple(row["case"]["target"]))
        excluded_xy = np.array(sorted(seen | set(map(tuple, fixed_xy))))
        expansion = []
        rejected = []
        expansion_rng = np.random.default_rng(EXPANSION_SEED)
        directory = output / "expanded-states"
        directory.mkdir()
        for index in range(300):
            start = new_start(reset_env, expansion_rng, excluded_xy, seen, index)
            if start is None:
                continue
            row, state, _ = start
            if state is None:
                rejected.append(row)
                continue
            np.savez(directory / f"{index:03}.npz", physics=state)
            expansion.append(start)
            if len(expansion) == 30:
                break
        if len(expansion) != 30:
            raise RuntimeError("Could not generate 30 disjoint held insertion starts")
        save_json(directory / "manifest.json", dict(
            count=len(expansion), seed=EXPANSION_SEED, zero_training_overlap=True,
            episodes=[row for row, _, _ in expansion], rejected=rejected,
        ))
        base_new = deterministic_eval(eval_env, features, actions, expansion)
        save_json(output / "expanded-smolvla.json", base_new)
        best_new = deterministic_eval(eval_env, features, actions, expansion, policy)
        save_json(output / "expanded-best.json", dict(step=best_step, **best_new))
        curve = [json.loads(line) for line in (output / "online-updates.jsonl").read_text().splitlines()]
        plot(curve, output / "online-loss-and-q.png")
        plot_eval(output, evaluations, baseline, episodes)
        save_json(output / "summary.json", dict(
            final_step=step, best_step=best_step, online_episodes=len(episodes),
            completed_online_episodes=sum(not r["budget_exhausted"] for r in episodes),
            online_action_chunks=required, online_positive_chunks=positive,
            online_insertions=sum(r["success"] for r in episodes),
            online_closed_gripper_slips=sum(r["slipped"] for r in episodes),
            elapsed_s=time.monotonic() - started,
        ))
    finally:
        reset_env.close()
        eval_env.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
