"""Run the separate qpos-relative RLT-TD3 experiment from warmup to evaluation."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .smolvla25_data import TARGET_RANGE_MM, SmolVLAInsertion
from .smolvla_rlt_delta import GRIP, SIGMA, SLIP_REWARD, DeltaActionMap, rollout_delta
from .smolvla_rlt_delta_audit import demonstrations
from .smolvla_rlt_offline import MAX_STEPS, demonstration_limits, evaluation_starts
from .smolvla_rlt_online import Replay, log_row, new_start, write_checkpoint
from .smolvla_rlt_stage2 import ActionMap, FrozenSmolVLA, save_json
from .smolvla_rlt_td3 import learner, load_rlinf, plot, take_batch, train_step, update_td3


WARMUP_SEED = 20261206
OFFLINE_STEPS = 3000
FINAL_STEP = 12000
EVAL_EVERY = 1500


def summarize(rows):
    motion = np.concatenate([row["wrist_mm"] for row in rows])
    return dict(
        episodes=len(rows), insertions=sum(row["success"] for row in rows),
        closed_gripper_slips=sum(row["slipped"] for row in rows),
        mean_steps=float(np.mean([row["steps"] for row in rows])),
        wrist_p95_mm=float(np.quantile(motion, .95)), wrist_max_mm=float(motion.max()),
        executed_delta_max_rad=np.max([row["peak_delta_rad"] for row in rows], axis=0).tolist(),
    )


def evaluate(env, features, actions, starts, policy):
    queues = features.policy._queues
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            rows = [rollout_delta(env, features, actions, start, policy=policy) for start in starts]
    finally:
        features.policy._queues = queues
    return dict(**summarize(rows), rows=rows)


def collect_warmup(env, reset, features, actions, reserved, rng, seen, output):
    replay, rows = [], []

    def collect(transition):
        replay.append(transition)
        return False

    with (output / "warmup-episodes.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for attempt in range(500):
            start = new_start(reset, rng, reserved, seen, attempt)
            if start is None:
                continue
            if start[1] is None:
                log_row(log, start[0])
                continue
            before = len(replay)
            stats = rollout_delta(env, features, actions, start, on_transition=collect)
            row = {**start[0], **stats, "action_chunks": len(replay) - before}
            rows.append(row)
            log_row(log, row)
            print("WARMUP", len(rows), len(replay), row["success"], row["slipped"], flush=True)
            if len(rows) >= 30 and len(replay) >= 1000:
                break
        else:
            raise RuntimeError("Warmup did not reach the approved minimums")
    result = dict(**summarize(rows), action_chunks=len(replay),
                  positive_chunks=sum(bool((r["rewards"] > 0).any()) for r in replay),
                  negative_chunks=sum(bool((r["rewards"] < 0).any()) for r in replay))
    if result["negative_chunks"] != result["closed_gripper_slips"]:
        raise RuntimeError("Slip penalties and Warmup slip outcomes differ")
    save_json(output / "warmup.json", result)
    return replay


def online(env, eval_env, reset, features, actions, eval_actions, fixed, reserved, rng, seen, output, offline_policy, run):
    rlinf = Path(run["rlinf"])
    device = actions.half.device
    offline = torch.load(output / "td3-offline.pt", weights_only=True, map_location="cpu")
    alg, args = offline["algorithm"], offline["model_config"]
    template = yaml.safe_load(
        (rlinf / "examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml").read_text(encoding="utf-8")
    )
    actor_cfg = template["actor"]
    chunks_per_group = alg["rlt_schedule"]["train_every_transitions"]
    updates_per_group = alg["update_epoch"]
    if (chunks_per_group, updates_per_group, alg["rlt_schedule"]["train_every_episodes"]) != (5, 5, 0):
        raise ValueError("The upstream online UTD schedule changed")
    schedule = alg["actor_weight_schedule"]
    if (schedule["warmup_updates"], schedule["ramp_updates"], schedule["online_bc_weight"],
        schedule["online_q_weight"]) != (2000, 5000, 2.8, .50):
        raise ValueError("Unexpected BC/Q schedule")
    step = offline["update_step"]
    if step != OFFLINE_STEPS or args["actor_noise_sigma"] != SIGMA:
        raise ValueError("The offline checkpoint has the wrong update count or noise")
    required = (FINAL_STEP - step) // updates_per_group * chunks_per_group
    warm = torch.load(output / offline["replay_path"], weights_only=True, map_location="cpu")
    warm_count = len(warm["actions"])
    replay = Replay(warm, warm_count + required, device)
    replay_rng = random.Random()
    replay_rng.setstate(offline["replay_rng"])
    cpu_rng, cuda_rng = offline["torch_rng"], offline["cuda_rng"]
    model_class, mixin = load_rlinf(rlinf)
    loss, actor_optim = learner(model_class, mixin, actions, args, alg, actor_cfg, device)
    policy = loss.model
    policy.load_state_dict(offline["model"], strict=True)
    loss.target_model.load_state_dict(offline["target"], strict=True)
    loss.qf_optimizer.load_state_dict(offline["critic_optimizer"])
    actor_optim.load_state_dict(offline["actor_optimizer"])
    assert {int(v["step"].item()) for v in loss.qf_optimizer.state.values()} == {OFFLINE_STEPS}
    assert {int(v["step"].item()) for v in actor_optim.state.values()} == {OFFLINE_STEPS // 2}
    offline_eval = evaluate(eval_env, features, eval_actions, fixed, offline_policy)
    save_json(output / "fixed-3000.json", dict(step=step, **offline_eval))
    evaluations = [dict(step=step, **{k:v for k,v in offline_eval.items() if k != "rows"})]
    best = None
    best_step = None
    positive = negative = 0
    episodes = []
    started = time.monotonic()
    save_json(output / "online-launch.json", dict(
        offline_steps=step, final_step=FINAL_STEP, chunks_per_group=chunks_per_group,
        updates_per_group=updates_per_group, online_chunks=required, eval_every=EVAL_EVERY,
        bc_q_schedule=schedule, warmup_chunks=warm_count,
    ))
    with ((output / "online-updates.jsonl").open("x", encoding="utf-8", buffering=1) as updates,
          (output / "online-episodes.jsonl").open("x", encoding="utf-8", buffering=1) as episode_log,
          (output / "fixed-progress.jsonl").open("x", encoding="utf-8", buffering=1) as eval_log):
        log_row(eval_log, evaluations[0])

        def transition(row):
            nonlocal step, cpu_rng, cuda_rng, positive, negative, best, best_step
            replay.add(row)
            positive += int((row["rewards"] > 0).any())
            negative += int((row["rewards"] < 0).any())
            chunks = replay.count - warm_count
            if chunks % chunks_per_group:
                return False
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                for _ in range(updates_per_group):
                    sample = replay_rng.choices(range(replay.count), k=actor_cfg["global_batch_size"])
                    metrics = train_step(loss, actor_optim, actor_cfg, take_batch(replay.data, sample), step)
                    step += 1
                    log_row(updates, dict(**metrics, online_chunks=chunks, replay_chunks=replay.count,
                                          positive_chunks=positive, negative_chunks=negative,
                                          elapsed_s=time.monotonic() - started))
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            if (step - OFFLINE_STEPS) % EVAL_EVERY == 0:
                results = evaluate(eval_env, features, eval_actions, fixed, policy)
                save_json(output / f"fixed-{step}.json", dict(step=step, **results))
                summary = dict(step=step, online_chunks=chunks, **{k:v for k,v in results.items() if k != "rows"})
                evaluations.append(summary)
                log_row(eval_log, summary)
                score = (results["insertions"], -results["closed_gripper_slips"],
                         -results["wrist_p95_mm"], -results["wrist_max_mm"])
                checkpoint = dict(model=policy.state_dict(), model_config=args, step=step, run=run,
                                  fixed_eval=summary)
                if best is None or score > best:
                    best, best_step = score, step
                    write_checkpoint(output / "best.pt", checkpoint)
                write_checkpoint(output / "latest.pt", dict(
                    **checkpoint, target=loss.target_model.state_dict(), algorithm=alg,
                    critic_optimizer=loss.qf_optimizer.state_dict(), actor_optimizer=actor_optim.state_dict(),
                    replay=replay.snapshot(), replay_rng=replay_rng.getstate(),
                    torch_rng=cpu_rng, cuda_rng=cuda_rng, best_step=best_step,
                    training_rng=rng.bit_generator.state,
                    training_holes=[list(map(float, xy)) for xy in sorted(seen)],
                ))
            return step >= FINAL_STEP

        for index in range(required * 2):
            start = new_start(reset, rng, reserved, seen, index)
            if start is None:
                continue
            if start[1] is None:
                log_row(episode_log, start[0])
                continue
            before = replay.count
            stats = rollout_delta(env, features, actions, start, policy=policy, explore=True, on_transition=transition)
            episode = {**start[0], **stats, "step":step, "action_chunks":replay.count-before}
            episodes.append(episode)
            log_row(episode_log, episode)
            if len(episodes) % 10 == 0 or step == FINAL_STEP:
                print("ONLINE", len(episodes), step, positive, negative, flush=True)
            if step == FINAL_STEP:
                break
    if step != FINAL_STEP or replay.count - warm_count != required or best_step is None:
        raise RuntimeError("Online updates did not match the required schedule")
    if sum(r["slipped"] for r in episodes) != negative:
        raise RuntimeError("Online slip outcomes do not equal terminal penalty transitions")
    (output / "latest.pt").replace(output / "final.pt")
    final = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    assert final["step"] == FINAL_STEP and len(final["replay"]["actions"]) == warm_count + required
    best_model = torch.load(output / "best.pt", weights_only=True, map_location="cpu")
    assert best_model["step"] == best_step
    policy.load_state_dict(best_model["model"], strict=True)
    save_json(output / "online-summary.json", dict(
        **summarize(episodes), best_step=best_step, final_step=step, action_chunks=required,
        completed_episodes=sum(not r["budget_exhausted"] for r in episodes),
        positive_chunks=positive, negative_chunks=negative, elapsed_s=time.monotonic()-started,
    ))
    return policy, best_step, evaluations


def expanded_starts(folder):
    rows = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["episodes"]
    if len(rows) != 30 or len({tuple(row["case"]["target"]) for row in rows}) != 30:
        raise ValueError("Expected 30 distinct Round 1 expansion starts")
    starts = []
    for row in rows:
        with np.load(folder / f'{row["index"]:03}.npz') as state:
            starts.append((row, state["physics"], None))
    return starts


def figures(output, evaluations):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [json.loads(row) for row in (output / "online-updates.jsonl").read_text().splitlines()]
    plot(metrics, output / "online-loss-and-q.png")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for axis, key, label in zip(axes.flat, ("insertions", "closed_gripper_slips", "wrist_p95_mm", "wrist_max_mm"),
                               ("Insertions out of 10", "Closed-gripper slips", "Wrist P95 (mm)", "Wrist max (mm)"),
                               strict=True):
        axis.plot([r["step"] for r in evaluations], [r[key] for r in evaluations], marker="o")
        axis.set_title(label)
        axis.set_xlabel("Total Critic updates")
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "fixed-evaluation-curves.png")
    plt.close(fig)


def run(source, round1, output):
    output.mkdir()
    old = json.loads((source / "run-config.json").read_text(encoding="utf-8"))
    root = Path(old["training_dataset"])
    limits, evidence = demonstrations(root)
    fixed = evaluation_starts(Path(old["eval_starts"]))
    expanded = expanded_starts(round1 / "expanded-states")
    reserved = np.array([row["case"]["target"] for row, _, _ in fixed + expanded])
    if len(set(map(tuple, reserved))) != 40:
        raise ValueError("Evaluation holes are not disjoint")
    save_json(output / "delta-config.json", dict(
        mode="qpos_plus_delta", sigma=SIGMA, slip_reward=SLIP_REWARD, success_reward=1.,
        gripper=GRIP, max_steps=MAX_STEPS, fixed_holes=10, expanded_holes=30,
        reserved_eval_holes=reserved.tolist(), warmup_seed=WARMUP_SEED, range_mm=TARGET_RANGE_MM,
        delta_caps_rad=limits.tolist(), demonstration_frames=evidence["frame_pairs"],
    ))
    collect = SmolVLAInsertion(penetration_limit_mm=3.)
    eval_env = SmolVLAInsertion(penetration_limit_mm=3.)
    reset = SmolVLAInsertion(penetration_limit_mm=3.)
    collect.max_steps = eval_env.max_steps = MAX_STEPS
    try:
        absolute = ActionMap(collect, demonstration_limits(root))
        limits_from_offline = np.array([[j["min_rad"], j["max_rad"]] for j in old["joints"]])
        np.testing.assert_array_equal(np.array([absolute.low.cpu(), absolute.high.cpu()]).T, limits_from_offline)
        actions = DeltaActionMap(collect, absolute, limits)
        eval_actions = DeltaActionMap(eval_env, absolute, limits)
        features = FrozenSmolVLA(Path(old["smolvla"]))
        features.load_token(Path(old["token"]), Path(old["token_source"]), Path(old["smolvla"]))
        rng = np.random.default_rng(WARMUP_SEED)
        seen = set()
        replay = collect_warmup(collect, reset, features, actions, reserved, rng, seen, output)
        configuration = {**old, "action_mode":"qpos_plus_delta", "delta_caps_rad":limits.tolist(),
                         "sigma":SIGMA, "slip_reward":SLIP_REWARD, "warmup_seed":WARMUP_SEED,
                         "warmup_rng":rng.bit_generator.state, "round1_eval":str(round1)}
        offline_policy = update_td3(Path(old["rlinf"]), actions, replay, output, updates=OFFLINE_STEPS,
                                   offline_run=configuration)
        del replay
        best, best_step, evaluations = online(collect, eval_env, reset, features, actions, eval_actions, fixed, reserved,
                                               rng, seen, output, offline_policy, configuration)
        if any(np.any(np.linalg.norm(reserved - xy, axis=1) <= 1e-6) for xy in seen):
            raise RuntimeError("Training and evaluation hole isolation failed")
        r2 = evaluate(eval_env, features, eval_actions, expanded, best)
        history = {}
        indices = [row["index"] for row in r2["rows"]]
        for name, file in (("frozen_smolvla", "expanded-smolvla.json"), ("round1_best", "expanded-best.json")):
            old_eval = json.loads((round1 / file).read_text(encoding="utf-8"))
            if [row["index"] for row in old_eval["rows"]] != indices:
                raise ValueError("The policies did not evaluate the same ordered starts")
            history[name] = old_eval
        save_json(output / "expanded-comparison.json", dict(**history, round2_best=dict(step=best_step, **r2)))
        figures(output, evaluations)
        print("DELTA_RLT_COMPLETE", flush=True)
    finally:
        reset.close()
        eval_env.close()
        collect.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "round1", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    args = parser.parse_args()
    run(args.source.resolve(), args.round1.resolve(), args.output.resolve())
