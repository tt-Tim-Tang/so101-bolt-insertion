"""Train the unit-residual RLT policy on the full training split and evaluate held-out holes."""

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .smolvla25_data import CAMERAS, FPS
from .smolvla_padding_e2e30 import failure_mode, quantiles
from .smolvla_rlt_online import Replay, log_row, write_checkpoint
from .smolvla_rlt_stage2 import ActionMap, FrozenSmolVLA
from .smolvla_rlt_td3 import batch_replay, learner, load_rlinf, take_batch, train_step
from .smolvla_rlt_v3 import SIGMA, episode_result, handoff, read_snapshot
from .smolvla_rlt_v3_mixed import Exploration, q_by_outcome, scan_unit_critic
from .smolvla_rlt_v3_pool import VALIDATION_SEED, EpisodeTelemetry, fixed_cases, save_json
from .smolvla_rlt_v3_short import NoiseStream, rollout

SIGMA_RAW = .0015
CRITIC_ONLY = 2500
ONLINE = 10_000
INTERVAL = 2500
GROUPS = ("base_fail", "base_success")


def summary(rows):
    return dict(n=len(rows), insertions=sum(r["success"] for r in rows),
                slips=sum(r["slipped"] for r in rows), slip_episodes=sum(bool(r["slipped"]) for r in rows),
                timeouts=sum(r["timeout"] for r in rows),
                penetration_failures=sum(r["penetration_failure"] for r in rows),
                mean_steps=float(np.mean([r["steps"] for r in rows])))


def end_to_end(env, features, mapping, policy, case, seed, video=None):
    writer = None

    def frame(observation):
        nonlocal writer
        image = np.concatenate([observation[key] for key in CAMERAS], axis=1)
        if writer is None:
            size = (image.shape[1], image.shape[0])
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video {video}")
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with torch.inference_mode():
                snap, rejection = handoff(env, features, case, seed, frame=frame if video else None)
            continuation = None
            if snap is not None:
                continuation = rollout(env, features, mapping, policy, snap, live_handoff=True,
                                       frame=frame if video else None)
            elif video:
                frame(env.observation())
        _, stopped, timeout, info = env.last_step
        outcome = episode_result(info, timeout, stopped)
        if continuation is not None and continuation["slip"]:
            outcome["timeout"] = False
        contact = env.first_contact or {}
        crossing = env.height_crossing or {}
        wrist = quantiles(env.wrist_steps_mm[contact["step"]:] if contact else [])
        mode, detail = failure_mode(info, env)
        return dict(**outcome, steps=env.steps, seed=seed, x_mm=case["target"][0], y_mm=case["target"][1],
                    rejection=rejection, handoff_step=None if snap is None else snap[0]["step"],
                    crossing_xy_mm=crossing.get("xy_mm"), crossing_tilt_deg=crossing.get("tilt_deg"),
                    first_contact_step=contact.get("step"), first_contact_xy_mm=contact.get("xy_mm"),
                    first_contact_tilt_deg=contact.get("tilt_deg"), first_contact_time_s=contact.get("elapsed_s"),
                    max_held_depth_mm=env.max_near_depth_mm, longest_seated_ms=1000 * info["stable_s"],
                    max_penetration_mm=info["max_penetration_mm"], wrist_after_contact_p95_mm=wrist["p95"],
                    wrist_after_contact_max_mm=wrist["max"], first_slip_step=env.first_slip_step,
                    failure_mode=mode, failure_detail=detail,
                    mean_abs_residual_mrad=None if continuation is None else 1e3 * continuation["mean_residual_rad"],
                    max_abs_residual_mrad=None if continuation is None else 1e3 * continuation["max_residual_rad"])
    finally:
        if writer is not None:
            writer.release()


def heldout_eval(output, env, features, mapping, policy, held, zero):
    output.mkdir(exist_ok=True)
    videos = output / "failures"
    videos.mkdir(exist_ok=True)
    rows = []
    for old in held:
        i = old["index"]
        saved = output / f"{i:02d}.json"
        if saved.is_file():
            row = json.loads(saved.read_text())
        else:
            case = dict(target=[old["target_x_mm"], old["target_y_mm"]], source=[0, 0],
                        joints=[0] * 5, noise=old["seed"])
            scratch = output / "active.mp4"
            row = dict(index=i, **end_to_end(env, features, mapping, policy, case, old["seed"], scratch))
            row.update(zero_success=bool(zero[i]["success"]), zero_slipped=zero[i]["slipped"],
                       zero_timeout=zero[i]["timeout"], rescued=bool(row["success"] and not zero[i]["success"]),
                       lost=bool(zero[i]["success"] and not row["success"]), video=None)
            if not row["success"]:
                filename = videos / f"{i:02d}.mp4"
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(scratch),
                                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(filename)], check=True)
                row["video"] = str(filename.relative_to(output))
            scratch.unlink()
            save_json(saved, row)
        rows.append(row)
        print("HELDOUT", output.name, i, row["success"], row["slipped"], row["timeout"], flush=True)
    for name, selected in (("episodes.csv", rows), ("failures.csv", [r for r in rows if not r["success"]])):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    result = dict(**summary(rows), rescued=[r["index"] for r in rows if r["rescued"]],
                  lost=[r["index"] for r in rows if r["lost"]], rows=rows)
    save_json(output / "summary.json", result)
    return {k: v for k, v in result.items() if k != "rows"}


def grouped(rng, pools, episode):
    group = GROUPS[episode % len(GROUPS)]
    index, snap = rng.choice(pools[group])
    return group, index, snap


def load_pools(folder, train):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = manifest["rows"]
    if len(rows) != 150 or [(r["index"], r["case"]) for r in rows] != list(enumerate(train)):
        raise ValueError("Snapshot manifest differs from the verified training split")
    pools = {group: [] for group in GROUPS}
    for row in rows:
        if row["snapshot"] is not None:
            index = row["index"]
            if row["snapshot"]["case"] != train[index]:
                raise ValueError(f"Snapshot {index} has the wrong physical reset")
            snap = read_snapshot(folder / f"{index:03d}.npz", row["snapshot"])
            pools[row["group"]].append((index, snap))
    if any(len(pools[group]) < 10 for group in GROUPS):
        raise RuntimeError("Fewer than ten pre-contact snapshots in a required training stratum")
    return manifest, pools


def warmup(output, env, features, mapping, policy, pools, explore, rng):
    checkpoint = output / "warmup.pt"
    if checkpoint.is_file():
        return torch.load(checkpoint, weights_only=False, map_location="cpu")
    blocks, rows, labels = [], [], []
    while len(rows) < 200:
        group, index, snap = grouped(rng, pools, len(rows))
        offset = len(blocks)
        result = rollout(env, features, mapping, policy, snap, explore=explore if group == "base_fail" else None,
                         transition=blocks.append)
        labels.extend([result["success"]] * (len(blocks) - offset))
        rows.append(dict(group=group, index=index, offset=offset,
                         positive_blocks=sum(bool(r["rewards"].any()) for r in blocks[offset:]), **result))
        if len(rows) % 10 == 0:
            print("WARMUP_PROGRESS", len(rows), len(blocks), sum(r["positive_blocks"] for r in rows), flush=True)
        if len(rows) >= 160 and len(rows) % 2 == 0 and len(blocks) >= 3500:
            break
    positive = sum(r["positive_blocks"] for r in rows)
    result = dict(episodes=len(rows), blocks=len(blocks), positive_blocks=positive,
                  positive_fraction=positive / len(blocks), block_target_met=3500 <= len(blocks) <= 4500,
                  groups={g: dict(episodes=sum(r["group"] == g for r in rows),
                                  insertions=sum(r["group"] == g and r["success"] for r in rows),
                                  positive_blocks=sum(r["positive_blocks"] for r in rows if r["group"] == g))
                          for g in GROUPS}, rows=rows)
    replay = Replay(batch_replay(blocks, "cuda"), len(blocks), "cuda")
    saved = dict(replay=replay.snapshot(), labels=labels, rng=rng.getstate(), exploration_rng=explore.rng.get_state(),
                 summary=result)
    write_checkpoint(checkpoint, saved)
    save_json(output / "warmup.json", result)
    print("WARMUP_COMPLETE", len(rows), len(blocks), positive, flush=True)
    return saved


def gradient_summary(metrics):
    return {key: quantiles([row[key] for row in metrics if key in row])
            for key in ("grad_norm_q", "grad_norm_bc", "grad_norm_bc_over_q", "critic_loss", "actor_loss")}


def validation_rank(check):
    val = check["validation"]
    return val["insertions"], -val["slips"], -val["mean_steps"], -check["step"]


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    held, train, validation = fixed_cases(args)
    gate = json.loads(args.gate.read_text())
    zero = {row["index"]: row["locked"] for row in gate["rows"]}
    if set(zero) != {r["index"] for r in held} or sum(z["success"] for z in zero.values()) != 20 or \
            sum(z["slipped"] for z in zero.values()) != 1:
        raise ValueError("Expected the exact historical v3 20/30 locked-gripper base results")
    export, pools = load_pools(args.pool, train)
    default = yaml.safe_load((args.rlinf / "examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml").read_text())
    actor_config, algorithm = default["actor"], default["algorithm"]
    group_size = algorithm["rlt_schedule"]["train_every_transitions"]
    if group_size != 5 or INTERVAL % group_size:
        raise ValueError("This run expects the short test's UTD: five updates per five collected blocks")
    algorithm.update(actor_weight_schedule={"enable": False}, reference_dropout_prob=0.,
                     critic_in_residual_space=True, residual_unit_bc=True, log_actor_gradient_terms=True,
                     bc_weight=.10, q_weight=.50)
    config = gate["actor_config"] | dict(critic_in_residual_space=True, actor_noise_sigma=SIGMA_RAW)
    rng = random.Random(20261204)
    starts_rng = random.Random(20261206)
    monitor_rng = random.Random(20261208)
    monitor = {g: monitor_rng.sample(pools[g], 10) for g in GROUPS}
    update_rng = NoiseStream(20261205)
    explore = Exploration("block", SIGMA_RAW)
    features = FrozenSmolVLA(args.checkpoint)
    features.load_token(args.token, args.rlinf / "rlinf/models/embodiment/modules/rlt_token_transformer.py",
                        args.checkpoint)
    env = EpisodeTelemetry(penetration_limit_mm=3.)
    eval_env = EpisodeTelemetry(penetration_limit_mm=3.)
    try:
        mapping = ActionMap(env, sigma=SIGMA)
        model, mixin = load_rlinf(args.rlinf)
        torch.manual_seed(actor_config["seed"])
        loss, optimizer = learner(model, mixin, mapping, config, algorithm, actor_config, "cuda")
        policy = loss.model
        setup = dict(sigma_raw=SIGMA_RAW, bc_weight=.10, q_weight=.50, utd=[group_size, group_size],
                     seed_validation=VALIDATION_SEED, critic_warmup_updates=CRITIC_ONLY, online_updates=ONLINE,
                     eval_interval=INTERVAL, batch_size=actor_config["global_batch_size"],
                     reset_ratio=dict.fromkeys(GROUPS, .5),
                     monitor={g: [index for index, _ in cases] for g, cases in monitor.items()},
                     valid_handoffs=export["valid_handoffs"], training_pool={g: len(pools[g]) for g in GROUPS})
        save_json(output / "config.json", setup)
        print("RUN_CONFIG", setup, flush=True)
        warm = warmup(output, env, features, mapping, policy, pools, explore, starts_rng)
        warm_n = warm["summary"]["blocks"]
        labels = warm["labels"]
        replay = Replay(warm["replay"], warm_n + ONLINE, "cuda")
        starts_rng.setstate(warm["rng"])
        explore.rng.set_state(warm["exploration_rng"])
        del warm["replay"]

        def sample():
            return take_batch(replay.data, rng.choices(range(replay.count), k=actor_config["global_batch_size"]))

        def update(step, *, actor=True):
            return update_rng(train_step, loss, optimizer, actor_config, sample(), step, update_actor=actor)

        positive = dict.fromkeys(GROUPS, 0)
        step = episode = 0

        def checkpoint(path):
            write_checkpoint(path, dict(model=policy.state_dict(), target=loss.target_model.state_dict(),
                             critic_optimizer=loss.qf_optimizer.state_dict(), actor_optimizer=optimizer.state_dict(),
                             model_config=config, algorithm=loss.cfg.algorithm, replay=replay.snapshot(),
                             labels=labels, rng=rng.getstate(), starts_rng=starts_rng.getstate(),
                             exploration_rng=explore.rng.get_state(), update_rng=update_rng.state,
                             critic_only_updates=CRITIC_ONLY, online_updates=step, episode=episode, positive=positive))

        def restore(path):
            nonlocal replay, labels, step, episode, positive
            saved = torch.load(path, weights_only=False, map_location="cpu")
            policy.load_state_dict(saved["model"])
            loss.target_model.load_state_dict(saved["target"])
            loss.qf_optimizer.load_state_dict(saved["critic_optimizer"])
            optimizer.load_state_dict(saved["actor_optimizer"])
            replay = Replay(saved["replay"], warm_n + ONLINE, "cuda")
            labels, step, episode, positive = (saved[k] for k in ("labels", "online_updates", "episode", "positive"))
            rng.setstate(saved["rng"])
            starts_rng.setstate(saved["starts_rng"])
            explore.rng.set_state(saved["exploration_rng"])
            update_rng.state = saved["update_rng"]

        pure = output / "critic-only.pt"
        if pure.is_file():
            restore(pure)
        else:
            policy.actor.requires_grad_(False)
            with (output / "critic-only.jsonl").open("w", buffering=1) as log:
                for i in range(CRITIC_ONLY):
                    row = update(i, actor=False)
                    log_row(log, row)
                    if (i + 1) % 500 == 0:
                        print("CRITIC_PROGRESS", i + 1, row["critic_loss"], flush=True)
            checkpoint(pure)

        preheat_file = output / "critic-only-summary.json"
        if not preheat_file.is_file():
            q = q_by_outcome(policy, replay, labels, mapping)
            probes = [r for r in warm["summary"]["rows"] if r["group"] == "base_fail"]
            distinct = list({r["index"]: r for r in probes}.values())
            scans = []
            for row in random.Random(20261209).sample(distinct, 3):
                scan = scan_unit_critic(policy, take_batch(replay.data["curr_obs"], [row["offset"]]))
                scans.append(scan | {"index": row["index"], "replay_offset": row["offset"]})
            save_json(preheat_file, dict(updates=CRITIC_ONLY, q=q, scans=scans))
            print("CRITIC_COMPLETE", q["positive_trajectory"]["replay_q"], q["negative_trajectory"]["replay_q"],
                  [(s["index"], s["max_scan_span"]) for s in scans], flush=True)
        policy.actor.requires_grad_(True)

        checkpoints = sorted(output.glob("online-[0-9]*.pt"))
        if checkpoints:
            restore(checkpoints[-1])
        checks = [json.loads(path.read_text()) for path in sorted(output.glob("evaluation-*.json"))
                  if int(path.stem.rsplit("-", 1)[-1]) <= step]

        def evaluate(metrics):
            check_path = output / f"evaluation-{step:05d}.json"
            if check_path.is_file():
                return
            queues = features.policy._queues
            try:
                train_report = {}
                for group, cases in monitor.items():
                    rows = [dict(index=i, **rollout(eval_env, features, mapping, policy, snap)) for i, snap in cases]
                    train_report[group] = dict(n=len(rows), insertions=sum(r["success"] for r in rows),
                                               slips=sum(r["slip"] for r in rows), rows=rows)
                val = [dict(index=i, **end_to_end(eval_env, features, mapping, policy, case, VALIDATION_SEED + i))
                       for i, case in enumerate(validation)]
            finally:
                features.policy._queues = queues
            q = q_by_outcome(policy, replay, labels, mapping)
            record = dict(step=step, training=train_report, validation=summary(val) | {"rows": val},
                          replay_q=q, gradient=gradient_summary(metrics),
                          mean_abs_residual_mrad=q["mean_residual_rad"] * 1000,
                          max_abs_residual_mrad=q["max_residual_rad"] * 1000, online_positive_by_group=positive.copy())
            save_json(check_path, record)
            checks.append(record)
            best = max(checks, key=validation_rank)
            save_json(output / "best.json", dict(step=best["step"], checkpoint=f"online-{best['step']:05d}.pt",
                                                validation=best["validation"]))
            print("EVAL", step, record["validation"]["insertions"],
                  {g: t["insertions"] for g, t in train_report.items()}, "BEST", best["step"], flush=True)

        def metrics_since_checkpoint():
            path = output / "updates.jsonl"
            rows = {}
            if path.is_file():
                with path.open() as stream:
                    for line in stream:
                        if not line.endswith("\n"):
                            break
                        row = json.loads(line)
                        if step - INTERVAL < row["online_update"] <= step:
                            rows[row["online_update"]] = row
            return list(rows.values())

        if step:
            evaluate(metrics_since_checkpoint())
        print("ONLINE_START", step, "UTD", group_size, group_size, flush=True)
        with (output / "updates.jsonl").open("a", buffering=1) as log, \
                (output / "online-episodes.jsonl").open("a", buffering=1) as episode_log:
            metrics = []

            def add(row):
                nonlocal step
                replay.add(row)
                labels.append(None)
                positive[group] += int(bool(row["rewards"].any()))
                if (replay.count - warm_n) % group_size == 0:
                    for _ in range(group_size):
                        metric = update(CRITIC_ONLY + step)
                        step += 1
                        metric["online_update"] = step
                        log_row(log, metric)
                        metrics.append(metric)
                    if step % 100 == 0:
                        print("ONLINE_PROGRESS", step, replay.count - warm_n, positive, metric["critic_loss"],
                              flush=True)
                    if step % INTERVAL == 0:
                        checkpoint(output / f"online-{step:05d}.pt")
                        evaluate(metrics)
                        metrics.clear()
                return step == ONLINE

            while step < ONLINE:
                group, index, snap = grouped(starts_rng, pools, episode)
                first = replay.count
                episode += 1
                row = rollout(env, features, mapping, policy, snap, explore=explore, transition=add)
                if not row["budget_truncated"]:
                    labels[first:replay.count] = [row["success"]] * (replay.count - first)
                log_row(episode_log, dict(episode=episode, group=group, index=index, online_update=step, **row))
        best = max(checks, key=validation_rank)["step"]
        tests = {}
        for weight_step in sorted({best, ONLINE}):
            state = torch.load(output / f"online-{weight_step:05d}.pt", weights_only=False, map_location="cpu")
            policy.load_state_dict(state["model"])
            del state
            tests[weight_step] = heldout_eval(output / f"heldout-{weight_step:05d}", eval_env, features, mapping, policy,
                                              held, zero)
        save_json(output / "summary.json", dict(setup=setup, export={k: v for k, v in export.items() if k != "rows"},
                  warmup={k: v for k, v in warm["summary"].items() if k != "rows"},
                  critic_only=json.loads(preheat_file.read_text()), evaluations=checks,
                  online=dict(updates=step, blocks=replay.count - warm_n, episodes=episode,
                              positive_by_group=positive), baseline=summary(list(zero.values())),
                  best=dict(step=best, **tests[best]), final=dict(step=ONLINE, **tests[ONLINE])))
        print("FULL_COMPLETE", best, tests, flush=True)
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        if any(term in str(exc).lower() for term in ("nonfinite", "non-finite", "nan")):
            save_json(output / "fuse.json", dict(reason=str(exc), stage="td3", online_updates=locals().get("step")))
            print("NUMERICAL_FUSE", str(exc), flush=True)
        raise
    finally:
        eval_env.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "token", "rlinf", "episodes", "baseline", "splits", "output", "gate", "pool"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
