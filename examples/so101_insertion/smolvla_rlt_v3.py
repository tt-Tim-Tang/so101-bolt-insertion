"""Export 15 mm handoff starts and run the v3 zero-update RLT gate."""

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .scene import MOUTH_Z
from .smolvla25_data import FPS
from .smolvla_padding_ab import padding_velocity
from .smolvla_padding_e2e30 import EndToEndTelemetry
from .smolvla_rlt_stage2 import CHUNK, GRIP, STATE, ActionMap, FrozenSmolVLA
from .smolvla_rlt_td3 import load_rlinf


HANDOFF_LIMIT = 200
EPS_RAD = (.0045, .006, .0075, .0095, .013, 0.)
SIGMA = .0005
OUTCOMES = ("pickup", "arrival", "success", "slipped", "penetration_failure", "timeout")
COUNTERS = ("steps", "start_time", "pick", "arrive", "held_ticks", "highest_lift", "seated_ticks",
            "max_streak", "max_penetration", "open_count", "was_open", "drop_ticks", "slip_count")


def read_cases(episodes, baseline, splits):
    with episodes.open(newline="", encoding="utf-8-sig") as stream:
        test = list(csv.DictReader(stream))
    reference = json.loads(baseline.read_text())["rows"]
    data = json.loads(splits.read_text())
    for row, old in zip(test, reference, strict=True):
        if (int(row["index"]), int(row["seed"]), float(row["target_x_mm"]), float(row["target_y_mm"])) != (
            old["index"], old["seed"], old["target_x_mm"], old["target_y_mm"]
        ):
            raise ValueError("The held-out CSV and historical result differ")
    tests = np.array([[float(r["target_x_mm"]), float(r["target_y_mm"])] for r in test])
    validation = np.array([r["target"] for r in data["validation"]])
    train = np.array([r["target"] for r in data["train"]])
    if len(tests) != 30 or len(validation) != 10 or len(train) != 150:
        raise ValueError("Expected the 30 tests, 10 validation starts, and merged training data")
    reserved = np.vstack((tests, validation))
    if len(np.unique(reserved, axis=0)) != 40 or (np.linalg.norm(train[:, None] - reserved, axis=-1) <= 1e-6).any():
        raise ValueError("A training target overlaps the held-out or validation positions")
    return reference, data["train"]


def episode_result(info, timeout, done):
    return {key: info[key] for key in OUTCOMES if key != "timeout"} | {"timeout": bool(timeout and not done)}


def handoff(env, features, case, seed, *, frame=None):
    """Run the same five-command prefix and capture the first eligible boundary."""
    env.max_steps = 20 * FPS
    env.start(case)
    torch.manual_seed(seed)
    features.policy.reset()
    env.observation()  # Match the archived video's reset/render cadence.
    while True:
        if env.height_crossing is not None:
            step = env.steps
            crossing_step = math.ceil(env.height_crossing["elapsed_s"] * FPS - 1e-8)
            expected = math.ceil(crossing_step / CHUNK) * CHUNK
            if step != expected or env.first_contact is not None or not env.pickup_state()[-1]:
                return None, "Handoff did not occur on the first held, pre-contact replan boundary"
            physics = np.empty(mujoco.mj_stateSize(env.model, STATE))
            mujoco.mj_getState(env.model, env.data, physics, STATE)
            row = dict(case=case, seed=seed, counters={name: getattr(env, name) for name in COUNTERS},
                       crossing=env.height_crossing, crossing_step=crossing_step, step=step,
                       elapsed_s=float(env.data.time - env.start_time),
                       tip_height_mm=float((env.data.body(env.bolt_id).xpos[2] - MOUTH_Z) * 1000))
            return (row, physics, torch.get_rng_state(), torch.cuda.get_rng_state()), None
        if env.first_contact is not None:
            return None, "Contact occurred before a valid 15 mm handoff"
        raw = env.observation()
        frame_observation = raw
        for command in features.predict_reference(features.preprocess_observation(raw)).cpu().numpy():
            if frame is not None:
                frame(frame_observation)
            _, done, timeout, _ = env.step(command)
            if done or timeout:
                return None, "The prefix terminated before a valid 15 mm handoff"
            frame_observation = env.observation()


def restore_handoff(env, features, snapshot):
    """Restore a training snapshot without advancing physics or consuming inference RNG."""
    row, physics, cpu_rng, cuda_rng = snapshot
    env.start(row["case"])
    mujoco.mj_setState(env.model, env.data, physics, STATE)
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_setState(env.model, env.data, physics, STATE)
    for key, value in row["counters"].items():
        setattr(env, key, value)
    env.height_crossing = row["crossing"]
    env.reached_vicinity = True
    features.policy.reset()
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)


def read_snapshot(path, row):
    with np.load(path) as saved:
        return row, saved["physics"], torch.from_numpy(saved["cpu_rng"]), torch.from_numpy(saved["cuda_rng"])


def physical_commands(env, mapping, physical_reference, prediction, reference, *, lock_claw=True):
    # Apply only the learned difference to the original radian targets, preserving exact zero-residual actions.
    commands = np.clip(physical_reference, env.low, env.high)
    commands += ((prediction - reference) * mapping.half).cpu().numpy()
    if lock_claw:
        commands[:, -1] = GRIP
    return commands


def resume(env, features, mapping, actor, snapshot, *, lock_claw):
    """Execute a zero-update continuation with the same five-step policy cadence."""
    row = snapshot[0]
    env.max_steps = row["step"] + HANDOFF_LIMIT
    chunks = clipped = 0
    grip_change = 0.
    done = timeout = False
    while not (done or timeout):
        obs = features.observe(env, mapping, with_token=True, include_physical_reference=True)
        physical_reference = obs.pop("reference_rad").cpu().numpy().astype(np.float64)
        reference = obs["ref_chunk"]
        prediction, _, _ = actor.sac_forward({key: value.unsqueeze(0) for key, value in obs.items()},
                                              deterministic=True, apply_action_noise=False)
        prediction = prediction.reshape_as(reference)
        if not torch.equal(prediction, reference):
            raise RuntimeError("The zero-update Actor does not exactly match its reference")
        chunks += 1
        clipped += int(np.count_nonzero((physical_reference[:, :5] < env.low[:5]) |
                                         (physical_reference[:, :5] > env.high[:5])))
        if lock_claw:
            original_grip = np.clip(physical_reference[:, -1], env.low[-1], env.high[-1])
            grip_change = max(grip_change, float(np.max(np.abs(original_grip - GRIP))))
        command_chunk = physical_commands(env, mapping, physical_reference, prediction, reference, lock_claw=lock_claw)
        for command in command_chunk:
            _, done, timeout, info = env.step(command)
            if lock_claw and env.data.ctrl[-1] != GRIP:
                raise RuntimeError("The actuator did not receive the locked gripper target")
            if done or timeout or info["slipped"]:
                break
            env.observation()
        if info["slipped"]:
            break
    return dict(**episode_result(info, timeout, done), steps=env.steps, stage2_steps=env.steps - row["step"],
                stage2_chunks=chunks, max_gripper_lock_delta_rad=grip_change, arm_values_outside_actuator_limits=clipped,
                first_contact_s=None if env.first_contact is None else env.first_contact["elapsed_s"])


def verify_padding_and_snapshot(env, features, mapping, snapshot):
    """Check the shared sampler against the archived experiment and the recorded state restore."""
    batch = features.preprocess_observation(env.observation())
    shape = (1, features.policy.config.chunk_size, features.policy.config.max_action_dim)
    generator = torch.Generator(device="cuda").manual_seed(snapshot[0]["seed"])
    noise = torch.randn(shape, device="cuda", generator=generator)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        shared = features.policy.predict_action_chunk(dict(batch), noise=noise, correct_padding_velocity=True)
        with padding_velocity(features.policy, True, [], []):
            archived = features.policy.predict_action_chunk(dict(batch), noise=noise)
    torch.testing.assert_close(shared, archived, atol=0, rtol=0)
    before = features.observe(env, mapping, with_token=True)
    restore_handoff(env, features, snapshot)
    after = features.observe(env, mapping, with_token=True)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


@torch.inference_mode()
def run(args):
    prior, train = read_cases(args.episodes, args.baseline, args.splits)
    args.output.mkdir()
    features = FrozenSmolVLA(args.checkpoint)
    token_metadata = torch.load(args.token, weights_only=True, map_location="cpu")["metadata"]
    token_source = args.rlinf / "rlinf/models/embodiment/modules/rlt_token_transformer.py"
    features.load_token(args.token, token_source, args.checkpoint)
    env = EndToEndTelemetry(penetration_limit_mm=3.)
    try:
        mapping = ActionMap(env, sigma=SIGMA)
        eps_norm = (torch.tensor(EPS_RAD, device=mapping.half.device) / mapping.half).tolist()
        model, _ = load_rlinf(args.rlinf)
        template = yaml.safe_load((args.rlinf / "examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml").read_text())
        actor_config = dict(z_dim=token_metadata["model"]["embed_dim"], proprio_dim=len(EPS_RAD), action_dim=len(EPS_RAD),
                            num_action_chunks=CHUNK, residual_eps_norm=eps_norm, actor_noise_sigma=SIGMA,
                            ref_action_dropout=0., **{key: template["actor"]["model"][key]
                            for key in ("mlp_hidden_dim", "mlp_num_hidden_layers")})
        torch.manual_seed(template["actor"]["seed"])
        actor = model(**actor_config).cuda().eval()
        report = dict(updates=0, eps_rad=EPS_RAD, actor_config=actor_config, mapping=mapping.report(),
                      training_snapshots=[], rows=[], stage2_horizon=HANDOFF_LIMIT)
        if args.export_train:
            folder = args.output / "train-starts"
            folder.mkdir()
            for idx in sorted(set(args.export_train)):
                if not 0 <= idx < len(train):
                    raise ValueError(f"Training index is out of range: {idx}")
                snapshot, rejection = handoff(env, features, train[idx], 714290 + idx)
                entry = dict(index=idx, target=train[idx]["target"], rejection=rejection)
                if snapshot is not None:
                    row, state, cpu_rng, cuda_rng = snapshot
                    filename = folder / f"{idx:03d}.npz"
                    np.savez_compressed(filename, physics=state, cpu_rng=cpu_rng.numpy(), cuda_rng=cuda_rng.numpy())
                    verify_padding_and_snapshot(env, features, mapping, read_snapshot(filename, row))
                    entry.update(row=row, file=filename.name, exact_restored_reference=True, padding_matches_ab=True)
                report["training_snapshots"].append(entry)
            (folder / "manifest.json").write_text(json.dumps(report["training_snapshots"], indent=2) + "\n")
        if args.export_only:
            return

        with (args.output / "episodes.jsonl").open("x", buffering=1) as log:
            for old in prior:
                case = dict(target=[old["target_x_mm"], old["target_y_mm"]], source=[0, 0], joints=[0]*5, noise=old["seed"])
                snapshot, rejection = handoff(env, features, case, old["seed"])
                baseline = {key: old[key] for key in (*OUTCOMES, "steps")}
                entry = dict(index=old["index"], historical=baseline, rejection=rejection)
                if snapshot is not None:
                    snap = snapshot[0]
                    entry["handoff"] = {key: snap[key] for key in ("crossing", "step", "elapsed_s", "tip_height_mm")}
                    result = resume(env, features, mapping, actor, snapshot, lock_claw=True)
                    entry.update(locked=result, same_outcome=all(result[key] == old[key] for key in OUTCOMES))
                    if not entry["same_outcome"]:
                        restore_handoff(env, features, snapshot)
                        unlocked = resume(env, features, mapping, actor, snapshot, lock_claw=False)
                        entry.update(unlocked=unlocked, lock_changed_outcome=any(result[k] != unlocked[k] for k in OUTCOMES),
                                     unlocked_matches_history=all(unlocked[k] == old[k] for k in OUTCOMES))
                report["rows"].append(entry)
                print(json.dumps(entry), file=log)
                print("GATE", old["index"], None if snapshot is None else entry["locked"]["success"], rejection,
                      entry.get("same_outcome"), flush=True)
        rows = report["rows"]
        report.update(episodes=len(rows), valid_handoffs=sum("locked" in r for r in rows),
                      insertions=sum(r.get("locked", {}).get("success", False) for r in rows),
                      slips=sum(r.get("locked", {}).get("slipped", 0) for r in rows),
                      exact_actor_chunks=sum(r.get("locked", {}).get("stage2_chunks", 0) for r in rows),
                      mismatches=[r["index"] for r in rows if not r.get("same_outcome", False)])
        report["gate_passed"] = not report["mismatches"] and report["insertions"] == 21 and report["slips"] == 0
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GATE_COMPLETE", report["gate_passed"], report["insertions"], report["slips"], report["mismatches"], flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "token", "rlinf", "episodes", "baseline", "splits", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--export-train", nargs="*", type=int, default=[])
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.export_only and not args.export_train:
        parser.error("--export-only requires at least one training index")
    run(args)


if __name__ == "__main__":
    main()
