"""Read-only SO-101 delta-control, exploration and terminal-reward audit."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch

from .scene import JOINTS
from .smolvla25_data import SmolVLAInsertion
from .smolvla_rlt_offline import MAX_STEPS, SIGMA, demonstration_limits, evaluation_starts
from .smolvla_rlt_stage2 import CHUNK, GRIP, STATE, ActionMap, FrozenSmolVLA, restore, save_json
from .smolvla_rlt_td3 import load_rlinf


INSERTION_FRAME = 154  # 129 ticks to end transfer, then 25 ticks of air alignment.
EPS = 1e-8


def demonstrations(root):
    data = pq.read_table(
        [str(path) for path in sorted((root / "data").glob("*/*.parquet"))],
        columns=["episode_index", "frame_index", "action", "observation.state"],
    )
    ep, frame, cmd, q = (np.asarray(data[key].to_pylist()) for key in data.column_names)
    order = np.lexsort((frame, ep))
    ep, frame, cmd, q = ep[order], frame[order], cmd[order, :5], q[order, :5]
    assert len(set(ep)) == 150 and len(ep) == 28649
    mask = (ep[1:] == ep[:-1]) & (frame[:-1] >= INSERTION_FRAME)
    assert np.all(np.diff(frame)[mask] == 1)
    command_step = np.abs(np.diff(cmd, axis=0)[mask])
    actual_step = np.abs(np.diff(q, axis=0)[mask])
    tracking = np.abs((cmd - q)[1:][mask])
    limits = command_step.max(axis=0)
    red = np.abs(tracking) > limits + EPS
    return limits, dict(
        joints=list(JOINTS[:5]), frame_pairs=len(command_step), max_delta=limits.tolist(),
        observed_joint_step_max_rad=actual_step.max(axis=0).tolist(),
        qpos_referenced_delta_p95_rad=np.quantile(tracking, .95, axis=0).tolist(),
        qpos_referenced_delta_max_rad=tracking.max(axis=0).tolist(),
        qpos_reference_clip_any_fraction=float(red.any(axis=1).mean()),
        qpos_reference_clip_fraction_per_joint=red.mean(axis=0).tolist(),
    )


def command(mode, previous, qpos, desired, low, high, limit, normalized=None):
    anchor = previous if mode == "target" else qpos
    request = (desired - anchor) / limit if normalized is None else normalized
    delta = np.clip(request, -1, 1)
    target = np.clip(anchor + delta * limit, low, high)
    return target, request, delta


def baseline(env, features, mapping, starts, limits, output):
    low, high = mapping.low.cpu().numpy()[:5], mapping.high.cpu().numpy()[:5]
    scratch = mujoco.MjData(env.model)

    def wrist_z(q):
        scratch.qpos[env.joints[:5]] = q
        mujoco.mj_kinematics(env.model, scratch)
        return float(scratch.site("grasp").xpos[2])

    results = {}
    for mode in ("absolute", "target", "qpos"):
        episodes = []
        all_motion = []
        with (output / f"{mode}-steps.jsonl").open("x", encoding="utf-8") as log:
            for start in starts:
                restore(env, start)
                index = start[0]["index"]
                torch.manual_seed(start[0]["seed"])
                features.policy.reset()
                counters = dict(clipped_steps=0, nominal_down_steps=0, clipped_nominal_down_steps=0,
                                nominal_down_not_followed_steps=0, consecutive_targets_over_limit=0)
                previous_goal = env.data.ctrl[:5].copy()
                grasp = env.data.site("grasp")
                done = False
                motion = []
                while not done:
                    obs = features.observe(env, mapping, with_token=False)
                    actions = mapping.physical(obs["ref_chunk"]).cpu().numpy().astype(np.float64)
                    for goal in actions:
                        before = grasp.xpos.copy()
                        previous = env.data.ctrl[:5].copy()
                        qpos = env.data.qpos[env.joints[:5]].copy()
                        nominal_down = bool(wrist_z(goal[:5]) < wrist_z(previous_goal) - .0001
                                            and np.all(np.abs(goal[:5] - previous_goal) <= limits + EPS))
                        if mode == "absolute":
                            target = goal.copy()
                            clipped = False
                        else:
                            target = goal.copy()
                            target[:5], request, _ = command(mode, previous, qpos, goal[:5], low, high, limits)
                            clipped = bool(np.any(np.abs(request) > 1 + EPS))
                        target[-1] = GRIP
                        changed = np.abs(target[:5] - previous) > limits + EPS
                        if mode == "target" and changed.any():
                            raise RuntimeError("Target-anchored Delta exceeded its per-tick cap")
                        counters["clipped_steps"] += clipped
                        counters["nominal_down_steps"] += nominal_down
                        counters["clipped_nominal_down_steps"] += nominal_down and clipped
                        counters["nominal_down_not_followed_steps"] += (
                            nominal_down and wrist_z(target[:5]) >= wrist_z(previous)
                        )
                        counters["consecutive_targets_over_limit"] += changed.any()
                        _, stopped, timeout, info = env.step(target)
                        done = bool(stopped or timeout or info["slipped"])
                        move = float(np.linalg.norm(grasp.xpos - before) * 1000)
                        motion.append(move)
                        print(json.dumps(dict(
                            index=index, step=env.steps, desired=goal.tolist(), applied=target.tolist(),
                            qpos=qpos.tolist(), wrist_mm=move, clipped=clipped, nominal_down=nominal_down,
                        )), file=log)
                        previous_goal = goal[:5].copy()
                        if done:
                            break
                episode = dict(index=index, success=info["success"], slipped=info["slipped"],
                               steps=env.steps, penetration_failure=info["penetration_failure"],
                               wrist_peak_mm=max(motion), wrist_peak_step=int(np.argmax(motion)) + 1,
                               **{k: int(v) for k, v in counters.items()})
                print(mode, json.dumps(episode), flush=True)
                episodes.append(episode)
                all_motion.extend(motion)
        results[mode] = dict(insertions=sum(row["success"] for row in episodes),
                             closed_gripper_slips=sum(row["slipped"] for row in episodes),
                             wrist_mm=np.quantile(all_motion, [.5, .95, 1]).tolist(),
                             rows=episodes, **{k: sum(row[k] for row in episodes) for k in counters})
        save_json(output / f"{mode}.json", results[mode])
    return results


def calibrate(env, mapping, starts, eval_folder, limits, mode, output,
              sigmas=(0.025, 0.05, 0.075, 0.10, 0.15, 0.20)):
    low, high = mapping.low.cpu().numpy()[:5], mapping.high.cpu().numpy()[:5]
    n_seeds = 48
    samples = np.random.default_rng(723).standard_normal((n_seeds, CHUNK, 5))
    shifts = {sigma: [] for sigma in sigmas}
    zero = np.zeros((CHUNK, 5))
    for start in starts:
        restore(env, start)
        row, physics, _ = start
        with np.load(eval_folder / f'{row["index"]:03}.npz') as source:
            expert_actions = source["expert_actions"][:CHUNK, :5]

        def execute(perturbation, nominal=None):
            mujoco.mj_setState(env.model, env.data, physics, STATE)
            mujoco.mj_forward(env.model, env.data)
            tip = env.data.body("bolt")
            deltas, points = [], []
            for i, goal in enumerate(expert_actions):
                normalized = None if nominal is None else nominal[i] + perturbation[i]
                qpos = env.data.qpos[env.joints[:5]]
                target, _, delta = command(mode, env.data.ctrl[:5], qpos, goal, low, high, limits, normalized)
                env.data.ctrl[:5], env.data.ctrl[-1] = target, GRIP
                mujoco.mj_step(env.model, env.data, nstep=env.substeps)
                mujoco.mj_forward(env.model, env.data)
                deltas.append(delta)
                points.append(tip.xpos[:2].copy())
            return np.array(points), np.array(deltas)

        reference, nominal = execute(zero)
        clean, _ = execute(zero, nominal)
        np.testing.assert_allclose(clean, reference, rtol=0, atol=1e-12)
        for sigma in sigmas:
            for z in samples:
                points, _ = execute(z * sigma, nominal)
                shifts[sigma].append(float(np.linalg.norm(points - reference, axis=1).max() * 1000))
        print("NOISE_CASE_COMPLETE", row["index"], flush=True)
    result = dict(mode=mode, starts=len(starts), seeds_per_start=n_seeds, rows=[
        dict(sigma=sigma, peak_xy_mm=np.quantile(values, [.5, .95, 1]).tolist(),
             joint_sigma_rad=(limits * sigma).tolist()) for sigma, values in shifts.items()
    ])
    save_json(output / "noise.json", result)


def check_reward(root, output):
    _, mixin = load_rlinf(root)

    class RewardProbe(mixin):
        torch_dtype = torch.float32

        def _env_values(self, key, *_):
            return self.cfg.algorithm[key]

        def _next_actions_for_critic_target(self, _):
            return None, None, None

    probe = RewardProbe()
    probe.cfg = SimpleNamespace(algorithm=dict(bootstrap_type="standard", q_head_type="default", gamma=.99))
    probe.target_model = lambda **_: torch.full((CHUNK, 2), 3.)
    probe.model = lambda **_: torch.zeros((CHUNK, 2))
    rewards = -.5 * torch.eye(CHUNK)
    with torch.no_grad():
        loss, _ = probe.forward_critic(dict(
            curr_obs={}, next_obs={}, actions=None, rewards=rewards,
            dones=torch.ones(CHUNK, dtype=torch.bool), terminations=torch.ones(CHUNK, dtype=torch.bool),
        ))
    expected = -.5 * .99 ** torch.arange(CHUNK)
    torch.testing.assert_close(loss, expected.square().mean())
    save_json(output / "reward.json", dict(
        slip_reward=-.5, gamma=.99, tested_terminal_targets=expected.tolist(), critic_loss=loss.item(),
        bootstrap_after_terminal=False, parameter_updates=0,
    ))


def run(source, output):
    output.mkdir()
    config = json.loads((source / "run-config.json").read_text(encoding="utf-8"))
    root = Path(config["training_dataset"])
    folder = Path(config["eval_starts"])
    limits, demos = demonstrations(root)
    save_json(output / "demonstrations.json", demos)
    check_reward(Path(config["rlinf"]), output)
    env = SmolVLAInsertion(penetration_limit_mm=3.)
    env.max_steps = MAX_STEPS
    try:
        mapping = ActionMap(env, demonstration_limits(root), SIGMA)
        if mapping.report() != config["joints"]:
            raise ValueError("Unexpected absolute actuator guard ranges")
        starts = evaluation_starts(folder)
        features = FrozenSmolVLA(Path(config["smolvla"]))
        baseline(env, features, mapping, starts, limits, output)
        calibrate(env, mapping, starts, folder, limits, "target", output)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.source, arguments.output)
