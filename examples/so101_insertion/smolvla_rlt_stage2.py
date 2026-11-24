"""Run the SO-101 insertion-only evaluation and the RLT-TD3 gate."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .scene import JOINTS, MOUTH_Z
from .smolvla25_data import CAMERAS, SmolVLAInsertion, trial
from .train_smolvla_rlt import extract_prefix, load_rlt_class


CHUNK = 5
GRIP = -.055  # Original expert's held-bolt command, in radians.
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
NOISE_SIGMA = .1  # RLinf ManiSkill TD3 actor setting, in normalized units.


class ActionMap:
    def __init__(self, env):
        self.low = torch.as_tensor(env.low, dtype=torch.float32, device="cuda")
        self.high = torch.as_tensor(env.high, dtype=torch.float32, device="cuda")
        self.half = (self.high - self.low) / 2
        self.mid = (self.high + self.low) / 2
        self.closed = (GRIP - self.mid[-1]) / self.half[-1]
        if not bool((self.half > 0).all() and self.closed.abs() <= 1):
            raise ValueError("Invalid SO-101 actuator limits or lock command")

    def normalize(self, physical):
        physical = physical.to(self.mid)
        normalized = ((physical - self.mid) / self.half).clamp(-1, 1)
        return self.lock(normalized)

    def lock(self, normalized):
        return torch.cat((normalized[..., :5], self.closed.expand_as(normalized[..., 5:])), dim=-1)

    def physical(self, normalized):
        command = self.mid + self.lock(normalized).clamp(-1, 1) * self.half
        command[..., -1] = GRIP
        return command

    def report(self):
        return [
            dict(joint=name, min_rad=low, max_rad=high, raw_noise_sigma_rad=NOISE_SIGMA * (high - low) / 2,
                 executed_noise_sigma_rad=NOISE_SIGMA * (high - low) / 2 if i < 5 else 0.)
            for i, (name, low, high) in enumerate(zip(JOINTS, self.low.tolist(), self.high.tolist(), strict=True))
        ]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value), flush=True)


def collect_starts(env, cases, folder, label):
    folder.mkdir()
    starts = []
    for index, case in cases:
        state = None
        actions = []

        def record(action):
            nonlocal state
            if state is None and env.phase == "insert":
                tip = env.data.body("bolt").xpos.copy()
                offset = np.r_[tip[:2] - env.model.body("target").pos[:2], tip[2] - MOUTH_Z]
                if not env.pick or not env.pickup_state()[-1] or not 0 < offset[2] <= .010:
                    raise RuntimeError(f"Invalid held start at {label}/{index}: {offset.tolist()}")
                state = np.empty(mujoco.mj_stateSize(env.model, STATE))
                mujoco.mj_getState(env.model, env.data, state, STATE)
                row.update(tip_offset_mm=(offset * 1000).tolist(), tilt_rad=env.info()["tilt_rad"])
            if state is not None:
                actions.append(action.copy())

        row = dict(index=index, case=case, seed=714290 + index + (10000 if label == "baseline" else 0))
        result = trial(env, case, record)
        if state is None:
            raise RuntimeError(f"Original expert did not reach the insertion start at {label}/{index}: {result}")
        expert_actions = np.asarray(actions, dtype=np.float32)
        np.savez(folder / f"{index:03}.npz", physics=state, expert_actions=expert_actions)
        row["expert_success"] = result["success"]
        starts.append((row, state, expert_actions))
    offsets = np.array([row["tip_offset_mm"] for row, _, _ in starts])
    targets = np.array([row["case"]["target"] for row, _, _ in starts])
    save_json(folder / "manifest.json", dict(
        count=len(starts), target_xy_min_mm=targets.min(axis=0).tolist(), target_xy_max_mm=targets.max(axis=0).tolist(),
        tip_dx_dy_height_min_mm=offsets.min(axis=0).tolist(), tip_dx_dy_height_max_mm=offsets.max(axis=0).tolist(),
        episodes=[row for row, _, _ in starts],
    ))
    return starts


def restore(env, start):
    row, state, _ = start
    env.start(row["case"])
    mujoco.mj_setState(env.model, env.data, state, STATE)
    mujoco.mj_forward(env.model, env.data)
    env.start_time = env.data.time
    env.pick = True
    env.highest_lift = env.pickup_state()[0]
    env.phase = "insert"


class FrozenSmolVLA:
    def __init__(self, model_path):
        config = SmolVLAConfig.from_pretrained(model_path, local_files_only=True)
        config.device = "cuda"
        config.compile_model = False
        self.policy = SmolVLAPolicy.from_pretrained(
            model_path, config=config, strict=True, local_files_only=True
        ).eval().requires_grad_(False)
        if (config.chunk_size, config.n_action_steps) != (25, CHUNK):
            raise ValueError("The frozen policy must use a 25-step prediction and 5-step execution")
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(model_path))
        self.token = None

    def load_token(self, path, source, model_path):
        saved = torch.load(path, weights_only=True, map_location="cpu")
        meta = saved["metadata"]
        if Path(meta["smolvla_checkpoint"]).resolve() != model_path.resolve():
            raise ValueError("RLT checkpoint belongs to a different SmolVLA checkpoint")
        if hashlib.sha256(source.read_bytes()).hexdigest() != meta["rlt_source_sha256"]:
            raise ValueError("RLT source differs from the accepted Stage 1")
        self.token = load_rlt_class(source)(**meta["model"]).cuda().eval().requires_grad_(False)
        self.token.load_state_dict(saved["state_dict"], strict=True)

    @torch.inference_mode()
    def observe(self, env, actions, *, with_token):
        raw = env.observation()
        proprio = torch.from_numpy(raw["observation.state"])
        batch = {key: torch.from_numpy(raw[key]).permute(2, 0, 1).float() / 255 for key in CAMERAS}
        batch.update(task=raw["task"], **{"observation.state": proprio})
        batch = self.pre(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = self.policy.predict_action_chunk(batch)[:, :CHUNK]
            if with_token:
                prefix, mask = extract_prefix(self.policy, batch)
                z = self.token.encode_flat(prefix, mask)[0].float()
        reference = actions.normalize(self.post(reference.float())[0])
        if with_token:
            return dict(z_rl=z, proprio=batch["observation.state"][0].float(), ref_chunk=reference)
        return dict(ref_chunk=reference)


def rollout(env, features, mapping, start, replay=None, *, expert=False):
    restore(env, start)
    row, _, expert_actions = start
    torch.manual_seed(row["seed"])
    features.policy.reset()
    with_token = replay is not None
    obs = features.observe(env, mapping, with_token=with_token)
    done = False
    while not done:
        if expert:
            chunk = torch.as_tensor(expert_actions[env.steps:env.steps + CHUNK], device="cuda")
            if not len(chunk):
                raise RuntimeError("The expert continuation ended before termination")
            chunk = mapping.normalize(chunk)
        else:
            chunk = obs["ref_chunk"]
        reward = np.zeros(CHUNK, dtype=np.float32)
        executed = []
        for i, command in enumerate(mapping.physical(chunk).cpu().numpy()):
            reward[i], stopped, timeout, info = env.step(command)
            if not np.isclose(env.data.ctrl[-1], GRIP, atol=1e-7, rtol=0):
                raise RuntimeError("Gripper lock failed at the actuator")
            executed.append(env.data.ctrl.copy())
            done = bool(stopped or timeout or info["slipped"])
            if done:
                break
        if with_token:
            nxt = obs if done else features.observe(env, mapping, with_token=True)
            action = executed + [executed[-1]] * (CHUNK - len(executed))
            flags = np.arange(CHUNK) < len(executed) if expert else np.zeros(CHUNK, dtype=bool)
            replay.append(dict(curr_obs=obs, next_obs=nxt, actions=mapping.normalize(
                torch.as_tensor(np.array(action), dtype=torch.float32, device="cuda")),
                rewards=reward, dones=done, terminations=done, intervene_flags=flags, expert=expert))
            obs = nxt
        elif not done:
            obs = features.observe(env, mapping, with_token=False)
    return dict(index=row["index"], steps=env.steps, success=info["success"], slipped=info["slipped"],
                penetration_failure=info["penetration_failure"])


def collect_rollouts(env, features, mapping, starts, *, replay=None, expert=False):
    started = time.perf_counter()
    rows = []
    for start in starts:
        row = rollout(env, features, mapping, start, replay, expert=expert)
        rows.append(row)
        print(json.dumps(dict(phase="expert" if expert else "baseline" if replay is None else "warmup",
                              **row)), flush=True)
    total = sum(row["steps"] for row in rows)
    return dict(episodes=len(rows), insertions=sum(row["success"] for row in rows),
                closed_gripper_slips=sum(row["slipped"] for row in rows), mean_steps=total / len(rows),
                control_steps=total, steps_per_s=total / (time.perf_counter() - started), rows=rows)


def run(args):
    args.output.mkdir()
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    env = SmolVLAInsertion(penetration_limit_mm=3.)
    try:
        val = collect_starts(env, enumerate(splits["validation"]), args.output / "baseline-states", "baseline")
        actions = ActionMap(env)
        features = FrozenSmolVLA(args.smolvla)
        baseline = collect_rollouts(env, features, actions, val)
        save_json(args.output / "baseline.json", baseline)
        # Inject an open command at a held state and verify what reaches MuJoCo.
        # Reset this test state before collecting any replay.
        restore(env, val[0])
        requested = actions.normalize(torch.as_tensor(env.data.ctrl, dtype=torch.float32, device="cuda"))
        requested[-1] = 1.
        env.step(actions.physical(requested).cpu().numpy())
        assert np.isclose(env.data.ctrl[-1], GRIP, atol=1e-7, rtol=0)
        save_json(args.output / "action-map.json", dict(joints=actions.report(), normalized_sigma=NOISE_SIGMA,
                  locked_gripper_rad=GRIP, open_command_test_rad=float(env.data.ctrl[-1])))

        from .smolvla_rlt_td3 import update_td3

        features.load_token(args.token, args.token_source, args.smolvla)
        train_cases = [(i, splits["train"][i]) for i in splits["eval_train_indices"]]
        starts = collect_starts(env, train_cases, args.output / "warmup-states", "warmup")
        replay = []
        warmup = collect_rollouts(env, features, actions, starts, replay=replay)
        base_count = len(replay)
        base_positive = sum(bool(row["rewards"].any()) for row in replay)
        expert_stats = None
        if baseline["insertions"] / baseline["episodes"] < .10:
            expert_stats = collect_rollouts(env, features, actions, [s for s in starts if s[0]["expert_success"]],
                                            replay=replay, expert=True)
        positive = sum(bool(row["rewards"].any()) for row in replay)
        save_json(args.output / "warmup.json", dict(reference=warmup, expert=expert_stats,
                  reference_chunks=base_count, reference_positive_chunks=base_positive,
                  replay_chunks=len(replay), positive_chunks=positive, positive_fraction=positive / len(replay)))
        update_td3(args.rlinf, actions, replay, args.output)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("smolvla", "token", "token-source", "splits", "rlinf", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
