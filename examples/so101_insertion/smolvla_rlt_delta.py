"""Qpos-relative action chunks for the SO-101 insertion task."""

import mujoco
import numpy as np
import torch

from .smolvla_rlt_stage2 import CHUNK, GRIP, STATE, restore


SIGMA = .015
SLIP_REWARD = -.5


class DeltaActionMap:
    def __init__(self, env, absolute, limits):
        self.limits = np.asarray(limits, dtype=np.float64)
        self.low = absolute.low.cpu().numpy()[:5].astype(np.float64)
        self.high = absolute.high.cpu().numpy()[:5].astype(np.float64)
        self.absolute = absolute
        self.half = torch.as_tensor([*self.limits, 0.], device=absolute.low.device, dtype=torch.float32)
        self.sigma = SIGMA
        self.preview = mujoco.MjData(env.model)
        self.state = np.empty(mujoco.mj_stateSize(env.model, STATE))

    def lock(self, normalized):
        return torch.cat((normalized[..., :5], torch.zeros_like(normalized[..., 5:])), dim=-1)

    def target(self, qpos, delta):
        lower, upper = qpos - self.limits, qpos + self.limits
        desired = qpos + np.clip(delta, -1., 1.) * self.limits
        # If a measured joint overshoots the absolute training guard, move it
        # toward the guard by at most one permitted increment.
        return np.clip(np.clip(desired, self.low, self.high), lower, upper)

    def reference(self, env, absolute_chunk):
        # A frozen VLA emits absolute targets. Reconstruct how its five targets
        # would be encoded from qpos on a copy of the current simulator state.
        mujoco.mj_getState(env.model, env.data, self.state, STATE)
        mujoco.mj_setState(env.model, self.preview, self.state, STATE)
        mujoco.mj_forward(env.model, self.preview)
        reference = []
        for goal in absolute_chunk:
            qpos = self.preview.qpos[env.joints[:5]]
            target = self.target(qpos, (goal[:5] - qpos) / self.limits)
            reference.append([*((target - qpos) / self.limits), 0.])
            self.preview.ctrl[:5], self.preview.ctrl[-1] = target, GRIP
            mujoco.mj_step(env.model, self.preview, nstep=env.substeps)
            mujoco.mj_forward(env.model, self.preview)
        return torch.as_tensor(np.array(reference), dtype=torch.float32, device=self.half.device)


def observe(env, actions, features, with_token):
    obs = features.observe(env, actions.absolute, with_token=with_token)
    absolute = actions.absolute.physical(obs["ref_chunk"]).cpu().numpy()
    obs["ref_chunk"] = actions.reference(env, absolute)
    return obs, absolute


def rollout_delta(env, features, actions, start, *, policy=None, explore=False, on_transition=None):
    restore(env, start)
    row, _, _ = start
    torch.manual_seed(row["seed"])
    features.policy.reset()
    with_token = policy is not None or on_transition is not None
    obs, absolute = observe(env, actions, features, with_token)
    done = budget = False
    motion = []
    grasp = env.data.site("grasp")
    peak_delta = np.zeros(5)
    while not done:
        if policy is None:
            chunk = obs["ref_chunk"]
        else:
            with torch.no_grad():
                batch = {key: value.unsqueeze(0) for key, value in obs.items()}
                predicted, _, _ = policy.sac_forward(batch, deterministic=not explore, apply_action_noise=explore)
                chunk = predicted.reshape_as(obs["ref_chunk"])

        rewards = np.zeros(CHUNK, dtype=np.float32)
        executed = np.zeros((CHUNK, 6), dtype=np.float32)
        for tick, delta in enumerate(chunk.cpu().numpy()):
            before = grasp.xpos.copy()
            qpos = env.data.qpos[env.joints[:5]].copy()
            relative = (absolute[tick, :5] - qpos) / actions.limits if policy is None else delta[:5]
            target = np.r_[actions.target(qpos, relative), GRIP]
            reward, stopped, timeout, info = env.step(target)
            slipped = bool(info["slipped"])
            rewards[tick] = SLIP_REWARD if slipped and not reward else reward
            actual = env.data.ctrl[:5] - qpos
            if np.any(np.abs(actual) > actions.limits + 1e-8):
                raise RuntimeError("Actual qpos-relative target exceeded the accepted per-axis limit")
            if abs(env.data.ctrl[-1] - GRIP) > 1e-7:
                raise RuntimeError("Gripper lock failed")
            executed[tick, :5] = actual / actions.limits
            peak_delta = np.maximum(peak_delta, np.abs(actual))
            motion.append(float(np.linalg.norm(grasp.xpos - before) * 1000))
            done = bool(stopped or timeout or slipped)
            if done:
                break
        if policy is None:
            observed_reference = obs["ref_chunk"].clone()
            observed_reference[:tick + 1] = torch.from_numpy(executed[:tick + 1]).to(actions.half.device)
            obs["ref_chunk"] = observed_reference
        next_obs, next_absolute = (obs, absolute) if done else observe(env, actions, features, with_token)
        if on_transition is not None:
            stop_now = on_transition(dict(
                curr_obs=obs, next_obs=next_obs, actions=torch.from_numpy(executed).to(actions.half.device),
                rewards=rewards, dones=done, terminations=done, intervene_flags=np.zeros(CHUNK, dtype=bool),
            ))
            if stop_now:
                budget = not done
                break
        obs, absolute = next_obs, next_absolute
    return dict(index=row["index"], steps=env.steps, success=info["success"], slipped=info["slipped"],
                penetration_failure=info["penetration_failure"], budget_exhausted=budget,
                wrist_mm=motion, peak_delta_rad=peak_delta.tolist())
