"""Record unchanged SO-101 expert and SmolVLA rollouts at control and physics rates."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from .scene import JOINTS, MOUTH_Z
from .smolvla25_data import CAMERAS, FPS, SmolVLAInsertion


EPISODE = 20
SEEDS = (714310, 714311, 714312)
PHASES = ("approach", "grasp", "transport", "contact")
XYZ = ("x", "y", "z")
QUAT = ("w", "x", "y", "z")
VECTORS = {
    "qpos": JOINTS, "qvel": JOINTS, "qacc": JOINTS, "ctrl": JOINTS,
    "expert": JOINTS, "dataset_qpos": JOINTS, "normalized_action": JOINTS, "target": JOINTS,
    "target_error_pre": JOINTS, "target_error_post": JOINTS, "ctrl_error_pre": JOINTS, "ctrl_error_post": JOINTS,
    "actuator_force": JOINTS, "qfrc_actuator": JOINTS, "qfrc_constraint": JOINTS,
    "ee_pos": XYZ, "ee_quat": QUAT, "ee_linear": XYZ, "ee_angular": XYZ,
    "ee_target": XYZ, "ee_ctrl": XYZ, "bolt_pos": XYZ, "bolt_quat": QUAT,
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def columns(rows):
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}


def write_csv(path, arrays):
    flat = {}
    for name, values in arrays.items():
        if values.ndim == 1:
            flat[name] = values
        else:
            key = name.removeprefix("pre_").removeprefix("post_")
            names = VECTORS.get(key, range(values.shape[1]))
            flat.update({f"{name}.{label}": values[:, i] for i, label in enumerate(names)})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(flat)
        writer.writerows(zip(*flat.values()))


def task_data(root):
    import pyarrow.parquet as pq

    case = json.loads((root / "splits.json").read_text())["train"][EPISODE]
    rows = pq.read_table(root / "dataset/data", filters=[("episode_index", "=", EPISODE)],
                         columns=["frame_index", "action", "observation.state"]).to_pylist()
    rows.sort(key=lambda r: r["frame_index"])
    if [row["frame_index"] for row in rows] != list(range(191)):
        raise ValueError("Expected all 191 recorded controls of Episode 20")
    return case, np.asarray([r["action"] for r in rows]), np.asarray([r["observation.state"] for r in rows])


class TraceInsertion(SmolVLAInsertion):
    def start(self, case):
        super().start(case)
        self.t0 = self.data.time
        self.dt = float(self.model.opt.timestep)
        self.events = {}
        self.control, self.physics, self.contacts = [], [], []
        self.scratch = mujoco.MjData(self.model)
        joint_ids = np.array([self.model.joint(joint).id for joint in JOINTS])
        if not (np.array_equal(self.model.actuator_trnid[:, 0], joint_ids)
                and np.all(self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                and np.all(self.model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError("This audit requires the original six radian position-control channels")

    def event(self, name, time_s):
        self.events.setdefault(name, dict(time_s=float(time_s), step=len(self.control)))

    def joints_and_forces(self):
        d = self.data
        return dict(qpos=d.qpos[self.joints].copy(), qvel=d.qvel[self.dofs].copy(), qacc=d.qacc[self.dofs].copy(),
                    ctrl=d.ctrl.copy(), actuator_force=d.actuator_force.copy(),
                    qfrc_actuator=d.qfrc_actuator[self.dofs].copy(), qfrc_constraint=d.qfrc_constraint[self.dofs].copy())

    def snapshot(self):
        d = self.data
        grasp = d.site(self.grasp_id)
        bolt = d.body(self.bolt_id)
        velocity, quat = np.empty(6), np.empty(4)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_SITE, self.grasp_id, velocity, 0)
        mujoco.mju_mat2Quat(quat, grasp.xmat)
        head = bolt.xpos + .045 * bolt.xmat.reshape(3, 3)[:, 2]
        return dict(**self.joints_and_forces(), ee_pos=grasp.xpos.copy(), ee_quat=quat,
                    ee_linear=velocity[3:].copy(), ee_angular=velocity[:3].copy(),
                    bolt_pos=bolt.xpos.copy(), bolt_quat=bolt.xquat.copy(),
                    tip_depth=MOUTH_Z - bolt.xpos[2],
                    tip_xy_error=np.linalg.norm(bolt.xpos[:2] - self.model.body(self.target_id).pos[:2]),
                    bolt_tilt=np.arccos(np.clip(bolt.xmat.reshape(3, 3)[2, 2], -1, 1)),
                    head_jaw_distance=np.linalg.norm(grasp.xpos - head),
                    in_jaw_range=self.pickup_state()[-1], slip_count=self.slip_count)

    def endpoint(self, joints):
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[self.joints] = joints
        mujoco.mj_kinematics(self.model, self.scratch)
        return self.scratch.site(self.grasp_id).xpos.copy()

    def sample_seating(self):
        # This existing hook runs after each mj_step and before the control-step mj_forward.
        super().sample_seating()
        d = self.data
        state_t = float(d.time - self.t0)
        solver_t = state_t - self.dt
        substep = len(self.physics)
        bolt = d.body(self.bolt_id)
        self.physics.append(dict(**self.joints_and_forces(), step=len(self.control), substep=substep,
                                 state_time_s=state_t, solver_time_s=solver_t, contact_count=d.ncon,
                                 solver_tip_depth=MOUTH_Z - bolt.xpos[2],
                                 solver_tip_xy_error=np.linalg.norm(bolt.xpos[:2] - self.model.body(self.target_id).pos[:2])))
        for i in range(d.ncon):
            contact = d.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, d, i, wrench)
            bolt_target = {b1, b2} == {self.bolt_id, self.target_id}
            self.contacts.append(dict(step=len(self.control), substep=substep, solver_time_s=solver_t,
                                      geom1=g1, geom2=g2, body1=b1, body2=b2, contact_pos=contact.pos.copy(),
                                      contact_frame=contact.frame.copy(), force_contact=wrench[:3], torque_contact=wrench[3:],
                                      dist_m=float(contact.dist), penetration_m=max(0., -float(contact.dist)),
                                      bolt_target=bolt_target))
            if bolt_target and contact.dist <= 0:
                self.event("target_contact", solver_t)

    def command(self, action, *, chunk_id=-1, chunk_index=-1, normalized=None, expert=None, dataset_qpos=None):
        t_pre = float(self.data.time - self.t0)
        pre = self.snapshot()
        if pre["ctrl"][-1] > .15 and action[-1] <= .15:
            self.event("close_command", t_pre)
        result = super().step(action)
        if len(self.physics) != self.steps * self.substeps:
            raise RuntimeError("Physics and control traces are not aligned")
        t_post = float(self.data.time - self.t0)
        post = self.snapshot()
        actual_goal = post["ctrl"]
        if not np.array_equal(np.clip(action, self.low, self.high), actual_goal):
            raise RuntimeError("MuJoCo ctrl differs from the original action clipping")
        lift, _, _, at_jaws = self.pickup_state()
        if lift >= .025 and at_jaws:
            self.event("leave_source", t_post)
        if self.pick:
            self.event("pickup_confirmed", t_post)
        if "leave_source" in self.events and pre["in_jaw_range"] and not at_jaws:
            self.event("lost_jaw_range", t_post)
        if self.slip_count:
            self.event("slip_confirmed", t_post)
        empty = np.full(len(JOINTS), np.nan)
        self.control.append(dict(
            step=self.steps - 1, t_pre_s=t_pre, t_post_s=t_post, chunk_id=chunk_id, chunk_index=chunk_index,
            replanned=chunk_index == 0, normalized_action=normalized if normalized is not None else empty,
            expert=expert if expert is not None else empty, dataset_qpos=dataset_qpos if dataset_qpos is not None else empty,
            target=action, ctrl=actual_goal, target_error_pre=action - pre["qpos"], target_error_post=action - post["qpos"],
            ctrl_error_pre=actual_goal - pre["qpos"], ctrl_error_post=actual_goal - post["qpos"],
            ee_target=self.endpoint(action), ee_ctrl=self.endpoint(actual_goal),
            **{f"pre_{k}": v for k, v in pre.items()}, **{f"post_{k}": v for k, v in post.items()},
        ))
        return result


def phase_at(time_s, events):
    labels = np.full(len(time_s), PHASES[0], dtype="U10")
    for event, phase in zip(("close_command", "leave_source", "target_contact"), PHASES[1:], strict=True):
        if event in events:
            labels[time_s + 1e-9 >= events[event]["time_s"]] = phase
    return labels


def tensor_array(value):
    cpu = value.detach().cpu()
    return (cpu.float() if str(cpu.dtype) == "torch.bfloat16" else cpu).numpy().copy()


def save_input(folder, obs, processed, chunk):
    import torch

    arrays = {f"raw.{key}": value.copy() for key, value in obs.items() if isinstance(value, np.ndarray)}
    arrays.update({f"processed.{key}": tensor_array(value) for key, value in processed.items()
                   if isinstance(value, torch.Tensor)})
    np.savez_compressed(folder / f"{chunk:04d}.npz", **arrays)
    return {"task": obs["task"], "keys": {key: str(value.dtype) if isinstance(value, torch.Tensor) else value
                                       for key, value in processed.items()}}


def noise_probe(features, inputs, chunk_id, destination):
    import torch

    policy = features.policy
    device = next(policy.parameters()).device
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    noises, predictions, angles = [], [], []
    for seed in (*SEEDS, None, None):
        generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        noise = torch.zeros(shape, device=device) if generator is None else torch.randn(shape, generator=generator, device=device)
        policy.reset()
        out = policy.predict_action_chunk({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
                                          noise=noise.clone())
        noises.append(tensor_array(noise)[0])
        predictions.append(tensor_array(out)[0])
        angles.append(tensor_array(features.post(out.float()))[0])
    np.savez_compressed(destination / "noise-probe.npz", noise=noises, normalized=predictions, target_rad=angles)
    std = np.asarray(angles[:len(SEEDS)]).std(axis=0)
    write_json(destination / "noise-probe.json", dict(chunk_id=chunk_id, seeds=SEEDS, zero_noise_runs=2,
               mean_std_rad_by_joint=std.mean(axis=0).tolist(), max_std_rad_by_joint=std.max(axis=0).tolist(),
               zero_repeat_max_difference_rad=float(np.abs(angles[-1] - angles[-2]).max())))


def rollout(case, expert, observed, features, seed, out, *, record_inputs=True):
    import cv2
    import torch

    out.mkdir()
    inputs = out / "inputs"
    if features is not None and record_inputs:
        inputs.mkdir()
    env = TraceInsertion()
    env.start(case)
    chunks, normalized_chunks, chunk_index = [], [], []
    writer = None
    probe = None
    done = timeout = False
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if features is not None else []), \
                torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=features is not None):
            if features is not None:
                torch.manual_seed(seed)
                features.policy.reset()
            for _ in range(env.max_steps):
                obs = env.observation()
                if features is None:
                    actions = expert[env.steps:env.steps + 1]
                    if not len(actions):
                        break
                else:
                    batch = {key: torch.from_numpy(obs[key]).permute(2, 0, 1).float() / 255 for key in CAMERAS}
                    batch.update(task=obs["task"], **{"observation.state": torch.from_numpy(obs["observation.state"])})
                    processed = features.pre(batch)
                    current = len(chunks)
                    meta = save_input(inputs, obs, processed, current) if record_inputs else {"task": obs["task"]}
                    if record_inputs and "target_contact" not in env.events:
                        probe = (current, {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in processed.items()})
                    prediction = features.policy.predict_action_chunk(processed)
                    target = features.post(prediction.float())
                    saved_normalized, saved_target = tensor_array(prediction)[0], tensor_array(target)[0]
                    if saved_target.shape != (25, len(JOINTS)) or features.policy.config.n_action_steps != 5:
                        raise ValueError("Expected a 25-by-6 prediction with 5 execution steps")
                    normalized_chunks.append(saved_normalized)
                    chunks.append(saved_target)
                    chunk_index.append(dict(chunk_id=current, first_step=env.steps, t_pre_s=float(env.data.time-env.t0), **meta))
                    actions = saved_target[:5]
                for i, action in enumerate(actions):
                    rgb = np.concatenate([obs[key] for key in CAMERAS], axis=1)
                    if writer is None:
                        writer = cv2.VideoWriter(str(out / "evaluation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                                 (rgb.shape[1], rgb.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("Could not open evaluation video writer")
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    model_fields = (dict(chunk_id=current, chunk_index=i, normalized=saved_normalized[i])
                                    if features is not None else dict(expert=action, dataset_qpos=observed[env.steps]))
                    _, done, timeout, info = env.command(action, **model_fields)
                    if done or timeout:
                        break
                    # Match the original evaluate() observation cadence, including at chunk boundaries.
                    if features is not None:
                        obs = env.observation()
                if done or timeout:
                    break
            if probe is not None:
                noise_probe(features, probe[1], probe[0], out)
        controls, physics, contacts = map(columns, (env.control, env.physics, env.contacts))
        controls["phase"] = phase_at(controls["t_pre_s"], env.events)
        physics["phase"] = phase_at(physics["solver_time_s"], env.events)
        physics["state_phase"] = phase_at(physics["state_time_s"], env.events)
        for name, arrays in (("control", controls), ("physics", physics), ("contacts", contacts)):
            np.savez_compressed(out / f"{name}.npz", **arrays)
            write_csv(out / f"{name}.csv", arrays)
        if chunks:
            np.savez_compressed(out / "chunks.npz", target_rad=chunks, normalized=normalized_chunks,
                                start_steps=[row["first_step"] for row in chunk_index])
            write_json(out / "chunks.json", chunk_index)
        write_json(out / "rollout.json", dict(
            episode=EPISODE, initial=env.initial, case=case, seed=seed, events=env.events, end=info,
            steps=env.steps, control_hz=FPS, physics_dt=env.dt, physics_steps=len(env.physics),
            contact_rows=len(env.contacts), integrator=int(env.model.opt.integrator),
            joints=JOINTS, actuators=[env.model.actuator(i).name for i in range(len(JOINTS))],
            bodies=[env.model.body(i).name for i in range(env.model.nbody)],
            geoms=[env.model.geom(i).name for i in range(env.model.ngeom)],
            units=dict(joints="rad, rad/s, rad/s^2", xyz="m", force="N", torque="N m", xyz_velocity="m/s",
                       angle_velocity="rad/s", quat="wxyz", wrench="contact frame, normal component first"),
            alignment="Control snapshots at t_k and t_(k+1); physics qpos/qvel after integration at state_time_s. "
                      "Physics qacc, forces, contacts and solver bolt geometry from solver_time_s.",
        ))
        print(f"Recorded {out.name}: {env.steps} controls, {len(env.physics)} physics steps", flush=True)
    finally:
        if writer is not None:
            writer.release()
        env.close()


def checkpoint_models(specifications):
    for label, path, seeds in specifications:
        model = path / "pretrained_model" if (path / "pretrained_model").is_dir() else path
        yield label, model, seeds, (model / "model.safetensors").is_file() and (model / "config.json").is_file()


def main():
    from .smolvla_motion_metrics import analyze

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Original 100-episode SFT dataset and splits")
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", nargs=2, metavar=("LABEL", "PATH"), action="append", default=[])
    parser.add_argument("--baseline", type=Path, help="One rollout with the existing 20k multi-demo SFT")
    args = parser.parse_args()
    case, expert, observed = task_data(args.data)
    args.output.mkdir()
    specifications = [(label, Path(path), SEEDS) for label, path in args.checkpoint]
    if args.baseline is not None:
        specifications.append(("smolvla-20k", args.baseline, SEEDS[:1]))
    missing = []
    expert_dir = args.output / "expert"
    rollout(case, expert, observed, None, None, expert_dir)
    results = {"expert": analyze(expert_dir, expert_dir)}
    for label, path, seeds, available in checkpoint_models(specifications):
        if not available:
            missing.append(dict(label=label, expected_path=str(path), reason="no saved model weights"))
            continue
        from .smolvla_rlt_stage2 import FrozenSmolVLA

        features = FrozenSmolVLA(path)
        for seed in seeds:
            out = args.output / f"{label}-seed-{seed}"
            rollout(case, expert, observed, features, seed, out)
            results[out.name] = analyze(out, expert_dir)
        del features
    write_json(args.output / "summary.json", dict(results=results, unavailable_checkpoints=missing))


if __name__ == "__main__":
    main()
