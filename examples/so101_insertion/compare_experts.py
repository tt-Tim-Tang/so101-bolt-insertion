"""Collect and replay two SO-101 experts in the frozen Fast16 MuJoCo scene."""

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np

from .scene import JOINTS, MOUTH_Z
from .wide_env import WideInsertion, expert

FPS = 25
BLOCK = 10
SEEDS = (101, 102, 103, 104, 105)
STDS = (0.005, 0.01)
EXPERTS = ("continuous_ik", "maniskill_style")
CASES = {
    "center": (0, 0),
    "corner_xn_yn": (-18, -18),
    "corner_xn_yp": (-18, 18),
    "corner_xp_yn": (18, -18),
    "corner_xp_yp": (18, 18),
}
CONTACT_WARNING = re.compile(r"linesearch|line.search|solver|constraint|Hessian|contact", re.I)


def initial_case(xy):
    return {"source": [0, 0], "target": xy, "joints": [0] * 5}


class CollectedInsertion(WideInsertion):
    def __init__(self, images):
        super().__init__(target_bore_radius=0.0057)
        self.images = images

    def start(self, case):
        super().start(case)
        self.frames = []
        self.result = {}
        self.start_state = self.observation_state()

    def observation_state(self):
        bolt = self.data.body("bolt")
        tcp = self.data.site("grasp")
        return {
            "q": self.state(),
            "tip": bolt.xpos.copy(),
            "local_tip": tcp.xmat.reshape(3, 3).T @ (bolt.xpos - tcp.xpos),
            "in_jaws": self.pickup_state()[-1],
        }

    def step(self, action):
        if self.steps % 2 == 0:
            frame = self.observation_state()
            frame.update(action=np.clip(action, self.low, self.high), phase=self.phase)
            if self.images:
                obs = self.observation()
                frame.update(top=obs["observation.images.top"], wrist=obs["observation.images.wrist"])
            self.frames.append(frame)
        reward, done, timeout, info = super().step(self.frames[-1]["action"])
        self.result = info
        return reward, done, timeout, info


def maniskill_style(env, case):
    """Port ManiSkill's geometry-based grasp and three pose refinements to SO-101.

    Use Cartesian waypoints and MuJoCo IK in place of MPLib's Panda screw planner.
    Round-shaft yaw is free. Each segment is planned before it is executed.
    """
    env.start(case)
    bolt = env.data.body("bolt")
    tcp = env.data.site("grasp")
    target_xy = env.model.body("target").pos[:2].copy()
    up = np.array([0.0, 0.0, 1.0])

    def execute(phase, commands):
        env.phase = phase
        for cmd in commands:
            _, done, timeout, _ = env.step(cmd)
            if done or timeout:
                return True
        return False

    def move(phase, point, axis, grip, ticks, held=False, linear=False):
        if held:
            start_point = bolt.xpos.copy()
            start_axis = bolt.xmat.reshape(3, 3)[:, 2].copy()
            solve = env.bolt_ik
        else:
            start_point = tcp.xpos.copy()
            start_axis = tcp.xmat.reshape(3, 3)[:, 2].copy()
            solve = lambda p, a, q: env.inverse_kinematics(p, q, a)
        q = env.state().astype(float)
        grip_start = env.data.ctrl[-1]
        commands = []
        env.phase = phase
        for tick in range(0, ticks, 2):
            t = (tick + 2) / ticks
            fraction = t if linear else t * t * (3.0 - 2.0 * t)
            p = start_point + fraction * (point - start_point)
            direction = start_axis + fraction * (axis - start_axis)
            q = solve(p, direction / np.linalg.norm(direction), q)
            q[-1] = grip_start + fraction * (grip - grip_start)
            commands.extend((q.copy(), q.copy()))
        return execute(phase, commands)

    axis = bolt.xmat.reshape(3, 3)[:, 2].copy()
    head = bolt.xpos + 0.044 * axis
    if move("over_source", head + 0.048 * up, axis, 0.52, 44):
        return env.result
    if move("lower", head, axis, 0.52, 50):
        return env.result
    q = env.data.ctrl.copy()
    closed = q.copy()
    closed[-1] = -0.055
    fractions = np.arange(1, 37) / 36
    close_commands = q + (closed - q) * (fractions**2 * (3 - 2 * fractions))[:, None]
    if execute("close", close_commands):
        return env.result
    if move("lift", head + 0.05 * up, axis, -0.055, 64):
        return env.result
    if not env.pick:
        return dict(env.result, reason="pickup")

    held_axis = bolt.xmat.reshape(3, 3)[:, 2].copy()
    if move("transfer", np.r_[target_xy, MOUTH_Z + 0.030], held_axis, -0.055, 64, held=True):
        return env.result
    pre_insert = np.r_[target_xy, MOUTH_Z + 0.005]
    if move("air_align", pre_insert, up, -0.055, 26, held=True):
        return env.result
    for _ in range(3):
        if move("air_align", pre_insert, up, -0.055, 8, held=True):
            return env.result
    seated = np.r_[target_xy, env.floor_z + 0.0005]
    if not move("insert", seated, up, -0.055, 66, held=True, linear=True):
        execute("hold", [env.data.ctrl.copy()] * 34)
    return env.result


def reversals(values, deadband=0.0002):
    """Count direction reversals after travelling 0.2 mm back from an extremum."""
    count = direction = 0
    if not len(values):
        return count
    extreme = values[0]
    for value in values[1:]:
        delta = value - extreme
        if not direction and abs(delta) >= deadband:
            direction = int(np.sign(delta))
            extreme = value
        elif direction and direction * delta > 0:
            extreme = value
        elif direction and direction * delta <= -deadband:
            count += 1
            direction *= -1
            extreme = value
    return count


def summarize_nominal(env, result, warnings):
    frames = env.frames
    phase = np.array([f["phase"] for f in frames])
    q = np.array([f["q"] for f in frames], dtype=float)
    action = np.array([f["action"] for f in frames])
    tip = np.array([f["tip"] for f in frames])
    target = env.model.body("target").pos[:2]
    radial = np.linalg.norm(tip[:, :2] - target, axis=1)
    dq = np.diff(q, axis=0)
    ddq = np.diff(q, n=2, axis=0) * FPS**2
    transfer = np.flatnonzero(phase == "transfer")
    slip = None
    above = None
    above_idx = None
    if len(transfer):
        local_tip = np.array([f["local_tip"] for f in frames])
        transfer_stop = min(transfer[-1] + 2, len(frames))
        local_transfer = local_tip[transfer[0]:transfer_stop]
        slip = 1000 * np.max(np.linalg.norm(local_transfer - local_transfer[0], axis=1))
        align = np.flatnonzero(np.isin(phase, ("air_align", "insert", "hold")))
        if len(align):
            distances = np.hypot(radial[align], tip[align, 2] - MOUTH_Z - 0.005)
            above_idx = int(align[np.argmin(distances)])
            above = {
                "frame": above_idx,
                "q_rad": q[above_idx].tolist(),
                "action_rad": action[above_idx].tolist(),
                "height_above_mouth_mm": float((tip[above_idx, 2] - MOUTH_Z) * 1000),
                "radial_error_mm": float(radial[above_idx] * 1000),
            }
    crossings = np.flatnonzero(np.isin(phase, ("air_align", "insert", "hold")) & (tip[:, 2] <= MOUTH_Z))
    before_entry = np.arange(len(frames)) < (crossings[0] if len(crossings) else len(frames))
    near_entry = before_entry & (phase == "air_align") & (radial <= 0.005)
    near_entry |= before_entry & (phase == "insert") & (radial <= 0.005)
    reversal_counts = [reversals(tip[near_entry, i]) for i in range(2)]
    after_entry = np.arange(len(frames)) > (crossings[0] if len(crossings) else len(frames))
    opening = np.r_[False, np.diff(action[:, -1]) > 0.001]
    withdraw = np.isin(phase, ("release", "withdraw", "retreat", "retract"))
    return {
        "frames": len(frames), "physics_ticks": env.steps,
        "pickup": bool(result.get("pickup")), "insertion": bool(result.get("success")),
        "arrival": bool(result.get("arrival")), "reason": result.get("reason"),
        "static_frames": int(np.sum(np.max(abs(dq), axis=1) < 0.001)), "static_denominator": len(dq),
        "max_q_step_rad": float(np.max(abs(dq))),
        "max_action_step_rad": float(np.max(abs(np.diff(action, axis=0)))),
        "accel_squared_sum": float(np.sum(ddq ** 2)), "accel_count": int(ddq.size),
        "contact_solver_warnings": sum(bool(CONTACT_WARNING.search(w)) for w in warnings),
        "all_warnings": warnings, "max_transfer_slip_mm": None if slip is None else float(slip),
        "lateral_reversals_xy": reversal_counts, "near_entry_frames": int(np.sum(near_entry)),
        "extra_release_or_retreat_frames": int(np.sum(after_entry & (opening | withdraw))),
        "above_hole": above,
    }, {"phase": phase, "q_rad": q, "action_rad": action, "tip_m": tip,
        "grasp_relative_tip_m": np.array([f["local_tip"] for f in frames]),
        "in_jaws": np.array([f["in_jaws"] for f in frames]), "time_s": np.arange(len(frames)) / FPS}


def replay(env, case, action, std, seed):
    env.start(case)
    rng = np.random.default_rng(seed)
    if std:
        bias = rng.normal(0, std, size=((len(action) + BLOCK - 1) // BLOCK, action.shape[-1]))
        action = action + np.repeat(bias, BLOCK, axis=0)[:len(action)]
    # Preserve the same physics ticks and stopping predicate as nominal collection.
    for cmd in action:
        for _ in range(2):
            _, done, timeout, info = env.step(cmd)
            if done or timeout:
                return info
    return info


def write_video(path, left, right, scenario):
    import av
    from PIL import Image, ImageDraw, ImageFont

    width, height = 896, 304
    font = ImageFont.truetype("arial.ttf", 18)
    small = ImageFont.truetype("arial.ttf", 14)
    colors = ("#80cbc4", "#ffd180")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS, options={"crf": "20", "preset": "fast"})
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(max(len(left["top"]), len(right["top"]))):
            canvas = Image.new("RGB", (width, height), "#18212b")
            draw = ImageDraw.Draw(canvas)
            for j, (name, d) in enumerate(zip(EXPERTS, (left, right))):
                x = j * width // 2
                k = min(i, len(d["top"]) - 1)
                draw.text((x + 10, 8), name.replace("_", " "), font=font, fill=colors[j])
                for cam_idx, camera in enumerate(("top", "wrist")):
                    cam_x = x + cam_idx * width // 4
                    draw.text((cam_x + 10, 34), camera, font=small, fill="white")
                    canvas.paste(Image.fromarray(d[camera][k]), (cam_x, 55))
                phase_text = str(d["phase"][k]) if k == i else "end of episode (last frame held)"
                draw.text((x + 10, 283), f"{scenario}  |  {k/FPS:.2f} s  |  {phase_text}", font=small, fill="white")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def run(output, videos):
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    if videos:
        (output / "videos").mkdir()
    env = CollectedInsertion(videos)
    replay_env = WideInsertion(target_bore_radius=0.0057)
    warnings = []
    previous_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(warnings.append)
    metadata = dict(fps=FPS, physics_dt_s=env.model.opt.timestep, clearance_mm=0.7, joints=JOINTS,
                    cases=CASES, seeds=SEEDS, noise_std_rad=STDS, noise_block_frames=BLOCK, mujoco=mujoco.__version__)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def save_row(handle, row):
        handle.write(json.dumps(row) + "\n")
        handle.flush()

    try:
        with (output / "nominal.jsonl").open("w", encoding="utf-8") as nominal, \
             (output / "replay.jsonl").open("w", encoding="utf-8") as replays:
            for name, xy in CASES.items():
                case = initial_case(xy)
                clips = []
                for label, controller in zip(EXPERTS, (expert, maniskill_style)):
                    warnings.clear()
                    try:
                        result = controller(env, case)
                    except RuntimeError as error:
                        result = dict(env.result, reason=f"{env.phase}: {error}")
                    metrics, arrays = summarize_nominal(env, result, warnings.copy())
                    save_row(nominal, dict(case=name, hole_offset_mm=xy, expert=label, **metrics))
                    arrays["initial_q_rad"] = env.start_state["q"]
                    if videos:
                        for cam in ("top", "wrist"):
                            arrays[cam] = np.array([f[cam] for f in env.frames])
                        clips.append(arrays)
                    np.savez_compressed(output / "trajectories" / f"{name}_{label}.npz", **arrays)
                    plan = [(0.0, SEEDS[0])] + [(std, seed) for std in STDS for seed in SEEDS]
                    for std, seed in plan:
                        warnings.clear()
                        replay_result = replay(replay_env, case, arrays["action_rad"], std, seed)
                        save_row(replays, dict(case=name, expert=label, std_rad=std, seed=seed,
                                             pickup=replay_result["pickup"], insertion=replay_result["success"],
                                             arrival=replay_result["arrival"], physics_ticks=replay_env.steps,
                                             radial_error_mm=1000 * replay_result["radial_error_m"],
                                             warnings=warnings.copy()))
                    print(name, label, metrics["frames"], metrics["insertion"], metrics["reason"], flush=True)
                if videos:
                    write_video(output / "videos" / f"{name}.mp4", *clips, name)
    finally:
        mujoco.set_mju_user_warning(previous_warning)
        env.close()
        replay_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run(args.out, videos=not args.no_video)
