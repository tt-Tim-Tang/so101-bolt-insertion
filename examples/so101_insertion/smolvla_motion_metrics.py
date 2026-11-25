"""Compute phase-resolved jerk, seam, tracking, and frequency metrics from raw traces."""

import json
from pathlib import Path

import numpy as np

from .smolvla_motion_audit import JOINTS, PHASES, XYZ, write_csv, write_json


def finite_difference(values, dt, order):
    result = np.full(values.shape, np.nan, dtype=float)
    result[order:] = np.diff(values.astype(float), n=order, axis=0) / dt**order
    return result


def stats(values):
    x = np.asarray(values)
    x = np.abs(x[np.isfinite(x)])
    if not x.size:
        return dict(n=0, mean=None, rms=None, p95=None, max=None)
    return dict(n=int(x.size), mean=float(x.mean()), rms=float(np.sqrt(np.mean(x*x))),
                p95=float(np.quantile(x, .95)), max=float(x.max()))


def joint_stats(values):
    return dict(arm=stats(values[:, :5]), gripper=stats(values[:, 5]),
                joints={name: stats(values[:, i]) for i, name in enumerate(JOINTS)})


def same_history(labels, size):
    result = np.zeros(len(labels), dtype=bool)
    if len(labels) >= size:
        history = np.lib.stride_tricks.sliding_window_view(labels, size)
        result[size-1:] = np.all(history == history[:, :1], axis=1)
    return result


def phase_metrics(control, diffs, phase):
    label, chunk = control["phase"], control["chunk_id"]
    here = label == phase
    two = here & same_history(label, 2)
    three = here & same_history(label, 3)
    four = here & same_history(label, 4)
    within_accel = same_history(chunk, 3) & (chunk >= 0)
    within_jerk = same_history(chunk, 4) & (chunk >= 0)
    seams = np.flatnonzero(control["replanned"] & (control["step"] > 0))
    near_seam = np.zeros(len(label), bool)
    for k in seams:
        near_seam[max(0, k-2):k+3] = True
    velocities = control["post_qvel"]
    preceding, following = velocities[:-1], velocities[1:]
    reverse = (preceding*following < 0) & (np.abs(preceding) > .001) & (np.abs(following) > .001)
    phase_summary = {
        "steps": int(here.sum()),
        "joint_vel_sign_reversals": dict(zip(JOINTS, reverse[two[1:]].sum(axis=0).tolist(), strict=True)),
        "tracking_pre_rad": joint_stats(control["ctrl_error_pre"][here]),
        "tracking_post_rad": joint_stats(control["ctrl_error_post"][here]),
        "ee_jerk_m_per_s3": stats(np.linalg.norm(diffs["ee_jerk"][four], axis=1)),
        "clipped_steps": int(np.any(control["target"][here] != control["ctrl"][here], axis=1).sum()),
        "clip_delta_rad": joint_stats((control["ctrl"] - control["target"])[here]),
        "step_mod_5": {},
    }
    for stream in ("target", "ctrl", "qpos"):
        accel, jerk = diffs[f"{stream}_accel"], diffs[f"{stream}_jerk"]
        phase_summary[stream] = dict(
            jerk_rad_per_s3=joint_stats(jerk[four]),
            within_chunk_accel=joint_stats(accel[three & within_accel]),
            within_chunk_jerk=joint_stats(jerk[four & within_jerk]),
            cross_chunk_jerk=joint_stats(jerk[four & ~within_jerk & (chunk >= 0)]),
            seam_window_jerk=joint_stats(jerk[four & near_seam]),
        )
    phase_summary["jerk_from_measured_qacc"] = joint_stats(diffs["solver_jerk"][two])
    for offset in range(5):
        at_offset = control["step"] % 5 == offset
        phase_summary["step_mod_5"][str(offset)] = {
            stream: dict(step_rad=joint_stats(diffs[f"{stream}_step"][two & at_offset]),
                         jerk=joint_stats(diffs[f"{stream}_jerk"][four & at_offset]))
            for stream in ("target", "ctrl", "qpos")
        }
        phase_summary["step_mod_5"][str(offset)]["ee_step_m"] = stats(
            np.linalg.norm(diffs["ee_step"][two & at_offset], axis=1))
    return phase_summary


def cosine(before, after):
    length = np.linalg.norm(before) * np.linalg.norm(after)
    return float(np.clip(before @ after / length, -1, 1)) if length > 1e-12 else np.nan


def seams_and_overlap(out, control, chunks, dt):
    times, labels = control["t_pre_s"], control["phase"]
    seam_rows, overlap_rows = [], []
    predictions = chunks["target_rad"]
    starts = chunks["start_steps"]
    for i, start in enumerate(starts[1:], 1):
        row = dict(step=int(start), time_s=float(times[start]), phase=labels[start])
        for name, values in (("target", control["target"]), ("sent", control["ctrl"]),
                             ("actual", control["post_qpos"]), ("ee", control["post_ee_pos"])):
            before = values[start-1] - values[start-2]
            after = values[start] - values[start-1]
            row[f"{name}_jump_norm"] = np.linalg.norm(after[:5])
            row[f"{name}_arm_direction_cos"] = cosine(before[:5], after[:5])
            if name != "ee":
                row[f"{name}_gripper_jump_rad"] = abs(after[5])
                row[f"{name}_gripper_direction_cos"] = cosine(before[5:], after[5:])
                row.update({f"{name}_jump_rad.{joint}": after[j] for j, joint in enumerate(JOINTS)})
        seam_rows.append(row)
        offset = int(start - starts[i-1])
        for j in range(len(predictions[i-1]) - offset):
            delta = predictions[i, j] - predictions[i-1, offset+j]
            overlap_rows.append(dict(chunk=i, phase=labels[start], absolute_step=int(start+j),
                                     future_time_s=float(times[start]+j*dt),
                                     current_index=j, previous_index=offset+j,
                                     **dict(zip(JOINTS, delta, strict=True))))
    if seam_rows:
        write_csv(out / "seam-events.csv", {k: np.array([row[k] for row in seam_rows]) for k in seam_rows[0]})
    if overlap_rows:
        write_csv(out / "overlap.csv", {k: np.array([row[k] for row in overlap_rows]) for k in overlap_rows[0]})
    by_phase = {}
    for name in PHASES:
        events = [r for r in seam_rows if r["phase"] == name]
        overlap = np.array([[r[j] for j in JOINTS] for r in overlap_rows if r["phase"] == name]).reshape(-1, 6)
        selected = predictions[labels[starts] == name]
        by_phase[name] = dict(
            boundaries=len(events), same_future_disagreement_rad=joint_stats(overlap),
            full_forecast_accel=joint_stats((np.diff(selected, n=2, axis=1)/dt**2).reshape(-1, 6)),
            full_forecast_jerk=joint_stats((np.diff(selected, n=3, axis=1)/dt**3).reshape(-1, 6)),
            arm_position_jump_rad=stats([r["target_jump_norm"] for r in events]),
            arm_direction_cos=[float(r["target_arm_direction_cos"]) if np.isfinite(r["target_arm_direction_cos"]) else None
                               for r in events],
        )
    return by_phase


def spectrum(out, control, physics, dt, physics_dt):
    rows, metrics = [], {}
    for stream, trace, key, period in (("control", control, "post_qvel", dt),
                                       ("physics", physics, "qvel", physics_dt)):
        for phase in PHASES:
            x = trace[key][trace["phase"] == phase]
            if len(x) < 8:
                continue
            window = np.hanning(len(x))[:, None]
            transform = np.fft.rfft((x-x.mean(axis=0))*window, axis=0)
            freq = np.fft.rfftfreq(len(x), period)
            power = 2 * period * np.abs(transform)**2 / np.sum(window**2)
            power[0] /= 2
            if len(x) % 2 == 0:
                power[-1] /= 2
            for i, f in enumerate(freq):
                rows.append((stream, phase, f, *power[i]))
            near = int(np.argmin(np.abs(freq - 5)))
            peaks = 1 + power[1:].argmax(axis=0)
            metrics[f"{stream}_{phase}"] = dict(
                samples=len(x), duration_s=len(x)*period, bin_width_hz=float(freq[1]),
                nearest_to_5_hz=float(freq[near]), power_at_nearest_bin=power[near].tolist(),
                peak_hz_by_joint=freq[peaks].tolist(), three_replans_observed=len(x)*period >= .6,
            )
    if rows:
        names = ("stream", "phase", "frequency_hz", *JOINTS)
        write_csv(out / "joint_velocity_psd.csv", {key: np.array([row[i] for row in rows])
                                                   for i, key in enumerate(names)})
    return rows, metrics


def plot(out, ctl, phys, contacts, reference, info, derived, chunks, spectral_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre, post, physics_t = ctl["t_pre_s"], ctl["t_post_s"], phys["solver_time_s"]
    boundaries = pre[ctl["replanned"] & (ctl["step"] > 0)]
    markers = dict(close_command="#64748b", leave_source="#0891b2", pickup_confirmed="#047857",
                   target_contact="#d97706", lost_jaw_range="#db2777", slip_confirmed="#b91c1c")

    def decorate(ax):
        for t in boundaries:
            ax.axvline(t, color="#64748b", lw=.5, alpha=.26)
        for k, event in info["events"].items():
            ax.axvline(event["time_s"], color=markers[k], lw=.9, ls="--")
        ax.grid(alpha=.2)
        ax.set_xlabel("Time (s)")

    def save(fig, name):
        events = " | ".join(f"{k.replace('_', ' ')} {v['time_s']:.3f}s" for k, v in info["events"].items())
        fig.suptitle(f"{out.name}  ·  {name.replace('-', ' ')}\n{events}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(out / f"{name}.png", dpi=132)
        plt.close(fig)

    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for j, axis in enumerate(ax.flat):
        axis.plot(pre, ctl["target"][:, j], label="Model target (expert command on expert run)")
        axis.plot(pre, ctl["ctrl"][:, j], "--", label="Sent to ctrl")
        axis.plot(post, ctl["post_qpos"][:, j], label="Actual")
        axis.plot(reference["t_pre_s"], reference["expert"][:, j], ":", label="Expert reference")
        axis.set(title=JOINTS[j], ylabel="rad")
        decorate(axis)
    ax[0, 0].legend(fontsize=7)
    save(fig, "joint-angles")

    fig, axes = plt.subplots(6, 3, figsize=(13, 13), sharex=True)
    for j, row in enumerate(axes):
        for axis, (name, unit) in zip(row, (("velocity", "rad/s"), ("accel", "rad/s²"), ("jerk", "rad/s³")), strict=True):
            for stream, t in (("target", pre), ("ctrl", pre), ("qpos", post)):
                axis.plot(t, derived[f"{stream}_{name}"][:, j], lw=.8, label=stream)
            axis.set(title=f"{JOINTS[j]} {name}", ylabel=unit)
            decorate(axis)
    axes[0, 0].legend(fontsize=7)
    save(fig, "joint-derivatives")

    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for j, axis in enumerate(ax.flat):
        axis.plot(pre, ctl["ctrl_error_pre"][:, j], label="Sent − qpos before step")
        axis.plot(post, ctl["ctrl_error_post"][:, j], label="Sent − qpos after step")
        axis.set(title=JOINTS[j], ylabel="rad")
        decorate(axis)
    ax[0, 0].legend(fontsize=7)
    save(fig, "tracking-error")

    if chunks is not None:
        fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
        for start, chunk in zip(chunks["start_steps"], chunks["target_rad"], strict=True):
            t = pre[start] + np.arange(len(chunk)) / info["control_hz"]
            for j, axis in enumerate(ax.flat):
                axis.plot(t, chunk[:, j], alpha=.42, lw=.8)
        for j, axis in enumerate(ax.flat):
            axis.plot(pre, ctl["ctrl"][:, j], color="#111827", lw=1, label="Sent")
            axis.set(title=JOINTS[j], ylabel="rad")
            decorate(axis)
        save(fig, "action-chunk-overlap")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (name, unit, values) in zip(ax.flat, (
            ("Position", "m", ctl["post_ee_pos"]), ("Linear velocity", "m/s", ctl["post_ee_linear"]),
            ("Linear acceleration", "m/s²", derived["ee_accel"]), ("Linear jerk", "m/s³", derived["ee_jerk"]) ), strict=True):
        for i, label in enumerate(XYZ):
            axis.plot(post, values[:, i], label=label)
        axis.set(title=name, ylabel=unit)
        decorate(axis)
    ax[0, 0].legend(fontsize=8)
    save(fig, "end-effector")

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(projection="3d")
    for xyz, label in ((reference["post_ee_pos"], "Expert reference"), (ctl["post_ee_pos"], "Measured"),
                       (ctl["ee_ctrl"], "Sent joint targets via FK")):
        axis.plot(*(1000*xyz).T, label=label)
    span = np.vstack((reference["post_ee_pos"], ctl["post_ee_pos"], ctl["ee_ctrl"]))
    axis.set_box_aspect(np.maximum(np.ptp(span, axis=0), .002))
    axis.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    axis.legend()
    save(fig, "end-effector-3d")

    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for j, axis in enumerate(ax.flat):
        for field in ("actuator_force", "qfrc_actuator", "qfrc_constraint"):
            axis.plot(physics_t, phys[field][:, j], lw=.8, label=field)
        axis.set(title=JOINTS[j], ylabel="N·m / actuator effort")
        decorate(axis)
    ax[0, 0].legend(fontsize=7)
    save(fig, "physics-forces")

    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(pre, ctl["ctrl"][:, -1], label="Gripper command")
    ax[0].plot(post, ctl["post_qpos"][:, -1], label="Gripper actual")
    ax[0].set_ylabel("rad")
    ax[1].plot(post, 1000*ctl["post_tip_depth"], label="Insertion depth")
    ax[1].plot(post, 1000*ctl["post_tip_xy_error"], label="XY error")
    ax[1].plot(post, 1000*ctl["post_head_jaw_distance"], label="Jaw-to-head distance")
    ax[1].set_ylabel("mm")
    ax[2].plot(post, np.rad2deg(ctl["post_bolt_tilt"]), label="Bolt tilt")
    ax[2].set_ylabel("degree")
    if contacts:
        peak, peak_target = np.zeros(len(physics_t)), np.zeros(len(physics_t))
        np.maximum.at(peak, contacts["substep"], np.abs(contacts["force_contact"][:, 0]))
        match = contacts["bolt_target"]
        np.maximum.at(peak_target, contacts["substep"][match], np.abs(contacts["force_contact"][match, 0]))
        ax[3].plot(physics_t, peak, label="Any contact: peak normal force")
        ax[3].plot(physics_t, peak_target, label="Bolt–target: peak normal force")
        twin = ax[3].twinx()
        depth = np.zeros(len(physics_t))
        np.maximum.at(depth, contacts["substep"][match], contacts["penetration_m"][match])
        twin.plot(physics_t, depth*1000, color="#7e22ce", lw=.8, label="Bolt–target penetration (mm)")
        twin.set_ylabel("mm", color="#7e22ce")
    ax[3].set_ylabel("N")
    for axis in ax:
        decorate(axis)
        axis.legend(fontsize=8)
    save(fig, "gripper-bolt-contact")

    fig, ax = plt.subplots(4, 2, figsize=(12, 10), sharex="col")
    for i, phase in enumerate(PHASES):
        for j, stream in enumerate(("control", "physics")):
            axis = ax[i, j]
            band = np.array([r[2:] for r in spectral_rows if r[0] == stream and r[1] == phase])
            if len(band):
                for k, joint in enumerate(JOINTS):
                    axis.semilogy(band[:, 0], np.maximum(band[:, k+1], 1e-14), label=joint)
            axis.axvline(5, color="#b91c1c", ls=":", label="5 Hz replan reference")
            axis.set(title=f"{phase} {stream}", xlabel="Hz", ylabel="(rad/s)²/Hz",
                     xlim=(0, 12.5 if stream == "control" else 40))
    ax[0, 0].legend(fontsize=6, ncol=2)
    save(fig, "joint-velocity-spectrum")


def analyze(out: Path, expert: Path):
    info = json.loads((out / "rollout.json").read_text())
    arrays = {}
    for file in ("control", "physics", "contacts"):
        with np.load(out / f"{file}.npz") as data:
            arrays[file] = dict(data)
    ctl, phys, contacts = (arrays[key] for key in ("control", "physics", "contacts"))
    with np.load(expert / "control.npz") as reference_file:
        reference = dict(reference_file)
    dt = 1 / info["control_hz"]
    derived = {
        "ee_step": finite_difference(ctl["post_ee_pos"], 1, 1),
        "ee_accel": finite_difference(ctl["post_ee_linear"], dt, 1),
        "ee_jerk": finite_difference(ctl["post_ee_linear"], dt, 2),
        "solver_jerk": finite_difference(ctl["post_qacc"], dt, 1),
    }
    for stream, vec in (("target", ctl["target"]), ("ctrl", ctl["ctrl"]), ("qpos", ctl["post_qpos"])):
        derived[f"{stream}_step"] = finite_difference(vec, 1, 1)
        for order, name in enumerate(("velocity", "accel", "jerk"), 1):
            derived[f"{stream}_{name}"] = finite_difference(vec, dt, order)
    np.savez_compressed(out / "derivatives.npz", t_pre_s=ctl["t_pre_s"], t_post_s=ctl["t_post_s"], **derived)
    phases = {phase: phase_metrics(ctl, derived, phase) for phase in PHASES}
    for phase in PHASES:
        phase_key = (phys["phase"] == phase) & same_history(phys["phase"], 2)
        jerk = finite_difference(phys["qacc"], info["physics_dt"], 1)
        phases[phase]["physics_jerk_from_qacc"] = joint_stats(jerk[phase_key])
    chunks = None
    comparisons = {}
    if (out / "chunks.npz").exists():
        with np.load(out / "chunks.npz") as data:
            chunks = dict(data)
        comparisons = seams_and_overlap(out, ctl, chunks, dt)
    spectral_rows, frequency_peaks = spectrum(out, ctl, phys, dt, info["physics_dt"])
    result = dict(rollout=info, phase_metrics=phases, predicted_chunk_comparison=comparisons,
                  joint_velocity_spectrum=frequency_peaks,
                  convention="Finite differences are backward and unfiltered. Mean refers to mean magnitude. "
                             "Phase statistics exclude derivative stencils crossing phase changes. The arm combines five "
                             "radian joints; the gripper is separate. Boundary window is ±2 control steps from replan. "
                             "Strict cross-chunk jerk requires the derivative stencil to span two realized chunks. "
                             "The 5 Hz line on expert plots is a comparison reference. No expert replan events are created.")
    noise = out / "noise-probe.json"
    if noise.exists():
        result["noise_probe"] = json.loads(noise.read_text())
    write_json(out / "summary.json", result)
    plot(out, ctl, phys, contacts, reference, info, derived, chunks, spectral_rows)
    return result
