"""Frozen SmolVLA unload replays and control-rate proprioception audit."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .scene import JOINTS, MOUTH_Z, SHAFT_RADIUS
from .smolvla25_data import DEPTH_TOLERANCE_M, FPS, MAX_FLOOR_OVERRUN_M, MIN_AXIS_Z
from .smolvla_padding_e2e30 import EndToEndTelemetry, failure_mode, replay
from .smolvla_rlt_stage2 import GRIP, FrozenSmolVLA


FAILURES = (2, 5, 12, 13, 28)
TRIGGERS = ("on_seated", "on_bottom_load")
HOLDS = ("latched_qpos", "live_qpos", "half_unload")
PHASES = ("pre_grasp", "grasp_close", "lift_transport", "pre_contact", "insertion", "bottom_load", "seated")
SIGNALS = ("qvel", "qfrc_actuator", "ctrl_minus_qpos", "u_vla_minus_qpos")
QUANTILES = {"p10": .10, "p50": .50, "p90": .90, "p95": .95}


class ForceGateInsertion(EndToEndTelemetry):
    def __init__(self, trigger=None, hold=None):
        super().__init__(penetration_limit_mm=3.)
        self.trigger_kind = trigger
        self.hold = hold

    def start(self, case):
        super().start(case)
        self.trigger = None
        self.applied_step = None
        self.trace = []

    def seating_state(self):
        tip, axis, xy, depth, tilt = self.bolt_pose()
        gap = tip[2] - SHAFT_RADIUS * np.linalg.norm(axis[:2]) - self.floor_z
        reasons = []
        if depth <= 0 or gap > DEPTH_TOLERANCE_M:
            reasons.append("insufficient_depth")
        if gap < -MAX_FLOOR_OVERRUN_M:
            reasons.append("floor_overrun")
        if axis[2] < MIN_AXIS_Z:
            reasons.append("tilt")
        if axis[2] > 0:
            at_mouth = xy + depth * axis[:2] / axis[2]
            if np.any(self.normals @ xy > self.faces):
                reasons.append("tip_xy")
            if np.any(self.normals @ at_mouth > self.faces):
                reasons.append("axis_at_mouth_xy")
        return dict(depth_mm=depth * 1000, xy_mm=float(np.linalg.norm(xy) * 1000), tilt_deg=tilt,
                    gap_mm=float(gap * 1000), seating_blockers=reasons)

    def sample_seating(self):
        previous_ticks = self.seated_ticks
        previous_penetration = self.max_penetration
        super().sample_seating()
        self.substep += 1
        contacts = self.data.contact
        if len(contacts):
            bodies = self.model.geom_bodyid[contacts.geom]
            pair = (bodies == self.bolt_id).any(axis=1) & (bodies == self.target_id).any(axis=1)
            if pair.any():
                self.step_penetration = max(self.step_penetration, float(-contacts.dist[pair].min()))
        self.step_peak_ticks = max(self.step_peak_ticks, self.seated_ticks)
        broke_hold = previous_ticks > 0 and self.seated_ticks == 0
        crossed_limit = previous_penetration <= self.penetration_limit_m < self.max_penetration
        if broke_hold or crossed_limit:
            self.events.append(dict(
                substep=self.substep, previous_seated_ticks=previous_ticks, broke_hold=broke_hold,
                penetration_limit_crossed=crossed_limit, max_penetration_mm=self.max_penetration * 1000,
                **self.seating_state(),
            ))
        if self.trigger_kind is None or self.trigger is not None:
            return
        depth_mm = (MOUTH_Z - self.data.body(self.bolt_id).xpos[2]) * 1000
        seated = self.trigger_kind == "on_seated" and self.seated_ticks >= 1
        bottom = (self.trigger_kind == "on_bottom_load" and self.pick and self.first_contact is not None
                  and self.max_streak == 0
                  and depth_mm >= 28.5 and self.max_penetration >= .0018)
        if seated or bottom:
            self.trigger = dict(
                step=self.steps + 1, substep=self.substep, seated_ticks=self.seated_ticks,
                max_penetration_mm=self.max_penetration * 1000, qpos=self.data.qpos[self.joints].tolist(),
                ctrl=self.data.ctrl.tolist(), **self.seating_state(),
            )

    def phase_stage(self, depth_mm):
        if self.seated_ticks > 0:
            return "seated"
        if self.first_contact is not None:
            return "bottom_load" if depth_mm >= 28 else "insertion"
        if self.reached_vicinity:
            return "pre_contact"
        lift, at_source, _, jaws = self.pickup_state()
        closed = self.data.ctrl[5] <= .15
        if closed and jaws and (lift > .002 or self.pick):
            return "lift_transport"
        if closed and at_source:
            return "grasp_close"
        return "pre_grasp"

    def step(self, action):
        self.step_penetration = 0.
        self.step_peak_ticks = 0
        self.substep = 0
        self.events = []
        applied = action
        if self.trigger is not None:
            if self.applied_step is None:
                self.applied_step = self.steps + 1
            applied = np.empty(len(JOINTS))
            if self.hold == "live_qpos":
                applied[:5] = self.data.qpos[self.joints[:5]]
            else:
                applied[:5] = self.trigger["qpos"][:5]
                if self.hold == "half_unload":
                    applied[:5] = .5 * (applied[:5] + np.asarray(self.trigger["ctrl"][:5]))
            applied[5] = GRIP
        outcome = super().step(applied)
        if self.trigger_kind is None:
            qpos = self.data.qpos[self.joints]
            pose = self.seating_state()
            self.trace.append(dict(
                step=self.steps, time_s=float(self.data.time - self.start_time),
                phase_stage=self.phase_stage(pose["depth_mm"]), **pose,
                seated_ticks=self.seated_ticks, peak_step_seated_ticks=self.step_peak_ticks,
                step_penetration_mm=self.step_penetration * 1000, events=self.events,
                qpos=qpos.tolist(), qvel=self.data.qvel[self.dofs].tolist(),
                qfrc_actuator=self.data.qfrc_actuator[self.dofs].tolist(),
                ctrl_minus_qpos=(self.data.ctrl - qpos).tolist(), u_vla_minus_qpos=(action - qpos).tolist(),
            ))
        return outcome


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value, *, compact=False):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=None if compact else 2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def saved_cases(path):
    report = read_json(path)
    rows = report["rows"]
    assert report["insertions"] == 25 and [r["index"] for r in rows] == list(range(30))
    assert tuple(r["index"] for r in rows if not r["success"]) == FAILURES
    return rows


def run(args):
    models = FrozenSmolVLA(args.checkpoint)
    env = ForceGateInsertion()
    try:
        for row in saved_cases(args.baseline)[args.shard::args.shards]:
            index = row["index"]
            variants = [(None, None)]
            if index in FAILURES:
                variants.extend(itertools.product(TRIGGERS, HOLDS))
            for trigger, hold in variants:
                path = args.output / f"{index:02d}-{trigger or 'baseline'}{('-' + hold) if hold else ''}.json"
                if path.exists():
                    continue
                env.trigger_kind, env.hold = trigger, hold
                end, _ = replay(env, models, row["case"], row["seed"], corrected=True)
                physical_reason = failure_mode(end, env)[0]
                terminal = ("success" if end["success"] else "penetration" if end["penetration_failure"]
                            else "slip" if end["slipped"] else "timeout" if end["timeout"] else "other")
                result = dict(
                    index=index, seed=row["seed"], case=row["case"], trigger_kind=trigger, hold=hold,
                    trigger=env.trigger, first_hold_step=env.applied_step, terminal=terminal,
                    failure_mode=physical_reason, max_held_depth_mm=env.max_near_depth_mm, **end,
                )
                if trigger is None:
                    result["trace"] = env.trace
                    result["wrist_steps_mm"] = env.wrist_steps_mm
                else:
                    result["triggered_but_no_hold"] = env.trigger is not None and env.applied_step is None
                write_json(path, result, compact=True)
                print(index, trigger or "baseline", hold or "-", terminal, end["steps"], end["stable_s"],
                      round(end["max_penetration_mm"], 3), "trigger", (env.trigger or {}).get("step"), flush=True)
    finally:
        env.close()


def reduced_signals(frame):
    columns = {f"{signal}_l2": float(np.linalg.norm(frame[signal][:5])) for signal in SIGNALS}
    for signal in ("qpos", *SIGNALS):
        for axis, value in zip(JOINTS[:4], frame[signal][:4], strict=True):
            columns[f"{signal}/{axis}"] = abs(value)
    return columns


def signal_summary(rows):
    groups = {phase: [] for phase in (*PHASES, "air_combined", "bottom_combined")}
    for row in rows:
        for frame in row["trace"]:
            phase = frame["phase_stage"]
            features = reduced_signals(frame)
            groups[phase].append(features)
            if phase in ("grasp_close", "lift_transport"):
                groups["air_combined"].append(features)
            elif phase in ("bottom_load", "seated"):
                groups["bottom_combined"].append(features)
    stats, columns = {}, {}
    for group, frames in groups.items():
        columns[group] = {key: np.array([f[key] for f in frames]) for key in frames[0]} if frames else {}
        stats[group] = dict(frames=len(frames), signals={
            key: dict(zip(QUANTILES, np.quantile(values, list(QUANTILES.values())).tolist(), strict=True))
            for key, values in columns[group].items()
        })
    air, bottom = columns["air_combined"], columns["bottom_combined"]
    auc = {}
    for key in air.keys() & bottom.keys():
        ordered = np.sort(air[key])
        left = np.searchsorted(ordered, bottom[key], side="left")
        right = np.searchsorted(ordered, bottom[key], side="right")
        auc[key] = float(np.mean((left + right) / (2 * len(ordered))))
    return stats, dict(sorted(auc.items()))


def summarize(args):
    previous = saved_cases(args.baseline)
    rows = [read_json(args.output / f"{i:02d}-baseline.json") for i in range(30)]
    assert all((a["index"], a["seed"], a["case"]) == (b["index"], b["seed"], b["case"])
               for a, b in zip(rows, previous, strict=True))
    oracle = [read_json(args.output / f"{i:02d}-{trigger}-{hold}.json")
              for i, trigger, hold in itertools.product(FAILURES, TRIGGERS, HOLDS)]
    phases, auc = signal_summary(rows)
    oracle_comparison = []
    for trigger, hold in itertools.product(TRIGGERS, HOLDS):
        matched = [r for r in oracle if (r["trigger_kind"], r["hold"]) == (trigger, hold)]
        oracle_comparison.append(dict(
            trigger=trigger, hold=hold, triggered=sum(r["trigger"] is not None for r in matched),
            applied=sum(r["first_hold_step"] is not None for r in matched),
            successful_with_hold=[r["index"] for r in matched if r["success"] and r["first_hold_step"] is not None],
            successful_without_hold=[r["index"] for r in matched if r["success"] and r["first_hold_step"] is None],
            slips=sum(bool(r["slipped"]) for r in matched),
        ))
    detail = {r["index"]: [
        {key: f[key] for key in ("step", "depth_mm", "xy_mm", "tilt_deg", "gap_mm", "seated_ticks",
                                "peak_step_seated_ticks", "step_penetration_mm", "seating_blockers", "events")}
        | {f"{signal}_l2": float(np.linalg.norm(f[signal][:5])) for signal in SIGNALS}
        for f in r["trace"] if f["step"] >= 150
    ] for r in rows if r["index"] in FAILURES}
    report = dict(
        configuration=dict(corrected=True, control_hz=FPS, penetration_limit_mm=3.,
                           bottom_trigger_depth_mm=28.5, bottom_trigger_cumulative_penetration_mm=1.8,
                           bottom_trigger_requires_no_prior_seating=True, fixed_gripper_after_trigger=GRIP,
                           signal_sample="post-step qpos/qvel/qfrc; errors against the same post-step qpos",
                           frame_norm="L2 on first five axes; per-axis percentiles are absolute",
                           actuator_effort="simulated generalized torque in N m; a proxy for motor load",
                           auroc="bottom/seated positive; pooled descriptive AUROC with ties at half weight"),
        baseline=dict(
            episodes=30, archived_insertions=25, insertions=sum(r["success"] for r in rows),
            slip_episodes=sum(bool(r["slipped"]) for r in rows),
            outcomes=Counter(r["terminal"] for r in rows),
            failures=[r["index"] for r in rows if not r["success"]],
            flips=[r["index"] for r, p in zip(rows, previous, strict=True) if r["success"] != p["success"]],
            wrist_p95_mm=float(np.quantile(np.concatenate([r["wrist_steps_mm"] for r in rows]), .95)),
            rows=[{k: v for k, v in r.items() if k not in ("trace", "wrist_steps_mm")} for r in rows],
        ),
        oracle=oracle, oracle_comparison=oracle_comparison, failure_frames=detail,
        phase_quantiles=phases, air_vs_bottom_auroc=auc,
    )
    write_json(args.output / "report.json", report)
    print("BASELINE", {k: v for k, v in report["baseline"].items() if k != "rows"})
    print("TRIGGER / HOLD                 ARMED USED RESCUED")
    for r in oracle_comparison:
        print(f'{r["trigger"]:14} {r["hold"]:16} {r["triggered"]:5} {r["applied"]:4} {r["successful_with_hold"]}')
    for signal in SIGNALS:
        key = f"{signal}_l2"
        print("SIGNAL", key, phases["air_combined"]["signals"][key],
              phases["bottom_combined"]["signals"][key], "AUROC", round(auc[key], 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "summarize"))
    for name in ("checkpoint", "baseline", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
