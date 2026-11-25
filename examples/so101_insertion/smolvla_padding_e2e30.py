"""Run corrected SmolVLA from the pickup fixture at the archived expansion targets."""

import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .smolvla25_data import FPS
from .smolvla_motion_audit import write_json
from .smolvla_padding_sft_eval import replay
from .smolvla_rlt_expanded_audit import TelemetryInsertion
from .smolvla_rlt_stage2 import FrozenSmolVLA


NEAR_RADIUS_MM = 15.
APPROACH_HEIGHT_MM = 15.
DEEP_INSERTION_MM = 15.


class EndToEndTelemetry(TelemetryInsertion):
    def start(self, case):
        super().start(case)
        self.previous_held = None
        self.height_crossing = None
        self.reached_vicinity = False
        self.max_near_depth_mm = 0.
        self.first_slip_step = None
        self.first_open_step = None
        self.wrist_steps_mm = []

    def record_held(self, pose):
        tip, axis, xy, depth, _ = pose
        head = tip + .045 * axis
        if not self.pick or np.linalg.norm(self.data.site(self.grasp_id).xpos - head) >= .025:
            self.previous_held = None
            return
        height = -depth * 1000
        radial = np.linalg.norm(xy) * 1000
        if radial <= NEAR_RADIUS_MM and height <= APPROACH_HEIGHT_MM:
            self.reached_vicinity = True
            self.max_near_depth_mm = max(self.max_near_depth_mm, -height)
        time_s = float(self.data.time - self.start_time)
        if self.height_crossing is None and self.previous_held is not None and height <= APPROACH_HEIGHT_MM:
            prev_height, prev_xy, prev_axis, prev_time_s = self.previous_held
            if prev_height > APPROACH_HEIGHT_MM:
                fraction = (prev_height - APPROACH_HEIGHT_MM) / (prev_height - height)
                offset = (1 - fraction) * prev_xy + fraction * xy
                if np.linalg.norm(offset) * 1000 <= NEAR_RADIUS_MM:
                    direction = (1 - fraction) * prev_axis + fraction * axis
                    direction /= np.linalg.norm(direction)
                    self.height_crossing = dict(elapsed_s=prev_time_s + fraction * (time_s - prev_time_s),
                                                xy_mm=float(np.linalg.norm(offset) * 1000),
                                                tilt_deg=math.degrees(math.acos(np.clip(direction[2], -1, 1))))
        self.previous_held = (height, xy.copy(), axis.copy(), time_s)

    def step(self, action):
        wrist_before = self.data.site(self.grasp_id).xpos.copy()
        outcome = super().step(action)
        wrist_after = self.data.site(self.grasp_id).xpos
        self.wrist_steps_mm.append(float(np.linalg.norm(wrist_after - wrist_before) * 1000))
        info = outcome[-1]
        if info["slipped"] and self.first_slip_step is None:
            self.first_slip_step = self.steps
        if info["opened_early"] and self.first_open_step is None:
            self.first_open_step = self.steps
        return outcome


def failure_mode(end, env):
    if end["success"]:
        return None, None
    if not end["pickup"]:
        return "pickup_failure", "Grasp-and-lift hold criterion never met"
    first = env.first_contact
    releases = [step for step in (env.first_slip_step, env.first_open_step) if step is not None]
    if releases and (first is None or min(releases) < first["step"]):
        detail = "Closed-gripper slip" if min(releases) == env.first_slip_step else "Early gripper opening"
        return "transport_drop", f"{detail} before first target contact"
    if end["stable_s"] > 0:
        return "seating_near_miss", "Entered the seating envelope without passing the original success check"
    if env.max_near_depth_mm >= DEEP_INSERTION_MM:
        return "deep_jam", "Held insertion reached at least 15 mm without seating"
    if first is None:
        return "rim_misalignment", "No target contact observed, so rim occlusion or jamming is unconfirmed"
    return "rim_misalignment", "Target contact followed by less than 15 mm held insertion"


def quantiles(values):
    if not len(values):
        return dict(n=0, median=None, p90=None, p95=None, max=None)
    q = np.quantile(values, (.5, .9, .95, 1.))
    return dict(n=len(values), **dict(zip(("median", "p90", "p95", "max"), q.tolist(), strict=True)))


def location_summary(rows, prefix):
    return {field: quantiles([row[f"{prefix}_{field}"] for row in rows if row[f"{prefix}_{field}"] is not None])
            for field in ("xy_mm", "tilt_deg")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("hole_mouth_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())["episodes"]
    prior = json.loads(args.hole_mouth_summary.read_text())["pairs"]
    keys = [(row["index"], row["seed"], row["case"]["target"]) for row in cases]
    if len(keys) != 30 or keys != [(r["index"], r["seed"], r["target_xy_mm"]) for r in prior]:
        raise ValueError("Expected the exact original 30 held-out positions and seeds")
    if sum(r["corrected"]["success"] for r in prior) != 27:
        raise ValueError("Expected the accepted 27/30 corrected hole-mouth result")
    if any(case["case"]["source"] != [0, 0] or case["case"]["joints"] != [0] * 5 for case in cases):
        raise ValueError("Expected the original fixed pickup fixture and default arm pose")

    args.output.mkdir()
    videos = args.output / "failures"
    videos.mkdir()
    features = FrozenSmolVLA(args.checkpoint)
    env = EndToEndTelemetry(penetration_limit_mm=3.)
    rows = []
    scratch_video = args.output / "active-rollout.mp4"
    try:
        with (args.output / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for case, previous in zip(cases, prior, strict=True):
                index, seed = case["index"], case["seed"]
                end, _ = replay(env, features, case["case"], seed, True, scratch_video)
                first = env.first_contact or {}
                crossing = env.height_crossing or {}
                nearby = bool(end["arrival"] or env.reached_vicinity)
                # Skip the control interval that straddled first contact.
                after_contact = env.wrist_steps_mm[first["step"]:] if first else []
                wrist = quantiles(after_contact)
                mode, detail = failure_mode(end, env)
                video = None
                if mode is not None:
                    video = f"failures/{index:02d}.mp4"
                    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(scratch_video),
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                    str(args.output / video)], check=True)
                scratch_video.unlink()
                prior_success = bool(previous["corrected"]["success"])
                row = dict(index=index, seed=seed, target_x_mm=case["case"]["target"][0],
                           target_y_mm=case["case"]["target"][1], pickup=end["pickup"], arrival=end["arrival"],
                           success=end["success"], slipped=end["slipped"], opened_early=end["opened_early"],
                           timeout=end["timeout"], penetration_failure=end["penetration_failure"], steps=end["steps"],
                           hole_mouth_success=prior_success, passed_hole_mouth_failed_e2e=prior_success and not end["success"],
                           reached_vicinity=nearby, approach15_xy_mm=crossing.get("xy_mm"),
                           approach15_tilt_deg=crossing.get("tilt_deg"), approach15_time_s=crossing.get("elapsed_s"),
                           first_contact_xy_mm=first.get("xy_mm"), first_contact_tilt_deg=first.get("tilt_deg"),
                           first_contact_depth_mm=first.get("depth_mm"), first_contact_time_s=first.get("elapsed_s"),
                           first_contact_step=first.get("step"), first_slip_step=env.first_slip_step,
                           first_open_step=env.first_open_step, max_held_depth_mm=env.max_near_depth_mm,
                           max_seated_ms=end["stable_s"] * 1000, postcontact_wrist_samples=wrist["n"],
                           postcontact_wrist_p95_mm=wrist["p95"], postcontact_wrist_max_mm=wrist["max"],
                           max_penetration_mm=end["max_penetration_mm"], failure_mode=mode, failure_detail=detail, video=video)
                print(json.dumps(row), file=log)
                rows.append(row)
                print(f"E2E {index:02d} {'inserted' if end['success'] else mode} steps={end['steps']} "
                      f"pickup={int(end['pickup'])} arrival={int(end['arrival'])} slip={end['slipped']}", flush=True)
    finally:
        env.close()

    metrics = ("pickup", "arrival", "success", "slipped", "timeout", "penetration_failure")
    near = [r for r in rows if r["reached_vicinity"]]
    paired = Counter((r["hole_mouth_success"], r["success"]) for r in rows)
    summary = dict(
        checkpoint=str(args.checkpoint), episodes=len(rows), control_hz=FPS, max_steps=20 * FPS,
        penetration_limit_mm=3., full_reset_from_pickup_fixture=True, restored_hole_mouth_state=False,
        totals={**{k: sum(r[k] for r in rows) for k in metrics},
                "slip_episodes": sum(r["slipped"] > 0 for r in rows), "mean_steps": float(np.mean([r["steps"] for r in rows]))},
        paired=dict(both_success=paired[True, True], both_failure=paired[False, False],
                    hole_mouth_only=[r["index"] for r in rows if r["passed_hole_mouth_failed_e2e"]],
                    e2e_only=[r["index"] for r in rows if not r["hole_mouth_success"] and r["success"]]),
        arrival_errors=dict(vicinity_episodes=len(near), vicinity_xy_radius_mm=NEAR_RADIUS_MM,
                            approach_height_mm=APPROACH_HEIGHT_MM, approach15=location_summary(near, "approach15"),
                            first_contact=location_summary(near, "first_contact")),
        failures=[r for r in rows if r["failure_mode"] is not None],
        failure_counts=dict(Counter(r["failure_mode"] for r in rows if r["failure_mode"] is not None)), rows=rows,
    )
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as log:
        writer = csv.DictWriter(log, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "summary.json", summary)
    print("EVAL_COMPLETE", summary["totals"], summary["paired"], flush=True)


if __name__ == "__main__":
    main()
