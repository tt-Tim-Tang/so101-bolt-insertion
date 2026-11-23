"""Zero-update online-weight evaluation with post-grasp motion measurements."""
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np

from .fast16 import FastMetrics
from .fast16_train import evaluate
from .online import Policy


class MotionMetrics(FastMetrics):
    def start(self, case):
        super().start(case)
        self.pick_step = None
        self.peak_lift = 0.
        self.min_xy = self.min_held_xy = float("inf")
        self.closest_held_tip = self.last_held_tip = None
        self.lost_ticks = 0
        self.dropped = False
        self.trace = []
        self.initial_tip = self.data.body("bolt").xpos.copy()
        self.max_toward = 0.

    def step(self, action):
        reward, done, timeout, info = super().step(action)
        tip = self.data.body("bolt").xpos.copy()
        target = self.model.body("target").pos[:2]
        lift, _, _, jaws = self.pickup_state()
        if self.pick:
            if self.pick_step is None:
                self.pick_step = self.steps
            self.peak_lift = max(self.peak_lift, lift)
            distance = float(np.linalg.norm(tip[:2]-target))
            self.min_xy = min(self.min_xy, distance)
            if jaws:
                self.last_held_tip = tip.tolist()
                if distance < self.min_held_xy:
                    self.min_held_xy = distance
                    self.closest_held_tip = tip.tolist()
                direction = target-self.initial_tip[:2]
                direction /= np.linalg.norm(direction)
                self.max_toward = max(self.max_toward, float((tip[:2]-self.initial_tip[:2])@direction))
            # Count a persistent loss of the object before arrival, not commanded lowering.
            lost = not jaws and self.peak_lift-lift > .010 and not self.arrive
            self.lost_ticks = self.lost_ticks+1 if lost else 0
            self.dropped |= self.lost_ticks >= 10
        if self.steps % 10 == 0 or done or timeout:
            self.trace.append(dict(step=self.steps, tip_m=tip.tolist(),
                                   tcp_m=self.data.site("grasp").xpos.tolist(),
                                   picked=bool(self.pick), at_jaws=bool(jaws)))
        if done or timeout:
            info.update(pick_step=self.pick_step,
                        max_lift_after_pick_mm=1000*self.peak_lift if self.pick else None,
                        dropped_before_arrival=bool(self.dropped),
                        min_xy_after_pick_mm=1000*self.min_xy if self.pick else None,
                        min_xy_held_mm=1000*self.min_held_xy if np.isfinite(self.min_held_xy) else None,
                        max_toward_target_mm=1000*self.max_toward if self.pick else None,
                        initial_tip_m=self.initial_tip.tolist(), target_xy_m=target.tolist(),
                        closest_held_tip_m=self.closest_held_tip, last_held_tip_m=self.last_held_tip,
                        motion_trace=self.trace)
        return reward, done, timeout, info


def median(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else None


def main():
    root, out = Path(os.environ["FAST"]), Path(os.environ["DR"])
    checkpoint = Path(os.environ["DG"])/"update-500/pretrained_model"
    cases = json.loads((root/"splits.json").read_text())
    clearance = json.loads((root/"environment.json").read_text())["clearance_mm"]
    started = time.monotonic()
    wrapper = Policy(checkpoint, None, 0, "cuda", str(Path(os.environ["SRC"])/"tokenizer"))
    loaded = time.monotonic()
    summary = dict(weights="online_update_500", additional_optimizer_updates=0,
                   clearance_mm=clearance, groups={})
    for group in ("train", "validation"):
        path = out/f"{group}.jsonl"
        evaluate(wrapper.policy, (wrapper.preprocess, wrapper.postprocess), cases[group], path,
                 partial(MotionMetrics, target_bore_radius=.005+clearance/1000))
        rows = sorted([json.loads(x) for x in path.read_text().splitlines()], key=lambda r:r["index"])
        assert [r["case"] for r in rows] == cases[group]
        summary["groups"][group] = dict(n=len(rows), **{k:sum(r[k] for r in rows)
            for k in ("pickup", "arrival", "success", "dropped_before_arrival")},
            closure_observed=sum(r["closure_xy_mm"] is not None for r in rows),
            closure_xy_median_mm=median(rows,"closure_xy_mm"),
            max_lift_after_pick_median_mm=median(rows,"max_lift_after_pick_mm"),
            min_xy_after_pick_median_mm=median(rows,"min_xy_after_pick_mm"),
            min_xy_held_median_mm=median(rows,"min_xy_held_mm"),
            max_toward_target_median_mm=median(rows,"max_toward_target_mm"))
        print(group, summary["groups"][group], flush=True)
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
    summary.update(model_load_seconds=loaded-started, evaluation_seconds=time.monotonic()-loaded)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
