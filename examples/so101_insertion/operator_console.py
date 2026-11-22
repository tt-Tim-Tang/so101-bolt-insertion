"""File-based operator decisions at paused MuJoCo chunk boundaries."""

import json
import time

import numpy as np

from .scene import MOUTH_Z


class InsertionOperator:
    def __init__(self, env, output):
        self.env = env
        self.output = output

    def reset(self, seed):
        self.seed = seed
        self.next_review = 0
        self.human_steps = 0
        self.end_reason = None
        self.events = []
        self.plan = iter(())

    def review(self, critical, human_active):
        env = self.env
        if env.steps < self.next_review or not critical and env.steps < 250:
            return None
        info = env.info()
        reason_id = f"{self.seed}-{env.steps}"
        lift, _, _, at_jaws = env.pickup_state()
        tip = env.data.body("bolt").xpos.copy()
        grasp = env.data.site("grasp").xpos.copy()
        target = env.model.body("target").pos[:2].copy()
        state = dict(request=reason_id, seed=self.seed, step=env.steps, human_active=human_active,
                     bolt_at_jaws=at_jaws, lift_m=lift, bolt_tip=tip.tolist(), grasp=grasp.tolist(),
                     target_xy=target.tolist(), human_steps=self.human_steps, **info)
        temp = self.output / "operator-state.tmp"
        temp.write_text(json.dumps(state), encoding="utf-8")
        temp.replace(self.output / "operator-state.json")
        print("OPERATOR WAIT", json.dumps(state), flush=True)
        path = self.output / "operator-command.json"
        while True:
            if not path.is_file():
                time.sleep(.15)
                continue
            try:
                command = json.loads(path.read_text(encoding="utf-8"))
                path.unlink()
                if command["request"] != reason_id:
                    raise ValueError("stale request")
                mode, reason = command["action"], command["reason"]
                if mode not in {"policy", "align", "insert", "finish"} or not isinstance(reason, str) or not reason.strip():
                    raise ValueError("expected an operator action and reason")
                if mode != "finish":
                    blocks = command["blocks"]
                    if type(blocks) is not int or not 1 <= blocks <= 100:
                        raise ValueError("blocks must be an integer in [1, 100]")
                if mode in {"align", "insert"} and (not critical or not at_jaws):
                    raise ValueError("insertion correction requires a held bolt and an active critical phase")
            except (KeyError, TypeError, ValueError) as exc:
                print("OPERATOR rejected:", exc, flush=True)
                continue
            break
        self.events.append(command)
        with (self.output / "operator-events.jsonl").open("a", encoding="utf-8") as log:
            print(json.dumps(dict(state=state, command=command)), file=log)
        if mode == "finish":
            self.end_reason = reason
            return "finish"
        count = 10 * blocks
        self.next_review = env.steps + count
        if mode == "policy":
            return "resume" if human_active else None

        # Live bolt-to-jaws offset, as in demonstrate.py. Actions start at the
        # last commanded joints so opening the console cannot jump the arm.
        start = env.data.ctrl.copy()
        offset = tip - grasp
        tip_z = MOUTH_Z + .012 if mode == "align" else MOUTH_Z - .024
        end = env.inverse_kinematics(np.r_[target - offset[:2], tip_z - offset[2]], start)
        end[-1] = start[-1]
        self.plan = iter((1 - u) * start + u * end for i in range(1, count + 1)
                         for u in [(i/count)**2 * (3 - 2*i/count)])
        return None if human_active else "takeover"

    def joint_target(self):
        return next(self.plan)
