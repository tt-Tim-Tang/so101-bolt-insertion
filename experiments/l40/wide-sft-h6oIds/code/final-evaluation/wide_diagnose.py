"""Localize acquisition failures on the locked policy and compare seen/unseen starts."""
import argparse
import io
import json
from pathlib import Path

import numpy as np

from . import wide_eval
from .online import Policy
from .wide_env import WideInsertion
from .online import CAMERA_MAP, diffusion_noise
from .scene import TASK


def reference_actions(policy, root):
    import mujoco
    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    episodes=np.linspace(0,63,16,dtype=int).tolist()
    ticks=[35,75,94,110,125,145,185,225,290,395]
    files=sorted((root/"dataset/data").rglob("*.parquet"))
    inputs=pq.read_table(files,columns=["episode_index","frame_index","observation.state",*CAMERA_MAP],
                         filters=[("episode_index","in",episodes),("frame_index","in",ticks)])
    targets=pq.read_table(files,columns=["episode_index","frame_index","action"],
                          filters=[("episode_index","in",episodes)])
    actions={(r["episode_index"],r["frame_index"]):r["action"] for r in targets.to_pylist()}
    env=WideInsertion()
    try:
        with (root/"reference-actions.jsonl").open("x") as log:
            for row in inputs.to_pylist():
                e,t=row["episode_index"],row["frame_index"]
                obs={"task":TASK,"observation.state":np.asarray(row["observation.state"],dtype=np.float32)}
                for key in CAMERA_MAP:
                    obs[key]=np.array(Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB"))
                gen=torch.Generator(device="cuda").manual_seed(922+t+e*1000)
                noise=diffusion_noise(policy.policy.config,gen)
                pred=policy.joint_targets(policy.decide(obs,False,noise=noise).reference)
                ref=np.array([actions[e,t+j] for j in range(10)])
                def grasp(q):
                    env.ik_data.qpos[env.joints]=q
                    mujoco.mj_forward(env.model,env.ik_data)
                    return env.ik_data.site("grasp").xpos.copy()
                xy=1000*np.linalg.norm((grasp(pred[-1])-grasp(ref[-1]))[:2])
                print(json.dumps(dict(episode=e,tick=t,arm_mae_rad=float(abs(pred[:,:5]-ref[:,:5]).mean()),
                                      gripper_mae_rad=float(abs(pred[:,5]-ref[:,5]).mean()),
                                      final_gripper_rad=float(pred[-1,5]),expert_gripper_rad=float(ref[-1,5]),
                                      requested_grasp_xy_difference_mm=float(xy))),file=log)
    finally:
        env.close()


class MeasuredInsertion(WideInsertion):
    def start(self, case):
        super().start(case)
        self.max_lift = 0.
        self.nearest_pick = None
        self.nearest_close = None
        self.close_ticks = 0
        self.max_lift_at_jaws = 0.

    def step(self, action):
        reward,done,timeout,info=super().step(action)
        grasp=self.data.site("grasp").xpos
        bolt=self.data.body("bolt")
        head=bolt.xpos+.045*bolt.xmat.reshape(3,3)[:,2]
        lift=self.pickup_state()[0]
        self.max_lift=max(self.max_lift,lift)
        if self.pickup_state()[-1]:
            self.max_lift_at_jaws=max(self.max_lift_at_jaws,lift)
        if abs(head[2]-grasp[2])<=.01 and lift<.012:
            xy=float(np.linalg.norm(head[:2]-grasp[:2]))*1000
            self.nearest_pick=xy if self.nearest_pick is None else min(xy,self.nearest_pick)
            if action[-1]<0:
                self.close_ticks+=1
                self.nearest_close=xy if self.nearest_close is None else min(xy,self.nearest_close)
        info.update(max_lift_mm=1000*self.max_lift,max_lift_at_jaws_mm=1000*self.max_lift_at_jaws,
                    closest_at_head_height_mm=self.nearest_pick,closest_while_closing_mm=self.nearest_close,
                    close_ticks_at_head_height=self.close_ticks)
        return reward,done,timeout,info


def diagnose(policy, root, validation):
    reference_actions(policy,root)
    attempts=[json.loads(line) for line in (root/"collection.jsonl").read_text().splitlines()]
    train=[r["case"] for r in attempts if r["result"]["success"]]
    for name,cases in (("seen",[train[i] for i in np.linspace(0,63,16,dtype=int)]),("validation",validation)):
        wide_eval.evaluate(policy,cases,root/f"diagnose-{name}.jsonl",env_type=MeasuredInsertion)
    wide_eval.evaluate(policy, validation[:8], root/"diagnose-single.jsonl", env_type=MeasuredInsertion,batch_size=1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--tokenizer",required=True)
    args=p.parse_args()
    root=args.root
    selection=json.loads((root/"selection.json").read_text())
    policy=Policy(Path(selection["checkpoint"]),None,0,"cuda",args.tokenizer)
    val=json.loads((root/"splits.json").read_text())["validation"]
    diagnose(policy,root,val)


if __name__=="__main__":
    main()
