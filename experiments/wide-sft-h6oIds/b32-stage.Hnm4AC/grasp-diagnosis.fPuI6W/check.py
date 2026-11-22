import json,os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from examples.so101_insertion.online import Policy,diffusion_noise
from examples.so101_insertion.wide_diagnose import MeasuredInsertion

root=Path(os.environ["BASE"])
stage=Path(os.environ["BATCH_RUN"])
out=Path(os.environ["GRASP_DIAG"])
ids=np.linspace(0,63,16,dtype=int)[::2].tolist()
frames=pq.read_table(sorted((root/"dataset/data").rglob("*.parquet")),columns=["episode_index","frame_index","action"]).to_pylist()
cases=[r["case"] for r in map(json.loads,(root/"collection.jsonl").read_text().splitlines()) if r["result"]["success"]]
env=MeasuredInsertion()
policy=Policy(stage/"train/checkpoints/last/pretrained_model",None,0,"cuda",Path(os.environ["SRC"])/"tokenizer",execution_steps=10)
with torch.inference_mode(),(out/"results.jsonl").open("x",buffering=1) as f:
 for e in ids:
  demo=np.array([r["action"] for r in sorted((r for r in frames if r["episode_index"]==e),key=lambda r:r["frame_index"])],np.float32)
  for start in (0,94):
   for mode in (("expert","model","expert_arm","expert_grip") if start==0 else ("model","expert_arm","expert_grip")):
    env.start(cases[e])
    policy.policy.reset()
    for a in demo[:start]: env.step(a)
    gen=torch.Generator(device="cuda").manual_seed(cases[e]["noise"])
    first=None
    for t in range(start,224,10):
     acts=demo[t:min(t+10,224)].copy()
     if mode!="expert":
      z=diffusion_noise(policy.policy.config,gen)
      pred=policy.joint_targets(policy.decide(env.observation(),False,noise=z).reference)
      if mode in ("model","expert_grip"): acts[:,:5]=pred[:len(acts),:5]
      if mode in ("model","expert_arm"): acts[:,5]=pred[:len(acts),5]
     for j,a in enumerate(acts):
      _,_,_,info=env.step(a)
      if first is None and env.state()[-1]<.15:
       bolt=env.data.body("bolt")
       delta=1000*(env.data.site("grasp").xpos-bolt.xpos-.045*bolt.xmat.reshape(3,3)[:,2])
       first=dict(t=t+j,xy_mm=float(np.linalg.norm(delta[:2])),z_mm=float(delta[2]))
      if env.pick: break
     if env.pick: break
    print(json.dumps(dict(e=e,start=start,mode=mode,pick=env.pick,first_close=first,near_mm=info["closest_while_closing_mm"],lift_mm=info["max_lift_at_jaws_mm"])),file=f)
  print("completed",e,flush=True)
env.close()
