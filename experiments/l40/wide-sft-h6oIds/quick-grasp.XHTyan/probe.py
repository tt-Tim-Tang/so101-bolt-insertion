import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_model

from examples.so101_insertion.online import CAMERA_MAP, diffusion_noise, sft_frame
from examples.so101_insertion.wide_env import WideInsertion


@torch.inference_mode()
def probe(root, out, policy):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    frames=pq.read_table(sorted((root/"dataset/data").rglob("*.parquet")),
                         columns=["index","episode_index","frame_index","action"]).to_pylist()
    ids=np.linspace(0,63,16,dtype=int).tolist()
    seq={e:sorted((r for r in frames if r["episode_index"]==e),key=lambda r:r["frame_index"]) for e in ids}
    demos={e:np.array([r["action"] for r in seq[e]],np.float32) for e in ids}
    attempts=[json.loads(l) for l in (root/"collection.jsonl").read_text().splitlines()]
    cases=[r["case"] for r in attempts if r["result"]["success"]]
    ds=LeRobotDataset("local/so101_wide_sft",root=root/"dataset",video_backend="pyav")
    env=WideInsertion()
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    ticks=(35,75,94,110,118,125)
    snapshots={}
    observations={}
    checks=[]

    def restore(e,t):
        env.start(cases[e])
        mujoco.mj_setState(env.model,env.data,snapshots[e,t],spec)
        mujoco.mj_forward(env.model,env.data)
        env.steps=t

    def issue(obs,seed):
        gen=torch.Generator(device="cuda").manual_seed(seed)
        z=diffusion_noise(policy.policy.config,gen)
        return policy.joint_targets(policy.decide(obs,False,noise=z).reference)

    def run10(actions):
        q=[]
        for action in actions:
            env.step(action)
            q.append(float(env.state()[-1]))
        return env.data.site("grasp").xpos.copy(),np.array(q)

    for e in ids:
        env.start(cases[e])
        for t in range(136):
            if t in ticks:
                st=np.empty(mujoco.mj_stateSize(env.model,spec))
                mujoco.mj_getState(env.model,env.data,st,spec)
                snapshots[e,t]=st
                obs=env.observation()
                observations[e,t]=obs
                if e in (ids[0],ids[len(ids)//2],ids[-1]) and t in (75,118):
                    r=ds[int(seq[e][t]["index"])]
                    b1=policy.preprocess({k:r[k] for k in ("task","observation.state",*CAMERA_MAP)})
                    b2=policy.preprocess(sft_frame(obs))
                    diff={k:float((v.float()-b2[k].float()).abs().max()) for k,v in b1.items()
                          if isinstance(v,torch.Tensor)}
                    seed=torch.Generator(device="cuda").manual_seed(922+t+e*1000)
                    noise=diffusion_noise(policy.policy.config,seed)
                    def cmd(b):
                        return policy.joint_targets(policy.policy.predict_action_chunk(b,noise=noise)[0,:10].float().cpu())
                    checks.append(dict(e=e,t=t,delta=diff,action_max_rad=float(np.abs(cmd(b1)-cmd(b2)).max())))
            env.step(demos[e][t])
    (out/"path.json").write_text(json.dumps(checks,indent=2))
    def fingerprint(ck):
        paths=[*ck.glob("policy_*"),ck/"config.json"]
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!="config.json"}
    ck0=root/"train/checkpoints/002000/pretrained_model"
    for step in (2000,4000,6000):
        ck=root/f"train/checkpoints/{step:06d}/pretrained_model"
        if step!=2000:
            assert fingerprint(ck)==fingerprint(ck0)
            load_model(policy.policy,ck/"model.safetensors",strict=True)
        with (out/f"chunks-{step}.jsonl").open("x",buffering=1) as f:
            for e in ids:
                for t in ticks:
                    obs=observations[e,t]
                    pred=issue(obs,922+t+e*1000)
                    ref=demos[e][t:t+10]
                    restore(e,t)
                    xp,qp=run10(pred)
                    restore(e,t)
                    xr,qr=run10(ref)
                    def cross(q,th):
                        hits=np.flatnonzero(q<th)
                        return int(hits[0]) if len(hits) else None
                    print(json.dumps(dict(e=e,t=t,xy_mm=1000*np.linalg.norm((xp-xr)[:2]),
                                          z_err_mm=1000*(xp-xr)[2],grip_rad=float(qp[-1]-qr[-1]),
                                          pred_cmd_cross=cross(pred[:,5],0),ref_cmd_cross=cross(ref[:,5],0),
                                          pred_measured_cross=cross(qp,.15),ref_measured_cross=cross(qr,.15))),file=f)
        print("chunks",step,flush=True)
    with (out/"takeover.jsonl").open("x",buffering=1) as log:
        for e in ids[::2]:
            for mode in ("expert","model","expert_arm","expert_grip"):
                restore(e,94)
                policy.policy.reset()
                first_close=None
                first_measured=None
                # 130 ticks is 2.6s, including 0.6s beyond the demonstrated lift.
                for t in range(94,224,10):
                    acts=demos[e][t:t+10].copy()
                    if mode!="expert":
                        model=issue(env.observation(),922+t+e*1000)
                        if mode in ("model","expert_grip"):
                            acts[:,:5]=model[:,:5]
                        if mode in ("model","expert_arm"):
                            acts[:,5]=model[:,5]
                    for j,a in enumerate(acts):
                        env.step(a)
                        if first_close is None and a[-1]<0:
                            first_close=t+j
                        if first_measured is None and env.state()[-1]<.15:
                            first_measured=t+j
                    if env.pick:
                        break
                print(json.dumps(dict(e=e,mode=mode,pick=env.pick,close=first_close,
                                      measured=first_measured)),file=log)
        print("takeover complete",flush=True)
    env.close()
