"""Evaluate full-task policies, fixed commands and a paired camera-only intervention."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .online import Policy, diffusion_noise, sft_frame
from .scene import TARGET
from .wide_env import WideInsertion, expert


@torch.inference_mode()
def evaluate(policy, cases, output):
    envs=[WideInsertion() for _ in range(4)]
    try:
        with output.open("x", buffering=1) as f:
            for start in range(0,len(cases),len(envs)):
                group=cases[start:start+len(envs)]
                for e,c in zip(envs,group):
                    e.start(c)
                generators=[torch.Generator(device="cuda").manual_seed(c["noise"]) for c in group]
                active=list(range(len(group)))
                while active:
                    batches=[policy.preprocess(sft_frame(envs[i].observation())) for i in active]
                    batch={k:torch.cat([b[k] for b in batches]) for k,v in batches[0].items()
                           if isinstance(v,torch.Tensor)}
                    noise=torch.cat([diffusion_noise(policy.policy.config,generators[i]) for i in active])
                    predictions=policy.policy.predict_action_chunk(batch,noise=noise)[:,:10].float().cpu()
                    remaining=[]
                    for i,pred in zip(active,predictions):
                        env=envs[i]
                        for action in policy.joint_targets(pred):
                            _,done,timeout,info=env.step(action)
                            if done or timeout:
                                print(json.dumps(dict(index=start+i,case=group[i],steps=env.steps,**info)),file=f)
                                break
                        else:
                            remaining.append(i)
                    active=remaining
    finally:
        for e in envs:
            e.close()


def controls(root,cases):
    import pyarrow.parquet as pq
    frames=pq.read_table(sorted((root/"dataset/data").rglob("*.parquet")),columns=["action","frame_index"],
                         filters=[("episode_index","=",0)]).sort_by("frame_index")
    commands=np.asarray(frames["action"].to_pylist())
    class Picked(Exception):
        pass
    def interrupt(env):
        raise Picked
    env=WideInsertion()
    try:
        with (root/"fixed-validation.jsonl").open("x",buffering=1) as log:
            for i,c in enumerate(cases):
                for mode in ("full","after_pick"):
                    env.start(c)
                    if mode=="after_pick":
                        try:
                            info=expert(env,c,at_pick=interrupt)
                        except Picked:
                            assert env.steps==194
                        else:
                            print(json.dumps(dict(index=i,mode=mode,setup_failed=True,**info)),file=log)
                            continue
                    while True:
                        _,done,timeout,info=env.step(commands[min(env.steps,len(commands)-1)])
                        if done or timeout:
                            break
                    print(json.dumps(dict(index=i,mode=mode,setup_failed=False,**info)),file=log)
    finally:
        env.close()


@torch.inference_mode()
def intervention(policy,cases,output):
    env=WideInsertion()
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    try:
        with output.open("x",buffering=1) as log:
            for i,c in enumerate(cases):
                # Obtain a held state near the original hole, then swap only the
                # hole's visible position. All compared actions see identical physics.
                nominal=dict(c,target=[0,0])
                class Captured(Exception):
                    pass
                def capture(action):
                    if env.phase=="slow_approach":
                        raise Captured
                try:
                    expert(env,nominal,record=capture)
                except Captured:
                    pass
                else:
                    print(json.dumps(dict(index=i,available=False)),file=log)
                    continue
                state=np.empty(mujoco.mj_stateSize(env.model,spec))
                mujoco.mj_getState(env.model,env.data,state,spec)
                delta=np.asarray(c["target"])/1000
                endpoints=[]
                for sign in (1,-1):
                    env.model.body("target").pos[:2]=np.asarray(TARGET)+sign*delta
                    mujoco.mj_setState(env.model,env.data,state,spec)
                    mujoco.mj_forward(env.model,env.data)
                    gen=torch.Generator(device="cuda").manual_seed(c["noise"])
                    obs=env.observation()
                    act=policy.decide(obs,False,noise=diffusion_noise(policy.policy.config,gen))
                    env.model.body("target").pos[:2]=TARGET
                    mujoco.mj_setState(env.model,env.data,state,spec)
                    mujoco.mj_forward(env.model,env.data)
                    for a in policy.joint_targets(act.reference):
                        env.step(a)
                    endpoints.append(env.data.body("bolt").xpos[:2].copy())
                separation=float((endpoints[0]-endpoints[1]) @ (delta/np.linalg.norm(delta)))
                print(json.dumps(dict(index=i,available=True,signed_mm=1000*separation,
                                      target_separation_mm=2000*np.linalg.norm(delta))),file=log)
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("controls","validation","test","intervention","select"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--tokenizer")
    args=p.parse_args()
    splits=json.loads((args.root/"splits.json").read_text())
    if args.mode=="controls":
        controls(args.root,splits["validation"])
        return
    if args.mode=="select":
        scores={}
        for step in (2000,4000,6000):
            rows=[json.loads(l) for l in (args.root/f"validation-{step}.jsonl").read_text().splitlines()]
            assert sorted(r["index"] for r in rows)==list(range(30))
            scores[step]=[sum(r[key] for r in rows) for key in ("success","alignment","arrival","pickup")]
        best=max(scores,key=lambda step:(scores[step],-step))
        candidates=[p for p in (args.root/"train/checkpoints").iterdir() if p.name.isdecimal() and int(p.name)==best]
        assert len(candidates)==1
        args.checkpoint=candidates[0]/"pretrained_model"
        with (args.root/"selection.json").open("x") as log:
            json.dump(dict(scores=scores,step=best,checkpoint=str(args.checkpoint)),log,indent=2)
        policy=Policy(args.checkpoint,None,0,"cuda",args.tokenizer)
        intervention(policy,splits["validation"],args.root/"image-intervention.jsonl")
        evaluate(policy,splits["test"],args.root/"final-test.jsonl")
        return
    policy=Policy(args.checkpoint,None,0,"cuda",args.tokenizer)
    if args.mode=="intervention":
        intervention(policy,splits["validation"],args.root/"image-intervention.jsonl")
    else:
        evaluate(policy,splits[args.mode],args.root/f"{args.mode}-{args.checkpoint.parent.name}.jsonl")


if __name__=="__main__":
    main()
