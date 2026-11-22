"""Audit workspace, freeze independent splits and record the 64-cell SFT dataset."""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from .scene import HOME, IMAGE_SIZE, JOINTS, MOUTH_Z, SOURCE, TARGET
from .wide_env import WideInsertion, expert


def make_case(rng, source, target):
    return dict(source=np.asarray(source).tolist(), target=np.asarray(target).tolist(),
                joints=rng.uniform(-.05, .05, 5).tolist(), noise=int(rng.integers(2**31)))


def trial(env, case, **kwargs):
    try:
        return expert(env, case, **kwargs)
    except RuntimeError as exc:
        if "Unreachable grasp waypoint" not in str(exc):
            raise
        return dict(success=False, reason=str(exc), phase=env.phase, pickup=env.pick, arrival=env.arrive)


def projected(env, cam, point):
    camera = env.data.camera(cam)
    x, y, z = camera.xmat.reshape(3,3).T @ (point-camera.xpos)
    f = 1/np.tan(np.deg2rad(env.model.camera(cam).fovy[0]/2))
    return bool(z < 0 and max(abs(x), abs(y))*f < -z)


def calibration(env):
    rows = []
    for offset in (0., .5, .9, 1.25):
        env.start(dict(source=[0,0], target=[0,0], joints=[0]*5))
        i = env.bolt_address
        env.data.qpos[i:i+7] = [TARGET[0]+offset/1000, TARGET[1], MOUTH_Z+.0005, 1,0,0,0]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        z = []
        for _ in range(100):
            _, _, _, info = env.step(HOME)
            z.append(float(env.data.body("bolt").xpos[2]))
        rows.append(dict(offset_mm=offset, settled_z_range_mm=1000*np.ptp(z[-25:]), **info))
    return rows


def audit(env, args, dst):
    rng = np.random.default_rng(632310)
    source, target = args.source, args.target
    grid = list(itertools.product((-1,0,1), repeat=2))
    cases = [make_case(rng,np.array(s)*source,np.array(t)*target) for s,t in itertools.product(grid,grid)]
    cases += [make_case(rng,rng.uniform(-source,source,2),rng.uniform(-target,target,2)) for _ in range(20)]
    with (dst/"audit.jsonl").open("x", buffering=1) as f:
        for c in cases:
            result = trial(env,c)
            retries=[]
            if not result["success"]:
                for _ in range(4):
                    retry=make_case(rng,c["source"],c["target"])
                    outcome=trial(env,retry)
                    retries.append(dict(case=retry,result=outcome))
                    if outcome["success"]:
                        break
            s, t = [env.model.body(n).pos[:2] for n in ("source","target")]
            overlap = bool(abs(s[0]-t[0]) < .060 and abs(s[1]-t[1]) < .045)
            visible = all(projected(env,"top",np.r_[p+[dx,dy],MOUTH_Z])
                          for p in (s,t) for dx,dy in itertools.product((-.030,.030),(-.0225,.0225)))
            print(json.dumps(dict(case=c, **result, retries=retries, overlap=overlap, top_in_frame=visible)),file=f)
    (dst/"calibration.json").write_text(json.dumps(calibration(env),indent=2))


def freeze(args,dst):
    rng = np.random.default_rng(228115)
    grid = np.asarray(list(itertools.product(range(8), repeat=2)))
    source_cells = rng.permutation(grid)
    def strat(cell,radius):
        return ((cell+rng.random(2))/8*2-1)*radius
    train = [dict(**make_case(rng,strat(sc,args.source),strat(tc,args.target)),
                  source_cell=sc.tolist(),target_cell=tc.tolist()) for sc,tc in zip(source_cells,grid)]
    def draw(n):
        return [make_case(rng,rng.uniform(-args.source,args.source,2),rng.uniform(-args.target,args.target,2))
                for _ in range(n)]
    doc=dict(source_range_mm=args.source,target_range_mm=args.target,train=train,validation=draw(30),test=draw(50),
             selection="maximize validation insertions, then alignments, arrivals, pickups; earlier step breaks ties")
    with (dst/"splits.json").open("x") as stream:
        json.dump(doc,stream,indent=2)
    with (dst/"environment.json").open("x") as stream:
        json.dump(dict(action="6 absolute joint positions, radians, gripper last",control_hz=50,
                       demonstration_hz=50,chunk_executed=10,episode_ticks=1000,physics_dt=.002,
                       shaft_radius_mm=5,hole_apothem_mm=6,straight_walls=16,chamfer=False,
                       floor_top_mm=13,mouth_mm=45,target_appearance="rendered collision geoms",
                       friction="unchanged from scene.py",stable_ticks=25,
                       seat="bottom rim gap [-0.2,2] mm; shaft cross sections within all bore faces; tilt < 0.10 rad",
                       initial_arm_jitter_rad=.05,source_xy_mm=list(np.array(SOURCE)*1000),
                       target_xy_mm=list(np.array(TARGET)*1000)),stream,indent=2)


def collect(env,dst):
    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    splits=json.loads((dst/"splits.json").read_text())
    features={k:dict(dtype="float32",shape=(6,),names=list(JOINTS)) for k in ("observation.state","action")}
    for name in ("top","wrist"):
        features["observation.images."+name]=dict(dtype="image",shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                                    names=["height","width","channel"])
    ds=LeRobotDataset.create(repo_id="local/so101_wide_sft",fps=50,root=dst/"dataset",features=features,
                             robot_type="so101_mujoco",use_videos=False,video_backend="pyav")
    rng=np.random.default_rng(100805)
    try:
        with (dst/"collection.jsonl").open("x", buffering=1) as log:
            for i,original in enumerate(splits["train"]):
                case=original
                for attempt in range(20):
                    frames=[]
                    def record(action):
                        frames.append(dict(**env.observation(),action=action))
                    result=trial(env,case,record=record)
                    print(json.dumps(dict(cell=i,attempt=attempt,case=case,actual_initial=env.initial,result=result)),file=log)
                    if result["success"]:
                        for frame in frames:
                            ds.add_frame(frame.copy())
                        ds.save_episode()
                        if i in (0,7,27,56,63):
                            movie=cv2.VideoWriter(str(dst/f"demo-{i}.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),
                                                  50,(IMAGE_SIZE*2,IMAGE_SIZE))
                            if not movie.isOpened():
                                raise RuntimeError("Could not open video writer")
                            for frame in frames:
                                rgb=np.concatenate([frame["observation.images."+c] for c in ("top","wrist")],axis=1)
                                movie.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                            movie.release()
                        print(i,attempt,len(frames),flush=True)
                        break
                    def resample(key):
                        return ((np.array(original[key+"_cell"])+rng.random(2))/8*2-1)*splits[key+"_range_mm"]
                    case=dict(**make_case(rng,resample("source"),resample("target")),
                              source_cell=original["source_cell"],target_cell=original["target_cell"])
                else:
                    raise RuntimeError(f"Unfilled target cell {i}")
    finally:
        ds.finalize()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode", choices=("audit","freeze","collect"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source",type=float,default=20)
    p.add_argument("--target",type=float,default=40)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    if args.mode=="freeze":
        freeze(args,args.out)
        return
    env=WideInsertion()
    try:
        if args.mode=="audit":
            audit(env,args,args.out)
        else:
            collect(env,args.out)
    finally:
        env.close()


if __name__=="__main__":
    main()
