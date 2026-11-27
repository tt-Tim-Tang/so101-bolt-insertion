"""Screen corrected SmolVLA failures and record continuous-IK DAgger corrections."""

import argparse
import json
import subprocess
from collections import Counter, deque
from pathlib import Path

import mujoco
import numpy as np

from examples.so101_insertion.scene import IMAGE_SIZE, MOUTH_Z, TASK
from examples.so101_insertion.smolvla25_data import CAMERAS, FPS, MERGED_REPO_ID, create_dataset, make_case
from examples.so101_insertion.smolvla_padding_e2e30 import EndToEndTelemetry, failure_mode, replay
from examples.so101_insertion.smolvla_rlt_stage2 import GRIP, FrozenSmolVLA
from examples.so101_insertion.smolvla_rlt_v3_pool import fixed_cases, save_json


DAGGER_ID = "local/so101_smolvla25_dagger"
MIXED_ID = "local/so101_smolvla25_dagger_mixed"
LIMIT_MM = 3.
GATE_SIZE = 25
COLLECTION_SIZE = 120
STATS = "observation.state"


class StallInsertion(EndToEndTelemetry):
    """Hand control to the teacher at a stall in the same physics rollout."""

    def start(self, case):
        super().start(case)
        self.depths = deque(maxlen=16)
        self.teacher = None
        self.takeover = None
        self.frames = []
        self.peak_rise_mm = 0.
        self.first_jump_mrad = 0.
        self.peak_jump_mrad = 0.

    def step(self, action):
        if self.teacher is None:
            if self.first_contact is not None and self.pick and self.pickup_state()[-1]:
                _, _, xy, depth, tilt = self.bolt_pose()
                self.depths.append(depth)
                if len(self.depths) == 16 and depth-self.depths[0] < .0006 and self.seated_ticks == 0:
                    self.takeover = dict(step=self.steps, first_contact_step=self.first_contact["step"],
                                         xy_mm=float(np.linalg.norm(xy)*1000), tilt_deg=tilt, depth_mm=1000*depth,
                                         advance_mm=1000*(depth-self.depths[0]))
                    self.teacher = LockedRollTeacher(self)
            else:
                self.depths.clear()
        if self.teacher is None:
            return super().step(action)

        action = self.teacher.next_command(self)
        observation = self.observation()
        if observation["task"] != TASK or any(observation[k].shape != (IMAGE_SIZE,IMAGE_SIZE,3) for k in CAMERAS):
            raise RuntimeError("DAgger recording does not match the original observation schema")
        self.frames.append(observation | dict(action=action.copy()))
        jump = 1000*float(np.abs(action-self.data.ctrl).max())
        if len(self.frames) == 1:
            self.first_jump_mrad = jump
        self.peak_jump_mrad = max(self.peak_jump_mrad,jump)
        value, done, timeout, info = super().step(action)
        np.testing.assert_allclose(self.data.ctrl,action,atol=2e-7,rtol=0)
        if self.data.ctrl[4] != self.teacher.wrist_roll:
            raise RuntimeError("Teacher changed the locked wrist_roll command")
        rise = float(self.data.body(self.bolt_id).xpos[2]-self.teacher.start_tip[2])*1000
        self.peak_rise_mm = max(self.peak_rise_mm,rise)
        return value, done or bool(info["slipped"]), timeout or self.teacher.tick >= self.teacher.ticks, info


def locked_bolt_ik(env, position, axis, wrist_roll):
    """Fit the held tip and softly align its axis using the first four joints."""
    tcp = env.data.site("grasp")
    bolt = env.data.body(env.bolt_id)
    rot = tcp.xmat.reshape(3,3)
    local_tip = rot.T@(bolt.xpos-tcp.xpos)
    local_axis = rot.T@bolt.xmat.reshape(3,3)[:,2]
    posture = env.state().astype(float)
    posture[4:] = wrist_roll, GRIP
    q, best = posture.copy(), posture.copy()
    best_cost = np.inf
    jp = np.zeros((3,env.model.nv))
    jr = np.zeros_like(jp)
    body = int(env.model.site_bodyid[env.grasp_id])
    dofs = env.dofs[:4]
    axis_weight, posture_weight = .024, .016
    reg = posture_weight*np.eye(4)
    damping = .00004*np.eye(4)
    for _ in range(32):
        env.ik_data.qpos[env.joints] = q
        mujoco.mj_forward(env.model,env.ik_data)
        site = env.ik_data.site("grasp")
        rot = site.xmat.reshape(3,3)
        tip, up = site.xpos+rot@local_tip, rot@local_axis
        error = position-tip
        anchor = posture[:4]-q[:4]
        cost = np.sum(error**2)+axis_weight**2*np.sum((axis-up)**2)+posture_weight**2*np.sum(anchor**2)
        if cost < best_cost:
            best_cost, best = cost, q.copy()
        mujoco.mj_jac(env.model,env.ik_data,jp,jr,tip,body)
        jac = np.vstack((jp[:,dofs],axis_weight*(np.eye(3)-np.outer(up,up))@jr[:,dofs],reg))
        residual = np.r_[error,axis_weight*np.cross(up,axis),posture_weight*anchor]
        delta = np.linalg.solve(jac.T@jac+damping,jac.T@residual)
        following = np.clip(q[:4]+np.clip(delta,-.045,.045),env.low[:4],env.high[:4])
        if np.max(np.abs(following-q[:4])) < 1e-5:
            break
        q[:4] = following
    return best


class LockedRollTeacher:
    def __init__(self, env):
        tip, axis, _, _, _ = env.bolt_pose()
        self.start_tip, self.start_axis = tip.copy(),axis.copy()
        self.lift_z = tip[2]+.00065
        self.bottom = env.floor_z+.0005
        self.center = env.model.body(env.target_id).pos[:2].copy()
        self.command = env.data.ctrl.copy()
        self.wrist_roll = float(np.float32(self.command[4]))
        self.command[4:] = self.wrist_roll, GRIP
        self.xy_bias = np.zeros(2)
        self.last_delta = np.zeros(4)
        self.tick = 0
        self.align_ticks = FPS
        # Match the original expert's 65 ticks at 50 Hz, scaled to the local descent distance.
        self.down_per_tick = (MOUTH_Z+.005-self.bottom)/np.ceil(65*FPS/50)
        descent_ticks = int(np.ceil(max(0.,self.lift_z-self.bottom)/self.down_per_tick))
        self.ticks = self.align_ticks + descent_ticks + 75

    def next_command(self, env):
        if self.tick:
            blend = min(self.tick/self.align_ticks,1.)
            blend = blend*blend*(3.-2.*blend)
            up = (1-blend)*self.start_axis+blend*np.array([0.,0.,1.])
            up /= np.linalg.norm(up)
            tip, _, _, _, _ = env.bolt_pose()
            xy = (1-blend)*self.start_tip[:2]+blend*self.center
            self.xy_bias = np.clip(self.xy_bias+np.clip(.2*(xy-tip[:2]),-.0001,.0001),-.002,.002)
            z = (1-blend)*self.start_tip[2]+blend*self.lift_z
            if self.tick > self.align_ticks:
                z = max(self.bottom,self.lift_z-(self.tick-self.align_ticks)*self.down_per_tick)
            goal = locked_bolt_ik(env,np.r_[xy+self.xy_bias,z],up,self.wrist_roll)
            delta = np.clip(goal[:4]-self.command[:4],-.05,.05)
            self.last_delta += np.clip(delta-self.last_delta,-.006,.006)
            self.command[:4] = np.clip(self.command[:4]+self.last_delta,env.low[:4],env.high[:4])
        self.tick += 1
        return self.command.astype(np.float32)


def save_frames(path,frames):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**{k:np.stack([frame[k] for frame in frames]) for k in (*CAMERAS,STATS,"action")})
    tmp.replace(path)


def candidate_plan(args,held,train,val):
    reserved = np.array([r["target"] for r in train+val]
                        +[[r["target_x_mm"],r["target_y_mm"]] for r in held])
    monitor = json.loads(args.monitor.read_text())["monitor"]["base_success"]
    assert len(set(monitor))==10
    assert all(0<=i<len(train) for i in monitor)
    rng = np.random.default_rng(713309)
    cases = []
    for index in range(2600):
        while True:
            xy = rng.uniform(-18,18,2)
            if np.linalg.norm(xy-reserved,axis=-1).min()>1e-6:
                break
        reserved = np.vstack((reserved,xy))
        cases.append(dict(index=index,seed=878100+index,case=make_case(xy.tolist(),rng)))
    save_json(args.output/"plan.json",dict(monitor_holes=monitor,cases=cases))


def paths(args,folder):
    return sorted((args.output/folder).glob("[0-9]*.json"))


def screen(args, features, env):
    plan = json.loads((args.output/"plan.json").read_text())["cases"]
    folder = args.output/args.phase
    folder.mkdir(exist_ok=True)
    for i in range(args.start+args.shard,args.stop,args.shards):
        path = folder/f"{i:04d}.json"
        if path.exists():
            continue
        row = plan[i].copy()
        baseline,_ = replay(env,features,row["case"],row["seed"],corrected=True)
        row["base"] = {k:baseline[k] for k in ("success","slipped","opened_early","timeout","steps","max_penetration_mm")}
        row["base"]["reason"] = failure_mode(baseline,env)[0]
        if args.phase == "collect" and not baseline["success"]:
            teacher_env = StallInsertion(penetration_limit_mm=LIMIT_MM)
            try:
                record_takeover(teacher_env,features,row,path)
            finally:
                teacher_env.close()
        else:
            save_json(path,row)
        print("SCREEN",i,row["base"]["reason"] or "success",flush=True)


def dry_run(args,features,env):
    source = args.screened or args.output/"screen"
    failed = [p for p in sorted(source.glob("[0-9]*.json"))
              if not json.loads(p.read_text())["base"]["success"]][:GATE_SIZE]
    if len(failed) != GATE_SIZE:
        raise ValueError("Expected the original 25 screened failures")
    folder = args.output/args.gate_dir
    folder.mkdir(exist_ok=True)
    for path in failed[args.shard::args.shards]:
        result = folder/path.name
        if result.exists():
            continue
        row = json.loads(path.read_text())
        record_takeover(env,features,row,result)


def record_takeover(env,features,row,result):
    end,_ = replay(env,features,row["case"],row["seed"],corrected=True)
    clean = bool(env.takeover and end["success"] and not end["slipped"] and not end["opened_early"]
                 and not end["penetration_failure"])
    reason = failure_mode(end,env)[0]
    if clean:
        save_frames(result.with_suffix(".npz"),env.frames)
    teacher = dict(success=clean,inserted_without_takeover=bool(end["success"] and not env.takeover),
                   reason="success" if clean else reason or "no_clean_takeover",slipped=end["slipped"],
                   opened_early=end["opened_early"],timeout=end["timeout"],frames=len(env.frames) if clean else 0,
                   takeover_frames=len(env.frames),first_jump_mrad=env.first_jump_mrad,peak_jump_mrad=env.peak_jump_mrad,
                   peak_rise_mm=env.peak_rise_mm,max_penetration_mm=end["max_penetration_mm"],
                   max_depth_mm=env.max_near_depth_mm,
                   wrist_roll_rad=env.teacher.wrist_roll if env.teacher else None)
    case = {k:row[k] for k in ("index","seed","case","base")}
    save_json(result,case | dict(takeover=env.takeover,teacher=teacher))
    print("TEACH_GATE",row["index"],env.takeover,teacher,flush=True)

def report_gate(args):
    rows = [json.loads(p.read_text()) for p in paths(args,args.gate_dir)]
    assert len(rows) == GATE_SIZE
    saved = [r["teacher"] for r in rows]
    rescued = sum(r["success"] for r in saved)
    result = dict(episodes=len(rows),rescued=rescued,slips=sum(bool(r["slipped"]) for r in saved),
                  passed=rescued>=18 and not any(r["slipped"] for r in saved),
                  takeovers=sum(r["takeover"] is not None for r in rows),
                  inserted_without_takeover=sum(r["inserted_without_takeover"] for r in saved),
                  average_success_frames=float(np.mean([r["frames"] for r in saved if r["success"]])) if rescued else None,
                  first_max_jump_mrad=max(r["first_jump_mrad"] for r in saved),
                  reasons=dict(Counter(r["reason"] for r in saved)),rows=rows)
    save_json(args.output/f"{args.gate_dir}.json",result)
    print("TEACH_GATE_SUMMARY",{k:v for k,v in result.items() if k!="rows"},flush=True)


def build_dataset(args):
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    original = LeRobotDatasetMetadata(MERGED_REPO_ID,root=args.original/"dataset")
    if (original.total_episodes,original.total_frames,original.fps)!=(150,28649,FPS):
        raise RuntimeError("Wrong original 150-episode dataset")
    if list(original.tasks.index)!=[TASK] or set(original.camera_keys)!=set(CAMERAS):
        raise RuntimeError("Original dataset task text or camera keys differ")
    rows = [(json.loads(p.read_text()),p.with_suffix(".npz")) for p in paths(args,args.gate_dir if args.audit_dataset else "collect")]
    selected = [(r,p) for r,p in rows if r.get("teacher",{}).get("success")]
    if args.audit_dataset:
        assert len(rows)==GATE_SIZE
    else:
        selected = selected[:COLLECTION_SIZE]
        assert len(selected)==COLLECTION_SIZE
    if not selected:
        print("No successful teacher clips to audit",flush=True)
        return
    path = args.output/"dagger-dataset"
    dataset = create_dataset(path,DAGGER_ID)
    frames = 0
    try:
        for row,file in selected:
            assert row["teacher"]["slipped"]==0 and row["teacher"]["max_penetration_mm"]<=LIMIT_MM
            with np.load(file) as pack:
                arrays = {k:pack[k] for k in (*CAMERAS,STATS,"action")}
                count = row["teacher"]["frames"]
                assert all(len(v)==count for v in arrays.values())
                assert all(arrays[k].shape==(count,IMAGE_SIZE,IMAGE_SIZE,3) for k in CAMERAS)
                assert np.allclose(arrays["action"][:,-1],GRIP,rtol=0,atol=1e-8)
                assert np.all(arrays["action"][:,4]==row["teacher"]["wrist_roll_rad"])
                for i in range(count):
                    dataset.add_frame({k:v[i] for k,v in arrays.items()}|dict(task=TASK))
                dataset.save_episode()
                frames += count
    finally:
        dataset.finalize()
    destination = args.output/"mixed-dataset"
    aggregate_datasets([MERGED_REPO_ID,DAGGER_ID],MIXED_ID,roots=[args.original/"dataset",path],
                       aggr_root=destination,concatenate_videos=False,concatenate_data=False)
    added = LeRobotDatasetMetadata(DAGGER_ID,root=path)
    merged = LeRobotDatasetMetadata(MIXED_ID,root=destination)
    assert merged.total_frames==original.total_frames+frames and merged.total_episodes==150+len(selected)
    assert merged.fps==added.fps==FPS and list(merged.tasks.index)==list(added.tasks.index)==[TASK]
    for key in (*CAMERAS,STATS,"action"):
        assert tuple(original.features[key]["shape"])==tuple(added.features[key]["shape"])==tuple(merged.features[key]["shape"])
    report = dict(repo_id=MIXED_ID,original_episodes=original.total_episodes,original_frames=original.total_frames,
                  dagger_episodes=added.total_episodes,dagger_frames=frames,fps=FPS,task=TASK,camera_keys=CAMERAS,
                  attempted=len(rows),base_failures=sum(not r["base"]["success"] for r,_ in rows),
                  teacher_successes=sum(bool(r.get("teacher",{}).get("success")) for r,_ in rows),
                  selected=[r for r,_ in selected])
    save_json(args.output/"dataset.json",report)
    print("DAGGER_DATASET",{k:v for k,v in report.items() if k!="selected"},flush=True)

    if not args.audit_dataset:
        return
    videos = args.output/"videos"
    videos.mkdir(exist_ok=True)
    for row,file in selected[:3]:
        with np.load(file) as pack:
            rgb = np.concatenate([pack[key] for key in CAMERAS],axis=2)
        video = videos/f'{row["index"]:04d}.mp4'
        subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
                        "-video_size",f"{2*IMAGE_SIZE}x{IMAGE_SIZE}","-framerate",str(FPS),"-i","pipe:0",
                        "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart","-threads","1",str(video)],
                       input=rgb.tobytes(),check=True)


def evaluate(args,env,features,held):
    folder = args.output/f"eval-{args.label}"
    (folder/"videos").mkdir(parents=True,exist_ok=True)
    for base in held[args.shard::args.shards]:
        i = base["index"]
        path = folder/f"{i:02d}.json"
        if path.exists():
            continue
        video = folder/"videos"/f"{i:02d}.mp4"
        raw = video.with_suffix(".raw.mp4")
        case = dict(target=[base["target_x_mm"],base["target_y_mm"]],source=[0,0],joints=[0]*5,noise=base["seed"])
        end,_ = replay(env,features,case,base["seed"],corrected=True,video=raw)
        subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(raw),"-c:v","libx264",
                        "-pix_fmt","yuv420p","-movflags","+faststart","-threads","1",str(video)],check=True)
        raw.unlink()
        mode,_ = failure_mode(end,env)
        save_json(path,dict(index=i,seed=base["seed"],target=case["target"],base_success=base["success"],
                            failure_mode=mode,max_held_depth_mm=env.max_near_depth_mm,
                            wrist_steps_mm=env.wrist_steps_mm,video=video.name,**end))
        print("DAGGER_EVAL",args.label,i,mode or "success",flush=True)


def summarize(args):
    report = {}
    for label in (21000,22500,25000):
        folder = args.output/f"eval-{label}"
        rows = [json.loads(p.read_text()) for p in sorted(folder.glob("[0-9][0-9].json"))]
        assert len(rows)==30 and sum(r["base_success"] for r in rows)==21
        report[label] = dict(insertions=sum(r["success"] for r in rows),slips=sum(bool(r["slipped"]) for r in rows),
                             timeouts=sum(r["timeout"] for r in rows),failures=dict(Counter(
                                 r["failure_mode"] for r in rows if r["failure_mode"])),
                             wrist_p95_mm=float(np.quantile([x for r in rows for x in r["wrist_steps_mm"]],.95)),
                             rescued=[r["index"] for r in rows if r["success"] and not r["base_success"]],
                             lost=[r["index"] for r in rows if not r["success"] and r["base_success"]])
    save_json(args.output/"evaluations.json",report)
    print("DAGGER_FINAL",report,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","monitor","episodes","baseline","splits","output","original"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--phase",required=True,choices=("plan","screen","gate","gate-report","collect","dataset","eval","summary"))
    parser.add_argument("--gate-dir",default="gate")
    parser.add_argument("--screened",type=Path)
    parser.add_argument("--audit-dataset",action="store_true")
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--stop",type=int)
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--label",type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=="dataset":
        build_dataset(args)
    elif args.phase=="gate-report":
        report_gate(args)
    elif args.phase=="summary":
        summarize(args)
    else:
        held,train,val = fixed_cases(args)
        if args.phase=="plan":
            candidate_plan(args,held,train,val)
            return
        env_type = StallInsertion if args.phase=="gate" else EndToEndTelemetry
        env = env_type(penetration_limit_mm=LIMIT_MM)
        try:
            features = FrozenSmolVLA(args.checkpoint)
            if args.phase=="gate":
                dry_run(args,features,env)
            elif args.phase=="eval":
                evaluate(args,env,features,held)
            else:
                screen(args,features,env)
        finally:
            env.close()


if __name__=="__main__":
    main()
