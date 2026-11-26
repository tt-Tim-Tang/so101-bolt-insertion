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
from examples.so101_insertion.smolvla_rlt_stage2 import GRIP, STATE, FrozenSmolVLA
from examples.so101_insertion.smolvla_rlt_v3 import COUNTERS
from examples.so101_insertion.smolvla_rlt_v3_pool import fixed_cases, save_json


DAGGER_ID = "local/so101_smolvla25_dagger"
MIXED_ID = "local/so101_smolvla25_dagger_mixed"
LIMIT_MM = 3.
GATE_SIZE = 25
COLLECTION_SIZE = 120
STATS = "observation.state"


class ScreenInsertion(EndToEndTelemetry):
    def start(self, case):
        super().start(case)
        self.depths = deque(maxlen=16)
        self.takeover = None
        self.teaching = False

    def step(self, action):
        if not self.teaching and self.takeover is None and self.pick and self.pickup_state()[-1]:
            _, _, xy, depth, tilt = self.bolt_pose()
            radial = np.linalg.norm(xy)*1000
            reason = None
            if self.first_contact is None:
                if -6. <= depth*1000 < 0 and radial <= 15. and (radial > 1.5 or tilt > 6.):
                    reason = "approach"
            else:
                self.depths.append(depth)
                if len(self.depths)==16 and depth-self.depths[0]<.0006 and self.seated_ticks==0:
                    reason = "stall"
            if reason is not None:
                physics = np.empty(mujoco.mj_stateSize(self.model, STATE))
                mujoco.mj_getState(self.model,self.data,physics,STATE)
                self.takeover = dict(physics=physics,counters={k:getattr(self,k) for k in COUNTERS},
                                     reason=reason,contact=self.first_contact,step=self.steps,
                                     xy_mm=float(radial),tilt_deg=tilt,depth_mm=1000*depth)
        return super().step(action)

    def restore(self, case, saved):
        self.start(case)
        mujoco.mj_setState(self.model,self.data,saved["physics"],STATE)
        mujoco.mj_forward(self.model,self.data)
        mujoco.mj_setState(self.model,self.data,saved["physics"],STATE)
        for key,value in saved["counters"].items():
            setattr(self,key,value)
        self.first_contact = saved["contact"]
        self.teaching = True


def smoothstep(value):
    x = np.clip(value,0.,1.)
    return x*x*(3.-2.*x)


def teach(env,case,saved,*,record,unload_mm=1.2,align_ticks=26,descent_mm=.55):
    if not saved:
        return dict(success=False,slipped=0,reason="no_eligible_takeover",frames=0), []
    env.restore(case,saved)
    start_tip, start_axis, _, _, _ = env.bolt_pose()
    start_tip,start_axis = start_tip.copy(), start_axis.copy()
    initial_command = env.data.ctrl.copy()
    command = initial_command.copy()
    command[-1] = GRIP
    center = env.model.body(env.target_id).pos[:2].copy()
    up = np.array([0.,0.,1.])
    # Align at the approach height or unload slightly from the captured stall.
    top = start_tip[2]+(.001*unload_mm if saved["reason"]=="stall" else 0.)
    bottom = env.floor_z+.0006
    descent_ticks = int(np.ceil(1.5*max(0.,top-bottom)/(.001*descent_mm)))
    xy_bias = np.zeros(2)
    limit = np.array([.010,.012,.012,.014,.019])
    difference = np.zeros(5)
    frames, info = [], {}
    first_jump = arm_jump = peak_jump = rise_mm = 0.
    ik_failures = 0
    for i in range(min(env.max_steps-env.steps,align_ticks+descent_ticks+75)):
        if i:
            alpha = float(smoothstep(i/align_ticks))
            end = float(smoothstep((i-align_ticks)/max(1,descent_ticks)))
            tip, axis, _, _, _ = env.bolt_pose()
            wanted_xy = (1-alpha)*start_tip[:2]+alpha*center
            xy_bias = np.clip(xy_bias+np.clip(.13*(wanted_xy-tip[:2]),-.00008,.00008),-.0014,.0014)
            wanted_z = (1-end)*((1-alpha)*start_tip[2]+alpha*top)+end*bottom
            wanted_axis = (1-alpha)*start_axis+alpha*up
            wanted_axis /= np.linalg.norm(wanted_axis)
            goal = np.r_[wanted_xy+xy_bias,wanted_z]
            qpos = env.data.qpos[env.joints].copy()
            try:
                next_q = env.bolt_ik(goal,wanted_axis,qpos)
            except RuntimeError as error:
                if not str(error).startswith("Held-bolt IK did not converge"):
                    raise
                ik_failures += 1
                # Keep the local step small when a fully vertical IK target is infeasible.
                eased_axis = wanted_axis+axis
                eased_axis /= np.linalg.norm(eased_axis)
                try:
                    next_q = env.bolt_ik(.55*goal+.45*tip,eased_axis,qpos)
                except RuntimeError as fallback_error:
                    if not str(fallback_error).startswith("Held-bolt IK did not converge"):
                        raise
                    next_q = command
            wanted = np.clip(next_q[:5]-command[:5],-limit,limit)
            difference += np.clip(wanted-difference,-limit*.4,limit*.4)
            command[:5] += difference
        command = np.clip(command,env.low,env.high)
        prev = env.data.ctrl.copy()
        # Match smolvla25_data.record_trial: observation before executing a clipped target.
        action = command.astype(np.float32)
        if record:
            observation = env.observation()
            if observation["task"]!=TASK or any(observation[k].shape!=(IMAGE_SIZE,IMAGE_SIZE,3) for k in CAMERAS):
                raise RuntimeError("DAgger recording does not match the original observation schema")
            frames.append(observation|dict(action=action.copy()))
        _, done, timeout, info = env.step(action)
        np.testing.assert_allclose(env.data.ctrl,action,atol=2e-7,rtol=0)
        step_jump = 1000*np.abs(env.data.ctrl-prev)
        peak_jump = max(peak_jump,float(step_jump.max()))
        if i==0:
            first_jump,arm_jump = float(step_jump.max()),float(step_jump[:5].max())
        if i<=align_ticks and saved["reason"]=="stall":
            rise_mm = max(rise_mm,1000*float(env.data.body(env.bolt_id).xpos[2]-start_tip[2]))
        if done or timeout or info["slipped"]:
            break
    success = bool(info.get("success") and not info["slipped"] and not info["opened_early"]
                   and not info["penetration_failure"] and rise_mm<=1.5)
    reason = "success" if success else "unload_overrun" if rise_mm>1.5 else "slip" if env.slip_count else (
        "penetration" if info.get("penetration_failure") else "deep_jam" if env.max_near_depth_mm>=15 else "rim_jam")
    result = dict(success=success,reason=reason,slipped=int(env.slip_count),opened_early=int(env.open_count),
                  frames=env.steps-saved["step"],first_jump_mrad=first_jump,first_arm_jump_mrad=arm_jump,
                  peak_jump_mrad=peak_jump,unload_rise_mm=rise_mm,ik_failures=ik_failures,
                  max_penetration_mm=float(env.max_penetration)*1000,max_depth_mm=env.max_near_depth_mm)
    return result,frames


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
        if not baseline["success"] and env.takeover is not None:
            saved = env.takeover.copy()
            physics = saved.pop("physics")
            row["takeover"] = saved
            if args.phase=="screen":
                np.save(path.with_suffix(".npy"),physics)
            else:
                result, frames = teach(env,row["case"],saved|dict(physics=physics),record=True,
                                       unload_mm=args.unload_mm,align_ticks=args.align_ticks,descent_mm=args.descent_mm)
                row["teacher"] = result
                if result["success"]:
                    save_frames(path.with_suffix(".npz"),frames)
        save_json(path,row)
        print(args.phase,i,"base",row["base"]["reason"] or "success",
              "teacher",row.get("teacher",{}).get("reason"),flush=True)


def dry_run(args,env):
    failed = [p for p in paths(args,"screen") if not json.loads(p.read_text())["base"]["success"]][:GATE_SIZE]
    assert len(failed)==GATE_SIZE
    folder = args.output/args.gate_dir
    folder.mkdir(exist_ok=True)
    for path in failed[args.shard::args.shards]:
        row = json.loads(path.read_text())
        saved = row.get("takeover")
        if saved:
            saved |= dict(physics=np.load(path.with_suffix(".npy")))
        result,_ = teach(env,row["case"],saved,record=False,unload_mm=args.unload_mm,
                         align_ticks=args.align_ticks,descent_mm=args.descent_mm)
        if saved is None:
            result["slipped"] = row["base"]["slipped"]
        save_json(folder/path.name,row|dict(teacher=result))
        print("TEACH_GATE",row["index"],row["base"]["reason"],result,flush=True)


def report_gate(args):
    rows = [json.loads(p.read_text()) for p in paths(args,args.gate_dir)]
    assert len(rows)==GATE_SIZE
    saved = [r["teacher"] for r in rows]
    rescued = sum(r["success"] for r in saved)
    slips = sum(bool(r["slipped"]) for r in saved)
    result = dict(episodes=len(rows),rescued=rescued,slips=slips,passed=rescued>=18 and slips==0,
                  first_max_jump_mrad=max(r.get("first_jump_mrad",0.) for r in saved),
                  first_max_arm_jump_mrad=max(r.get("first_arm_jump_mrad",0.) for r in saved),
                  reasons=dict(Counter(r["reason"] for r in saved)),teacher_parameters=dict(
                      unload_mm=args.unload_mm,align_ticks=args.align_ticks,descent_mm=args.descent_mm),
                  rows=rows)
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
    rows = [(json.loads(p.read_text()),p.with_suffix(".npz")) for p in paths(args,"collect")]
    selected = [(r,p) for r,p in rows if r.get("teacher",{}).get("success")][:COLLECTION_SIZE]
    assert len(selected)==COLLECTION_SIZE
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
    assert merged.total_frames==original.total_frames+frames and merged.total_episodes==150+COLLECTION_SIZE
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
    parser.add_argument("--unload-mm",type=float,default=1.2)
    parser.add_argument("--align-ticks",type=int,default=26)
    parser.add_argument("--descent-mm",type=float,default=.55)
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
        env_class = EndToEndTelemetry if args.phase=="eval" else ScreenInsertion
        env = env_class(penetration_limit_mm=LIMIT_MM)
        try:
            if args.phase=="gate":
                dry_run(args,env)
            else:
                features = FrozenSmolVLA(args.checkpoint)
                if args.phase=="eval":
                    evaluate(args,env,features,held)
                else:
                    screen(args,features,env)
        finally:
            env.close()


if __name__=="__main__":
    main()
