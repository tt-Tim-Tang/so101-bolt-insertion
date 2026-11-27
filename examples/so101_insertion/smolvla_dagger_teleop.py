"""Export stalled SmolVLA states and let a human move the held bolt at 25 Hz."""

import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from examples.so101_insertion.scene import MOUTH_Z
from examples.so101_insertion.smolvla25_data import CAMERAS, FPS, SmolVLAInsertion


STATE = mujoco.mjtState.mjSTATE_INTEGRATION
GRIP = -.055
LIMIT_MM = 3.
COUNTERS = ("steps","start_time","pick","arrive","held_ticks","highest_lift","seated_ticks",
            "max_streak","max_penetration","open_count","was_open","drop_ticks","slip_count")
MOVE = {"w":(1,0,0), "s":(-1,0,0), "a":(0,1,0), "d":(0,-1,0), "q":(0,0,1), "e":(0,0,-1)}


def write_json(path, value):
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def export(args):
    from collections import deque

    from examples.so101_insertion.smolvla_padding_e2e30 import EndToEndTelemetry, replay
    from examples.so101_insertion.smolvla_rlt_stage2 import FrozenSmolVLA

    class Stalled(Exception):
        pass

    class Capture(EndToEndTelemetry):
        def start(self, case):
            super().start(case)
            self.depths = deque(maxlen=16)

        def step(self, action):
            if self.first_contact is not None and self.pick and self.pickup_state()[-1]:
                depth = self.bolt_pose()[3]
                self.depths.append(depth)
                if len(self.depths)==16 and depth-self.depths[0]<.0006 and self.seated_ticks==0:
                    raise Stalled
            else:
                self.depths.clear()
            return super().step(action)

    recent = json.loads(args.cases.read_text())["rows"]
    cases = [row for row in recent if row["takeover"] is not None]
    args.output.mkdir(parents=True,exist_ok=True)
    features = FrozenSmolVLA(args.checkpoint)
    env = Capture(penetration_limit_mm=LIMIT_MM)
    try:
        for row in cases[args.shard::args.shards]:
            path = args.output/f'{row["index"]:04d}.json'
            if path.exists():
                continue
            report = {k: row[k] for k in ("index","case","seed")}
            report["previous_reason"] = row["teacher"]["reason"]
            try:
                end,_ = replay(env,features,row["case"],row["seed"],corrected=True)
            except Stalled:
                state = np.empty(mujoco.mj_stateSize(env.model,STATE))
                mujoco.mj_getState(env.model,env.data,state,STATE)
                tip,axis,xy,depth,tilt = env.bolt_pose()
                report.update(captured=True,physics=path.with_suffix(".npy").name,
                              counters={k:getattr(env,k) for k in COUNTERS},
                              contact_step=env.first_contact["step"],tip=tip.tolist(),axis=axis.tolist(),
                              depth_mm=1000*depth,xy_mm=1000*float(np.linalg.norm(xy)),tilt_deg=tilt,
                              advance_mm=1000*(depth-env.depths[0]),initial_ctrl=env.data.ctrl.tolist())
                np.save(path.with_suffix(".npy"),state)
            else:
                report.update(captured=False,inserted=bool(end["success"]),slipped=int(end["slipped"]))
            write_json(path,report)
            print("STALL_CAPTURE",row["index"],report.get("depth_mm"),report["captured"],flush=True)
    finally:
        env.close()


def summarize(folder):
    rows = [json.loads(f.read_text()) for f in sorted(folder.glob("[0-9]*.json"))]
    captured = [row for row in rows if row["captured"]]
    captured.sort(key=lambda row:(row["previous_reason"]!="seating_near_miss",-row["depth_mm"],row["index"]))
    first = captured[:10]
    write_json(folder/"manifest.json",dict(fps=FPS,penetration_limit_mm=LIMIT_MM,
               attempted=len(rows),captured=len(captured),cases=first,reserve=captured[10:]))
    print("MANUAL_STARTS",len(first),"available",len(captured),"attempted",len(rows),flush=True)


def restore(env, folder, row):
    env.start(row["case"])
    state = np.load(folder/row["physics"],allow_pickle=False)
    if state.shape != (mujoco.mj_stateSize(env.model,STATE),):
        raise ValueError("Snapshot size does not match this MuJoCo model")
    mujoco.mj_setState(env.model,env.data,state,STATE)
    mujoco.mj_forward(env.model,env.data)
    mujoco.mj_setState(env.model,env.data,state,STATE)
    for key,value in row["counters"].items():
        setattr(env,key,value)
    env.max_steps = env.steps+20*FPS
    bolt = env.data.body(env.bolt_id)
    np.testing.assert_allclose(bolt.xpos,row["tip"],rtol=0,atol=1e-8)
    np.testing.assert_allclose(bolt.xmat.reshape(3,3)[:,2],row["axis"],rtol=0,atol=1e-8)
    np.testing.assert_allclose(env.data.ctrl,row["initial_ctrl"],rtol=0,atol=1e-8)


def jog(env, key):
    cmd = env.data.ctrl.copy()
    cmd[-1] = GRIP
    if key not in MOVE and key not in ("r","f"):
        return np.clip(cmd,env.low,env.high).astype(np.float32)
    jp = np.zeros((3,env.model.nv))
    jr = np.zeros_like(jp)
    mujoco.mj_jacSite(env.model,env.data,jp,jr,env.grasp_id)
    dofs = env.dofs[:4]
    delta_xyz = .00025*np.asarray(MOVE.get(key,(0,0,0)))
    tangential = env.data.site(env.grasp_id).xpos[:2].copy()
    axis = np.array([-tangential[1],tangential[0],0.])
    axis /= max(float(np.linalg.norm(axis)),1e-8)
    delta_rot = (.004 if key=="r" else -.004 if key=="f" else 0.)*axis
    jac = np.vstack((jp[:,dofs],.05*jr[:,dofs]))
    q_delta = np.linalg.solve(jac.T@jac+.00015*np.eye(4),jac.T@np.r_[delta_xyz,.05*delta_rot])
    cmd[:4] += np.clip(q_delta,-.008,.008)
    return np.clip(cmd,env.low,env.high).astype(np.float32)


def save_success(folder,row,frames):
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/f'{row["index"]:04d}-{time.time_ns()}.npz'
    fields = (*CAMERAS,"observation.state","action")
    data = {key:np.stack([frame[key] for frame in frames]) for key in fields}
    np.savez_compressed(path,**data,task=frames[0]["task"],fps=FPS,case=json.dumps(row["case"]))
    print("SAVED",path,len(frames),flush=True)
    return path.name


def display(env, obs, title, index, count, frames, status):
    rgb = np.concatenate([obs[k] for k in CAMERAS],axis=1)
    image = cv2.cvtColor(cv2.resize(rgb,(896,448)),cv2.COLOR_RGB2BGR)
    image = cv2.copyMakeBorder(image,0,105,0,0,cv2.BORDER_CONSTANT,value=(22,22,22))
    info = env.info()
    caption = f"Case {index}   {count}   depth {1000*info['insertion_depth_m']:.2f} mm" \
              f"   XY {1000*info['radial_error_m']:.2f} mm   tilt {np.degrees(info['tilt_rad']):.1f} deg"
    lines = ["TOP                                         WRIST",caption,
             "WASD move XY  |  Q up / E down  |  R/F pitch  |  Space hold 0.20 s",
             f"N next  |  B previous  |  Backspace reset  |  Esc exit     Frames: {frames}   {status}"]
    for i,line in enumerate(lines):
        cv2.putText(image,line,(14,474+22*i),cv2.FONT_HERSHEY_SIMPLEX,.56,(235,235,235),1,cv2.LINE_AA)
    cv2.imshow(title,image)


def play(args):
    manifest = json.loads((args.snapshots/"manifest.json").read_text())
    cases = manifest["cases"]
    if not cases:
        raise ValueError("No real stalled states were captured")
    env = SmolVLAInsertion(penetration_limit_mm=LIMIT_MM)
    title = "SO-101 insertion | Human takeover"
    try:
        if args.check:
            for row in cases:
                restore(env,args.snapshots,row)
                action = jog(env,"e")
                assert np.isfinite(action).all() and np.allclose(action[4],row["initial_ctrl"][4])
            print("RESTORE_OK",len(cases),flush=True)
            return
        cv2.namedWindow(title,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title,896,553)
        current = 0
        row = cases[current]
        restore(env,args.snapshots,row)
        frames, status = [], "Paused"
        finished = False
        while True:
            display(env,env.observation(),title,row["index"],f"{current+1}/{len(cases)}",len(frames),status)
            code = cv2.waitKey(50)
            while code < 0:
                if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE) < 1:
                    return
                code = cv2.waitKey(50)
            code &= 255
            if code == 27:
                break
            key = chr(code).lower()
            if key in ("n","b") or code in (8,127):
                current = (current+(key=="n")-(key=="b"))%len(cases)
                row = cases[current]
                restore(env,args.snapshots,row)
                frames,status,finished = [],"Paused",False
                continue
            if finished or (key not in MOVE and key not in ("r","f"," ")):
                continue
            for _ in range(5 if key==" " else 1):
                action = jog(env,key)
                frames.append(env.observation()|dict(action=action.copy()))
                _, done, timeout, info = env.step(action)
                np.testing.assert_allclose(env.data.ctrl,action,rtol=0,atol=2e-7)
                if done or timeout or info["slipped"]:
                    finished = True
                    if info["success"] and not info["slipped"] and not info["opened_early"]:
                        filename = save_success(args.output,row,frames)
                        status = "Saved: "+filename
                    else:
                        status = "Slip" if info["slipped"] else "Penetration" if info["penetration_failure"] else "Time limit"
                    break
                status = "Paused"
    finally:
        env.close()
        if not args.check:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase",required=True)
    remote = sub.add_parser("export")
    for k in ("checkpoint","cases","output"):
        remote.add_argument(f"--{k}",type=Path,required=True)
    remote.add_argument("--shard",default=0,type=int)
    remote.add_argument("--shards",default=1,type=int)
    summary = sub.add_parser("summarize")
    summary.add_argument("--snapshots",type=Path,required=True)
    manual = sub.add_parser("play")
    manual.add_argument("--snapshots",type=Path,required=True)
    manual.add_argument("--output",type=Path,required=True)
    manual.add_argument("--check",action="store_true")
    args = parser.parse_args()
    if args.phase=="export":
        export(args)
    elif args.phase=="summarize":
        summarize(args.snapshots)
    else:
        play(args)


if __name__=="__main__":
    main()
