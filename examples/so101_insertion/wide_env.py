"""50 Hz SO-101 task with independent fixtures and stable, geometry-checked seating."""
import mujoco
import numpy as np

from .scene import HOME, MOUTH_Z, SHAFT_RADIUS, SOURCE, TARGET, Insertion


class WideInsertion(Insertion):
    def __init__(self, *, target_bore_radius=.006):
        super().__init__(max_steps=1000, control_fps=50, target_bore_radius=target_bore_radius, show_target_colliders=True)
        ids = np.flatnonzero(self.model.geom_bodyid == self.model.body("target").id)
        self.walls, self.floor = ids[:-1], ids[-1]
        self.model.opt.timestep = .002
        self.substeps = 10
        self.model.geom_solref[ids, 0] = .006
        self.model.geom_solmix[ids] = 50
        self.floor_z = (self.model.body("target").pos[2] + self.model.geom_pos[self.floor, 2]
                        + self.model.geom_size[self.floor, 2])
        centers = self.model.geom_pos[self.walls, :2]
        self.normals = centers / np.linalg.norm(centers, axis=1)[:, None]
        self.faces = np.linalg.norm(centers, axis=1) - self.model.geom_size[self.walls, 0]
        assert len(ids) == 17 and np.allclose(self.faces, target_bore_radius) and np.isclose(self.floor_z, .013)

    def start(self, case):
        mujoco.mj_resetData(self.model, self.data)
        for name, origin in (("source", SOURCE), ("target", TARGET)):
            self.model.body(name).pos[:2] = np.array(origin) + np.array(case[name]) / 1000
        joints = HOME.copy()
        joints[:5] += case["joints"]
        self.data.qpos[self.joints] = self.data.ctrl[:] = joints
        self.data.qpos[self.bolt_address:self.bolt_address+2] = self.model.body("source").pos[:2]
        mujoco.mj_step(self.model, self.data, nstep=25)
        mujoco.mj_forward(self.model, self.data)
        self.bolt_rest_z = float(self.data.body("bolt").xpos[2])
        self.steps = self.held_ticks = self.seated_ticks = 0
        self.pick = self.arrive = self.align = False
        self.max_streak = 0
        self.max_jump = 0.
        self.max_penetration = 0.
        self.phase = "reset"
        self.initial = dict(joints=self.state().tolist(), bolt_qpos=self.data.qpos[
            self.bolt_address:self.bolt_address+7].tolist(), source=self.model.body("source").pos.tolist(),
            target=self.model.body("target").pos.tolist())

    def geometry(self):
        bolt = self.data.body("bolt")
        tip = bolt.xpos
        axis = bolt.xmat.reshape(3, 3)[:, 2]
        xy = tip[:2] - self.model.body("target").pos[:2]
        tilt = np.arccos(np.clip(axis[2], -1, 1))
        bottom = tip[2] - SHAFT_RADIUS * np.linalg.norm(axis[:2]) - self.floor_z
        contained = False
        if tilt < .10 and -.0002 <= bottom <= .002:
            slope = axis[:2] / axis[2]
            upper = xy + (MOUTH_Z - tip[2]) * slope
            support = SHAFT_RADIUS * np.sqrt(1 + (self.normals @ slope)**2)
            contained = np.all(np.maximum(self.normals @ xy, self.normals @ upper) + support <= self.faces)
        return float(tilt), float(np.linalg.norm(xy)), float(bottom), bool(contained)

    def step(self, action):
        self.max_jump = max(self.max_jump, float(np.max(np.abs(action - self.data.ctrl))))
        _, _, timeout, info = super().step(action)
        tilt, radial, bottom, inside = self.geometry()
        lift, _, _, jaws = self.pickup_state()
        self.held_ticks = self.held_ticks + 1 if lift >= .025 and jaws else 0
        self.pick |= self.held_ticks >= 20
        approach = (self.pick and jaws and 0 <= -info["insertion_depth_m"] <= .010 and tilt <= .15)
        self.arrive |= approach and radial <= .005
        self.align |= approach and radial <= .002
        self.seated_ticks = self.seated_ticks + 1 if inside else 0
        self.max_streak = max(self.max_streak, self.seated_ticks)
        contacts = self.data.contact
        if len(contacts):
            bodies = self.model.geom_bodyid[contacts.geom]
            mask = ((bodies == self.model.body("bolt").id).any(axis=1)
                    & (bodies == self.model.body("target").id).any(axis=1))
            if mask.any():
                self.max_penetration = max(self.max_penetration, float(-contacts.dist[mask].min()))
        info.update(success=self.seated_ticks >= 25, pickup=self.pick, arrival=self.arrive, alignment=self.align,
                    bottom_gap_mm=bottom*1000, stable_ticks=self.max_streak, phase=self.phase,
                    max_joint_step_rad=self.max_jump, max_penetration_mm=self.max_penetration*1000)
        return float(info["success"]), info["success"], timeout, info


def expert(env, case, record=None, at_pick=None):
    env.start(case)
    cmd = env.state().astype(float)
    source, target = [env.model.body(n).pos[:2].copy() for n in ("source", "target")]
    info = {}

    def move(phase, point, grip, ticks, up=(0,0,1)):
        nonlocal cmd, info
        env.phase = phase
        try:
            end = env.inverse_kinematics(point, cmd, up)
        except RuntimeError:
            if np.array_equal(up, (0,0,1)):
                raise
            end = env.inverse_kinematics(point, cmd)
        end[-1] = grip
        for t in range(1, ticks+1):
            f = t / ticks
            action = (cmd + (end-cmd) * (f*f*(3-2*f))).astype(np.float32)
            if record:
                record(action)
            _, done, timeout, info = env.step(action)
            if done or timeout:
                return True
        cmd = end.copy()
        return False

    for phase, z, grip, ticks in (("over_source", .105, .52, 44), ("lower", .057, .52, 50),
                                  ("close", .057, -.055, 36), ("lift", .107, -.055, 64)):
        if move(phase, [*source, z], grip, ticks):
            return info
    if not env.pick:
        return dict(info, reason="pickup")
    if at_pick:
        at_pick(env)
    # The bolt body origin is the shaft tip; compensate its measured TCP offset.
    def tip_offset():
        return env.data.body("bolt").xpos - env.data.site("grasp").xpos

    if move("transfer", np.r_[target, MOUTH_Z+.030]-tip_offset(), -.055, 64):
        return info
    xy_correction = np.zeros(2)

    def track_tip(z):
        nonlocal cmd, info
        tip = env.data.body("bolt").xpos
        xy_correction[:] = np.clip(xy_correction + np.clip(.2*(target-tip[:2]), -.0001, .0001),
                                   -.002, .002)
        point = np.r_[target+xy_correction, z]-tip_offset()
        cmd = env.inverse_kinematics(point, cmd)
        cmd[-1] = -.055
        action = cmd.astype(np.float32)
        if record:
            record(action)
        _, done, timeout, info = env.step(action)
        return done or timeout

    env.phase = "air_align"
    start_z = float(env.data.body("bolt").xpos[2])
    for tick in range(1, 51):
        f = tick/50
        z = start_z + (MOUTH_Z+.005-start_z)*(f*f*(3-2*f))
        if track_tip(z):
            return info
    # 65 constant-speed descent ticks plus at most 35 settling ticks: 100 total.
    for tick in range(1, 101):
        env.phase = "insert" if tick <= 65 else "hold"
        z = MOUTH_Z+.005 + (env.floor_z+.0005-MOUTH_Z-.005)*min(tick/65, 1)
        if track_tip(z):
            return info
    return dict(info, reason="insertion_100_tick_limit")
