"""SO-101 tabletop insertion using the upstream arm and printed parts."""

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARM = ROOT / "tmp/mujoco_menagerie/robotstudio_so101/so101.xml"
PARTS = ROOT / "tmp/RL-Token-Pi05-open/assets/stl"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "Insert the bolt into the hole"
FPS = 25
IMAGE_SIZE = 224
SOURCE = (0.19, -0.057)
TARGET = (0.19, 0.057)
MOUTH_Z = 0.045
SHAFT_RADIUS = 0.005
BORE_RADIUS = 0.0065
HOME = np.array([0, -0.75, 1.15, 1.15, 0, 0.6])


def scene_model(arm: Path, parts: Path, *, target_bore_radius=BORE_RADIUS, show_target_colliders=False) -> mujoco.MjModel:
    root = ET.parse(arm).getroot()
    root.find("option").set("timestep", "0.004")
    assets = root.find("asset")
    mesh_dir = (arm.parent / root.find("compiler").attrib.pop("meshdir")).resolve()
    for mesh in assets.findall("mesh"):
        mesh.set("file", (mesh_dir / mesh.get("file")).as_posix())
    for name in ("base_1", "base_2", "bolt"):
        ET.SubElement(assets, "mesh", name=name, file=(parts / f"{name}.stl").resolve().as_posix(),
                      scale=".001 .001 .001", maxhullvert="128")

    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="plane", size=".65 .65 .03", rgba=".22 .25 .28 1")
    ET.SubElement(world, "geom", name="arm_stand", type="box", size=".072 .065 .02", pos="0 0 .02",
                  rgba=".3 .32 .35 1")
    world.find("body[@name='base']").set("pos", "0 0 .04")
    ET.SubElement(world, "light", pos="0 -.4 .7", dir=".2 .3 -1", directional="true")
    ET.SubElement(world, "camera", name="top", pos=".39 -.43 .48", fovy="43",
                  xyaxes=".93 .37 0  -.28 .68 .68")
    gripper = world.find(".//body[@name='gripper']")
    ET.SubElement(gripper, "site", name="grasp", pos=".002 0 -.098", size=".002", group="3")

    for name, xy, mesh, color in (
        ("source", SOURCE, "base_1", ".62 .65 .68 1"),
        ("target", TARGET, "base_2", ".08 .62 .32 1"),
    ):
        fixture = ET.SubElement(world, "body", name=name, pos=f"{xy[0]} {xy[1]} .005")
        visible_ring = name == "target" and show_target_colliders
        if not visible_ring:
            ET.SubElement(fixture, "geom", mesh=mesh, type="mesh", rgba=color, contype="0", conaffinity="0")
        collider_style = {"group": "0", "rgba": color} if visible_ring else {"group": "3"}
        # A concave hole needs multiple convex colliders. Mesh geoms alone would fill the bore.
        bore_radius = target_bore_radius if name == "target" else BORE_RADIUS
        for i in range(16):
            angle = i * 2 * np.pi / 16
            center = (bore_radius + .002) * np.array([np.cos(angle), np.sin(angle)])
            ET.SubElement(fixture, "geom", type="box", size=".002 .0021 .016", **collider_style,
                          pos=f"{center[0]} {center[1]} .024", euler=f"0 0 {angle}", friction=".5 .005 .0001")
        ET.SubElement(fixture, "geom", type="box", size=".03 .0225 .004", pos="0 0 .004", **collider_style)

    bolt = ET.SubElement(world, "body", name="bolt", pos=f"{SOURCE[0]} {SOURCE[1]} .013")
    ET.SubElement(bolt, "freejoint", name="bolt_free")
    ET.SubElement(bolt, "inertial", pos="0 0 .025", mass=".012", diaginertia=".000003 .000003 .0000006")
    ET.SubElement(bolt, "geom", type="mesh", mesh="bolt", euler="1.57079632679 0 0", rgba=".14 .15 .17 1",
                  contype="0", conaffinity="0")
    ET.SubElement(bolt, "geom", name="shaft", type="cylinder", size=f"{SHAFT_RADIUS} .02", pos="0 0 .02",
                  friction=".65 .005 .0001", group="3")
    ET.SubElement(bolt, "geom", name="head", type="cylinder", size=".009 .005", pos="0 0 .045",
                  friction="1.3 .01 .0005", group="3")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class Insertion:
    def __init__(self, arm=ARM, parts=PARTS, max_steps=300, *, control_fps=FPS, target_bore_radius=BORE_RADIUS,
                 show_target_colliders=False):
        self.model = scene_model(Path(arm), Path(parts), target_bore_radius=target_bore_radius,
                                 show_target_colliders=show_target_colliders)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(name).qposadr[0] for name in JOINTS])
        self.dofs = np.array([self.model.joint(name).dofadr[0] for name in JOINTS])
        self.low, self.high = self.model.actuator_ctrlrange.T.copy()
        self.grasp_id = self.model.site("grasp").id
        self.bolt_address = self.model.joint("bolt_free").qposadr[0]
        self.substeps = round(1 / (control_fps * self.model.opt.timestep))
        self.max_steps = max_steps
        self.renderer = None
        self.view = mujoco.MjvOption()
        self.view.geomgroup[3:] = 0
        self.view.sitegroup[:] = 0

    def reset(self, seed=0, jitter=0.001):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        for name, center in (("source", SOURCE), ("target", TARGET)):
            self.model.body(name).pos[:2] = center + rng.uniform(-jitter, jitter, size=2)
        self.data.qpos[self.bolt_address:self.bolt_address + 2] = self.model.body("source").pos[:2]
        self.data.qpos[self.joints] = HOME
        self.data.ctrl[:] = HOME
        self.steps = 0
        mujoco.mj_step(self.model, self.data, nstep=25)
        mujoco.mj_forward(self.model, self.data)
        self.bolt_rest_z = float(self.data.body("bolt").xpos[2])
        return self.info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(JOINTS),) or not np.isfinite(action).all():
            raise ValueError("Expected six finite absolute joint targets in radians")
        self.data.ctrl[:] = np.clip(action, self.low, self.high)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        info = self.info()
        return float(info["success"]), info["success"], self.steps >= self.max_steps, info

    def info(self):
        tip = self.data.body("bolt").xpos
        axis = self.data.body("bolt").xmat.reshape(3, 3)[:, 2]
        target = self.model.body("target").pos[:2]
        radial_error = float(np.linalg.norm(tip[:2] - target))
        depth = float(MOUTH_Z - tip[2])
        tilt = float(np.arccos(np.clip(axis[2], -1, 1)))
        return {
            "success": bool(0.021 <= depth <= 0.034 and radial_error < 0.002 and tilt < 0.13),
            "critical": bool(-0.030 <= depth and radial_error < 0.018 and tilt < 0.35),
            "radial_error_m": radial_error,
            "insertion_depth_m": depth,
            "tilt_rad": tilt,
        }

    def state(self):
        return self.data.qpos[self.joints].astype(np.float32)

    def pickup_state(self):
        bolt = self.data.body("bolt")
        lift = float(bolt.xpos[2] - self.bolt_rest_z)
        head = bolt.xpos + .045 * bolt.xmat.reshape(3, 3)[:, 2]
        grasp = self.data.site("grasp").xpos
        source = self.model.body("source").pos[:2]
        target_direction = self.model.body("target").pos[:2] - source
        target_direction /= np.linalg.norm(target_direction)
        offset = grasp[:2] - source
        at_source = bool(np.linalg.norm(offset) < .026)
        toward_target = bool(offset @ target_direction > .035)
        bolt_at_jaws = bool(np.linalg.norm(grasp - head) < .025)
        return lift, at_source, toward_target, bolt_at_jaws

    def observation(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        obs = {"observation.state": self.state(), "task": TASK}
        for name, camera in (("top", "top"), ("wrist", "wrist_cam")):
            self.renderer.update_scene(self.data, camera=camera, scene_option=self.view)
            obs[f"observation.images.{name}"] = self.renderer.render().copy()
        return obs

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def inverse_kinematics(self, position, start, up=(0, 0, 1)):
        """Match grasp position and downwards approach while leaving yaw free."""
        q = np.asarray(start, dtype=np.float64).copy()
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        for _ in range(100):
            self.ik_data.qpos[self.joints] = q
            mujoco.mj_forward(self.model, self.ik_data)
            site = self.ik_data.site("grasp")
            position_error = position - site.xpos
            axis = site.xmat.reshape(3, 3)[:, 2]
            rotation_error = np.cross(axis, up)
            if np.linalg.norm(position_error) < .0003 and np.dot(axis, up) > .9999:
                return q
            mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.grasp_id)
            jac = np.vstack([jp, .12 * (np.eye(3) - np.outer(axis, axis)) @ jr])[:, self.dofs[:5]]
            error = np.r_[position_error, .12 * rotation_error]
            delta = np.linalg.solve(jac.T @ jac + .0004 * np.eye(5), jac.T @ error)
            q[:5] = np.clip(q[:5] + np.clip(delta, -.16, .16), self.low[:5], self.high[:5])
        raise RuntimeError(f"Unreachable grasp waypoint {position}: position error {position_error}")
