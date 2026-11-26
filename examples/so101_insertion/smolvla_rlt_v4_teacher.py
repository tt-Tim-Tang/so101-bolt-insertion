"""Use the native RLinf takeover route with a bounded privileged MuJoCo teacher."""

import ast
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import mujoco
import numpy as np
import torch

from .scene import MOUTH_Z, SHAFT_RADIUS
from .smolvla25_data import DEPTH_TOLERANCE_M, MAX_FLOOR_OVERRUN_M, MIN_AXIS_Z
from .smolvla_rlt_stage2 import CHUNK, GRIP


def load_route(root):
    """Load the native single-process route without importing RLinf's simulator runtime."""
    name = "_so101_rlinf_simulator_route"
    module = sys.modules[name] = ModuleType(name)
    module.__dict__.update(torch=torch, np=np, ABC=ABC, abstractmethod=abstractmethod,
                           dataclass=dataclass, Any=Any, Literal=Literal)
    folder = root / "rlinf/algorithms/rlt"
    for path in (folder / "expert.py", folder / "route.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.SimulatorRLTRoute(use_schedule=False, warmup_updates=0), module.RLTRouteContext


class StallGate:
    def __init__(self):
        self.depths = []
        self.active = False

    def request(self, env):
        if env.first_contact is None or not env.pickup_state()[-1]:
            return False
        _, _, xy, depth, tilt = env.bolt_pose()
        self.depths.append(depth * 1000)
        if self.active:
            return True
        if env.seated_ticks:
            return False
        if len(self.depths) >= 3:
            progress = self.depths[-1] - self.depths[-3]
            misaligned = np.linalg.norm(xy) > .0012 or tilt > 5.
            self.active = progress < .6 or (misaligned and progress < 1.3)
        return self.active


class ScriptTeacher:
    """Choose short, bounded corrections using the held-bolt Jacobian and shadow physics."""

    def __init__(self, env, eps_rad, *, scale=1.):
        self.env = env
        self.bounds = np.asarray(eps_rad[:5]) * scale
        self.shadow = mujoco.MjData(env.model)
        self.previous = None
        self.wedged = 0
        self.progress = None
        self.unload_until = None
        self.metrics = []

    def predict_action_batch(self, *, env_obs, mode, compute_values):
        return self.chunk(env_obs["reference"])[None], {}

    def limit(self, reference, residual):
        return np.clip(residual, -1, 1)

    def score(self, reference, residual, *, unload):
        env, shadow = self.env, self.shadow
        mujoco.mj_copyData(shadow, env.model, env.data)
        target = env.model.body(env.target_id).pos[:2]
        streak, peak = env.seated_ticks, env.seated_ticks
        worst_penetration = 0.
        for ref, unit in zip(reference, residual, strict=True):
            shadow.ctrl[:5] = np.clip(ref[:5] + self.bounds * unit, env.low[:5], env.high[:5])
            shadow.ctrl[-1] = GRIP
            for _ in range(env.substeps):
                mujoco.mj_step(env.model, shadow)
                bolt = shadow.body(env.bolt_id)
                tip, axis = bolt.xpos, bolt.xmat.reshape(3, 3)[:, 2]
                gap = tip[2] - SHAFT_RADIUS*np.linalg.norm(axis[:2]) - env.floor_z
                xy = tip[:2]-target
                at_mouth = xy + (MOUTH_Z-tip[2])*axis[:2]/max(axis[2], .01)
                seated = (tip[2] < MOUTH_Z and axis[2] >= MIN_AXIS_Z
                          and -MAX_FLOOR_OVERRUN_M <= gap <= DEPTH_TOLERANCE_M
                          and np.all(np.maximum(env.normals@xy, env.normals@at_mouth) <= env.faces))
                streak = streak+1 if seated else 0
                peak = max(peak, streak)
                if len(shadow.contact):
                    bodies = env.model.geom_bodyid[shadow.contact.geom]
                    pair = ((bodies == env.bolt_id).any(axis=1) & (bodies == env.target_id).any(axis=1))
                    if pair.any():
                        worst_penetration = max(worst_penetration, -float(shadow.contact.dist[pair].min()))
        grasp_error = np.linalg.norm(shadow.site(env.grasp_id).xpos - (tip + .045*axis))
        risk = 10000*max(0., grasp_error-.019) + 30000*max(0., worst_penetration-env.penetration_limit_m*.55)
        lateral = 1000*(1.6*np.linalg.norm(xy) + .8*np.linalg.norm(at_mouth))
        depth_term = 1000*abs(gap-.00065) + 5000*max(0., -gap-.0003)
        if unload is not None:
            depth_term = 2800*abs(tip[2]-unload) + 7000*max(0., unload-tip[2])
        return (risk + lateral + depth_term + 7*np.linalg.norm(axis[:2])
                - .65*(streak+peak) + .04*np.square(residual).mean())

    def chunk(self, reference):
        env = self.env
        if env.first_contact is None:
            raise RuntimeError("Expert route attempted pre-contact intervention")
        tip, axis, xy, depth, _ = env.bolt_pose()
        q = env.data.qpos[env.joints[:5]]
        if self.progress is not None:
            self.wedged = self.wedged + 1 if abs(depth-self.progress) < .00035 else 0
        self.progress = depth
        if env.seated_ticks:
            self.unload_until = None
        elif self.unload_until is None and self.wedged >= 3:
            self.unload_until = (env.steps + 2*CHUNK, min(tip[2]+.0011, MOUTH_Z+.001))
            self.wedged = 0
        if self.unload_until is not None and env.steps >= self.unload_until[0]:
            self.unload_until = None
        unload = None if self.unload_until is None else self.unload_until[1]

        jp = np.zeros((3, env.model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jac(env.model, env.data, jp, jr, tip, int(env.model.site_bodyid[env.grasp_id]))
        jac = np.vstack([jp, .045*(np.eye(3)-np.outer(axis, axis))@jr])[:, env.dofs[:5]]
        inverse = np.linalg.solve(jac.T@jac + .00004*np.eye(5), jac.T)
        candidates = [np.zeros((CHUNK, 5))]
        if self.previous is not None:
            candidates.append(self.previous)
        # Try steady hold, incremental descent, and a short unload with lateral correction.
        requested = []
        for shift in (0., -.0008, -.0016, .0009):
            desired_z = unload if unload is not None else max(env.floor_z+.0005, tip[2]+shift)
            for align in (.55, 1.1):
                target = np.r_[np.clip(-align*xy, -.002, .002), desired_z-tip[2],
                               .045*np.cross(axis, (0., 0., 1.))]
                desired = q + inverse@target
                requested.append((desired-reference[:, :5])/self.bounds)
                candidates.append(self.limit(reference, requested[-1]))
        scored = [(self.score(reference, u, unload=unload), u) for u in candidates]
        best_score, best = min(scored, key=lambda entry: entry[0])
        for axis_index in range(5):
            axis_best = best
            for delta in (-.6, .6):
                candidate = axis_best.copy()
                candidate[:, axis_index] += delta
                candidate = self.limit(reference, candidate)
                score = self.score(reference, candidate, unload=unload)
                if score < best_score:
                    best_score, best = score, candidate
        self.previous = best
        self.metrics.append(dict(step=env.steps, unload=unload is not None, depth_mm=1000*depth,
                                 xy_mm=1000*float(np.linalg.norm(xy)), limited=(np.abs(best) >= .999).mean(),
                                 requested_scale=min(float(np.abs(c).max()) for c in requested),
                                 selected_scale=float(np.abs(best).max())))
        return np.pad(best, ((0, 0), (0, 1))).astype(np.float32)


def routed_chunk(route, context, ref_obs, physical_reference, teacher, requested, *, mode="train", student=None):
    unit = torch.zeros_like(ref_obs) if student is None else student
    output = route.route(context(env_obs={"reference": physical_reference}, rlt_obs={"ref_chunk": ref_obs[None]},
                                 student_actions=unit[None], result={"forward_inputs": {"ref_chunk": ref_obs[None]}},
                                 mode=mode, expert_model=teacher, rlt_switch_flags=torch.tensor(True),
                                 intervene_requested=torch.tensor(requested), actions_in_residual_space=True))
    if not torch.equal(output.result["forward_inputs"]["ref_chunk"][0], ref_obs):
        raise RuntimeError("The takeover route must preserve the frozen reference chunk")
    return output.actions[0], output.result["intervene_flags"][0]
