"""Score recorded insertion actions with the soft wall contacts resolved by MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .compare_experts import initial_case
from .scene import MOUTH_Z, SHAFT_RADIUS
from .wide_env import WideInsertion

CONTACT_TOLERANCE_M = 0.0002  # Match the seating check's existing floor penetration allowance.
TASK_CENTERLINE_LIMIT_M = 0.002  # Match the base task's radial acceptance tolerance.
SEAT_TICKS = 25


def seating_snapshot(env):
    tilt, _, bottom, hard_containment = env.geometry()
    depth_and_tilt = -CONTACT_TOLERANCE_M <= bottom <= 0.002 and tilt < 0.10
    bolt = env.data.body("bolt")
    axis = bolt.xmat.reshape(3, 3)[:, 2]
    contact_penetration = 0.0
    geometric_overclosure = 0.0
    soft_containment = False
    task_seated = False
    if depth_and_tilt:
        tip = bolt.xpos[:2] - env.model.body("target").pos[:2]
        slope = axis[:2] / axis[2]
        mouth_center = tip + (MOUTH_Z - bolt.xpos[2]) * slope
        center_extents = np.maximum(env.normals @ tip, env.normals @ mouth_center)
        task_seated = (np.all(center_extents <= env.faces)
                       and max(np.linalg.norm(tip), np.linalg.norm(mouth_center)) < TASK_CENTERLINE_LIMIT_M)
        edge_extents = center_extents + SHAFT_RADIUS * np.sqrt(1 + (env.normals @ slope)**2)
        outside = edge_extents > env.faces
        geometric_overclosure = max(0.0, float(np.max(edge_extents - env.faces)))
        contacts = env.data.contact
        shaft_id = env.model.geom("shaft").id
        shaft_contacts = contacts.geom == shaft_id
        touching = shaft_contacts.any(axis=1)
        partner = np.where(shaft_contacts[touching, 0], contacts.geom[touching, 1], contacts.geom[touching, 0])
        distance = contacts.dist[touching]
        wall_contact = np.isin(partner, env.walls)
        contact_penetration = max(0.0, float(-distance[wall_contact].min())) if wall_contact.any() else 0.0
        closed_wall_contacts = partner[wall_contact & (distance <= 0.0)]
        # Permit measured soft contact; any overrun without a corresponding wall
        # contact remains a geometric failure. Still reject a shaft axis outside the bore.
        soft_containment = (np.all(center_extents <= env.faces)
                            and np.all(np.isin(env.walls[outside], closed_wall_contacts))
                            and contact_penetration <= CONTACT_TOLERANCE_M)
    return bool(hard_containment), bool(soft_containment), bool(task_seated), contact_penetration, geometric_overclosure


def evaluate(env, case, actions):
    env.start(case)
    current = np.zeros(3, dtype=int)
    longest = np.zeros(3, dtype=int)
    revised_window_penetration = None
    last_window = []
    peak_contact = 0.0
    for action in actions:
        _, _, timeout, _ = env.step(action)
        strict, soft, task, penetration, overclosure = seating_snapshot(env)
        current = (current + 1) * np.array([strict, soft, task])
        longest = np.maximum(longest, current)
        last_window.append(penetration)
        if revised_window_penetration is None and current[1] == SEAT_TICKS:
            revised_window_penetration = max(last_window[-SEAT_TICKS:]) * 1000
        peak_contact = max(peak_contact, penetration)
        if timeout:
            break
    return dict(
        strict_streak=int(longest[0]), soft_contact_streak=int(longest[1]), task_seated_streak=int(longest[2]),
        strict_success=bool(longest[0] >= SEAT_TICKS), soft_contact_success=bool(longest[1] >= SEAT_TICKS),
        task_seated_success=bool(longest[2] >= SEAT_TICKS), task_seated_final_streak=int(current[2]),
        peak_wall_penetration_mm=peak_contact * 1000, last_wall_penetration_mm=penetration * 1000,
        last_geometric_overclosure_mm=overclosure * 1000,
        successful_window_max_wall_penetration_mm=revised_window_penetration,
    )


def verify_poses(env):
    """Check physically the same scene for passing and failing near-floor poses."""
    test_cases = (
        ("centered", 0.0, 0.0005, 0.0, True, True),
        ("touching", 0.0008, 0.0005, 0.0, True, True),
        ("heavy_wall_overlap", 0.0015, 0.0005, 0.0, False, True),
        ("off_center", 0.0025, 0.0005, 0.0, False, False),
        ("outside_bore", 0.01, 0.0005, 0.0, False, False),
        ("too_shallow", 0.0, 0.003, 0.0, False, False),
        ("too_tilted", 0.0, 0.001, 0.12, False, False),
        ("beyond_floor", 0.0, -0.001, 0.0, False, False),
    )
    result = {}
    for label, x, gap, tilt, expected_soft, expected_task in test_cases:
        env.start(initial_case((0, 0)))
        target = env.model.body("target").pos[:2]
        env.data.qpos[env.bolt_address:env.bolt_address+7] = (
            target[0] + x, target[1], env.floor_z + gap, np.cos(tilt / 2), 0, np.sin(tilt / 2), 0,
        )
        mujoco.mj_forward(env.model, env.data)
        strict, soft, task, penetration, overclosure = seating_snapshot(env)
        assert (soft, task) == (expected_soft, expected_task), (label, strict, soft, task, penetration, overclosure)
        result[label] = dict(strict=strict, soft_contact=soft, task_seated=task,
                             measured_wall_penetration_mm=penetration * 1000)
    return result


def audit(root):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    env = WideInsertion(target_bore_radius=SHAFT_RADIUS + metadata["clearance_mm"] / 1000)
    checks = verify_poses(env)
    results = []
    try:
        nominal = [json.loads(line) for line in (root / "nominal.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in nominal:
            name, label = row["case"], row["expert"]
            case = initial_case(row["hole_offset_mm"])
            with np.load(root / "trajectories" / f"{name}_{label}.npz") as saved:
                full = saved["control_action_50hz_rad"]
                held = np.repeat(saved["action_rad"], 2, axis=0)[:len(full)]
            for hz, nominal_actions in ((50, full), (25, held)):
                runs = [(0.0, metadata["seeds"][0])]
                if hz == 25:
                    runs += [(std, seed) for std in metadata["noise_std_rad"] for seed in metadata["seeds"]]
                for std, seed in runs:
                    actions = nominal_actions
                    if std:
                        block_ticks = metadata["noise_block_frames"] * metadata["control_hz"] // hz
                        blocks = (len(actions) + block_ticks - 1) // block_ticks
                        rng = np.random.default_rng(seed)
                        bias = rng.normal(0, std, size=(blocks, actions.shape[1]))
                        actions = actions + np.repeat(bias, block_ticks, axis=0)[:len(actions)]
                    result = evaluate(env, case, actions)
                    if hz == 50:
                        assert row["insertion"] == result["strict_success"], (row, result)
                        print(name, label, result, flush=True)
                    results.append(dict(case=name, expert=label, hz=hz, std_rad=std, seed=seed, **result))
    finally:
        env.close()
    prior = [json.loads(line) for line in (root / "replay.jsonl").read_text(encoding="utf-8").splitlines()]
    by_key = {(r["case"], r["expert"], r["command_hz"], r["std_rad"], r["seed"]): r["insertion"] for r in prior}
    assert len(results) == len(by_key)
    for r in results:
        assert r["strict_success"] == by_key[r["case"], r["expert"], r["hz"], r["std_rad"], r["seed"]], r
    (root / "seating-audit-task.json").write_text(json.dumps(
        dict(contact_tolerance_mm=CONTACT_TOLERANCE_M * 1000,
             task_centerline_limit_mm=TASK_CENTERLINE_LIMIT_M * 1000,
             pose_checks=checks, results=results), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root)
