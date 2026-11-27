"""Extend 25 Hz SO-101 insertion holds, audit 20k flow loss, fine-tune, and evaluate."""

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter
from math import ceil
from pathlib import Path

import mujoco
import numpy as np

from .scene import JOINTS, MOUTH_Z
from .smolvla25_data import (
    CAMERAS,
    FPS,
    MERGED_REPO_ID,
    TARGET_RANGE_MM,
    SmolVLAInsertion,
    create_dataset,
    make_case,
)


HOLD_REPO_ID = "local/so101_smolvla25_hold50"
MIXED_REPO_ID = "local/so101_smolvla25_hold_mixed"
STATS = "observation.state"
PENETRATION_LIMIT_MM = 3.0
START = 20_000
WARMUP = 100
EFFECTIVE_BATCH = 64
SAVE_AT = (1000, 2000, 3000)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_straight_solref011(env: SmolVLAInsertion) -> None:
    """Verify the target bore matches the 90-degree straight solref=0.011 physics."""
    assert len(env.walls) == 16 and np.allclose(env.faces, 0.0057), "Expected 16 straight walls with 0.7 mm clearance"
    assert all("chamfer" not in env.model.geom(i).name for i in range(env.model.ngeom)), "Unexpected chamfer collider"
    assert np.allclose(
        env.model.body("target").pos[2] + env.model.geom_pos[env.walls, 2] + env.model.geom_size[env.walls, 2],
        MOUTH_Z,
    )
    assert np.isclose(env.floor_z, 0.013) and env.substeps == 20
    assert np.allclose(env.model.geom_solref[env.walls, 0], 0.011)
    assert np.allclose(env.model.geom_friction[env.walls, 0], 0.28)


class HoldCollectionEnv(SmolVLAInsertion):
    """Do not early-stop on success so the expert records the full bottom hold tail."""

    def step(self, action):
        reward, _, timeout, info = super().step(action)
        return reward, bool(info["penetration_failure"]), timeout, info


def locked_bolt_ik(env: SmolVLAInsertion, position: np.ndarray, axis: np.ndarray, wrist_roll: float, grip: float = -0.055) -> np.ndarray:
    """Solve 4-DOF tip+axis IK with wrist_roll held strictly constant."""
    tcp = env.data.site("grasp")
    bolt = env.data.body(env.bolt_id)
    rot = tcp.xmat.reshape(3, 3)
    local_tip = rot.T @ (bolt.xpos - tcp.xpos)
    local_axis = rot.T @ bolt.xmat.reshape(3, 3)[:, 2]
    posture = env.state().astype(float)
    posture[4:] = wrist_roll, grip
    q, best = posture.copy(), posture.copy()
    best_cost = np.inf
    jp = np.zeros((3, env.model.nv))
    jr = np.zeros_like(jp)
    body = int(env.model.site_bodyid[env.grasp_id])
    dofs = env.dofs[:4]
    axis_weight, posture_weight = 0.024, 0.004
    reg = posture_weight * np.eye(4)
    damping = 0.00004 * np.eye(4)
    for _ in range(32):
        env.ik_data.qpos[env.joints] = q
        mujoco.mj_forward(env.model, env.ik_data)
        site = env.ik_data.site("grasp")
        rot = site.xmat.reshape(3, 3)
        tip, up = site.xpos + rot @ local_tip, rot @ local_axis
        error = position - tip
        anchor = posture[:4] - q[:4]
        cost = np.sum(error**2) + axis_weight**2 * np.sum((axis - up) ** 2) + posture_weight**2 * np.sum(anchor**2)
        if cost < best_cost:
            best_cost, best = cost, q.copy()
        mujoco.mj_jac(env.model, env.ik_data, jp, jr, tip, body)
        jac = np.vstack((jp[:, dofs], axis_weight * (np.eye(3) - np.outer(up, up)) @ jr[:, dofs], reg))
        residual = np.r_[error, axis_weight * np.cross(up, axis), posture_weight * anchor]
        delta = np.linalg.solve(jac.T @ jac + damping, jac.T @ residual)
        following = np.clip(q[:4] + np.clip(delta, -0.045, 0.045), env.low[:4], env.high[:4])
        if np.max(np.abs(following - q[:4])) < 1e-5:
            break
        q[:4] = following
    return best


def clean_hold_expert(
    env: SmolVLAInsertion,
    case: dict,
    record=None,
    *,
    seat_offset: float = 0.0005,
    settle_ticks: int = 4,
    hold_frames: int = 40,
):
    """Run the 25 Hz expert with locked wrist_roll during descent and a static bottom hold."""
    env.start(case)
    cmd = env.state().astype(float)
    source, target = [env.model.body(n).pos[:2].copy() for n in ("source", "target")]
    info = {}
    scale = 50 / env.control_fps

    def move(phase, point, grip, ticks, up=(0, 0, 1)):
        nonlocal cmd, info
        ticks = ceil(ticks / scale)
        env.phase = phase
        try:
            end = env.inverse_kinematics(point, cmd, up)
        except RuntimeError:
            if np.array_equal(up, (0, 0, 1)):
                raise
            end = env.inverse_kinematics(point, cmd)
        end[-1] = grip
        for t in range(1, ticks + 1):
            f = t / ticks
            action = (cmd + (end - cmd) * (f * f * (3 - 2 * f))).astype(np.float32)
            if record:
                record(action)
            _, done, timeout, info = env.step(action)
            if done or timeout:
                return True
        cmd = end.copy()
        return False

    for phase, z, grip, ticks in (
        ("over_source", 0.105, 0.52, 44),
        ("lower", 0.057, 0.52, 50),
        ("close", 0.057, -0.055, 36),
        ("lift", 0.107, -0.055, 64),
    ):
        if move(phase, [*source, z], grip, ticks):
            return info, float(np.abs(cmd[4]))
    if not env.pick:
        return dict(info, reason="pickup"), float(np.abs(cmd[4]))

    def tip_offset():
        return env.data.body("bolt").xpos - env.data.site("grasp").xpos

    if move("transfer", np.r_[target, MOUTH_Z + 0.030] - tip_offset(), -0.055, 64):
        return info, float(np.abs(cmd[4]))

    locked_roll = float(cmd[4])
    max_abs_roll = abs(locked_roll)
    xy_correction = np.zeros(2)

    def track_tip(z, axis=(0.0, 0.0, 1.0)):
        nonlocal cmd, info, max_abs_roll
        tip = env.data.body("bolt").xpos
        xy_correction[:] = np.clip(xy_correction + np.clip(0.2 * (target - tip[:2]), -0.0001, 0.0001), -0.002, 0.002)
        point = np.r_[target + xy_correction, z]
        target_q = locked_bolt_ik(env, point, np.asarray(axis), locked_roll)
        cmd[:4] += np.clip(target_q[:4] - cmd[:4], -0.025 * scale, 0.025 * scale)
        cmd[4] = locked_roll
        cmd[-1] = -0.055
        max_abs_roll = max(max_abs_roll, abs(float(cmd[4])))
        action = cmd.astype(np.float32)
        if record:
            record(action)
        _, done, timeout, info = env.step(action)
        return done or timeout

    env.phase = "air_align"
    start_z = float(env.data.body("bolt").xpos[2])
    start_axis = env.data.body("bolt").xmat.reshape(3, 3)[:, 2].copy()
    align_ticks = ceil(50 / scale)
    for tick in range(1, align_ticks + 1):
        f = tick / align_ticks
        blend = f * f * (3 - 2 * f)
        z = start_z + (MOUTH_Z + 0.005 - start_z) * blend
        axis = (1 - blend) * start_axis + blend * np.array([0.0, 0.0, 1.0])
        if track_tip(z, axis / np.linalg.norm(axis)):
            return info, max_abs_roll

    descent_ticks = ceil(65 / scale)
    for tick in range(1, descent_ticks + settle_ticks + 1):
        env.phase = "insert" if tick <= descent_ticks else "settle"
        z = MOUTH_Z + 0.005 + (env.floor_z + seat_offset - MOUTH_Z - 0.005) * min(tick / descent_ticks, 1.0)
        if track_tip(z):
            return info, max_abs_roll

    env.phase = "hold"
    hold_cmd = cmd.astype(np.float32).copy()
    for _ in range(hold_frames):
        if record:
            record(hold_cmd)
        _, done, timeout, info = env.step(hold_cmd)
        if done or timeout:
            return info, max_abs_roll
    return dict(info, reason=f"hold_{hold_frames}_complete"), max_abs_roll


def record_hold_episode(env: HoldCollectionEnv, case: dict, settle_ticks: int, hold_frames: int):
    frames = []

    def record(action):
        frames.append(dict(**env.observation(), action=np.clip(action, env.low, env.high).astype(np.float32)))

    max_abs_roll = 0.0
    try:
        info, max_abs_roll = clean_hold_expert(
            env, case, record=record, settle_ticks=settle_ticks, hold_frames=hold_frames
        )
    except RuntimeError as exc:
        if not str(exc).startswith(("Held-bolt IK did not converge", "Unreachable grasp waypoint")):
            raise
        info = dict(success=False, penetration_failure=False, slipped=0, opened_early=0, stable_s=0.0,
                    max_penetration_mm=0.0, reason=str(exc))
    elapsed = env.data.time - env.start_time
    finite = bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all())
    clock_ok = bool(np.isclose(elapsed, env.steps / FPS, atol=1e-9))
    info = dict(info, steps=env.steps, finite=finite, clock_ok=clock_ok, max_abs_roll=max_abs_roll)
    return frames, info


def sample_stratified_holes(count: int, reserved: np.ndarray, seed: int = 20261127) -> list[list[float]]:
    if count % 25 != 0:
        raise ValueError(f"Expected a multiple of 25 for 5x5 stratification, got {count}")
    per_cell = count // 25
    rng = np.random.default_rng(seed)
    edges = np.linspace(-TARGET_RANGE_MM, TARGET_RANGE_MM, 6)
    targets = []
    seen = reserved.copy()
    for gx in range(5):
        for gy in range(5):
            for _ in range(per_cell):
                for _attempt in range(200):
                    xy = rng.uniform([edges[gx], edges[gy]], [edges[gx + 1], edges[gy + 1]])
                    if np.linalg.norm(seen - xy, axis=1).min() > 1e-3:
                        seen = np.vstack((seen, xy))
                        targets.append(xy.tolist())
                        break
                else:
                    raise RuntimeError(f"Could not sample distinct hole in cell {(gx, gy)}")
    return targets


def run_collect(args) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    splits = read_json(args.splits)
    historical = read_json(args.historical)["rows"]
    reserved = np.array(
        [c["target"] for c in splits["train"] + splits["validation"]]
        + [[r["target_x_mm"], r["target_y_mm"]] for r in historical],
        dtype=float,
    )
    candidates = sample_stratified_holes(args.episodes * 3, reserved)
    rng = np.random.default_rng(20261128)

    args.output.mkdir(parents=True, exist_ok=True)
    hold_root = args.output / "hold50-dataset"
    if hold_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset at {hold_root}")
    mixed_root = args.scratch / "mixed-dataset"
    if mixed_root.exists():
        shutil.rmtree(mixed_root)
    mixed_root.parent.mkdir(parents=True, exist_ok=True)

    env = HoldCollectionEnv(penetration_limit_mm=PENETRATION_LIMIT_MM)
    assert_straight_solref011(env)
    replay_env = SmolVLAInsertion(penetration_limit_mm=PENETRATION_LIMIT_MM)
    assert_straight_solref011(replay_env)
    dataset = create_dataset(hold_root, HOLD_REPO_ID)

    accepted_cases = []
    lengths = []
    stables = []
    max_pens = []
    max_rolls = []
    candidate_idx = 0
    try:
        with (args.output / "collection.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            while len(accepted_cases) < args.episodes:
                if candidate_idx >= len(candidates):
                    raise RuntimeError("Exhausted candidate holes before collecting required episodes")
                target = candidates[candidate_idx]
                candidate_idx += 1
                case = make_case(target, rng)
                frames, info = record_hold_episode(env, case, args.settle_ticks, args.hold_frames)
                ok = (
                    bool(info["success"])
                    and bool(info["clock_ok"])
                    and bool(info["finite"])
                    and not bool(info["penetration_failure"])
                    and info["slipped"] == 0
                    and info["opened_early"] == 0
                    and info["stable_s"] >= 1.2
                    and info["max_penetration_mm"] <= 1.5
                    and info["max_abs_roll"] <= 0.15
                    and len(frames) >= 225
                )
                replay_ok = None
                if ok and len(accepted_cases) < 3:
                    replay_env.start(case)
                    r_info = {}
                    for frame in frames:
                        _, r_done, r_to, r_info = replay_env.step(frame["action"])
                        if r_done or r_to:
                            break
                    replay_ok = bool(
                        r_info.get("success") and not r_info.get("slipped") and not r_info.get("penetration_failure")
                    )
                    if not replay_ok:
                        raise RuntimeError(f"Open-loop replay gate failed on episode {len(accepted_cases)}: {r_info}")

                print(
                    json.dumps(
                        dict(
                            candidate=candidate_idx - 1,
                            accepted=ok,
                            frames=len(frames),
                            stable_s=info["stable_s"],
                            max_penetration_mm=info["max_penetration_mm"],
                            max_abs_roll=info["max_abs_roll"],
                            replay_ok=replay_ok,
                            target=target,
                        )
                    ),
                    file=log,
                )
                if not ok:
                    continue
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                accepted_cases.append(case)
                lengths.append(len(frames))
                stables.append(float(info["stable_s"]))
                max_pens.append(float(info["max_penetration_mm"]))
                max_rolls.append(float(info["max_abs_roll"]))
                print(
                    f"HOLD_COLLECT {len(accepted_cases)}/{args.episodes} frames={len(frames)} "
                    f"stable_s={info['stable_s']:.2f} max_pen={info['max_penetration_mm']:.2f} "
                    f"max_roll={info['max_abs_roll']:.3f}",
                    flush=True,
                )
    finally:
        dataset.finalize()
        env.close()
        replay_env.close()

    aggregate_datasets(
        [MERGED_REPO_ID, HOLD_REPO_ID],
        MIXED_REPO_ID,
        roots=[args.original_dataset, hold_root],
        aggr_root=mixed_root,
        concatenate_videos=False,
        concatenate_data=False,
    )
    orig_meta = LeRobotDatasetMetadata(MERGED_REPO_ID, root=args.original_dataset)
    add_meta = LeRobotDatasetMetadata(HOLD_REPO_ID, root=hold_root)
    mix_meta = LeRobotDatasetMetadata(MIXED_REPO_ID, root=mixed_root)
    assert orig_meta.total_episodes == 150 and orig_meta.total_frames == 28_649
    assert add_meta.total_episodes == args.episodes and add_meta.total_frames == sum(lengths)
    assert mix_meta.total_episodes == 150 + args.episodes and mix_meta.total_frames == 28_649 + sum(lengths)
    assert mix_meta.fps == FPS

    manifest = dict(
        repo_id=MIXED_REPO_ID,
        hold_repo_id=HOLD_REPO_ID,
        original_episodes=orig_meta.total_episodes,
        original_frames=orig_meta.total_frames,
        new_episodes=add_meta.total_episodes,
        new_frames=add_meta.total_frames,
        total_episodes=mix_meta.total_episodes,
        total_frames=mix_meta.total_frames,
        fps=FPS,
        settle_ticks=args.settle_ticks,
        hold_frames=args.hold_frames,
        mean_new_frames=float(np.mean(lengths)),
        min_new_frames=int(min(lengths)),
        max_new_frames=int(max(lengths)),
        mean_stable_s=float(np.mean(stables)),
        min_stable_s=float(min(stables)),
        max_penetration_mm=float(max(max_pens)),
        max_abs_roll=float(max(max_rolls)),
        episode_lengths=lengths,
        cases=accepted_cases,
    )
    save_json(args.output / "dataset.json", manifest)
    print("HOLD_DATASET_READY", {k: v for k, v in manifest.items() if k not in ("episode_lengths", "cases")}, flush=True)


def run_audit(args) -> None:
    import torch

    from examples.so101_insertion.smolvla25_train import CAMERA_MAP, verify_processor_stats
    try:
        from smolvla_dagger_loss_audit import batch_for, flow_loss, input_diff, load_dataset
    except ImportError:
        from examples.so101_insertion.smolvla_dagger_loss_audit import batch_for, flow_loss, input_diff, load_dataset
    from examples.so101_insertion.smolvla_rlt_stage2 import FrozenSmolVLA

    if args.checkpoint.resolve().parent.name != "020000":
        raise ValueError("Expected the accepted 020000/pretrained_model")
    manifest = read_json(args.manifest)
    frozen = FrozenSmolVLA(args.checkpoint)
    verify_processor_stats((frozen.pre, frozen.post), args.checkpoint)
    policy = frozen.policy.train()
    assert not any(p.requires_grad for p in policy.parameters())
    config = policy.config

    original = load_dataset(args.original_dataset, MERGED_REPO_ID, config)
    mixed = load_dataset(args.mixed_dataset, manifest["repo_id"], config)
    boundary = manifest["original_frames"]
    if (len(original), len(mixed) - boundary) != (boundary, manifest["new_frames"]):
        raise ValueError("Frame counts do not match dataset.json")

    episodes = mixed.meta.episodes
    starts = list(episodes["dataset_from_index"])
    stops = list(episodes["dataset_to_index"])
    tail_indices = [
        idx
        for ep in range(manifest["original_episodes"], manifest["total_episodes"])
        for idx in range(starts[ep] + 170, stops[ep])
    ]
    if len(tail_indices) < args.batch:
        raise RuntimeError(f"Expected >= {args.batch} tail frames, found {len(tail_indices)}")

    rng = np.random.default_rng(args.seed)
    orig_pick = rng.choice(boundary, args.batch, replace=False)
    new_all_pick = boundary + rng.choice(len(mixed) - boundary, args.batch, replace=False)
    new_tail_pick = rng.choice(tail_indices, args.batch, replace=False)

    torch.manual_seed(args.seed)
    noise = policy.model.sample_noise((args.batch, config.chunk_size, config.max_action_dim), "cuda")
    t_sample = policy.model.sample_time(args.batch, "cuda")

    standalone = batch_for(original, orig_pick, frozen.pre)
    merged = batch_for(mixed, orig_pick, frozen.pre)
    hold_all = batch_for(mixed, new_all_pick, frozen.pre)
    hold_tail = batch_for(mixed, new_tail_pick, frozen.pre)
    difference = input_diff(standalone[0], merged[0], tuple(CAMERA_MAP.values()))

    report = dict(
        checkpoint=str(args.checkpoint),
        seed=args.seed,
        batch=args.batch,
        optimizer_updates=0,
        original_input_max_diff=difference,
    )
    for tag, (batch, source) in (
        ("original", standalone),
        ("original_from_mixed", merged),
        ("hold50_all", hold_all),
        ("hold50_tail", hold_tail),
    ):
        report[tag] = flow_loss(policy, batch, source, noise, t_sample, args.seed)
        print("AUDIT", tag, json.dumps(report[tag]), flush=True)

    save_json(args.output / "frozen-loss-audit.json", report)
    if max(difference.values()) > 1e-6:
        raise RuntimeError(f"Merged original frames differ from standalone original: {difference}")
    max_all_axis = max(report["hold50_all"]["axes"].values())
    max_tail_axis = max(report["hold50_tail"]["axes"].values())
    if max_all_axis > 0.1 or max_tail_axis > 0.25:
        raise RuntimeError(
            f"Safety gate tripped: hold50_all max axis={max_all_axis:.4f}, hold50_tail max axis={max_tail_axis:.4f}"
        )
    print("HOLD_AUDIT_PASSED", dict(max_all_axis=max_all_axis, max_tail_axis=max_tail_axis), flush=True)


def run_train(args) -> None:
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from lerobot.scripts import lerobot_train as trainer
    from examples.so101_insertion.smolvla25_train import CAMERA_MAP, verify_processor_stats

    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.parent.name != f"{START:06d}":
        raise RuntimeError(f"Expected checkpoint {START:06d}, got {checkpoint}")
    data = read_json(args.manifest)
    cfg = read_json(checkpoint / "config.json")
    assert (cfg["chunk_size"], cfg["n_action_steps"], cfg["empty_cameras"]) == (25, 5, 1)
    assert cfg["resize_imgs_with_padding"] == [512, 512]
    accumulation = EFFECTIVE_BATCH // args.micro_batch

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_root = args.scratch / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    make_policy = trainer.make_policy
    make_processors = trainer.make_pre_post_processors
    make_dataloaders = trainer.make_dataloaders
    train_step = trainer.update_policy
    save_checkpoint = trainer.save_checkpoint

    updates = 0
    micro_losses = []
    new_count = observed = probe_count = 0
    prev = time.perf_counter()

    def mixed_loader(loader, seed):
        dataset = loader.dataset
        meta = dataset.meta.episodes
        starts, stops = list(meta["dataset_from_index"]), list(meta["dataset_to_index"])
        orig_ep, orig_frames = data["original_episodes"], data["original_frames"]
        if (
            list(meta["episode_index"]) != list(range(len(starts)))
            or starts != [0, *stops[:-1]]
            or stops[-1] != len(dataset)
            or starts[orig_ep] != orig_frames
            or dataset.absolute_to_relative_idx is not None
        ):
            raise ValueError("Mixed dataset does not match recorded episode boundary")
        new_frames = len(dataset) - orig_frames
        if new_frames != data["new_frames"]:
            raise ValueError(f"Expected {data['new_frames']} new frames, got {new_frames}")

        raw_new = np.ones(new_frames, dtype=np.float64)
        for s, e in zip(starts[orig_ep:], stops[orig_ep:], strict=True):
            rel = np.arange(e - s)
            tail_mask = rel >= args.tail_start_frame
            raw_new[s - orig_frames : e - orig_frames][tail_mask] = args.tail_boost

        weights = torch.empty(len(dataset), dtype=torch.double)
        weights[:orig_frames] = (1.0 - args.new_prob) / orig_frames
        weights[orig_frames:] = torch.from_numpy(raw_new * (args.new_prob / raw_new.sum()))

        samples_per_epoch = len(dataset) // EFFECTIVE_BATCH * EFFECTIVE_BATCH
        sampler = WeightedRandomSampler(
            weights, samples_per_epoch, replacement=True, generator=torch.Generator().manual_seed(seed)
        )
        print(
            "HOLD_SAMPLER",
            dict(
                original_frames=orig_frames,
                new_frames=new_frames,
                new_prob=float(weights[orig_frames:].sum()),
                tail_start_frame=args.tail_start_frame,
                tail_boost=args.tail_boost,
            ),
            flush=True,
        )
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=True,
            collate_fn=loader.collate_fn,
            prefetch_factor=loader.prefetch_factor,
            persistent_workers=loader.persistent_workers,
            multiprocessing_context=loader.multiprocessing_context,
            worker_init_fn=loader.worker_init_fn,
        )

    def lr_factor(step):
        if step < WARMUP:
            return max(1, step) / WARMUP
        progress = (min(step, args.updates) - WARMUP) / max(1, args.updates - WARMUP)
        return args.decay_lr / args.peak_lr + (1.0 - args.decay_lr / args.peak_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def policy_factory(*a, **kw):
        policy = make_policy(*a, **kw).float().requires_grad_(True)
        params = list(policy.parameters())
        assert params and all(p.dtype == torch.float32 and p.requires_grad for p in params)
        print("HOLD_POLICY", dict(parameters=sum(p.numel() for p in params), micro_batch=args.micro_batch), flush=True)
        return policy

    def processors(policy_cfg, pretrained_path=None, **_):
        if pretrained_path is None or Path(pretrained_path).resolve() != checkpoint:
            raise RuntimeError("Processors must load exact 20k checkpoint")
        result = make_processors(policy_cfg, pretrained_path=str(checkpoint))
        verify_processor_stats(result, checkpoint)
        return result

    def dataloaders(config, dataset, eval_dataset, step, parallel):
        train_dl, other = make_dataloaders(config, dataset, eval_dataset, step, parallel)
        return mixed_loader(train_dl, config.seed), other

    def optimizer(config, policy):
        opt = config.optimizer.build(policy.get_optim_params())
        assert all(math.isclose(group["lr"], args.peak_lr) for group in opt.param_groups)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    def update(*a, **kw):
        nonlocal updates, new_count, observed, probe_count, prev
        accelerator = kw["accelerator"]
        scheduler = kw.pop("lr_scheduler")
        assert accelerator.mixed_precision == "bf16" and accelerator.num_processes == 1
        tracker, metrics = train_step(*a, **kw)
        sources = a[2]["episode_index"]
        new_count += int((sources >= data["original_episodes"]).sum().item())
        observed += sources.numel()
        loss = float(tracker.loss.val)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {updates}")
        micro_losses.append(loss)
        if accelerator.sync_gradients:
            if accelerator.optimizer_step_was_skipped or observed != EFFECTIVE_BATCH:
                raise RuntimeError("Skipped or partial gradient sync step")
            lr_used = a[3].param_groups[0]["lr"]
            scheduler.step()
            updates += 1
            now = time.perf_counter()
            row = dict(
                update=updates,
                total_step=START + updates,
                train_loss=float(np.mean(micro_losses)),
                lr=lr_used,
                seconds=now - prev,
                original_samples=observed - new_count,
                new_samples=new_count,
            )
            with (args.output / "train-metrics.jsonl").open("a", encoding="utf-8") as log:
                print(json.dumps(row), file=log)
            if updates <= 10:
                probe_count += new_count
                if updates == 10:
                    measured = probe_count / (10 * EFFECTIVE_BATCH)
                    print("HOLD_MIX_CHECK", dict(new_fraction=measured, samples=probe_count), flush=True)
                    if not 0.20 <= measured <= 0.40:
                        raise RuntimeError(f"First 10 batches new fraction {measured:.3f} outside [0.20, 0.40]")
            if updates == 1 or updates % 50 == 0:
                print("HOLD_TRAIN", json.dumps(row), flush=True)
            micro_losses.clear()
            new_count = observed = 0
            prev = now
        return tracker, metrics

    def directory(output_dir, total_steps, step):
        return ckpt_root / f"{START + step // accumulation:06d}"

    def save(**kwargs):
        final = kwargs["checkpoint_dir"]
        stage = int(final.name) - START
        assert stage in SAVE_AT and updates == stage and not final.exists()
        temporary = final.with_name(f"{final.name}.writing")
        if temporary.exists():
            shutil.rmtree(temporary)
        save_checkpoint(**(kwargs | dict(checkpoint_dir=temporary)))
        model = temporary / "pretrained_model"
        verify_processor_stats((kwargs["preprocessor"], kwargs["postprocessor"]), checkpoint)
        for state in checkpoint.glob("policy_*processor_step_*.safetensors"):
            actual, expected = load_file(model / state.name), load_file(state)
            if actual.keys() != expected.keys() or any(not actual[k].equal(v) for k, v in expected.items()):
                raise RuntimeError(f"Saved processor changed {state.name}")
        assert (model / "model.safetensors").stat().st_size > 0
        # Strip optimizer training_state (~3.0G each) to stay well within /dev/shm budget.
        shutil.rmtree(temporary / "training_state", ignore_errors=True)
        temporary.rename(final)
        print("HOLD_CHECKPOINT", str(final), flush=True)

    trainer.make_policy = policy_factory
    trainer.make_pre_post_processors = processors
    trainer.make_dataloaders = dataloaders
    trainer.make_optimizer_and_scheduler = optimizer
    trainer.update_policy = update
    trainer.get_step_checkpoint_dir = directory
    trainer.should_save_checkpoint = lambda step, *_: step in {n * accumulation for n in SAVE_AT}
    trainer.save_checkpoint = save

    flags = {
        "dataset.repo_id": data["repo_id"],
        "dataset.root": str(args.mixed_dataset),
        "dataset.video_backend": "torchcodec",
        "policy.path": str(checkpoint),
        "policy.device": "cuda",
        "policy.freeze_vision_encoder": "false",
        "policy.train_expert_only": "false",
        "policy.load_vlm_weights": "false",
        "policy.use_amp": "false",
        "policy.compile_model": "true",
        "policy.compile_mode": "default",
        "policy.optimizer_lr": str(args.peak_lr),
        "policy.scheduler_decay_lr": str(args.decay_lr),
        "policy.scheduler_warmup_steps": str(WARMUP),
        "policy.scheduler_decay_steps": str(args.updates),
        "policy.push_to_hub": "false",
        "accelerator.mixed_precision": "bf16",
        "accelerator.gradient_accumulation.steps": str(accumulation),
        "ema.enable": "false",
        "wandb.enable": "false",
        "batch_size": str(args.micro_batch),
        "num_workers": "4",
        "steps": str(args.updates * accumulation),
        "output_dir": str(args.scratch / "trainer_out"),
        "rename_map": json.dumps(CAMERA_MAP),
        "save_freq": str(1000 * accumulation),
        "log_freq": str(50 * accumulation),
        "save_checkpoint": "true",
        "env_eval_freq": "0",
    }
    sys.argv = [sys.argv[0], *[f"--{key}={value}" for key, value in flags.items()]]
    trainer.main()
    if updates != args.updates or any(not (ckpt_root / f"{START + n:06d}" / "pretrained_model").is_dir() for n in SAVE_AT):
        raise RuntimeError("Training ended without all expected checkpoints")
    print("HOLD_TRAIN_COMPLETE", updates, flush=True)


def run_eval(args) -> None:
    from examples.so101_insertion.smolvla_padding_e2e30 import EndToEndTelemetry, failure_mode, replay
    from examples.so101_insertion.smolvla_rlt_stage2 import FrozenSmolVLA

    old = read_json(args.historical)["rows"]
    assert len(old) == 30 and [r["index"] for r in old] == list(range(30))
    cases = [
        dict(
            index=r["index"],
            seed=r["seed"],
            case=dict(target=[r["target_x_mm"], r["target_y_mm"]], source=[0, 0], joints=[0] * 5, noise=r["seed"]),
        )
        for r in old
    ][args.shard :: args.shards]
    if not cases:
        return
    features = FrozenSmolVLA(args.checkpoint)
    env = EndToEndTelemetry(penetration_limit_mm=PENETRATION_LIMIT_MM)
    assert_straight_solref011(env)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for row in cases:
            path = args.output / f"{row['index']:02d}.json"
            if path.exists():
                continue
            end, _ = replay(env, features, row["case"], row["seed"], corrected=True)
            mode, detail = failure_mode(end, env)
            result = (
                row
                | end
                | dict(
                    failure_mode=mode,
                    failure_detail=detail,
                    max_held_depth_mm=env.max_near_depth_mm,
                    first_contact=env.first_contact,
                    wrist_steps_mm=env.wrist_steps_mm,
                )
            )
            save_json(path, result)
            print("HOLD_EVAL", args.checkpoint.parent.name, row["index"], end["success"], mode, end["steps"], flush=True)
    finally:
        env.close()


def run_eval_summary(args) -> None:
    rows = [read_json(p) for p in sorted(args.eval_dir.glob("[0-9][0-9].json"))]
    base_rows = read_json(args.baseline)["rows"]
    assert [(r["index"], r["seed"]) for r in rows] == [(r["index"], r["seed"]) for r in base_rows]
    movement = np.concatenate([r["wrist_steps_mm"] for r in rows])
    report = dict(
        checkpoint=args.label,
        control_hz=FPS,
        penetration_limit_mm=PENETRATION_LIMIT_MM,
        corrected=True,
        episodes=len(rows),
        insertions=sum(r["success"] for r in rows),
        slips=sum(bool(r["slipped"]) for r in rows),
        timeouts=sum(r["timeout"] for r in rows),
        penetration=sum(r["penetration_failure"] for r in rows),
        wrist_p95_mm=float(np.quantile(movement, 0.95)),
        median_max_penetration_mm=float(np.median([r["max_penetration_mm"] for r in rows])),
        failed=dict(Counter(r["failure_mode"] for r in rows if not r["success"])),
        recovered=[r["index"] for r, b in zip(rows, base_rows, strict=True) if r["success"] and not b["success"]],
        regressed=[r["index"] for r, b in zip(rows, base_rows, strict=True) if not r["success"] and b["success"]],
        rows=rows,
    )
    save_json(args.output, report)
    table = [
        {k: v for k, v in r.items() if k not in ("case", "wrist_steps_mm", "first_contact")}
        | dict(x_mm=r["case"]["target"][0], y_mm=r["case"]["target"][1])
        for r in rows
    ]
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("EVAL_SUMMARY", args.label, {k: v for k, v in report.items() if k != "rows"}, flush=True)


def run_select_best(args) -> None:
    baseline = read_json(args.baseline)
    candidates = []
    for label in ("021000", "022000", "023000"):
        summary_path = args.output / f"eval-{label}-summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        candidates.append(
            dict(
                label=label,
                insertions=summary["insertions"],
                slips=summary["slips"],
                penetration=summary["penetration"],
                timeouts=summary["timeouts"],
                wrist_p95_mm=summary["wrist_p95_mm"],
                median_max_penetration_mm=summary["median_max_penetration_mm"],
                recovered=summary["recovered"],
                regressed=summary["regressed"],
            )
        )

    def rank_key(item):
        return (item["insertions"], -item["slips"], -item["penetration"], -item["wrist_p95_mm"])

    best = max(candidates, key=rank_key) if candidates else None
    beats_or_ties = bool(best and (best["insertions"] > baseline["insertions"] or (
        best["insertions"] == baseline["insertions"]
        and best["slips"] <= baseline["slips"]
        and best["wrist_p95_mm"] < baseline["wrist_p95_mm"]
    )))
    persisted = None
    if beats_or_ties and args.scratch is not None:
        src = args.scratch / "checkpoints" / best["label"] / "pretrained_model"
        dst = args.output / "best_checkpoint" / "pretrained_model"
        free_gb = shutil.disk_usage(args.output).free / 1024**3
        if src.is_dir() and free_gb >= 3.0 and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
            persisted = str(dst)

    comparison = dict(
        baseline_20k=dict(
            insertions=baseline["insertions"],
            slips=baseline["slips"],
            penetration=baseline["penetration"],
            timeouts=baseline["timeouts"],
            wrist_p95_mm=baseline["wrist_p95_mm"],
        ),
        candidates=candidates,
        best=best,
        persisted_best_checkpoint=persisted,
    )
    save_json(args.output / "comparison.json", comparison)
    print("COMPARISON", json.dumps(comparison), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_col = sub.add_parser("collect")
    for k in ("original-dataset", "splits", "historical", "output", "scratch"):
        p_col.add_argument(f"--{k}", type=Path, required=True)
    p_col.add_argument("--episodes", type=int, default=50)
    p_col.add_argument("--settle-ticks", type=int, default=4)
    p_col.add_argument("--hold-frames", type=int, default=40)

    p_aud = sub.add_parser("audit")
    for k in ("checkpoint", "original-dataset", "mixed-dataset", "manifest", "output"):
        p_aud.add_argument(f"--{k}", type=Path, required=True)
    p_aud.add_argument("--batch", type=int, default=32)
    p_aud.add_argument("--seed", type=int, default=1000)

    p_trn = sub.add_parser("train")
    for k in ("checkpoint", "manifest", "mixed-dataset", "output", "scratch"):
        p_trn.add_argument(f"--{k}", type=Path, required=True)
    p_trn.add_argument("--micro-batch", type=int, choices=(8, 16), default=16)
    p_trn.add_argument("--updates", type=int, default=3000)
    p_trn.add_argument("--peak-lr", type=float, default=5e-6)
    p_trn.add_argument("--decay-lr", type=float, default=5e-7)
    p_trn.add_argument("--tail-start-frame", type=int, default=150)
    p_trn.add_argument("--tail-boost", type=float, default=2.0)
    p_trn.add_argument("--new-prob", type=float, default=0.30)

    p_ev = sub.add_parser("eval")
    for k in ("checkpoint", "historical", "output"):
        p_ev.add_argument(f"--{k}", type=Path, required=True)
    p_ev.add_argument("--shard", type=int, default=0)
    p_ev.add_argument("--shards", type=int, default=1)

    p_sum = sub.add_parser("eval-summary")
    for k in ("eval-dir", "baseline", "output"):
        p_sum.add_argument(f"--{k}", type=Path, required=True)
    p_sum.add_argument("--label", required=True)

    p_best = sub.add_parser("select-best")
    for k in ("baseline", "output"):
        p_best.add_argument(f"--{k}", type=Path, required=True)
    p_best.add_argument("--scratch", type=Path, default=None)

    args = parser.parse_args()
    dispatch = {
        "collect": run_collect,
        "audit": run_audit,
        "train": run_train,
        "eval": run_eval,
        "eval-summary": run_eval_summary,
        "select-best": run_select_best,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
