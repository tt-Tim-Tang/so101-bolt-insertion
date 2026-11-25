"""Compare native SmolVLA inference with the analytic padding velocity on Episode 20."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .smolvla_motion_audit import CAMERAS, FPS, PHASES, SEEDS, columns, rollout, task_data, tensor_array, write_csv, write_json
from .smolvla_motion_metrics import finite_difference, same_history, stats


@contextmanager
def padding_velocity(policy, corrected, noises, ratios):
    model = policy.model
    config = policy.config
    if config.compile_model or config.rtc_config is not None:
        raise ValueError("This experiment uses eager inference with RTC disabled")
    active_dim = config.action_feature.shape[0]
    if not 0 < active_dim < config.max_action_dim:
        raise ValueError("Expected both physical action and padding channels")
    sample, denoise = model.sample_actions, model.denoise_step
    tail = None

    def sample_and_record(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        nonlocal tail
        if noise is None:
            noise = model.sample_noise((state.shape[0], config.chunk_size, config.max_action_dim), state.device)
        tail = noise[..., active_dim:]
        try:
            output = sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)
            noises.append(tensor_array(noise)[0])
            numerator = torch.linalg.vector_norm(output[..., active_dim:].float())
            denominator = torch.linalg.vector_norm(tail.float())
            ratios.append(float((numerator / denominator).item()))
            return output
        finally:
            tail = None

    def corrected_denoise(*args, **kwargs):
        velocity = denoise(*args, **kwargs)
        # x_pad(t) = t * noise_pad since the clean training target is zero.
        return torch.cat((velocity[..., :active_dim], tail.to(velocity.dtype)), dim=-1)

    model.sample_actions = sample_and_record
    if corrected:
        model.denoise_step = corrected_denoise
    try:
        yield
    finally:
        del model.sample_actions
        if corrected:
            del model.denoise_step


def open_loop(features, source, output):
    policy = features.policy
    metadata = json.loads((source / "chunks.json").read_text())
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    device = next(policy.parameters()).device
    rows, noises, normalized, targets = [], [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk_id in (6, 14, 24, 39):
            with np.load(source / "inputs" / f"{chunk_id:04d}.npz") as raw:
                batch = {key: torch.from_numpy(raw[f"raw.{key}"]).permute(2, 0, 1).float() / 255 for key in CAMERAS}
                batch.update(task=metadata[chunk_id]["task"], **{
                    "observation.state": torch.from_numpy(raw["raw.observation.state"])
                })
            inputs = features.pre(batch)
            for seed in SEEDS:
                generator = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
                pair = []
                stock = None
                if not rows:
                    policy.reset()
                    stock = tensor_array(policy.predict_action_chunk(dict(inputs), noise=noise.clone()))[0]
                for corrected in (False, True):
                    used, ratios = [], []
                    policy.reset()
                    with padding_velocity(policy, corrected, used, ratios):
                        prediction = policy.predict_action_chunk(dict(inputs), noise=noise.clone())
                    norm = tensor_array(prediction)[0]
                    target = tensor_array(features.post(prediction.float()))[0]
                    np.testing.assert_array_equal(used[0], tensor_array(noise)[0])
                    if stock is not None and not corrected:
                        np.testing.assert_array_equal(norm, stock)
                    jerk = finite_difference(target[:, :5], 1 / FPS, 3)
                    pair.append((norm, target, ratios[0], stats(jerk)["rms"], stats(jerk[:5])["rms"]))
                if not 0 <= pair[1][2] < .01:
                    raise RuntimeError(f"Unexpected corrected padding residual: {pair[1][2]}")
                a, b = pair
                diff = b[1] - a[1]
                rows.append(dict(observation_chunk=chunk_id, time_s=metadata[chunk_id]["t_pre_s"], seed=seed,
                                 native_tail_ratio=a[2], corrected_tail_ratio=b[2],
                                 arm_difference_5_rad=stats(diff[:5, :5])["rms"],
                                 arm_difference_25_rad=stats(diff[:, :5])["rms"],
                                 native_arm_jerk_5=a[4], corrected_arm_jerk_5=b[4],
                                 native_arm_jerk_25=a[3], corrected_arm_jerk_25=b[3]))
                noises.append(tensor_array(noise)[0])
                normalized.append([a[0], b[0]])
                targets.append([a[1], b[1]])
    write_csv(output / "open-loop.csv", columns(rows))
    np.savez_compressed(output / "open-loop.npz", initial_noise=noises, normalized=normalized, target_rad=targets)
    medians = {key: float(np.median([row[key] for row in rows])) for key in rows[0] if key not in (
        "observation_chunk", "time_s", "seed"
    )}
    result = dict(pairs=len(rows), native_matches_stock=True, medians=medians)
    write_json(output / "open-loop.json", result)
    print(json.dumps(dict(model=output.name, open_loop=result)), flush=True)
    return result


def closed_metrics(folder):
    result = json.loads((folder / "rollout.json").read_text())
    with np.load(folder / "control.npz") as arrays:
        control = dict(arrays)
    dt = 1 / result["control_hz"]
    labels = control["phase"]
    within = same_history(control["chunk_id"], 4)
    target = finite_difference(control["target"][:, :5], dt, 3)
    sent = finite_difference(control["ctrl"][:, :5], dt, 3)
    actual = finite_difference(control["post_qpos"][:, :5], dt, 3)
    wrist = np.linalg.norm(control["post_ee_pos"] - control["pre_ee_pos"], axis=1) * 1000
    phases = {}
    for phase in ("all", *PHASES):
        selected = np.ones(len(labels), dtype=bool) if phase == "all" else labels == phase
        valid = selected if phase == "all" else selected & same_history(labels, 4)
        phases[phase] = dict(steps=int(selected.sum()), wrist_step_mm=stats(wrist[selected]),
                             target_jerk=stats(target[valid]), within_chunk_target_jerk=stats(target[valid & within]),
                             cross_chunk_target_jerk=stats(target[valid & ~within]),
                             sent_jerk=stats(sent[valid]), actual_jerk=stats(actual[valid]))
    return dict(end=result["end"], steps=result["steps"], events=result["events"], phases=phases)


def main():
    from .smolvla_rlt_stage2 import FrozenSmolVLA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", nargs=2, metavar=("LABEL", "PATH"), required=True, action="append")
    args = parser.parse_args()
    case, expert, observed = task_data(args.data)
    args.output.mkdir()
    summary = dict(episode=20, seeds=SEEDS, models={})
    for label, path in args.checkpoint:
        output = args.output / label
        output.mkdir()
        features = FrozenSmolVLA(Path(path))
        entry = dict(checkpoint=path, open_loop=open_loop(features, args.observations, output), closed_loop=[])
        summary["models"][label] = entry
        for seed in SEEDS:
            pair = dict(seed=seed)
            noise_pair, initial_pair = [], []
            for mode in ("native", "corrected"):
                folder = output / f"{mode}-{seed}"
                noises, ratios = [], []
                with padding_velocity(features.policy, mode == "corrected", noises, ratios):
                    rollout(case, expert, observed, features, seed, folder, record_inputs=False)
                np.savez_compressed(folder / "initial-noise.npz", initial_noise=noises, tail_ratio=ratios)
                if mode == "corrected" and not all(0 <= ratio < .01 for ratio in ratios):
                    raise RuntimeError("Corrected rollout padding residual exceeds 1%")
                pair[mode] = closed_metrics(folder)
                noise_pair.append(np.asarray(noises))
                initial_pair.append(json.loads((folder / "rollout.json").read_text())["initial"])
            common = min(map(len, noise_pair))
            np.testing.assert_array_equal(noise_pair[0][:common], noise_pair[1][:common])
            if initial_pair[0] != initial_pair[1]:
                raise RuntimeError("The simulation initial conditions differ")
            pair["matched_noise_chunks"] = common
            entry["closed_loop"].append(pair)
            write_json(args.output / "summary.json", summary)
            print(json.dumps(dict(model=label, seed=seed, matched_noise_chunks=common,
                                  native=pair["native"]["end"], corrected=pair["corrected"]["end"])), flush=True)
        del features


if __name__ == "__main__":
    main()
