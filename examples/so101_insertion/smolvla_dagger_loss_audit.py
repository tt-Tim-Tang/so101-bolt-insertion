"""Compare frozen 20k flow losses by data source, using the continuation processors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from examples.so101_insertion.scene import JOINTS
from examples.so101_insertion.smolvla25_train import CAMERA_MAP, verify_processor_stats
from examples.so101_insertion.smolvla_rlt_stage2 import FrozenSmolVLA
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch


def load_dataset(root, repo_id, config):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    dataset = LeRobotDataset(repo_id, root=root, video_backend="torchcodec", return_uint8=True,
                             delta_timestamps=resolve_delta_timestamps(config, meta, CAMERA_MAP))
    if meta.fps != 25 or set(meta.camera_keys) != set(CAMERA_MAP):
        raise ValueError(f"Wrong camera keys or control rate in {root}")
    return dataset


def batch_for(dataset, indices, preprocessor):
    batch = default_collate([dataset[int(i)] for i in indices])
    sources = dict(episodes=batch["episode_index"].tolist(), frame_indices=batch["frame_index"].tolist())
    batch = _preprocess_dataset_batch(batch, dataset.meta.camera_keys, CAMERA_MAP, preprocessor)
    return batch, sources


def input_diff(left, right, camera_keys):
    keys = ("action", "action_is_pad", "observation.state", *camera_keys,
            "observation.language.tokens", "observation.language.attention_mask")
    return {key: float((left[key].float()-right[key].float()).abs().max()) for key in keys}


@torch.no_grad()
def flow_loss(policy, batch, sources, noise, time, seed):
    forward = policy.model.forward
    captured = []

    def record(*args, **kwargs):
        losses = forward(*args, **kwargs)
        captured.append(losses[..., :len(JOINTS)].float())
        return losses

    policy.model.forward = record
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(seed)
            total, _ = policy(batch, noise=noise, time=time)
    finally:
        policy.model.forward = forward
    raw, = captured
    valid = ~batch["action_is_pad"]
    per_axis = raw[valid].mean(0)
    torch.testing.assert_close(per_axis.mean(), total.float(), rtol=1e-4, atol=1e-6)
    sample = (raw.mean(-1)*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    max_at = sample.argmax().item()
    return dict(loss=total.item(), axes=dict(zip(JOINTS, per_axis.tolist(), strict=True)),
                valid_steps=int(valid.sum()), samples=len(valid),
                worst_sample=dict(episode=sources["episodes"][max_at], frame=sources["frame_indices"][max_at],
                                  loss=sample[max_at].item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "original", "mixed", "manifest", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.checkpoint.resolve().parent.name != "020000":
        raise ValueError("Expected the accepted 020000/pretrained_model")
    manifest = json.loads(args.manifest.read_text())
    frozen = FrozenSmolVLA(args.checkpoint)
    verify_processor_stats((frozen.pre, frozen.post), args.checkpoint)
    policy = frozen.policy.train()
    assert not any(p.requires_grad for p in policy.parameters())
    config = policy.config
    original = load_dataset(args.original, "local/so101_smolvla25_150", config)
    mixed = load_dataset(args.mixed, manifest["repo_id"], config)
    boundary = manifest["original_frames"]
    if (len(original), len(mixed)-boundary) != (boundary, manifest["dagger_frames"]):
        raise ValueError("Frame counts do not match the audited DAgger manifest")
    rng = np.random.default_rng(args.seed)
    originals = rng.choice(boundary, args.batch, replace=False)
    corrections = boundary+rng.choice(len(mixed)-boundary, args.batch, replace=False)
    torch.manual_seed(args.seed)
    noise = policy.model.sample_noise((args.batch, config.chunk_size, config.max_action_dim), "cuda")
    time = policy.model.sample_time(args.batch, "cuda")
    standalone = batch_for(original, originals, frozen.pre)
    merged = batch_for(mixed, originals, frozen.pre)
    dagger = batch_for(mixed, corrections, frozen.pre)
    difference = input_diff(standalone[0], merged[0], tuple(CAMERA_MAP.values()))
    report = dict(checkpoint=str(args.checkpoint), seed=args.seed, optimizer_updates=0, original_input_max_diff=difference)
    for tag, (batch, source) in (("original", standalone), ("original_from_mixed", merged), ("dagger", dagger)):
        report[tag] = flow_loss(policy, batch, source, noise, time, args.seed)
        print(tag, json.dumps(report[tag]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("ORIGINAL_INPUT_MAX_DIFF", difference, flush=True)


if __name__ == "__main__":
    main()
