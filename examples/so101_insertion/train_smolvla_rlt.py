"""Train an RLinf token autoencoder on frozen SmolVLA observation features."""

import argparse
import hashlib
import importlib.util
import json
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import torch
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def load_rlt_class(source):
    # This upstream module depends only on PyTorch. Loading it directly avoids
    # initializing RLinf's distributed workers for the offline reconstruction run.
    spec = importlib.util.spec_from_file_location("rlinf_token_transformer", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the RLinf token module at {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RLTTokenTransformer


@torch.no_grad()
def extract_prefix(policy, batch):
    batch = policy._prepare_batch(batch)
    images, image_masks = policy.prepare_images(batch)
    model = policy.model
    embeddings, valid_tokens, attention_groups = model.embed_prefix(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state=policy.prepare_state(batch),
    )
    (prefix, _), _ = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(valid_tokens, attention_groups),
        position_ids=valid_tokens.cumsum(dim=1) - 1,
        inputs_embeds=[embeddings, None],
        use_cache=True,
    )
    return prefix, valid_tokens


def run(args):
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    from .smolvla25_data import MERGED_REPO_ID, VALIDATION_REPO_ID
    from .smolvla25_train import CAMERA_MAP

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_path = args.checkpoint / "pretrained_model"
    config = SmolVLAConfig.from_pretrained(model_path, local_files_only=True)
    config.device = args.device
    config.compile_model = False
    config.load_vlm_weights = False  # Initialize architecture, then load every weight from the SFT checkpoint.
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config, strict=True, local_files_only=True)
    policy.eval().requires_grad_(False)
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(model_path))
    dataset = LeRobotDataset(
        MERGED_REPO_ID, root=args.dataset, return_uint8=True, video_backend=args.video_backend
    )
    if dataset.meta.total_episodes != 150 or dataset.meta.fps != 25:
        raise ValueError("Expected the merged 150-episode SO-101 demonstration dataset at 25 Hz")
    if set(dataset.meta.camera_keys) != set(CAMERA_MAP):
        raise ValueError(f"Unexpected dataset cameras: {dataset.meta.camera_keys}")
    train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation = None
    if args.validation:
        val_data = LeRobotDataset(
            VALIDATION_REPO_ID, root=args.validation, return_uint8=True, video_backend=args.video_backend
        )
        validation = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)

    args.output.mkdir(parents=True)
    token_class = load_rlt_class(args.rlt_source)

    def autocast():
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def features(raw):
        batch = _preprocess_dataset_batch(raw, dataset.meta.camera_keys, CAMERA_MAP, preprocessor)
        with autocast():
            return extract_prefix(policy, batch)

    started = time.monotonic()
    step = 0
    for epoch in range(1, 2 if args.smoke else args.epochs + 1):
        loss_sum = 0.0
        frames = 0
        batches = islice(train, 4) if args.smoke else train
        for raw in batches:
            prefix, mask = features(raw)
            if step == 0:
                spec = dict(input_dim=prefix.size(-1), embed_dim=512, prefix_seq_len=prefix.size(1))
                token = token_class(**spec).to(device)
                optimizer = torch.optim.AdamW(token.parameters(), lr=args.lr)
                metadata = {
                    "model": spec,
                    "smolvla_checkpoint": str(model_path.resolve()),
                    "rlt_source_sha256": hashlib.sha256(args.rlt_source.read_bytes()).hexdigest(),
                    "camera_map": CAMERA_MAP,
                    "episodes": dataset.meta.total_episodes,
                    "dataset_frames": len(dataset),
                    "seed": args.seed,
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "smolvla_trainable_params": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                    "rlt_trainable_params": sum(p.numel() for p in token.parameters()),
                }
                print(json.dumps(metadata), flush=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss, _ = token(prefix, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(token.parameters(), 1.0, error_if_nonfinite=True)
            if step == 0:
                for module in (token.encoder, token.decoder):
                    if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()):
                        raise RuntimeError(f"No training gradient for {type(module).__name__}")
            optimizer.step()
            step += 1
            batch_frames = prefix.size(0)
            loss_sum += loss.detach().item() * batch_frames
            frames += batch_frames
            if step == 1 or step % 50 == 0 or args.smoke:
                print(json.dumps(dict(step=step, epoch=epoch, train_mse=loss.item(),
                                      elapsed_s=round(time.monotonic() - started, 1))), flush=True)

        token.eval()
        metrics = dict(epoch=epoch, step=step, train_mse=loss_sum / frames, frames=frames,
                       elapsed_s=round(time.monotonic() - started, 1))
        with torch.no_grad():
            if validation is not None:
                val_sum = 0.0
                val_frames = 0
                val_batches = islice(validation, 1) if args.smoke else validation
                for raw in val_batches:
                    val_prefix, val_mask = features(raw)
                    with autocast():
                        val_loss, _ = token(val_prefix, val_mask)
                    val_sum += val_loss.item() * val_prefix.size(0)
                    val_frames += val_prefix.size(0)
                metrics["val_mse"] = val_sum / val_frames

            if any(p.grad is not None for p in policy.parameters()):
                raise RuntimeError("The frozen SmolVLA received gradients")
            destination = args.output / "rlt.pt"
            temporary = destination.with_suffix(".tmp")
            torch.save(dict(metadata=metadata, metrics=metrics, state_dict=token.state_dict()), temporary)
            if args.smoke or epoch == args.epochs:
                saved = torch.load(temporary, map_location=device, weights_only=True)
                reloaded = token_class(**saved["metadata"]["model"]).to(device).eval()
                reloaded.load_state_dict(saved["state_dict"], strict=True)
                with autocast():
                    z = reloaded.encode_flat(prefix[:1], mask[:1])
                    torch.testing.assert_close(z, token.encode_flat(prefix[:1], mask[:1]), atol=0, rtol=0)
                    loss_reloaded, _ = reloaded(prefix[:1], mask[:1])
                    if not torch.isfinite(loss_reloaded):
                        raise RuntimeError("Reconstruction loss is non-finite after checkpoint reload")
                metrics["reload_verified"] = True
                metrics["z_dim"] = z.size(1)
            temporary.replace(destination)
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        token.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--rlt-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--smoke", action="store_true", help="Run four updates and check checkpoint reload.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
