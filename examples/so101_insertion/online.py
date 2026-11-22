"""Evaluate π0.5 or train and evaluate RL Token in the SO-101 simulation."""

import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from .scene import ARM, FPS, JOINTS, PARTS, Insertion

CHUNK = 10
CAMERA_MAP = {
    "observation.images.top": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}


class Decision(NamedTuple):
    token: torch.Tensor
    state: torch.Tensor
    reference: torch.Tensor


def diffusion_noise(config, generator):
    return torch.normal(0., 1., size=(1, config.chunk_size, config.max_action_dim),
                        dtype=torch.float32, device=generator.device, generator=generator)


def sft_frame(observation):
    frame = {"task": observation["task"], "observation.state": torch.from_numpy(observation["observation.state"])}
    for camera in CAMERA_MAP:
        frame[camera] = torch.from_numpy(observation[camera]).permute(2, 0, 1).float() / 255
    return frame


class Policy:
    def __init__(self, checkpoint, encoder, token_dim, device, tokenizer, execution_steps=CHUNK):
        from safetensors.torch import load_model

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        config = PI05Config.from_pretrained(checkpoint)
        config.device = device
        config.gradient_checkpointing = False
        config.compile_model = False
        if config.use_relative_actions or config.output_features["action"].shape != (len(JOINTS),):
            raise ValueError("Use the insertion π0.5 checkpoint with six absolute joint targets")
        if not 1 <= execution_steps <= config.chunk_size:
            raise ValueError(f"execution_steps must be in [1, {config.chunk_size}]")
        self.policy = PI05Policy(config)
        load_model(self.policy, checkpoint / "model.safetensors", strict=True)
        self.policy.to(device).eval().requires_grad_(False)
        overrides = {"device_processor": {"device": device}}
        if tokenizer:
            overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer}
        self.preprocess, self.postprocess = make_pre_post_processors(config, str(checkpoint),
                                                                   preprocessor_overrides=overrides)
        self.encoder = encoder
        self.token_dim = token_dim
        self.execution_steps = execution_steps

    @torch.no_grad()
    def decide(self, observation, active, *, noise=None):
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = self.preprocess(sft_frame(observation))
        reference = self.policy.predict_action_chunk(batch, **({"noise": noise} if noise is not None else {}))
        token = reference.new_zeros(self.token_dim, dtype=torch.float32)
        if active and self.encoder is not None:
            from lerobot.policies.pi05_rlt.vla_compat import extract_embeddings

            config = self.policy.config
            images, masks = self.policy._preprocess_images(batch)
            suffix = reference.new_zeros((1, config.chunk_size, config.max_action_dim))
            features, _, feature_mask = extract_embeddings(
                self.policy, images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                suffix, noise=suffix, time=reference.new_ones(1),
                chunk_size=config.chunk_size, max_action_dim=config.max_action_dim, image_only=True,
            )
            token = self.encoder(features.float(), feature_mask)[0]
        return Decision(token.cpu(), batch["observation.state"][0].float().cpu(),
                        reference[0, :self.execution_steps].float().cpu())

    def joint_targets(self, action):
        return self.postprocess(action.unsqueeze(0))[0].numpy()


class EvaluationActor:
    """Read actor.pt files produced by earlier simulation experiments."""

    def __init__(self, config, device):
        from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTChunkActor

        self.device = device
        self.actor = RLTChunkActor(config).to(device).eval()

    @torch.no_grad()
    def select(self, point):
        mean, _ = self.actor(*(x.unsqueeze(0).to(self.device) for x in point), training=False)
        return mean[0].cpu()


def rollout(env, policy, agent, seed, jitter, *, use_actor, video=None, stop_empty=False,
            scripted_pickup=False):
    torch.manual_seed(seed)
    entry_step = None
    near_hole_step = None
    actor_chunks = 0
    pickup_step = empty_transfer_step = None
    max_lift = 0.0
    lifted_steps = 0
    visited_source = False
    min_radial_error = float("inf")

    def track_pickup(info):
        nonlocal pickup_step, max_lift, lifted_steps, visited_source, min_radial_error
        lift, at_source, toward_target, bolt_at_jaws = env.pickup_state()
        visited_source |= at_source
        max_lift = max(max_lift, lift)
        min_radial_error = min(min_radial_error, info["radial_error_m"])
        lifted_steps = lifted_steps + 1 if lift >= .025 and bolt_at_jaws else 0
        if pickup_step is None and lifted_steps >= 10:
            pickup_step = env.steps - lifted_steps + 1
        return visited_source and toward_target and not bolt_at_jaws and not info["success"]

    def result(info, *, stopped_empty=False, pickup_failed=False):
        return {"seed": seed, "steps": env.steps, "critical_entry": entry_step,
                "near_hole_step": near_hole_step, "min_radial_error_m": min_radial_error,
                "pickup_step": pickup_step, "max_lift_m": max_lift,
                "empty_transfer_step": empty_transfer_step, "stopped_empty": stopped_empty,
                "pickup_failed": pickup_failed, "actor_chunks": actor_chunks, **info}

    def capture(observation):
        video.append(np.concatenate([observation[key] for key in CAMERA_MAP], axis=1))

    if scripted_pickup:
        from .demonstrate import demonstrate

        info = demonstrate(env, seed, jitter=jitter, pickup_only=True,
                           record=None if video is None else capture, after_step=track_pickup)
        if lifted_steps < 10 or env.steps >= env.max_steps:
            return result(info, pickup_failed=True)
    else:
        info = env.reset(seed, jitter=jitter)
        min_radial_error = info["radial_error_m"]
    point = policy.decide(env.observation(), False)
    while True:
        active = entry_step is not None
        if active and use_actor:
            action = agent.select(point)
            actor_chunks += 1
        else:
            action = point.reference
        targets = policy.joint_targets(action)
        for index, target in enumerate(targets):
            if video is not None:
                capture(env.observation())
            reward, terminated, truncated, info = env.step(target)
            empty_transfer = track_pickup(info)
            if empty_transfer and empty_transfer_step is None:
                empty_transfer_step = env.steps
            stopped_empty = stop_empty and empty_transfer
            if near_hole_step is None and info["critical"]:
                near_hole_step = env.steps
            if terminated or truncated or stopped_empty:
                break
        done = terminated or truncated or stopped_empty
        if not done and entry_step is None and info["critical"]:
            entry_step = env.steps
        following = point if done else policy.decide(env.observation(), entry_step is not None)
        point = following
        if done:
            return result(info, stopped_empty=stopped_empty)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05", type=Path, required=True)
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "proprio", "token"), default="baseline")
    parser.add_argument("--execution-steps", type=int, default=CHUNK, help="Actions per π0.5 prediction")
    parser.add_argument("--stop-empty", action="store_true", help="End simulation when the gripper travels empty")
    parser.add_argument("--scripted-pickup", action="store_true", help="Verify a scripted pickup before π0.5 takes over")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved actor instead of training")
    parser.add_argument("--tokenizer", help="Optional local PaliGemma tokenizer path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", type=Path, default=ARM)
    parser.add_argument("--parts", type=Path, default=PARTS)
    parser.add_argument("--episodes", type=int, help="Evaluation episode count; training uses the upstream template")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--run-stage2", action="store_true", help="Execute Stage 2 in MuJoCo after preflight")
    parser.add_argument("--operator-resume", type=Path, help="Resume a training checkpoint with interactive insertion corrections")
    parser.add_argument("--jitter", type=float, default=.001)
    parser.add_argument("--video", action="store_true", help="Save a clip for the first episode")
    args = parser.parse_args()
    if (args.episodes is not None and args.episodes < 1) or args.jitter < 0:
        parser.error("episodes must be positive and jitter must be nonnegative")
    if args.operator_resume and (not args.run_stage2 or args.evaluate or args.variant != "token" or args.episodes is None):
        parser.error("--operator-resume requires --run-stage2 --variant token and a positive --episodes pilot length")
    legacy_evaluation = args.evaluate is not None and args.evaluate.name == "actor.pt"
    stage2 = (args.variant != "baseline" and not legacy_evaluation) or args.run_stage2 or (
        args.variant == "baseline" and args.stage1 is not None
    )
    if stage2 or args.variant != "baseline":
        if args.execution_steps != CHUNK or args.stop_empty or args.scripted_pickup:
            parser.error("execution-step overrides and scripted pickup checks require standalone baseline evaluation")
    if stage2:
        if args.stage1 is None:
            parser.error("upstream Stage 2 requires a Stage 1 checkpoint")
        if args.variant == "baseline" and args.evaluate:
            parser.error("the baseline does not use actor weights")
        if legacy_evaluation:
            parser.error("use --evaluate actor.pt without --run-stage2 for historical checkpoints")
        from .stage2_sim import run

        run(args)
        return
    if args.variant == "token" and args.stage1 is None:
        parser.error("token evaluation requires a Stage 1 checkpoint")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")

    args.episodes = args.episodes or 10
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder

    random.seed(args.start_seed)
    torch.manual_seed(args.start_seed)
    config = PI05RLTConfig(device=args.device, n_action_steps_rl=CHUNK)
    encoder = None
    if args.variant == "token":
        # Stage 1 checkpoints contain optimizer/RNG metadata alongside the encoder weights.
        stage1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
        if not stage1["image_only"]:
            raise ValueError("Expected an image-only Stage 1 encoder")
        config = PI05RLTConfig(**{**stage1["config"], "device": args.device, "n_action_steps_rl": CHUNK})
        encoder = RLTokenEncoder(config)
        encoder.load_state_dict(stage1["encoder_state_dict"])
        encoder.to(args.device).eval().requires_grad_(False)

    policy = Policy(args.pi05, encoder, config.rlt_hidden_dim, args.device, args.tokenizer, args.execution_steps)
    agent = None
    if args.evaluate:
        saved = torch.load(args.evaluate, map_location=args.device, weights_only=True)
        if saved["variant"] != args.variant or args.variant == "baseline":
            raise ValueError("actor checkpoint and evaluation variant differ")
        agent = EvaluationActor(config, args.device)
        agent.actor.load_state_dict(saved["actor"])

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    env = Insertion(args.arm, args.parts)
    actor_chunks = 0
    successes = 0
    entries = 0
    arrivals = 0
    pickups = empty_transfers = stopped_empty = 0
    try:
        with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as log:
            for episode in range(args.episodes):
                capture = [] if args.video and episode == 0 else None
                metrics = rollout(env, policy, agent, args.start_seed + episode, args.jitter,
                                  use_actor=agent is not None, video=capture,
                                  stop_empty=args.stop_empty, scripted_pickup=args.scripted_pickup)
                actor_chunks += metrics["actor_chunks"]
                if capture:
                    from lerobot.utils.io_utils import write_video

                    write_video(args.output / "episode-0.mp4", capture, FPS)
                successes += metrics["success"]
                entries += metrics["critical_entry"] is not None
                arrivals += metrics["near_hole_step"] is not None
                pickups += metrics["pickup_step"] is not None
                empty_transfers += metrics["empty_transfer_step"] is not None
                stopped_empty += metrics["stopped_empty"]
                line = json.dumps(metrics)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
        summary = {"episodes": args.episodes, "successes": successes, "critical_entries": entries,
                   "near_hole_arrivals": arrivals, "actor_chunks": actor_chunks, "pickups": pickups,
                   "empty_transfers": empty_transfers, "stopped_empty": stopped_empty}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
