"""MuJoCo I/O for the pinned upstream Stage 2 trainer."""
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from .scene import HOME, JOINTS, ROOT, TASK, Insertion


UPSTREAM = ROOT / "tmp/RL-Token-Pi05-open"


def joint_dict(values):
    return {f"{name}.pos": float(value) for name, value in zip(JOINTS, values, strict=True)}


class Simulator:
    def __init__(self, options):
        self.options = options
        self.proprio_only = options.variant == "proprio"
        self.baseline = options.variant == "baseline"
        self.evaluating = self.baseline or options.evaluate is not None
        self.static_leader = SimpleNamespace(get_action=lambda: joint_dict(HOME))
        self.env = None
        self.results = []
        self.preflight_complete = False
        self.operator = None
        self.reference_rng = None

    def configure(self, args):
        from .online import CAMERA_MAP

        opt = self.options
        args.task = TASK
        args.pi05_path = str(opt.pi05.resolve())
        args.rlt_checkpoint = str(opt.stage1.resolve())
        args.stats_path = None  # Use the quantiles saved with this SFT policy.
        args.tokenizer_path = str(opt.tokenizer) if opt.tokenizer else None
        args.camera_map = json.dumps({key.split(".")[-1]: value for key, value in CAMERA_MAP.items()})
        args.pad_missing_cameras = True  # The SFT checkpoint has an explicit empty third slot.
        args.output_dir = str(opt.output.resolve())
        args.device = opt.device
        args.display_data = False
        args.dry_run = not opt.run_stage2
        if self.evaluating:
            args.max_episodes = opt.episodes if opt.episodes is not None else 30
            args.inference_only = True
            args.inference_score = True
            args.actor_execute_mean = True  # Frozen-policy, paired benchmark.
            args.resume_from = str(opt.evaluate.resolve()) if opt.evaluate else None
        elif opt.operator_resume:
            args.resume_from = str(opt.operator_resume.resolve())
            args.max_episodes = opt.episodes
        elif opt.episodes is not None and opt.episodes != args.max_episodes:
            raise ValueError(f"Stage 2 training uses upstream max_episodes={args.max_episodes}.")
        self.args = args
        self.expected = args.max_episodes

    @staticmethod
    def processor_overrides(tokenizer, device):
        return {"preprocessor_overrides": {"tokenizer_processor": {"tokenizer_name": tokenizer},
                                           "device_processor": {"device": str(device)}}}

    @staticmethod
    def load_policy(checkpoint, device):
        from safetensors.torch import load_model
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        config = PI05Config.from_pretrained(checkpoint)
        if config.use_relative_actions or tuple(config.output_features["action"].shape) != (len(JOINTS),):
            raise ValueError("Expected the six-dimensional absolute-joint insertion policy")
        config.device = str(device)
        model = PI05Policy(config)
        load_model(model, checkpoint / "model.safetensors", strict=True)
        return model

    def connect(self):
        self.env = Insertion(self.options.arm, self.options.parts, max_steps=self.args.steps_per_episode,
                             control_fps=self.args.control_fps)
        self.env.reset(self.options.start_seed, self.options.jitter)
        if self.options.operator_resume:
            from .operator_console import InsertionOperator

            self.operator = InsertionOperator(self.env, self.options.output)
        (self.options.output / "run_config.json").write_text(
            json.dumps(vars(self.args), indent=2), encoding="utf-8"
        )

    def reference_policy(self, model, device):
        if self.evaluating:
            from .online import diffusion_noise

            self.reference_rng = torch.Generator(device=device)
            predict = model.predict_action_chunk
            def predict_matched(batch, **kwargs):
                return predict(batch, noise=diffusion_noise(model.config, self.reference_rng), **kwargs)
            model.predict_action_chunk = predict_matched
        return model

    def start(self):
        pass  # No keyboard thread in the simulator.

    def stop(self):
        pass

    def disconnect(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def begin_episode(self):
        self.seed = self.options.start_seed + len(self.results)
        if self.evaluating:
            torch.manual_seed(self.seed)
            self.reference_rng.manual_seed(self.seed)
        self.env.reset(self.seed, self.options.jitter)
        if self.operator:
            self.operator.reset(self.seed)
        self.done = False
        self.actor_chunks = 0
        self.entry = None
        self.video = [] if self.options.video and not self.results else None

    def begin_chunk(self, source, active):
        if self.entry is None and active:
            self.entry = self.env.steps
        self.actor_chunks += int(source == "ACTOR")

    def get_observation(self):
        obs = self.env.observation()
        return {**joint_dict(obs["observation.state"]),
                "top": obs["observation.images.top"], "wrist": obs["observation.images.wrist"]}

    @staticmethod
    def sft_input(obs, state):
        from .online import CAMERA_MAP, sft_frame

        return sft_frame({"task": TASK, "observation.state": state.detach().cpu().numpy(),
                          **{key: obs[key.split(".")[-1]] for key in CAMERA_MAP}})

    def send_action(self, target):
        if self.video is not None:
            import numpy as np
            obs = self.get_observation()
            self.video.append(np.concatenate((obs["top"], obs["wrist"]), axis=1))
        _, terminated, truncated, _ = self.env.step([target[f"{name}.pos"] for name in JOINTS])
        self.done = terminated or truncated
        return joint_dict(self.env.data.ctrl)

    def prompt_episode_reward(self):
        result = dict(seed=self.seed, steps=self.env.steps, critical_entry=self.entry,
                      actor_chunks=self.actor_chunks, **self.env.info())
        if self.operator:
            result.update(operator_steps=self.operator.human_steps, operator_end=self.operator.end_reason,
                          unaided_success=result["success"] and self.operator.human_steps == 0)
        self.results.append(result)
        with (self.options.output / "episodes.jsonl").open("a", encoding="utf-8") as log:
            print(json.dumps(result), file=log)
        if self.video:
            from lerobot.utils.io_utils import write_video
            write_video(self.options.output / "episode-0.mp4", self.video, int(self.args.control_fps))
        return float(result["success"]), True, False


def run(options):
    import importlib.util
    from unittest.mock import patch

    scripts = UPSTREAM / "scripts"
    filename = "train_rlt_stage2_pi05.py"
    simulator = Simulator(options)
    # Apply only the simulation I/O patch to a private source copy. The real-robot checkout stays pristine.
    with TemporaryDirectory(prefix="so101-stage2-") as temporary:
        source = Path(temporary) / filename
        source.write_text((scripts / filename).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(Path(__file__).with_name("stage2_mujoco.patch"))],
                       cwd=temporary, check=True)
        spec = importlib.util.spec_from_file_location("so101_upstream_stage2", source)
        trainer = importlib.util.module_from_spec(spec)
        original_path = sys.path.copy()
        try:
            sys.path.insert(0, str(scripts))
            sys.modules[spec.name] = trainer
            with patch.object(sys, "argv", [filename, "--config", str(UPSTREAM / "configs/stage2_pi05.json")]):
                spec.loader.exec_module(trainer)
                trainer.ACTOR_ENABLED = not simulator.baseline  # Equivalent to enabling the upstream 'a' switch.
                trainer.main(simulator=simulator)
        finally:
            sys.modules.pop(spec.name, None)
            sys.path[:] = original_path
            simulator.disconnect()
    if not simulator.preflight_complete:
        raise RuntimeError("Stage 2 preflight failed; see the upstream log")
    if simulator.args.dry_run:
        return
    if len(simulator.results) != simulator.expected:
        raise RuntimeError(f"Stage 2 stopped after {len(simulator.results)}/{simulator.expected} episodes")
    if not simulator.evaluating and not (options.output / "final_checkpoint.pt").is_file():
        raise RuntimeError("Upstream Stage 2 did not save its final checkpoint")
    summary = dict(episodes=len(simulator.results), successes=sum(r["success"] for r in simulator.results),
                   actor_chunks=sum(r["actor_chunks"] for r in simulator.results))
    (options.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
