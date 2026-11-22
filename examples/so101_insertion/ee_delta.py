"""Train on grasp-site pose deltas and execute them through MuJoCo IK.

Each action is [dx, dy, dz, rx, ry, rz, gripper]: world-frame translation
in metres, rotation vector in the current grasp frame in radians, and an
absolute gripper joint target in radians. Each step uses the measured pose
at that step as its reference, including steps within a predicted chunk.
Observations and task text are copied from the joint-target demonstrations.
"""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import mujoco
import numpy as np

from .scene import FPS, Insertion

if TYPE_CHECKING:
    from torch import Tensor

ACTION_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")
CAMERAS = ("observation.images.top", "observation.images.wrist")
DATASET_ID = "local/so101_insertion_ee_delta"


class DeltaInsertion(Insertion):
    def _grasp_pose(self, joint_targets):
        self.ik_data.qpos[self.joints] = joint_targets
        mujoco.mj_kinematics(self.model, self.ik_data)
        site = self.ik_data.site(self.grasp_id)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        return site.xpos.copy(), quat

    def encode(self, state, joint_action):
        position, orientation = self._grasp_pose(state)
        target_position, target_orientation = self._grasp_pose(np.clip(joint_action, self.low, self.high))
        rotation = np.empty(3)
        mujoco.mju_subQuat(rotation, target_orientation, orientation)
        return np.r_[target_position - position, rotation, joint_action[-1]].astype(np.float32)

    def joint_targets(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(ACTION_NAMES),) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite end-effector delta and gripper values")

        q = np.clip(self.data.qpos[self.joints], self.low, self.high)
        site = self.data.site(self.grasp_id)
        goal_position = site.xpos + action[:3]
        goal_quat = np.empty(4)
        mujoco.mju_mat2Quat(goal_quat, site.xmat)
        mujoco.mju_quatIntegrate(goal_quat, action[3:6], 1.0)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        rotation = np.empty(3)
        dofs = self.dofs[:-1]
        damping = 1e-5 * np.eye(len(dofs))
        for _ in range(12):
            position, quat = self._grasp_pose(q)
            position_error = goal_position - position
            mujoco.mju_subQuat(rotation, goal_quat, quat)
            if np.linalg.norm(position_error) < 1e-5 and np.linalg.norm(rotation) < 1e-4:
                break
            axes = self.ik_data.site(self.grasp_id).xmat.reshape(3, 3)
            mujoco.mj_comPos(self.model, self.ik_data)
            mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.grasp_id)
            # The 5-DOF arm projects an arbitrary 6D pose prediction onto its reachable motion.
            jac = np.vstack((jp[:, dofs], 0.12 * jr[:, dofs]))
            error = np.r_[position_error, 0.12 * (axes @ rotation)]
            update = np.linalg.solve(jac.T @ jac + damping, jac.T @ error)
            next_q = np.clip(q[:-1] + np.clip(update, -0.12, 0.12), self.low[:-1], self.high[:-1])
            if np.linalg.norm(next_q - q[:-1]) < 1e-6:
                break
            q[:-1] = next_q
        q[-1] = action[-1]
        return q

    def step(self, action):
        return super().step(self.joint_targets(action))


def convert(source_path, output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset("local/so101_insertion", root=source_path, video_backend="pyav")
    features = {key: source.features[key] for key in ("observation.state", *CAMERAS)}
    features["action"] = {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": list(ACTION_NAMES)}
    dataset = LeRobotDataset.create(
        repo_id=DATASET_ID, fps=source.fps, root=output, features=features,
        robot_type=source.meta.robot_type, use_videos=False, video_backend="pyav",
    )
    env = DeltaInsertion()
    previous_episode = 0
    try:
        for index in range(len(source)):
            sample = source[index]
            episode = int(sample["episode_index"])
            if episode != previous_episode:
                dataset.save_episode()
                previous_episode = episode
            frame = {key: sample[key] for key in (*features, "task")}
            for camera in CAMERAS:
                rgb = sample[camera].permute(1, 2, 0).numpy()
                frame[camera] = np.rint(rgb * 255).astype(np.uint8)
            frame["action"] = env.encode(sample["observation.state"].numpy(), sample["action"].numpy())
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        dataset.finalize()
        env.close()
    if dataset.num_frames != source.num_frames or dataset.num_episodes != source.num_episodes:
        raise RuntimeError("Delta conversion changed the demonstration counts")
    print(json.dumps({"episodes": dataset.num_episodes, "frames": dataset.num_frames}), flush=True)


class Prediction(NamedTuple):
    reference: "Tensor"


class DeltaPolicy:
    def __init__(self, checkpoint, tokenizer, device, execution_steps):
        from safetensors.torch import load_model

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        config = PI05Config.from_pretrained(checkpoint)
        config.device = device
        config.gradient_checkpointing = False
        config.compile_model = False
        if config.use_relative_actions or config.output_features["action"].shape != (len(ACTION_NAMES),):
            raise ValueError("Expected a seven-dimensional pose delta checkpoint with no joint-delta processor")
        if not 1 <= execution_steps <= config.chunk_size:
            raise ValueError(f"execution_steps must be in [1, {config.chunk_size}]")
        self.policy = PI05Policy(config)
        load_model(self.policy, checkpoint / "model.safetensors", strict=True)
        self.policy.to(device).eval().requires_grad_(False)
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, str(checkpoint), preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
            },
        )
        self.execution_steps = execution_steps

    def decide(self, observation, _active):
        import torch

        frame = {"task": observation["task"], "observation.state": torch.from_numpy(observation["observation.state"])}
        for camera in CAMERAS:
            frame[camera] = torch.from_numpy(observation[camera]).permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(self.preprocess(frame))[0, :self.execution_steps]
            return Prediction(actions.float())

    def joint_targets(self, action):
        # The simulator resolves each delta against its live pose after previous steps execute.
        return self.postprocess(action).numpy()


def evaluate(args):
    from lerobot.utils.io_utils import write_video

    from .online import rollout

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run_config.json").write_text(
        json.dumps({key: value for key, value in vars(args).items() if key != "command"}, default=str, indent=2),
        encoding="utf-8",
    )
    env = DeltaInsertion()
    try:
        policy = DeltaPolicy(args.pi05, args.tokenizer, args.device, args.execution_steps)
        with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as log:
            results = []
            for index in range(args.episodes):
                video = [] if index == 0 else None
                result = rollout(env, policy, None, args.start_seed + index, .001, use_actor=False, video=video)
                if video:
                    write_video(args.output / "episode-0.mp4", video, FPS)
                results.append(result)
                line = json.dumps(result)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
    finally:
        env.close()
    summary = {
        "episodes": len(results),
        "pickups": sum(result["pickup_step"] is not None for result in results),
        "near_hole_arrivals": sum(result["near_hole_step"] is not None for result in results),
        "successes": sum(result["success"] for result in results),
        "empty_transfers": sum(result["empty_transfer_step"] is not None for result in results),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    conversion = subcommands.add_parser("convert")
    conversion.add_argument("--source", type=Path, required=True)
    conversion.add_argument("--output", type=Path, required=True)
    evaluation = subcommands.add_parser("evaluate")
    evaluation.add_argument("--pi05", type=Path, required=True)
    evaluation.add_argument("--tokenizer", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--execution-steps", type=int, default=10)
    evaluation.add_argument("--episodes", type=int, default=30)
    evaluation.add_argument("--start-seed", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "convert":
        convert(args.source, args.output)
    else:
        if args.episodes < 1:
            parser.error("episodes must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
