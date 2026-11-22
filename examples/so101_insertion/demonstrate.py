"""Generate scripted SO-101 insertion demonstrations in LeRobot format."""

import argparse
import json
from pathlib import Path

import numpy as np

from .scene import ARM, FPS, IMAGE_SIZE, JOINTS, PARTS, Insertion


def demonstrate(env, seed, *, jitter, record=None, pickup_only=False, after_step=None):
    env.reset(seed, jitter=jitter)
    source = env.model.body("source").pos[:2].copy()
    target = env.model.body("target").pos[:2].copy()
    command = env.state()
    info = env.info()

    def move(position, grip, steps):
        nonlocal command, info
        end = env.inverse_kinematics(np.array(position), command)
        end[-1] = grip
        for alpha in np.linspace(0, 1, steps + 1)[1:]:
            fraction = alpha * alpha * (3 - 2 * alpha)
            action = ((1 - fraction) * command + fraction * end).astype(np.float32)
            if record is not None:
                record({**env.observation(), "action": action})
            _, terminated, truncated, info = env.step(action)
            if after_step is not None:
                after_step(info)
            if terminated or truncated:
                return True
        command = end
        return False

    for xy, z, grip, steps in (
        (source, .105, .52, 22),
        (source, .057, .52, 25),
        (source, .057, -.055, 18),
        (source, .107, -.055, 32),
    ):
        if move((*xy, z), grip, steps):
            return info

    # Account for the actual offset between the jaws and the grasped bolt.
    if pickup_only or env.data.body("bolt").xpos[2] < .045:
        return info
    if move((*target, .107), -.055, 32):
        return info
    for _ in range(3):
        tip = env.data.body("bolt").xpos.copy()
        grip_position = env.data.site("grasp").xpos.copy()
        offset = tip - grip_position
        xy = target - offset[:2]
        if move((*xy, .089), -.055, 17):
            return info
    move((*xy, .018 - offset[2]), -.055, 39)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, default=ARM)
    parser.add_argument("--parts", type=Path, default=PARTS)
    parser.add_argument("--output", type=Path, help="Optional path for a new LeRobot dataset")
    parser.add_argument("--video", type=Path, help="Save the first successful rollout as an MP4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=.001)
    args = parser.parse_args()
    if args.episodes < 1 or args.jitter < 0:
        parser.error("episodes must be positive and jitter must be nonnegative")

    env = Insertion(args.arm, args.parts)
    dataset = None
    successes = 0
    try:
        if args.output is not None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            features = {
                "observation.state": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
                "action": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
                **{
                    f"observation.images.{camera}": {
                        "dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                        "names": ["height", "width", "channel"],
                    }
                    for camera in ("top", "wrist")
                },
            }
            dataset = LeRobotDataset.create(
                repo_id="local/so101_insertion", fps=FPS, root=args.output, features=features,
                robot_type="so101_mujoco", use_videos=False, video_backend="pyav",
            )
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            frames = []
            capture = dataset is not None or (args.video is not None and not successes)
            info = demonstrate(env, seed, jitter=args.jitter, record=frames.append if capture else None)
            print(json.dumps({"seed": seed, "steps": env.steps, **info}), flush=True)
            if info["success"]:
                successes += 1
                if args.video is not None and successes == 1:
                    from lerobot.utils.io_utils import write_video

                    args.video.parent.mkdir(parents=True, exist_ok=True)
                    video = [np.concatenate([f["observation.images.top"], f["observation.images.wrist"]], axis=1)
                             for f in frames]
                    write_video(args.video, video, FPS)
                if dataset is not None:
                    for frame in frames:
                        dataset.add_frame(frame)
                    dataset.save_episode()
        print(f"Successful demonstrations: {successes}/{args.episodes}", flush=True)
        if not successes:
            raise SystemExit(1)
    finally:
        if dataset is not None:
            dataset.finalize()
        env.close()


if __name__ == "__main__":
    main()
