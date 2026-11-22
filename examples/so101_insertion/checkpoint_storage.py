"""Keep one native LeRobot checkpoint with its large AdamW tensors on a separate mount."""

import os
from pathlib import Path
from shutil import copy2, rmtree


def configure_checkpoint_storage(trainer, storage: Path) -> None:
    from lerobot.common import train_utils
    from lerobot.utils.constants import (
        LAST_CHECKPOINT_LINK,
        OPTIMIZER_PARAM_GROUPS,
        OPTIMIZER_STATE,
        PRETRAINED_MODEL_DIR,
        TRAINING_STATE_DIR,
    )

    storage = storage.resolve(strict=True)
    save_optimizer = train_utils.save_optimizer_state

    def save_split_optimizer(optimizer, save_dir: Path):
        step = save_dir.parent.name
        if save_dir.name != TRAINING_STATE_DIR or not step.isdecimal():
            raise ValueError(f"Unexpected optimizer checkpoint directory {save_dir}")
        target = storage / step
        target.mkdir()
        save_optimizer(optimizer, target)
        copy2(target / OPTIMIZER_PARAM_GROUPS, save_dir / OPTIMIZER_PARAM_GROUPS)
        (save_dir / OPTIMIZER_STATE).symlink_to(target / OPTIMIZER_STATE)

    def advance_checkpoint(checkpoint_dir: Path):
        checkpoint_dir = checkpoint_dir.resolve(strict=True)
        parent = checkpoint_dir.parent
        new_optimizer = checkpoint_dir / TRAINING_STATE_DIR / OPTIMIZER_STATE
        if new_optimizer.resolve(strict=True) != storage / checkpoint_dir.name / OPTIMIZER_STATE:
            raise RuntimeError(f"Optimizer storage does not match {checkpoint_dir}")
        if not (checkpoint_dir / PRETRAINED_MODEL_DIR / "model.safetensors").is_file():
            raise FileNotFoundError(f"Model weights are missing from {checkpoint_dir}")

        last = parent / LAST_CHECKPOINT_LINK
        previous = last.resolve(strict=True) if last.is_symlink() else None
        if previous is not None and previous != checkpoint_dir:
            if (
                previous.parent != parent
                or not previous.name.isdecimal()
                or (previous / TRAINING_STATE_DIR / OPTIMIZER_STATE).resolve(strict=True)
                != storage / previous.name / OPTIMIZER_STATE
            ):
                raise RuntimeError(f"Existing checkpoint is outside this run's storage: {previous}")
        staged_link = parent / ".last.next"
        staged_link.symlink_to(checkpoint_dir.name, target_is_directory=True)
        os.replace(staged_link, last)
        if previous is not None and previous != checkpoint_dir:
            rmtree(previous)
            rmtree(storage / previous.name)

    train_utils.save_optimizer_state = save_split_optimizer
    trainer.update_last_checkpoint = advance_checkpoint
