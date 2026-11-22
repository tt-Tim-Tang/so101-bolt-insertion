"""Fresh full π0.5 SFT; evaluate the three requested checkpoints on the frozen validation set."""
import json
import os
import sys
from pathlib import Path

import torch

from . import upstream
from .online import Policy
from .wide_eval import evaluate


def main():
    from lerobot.scripts import lerobot_train as trainer
    root=Path(os.environ["BASE"])
    scratch=Path(os.environ["WB"])
    tokenizer=Path(os.environ["SRC"])/"tokenizer"
    cases=json.loads((root/"splits.json").read_text())["validation"]
    if len(cases)!=30 or not (root/"reviewed-videos.json").exists():
        raise RuntimeError("Missing frozen validation split or demonstration review")
    save=trainer.save_checkpoint

    def save_and_validate(**kw):
        ck=kw["checkpoint_dir"]
        state=scratch/"optimizer"/ck.name
        state.mkdir(parents=True,exist_ok=False)
        ck.mkdir(parents=True,exist_ok=True)
        (ck/"training_state").symlink_to(state,target_is_directory=True)
        save(**kw)
        raw=kw["accelerator"].unwrap_model(kw["policy"])
        kw["optimizer"].zero_grad(set_to_none=True)
        policy=Policy.__new__(Policy)
        policy.policy=raw.eval()
        policy.preprocess=kw["preprocessor"]
        policy.postprocess=kw["postprocessor"]
        policy.execution_steps,policy.encoder,policy.token_dim=10,None,0
        torch.cuda.empty_cache()
        try:
            with torch.random.fork_rng(devices=[0]):
                evaluate(policy,cases,root/f"validation-{kw['step']}.jsonl")
        finally:
            raw.train()
        print("VALIDATED",kw["step"],flush=True)

    trainer.save_checkpoint=save_and_validate
    sys.argv=[sys.argv[0],"finetune","--dataset.repo_id=local/so101_wide_sft",f"--dataset.root={root/'dataset'}",
              "--dataset.video_backend=pyav","--policy.type=pi05","--policy.pretrained_path=/root/so101-pi05-base",
              "--policy.device=cuda","--policy.dtype=bfloat16","--policy.freeze_vision_encoder=false",
              "--policy.train_expert_only=false","--policy.gradient_checkpointing=true","--policy.empty_cameras=1",
              "--policy.use_relative_actions=false","--policy.push_to_hub=false",f"--policy.text_tokenizer_name={tokenizer}",
              '--rename_map={"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}',
              "--policy.optimizer_lr=2.5e-5","--policy.scheduler_warmup_steps=100","--policy.scheduler_decay_steps=6000",
              "--policy.scheduler_decay_lr=1e-5","--batch_size=1","--num_workers=0","--seed=1000","--wandb.enable=false",
              "--steps=6000","--save_freq=2000","--log_freq=50",f"--output_dir={root/'train'}"]
    upstream.main()


if __name__=="__main__":
    main()
