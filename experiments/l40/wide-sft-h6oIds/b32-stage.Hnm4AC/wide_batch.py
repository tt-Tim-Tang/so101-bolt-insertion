"""Measure physical batch capacity and run a larger-batch comparison from π0.5 base."""
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import upstream


def main():
    from lerobot.scripts import lerobot_train as trainer

    probe=sys.argv[1]=="probe"
    benchmark=sys.argv[1]=="benchmark"
    staged=sys.argv[1] in ("stage","resume")
    resume=sys.argv[1]=="resume"
    directory=Path(os.environ["BATCH_RUN"])
    root=Path(os.environ["BASE"])
    physical=32 if probe else int(sys.argv[2])
    effective=physical if probe else int(sys.argv[3])
    updates=2 if probe else int(sys.argv[4])
    accumulation=effective//physical
    assert accumulation*physical==effective
    previous=directory/"train/checkpoints/last"
    if resume and not (previous/"training_state/training_step.json").exists():
        count=int(previous.resolve(strict=True).name)//accumulation
        archive=f"state-{count}"
        subprocess.run(["sha256sum","-c",f"{archive}.sha256"],cwd=directory,check=True)
        state=(previous/"training_state").resolve()
        if not state.is_relative_to("/dev/shm") or state.name!="training_state" or state.exists():
            raise RuntimeError(f"Cannot restore checkpoint state to {state}")
        state.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(["tar","-I","zstd","-xf",str(directory/f"{archive}.tar.zst"),
                        "-C",str(state.parent)],check=True)
    previous_steps=json.loads((previous/"training_state/training_step.json").read_text())["step"] if resume else 0
    native_save=trainer.save_checkpoint
    update=trainer.update_policy
    if probe:
        def check(tracker,policy,batch,optimizer,*args,**kwargs):
            records=[]
            def two_updates(size):
                b={k:v[:size] if isinstance(v,(torch.Tensor,list,tuple)) else v for k,v in batch.items()}
                assert b["action"].shape[0]==size
                elapsed=[]
                peaks=[]
                for _ in range(2):
                    torch.cuda.synchronize()
                    start=time.monotonic()
                    update(tracker,policy,b,optimizer,*args,**kwargs)
                    torch.cuda.synchronize()
                    elapsed.append(time.monotonic()-start)
                    peaks.append(torch.cuda.max_memory_allocated()/2**30)
                return dict(fits=True,seconds=elapsed,peak_gib=max(peaks))
            # Allocate AdamW's persistent states before testing large activations.
            two_updates(1)
            for size in (32,16,8,4,2,1):
                try:
                    row=two_updates(size)
                except torch.OutOfMemoryError as exc:
                    row=dict(fits=False,error=str(exc).splitlines()[0][:180])
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                records.append(dict(physical=size,**row))
                (directory/"capacity.json").write_text(json.dumps(records,indent=2))
                print("CAPACITY",records[-1],flush=True)
                if size<=16 and row["fits"]:
                    break
            raise SystemExit(0)
        trainer.update_policy=check
    else:
        from .online import Policy
        from .wide_eval import evaluate
        from lerobot.utils.constants import PRETRAINED_MODEL_DIR
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

        # A short comparison uses exactly the beginning of the original LR curve.
        original_build=CosineDecayWithWarmupSchedulerConfig.build
        def fixed_schedule(config,optimizer,num_training_steps):
            return original_build(config,optimizer,max(num_training_steps,config.num_decay_steps))
        CosineDecayWithWarmupSchedulerConfig.build=fixed_schedule
        completed=previous_steps//accumulation
        samples=previous_steps*physical
        last_update=None
        def counted(tracker,policy,batch,optimizer,*args,**kwargs):
            nonlocal samples,completed,last_update
            if benchmark and last_update is None:
                last_update=time.perf_counter()-tracker.dataloading_s.val-tracker.preprocessing_s.val
            scheduler=kwargs.pop("lr_scheduler")
            result=update(tracker,policy,batch,optimizer,*args,**kwargs)
            samples+=batch["action"].shape[0]
            accelerator=kwargs["accelerator"]
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                scheduler.step()
                completed+=1
                if benchmark:
                    torch.cuda.synchronize()
                    now=time.perf_counter()
                    with (directory/"throughput.jsonl").open("a") as f:
                        print(json.dumps(dict(updates=completed,samples=samples,seconds=now-last_update,
                                              lr=optimizer.param_groups[0]["lr"],loss=float(tracker.loss.val),
                                              allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                              reserved_gib=torch.cuda.max_memory_reserved()/2**30)),file=f)
                    last_update=now
                elif completed in (1,2) or completed%25==0 or completed==updates:
                    with (directory/"progress.jsonl").open("a") as f:
                        print(json.dumps(dict(updates=completed,samples=samples,lr=optimizer.param_groups[0]["lr"],
                                              loss=float(tracker.loss.val),time=time.time())),file=f)
            return result
        trainer.update_policy=counted

        def checkpoint(checkpoint_dir,step,cfg,policy,optimizer,preprocessor,postprocessor,accelerator,**kw):
            count=step//accumulation
            assert step%accumulation==0 and accelerator.sync_gradients and completed==count
            raw=accelerator.unwrap_model(policy)
            checkpoint_dir.mkdir(parents=True,exist_ok=True)
            if staged:
                state=Path(os.environ["BATCH_WEIGHTS"])/f"{count:06d}"/"training_state"
                state.mkdir(parents=True,exist_ok=False)
                (checkpoint_dir/"training_state").symlink_to(state,target_is_directory=True)
                native_save(checkpoint_dir,step,cfg,policy,optimizer,preprocessor=preprocessor,
                            postprocessor=postprocessor,accelerator=accelerator,**kw)
                assert json.loads((state/"scheduler_state.json").read_text())["last_epoch"]==count
            else:
                dest=Path(os.environ["BATCH_WEIGHTS"])/f"{count:06d}"/PRETRAINED_MODEL_DIR
                dest.mkdir(parents=True,exist_ok=False)
                raw.save_pretrained(dest)
                cfg.save_pretrained(dest)
                preprocessor.save_pretrained(dest)
                postprocessor.save_pretrained(dest)
                (checkpoint_dir/PRETRAINED_MODEL_DIR).symlink_to(dest,target_is_directory=True)
            wrapper=Policy.__new__(Policy)
            wrapper.policy=raw.eval()
            wrapper.preprocess,wrapper.postprocess=preprocessor,postprocessor
            wrapper.execution_steps,wrapper.encoder,wrapper.token_dim=10,None,0
            try:
                with torch.random.fork_rng(devices=[0]):
                    cases=json.loads((root/"splits.json").read_text())["validation"]
                    episodes=[json.loads(line) for line in (root/"collection.jsonl").read_text().splitlines()]
                    train=[r["case"] for r in episodes if r["result"]["success"]]
                    assert len(train)==64 and len(cases)==30
                    seen=[train[i*63//15] for i in range(16)]
                    summary={}
                    for name,starts in (("seen",seen),("validation",cases)):
                        path=directory/f"{name}-{count}.jsonl"
                        evaluate(wrapper,starts,path)
                        rows=[json.loads(line) for line in path.read_text().splitlines()]
                        assert sorted(row["index"] for row in rows)==list(range(len(starts)))
                        summary[name]=dict(n=len(rows),pickup=sum(r["pickup"] for r in rows),
                                           arrival=sum(r["arrival"] for r in rows))
                    (directory/f"metrics-{count}.json").write_text(json.dumps(summary,indent=2))
            finally:
                raw.train()
            print("VALIDATED",count,flush=True)
        trainer.save_checkpoint=checkpoint
    save_every=int(os.environ.get("BATCH_SAVE_UPDATES","2000"))
    sys.argv=[sys.argv[0],"finetune","--dataset.repo_id=local/so101_wide_sft",f"--dataset.root={root/'dataset'}",
              "--dataset.video_backend=pyav","--policy.type=pi05","--policy.pretrained_path=/root/so101-pi05-base",
              "--policy.dtype=bfloat16","--policy.device=cuda","--policy.gradient_checkpointing=true",
              "--policy.freeze_vision_encoder=false","--policy.train_expert_only=false","--policy.empty_cameras=1",
              "--policy.use_relative_actions=false","--policy.push_to_hub=false",
              f"--policy.text_tokenizer_name={Path(os.environ['SRC'])/'tokenizer'}",
              '--rename_map={"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}',
              f"--policy.optimizer_lr={5e-5 if staged else 2.5e-5}","--policy.scheduler_warmup_steps=100",
              f"--policy.scheduler_decay_steps={4000 if staged else 6000}","--policy.scheduler_decay_lr=1e-5",
              f"--batch_size={physical}",f"--accelerator.gradient_accumulation.steps={accumulation}",
              "--num_workers=0","--seed=1000","--wandb.enable=false",f"--steps={updates*accumulation}",
              f"--save_freq={save_every*accumulation}",f"--log_freq={10*accumulation}",
              f"--output_dir={directory/('probe' if probe else 'benchmark' if benchmark else 'train')}"]
    if benchmark:
        sys.argv.append("--save_checkpoint=false")
    if resume:
        sys.argv=[sys.argv[0],"finetune","--resume=true",f"--config_path={previous/'pretrained_model/train_config.json'}",
                  f"--steps={updates*accumulation}",f"--save_freq={save_every*accumulation}"]
    upstream.main()


if __name__=="__main__":
    main()
