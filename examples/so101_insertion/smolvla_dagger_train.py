"""Continue the 20k SmolVLA weights for 5k updates with frame-balanced DAgger replay."""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, WeightedRandomSampler

from examples.so101_insertion.smolvla25_train import CAMERA_MAP, verify_processor_stats


START = 20000
UPDATES = 5000
SAVE_AT = (1000,2500,5000)
WARMUP = 100
PEAK = 7.5e-6
DECAY = 7.5e-7
EFFECTIVE_BATCH = 64


def mixed_loader(loader,boundary,seed):
    dataset = loader.dataset
    meta = dataset.meta.episodes
    starts, stops = list(meta["dataset_from_index"]), list(meta["dataset_to_index"])
    expected_episodes, orig_frames = boundary["original_episodes"], boundary["original_frames"]
    if (list(meta["episode_index"])!=list(range(len(starts))) or starts!=[0,*stops[:-1]]
            or stops[-1]!=len(dataset) or starts[expected_episodes]!=orig_frames
            or dataset.absolute_to_relative_idx is not None):
        raise ValueError("Mixed dataset does not preserve the recorded original-to-DAgger frame boundary")
    new_frames = len(dataset)-orig_frames
    if new_frames!=boundary["dagger_frames"]:
        raise ValueError("Unexpected DAgger frame count")
    weights = torch.full((len(dataset),), .20/new_frames, dtype=torch.double)
    weights[:orig_frames] = .80/orig_frames
    sampler = WeightedRandomSampler(weights,len(dataset),replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    print("DAGGER_SAMPLER",dict(original_frame_weight=.80/orig_frames,dagger_frame_weight=.20/new_frames,
                                original_frames=orig_frames,dagger_frames=new_frames,
                                dagger_probability=float(weights[orig_frames:].sum())),flush=True)
    return DataLoader(dataset,batch_size=loader.batch_size,sampler=sampler,num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory,drop_last=loader.drop_last,collate_fn=loader.collate_fn,
                      prefetch_factor=loader.prefetch_factor,persistent_workers=loader.persistent_workers,
                      multiprocessing_context=loader.multiprocessing_context,worker_init_fn=loader.worker_init_fn)


def lr_factor(step):
    if step<WARMUP:
        return max(1,step)/WARMUP
    progress = (min(step,UPDATES)-WARMUP)/(UPDATES-WARMUP)
    return DECAY/PEAK + (1.-DECAY/PEAK)*.5*(1.+math.cos(math.pi*progress))


def run(args):
    from lerobot.scripts import lerobot_train as trainer

    checkpoint = args.checkpoint.resolve(strict=True)
    data = json.loads((args.data/"dataset.json").read_text())
    cfg = json.loads((checkpoint/"config.json").read_text())
    assert (cfg["chunk_size"],cfg["n_action_steps"],cfg["empty_cameras"])==(25,5,1)
    assert cfg["resize_imgs_with_padding"]==[512,512]
    accumulation = EFFECTIVE_BATCH//args.micro_batch
    make_policy = trainer.make_policy
    make_processors = trainer.make_pre_post_processors
    make_dataloaders = trainer.make_dataloaders
    train_step = trainer.update_policy
    save_checkpoint = trainer.save_checkpoint
    updates = 0
    micro_losses = []
    new_count = observed = probe_count = 0
    prev = time.perf_counter()
    config_bytes = sum(f.stat().st_size for f in checkpoint.glob("*.safetensors"))
    full_bytes = 3*config_bytes
    args.scratch.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.scratch).free<2*full_bytes or shutil.disk_usage(args.output.parent).free<full_bytes:
        raise RuntimeError("Insufficient space to keep all three full DAgger checkpoints")

    def policy_factory(*a,**kw):
        policy = make_policy(*a,**kw).float().requires_grad_(True)
        params = list(policy.parameters())
        assert params and all(p.dtype==torch.float32 and p.requires_grad for p in params)
        print("DAGGER_POLICY",dict(parameters=sum(p.numel() for p in params),dtype="float32",
                                    compute="bf16",micro_batch=args.micro_batch,accumulation=accumulation),flush=True)
        return policy

    def processors(policy_cfg,pretrained_path=None,**_):
        if pretrained_path is None or Path(pretrained_path).resolve()!=checkpoint:
            raise RuntimeError("DAgger processors must load the accepted 20k checkpoint")
        result = make_processors(policy_cfg,pretrained_path=str(checkpoint))
        verify_processor_stats(result,checkpoint)
        return result

    def dataloaders(config,dataset,eval_dataset,step,parallel):
        train,other = make_dataloaders(config,dataset,eval_dataset,step,parallel)
        return mixed_loader(train,data,config.seed),other

    def optimizer(config,policy):
        opt = config.optimizer.build(policy.get_optim_params())
        assert all(math.isclose(group["lr"],PEAK) for group in opt.param_groups)
        return opt,torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)

    def update(*a,**kw):
        nonlocal updates, new_count, observed, probe_count, prev
        accelerator = kw["accelerator"]
        scheduler = kw.pop("lr_scheduler")
        assert accelerator.mixed_precision=="bf16" and accelerator.num_processes==1
        tracker, metrics = train_step(*a,**kw)
        sources = a[2]["episode_index"]
        new_count += int((sources>=data["original_episodes"]).sum().item())
        observed += sources.numel()
        loss = float(tracker.loss.val)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite DAgger loss at {updates}")
        micro_losses.append(loss)
        if accelerator.sync_gradients:
            if accelerator.optimizer_step_was_skipped or observed!=EFFECTIVE_BATCH:
                raise RuntimeError("DAgger update did not use all 64 frames")
            lr_used = a[3].param_groups[0]["lr"]
            scheduler.step()
            updates += 1
            now = time.perf_counter()
            row = dict(update=updates,total_step=START+updates,train_loss=float(np.mean(micro_losses)),
                       lr=lr_used,next_lr=a[3].param_groups[0]["lr"],seconds=now-prev,
                       original_samples=observed-new_count,dagger_samples=new_count)
            with (args.output/"train-metrics.jsonl").open("a",encoding="utf-8") as log:
                print(json.dumps(row),file=log)
            if updates<=10:
                probe_count += new_count
                print("DAGGER_MIX_BATCH",updates,observed-new_count,new_count,flush=True)
                if updates==10:
                    measured = probe_count/(10*EFFECTIVE_BATCH)
                    print("DAGGER_MIX_CHECK",dict(dagger_fraction=measured,dagger_samples=probe_count,
                                                 total_samples=10*EFFECTIVE_BATCH),flush=True)
                    if not .15<=measured<=.25:
                        raise RuntimeError("First ten effective batches miss the expected 80:20 mix")
            if updates==1 or updates%50==0:
                print("DAGGER_TRAIN",json.dumps(row),flush=True)
            micro_losses.clear()
            new_count = observed = 0
            prev = now
        return tracker,metrics

    def directory(output_dir,total_steps,step):
        return output_dir/"checkpoints"/f"{START+step//accumulation:06d}"

    def save(**kwargs):
        final = kwargs["checkpoint_dir"]
        stage = int(final.name)-START
        assert stage in SAVE_AT and updates==stage and not final.exists()
        destination = args.scratch/final.name if stage<UPDATES else final
        temporary = destination.with_name(f"{destination.name}.writing")
        if destination.exists() or temporary.exists():
            raise FileExistsError(destination)
        save_checkpoint(**(kwargs|dict(checkpoint_dir=temporary)))
        model = temporary/"pretrained_model"
        # Verify normalization both in the live processors and on disk after serialization.
        verify_processor_stats((kwargs["preprocessor"],kwargs["postprocessor"]),checkpoint)
        for state in checkpoint.glob("policy_*processor_step_*.safetensors"):
            actual, expected = load_file(model/state.name), load_file(state)
            if actual.keys()!=expected.keys() or any(not actual[k].equal(v) for k,v in expected.items()):
                raise RuntimeError(f"Saved processor changed {state.name}")
        assert (model/"model.safetensors").stat().st_size>0
        temporary.rename(destination)
        if destination!=final:
            final.parent.mkdir(parents=True,exist_ok=True)
            final.symlink_to(destination,target_is_directory=True)
        print("DAGGER_CHECKPOINT",final,"storage",destination,flush=True)

    trainer.make_policy = policy_factory
    trainer.make_pre_post_processors = processors
    trainer.make_dataloaders = dataloaders
    trainer.make_optimizer_and_scheduler = optimizer
    trainer.update_policy = update
    trainer.get_step_checkpoint_dir = directory
    trainer.should_save_checkpoint = lambda step,*_: step in {n*accumulation for n in SAVE_AT}
    trainer.save_checkpoint = save
    flags = dict(
        **{"dataset.repo_id":data["repo_id"],"dataset.root":args.data/"mixed-dataset",
           "dataset.video_backend":"torchcodec","policy.path":checkpoint,"policy.device":"cuda",
           "policy.freeze_vision_encoder":"false","policy.train_expert_only":"false",
           "policy.load_vlm_weights":"false","policy.use_amp":"false","policy.compile_model":"true",
           "policy.compile_mode":"default","policy.optimizer_lr":PEAK,"policy.scheduler_decay_lr":DECAY,
           "policy.scheduler_warmup_steps":WARMUP,"policy.scheduler_decay_steps":UPDATES,
           "policy.push_to_hub":"false","accelerator.mixed_precision":"bf16",
           "accelerator.gradient_accumulation.steps":accumulation,"ema.enable":"false","wandb.enable":"false"},
        batch_size=args.micro_batch,num_workers=4,steps=UPDATES*accumulation,
        output_dir=args.output,rename_map=json.dumps(CAMERA_MAP),save_freq=1000*accumulation,
        log_freq=50*accumulation,save_checkpoint="true",env_eval_freq=0)
    sys.argv = [sys.argv[0],*[f"--{key}={value}" for key,value in flags.items()]]
    trainer.main()
    if updates!=UPDATES or any(not (args.output/"checkpoints"/f"{START+n:06d}"/"training_state").is_dir()
                              for n in SAVE_AT):
        raise RuntimeError("DAgger run ended without 5k updates and all three full checkpoints")
    print("DAGGER_TRAIN_COMPLETE",updates,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","data","output","scratch"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    parser.add_argument("--micro-batch",type=int,choices=(8,16),default=16)
    run(parser.parse_args())
