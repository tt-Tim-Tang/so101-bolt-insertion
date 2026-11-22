#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
root="$PWD"
py="${1:?pass the L40 Python interpreter path}"
base="${2:?pass the local pi05_base checkpoint directory}"
out="$root/outputs/so101_insertion"
run="$out/pi05-full-bc"
status="$out/full-bc.status"
test ! -e "$run"
test -s "$base/model.safetensors"
test -s "$out/demos50/meta/stats.json"
test "$(df --output=avail -B1 "$out" | tail -1)" -ge 19500000000
test "$(df --output=avail -B1 /dev/shm | tail -1)" -ge 42000000000

trap 'rc=$?; printf "exit_code=%s\n" "$rc" >> "$status"' EXIT
export SO101_OPTIMIZER_STATE_DIR
SO101_OPTIMIZER_STATE_DIR="$(mktemp -d /dev/shm/so101-full-bc.XXXXXX)"
printf 'pid=%s\noptimizer_dir=%s\n' "$$" "$SO101_OPTIMIZER_STATE_DIR" > "$status"

export PYTHONPATH="$root/tmp/RL-Token-Pi05-open/lerobot/src" HF_HOME=/root/gpufree-data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_DATASETS_DISABLE_PROGRESS_BARS=1
cam='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

"$py" -m examples.so101_insertion.upstream finetune \
  --dataset.repo_id=local/so101_insertion --dataset.root="$out/demos50" --dataset.video_backend=pyav \
  --policy.type=pi05 --policy.pretrained_path="$base" --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false --policy.gradient_checkpointing=true \
  --policy.empty_cameras=1 --policy.use_relative_actions=false --policy.push_to_hub=false --rename_map="$cam" \
  --policy.optimizer_lr=5e-4 --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=1e-5 --batch_size=1 --num_workers=8 --wandb.enable=false \
  --steps=30000 --save_freq=1000 --log_freq=10 --output_dir="$run"

checkpoint="$run/checkpoints/last"
test -s "$checkpoint/pretrained_model/model.safetensors"
test -s "$checkpoint/training_state/optimizer_state.safetensors"
echo FULL_BC_TRAINING_OK | tee -a "$status"

"$py" -m examples.so101_insertion.online \
  --pi05 "$checkpoint/pretrained_model" --tokenizer "$root/tokenizer" \
  --variant baseline --execution-steps 10 --episodes 30 --start-seed 10000 --video \
  --output "$out/pi05-full-bc-heldout"
test -s "$out/pi05-full-bc-heldout/summary.json"
echo FULL_BC_EVALUATION_OK | tee -a "$status"
