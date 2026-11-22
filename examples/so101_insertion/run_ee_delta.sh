#!/usr/bin/env bash
# A 3,000-step, full-parameter π0.5 pilot with end-effector delta action labels.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
root="$PWD"
py="${1:?pass the L40 Python interpreter}"
base="${2:?pass the local pi05_base checkpoint}"
run="${3:?pass a new experiment output directory}"
delta_data="${4:?pass the prepared end-effector delta dataset directory}"
source_data="$root/outputs/so101_insertion/demos50"
reference_weights="$root/outputs/so101_insertion/pi05-full-bc/checkpoints/030000/pretrained_model/model.safetensors"
test -s "$base/model.safetensors"
test -s "$delta_data/meta/stats.json"
"$py" - "$source_data/meta/info.json" "$delta_data/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

source, delta = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
for key in ("total_episodes", "total_frames", "fps"):
    if delta[key] != source[key]:
        raise SystemExit(f"Incomplete or mismatched delta dataset: {key}")
if delta["features"]["action"]["names"] != ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]:
    raise SystemExit("Expected seven end-effector delta action fields")
PY
weight_bytes=$(stat -c %s "$reference_weights")
if [[ $(df --output=avail -B1 "$(dirname "$run")" | tail -1) -lt $((weight_bytes + 256 * 1024 * 1024)) ]]; then
  echo INSUFFICIENT_CHECKPOINT_SPACE
  exit 1
fi
if [[ $(df --output=avail -B1 /dev/shm | tail -1) -lt 20000000000 ]]; then
  echo INSUFFICIENT_OPTIMIZER_SPACE
  exit 1
fi
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$gpu_processes" ]]; then
  echo GPU_ALREADY_IN_USE
  exit 1
fi
mkdir "$run"
status="$run/status"
trap 'rc=$?; printf "exit_code=%s\n" "$rc" >> "$status"' EXIT
printf 'pid=%s\n' "$$" > "$status"
scratch=$(mktemp -d /dev/shm/so101-ee-delta.XXXXXX)
export SO101_OPTIMIZER_STATE_DIR="$scratch/optimizer"
export HF_DATASETS_CACHE="$scratch/datasets"
mkdir "$SO101_OPTIMIZER_STATE_DIR"
printf 'scratch=%s\n' "$scratch" >> "$status"

export PYTHONPATH="$root/tmp/RL-Token-Pi05-open/lerobot/src" HF_HOME=/root/gpufree-data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_DATASETS_DISABLE_PROGRESS_BARS=1
cam='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

echo TRAINING_EE_DELTA_SFT | tee -a "$status"
"$py" -m examples.so101_insertion.upstream finetune \
  --dataset.repo_id=local/so101_insertion_ee_delta --dataset.root="$delta_data" --dataset.video_backend=pyav \
  --policy.type=pi05 --policy.pretrained_path="$base" --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false --policy.gradient_checkpointing=true \
  --policy.empty_cameras=1 --policy.use_relative_actions=false --policy.push_to_hub=false --rename_map="$cam" \
  --policy.optimizer_lr=2.5e-5 --policy.scheduler_warmup_steps=100 --policy.scheduler_decay_steps=3000 \
  --policy.scheduler_decay_lr=2.5e-6 --batch_size=1 --num_workers=8 --wandb.enable=false \
  --steps=3000 --save_freq=3000 --log_freq=10 --output_dir="$run/checkpoint"

echo EVALUATING_EE_DELTA_SFT | tee -a "$status"
"$py" -m examples.so101_insertion.ee_delta evaluate \
  --pi05 "$run/checkpoint/checkpoints/last/pretrained_model" --tokenizer "$root/tokenizer" \
  --execution-steps 10 --episodes 30 --start-seed 10000 --output "$run/heldout"
test -s "$run/heldout/summary.json"
echo EE_DELTA_EVALUATION_OK | tee -a "$status"
