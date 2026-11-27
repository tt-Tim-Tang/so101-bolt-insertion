#!/usr/bin/env bash
set -euo pipefail
out=$(cd "$(dirname "$0")" && pwd)
project=/root/gpufree-data/so101-insertion
p=$project/outputs
work=$p/smolvla-rlt-v3-full.dckbrV
base=$p/smolvla25-r18-smoke
gate=$p/smolvla-rlt-v3-gate.QtW9Dy
hist=$p/smolvla-padding-e2e30.glynhJ
online=$p/rlt-td3-online.6yxPz6
solref011=$p/smolvla-straight-solref011.Slx8fE
py=/root/gpufree-data/lerobot/.venv/bin/python
scratch=/dev/shm/smolvla-hold-finetune

export HF_HOME=/root/gpufree-data/hf_cache HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$out/code:$solref011/code:$work/dagger25-clean/code:$work/quantile-base/physical-only:$work/full-action/code:$work/v4-full/code:$work/v4-full:$work/v4-calibration:$work/code:$p/smolvla-rlt-v3-ucrit.Lti5Ic/code:$p/smolvla-rlt-v3-short.NkieVx/code:$gate/code:$gate/lerobot-src:$hist/code:$p/smolvla-padding-sft150.8XW7Gj/code:$p/smolvla-padding-ab.HETrxJ/code:$p/smolvla-motion-audit.wIrrVU/code:$online/expanded-telemetry.bjm05V:$p/rlt-td3-delta-round2.DCPcgP:$online:$p/rlt-td3-offline.pdjkkO:$p/rlt-smolvla20k.XIccAl:$base:$base/deps"

ckpt20k="$base/increment-20k/training/checkpoints/020000/pretrained_model"
orig_ds="$base/increment-20k/data/dataset"
splits="$base/increment-20k/data/splits.json"
hist_sum="$hist/results/summary.json"
base_sum="$solref011/baseline-summary.json"

phase=${1:-all}

if [[ "$phase" == "collect" || "$phase" == "all" ]]; then
  "$py" -u -m examples.so101_insertion.smolvla_hold_finetune collect \
    --original-dataset "$orig_ds" \
    --splits "$splits" \
    --historical "$hist_sum" \
    --output "$out" \
    --scratch "$scratch" \
    --episodes 50 \
    --insertion-seconds 2.8
fi

if [[ "$phase" == "audit" || "$phase" == "all" ]]; then
  "$py" -u -m examples.so101_insertion.smolvla_hold_finetune audit \
    --checkpoint "$ckpt20k" \
    --original-dataset "$orig_ds" \
    --mixed-dataset "$scratch/mixed-dataset" \
    --manifest "$out/dataset.json" \
    --output "$out" \
    --batch 32 \
    --seed 1000
fi

if [[ "$phase" == "train" || "$phase" == "all" ]]; then
  "$py" -u -m examples.so101_insertion.smolvla_hold_finetune train \
    --checkpoint "$ckpt20k" \
    --manifest "$out/dataset.json" \
    --mixed-dataset "$scratch/mixed-dataset" \
    --output "$out" \
    --scratch "$scratch" \
    --micro-batch 16 \
    --updates 3000 \
    --peak-lr 5e-6 \
    --decay-lr 5e-7 \
    --tail-start-frame 150 \
    --tail-boost 2.0 \
    --new-prob 0.30
fi

if [[ "$phase" == "eval" || "$phase" == "all" ]]; then
  for label in 021000 022000 023000; do
    edir="$out/eval-$label"
    mkdir -p "$edir"
    ckpt="$scratch/checkpoints/$label/pretrained_model"
    status=0
    pids=()
    for ((i=0; i<4; i++)); do
      "$py" -u -m examples.so101_insertion.smolvla_hold_finetune eval \
        --checkpoint "$ckpt" \
        --historical "$hist_sum" \
        --output "$edir" \
        --shard "$i" \
        --shards 4 > "$out/eval-$label-$i.log" 2>&1 &
      pids+=("$!")
      sleep 0.5
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    if ((status)); then
      tail -n 20 "$out/eval-$label"-*.log
      exit "$status"
    fi
    "$py" -u -m examples.so101_insertion.smolvla_hold_finetune eval-summary \
      --eval-dir "$edir" \
      --baseline "$base_sum" \
      --output "$out/eval-$label-summary.json" \
      --label "$label"
  done
  "$py" -u -m examples.so101_insertion.smolvla_hold_finetune select-best \
    --baseline "$base_sum" \
    --output "$out" \
    --scratch "$scratch"
  echo "HOLD_FINETUNE_PIPELINE_COMPLETE"
fi
