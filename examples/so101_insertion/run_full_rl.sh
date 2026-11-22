#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
root="$PWD"
py="${1:?pass the L40 Python interpreter path}"
out="$root/outputs/so101_insertion"
sft_status="$out/full-bc.status"
heldout="$out/pi05-full-bc-heldout"
run="$out/pi05-full-rl"
mkdir "$run"
prefix=""
trap 'rc=$?; if [[ -n "$prefix" && -d "$prefix" ]]; then rm -r -- "$prefix"; fi; printf "exit_code=%s\n" "$rc" >> "$run/status"' EXIT
printf 'pid=%s\nWAITING_FOR_SFT_EVALUATION\n' "$$" > "$run/status"

sft_pid=$(sed -n 's/^pid=\([0-9]\+\)$/\1/p' "$sft_status")
until grep -qx 'exit_code=[0-9]\+' "$sft_status"; do
  if ! kill -0 "$sft_pid" 2>/dev/null; then
    grep -qx 'exit_code=[0-9]\+' "$sft_status" || { echo 'SFT process exited without a status'; exit 1; }
    break
  fi
  sleep 20
done
grep -qx 'exit_code=0' "$sft_status"
grep -qx 'FULL_BC_EVALUATION_OK' "$sft_status"
checkpoint="$out/pi05-full-bc/checkpoints/030000/pretrained_model"
test -s "$checkpoint/model.safetensors"
test -s "$heldout/summary.json"

gate_code=0
"$py" - "$heldout" "$checkpoint" <<'PY' | tee "$run/gate.json" || gate_code=$?
import json
import sys
from pathlib import Path

heldout, checkpoint = map(Path, sys.argv[1:])
config = json.loads((heldout / "run_config.json").read_text())
episodes = [json.loads(line) for line in (heldout / "episodes.jsonl").read_text().splitlines()]
if (config["variant"] != "baseline" or config["execution_steps"] != 10
    or config["scripted_pickup"] or config["stop_empty"] or config["evaluate"] is not None
    or Path(config["pi05"]).resolve() != checkpoint.resolve()
    or [episode["seed"] for episode in episodes] != list(range(10000, 10030))):
    raise RuntimeError("Expected the 30 held-out episodes from the final pure π0.5 checkpoint")

def carried_at(episode, step):
    pickup, empty = episode["pickup_step"], episode["empty_transfer_step"]
    return pickup is not None and step is not None and pickup <= step and (empty is None or empty > step)

arrivals = [episode for episode in episodes if carried_at(episode, episode["near_hole_step"])]
entries = [episode for episode in arrivals if carried_at(episode, episode["critical_entry"])]
positive_rewards = sum(episode["success"] for episode in entries)
eligible = len(arrivals) >= 27 and len(entries) >= 24 and positive_rewards >= 1
print(json.dumps({"episodes": len(episodes), "carried_to_mouth": len(arrivals), "rl_entries": len(entries),
                  "insertions": sum(episode["success"] for episode in episodes),
                  "insertions_after_rl_entry": positive_rewards, "eligible": eligible}, indent=2))
sys.exit(0 if eligible else 10)
PY
if [[ "$gate_code" == 10 ]]; then
  echo RL_GATE_NOT_MET | tee -a "$run/status"
  exit 0
fi
test "$gate_code" == 0
echo RL_GATE_PASSED | tee -a "$run/status"

rlt="$root/tmp/RL-Token-Pi05-open"
export PYTHONPATH="$rlt/lerobot/src" HF_HOME=/root/gpufree-data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_DATASETS_DISABLE_PROGRESS_BARS=1
cam='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

# Three 256-token camera inputs, 2048 fp16 features per token, and a validity mask.
frames=$("$py" -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_frames"])' "$out/demos50/meta/info.json")
test "$(df --output=avail -B1 /dev/shm | tail -1)" -ge "$((frames * 768 * (2048 * 2 + 1)))"
prefix=$(mktemp -d /dev/shm/so101-full-rl-prefix.XXXXXX)
echo PRECOMPUTING_PREFIX | tee -a "$run/status"
"$py" -m examples.so101_insertion.upstream embeddings \
  --pi05_path "$checkpoint" --tokenizer_path "$root/tokenizer" --dataset_repo_id "$out/demos50" \
  --camera_map "$cam" --output_dir "$prefix" --device cuda --batch_size 2 --num_workers 0

echo TRAINING_TOKEN | tee -a "$run/status"
"$py" "$rlt/scripts/train_rlt_stage1_pi05.py" \
  --precomputed_path "$prefix" --output_dir "$run/stage1" \
  --steps 1000 --warmup_steps 100 --save_every 500 --batch_size 8 --num_workers 0 --device cuda --use_amp
stage1="$run/stage1/best_checkpoint.pt"
test -s "$stage1"
rm -r -- "$prefix"
prefix=""

rollout() {
  "$py" -m examples.so101_insertion.online --pi05 "$checkpoint" --tokenizer "$root/tokenizer" \
    --run-stage2 --stage1 "$stage1" "$@"
}
echo EVALUATING_STAGE2_BASELINE | tee -a "$run/status"
rollout --variant baseline --output "$run/baseline" --episodes 30 --start-seed 10000
echo TRAINING_ONLINE_RL | tee -a "$run/status"
rollout --variant token --output "$run/online" --start-seed 1000
test -s "$run/online/final_checkpoint.pt"
echo EVALUATING_ONLINE_RL | tee -a "$run/status"
rollout --variant token --evaluate "$run/online/final_checkpoint.pt" --output "$run/heldout" --episodes 30 --start-seed 10000 --video
test -s "$run/heldout/summary.json"
echo FULL_RL_EVALUATION_OK | tee -a "$run/status"
