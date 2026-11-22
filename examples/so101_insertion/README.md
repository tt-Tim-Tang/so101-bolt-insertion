# SO-101 插螺栓仿真与 RL Token

单个 SO-101 抓起螺栓，移动到固定孔座上方，再对准并插入。顶部和腕部 RGB 相机与六维关节状态供 π0.5 使用。动作是六维绝对关节目标，单位为弧度。脚本示教使用逆运动学和 MuJoCo 接触抓取。场景沿用原仓库的 STL 外观，孔座用凸碰撞体组合近似，螺纹接触未建模。仿真用插入深度、横向误差和倾角判定成功，并用孔口附近的几何条件触发在线 RL。

第二阶段读取固定版本上游的 `configs/stage2_pi05.json`，运行其原始 Actor、Critic、异步更新、步幅为 2 的重叠回放和检查点逻辑。MuJoCo 提供机器人和相机读数、回合重置、孔口开关和终点成败。训练使用完整动作 Actor。proprio 对照组仅在显式选择时将 token 替换为零向量。[RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open) · [MuJoCo Menagerie SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)

## 本地 Windows 验收

从 LeRobot 仓库根目录执行。当前工作区已克隆两个资产来源，并已安装下列独立 MuJoCo 依赖。如果移动了工作区，可用下一节的 Git 命令恢复相同版本。

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$env:HF_HOME = "$PWD\tmp\so101-hf-cache"
uv pip install --target "$PWD\tmp\so101-insertion-deps" mujoco==3.10.0 "numpy>=2.0,<2.3"
$env:PYTHONPATH = "$PWD\tmp\so101-insertion-deps;$PWD\tmp\RL-Token-Pi05-open\lerobot\src"

uv run --no-sync python -m examples.so101_insertion.demonstrate --episodes 5
uv run --no-sync python -m examples.so101_insertion.demonstrate --episodes 1 `
  --output outputs/so101_insertion/local-smoke --video outputs/so101_insertion/local-smoke.mp4
```

## L40 上的 π0.5 与 RL Token 训练

以下使用 Linux Bash，在 LeRobot 根目录运行。正式基线使用 50 条插接示教、30,000 步全参微调，对齐原仓库的示例训练规模。已有的 32 条示教与 3,000 步动作专家试跑另存，可用来复查早期结果。

```bash
RLT="$PWD/tmp/RL-Token-Pi05-open"
MENAGERIE="$PWD/tmp/mujoco_menagerie"
git clone https://github.com/Zhengsw03/RL-Token-Pi05-open.git "$RLT"
git -C "$RLT" checkout e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE"
git -C "$MENAGERIE" sparse-checkout set robotstudio_so101
git -C "$MENAGERIE" checkout 370c43965323c0944b433b202aac5058b5f907a9
uv sync --project "$RLT/lerobot" --locked --extra dataset --extra pi
uv pip install --python "$RLT/lerobot/.venv/bin/python" mujoco==3.10.0
export MUJOCO_GL=egl

run() { uv run --project "$RLT/lerobot" --no-sync "$@"; }
run python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

OUT="$PWD/outputs/so101_insertion"
DEMO="$OUT/demos50"
CAMERAS='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'
test -d "$DEMO" || run python -m examples.so101_insertion.demonstrate --episodes 50 --output "$DEMO"
```

先用 3 步全参更新验证权重、计算和显存。正式训练前，确认输出盘能写入模型和优化器状态。LeRobot 的 base checkpoint 自带的处理器从 Hugging Face 缓存加载 tokenizer，需提前把其 tokenizer 文件下载到同一缓存。 [π0.5 基座文件](https://huggingface.co/lerobot/pi05_base/tree/main)

```bash
TOKENIZER=$(run hf download google/paligemma-3b-pt-224 \
  tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model)
fit() {
  local steps="$1" output="$2"
  shift 2
  run python -m examples.so101_insertion.upstream finetune \
    --dataset.repo_id=local/so101_insertion --dataset.root="$DEMO" \
    --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true --policy.empty_cameras=1 --policy.push_to_hub=false \
    --policy.use_relative_actions=false --rename_map="$CAMERAS" \
    --policy.optimizer_lr=5e-4 --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=30000 --policy.scheduler_decay_lr=1e-5 \
    --batch_size=1 --num_workers=8 --wandb.enable=false \
    --steps="$steps" --output_dir="$output" "$@"
}

fit 3 "$OUT/pi05-full-preflight" --save_checkpoint=false --log_freq=1
fit 30000 "$OUT/pi05-full-bc" --save_freq=30000 --log_freq=10
BC="$OUT/pi05-full-bc/checkpoints/last/pretrained_model"
run python -m examples.so101_insertion.online --pi05 "$BC" --tokenizer "$TOKENIZER" \
  --output "$OUT/baseline-check" --episodes 10 --start-seed 2000 --video
```

当前 L40 实机用下列入口在后台执行同一组训练超参数，每 1,000 步保存一次。数据盘保留最新模型和处理器，AdamW 张量放在 `/dev/shm`，通过 checkpoint 下的链接供原版加载器读取。SSH 断开后仍可继续训练和接续保存。实例重启会清空 `/dev/shm`，此时持久模型仍可评估，原优化器状态无法恢复。训练完成后脚本自动做 30 个留出初态的纯 π0.5 评估。

```bash
nohup bash examples/so101_insertion/run_full_bc.sh \
  /root/gpufree-data/lerobot/.venv/bin/python /root/so101-pi05-base \
  > outputs/so101_insertion/full-bc.log 2>&1 </dev/null &
```

用 `summary.json` 的 `near_hole_arrivals` 检查到达率，用 `successes` 检查插入率。`near_hole_step` 记录螺栓头首次满足孔口几何条件的控制步，条件是横向误差小于 18 mm、螺栓头在孔口高度以上最多 30 mm、倾角小于 0.35 rad。`critical_entry` 记录动作块边界上的门控时刻。Stage 1、在线训练和最终评估使用同一个 BC checkpoint，示教、调试和在线训练使用不同于留出评估的初态种子。

抓取诊断看 `pickup_step` 和 `max_lift_m`。螺栓抬升至少 25 mm，且头部连续 10 步处于夹爪附近，才记为一次持稳提起。纯 π0.5 评估可用 `--execution-steps` 比较重规划频率。`--stop-empty` 可在夹爪访问取料区后向孔座方向移动、但螺栓已离开夹爪时中止回合。该选项使用仿真几何作为外部检查，触发回合记为失败。

`--scripted-pickup --stop-empty` 可评估从可靠抓取之后交接给 π0.5 的效果。起始抓取由示教脚本执行并通过仿真检查，剩余搬运与插入由 π0.5 执行。报告这种设置时注明脚本参与，成功率不能归给纯 π0.5。

```bash
run python -m examples.so101_insertion.upstream embeddings \
  --pi05_path "$BC" --tokenizer_path "$TOKENIZER" --dataset_repo_id "$DEMO" \
  --camera_map "$CAMERAS" --output_dir "$OUT/prefix" --device cuda --batch_size 2 --num_workers 0

run python "$RLT/scripts/train_rlt_stage1_pi05.py" \
  --precomputed_path "$OUT/prefix" --output_dir "$OUT/stage1" \
  --steps 1000 --warmup_steps 100 --save_every 500 --batch_size 8 --num_workers 0 --device cuda --use_amp

STAGE1="$OUT/stage1/best_checkpoint.pt"
rollout() {
  run python -m examples.so101_insertion.online --pi05 "$BC" --tokenizer "$TOKENIZER" --stage1 "$STAGE1" "$@"
}

rollout --variant token --output "$OUT/online-token"  # 仅预检
rollout --run-stage2 --variant baseline --output "$OUT/eval-baseline" --episodes 30 --start-seed 10000 --video
rollout --run-stage2 --variant token --output "$OUT/online-token" --start-seed 1000
rollout --run-stage2 --variant token --evaluate "$OUT/online-token/final_checkpoint.pt" \
  --output "$OUT/eval-token" --episodes 30 --start-seed 10000 --video
```

预热与接管分别按原模板的 5 个回合和 1,000 次学习器更新判断。每批 256 个样本，仿真控制频率 50 Hz，每回合上限 1,000 控制步，训练上限 1,000 局。独立评估用相同控制频率和回合长度，固定 30 个初态，执行 Actor 均值。原版回放日志与完整检查点保存在输出目录，磁盘用量会随训练增长。
