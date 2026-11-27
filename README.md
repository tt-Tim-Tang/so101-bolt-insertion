# SO-101 螺栓插入仿真

在 MuJoCo 中用双摄 SmolVLA 控制 SO-101，从固定取件座抓起螺栓，搬运到随机孔位并完成插入。最终展示使用 **20,000 步 SmolVLA + padding 去噪修复**。

<p align="center">
  <img src="media/smolvla-20k-hole-19.gif" width="400" alt="修复版 SmolVLA 从取件座抓取并插入孔位 19">
  <img src="media/smolvla-20k-hole-24.gif" width="400" alt="修复版 SmolVLA 从取件座抓取并插入孔位 24">
</p>

两段均为留出孔全流程回放。每段左侧是外部相机，右侧是腕部相机。 [孔位、种子与回放记录](experiments/smolvla-20k/demo-episodes.json)

## 评测结果

训练数据为 **150 条示教、28,649 帧**。策略观测双摄 RGB 和六维关节状态，输出六维绝对关节目标。仿真以 **25 Hz** 执行，每 5 步重规划。下表使用同一批 30 个留出孔，全部从固定取件座起步，评测穿壁阈值为 3.0 mm。 [数据统计](experiments/smolvla-20k/dataset.json)

| 端到端策略 | 插入 | 滑落 | 超时 |
| --- | ---: | ---: | ---: |
| [修复版 SmolVLA 20k](experiments/smolvla-20k/end-to-end-30.json) | **21/30** | **0** | 9 |
| [修复版 SmolVLA 20k，同种子复测](experiments/smolvla-20k/end-to-end-30-repeat.json) | **23/30** | 1 | 7 |
| [全动作 BC Actor，6 mm 接管](experiments/smolvla-20k/stage-a-bc.json) | 21/30 | 0 | 9 |
| [在线 RL Actor，10k 更新，同一断点两次回放](experiments/smolvla-20k/online-rl.json) | 11/30、12/30 | 0、1 | 19、17 |
| [DAgger 续训 25k](experiments/smolvla-20k/dagger-eval.json) | 18/30 | 9 | 12 |

从孔口预对准状态起步时，修复版 20k 插入 **27/30**。这组测量覆盖接触插入，端到端评测还包含抓取和搬运。 [孔口评测](experiments/smolvla-20k/hole-mouth-30.json)

## 排错记录

- **推理 padding**。SmolVLA 对补齐的动作维度使用解析速度场，原六维关节动作继续使用模型预测。 [实现补丁](patches/smolvla-padding-velocity.patch)
- **在线 RL**。接管后的全动作 Actor 在固定监控孔上从 8/10 提到 9/10；30 个留出孔的端到端评测低于 BC 起点。 [逐孔挽回与丢失](experiments/smolvla-20k/online-rl.json)
- **DAgger**。固定 20k 权重拆开算 flow loss，原数据为 0.000331，新增老师数据为 23.58。原始记录核查显示，老师标签的 `wrist_roll` 范围显著扩大，该轴占新增批次 loss 的 81.1%。改为卡滞时原地接管后，脚本老师在 20 个实际接管回合里救回 4 局。 [卡滞体检](experiments/smolvla-20k/dagger-stall-recenter.json) [逐轴 loss](experiments/smolvla-20k/dagger-frozen-loss.json) · [动作范围](experiments/smolvla-20k/dagger-action-ranges.json)

## 代码与出处

将 `examples/so101_insertion/` 放在 [LeRobot checkout](https://github.com/huggingface/lerobot/tree/4aaff99be4a1d81568c08c8f0296b41b40c99ec4) 内。环境沿用 [MuJoCo Menagerie 的 SO-101 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101)。

- [仿真与 25 Hz 示教采集](examples/so101_insertion/smolvla25_data.py)
- [SmolVLA 微调](examples/so101_insertion/smolvla25_train.py) · [修复版端到端评测](examples/so101_insertion/smolvla_padding_e2e30.py)
- [RL-Token 动作头](examples/so101_insertion/smolvla_rlt_full_action_core.py) · [上游 RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open/tree/e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3)
- [冻结权重 loss 核查](examples/so101_insertion/smolvla_dagger_loss_audit.py) · [早期 π0.5 / 50 Hz 实验](README-pi05-50hz.md) · [第三方许可](third_party)
