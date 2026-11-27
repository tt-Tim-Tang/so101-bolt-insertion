# SO-101 螺栓插入仿真与 π0.5 实验

MuJoCo 中的 SO-101 从取件座抓起螺栓，搬运到独立随机化的目标孔并尝试插入。本项目记录仿真搭建、示教采集、π0.5 监督微调、评估与抓取失败的诊断过程。

## 代码

- [环境与控制](examples/so101_insertion/wide_env.py)定义 50 Hz 控制、六维绝对关节目标和插入成功条件。
- [示教与初态划分](examples/so101_insertion/wide_data.py)包含工作区检查、分层采样和 LeRobot 数据集采集。
- [Batch 1 训练](examples/so101_insertion/wide_train.py)、[Batch 32 训练](examples/so101_insertion/wide_batch.py)、[评估](examples/so101_insertion/wide_eval.py)和[抓取诊断](examples/so101_insertion/wide_diagnose.py)对应当前实验。
- [早期版本与运行说明](examples/so101_insertion/README.md)记录初版 25 Hz 实验和环境依赖。

## 实验结果

2026-11-22 核对 L40 记录。两轮 SFT 使用同一份 50 Hz 数据，取件座与孔座独立采样。64 条成功示教共 32,271 帧，采集尝试 67 次，3 次失败均发生在就位阶段。工作区审计首试通过 98/101，保持座位不变并重采初始关节后通过 101/101。

### 闭环评估

| 权重或对照 | 初态 | 抓稳 | 到孔 | 精对准 | 插入 |
| --- | --- | ---: | ---: | ---: | ---: |
| [脚本专家](experiments/l40/wide-sft-h6oIds/expert-validation.jsonl) | 验证 | 30/30 | 30/30 | 30/30 | 27/30 |
| [固定动作回放](experiments/l40/wide-sft-h6oIds/fixed-validation.jsonl) | 验证 | 6/30 | 0/30 | 0/30 | 0/30 |
| [脚本抓稳后固定回放](experiments/l40/wide-sft-h6oIds/fixed-validation.jsonl) | 验证 | 30/30 | 1/30 | 0/30 | 0/30 |
| [Batch 1 · 2,000 步](experiments/l40/wide-sft-h6oIds/validation-2000.jsonl) | 验证 | 1/30 | 0/30 | 0/30 | 0/30 |
| [Batch 1 · 4,000 步](experiments/l40/wide-sft-h6oIds/validation-4000.jsonl) | 验证 | 1/30 | 0/30 | 0/30 | 0/30 |
| [Batch 1 · 6,000 步](experiments/l40/wide-sft-h6oIds/validation-6000.jsonl) | 验证 | 0/30 | 0/30 | 0/30 | 0/30 |
| [Batch 1 · 2,000 步](experiments/l40/wide-sft-h6oIds/diagnose-seen.jsonl) | 已见 | 0/16 | 0/16 | 0/16 | 0/16 |
| [Batch 1 · 6,000 步](experiments/l40/wide-sft-h6oIds/seen-006000.B00yfq/episodes.jsonl) | 已见 | 0/16 | 0/16 | 0/16 | 0/16 |
| [Batch 1 · 2,000 步](experiments/l40/wide-sft-h6oIds/final-test.jsonl) | 最终测试 | 1/50 | 0/50 | 0/50 | 0/50 |
| [Batch 32 · 1,000 次更新](experiments/l40/wide-sft-h6oIds/b32-stage.Hnm4AC/seen-1000.jsonl) | 已见 | 2/16 | 0/16 | 0/16 | 0/16 |
| [Batch 32 · 1,000 次更新](experiments/l40/wide-sft-h6oIds/b32-stage.Hnm4AC/validation-1000.jsonl) | 验证 | 4/30 | 0/30 | 0/30 | 0/30 |

**最新 Batch 32** 使用实际 batch 16，梯度累积 2 次。1,000 次参数更新累计采样了 32,000 个训练窗口，约 0.99 epoch。峰值学习率为 5×10⁻⁵。30 个相同验证初态上的抓稳从 Batch 1 最好的 1/30 增至 4/30，到孔仍为 0/30。两轮同时改变了学习率和训练样本量，差异不能单独归因于 batch。当前结果未达到接触 RL 的抓稳率 95% 和到孔率 90% 门槛。

[Batch 32 训练日志](experiments/l40/wide-sft-h6oIds/b32-stage.Hnm4AC/train.log) · [实际训练配置](experiments/l40/wide-sft-h6oIds/b32-stage.Hnm4AC/train-config.json) · [每 25 次更新的进度](experiments/l40/wide-sft-h6oIds/b32-stage.Hnm4AC/progress.jsonl) · [原始验证汇总](experiments/l40/wide-sft-h6oIds/b32-stage.Hnm4AC/metrics-1000.json)

### 抓取诊断与对照

- 30 个示教帧的重放检查中，关节状态最大绝对差为 0，外部相机与腕部相机的最大像素差分别为 0 和 1（0–255）。 [重放对齐](experiments/l40/wide-sft-h6oIds/roundtrip.json)
- 6,000 步 Batch 1 权重在 16 个已见初态上抓稳 0 局。6 局在螺栓头高度附近合爪，其中夹爪与螺栓头的最小横向距离中位数为 26.8 mm。 [逐局诊断](experiments/l40/wide-sft-h6oIds/seen-006000.B00yfq/episodes.jsonl)
- 从第 94 帧的专家状态接管，测了 8 条示教。纯模型接管抓稳 3/8，专家手臂配模型夹爪抓稳 8/8，模型手臂配专家夹爪抓稳 4/8。 [接管记录](experiments/l40/wide-sft-h6oIds/quick-grasp.XHTyan/takeover.jsonl)
- 孔位图像干预 30/30 局方向响应为正，沿干预方向的位移差中位数为 3.67 mm。 [干预记录](experiments/l40/wide-sft-h6oIds/image-intervention.jsonl)

[结构化结果汇总](experiments/l40/wide-sft-h6oIds/experiment-results.json) · [工作区与间隙审计](experiments/l40/wide-sft-h6oIds/audit40-center) · [Batch 容量与吞吐](experiments/l40/wide-sft-h6oIds/batch-comparison.PQ876r) · [示教视频及本地记录](experiments/wide-sft-h6oIds)

## 依赖与出处

将示例放入 LeRobot checkout 的 `examples/so101_insertion/`。运行依赖 LeRobot 的 π0.5、MuJoCo、SO-101 机械臂模型、上游 STL 和本地模型权重。依赖的上游源码和资产由相应项目提供。

- [LeRobot](https://github.com/huggingface/lerobot/tree/4aaff99be4a1d81568c08c8f0296b41b40c99ec4)
- [RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open/tree/e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3)
- [MuJoCo Menagerie 的 SO-101 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101)

[第三方许可和声明](third_party)
