# SO-101 螺栓插入仿真与 π0.5 实验

MuJoCo 中的 SO-101 从取件座抓起螺栓，搬运到独立随机化的目标孔并尝试插入。本项目记录仿真搭建、示教采集、π0.5 监督微调、评估与抓取失败的诊断过程。

## 代码

- [环境与控制](examples/so101_insertion/wide_env.py)定义 50 Hz 控制、六维绝对关节目标和插入成功条件。
- [示教与初态划分](examples/so101_insertion/wide_data.py)包含工作区检查、分层采样和 LeRobot 数据集采集。
- [训练](examples/so101_insertion/wide_train.py)、[评估](examples/so101_insertion/wide_eval.py)和[抓取诊断](examples/so101_insertion/wide_diagnose.py)对应当前实验。
- [早期版本与运行说明](examples/so101_insertion/README.md)记录初版 25 Hz 实验和环境依赖。

## 实验记录

[本轮实验](experiments/wide-sft-h6oIds)保存 64 条脚本示教的采集清单、冻结的初态划分、训练统计、对照、验证和最终测试。数据集共 32,271 帧。2,000、4,000、6,000 步权重在 30 个验证初态的抓稳结果分别为 1/30、1/30、0/30。

[L40 诊断快照](experiments/l40/wide-sft-h6oIds)包含训练日志和逐局诊断。6,000 步权重在 16 个已见初态上的抓稳结果为 0/16。实验记录保留了失败结果。

## 依赖与出处

将示例放入 LeRobot checkout 的 `examples/so101_insertion/`。运行依赖 LeRobot 的 π0.5、MuJoCo、SO-101 机械臂模型、上游 STL 和本地模型权重。依赖的上游源码和资产由相应项目提供。

- [LeRobot](https://github.com/huggingface/lerobot/tree/4aaff99be4a1d81568c08c8f0296b41b40c99ec4)
- [RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open/tree/e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3)
- [MuJoCo Menagerie 的 SO-101 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101)

[第三方许可和声明](third_party)
