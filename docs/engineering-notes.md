# 工程排错与消融记录（Engineering Notes）

[← 返回中文 README](../README.md) · [English README](../README_EN.md)

本文档记录项目推进过程中三次关键的定位、物理对齐与路线证伪实验，所有结论均对应仓库内的原始数据凭证。

---

## 一、LeRobot SmolVLA Padding 推理速度漂移缺陷（`#5335`）

### 1. 现象与代码根因

SO-101 机械臂仅使用 **6 维**物理关节动作，而 `SmolVLAConfig.max_action_dim = 32`，因此高达 **26/32（81.25%）** 的动作通道属于零填充（padding）维度：

1. **训练阶段**：`SmolVLAPolicy` 只对前 `original_action_dim = 6` 维真实关节计算 Flow-Matching loss，后 26 维不参与梯度监督。
2. **推理阶段**：原生 `euler_integrate` 对全部 `32` 维同时积分；每一步 `denoise_step` 会先通过 `action_in_proj`（`32 -> hidden`）读取全部 32 维状态再经注意力交互。
3. **缺陷爆发**：无监督的 26 维 padding 通道在积分过程中产生明显漂移（实测开环尾维相对残留率高达 **`0.506`**），并通过联合隐空间把高频噪声直接混回前 6 维物理动作，导致真机/仿真机械臂在空中就剧烈抖动并在接触孔口时频繁甩飞螺栓。

### 2. 零重训解析修复

由于训练时 padding 通道的目标干净动作恒为 `0`，在流匹配插值 $x_{\text{pad}}(t) = t \cdot \epsilon_{\text{pad}}$ 下，其精确解析速度场为：

$$\frac{dx_{\text{pad}}}{dt} = \epsilon_{\text{pad}}$$

我们在 [patches/smolvla-padding-velocity.patch](../patches/smolvla-padding-velocity.patch) 中增加 `correct_padding_velocity=True` 开关：推理每步去噪后保留前 6 维模型预测速度，将后 26 维替换为初始采样噪声 `noise[..., active_dim:]`。**完全不修改模型权重、零额外训练**，即可将尾维残留从 `0.506` 压到 `0.0023`（**-99.6%**）。

### 3. 严格同种子 A/B 实测数据

| 评测基准 | 原生 SmolVLA 20k | + 解析 Padding 修复（零重训） | 变化幅度 | 凭证文件 |
| --- | ---: | ---: | ---: | --- |
| Ep 20 开环尾维残留中位数 | `0.5056` | **`0.0023`** | **-99.6%** | [padding-ab-ep20.json](../experiments/smolvla-20k/padding-ab-ep20.json) |
| Ep 20 开环 5 步关节 Jerk 中位数 | `262.6 rad/s³` | **`41.9 rad/s³`** | **-84.1%** | [padding-ab-ep20.json](../experiments/smolvla-20k/padding-ab-ep20.json) |
| Ep 20 闭环实际 Jerk RMS 中位数（3 种子） | `326.5 rad/s³` | **`97.2 rad/s³`** | **-70.2%** | [padding-ab-ep20.json](../experiments/smolvla-20k/padding-ab-ep20.json) |
| Ep 20 闭环插入（3 种子） | `0/3` | **`3/3`** | **0% → 100%** | [padding-ab-ep20.json](../experiments/smolvla-20k/padding-ab-ep20.json) |
| SFT 10 验证孔端到端（同种子） | `0/10`（9 滑落） | **`9/10`（0 滑落）** | **0% → 90%** | [padding-sft-val10.json](../experiments/smolvla-20k/padding-sft-val10.json) |
| 孔口 30 留出孔（同种子） | `7/30`（12 滑落） | **`27/30`（0 滑落）** | **23% → 90%** | [hole-mouth-30.json](../experiments/smolvla-20k/hole-mouth-30.json) |

---

## 二、MuJoCo 接触物理对齐与防作弊审计

早期评测为了防止模型硬塞穿模，将目标孔壁 `solref` 设为极其刚硬的 `0.006 1`（等效刚度约为真实 SO-101 夹爪胶垫 `0.01` 的 2.8 倍、MuJoCo 默认 `0.02` 的 11 倍）且侧壁摩擦为 `0.50`，导致轻微碰沿即咬死/弹滑。

为避免“调软材料或加倒角作弊”，在 [scene.py](../examples/so101_insertion/scene.py#L58-L71) 中完成以下对齐并固死：
- **几何防作弊**：移除孔口倒角测试项，严格保留 **90° 直壁孔口**与 **0.7 mm 单边径向间隙**（螺栓半径 `5.0 mm`，孔半径 `5.7 mm`）。
- **材料与摩擦对齐**：孔壁采用贴合 3D 打印塑料与夹爪的 `solref="0.011 1"`、`friction="0.28 0.005 0.0001"` 与 `priority="1"`、`solmix="50"`。
- **穿壁过载护栏**：保留 `penetration_limit_mm = 3.0` 熔断判负线与 `0.200 s`（500 Hz 下连续 `100` 步）孔底就位判定。

在该诚实直壁基准上，冻结 `SmolVLA 20k` 在 30 个未见留出孔两轮实测分别取得 **`25/30`（83.3%，0 滑落）** 与 **`27/30`（90.0%，0 滑落）**，且 **`30/30` 全部进洞至 ~30 mm**。

---

## 三、三条支线实证排雷记录

在决定收手于干净 SFT 基准前，我们对三条常见“后训练提分路线”逐一进行了低成本诊断与证伪：

### 1. 在线 RL-Token（残差空间 & 全动作空间）

- **全动作 Stage A 纯 BC 对齐**：采用分位数窄归一化尺度后，在 30 留出孔上取得 `21/30`（0 滑落），验证了动作头自身拟合能力（[stage-a-bc.json](../experiments/smolvla-20k/stage-a-bc.json)）。
- **进入在线 RL（10,000 步）**：虽然固定监控孔从 `8/10` 涨至 `9/10`，但 30 个未见留出孔泛化成绩反而回落至 **`11/30`** 与 **`12/30`**（[online-rl.json](../experiments/smolvla-20k/online-rl.json)）。
- **核心阻因**：5 轴无力控视觉策略在稀疏 `0/1` 奖励下容易在原本对齐的好孔上叠加额外偏置（首次触孔跳变 P95 达 `26.1 mrad`），导致在未见孔上“救少丢多”。

### 2. DAgger 连续 IK 纠偏微调（25k）

- **现象**：从 `20k` 底座混采 DAgger 片段续训至 `25k`，端到端成功率反而从 `21–23/30` 跌至 **`18/30`（9 滑落）**（[dagger-eval.json](../experiments/smolvla-20k/dagger-eval.json)）。
- **冻结权重逐轴 Loss 审计**（[dagger-frozen-loss.json](../experiments/smolvla-20k/dagger-frozen-loss.json)）：不更新参数直接测 `20k` 权重在两批数据上的 Flow Loss，原 150 局示教为 **`0.000331`**，而新增老师数据高达 **`23.58`**；其中仅 **`wrist_roll`** 一轴 loss 达 **`114.74`**，占新增 6 轴总 loss 的 **`81.1%`**（5 轴 IK 在零空间内大幅拧转手腕）。
- **锁腕原地接管审计**（[dagger-stall-recenter.json](../experiments/smolvla-20k/dagger-stall-recenter.json)）：即使锁死 `wrist_roll` 并改为仅在孔内卡滞时接管，脚本老师在 20 个真实卡滞局中也仅救回 `4/20`，无法达到批量提升底座的门槛。

### 3. 末端力反馈“卸力停”门控（第 0 步零训练证伪）

在最后观察到 `25/30` 中剩余 5 局均已插进 `29.5–31.0 mm` 后，我们先用 **0 训练成本的第 0 步诊断**（[straight-solref011-run2-force-gate.json](../experiments/smolvla-20k/straight-solref011-run2-force-gate.json)）检验“收到力/位置滞后信号就卸力保持”的物理上限，结果直接证伪：

- **Oracle 上限测试 `0/14`**：分别在 `on_seated` 与 `on_bottom_load` 触发条件下测试 3 种零参数卸力保持（`latched_qpos` / `live_qpos` / `half_unload`），14 次实际触发全部未能撑满 `200 ms`（最长 `100 ms`），`0/14` 成功。
- **逐帧物理根因**：拆解 `02 / 05 / 13` 失败窗口发现，它们**并不是已经压实孔底顶死**，而是距离孔底仍有 `gap = 1.42–2.34 mm`（就位门槛 `≤ 2.0 mm`）时，**191 步附近落出训练示教时间窗**的 SmolVLA 轻微回抬，打断了 500 Hz 连续 `200 ms` 单帧清零计时；一旦中途撤掉向下推力，螺栓立刻回弹脱离就位区。
- **本体/力矩信号可分性不足**：对比“闭爪抓举与搬运”与“孔底与就位”两阶段前 5 轴 L2 范数，关节速度 AUROC `0.456`，`ctrl − qpos` AUROC `0.687`，模拟舵机力矩 `qfrc_actuator` AUROC 仅 **`0.729`**（且进就位包络瞬间如 `02` 号第 `186–188` 步力矩仅 `0.70–0.98 N·m`，低于空中搬运中位数 `1.66 N·m`）。

### 4. 孔底保持帧补采续训消融（v1 4 轴锁腕 vs v2 5 轴同源）与 SO-101 硬件边界

为了实证检验“第 191 帧早停剪掉孔底保持帧”能否通过补采干净数据继续提分，我们编写了 [smolvla_hold_finetune.py](../examples/so101_insertion/smolvla_hold_finetune.py) 对 `020000` 做了两轮严格对照续训：

#### (1) 采数阶段发现的两项物理细节
- **孔底专家 `wrist_roll` 空转**：若只关闭早停并延长 `wide_env.expert()`，5 轴 `bolt_ik()` 会在坐底后继续调用零空间，把 `wrist_roll` 拧至 `-1.53 rad`（冻结 `020000` 审计尾段 `wrist_roll` Loss 高达 `33.45`）。
- **铰链夹爪慢挤上滑**：保持末端 `tcp_z` 静态锁死在 `54.00 ± 0.03 mm` 时，SO-101 单边铰链两指硬爪会对光滑螺帽产生约 **`2 mm/s`** 的向上挤推分力；因此在 `settle_ticks = 4` 后静态保持 **`25` 帧**（总长 `216` 帧 > `chunk_size = 25`）刚好能在上滑超出 `2.0 mm` 前录出 `~0.98 s` 稳态窗口，v2 补采 `50` 条实现 **0 滑落**、平均稳态 `0.981 s`（最短 `0.782 s`）、最大峰值穿壁仅 **`0.622 mm`**（中位数 `0.452 mm`）。

#### (2) 冻结 `020000` 前/尾段拆分 6 轴 Flow Loss 体检

| 消融版本与切片 | 总 Loss | `shoulder_pan` | `shoulder_lift` | `elbow_flex` | `wrist_flex` | `wrist_roll` | `gripper` | 凭证 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原训练集 `original`（0–191 帧） | `0.0022` | `0.00057` | `0.00256` | `0.00320` | `0.00356` | `0.00300` | `0.00043` | [v2 audit](../experiments/smolvla-20k/hold-finetune-v2-audit.json) |
| 原验证集 `val_prefix_0_190` | `0.0303` | `0.00035` | `0.00233` | `0.00092` | `0.00050` | **`0.17707`** | `0.00035` | [v2 audit](../experiments/smolvla-20k/hold-finetune-v2-audit.json) |
| v1 `hold50_all`（下插段改 4 轴锁腕） | `0.7786` | `0.00265` | `0.09320` | `0.04555` | `0.01831` | **`4.51056`** | `0.00128` | [v1 audit](../experiments/smolvla-20k/hold-finetune-v1-audit.json) |
| **v2 `hold50_prefix_0_190`（5 轴同源前段）** | **`0.0362`** | `0.00075` | `0.00518` | `0.02544` | `0.01522` | **`0.17005`** | `0.00079` | [v2 audit](../experiments/smolvla-20k/hold-finetune-v2-audit.json) |
| v2 `hold50_tail_191_216`（新增静止保持） | `1.9055` | `0.06723` | `0.15372` | `0.28548` | `0.29975` | `10.61701` | `0.01000` | [v2 audit](../experiments/smolvla-20k/hold-finetune-v2-audit.json) |

#### (3) 两轮 30 孔直壁基准（`solref=0.011`, `friction=0.28`）续训对比

| 实验与检查点 | 配方 | 插入 | 滑落 | 穿透超限 / 超时 | 中位峰值穿壁 | 相比 `020000` 救回 / 退化 | 凭证 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `020000` 基线 | 原 150 条 SFT | **25/30** | **0** | 2 / 3 | **`0.94 mm`** | 失败 `[2, 5, 12, 13, 28]` | [run1](../experiments/smolvla-20k/straight-solref011-run1.json) |
| v1 `021000` / `022000` / `023000` | 4 轴锁腕，`lr=5e-6` | `14 → 1 → 22/30` | `0 / 3 / 0` | `6 / 2`（23k） | `1.68 mm`（23k） | 23k 全数救回 `[2,5,12,13,28]`，另退化 `8` 孔 | [v1 comp](../experiments/smolvla-20k/hold-finetune-v1-comparison.json) |
| v2 `020500` / `021000` | 5 轴同源前缀，`lr=3e-6` | `10 → 13/30` | **0 / 0** | `11/9` · `10/7` | `2.83 · 2.57 mm` | 新热身阶段先震散收敛态，再重新爬坡 | [v2 comp](../experiments/smolvla-20k/hold-finetune-v2-comparison.json) |
| **v2 `022000`** | **5 轴同源前缀，`lr=3e-6→3e-7`** | **24/30** | **0** | **3 / 3** | **`1.40 mm`** | **救回 `[2, 12, 13]`，另 4 孔 `[1, 9, 19, 22]` 擦线轮换** | [v2 comp](../experiments/smolvla-20k/hold-finetune-v2-comparison.json) |

#### (4) 归因结论：SO-101 硬件四重约束决定了 `24–27/30` 平台
1. **纯视觉定位误差 > 孔隙**：`224×224` 首次触孔中位误差 `1.20 mm` > 单边 `0.7 mm` 间隙，进洞 30 mm 后爪座遮挡洞口，亚毫米歪斜几乎不可分辨；
2. **无力觉传感器**：舵机负载估算 AUROC 仅 `0.729`，模型无法感知“哪一侧擦壁 / 是否坐实底板”；
3. **高刚度硬位置伺服**（`kp=998.22`）：偏 `1 mm` 不会被动让步顺滑进孔，而是直接往壁内硬顶出 `3 mm+` 穿壁超限；
4. **5 轴欠驱动 + 铰链爪**：缺少手腕侧向摆正自由度（不能原地侧倾扶正），加之铰链夹爪受力慢挤上滑，导致边际难例在 30 孔间此消彼长、无法通过继续 SFT 突破硬件平台。

因此，本项目明确保留 **`SmolVLA 20k + #5335 解析修复`（直壁基准 `25–27/30`、0 滑落）** 作为最轻柔、最高效的定版基线，不做仿真物理或硬件作弊修改；若面向 99% 工业装配需求，工程上的正确升级路径是切换至 7 轴带末端柔顺阻抗控制与腕部力觉的平台（如 Franka + OSC/Impedance）。
