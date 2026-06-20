# GRPO + Replay Buffer 数值失稳问题 — 配置对比与根因

## TL;DR

新版 `grpo_aime/exp4,5` 的训练效果显著差于旧版 `math_aime_8h100_replaybuffer` 的同类实验（val/score 0.13 vs 更好），并在 replay 阶段观测到 **importance ratio 极端值 1e9~1e19** 和 **grad_norm 偶发爆炸到 1e8**。

**根因单点**：`adv_estimator` 从 `"reinforce"` 改成了 `"grpo"`。REINFORCE 不计算 importance ratio，对 off-policy replay 天然稳定；GRPO 是 PPO 系目标函数，含 `ratio = exp(new_logp - old_logp)`，对 old policy 距离极度敏感。当 replay 采样到较老的 trajectory 且当前 policy 已漂离时，极端 token 的 `ratio` 数值溢出。

本该缓解这个问题的两个机制 **都没接上**：
1. `importance_sampling_correction: true` 生成 `importance_weights`，但 pipeline 的 `actor_train.train_step` 侧没消费这个字段（断点 3，前面调研过）
2. `init_kl_coef: 0`，没有 KL penalty 约束 policy drift

两个机制缺席下，GRPO + replay = 纯爆炸。而**旧版 REINFORCE 配置下同样两个开关也是 off，但因为 REINFORCE 根本不算 ratio，所以稳定**。

---

## 配置对比

路径：
- 旧（效果好）：`experiments/math_aime_8h100_replaybuffer/aime_traj_reward_fresh_8gpu.yaml`
- 新（数值失稳）：`experiments/grpo_aime/exp5_grpo_replay_fresh_per.yaml`

| 配置 | 旧版 | 新版 | 关键性 |
|---|---|---|---|
| **`adv_estimator`** | **`"reinforce"`** | **`"grpo"`** | ⚠️ **决定性** |
| `rollout_batch_size` | 512 | 512 | 相同 |
| `advantage_clip` | 0.2 | 0.2 | 相同 |
| `init_kl_coef` | 0 | 0 | 旧版 OK，新版致命 |
| `entropy_loss_coef` | 0 | 0 | 相同 |
| `ppo_epochs` | 1 | 1 | 相同 |
| `whiten_advantages` | true | true | 相同 |
| `group_size` (train/val) | 4 / 1 | 4 / 1 | 相同 |
| `train_steps_per_env_step` | 2 | 1 | 旧版 off-policy 更重，但 REINFORCE 不怕 |
| `importance_sampling_correction` | **false** | **true** | 新版开启但 pipeline 不消费 → 半空转 |
| `priority_function` | reward_fresh | reward_fresh | 相同 |
| `priority_exponent` | 0.6 | 0.6 | 相同 |
| `enable_age_decay` | true | true | 相同 |

---

## 为什么 REINFORCE 下 replay 稳定、GRPO 下不稳定

### REINFORCE

Loss ≈ `-advantage * log π(a|s)`。训练时**不计算 new/old policy 比率**。

对 replay buffer 里采样出的老样本：
- 直接用当前 policy 的 `log π` 做 policy gradient
- 老样本的 "old policy" 从未进入 loss
- 没有 ratio → 没有 ratio 爆炸 → grad_norm 稳定
- PER 的 priority 加权只决定**哪些样本被重训**，不放大单 token 梯度

### GRPO（PPO 系）

Loss 含 `ratio = exp(new_logp - old_logp)`：
- Buffer 里采样的老样本 → `old_logp` 是 rollout 时的 policy
- 经过 N 个训练步后 `new_logp` 漂远 → 某些 token 的 `ratio` 数值爆炸
- PPO clip 按 mean 层面裁（`clipfrac ≈ 0.5%`），无法拦住**单 token 极端 ratio**
- 极端 token 的 `ratio * advantage` 贡献极大梯度 → `grad_norm` 爆
- 梯度反传后 policy 进一步漂 → 下一步 ratio 更极端 → 正反馈失稳

---

## 运行时数据佐证

三个 run 的 driver log 提取指标（`training_records/analysis_report.txt`）：

### Importance ratio 极端值（`ratio_max@max`）

| 指标 | exp4 (GRPO, reward) | exp5 (GRPO, reward_fresh) | vlm (GRPO) | 正常范围 |
|---|---|---|---|---|
| `actor/ratio_max@max` median | 24 | 117 | 1.68 | < 10 |
| `actor/ratio_max@max` max | **1.3e9** | **8.9e19** | 19.18 | < 100 报警 |
| `replay/actor/ratio_max@max` max | **5.2e9** | **2.1e12** | 5.4e4 | < 100 |

### Gradient norm（偶发爆炸）

| 指标 | exp4 | exp5 | vlm | 正常 |
|---|---|---|---|---|
| `actor_train/grad_norm` median / max | 0.09 / 1.15e5 | 0.001 / 78 | 3.48 / 35.6 | median/max 都 < 10 |
| `replay/actor_train/grad_norm` median / max | 0.13 / **1.43e8** | 0.68 / **2.01e7** | 4.52 / 911 | median/max 都 < 10 |

median 正常但 max 爆 → 偶发性数值爆炸（极端 ratio 引发）。

### 任务分数（val/score/mean）

| 实验 | 末期 val score | 最高 val score |
|---|---|---|
| exp4 (reward PER) | 0.129 | 0.176 |
| exp5 (reward_fresh PER + age_decay) | 0.051 | 0.082 |
| vlm_fresh_per | 0.768 | 0.768 |

exp5 （对应旧 `aime_traj_reward_fresh_8gpu` 配置）的 val 分数显著低于 exp4（0.05 vs 0.13）。问题**不是** "reward_fresh 不好"，是 GRPO 在两个缓解机制缺席下被 off-policy replay 反复冲击。

### critic/kl（独立佐证）

- exp4 / exp5：`critic/kl` 和 `critic/kl_coef` 全程为 0（KL penalty 彻底关闭）
- vlm：`critic/kl` median 0.157（有 KL 约束，稳定性相对好；见 `vlm/ratio_max` 远低于 exp4/exp5）

---

## 为什么 vlm 的 ratio 比 exp4/exp5 温和

vlm 也是 `adv_estimator: "grpo"` + `init_kl_coef: 0`，按理应也爆炸。但观测到的 ratio_max 最大 5.4e4（比 exp5 的 2.1e12 小 8 个数量级），grad_norm max 只有 911。

可能原因（待确认）：
- vlm 的 response 长度更短（FrozenLake 每轮回复几十 token vs AIME 动辄几百~上千），极端 token 积累机会更少
- vlm 只跑了 100 steps（vs 400），少 4 倍爆炸机会
- VLM 模型参数较少，policy drift 速率不同

但 vlm 的 `critic/entropy` 从 0.61 坍到 0.016 —— 熵崩塌意味着快速过拟合到单一策略，泛化性存疑。这**不是** ratio 爆炸，是另一个问题（entropy_loss_coef=0 导致无 entropy bonus）。

---

## 修复方案（按优先级）

### 方案 A — 最简单，复刻旧实现稳定性

把 `adv_estimator` 改回 `"reinforce"`。其它保持不变。
- 优点：零侵入，确定稳定
- 缺点：失去 GRPO 的组内 advantage 归一化特性

### 方案 B — 保留 GRPO，补齐两个缓解机制

1. **开 KL penalty**：`init_kl_coef: 0.05 ~ 0.1`（配合 `target_kl` 做自适应），约束 policy drift
2. **真正接通 IS correction**：修改 `actor_train.train_step` 的 PPO loss，把 `replay_batch.batch["importance_weights"]` 乘到 pg loss 项里
3. **收紧 PPO clip**：`ratio_clip_high: 1.2`（若 ROLL 支持），避免极端 token 漏过

改动面较大（需要改 worker/strategy 侧 loss 代码），见下一节"断点 3 修复路径"。

### 方案 C — 混合方案（推荐）

先用 A 复刻旧稳定性跑出对照（确认 PER 闭环和 VLM 多模态修复生效），再逐步引入 B 的 KL penalty 和 IS correction 验证 GRPO 能否稳定工作。

---

## 断点 3 修复路径（方案 B 第 2 项）

目标：让 `replay_batch.batch["importance_weights"]` 真正作用于 loss。

1. `replay_buffer/{trajectory,group,step}_buffer.py` 已经在 `sample_for_training` 里生成了 `importance_weights`（当 `compute_importance_weights=True`）
2. `agentic_pipeline.py` 传 `replay_batch` 给 `actor_train.train_step`
3. **缺失**：`actor_train` 的 loss 实现（可能在 `roll/pipeline/agentic/agentic_actor_worker.py` 或 strategy 层）需要：
   - 检测 `replay_batch.batch` 里是否有 `"importance_weights"` 字段
   - 若有，把它 broadcast 到 per-token 级别并乘到 `pg_loss` 项上
   - 典型写法：`weighted_pg_loss = pg_loss_per_token * importance_weights[:, None]`

这个改动跨 pipeline / worker / strategy 三层，动之前建议先读 PPO loss 的 per-token reduction 逻辑，确认加权层级正确（sample-level 权重乘到每个 token 的 loss 项上，然后 mean 出最终 loss）。

---

## 相关文件

- 对比 yaml：
  - `experiments/math_aime_8h100_replaybuffer/aime_traj_reward_fresh_8gpu.yaml`（旧，REINFORCE）
  - `experiments/grpo_aime/exp5_grpo_replay_fresh_per.yaml`（新，GRPO）
- 运行时证据：`training_records/analysis_report.txt`（本地 driver log 提取）
- 断点 3 涉及 pipeline 层：`roll/pipeline/agentic/agentic_pipeline.py` PHASE 15
- 断点 3 涉及 worker/strategy 层：`roll/pipeline/agentic/agentic_actor_worker.py`、`roll/distributed/strategy/*.py` 的 PPO loss 实现

---

## 后续

- 如果你只想快速得到可用结果 → 方案 A（换 REINFORCE）跑一轮对照，同时可用本次已完成的 PER 闭环 + VLM 多模态修复
- 如果要系统性让 GRPO + replay 稳定 → 方案 B 需要一次性修 KL + IS correction 两处，建议放在下一个迭代
