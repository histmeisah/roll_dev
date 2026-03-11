# Step Level 训练问题分析报告

> 日期: 2026-01-25
> 实验环境: FrozenLake + StepEnvManager + 2×A100 40GB
> 模型: Qwen2.5-0.5B-Instruct

---

## 1. 实验概述

我们测试了三种 Step Level 的训练配置:

| 实验名称 | 配置 | Replay Buffer | 优先级函数 |
|---------|------|---------------|-----------|
| step_baseline | 无Replay | - | - |
| step_per | 标准PER | 开启 | reward (priority=\|reward\|) |
| step_reward_fresh | Reward-Fresh | 开启 | reward×freshness |

对应日志目录:
- `20260123_164201/` - step_baseline
- `20260124_155601/` - step_per
- `20260124_160415/` - step_reward_fresh

---

## 2. 核心问题发现

### 2.1 关键指标对比

| 指标 | step_baseline | step_per | step_reward_fresh |
|------|---------------|----------|-------------------|
| **最终成功率** | 0% | 0% | **11.7%** |
| **action_is_valid** | 0% (崩溃) | 0% | **99.6%** |
| **old_log_prob/mean** | **NaN** | **-31.6** | -0.31 (正常) |
| **entropy/mean** | **NaN** | 0.27 | 0.39 |
| **importance_weight/max** | NaN | **1.4×10^28** | 2387 |
| **importance_weight/mean** | - | 1.3×10^27 | 6.4 |
| **sample_importance_weight/max** | - | 1.4×10^14 | 5.9 |

### 2.2 问题严重性排序

1. **step_baseline**: 完全崩溃 (产生NaN)
2. **step_per**: 严重不稳定 (importance weight 爆炸到 10^28)
3. **step_reward_fresh**: 相对稳定，保持了11.7%成功率

---

## 3. 根本原因分析

### 3.1 模式崩溃 (Mode Collapse)

所有问题的根源是 **训练过程中模型发生模式崩溃**:

```
正常状态                    崩溃状态
────────────────────────────────────────────────────
输出: "<answer>Right</answer>"  →  输出: "!!!!!!!!!!!!!..."
log_prob: -0.31                 →  log_prob: NaN 或 极负值
action_is_valid: ~1.0           →  action_is_valid: 0.0
```

**日志证据** (step_baseline 最后阶段):
```json
{
  "response": "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!...",
  "episode_score": 0,
  "penalty": -0.15
}
```

### 3.2 各实验的崩溃机制

#### step_baseline: NaN 传播

```
模型输出退化 (全是 "!" 等无效字符)
    ↓
softmax 计算时概率分布异常
    ↓
log_prob 计算产生 NaN
    ↓
ValueError: autodetected range of [nan, nan] is not finite
    ↓
训练无法继续
```

#### step_per: Importance Weight 爆炸

```
behavior_policy (数据生成时) 的 log_prob ≈ -31.6
current_policy (当前) 的 log_prob ≈ -0.3

importance_weight = exp(current_log_prob - behavior_log_prob)
                  = exp(-0.3 - (-31.6))
                  = exp(31.3)
                  ≈ 4×10^13

对于多token序列,乘积后达到 10^28
```

**标准PER的问题**: 优先采样高reward的样本,但这些样本往往policy drift最大,形成恶性循环。

#### step_reward_fresh: 相对稳定

```
priority = (|reward| + ε) × exp(-age / 500)

age_decay 机制:
- 旧数据自动降权
- 采样更倾向于新鲜数据
- 策略漂移较小
- importance_weight 保持在合理范围
```

---

## 4. 技术细节

### 4.1 Importance Weight 计算

位置: `roll/pipeline/agentic/offpolicy_monitor.py:122-123`

```python
log_ratio = valid_current - valid_behavior
ratio = log_ratio.exp()  # This is the importance weight
```

当 `valid_behavior` 极负 (如 -31.6) 时:
- `log_ratio` 会很大 (如 31.3)
- `ratio = exp(31.3)` 会爆炸

### 4.2 为什么 old_log_prob 会变得极负?

1. **模型输出退化**: 模型不再输出有效格式的答案
2. **概率极低**: 当前策略给旧数据中的token极低概率
3. **累积效应**: 序列越长,累积的log_prob越负

### 4.3 PER 的放大效应

标准PER按 `priority = |reward|` 采样:
- 成功episode有高reward → 高优先级
- 但成功episode往往是早期生成的 → 策略已漂移
- 导致采样的数据与当前策略差异最大
- **形成恶性循环**

---

## 5. 配置对比

### 5.1 关键配置差异

```yaml
# step_baseline
replay:
  enabled: false

# step_per
replay:
  enabled: true
  priority_function: "reward"      # 标准PER
  priority_exponent: 0.6
  # 无 age_decay

# step_reward_fresh
replay:
  enabled: true
  priority_function: "reward_fresh"  # 我们的扩展
  priority_exponent: 0.6
  age_decay: 500.0                   # 关键: 新鲜度衰减
```

### 5.2 共同配置 (可能有问题)

```yaml
# 这些配置对所有实验都一样
advantage_clip: 20           # 过大?
init_kl_coef: 0.0            # 无KL约束初始化
kl_loss_coef: 0.01           # KL正则化较弱
entropy_loss_coef: 0         # 无entropy bonus!
learning_rate: 1.0e-6        # 可能过小
```

---

## 6. 问题总结

### 6.1 已确认的问题

| 问题 | 严重性 | 影响 |
|------|--------|------|
| 模式崩溃 | 严重 | 模型输出退化,成功率归零 |
| NaN传播 | 严重 | 训练完全停止 |
| IW爆炸 | 严重 | 梯度不稳定,训练崩溃 |
| PER放大效应 | 中等 | 加速模型退化 |
| 无entropy bonus | 中等 | 模型容易过拟合到特定模式 |

### 6.2 可能的问题

| 问题 | 待验证 |
|------|--------|
| advantage_clip=20 是否过大 | 需要实验验证 |
| 学习率1e-6是否过小 | 需要实验验证 |
| StepEnvManager的reward shaping | 需要检查step-level reward分配 |

---

## 7. 建议修复措施

### 7.1 紧急修复 (短期)

```yaml
# 1. 添加 importance weight clipping
replay:
  importance_weight_clip: 10.0  # 截断极端值

# 2. 添加 entropy bonus 防止模式崩溃
entropy_loss_coef: 0.01  # 或更高

# 3. 增强 KL 约束
init_kl_coef: 0.1
kl_loss_coef: 0.05
```

### 7.2 架构改进 (中期)

1. **使用 V-trace 替代裸 PER**
```yaml
adv_estimator: "vtrace"
vtrace:
  rho_bar: 1.5   # 截断 importance ratio
  c_bar: 1.0     # 截断 trace coefficient
```

2. **添加 Importance Weight 监控和早停**
```python
if importance_weight_max > 1000:
    logger.warning("IW exploding, clearing old buffer entries")
    replay_buffer.clear_old_entries(keep_ratio=0.5)
```

3. **混合训练策略**
```yaml
replay:
  fresh_data_ratio: 0.5  # 50% fresh + 50% replay
```

### 7.3 长期优化

1. **动态调整 age_decay**
   - 根据 IW 统计自动调整
   - IW过大时减小age_decay

2. **自适应优先级**
   - 不只用reward,还考虑policy drift
   - `priority = |reward| × (1 / (1 + policy_drift))`

3. **模型健康监控**
   - 监控 action_is_valid 趋势
   - 低于阈值时触发恢复机制

---

## 8. 下一步实验计划

### 8.1 优先级高

1. [ ] 运行 `step_reward_fresh_vtrace` 配置
2. [ ] 添加 `entropy_loss_coef: 0.01`
3. [ ] 测试 `importance_weight_clip`

### 8.2 优先级中

4. [ ] 对比不同 `age_decay` 值 (200, 500, 1000)
5. [ ] 测试混合训练 (fresh + replay)
6. [ ] 分析 StepEnvManager 的 reward shaping

### 8.3 优先级低

7. [ ] 尝试不同学习率 (5e-7, 1e-6, 5e-6)
8. [ ] 测试不同 `advantage_clip` 值

---

## 9. 附录

### 9.1 日志关键片段

**step_baseline 崩溃时刻**:
```
[2026-01-24 13:11:32] ValueError: autodetected range of [nan, nan] is not finite
```

**step_per IW 爆炸**:
```
[HISTOGRAM] Created replay_iw histogram:
  shape=(260,),
  min=0.000,
  max=14960311804961687795398606848.000,  # 1.4×10^28
  mean=1310672894836870422745579520.000
```

**step_reward_fresh 正常运行**:
```
[HISTOGRAM] Created replay_iw histogram:
  shape=(1541,),
  min=0.000,
  max=2387.721,  # 正常范围
  mean=6.464
```

### 9.2 参考文献

- Schaul et al., "Prioritized Experience Replay", ICLR 2016
- Espeholt et al., "IMPALA: Importance Weighted Actor-Learner Architectures", ICML 2018
- Fedus et al., "Revisiting Fundamentals of Experience Replay", ICML 2020
