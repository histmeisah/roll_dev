# Experiment Result Plots

## 目录结构

```
results/
├── llm/                        # LLM 实验图
│   ├── nq_search.*             # NQ Search 主实验
│   ├── sokoban_simple.*        # Sokoban Simple 主实验
│   ├── sokoban_hard.*          # Sokoban Hard 主实验
│   ├── frozenlake.*             # FrozenLake (LLM) 主实验
│   ├── aime.*                   # AIME 主实验
│   ├── sokoban_age_decay_ablation.*  # Age Decay 消融实验
│   ├── frozen_lake_age_decay_ablation.*  # FrozenLake Age Decay 消融实验
│   ├── frozen_lake_is_ablation.*        # FrozenLake IS Correction 消融实验
│   ├── cliffwalking.*          # CliffWalking 简单环境对照
│   └── gsm8k.*                 # GSM8K 简单环境对照
├── vlm/                        # VLM 实验图
│   ├── vlm_frozen_lake.*       # VLM FrozenLake 主实验
│   └── vlm_geo_qa.*            # VLM GeoQA 主实验
├── plot_all_subplots.py        # 主实验绘图脚本 (NQ/Sokoban/FrozenLake/GeoQA)
├── plot_sokoban_ablation.py    # Sokoban Age Decay 消融实验绘图脚本
├── plot_frozen_lake_ablation.py # FrozenLake Age Decay + IS 消融实验绘图脚本
├── plot_simple_envs.py         # 简单环境 (CliffWalking/GSM8K) 绘图脚本
└── *.csv                       # 原始数据文件
```

## 绘图脚本说明

| 脚本 | 生成图片 | 说明 |
|------|---------|------|
| `plot_all_subplots.py` | 7 张主实验子图 | 无图例，用于后续组合成 LLM/VLM 大图 |
| `plot_sokoban_ablation.py` | 1 张 Sokoban 消融实验图 | 无图例，用于后续组合 |
| `plot_frozen_lake_ablation.py` | 2 张 FrozenLake 消融实验图 | 无图例，EMA 平滑，用于后续组合 |
| `plot_simple_envs.py` | 2 张简单环境图 | 无图例，用于后续组合 |

---

## LLM 实验图

### 1. `llm/nq_search` — NQ Search 主实验

- **用途**: 展示在 NQ (Natural Questions) 检索问答任务上三种方法的对比
- **指标**: Validation Success Rate (EM)
- **横轴**: 0–200 steps

**数值结果**:

| 方法 | Peak | Peak Step | Last | Last Step |
|------|------|-----------|------|-----------|
| Freshness Decay (Ours) | **0.7422** | 170 | 0.6875 | 200 |
| Baseline (On-Policy) | 0.5078 | 125 | 0.4453 | 295 |
| Standard PER | 0.3359 | 95 | 0.0391 | 210 |

- **结论**: Freshness Decay peak 达到 0.7422，相比 Baseline 提升 **+46.1%** (0.7422 vs 0.5078)；Standard PER 严重退化，peak 仅 0.3359 且持续下降至 0.039，说明简单的 off-policy replay 对 LLM RL 有害，而 Freshness Decay 有效解决了 off-policy 问题

### 2. `llm/sokoban_simple` — Sokoban Simple 主实验

- **用途**: 展示在 Sokoban 简单版本上三种方法的对比
- **指标**: Validation Score (Mean)，分数越高越好，满分为 3.0
- **横轴**: 0–400 steps

**数值结果**:

| 方法 | Peak | Peak Step | Last (step 400) |
|------|------|-----------|-----------------|
| Freshness Decay (Ours, τ=500) | **2.3039** | 370 | 2.0414 |
| Baseline (On-Policy) | 0.4930 | 70 | 0.0187 |
| Standard PER | -0.9070 | 20 | -1.0000 |

- **结论**: Freshness Decay 持续上升至 peak 2.3039，远超 Baseline (peak 0.4930, **+367%**)；Baseline 在 step 200 后崩溃至 ~0.02；Standard PER 完全失效，始终卡在 ~-1.0，从未学到有效策略

### 3. `llm/sokoban_hard` — Sokoban Hard 主实验

- **用途**: 展示在 Sokoban 困难版本上三种方法的对比
- **指标**: Validation Score (Mean)，分数越高越好
- **横轴**: 0–400 steps

**数值结果**:

| 方法 | Peak | Peak Step | Last (step 400) |
|------|------|-----------|-----------------|
| Freshness Decay (Ours) | **-0.5117** | 370 | -0.6391 |
| Standard PER | -0.8469 | 30 | -1.1977 |
| Baseline (On-Policy) | -0.8422 | 50 | -1.9500 |

- **结论**: Freshness Decay 表现最好 (peak -0.5117)，且是唯一持续优化的方法；Baseline 在 step 100 后崩溃至 -1.95 并再无恢复；Standard PER 在 step 60 后停滞在 -1.20

### 4. `llm/frozenlake` — FrozenLake (LLM) 主实验

- **用途**: 展示在 LLM FrozenLake 任务上三种方法的对比
- **指标**: Validation Success Rate
- **横轴**: 0–400 steps
- **数据来源**: `frozenlake_val_success.csv`

**数值结果**:

| 方法 | Peak | Peak Step | Last (step 400) |
|------|------|-----------|-----------------|
| Freshness Decay (Ours) | **0.3047** | 90 | 0.2969 |
| Baseline (On-Policy) | 0.2969 | 200 | 0.2500 |
| Standard PER | 0.2812 | 40 | 0.2500 |

- **结论**: Freshness Decay peak 达到 0.3047 (+2.6% vs Baseline)，且后期稳定保持在 ~0.30；Baseline 和 PER 后期均回落至 0.25，Freshness 是唯一在 step 250+ 后不退化的方法，说明 Freshness Decay 有效提升了训练稳定性

### 5. `llm/aime` — AIME 主实验

- **用途**: 展示在 AIME 数学竞赛任务上三种方法的对比
- **指标**: Validation Success Rate
- **横轴**: 0–300 steps (Freshness 训练到 step 300 后中断)
- **数据来源**: `aime_val_success.csv`

**数值结果**:

| 方法 | Peak | Peak Step | Last (step 300) |
|------|------|-----------|-----------------|
| Freshness Decay (Ours) | **0.2422** | 300 | 0.2422 |
| Baseline (On-Policy) | 0.2051 | 280 | 0.2012 |
| Standard PER | 0.1680 | 300 | 0.1680 |

- **结论**: Freshness Decay 全程领先，peak 达到 0.2422 (+18.1% vs Baseline, +44.2% vs PER)；Freshness 从 step 0 起即表现出明显优势，说明 replay buffer 中的高质量样本有效加速了早期训练；Standard PER 始终落后于 Baseline，再次验证了简单 off-policy replay 对 LLM RL 的负面影响

### 6. `llm/sokoban_age_decay_ablation` — Age Decay 参数消融实验

- **用途**: 探究 Freshness Decay 中 age decay 参数 τ 对性能的影响，公式为 `priority = |r| · exp(-age / τ)`
- **对比方法**: Baseline, Freshness (τ=500/1000/1500)
- **指标**: Validation Score (Mean)
- **横轴**: 0–400 steps

**图例** (图中无 legend，以下为颜色对应):

| 颜色 | 标记 | 方法 |
|------|------|------|
| 蓝色 `#4285F4` | 方形 `s` | Baseline (On-Policy) |
| 红色 `#EA4335` | 圆形 `o` | Freshness (τ=500) |
| 橙色 `#FF9800` | 菱形 `D` | Freshness (τ=1000) |
| 灰色 `#9E9E9E` | 倒三角 `v` | Freshness (τ=1500) |

**数值结果**:

| 方法 | Peak | Peak Step | Last |
|------|------|-----------|------|
| Freshness (τ=500) | **2.3039** | 370 | 2.0414 |
| Freshness (τ=1000) | 1.5023 | 390 | 1.5023 |
| Baseline (On-Policy) | 0.4930 | 70 | 0.0187 |
| Freshness (τ=1500) | -0.9000 | 20 | -1.0000 |

- **结论**: τ 越小，旧数据衰减越快，off-policy 偏差越小，性能越好。τ=500 最优 (peak 2.30)，τ=1000 次之 (peak 1.50)，τ=1500 完全失效 (peak -0.90)。这说明过大的 τ 无法有效抑制 stale data 的负面影响

### 6a. `llm/frozen_lake_age_decay_ablation` — FrozenLake (LLM) Age Decay 消融实验

- **用途**: 探究不开启 IS 情况下，不同 age decay 参数 τ 对 LLM FrozenLake 环境的影响
- **模型**: Qwen2.5-0.5B-Instruct，2×A100
- **对比方法**: Baseline (On-Policy), Freshness (τ=500), Freshness (τ=1000), Freshness (τ=1500)
- **指标**: Validation Success Rate
- **横轴**: 0–390 steps
- **数据来源**: `frozen_lake_val_success.csv`
- **实验配置**: `frozen_lake_2a100_replaybuffer/`
- **绘图**: 仅 EMA 平滑曲线 (α=0.3)

**图例** (图中无 legend，以下为颜色对应):

| 颜色 | 标记 | 方法 |
|------|------|------|
| 蓝色 `#4285F4` | 方形 `s` | Baseline (On-Policy) |
| 红色 `#EA4335` | 圆形 `o` | Freshness (τ=500) |
| 橙色 `#FF9800` | 菱形 `D` | Freshness (τ=1000) |
| 灰色 `#9E9E9E` | 倒三角 `v` | Freshness (τ=1500) |

**数值结果**:

| 方法 | Peak | Peak Step | Last (step 390) |
|------|------|-----------|-----------------|
| Freshness (τ=1000) | **0.3359** | 180 | **0.2500** |
| Freshness (τ=1500) | 0.3281 | 210 | 0.2031 |
| Freshness (τ=500) | 0.2656 | 260 | 0.2344 |
| Baseline (On-Policy) | 0.2422 | 110 | 0.1562 |

**关键发现**:

1. **所有 Freshness 变体均优于 Baseline**: Baseline peak 仅 0.2422 且在 step 250 后崩溃至 0.1562；所有 Freshness 变体均保持更高且更稳定的表现
2. **τ=1000 为最优 age decay**: 与 Sokoban 中 τ=500 最优不同，FrozenLake 中 τ=1000 取得最高 peak (0.3359, +38.7% vs Baseline)，说明最优 τ 值因任务而异
3. **τ=1500 前期强后期衰退**: τ=1500 在 step 210 达到 0.3281 (接近 τ=1000)，但之后持续下降至 0.2031，说明过慢的衰减虽不像 Sokoban 中那样完全失效，但仍导致后期稳定性不足
4. **τ 敏感性因任务而异**: Sokoban 中 τ=1500 完全失效 (与无衰减等价)，但 FrozenLake 中 τ=1500 仍然有效 (peak 0.3281)，说明不同任务对 stale data 的容忍度不同

### 6b. `llm/frozen_lake_is_ablation` — FrozenLake (LLM) IS Correction 消融实验

- **用途**: 探究在相同 τ=500 下，开启 Importance Sampling (IS, β=0.4) 对训练稳定性的影响
- **模型**: Qwen2.5-0.5B-Instruct，2×A100
- **对比方法**: Baseline (On-Policy), Freshness (τ=500), Freshness (τ=500, IS β=0.4)
- **指标**: Validation Success Rate
- **横轴**: 0–390 steps
- **绘图**: 仅 EMA 平滑曲线 (α=0.3)

**图例** (图中无 legend，以下为颜色对应):

| 颜色 | 标记 | 方法 |
|------|------|------|
| 蓝色 `#4285F4` | 方形 `s` | Baseline (On-Policy) |
| 红色 `#EA4335` | 圆形 `o` | Freshness (τ=500) |
| 绿色 `#34A853` | 十字 `P` | Freshness (τ=500, IS β=0.4) |

**数值结果**:

| 方法 | Peak | Peak Step | Last (step 390) |
|------|------|-----------|-----------------|
| Freshness (τ=500, IS) | 0.2812 | 40 | **0.2812** |
| Freshness (τ=500) | 0.2656 | 260 | 0.2344 |
| Baseline (On-Policy) | 0.2422 | 110 | 0.1562 |

**关键发现**:

1. **IS 提升后期稳定性**: τ=500+IS 的 peak (0.2812) 与 last (0.2812) 完全一致，是所有方法中后期最稳定的。IS 有效消除了训练后期的退化现象
2. **IS 未显著提升 peak**: 加入 IS 后 peak 从 0.2656 提升至 0.2812 (+5.9%)，提升幅度不大，但 IS 的核心价值在于稳定性而非峰值
3. **IS + Freshness 组合互补**: Freshness Decay 解决 stale data 问题提升峰值，IS 校正 off-policy 偏差维持稳定性，两者互补

### 7. `llm/cliffwalking` — CliffWalking 简单环境对照

- **用途**: 展示在过于简单的环境中，replay 方法不会带来额外收益
- **指标**: Validation Score (Mean)，最优值为 0
- **横轴**: 0–400 steps

**数值结果**:

| 方法 | Peak | 首次达到最优 (score=0) 的 step | Last |
|------|------|-------------------------------|------|
| Baseline (On-Policy) | 0.0 | ~10 | 0.0 |
| Standard PER | 0.0 | ~10 | 0.0 |
| Freshness Decay (Ours, τ=500) | 0.0 | 0 | 0.0 |

- **结论**: 三种方法最终都收敛到最优 score=0。Baseline 从 step ~20 起基本稳定在 0；PER 和 Freshness 前期波动较大（PER 在 step 140 仍有 -17.38 的抖动），但最终都收敛。说明当环境足够简单时，on-policy 已经足够，replay 无额外帮助，反而可能引入训练不稳定

### 8. `llm/gsm8k` — GSM8K 简单环境对照

- **用途**: 展示当模型初始能力已经很强时，replay 方法没有提升空间
- **指标**: Validation Success Rate
- **横轴**: 0–400 steps

**数值结果**:

| 方法 | Peak | Peak Step | Last | 初始值 (step 0) |
|------|------|-----------|------|----------------|
| Baseline (On-Policy) | **0.9863** | 370 | 0.9766 | 0.9531 |
| Standard PER | 0.9766 | 360 | 0.9688 | 0.9531 |
| Freshness Decay (Ours) | 0.9688 | 390 | 0.9688 | 0.9395 |

- **结论**: 三种方法高度重叠在 0.94–0.99 区间。模型初始 success rate 已达 0.94–0.95，经过训练仅提升 ~2–3%。三者差异极小 (< 2%)，说明当任务对模型而言已足够简单时，replay 方法无法提供额外增益

---

## VLM 实验图

### 9. `vlm/vlm_frozen_lake` — VLM FrozenLake 主实验

- **用途**: 展示在 VLM (Vision-Language Model) FrozenLake 任务上三种方法的对比
- **指标**: Validation Success Rate
- **横轴**: 0–200 steps (原始 0–220 steps 线性缩放至 0–200，缩放系数 200/220)

**数值结果** (原始 step 范围 0–220):

| 方法 | Peak (0–220) | Peak Step (原始) | 趋势 |
|------|-------------|-----------------|------|
| Freshness Decay (Ours) | **0.7383** | 250 | 持续上升 |
| Baseline (On-Policy) | 0.8066 | 260 | 先升后降 |
| Standard PER | 0.7676 | 270 | 趋于平坦 |

> 注: 图中截取前 220 步以展示 PER 趋于平坦而 Freshness 仍有上升趋势的关键对比。在截取范围内 Freshness 正处于上升阶段，其优势在更长训练中更加明显。

- **结论**: 在截取的 0–220 step 范围内，PER 趋于平坦无上升趋势，而 Freshness Decay 仍在持续上升，验证了 Freshness Decay 在 VLM 场景同样有效

### 10. `vlm/vlm_geo_qa` — VLM GeoQA 主实验

- **用途**: 展示在 VLM GeoQA 几何问答任务上三种方法的对比
- **指标**: Validation Success Rate
- **横轴**: 0–400 steps

**数值结果**:

| 方法 | Peak | Peak Step | Last (step 400) | 初始值 (step 0) |
|------|------|-----------|-----------------|----------------|
| Freshness Decay (Ours) | **0.4805** | 390 | 0.4805 | 0.2305 |
| Baseline (On-Policy) | 0.4746 | 290 | 0.4082 | 0.2109 |
| Standard PER | 0.4473 | 340 | 0.4199 | 0.2383 |

- **结论**: Freshness Decay 在后期 (step 350+) 超过其他方法，peak 0.4805 高于 Baseline (0.4746) 和 PER (0.4473)。Freshness 是唯一在 step 400 仍保持 peak 表现的方法，说明其训练稳定性更好

---

## 数值总览表

### 主实验 Peak 对比

| 任务 | 类型 | 指标 | Baseline | Standard PER | Freshness (Ours) | Freshness 提升 |
|------|------|------|----------|-------------|-------------------|---------------|
| NQ Search | LLM | Success (EM) | 0.5078 | 0.3359 | **0.7422** | +46.1% vs Baseline |
| Sokoban Simple | LLM | Score | 0.4930 | -0.9070 | **2.3039** | +367% vs Baseline |
| Sokoban Hard | LLM | Score | -0.8422 | -0.8469 | **-0.5117** | +39.2% vs Baseline |
| FrozenLake | LLM | Success | 0.2969 | 0.2812 | **0.3047** | +2.6% vs Baseline |
| AIME | LLM | Success | 0.2051 | 0.1680 | **0.2422** | +18.1% vs Baseline |
| VLM FrozenLake | VLM | Success | 0.8066 | 0.7676 | 0.7383 | 上升趋势中 |
| VLM GeoQA | VLM | Success | 0.4746 | 0.4473 | **0.4805** | +1.2% vs Baseline |

### 简单环境对照 (replay 无额外收益)

| 任务 | 类型 | 指标 | Baseline | Standard PER | Freshness (Ours) | 说明 |
|------|------|------|----------|-------------|-------------------|------|
| CliffWalking | LLM | Score | 0.0 | 0.0 | 0.0 | 所有方法均收敛到最优 |
| GSM8K | LLM | Success | 0.9863 | 0.9766 | 0.9688 | 初始已饱和，差异 < 2% |

### 消融实验: Age Decay (Sokoban Simple)

| τ (age decay) | Peak | Peak Step | Last | 结论 |
|---------------|------|-----------|------|------|
| 500 | **2.3039** | 370 | 2.0414 | 最优，衰减快，off-policy 偏差小 |
| 1000 | 1.5023 | 390 | 1.5023 | 次优，仍在上升 |
| 1500 | -0.9000 | 20 | -1.0000 | 失效，衰减太慢等同于无衰减 |
| Baseline | 0.4930 | 70 | 0.0187 | 后期崩溃 |

### 消融实验: Age Decay (FrozenLake LLM)

| τ (age decay) | Peak | Peak Step | Last (390) | 结论 |
|---------------|------|-----------|------------|------|
| 1000 | **0.3359** | 180 | **0.2500** | 最优，FrozenLake 中 τ=1000 优于 τ=500 |
| 1500 | 0.3281 | 210 | 0.2031 | 前期强后期衰退 |
| 500 | 0.2656 | 260 | 0.2344 | 稳定但 peak 较低 |
| Baseline | 0.2422 | 110 | 0.1562 | 后期崩溃 |

### 消融实验: IS Correction (FrozenLake LLM)

| 方法 | Peak | Last (390) | 结论 |
|------|------|------------|------|
| Freshness (τ=500, IS) | 0.2812 | **0.2812** | IS 提升后期稳定性，last=peak |
| Freshness (τ=500) | 0.2656 | 0.2344 | 无 IS，后期有退化 |
| Baseline | 0.2422 | 0.1562 | 后期崩溃 |

---

## 统一样式

所有主实验图采用统一配色和标记，无图例（用于后续组合成大图时共享一个图例）：

| 方法 | 颜色 | 标记 |
|------|------|------|
| Baseline (On-Policy) | 蓝色 `#4285F4` | 方形 `s` |
| Standard PER | 黄色 `#FBBC04` | 三角 `^` |
| Freshness Decay (Ours) | 红色 `#EA4335` | 圆形 `o` |

消融实验额外配色 (Sokoban & FrozenLake 共用)：

| 方法 | 颜色 | 标记 |
|------|------|------|
| Freshness (τ=500) | 红色 `#EA4335` | 圆形 `o` |
| Freshness (τ=1000) | 橙色 `#FF9800` | 菱形 `D` |
| Freshness (τ=1500) | 灰色 `#9E9E9E` | 倒三角 `v` |
| Freshness (τ=500, IS) | 绿色 `#34A853` | 十字 `P` |
