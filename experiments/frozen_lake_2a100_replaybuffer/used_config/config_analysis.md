# Frozen Lake 2A100 Replay Buffer 实验配置分析

## 配置文件列表

| 日期 | 实验名称 | 文件名 |
|------|---------|--------|
| 01/15 | traj_per | 20260115_193211_traj_per_config.yaml |
| 01/19 | step_per_nstep | 20260119_204444_step_per_nstep_config.yaml |
| 01/19 | step_reward_fresh_nstep | 20260119_223447_step_reward_fresh_nstep_config.yaml |
| 01/20 | traj_reward_fresh | 20260120_142505_traj_reward_fresh_config.yaml |
| 01/22 | traj_baseline (v1) | 20260122_145417_traj_baseline_config.yaml |
| 01/22 | step_baseline (v1) | 20260122_145133_step_baseline_config.yaml |
| 01/23 | step_baseline (v2) | 20260123_164201_step_baseline_config.yaml |
| 01/24 | step_per | 20260124_155601_step_per_config.yaml |
| 01/24 | step_reward_fresh | 20260124_160415_step_reward_fresh_config.yaml |
| 01/26 | step_reward_fresh (v2) | 20260126_050125_step_reward_fresh_config.yaml |
| 01/26 | traj_baseline (v2) | 20260126_202655_traj_baseline_config.yaml |
| 01/27 | traj_reward_fresh (v2) | 20260127_215523_traj_reward_fresh_config.yaml |

## 关键参数对比表

### Replay Buffer 配置

| 日期 | 实验名称 | Level | Replay | Priority | N-Step | IS Correction | Age Decay |
|------|---------|-------|--------|----------|--------|---------------|-----------|
| 01/15 | traj_per | Traj | ✅ | reward | ❌ | ✅ | 1000 |
| 01/19 | step_per_nstep | Step | ✅ | reward | ✅ (n=5) | ✅ | 1000 |
| 01/19 | step_reward_fresh_nstep | Step | ✅ | reward_fresh | ✅ (n=5) | ✅ | 500 |
| 01/20 | traj_reward_fresh | Traj | ✅ | reward_fresh | ❌ | ✅ | 500 |
| 01/22 | traj_baseline (v1) | Traj | ❌ | - | - | - | - |
| 01/22 | step_baseline (v1) | Step | ❌ | - | - | - | - |
| 01/23 | step_baseline (v2) | Step | ❌ | - | - | - | - |
| 01/24 | step_per | Step | ✅ | reward | ❌ | ✅ | 1000 |
| 01/24 | step_reward_fresh | Step | ✅ | reward_fresh | ❌ | ✅ | 500 |
| 01/26 | step_reward_fresh (v2) | Step | ✅ | reward_fresh | ❌ | ✅ | 500 |
| 01/26 | traj_baseline (v2) | Traj | ❌ | - | - | - | - |
| 01/27 | traj_reward_fresh (v2) | Traj | ✅ | reward_fresh | ❌ | ✅ | 500 |

### 训练参数配置

| 日期 | 实验名称 | adv_clip | init_kl | use_kl_loss | kl_loss_coef | entropy_loss | batch (per×accum) | max_steps |
|------|---------|----------|---------|-------------|--------------|--------------|-------------------|-----------|
| 01/15 | traj_per | 0.2 | 0.0 | ❌ | 0 | 0 | 4×32 | 400 |
| 01/19 | step_per_nstep | 0.2 | 0.0 | ❌ | 0 | 0 | 4×32 | 400 |
| 01/19 | step_reward_fresh_nstep | 0.2 | 0.0 | ❌ | 0 | 0 | 4×32 | 400 |
| 01/20 | traj_reward_fresh | 0.2 | 0.0 | ❌ | 0 | 0 | 4×32 | 400 |
| 01/22 | traj_baseline (v1) | 0.2 | 0.0 | ❌ | 0 | 0 | 4×32 | 400 |
| 01/22 | step_baseline (v1) | 0.2 | 0.0 | ❌ | 0 | 0 | 4×32 | 400 |
| 01/23 | step_baseline (v2) | 20 | 0.0 | ✅ | 0.01 | 0 | 4×32 | 400 |
| 01/24 | step_per | 20 | 0.0 | ✅ | 0.01 | 0 | 4×32 | 400 |
| 01/24 | step_reward_fresh | 20 | 0.0 | ✅ | 0.01 | 0 | 4×32 | 400 |
| 01/26 | step_reward_fresh (v2) | 20 | 0.1 | ✅ | 0.05 | 0.01 | 2×64 | 100 |
| 01/26 | traj_baseline (v2) | 20 | 0.1 | ✅ | 0.05 | 0.01 | 2×64 | 400 |
| 01/27 | traj_reward_fresh (v2) | 20 | 0.1 | ✅ | 0.05 | 0.01 | 2×64 | 400 |

## 配置演变时间线

### Phase 1: 01/15 - 01/22 (原始配置)
```yaml
advantage_clip: 0.2
init_kl_coef: 0.0
use_kl_loss: false
kl_loss_coef: 0
entropy_loss_coef: 0
per_device_train_batch_size: 4
gradient_accumulation_steps: 32
```

### Phase 2: 01/23 - 01/24 (引入KL Loss)
```yaml
advantage_clip: 20          # 改变
init_kl_coef: 0.0
use_kl_loss: true           # 改变
kl_loss_coef: 0.01          # 改变
entropy_loss_coef: 0
per_device_train_batch_size: 4
gradient_accumulation_steps: 32
```

### Phase 3: 01/26 - 01/27 (进一步增加正则化)
```yaml
advantage_clip: 20
init_kl_coef: 0.1           # 改变
use_kl_loss: true
kl_loss_coef: 0.05          # 改变
entropy_loss_coef: 0.01     # 改变
per_device_train_batch_size: 2   # 改变
gradient_accumulation_steps: 64  # 改变
```

## 实验分组

### 按 Replay Buffer 类型

| 类型 | 实验 |
|------|------|
| **Baseline (无Replay)** | traj_baseline (v1, v2), step_baseline (v1, v2) |
| **Standard PER** | traj_per, step_per |
| **Reward-Fresh** | traj_reward_fresh (v1, v2), step_reward_fresh (v1, v2) |
| **PER + N-Step** | step_per_nstep |
| **Reward-Fresh + N-Step** | step_reward_fresh_nstep |

### 按 EnvManager 类型

| 类型 | 实验 |
|------|------|
| **Trajectory Level** | traj_baseline, traj_per, traj_reward_fresh |
| **Step Level** | step_baseline, step_per, step_reward_fresh, step_per_nstep, step_reward_fresh_nstep |

---
*生成时间: 2026-01-28*
*待补充: wandb实际训练效果数据*
