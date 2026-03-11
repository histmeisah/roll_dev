# FrozenLake Replay Buffer Experiments - 2xA100 40GB

## 实验设计

```
实验层级：

1. Baseline（无Replay）
   ├── traj_baseline    : TrajEnvManager, replay.enabled=false
   └── step_baseline    : StepEnvManager, replay.enabled=false

2. Off-Policy with Standard PER（标准PER）
   ├── traj_per         : TrajEnvManager, priority=reward, IS correction
   └── step_per         : StepEnvManager, priority=reward, IS correction

3. Off-Policy with Reward-Fresh（我们的扩展）
   ├── traj_reward_fresh : TrajEnvManager, priority=reward_fresh, IS correction
   └── step_reward_fresh : StepEnvManager, priority=reward_fresh, IS correction

4. Off-Policy with N-Step Returns（多步回报）
   ├── step_per_nstep         : StepEnvManager, PER + 5-step returns
   └── step_reward_fresh_nstep: StepEnvManager, Reward-Fresh + 5-step returns

5. Off-Policy with V-trace（重要性采样校正）
   ├── step_per_vtrace         : StepEnvManager, PER + V-trace advantage
   └── step_reward_fresh_vtrace: StepEnvManager, Reward-Fresh + V-trace advantage
```

## 配置文件

### 基础配置

| 配置文件 | EnvManager | Replay | Priority | 说明 |
|---------|-----------|--------|----------|------|
| `traj_baseline.yaml` | Traj | ❌ | - | Trajectory级别baseline |
| `step_baseline.yaml` | Step | ❌ | - | Step级别baseline |
| `traj_per.yaml` | Traj | ✅ | `reward` | 标准PER (Schaul et al. 2015) |
| `step_per.yaml` | Step | ✅ | `reward` | 标准PER Step版本 |
| `traj_reward_fresh.yaml` | Traj | ✅ | `reward_fresh` | 我们的扩展：reward × age_decay |
| `step_reward_fresh.yaml` | Step | ✅ | `reward_fresh` | 我们的扩展 Step版本 |

### N-Step 配置（Step级别）

| 配置文件 | Priority | N-Step | 说明 |
|---------|----------|--------|------|
| `step_per_nstep.yaml` | `reward` | 5 | PER + N-step returns |
| `step_reward_fresh_nstep.yaml` | `reward_fresh` | 5 | Reward-Fresh + N-step returns |

### V-trace 配置（Step级别）

| 配置文件 | Priority | V-trace | 说明 |
|---------|----------|---------|------|
| `step_per_vtrace.yaml` | `reward` | ✅ | PER + V-trace off-policy correction |
| `step_reward_fresh_vtrace.yaml` | `reward_fresh` | ✅ | Reward-Fresh + V-trace off-policy correction |

## 运行方式

```bash
# 通用脚本 (修改run.sh中的CONFIG_NAME变量)
./run.sh

# 可用配置:
# Baseline
#   traj_baseline       - Trajectory baseline
#   step_baseline       - Step baseline
# Standard PER
#   traj_per           - Trajectory + 标准PER
#   step_per           - Step + 标准PER
# Reward-Fresh
#   traj_reward_fresh  - Trajectory + Reward-Fresh
#   step_reward_fresh  - Step + Reward-Fresh
# N-Step (Step level only)
#   step_per_nstep          - Step + PER + N-step
#   step_reward_fresh_nstep - Step + Reward-Fresh + N-step
# V-trace (Step level only)
#   step_per_vtrace          - Step + PER + V-trace
#   step_reward_fresh_vtrace - Step + Reward-Fresh + V-trace
```

## Priority Functions

| 函数 | 公式 | 说明 |
|-----|------|------|
| `reward` | `\|reward\| + ε` | 标准PER，基于奖励的优先级 |
| `reward_fresh` | `(\|reward\| + ε) × exp(-age/age_decay)` | **我们的扩展**：reward × 新鲜度 |

### reward_fresh 特点

解决 off-policy LLM RL 的两个关键问题：
1. **高奖励样本更有价值**：成功的trajectory包含更多学习信号
2. **新鲜样本策略漂移更小**：旧样本与当前策略差距大

优先级分布：
- 高奖励 + 新鲜 → 最高优先级
- 低奖励 + 新鲜 → 中等优先级
- 高奖励 + 陈旧 → 中等优先级（被age衰减）
- 低奖励 + 陈旧 → 最低优先级

## N-Step Returns

| 函数 | 公式 | 说明 |
|-----|------|------|
| N-Step | `R_t^(n) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}` | 多步累积回报 |

### N-Step 特点

1. **更好的长期信用分配**：考虑多个步骤的奖励，而不是单步
2. **减少偏差**：相比单步TD，n-step可以减少bootstrap偏差
3. **适合序列决策**：FrozenLake需要连续正确动作才能到达目标

N-Step 参数：
- `n_step: 5` - 使用5步回报
- `nstep_gamma: 0.99` - 步级别折扣因子
- `use_nstep_in_advantage: true` - 在优势计算中使用n-step回报

## V-trace

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `rho_bar` | 1.5 | 重要性采样比率截断阈值 |
| `c_bar` | 1.0 | 迹系数截断阈值 |

### V-trace 特点 (IMPALA, Espeholt et al. 2018)

1. **Off-Policy 校正**：通过截断的重要性采样校正策略漂移
2. **稳定性**：截断防止方差爆炸
3. **收敛保证**：理论上收敛到最优策略不动点

V-trace 公式：
```
v_s = V(x_s) + Σ γ^{t-s} (Π c_i) δ_t V
δ_t V = ρ_t (r_t + γV(x_{t+1}) - V(x_t))
ρ_t = min(ρ̄, π(a_t|x_t)/μ(a_t|x_t))
c_t = min(c̄, π(a_t|x_t)/μ(a_t|x_t))
```

使用场景：
- 大型Replay Buffer
- 分布式训练（策略滞后）
- 异步训练

## Hardware Configuration

- **GPUs**: 2x NVIDIA A100 40GB
- **GPU 0**: actor_train + reference (DeepSpeed ZeRO-2)
- **GPU 1**: actor_infer (vLLM)

## Model

- **Base Model**: Qwen2.5-0.5B-Instruct
- **Path**: `/mnt/nasdata/weiyu/llm_model/qwen_models/Qwen2.5-0.5B-Instruct`

## 关键参数对比

### 基础配置对比

| Parameter | Baseline | Standard PER | Reward-Fresh |
|-----------|----------|--------------|--------------|
| replay.enabled | false | true | true |
| replay.capacity | - | 50000 | 50000 |
| priority_function | - | `reward` | `reward_fresh` |
| priority_exponent | - | 0.6 | 0.6 |
| age_decay | - | - | 500.0 |
| importance_sampling_correction | - | true | true |
| importance_beta | - | 0.4 | 0.4 |
| train_steps_per_env_step | - | 2 | 2 |

### N-Step 配置对比

| Parameter | PER + N-Step | Reward-Fresh + N-Step |
|-----------|--------------|----------------------|
| priority_function | `reward` | `reward_fresh` |
| enable_nstep | true | true |
| n_step | 5 | 5 |
| nstep_gamma | 0.99 | 0.99 |
| use_nstep_in_advantage | true | true |
| age_decay | - | 500.0 |

### V-trace 配置对比

| Parameter | PER + V-trace | Reward-Fresh + V-trace |
|-----------|---------------|------------------------|
| adv_estimator | `vtrace` | `vtrace` |
| priority_function | `reward` | `reward_fresh` |
| vtrace.rho_bar | 1.5 | 1.5 |
| vtrace.c_bar | 1.0 | 1.0 |
| age_decay | - | 500.0 |

## Replay Buffer Configuration

### Standard PER (traj_per.yaml / step_per.yaml)
```yaml
replay:
  enabled: true
  capacity: 50000
  train_steps_per_env_step: 2
  priority_function: "reward"        # Priority = |reward|
  priority_exponent: 0.6             # alpha
  importance_sampling_correction: true
  importance_beta: 0.4
```

### Reward-Fresh (traj_reward_fresh.yaml / step_reward_fresh.yaml)
```yaml
replay:
  enabled: true
  capacity: 50000
  train_steps_per_env_step: 2
  priority_function: "reward_fresh"  # Priority = |reward| × exp(-age/age_decay)
  priority_exponent: 0.6
  age_decay: 500.0                   # Freshness decay constant
  importance_sampling_correction: true
  importance_beta: 0.4
```

### N-Step Returns (step_per_nstep.yaml / step_reward_fresh_nstep.yaml)
```yaml
replay:
  enabled: true
  capacity: 50000
  priority_function: "reward"        # Or "reward_fresh"
  enable_nstep: true
  n_step: 5                          # 5-step returns
  nstep_gamma: 0.99                  # Step-level discount
  use_nstep_in_advantage: true       # Use n-step in advantage computation
  use_bootstrap: false
```

### V-trace (step_per_vtrace.yaml / step_reward_fresh_vtrace.yaml)
```yaml
adv_estimator: "vtrace"              # V-trace advantage estimator

vtrace:
  rho_bar: 1.5                       # Truncation for IS ratio
  c_bar: 1.0                         # Truncation for trace coefficient

replay:
  enabled: true
  capacity: 50000
  priority_function: "reward"        # Or "reward_fresh"
```

## Running Experiments

1. **Sync code to server**:
   ```bash
   powershell.exe -Command "cd e:\code_project\python_code\local_roll_dev; .\sync.bat push-all"
   ```

2. **SSH to server and run**:
   ```bash
   ssh aicoder
   cd /mnt/project_modelware_roce/zhaojian/liangsirui/weiyu/projects/local_roll_dev/roll_dev/experiments/frozen_lake_2a100_replaybuffer
   chmod +x run.sh
   ./run.sh traj_baseline
   ```

3. **Monitor training**:
   ```bash
   tail -f output/*/logs/training_*.log
   watch -n 1 nvidia-smi
   ```

## Output Structure
```
output/
└── YYYYMMDD_HHMMSS/
    ├── logs/
    │   └── training_*.log
    ├── models/
    ├── tensorboard/
    ├── render/
    └── wandb/
```

## 实验对比要点

1. **Baseline vs PER**: 验证 replay buffer + priority sampling 的效果
2. **PER vs Reward-Fresh**: 验证 age decay 对 off-policy 训练的帮助
3. **Traj vs Step**: 比较不同 EnvManager 的数据组织方式
4. **N-Step Returns**: 验证多步回报对长期信用分配的帮助
5. **V-trace**: 验证截断重要性采样对 off-policy 校正的效果
6. **组合实验**: Reward-Fresh + N-Step / V-trace 的协同效果
