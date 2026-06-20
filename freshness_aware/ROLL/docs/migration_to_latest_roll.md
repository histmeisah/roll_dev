# FreshPER 重构方案：基于官方最新 ROLL 重新实现

## 1. 概述

### 1.1 策略
在官方最新 ROLL 代码上**重新实现** FreshPER 功能，旧版代码仅作参考。
不是"复制旧代码过来改 import"，而是理解新版架构后用新版的方式写。

### 1.2 工作目录
```
roll_dev/ROLL/                          ← 旧版（参考，不动）
roll_dev/newest_roll/ROLL_latest/       ← 官方最新版（纯净参考）
roll_dev/newest_roll/ROLL_freshper/     ← 工作副本（在此开发）
```

### 1.3 核心发现：新版已有 infer_logprobs 完整链路

官方最新版已实现 engine logprobs 从 VLLM 到训练的完整数据流：

```
VLLM (logprobs>0)
  → VllmStrategy.generate_request(): 提取 output_logprobs
    → RouterClient._postprocess_generate(): 传递
      → PolicyProxy → TrajEnvManager.make_decision(): 存入 history["infer_logprobs"]
        → formulate_rollouts(): 写入 batch["infer_logprobs"]
          → ActorWorker.loss_func(): 用于 train-infer correction
```

这意味着我们**不需要修改推理链路**，只需：
1. 在 generation_config 中设置 `logprobs=1`
2. 将 `infer_logprobs` 作为 `behavior_log_probs` 存入 replay buffer

---

## 2. 新版训练循环结构（需要理解的基础）

```
AgenticPipeline.run() 每一步:
  PHASE 1:  offload_states (actor_train, critic)
  PHASE 2:  suspend rollout scheduler
  PHASE 3:  model_update (sync weights actor_train → actor_infer)
  PHASE 4:  load_states on actor_infer (init KV cache)
  PHASE 5:  expand_sampler (partial GPU mode)
  PHASE 6:  validation (async, ThreadPoolExecutor)
  PHASE 7:  rollout get_batch → fresh batch
  PHASE 8:  wait for val completion
  PHASE 9:  shrink_sampler (free training GPUs)
  ─── 以下是我们需要注入 replay buffer 的位置 ───
  PHASE 10: compute_discounted_returns
  PHASE 11: ref_log_probs
  PHASE 12: old_log_probs & values
  PHASE 13: advantage computation
  PHASE 14: actor_train.train_step + critic.train_step
```

---

## 3. 分模块实现计划

### Module A: Replay Buffer 核心（纯逻辑，可直接复制）

**目标路径**: `roll/pipeline/agentic/replay_buffer/`

这些文件是纯算法逻辑，只依赖 `DataProto` 和 `torch`，与框架无耦合：
- `base_buffer.py` — 抽象基类
- `trajectory_buffer.py` — 轨迹级缓冲
- `step_buffer.py` — 步级缓冲
- `buffer_factory.py` — 工厂函数
- `priority_functions.py` — FreshPER / PER 优先级计算
- `segment_tree.py` — 优先级采样的底层数据结构

**操作**: 直接复制，更新 `__init__.py` 中的 import 路径。
**验证**: 单元测试通过即可。

---

### Module B: Config 扩展

**文件**: `roll/pipeline/agentic/agentic_config.py`

在 `AgenticConfig` 中新增两个 dataclass：

```python
@dataclass
class ReplayConfig:
    """Off-policy replay buffer configuration."""
    enabled: bool = False
    capacity: int = 100000
    min_size: int = 128                    # buffer 最少多少条才开始采样
    train_steps_per_env_step: int = 2      # 每次 rollout 后训练几轮
    sampling_mode: str = "trajectory"      # trajectory / step
    eviction_strategy: str = "fifo"
    # FreshPER
    priority_function: str = "reward_fresh"
    priority_exponent: float = 0.6
    enable_age_decay: bool = True
    age_decay: float = 1000.0
    # IS correction
    importance_sampling_correction: bool = False
    importance_beta: float = 0.4

@dataclass
class OffPolicyMonitorConfig:
    """Off-policy diagnostics monitoring."""
    enabled: bool = False
    monitor_interval: int = 1
```

在 `AgenticConfig` 中添加字段：
```python
replay: ReplayConfig = field(default_factory=ReplayConfig)
offpolicy_monitor: OffPolicyMonitorConfig = field(default_factory=OffPolicyMonitorConfig)
```

---

### Module C: Pipeline 训练循环集成（核心工作）

**文件**: `roll/pipeline/agentic/agentic_pipeline.py`

#### C.1 初始化 replay buffer
在 `__init__` 中，当 `replay.enabled=True` 时创建 buffer：
```python
if self.pipeline_config.replay.enabled:
    from roll.pipeline.agentic.replay_buffer import create_replay_buffer
    self.replay_buffer = create_replay_buffer(self.pipeline_config.replay)
```

#### C.2 开启 engine logprobs
在 `TrajEnvManager.make_decision()` 中，当 replay 开启时设置 `logprobs=1`：
```python
if self.mode == "train" and self.pipeline_config.replay.enabled:
    generation_config["logprobs"] = 1
```
新版 `create_sampling_params_for_vllm()` 已支持 `gen_kwargs.get("logprobs", 0)`，无需改 vllm_strategy。

#### C.3 注入 replay 训练循环
在 PHASE 14 之后（或替代 PHASE 10-14），实现 replay 训练：

```
原始流程:                 Replay 模式:
──────────               ──────────
PHASE 7: get_batch       PHASE 7: get_batch (fresh_batch)
                         PHASE 9.1: store fresh_batch to replay buffer
                           - batch["behavior_log_probs"] = batch["infer_logprobs"]
                           - replay_buffer.add(batch, global_step)
PHASE 10-14: 单次训练    PHASE 9.2: for i in range(train_steps_per_env_step):
                           - 第1轮: 用 fresh_batch 训练（on-policy）
                           - 第2+轮: 从 buffer 采样 replay_batch
                             - 重新计算 old_log_probs（当前策略）
                             - 重新计算 ref_log_probs
                             - 重新计算 advantages
                             - actor_train.train_step(replay_batch)
```

关键设计决策：
- **第1轮用 fresh data 训练**：保持 on-policy 基线性能
- **第2+轮从 buffer 采样**：额外的 off-policy 训练步，提高样本效率
- **每轮都重新计算 log_probs 和 advantages**：off-policy 数据需要用当前策略重计算

#### C.4 Off-policy 监控指标
计算并记录：
- `offpolicy/importance_weight_mean`: IS ratio 均值
- `offpolicy/importance_weight_max`: IS ratio 最大值
- `offpolicy/approx_kl_divergence`: 近似 KL 散度
- `offpolicy/fraction_in_clip_range`: IS ratio 在 PPO clip 范围内的比例
- `replay/buffer_size`: 当前 buffer 大小
- `replay/buffer_utilization`: buffer 使用率
- `replay/avg_age`: 采样数据的平均年龄

---

### Module D: Off-policy Monitor（独立模块）

**目标路径**: `roll/pipeline/agentic/offpolicy_monitor.py`

从旧版复制核心计算函数（纯数学计算，无框架依赖）：
- `compute_offpolicy_metrics(current_log_probs, behavior_log_probs, response_mask)`
- `log_offpolicy_diagnostics(metrics, batch, global_step)`

---

### Module E: TrajEnvManager 适配

**文件**: `roll/pipeline/agentic/env_manager/traj_env_manager.py`

需要的改动很小：

#### E.1 开启 logprobs
在 `make_decision()` 中：
```python
# 在构建 generation_config 之后
if self.mode == "train" and getattr(self.pipeline_config, 'replay', None) and self.pipeline_config.replay.enabled:
    generation_config["logprobs"] = 1
```

#### E.2 确认 infer_logprobs 已在 formulate_rollouts 输出中
新版 `formulate_rollouts()` 已经把 `infer_logprobs` 写入 batch。只需确认：
- 格式是否与 `old_log_probs` 兼容（next-token format, shape `[batch, seq_len-1]`）
- 是否正确 padding 到 sequence_length

---

### Module F: 实验配置

**目录**: `roll_dev/experiments/`

更新 YAML 配置以适配新版格式。主要变化：
- Hydra defaults 路径
- 新增 `replay:` 配置块
- 环境配置适配 `gem.make` 方式
- `use_engine_logprobs` 不再需要（新版直接用 `logprobs` 字段）

---

## 4. 实施顺序

```
Step 1: Module A — 复制 replay buffer 核心          [0.5天]
  └ 更新 import 路径，跑单元测试

Step 2: Module B — 扩展 Config                      [0.5天]
  └ 新增 ReplayConfig、OffPolicyMonitorConfig

Step 3: Module E — TrajEnvManager 添加 logprobs=1   [0.5天]
  └ 验证 infer_logprobs 格式

Step 4: Module C — Pipeline 训练循环集成             [2-3天]
  └ C.1: 初始化 replay buffer
  └ C.2: 存储 fresh batch
  └ C.3: replay 训练子循环
  └ C.4: 监控指标

Step 5: Module D — Off-policy Monitor               [0.5天]
  └ 复制计算函数，集成到 pipeline

Step 6: Module F — 实验配置 + 集成测试               [1-2天]
  └ FrozenLake baseline（无 replay）验证新版可运行
  └ FrozenLake + replay buffer 验证 off-policy 训练
  └ 对比旧版结果

总计: 约 5-7 个工作日
```

---

## 5. 不需要做的事情（相比旧版的简化）

| 旧版需要做的 | 新版不需要了 | 原因 |
|-------------|------------|------|
| 修改 vllm_strategy.py 添加 logprobs | ❌ | 新版已支持 `gen_kwargs.get("logprobs", 0)` |
| 修改 generate_scheduler 传递 logprobs | ❌ | RouterClient 已处理 |
| 在 step() 中提取 per-turn logprobs | ❌ | make_decision() 已存 infer_logprobs |
| 在 formulate_rollouts() 中重建全序列 logprobs | ❌ | 增量 tokenization 直接拼接 |
| 修改 policy_proxy.py 传递 use_engine_logprobs | ❌ | generation_config 直接传 logprobs 字段 |
| 处理 tokenization mismatch | ❌ | 增量式 tokenization 无 mismatch |
| rollout_scheduler 中 padding behavior_log_probs | ❌ | formulate_rollouts 统一 padding |

---

## 6. 风险与注意事项

### 6.1 infer_logprobs 格式验证
需要确认 `batch["infer_logprobs"]` 的 tensor shape 和语义：
- 是否是 next-token format `[batch, seq_len-1]`？
- 还是 aligned format `[batch, seq_len]`（position i = logprob of token i）？
- 与 `old_log_probs`（来自 `compute_log_probs()`）的格式是否一致？

**行动**: 在 Step 3 中用 debug 打印确认。

### 6.2 Partial GPU Mode 兼容性
Replay 训练子循环发生在 PHASE 9 shrink 之后，此时 training GPUs 已可用。
多轮训练不需要额外的 expand/shrink。

### 6.3 Async Pipeline 兼容性
当 `async_pipeline=True` 时，rollout 和 training 可能重叠。
Replay 训练子循环需要确保不与 async rollout 冲突。
**建议**：初期只在 sync mode (`async_generation_ratio=0`) 下支持 replay。

### 6.4 VLLM Temperature Scaling
当前 VLLM 已更新到最新版，需要确认是否支持 `logprobs_mode="processed_logprobs"`。
如果支持，在 `create_sampling_params_for_vllm` 中添加该参数以获取 temperature 缩放后的真实 logprobs。

---

## 7. 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `roll/pipeline/agentic/replay_buffer/` | **新建目录** | 复制核心模块 + 更新 imports |
| `roll/pipeline/agentic/agentic_config.py` | **修改** | 新增 ReplayConfig, OffPolicyMonitorConfig |
| `roll/pipeline/agentic/agentic_pipeline.py` | **修改** | 注入 replay buffer 逻辑 |
| `roll/pipeline/agentic/env_manager/traj_env_manager.py` | **修改** | 添加 `logprobs=1` 条件 |
| `roll/pipeline/agentic/offpolicy_monitor.py` | **新建** | off-policy 监控计算 |
| `examples/config/` | **修改** | 添加 replay 相关默认配置 |
