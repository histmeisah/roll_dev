# Trajectory 信息存储现状分析

> 日期: 2026-01-25

---

## 1. 当前数据结构

### 1.1 RolloutCache（核心缓存结构）

位置: `roll/agentic/rollout/base_env_manager.py:32-42`

```python
@dataclass
class RolloutCache:
    env_id: int
    group_id: int
    tag: str

    history: List[Dict]  # 每个step的完整信息
    frames: List         # 渲染帧（用于GIF）

    truncated: bool = False
    terminated: bool = False
    step: int = 0
```

### 1.2 history 中每个 step 包含的信息

位置: `step_env_manager.py:94-123`, `traj_env_manager.py:191-217`

```python
{
    "state": "环境状态（如FrozenLake的grid）",
    "actions_left": 10,
    "reward": 0.0,
    "penalty": -0.15,           # 格式错误惩罚
    "llm_response": "Right",    # 模型原始输出
    "observation": [            # 完整对话历史
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "action_content": "Right",  # 解析出的action
    "metrics": {                # 环境指标
        "success": False,
        "action_is_valid": True
    }
}
```

---

## 2. 数据流向

```
┌─────────────────────────────────────────────────────────────────┐
│                        RolloutCache                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ history: [step0, step1, step2, ...]                         ││
│  │   - state, action, reward, penalty, llm_response, metrics   ││
│  │   - observation (完整对话)                                   ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼ formulate_rollouts()
┌─────────────────────────────────────────────────────────────────┐
│                         DataProto                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ batch (TensorDict):                                         ││
│  │   - input_ids, attention_mask, position_ids                 ││
│  │   - response_mask, prompt_mask                              ││
│  │   - scores, penalty                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ non_tensor_batch:                                           ││
│  │   - messages_list (对话历史)                                 ││
│  │   - episode_scores, step_scores                             ││
│  │   - tags, env_ids, group_ids, traj_id                       ││
│  │   - frames, state_hash                                      ││
│  │   - step, done, terminated, truncated                       ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────┬──────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────┐              ┌───────────────────────┐
│    Replay Buffer      │              │      日志输出          │
│ (StepReplayBuffer)    │              │ (agentic_pipeline.py) │
│                       │              │                       │
│ 保存:                 │              │ 保存:                 │
│ - input_ids (tokens)  │              │ - 每step仅10条样本    │
│ - response_mask       │              │ - prompt/response文本 │
│ - scores, penalty     │              │ - episode_score       │
│ - messages_list       │              │ - penalty             │
│ - step_scores         │              │                       │
│ - behavior_log_probs  │              │ 格式: JSON (日志行)   │
└───────────────────────┘              └───────────────────────┘
```

---

## 3. 当前保存机制

### 3.1 日志输出（有限采样）

位置: `agentic_pipeline.py:1165-1182`

```python
# 每个global_step只保存10条样本
for prompt, response, episode_score, penalty in ...:
    log_res.append({
        "prompt": prompt,
        "response": response,
        "episode_score": episode_score,
        "penalty": penalty,
    })
    if len(log_res) >= 10:
        break
logger.info(json.dumps(log_res, ensure_ascii=False))
```

**问题**: 只采样10条，无法分析完整行为

### 3.2 渲染保存（仅GIF）

位置: `agentic_pipeline.py:1223-1232`

```python
if self.pipeline_config.render_save_dir and "frames" in eval_batch.non_tensor_batch:
    dump_rollout_render(
        save_dir=self.pipeline_config.render_save_dir,
        step=global_step,
        frames=eval_batch.non_tensor_batch["frames"],
        env_ids=eval_batch.non_tensor_batch["env_ids"],
        ...
    )
```

**问题**: 只保存GIF渲染，不保存文本trajectory

### 3.3 Replay Buffer 存储

位置: `step_buffer.py:156-204`

```python
@dataclass
class StepEntry:
    input_ids: np.ndarray           # Tokenized
    attention_mask: np.ndarray
    response_mask: np.ndarray
    prompt_mask: np.ndarray
    scores: np.ndarray
    penalty: float

    # Metadata
    env_id: str
    messages_list: List[Dict]       # 对话历史 ✓
    step_scores: float
    behavior_log_probs: np.ndarray
    ...
```

**有**: messages_list 包含完整对话
**没有**: 直接可读的 prompt/response 文本（需要 decode）

---

## 4. 主要问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **采样不完整** | 日志只保存10条/step | 无法分析全部模型行为 |
| **无文本保存** | 只有tokenized数据 | 需要额外decode才能分析 |
| **无trajectory聚合** | 数据按step分散 | 难以追踪单个episode的完整过程 |
| **无分析工具** | 缺少trajectory分析功能 | 难以定位问题 |
| **RolloutCache丢失** | 训练后内存释放 | 无法回溯分析 |

---

## 5. 已有的信息来源

### 5.1 可用于分析的字段

```python
# 在 DataProto.non_tensor_batch 中
"messages_list"     # 完整对话历史 (可用于重建trajectory)
"traj_id"           # Trajectory标识
"traj_group_id"     # Trajectory组标识
"step"              # 当前step序号
"episode_scores"    # 整episode累计奖励
"step_scores"       # 当前step奖励
"done"              # 是否为episode最后一步
"state_hash"        # 状态哈希（用于去重）
```

### 5.2 可以从 input_ids 恢复的信息

```python
# 需要 tokenizer.decode()
prompt = tokenizer.decode(input_ids[prompt_mask.bool()])
response = tokenizer.decode(input_ids[response_mask.bool()])
```

---

## 6. 建议改进方向

### 6.1 短期方案：增强日志输出

```python
# 在 agentic_pipeline.py 中添加配置
trajectory_log:
  enabled: true
  save_dir: ${output_dir}/trajectories
  save_interval: 10          # 每10个step保存一次
  max_trajectories: 1000     # 最多保存1000条
  include_failed: true       # 包含失败的trajectory
```

### 6.2 中期方案：添加 Trajectory Logger

```python
class TrajectoryLogger:
    """
    专门用于记录和分析trajectory的组件
    """
    def log_trajectory(self, traj_id, history, episode_score, success):
        """保存单个trajectory到JSONL文件"""
        entry = {
            "traj_id": traj_id,
            "timestamp": time.time(),
            "episode_score": episode_score,
            "success": success,
            "steps": [
                {
                    "step": i,
                    "state": h["state"],
                    "action": h["action_content"],
                    "llm_response": h["llm_response"],
                    "reward": h["reward"],
                    "penalty": h["penalty"],
                }
                for i, h in enumerate(history)
            ]
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### 6.3 长期方案：Trajectory 分析工具

```python
class TrajectoryAnalyzer:
    """分析保存的trajectory数据"""

    def load_trajectories(self, log_path):
        """加载trajectory日志"""

    def analyze_success_patterns(self):
        """分析成功trajectory的共同模式"""

    def analyze_failure_patterns(self):
        """分析失败trajectory的失败点"""

    def compare_checkpoints(self, ckpt1, ckpt2):
        """对比不同checkpoint的行为差异"""
```

---

## 7. 现有可用的分析方法

### 7.1 从日志提取样本

```python
import json
import re

def extract_samples_from_log(log_path):
    samples = []
    with open(log_path, 'r') as f:
        for line in f:
            if '"prompt":' in line and '"response":' in line:
                try:
                    # 尝试解析JSON数组
                    match = re.search(r'\[.*\]', line)
                    if match:
                        samples.extend(json.loads(match.group()))
                except:
                    pass
    return samples
```

### 7.2 从 Replay Buffer 恢复trajectory

```python
def recover_trajectory_from_buffer(buffer, traj_id, tokenizer):
    """从replay buffer恢复trajectory文本"""
    steps = [s for s in buffer.entries if s.traj_id == traj_id]
    steps.sort(key=lambda x: x.step)

    trajectory = []
    for step in steps:
        prompt = tokenizer.decode(step.input_ids[step.prompt_mask.astype(bool)])
        response = tokenizer.decode(step.input_ids[step.response_mask.astype(bool)])
        trajectory.append({
            "step": step.step,
            "prompt": prompt,
            "response": response,
            "reward": step.step_scores,
        })
    return trajectory
```

---

## 8. 总结

当前框架**有**trajectory信息，但**分散**在多个地方，**没有**统一的保存和分析机制。

主要需要添加的功能：
1. **Trajectory保存器** - 将完整trajectory保存到文件
2. **Trajectory分析器** - 分析成功/失败模式
3. **配置选项** - 控制保存粒度和范围

这些改进可以帮助：
- 分析模型在哪个step开始出错
- 对比不同实验的行为差异
- 定位模式崩溃的根本原因
