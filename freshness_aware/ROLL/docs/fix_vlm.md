# GroupReplayBuffer: VLM 兼容性修复方案

## 1. 背景

旧版 `bandit-upstream-v2` 分支的 `TrajectoryReplayBuffer` 在 VLM agentic
训练中跑通过（commit `c5e876ad`：VLM multimodal bandit support，配套修了
Qwen3-VL encoder image token mismatch）。

当前 `replay-buffer-clean` 分支基于最新 upstream 重建的 `GroupReplayBuffer`
**不能在 VLM 上使用**。原因不是数据丢失，而是**我们的 buffer 只保存了一个
硬编码的非张量字段子集**，把 VLM 图片信息和 env 动态注入的字段都丢了。

本文档梳理原因，并给出**完整修复方案**与**分阶段实施步骤**。

## 2. VLM 数据流（upstream 最新版本）

### 2.1 图片数据驻留位置

VLM 在 ROLL 中的关键设计：**PIL 图片对象不是 tensor，不放在 `batch.batch`，
而是藏在 `batch.non_tensor_batch["messages_list"]` 里**。

`vl_traj_env_manager.py:477-484` 在 rollout 结束时：

```python
lm_input.non_tensor_batch.update({
    "env_ids": ...,
    "group_ids": ...,
    "messages_list": np.array([messages], dtype=object),  # ← PIL 图片在这里
    "tags": ...,
    "step_scores": ...,
    "episode_scores": ...,
})
```

`messages_list` 的元素形如：
```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
        {"type": "image", "image_PIL": <PIL.Image>},   # ← 图片对象本身
        {"type": "text", "text": "..."},
    ]},
    {"role": "assistant", "content": "..."},
]
```

### 2.2 `add_extra_data()` 动态注入（新机制）

`vl_traj_env_manager.py:495-496`：

```python
if callable(getattr(self.env, "add_extra_data", None)):
    self.env.add_extra_data(lm_input, messages)
```

不同 VLM env 可能注入**不同字段名**。例如 `deepeyes/env.py:444-451`：

```python
def add_extra_data(self, data, messages):
    data.non_tensor_batch.update({
        "question": ...,
        "ground_truth": ...,
        "message": ...,      # 注意：是 "message"，不是 "messages_list"
    })
```

而另一个 VLM env 可能注入 `multi_modal_data` 或 `images` 这类完全不同的 key。

### 2.3 图片 → pixel_values 的重新编码

VLM 训练需要 `pixel_values`，它**不是从 rollout batch 继承**的，而是在
训练前由 collator 重新生成：

路径：`messages_list` → `custom_vl_apply_chat_template()`
→ `collator.processor(images=..., text=...)` → `pixel_values`

关键函数：`roll/pipeline/agentic/env_manager/token_mask_utils.py:39-72`
`custom_vl_apply_chat_template()`。

**结论**：只要 replay buffer 能保留完整的 `messages_list`（里面有 PIL 图片），
训练路径就能自动重建 `pixel_values`。我们不需要存 tensor 形式的图片。

### 2.4 3D position_ids（Qwen2.5-VL / Qwen3-VL）

VLM 的 `position_ids` 形状是 `[C, seq_len]`（C=3 或 4，对应时间+空间维度），
不是普通的 `[seq_len]`。`megatron_strategy.py:436-442` 通过
`position_ids.dim() == 3` 自动检测多维情况。

**当前 `GroupReplayBuffer` 已处理**（`group_buffer.py:358-363` 的
`first_pos.ndim > 1` 分支），这部分**不用改**。

## 3. 当前 `GroupReplayBuffer` 的三个致命缺陷

### 3.1 非张量字段硬编码

`group_buffer.py:_extract_trajectory_entry()` 只读 8 个固定字段：
`env_ids / group_ids / messages_list / tags / frames / step_scores /
episode_scores / traj_group_id / traj_id`。

**问题**：`add_extra_data` 注入的 env-specific 字段（`multi_modal_data`、
`question`、`message`、`image_grid_thw` 等）**一个都读不到**。

### 3.2 采样时重建 DataProto 也是硬编码

`group_buffer.py:421-437` `sample_for_training()` 的 `dataproto.non_tensor_batch`
赋值也只有 8 个固定 key。**采样出的 batch 会缺少 VLM 训练必需的 `multi_modal_data`
等字段**。

### 3.3 `TrajectoryEntry` dataclass 字段闭合

`trajectory_buffer.py:26-60` 的 `TrajectoryEntry` 字段是 hard-coded 的，没有
预留给 env-specific 字段的通用容器。

## 4. 修复方案

### 4.1 核心原则

**动态保留 `non_tensor_batch` 的全部键，不做硬编码**。

### 4.2 具体改动

#### 改动 1：`TrajectoryEntry` 增加通用容器

`trajectory_buffer.py`：

```python
@dataclass
class TrajectoryEntry:
    # ... 原有字段保持 ...

    # 通用容器：存储所有 env-specific / VLM 额外的非张量字段
    extra_non_tensor: Dict[str, Any] = field(default_factory=dict)
```

#### 改动 2：`_extract_trajectory_entry` 动态遍历

`group_buffer.py`：

```python
# 已知的标准字段（放入 TrajectoryEntry 专有字段）
STANDARD_NT_KEYS = {
    "env_ids", "group_ids", "messages_list", "tags", "frames",
    "step_scores", "episode_scores", "traj_group_id", "traj_id",
}

def _extract_trajectory_entry(self, batch, idx, global_step):
    # 1. 标准字段照旧提取（保留向后兼容）
    env_id = _nt("env_ids", "")
    # ...

    # 2. 动态收集所有非标准字段到 extra_non_tensor
    extra_non_tensor = {}
    for key, arr in batch.non_tensor_batch.items():
        if key not in STANDARD_NT_KEYS:
            try:
                extra_non_tensor[key] = arr[idx]
            except Exception as e:
                logger.warning(f"Failed to extract non_tensor[{key}] at idx {idx}: {e}")

    return TrajectoryEntry(
        ...,
        extra_non_tensor=extra_non_tensor,
    )
```

#### 改动 3：`sample_for_training` 动态还原

`group_buffer.py:sample_for_training()`：

```python
# 原有硬编码字段
dataproto.non_tensor_batch = {
    "env_ids": np.array(env_ids, dtype=object),
    "group_ids": ...,
    # ...
}

# 新增：动态还原 extra_non_tensor（VLM 的 multi_modal_data 等）
# 首先收集所有 entries 的 extra keys
extra_keys = set()
for traj in all_trajectories:
    extra_keys.update(traj.extra_non_tensor.keys())

for key in extra_keys:
    values = [traj.extra_non_tensor.get(key, None) for traj in all_trajectories]
    dataproto.non_tensor_batch[key] = np.array(values, dtype=object)
```

#### 改动 4：同样修改 `trajectory_buffer.py` 和 `step_buffer.py`

保持三种 buffer 行为一致，便于未来切换。

#### 改动 5：张量字段也需要检查

虽然 VLM 图片主要走 non_tensor 路径，但某些 env 可能在 `batch.batch` 里
加自定义 tensor（如 pre-computed embeddings）。**建议不做硬性保留**，
否则会浪费内存；改为在 YAML 配置中声明需要额外保留的 tensor key：

```yaml
replay:
  preserve_tensor_keys: []  # 默认空；VLM 如需要可列 ["pixel_values"]
```

### 4.3 兼容性保证

| 场景 | 改动后的行为 |
|------|-------------|
| 文本 GRPO (当前 AIME) | `extra_non_tensor` 为空 dict，与原行为完全一致 |
| VLM with `add_extra_data` | `multi_modal_data`/`message`/`image_PIL` 等字段自动保留 |
| 自定义 env 新增字段 | 无需改 buffer 代码，自动兼容 |

### 4.4 需要迁移的旧版补丁

从 `bandit-upstream-v2` commit `c5e876ad` 中提取：

1. **Qwen3-VL encoder image token mismatch 修复**：涉及
   `roll/algorithms/bandit/encoder_actor.py`，但新版已无 bandit，
   需判断 upstream 主干是否已经修复相同问题；如未修，移植到
   对应的 actor path。

2. **`vl_traj_env_manager.py` 的几处小修**（c5e876ad diff 显示）：
   - `pre_step_template` / `next_step_template` 改为 optional（单轮 math VLM）
   - `obs.get("prompt", obs.get("text", ""))` 容错
   - 自动把纯文本 obs 包成 `[{"type":"image"}, {"type":"text"}]` 
   - `system_prompt_override` 支持

   **建议**：这些小修在上游可能已通过其他 PR 修复，迁移前先 `git diff`
   当前上游版本与旧版本，只 cherry-pick 上游确实缺失的部分。

## 5. 实施分阶段

### Phase 1：通用化非张量字段（必须）
- [ ] 修改 `TrajectoryEntry` 加 `extra_non_tensor`
- [ ] 修改 `group_buffer._extract_trajectory_entry` 动态抓取
- [ ] 修改 `group_buffer.sample_for_training` 动态还原
- [ ] 同步改 `trajectory_buffer.py` / `step_buffer.py`
- [ ] 跑 AIME math exp1 回归测试（确保文本场景无影响）

### Phase 2：VLM 小环境验证（推荐）
- [ ] 用最简 VLM env（如 DeepEyes 或一个 sokoban VL 变体）跑 replay
- [ ] 确认训练 loss 曲线合理、图片 pixel_values 正确重建
- [ ] 对比 baseline（无 replay）和 replay 两条曲线

### Phase 3：Qwen3-VL 相关修复迁移（按需）
- [ ] 查 upstream 当前 `encoder_actor.py` / VLM 训练 path 是否有 image token
      mismatch 问题
- [ ] 若有，从 `c5e876ad` 提取对应 hunk 做 cherry-pick

### Phase 4：配置层扩展（可选）
- [ ] `ReplayConfig` 加 `preserve_tensor_keys: List[str]` 字段
- [ ] buffer 按配置保留额外 tensor

## 6. 风险与已知问题

### 6.1 内存消耗

`extra_non_tensor` 里如果含 PIL Image 对象（VLM messages），单条轨迹的内存
占用会显著高于纯文本场景。建议：

- 调小 `replay.capacity`（VLM 场景 500-2000 groups 比较合理）
- 或实现 `lazy_load` 模式：图片以路径/bytes 形式存，采样时按需 decode

### 6.2 跨 Ray worker 序列化

`non_tensor_batch` 的 PIL 对象在 Ray 传输时走 pickle。**实测上游已验证可行**
（rollout 本身就会 pickle 这些对象），我们的 buffer 不会引入新的序列化问题。

### 6.3 `messages_list` 的 PIL 图片生命周期

需确认 PIL Image 对象被 numpy object array 持有后，不会被 GC。目前通过
`np.array([...], dtype=object)` 存储保留了强引用，这条**不需要额外处理**。

## 7. 回归测试清单

修复完成后，必须通过以下测试：

1. [ ] AIME text GRPO baseline（exp1）：训练曲线与修复前完全一致
2. [ ] AIME text GRPO + replay uniform（exp3）：可正常训练 50+ steps
3. [ ] VLM smoke test：某个 VLM env 能完成 push + sample + train 一轮
4. [ ] 单测：`test_group_buffer_vlm.py` 模拟 VLM non_tensor 字段并验证采样一致性

## 8. 参考

- 旧版 VLM 实现：`bandit-upstream-v2` 分支 commit `c5e876ad`
- upstream VLM env manager：`roll/pipeline/agentic/env_manager/vl_traj_env_manager.py`
- upstream VLM tokenizer：`roll/pipeline/agentic/env_manager/token_mask_utils.py:39`
- 当前 GroupReplayBuffer：`roll/pipeline/agentic/replay_buffer/group_buffer.py`
- 设计文档：`roll/pipeline/agentic/replay_buffer/docs/group_replay_buffer_design.md`
