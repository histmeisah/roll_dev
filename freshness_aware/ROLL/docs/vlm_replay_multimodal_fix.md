# VLM Replay 多模态字段丢失问题 — 调研与修复方案

## TL;DR

`GroupReplayBuffer` 在持久化 rollout 数据时**静默丢弃了 `non_tensor_batch["multi_modal_inputs"]`**（即 VLM 的 `pixel_values` / `image_grid_thw` 等 tensor 字段的宿主）。

由于下游 strategy 读取时做了 `if "multi_modal_inputs" in ...` 判空，**replay train_step 不崩，只是静默降级为纯文本训练**——VL 模型拿到 input_ids（其中 `<image>` 位置是 placeholder token）但**没有视觉 embedding 进入 forward**。那一步的梯度对 VL 权重是噪声，不是正确训练信号。

修复：`_extract_trajectory_entry` 增加对 `multi_modal_inputs` 的抓取，`sample_for_training` 采样时按 group 顺序写回 `replay_batch.non_tensor_batch`。改动约 20 行，全在 `group_buffer.py`。

---

## 问题背景

`roll/pipeline/agentic/replay_buffer/group_buffer.py:210-213` 的 `TODO(VLM)` 注释承认"多模态 tensor 未持久化"。但 pipeline 侧的 exp5 / vlm_fresh_per 运行时**能 pipeline_complete**，不崩。那它到底发生了什么？

---

## 调研链路

### 1. VLM rollout 阶段 pixel_values 的真实位置

`roll/pipeline/agentic/env_manager/vl_traj_env_manager.py:288-406` 的 `format_messages` 里：

- L312: PIL Image 收集进 `mm_data["image"]`，**与 `messages` 分开**
- L353: `messages` 里只有文字 dict + text placeholder token，PIL 图**不在 messages 里**
- L389: `self.collator([feature])` 在 **rollout 阶段就调用**，processor 把 PIL 处理成 tensor

关键在 collator：

### 2. Collator 的输出路径（`roll/datasets/collator.py:136-240`）

```python
# processor 产出的 pixel_values / image_grid_thw 是 torch.Tensor
processor(images=..., text=prompt)
# 每 sample 的 tensors 组成一个 dict，append 到 list
un_padded_features["multi_modal_inputs"].append(dict(model_inputs))
# list 被 wrap 成 np.array(dtype=object)
batch["multi_modal_inputs"] = np.array(list_of_dicts, dtype=object)
```

**设计原因**（collator L144-145 注释）：`pixel_values` 每 sample 的 `n_patches` 不同，没法 stack 进 TensorDict，所以塞进非 tensor 字段。

### 3. DataProto 分流（`roll/distributed/scheduler/protocol.py:265-319`）

`DataProto.from_single_dict` 看每个 key：

```python
if isinstance(val, torch.Tensor):       → batch.batch (TensorDict)
elif isinstance(val, np.ndarray):       → non_tensor_batch (dict of object arrays)
```

`multi_modal_inputs` 是 `np.array(dtype=object)`，**进 `non_tensor_batch`**。

> **修正了本次调研最初的误判**：第二轮探测里说 "pixel_values 存在 batch.batch 里"，实际是 rollout batch 的 **non_tensor_batch**。

### 4. On-policy 训练如何消费（`hf_strategy.py` / `deepspeed_strategy.py`）

```python
if "multi_modal_inputs" in data.non_tensor_batch:
    multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
    multi_modal_data = defaultdict(list)
    for sample_mm_inputs in multi_modal_inputs:          # iterate per sample
        for key in sample_mm_inputs.keys():              # "pixel_values", "image_grid_thw" ...
            multi_modal_data[key].append(sample_mm_inputs[key])
    for key in multi_modal_data:
        forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(device)
model(**forward_args)
```

### 5. GroupReplayBuffer 现状

`_extract_trajectory_entry` 只白名单式提取 9 个 non_tensor 字段：`env_ids / group_ids / messages_list / tags / frames / step_scores / episode_scores / traj_group_id / traj_id`。

**`multi_modal_inputs` 不在列表里 → 入库时丢弃 → 采样后 `replay_batch.non_tensor_batch` 缺这个字段**。

### 6. Replay train 时发生了什么

回到第 4 步，`if "multi_modal_inputs" in data.non_tensor_batch` 的判空命中 **else 分支 → forward_args 里不含 pixel_values → 模型只拿 input_ids 前向**。

对 Qwen2.5-VL：`<image>` 占位 token 在 input_ids 里存在（这部分 token 被保留了），但没有对应的视觉特征注入。模型会把它当普通 token 处理，forward 出一个语义错位的输出，loss 仍能正常算，反传正常——**不崩，但对 VL 权重的梯度信号是污染**。

这就是为什么 vlm_fresh_per 那个 85G run **能走到 pipeline complete**。

---

## 影响面

| 实验 | 是否启用 replay | 是否 VLM | 本次修复前 PHASE 15 行为 |
|---|---|---|---|
| grpo_aime / exp1-3 | 视配置 | ❌ | 正常（无多模态） |
| grpo_aime / exp4 / exp5 | ✅ | ❌ | 正常（无多模态） |
| vlm_frozen_lake / vlm_fresh_per | ✅ | ✅ | **静默降级为 text-only train**（污染） |
| 任何其他 VLM + replay 的实验 | ✅ | ✅ | 同上 |

---

## 修复方案

### 核心改动

`roll/pipeline/agentic/replay_buffer/group_buffer.py`

**A. `TrajectoryEntry` 增加字段**（借用 `trajectory_buffer.py` 的同名结构）
```python
multi_modal_inputs: Optional[dict] = None    # per-trajectory dict, keys: pixel_values, image_grid_thw, ...
```

**B. `_extract_trajectory_entry` 抓取**
```python
mm_inputs_arr = batch.non_tensor_batch.get("multi_modal_inputs", None)
multi_modal_inputs = mm_inputs_arr[idx] if mm_inputs_arr is not None else None
```

**C. `sample_for_training` 回写**
```python
mm_list = [t.multi_modal_inputs for t in all_trajectories]
if any(x is not None for x in mm_list):
    dataproto.non_tensor_batch["multi_modal_inputs"] = np.array(mm_list, dtype=object)
```

### 为什么这个方案够简单

1. **字段本身就是 object array**：每元素是 dict，dict 里才有 tensor。GroupBuffer 存的时候就是 Python 对象，不涉及 tensor stack / pad / pack 的边界问题。和 `messages_list` 的存储机制完全一致。
2. **不需要改 env_manager**：rollout 阶段 collator 已经把多模态处理完了，我们只需把结果搬运下去。
3. **不需要改 pipeline**：strategy 层已经做了 `if ... in` 判空，只要 non_tensor_batch 里有这个字段，多模态 forward 就自动启用。
4. **对非 VLM 路径零影响**：text-only 的 batch 里压根没 `multi_modal_inputs`，抓取时 `arr is None`、回写时 `any(x is not None)` 为 False，整个分支不触发。

### 改动边界

- 只改 `group_buffer.py`（约 20 行）
- `trajectory_buffer.py` / `step_buffer.py` 如有类似问题可参照移植（目前 VLM 实验默认 `group_level=true`，优先级低）
- 不改 `agentic_pipeline.py`、`vl_traj_env_manager.py`、`collator.py`、`protocol.py`

---

## 验证计划

1. `py_compile` 本地 → 服务端
2. 跑一次 `vlm_fresh_per`（50-100 步即可），抓 PHASE 15 的 debug 日志，确认 `multi_modal_inputs` 已在 replay_batch 里
3. 对比有无修复下的 VL 权重梯度分布：应有显著差异（污染 vs 正常训练信号）

---

## 相关文件

- `roll/pipeline/agentic/replay_buffer/group_buffer.py` — 本次修改点
- `roll/pipeline/agentic/env_manager/vl_traj_env_manager.py:288-498` — VLM rollout 源
- `roll/datasets/collator.py:136-240` — multimodal 打包
- `roll/distributed/scheduler/protocol.py:265-319` — DataProto 分流
- `roll/distributed/strategy/hf_strategy.py` / `deepspeed_strategy.py` — 消费侧 `if ... in` 判空
