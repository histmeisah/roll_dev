# aicoder_ib vs aicoder 功能对齐审计

**Audit scope**：以 aicoder_ib（`dev_12_26` 分支，`math_aime_8h100_replaybuffer` 效果好的参考实现）为 ground truth，系统检查当前 aicoder（`freshness_ReplayBuffer` 分支）在 pipeline / worker / config / buffer 四层有哪些功能**缺失**或**未集成**，不看我本轮修复过的位置，而是**横向对比文件夹**。

**背景**：之前几轮"认真检查"我只审查了自己改过的代码，默认 aicoder_ib 和 aicoder 其它位置一致，漏掉了关键 delta。这份文档补齐这个盲区。

---

## 概览：三层缺失

| 层 | aicoder_ib | 当前 aicoder | 差距性质 |
|---|---|---|---|
| **Loss（worker）** | `base_worker.py:loss_func` 消费 `importance_weights` | 完全没消费 | **关键 bug，单点根因** |
| **Filter（pipeline+utils）** | `filter_utils.py`（666 行）+ pipeline 调用 | 文件缺失，pipeline 未集成 | **第二层保护缺失** |
| **Monitor（pipeline）** | pipeline 调用 `offpolicy_monitor.compute_offpolicy_metrics` | 文件存在（md5 一致）但 pipeline 未调用 | **可观测性缺失** |
| **Hierarchical（可选）** | `hierarchical_config/computer.py`（1100 行）+ pipeline 集成 | 缺失 | **可选功能，AIME 实验不一定必需** |

---

## 1. 缺失文件

aicoder_ib 独有、当前 aicoder 缺失：

| 文件 | 行数 | 作用 |
|---|---|---|
| `roll/pipeline/agentic/filter_utils.py` | **666** | `filter_replay_batch_with_mini_batches`：按 IS ratio 阈值过滤极端 off-policy 样本（mini-batch 前向 + early-stop 保守抽样）|
| `roll/pipeline/agentic/hierarchical_config.py` | 296 | `HierarchicalRLConfig` dataclass |
| `roll/pipeline/agentic/hierarchical_computer.py` | 804 | `HierarchicalAdvantageComputer`：双层 GAE（step-level + token-level）|

当前 aicoder 独有（上游重构后新增）：
- `replay_buffer/` 子目录（原来在 `roll/agentic/replay_buffer/`，现在挪到 pipeline 内；内容我们已改过）
- `agentic_actor_worker.py` / `agentic_actor_pg_worker.py`
- `env_manager/` 下的 `base_env_manager.py`, `step_concat_env_manager.py`, `agent_native_env_manager.py`, `token_mask_utils.py`
- `env/` / `llm_proxy/` / `tools/` 子目录

重构本身不是问题；问题是重构**只搬了框架，没搬 replay 相关的训练时治理逻辑**。

---

## 2. Worker 层：`base_worker.py:loss_func` 的 IS correction

### aicoder_ib（L340-420，直接决定稳定性）

```python
# 在 pg_loss 计算之后
per_importance_weights = data.batch.get("importance_weights", None)

pg_loss = agg_loss(
    loss_mat=pg_loss,
    loss_mask=response_mask,
    loss_agg_mode=self.pipeline_config.loss_agg_mode,
    weights=per_importance_weights,        # ← 乘 IS weight
)

kl_loss = compute_approx_kl(...)
kl_loss = agg_loss(
    loss_mat=kl_loss,
    loss_mask=response_mask,
    loss_agg_mode=self.pipeline_config.loss_agg_mode,
    weights=per_importance_weights,        # ← 同样乘
)

# pg_metrics 末尾
if per_importance_weights is not None:
    pg_metrics["actor/per_importance_weights_mean"] = ...
    pg_metrics["actor/per_importance_weights_max"]  = ...
    pg_metrics["actor/per_importance_weights_min"]  = ...

# 返回 3 元组，第三项供 off-policy monitor 用
return total_loss, pg_metrics, {"log_probs": log_probs.detach()}
```

### 当前 aicoder（L248-359）

```python
pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask,
                   loss_agg_mode=self.pipeline_config.loss_agg_mode,
                   batch_num_tokens=batch_num_tokens['response_mask'],
                   global_valid_samples=global_valid_samples['response_mask'])
# ❌ 无 weights= 参数

kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, ...)
# ❌ 无 weights= 参数

# ❌ 整个函数没有任何 importance_weights 的引用
# ❌ 返回 2 元组：return total_loss, pg_metrics
```

**好消息**：`roll/utils/functionals.py:229` 的 `agg_loss` 签名已经有 `weights: Optional[torch.Tensor] = None`——上游已经保留了这个参数位，只是调用侧没传。所以修复**不需要改 agg_loss 本身**。

### 修复所需改动

- `base_worker.py` 中加 4 处：提取 `per_importance_weights`、两处 `agg_loss(..., weights=...)`、三行 metric
- 如果要让 off-policy monitor 能拿到 `log_probs`，还要改 return 三元组（牵涉调用链，量大）；但**监控缺失不影响训练稳定性**，这部分可以暂缓

### 影响

这是 exp4/exp5 `ratio_max@max` 飙到 1e9~1e19 的**单点根因**。IS correction 的设计就是给 PER 抽样的样本乘一个 `(N·P(i))^(-β)/max_w` 的权重——高 priority 样本概率大、权重小——**抵消**过度采样导致的加权偏差。现在权重被"生成但扔掉"，PER 抽到的"高 reward 样本"loss 里的权重远大于它们应有的比例，gradient 被过度放大，policy drift → ratio 爆炸。

---

## 3. Pipeline 层：`agentic_pipeline.py` 486 行差距

- aicoder_ib: 1931 行
- 当前 aicoder: 1445 行（含我本轮加的 PER 闭环、age decay、VLM round-trip、GroupBuffer stats）

差的 486 行主要是以下 5 个功能块：

### 3a. Off-policy filter 集成（~150 行，缺失）

aicoder_ib 的 pipeline 在 PHASE 15 采样循环内有一段：
```python
if (hasattr(rb_cfg, 'enable_offpolicy_filter')
        and rb_cfg.enable_offpolicy_filter
        and hasattr(rb_cfg, 'ratio_clip_max')
        and rb_cfg.ratio_clip_max is not None):
    from roll.pipeline.agentic.filter_utils import filter_replay_batch_with_mini_batches
    filter_result = filter_replay_batch_with_mini_batches(
        replay_buffer=self.replay_buffer,
        actor_train=self.actor_train,
        num_groups_to_sample=...,
        ratio_clip_max=rb_cfg.ratio_clip_max,
        ...
    )
    replay_batch = filter_result['batch']
```

**当前 aicoder 完全没这段**（`filter_utils.py` 文件也不在）。

Config schema 在 aicoder 的 `agentic_config.py:317-339` **已经有**这些字段：`enable_offpolicy_filter / ratio_clip_max / filter_mini_batch_size / filter_max_attempts / filter_oversample_ratio / filter_min_acceptable_batch`——但**pipeline 不读、`filter_utils.py` 不存在**，等于配置字段写了没人用。

### 3b. Off-policy monitor 调用（~80 行，缺失）

aicoder_ib 的 pipeline 里有 6 处调用 `compute_offpolicy_metrics` / `log_offpolicy_diagnostics`：
- PHASE 12 (fresh batch 算 old_log_probs 之后) 计算 fresh-side token-level ratio 分布
- PHASE 15 每次 replay sample 后、train 前算 replay-side ratio
- Pipeline 末尾 dump raw importance weights 到 metrics（`_raw_importance_weights` / `_raw_log_importance_weights` / `_raw_sample_importance_weights`）

当前 aicoder 的 pipeline 有 import（`try/except` 写了，前面调研过），但**实际没调用**。`offpolicy_monitor.py` 文件 md5 一致 → 函数在但没接线。

### 3c. `store_fresh_data_to_replay_buffer` 里的 behavior_log_probs 特殊处理（~50 行，缺失）

aicoder_ib：
```python
if actor_train_metrics.batch is not None and "log_probs" in actor_train_metrics.batch:
    batch.batch["behavior_log_probs"] = actor_train_metrics.batch["log_probs"]
    logger.debug("Attached training log_probs as behavior_log_probs for replay buffer storage")

self.store_fresh_data_to_replay_buffer(batch, global_step)
```

入库前把训练时算的 `log_probs` 保存为 `behavior_log_probs`（供将来采样时做 IS ratio 计算）。

当前 aicoder 的 PHASE 10 只直接 `push_from_dataproto(batch, ...)`，没 attach `behavior_log_probs`（GroupBuffer 的 extract 会发现缺字段，退而填 0）。这让后续 IS ratio 计算失去了正确的分母。

### 3d. Hierarchical RL 集成（~200 行，可选缺失）

aicoder_ib pipeline L74-89：
```python
if self.pipeline_config.hierarchical.enabled:
    validate_hierarchical_config(...)
    self.hierarchical_computer = HierarchicalAdvantageComputer(...)
```
和 PHASE 13 的双层 GAE 调度。

AIME 实验 yaml 没开这个，但当前 aicoder **连 config schema 都没有**（`HierarchicalRLConfig` 未 import）——如果你们要做 hierarchical RL 实验就会直接 AttributeError。

### 3e. `integrate_replay_buffer_data` 辅助方法（~100 行）

aicoder_ib 有一个专门的 helper 方法做 replay sample + balance + 准备 batch 的事；当前 aicoder 把这段逻辑平铺在 PHASE 15 里。属于组织差异，不是功能缺失。

---

## 4. Config 层：`agentic_config.py`（670 vs 654 行）

### aicoder 已有 ✅
- `OffPolicyMonitorConfig`（L136 附近）
- `ReplayConfig` 里的 `enable_offpolicy_filter` / `ratio_clip_max` / `filter_*` 字段（L317-339）
- `VTraceConfig`（L112）

### aicoder 缺失 ❌
- `HierarchicalRLConfig` import & `hierarchical: HierarchicalRLConfig` 字段（aicoder_ib 的 `AgenticConfig.hierarchical`）

所以 **config schema 的 filter 那套字段是齐的**，**只差 pipeline 实现和 filter_utils.py 文件**。hierarchical 需要同时补 config 和 impl。

---

## 5. Buffer 层（本轮已处理）

我本轮已改：
- GroupBuffer / TrajectoryBuffer / StepBuffer 的 `multi_modal_inputs` 持久化（VLM fix）
- GroupBuffer 的 `refresh_all_age_decay` 和 `update_priorities` 的 age-decay 分支
- GroupBuffer 的 `get_stats` 监控字段对齐

Pipeline 层本轮也加了 `_update_replay_priorities` / `_async_refresh_age_decay` / `_wait_age_decay_refresh` / `_extract_priority_signal` 4 个方法。

**这部分已经对齐 aicoder_ib 的对应行为**。

---

## 6. 严重程度 & 移植优先级

| 项目 | 代码量 | 是否为 exp4/exp5 爆炸根因 | 优先级 |
|---|---|---|---|
| `base_worker.py:loss_func` IS 消费 | ~10 行 | **✅ 单点根因** | **P0** |
| `store_fresh_data` 里 attach `behavior_log_probs` | ~15 行 | 影响 IS ratio 分母准确性，间接影响 | **P0** |
| `filter_utils.py` + pipeline 集成 | ~666+150 行 | 第二层保护，能进一步压 ratio 极端值 | P1 |
| `offpolicy_monitor` 在 pipeline 的调用 | ~80 行 | 纯可观测，不影响训练稳定性 | P2 |
| `hierarchical_*` 文件 + config | ~1100 行 | AIME 不需要 | P3（按需）|

---

## 7. 推荐的对齐计划

### Phase A — P0 修复（~1 小时，最小改动）
1. `base_worker.py:loss_func`：加 4 处 IS 消费 + 3 行 metric
2. `agentic_pipeline.py` PHASE 10：train_step 返回的 `log_probs` attach 为 `behavior_log_probs` 再 push
3. py_compile + scp + smoke run 一个短 config（10 step）验证 `actor/per_importance_weights_mean` 出现在日志里
4. 观察 `actor/ratio_max` 应该回落到百以内

### Phase B — P1 filter 移植（~半天）
1. scp `aicoder_ib:roll/pipeline/agentic/filter_utils.py` → 本地对应路径
2. 小改：aicoder_ib 的 filter 读 `rb_cfg.enable_offpolicy_filter` 等字段——当前 aicoder config 已经有这些字段，应该直接可用
3. pipeline PHASE 15：在 `sample_for_training` 之后加一段 `if enable_offpolicy_filter:` 分支，调用 filter_utils
4. 再起一次 smoke run 验证 ratio 被 filter 压低

### Phase C — P2 offpolicy_monitor（~1 小时）
从 aicoder_ib 移植 3 处调用进 agentic_pipeline（PHASE 12 fresh-side、PHASE 15 replay-side、末尾 raw_iw dump）。纯监控无风险。

### Phase D — P3 hierarchical（按需，~1 天）
移植 `hierarchical_config.py` / `hierarchical_computer.py` + 改 config schema + pipeline 集成。AIME 实验不需要，可延后。

---

## 8. 验证 checklist（每 Phase 完成后跑）

```bash
# 语法 + import
python -m py_compile roll/pipeline/{base_worker,agentic/agentic_pipeline,agentic/filter_utils}.py

# 服务端 md5 对比
md5sum ...     # 本地 vs 服务端

# 短 smoke run（10 step）观察 log 关键字
grep -E 'actor/per_importance_weights_mean|actor/ratio_max' run_log

# 期望：Phase A 后 ratio_max 应在 1e2~1e3 量级而非 1e9+
# 期望：Phase B 后 filter_kept_ratio 和 ratio_max 都有显著改善
# 期望：Phase C 后 wandb 里出现 off-policy monitor 曲线
```

---

## 9. 会话失误复盘（避免下次再漏）

**失误模式**：每一轮"再检查"时只审查本次改过的代码，没做 aicoder_ib 作为 ground truth 的横向对比。
**根本原因**：早期我自己就识别了"断点 3"（IS correction 未消费），当时说"这是下个迭代的事"，之后的 audit 把这个**已知未修项从 scope 里悄悄剔除了**。

**下次 audit 的强制项**（建议写进 `.claude/CLAUDE.md` 或 memory）：
1. 每次"认真检查"必须包含：**aicoder_ib 参考实现的横向对比**，不只看本次修改位置
2. 所有本会话中识别过但未修的"断点 / 风险项"必须在后续 audit 报告里显式列为未解决项，不能默认跳过
3. 有训练数据（不只是代码走查）时，**反推数值异常到代码缺失**，而不是给数值异常找外部解释（如"GRPO 本身不稳"）

---

## 10. 相关文件

- aicoder_ib（参考）：`/mnt/project_modelware/zhaojian/liangsirui/weiyu/local_roll_dev/roll_dev/ROLL/`
- 当前 aicoder：`/mnt/project_modelware_roce/zhaojian/weiyu/freshness_replaybuffer/ROLL/`
- 本地：`E:\code_project\python_code\local_roll_dev\roll_dev\ROLL\`
- 本次审计涉及的关键对照：
  - `base_worker.py:loss_func` 两侧对比（section 2）
  - `agentic_pipeline.py` PHASE 10/12/15 差异（section 3）
  - `agentic_config.py` schema 差异（section 4）
- 训练现象反证（让 audit 必要性可信）：`training_records/analysis_report.txt` 中 exp4/exp5 的 `ratio_max@max` 极端值
- 之前误诊文档：`docs/grpo_replay_offpolicy_instability.md`（根因归错到 GRPO，应修正为"与 aicoder_ib 功能未对齐"）

---

## 11. 还需要你决定的事

- Phase A / B / C / D 是否按顺序做？还是只做 P0 先回跑一次看效果？
- Hierarchical 要不要现在对齐（下个迭代你是否需要 hierarchical RL 实验）？
- 旧的 `grpo_replay_offpolicy_instability.md` 要不要我改：把根因从"GRPO vs REINFORCE"修正为"IS correction 未接通"？
- `CLAUDE.md` 里要不要加一条 hard rule："每次审查必须以 aicoder_ib 为 ground truth 横向对比，不只看本次修改"？
