
  🟢 P0 - 必做(高 rebuttal 价值,中等算力)

  ┌─────┬──────────────────────┬────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────┬────────────────────────────┐
  │  #  │         实验         │                                  现有配置                                  │           需要做的事            │            算力            │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │     │ Sokoban Simple       │ ✅                                                                         │                                 │                            │
  │ 1   │ FreshPER τ=1000 多   │ sokoban_2a100_replaybuffer/sokoban_traj_reward_fresh_configA_age1000.yaml  │ 改 seed 跑 2-3 次               │ 0.5B 模型,2×A100,~12h/seed │
  │     │ seed                 │                                                                            │                                 │                            │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ 2   │ AIME FreshPER 多     │ ✅ math_aime_2a100_replaybuffer/ 或 math_aime_8h100_replaybuffer/          │ 改 seed 跑 2-3 次               │ 7B 模型,2×A100 或          │
  │     │ seed                 │                                                                            │                                 │ 8×H100,~24h/seed           │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ 3   │ FreshPER + GRPO on   │ ✅ 已有完整配置:grpo_aime/exp5_grpo_replay_fresh_per.yaml                  │ 直接跑 + 配 baseline            │ 7B,8×H100,~24h             │
  │     │ AIME                 │                                                                            │ exp1_grpo_baseline.yaml         │                            │
  └─────┴──────────────────────┴────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────┴────────────────────────────┘

  ▎ #3 是最高性价比:配置已写好(grpo_aime/ 下 5 个 exp 配套),直接覆盖 QUzc + 7Zqh 的"REINFORCE++ only"批评

  🟡 P1 - 强烈推荐(若算力充足)

  ┌─────┬─────────────────────────────────────────────┬────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
  │  #  │                    实验                     │                  现有配置                  │                            备注                             │
  ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ 4   │ Sokoban Simple Standard PER 多 seed(对照组) │ ✅ sokoban_traj_advantage_per_configA.yaml │ 配合 #1,multi-seed band 直接展示 PER vs FreshPER 鲁棒性差距 │
  ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ 5   │ Sokoban Hard FreshPER 多 seed               │ 需查或仿照 simple 版改                     │ 论文 -0.51 vs -0.84 的提升幅度小,需 error bar 才有说服力    │
  ├─────┼─────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ 6   │ AIME GRPO baseline + GRPO+Standard PER 对照 │ ✅ grpo_aime/exp1 + exp4                   │ 配合 #3,完整 GRPO 对照三件套                                │
  └─────┴─────────────────────────────────────────────┴────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘

  🟠 P2 - 加分项(可选)

  ┌─────┬───────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────┐
  │  #  │                 实验                  │                                   现状                                   │                     备注                      │
  ├─────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ 7   │ FreshPER + PPO on Sokoban             │ 需创建配置(改 sokoban_traj_reward_fresh_configA_age1000.yaml 的 algo 为  │ 配合 #3 形成"PPO + GRPO + REINFORCE++         │
  │     │                                       │ PPO)                                                                     │ 三算法验证"                                   │
  ├─────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ 8   │ Adaptive τ POC(Sokoban 或 LLM         │ 需小代码改动:τ = log(2) / observed_KL                                    │ QUzc τ 启发式诉求                             │
  │     │ FrozenLake)                           │                                                                          │                                               │
  ├─────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ 9   │ GRPO + FreshPER on Sokoban            │ 仿 exp5_grpo_replay_fresh_per.yaml 改 env                                │ 0.5B 模型,比 AIME 便宜很多                    │
  └─────┴───────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────┘

  🔴 不做(确定)

  ┌──────────────────────────┬──────────────────────────────┐
  │           实验           │             原因             │
  ├──────────────────────────┼──────────────────────────────┤
  │ ~~NQ Search 任何新实验~~ │ 用户明确算力不够             │
  ├──────────────────────────┼──────────────────────────────┤
  │ ~~RLEP 严格模式~~        │ 机制等价数学证明已足够       │
  ├──────────────────────────┼──────────────────────────────┤
  │ ~~Fatemi 复现~~          │ 它不是 buffer                │
  ├──────────────────────────┼──────────────────────────────┤
  │ ~~VLM GeoQA multi-seed~~ │ GeoQA +1.3% 本身弱,救不了 W5 │
  └──────────────────────────┴──────────────────────────────┘

  ---
  推荐执行顺序

  │ 8   │ Adaptive τ POC(Sokoban 或 LLM         │ 需小代码改动:τ = log(2) / observed_KL                                    │ QUzc τ 启发式诉求                             │
  │     │ FrozenLake)                           │                                                                          │                                               │
  ├─────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ 9   │ GRPO + FreshPER on Sokoban            │ 仿 exp5_grpo_replay_fresh_per.yaml 改 env                                │ 0.5B 模型,比 AIME 便宜很多                    │
  └─────┴───────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────┘
          
  🔴 不做(确定)                                                                  
                                              
  ┌──────────────────────────┬──────────────────────────────┐
  │           实验           │             原因             │
  ├──────────────────────────┼──────────────────────────────┤
  │ ~~NQ Search 任何新实验~~ │ 用户明确算力不够             │
  ├──────────────────────────┼──────────────────────────────┤
  │ ~~RLEP 严格模式~~        │ 机制等价数学证明已足够       │
  ├──────────────────────────┼──────────────────────────────┤
  │ ~~Fatemi 复现~~          │ 它不是 buffer                │
  ├──────────────────────────┼──────────────────────────────┤                        
  │ ~~VLM GeoQA multi-seed~~ │ GeoQA +1.3% 本身弱,救不了 W5 │                                                   
  └──────────────────────────┴──────────────────────────────┘                   

---

# 实施日志:P0 #1 Sokoban Simple multi-seed 阻碍分析(2026-05-04)

## TL;DR

P0 #1 多 seed run 跑通了**整条 SLURM/ROLL/replay buffer pipeline**,但 4 个 chained run 中 3 个完成、全部 **400 步零学习**:`action_is_valid=0`、`success=0`、`pg_loss=0`、`grad_norm=0` 始终为 0。原因不是单点 bug,是 **0.5B 弱模型 + 严格 parser + 稀疏 reward + 非确定性采样** 四者叠加造成的 "bootstrap 二元结局" — 同 yaml 同 seed 在不同硬件/库版本下,要么模型早期偶然产出 `<answer>Right</answer>` 触发学习(历史 28% 成功),要么从未触发(我们的现状)。

## 历史 vs 当前对比

5 次历史 run(都用同一份 yaml,seed=42):

| run | 时间 | 硬件 | step | action_is_valid | success | val_score_max |
|---|---|---|---|---|---|---|
| 08qz0xao | 2026-02-09 04:12 | A100 | 399 | **1.0** | 7.8% | 10.9 |
| d9qrat01 | 2026-02-09 05:44 | A100 | 399 | **1.0** | **28%** | 10.9 |
| posn0bjl | 2026-02-11 03:36 | A100 | 399 | **1.0** | 25% | 10.9 |
| 8ebrzp3h | 2026-02-11 05:36 | A100 | 359 | **0** | 0 | -1.0 |
| d11pcwqm | 2026-03-01 22:16 | A100 | 399 | **0** | 0 | -1.0 |

我们当前(2026-05-04, H200, vLLM 0.10.2):

| run | step | action_is_valid | success |
|---|---|---|---|
| 24268 reward_fresh seed43 | 399 | 0 | 0 |
| 24269 reward_fresh seed44 | 399 | 0 | 0 |
| 24270 advantage_per seed43 | 399 | 0 | 0 |

历史成功率 3/5(同 yaml),失败的 2 次和我们一样:**`action_is_valid` 永远为 0**。

## 核心机制:为什么会卡在零学习

```
模型 random init → 一次 rollout 8 trajectories × 10 steps
                ↓
每一步:模型生成 response,parser 用 `<answer>(.*?)</answer>` 提取动作
                ↓
没匹配 → action=None → SokobanEnv no-op,action_is_valid=False,reward=-1
匹配但内容不在 {Up,Down,Left,Right} → 同上
匹配且内容合法 → 真正执行,大概率 reward=-1(没解)+ 极小概率 reward=10.9(解了)
                ↓
全部 -1 → reward_normalization (mean/std) 后 advantage 全 0
       → pg_loss = E[A * log_prob] = 0
       → grad_norm = 0
       → 模型不更新
       → 下一步同样的 random 行为,recursion
```

**bootstrap 触发条件**:模型至少要在 400 步内偶然产出**一次** `<answer>X</answer>`(X∈{Up,Down,Left,Right}),且因为这是 **rollout group of size 2 + reward whitening**,需要同 group 内有差异,才能产生非零 advantage,才能开始学习。

0.5B Instruct 模型对 system prompt 中 "Output must be wrapped as <answer>Action</answer>" 的遵循度,**完全取决于采样链路的 floating-point 行为**:
- A100 + vLLM 0.8.4 + transformers 4.51:3/5 概率早期触发 → 学到 25-28% 成功
- H200 + vLLM 0.10.2 + transformers 4.57:0/3 概率触发 → 零学习

这不是 bug — 是该实验设计对 stochastic bootstrap 的**脆弱依赖**。

## 诊断过程(为留档)

按时间顺序排查的 fail-fast 阶段:

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | 登录节点跑 4 run × 30s 全崩,raylet `dashboard_agent failed` | 登录节点 `nvidia-smi` 是 0 字节空文件,Ray 启动校验抛 `OSError`,torch 也看不到 GPU(无 `/dev/nvidia*`) | 用 SLURM `sbatch` 提交到 GPU worker |
| 2 | 4 个 SLURM job 14s 全 FAILED | `source .../activate` 没传 env 名,`$1`(CONFIG_NAME)被当 env 名 | `source ... activate roll` 显式传参 |
| 3 | Ray cluster 撞名 / `available_gpu=0` | 4 job 抢同一节点的 6379/8265 端口 + `/tmp/ray/` | 改 `--dependency=afterany` 串行 |
| 4 | conda activate 通过 / pi-elhosemh 4 job × 5 min 全崩 | hydra `initialize()` 拒绝绝对路径 config_path | 改回相对路径 `../../experiments/...` |
| 5 | hydra 通过 / 6 min 崩在 env worker create | `pkg_resources` import 失败:setuptools 82.0.1 已删 | `pip install 'setuptools<81'` → 80.10.2 |
| 6 | env init 通过 / 5 min 崩在 SokobanEnv | `dim_x/dim_y` 已被 ROLL 改成 `dim_room: [6,6]` | yaml 4 处批量改 |
| 7 | Sokoban 创建通过 / rollout 启动崩 `Missing key max_steps` | yaml 把 `max_steps` 只放在 `env_config:` 下,顶层缺失 | yaml 4 处补顶层 `max_steps: ${max_actions_per_traj}` |
| 8 | rollout 启动通过 / `make_decision` 崩 `KeyError: 'state'` | ROLL `format_messages` 提供 `{observation}`,我们 yaml 用 `{state}` | yaml 4 处 `{state}` → `{observation}` |
| 9 | step 0 完成 / `compute_token_reward` 崩 `bool tensor 减法` | replay buffer 用 `dtype=torch.bool` 重建 attention_mask,而 `agentic_pipeline.py` 用 `torch.zeros_like(attention_mask)` 创建 `old_log_probs`/`ref_log_probs` → 两个 bool tensor 相减,PyTorch 2.x 拒绝 | 改 2 处 `zeros_like(..., dtype=torch.float32)` + `compute_approx_kl` 入口防御 cast |
| 10 | pi-elhosemh 队列 21h 等待 | 同 PI 组 feij0a 用满 64 CPU/4 GPU 配额 | 切到 batch-h200 (qos=batch),无队列直接 RUN |
| 11 | **训练通跑 400 步,零学习** | **本节核心问题** | 待修(见下) |

## 修复方案

3 选 1 或叠加:

**A. parser fallback(改 `roll/pipeline/agentic/env/parse_action_utils.py`)**
当 `<answer>` 正则失败时,把 stripped 文本直接和 `action_lookup` 比对(case-insensitive)。
- 优点:历史 working runs 完全不受影响(它们 regex 命中,走原路径);unlucky runs 能识别裸 `Right` 进入学习
- 缺点:违反"必须输出标签"的设计意图(但目前这个意图本身就让实验脆弱)

**B. env_config 加 `format_penalty: -0.15`(yaml 改动)**
让格式不对的 turn 至少有 -0.15 reward,哪怕 action 永远不 valid,也存在 reward 方差。
- 现状:env_config 里没设 → SokobanEnv 默认 0 → 没惩罚 → reward 全 -1 完全等价
- 注:yaml 顶层 `train_env_manager.format_penalty: -0.15` 是 env_manager 层面,**不传到 SokobanEnv**

**C. 换 7B 模型(本地有 `Qwen2.5-7B-Instruct`)**
- 优点:7B Instruct 对 system prompt 格式遵循度远高于 0.5B,bootstrap 概率从"看运气"变成"近确定"
- 缺点:H200 80GB 装得下但每 run 显存压力大,2× H200 跑可能要降 batch 或 sequence_length;成本约 ×10-14

**推荐:A + B 同改,模型保持 0.5B**(rebuttal 价值在于复现作者声称的 0.5B 结果,换 7B 等于改实验设置)。

## 已落地的代码/配置变更(可保留)

下面这些不是引入的问题,是把环境/接口对齐到当前 ROLL 的必要修复,无需回滚:

- `experiments/sokoban_hard_2a100_replaybuffer/`:
  - `sbatch_one_seed.sh`(新增,SLURM 单 seed 模板,`batch-h200 qos=batch --gres=gpu:2`)
  - `submit_4_seeds.sh`(新增,`--dependency=afterany` 串行 4 job)
  - 4 个 seed yaml: `dim_x/dim_y → dim_room: [6, 6]`,顶层加 `max_steps`,`{state} → {observation}`,`pretrain` 指向本地 0.5B
- `ROLL/roll/utils/functionals.py`:`compute_approx_kl` 入口对 bool tensor 防御性 cast 到 float
- `ROLL/roll/pipeline/agentic/agentic_pipeline.py`:2 处 `zeros_like(attention_mask)` 加 `dtype=torch.float32`
- env: `pip install 'setuptools<81'`(80.10.2)恢复 `pkg_resources`

## 待决策

1. **修复方案 A/B/C 三选一**(或 A+B 叠加),然后重跑 P0 #1 的 4 seed chain
2. **要不要顺手 sokoban_hard 跑同样 multi-seed**(P1 #5),因为 yaml 已就绪、所有底层 bug 已修
3. AIME 任务(P0 #2, #3)不依赖 Sokoban,bootstrap 问题可能不存在(7B + 数学题 reward 不那么稀疏);可以**并行启动 P0 #3** 不等 Sokoban 修完
