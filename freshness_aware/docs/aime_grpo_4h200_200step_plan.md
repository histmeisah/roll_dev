# 4-H200 AIME GRPO 200-Step Plan

## Goal

Use public `freecycle-h200/freecycle` resources to run a clean 4-H200 AIME GRPO comparison for:

1. GRPO baseline with matched update budget
2. GRPO + reward PER
3. GRPO + FreshPER

Each run should train for at least 200 steps, use the current fixed replay/log-prob code path, and produce clean logs with no checkpoint teardown failure or behavior-log-prob fallback.

## Main Comparison

| Run | Config name | Replay | Priority | Age decay | Update budget |
|---|---|---|---|---|---|
| Baseline | `aime_grpo_4h200_baseline_ppo2_200step` | disabled | none | none | `ppo_epochs=2` |
| PER | `aime_grpo_4h200_reward_per_200step` | enabled | `reward` | disabled | `ppo_epochs=1` + 1 replay step |
| FreshPER | `aime_grpo_4h200_reward_fresh_per_200step` | enabled | `reward_fresh` | enabled, `age_decay=1000` | `ppo_epochs=1` + 1 replay step |

Why baseline uses `ppo_epochs=2`: PER/FreshPER do one on-policy actor update and one replay actor update per env step. A ppo-epochs-1 no-replay baseline is useful as an anchor, but it is not the fairest compute-matched baseline.

## 20-Step Tuning Result

The 4-H200 probe on public `freecycle-h200/freecycle` completed cleanly with:

- Job `25996`: `aime_grpo_4h200_freshper_tune_b192_3t1i_ga16_20step`
- Layout: 3 actor-train GPUs + 1 vLLM GPU
- `rollout_batch_size=192`, `num_env_groups=48`, `group_size=4`
- `per_device_train_batch_size=2`, `gradient_accumulation_steps=16`
- Slurm result: `COMPLETED 0:0`, elapsed `00:37:34`
- Clean log: no `actor_train did not return log_probs`, no train-infer correction fallback, no traceback, no OOM
- Warmup-excluded averages: `old_log_probs=10.44s`, `train=30.83s`, `replay_train=41.23s`, `system/tps=1484.32`

The slower 2-train/2-vLLM probe with `rollout_batch_size=256` also completed cleanly, but it was train-bound: job `25990` took `00:59:20`, with warmup-excluded `system/tps=1159.67` and train/replay phases around `56s/74s`. Use the 3-train/1-vLLM layout for the 200-step comparison.

## Shared 4-H200 Settings

Base these configs on the cleaned AIME GRPO/FreshPER config, not on the old failed 8-GPU run.

```yaml
num_gpus_per_node: 4
max_steps: 200
save_steps: 10000
logging_steps: 1
eval_steps: 10
checkpoint_config: null

rollout_batch_size: 192
val_batch_size: 60
sequence_length: 4096

adv_estimator: "grpo"
max_tokens_per_step: 512
max_actions_per_traj: 3

train_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 48
  group_size: 4
  num_groups_partition: [48]

val_env_manager:
  max_env_num_per_worker: 64
  num_env_groups: 60
  group_size: 1
  num_groups_partition: [60]
```

Recommended placement:

```yaml
actor_train:
  strategy_args:
    strategy_name: deepspeed_train
    strategy_config: ${deepspeed_zero2}
  device_mapping: list(range(0,3))
  training_args:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 16
  infer_batch_size: 4
  max_tokens_per_microbatch_in_train: 32768
  max_tokens_per_microbatch_in_infer: 32768

actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.92
      block_size: 16
      load_format: auto
  device_mapping: list(range(3,4))

reference:
  device_mapping: list(range(0,3))
  infer_batch_size: 4
  max_tokens_per_microbatch_in_infer: 32768
```

Use `actor_train.device_mapping: list(range(0,3))` for the train workers. With 3 train ranks, `per_device_train_batch_size=2`, and `gradient_accumulation_steps=16`, each actor-train or replay-train call still performs two optimizer steps for a 192-sample rollout batch. That keeps the update budget matched to the 2-train/2-vLLM b256 probe while improving throughput.

Reference may remain configured even when `enable_reference` resolves false from KL settings. Keep `init_kl_coef=0.0`, `use_kl_loss=false`, and `kl_loss_coef=0` to match the prior AIME GRPO setup.

## GPU Utilization Plan

The ROLL agentic pipeline is mostly synchronous: vLLM inference GPUs are busiest during rollout, while actor-train GPUs are busiest during log-prob recomputation and train/replay updates. This means per-GPU utilization will naturally be sawtooth rather than flat 95-100%. The goal is not perfect simultaneous utilization; the goal is avoiding long phases where either side is mostly idle because the batch is too small or the GPU split is badly balanced.

Recommended 4-H200 split from the completed 20-step probe:

| Role | GPUs | Expected busy phase | Main risk |
|---|---:|---|---|
| actor_train + old logprob + optional reference path | 0-2 | logprob, on-policy train, replay train | train side dominates wall time |
| actor_infer / vLLM | 3 | rollout and validation | bursty utilization is expected |

The tested `rollout_batch_size=192`, `num_env_groups=48`, `group_size=4`, and `max_tokens_per_step=512` keep the train side active without starving vLLM. Avoid copying the PPO 2-GPU probe settings here; `rollout_batch_size=8` was far too small for utilization and learning signal.

Utilization monitoring during the first 20-30 steps:

```bash
squeue -u maw0a -o '%i %j %T %M %l %D %R %P %q'
sstat -j <jobid>.batch --format=JobID,AveCPU,AveRSS,MaxRSS
srun --jobid=<jobid> nvidia-smi dmon -s pucm -c 20
```

Interpretation:

- If vLLM is active but train GPUs sit idle for long periods, rollout is the bottleneck. Keep the 3 train / 1 infer split, but consider reducing validation frequency or increasing `num_env_groups` only if env scheduling is the bottleneck.
- If train GPUs are active for much longer than rollout and vLLM GPUs sit idle, training/replay is the bottleneck. Keep the comparison fair, but do not move back to 2 train / 2 vLLM unless the 3 / 1 layout fails.
- If all GPUs show low utilization and CPU load is high, env workers or Ray scheduling are bottlenecking. Increase `max_env_num_per_worker` carefully, or reduce logging/trajectory dump overhead.
- If GPU memory is low but utilization is low, increase microbatch token limits cautiously: `max_tokens_per_microbatch_in_train` and `max_tokens_per_microbatch_in_infer`.

Do not change the GPU split between variants unless a run fails. Baseline, PER, and FreshPER should share the same 3-train / 1-infer layout so utilization differences do not become a confounder.

## Replay Settings

Shared PER/FreshPER replay settings:

```yaml
replay:
  enabled: true
  group_level: true
  capacity: 2000
  min_size: 48
  train_steps_per_env_step: 1
  use_rollout_batch_size: true
  sampling_mode: "trajectory"
  storage_mode: tokens_only
  lazy_tokenization: false
  eviction_strategy: "fifo"
  sample_method: "uniform"
  priority_exponent: 0.6
  importance_sampling_correction: true
  importance_beta: 0.4
```

PER:

```yaml
priority_function: "reward"
enable_age_decay: false
```

FreshPER:

```yaml
priority_function: "reward_fresh"
enable_age_decay: true
age_decay: 1000.0
refresh_interval: 1
```

Baseline:

```yaml
ppo_epochs: 2
replay:
  enabled: false
```

## Submission

Create `experiments/grpo_aime/sbatch_4gpu.sh` from the current AIME sbatch script:

```bash
#SBATCH --partition=freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=24:00:00
```

Submit commands:

```bash
sbatch --partition=freecycle-h200 --qos=freecycle --time=24:00:00 -J aime-grpo4-baseline experiments/grpo_aime/sbatch_4gpu.sh aime_grpo_4h200_baseline_ppo2_200step
sbatch --partition=freecycle-h200 --qos=freecycle --time=24:00:00 -J aime-grpo4-per experiments/grpo_aime/sbatch_4gpu.sh aime_grpo_4h200_reward_per_200step
sbatch --partition=freecycle-h200 --qos=freecycle --time=24:00:00 -J aime-grpo4-freshper experiments/grpo_aime/sbatch_4gpu.sh aime_grpo_4h200_reward_fresh_per_200step
```

If the cluster is fragmented, submit all three and let Slurm queue them. If only one 4-GPU slot is available, run FreshPER first, then PER, then baseline.

## Monitoring Checklist

Run these checks after startup and every 20-50 steps:

```bash
squeue -u maw0a -o '%i %j %T %M %l %D %R %P %q'
rg -n "Traceback|RuntimeError|CUDA out of memory|actor_train did not return|Skipping train-infer|pipeline step [0-9]+ finished|pipeline complete|replay/train_steps|replay/priority_fn|env/AIME/success|val/env/AIME/success" experiments/grpo_aime/output/<timestamp>/logs/*.log
```

Clean-run requirements:

- Slurm exits `COMPLETED 0:0`.
- No `Traceback`, `RuntimeError`, CUDA OOM, or `KeyError`.
- No `actor_train did not return log_probs`.
- No `Skipping train-infer correction`.
- PER logs `replay/priority_fn=reward_priority`.
- FreshPER logs `replay/priority_fn=reward_fresh_priority`.
- FreshPER logs nontrivial `replay/age/*` and `replay/freshness/mean < 1` after warmup.
- `offpolicy/reused_log_probs=1.0` and `replay/offpolicy/reused_log_probs=1.0` when monitor metrics are present.

## Metrics To Compare

Primary:

- `val/env/AIME/success`
- `val/env/AIME/action_is_valid`
- `rollout/score/mean`
- `env/AIME/success`
- `env/AIME/action_is_valid`

Replay/health:

- `replay/buffer_utilization`
- `replay/num_groups`
- `replay/priority/mean`, `max`, `min`
- `replay/age/mean`, `replay/freshness/mean`
- `replay/offpolicy/effective_sample_size_ratio`
- `replay/offpolicy/fraction_outside_ppo_clip_range`
- `actor_train/grad_norm`
- `actor/pg_loss@sum`

Suggested summary table:

| Run | Completed | final val success | best val success | final train success | best train success | final valid | replay groups | fallback count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | | | | | | | n/a | |
| PER | | | | | | | | |
| FreshPER | | | | | | | | |

## Expected Runtime

The completed 20-step 4-H200 probe with `rollout_batch_size=192` took `00:37:34` including model startup and warmup. Steady-state train time is about 83 seconds per step before evaluation overhead, so a 200-step run should be budgeted around 5-7 hours per job. Use `12:00:00` as the practical request, or `24:00:00` if the queue permits and we want maximum safety.

## Optional Follow-Up

If these three runs are clean and FreshPER is promising, launch two more seeds for the best two variants:

- seed 43
- seed 44

If PPO remains interesting, only compare it after matching the 4-H200 GRPO settings: `rollout_batch_size=192`, `max_actions_per_traj=3`, and at least 200 steps.
