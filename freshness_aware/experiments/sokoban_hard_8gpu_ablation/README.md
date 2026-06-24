# Sokoban Hard 8GPU Ablation

Four compute-matched runs for Sokoban hard with Qwen3-8B:

| Run | Script | Config |
| --- | --- | --- |
| Reinforce baseline | `run_reinforce_baseline.sh` | `sokoban_hard_reinforce_baseline_qwen3_8b_8gpu` |
| GRPO baseline | `run_grpo_baseline.sh` | `sokoban_hard_grpo_baseline_qwen3_8b_8gpu` |
| Reinforce + FreshPER | `run_reinforce_freshper.sh` | `sokoban_hard_reinforce_freshper_qwen3_8b_8gpu` |
| GRPO + FreshPER | `run_grpo_freshper.sh` | `sokoban_hard_grpo_freshper_qwen3_8b_8gpu` |

Follow-up runs after the 400-step results showed replay underperforming the ppo2
baseline:

| Run | Script | Config |
| --- | --- | --- |
| Reinforce ppo2 + Reward-FreshPER + filter | `run_reinforce_ppo2_reward_fresh_filter.sh` | `sokoban_hard_reinforce_ppo2_reward_fresh_filter_qwen3_8b_8gpu` |
| Reinforce ppo2 + KL-FreshPER + filter | `run_reinforce_ppo2_kl_fresh_filter.sh` | `sokoban_hard_reinforce_ppo2_kl_fresh_filter_qwen3_8b_8gpu` |
| Reinforce old hard recipe reproduction | `run_reinforce_oldrecipe_reward_fresh.sh` | `sokoban_hard_reinforce_oldrecipe_reward_fresh_qwen3_8b_8gpu` |

Baseline runs disable replay and use `ppo_epochs=2`.

Replay buffer type is explicit:

- Reinforce replay configs use `replay.group_level: false` and sample trajectories.
- GRPO replay configs use `replay.group_level: true` when group-preserving replay is needed.
- The shared base defaults to `replay.group_level: false` to avoid accidental group replay inheritance.

FreshPER runs use the current KL-FreshPER implementation:

```yaml
replay.enabled: true
replay.group_level: false  # Reinforce; use true only for GRPO group replay.
replay.priority_function: kl_fresh
replay.enable_age_decay: true
replay.use_engine_logprobs: true
ppo_epochs: 1
```

The recommended next run keeps the strong `ppo_epochs=2` on-policy update and
adds one filtered reward-fresh replay update:

```yaml
ppo_epochs: 2
replay.priority_function: reward_fresh
replay.train_steps_per_env_step: 1
replay.enable_age_decay: true
replay.age_decay: 300
replay.enable_offpolicy_filter: true
replay.ratio_clip_max: 5
```

Run one experiment on the Wuwen 8-GPU shell:

```bash
tmux new -s roll
source /mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/project_modelware_roce/zhaojian/envs/roll
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/sokoban_hard_8gpu_ablation
bash run_reinforce_baseline.sh
```

Recommended next submission:

```bash
bash run_reinforce_ppo2_reward_fresh_filter.sh
```

For a non-interactive Wuwen job command, do not start `tmux`; submit the shell
body directly:

```bash
source /mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/project_modelware_roce/zhaojian/envs/roll
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/sokoban_hard_8gpu_ablation
bash run_reinforce_baseline.sh
```

Run all four sequentially:

```bash
bash run_all_four.sh
```

Sync the latest offline W&B run:

```bash
export WANDB_API_KEY="<your-key>"
bash sync_latest_wandb.sh
```

Outputs are written under:

```text
/mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/sokoban_hard_8gpu_ablation/output
```
