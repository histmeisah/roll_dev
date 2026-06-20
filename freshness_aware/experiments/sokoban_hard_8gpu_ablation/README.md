# Sokoban Hard 8GPU Ablation

Four compute-matched runs for Sokoban hard with Qwen3-8B:

| Run | Script | Config |
| --- | --- | --- |
| Reinforce baseline | `run_reinforce_baseline.sh` | `sokoban_hard_reinforce_baseline_qwen3_8b_8gpu` |
| GRPO baseline | `run_grpo_baseline.sh` | `sokoban_hard_grpo_baseline_qwen3_8b_8gpu` |
| Reinforce + FreshPER | `run_reinforce_freshper.sh` | `sokoban_hard_reinforce_freshper_qwen3_8b_8gpu` |
| GRPO + FreshPER | `run_grpo_freshper.sh` | `sokoban_hard_grpo_freshper_qwen3_8b_8gpu` |

Baseline runs disable replay and use `ppo_epochs=2`.

FreshPER runs use the current KL-FreshPER implementation:

```yaml
replay.enabled: true
replay.priority_function: kl_fresh
replay.enable_age_decay: true
replay.use_engine_logprobs: true
ppo_epochs: 1
```

Run one experiment on the Wuwen 8-GPU shell:

```bash
tmux new -s roll
source /mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/project_modelware_roce/zhaojian/envs/roll
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/sokoban_hard_8gpu_ablation
bash run_reinforce_baseline.sh
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
