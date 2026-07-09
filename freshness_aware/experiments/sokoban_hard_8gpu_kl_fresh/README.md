# Sokoban Hard KL-FreshPER 8GPU

This directory is for ORIX Slurm submission.

Repo path:

```bash
/mnt/data/u/maw0a/python_project/freshness_aware
```

Model path:

```bash
/mnt/data/u/maw0a/models/Qwen3-8B
```

Submit an 8-GPU H100 job:

```bash
cd /mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_8gpu_kl_fresh
sbatch sbatch_8gpu.sh
```

Equivalent explicit command:

```bash
sbatch sbatch_8gpu.sh sokoban_hard_reinforce_kl_fresh_qwen3_8b_8gpu_smoke
```

Use `MODEL_PATH=/path/to/Qwen3-8B sbatch sbatch_8gpu.sh` if the model is stored
somewhere else.

Key settings:

- Algorithm: Reinforce++ style Config A (`adv_estimator: reinforce`)
- Environment: `LargerSokoban`, 10x10 room, 2 boxes
- GPUs: train/reference on 0-3, vLLM inference on 4-7
- Replay priority: `kl_fresh`
- Behavior logprobs: `replay.use_engine_logprobs: true`
- Current length: `max_steps: 400`

This directory is a standalone KL-FreshPER Config-A run. The final two
ppo2/filter ablation runs live in `../sokoban_hard_8gpu_ablation`.
