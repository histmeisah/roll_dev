# Sokoban Hard KL-FreshPER 8GPU

This directory is for Wuwen platform shell-window submission.

Repo path:

```bash
/mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware
```

Model path:

```bash
/mnt/project_modelware_roce/zhaojian/liangsirui/Model/Qwen3-8B
```

Run command on the allocated 8-GPU node:

```bash
tmux new -s roll
source /mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/project_modelware_roce/zhaojian/envs/roll
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/sokoban_hard_8gpu_kl_fresh
bash run_sokoban_hard_kl_fresh_8gpu.sh
```

Equivalent generic command:

```bash
bash run_8gpu.sh sokoban_hard_reinforce_kl_fresh_qwen3_8b_8gpu_smoke
```

Key settings:

- Algorithm: Reinforce++ style Config A (`adv_estimator: reinforce`)
- Environment: `LargerSokoban`, 10x10 room, 2 boxes
- GPUs: train/reference on 0-3, vLLM inference on 4-7
- Replay priority: `kl_fresh`
- Behavior logprobs: `replay.use_engine_logprobs: true`
- Smoke length: `max_steps: 50`

For a longer formal run, increase `max_steps` to 400 and consider changing
`replay.train_steps_per_env_step` from 1 to 2.
