# VLM FrozenLake Experiments

## Overview

This experiment validates VLM (Vision Language Model) support with the off-policy replay buffer system.
The environment renders visual states (RGB images) instead of text, and uses `VLTrajEnvManager` to process multimodal inputs.

## Architecture

```
VLM Training Pipeline:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FrozenLake    │───>│ VLTrajEnvManager │───>│  VLM (Qwen-VL)  │
│  (rgb_array)    │    │  (image + text)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                      │                      │
         │              ┌───────▼───────┐              │
         │              │ ReplayBuffer  │<─────────────┘
         │              │ (off-policy)  │
         │              └───────────────┘
         │
    Visual State: numpy RGB array
    Message Format: [{"type": "image", "image": base64}, {"type": "text", ...}]
```

## Key Differences from Text Version

| Component | Text Version | VLM Version |
|-----------|-------------|-------------|
| EnvManager | TrajEnvManager / StepEnvManager | **VLTrajEnvManager** |
| render_mode | "text" | **"rgb_array"** |
| State Format | String (grid ASCII) | **numpy RGB array** |
| Message Format | `{"role": "user", "content": str}` | `{"role": "user", "content": [{"type": "image"}, {"type": "text"}]}` |
| Model | LLM (Qwen2.5) | **VLM (Qwen2.5-VL)** |

## Configurations

| Config | Replay | Priority | Description |
|--------|--------|----------|-------------|
| `traj_baseline.yaml` | ❌ | - | VLM baseline, no replay buffer |
| `traj_per.yaml` | ✅ | `reward` | VLM + Standard PER |
| `traj_reward_fresh.yaml` | ✅ | `reward_fresh` | VLM + Reward-Fresh (our extension) |

## Environment Configuration

```yaml
custom_envs:
  FrozenLake:
    # Use VLTrajEnvManager for multimodal processing
    env_manager_cls: roll.pipeline.agentic.env_manager.vl_traj_env_manager.VLTrajEnvManager

    # VLM-specific templates
    pre_step_template: "Turn {turn_idx}:\nState:"
    next_step_template: |
      You have {actions_left} actions left.
      Decide the next action within {max_response_length} tokens.
    reward_template: "Reward: {reward}\n"

    env_config:
      render_mode: "rgb_array"  # CRITICAL: Enable image rendering
```

## Model Requirements

VLM models that support the framework:
- **Qwen2.5-VL** series (recommended): Qwen2.5-VL-3B, Qwen2.5-VL-7B
- **LLaVA** series
- Any model compatible with `ProcessorMixin` from transformers

## Running Experiments

```bash
# 1. Sync code to server
powershell.exe -Command "cd e:\code_project\python_code\local_roll_dev; .\sync.bat push-all"

# 2. SSH and run
ssh zgc_server
cd /data1/Chengyang_project/roll_dev/experiments/vlm_frozen_lake
chmod +x run.sh

# Edit CONFIG_NAME in run.sh, then:
./run.sh
```

## Replay Buffer Compatibility

The off-policy replay buffer system is **fully compatible** with VLM training:

1. **Data Storage**: `TrajectoryEntry` stores tokenized `input_ids` (including image tokens)
2. **Priority Functions**: Based on rewards, modality-agnostic
3. **Importance Sampling**: Works with any token sequence

No modifications needed to the replay buffer code.

## Hardware Requirements

- **Minimum**: 2x A100 40GB (or equivalent)
- **Recommended**: 4x A100 80GB for larger VLM models
- VLM inference requires more VRAM than text-only LLM

## TODO

- [ ] Validate with Qwen2.5-VL-3B
- [ ] Add Sokoban VLM environment
- [ ] Implement VLStepEnvManager for step-level VLM training
- [ ] Benchmark VLM vs Text performance
