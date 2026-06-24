# Freshness Aware Server Workflow

Last updated: 2026-06-20

This document records the project-specific workflow for `freshness_aware` on the Wuwen J-02/RoCE cluster. It is intentionally separate from older `freshness_replaybuffer`, `local_roll_dev`, and `world_model` paths.

## Current Machine

```text
resource pool: project_modelware_roce
devmachine ip: 172.27.250.202
remote repo:   /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware
model root:    /mnt/project_modelware_roce/zhaojian/liangsirui/Model
conda init:    /mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh
roll env:      /mnt/project_modelware_roce/zhaojian/envs/roll
```

Connection path:

```text
local Windows PowerShell
-> ssh aicoder
-> ssh root@172.27.250.202
-> /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware
```

Quick check from local:

```bash
ssh aicoder "ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@172.27.250.202 'hostname; hostname -i; whoami; cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware && pwd'"
```

## Sync From Windows

Use `rsync` for normal code sync. Do not sync logs, checkpoints, wandb runs, datasets, caches, local virtual envs, or temporary outputs.

PowerShell setup:

```powershell
$env:MSYS2_ARG_CONV_EXCL='*'
```

Sync `freshness_aware/`:

```powershell
C:\msys64\usr\bin\rsync.exe -az --delete --human-readable --stats --itemize-changes `
  --exclude ".git/" `
  --exclude "__pycache__/" `
  --exclude "*.pyc" `
  --exclude ".pytest_cache/" `
  --exclude ".ruff_cache/" `
  --exclude ".mypy_cache/" `
  --exclude ".venv*/" `
  --exclude "wandb/" `
  --exclude "outputs/" `
  --exclude "logs/" `
  --exclude "runs/" `
  --exclude "checkpoints/" `
  --exclude "download_logs/" `
  --exclude "datasets/" `
  --rsync-path=/usr/bin/rsync `
  -e "ssh" `
  ./freshness_aware/ aicoder:/mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/
```

The old remote repo is reference-only:

```text
/mnt/project_modelware_roce/zhaojian/weiyu/freshness_replaybuffer
```

Do not sync new code into that path.

## PowerShell Remote Commands

Simple checks can use one-line nested SSH:

```powershell
ssh aicoder "ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@172.27.250.202 'hostname && nvidia-smi -L'"
```

For commands with pipes, regex, `grep/sed/awk`, loops, here-docs, multi-line exports, or `tmux new-session`, send a script through stdin instead of nesting quotes:

```powershell
$script = @'
set -euo pipefail
hostname
nvidia-smi -L
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware
pwd
'@
$script = $script.Replace("`r`n", "`n").Replace("`r", "")

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = 'ssh'
$psi.Arguments = 'aicoder "ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@172.27.250.202 bash -s"'
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$p = [System.Diagnostics.Process]::Start($psi)
$p.StandardInput.Write($script)
$p.StandardInput.Close()
$out = $p.StandardOutput.ReadToEnd()
$err = $p.StandardError.ReadToEnd()
$p.WaitForExit()
if ($out) { Write-Output $out }
if ($err) { Write-Error $err }
if ($p.ExitCode -ne 0) { throw "remote command failed with exit code $($p.ExitCode)" }
```

Rules:

```text
1. Normalize CRLF before sending scripts to remote bash.
2. Keep BatchMode=yes and ConnectTimeout to avoid hangs when a devmachine is down.
3. Use StrictHostKeyChecking=accept-new for newly created devmachines.
4. Do not use $script | ssh ... for complex scripts from Windows PowerShell.
```

## Wuwen Training Submission

The web platform command should stay simple and reproducible:

```bash
tmux new -s roll
source /mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/project_modelware_roce/zhaojian/envs/roll
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/<experiment_name>
bash run_<experiment_name>.sh
```

Sokoban Hard 8GPU ablation:

```bash
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/sokoban_hard_8gpu_ablation
bash run_reinforce_baseline.sh
bash run_grpo_baseline.sh
bash run_reinforce_freshper.sh
bash run_grpo_freshper.sh
```

KL-Fresh:

```bash
cd /mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware/experiments/sokoban_hard_8gpu_kl_fresh
bash run_sokoban_hard_kl_fresh_8gpu.sh
```

## Log Checks

Do not judge success from `tail` alone. Check process exit, expected step count, wandb/offline run files, and strict error patterns.

Use specific error patterns:

```bash
grep -E 'Traceback|ERROR|Exception|RuntimeError|CUDA out of memory|Killed|torchrun.*failed|NCCL.*(error|Error)' "$LOG" || true
```

Do not use bare `inf`; it matches normal `INFO` log lines.

## Safety

Do not commit or write into docs:

```text
platform passwords
Access Key / Secret Key
tokens
private keys
complete authorized_keys content
```

When the devmachine is recreated, update this document with the new IP and re-check AIcoder to devmachine SSH access.
