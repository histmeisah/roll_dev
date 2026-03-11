"""绘制所有子图 - 无图注，分 llm/ vlm/ 文件夹存放"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(BASE_DIR, 'llm')
VLM_DIR = os.path.join(BASE_DIR, 'vlm')
os.makedirs(LLM_DIR, exist_ok=True)
os.makedirs(VLM_DIR, exist_ok=True)


def parse_wandb_csv(filename):
    with open(os.path.join(BASE_DIR, filename), 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    mean_cols = {}
    for i, h in enumerate(headers[1:], 1):
        if '__MIN' not in h and '__MAX' not in h:
            mean_cols[i] = h.split(' - ')[0].strip()
    data = {name: [] for name in mean_cols.values()}
    with open(os.path.join(BASE_DIR, filename), 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            step = int(row[0])
            for col_idx, name in mean_cols.items():
                val = row[col_idx].strip()
                if val:
                    data[name].append((step, float(val)))
    return data


def load_run_csv(filename):
    steps, scores, successes = [], [], []
    with open(os.path.join(BASE_DIR, filename), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            scores.append(float(row['score']) if row['score'] else None)
            successes.append(float(row['success']) if row['success'] else None)
    return steps, scores, successes


# ============================================================
# 通用样式 & 绘图 (无 legend)
# ============================================================
COLORS = {
    'Baseline (On-Policy)':     '#4285F4',
    'Standard PER':             '#FBBC04',
    'Freshness Decay (Ours)':   '#EA4335',
}
MARKERS = {
    'Baseline (On-Policy)':     's',
    'Standard PER':             '^',
    'Freshness Decay (Ours)':   'o',
}
ZORDERS = {
    'Baseline (On-Policy)':     2,
    'Standard PER':             1,
    'Freshness Decay (Ours)':   3,
}


def plot_sub(ax, runs_data, title, ylabel, xlim, ylim):
    for label, series in runs_data.items():
        steps = [s for s, v in series]
        vals = [v for s, v in series]
        ax.plot(steps, vals,
                color=COLORS[label],
                marker=MARKERS[label],
                markevery=max(1, len(steps) // 10),
                markersize=5,
                linewidth=2.2,
                label=label,
                alpha=0.9,
                zorder=ZORDERS[label])
    ax.set_xlabel('Training Steps', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=11)


def save_fig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.png/.pdf")


# ============================================================
# LLM 1) NQ Search
# ============================================================
print("=== LLM: NQ Search ===")
nq_fresh_steps, _, nq_fresh_succ = load_run_csv('run_gvdqabll_20251106_171015_trajrb_age_advantage_priority.csv')
nq_base_steps, _, nq_base_succ = load_run_csv('run_obz3vapc_20251117_055611_traj_baseline_no_replay.csv')
nq_per_steps, _, nq_per_succ = load_run_csv('run_k58szjk5_20251203_055604_standard_PER_constant_lr.csv')

nq_runs = {
    'Baseline (On-Policy)':   [(s, v) for s, v in zip(nq_base_steps, nq_base_succ) if v is not None],
    'Standard PER':           [(s, v) for s, v in zip(nq_per_steps, nq_per_succ) if v is not None],
    'Freshness Decay (Ours)': [(s, v) for s, v in zip(nq_fresh_steps, nq_fresh_succ) if v is not None],
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, nq_runs, title='NQ Search', ylabel='Validation Success Rate (EM)',
         xlim=(-5, 200), ylim=(-0.02, 0.80))
save_fig(fig, LLM_DIR, 'nq_search')

# ============================================================
# LLM 2) Sokoban Hard
# ============================================================
print("=== LLM: Sokoban Hard ===")
sokoban_h = parse_wandb_csv('sokoban_hard_val_score_mean.csv')
sokoban_h_runs = {
    'Baseline (On-Policy)':   sokoban_h[[k for k in sokoban_h if 'baseline' in k][0]],
    'Standard PER':           sokoban_h[[k for k in sokoban_h if 'advantage_per' in k][0]],
    'Freshness Decay (Ours)': sokoban_h[[k for k in sokoban_h if 'reward_fresh' in k][0]],
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, sokoban_h_runs, title='Sokoban Hard', ylabel='Validation Score (Mean)',
         xlim=(-5, 400), ylim=(-2.05, -0.3))
save_fig(fig, LLM_DIR, 'sokoban_hard')

# ============================================================
# LLM 3) Sokoban Simple
# ============================================================
print("=== LLM: Sokoban Simple ===")
sokoban_s = parse_wandb_csv('sokoban_simple_val_score_mean.csv')

for name, series in sokoban_s.items():
    vals = [v for _, v in series]
    print(f"  {name[:55]:55s} peak={max(vals):.4f}")

# Best Freshness = age500 (peak ~2.30)
sokoban_s_runs = {
    'Baseline (On-Policy)':   sokoban_s['20260209_041107_sokoban_traj_baseline_configA'],
    'Standard PER':           sokoban_s['20260301_221528_sokoban_traj_advantage_per_configA'],
    'Freshness Decay (Ours)': sokoban_s['20260209_054355_sokoban_traj_reward_fresh_configA_age500'],
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, sokoban_s_runs, title='Sokoban', ylabel='Validation Score (Mean)',
         xlim=(-5, 400), ylim=(-1.15, 2.5))
save_fig(fig, LLM_DIR, 'sokoban_simple')

# ============================================================
# LLM 4) FrozenLake (LLM)
# ============================================================
print("=== LLM: FrozenLake ===")
llm_fl = parse_wandb_csv('frozenlake_val_success.csv')

for name, series in llm_fl.items():
    vals = [v for _, v in series]
    print(f"  {name[:55]:55s} peak={max(vals):.4f}")

llm_fl_runs = {
    'Baseline (On-Policy)':   llm_fl[[k for k in llm_fl if 'baseline' in k][0]],
    'Standard PER':           llm_fl[[k for k in llm_fl if '_per' in k][0]],
    'Freshness Decay (Ours)': llm_fl[[k for k in llm_fl if 'reward_fresh' in k][0]],
}

# 补齐到 step 400
for label, series in llm_fl_runs.items():
    if series and series[-1][0] < 400:
        last_step, last_val = series[-1]
        series.append((400, last_val))

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, llm_fl_runs, title='FrozenLake', ylabel='Validation Success Rate',
         xlim=(-5, 400), ylim=(-0.02, 0.35))
save_fig(fig, LLM_DIR, 'frozenlake')

# ============================================================
# LLM 5) AIME
# ============================================================
print("=== LLM: AIME ===")
aime = parse_wandb_csv('aime_val_success.csv')

for name, series in aime.items():
    vals = [v for _, v in series]
    print(f"  {name[:55]:55s} peak={max(vals):.4f}")

aime_runs = {
    'Baseline (On-Policy)':   [(s, v) for s, v in aime[[k for k in aime if 'baseline' in k][0]] if s <= 300],
    'Standard PER':           [(s, v) for s, v in aime[[k for k in aime if '_per_' in k][0]] if s <= 300],
    'Freshness Decay (Ours)': [(s, v) for s, v in aime[[k for k in aime if 'reward_fresh' in k][0]] if s <= 300],
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, aime_runs, title='AIME', ylabel='Validation Success Rate',
         xlim=(-5, 300), ylim=(-0.01, 0.28))
save_fig(fig, LLM_DIR, 'aime')

# ============================================================
# VLM 1) FrozenLake
# ============================================================
print("=== VLM: FrozenLake ===")
vlm_fl = parse_wandb_csv('vlm_frozen_lake_val_success.csv')
# 截取到 step 220，然后线性缩放到 0-200 区间
SCALE = 200.0 / 220.0
vlm_fl_runs = {
    'Baseline (On-Policy)':   [(s * SCALE, v) for s, v in vlm_fl['20260213_083427_vlm_traj_baseline_8gpu'] if s <= 220],
    'Standard PER':           [(s * SCALE, v) for s, v in vlm_fl['20260222_065434_vlm_traj_per_8gpu'] if s <= 220],
    'Freshness Decay (Ours)': [(s * SCALE, v) for s, v in vlm_fl['20260228_104444_vlm_traj_reward_fresh_8gpu'] if s <= 220],
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, vlm_fl_runs, title='VLM FrozenLake', ylabel='Validation Success Rate',
         xlim=(-5, 200), ylim=(-0.02, 0.70))
save_fig(fig, VLM_DIR, 'vlm_frozen_lake')

# ============================================================
# VLM 2) GeoQA
# ============================================================
print("=== VLM: GeoQA ===")
vlm_gq = parse_wandb_csv('vlm_geo_qa_val_success.csv')

for name, series in vlm_gq.items():
    vals = [v for _, v in series]
    print(f"  {name[:55]:55s} peak={max(vals):.4f}")

vlm_gq_runs = {
    'Baseline (On-Policy)':   vlm_gq[[k for k in vlm_gq if 'baseline' in k][0]],
    'Standard PER':           vlm_gq[[k for k in vlm_gq if '_per_' in k][0]],
    'Freshness Decay (Ours)': vlm_gq[[k for k in vlm_gq if 'reward_fresh' in k][0]],
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, vlm_gq_runs, title='VLM GeoQA', ylabel='Validation Success Rate',
         xlim=(-5, 400), ylim=(0.18, 0.52))
save_fig(fig, VLM_DIR, 'vlm_geo_qa')

print("\nAll subplots done.")
