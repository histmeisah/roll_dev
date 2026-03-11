"""Sokoban Hard + VLM FrozenLake 绘图 - 论文用 (Best Run per Method)"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

def parse_wandb_csv(filename):
    """解析 wandb 导出的 CSV，返回 {run_short_name: [(step, value), ...]}"""
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

    # 找 mean 列 (排除 MIN/MAX)
    mean_cols = {}
    for i, h in enumerate(headers[1:], 1):
        if '__MIN' not in h and '__MAX' not in h:
            mean_cols[i] = h.split(' - ')[0].strip()

    data = {name: [] for name in mean_cols.values()}

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            step = int(row[0])
            for col_idx, name in mean_cols.items():
                val = row[col_idx].strip()
                if val:
                    data[name].append((step, float(val)))
    return data


# ============================================================
# 通用绘图函数
# ============================================================
def plot_comparison(ax, runs_data, title, ylabel, xlim, ylim, legend_loc='lower right'):
    colors = {
        'Baseline (On-Policy)':     '#4285F4',
        'Standard PER':             '#FBBC04',
        'Freshness Decay (Ours)':   '#EA4335',
    }
    markers = {
        'Baseline (On-Policy)':     's',
        'Standard PER':             '^',
        'Freshness Decay (Ours)':   'o',
    }
    zorders = {
        'Baseline (On-Policy)':     2,
        'Standard PER':             1,
        'Freshness Decay (Ours)':   3,
    }

    for label, series in runs_data.items():
        steps = [s for s, v in series]
        vals = [v for s, v in series]
        ax.plot(steps, vals,
                color=colors[label],
                marker=markers[label],
                markevery=max(1, len(steps)//10),
                markersize=5,
                linewidth=2.2,
                label=label,
                alpha=0.9,
                zorder=zorders[label])

    ax.set_xlabel('Training Steps', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc=legend_loc)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=11)


# ============================================================
# 1) Sokoban Hard
# ============================================================
sokoban = parse_wandb_csv('sokoban_hard_val_score_mean.csv')

print("=== Sokoban Hard runs ===")
for name, series in sokoban.items():
    vals = [v for _, v in series]
    print(f"  {name[:60]}: max={max(vals):.4f}, last={vals[-1]:.4f}, steps={len(vals)}")

# 每个方法只有1个run，直接映射
sokoban_runs = {
    'Freshness Decay (Ours)': sokoban[[k for k in sokoban if 'reward_fresh' in k][0]],
    'Standard PER':           sokoban[[k for k in sokoban if 'advantage_per' in k][0]],
    'Baseline (On-Policy)':   sokoban[[k for k in sokoban if 'baseline' in k][0]],
}

# ============================================================
# 2) VLM FrozenLake - 选 best run per method
# ============================================================
vlm = parse_wandb_csv('vlm_frozen_lake_val_success.csv')

print("\n=== VLM FrozenLake runs ===")
for name, series in vlm.items():
    vals = [v for _, v in series]
    print(f"  {name[:60]}: max={max(vals):.4f}, steps={len(vals)}")

# 按方法分组，选 best (最高 peak)
def best_run(data, keyword):
    candidates = {k: v for k, v in data.items() if keyword in k}
    best_name = max(candidates, key=lambda k: max(v for _, v in candidates[k]))
    peak = max(v for _, v in candidates[best_name])
    print(f"  Best '{keyword}': {best_name[:50]}... (peak={peak:.4f})")
    return candidates[best_name]

print("\nVLM FrozenLake picks (best Freshness vs others):")
# 最好的 Freshness run
vlm_fresh_best = vlm['20260228_104444_vlm_traj_reward_fresh_8gpu']
# 其他方法选非同批次 run
vlm_per_pick   = vlm['20260222_065434_vlm_traj_per_8gpu']
vlm_base_pick  = vlm['20260213_083427_vlm_traj_baseline_8gpu']

for tag, s in [('Freshness', vlm_fresh_best), ('PER', vlm_per_pick), ('Baseline', vlm_base_pick)]:
    peak = max(v for _, v in s)
    print(f"  {tag}: peak={peak:.4f}, steps={len(s)}")

vlm_runs = {
    'Baseline (On-Policy)':   vlm_base_pick,
    'Standard PER':           vlm_per_pick,
    'Freshness Decay (Ours)': vlm_fresh_best,
}

# ============================================================
# 绘图: 2个子图
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# Sokoban Hard
plot_comparison(ax1, sokoban_runs,
                title='Sokoban Hard: Validation Score',
                ylabel='Validation Score (Mean)',
                xlim=(-5, 400),
                ylim=(-2.05, -0.3),
                legend_loc='lower right')

# VLM FrozenLake
plot_comparison(ax2, vlm_runs,
                title='VLM FrozenLake: Validation Success Rate',
                ylabel='Validation Success Rate',
                xlim=(-5, 300),
                ylim=(-0.02, 0.85),
                legend_loc='upper left')

plt.tight_layout()
plt.savefig('sokoban_vlm_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('sokoban_vlm_comparison.pdf', bbox_inches='tight')
print("\nSaved: sokoban_vlm_comparison.png / .pdf")
