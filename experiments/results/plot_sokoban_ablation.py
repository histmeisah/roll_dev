"""Sokoban Age Decay Ablation 独立绘图脚本 (含 PER 对比)"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(BASE_DIR, 'llm')
os.makedirs(LLM_DIR, exist_ok=True)


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


# ============================================================
# 加载数据
# ============================================================
sokoban_s = parse_wandb_csv('sokoban_simple_val_score_mean.csv')

print("=== Sokoban Simple 所有 runs ===")
for name, series in sokoban_s.items():
    vals = [v for _, v in series]
    print(f"  {name[:55]:55s} peak={max(vals):.4f}  steps={len(series)}")

# ============================================================
# 构建 ablation runs (仅 age decay 参数对比)
# ============================================================
LABELS = [
    'Baseline (On-Policy)',
    'Freshness (age=500)',
    'Freshness (age=1000)',
    'Freshness (age=1500)',
]
COLORS = {
    'Baseline (On-Policy)':      '#4285F4',
    'Freshness (age=500)':       '#EA4335',
    'Freshness (age=1000)':      '#FF9800',
    'Freshness (age=1500)':      '#9E9E9E',
}
MARKERS = {
    'Baseline (On-Policy)':      's',
    'Freshness (age=500)':       'o',
    'Freshness (age=1000)':      'D',
    'Freshness (age=1500)':      'v',
}
ZORDERS = {
    'Baseline (On-Policy)':      1,
    'Freshness (age=500)':       4,
    'Freshness (age=1000)':      3,
    'Freshness (age=1500)':      2,
}

ablation_runs = {
    'Baseline (On-Policy)':  sokoban_s['20260209_041107_sokoban_traj_baseline_configA'],
    'Freshness (age=500)':   sokoban_s['20260209_054355_sokoban_traj_reward_fresh_configA_age500'],
    'Freshness (age=1000)':  sokoban_s['20260211_033527_sokoban_traj_reward_fresh_configA_age1000'],
    'Freshness (age=1500)':  sokoban_s['20260211_053616_sokoban_traj_reward_fresh_configA_age1500'],
}

# 补齐到 step 400
for label, series in ablation_runs.items():
    if series and series[-1][0] < 400:
        last_step, last_val = series[-1]
        for s in range(int(last_step) + 10, 401, 10):
            series.append((s, last_val))

# ============================================================
# 绘图
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 5.2))

for label in LABELS:
    series = ablation_runs[label]
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
ax.set_ylabel('Validation Score (Mean)', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-5, 400)
ax.set_ylim(-1.15, 2.5)
ax.tick_params(axis='both', which='major', labelsize=11)

fig.tight_layout()
fig.savefig(os.path.join(LLM_DIR, 'sokoban_age_decay_ablation.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(LLM_DIR, 'sokoban_age_decay_ablation.pdf'), bbox_inches='tight')
plt.close(fig)
print("\nSaved: llm/sokoban_age_decay_ablation.png/.pdf")
