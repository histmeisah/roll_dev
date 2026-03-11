"""NQ Search 实验结果绘图 - 论文用 (Best Run per Method)"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

def load_run_csv(filepath):
    steps, scores, successes = [], [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row['step'])
            sc = float(row['score']) if row['score'] else None
            su = float(row['success']) if row['success'] else None
            steps.append(s)
            scores.append(sc)
            successes.append(su)
    return steps, scores, successes

# 加载3个best run
runs = {
    'Baseline (On-Policy)': 'run_obz3vapc_20251117_055611_traj_baseline_no_replay.csv',
    'Standard PER':         'run_k58szjk5_20251203_055604_standard_PER_constant_lr.csv',
    'Freshness Decay (Ours)': 'run_gvdqabll_20251106_171015_trajrb_age_advantage_priority.csv',
}

# 打印统计
for label, fname in runs.items():
    steps, scores, successes = load_run_csv(fname)
    vals = [v for v in successes if v is not None]
    valid_pairs = [(s, v) for s, v in zip(steps, successes) if v is not None and v > 0]
    peak = max(v for _, v in valid_pairs)
    peak_step = [s for s, v in valid_pairs if v == peak][0]
    converged = [v for s, v in valid_pairs if 50 <= s <= 165]
    conv_mean = np.mean(converged) if converged else 0
    print(f"{label:30s}: peak={peak:.4f} @ step {peak_step}, converged_avg(50-165)={conv_mean:.4f}, total_steps={len(vals)}")

# ========== 绘图 ==========
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

colors = {
    'Baseline (On-Policy)':     '#4285F4',  # Google Blue
    'Standard PER':             '#FBBC04',  # Google Yellow
    'Freshness Decay (Ours)':   '#EA4335',  # Google Red
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

for label, fname in runs.items():
    steps, scores, successes = load_run_csv(fname)
    # 使用 success rate
    valid_steps = [s for s, v in zip(steps, successes) if v is not None]
    valid_vals = [v for v in successes if v is not None]

    ax.plot(valid_steps, valid_vals,
            color=colors[label],
            marker=markers[label],
            markevery=4,
            markersize=5,
            linewidth=2.2,
            label=label,
            alpha=0.9,
            zorder=zorders[label])

ax.set_xlabel('Training Steps', fontsize=14)
ax.set_ylabel('Validation Success Rate (EM)', fontsize=14)
ax.set_title('NQ Search: Validation Performance Comparison', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(-0.02, 0.80)
ax.set_xlim(-5, 200)
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig('nq_search_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('nq_search_comparison.pdf', bbox_inches='tight')
print("\nSaved: nq_search_comparison.png / .pdf")
