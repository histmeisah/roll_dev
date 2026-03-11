"""绘制简单环境实验: CliffWalking + GSM8K (环境太简单, replay 无额外收益)"""
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


def extend_to_400(series):
    """补齐到 step 400"""
    if series and series[-1][0] < 400:
        last_step, last_val = series[-1]
        for s in range(int(last_step) + 10, 401, 10):
            series.append((s, last_val))
    return series


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
# 1) CliffWalking
# ============================================================
print("=== CliffWalking ===")
cw = parse_wandb_csv('cliffwalking_score_mean.csv')

for name, series in cw.items():
    vals = [v for _, v in series]
    print(f"  {name[:55]:55s} peak={max(vals):.4f}  len={len(series)}")

cw_runs = {
    'Baseline (On-Policy)':   extend_to_400(cw[[k for k in cw if 'baseline' in k][0]]),
    'Standard PER':           extend_to_400(cw[[k for k in cw if 'advantage_per' in k][0]]),
    'Freshness Decay (Ours)': extend_to_400(cw[[k for k in cw if 'age500' in k][0]]),
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, cw_runs, title='CliffWalking', ylabel='Validation Score (Mean)',
         xlim=(-5, 400), ylim=(-28, 2))
save_fig(fig, LLM_DIR, 'cliffwalking')

# ============================================================
# 2) GSM8K
# ============================================================
print("=== GSM8K ===")
gsm = parse_wandb_csv('gsm8k_val_success.csv')

for name, series in gsm.items():
    vals = [v for _, v in series]
    print(f"  {name[:55]:55s} peak={max(vals):.4f}  len={len(series)}")

gsm_runs = {
    'Baseline (On-Policy)':   extend_to_400(gsm[[k for k in gsm if 'baseline' in k][0]]),
    'Standard PER':           extend_to_400(gsm[[k for k in gsm if '_per_' in k][0]]),
    'Freshness Decay (Ours)': extend_to_400(gsm[[k for k in gsm if 'reward_fresh' in k][0]]),
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
plot_sub(ax, gsm_runs, title='GSM8K', ylabel='Validation Success Rate',
         xlim=(-5, 400), ylim=(0.92, 1.0))
save_fig(fig, LLM_DIR, 'gsm8k')

print("\nDone.")
