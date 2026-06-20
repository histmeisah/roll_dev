"""
FrozenLake (LLM, 0.5B) Age Decay + IS Ablation 绘图脚本

数据来源: frozen_lake_val_success.csv
实验配置: frozen_lake_2a100_replaybuffer/

生成两张图:
  1. llm/frozen_lake_age_decay_ablation.png/.pdf
     — Age Decay 消融 (无 IS): Baseline vs τ=500 vs τ=1000 vs τ=1500
  2. llm/frozen_lake_is_ablation.png/.pdf
     — IS 消融: Baseline vs τ=500 vs τ=500+IS(β=0.4)
"""
import csv
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

# EMA 平滑
EMA_ALPHA = 0.3  # 平滑系数, 越小越平滑


def ema_smooth(values, alpha=EMA_ALPHA):
    """Exponential moving average smoothing."""
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(BASE_DIR, 'llm')
os.makedirs(LLM_DIR, exist_ok=True)

# ============================================================
# 解析 CSV
# ============================================================
CSV_FILE = os.path.join(BASE_DIR, 'frozen_lake_val_success.csv')

with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    # 跳过开头的空行/垃圾行，找到真正的 header
    headers = next(reader)
    while not headers or headers[0].strip() != 'Step':
        headers = next(reader)

# 识别 mean 列 (排除 __MIN / __MAX)
mean_cols = {}
for i, h in enumerate(headers[1:], 1):
    if '__MIN' not in h and '__MAX' not in h:
        mean_cols[i] = h.split(' - ')[0].strip()

data = {name: [] for name in mean_cols.values()}
with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    # 跳过垃圾行和 header
    for row in reader:
        if row and row[0].strip() == 'Step':
            break
    for row in reader:
        if not row or not row[0].strip():
            continue
        step = int(row[0])
        for col_idx, name in mean_cols.items():
            if col_idx < len(row):
                val = row[col_idx].strip()
                if val:
                    data[name].append((step, float(val)))

# 打印所有 runs
print("=== FrozenLake (LLM) 所有 runs ===")
for name, series in data.items():
    vals = [v for _, v in series]
    peak = max(vals) if vals else 0
    peak_step = series[vals.index(peak)][0] if vals else 0
    last = vals[-1] if vals else 0
    last_step = series[-1][0] if series else 0
    print(f"  {name[:60]:60s}  peak={peak:.4f}@{peak_step}  last={last:.4f}@{last_step}")

# ============================================================
# 构建绘图 runs
# ============================================================
RUN_MAP = {}
for name in data.keys():
    if 'step_baseline_configA' in name:
        RUN_MAP['Baseline (On-Policy)'] = name
    elif 'reward_fresh_configA_age1500' in name:
        RUN_MAP[r'Freshness ($\tau$=1500)'] = name
    elif 'reward_fresh_configA_age1000' in name:
        RUN_MAP[r'Freshness ($\tau$=1000)'] = name
    elif 'reward_fresh_configA_IS' in name:
        RUN_MAP[r'Freshness ($\tau$=500, IS)'] = name
    elif 'reward_fresh_configA' in name:
        RUN_MAP[r'Freshness ($\tau$=500)'] = name

print("\n=== Label -> Run 映射 ===")
for label, name in RUN_MAP.items():
    print(f"  {label:30s} -> {name[:55]}")

# ============================================================
# 通用样式
# ============================================================
COLORS = {
    'Baseline (On-Policy)':            '#4285F4',
    r'Freshness ($\tau$=500)':         '#EA4335',
    r'Freshness ($\tau$=1000)':        '#FF9800',
    r'Freshness ($\tau$=1500)':        '#9E9E9E',
    r'Freshness ($\tau$=500, IS)':     '#34A853',
}
MARKERS = {
    'Baseline (On-Policy)':            's',
    r'Freshness ($\tau$=500)':         'o',
    r'Freshness ($\tau$=1000)':        'D',
    r'Freshness ($\tau$=1500)':        'v',
    r'Freshness ($\tau$=500, IS)':     'P',
}


def plot_lines(ax, labels, zorder_start=1):
    """在 ax 上绘制指定 labels 的平滑曲线 (仅平滑, 无原始数据)."""
    for i, label in enumerate(labels):
        if label not in RUN_MAP:
            print(f"  WARNING: {label} not found in data, skipping")
            continue
        series = data[RUN_MAP[label]]
        steps = [s for s, v in series]
        vals = [v for s, v in series]
        smoothed = ema_smooth(vals)
        zo = zorder_start + i
        ax.plot(steps, smoothed,
                color=COLORS[label],
                marker=MARKERS[label],
                markevery=max(1, len(steps) // 8),
                markersize=6,
                linewidth=2.2,
                label=label,
                alpha=0.9,
                zorder=zo)


def style_ax(ax):
    """统一坐标轴样式."""
    ax.set_xlabel('Training Steps', fontsize=13)
    ax.set_ylabel('Validation Success Rate', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-5, 400)
    ax.tick_params(axis='both', which='major', labelsize=11)


# ============================================================
# 图 1: Age Decay 消融 (无 IS)
# ============================================================
ORDER_AGE = [
    'Baseline (On-Policy)',
    r'Freshness ($\tau$=1500)',
    r'Freshness ($\tau$=500)',
    r'Freshness ($\tau$=1000)',
]

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5.2))
plot_lines(ax1, ORDER_AGE)
style_ax(ax1)
fig1.tight_layout()
fig1.savefig(os.path.join(LLM_DIR, 'frozen_lake_age_decay_ablation.png'), dpi=300, bbox_inches='tight')
fig1.savefig(os.path.join(LLM_DIR, 'frozen_lake_age_decay_ablation.pdf'), bbox_inches='tight')
plt.close(fig1)
print("\nSaved: llm/frozen_lake_age_decay_ablation.png/.pdf")

# ============================================================
# 图 2: IS 消融
# ============================================================
ORDER_IS = [
    'Baseline (On-Policy)',
    r'Freshness ($\tau$=500)',
    r'Freshness ($\tau$=500, IS)',
]

fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5.2))
plot_lines(ax2, ORDER_IS)
style_ax(ax2)
fig2.tight_layout()
fig2.savefig(os.path.join(LLM_DIR, 'frozen_lake_is_ablation.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(LLM_DIR, 'frozen_lake_is_ablation.pdf'), bbox_inches='tight')
plt.close(fig2)
print("Saved: llm/frozen_lake_is_ablation.png/.pdf")

# ============================================================
# 打印数值总结
# ============================================================
ALL_LABELS = list(dict.fromkeys(ORDER_AGE + ORDER_IS))  # 去重保序
print("\n=== 数值总结 ===")
print(f"{'Method':<32s}  {'Peak':>6s}  {'Peak Step':>9s}  {'Last':>6s}  {'Last Step':>9s}")
print("-" * 72)
for label in ALL_LABELS:
    if label not in RUN_MAP:
        continue
    series = data[RUN_MAP[label]]
    vals = [v for _, v in series]
    peak = max(vals)
    peak_idx = vals.index(peak)
    peak_step = series[peak_idx][0]
    last = vals[-1]
    last_step = series[-1][0]
    print(f"{label:<32s}  {peak:>6.4f}  {peak_step:>9d}  {last:>6.4f}  {last_step:>9d}")
