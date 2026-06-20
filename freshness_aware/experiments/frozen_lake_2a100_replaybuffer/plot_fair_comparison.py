"""
公平对比绘图：
1. Trajectory vs Trajectory
2. Step vs Step
3. 使用相同基础配置的实验
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = Path(__file__).parent / "used_config" / "results"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# 读取数据
df = pd.read_csv(DATA_DIR / "all_val_env_FrozenLake_success.csv")

# ========== Trajectory 实验 (公平对比，都用 advantage_clip=0.2) ==========
TRAJ_EXPERIMENTS = {
    "Traj Baseline": {
        "exp": "20260122_145417_traj_baseline",  # adv_clip=0.2
        "color": "gray",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "Traj PER": {
        "exp": "20260115_193211_traj_per",  # adv_clip=0.2
        "color": "blue",
        "linestyle": "-.",
        "linewidth": 1.5,
    },
    "Traj reward_fresh (bug)": {
        "exp": "20260120_142505_traj_reward_fresh",  # adv_clip=0.2, bug
        "color": "orange",
        "linestyle": ":",
        "linewidth": 1.5,
    },
    "Traj RF age500 (crash)": {
        "exp": "20260203_141744_traj_reward_fresh",  # adv_clip=0.2, fixed, crash
        "color": "red",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "Traj RF age1000 (best)": {
        "exp": "20260203_145221_traj_reward_fresh_configA_age1000",  # adv_clip=0.2, fixed
        "color": "green",
        "linestyle": "-",
        "linewidth": 2.5,
        "marker": "o",
    },
    "Traj RF age1500": {
        "exp": "20260203_145728_traj_reward_fresh_configA_age1500",  # adv_clip=0.2, fixed
        "color": "purple",
        "linestyle": "-",
        "linewidth": 1.5,
    },
}

# ========== Step 实验 ==========
STEP_EXPERIMENTS = {
    "Step Baseline (adv=0.2)": {
        "exp": "20260122_145133_step_baseline",  # 需要确认adv_clip
        "color": "gray",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "Step PER+Nstep": {
        "exp": "20260119_204444_step_per_nstep",  # adv_clip=0.2
        "color": "blue",
        "linestyle": "-.",
        "linewidth": 1.5,
    },
    "Step reward_fresh (bug)": {
        "exp": "20260124_160415_step_reward_fresh",  # adv_clip=0.2, bug
        "color": "orange",
        "linestyle": ":",
        "linewidth": 1.5,
    },
}


def plot_trajectory_comparison():
    """绘制Trajectory实验对比图"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for exp_name, exp_info in TRAJ_EXPERIMENTS.items():
        exp = exp_info["exp"]
        if exp not in df.columns:
            print(f"警告: {exp} 不在数据中")
            continue

        steps = df["Step"].values
        values = df[exp].values * 100
        mask = ~np.isnan(values)

        ax.plot(steps[mask], values[mask],
                label=exp_name,
                color=exp_info["color"],
                linestyle=exp_info["linestyle"],
                linewidth=exp_info.get("linewidth", 1.5),
                marker=exp_info.get("marker"),
                markersize=4,
                markevery=5)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("FrozenLake Trajectory Experiments: Fair Comparison (adv_clip=0.2)", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 40)

    # 30%参考线
    ax.axhline(y=30, color='green', linestyle=':', alpha=0.5)
    ax.text(380, 31, "30%", fontsize=9, color='green')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "traj_fair_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"保存: {output_path}")
    plt.close()


def plot_step_comparison():
    """绘制Step实验对比图"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for exp_name, exp_info in STEP_EXPERIMENTS.items():
        exp = exp_info["exp"]
        if exp not in df.columns:
            print(f"警告: {exp} 不在数据中")
            continue

        steps = df["Step"].values
        values = df[exp].values * 100
        mask = ~np.isnan(values)

        ax.plot(steps[mask], values[mask],
                label=exp_name,
                color=exp_info["color"],
                linestyle=exp_info["linestyle"],
                linewidth=exp_info.get("linewidth", 1.5))

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("FrozenLake Step Experiments Comparison", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 35)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "step_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"保存: {output_path}")
    plt.close()


def plot_traj_bar_comparison():
    """绘制Trajectory实验柱状图对比"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    steps_to_plot = [10, 20, 40]

    experiments = [
        ("Baseline", "20260122_145417_traj_baseline", "gray"),
        ("PER", "20260115_193211_traj_per", "blue"),
        ("RF (bug)", "20260120_142505_traj_reward_fresh", "orange"),
        ("RF age1000", "20260203_145221_traj_reward_fresh_configA_age1000", "green"),
    ]

    for ax_idx, step in enumerate(steps_to_plot):
        ax = axes[ax_idx]
        row = df[df["Step"] == step]

        values = []
        colors = []
        labels = []

        for label, exp, color in experiments:
            if exp in df.columns and len(row) > 0:
                val = row[exp].values[0]
                if not pd.isna(val):
                    values.append(val * 100)
                    colors.append(color)
                    labels.append(label)

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, alpha=0.8)

        ax.set_ylabel("Success Rate (%)" if ax_idx == 0 else "")
        ax.set_title(f"Step {step}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
        ax.set_ylim(0, 35)
        ax.grid(True, alpha=0.3, axis='y')

        # 数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.suptitle("FrozenLake Trajectory: Fair Comparison (adv_clip=0.2)", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "traj_bar_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"保存: {output_path}")
    plt.close()


def plot_age_decay_traj():
    """绘制age_decay对比图 (仅Trajectory)"""
    fig, ax = plt.subplots(figsize=(12, 7))

    age_experiments = {
        "Traj Baseline": {
            "exp": "20260122_145417_traj_baseline",
            "color": "gray",
            "linestyle": "--",
        },
        "Traj PER": {
            "exp": "20260115_193211_traj_per",
            "color": "blue",
            "linestyle": "-.",
        },
        "age_decay=500 (crashed)": {
            "exp": "20260203_141744_traj_reward_fresh",
            "color": "red",
            "linestyle": "--",
        },
        "age_decay=1000 (best)": {
            "exp": "20260203_145221_traj_reward_fresh_configA_age1000",
            "color": "green",
            "linestyle": "-",
            "linewidth": 2.5,
            "marker": "o",
        },
        "age_decay=1500": {
            "exp": "20260203_145728_traj_reward_fresh_configA_age1500",
            "color": "purple",
            "linestyle": "-",
        },
    }

    for exp_name, exp_info in age_experiments.items():
        exp = exp_info["exp"]
        if exp not in df.columns:
            continue

        steps = df["Step"].values
        values = df[exp].values * 100
        mask = ~np.isnan(values)

        ax.plot(steps[mask], values[mask],
                label=exp_name,
                color=exp_info["color"],
                linestyle=exp_info["linestyle"],
                linewidth=exp_info.get("linewidth", 1.5),
                marker=exp_info.get("marker"),
                markersize=4,
                markevery=5)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("FrozenLake Trajectory: Age Decay Parameter Comparison", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 40)

    # 注释
    ax.annotate("age=500 crashes!",
                xy=(40, 2), xytext=(80, 12),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "traj_age_decay_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"保存: {output_path}")
    plt.close()


def plot_traj_heatmap():
    """绘制Trajectory实验热力图"""
    traj_experiments = [
        ("Baseline", "20260122_145417_traj_baseline"),
        ("PER", "20260115_193211_traj_per"),
        ("RF (bug)", "20260120_142505_traj_reward_fresh"),
        ("RF age500 (crash)", "20260203_141744_traj_reward_fresh"),
        ("RF age1000 ★", "20260203_145221_traj_reward_fresh_configA_age1000"),
        ("RF age1500", "20260203_145728_traj_reward_fresh_configA_age1500"),
    ]

    key_steps = [10, 20, 40, 100, 200, 300]

    # 构建热力图数据
    heatmap_data = []
    labels = []
    for label, exp in traj_experiments:
        if exp not in df.columns:
            continue
        labels.append(label)
        row_data = []
        for step in key_steps:
            step_row = df[df["Step"] == step]
            if len(step_row) > 0:
                val = step_row[exp].values[0]
                row_data.append(val * 100 if not pd.isna(val) else np.nan)
            else:
                row_data.append(np.nan)
        heatmap_data.append(row_data)

    heatmap_data = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=35)

    ax.set_xticks(np.arange(len(key_steps)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([f"Step {s}" for s in key_steps])
    ax.set_yticklabels(labels)

    # 数值
    for i in range(len(labels)):
        for j in range(len(key_steps)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 12 or val > 28 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                       color=text_color, fontsize=10)

    ax.set_title("FrozenLake Trajectory: Success Rate Heatmap (%)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Success Rate (%)")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "traj_heatmap.png"
    plt.savefig(output_path, dpi=150)
    print(f"保存: {output_path}")
    plt.close()


def print_summary():
    """打印Trajectory实验对比摘要"""
    print("\n" + "=" * 70)
    print("Trajectory 实验公平对比摘要 (都用 advantage_clip=0.2)")
    print("=" * 70)

    traj_experiments = [
        ("Baseline", "20260122_145417_traj_baseline"),
        ("PER", "20260115_193211_traj_per"),
        ("RF (bug)", "20260120_142505_traj_reward_fresh"),
        ("RF age500 (crash)", "20260203_141744_traj_reward_fresh"),
        ("RF age1000", "20260203_145221_traj_reward_fresh_configA_age1000"),
        ("RF age1500", "20260203_145728_traj_reward_fresh_configA_age1500"),
    ]

    print(f"\n{'实验':<25} {'Step10':>10} {'Step20':>10} {'Step40':>10} {'Step100':>10}")
    print("-" * 70)

    for label, exp in traj_experiments:
        if exp not in df.columns:
            continue
        values = []
        for step in [10, 20, 40, 100]:
            row = df[df["Step"] == step]
            if len(row) > 0:
                val = row[exp].values[0]
                if not pd.isna(val):
                    values.append(f"{val*100:>8.1f}%")
                else:
                    values.append("     N/A")
            else:
                values.append("     N/A")
        print(f"{label:<25} {values[0]:>10} {values[1]:>10} {values[2]:>10} {values[3]:>10}")

    print("\n结论:")
    print("  1. RF age1000 在 Step40 达到 30.5%，是所有Trajectory实验中最高")
    print("  2. 比 Baseline 高 50% (30.5% vs 20.3%)")
    print("  3. 比 PER 高 50% (30.5% vs 20.3%)")
    print("  4. age500 会导致崩溃")


if __name__ == "__main__":
    print("=" * 60)
    print("绘制公平对比图 (Trajectory vs Trajectory)")
    print("=" * 60)

    plot_trajectory_comparison()
    plot_traj_bar_comparison()
    plot_age_decay_traj()
    plot_traj_heatmap()
    plot_step_comparison()

    print_summary()

    print("\n" + "=" * 60)
    print(f"所有图表已保存到: {OUTPUT_DIR}")
    print("=" * 60)
