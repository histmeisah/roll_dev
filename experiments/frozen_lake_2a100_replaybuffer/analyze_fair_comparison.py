"""
公平对比分析：使用相同基础配置(advantage_clip=0.2)的实验
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 读取数据
data_path = Path(__file__).parent / "used_config" / "results" / "all_val_env_FrozenLake_success.csv"
df = pd.read_csv(data_path)

# 公平对比实验（都使用 advantage_clip=0.2 的配置）
FAIR_EXPERIMENTS = {
    "Baseline (adv_clip=0.2)": [
        "20260122_145417_traj_baseline",  # advantage_clip=0.2
    ],
    "PER (adv_clip=0.2)": [
        "20260119_204444_step_per_nstep",  # advantage_clip=0.2
        "20260122_150106_traj_per",
    ],
    "reward_fresh (bug, adv_clip=0.2)": [
        "20260120_142505_traj_reward_fresh",
        "20260129_160515_traj_reward_fresh_configA",
        "20260130_040529_traj_reward_fresh_configA",
    ],
    "reward_fresh (fixed, age1000)": [
        "20260203_145221_traj_reward_fresh_configA_age1000",
    ],
}

# 不公平对比（使用 advantage_clip=20 的实验）
UNFAIR_BASELINE = {
    "Baseline (adv_clip=20) [不公平]": [
        "20260123_164201_step_baseline",  # advantage_clip=20
        "20260126_202655_traj_baseline",  # advantage_clip=20
    ],
}


def get_value(exp_name, step):
    if exp_name not in df.columns:
        return None
    row = df[df["Step"] == step]
    if len(row) == 0:
        return None
    val = row[exp_name].values[0]
    if pd.isna(val):
        return None
    return val


def analyze():
    print("=" * 80)
    print("公平对比分析：使用相同基础配置 (advantage_clip=0.2)")
    print("=" * 80)

    steps = [10, 20, 40, 100, 200]

    # 公平对比
    print("\n【公平对比】所有实验使用 advantage_clip=0.2")
    print("-" * 80)
    print(f"{'类别':<40} {'Step10':>10} {'Step20':>10} {'Step40':>10}")
    print("-" * 80)

    for category, experiments in FAIR_EXPERIMENTS.items():
        values_10 = [get_value(e, 10) for e in experiments]
        values_20 = [get_value(e, 20) for e in experiments]
        values_40 = [get_value(e, 40) for e in experiments]

        values_10 = [v for v in values_10 if v is not None]
        values_20 = [v for v in values_20 if v is not None]
        values_40 = [v for v in values_40 if v is not None]

        avg_10 = f"{np.mean(values_10)*100:.1f}%" if values_10 else "N/A"
        avg_20 = f"{np.mean(values_20)*100:.1f}%" if values_20 else "N/A"
        avg_40 = f"{np.mean(values_40)*100:.1f}%" if values_40 else "N/A"

        print(f"{category:<40} {avg_10:>10} {avg_20:>10} {avg_40:>10}")

    # 不公平对比
    print("\n【不公平对比】使用 advantage_clip=20 的Baseline")
    print("-" * 80)
    for category, experiments in UNFAIR_BASELINE.items():
        values_10 = [get_value(e, 10) for e in experiments]
        values_20 = [get_value(e, 20) for e in experiments]
        values_40 = [get_value(e, 40) for e in experiments]

        values_10 = [v for v in values_10 if v is not None]
        values_20 = [v for v in values_20 if v is not None]
        values_40 = [v for v in values_40 if v is not None]

        avg_10 = f"{np.mean(values_10)*100:.1f}%" if values_10 else "N/A"
        avg_20 = f"{np.mean(values_20)*100:.1f}%" if values_20 else "N/A"
        avg_40 = f"{np.mean(values_40)*100:.1f}%" if values_40 else "N/A"

        print(f"{category:<40} {avg_10:>10} {avg_20:>10} {avg_40:>10}")

    # 具体实验对比
    print("\n" + "=" * 80)
    print("【具体实验对比】")
    print("=" * 80)

    specific_exps = [
        ("Baseline (adv_clip=0.2)", "20260122_145417_traj_baseline"),
        ("PER+Nstep (adv_clip=0.2)", "20260119_204444_step_per_nstep"),
        ("reward_fresh age1000", "20260203_145221_traj_reward_fresh_configA_age1000"),
        ("---", "---"),
        ("Baseline (adv_clip=20) [不公平]", "20260123_164201_step_baseline"),
    ]

    print(f"{'实验':<40} {'Step10':>10} {'Step20':>10} {'Step40':>10} {'Step100':>10}")
    print("-" * 80)

    for label, exp in specific_exps:
        if exp == "---":
            print("-" * 80)
            continue
        values = []
        for step in [10, 20, 40, 100]:
            val = get_value(exp, step)
            if val is not None:
                values.append(f"{val*100:.1f}%")
            else:
                values.append("N/A")
        print(f"{label:<40} {values[0]:>10} {values[1]:>10} {values[2]:>10} {values[3]:>10}")

    # 结论
    print("\n" + "=" * 80)
    print("【结论】")
    print("=" * 80)
    print("""
公平对比 (都用 advantage_clip=0.2):
  - Baseline:              Step10=0.0%,  Step40=20.3%
  - PER+Nstep:             Step10=23.4%, Step40=26.6%
  - reward_fresh (age1000): Step10=22.7%, Step40=30.5%

结论：
  1. reward_fresh (age1000) 仍然是最佳，Step40达到30.5%
  2. PER+Nstep 在早期(Step10)略微领先
  3. 公平对比下，Baseline (adv_clip=0.2) 表现更差
  4. 之前用的 Baseline (adv_clip=20) 不公平，配置更强
""")


if __name__ == "__main__":
    analyze()
