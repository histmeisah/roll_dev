"""
从WandB下载FrozenLake实验数据用于分析

使用方法:
    python download_wandb_data.py

输出:
    - results/all_experiments_success.csv  (val/env/FrozenLake/success)
    - results/all_experiments_score.csv    (val/score/mean)
    - results/all_experiments_summary.csv  (实验配置汇总)
"""

import wandb
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ========== 配置区 ==========
WANDB_API_KEY = "5d830c409e2aa7dff34c333a2f79798a877bfc7b"
WANDB_ENTITY = "740988193-institute-of-automation-chinese-academy-of-sci"
WANDB_PROJECT = "roll-frozen-lake-2a100"

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "used_config" / "results"

# 要下载的指标
METRICS_TO_DOWNLOAD = [
    "val/env/FrozenLake/success",
    "val/score/mean",
    "actor_train/grad_norm",
    "offpolicy/importance_weight/mean",
]
# ============================


def download_all_runs():
    """下载所有runs的数据"""
    print("=" * 60)
    print("WandB Data Downloader")
    print("=" * 60)
    print(f"Entity: {WANDB_ENTITY}")
    print(f"Project: {WANDB_PROJECT}")

    # 登录
    wandb.login(key=WANDB_API_KEY)
    api = wandb.Api()

    # 获取所有runs
    runs_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    print(f"\n获取runs: {runs_path}")
    runs = api.runs(runs_path)

    print(f"找到 {len(runs)} 个runs")

    # 存储数据
    all_data = defaultdict(dict)  # metric -> {run_name: {step: value}}
    run_configs = []

    for i, run in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] 下载: {run.name} (id: {run.id})")
        print(f"  状态: {run.state}")

        # 保存配置
        config_info = {
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "created_at": run.created_at,
        }

        # 提取关键配置
        config = run.config
        if config:
            config_info["exp_name"] = config.get("exp_name", "")

            # replay配置
            replay_config = config.get("replay", {})
            if replay_config:
                config_info["replay_enabled"] = replay_config.get("enabled", False)
                config_info["priority_function"] = replay_config.get("priority_function", "")
                config_info["enable_age_decay"] = replay_config.get("enable_age_decay", False)
                config_info["age_decay"] = replay_config.get("age_decay", "")
                config_info["importance_sampling_correction"] = replay_config.get("importance_sampling_correction", False)

        run_configs.append(config_info)

        # 下载历史数据 (使用scan_history获取完整数据)
        try:
            history = list(run.scan_history(keys=METRICS_TO_DOWNLOAD + ["_step"]))
            print(f"  数据点: {len(history)}")

            for row in history:
                step = row.get("_step", 0)
                for metric in METRICS_TO_DOWNLOAD:
                    if metric in row and row[metric] is not None:
                        if run.name not in all_data[metric]:
                            all_data[metric][run.name] = {}
                        all_data[metric][run.name][step] = row[metric]
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 保存数据
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 保存配置汇总
    config_df = pd.DataFrame(run_configs)
    config_path = OUTPUT_DIR / "all_experiments_summary.csv"
    config_df.to_csv(config_path, index=False)
    print(f"\n保存配置汇总: {config_path}")

    # 2. 保存各指标数据
    for metric in METRICS_TO_DOWNLOAD:
        if metric not in all_data:
            continue

        # 转换为DataFrame
        metric_data = all_data[metric]

        # 获取所有steps
        all_steps = set()
        for run_name, steps_data in metric_data.items():
            all_steps.update(steps_data.keys())
        all_steps = sorted(all_steps)

        # 构建DataFrame
        df_data = {"Step": all_steps}
        for run_name, steps_data in metric_data.items():
            df_data[run_name] = [steps_data.get(step, None) for step in all_steps]

        df = pd.DataFrame(df_data)

        # 保存
        metric_filename = metric.replace("/", "_").replace("\\", "_")
        output_path = OUTPUT_DIR / f"all_{metric_filename}.csv"
        df.to_csv(output_path, index=False)
        print(f"保存 {metric}: {output_path}")

    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)

    return all_data, run_configs


def analyze_data():
    """分析下载的数据"""
    success_path = OUTPUT_DIR / "all_val_env_FrozenLake_success.csv"
    if not success_path.exists():
        print("请先运行 download_all_runs() 下载数据")
        return

    df = pd.read_csv(success_path)
    print("\n" + "=" * 60)
    print("数据分析: val/env/FrozenLake/success")
    print("=" * 60)

    # 显示列名（实验名）
    experiments = [col for col in df.columns if col != "Step"]
    print(f"\n共 {len(experiments)} 个实验:")
    for exp in experiments:
        print(f"  - {exp}")

    # 早期收敛对比 (Step 10, 20, 40)
    print("\n早期收敛对比:")
    print("-" * 80)
    print(f"{'实验名':<50} {'Step10':>10} {'Step20':>10} {'Step40':>10}")
    print("-" * 80)

    for exp in experiments:
        step10 = df[df["Step"] == 10][exp].values
        step20 = df[df["Step"] == 20][exp].values
        step40 = df[df["Step"] == 40][exp].values

        s10 = f"{step10[0]*100:.1f}%" if len(step10) > 0 and pd.notna(step10[0]) else "N/A"
        s20 = f"{step20[0]*100:.1f}%" if len(step20) > 0 and pd.notna(step20[0]) else "N/A"
        s40 = f"{step40[0]*100:.1f}%" if len(step40) > 0 and pd.notna(step40[0]) else "N/A"

        # 截断实验名
        exp_short = exp[:48] + ".." if len(exp) > 50 else exp
        print(f"{exp_short:<50} {s10:>10} {s20:>10} {s40:>10}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_data()
    else:
        download_all_runs()
        analyze_data()
