"""
同步offline wandb到线上 (Windows版本)
支持批量同步，自动删除.synced标记文件

直接运行: python sync_wandb_windows.py
"""

import subprocess
import sys
import os
from pathlib import Path

# ========== 配置区 ==========
# 要同步的offline run目录列表
OFFLINE_PATHS = [
    r"E:\code_project\python_code\local_roll_dev\roll_dev\experiments\grpo_aime\wandb\exp1_huj8ikin",
    r"E:\code_project\python_code\local_roll_dev\roll_dev\experiments\grpo_aime\wandb\exp2_6iwiqxzx",
    r"E:\code_project\python_code\local_roll_dev\roll_dev\experiments\grpo_aime\wandb\exp3_xuazinvp",
    r"E:\code_project\python_code\local_roll_dev\roll_dev\experiments\grpo_aime\wandb\exp4_mmv8u93h",
]

# WandB配置
WANDB_API_KEY = "5d830c409e2aa7dff34c333a2f79798a877bfc7b"
WANDB_PROJECT = "roll-grpo-replay-aime"  # 必须与训练时的项目名一致

ENTITY = ""  # 留空使用个人账户
# ============================


def remove_synced_files(path: Path):
    """删除.synced标记文件，确保数据能重新上传"""
    synced_files = list(path.glob("*.synced"))
    for f in synced_files:
        print(f"  删除标记文件: {f.name}")
        f.unlink()
    return len(synced_files)


def get_wandb_file_info(path: Path):
    """获取.wandb文件信息"""
    wandb_files = list(path.glob("*.wandb"))
    if wandb_files:
        f = wandb_files[0]
        size_mb = f.stat().st_size / 1024 / 1024
        return f.name, size_mb
    return None, 0


def sync_single(path: Path, index: int, total: int):
    """同步单个offline run"""
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] 同步: {path.name}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"  错误: 路径不存在")
        return False

    # 获取.wandb文件信息
    wandb_file, size_mb = get_wandb_file_info(path)
    if wandb_file:
        print(f"  数据文件: {wandb_file} ({size_mb:.2f} MB)")

    # 删除.synced文件
    removed = remove_synced_files(path)
    if removed > 0:
        print(f"  已删除 {removed} 个.synced标记文件")

    # 设置环境变量
    env = os.environ.copy()
    env["WANDB_API_KEY"] = WANDB_API_KEY

    # 构建命令
    cmd = [sys.executable, "-m", "wandb", "sync", str(path)]
    if WANDB_PROJECT:
        cmd.extend(["--project", WANDB_PROJECT])
    if ENTITY:
        cmd.extend(["--entity", ENTITY])

    print(f"  项目: {WANDB_PROJECT}")
    print(f"  开始同步...")

    # 执行同步
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        print(f"  [OK] 同步成功!")
        # 提取URL
        for line in result.stdout.split('\n'):
            if 'wandb.ai' in line:
                print(f"  {line.strip()}")
        return True
    else:
        print(f"  [FAIL] 同步失败!")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False


def sync_all():
    """同步所有配置的offline runs"""
    print("=" * 60)
    print("WandB Offline Sync Tool")
    print("=" * 60)
    print(f"项目: {WANDB_PROJECT}")
    print(f"待同步数量: {len(OFFLINE_PATHS)}")

    # 先登录确保API key有效
    print("\n检查登录状态...")
    login_cmd = [sys.executable, "-m", "wandb", "login", "--relogin", WANDB_API_KEY]
    subprocess.run(login_cmd, capture_output=True)

    success_count = 0
    fail_count = 0

    for i, path_str in enumerate(OFFLINE_PATHS, 1):
        path = Path(path_str)
        if sync_single(path, i, len(OFFLINE_PATHS)):
            success_count += 1
        else:
            fail_count += 1

    # 汇总
    print("\n" + "=" * 60)
    print("同步完成!")
    print("=" * 60)
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"\n访问 https://wandb.ai 查看项目 {WANDB_PROJECT}")

    return fail_count == 0


if __name__ == "__main__":
    sync_all()
