"""
检查wandb离线数据文件的实际内容
"""
import os
import json
from pathlib import Path
import struct

# 离线运行路径
OFFLINE_PATH = r"E:\code_project\python_code\local_roll_dev\roll_dev\experiments\frozen_lake_2a100_replaybuffer\20260119_204444\wandb\wandb\offline-run-20260119_204523-5f8bxd0q"

def check_wandb_file():
    path = Path(OFFLINE_PATH)
    wandb_file = path / "run-5f8bxd0q.wandb"

    if not wandb_file.exists():
        print(f"文件不存在: {wandb_file}")
        return

    file_size = wandb_file.stat().st_size
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")

    try:
        import wandb
        print(f"wandb版本: {wandb.__version__}")
    except:
        pass

    # 尝试用wandb的run history API
    try:
        import wandb
        from wandb.sdk.internal.datastore import DataStore
        from wandb.proto import wandb_internal_pb2 as pb

        ds = DataStore()
        ds.open_for_scan(str(wandb_file))

        record_counts = {}
        history_count = 0
        sample_records = []

        count = 0
        while True:
            try:
                data = ds.scan_record()
                if data is None:
                    break
                count += 1

                # data 是 Record protobuf
                record_type = data.WhichOneof("record_type")
                record_counts[record_type] = record_counts.get(record_type, 0) + 1

                if record_type == "history" and history_count < 3:
                    history_count += 1
                    hist = data.history
                    items = {}
                    for item in hist.item:
                        items[item.key] = item.value_json[:100] if item.value_json else "N/A"
                    sample_records.append(items)

            except Exception as e:
                print(f"读取记录 {count} 时出错: {e}")
                break

        ds.close()

        print(f"\n总记录数: {count}")
        print("\n=== 记录类型统计 ===")
        for record_type, cnt in sorted(record_counts.items()):
            print(f"  {record_type}: {cnt}")

        if sample_records:
            print("\n=== History样本 ===")
            for i, rec in enumerate(sample_records):
                print(f"\n样本 {i+1}: {len(rec)} 个字段")
                for k, v in list(rec.items())[:5]:
                    print(f"  {k}: {v}")

    except Exception as e:
        print(f"使用wandb库解析失败: {e}")
        import traceback
        traceback.print_exc()

        # 尝试直接读取文件头
        print("\n=== 直接读取文件 ===")
        with open(wandb_file, 'rb') as f:
            header = f.read(100)
            print(f"文件头(hex): {header[:50].hex()}")
            print(f"文件头(bytes): {header[:50]}")


def check_summary():
    """检查summary文件"""
    path = Path(OFFLINE_PATH)
    summary_file = path / "files" / "wandb-summary.json"

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"\n=== Summary文件 ===")
        print(f"字段数量: {len(summary)}")
        print(f"_step: {summary.get('_step', 'N/A')}")
        print(f"_runtime: {summary.get('_runtime', 'N/A')}")

        # 检查是否有实际的训练指标
        training_keys = [k for k in summary.keys() if not k.startswith('_') and not k.startswith('memory/')]
        print(f"训练相关字段数: {len(training_keys)}")
        print(f"前10个训练字段: {training_keys[:10]}")


if __name__ == "__main__":
    check_wandb_file()
    check_summary()
