"""
Expand AIME dataset by repeating problems
"""
import json

print("Reading AIME data...")
with open("aime/train.jsonl", 'r', encoding='utf-8') as f:
    problems = [json.loads(line) for line in f]

print(f"Original: {len(problems)} problems")

# 重复100次,从30个变成3000个
target_count = 3000
repeat_times = (target_count // len(problems)) + 1

expanded = problems * repeat_times
expanded = expanded[:target_count]  # 精确到3000个

print(f"Expanded: {len(expanded)} problems")

# 保存扩充后的数据
with open("aime/train_expanded.jsonl", 'w', encoding='utf-8') as f:
    for item in expanded:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Saved to aime/train_expanded.jsonl")
print("Done!")
