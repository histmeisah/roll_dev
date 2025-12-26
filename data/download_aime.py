"""
Download AIME 2024 dataset
"""
from datasets import load_dataset
import json
import os

# Create directory
os.makedirs("aime", exist_ok=True)

print("Downloading AIME 2024 dataset...")

# Load AIME 2024 from HuggingFace
dataset = load_dataset("HuggingFaceH4/aime_2024")

print(f"Dataset loaded! Splits: {list(dataset.keys())}")

# Process and save
for split_name in dataset.keys():
    data = dataset[split_name]
    print(f"\n{split_name}: {len(data)} problems")

    # Save to JSONL
    output_file = f"aime/{split_name}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ Saved to {output_file}")

print(f"\n✓ Complete! Files in: {os.path.abspath('aime')}")
