import csv
import statistics

def parse_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

    mean_cols = {}
    for i, h in enumerate(headers[1:], 1):
        if '__MIN' not in h and '__MAX' not in h:
            mean_cols[i] = h.split(' - ')[0].strip()

    data = {name: {} for name in mean_cols.values()}

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            step = int(row[0])
            for col_idx, name in mean_cols.items():
                val = row[col_idx].strip()
                if val:
                    data[name][step] = float(val)
    return data

def categorize(data):
    groups = {
        'Baseline (no replay)': [],
        'Baseline (PPO)': [],
        'Standard PER': [],
        'Freshness (early)': [],
        'Freshness (constantLR)': [],
        'Hierarchical GAE': [],
    }
    for name in data:
        if 'traj_baseline_no_replay' in name:
            groups['Baseline (no replay)'].append(name)
        elif 'traj_ppo_baseline' in name:
            groups['Baseline (PPO)'].append(name)
        elif 'standard_PER_constant_lr' in name and 'age_advantage' not in name:
            groups['Standard PER'].append(name)
        elif 'standard_PER_constantLR_trajrb_age_advantage' in name:
            groups['Freshness (constantLR)'].append(name)
        elif 'trajrb_age_advantage_priority' in name and 'standard_PER' not in name:
            groups['Freshness (early)'].append(name)
        elif 'hierarchical_gae' in name:
            groups['Hierarchical GAE'].append(name)
    return groups

def group_stats(data, names):
    all_steps = sorted(set(s for n in names for s in data[n]))
    results = {}
    for step in all_steps:
        vals = [data[n][step] for n in names if step in data[n] and (data[n][step] > 0 or step <= 15)]
        if vals:
            results[step] = {
                'mean': statistics.mean(vals),
                'std': statistics.stdev(vals) if len(vals) > 1 else 0,
                'n': len(vals),
                'max': max(vals),
                'min': min(vals),
            }
    return results

# ===== Parse =====
score = parse_csv('nq_search_val_score_mean.csv')
success = parse_csv('nq_search_val_success_mean.csv')

sg = categorize(score)
ug = categorize(success)

print("=" * 80)
print("NQ Search 实验结果分析")
print("=" * 80)

# Run overview
print("\n[1] 实验分组概览")
for g, names in sg.items():
    if names:
        print(f"\n  {g} ({len(names)} runs):")
        for n in names:
            steps = sorted(score[n].keys())
            nonzero = [s for s in steps if score[n][s] > 0 or s <= 15]
            print(f"    {n[:50]}...")
            print(f"      range: step 0-{steps[-1]}, last_valid: {nonzero[-1] if nonzero else 'N/A'}")

# Also baseline no replay (only in success CSV)
if ug.get('Baseline (no replay)'):
    print(f"\n  Baseline (no replay) ({len(ug['Baseline (no replay)'])} runs) [success CSV only]:")
    for n in ug['Baseline (no replay)']:
        steps = sorted(success[n].keys())
        print(f"    {n[:50]}...")
        print(f"      range: step 0-{steps[-1]}")

# Key comparison table
print("\n" + "=" * 80)
print("[2] Score Mean 对比表 (每5步)")
print("=" * 80)

key_groups = ['Baseline (PPO)', 'Standard PER', 'Freshness (early)', 'Freshness (constantLR)']
stats_cache = {g: group_stats(score, sg[g]) for g in key_groups if sg.get(g)}

print(f"\n{'Step':>5} | {'Baseline(PPO)':>16} | {'Standard PER':>16} | {'Fresh(early)':>16} | {'Fresh(constLR)':>16}")
print("-" * 80)

for step in range(0, 205, 5):
    line = f"{step:>5}"
    for g in key_groups:
        st = stats_cache.get(g, {})
        if step in st:
            s = st[step]
            line += f" | {s['mean']:>7.4f}(n={s['n']})"
        else:
            line += f" | {'---':>16}"
    print(line)

# Best performance
print("\n" + "=" * 80)
print("[3] 各方法最佳表现 (Score Mean)")
print("=" * 80)

for g in key_groups:
    if not sg.get(g):
        continue
    print(f"\n  {g}:")
    bests = []
    for name in sg[g]:
        valid = [(s, v) for s, v in score[name].items() if v > 0 or s <= 15]
        if valid:
            bs, bv = max(valid, key=lambda x: x[1])
            bests.append((name[:20], bs, bv))
            print(f"    {name[:20]}: best = {bv:.4f} @ step {bs}")
    if bests:
        avg_best = statistics.mean([v for _, _, v in bests])
        print(f"    >>> Avg best: {avg_best:.4f}")

# Converged performance (step 75-165)
print("\n" + "=" * 80)
print("[4] 收敛后平均表现 (Step 75-165)")
print("=" * 80)

print(f"\n  {'Method':>30} | {'Score Mean':>20} | {'N points':>10}")
print("  " + "-" * 65)
for g in key_groups:
    if not sg.get(g):
        continue
    vals = []
    for name in sg[g]:
        vals.extend([v for s, v in score[name].items() if 75 <= s <= 165 and v > 0])
    if vals:
        print(f"  {g:>30} | {statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f} | {len(vals):>10}")

# Success rate comparison (includes baseline no replay)
print("\n" + "=" * 80)
print("[5] Success Rate 对比 (收敛后 Step 75-165)")
print("=" * 80)

succ_key_groups = ['Baseline (no replay)', 'Baseline (PPO)', 'Standard PER', 'Freshness (early)', 'Freshness (constantLR)']
print(f"\n  {'Method':>30} | {'Success Rate':>20} | {'N points':>10}")
print("  " + "-" * 65)
for g in succ_key_groups:
    names = ug.get(g, [])
    if not names:
        continue
    vals = []
    for name in names:
        vals.extend([v for s, v in success[name].items() if 75 <= s <= 165 and v > 0])
    if vals:
        print(f"  {g:>30} | {statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f} | {len(vals):>10}")

# Late-stage stability
print("\n" + "=" * 80)
print("[6] 后期稳定性 (Step 165+)")
print("=" * 80)

for g in ['Standard PER', 'Freshness (early)', 'Freshness (constantLR)']:
    names = sg.get(g, [])
    if not names:
        continue
    print(f"\n  {g}:")
    for name in names:
        late = [(s, v) for s, v in score[name].items() if s >= 165]
        if late:
            all_v = [v for _, v in late]
            nonzero = [v for v in all_v if v > 0]
            crashed = len(all_v) - len(nonzero)
            if nonzero:
                print(f"    {name[:40]}: avg={statistics.mean(nonzero):.4f}, steps={len(late)}, crashed={crashed}")
            else:
                print(f"    {name[:40]}: ALL CRASHED")

# Improvement over baseline
print("\n" + "=" * 80)
print("[7] 相对于 Baseline 的提升幅度")
print("=" * 80)

# Use success rate step 75-165 as main metric
baseline_vals = []
for name in ug.get('Baseline (no replay)', []):
    baseline_vals.extend([v for s, v in success[name].items() if 75 <= s <= 165 and v > 0])

if baseline_vals:
    baseline_mean = statistics.mean(baseline_vals)
    print(f"\n  Baseline (no replay) success rate: {baseline_mean:.4f}")

    for g in ['Standard PER', 'Freshness (early)', 'Freshness (constantLR)']:
        names = ug.get(g, [])
        if not names:
            continue
        vals = []
        for name in names:
            vals.extend([v for s, v in success[name].items() if 75 <= s <= 165 and v > 0])
        if vals:
            m = statistics.mean(vals)
            impr = (m - baseline_mean) / baseline_mean * 100
            print(f"  {g:>30}: {m:.4f} (improvement: {impr:+.1f}%)")
