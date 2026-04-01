#!/usr/bin/env python3
"""
Create stratified train/test splits for recalibration experiments.

Splits all-benchmarks.json into:
- train.json (70% of each benchmark, stratified by healthy)
- test.json (30% of each benchmark, stratified by healthy)

Preserves healthy/unhealthy ratio within each benchmark.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from datetime import date

# Paths
INPUT_PATH = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper/data/unified/all-benchmarks.json")
OUTPUT_DIR = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper/data/unified/splits")

# Split ratio
TRAIN_RATIO = 0.70
RANDOM_SEED = 42


def compute_stats(traces: list) -> dict:
    """Compute stats for a set of traces."""
    stats = defaultdict(lambda: {"total": 0, "success": 0, "healthy": 0})
    
    for trace in traces:
        benchmark = trace["benchmark"]
        stats[benchmark]["total"] += 1
        if trace.get("success", False):
            stats[benchmark]["success"] += 1
        if trace.get("healthy", False):
            stats[benchmark]["healthy"] += 1
    
    # Compute rates
    result = {}
    total_traces = 0
    total_success = 0
    total_healthy = 0
    
    for benchmark, s in stats.items():
        result[benchmark] = {
            "total": s["total"],
            "success": s["success"],
            "healthy": s["healthy"],
            "success_rate": round(s["success"] / s["total"], 3) if s["total"] > 0 else 0,
            "healthy_rate": round(s["healthy"] / s["total"], 3) if s["total"] > 0 else 0,
        }
        total_traces += s["total"]
        total_success += s["success"]
        total_healthy += s["healthy"]
    
    result["total"] = {
        "traces": total_traces,
        "success": total_success,
        "healthy": total_healthy,
        "success_rate": round(total_success / total_traces, 3) if total_traces > 0 else 0,
        "healthy_rate": round(total_healthy / total_traces, 3) if total_traces > 0 else 0,
    }
    
    return result


def stratified_split(traces: list, train_ratio: float, seed: int) -> tuple:
    """
    Stratified split by benchmark and healthy status.
    Returns (train_traces, test_traces).
    """
    random.seed(seed)
    
    # Group traces by (benchmark, healthy)
    groups = defaultdict(list)
    for trace in traces:
        benchmark = trace["benchmark"]
        healthy = trace.get("healthy", False)
        groups[(benchmark, healthy)].append(trace)
    
    train_traces = []
    test_traces = []
    
    for (benchmark, healthy), group_traces in groups.items():
        # Shuffle the group
        shuffled = group_traces.copy()
        random.shuffle(shuffled)
        
        # Split
        n_train = int(len(shuffled) * train_ratio)
        train_traces.extend(shuffled[:n_train])
        test_traces.extend(shuffled[n_train:])
    
    return train_traces, test_traces


def main():
    print(f"Loading {INPUT_PATH}...")
    
    # Load the input file using streaming to handle large file
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)
    
    traces = data["traces"]
    print(f"Loaded {len(traces)} traces")
    
    # Show original stats
    print("\n=== Original Stats ===")
    for benchmark, s in data["stats"].items():
        if benchmark != "total":
            print(f"  {benchmark}: {s['total']} traces, healthy_rate={s['healthy_rate']:.3f}")
    print(f"  TOTAL: {data['stats']['total']['traces']} traces")
    
    # Perform stratified split
    print(f"\n=== Splitting (train={TRAIN_RATIO*100:.0f}%, test={100-TRAIN_RATIO*100:.0f}%) ===")
    train_traces, test_traces = stratified_split(traces, TRAIN_RATIO, RANDOM_SEED)
    
    # Compute stats for each split
    train_stats = compute_stats(train_traces)
    test_stats = compute_stats(test_traces)
    
    # Create output documents
    train_doc = {
        "version": "1.0.0",
        "created": str(date.today()),
        "description": "Training split (70%) for recalibration - stratified by benchmark and healthy",
        "split": "train",
        "stats": train_stats,
        "traces": train_traces,
    }
    
    test_doc = {
        "version": "1.0.0",
        "created": str(date.today()),
        "description": "Test split (30%) for recalibration - stratified by benchmark and healthy",
        "split": "test",
        "stats": test_stats,
        "traces": test_traces,
    }
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = OUTPUT_DIR / "train.json"
    test_path = OUTPUT_DIR / "test.json"
    
    print(f"\nSaving train split to {train_path}...")
    with open(train_path, 'w') as f:
        json.dump(train_doc, f, indent=2)
    
    print(f"Saving test split to {test_path}...")
    with open(test_path, 'w') as f:
        json.dump(test_doc, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    
    print("\n--- TRAIN SET ---")
    for benchmark in sorted([b for b in train_stats.keys() if b != "total"]):
        s = train_stats[benchmark]
        print(f"  {benchmark:12}: {s['total']:4} traces, healthy_rate={s['healthy_rate']:.3f}")
    print(f"  {'TOTAL':12}: {train_stats['total']['traces']:4} traces, healthy_rate={train_stats['total']['healthy_rate']:.3f}")
    
    print("\n--- TEST SET ---")
    for benchmark in sorted([b for b in test_stats.keys() if b != "total"]):
        s = test_stats[benchmark]
        print(f"  {benchmark:12}: {s['total']:4} traces, healthy_rate={s['healthy_rate']:.3f}")
    print(f"  {'TOTAL':12}: {test_stats['total']['traces']:4} traces, healthy_rate={test_stats['total']['healthy_rate']:.3f}")
    
    # Verify healthy rate preservation
    print("\n--- HEALTHY RATE PRESERVATION CHECK ---")
    all_ok = True
    for benchmark in sorted([b for b in data["stats"].keys() if b != "total"]):
        orig = data["stats"][benchmark]["healthy_rate"]
        train = train_stats.get(benchmark, {}).get("healthy_rate", 0)
        test = test_stats.get(benchmark, {}).get("healthy_rate", 0)
        train_diff = abs(train - orig) * 100
        test_diff = abs(test - orig) * 100
        status = "✓" if train_diff <= 5 and test_diff <= 5 else "✗"
        if status == "✗":
            all_ok = False
        print(f"  {benchmark:12}: orig={orig:.3f}, train={train:.3f} (Δ{train_diff:.1f}%), test={test:.3f} (Δ{test_diff:.1f}%) {status}")
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ All healthy rates preserved within ±5%")
    else:
        print("✗ Some healthy rates differ by more than ±5%")
    print("="*60)


if __name__ == "__main__":
    main()
