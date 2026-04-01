#!/usr/bin/env python3
"""Fast targeted tuning with reduced grid."""

import json
import statistics
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.metrics import roc_auc_score


def detect_signals(trace, config):
    turns = trace.get('turns', [])
    if len(turns) < 2:
        return False, 0
    
    signals = 0
    
    # Confidence
    confidences = [t.get('confidence', 0.5) for t in turns 
                   if t.get('role') in ('agent', 'ai', 'assistant') and t.get('confidence')]
    if confidences:
        if statistics.mean(confidences) < config['conf_min']:
            signals += 1
        elif len(confidences) >= 2 and (confidences[0] - confidences[-1]) > config['conf_drop']:
            signals += 1
    
    # Loop detection
    agent_turns = [t.get('content', '') for t in turns if t.get('role') in ('agent', 'ai', 'assistant')]
    if len(agent_turns) >= 4:
        all_content = ' '.join(agent_turns).lower()
        for marker in ['loop detected', 'stuck in a loop', 'early stop', 'same action']:
            if marker in all_content:
                signals += 1
                break
    
    # Context pressure 
    if len(turns) >= config['min_pressure']:
        tokens = []
        for t in turns:
            tc = t.get('token_count') or len(t.get('content', '')) // 4
            tokens.append(tc)
        if tokens and max(tokens) > 0:
            rates = [tokens[i]/tokens[i-1] for i in range(1, len(tokens)) if tokens[i-1] > 10]
            if rates and sum(1 for r in rates if r > 1.3) / len(rates) > 0.5:
                signals += 1
    
    return signals >= config['min_signals'], signals


def evaluate(traces, config):
    predictions = []
    tp = fp = tn = fn = 0
    
    for trace in traces:
        gt = trace.get('healthy')
        if gt is None:
            continue
        pred, sig = detect_signals(trace, config)
        predictions.append((gt, sig))
        
        if pred and not gt: tp += 1
        elif pred and gt: fp += 1
        elif not pred and not gt: fn += 1
        else: tn += 1
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    if predictions:
        y_true = np.array([0 if p[0] else 1 for p in predictions])
        y_scores = np.array([p[1] for p in predictions])
        try:
            auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) >= 2 else 0.5
        except:
            auc = 0.5
    else:
        auc = 0.5
    
    return {'tpr': tpr, 'fpr': fpr, 'auc': auc}


def main():
    workspace = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper")
    
    print("Loading train data...")
    with open(workspace / "data/unified/splits/train.json") as f:
        data = json.load(f)
    traces = data.get('traces', [])
    print(f"Loaded {len(traces)} traces")
    
    if_targets = {'gaia': 0.567, 'webarena': 0.747, 'agentbench': 0.739, 'swe-bench': 0.337}
    benchmarks = list(if_targets.keys())
    
    # Smaller grid
    grid = {
        'conf_min': [0.35, 0.45, 0.5],
        'conf_drop': [0.2, 0.3],
        'min_signals': [1, 2],
        'min_pressure': [3, 5],
    }
    
    combos = list(product(*grid.values()))
    print(f"Testing {len(combos)} combos...")
    
    best_wins = 0
    best_config = None
    best_results = {}
    
    for i, combo in enumerate(combos):
        config = dict(zip(grid.keys(), combo))
        wins = 0
        results = {}
        
        for bm in benchmarks:
            bm_traces = [t for t in traces if t.get('benchmark') == bm]
            if bm_traces:
                m = evaluate(bm_traces, config)
                results[bm] = m
                if m['auc'] > if_targets[bm]:
                    wins += 1
        
        if wins > best_wins:
            best_wins = wins
            best_config = config
            best_results = results
            print(f"New best: {wins} wins")
    
    print(f"\nBest: {best_wins}/4 wins")
    print(f"Config: {best_config}")
    for bm, m in best_results.items():
        print(f"  {bm}: {m['auc']:.3f} vs IF {if_targets[bm]:.3f}")
    
    # Save
    output = {
        'best_thresholds': {
            'confidence_threshold': best_config['conf_min'],
            'confidence_drop': best_config['conf_drop'],
            'min_signals': best_config['min_signals'],
            'min_turns_for_pressure': best_config['min_pressure'],
            'growth_rate': 2.0,
        },
        'train_metrics': {'wins_vs_if': best_wins, 'per_benchmark': best_results},
        'if_targets': if_targets,
    }
    
    with open(workspace / "results/tuned-thresholds.json", 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved thresholds")


if __name__ == "__main__":
    main()
