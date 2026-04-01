#!/usr/bin/env python3
"""
Proper Threshold Tuning on Train Set

Does a grid search over detector thresholds to find optimal configuration.
Targets maximizing AUC on the training set.
"""

import json
import statistics
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from itertools import product


@dataclass
class DetectorConfig:
    """Detector thresholds."""
    confidence_min_threshold: float = 0.4
    confidence_drop_threshold: float = 0.3
    contradiction_rate_threshold: float = 0.25
    ngram_overlap_threshold: float = 0.85
    ngram_min_turns: int = 4
    token_growth_rate_threshold: float = 1.5
    min_turns_for_pressure: int = 5
    latency_std_threshold: float = 3.0
    min_turns: int = 2
    min_signals_for_failure: int = 1


class GreyFailureDetector:
    """Simplified detector for fast evaluation."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
    
    def detect_confidence_collapse(self, turns: list) -> bool:
        confidences = [
            t.get('confidence', 0.5) 
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if not confidences:
            return False
        
        mean_conf = statistics.mean(confidences)
        if mean_conf < self.config.confidence_min_threshold:
            return True
        
        if len(confidences) >= 2:
            drop = confidences[0] - confidences[-1]
            if drop > self.config.confidence_drop_threshold:
                return True
        
        return False
    
    def detect_reasoning_loop(self, turns: list) -> bool:
        agent_turns = [
            t.get('content', '').strip()
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if len(agent_turns) < self.config.ngram_min_turns:
            return False
        
        loop_markers = [
            'same action has been', 'loop detected', 'repeated the same',
            'stuck in a loop', 'early stop', 'maximum iterations',
        ]
        
        all_content = ' '.join(agent_turns).lower()
        for marker in loop_markers:
            if marker in all_content:
                return True
        
        consecutive_identical = 0
        for i in range(1, len(agent_turns)):
            if agent_turns[i] == agent_turns[i-1] and len(agent_turns[i]) > 100:
                consecutive_identical += 1
                if consecutive_identical >= 3:
                    return True
            else:
                consecutive_identical = 0
        
        return False
    
    def detect_context_pressure(self, turns: list) -> bool:
        if len(turns) < self.config.min_turns_for_pressure:
            return False
        
        token_counts = []
        for t in turns:
            tc = t.get('token_count') or t.get('tokens_used')
            if tc is None:
                content = t.get('content', '')
                tc = len(content) // 4 if content else 0
            token_counts.append(tc)
        
        if not token_counts or max(token_counts) == 0:
            return False
        
        growth_rates = []
        for i in range(1, len(token_counts)):
            if token_counts[i-1] > 10:
                rate = token_counts[i] / token_counts[i-1]
                growth_rates.append(rate)
        
        if not growth_rates:
            return False
        
        n = len(growth_rates)
        if n >= 6:
            first_third = growth_rates[:n//3]
            last_third = growth_rates[-n//3:]
            avg_first = statistics.mean(first_third) if first_third else 1.0
            avg_last = statistics.mean(last_third) if last_third else 1.0
            
            if avg_first > 0 and avg_last / avg_first > self.config.token_growth_rate_threshold:
                return True
        
        high_growth_turns = sum(1 for r in growth_rates if r > 1.3)
        high_growth_ratio = high_growth_turns / len(growth_rates)
        
        return high_growth_ratio > 0.5
    
    def detect(self, trace: dict) -> tuple:
        """Returns (predicted_failure, num_signals)"""
        turns = trace.get('turns', [])
        
        if len(turns) < self.config.min_turns:
            return False, 0
        
        signals = 0
        if self.detect_confidence_collapse(turns):
            signals += 1
        if self.detect_reasoning_loop(turns):
            signals += 1
        if self.detect_context_pressure(turns):
            signals += 1
        
        predicted_failure = signals >= self.config.min_signals_for_failure
        return predicted_failure, signals


def compute_auc(predictions: list) -> float:
    """Compute AUC from predictions."""
    # predictions = [(gt_healthy, num_signals), ...]
    if not predictions:
        return 0.5
    
    y_true = np.array([0 if p[0] else 1 for p in predictions])  # 1 = failure
    y_scores = np.array([p[1] for p in predictions])  # num_signals
    
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    # Simple AUC calculation
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_scores)
    except:
        return 0.5


def evaluate_config(traces: list, config: DetectorConfig) -> dict:
    """Evaluate a config on traces."""
    detector = GreyFailureDetector(config)
    predictions = []
    tp = fp = tn = fn = 0
    
    for trace in traces:
        gt_healthy = trace.get('healthy')
        if gt_healthy is None:
            continue
        
        pred_failure, num_signals = detector.detect(trace)
        predictions.append((gt_healthy, num_signals))
        
        gt_failure = not gt_healthy
        if pred_failure and gt_failure:
            tp += 1
        elif pred_failure and not gt_failure:
            fp += 1
        elif not pred_failure and gt_failure:
            fn += 1
        else:
            tn += 1
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    auc = compute_auc(predictions)
    
    return {
        'tpr': tpr,
        'fpr': fpr,
        'f1': f1,
        'auc': auc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def grid_search(traces: list, benchmarks: list = None) -> dict:
    """Grid search over thresholds."""
    
    # Define grid
    param_grid = {
        'confidence_min_threshold': [0.3, 0.4, 0.5],
        'confidence_drop_threshold': [0.2, 0.3, 0.4],
        'token_growth_rate_threshold': [1.3, 1.5, 2.0],
        'min_signals_for_failure': [1, 2],
        'min_turns_for_pressure': [3, 5, 7],
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"Testing {len(combinations)} combinations...")
    
    best_score = -1
    best_config = None
    best_results = None
    
    for i, combo in enumerate(combinations):
        config = DetectorConfig()
        for k, v in zip(keys, combo):
            setattr(config, k, v)
        
        # Evaluate on each benchmark
        total_auc = 0
        benchmark_results = {}
        
        for benchmark in benchmarks or ['gaia', 'webarena', 'agentbench', 'swe-bench']:
            bm_traces = [t for t in traces if t.get('benchmark') == benchmark]
            if not bm_traces:
                continue
            
            metrics = evaluate_config(bm_traces, config)
            benchmark_results[benchmark] = metrics
            total_auc += metrics['auc']
        
        avg_auc = total_auc / len(benchmark_results) if benchmark_results else 0
        
        if avg_auc > best_score:
            best_score = avg_auc
            best_config = config
            best_results = benchmark_results
            
        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{len(combinations)}, best avg AUC: {best_score:.4f}")
    
    return {
        'best_config': best_config,
        'best_score': best_score,
        'best_results': best_results
    }


def main():
    workspace = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper")
    train_path = workspace / "data/unified/splits/train.json"
    output_path = workspace / "results/tuned-thresholds.json"
    
    print(f"Loading training data from {train_path}...")
    with open(train_path) as f:
        train_data = json.load(f)
    
    traces = train_data.get('traces', [])
    print(f"Loaded {len(traces)} training traces")
    
    # Show distribution
    benchmarks = ['gaia', 'webarena', 'agentbench', 'swe-bench']
    for bm in benchmarks:
        bm_traces = [t for t in traces if t.get('benchmark') == bm]
        n_healthy = sum(1 for t in bm_traces if t.get('healthy'))
        print(f"  {bm}: {len(bm_traces)} traces ({n_healthy} healthy)")
    
    # Run grid search
    print("\nRunning grid search...")
    results = grid_search(traces, benchmarks)
    
    best_config = results['best_config']
    print(f"\nBest configuration (avg AUC: {results['best_score']:.4f}):")
    print(f"  confidence_min_threshold: {best_config.confidence_min_threshold}")
    print(f"  confidence_drop_threshold: {best_config.confidence_drop_threshold}")
    print(f"  token_growth_rate_threshold: {best_config.token_growth_rate_threshold}")
    print(f"  min_signals_for_failure: {best_config.min_signals_for_failure}")
    print(f"  min_turns_for_pressure: {best_config.min_turns_for_pressure}")
    
    print("\nPer-benchmark training results:")
    for bm, metrics in results['best_results'].items():
        print(f"  {bm}: AUC={metrics['auc']:.3f}, TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
    
    # Save tuned thresholds
    output = {
        'best_thresholds': {
            'confidence_threshold': best_config.confidence_min_threshold,
            'confidence_drop': best_config.confidence_drop_threshold,
            'growth_rate': best_config.token_growth_rate_threshold,
            'min_signals': best_config.min_signals_for_failure,
            'min_turns_for_pressure': best_config.min_turns_for_pressure,
        },
        'train_metrics': {
            'avg_auc': results['best_score'],
            'per_benchmark': {
                bm: {k: round(v, 4) for k, v in m.items()} 
                for bm, m in results['best_results'].items()
            }
        },
        'note': 'Grid search on train set (70%)'
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved tuned thresholds to {output_path}")


if __name__ == "__main__":
    main()
