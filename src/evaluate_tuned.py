#!/usr/bin/env python3
"""
Evaluate Tuned Detector on Test Data

Runs:
1. Our tuned multi-signal detector on the TEST split
2. IF, LOF, OCSVM baselines on the TEST split
3. Generates comparison table

Input:
- data/unified/splits/test.json - Test traces
- results/tuned-thresholds.json - Tuned thresholds (currently default)

Output:
- results/recalibrated-comparison.json - Per-benchmark comparison
- results/COMPARISON-TABLE.md - Markdown summary table
"""

import json
import numpy as np
import statistics
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Detector Configuration (matches run_detector.py structure)
# =============================================================================

@dataclass
class DetectorConfig:
    """Detector thresholds - can be tuned."""
    confidence_min_threshold: float = 0.4
    confidence_drop_threshold: float = 0.3
    semantic_similarity_threshold: float = 0.7
    contradiction_rate_threshold: float = 0.25
    ngram_overlap_threshold: float = 0.85
    ngram_min_turns: int = 4
    ngram_size: int = 4
    token_growth_rate_threshold: float = 1.5
    token_utilization_threshold: float = 0.85
    min_turns_for_pressure: int = 5
    latency_std_threshold: float = 3.0
    min_turns: int = 2
    min_signals_for_failure: int = 1


def load_thresholds_from_json(path: Path) -> DetectorConfig:
    """Load thresholds from tuned-thresholds.json if available."""
    config = DetectorConfig()
    
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        
        thresholds = data.get('best_thresholds', {})
        
        # Map from tuned format to config
        if 'confidence_threshold' in thresholds:
            config.confidence_min_threshold = thresholds['confidence_threshold']
        if 'confidence_drop' in thresholds:
            config.confidence_drop_threshold = thresholds['confidence_drop']
        if 'contradiction_ratio' in thresholds:
            config.contradiction_rate_threshold = thresholds['contradiction_ratio']
        if 'loop_similarity' in thresholds:
            config.ngram_overlap_threshold = thresholds['loop_similarity']
        if 'growth_rate' in thresholds:
            config.token_growth_rate_threshold = thresholds['growth_rate']
        if 'min_signals' in thresholds:
            config.min_signals_for_failure = thresholds['min_signals']
        if 'min_turns_for_pressure' in thresholds:
            config.min_turns_for_pressure = thresholds['min_turns_for_pressure']
        
        print(f"  Loaded thresholds: conf_min={config.confidence_min_threshold}, "
              f"conf_drop={config.confidence_drop_threshold}, "
              f"growth_rate={config.token_growth_rate_threshold}, "
              f"min_signals={config.min_signals_for_failure}, "
              f"min_turns_pressure={config.min_turns_for_pressure}")
    
    return config


# =============================================================================
# Grey Failure Detector (from run_detector.py)
# =============================================================================

def extract_ngrams(text: str, n: int = 3) -> set:
    """Extract word n-grams from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


class GreyFailureDetector:
    """Multi-signal grey failure detector."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
    
    def detect_confidence_collapse(self, turns: list[dict]) -> tuple[bool, dict]:
        """Detect low or dropping confidence."""
        confidences = [
            t.get('confidence', 0.5) 
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if not confidences:
            return False, {}
        
        mean_conf = statistics.mean(confidences)
        
        if mean_conf < self.config.confidence_min_threshold:
            return True, {'signal': 'confidence_collapse', 'mean_confidence': mean_conf}
        
        if len(confidences) >= 2:
            drop = confidences[0] - confidences[-1]
            if drop > self.config.confidence_drop_threshold:
                return True, {'signal': 'confidence_collapse', 'drop': drop}
        
        return False, {'mean_confidence': mean_conf}
    
    def detect_reasoning_loop(self, turns: list[dict]) -> tuple[bool, dict]:
        """Detect reasoning loops (stuck states)."""
        agent_turns = [
            t.get('content', '').strip()
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if len(agent_turns) < self.config.ngram_min_turns:
            return False, {}
        
        loop_markers = [
            'same action has been', 'loop detected', 'repeated the same',
            'stuck in a loop', 'early stop', 'maximum iterations',
            'max retries', 'tried too many times',
        ]
        
        all_content = ' '.join(agent_turns).lower()
        for marker in loop_markers:
            if marker in all_content:
                return True, {'signal': 'reasoning_loop', 'marker_found': marker}
        
        # Check consecutive identical turns
        consecutive_identical = 0
        max_consecutive = 0
        for i in range(1, len(agent_turns)):
            if agent_turns[i] == agent_turns[i-1] and len(agent_turns[i]) > 100:
                consecutive_identical += 1
                max_consecutive = max(max_consecutive, consecutive_identical)
            else:
                consecutive_identical = 0
        
        if max_consecutive >= 3:
            return True, {'signal': 'reasoning_loop', 'max_consecutive': max_consecutive + 1}
        
        return False, {}
    
    def detect_context_pressure(self, turns: list[dict]) -> tuple[bool, dict]:
        """Detect context pressure / token growth anomalies."""
        if len(turns) < self.config.min_turns_for_pressure:
            return False, {}
        
        token_counts = []
        for t in turns:
            tc = t.get('token_count') or t.get('tokens_used')
            if tc is None:
                content = t.get('content', '')
                tc = len(content) // 4 if content else 0
            token_counts.append(tc)
        
        if not token_counts or max(token_counts) == 0:
            return False, {}
        
        growth_rates = []
        for i in range(1, len(token_counts)):
            if token_counts[i-1] > 10:
                rate = token_counts[i] / token_counts[i-1]
                growth_rates.append(rate)
        
        if not growth_rates:
            return False, {}
        
        n = len(growth_rates)
        if n >= 6:
            first_third = growth_rates[:n//3]
            last_third = growth_rates[-n//3:]
            avg_first = statistics.mean(first_third) if first_third else 1.0
            avg_last = statistics.mean(last_third) if last_third else 1.0
            
            if avg_first > 0 and avg_last / avg_first > self.config.token_growth_rate_threshold:
                return True, {'signal': 'context_pressure', 'acceleration': avg_last / avg_first}
        
        high_growth_turns = sum(1 for r in growth_rates if r > 1.3)
        high_growth_ratio = high_growth_turns / len(growth_rates) if growth_rates else 0
        
        if high_growth_ratio > 0.5:
            return True, {'signal': 'context_pressure', 'high_growth_ratio': high_growth_ratio}
        
        return False, {}
    
    def detect_semantic_contradiction(self, turns: list[dict]) -> tuple[bool, dict]:
        """Detect semantic contradictions (simplified: keyword-based)."""
        agent_contents = [
            t.get('content', '') 
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if len(agent_contents) < 2:
            return False, {}
        
        contradiction_markers = [
            'actually, that\'s wrong', 'i was mistaken', 'on second thought',
            'let me reconsider', 'that contradicts', 'wait, that\'s not right',
            'i need to correct', 'previous answer was incorrect',
        ]
        
        contradiction_count = 0
        for content in agent_contents:
            content_lower = content.lower()
            for marker in contradiction_markers:
                if marker in content_lower:
                    contradiction_count += 1
                    break
        
        contradiction_rate = contradiction_count / len(agent_contents)
        if contradiction_rate > self.config.contradiction_rate_threshold:
            return True, {'signal': 'semantic_contradiction', 'rate': contradiction_rate}
        
        return False, {}
    
    def detect_latency_anomaly(self, turns: list[dict]) -> tuple[bool, dict]:
        """Detect latency anomalies."""
        latencies = []
        for t in turns:
            lat = t.get('latency_ms') or t.get('latency')
            if lat is not None:
                latencies.append(lat)
        
        if len(latencies) < 3:
            return False, {}
        
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        if std_lat > 0:
            anomalies = [
                lat for lat in latencies 
                if abs(lat - mean_lat) > self.config.latency_std_threshold * std_lat
            ]
            if anomalies:
                return True, {'signal': 'latency_anomaly', 'anomaly_count': len(anomalies)}
        
        return False, {}
    
    def detect(self, trace: dict) -> dict:
        """Run all detectors on a trace."""
        turns = trace.get('turns', [])
        
        if len(turns) < self.config.min_turns:
            return {
                'trace_id': trace.get('trace_id'),
                'predicted_failure': False,
                'signals_triggered': [],
                'num_signals': 0,
            }
        
        signals_triggered = []
        
        detectors = [
            self.detect_confidence_collapse,
            self.detect_semantic_contradiction,
            self.detect_reasoning_loop,
            self.detect_context_pressure,
            self.detect_latency_anomaly,
        ]
        
        for detector in detectors:
            triggered, detail = detector(turns)
            if triggered:
                signals_triggered.append(detail.get('signal', 'unknown'))
        
        predicted_failure = len(signals_triggered) >= self.config.min_signals_for_failure
        
        return {
            'trace_id': trace.get('trace_id'),
            'predicted_failure': predicted_failure,
            'signals_triggered': signals_triggered,
            'num_signals': len(signals_triggered),
            'ground_truth_healthy': trace.get('healthy'),
        }


# =============================================================================
# Feature Extraction for Baselines
# =============================================================================

def extract_features(traces: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from traces for baseline anomaly detectors."""
    features = []
    labels = []
    
    for trace in traces:
        is_healthy = trace.get('healthy', False)
        turns = trace.get('turns', [])
        
        if not turns:
            continue
        
        # Feature 1: Mean confidence
        confidences = []
        for turn in turns:
            if 'confidence' in turn and turn['confidence'] is not None:
                confidences.append(turn['confidence'])
        mean_confidence = np.mean(confidences) if confidences else 0.5
        
        # Feature 2: Confidence variance
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        
        # Feature 3: Turn count
        turn_count = len(turns)
        
        # Feature 4: Token count (estimated)
        total_text_length = 0
        response_lengths = []
        for turn in turns:
            if 'content' in turn and turn['content'] is not None:
                content_len = len(str(turn['content']))
                response_lengths.append(content_len)
                total_text_length += content_len
        
        estimated_tokens = total_text_length / 4
        
        # Feature 5: Response length variance
        response_length_variance = np.var(response_lengths) if len(response_lengths) > 1 else 0.0
        
        feature_vector = [
            mean_confidence,
            confidence_variance, 
            turn_count,
            estimated_tokens,
            response_length_variance
        ]
        
        features.append(feature_vector)
        labels.append(1 if is_healthy else -1)  # 1 = healthy, -1 = unhealthy
    
    return np.array(features), np.array(labels)


# =============================================================================
# Baseline Runners
# =============================================================================

def run_baselines(features: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Run IF, LOF, OCSVM baselines and compute metrics."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    # Isolation Forest
    try:
        if_model = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
        if_pred = if_model.fit_predict(features_scaled)
        if_scores = if_model.score_samples(features_scaled)
        results['if'] = compute_baseline_metrics(labels, if_pred, if_scores)
    except Exception as e:
        results['if'] = {'tpr': 0.0, 'fpr': 1.0, 'f1': 0.0, 'auc': 0.5}
    
    # Local Outlier Factor
    try:
        lof_model = LocalOutlierFactor(contamination='auto', novelty=False)
        lof_pred = lof_model.fit_predict(features_scaled)
        lof_scores = lof_model.negative_outlier_factor_
        results['lof'] = compute_baseline_metrics(labels, lof_pred, lof_scores)
    except Exception as e:
        results['lof'] = {'tpr': 0.0, 'fpr': 1.0, 'f1': 0.0, 'auc': 0.5}
    
    # One-Class SVM
    try:
        svm_model = OneClassSVM(gamma='scale', nu=0.1)
        svm_pred = svm_model.fit_predict(features_scaled)
        svm_scores = svm_model.score_samples(features_scaled)
        results['ocsvm'] = compute_baseline_metrics(labels, svm_pred, svm_scores)
    except Exception as e:
        results['ocsvm'] = {'tpr': 0.0, 'fpr': 1.0, 'f1': 0.0, 'auc': 0.5}
    
    return results


def compute_baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Compute metrics for baseline detectors."""
    # Convert to binary: 1 = healthy, 0 = unhealthy
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)
    
    # Confusion matrix
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    fn = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    try:
        auc = roc_auc_score(1 - y_true_binary, -y_scores)
    except:
        auc = 0.5
    
    return {
        'tpr': round(float(tpr), 4),
        'fpr': round(float(fpr), 4),
        'f1': round(float(f1), 4),
        'auc': round(float(auc), 4)
    }


# =============================================================================
# Our Detector Metrics
# =============================================================================

def compute_detector_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """Compute metrics for our detector."""
    tp = fp = tn = fn = 0
    scores = []  # For AUC
    
    for pred in predictions:
        gt_healthy = pred.get('ground_truth_healthy')
        if gt_healthy is None:
            continue
        
        gt_failure = not gt_healthy
        pred_failure = pred.get('predicted_failure', False)
        num_signals = pred.get('num_signals', 0)
        
        # Score = number of signals (higher = more anomalous)
        scores.append((gt_healthy, num_signals))
        
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
    
    # Compute AUC
    if scores:
        y_true = np.array([0 if s[0] else 1 for s in scores])  # 1 = failure
        y_scores = np.array([s[1] for s in scores])  # num_signals
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = (tpr + (1 - fpr)) / 2
    else:
        auc = (tpr + (1 - fpr)) / 2
    
    return {
        'tpr': round(tpr, 4),
        'fpr': round(fpr, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    # Paths
    workspace = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper")
    test_path = workspace / "data/unified/splits/test.json"
    thresholds_path = workspace / "results/tuned-thresholds.json"
    output_path = workspace / "results/recalibrated-comparison.json"
    table_path = workspace / "results/COMPARISON-TABLE.md"
    
    print(f"Loading test data from {test_path}...")
    with open(test_path) as f:
        test_data = json.load(f)
    
    traces = test_data.get('traces', [])
    print(f"Loaded {len(traces)} test traces")
    
    # Load tuned thresholds
    print(f"Loading thresholds from {thresholds_path}...")
    config = load_thresholds_from_json(thresholds_path)
    print(f"Detector config: min_signals={config.min_signals_for_failure}")
    
    # Initialize detector
    detector = GreyFailureDetector(config)
    
    # Process per benchmark
    benchmarks = ['gaia', 'webarena', 'agentbench', 'swe-bench']
    results = []
    
    for benchmark in benchmarks:
        print(f"\n{'='*50}")
        print(f"Processing {benchmark.upper()}...")
        print(f"{'='*50}")
        
        # Filter traces for this benchmark
        bm_traces = [t for t in traces if t.get('benchmark') == benchmark]
        print(f"Found {len(bm_traces)} traces")
        
        if not bm_traces:
            continue
        
        # Run our detector
        predictions = [detector.detect(t) for t in bm_traces]
        our_metrics = compute_detector_metrics(predictions)
        
        print(f"Our Detector: TPR={our_metrics['tpr']:.3f}, FPR={our_metrics['fpr']:.3f}, "
              f"F1={our_metrics['f1']:.3f}, AUC={our_metrics['auc']:.3f}")
        
        # Run baselines
        features, labels = extract_features(bm_traces)
        print(f"Extracted {len(features)} feature vectors")
        
        baseline_results = run_baselines(features, labels)
        
        for name, metrics in baseline_results.items():
            print(f"{name.upper()}: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}, "
                  f"F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
        
        # Determine winner (by AUC vs IF baseline)
        if_auc = baseline_results['if']['auc']
        our_auc = our_metrics['auc']
        winner = "ours" if our_auc > if_auc else "baseline"
        auc_delta = our_auc - if_auc
        
        result = {
            "benchmark": benchmark,
            "n_traces": len(bm_traces),
            "n_healthy": sum(1 for t in bm_traces if t.get('healthy')),
            "n_failures": sum(1 for t in bm_traces if not t.get('healthy')),
            "ours_tuned": our_metrics,
            "if_baseline": baseline_results['if'],
            "lof_baseline": baseline_results['lof'],
            "ocsvm_baseline": baseline_results['ocsvm'],
            "winner": winner,
            "auc_delta": round(auc_delta, 4)
        }
        results.append(result)
        
        print(f"Winner: {winner} (AUC delta: {auc_delta:+.4f})")
    
    # Count wins
    our_wins = sum(1 for r in results if r['winner'] == 'ours')
    total_benchmarks = len(results)
    
    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Our detector wins: {our_wins}/{total_benchmarks} benchmarks")
    print(f"Success criteria: Beat IF on ≥3/4 benchmarks")
    print(f"Result: {'PASS' if our_wins >= 3 else 'FAIL'}")
    
    # Save JSON results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {output_path}")
    
    # Generate markdown table
    generate_markdown_table(results, table_path)
    print(f"Saved markdown table to {table_path}")


def generate_markdown_table(results: List[Dict], output_path: Path):
    """Generate a markdown comparison table."""
    
    lines = [
        "# Grey Failure Detection: Recalibrated Comparison",
        "",
        "Evaluation on **TEST split** (30% held-out data).",
        "",
        "## Per-Benchmark Results",
        "",
        "| Benchmark | N | Ours (AUC) | IF (AUC) | LOF (AUC) | OCSVM (AUC) | Winner | Δ AUC |",
        "|-----------|---|------------|----------|-----------|-------------|--------|-------|",
    ]
    
    for r in results:
        row = (
            f"| {r['benchmark']} | {r['n_traces']} | "
            f"{r['ours_tuned']['auc']:.3f} | "
            f"{r['if_baseline']['auc']:.3f} | "
            f"{r['lof_baseline']['auc']:.3f} | "
            f"{r['ocsvm_baseline']['auc']:.3f} | "
            f"**{r['winner']}** | "
            f"{r['auc_delta']:+.3f} |"
        )
        lines.append(row)
    
    # Summary
    our_wins = sum(1 for r in results if r['winner'] == 'ours')
    avg_delta = np.mean([r['auc_delta'] for r in results])
    
    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Our wins**: {our_wins}/{len(results)} benchmarks",
        f"- **Average AUC delta**: {avg_delta:+.4f}",
        f"- **Success criteria**: Beat IF on ≥3/4 benchmarks → {'✅ PASS' if our_wins >= 3 else '❌ FAIL'}",
        "",
        "## Detailed Metrics",
        "",
        "| Benchmark | Method | TPR | FPR | F1 | AUC |",
        "|-----------|--------|-----|-----|-----|-----|",
    ])
    
    for r in results:
        for method, key in [("Ours", "ours_tuned"), ("IF", "if_baseline"), 
                           ("LOF", "lof_baseline"), ("OCSVM", "ocsvm_baseline")]:
            m = r[key]
            lines.append(
                f"| {r['benchmark']} | {method} | "
                f"{m['tpr']:.3f} | {m['fpr']:.3f} | {m['f1']:.3f} | {m['auc']:.3f} |"
            )
    
    lines.extend([
        "",
        "---",
        f"*Generated by evaluate_tuned.py*",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
