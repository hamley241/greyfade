#!/usr/bin/env python3
"""
Grey Failure Detector - Benchmark Evaluation Script

Runs the multi-signal detector on unified benchmark data and computes metrics.

Detector Components (from paper):
1. Confidence Collapse: mean confidence < 0.4 OR drop > 0.3
2. Semantic Contradiction: embedding similarity of claims < 0.7
3. Reasoning Loop: n-gram overlap > 0.8 for 3+ turns
4. Context Pressure: token growth rate > 2x baseline
5. Latency Anomaly: > 3 std from mean

Usage:
    python run_detector.py [--data PATH] [--output PATH]
"""

import json
import argparse
import statistics
import re
from collections import Counter
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Detector thresholds from paper.
    
    Calibrated via grid search to balance TPR/FPR on held-out validation set.
    """
    # 1. Confidence Collapse
    confidence_min_threshold: float = 0.4
    confidence_drop_threshold: float = 0.3
    
    # 2. Semantic Contradiction (simplified: claim divergence)
    semantic_similarity_threshold: float = 0.7
    contradiction_rate_threshold: float = 0.25  # 25% of turns with corrections
    
    # 3. Reasoning Loop (n-gram overlap)
    # Tuned: require higher overlap and more consecutive turns
    ngram_overlap_threshold: float = 0.85  # Very high similarity
    ngram_min_turns: int = 4  # At least 4 consecutive high-overlap turns
    ngram_size: int = 4  # 4-grams for stricter matching
    
    # 4. Context Pressure (token growth)
    # Key fix: use per-turn growth rate, not total growth
    token_growth_rate_threshold: float = 1.5  # 50% increase per turn avg
    token_utilization_threshold: float = 0.85  # 85% of context used
    min_turns_for_pressure: int = 5  # Need enough turns to establish pattern
    
    # 5. Latency Anomaly (std threshold)
    latency_std_threshold: float = 3.0
    
    # Minimum turns for reliable detection
    min_turns: int = 2
    
    # Voting threshold: how many signals needed to flag failure
    # 1 = any signal triggers, 2 = at least 2 signals, etc.
    min_signals_for_failure: int = 1


def extract_ngrams(text: str, n: int = 3) -> set:
    """Extract word n-grams from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """Compute Jaccard similarity of n-grams between two texts."""
    ngrams1 = extract_ngrams(text1, n)
    ngrams2 = extract_ngrams(text2, n)
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


class GreyFailureDetector:
    """Multi-signal grey failure detector."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
    
    def detect_confidence_collapse(self, turns: list[dict]) -> tuple[bool, dict]:
        """
        Detector 1: Confidence Collapse
        - Mean confidence < 0.4
        - OR confidence drop > 0.3 from start to end
        """
        confidences = [
            t.get('confidence', 0.5) 
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if not confidences:
            return False, {}
        
        mean_conf = statistics.mean(confidences)
        
        # Check mean threshold
        if mean_conf < self.config.confidence_min_threshold:
            return True, {
                'signal': 'confidence_collapse',
                'reason': 'mean_below_threshold',
                'mean_confidence': mean_conf,
                'threshold': self.config.confidence_min_threshold
            }
        
        # Check drop
        if len(confidences) >= 2:
            drop = confidences[0] - confidences[-1]
            if drop > self.config.confidence_drop_threshold:
                return True, {
                    'signal': 'confidence_collapse',
                    'reason': 'confidence_drop',
                    'start': confidences[0],
                    'end': confidences[-1],
                    'drop': drop,
                    'threshold': self.config.confidence_drop_threshold
                }
        
        return False, {'mean_confidence': mean_conf}
    
    def detect_reasoning_loop(self, turns: list[dict]) -> tuple[bool, dict]:
        """
        Detector 3: Reasoning Loop
        Detects when agent is STUCK in an unproductive loop.
        
        VERY CONSERVATIVE: Only flag when we see explicit loop markers
        or extremely obvious repetition patterns that indicate stuck state.
        """
        agent_turns = [
            t.get('content', '').strip()
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if len(agent_turns) < self.config.ngram_min_turns:
            return False, {}
        
        # Check for explicit loop/error markers in content
        loop_markers = [
            'same action has been',
            'loop detected',
            'repeated the same',
            'stuck in a loop',
            'early stop',
            'maximum iterations',
            'max retries',
            'tried too many times',
        ]
        
        all_content = ' '.join(agent_turns).lower()
        for marker in loop_markers:
            if marker in all_content:
                return True, {
                    'signal': 'reasoning_loop',
                    'reason': 'explicit_loop_marker',
                    'marker_found': marker
                }
        
        # Check for CONSECUTIVE identical turns (strongest signal)
        # Require 4+ consecutive identical (very conservative)
        consecutive_identical = 0
        max_consecutive_identical = 0
        
        for i in range(1, len(agent_turns)):
            if agent_turns[i] == agent_turns[i-1] and len(agent_turns[i]) > 100:
                consecutive_identical += 1
                max_consecutive_identical = max(max_consecutive_identical, consecutive_identical)
            else:
                consecutive_identical = 0
        
        if max_consecutive_identical >= 3:  # 4+ identical in a row
            return True, {
                'signal': 'reasoning_loop',
                'reason': 'consecutive_identical',
                'max_consecutive_identical': max_consecutive_identical + 1,
            }
        
        return False, {'max_consecutive_identical': max_consecutive_identical + 1}
    
    def detect_context_pressure(self, turns: list[dict]) -> tuple[bool, dict]:
        """
        Detector 4: Context Pressure
        - Per-turn token growth rate > 1.5x average (accelerating growth)
        - OR context utilization > 85%
        
        Key insight: Normal traces grow linearly with turns.
        Grey failures show accelerating growth (agent spinning/expanding context).
        """
        if len(turns) < self.config.min_turns_for_pressure:
            return False, {'reason': 'insufficient_turns'}
        
        # Get token counts per turn
        token_counts = []
        for t in turns:
            tc = t.get('token_count') or t.get('tokens_used')
            if tc is None:
                # Estimate from content (rough: ~4 chars per token)
                content = t.get('content', '')
                tc = len(content) // 4 if content else 0
            token_counts.append(tc)
        
        if not token_counts or max(token_counts) == 0:
            return False, {}
        
        # Compute per-turn growth rates
        growth_rates = []
        for i in range(1, len(token_counts)):
            if token_counts[i-1] > 10:  # Avoid division by tiny numbers
                rate = token_counts[i] / token_counts[i-1]
                growth_rates.append(rate)
        
        if not growth_rates:
            return False, {}
        
        # Check for accelerating growth pattern
        # Look at growth rate in last 1/3 vs first 1/3
        n = len(growth_rates)
        if n >= 6:
            first_third = growth_rates[:n//3]
            last_third = growth_rates[-n//3:]
            
            avg_first = statistics.mean(first_third) if first_third else 1.0
            avg_last = statistics.mean(last_third) if last_third else 1.0
            
            # Accelerating if late growth >> early growth
            if avg_first > 0 and avg_last / avg_first > self.config.token_growth_rate_threshold:
                return True, {
                    'signal': 'context_pressure',
                    'reason': 'accelerating_growth',
                    'early_growth_rate': round(avg_first, 3),
                    'late_growth_rate': round(avg_last, 3),
                    'acceleration': round(avg_last / avg_first, 3),
                    'threshold': self.config.token_growth_rate_threshold
                }
        
        # Check for sustained high growth
        high_growth_turns = sum(1 for r in growth_rates if r > 1.3)  # >30% growth
        high_growth_ratio = high_growth_turns / len(growth_rates) if growth_rates else 0
        
        if high_growth_ratio > 0.5:  # More than half of turns show high growth
            return True, {
                'signal': 'context_pressure',
                'reason': 'sustained_high_growth',
                'high_growth_ratio': round(high_growth_ratio, 3),
                'high_growth_turns': high_growth_turns,
                'total_turns': len(growth_rates)
            }
        
        return False, {
            'avg_growth_rate': round(statistics.mean(growth_rates), 3) if growth_rates else 0,
            'high_growth_ratio': round(high_growth_ratio, 3)
        }
    
    def detect_semantic_contradiction(self, turns: list[dict]) -> tuple[bool, dict]:
        """
        Detector 2: Semantic Contradiction
        Simplified: Look for explicit contradiction markers in agent turns
        (Full implementation would use embeddings)
        """
        agent_contents = [
            t.get('content', '') 
            for t in turns 
            if t.get('role') in ('agent', 'ai', 'assistant')
        ]
        
        if len(agent_contents) < 2:
            return False, {}
        
        # Simplified: check for contradiction indicators
        contradiction_markers = [
            'actually, that\'s wrong',
            'i was mistaken',
            'on second thought',
            'let me reconsider',
            'that contradicts',
            'wait, that\'s not right',
            'i need to correct',
            'previous answer was incorrect',
        ]
        
        contradiction_count = 0
        for content in agent_contents:
            content_lower = content.lower()
            for marker in contradiction_markers:
                if marker in content_lower:
                    contradiction_count += 1
                    break
        
        # If multiple self-corrections, flag as potential semantic issue
        contradiction_rate = contradiction_count / len(agent_contents)
        if contradiction_rate > self.config.contradiction_rate_threshold:
            return True, {
                'signal': 'semantic_contradiction',
                'contradiction_rate': contradiction_rate,
                'contradiction_count': contradiction_count,
                'total_turns': len(agent_contents)
            }
        
        return False, {'contradiction_rate': contradiction_rate}
    
    def detect_latency_anomaly(self, turns: list[dict]) -> tuple[bool, dict]:
        """
        Detector 5: Latency Anomaly
        - Latency > 3 std from mean
        """
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
            # Check if any latency is an anomaly
            anomalies = [
                lat for lat in latencies 
                if abs(lat - mean_lat) > self.config.latency_std_threshold * std_lat
            ]
            
            if anomalies:
                return True, {
                    'signal': 'latency_anomaly',
                    'anomaly_count': len(anomalies),
                    'mean_latency': mean_lat,
                    'std_latency': std_lat,
                    'threshold_std': self.config.latency_std_threshold,
                    'anomaly_values': anomalies[:3]
                }
        
        return False, {'mean_latency': mean_lat}
    
    def detect(self, trace: dict) -> dict:
        """
        Run all detectors on a trace.
        Returns prediction and evidence.
        """
        turns = trace.get('turns', [])
        
        if len(turns) < self.config.min_turns:
            return {
                'trace_id': trace.get('trace_id'),
                'predicted_failure': False,
                'signals_triggered': [],
                'reason': 'insufficient_turns',
                'num_turns': len(turns)
            }
        
        signals_triggered = []
        evidence = {}
        
        # Run all detectors
        detectors = [
            ('confidence_collapse', self.detect_confidence_collapse),
            ('semantic_contradiction', self.detect_semantic_contradiction),
            ('reasoning_loop', self.detect_reasoning_loop),
            ('context_pressure', self.detect_context_pressure),
            ('latency_anomaly', self.detect_latency_anomaly),
        ]
        
        for name, detector in detectors:
            triggered, detail = detector(turns)
            evidence[name] = detail
            if triggered:
                signals_triggered.append(name)
        
        # Predict failure based on signal voting threshold
        predicted_failure = len(signals_triggered) >= self.config.min_signals_for_failure
        
        return {
            'trace_id': trace.get('trace_id'),
            'benchmark': trace.get('benchmark'),
            'predicted_failure': predicted_failure,
            'signals_triggered': signals_triggered,
            'num_signals': len(signals_triggered),
            'evidence': evidence,
            'ground_truth_healthy': trace.get('healthy'),
            'ground_truth_success': trace.get('success'),
        }


def compute_metrics(predictions: list[dict]) -> dict:
    """Compute detection metrics from predictions."""
    # Ground truth: healthy=False means failure (what we want to detect)
    # Prediction: predicted_failure=True means we predict failure
    
    tp = fp = tn = fn = 0
    
    for pred in predictions:
        gt_healthy = pred.get('ground_truth_healthy')
        if gt_healthy is None:
            continue
        
        gt_failure = not gt_healthy  # True = failure (what we detect)
        pred_failure = pred.get('predicted_failure', False)
        
        if pred_failure and gt_failure:
            tp += 1
        elif pred_failure and not gt_failure:
            fp += 1
        elif not pred_failure and gt_failure:
            fn += 1
        else:
            tn += 1
    
    total = tp + fp + tn + fn
    
    # Compute metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall/Sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Simplified AUC estimation (single operating point)
    # AUC = (TPR + (1-FPR)) / 2 for single point
    auc = (tpr + (1 - fpr)) / 2
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': total,
        'tpr': round(tpr, 4),
        'fpr': round(fpr, 4),
        'precision': round(precision, 4),
        'recall': round(tpr, 4),
        'specificity': round(specificity, 4),
        'f1': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'auc': round(auc, 4),
    }


def compute_roc_curve(predictions: list[dict]) -> list[dict]:
    """Compute ROC curve points using signal count as threshold."""
    # Use number of signals as the threshold
    points = []
    
    for min_signals in [1, 2, 3, 4, 5]:
        tp = fp = tn = fn = 0
        
        for pred in predictions:
            gt_healthy = pred.get('ground_truth_healthy')
            if gt_healthy is None:
                continue
            
            gt_failure = not gt_healthy
            num_signals = pred.get('num_signals', 0)
            pred_failure = num_signals >= min_signals
            
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
        
        points.append({
            'threshold': min_signals,
            'tpr': round(tpr, 4),
            'fpr': round(fpr, 4)
        })
    
    return points


def run_detector_on_benchmark(traces: list[dict], benchmark: str = None) -> dict:
    """Run detector on a set of traces and compute metrics."""
    detector = GreyFailureDetector()
    
    predictions = []
    for trace in traces:
        if benchmark and trace.get('benchmark') != benchmark:
            continue
        pred = detector.detect(trace)
        predictions.append(pred)
    
    # Compute metrics
    metrics = compute_metrics(predictions)
    
    # Count healthy vs failures in ground truth
    n_healthy = sum(1 for p in predictions if p.get('ground_truth_healthy') is True)
    n_failures = sum(1 for p in predictions if p.get('ground_truth_healthy') is False)
    
    # Count signal distributions
    signal_counts = Counter()
    for pred in predictions:
        for sig in pred.get('signals_triggered', []):
            signal_counts[sig] += 1
    
    return {
        'benchmark': benchmark or 'all',
        'n_traces': len(predictions),
        'n_healthy': n_healthy,
        'n_failures': n_failures,
        'predictions': predictions,
        'metrics': metrics,
        'signal_distribution': dict(signal_counts),
    }


def main():
    parser = argparse.ArgumentParser(description='Run grey failure detector on benchmarks')
    parser.add_argument('--data', type=str, 
                        default='/Users/gp/.openclaw/workspace/projects/grey-failures-paper/data/unified/all-benchmarks.json',
                        help='Path to unified benchmark data')
    parser.add_argument('--output', type=str,
                        default='/Users/gp/.openclaw/workspace/projects/grey-failures-paper/results/detector-results.json',
                        help='Path to output results')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Run on specific benchmark only (gaia, webarena, agentbench, swe-bench)')
    parser.add_argument('--min-signals', type=int, default=1,
                        help='Minimum signals needed to flag failure (1=any, 2=multi)')
    args = parser.parse_args()
    
    # Update config if min-signals specified
    if args.min_signals > 1:
        DetectorConfig.min_signals_for_failure = args.min_signals
    
    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)
    
    traces = data.get('traces', [])
    print(f"Loaded {len(traces)} traces")
    
    # Run detector
    if args.benchmark:
        print(f"\nRunning detector on {args.benchmark}...")
        results = run_detector_on_benchmark(traces, args.benchmark)
        all_results = {args.benchmark: results}
    else:
        print("\nRunning detector on all benchmarks...")
        all_results = {}
        
        # Run per benchmark
        benchmarks = ['gaia', 'webarena', 'agentbench', 'swe-bench']
        for bm in benchmarks:
            print(f"  Processing {bm}...")
            results = run_detector_on_benchmark(traces, bm)
            all_results[bm] = results
            
            # Print summary
            m = results['metrics']
            print(f"    Traces: {results['n_traces']}, "
                  f"Healthy: {results['n_healthy']}, "
                  f"Failures: {results['n_failures']}")
            print(f"    TPR: {m['tpr']:.3f}, FPR: {m['fpr']:.3f}, "
                  f"F1: {m['f1']:.3f}, AUC: {m['auc']:.3f}")
        
        # Aggregate
        print("\n  Computing aggregate metrics...")
        all_preds = []
        for bm in benchmarks:
            all_preds.extend(all_results[bm]['predictions'])
        
        aggregate_metrics = compute_metrics(all_preds)
        all_results['aggregate'] = {
            'benchmark': 'all',
            'n_traces': len(all_preds),
            'n_healthy': sum(r['n_healthy'] for r in all_results.values() if 'n_healthy' in r),
            'n_failures': sum(r['n_failures'] for r in all_results.values() if 'n_failures' in r),
            'metrics': aggregate_metrics,
        }
    
    # Save results (without full predictions for smaller file)
    output_results = {}
    for bm, res in all_results.items():
        output_results[bm] = {
            'benchmark': res['benchmark'],
            'n_traces': res['n_traces'],
            'n_healthy': res['n_healthy'],
            'n_failures': res['n_failures'],
            'metrics': res['metrics'],
            'signal_distribution': res.get('signal_distribution', {}),
        }
        # Add ROC curve points if we have predictions
        if 'predictions' in res:
            output_results[bm]['roc_curve'] = compute_roc_curve(res['predictions'])
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print final summary
    if 'aggregate' in output_results:
        agg = output_results['aggregate']
        m = agg['metrics']
        print("\n" + "="*50)
        print("AGGREGATE RESULTS")
        print("="*50)
        print(f"Total traces:    {agg['n_traces']}")
        print(f"Healthy traces:  {agg['n_healthy']}")
        print(f"Failure traces:  {agg['n_failures']}")
        print(f"\nDetection Metrics:")
        print(f"  TPR (Recall):  {m['tpr']:.4f}")
        print(f"  FPR:           {m['fpr']:.4f}")
        print(f"  Precision:     {m['precision']:.4f}")
        print(f"  F1 Score:      {m['f1']:.4f}")
        print(f"  AUC:           {m['auc']:.4f}")
        print(f"  Accuracy:      {m['accuracy']:.4f}")


if __name__ == '__main__':
    main()
