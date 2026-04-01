#!/usr/bin/env python3
"""
Evaluate Detectors at Realistic Operating Points (Low FPR)

Grey failures are RARE events in production (1-5% of runs).
High FPR = alert fatigue. We need to compare detectors at realistic operating points.

Metrics computed:
1. TPR @ FPR=0.01 (very strict, production-critical)
2. TPR @ FPR=0.05 (realistic production)
3. TPR @ FPR=0.10 (lenient)
4. Precision @ Recall=0.50 (catch half the failures, how precise?)

Usage:
    python evaluate_low_fpr.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Paths
WORKSPACE = Path("/Users/gp/.openclaw/workspace")
DATA_PATH = WORKSPACE / "projects/grey-failures-paper/data/unified/splits/test.json"
THRESHOLDS_PATH = WORKSPACE / "projects/grey-failures-paper/results/tuned-thresholds.json"
OUTPUT_JSON = WORKSPACE / "projects/grey-failures-paper/results/low-fpr-comparison.json"
OUTPUT_MD = WORKSPACE / "projects/grey-failures-paper/results/LOW-FPR-TABLE.md"


def extract_features_and_filter(traces: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extract features from traces for anomaly detection.
    Also returns filtered traces list to ensure consistency.
    
    Returns:
        features: Feature matrix (n_samples, n_features)
        labels: Binary labels (1 = failure/unhealthy, 0 = healthy)
        filtered_traces: List of traces that were successfully processed
    """
    features = []
    labels = []
    filtered_traces = []
    
    for trace in traces:
        try:
            is_healthy = trace.get('healthy', False)
            turns = trace.get('turns', [])
            
            if not turns:
                continue
            
            # Feature 1: Mean confidence per trace
            confidences = []
            for turn in turns:
                if 'confidence' in turn and turn['confidence'] is not None:
                    confidences.append(turn['confidence'])
            mean_confidence = np.mean(confidences) if confidences else 0.5
            
            # Feature 2: Confidence variance
            confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
            
            # Feature 3: Turn count
            turn_count = len(turns)
            
            # Feature 4: Token count (estimated from text length)
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
            
            # Compile features
            feature_vector = [
                mean_confidence,
                confidence_variance, 
                turn_count,
                estimated_tokens,
                response_length_variance
            ]
            
            features.append(feature_vector)
            # 1 = failure (what we want to detect), 0 = healthy
            labels.append(0 if is_healthy else 1)
            filtered_traces.append(trace)
            
        except Exception as e:
            continue
    
    return np.array(features), np.array(labels), filtered_traces


def get_our_detector_scores(traces: List[Dict], thresholds: Dict) -> np.ndarray:
    """
    Get anomaly scores from our multi-signal detector.
    Higher score = more likely to be failure.
    
    Uses tuned thresholds and computes a combined score based on signals.
    """
    best_params = thresholds.get('best_thresholds', {})
    conf_thresh = best_params.get('confidence_threshold', 0.45)
    conf_drop = best_params.get('confidence_drop', 0.2)
    min_signals = best_params.get('min_signals', 1)
    min_turns = best_params.get('min_turns_for_pressure', 3)
    growth_rate = best_params.get('growth_rate', 2.0)
    
    scores = []
    
    for trace in traces:
        turns = trace.get('turns', [])
        if not turns:
            scores.append(0.0)
            continue
        
        # Compute individual signal scores
        signal_scores = []
        
        # 1. Confidence collapse
        confidences = [t.get('confidence', 0.5) for t in turns 
                      if t.get('role') in ('agent', 'ai', 'assistant') and t.get('confidence') is not None]
        if confidences:
            mean_conf = np.mean(confidences)
            # Score: how much below threshold
            conf_score = max(0, conf_thresh - mean_conf) / conf_thresh
            signal_scores.append(conf_score)
            
            # Confidence drop
            if len(confidences) >= 2:
                drop = confidences[0] - confidences[-1]
                drop_score = max(0, drop - conf_drop * 0.5) / 0.5  # Scale
                signal_scores.append(drop_score)
        
        # 2. Context pressure (token growth)
        if len(turns) >= min_turns:
            token_counts = []
            for t in turns:
                content = t.get('content', '')
                tc = len(content) // 4 if content else 0
                token_counts.append(tc)
            
            if len(token_counts) >= 2 and token_counts[0] > 10:
                growth = token_counts[-1] / token_counts[0] if token_counts[0] > 0 else 1.0
                growth_score = max(0, (growth - 1) / (growth_rate - 1)) if growth_rate > 1 else 0.0
                signal_scores.append(min(1.0, growth_score))
        
        # 3. Reasoning loop detection
        agent_turns = [t.get('content', '').strip() for t in turns 
                      if t.get('role') in ('agent', 'ai', 'assistant')]
        
        if len(agent_turns) >= 3:
            # Check for repetition
            repetition_count = 0
            for i in range(1, len(agent_turns)):
                if agent_turns[i] == agent_turns[i-1] and len(agent_turns[i]) > 50:
                    repetition_count += 1
            loop_score = min(1.0, repetition_count / 2)
            signal_scores.append(loop_score)
            
            # Check for loop markers
            all_content = ' '.join(agent_turns).lower()
            loop_markers = ['loop detected', 'same action', 'stuck', 'max retries', 'repeated']
            marker_found = any(m in all_content for m in loop_markers)
            if marker_found:
                signal_scores.append(0.8)
        
        # Combined score: mean of signal scores
        final_score = np.mean(signal_scores) if signal_scores else 0.0
        scores.append(final_score)
    
    return np.array(scores)


def get_baseline_scores(features: np.ndarray, detector_name: str) -> np.ndarray:
    """
    Get anomaly scores from baseline detectors.
    Higher score = more likely to be anomaly/failure.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if detector_name == 'if':
        model = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
        model.fit(features_scaled)
        # score_samples returns higher for normal, so negate
        return -model.score_samples(features_scaled)
    
    elif detector_name == 'lof':
        model = LocalOutlierFactor(contamination='auto', novelty=False, n_neighbors=20)
        model.fit_predict(features_scaled)
        # negative_outlier_factor_ is more negative for outliers, so negate
        return -model.negative_outlier_factor_
    
    elif detector_name == 'ocsvm':
        model = OneClassSVM(gamma='scale', nu=0.1)
        model.fit(features_scaled)
        # score_samples returns higher for normal, so negate
        return -model.score_samples(features_scaled)
    
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """
    Compute TPR at a specific FPR operating point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    # Find closest FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    
    # Get the highest TPR at or below target FPR
    best_idx = valid_idx[np.argmax(tpr[valid_idx])]
    return float(tpr[best_idx])


def precision_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """
    Compute Precision at a specific Recall operating point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Find closest recall >= target_recall
    valid_idx = np.where(recall >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.0
    
    # Get highest precision at or above target recall
    best_idx = valid_idx[np.argmax(precision[valid_idx])]
    return float(precision[best_idx])


def evaluate_benchmark(traces: List[Dict], thresholds: Dict, benchmark_name: str) -> Dict[str, Any]:
    """
    Evaluate all detectors on a single benchmark at low FPR operating points.
    """
    print(f"\n  Evaluating {benchmark_name}...")
    
    # Extract features and labels - this also returns filtered traces for consistency
    features, labels, filtered_traces = extract_features_and_filter(traces)
    
    if len(features) == 0:
        print(f"    No valid traces for {benchmark_name}")
        return {}
    
    n_failures = np.sum(labels == 1)
    n_healthy = np.sum(labels == 0)
    print(f"    Traces: {len(features)}, Failures: {n_failures}, Healthy: {n_healthy}")
    
    if n_failures == 0 or n_healthy == 0:
        print(f"    Skipping - need both classes")
        return {}
    
    results = {
        'benchmark': benchmark_name,
        'n_traces': len(features),
        'n_failures': int(n_failures),
        'n_healthy': int(n_healthy),
        'detectors': {}
    }
    
    # Target FPR levels
    fpr_levels = [0.01, 0.05, 0.10]
    
    # 1. Our detector - use filtered_traces to ensure same length
    print("    Running our detector...")
    our_scores = get_our_detector_scores(filtered_traces, thresholds)
    
    our_metrics = {
        f'tpr_at_fpr_{int(fpr*100):02d}': tpr_at_fpr(labels, our_scores, fpr)
        for fpr in fpr_levels
    }
    our_metrics['precision_at_recall_50'] = precision_at_recall(labels, our_scores, 0.50)
    results['detectors']['ours'] = our_metrics
    
    # 2. Baselines
    for baseline in ['if', 'lof', 'ocsvm']:
        print(f"    Running {baseline.upper()}...")
        try:
            scores = get_baseline_scores(features, baseline)
            metrics = {
                f'tpr_at_fpr_{int(fpr*100):02d}': tpr_at_fpr(labels, scores, fpr)
                for fpr in fpr_levels
            }
            metrics['precision_at_recall_50'] = precision_at_recall(labels, scores, 0.50)
            results['detectors'][baseline] = metrics
        except Exception as e:
            print(f"    {baseline} failed: {e}")
            results['detectors'][baseline] = {
                f'tpr_at_fpr_{int(fpr*100):02d}': 0.0
                for fpr in fpr_levels
            }
            results['detectors'][baseline]['precision_at_recall_50'] = 0.0
    
    return results


def generate_markdown_table(results: Dict[str, Any]) -> str:
    """
    Generate a markdown table from evaluation results.
    """
    lines = [
        "# Low FPR Detector Comparison",
        "",
        "Grey failures are **rare events** in production (1-5% of runs).",
        "High FPR = alert fatigue. This table compares detectors at **realistic operating points**.",
        "",
        "## TPR at Fixed FPR Operating Points",
        "",
        "| Benchmark | Detector | TPR@FPR=0.01 | TPR@FPR=0.05 | TPR@FPR=0.10 | Prec@Rec=0.50 |",
        "|-----------|----------|--------------|--------------|--------------|---------------|",
    ]
    
    benchmarks = ['gaia', 'webarena', 'agentbench', 'swe-bench', 'aggregate']
    detector_order = ['ours', 'if', 'lof', 'ocsvm']
    detector_names = {
        'ours': '**Ours**',
        'if': 'Isolation Forest',
        'lof': 'LOF',
        'ocsvm': 'One-Class SVM'
    }
    
    for bm in benchmarks:
        if bm not in results:
            continue
        
        bm_results = results[bm]
        bm_display = bm.upper() if bm != 'aggregate' else '**AGGREGATE**'
        
        for i, det in enumerate(detector_order):
            if det not in bm_results.get('detectors', {}):
                continue
            
            metrics = bm_results['detectors'][det]
            
            row_bm = bm_display if i == 0 else ""
            det_name = detector_names.get(det, det)
            
            tpr_01 = metrics.get('tpr_at_fpr_01', 0.0)
            tpr_05 = metrics.get('tpr_at_fpr_05', 0.0)
            tpr_10 = metrics.get('tpr_at_fpr_10', 0.0)
            prec_50 = metrics.get('precision_at_recall_50', 0.0)
            
            lines.append(
                f"| {row_bm} | {det_name} | {tpr_01:.3f} | {tpr_05:.3f} | {tpr_10:.3f} | {prec_50:.3f} |"
            )
        
        # Add separator after each benchmark
        if bm != 'aggregate':
            lines.append("|  |  |  |  |  |  |")
    
    lines.extend([
        "",
        "## Key Insights",
        "",
        "- **TPR @ FPR=0.01**: Production-critical. At 1% false alarm rate, how many failures do we catch?",
        "- **TPR @ FPR=0.05**: Realistic production setting. 5% false alarms may be tolerable.",
        "- **TPR @ FPR=0.10**: Lenient setting for less critical monitoring.",
        "- **Precision @ Recall=0.50**: If we catch half the failures, how precise are we?",
        "",
        "## Interpretation",
        "",
        "Higher TPR at low FPR = better detector for production use.",
        "Our multi-signal detector should excel at distinguishing grey failures from normal variation.",
    ])
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("LOW FPR DETECTOR EVALUATION")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading test data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    traces = data.get('traces', [])
    print(f"Loaded {len(traces)} test traces")
    
    # Load tuned thresholds
    print(f"Loading tuned thresholds from {THRESHOLDS_PATH}...")
    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    
    # Group traces by benchmark
    benchmarks = {}
    for trace in traces:
        bm = trace.get('benchmark', 'unknown')
        if bm not in benchmarks:
            benchmarks[bm] = []
        benchmarks[bm].append(trace)
    
    print(f"Benchmarks found: {list(benchmarks.keys())}")
    
    # Evaluate each benchmark
    results = {}
    for bm_name, bm_traces in benchmarks.items():
        if bm_name == 'unknown':
            continue
        bm_results = evaluate_benchmark(bm_traces, thresholds, bm_name)
        if bm_results:
            results[bm_name] = bm_results
    
    # Compute aggregate
    print("\n  Computing aggregate metrics...")
    all_traces = traces
    aggregate_results = evaluate_benchmark(all_traces, thresholds, 'aggregate')
    if aggregate_results:
        results['aggregate'] = aggregate_results
    
    # Save JSON results
    print(f"\nSaving JSON results to {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save markdown table
    print(f"Saving markdown table to {OUTPUT_MD}...")
    md_content = generate_markdown_table(results)
    with open(OUTPUT_MD, 'w') as f:
        f.write(md_content)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if 'aggregate' in results:
        agg = results['aggregate']
        print(f"\nAggregate Results ({agg['n_traces']} traces):")
        print(f"  Failures: {agg['n_failures']}, Healthy: {agg['n_healthy']}")
        print("\n  TPR @ FPR Operating Points:")
        
        for det in ['ours', 'if', 'lof', 'ocsvm']:
            if det in agg['detectors']:
                m = agg['detectors'][det]
                det_name = 'Ours' if det == 'ours' else det.upper()
                print(f"    {det_name:10s}: FPR=0.01→{m.get('tpr_at_fpr_01', 0):.3f}, "
                      f"FPR=0.05→{m.get('tpr_at_fpr_05', 0):.3f}, "
                      f"FPR=0.10→{m.get('tpr_at_fpr_10', 0):.3f}")
    
    print(f"\nResults saved to:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {OUTPUT_MD}")


if __name__ == '__main__':
    main()
