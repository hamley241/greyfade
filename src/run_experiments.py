"""Main experiment runner for grey failure detection paper.

Runs four experiments:
- E1: Detection capability (TPR across patterns and severities)
- E2: False positive rate (FPR across noise levels)
- E3: Baseline comparison (compare with threshold, moving avg, etc.)
- E4: Ablation study (contribution of each component)

Usage:
    python run_experiments.py [--quick] [--experiment E1|E2|E3|E4] [--output-dir DIR]
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from grey_detector import (
    GreyFailureDetector,
    AgentTurn,
    GreyFailureType,
    HealthStatus,
    DetectorConfig,
)
from ablation_variants import AblationDetector, create_ablation_variants, reset_all_variants
from stats_utils import wilson_ci, compute_tpr_fpr, compute_auc_roc, chi_square_independence
from format_results import generate_all_outputs, plot_roc_curves

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed. Some features may be limited.")


# =============================================================================
# Synthetic Trace Generation
# =============================================================================

def make_turn(
    turn_id: int,
    confidence: float = 0.9,
    latency_ms: float = 150,
    tokens_used: int = 2000,
    tokens_available: int = 8000,
    output_quality: float = 0.85,
    reasoning_consistent: bool = True,
    tool_calls: int = 0,
    tool_failures: int = 0
) -> AgentTurn:
    """Create a turn with specified parameters."""
    return AgentTurn(
        turn_id=turn_id,
        timestamp=datetime.now() + timedelta(seconds=turn_id * 2),
        confidence=confidence,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        tokens_available=tokens_available,
        output_quality=output_quality,
        reasoning_consistent=reasoning_consistent,
        tool_calls=tool_calls,
        tool_failures=tool_failures
    )


def generate_healthy_trace(n_turns: int = 10, noise_level: float = 0.0, seed: Optional[int] = None) -> List[AgentTurn]:
    """Generate a healthy trace with optional noise.
    
    Args:
        n_turns: Number of turns
        noise_level: Standard deviation of noise (0.0 = no noise)
        seed: Random seed for reproducibility
    
    Returns:
        List of AgentTurn representing healthy execution
    """
    if seed is not None:
        random.seed(seed)
    
    turns = []
    for i in range(n_turns):
        # Healthy baseline with noise
        conf = max(0.5, min(1.0, 0.88 + random.gauss(0, noise_level)))
        latency = max(50, 150 + random.gauss(0, noise_level * 100))
        quality = max(0.5, min(1.0, 0.87 + random.gauss(0, noise_level)))
        tokens = int(2000 + random.gauss(0, noise_level * 500))
        
        turns.append(make_turn(
            i,
            confidence=conf,
            latency_ms=latency,
            tokens_used=tokens,
            output_quality=quality,
            reasoning_consistent=True
        ))
    
    return turns


def generate_failure_trace(
    pattern: str,
    severity: str,
    n_turns: int = 10,
    seed: Optional[int] = None
) -> List[AgentTurn]:
    """Generate a trace with a specific failure pattern.
    
    Args:
        pattern: Type of failure (confidence_degradation, latency_creep, etc.)
        severity: low, medium, or high
        n_turns: Number of turns
        seed: Random seed
    
    Returns:
        List of AgentTurn with failure pattern
    """
    if seed is not None:
        random.seed(seed)
    
    # Severity multipliers
    sev_mult = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
    mult = sev_mult.get(severity, 1.0)
    
    turns = []
    
    for i in range(n_turns):
        # Base healthy values
        conf = 0.9
        latency = 150
        quality = 0.85
        tokens = 2000
        consistent = True
        
        # Apply failure pattern
        progress = i / max(n_turns - 1, 1)  # 0 to 1
        
        if pattern == 'confidence_degradation':
            # Confidence drops over time
            conf = 0.9 - (progress * 0.4 * mult)
            
        elif pattern == 'latency_creep':
            # Latency increases over time
            latency = 100 + (progress * 300 * mult)
            
        elif pattern == 'quality_drift':
            # Quality decreases
            quality = 0.9 - (progress * 0.4 * mult)
            
        elif pattern == 'context_exhaustion':
            # Tokens fill up
            tokens = int(1000 + progress * 6500 * mult)
            # Also affects quality at high utilization
            if tokens > 6000:
                quality = 0.85 - ((tokens - 6000) / 2000) * 0.3
                
        elif pattern == 'reasoning_inconsistency':
            # Reasoning becomes inconsistent mid-way
            if progress > 0.3:
                consistent = random.random() > (0.7 - progress * 0.5 * mult)
                conf = 0.9 - (progress * 0.3 * mult)
                
        elif pattern == 'composite':
            # Multiple signals degrade together
            conf = 0.9 - (progress * 0.3 * mult)
            latency = 150 + (progress * 200 * mult)
            quality = 0.88 - (progress * 0.3 * mult)
        
        # Add small noise
        conf = max(0.3, min(1.0, conf + random.gauss(0, 0.02)))
        latency = max(50, latency + random.gauss(0, 10))
        quality = max(0.3, min(1.0, quality + random.gauss(0, 0.02)))
        
        turns.append(make_turn(
            i,
            confidence=conf,
            latency_ms=latency,
            tokens_used=min(tokens, 7900),
            output_quality=quality,
            reasoning_consistent=consistent
        ))
    
    return turns


def generate_test_dataset(
    n_per_category: int = 50,
    seed: int = 42
) -> Tuple[List[Tuple[List[AgentTurn], str, str]], List[List[AgentTurn]]]:
    """Generate full test dataset with failures and healthy traces.
    
    Args:
        n_per_category: Number of traces per pattern/severity combination
        seed: Random seed
    
    Returns:
        (failure_traces, healthy_traces) where failure_traces are (trace, pattern, severity) tuples
    """
    random.seed(seed)
    
    patterns = [
        'confidence_degradation',
        'latency_creep',
        'quality_drift',
        'context_exhaustion',
        'reasoning_inconsistency',
        'composite'
    ]
    severities = ['low', 'medium', 'high']
    
    failure_traces = []
    for pattern in patterns:
        for severity in severities:
            for i in range(n_per_category):
                trace = generate_failure_trace(pattern, severity, n_turns=10, seed=seed + hash(f"{pattern}{severity}{i}") % 10000)
                failure_traces.append((trace, pattern, severity))
    
    # Generate healthy traces with varying noise
    healthy_traces = []
    noise_levels = [0.0, 0.05, 0.1, 0.15]
    for noise in noise_levels:
        for i in range(n_per_category):
            trace = generate_healthy_trace(n_turns=10, noise_level=noise, seed=seed + hash(f"healthy{noise}{i}") % 10000)
            healthy_traces.append(trace)
    
    return failure_traces, healthy_traces


# =============================================================================
# Baseline Detectors
# =============================================================================

class ThresholdDetector:
    """Simple threshold-based detector (baseline)."""
    
    def __init__(self, conf_thresh: float = 0.7, latency_thresh: float = 300):
        self.conf_thresh = conf_thresh
        self.latency_thresh = latency_thresh
    
    def detect(self, turns: List[AgentTurn]) -> bool:
        """Return True if failure detected."""
        for turn in turns:
            if turn.confidence < self.conf_thresh or turn.latency_ms > self.latency_thresh:
                return True
        return False
    
    def get_score(self, turns: List[AgentTurn]) -> float:
        """Return anomaly score (higher = more anomalous)."""
        if not turns:
            return 0.0
        
        max_score = 0.0
        for turn in turns:
            conf_score = max(0, 1 - turn.confidence / self.conf_thresh)
            latency_score = max(0, turn.latency_ms / self.latency_thresh - 1)
            max_score = max(max_score, conf_score + latency_score)
        
        return min(max_score, 1.0)


class MovingAverageDetector:
    """Moving average based detector (baseline)."""
    
    def __init__(self, window: int = 3, threshold: float = 0.15):
        self.window = window
        self.threshold = threshold
    
    def detect(self, turns: List[AgentTurn]) -> bool:
        """Return True if significant drop from moving average."""
        if len(turns) < self.window + 1:
            return False
        
        for metric in ['confidence', 'output_quality']:
            values = [getattr(t, metric) or 0.85 for t in turns]
            
            for i in range(self.window, len(values)):
                window_avg = sum(values[i-self.window:i]) / self.window
                current = values[i]
                
                if window_avg - current > self.threshold:
                    return True
        
        return False
    
    def get_score(self, turns: List[AgentTurn]) -> float:
        """Return anomaly score based on deviation from moving average."""
        if len(turns) < self.window + 1:
            return 0.0
        
        max_deviation = 0.0
        for metric in ['confidence', 'output_quality']:
            values = [getattr(t, metric) or 0.85 for t in turns]
            
            for i in range(self.window, len(values)):
                window_avg = sum(values[i-self.window:i]) / self.window
                deviation = window_avg - values[i]
                max_deviation = max(max_deviation, deviation)
        
        return min(max_deviation / self.threshold, 1.0)


class GreyDetectorWrapper:
    """Wrapper to match baseline interface."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
    
    def detect(self, turns: List[AgentTurn]) -> bool:
        """Run detector and return if any alerts."""
        detector = GreyFailureDetector(self.config)
        for turn in turns:
            detector.record_turn(turn)
        return len(detector.alerts) > 0
    
    def get_score(self, turns: List[AgentTurn]) -> float:
        """Return 1 - health_score as anomaly score."""
        detector = GreyFailureDetector(self.config)
        for turn in turns:
            detector.record_turn(turn)
        return 1.0 - detector.get_health_score()


class AblationWrapper:
    """Wrapper for ablation detectors to match baseline interface."""
    
    def __init__(self, ablation_detector: AblationDetector):
        self.ablation = ablation_detector
    
    def detect(self, turns: List[AgentTurn]) -> bool:
        """Run detector and return if any alerts."""
        # Create fresh detector
        detector = AblationDetector(self.ablation.config, list(self.ablation.disable))
        for turn in turns:
            detector.record_turn(turn)
        return len(detector.alerts) > 0
    
    def get_score(self, turns: List[AgentTurn]) -> float:
        """Return 1 - health_score as anomaly score."""
        detector = AblationDetector(self.ablation.config, list(self.ablation.disable))
        for turn in turns:
            detector.record_turn(turn)
        return 1.0 - detector.get_health_score()


# =============================================================================
# Experiment Functions
# =============================================================================

def run_e1_detection_capability(
    failure_traces: List[Tuple[List[AgentTurn], str, str]],
    detector: Optional[Any] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """E1: TPR across patterns and severities.
    
    Args:
        failure_traces: List of (trace, pattern, severity) tuples
        detector: Detector to use (default: GreyDetectorWrapper)
    
    Returns:
        {pattern: {severity: {tpr, ci_low, ci_high}}}
    """
    if detector is None:
        detector = GreyDetectorWrapper()
    
    # Group by pattern and severity
    results: Dict[str, Dict[str, List[bool]]] = {}
    
    for trace, pattern, severity in failure_traces:
        if pattern not in results:
            results[pattern] = {}
        if severity not in results[pattern]:
            results[pattern][severity] = []
        
        detected = detector.detect(trace)
        results[pattern][severity].append(detected)
    
    # Compute TPR with confidence intervals
    output = {}
    for pattern, severities in results.items():
        output[pattern] = {}
        for severity, detections in severities.items():
            successes = sum(detections)
            total = len(detections)
            tpr, ci_low, ci_high = wilson_ci(successes, total)
            output[pattern][severity] = {
                'tpr': tpr,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n': total
            }
    
    return output


def run_e2_false_positive(
    healthy_traces: List[List[AgentTurn]],
    detector: Optional[Any] = None,
    noise_levels: Optional[List[float]] = None
) -> Dict[str, Dict[str, float]]:
    """E2: FPR across noise levels.
    
    Args:
        healthy_traces: List of healthy traces
        detector: Detector to use (default: GreyDetectorWrapper)
        noise_levels: If provided, partition traces by noise (requires matching order)
    
    Returns:
        {noise_level: {fpr, ci_low, ci_high}}
    """
    if detector is None:
        detector = GreyDetectorWrapper()
    
    if noise_levels is None:
        noise_levels = ['all']
    
    # For simplicity, we'll compute overall FPR
    # In full implementation, we'd need to track which traces have which noise
    false_positives = []
    
    for trace in healthy_traces:
        detected = detector.detect(trace)
        false_positives.append(detected)
    
    # Compute overall FPR
    fp_count = sum(false_positives)
    total = len(false_positives)
    fpr, ci_low, ci_high = wilson_ci(fp_count, total)
    
    # Partition by noise level if we know the order
    # (Assuming traces were generated in order: n_per_category per noise level)
    n_per_level = len(healthy_traces) // 4 if len(healthy_traces) >= 4 else len(healthy_traces)
    
    output = {}
    noise_vals = [0.0, 0.05, 0.1, 0.15]
    
    for i, noise in enumerate(noise_vals):
        start = i * n_per_level
        end = start + n_per_level
        if end > len(false_positives):
            break
        
        level_fps = false_positives[start:end]
        fp_count = sum(level_fps)
        total = len(level_fps)
        fpr, ci_low, ci_high = wilson_ci(fp_count, total)
        
        output[str(noise)] = {
            'fpr': fpr,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': total
        }
    
    return output


def run_e3_baseline_comparison(
    failure_traces: List[Tuple[List[AgentTurn], str, str]],
    healthy_traces: List[List[AgentTurn]],
    detectors: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """E3: Compare all methods.
    
    Args:
        failure_traces: Traces with failures
        healthy_traces: Healthy traces
        detectors: Dict of {name: detector}
    
    Returns:
        {method: {tpr, fpr, f1, auc}}
    """
    if detectors is None:
        detectors = {
            'Grey': GreyDetectorWrapper(),
            'Threshold': ThresholdDetector(),
            'MovingAvg': MovingAverageDetector(),
        }
    
    # Build labels and run all detectors
    all_traces = []
    labels = []
    
    # Failures (label=1)
    for trace, pattern, severity in failure_traces:
        all_traces.append(trace)
        labels.append(1)
    
    # Healthy (label=0)
    for trace in healthy_traces:
        all_traces.append(trace)
        labels.append(0)
    
    if HAS_NUMPY:
        labels = np.array(labels)
    
    results = {}
    roc_curves = {}
    
    for name, detector in detectors.items():
        predictions = []
        scores = []
        
        for trace in all_traces:
            predictions.append(1 if detector.detect(trace) else 0)
            scores.append(detector.get_score(trace))
        
        if HAS_NUMPY:
            predictions = np.array(predictions)
            scores = np.array(scores)
            
            # Compute metrics
            metrics = compute_tpr_fpr(predictions, labels)
            auc, fpr_curve, tpr_curve = compute_auc_roc(scores, labels)
            
            # compute_auc_roc already returns lists, not numpy arrays
            roc_curves[name] = (fpr_curve, tpr_curve, auc)
        else:
            # Manual computation without numpy
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
            
            metrics = {'tpr': tpr, 'fpr': fpr, 'precision': precision, 'f1': f1}
            auc = 0.5  # Placeholder
        
        results[name] = {
            'tpr': metrics['tpr'],
            'fpr': metrics['fpr'],
            'f1': metrics['f1'],
            'auc': auc if HAS_NUMPY else 0.5,
            'precision': metrics['precision']
        }
    
    return results


def run_e4_ablation(
    failure_traces: List[Tuple[List[AgentTurn], str, str]],
    healthy_traces: List[List[AgentTurn]],
    variants: Optional[Dict[str, AblationDetector]] = None
) -> Dict[str, Dict[str, float]]:
    """E4: Ablation study.
    
    Args:
        failure_traces: Traces with failures
        healthy_traces: Healthy traces
        variants: Dict of {variant_name: AblationDetector}
    
    Returns:
        {variant: {tpr, fpr, f1}}
    """
    if variants is None:
        variants = create_ablation_variants()
    
    # Wrap ablation detectors
    detectors = {name: AblationWrapper(det) for name, det in variants.items()}
    
    # Build labels
    all_traces = []
    labels = []
    
    for trace, pattern, severity in failure_traces:
        all_traces.append(trace)
        labels.append(1)
    
    for trace in healthy_traces:
        all_traces.append(trace)
        labels.append(0)
    
    results = {}
    
    for name, detector in detectors.items():
        predictions = []
        
        for trace in all_traces:
            predictions.append(1 if detector.detect(trace) else 0)
        
        # Compute metrics
        if HAS_NUMPY:
            predictions = np.array(predictions)
            labels_arr = np.array(labels)
            metrics = compute_tpr_fpr(predictions, labels_arr)
        else:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
            
            metrics = {'tpr': tpr, 'fpr': fpr, 'f1': f1}
        
        results[name] = {
            'tpr': metrics['tpr'],
            'fpr': metrics['fpr'],
            'f1': metrics['f1']
        }
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run grey failure detection experiments')
    parser.add_argument('--quick', action='store_true', help='Run with reduced sample size')
    parser.add_argument('--experiment', choices=['E1', 'E2', 'E3', 'E4', 'all'], default='all',
                       help='Which experiment to run')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    n_per_category = 10 if args.quick else 50
    print(f"\n📊 Generating test dataset (n_per_category={n_per_category})...")
    failure_traces, healthy_traces = generate_test_dataset(n_per_category, seed=args.seed)
    print(f"  Failure traces: {len(failure_traces)}")
    print(f"  Healthy traces: {len(healthy_traces)}")
    
    results = {}
    
    # E1: Detection Capability
    if args.experiment in ['E1', 'all']:
        print("\n" + "=" * 60)
        print("E1: Detection Capability (TPR by Pattern/Severity)")
        print("=" * 60)
        e1_results = run_e1_detection_capability(failure_traces)
        results['e1'] = e1_results
        
        for pattern, severities in e1_results.items():
            print(f"\n  {pattern}:")
            for sev, metrics in severities.items():
                print(f"    {sev}: TPR={metrics['tpr']:.3f} [{metrics['ci_low']:.3f}, {metrics['ci_high']:.3f}]")
    
    # E2: False Positive Rate
    if args.experiment in ['E2', 'all']:
        print("\n" + "=" * 60)
        print("E2: False Positive Rate by Noise Level")
        print("=" * 60)
        e2_results = run_e2_false_positive(healthy_traces)
        results['e2'] = e2_results
        
        for noise, metrics in e2_results.items():
            print(f"  Noise {noise}: FPR={metrics['fpr']:.3f} [{metrics['ci_low']:.3f}, {metrics['ci_high']:.3f}]")
    
    # E3: Baseline Comparison
    if args.experiment in ['E3', 'all']:
        print("\n" + "=" * 60)
        print("E3: Baseline Comparison")
        print("=" * 60)
        e3_results = run_e3_baseline_comparison(failure_traces, healthy_traces)
        results['e3'] = e3_results
        
        print(f"\n  {'Method':<12} {'TPR':<8} {'FPR':<8} {'F1':<8} {'AUC':<8}")
        print(f"  {'-'*44}")
        for method, metrics in e3_results.items():
            print(f"  {method:<12} {metrics['tpr']:<8.3f} {metrics['fpr']:<8.3f} {metrics['f1']:<8.3f} {metrics['auc']:<8.3f}")
    
    # E4: Ablation Study
    if args.experiment in ['E4', 'all']:
        print("\n" + "=" * 60)
        print("E4: Ablation Study")
        print("=" * 60)
        e4_results = run_e4_ablation(failure_traces, healthy_traces)
        results['e4'] = e4_results
        
        full_f1 = e4_results.get('full', {}).get('f1', 0)
        print(f"\n  {'Variant':<20} {'TPR':<8} {'FPR':<8} {'F1':<8} {'ΔF1':<8}")
        print(f"  {'-'*52}")
        for variant in ['full'] + sorted([v for v in e4_results.keys() if v != 'full']):
            if variant not in e4_results:
                continue
            metrics = e4_results[variant]
            delta = metrics['f1'] - full_f1
            delta_str = f"{delta:+.3f}" if variant != 'full' else "-"
            print(f"  {variant:<20} {metrics['tpr']:<8.3f} {metrics['fpr']:<8.3f} {metrics['f1']:<8.3f} {delta_str:<8}")
    
    # Generate outputs
    print("\n" + "=" * 60)
    print("Generating output files...")
    print("=" * 60)
    
    outputs = generate_all_outputs(
        results.get('e1', {}),
        results.get('e2', {}),
        results.get('e3', {}),
        results.get('e4', {}),
        str(output_dir)
    )
    
    # Also save raw results
    raw_path = output_dir / "experiment_results.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Raw results: {raw_path}")
    
    print("\n✅ Experiments complete!")
    return results


if __name__ == "__main__":
    main()
