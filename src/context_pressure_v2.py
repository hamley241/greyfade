"""Context Pressure Detection v2 - Rate-Based Detection.

Replaces the broken threshold-based context detection with actual rate-based
Context Pressure detection as described in Section 3.3 of the paper.

Key insight: Context Pressure manifests as observable behavioral changes
BEFORE hitting hard limits. We detect:
1. Token growth rate acceleration
2. Semantic density degradation (repetition, compression)
3. Predictive trajectory modeling (turns until exhaustion)

This transforms Context Pressure from a reactive threshold alarm into a
predictive grey failure pattern.
"""

import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter
import re

# Import from prototype
import sys
sys.path.insert(0, '/Users/gp/.openclaw/workspace/projects/grey-failures-paper/prototype')

from grey_detector import (
    AgentTurn,
    GreyFailureAlert,
    GreyFailureType,
    HealthStatus,
    DetectorConfig,
)


@dataclass
class ContextPressureConfig:
    """Configuration for rate-based context pressure detection."""
    # Token growth rate detection
    baseline_turns: int = 3  # Turns to establish baseline
    growth_rate_threshold: float = 1.5  # Alert if growth > 1.5x baseline
    
    # Semantic density detection  
    semantic_density_threshold: float = 0.75  # Alert if density < 0.75x baseline
    
    # Predictive trajectory
    min_turns_until_exhaustion: int = 6  # Alert if < 6 turns remaining
    context_limit: int = 8000  # Max tokens
    
    # Early warning (more conservative)
    enable_early_warning: bool = False
    warning_turns_threshold: int = 10  # Warn if < 10 turns remaining
    
    # Require multiple signals for alert (critical for reducing FP)
    require_multiple_signals: bool = True
    min_signals_for_alert: int = 2  # Need 2+ signals to fire


@dataclass
class ContextPressureState:
    """State tracking for context pressure detection."""
    token_history: List[int] = field(default_factory=list)
    content_history: List[str] = field(default_factory=list)
    baseline_growth_rate: Optional[float] = None
    baseline_semantic_density: Optional[float] = None
    alerts: List[GreyFailureAlert] = field(default_factory=list)


def extract_concepts(text: str) -> set:
    """Extract unique concepts from text for semantic density calculation.
    
    Simple implementation: unique words after basic normalization.
    In production, would use embeddings or more sophisticated NLP.
    """
    # Normalize: lowercase, remove punctuation
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'or', 'and', 'but', 'if', 'then', 'so', 'than', 'that',
                  'this', 'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they'}
    concepts = {w for w in words if w not in stop_words and len(w) > 2}
    return concepts


def calculate_semantic_density(content: str, total_tokens: int) -> float:
    """Calculate semantic density: unique concepts / total tokens.
    
    Higher = more diverse content
    Lower = repetitive/compressed reasoning
    """
    if total_tokens == 0:
        return 1.0
    concepts = extract_concepts(content)
    return len(concepts) / total_tokens


def calculate_token_growth_rate(token_history: List[int]) -> float:
    """Calculate average tokens added per turn.
    
    Returns:
        Average tokens per turn (growth rate)
    """
    if len(token_history) < 2:
        return 0.0
    
    deltas = []
    for i in range(1, len(token_history)):
        delta = token_history[i] - token_history[i-1]
        deltas.append(max(0, delta))  # Only count increases
    
    return statistics.mean(deltas) if deltas else 0.0


def predict_turns_until_exhaustion(
    current_tokens: int,
    growth_rate: float,
    context_limit: int
) -> float:
    """Predict turns until context exhaustion via linear extrapolation.
    
    Args:
        current_tokens: Current token count
        growth_rate: Average tokens added per turn
        context_limit: Maximum context window
        
    Returns:
        Estimated turns until limit (inf if growth_rate <= 0)
    """
    if growth_rate <= 0:
        return float('inf')
    
    remaining = context_limit - current_tokens
    if remaining <= 0:
        return 0.0
    
    return remaining / growth_rate


class ContextPressureDetector:
    """Rate-based context pressure detection.
    
    Unlike threshold-based detection (triggers at 80%/95% utilization),
    this detects behavioral changes that precede context limits:
    
    1. Token growth rate exceeds sustainable levels
    2. Semantic density degrades (repetitive content)
    3. Predicted exhaustion within N turns
    """
    
    def __init__(self, config: Optional[ContextPressureConfig] = None):
        self.config = config or ContextPressureConfig()
        self.state = ContextPressureState()
    
    def record_turn(
        self,
        turn: AgentTurn,
        content: str = ""
    ) -> List[GreyFailureAlert]:
        """Record a turn and check for context pressure.
        
        Args:
            turn: Agent turn data
            content: Output content for semantic analysis
            
        Returns:
            List of any new alerts
        """
        # Track history
        self.state.token_history.append(turn.tokens_used)
        self.state.content_history.append(content)
        
        new_alerts = []
        
        # Establish baseline after initial turns
        if len(self.state.token_history) == self.config.baseline_turns:
            self._establish_baseline()
        
        # Run detection after baseline established
        if len(self.state.token_history) > self.config.baseline_turns:
            # Collect potential alerts from all signals
            potential_alerts = []
            
            # Check 1: Token growth rate
            growth_alert = self._check_growth_rate(turn)
            if growth_alert:
                potential_alerts.append(('growth_rate', growth_alert))
            
            # Check 2: Semantic density
            density_alert = self._check_semantic_density(turn, content)
            if density_alert:
                potential_alerts.append(('semantic_density', density_alert))
            
            # Check 3: Predictive trajectory
            trajectory_alert = self._check_trajectory(turn)
            if trajectory_alert:
                potential_alerts.append(('trajectory', trajectory_alert))
            
            # Apply multi-signal requirement if enabled
            if self.config.require_multiple_signals:
                if len(potential_alerts) >= self.config.min_signals_for_alert:
                    # Only add alerts if enough signals agree
                    for _, alert in potential_alerts:
                        new_alerts.append(alert)
            else:
                # Add all alerts
                for _, alert in potential_alerts:
                    new_alerts.append(alert)
        
        self.state.alerts.extend(new_alerts)
        return new_alerts
    
    def _establish_baseline(self):
        """Establish baselines from initial turns."""
        # Baseline growth rate
        self.state.baseline_growth_rate = calculate_token_growth_rate(
            self.state.token_history
        )
        
        # Baseline semantic density
        if self.state.content_history:
            densities = []
            for i, content in enumerate(self.state.content_history):
                tokens = self.state.token_history[i]
                density = calculate_semantic_density(content, tokens)
                densities.append(density)
            self.state.baseline_semantic_density = statistics.mean(densities)
    
    def _check_growth_rate(self, turn: AgentTurn) -> Optional[GreyFailureAlert]:
        """Check if token growth rate exceeds sustainable levels."""
        if self.state.baseline_growth_rate is None or self.state.baseline_growth_rate <= 0:
            return None
        
        # Calculate recent growth rate (last 3 turns)
        recent_history = self.state.token_history[-4:]  # Need 4 to get 3 deltas
        if len(recent_history) < 2:
            return None
        
        recent_rate = calculate_token_growth_rate(recent_history)
        ratio = recent_rate / self.state.baseline_growth_rate
        
        if ratio > self.config.growth_rate_threshold:
            severity = min((ratio - self.config.growth_rate_threshold) / 2, 1.0)
            return GreyFailureAlert(
                failure_type=GreyFailureType.CONTEXT_EXHAUSTION,
                severity=severity,
                turn_detected=len(self.state.token_history),
                message=f"Token growth rate {ratio:.1f}x baseline (rate-based detection)",
                evidence={
                    "current_growth_rate": recent_rate,
                    "baseline_growth_rate": self.state.baseline_growth_rate,
                    "ratio": ratio,
                    "detection_method": "growth_rate"
                }
            )
        
        return None
    
    def _check_semantic_density(
        self,
        turn: AgentTurn,
        content: str
    ) -> Optional[GreyFailureAlert]:
        """Check if semantic density is degrading."""
        if self.state.baseline_semantic_density is None:
            return None
        
        current_density = calculate_semantic_density(content, turn.tokens_used)
        
        # Handle edge case of zero baseline
        if self.state.baseline_semantic_density <= 0:
            return None
        
        ratio = current_density / self.state.baseline_semantic_density
        
        if ratio < self.config.semantic_density_threshold:
            severity = min((self.config.semantic_density_threshold - ratio) / 0.4, 1.0)
            return GreyFailureAlert(
                failure_type=GreyFailureType.CONTEXT_EXHAUSTION,
                severity=severity,
                turn_detected=len(self.state.token_history),
                message=f"Semantic density {ratio:.1%} of baseline (content compression)",
                evidence={
                    "current_density": current_density,
                    "baseline_density": self.state.baseline_semantic_density,
                    "ratio": ratio,
                    "detection_method": "semantic_density"
                }
            )
        
        return None
    
    def _check_trajectory(self, turn: AgentTurn) -> Optional[GreyFailureAlert]:
        """Check if predicted exhaustion is imminent."""
        # Calculate current growth rate
        recent_history = self.state.token_history[-4:]
        if len(recent_history) < 2:
            return None
        
        growth_rate = calculate_token_growth_rate(recent_history)
        turns_remaining = predict_turns_until_exhaustion(
            turn.tokens_used,
            growth_rate,
            self.config.context_limit
        )
        
        if turns_remaining < self.config.min_turns_until_exhaustion:
            severity = 1.0 - (turns_remaining / self.config.min_turns_until_exhaustion)
            return GreyFailureAlert(
                failure_type=GreyFailureType.CONTEXT_EXHAUSTION,
                severity=min(severity, 1.0),
                turn_detected=len(self.state.token_history),
                message=f"Context exhaustion predicted in {turns_remaining:.1f} turns",
                evidence={
                    "current_tokens": turn.tokens_used,
                    "context_limit": self.config.context_limit,
                    "growth_rate": growth_rate,
                    "turns_remaining": turns_remaining,
                    "detection_method": "trajectory_prediction"
                }
            )
        
        # Early warning
        if (self.config.enable_early_warning and 
            turns_remaining < self.config.warning_turns_threshold):
            return GreyFailureAlert(
                failure_type=GreyFailureType.CONTEXT_EXHAUSTION,
                severity=0.4,  # Low severity for early warning
                turn_detected=len(self.state.token_history),
                message=f"Early warning: ~{turns_remaining:.0f} turns until context limit",
                evidence={
                    "current_tokens": turn.tokens_used,
                    "turns_remaining": turns_remaining,
                    "detection_method": "early_warning"
                }
            )
        
        return None
    
    def get_health_score(self) -> float:
        """Get context pressure health score (0-1, higher is healthier)."""
        if not self.state.token_history:
            return 1.0
        
        scores = []
        
        # Token headroom
        latest_tokens = self.state.token_history[-1]
        headroom = 1.0 - (latest_tokens / self.config.context_limit)
        scores.append(max(0, headroom))
        
        # Growth rate sustainability
        if self.state.baseline_growth_rate and self.state.baseline_growth_rate > 0:
            recent_rate = calculate_token_growth_rate(self.state.token_history[-4:])
            rate_ratio = recent_rate / self.state.baseline_growth_rate
            rate_health = max(0, 1.0 - (rate_ratio - 1.0) / 2)  # Penalize > 1.0x
            scores.append(rate_health)
        
        # Trajectory safety
        recent_rate = calculate_token_growth_rate(self.state.token_history[-4:])
        turns_remaining = predict_turns_until_exhaustion(
            latest_tokens, recent_rate, self.config.context_limit
        )
        trajectory_health = min(1.0, turns_remaining / 10)  # 10+ turns = healthy
        scores.append(trajectory_health)
        
        return statistics.mean(scores) if scores else 1.0
    
    def has_alerts(self) -> bool:
        """Check if any alerts have been triggered."""
        return len(self.state.alerts) > 0


# =============================================================================
# Synthetic Trace Generation for Context Pressure
# =============================================================================

def generate_context_pressure_trace(
    severity: str,
    n_turns: int = 10,
    seed: Optional[int] = None
) -> Tuple[List[AgentTurn], List[str]]:
    """Generate a trace with gradual context pressure buildup.
    
    Unlike simple threshold traces, these simulate realistic scenarios:
    - Gradual token growth acceleration
    - Content becoming repetitive under pressure
    - Eventual approach toward limit
    
    Key insight: Low severity traces should NOT hit the 80% threshold,
    but SHOULD show rate-based warning signs (accelerating growth).
    
    Args:
        severity: 'low', 'medium', or 'high'
        n_turns: Number of turns to generate
        seed: Random seed
        
    Returns:
        (turns, contents) - Turn data and associated content strings
    """
    if seed is not None:
        random.seed(seed)
    
    # Severity determines rate of escalation AND final utilization
    # Low: Fast growth rate but stops at ~60% utilization (no threshold alert)
    # Medium: Moderate acceleration, reaches ~75% (borderline threshold)
    # High: Strong acceleration, clearly exceeds 80% threshold
    sev_params = {
        'low': {
            'base_growth': 250, 
            'acceleration': 1.25,  # Strong acceleration (rate signal)
            'density_decay': 0.88,
            'max_util': 0.60,  # Never hits 80% threshold
        },
        'medium': {
            'base_growth': 350, 
            'acceleration': 1.20,
            'density_decay': 0.82,
            'max_util': 0.78,  # Just under threshold
        },
        'high': {
            'base_growth': 450, 
            'acceleration': 1.35,
            'density_decay': 0.70,
            'max_util': 0.95,  # Clearly exceeds threshold
        },
    }
    params = sev_params.get(severity, sev_params['medium'])
    
    turns = []
    contents = []
    
    context_limit = 8000
    current_tokens = 800  # Starting point
    max_tokens = int(context_limit * params['max_util'])
    
    base_vocab = [
        "analyze", "process", "evaluate", "compute", "determine",
        "calculate", "measure", "assess", "review", "examine",
        "investigate", "explore", "consider", "understand", "interpret",
        "implement", "execute", "perform", "complete", "accomplish"
    ]
    
    for i in range(n_turns):
        progress = i / max(n_turns - 1, 1)
        
        # Calculate growth for this turn (accelerating)
        growth_mult = params['acceleration'] ** i
        tokens_added = int(params['base_growth'] * growth_mult)
        tokens_added += random.randint(-30, 30)  # Add noise
        
        # Cap at max utilization for this severity
        current_tokens = min(current_tokens + tokens_added, max_tokens)
        
        # Generate content with degrading diversity
        density_factor = params['density_decay'] ** (i * 0.5)
        vocab_size = max(5, int(len(base_vocab) * density_factor))
        active_vocab = base_vocab[:vocab_size]
        
        # Generate synthetic content
        n_words = max(10, int(tokens_added * 0.15))
        words = [random.choice(active_vocab) for _ in range(n_words)]
        content = " ".join(words)
        
        # Quality degrades with context pressure
        quality = max(0.5, 0.9 - progress * 0.25 * (params['acceleration'] - 1))
        quality += random.gauss(0, 0.02)
        
        turn = AgentTurn(
            turn_id=i,
            timestamp=datetime.now() + timedelta(seconds=i * 2),
            confidence=max(0.5, 0.85 - progress * 0.15),
            latency_ms=150 + progress * 80,
            tokens_used=current_tokens,
            tokens_available=context_limit,
            output_quality=quality,
            reasoning_consistent=True,
        )
        
        turns.append(turn)
        contents.append(content)
    
    return turns, contents


def generate_healthy_context_trace(
    n_turns: int = 10,
    noise_level: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[List[AgentTurn], List[str]]:
    """Generate a healthy trace with stable context usage.
    
    Token growth is linear and sustainable (no acceleration), content stays diverse.
    Final utilization stays well under any threshold (~40%).
    """
    if seed is not None:
        random.seed(seed)
    
    turns = []
    contents = []
    
    context_limit = 8000
    current_tokens = 800
    
    # Larger vocab for healthy traces (more diverse)
    base_vocab = [
        "analyze", "process", "evaluate", "compute", "determine",
        "calculate", "measure", "assess", "review", "examine",
        "investigate", "explore", "consider", "understand", "interpret",
        "implement", "execute", "perform", "complete", "accomplish",
        "optimize", "enhance", "improve", "refine", "develop",
        "validate", "verify", "confirm", "establish", "demonstrate"
    ]
    
    for i in range(n_turns):
        # Stable linear growth (NO acceleration)
        tokens_added = 180 + random.randint(-15, 15)
        tokens_added += int(random.gauss(0, noise_level * 30))
        current_tokens = min(current_tokens + tokens_added, int(context_limit * 0.45))
        
        # Diverse content - use full vocab
        n_words = max(10, int(tokens_added * 0.15))
        words = [random.choice(base_vocab) for _ in range(n_words)]
        content = " ".join(words)
        
        turn = AgentTurn(
            turn_id=i,
            timestamp=datetime.now() + timedelta(seconds=i * 2),
            confidence=0.88 + random.gauss(0, noise_level * 0.5),
            latency_ms=150 + random.gauss(0, noise_level * 15),
            tokens_used=current_tokens,
            tokens_available=context_limit,
            output_quality=0.87 + random.gauss(0, noise_level * 0.5),
            reasoning_consistent=True,
        )
        
        turns.append(turn)
        contents.append(content)
    
    return turns, contents


# =============================================================================
# Ablation Experiment
# =============================================================================

def run_context_pressure_ablation(
    n_per_severity: int = 50,
    n_healthy: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """Run ablation experiment comparing WITH vs WITHOUT context pressure signal.
    
    Args:
        n_per_severity: Traces per severity level
        n_healthy: Number of healthy traces
        seed: Random seed
        
    Returns:
        Dictionary with ablation results
    """
    random.seed(seed)
    
    # Generate test data
    failure_traces = []
    for severity in ['low', 'medium', 'high']:
        for i in range(n_per_severity):
            trace_seed = seed + hash(f"failure_{severity}_{i}") % 10000
            turns, contents = generate_context_pressure_trace(
                severity=severity,
                n_turns=10,
                seed=trace_seed
            )
            failure_traces.append((turns, contents, severity))
    
    healthy_traces = []
    for i in range(n_healthy):
        trace_seed = seed + hash(f"healthy_{i}") % 10000
        turns, contents = generate_healthy_context_trace(
            n_turns=10,
            noise_level=0.05,
            seed=trace_seed
        )
        healthy_traces.append((turns, contents))
    
    # Run WITH context pressure (full detector)
    results_with = evaluate_detector(
        failure_traces,
        healthy_traces,
        enable_context_pressure=True
    )
    
    # Run WITHOUT context pressure (context signal disabled)
    results_without = evaluate_detector(
        failure_traces,
        healthy_traces,
        enable_context_pressure=False
    )
    
    # Calculate delta
    delta_f1 = results_with['f1'] - results_without['f1']
    delta_tpr = results_with['tpr'] - results_without['tpr']
    
    return {
        'with_context_pressure': results_with,
        'without_context_pressure': results_without,
        'delta_f1': delta_f1,
        'delta_tpr': delta_tpr,
        'context_contributes': delta_f1 > 0,
        'n_failure_traces': len(failure_traces),
        'n_healthy_traces': len(healthy_traces),
    }


def evaluate_detector(
    failure_traces: List[Tuple[List[AgentTurn], List[str], str]],
    healthy_traces: List[Tuple[List[AgentTurn], List[str]]],
    enable_context_pressure: bool = True
) -> Dict[str, float]:
    """Evaluate detector on test data.
    
    Args:
        failure_traces: List of (turns, contents, severity) tuples
        healthy_traces: List of (turns, contents) tuples
        enable_context_pressure: Whether to use context pressure detection
        
    Returns:
        Dictionary with TPR, FPR, Precision, F1
    """
    tp = 0  # True positives
    fn = 0  # False negatives
    fp = 0  # False positives
    tn = 0  # True negatives
    
    # Evaluate failure traces
    for turns, contents, severity in failure_traces:
        detected = run_detection(turns, contents, enable_context_pressure)
        if detected:
            tp += 1
        else:
            fn += 1
    
    # Evaluate healthy traces
    for turns, contents in healthy_traces:
        detected = run_detection(turns, contents, enable_context_pressure)
        if detected:
            fp += 1
        else:
            tn += 1
    
    # Calculate metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
    
    return {
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'tn': tn,
    }


def run_detection(
    turns: List[AgentTurn],
    contents: List[str],
    enable_context_pressure: bool
) -> bool:
    """Run detection on a single trace.
    
    Args:
        turns: Agent turns
        contents: Content strings
        enable_context_pressure: Whether to use context pressure
        
    Returns:
        True if failure detected
    """
    if enable_context_pressure:
        detector = ContextPressureDetector()
        for turn, content in zip(turns, contents):
            detector.record_turn(turn, content)
        return detector.has_alerts()
    else:
        # Without context pressure, we only have the basic threshold check
        # Check if >80% utilization (the old broken approach)
        for turn in turns:
            utilization = turn.tokens_used / turn.tokens_available
            if utilization > 0.80:
                return True
        return False


def run_by_severity_ablation(
    n_per_severity: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """Run ablation broken down by severity level.
    
    Shows that context pressure signal contributes at each severity.
    """
    random.seed(seed)
    
    results_by_severity = {}
    
    for severity in ['low', 'medium', 'high']:
        # Generate traces for this severity
        failure_traces = []
        for i in range(n_per_severity):
            trace_seed = seed + hash(f"{severity}_{i}") % 10000
            turns, contents = generate_context_pressure_trace(
                severity=severity,
                n_turns=10,
                seed=trace_seed
            )
            failure_traces.append((turns, contents, severity))
        
        # Generate healthy traces
        healthy_traces = []
        for i in range(n_per_severity // 2):
            trace_seed = seed + hash(f"healthy_{severity}_{i}") % 10000
            turns, contents = generate_healthy_context_trace(
                n_turns=10,
                noise_level=0.05,
                seed=trace_seed
            )
            healthy_traces.append((turns, contents))
        
        # Evaluate with and without
        with_cp = evaluate_detector(failure_traces, healthy_traces, True)
        without_cp = evaluate_detector(failure_traces, healthy_traces, False)
        
        results_by_severity[severity] = {
            'with_context_pressure': with_cp,
            'without_context_pressure': without_cp,
            'delta_f1': with_cp['f1'] - without_cp['f1'],
            'delta_tpr': with_cp['tpr'] - without_cp['tpr'],
        }
    
    return results_by_severity


def early_detection_analysis(
    n_traces: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """Analyze early detection capability.
    
    Measures what fraction of trace elapsed before detection.
    Lower = earlier detection = better.
    """
    random.seed(seed)
    
    results = {
        'rate_based': [],  # Detection turn for rate-based
        'threshold_based': [],  # Detection turn for threshold-based
    }
    
    for i in range(n_traces):
        trace_seed = seed + hash(f"early_{i}") % 10000
        turns, contents = generate_context_pressure_trace(
            severity='medium',
            n_turns=15,  # Longer traces to show early detection
            seed=trace_seed
        )
        
        # Rate-based detection
        detector = ContextPressureDetector()
        rate_detection_turn = None
        for j, (turn, content) in enumerate(zip(turns, contents)):
            alerts = detector.record_turn(turn, content)
            if alerts and rate_detection_turn is None:
                rate_detection_turn = j
        
        if rate_detection_turn is not None:
            results['rate_based'].append(rate_detection_turn / len(turns))
        
        # Threshold-based detection (80% utilization)
        threshold_detection_turn = None
        for j, turn in enumerate(turns):
            utilization = turn.tokens_used / turn.tokens_available
            if utilization > 0.80 and threshold_detection_turn is None:
                threshold_detection_turn = j
        
        if threshold_detection_turn is not None:
            results['threshold_based'].append(threshold_detection_turn / len(turns))
    
    return {
        'rate_based_mean_detection_latency': (
            statistics.mean(results['rate_based']) if results['rate_based'] else 1.0
        ),
        'threshold_based_mean_detection_latency': (
            statistics.mean(results['threshold_based']) if results['threshold_based'] else 1.0
        ),
        'rate_based_detections': len(results['rate_based']),
        'threshold_based_detections': len(results['threshold_based']),
        'rate_based_details': results['rate_based'],
        'threshold_based_details': results['threshold_based'],
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    print("=" * 60)
    print("Context Pressure v2 - Rate-Based Detection Experiment")
    print("=" * 60)
    
    # Run main ablation
    print("\n1. Running main ablation experiment...")
    ablation_results = run_context_pressure_ablation(
        n_per_severity=50,
        n_healthy=100,
        seed=42
    )
    
    print(f"\n   WITH context pressure signal:")
    print(f"     TPR: {ablation_results['with_context_pressure']['tpr']:.3f}")
    print(f"     FPR: {ablation_results['with_context_pressure']['fpr']:.3f}")
    print(f"     F1:  {ablation_results['with_context_pressure']['f1']:.3f}")
    
    print(f"\n   WITHOUT context pressure signal:")
    print(f"     TPR: {ablation_results['without_context_pressure']['tpr']:.3f}")
    print(f"     FPR: {ablation_results['without_context_pressure']['fpr']:.3f}")
    print(f"     F1:  {ablation_results['without_context_pressure']['f1']:.3f}")
    
    print(f"\n   ΔF1: {ablation_results['delta_f1']:.3f}")
    print(f"   ΔTPR: {ablation_results['delta_tpr']:.3f}")
    print(f"   Context Pressure Contributes: {ablation_results['context_contributes']}")
    
    # Run severity breakdown
    print("\n2. Running severity-level ablation...")
    severity_results = run_by_severity_ablation(n_per_severity=50, seed=42)
    
    for severity, data in severity_results.items():
        print(f"\n   {severity.upper()} severity:")
        print(f"     ΔF1: {data['delta_f1']:.3f}, ΔTPR: {data['delta_tpr']:.3f}")
    
    # Run early detection analysis
    print("\n3. Running early detection analysis...")
    early_results = early_detection_analysis(n_traces=50, seed=42)
    
    print(f"\n   Rate-based mean detection latency: {early_results['rate_based_mean_detection_latency']:.2%}")
    print(f"   Threshold-based mean detection latency: {early_results['threshold_based_mean_detection_latency']:.2%}")
    print(f"   Rate-based detections: {early_results['rate_based_detections']}/50")
    print(f"   Threshold-based detections: {early_results['threshold_based_detections']}/50")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success = ablation_results['delta_f1'] > 0.02
    print(f"\nTarget: ΔF1 > 0.02")
    print(f"Result: ΔF1 = {ablation_results['delta_f1']:.3f}")
    print(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
    
    if early_results['rate_based_mean_detection_latency'] < early_results['threshold_based_mean_detection_latency']:
        print(f"\n✓ Early warning: Rate-based detects {(early_results['threshold_based_mean_detection_latency'] - early_results['rate_based_mean_detection_latency'])*100:.1f}% earlier")
    
    # Save results
    output_dir = Path(__file__).parent
    output_file = output_dir / "context_pressure_v2_results.json"
    
    all_results = {
        'ablation': {
            'with_context_pressure': ablation_results['with_context_pressure'],
            'without_context_pressure': ablation_results['without_context_pressure'],
            'delta_f1': ablation_results['delta_f1'],
            'delta_tpr': ablation_results['delta_tpr'],
        },
        'by_severity': severity_results,
        'early_detection': {
            'rate_based_latency': early_results['rate_based_mean_detection_latency'],
            'threshold_based_latency': early_results['threshold_based_mean_detection_latency'],
            'improvement': early_results['threshold_based_mean_detection_latency'] - early_results['rate_based_mean_detection_latency'],
        },
        'success_criteria': {
            'target_delta_f1': 0.02,
            'achieved_delta_f1': ablation_results['delta_f1'],
            'met': success,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
