"""Ablation variants for grey failure detector.

Creates modified versions of the detector with specific components disabled
to measure the contribution of each signal to detection performance.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set
from datetime import datetime

from grey_detector import (
    GreyFailureDetector,
    AgentTurn,
    GreyFailureAlert,
    GreyFailureType,
    HealthStatus,
    DetectorConfig
)


class AblationDetector:
    """Detector wrapper that disables specific detection components.
    
    This allows measuring the contribution of each signal to overall detection.
    
    disable options:
    - 'confidence': Remove confidence signal (both slope and threshold)
    - 'latency': Remove latency signal
    - 'tokens': Remove token/context tracking
    - 'quality': Remove quality scoring
    - 'trend': Point-in-time only (no slope/trend analysis, just thresholds)
    - 'reasoning': Remove reasoning consistency check
    """
    
    VALID_COMPONENTS = {'confidence', 'latency', 'tokens', 'quality', 'trend', 'reasoning'}
    
    def __init__(self, config: Optional[DetectorConfig] = None, disable: Optional[List[str]] = None):
        """Initialize ablation detector.
        
        Args:
            config: Base detector configuration
            disable: List of components to disable
        """
        self.config = config or DetectorConfig()
        self.disable = set(disable or [])
        
        # Validate disable options
        invalid = self.disable - self.VALID_COMPONENTS
        if invalid:
            raise ValueError(f"Invalid disable options: {invalid}. Valid: {self.VALID_COMPONENTS}")
        
        # Internal state (mirrors GreyFailureDetector)
        self.turns: List[AgentTurn] = []
        self.alerts: List[GreyFailureAlert] = []
        self.baseline_latency: Optional[float] = None
        
    def record_turn(self, turn: AgentTurn) -> List[GreyFailureAlert]:
        """Record a turn and check for grey failures (with ablations)."""
        self.turns.append(turn)
        new_alerts = []
        
        # Establish baselines
        if len(self.turns) == self.config.min_turns_for_trend:
            self._establish_baselines()
        
        # Run enabled detectors
        if len(self.turns) >= self.config.min_turns_for_trend:
            if 'confidence' not in self.disable:
                new_alerts.extend(self._check_confidence_degradation())
            if 'latency' not in self.disable:
                new_alerts.extend(self._check_latency_creep())
            if 'tokens' not in self.disable:
                new_alerts.extend(self._check_context_exhaustion())
            if 'quality' not in self.disable:
                new_alerts.extend(self._check_quality_drift())
            if 'reasoning' not in self.disable:
                new_alerts.extend(self._check_reasoning_consistency())
        
        self.alerts.extend(new_alerts)
        return new_alerts
    
    def get_health_status(self) -> HealthStatus:
        """Get overall health status based on recent alerts."""
        if not self.alerts:
            return HealthStatus.HEALTHY
        
        recent_alerts = [a for a in self.alerts 
                        if a.turn_detected >= len(self.turns) - 3]
        
        if any(a.severity > 0.8 for a in recent_alerts):
            return HealthStatus.FAILED
        elif recent_alerts:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
    
    def get_health_score(self) -> float:
        """Multi-dimensional health score (0.0 - 1.0) with ablations."""
        if not self.turns:
            return 1.0
            
        latest = self.turns[-1]
        scores = []
        
        # Confidence component
        if 'confidence' not in self.disable:
            scores.append(latest.confidence)
        
        # Latency component
        if 'latency' not in self.disable and self.baseline_latency:
            latency_ratio = self.baseline_latency / max(latest.latency_ms, 1)
            scores.append(min(latency_ratio, 1.0))
        
        # Token headroom component
        if 'tokens' not in self.disable:
            token_ratio = 1 - (latest.tokens_used / max(latest.tokens_available, 1))
            scores.append(max(token_ratio, 0))
        
        # Quality component
        if 'quality' not in self.disable and latest.output_quality is not None:
            scores.append(latest.output_quality)
        
        # Tool success component (not disabled)
        if latest.tool_calls > 0:
            tool_success = 1 - (latest.tool_failures / latest.tool_calls)
            scores.append(tool_success)
        
        return sum(scores) / len(scores) if scores else 1.0
    
    def _establish_baselines(self):
        """Establish baselines from initial turns."""
        latencies = [t.latency_ms for t in self.turns]
        self.baseline_latency = sum(latencies) / len(latencies)
    
    def _check_confidence_degradation(self) -> List[GreyFailureAlert]:
        """Detect declining confidence (with trend ablation support)."""
        alerts = []
        confidences = [t.confidence for t in self.turns]
        
        n = len(confidences)
        if n < self.config.min_turns_for_trend:
            return alerts
        
        # If trend is disabled, only check absolute threshold
        if 'trend' not in self.disable:
            # Calculate slope
            slopes = []
            for i in range(1, len(confidences)):
                slopes.append(confidences[i] - confidences[i-1])
            avg_slope = sum(slopes) / len(slopes)
            
            if avg_slope < self.config.confidence_slope_threshold:
                severity = min(abs(avg_slope) / 0.1, 1.0)
                alerts.append(GreyFailureAlert(
                    failure_type=GreyFailureType.CONFIDENCE_DEGRADATION,
                    severity=severity,
                    turn_detected=len(self.turns),
                    message=f"Confidence declining at {avg_slope:.2%} per turn",
                    evidence={
                        "slope": avg_slope,
                        "start_confidence": confidences[0],
                        "current_confidence": confidences[-1],
                    }
                ))
        
        # Always check absolute threshold (unless entire confidence check disabled)
        if confidences[-1] < self.config.confidence_min_threshold:
            alerts.append(GreyFailureAlert(
                failure_type=GreyFailureType.CONFIDENCE_DEGRADATION,
                severity=0.7,
                turn_detected=len(self.turns),
                message=f"Confidence below threshold: {confidences[-1]:.2%}",
                evidence={"current_confidence": confidences[-1]}
            ))
        
        return alerts
    
    def _check_latency_creep(self) -> List[GreyFailureAlert]:
        """Detect gradually increasing latency (with trend ablation)."""
        alerts = []
        if not self.baseline_latency:
            return alerts
            
        latest = self.turns[-1]
        ratio = latest.latency_ms / self.baseline_latency
        
        # If trend disabled, only check if current exceeds baseline significantly
        if 'trend' in self.disable:
            # Point-in-time: only check if way above baseline (no trend)
            if ratio > self.config.latency_baseline_multiplier * 1.5:
                alerts.append(GreyFailureAlert(
                    failure_type=GreyFailureType.LATENCY_CREEP,
                    severity=min((ratio - 1) / 3, 1.0),
                    turn_detected=len(self.turns),
                    message=f"Latency {ratio:.1f}x baseline (point-in-time)",
                    evidence={
                        "current_latency": latest.latency_ms,
                        "baseline_latency": self.baseline_latency,
                        "ratio": ratio
                    }
                ))
        else:
            if ratio > self.config.latency_baseline_multiplier:
                severity = min((ratio - 1) / 2, 1.0)
                alerts.append(GreyFailureAlert(
                    failure_type=GreyFailureType.LATENCY_CREEP,
                    severity=severity,
                    turn_detected=len(self.turns),
                    message=f"Latency {ratio:.1f}x baseline",
                    evidence={
                        "current_latency": latest.latency_ms,
                        "baseline_latency": self.baseline_latency,
                        "ratio": ratio
                    }
                ))
        
        return alerts
    
    def _check_context_exhaustion(self) -> List[GreyFailureAlert]:
        """Detect approaching token limits."""
        alerts = []
        latest = self.turns[-1]
        utilization = latest.tokens_used / max(latest.tokens_available, 1)
        
        if utilization > self.config.token_utilization_critical:
            alerts.append(GreyFailureAlert(
                failure_type=GreyFailureType.CONTEXT_EXHAUSTION,
                severity=0.9,
                turn_detected=len(self.turns),
                message=f"Critical token utilization: {utilization:.1%}",
                evidence={
                    "tokens_used": latest.tokens_used,
                    "tokens_available": latest.tokens_available,
                    "utilization": utilization
                }
            ))
        elif utilization > self.config.token_utilization_warning:
            alerts.append(GreyFailureAlert(
                failure_type=GreyFailureType.CONTEXT_EXHAUSTION,
                severity=0.5,
                turn_detected=len(self.turns),
                message=f"High token utilization: {utilization:.1%}",
                evidence={
                    "tokens_used": latest.tokens_used,
                    "tokens_available": latest.tokens_available,
                    "utilization": utilization
                }
            ))
        
        return alerts
    
    def _check_quality_drift(self) -> List[GreyFailureAlert]:
        """Detect declining output quality (with trend ablation)."""
        alerts = []
        qualities = [t.output_quality for t in self.turns if t.output_quality is not None]
        
        if len(qualities) < self.config.min_turns_for_trend:
            return alerts
        
        current = qualities[-1]
        
        if 'trend' in self.disable:
            # Point-in-time: only check absolute quality
            if current < 0.6:  # Absolute threshold
                alerts.append(GreyFailureAlert(
                    failure_type=GreyFailureType.QUALITY_DRIFT,
                    severity=min((0.9 - current) / 0.4, 1.0),
                    turn_detected=len(self.turns),
                    message=f"Low quality: {current:.2%} (point-in-time)",
                    evidence={"current_quality": current}
                ))
        else:
            initial_avg = sum(qualities[:self.config.min_turns_for_trend]) / self.config.min_turns_for_trend
            drop = initial_avg - current
            
            if drop > self.config.quality_drop_threshold:
                alerts.append(GreyFailureAlert(
                    failure_type=GreyFailureType.QUALITY_DRIFT,
                    severity=min(drop / 0.3, 1.0),
                    turn_detected=len(self.turns),
                    message=f"Quality dropped {drop:.1%} from baseline",
                    evidence={
                        "initial_quality": initial_avg,
                        "current_quality": current,
                        "drop": drop
                    }
                ))
        
        return alerts
    
    def _check_reasoning_consistency(self) -> List[GreyFailureAlert]:
        """Detect inconsistent reasoning patterns."""
        alerts = []
        recent = self.turns[-3:] if len(self.turns) >= 3 else self.turns
        inconsistent = [t for t in recent if t.reasoning_consistent is False]
        
        if len(inconsistent) >= 2:
            alerts.append(GreyFailureAlert(
                failure_type=GreyFailureType.REASONING_INCONSISTENCY,
                severity=len(inconsistent) / len(recent),
                turn_detected=len(self.turns),
                message=f"Reasoning inconsistent in {len(inconsistent)}/{len(recent)} recent turns",
                evidence={"inconsistent_turns": [t.turn_id for t in inconsistent]}
            ))
        
        return alerts


def create_ablation_variants(config: Optional[DetectorConfig] = None) -> dict:
    """Create all ablation variants for comparison.
    
    Returns:
        dict mapping variant name to AblationDetector instance
    """
    config = config or DetectorConfig()
    
    variants = {
        'full': AblationDetector(config, disable=[]),
        'no_confidence': AblationDetector(config, disable=['confidence']),
        'no_latency': AblationDetector(config, disable=['latency']),
        'no_tokens': AblationDetector(config, disable=['tokens']),
        'no_quality': AblationDetector(config, disable=['quality']),
        'no_trend': AblationDetector(config, disable=['trend']),
        'no_reasoning': AblationDetector(config, disable=['reasoning']),
        # Minimal variants for extreme ablation
        'confidence_only': AblationDetector(config, disable=['latency', 'tokens', 'quality', 'reasoning']),
        'latency_only': AblationDetector(config, disable=['confidence', 'tokens', 'quality', 'reasoning']),
        'quality_only': AblationDetector(config, disable=['confidence', 'latency', 'tokens', 'reasoning']),
    }
    
    return variants


def reset_all_variants(variants: dict) -> dict:
    """Reset all variants to fresh state (for rerunning experiments).
    
    Returns:
        Fresh dict of variants
    """
    config = variants['full'].config if variants else DetectorConfig()
    return create_ablation_variants(config)


if __name__ == "__main__":
    from datetime import timedelta
    
    print("Testing ablation variants...")
    
    # Create test turn
    def make_turn(turn_id, confidence=0.9, latency_ms=150, tokens_used=2000,
                  tokens_available=8000, output_quality=0.85, reasoning_consistent=True):
        return AgentTurn(
            turn_id=turn_id,
            timestamp=datetime.now() + timedelta(seconds=turn_id * 2),
            confidence=confidence,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            tokens_available=tokens_available,
            output_quality=output_quality,
            reasoning_consistent=reasoning_consistent,
            tool_calls=0,
            tool_failures=0
        )
    
    # Create degradation scenario
    def run_degradation_scenario(detector):
        for i in range(10):
            conf = 0.9 - (i * 0.05)  # Declining confidence
            quality = 0.9 - (i * 0.04)  # Declining quality
            latency = 100 + (i * 50)  # Increasing latency
            
            detector.record_turn(make_turn(
                i,
                confidence=conf,
                latency_ms=latency,
                output_quality=quality
            ))
        return len(detector.alerts)
    
    # Test all variants
    variants = create_ablation_variants()
    
    print("\nRunning degradation scenario on all variants:")
    results = {}
    for name, detector in variants.items():
        alert_count = run_degradation_scenario(detector)
        results[name] = alert_count
        print(f"  {name}: {alert_count} alerts")
    
    # Verify full has most alerts
    assert results['full'] >= results['no_confidence'], "Full should detect >= no_confidence"
    assert results['full'] >= results['no_latency'], "Full should detect >= no_latency"
    assert results['full'] >= results['no_quality'], "Full should detect >= no_quality"
    
    print("\n✅ Ablation variants working correctly!")
    print("\nAlert counts show impact of each component:")
    for name, count in sorted(results.items(), key=lambda x: -x[1]):
        diff = results['full'] - count
        print(f"  {name}: {count} alerts (full - {diff} = {results['full'] - count})")
