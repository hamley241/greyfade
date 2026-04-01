"""Grey Failure Detector for AI Agents.

Real-time detection of subtle failures that don't trigger traditional alerts.
This is the core prototype for the paper.

Key insight: Traditional monitoring is binary (up/down). Grey failures are
gradual degradations that pass health checks but hurt users.

Detection approach: Multi-signal health scoring with trend analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import statistics


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Grey failure detected
    FAILED = "failed"      # Hard failure


class GreyFailureType(Enum):
    CONFIDENCE_DEGRADATION = "confidence_degradation"
    LATENCY_CREEP = "latency_creep"
    QUALITY_DRIFT = "quality_drift"
    CONTEXT_EXHAUSTION = "context_exhaustion"
    REASONING_INCONSISTENCY = "reasoning_inconsistency"


@dataclass
class AgentTurn:
    """Single turn of agent execution."""
    turn_id: int
    timestamp: datetime
    confidence: float  # 0.0 - 1.0
    latency_ms: float
    tokens_used: int
    tokens_available: int
    output_quality: Optional[float] = None  # 0.0 - 1.0 if evaluated
    reasoning_consistent: Optional[bool] = None
    tool_calls: int = 0
    tool_failures: int = 0


@dataclass
class GreyFailureAlert:
    """Alert when grey failure detected."""
    failure_type: GreyFailureType
    severity: float  # 0.0 - 1.0
    turn_detected: int
    message: str
    evidence: dict


@dataclass
class DetectorConfig:
    """Tunable detection thresholds.
    
    These defaults were selected through grid search over 500 parameter
    combinations evaluated against 13 scenarios (8 failure modes, 5 healthy).
    See sensitivity_analysis.py for methodology.
    """
    # Confidence degradation
    # -0.04 achieves F1=0.933 vs -0.03 at F1=0.875 (avoids brief-dip FPs)
    confidence_slope_threshold: float = -0.04  # Per-turn drop
    confidence_min_threshold: float = 0.7  # Absolute threshold
    
    # Latency creep
    latency_baseline_multiplier: float = 2.0
    latency_trend_threshold: float = 1.2  # 20% increase per turn
    
    # Context exhaustion
    token_utilization_warning: float = 0.8
    token_utilization_critical: float = 0.95
    
    # Quality drift
    quality_drop_threshold: float = 0.15
    
    # Minimum turns before detection
    min_turns_for_trend: int = 3


class GreyFailureDetector:
    """Real-time grey failure detection for AI agents.
    
    Unlike post-hoc analysis tools, this detects failures DURING execution,
    enabling intervention before complete task failure.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.turns: list[AgentTurn] = []
        self.alerts: list[GreyFailureAlert] = []
        self.baseline_latency: Optional[float] = None
        
    def record_turn(self, turn: AgentTurn) -> list[GreyFailureAlert]:
        """Record a turn and check for grey failures.
        
        Returns any new alerts detected this turn.
        """
        self.turns.append(turn)
        new_alerts = []
        
        # Set baseline from first few turns
        if len(self.turns) == self.config.min_turns_for_trend:
            self._establish_baselines()
        
        # Run all detectors
        if len(self.turns) >= self.config.min_turns_for_trend:
            new_alerts.extend(self._check_confidence_degradation())
            new_alerts.extend(self._check_latency_creep())
            new_alerts.extend(self._check_context_exhaustion())
            new_alerts.extend(self._check_quality_drift())
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
        """Multi-dimensional health score (0.0 - 1.0).
        
        This is the key innovation: instead of binary health,
        we compute a continuous score from multiple signals.
        """
        if not self.turns:
            return 1.0
            
        latest = self.turns[-1]
        scores = []
        
        # Confidence component
        scores.append(latest.confidence)
        
        # Latency component (inverse, normalized)
        if self.baseline_latency:
            latency_ratio = self.baseline_latency / max(latest.latency_ms, 1)
            scores.append(min(latency_ratio, 1.0))
        
        # Token headroom component
        token_ratio = 1 - (latest.tokens_used / max(latest.tokens_available, 1))
        scores.append(max(token_ratio, 0))
        
        # Quality component (if available)
        if latest.output_quality is not None:
            scores.append(latest.output_quality)
        
        # Tool success component
        if latest.tool_calls > 0:
            tool_success = 1 - (latest.tool_failures / latest.tool_calls)
            scores.append(tool_success)
        
        return statistics.mean(scores) if scores else 1.0
    
    def _establish_baselines(self):
        """Establish baselines from initial turns."""
        latencies = [t.latency_ms for t in self.turns]
        self.baseline_latency = statistics.mean(latencies)
    
    def _check_confidence_degradation(self) -> list[GreyFailureAlert]:
        """Detect declining confidence over turns."""
        alerts = []
        confidences = [t.confidence for t in self.turns]
        
        # Check slope (linear regression simplified)
        n = len(confidences)
        if n < self.config.min_turns_for_trend:
            return alerts
            
        # Calculate slope: (y2 - y1) / (x2 - x1) averaged
        slopes = []
        for i in range(1, len(confidences)):
            slopes.append(confidences[i] - confidences[i-1])
        avg_slope = statistics.mean(slopes)
        
        if avg_slope < self.config.confidence_slope_threshold:
            severity = min(abs(avg_slope) / 0.1, 1.0)  # Normalize
            alerts.append(GreyFailureAlert(
                failure_type=GreyFailureType.CONFIDENCE_DEGRADATION,
                severity=severity,
                turn_detected=len(self.turns),
                message=f"Confidence declining at {avg_slope:.2%} per turn",
                evidence={
                    "slope": avg_slope,
                    "start_confidence": confidences[0],
                    "current_confidence": confidences[-1],
                    "trend": confidences[-5:] if len(confidences) >= 5 else confidences
                }
            ))
        
        # Also check absolute threshold
        if confidences[-1] < self.config.confidence_min_threshold:
            alerts.append(GreyFailureAlert(
                failure_type=GreyFailureType.CONFIDENCE_DEGRADATION,
                severity=0.7,
                turn_detected=len(self.turns),
                message=f"Confidence below threshold: {confidences[-1]:.2%}",
                evidence={"current_confidence": confidences[-1]}
            ))
        
        return alerts
    
    def _check_latency_creep(self) -> list[GreyFailureAlert]:
        """Detect gradually increasing latency."""
        alerts = []
        if not self.baseline_latency:
            return alerts
            
        latest = self.turns[-1]
        ratio = latest.latency_ms / self.baseline_latency
        
        if ratio > self.config.latency_baseline_multiplier:
            severity = min((ratio - 1) / 2, 1.0)
            alerts.append(GreyFailureAlert(
                failure_type=GreyFailureType.LATENCY_CREEP,
                severity=severity,
                turn_detected=len(self.turns),
                message=f"Latency {ratio:.1f}x baseline ({latest.latency_ms:.0f}ms vs {self.baseline_latency:.0f}ms)",
                evidence={
                    "current_latency": latest.latency_ms,
                    "baseline_latency": self.baseline_latency,
                    "ratio": ratio
                }
            ))
        
        return alerts
    
    def _check_context_exhaustion(self) -> list[GreyFailureAlert]:
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
    
    def _check_quality_drift(self) -> list[GreyFailureAlert]:
        """Detect declining output quality."""
        alerts = []
        qualities = [t.output_quality for t in self.turns if t.output_quality is not None]
        
        if len(qualities) < self.config.min_turns_for_trend:
            return alerts
        
        initial_avg = statistics.mean(qualities[:self.config.min_turns_for_trend])
        current = qualities[-1]
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
    
    def _check_reasoning_consistency(self) -> list[GreyFailureAlert]:
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


# Convenience function for simple usage
def create_detector(
    confidence_threshold: float = 0.6,
    latency_multiplier: float = 2.0,
    token_warning: float = 0.8
) -> GreyFailureDetector:
    """Create a detector with custom thresholds."""
    config = DetectorConfig(
        confidence_min_threshold=confidence_threshold,
        latency_baseline_multiplier=latency_multiplier,
        token_utilization_warning=token_warning
    )
    return GreyFailureDetector(config)
