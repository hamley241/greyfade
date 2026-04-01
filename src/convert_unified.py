#!/usr/bin/env python3
"""
Convert benchmark traces (GAIA, WebArena, AgentBench, SWE-bench) to unified format.

Unified Schema:
{
  "trace_id": "benchmark-XXXX",
  "benchmark": "swe-bench|gaia|webarena|agentbench",
  "turns": [{"role": "...", "content": "...", "confidence": 0.0-1.0}],
  "success": true/false,
  "healthy": true/false,  // success=true AND no obvious degradation
  "metadata": {...}
}
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Paths
BASE_DIR = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper/data")
GAIA_PATH = BASE_DIR / "gaia/gaia-traces.json"
WEBARENA_PATH = BASE_DIR / "webarena/traces/webarena-traces.json"
AGENTBENCH_PATH = BASE_DIR / "agentbench/agentbench-traces.json"
SWEBENCH_PATH = BASE_DIR / "swe-bench/swe-bench-labelled.json"
OUTPUT_PATH = BASE_DIR / "unified/all-benchmarks.json"


def estimate_confidence(text: str) -> float:
    """
    Estimate confidence from text patterns.
    Returns a value between 0.0 and 1.0.
    """
    if not text:
        return 0.5
    
    text_lower = text.lower()
    
    # High confidence indicators
    high_conf_patterns = [
        r'\bi(?:\'m| am) (?:confident|certain|sure)\b',
        r'\bthe answer is\b',
        r'\bdefinitely\b',
        r'\bclearly\b',
        r'\bwithout doubt\b',
    ]
    
    # Low confidence indicators
    low_conf_patterns = [
        r'\bi(?:\'m| am) (?:not sure|uncertain|unsure)\b',
        r'\bmaybe\b',
        r'\bperhaps\b',
        r'\bi think\b',
        r'\bpossibly\b',
        r'\bmight be\b',
        r'\bcould be\b',
        r'\bi don\'t know\b',
        r'\bnot certain\b',
    ]
    
    # Degradation indicators
    degradation_patterns = [
        r'\bi (?:cannot|can\'t|am unable to)\b',
        r'\berror\b',
        r'\bfailed\b',
        r'\bunable to\b',
        r'\btask cannot be completed\b',
        r'\bearly stop\b',
        r'\bsame.*action.*times\b',
    ]
    
    score = 0.5
    
    for pattern in high_conf_patterns:
        if re.search(pattern, text_lower):
            score += 0.1
    
    for pattern in low_conf_patterns:
        if re.search(pattern, text_lower):
            score -= 0.1
    
    for pattern in degradation_patterns:
        if re.search(pattern, text_lower):
            score -= 0.15
    
    return max(0.0, min(1.0, score))


def is_healthy(success: bool, turns: List[Dict], avg_confidence: float) -> bool:
    """
    Determine if a trace is healthy (successful without degradation).
    """
    if not success:
        return False
    
    if avg_confidence < 0.4:
        return False
    
    # Check for degradation patterns in final turns
    if turns:
        last_turns_text = " ".join(
            t.get("content", "") for t in turns[-3:]
            if isinstance(t.get("content"), str)
        )
        degradation_patterns = [
            r'error',
            r'failed',
            r'unable to',
            r'cannot complete',
            r'task cannot be',
            r'early stop',
            r'same.*action.*times',
            r'loop detected',
        ]
        for pattern in degradation_patterns:
            if re.search(pattern, last_turns_text.lower()):
                return False
    
    return True


def convert_gaia(data: Dict) -> List[Dict]:
    """Convert GAIA format to unified format."""
    traces = data.get("traces", [])
    unified = []
    
    for i, trace in enumerate(traces):
        trace_id = f"gaia-{i:04d}"
        task_id = trace.get("internal_task_id", "")
        
        # Normalize turns
        raw_turns = trace.get("turns", [])
        turns = []
        confidences = []
        
        for turn in raw_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            
            # Map roles
            if role == "assistant":
                role = "ai"
            elif role == "user":
                role = "user"
            
            conf = estimate_confidence(content) if role == "ai" else None
            if conf is not None:
                confidences.append(conf)
            
            turns.append({
                "role": role,
                "content": content,
                "confidence": conf,
            })
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        
        # GAIA doesn't have explicit success in the trace structure we see
        # Estimate based on patterns and metadata
        success = trace.get("success", None)
        if success is None:
            # Check for final_answer in last turns
            has_final = any(
                "final_answer" in t.get("content", "").lower()
                for t in raw_turns[-3:]
            )
            success = has_final and avg_conf > 0.5
        
        healthy = is_healthy(success, turns, avg_conf)
        
        unified.append({
            "trace_id": trace_id,
            "benchmark": "gaia",
            "turns": turns,
            "success": success,
            "healthy": healthy,
            "metadata": {
                "task_id": task_id,
                "model": trace.get("model", ""),
                "avg_confidence": round(avg_conf, 3),
            }
        })
    
    return unified


def convert_webarena(data: List) -> List[Dict]:
    """Convert WebArena format to unified format."""
    unified = []
    
    for trace in data:
        trace_id = trace.get("trace_id", f"webarena-{len(unified):04d}")
        
        # Normalize turns
        raw_turns = trace.get("turns", [])
        turns = []
        confidences = []
        
        for turn in raw_turns:
            # WebArena has observation + action format
            observation = turn.get("observation", "")
            action = turn.get("action", "")
            
            # Environment/observation turn
            if observation:
                turns.append({
                    "role": "environment",
                    "content": observation[:2000],  # Truncate very long observations
                    "confidence": None,
                })
            
            # Agent action turn
            if action:
                conf = estimate_confidence(action)
                confidences.append(conf)
                turns.append({
                    "role": "ai",
                    "content": action,
                    "confidence": conf,
                })
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        success = trace.get("success", False)
        healthy = is_healthy(success, turns, avg_conf)
        
        unified.append({
            "trace_id": trace_id,
            "benchmark": "webarena",
            "turns": turns,
            "success": success,
            "healthy": healthy,
            "metadata": {
                "task_id": trace.get("task_id", ""),
                "model": trace.get("model", ""),
                "intent": trace.get("intent", ""),
                "reward": trace.get("reward", 0.0),
                "avg_confidence": round(avg_conf, 3),
            }
        })
    
    return unified


def convert_agentbench(data: List) -> List[Dict]:
    """Convert AgentBench format to unified format."""
    unified = []
    
    for trace in data:
        trace_id = trace.get("trace_id", f"agentbench-{len(unified):04d}")
        
        # Normalize turns
        raw_turns = trace.get("turns", [])
        turns = []
        confidences = []
        
        for turn in raw_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            
            # Map roles
            if role == "agent":
                role = "ai"
            elif role == "environment":
                role = "environment"
            # user stays as user
            
            conf = None
            if role == "ai":
                conf = estimate_confidence(content)
                confidences.append(conf)
            
            turns.append({
                "role": role,
                "content": content,
                "confidence": conf,
            })
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        success = trace.get("success", False)
        healthy = is_healthy(success, turns, avg_conf)
        
        unified.append({
            "trace_id": trace_id,
            "benchmark": "agentbench",
            "turns": turns,
            "success": success,
            "healthy": healthy,
            "metadata": {
                "domain": trace.get("domain", ""),
                "task_id": trace.get("task_id", ""),
                "score": trace.get("score", 0.0),
                "num_turns": trace.get("num_turns", len(raw_turns)),
                "source": trace.get("source", ""),
                "avg_confidence": round(avg_conf, 3),
            }
        })
    
    return unified


def convert_swebench(data: List) -> List[Dict]:
    """Convert SWE-bench format to unified format (already well-structured)."""
    unified = []
    
    for trace in data:
        trace_id = trace.get("trace_id", f"swe-bench-{len(unified):04d}")
        
        # SWE-bench already has good turn structure with confidence
        raw_turns = trace.get("turns", [])
        turns = []
        confidences = []
        
        for turn in raw_turns:
            role = turn.get("role", "unknown")
            
            # Map roles
            if role == "ai":
                pass  # Already correct
            elif role == "user":
                pass  # Already correct
            # Keep other roles as-is
            
            conf = turn.get("confidence")
            if conf is not None and role == "ai":
                confidences.append(conf)
            
            content = turn.get("raw_text", turn.get("content", ""))
            
            turns.append({
                "role": role,
                "content": content,
                "confidence": conf,
            })
        
        # Use existing summary if available
        summary = trace.get("summary", {})
        avg_conf = summary.get("avg_confidence", 
                              sum(confidences) / len(confidences) if confidences else 0.5)
        
        # SWE-bench success determination
        # If trace has explicit success field, use it
        # Otherwise, infer from summary
        success = trace.get("success", None)
        if success is None:
            # Check if there are resolved indicators
            success = avg_conf >= 0.6  # Heuristic
        
        healthy = is_healthy(success, turns, avg_conf)
        
        # Check for inconsistent reasoning as degradation indicator
        inconsistent = summary.get("inconsistent_reasoning_turns", 0)
        if inconsistent > summary.get("ai_turns", 1) * 0.3:
            healthy = False
        
        unified.append({
            "trace_id": trace_id,
            "benchmark": "swe-bench",
            "turns": turns,
            "success": success,
            "healthy": healthy,
            "metadata": {
                "instance_id": trace.get("instance_id", ""),
                "model": trace.get("model", ""),
                "source": trace.get("source", ""),
                "total_turns": summary.get("total_turns", len(raw_turns)),
                "ai_turns": summary.get("ai_turns", 0),
                "total_tokens": summary.get("total_tokens", 0),
                "avg_confidence": round(avg_conf, 3),
                "inconsistent_reasoning_turns": inconsistent,
            }
        })
    
    return unified


def main():
    """Main conversion function."""
    all_traces = []
    stats = {}
    
    # Convert GAIA
    print("Loading GAIA traces...")
    if GAIA_PATH.exists():
        with open(GAIA_PATH) as f:
            gaia_data = json.load(f)
        gaia_traces = convert_gaia(gaia_data)
        all_traces.extend(gaia_traces)
        success_count = sum(1 for t in gaia_traces if t["success"])
        healthy_count = sum(1 for t in gaia_traces if t["healthy"])
        stats["gaia"] = {
            "total": len(gaia_traces),
            "success": success_count,
            "healthy": healthy_count,
            "success_rate": round(success_count / len(gaia_traces), 3) if gaia_traces else 0,
            "healthy_rate": round(healthy_count / len(gaia_traces), 3) if gaia_traces else 0,
        }
        print(f"  GAIA: {len(gaia_traces)} traces, {success_count} success, {healthy_count} healthy")
    else:
        print(f"  GAIA file not found: {GAIA_PATH}")
        stats["gaia"] = {"error": "file not found"}
    
    # Convert WebArena
    print("Loading WebArena traces...")
    if WEBARENA_PATH.exists():
        with open(WEBARENA_PATH) as f:
            webarena_data = json.load(f)
        webarena_traces = convert_webarena(webarena_data)
        all_traces.extend(webarena_traces)
        success_count = sum(1 for t in webarena_traces if t["success"])
        healthy_count = sum(1 for t in webarena_traces if t["healthy"])
        stats["webarena"] = {
            "total": len(webarena_traces),
            "success": success_count,
            "healthy": healthy_count,
            "success_rate": round(success_count / len(webarena_traces), 3) if webarena_traces else 0,
            "healthy_rate": round(healthy_count / len(webarena_traces), 3) if webarena_traces else 0,
        }
        print(f"  WebArena: {len(webarena_traces)} traces, {success_count} success, {healthy_count} healthy")
    else:
        print(f"  WebArena file not found: {WEBARENA_PATH}")
        stats["webarena"] = {"error": "file not found"}
    
    # Convert AgentBench
    print("Loading AgentBench traces...")
    if AGENTBENCH_PATH.exists():
        with open(AGENTBENCH_PATH) as f:
            agentbench_data = json.load(f)
        agentbench_traces = convert_agentbench(agentbench_data)
        all_traces.extend(agentbench_traces)
        success_count = sum(1 for t in agentbench_traces if t["success"])
        healthy_count = sum(1 for t in agentbench_traces if t["healthy"])
        stats["agentbench"] = {
            "total": len(agentbench_traces),
            "success": success_count,
            "healthy": healthy_count,
            "success_rate": round(success_count / len(agentbench_traces), 3) if agentbench_traces else 0,
            "healthy_rate": round(healthy_count / len(agentbench_traces), 3) if agentbench_traces else 0,
        }
        print(f"  AgentBench: {len(agentbench_traces)} traces, {success_count} success, {healthy_count} healthy")
    else:
        print(f"  AgentBench file not found: {AGENTBENCH_PATH}")
        stats["agentbench"] = {"error": "file not found"}
    
    # Convert SWE-bench
    print("Loading SWE-bench traces...")
    if SWEBENCH_PATH.exists():
        with open(SWEBENCH_PATH) as f:
            swebench_data = json.load(f)
        swebench_traces = convert_swebench(swebench_data)
        all_traces.extend(swebench_traces)
        success_count = sum(1 for t in swebench_traces if t["success"])
        healthy_count = sum(1 for t in swebench_traces if t["healthy"])
        stats["swe-bench"] = {
            "total": len(swebench_traces),
            "success": success_count,
            "healthy": healthy_count,
            "success_rate": round(success_count / len(swebench_traces), 3) if swebench_traces else 0,
            "healthy_rate": round(healthy_count / len(swebench_traces), 3) if swebench_traces else 0,
        }
        print(f"  SWE-bench: {len(swebench_traces)} traces, {success_count} success, {healthy_count} healthy")
    else:
        print(f"  SWE-bench file not found: {SWEBENCH_PATH}")
        stats["swe-bench"] = {"error": "file not found"}
    
    # Calculate totals
    total = len(all_traces)
    total_success = sum(1 for t in all_traces if t["success"])
    total_healthy = sum(1 for t in all_traces if t["healthy"])
    
    stats["total"] = {
        "traces": total,
        "success": total_success,
        "healthy": total_healthy,
        "success_rate": round(total_success / total, 3) if total else 0,
        "healthy_rate": round(total_healthy / total, 3) if total else 0,
    }
    
    # Save output
    print(f"\nSaving {total} unified traces to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "version": "1.0.0",
        "created": "2026-03-25",
        "description": "Unified benchmark traces for grey failure analysis",
        "stats": stats,
        "traces": all_traces,
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDone! Output saved to {OUTPUT_PATH}")
    print(f"\n=== Summary Stats ===")
    print(f"Total traces: {total}")
    print(f"Total success: {total_success} ({stats['total']['success_rate']*100:.1f}%)")
    print(f"Total healthy: {total_healthy} ({stats['total']['healthy_rate']*100:.1f}%)")
    print(f"\nPer-benchmark breakdown:")
    for bench in ["gaia", "webarena", "agentbench", "swe-bench"]:
        if bench in stats and "total" in stats[bench]:
            s = stats[bench]
            print(f"  {bench}: {s['total']} traces, {s['success_rate']*100:.1f}% success, {s['healthy_rate']*100:.1f}% healthy")


if __name__ == "__main__":
    main()
