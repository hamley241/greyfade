#!/usr/bin/env python3
"""
Simple threshold tuning script - uses first 100 traces for quick testing.
"""

import json
import itertools
from pathlib import Path

# Define search ranges for each threshold
THRESHOLD_RANGES = {
    "confidence_threshold": [0.2, 0.3, 0.4, 0.5, 0.6],
    "confidence_drop": [0.15, 0.2, 0.25, 0.3, 0.4],
    "contradiction_ratio": [0.1, 0.15, 0.2, 0.25, 0.3],
    "loop_similarity": [0.7, 0.75, 0.8, 0.85, 0.9],
    "growth_rate": [1.5, 2.0, 2.5, 3.0],
    "min_signals": [1, 2, 3]
}

def apply_detector(data, thresholds):
    """Apply detector with given thresholds to data."""
    predictions = []
    
    for trace in data:
        signals = []
        
        # Extract ground truth (a trace is grey failure if success=True but healthy=False)
        success = trace.get('success', False)
        healthy = trace.get('healthy', True)
        ground_truth = success and not healthy  # Grey failure: successful but not healthy
        
        # Get conversation turns for analysis
        turns = trace.get('turns', [])
        
        if not turns:
            predictions.append((ground_truth, False))
            continue
            
        # Signal 1: Confidence threshold - only check AI turns
        ai_turns = [turn for turn in turns if turn.get('role') == 'ai']
        confidences = [turn.get('confidence', 1.0) for turn in ai_turns if turn.get('confidence') is not None]
        if confidences and min(confidences) < thresholds['confidence_threshold']:
            signals.append('low_confidence')
            
        # Signal 2: Confidence drop
        if len(confidences) > 1:
            max_drop = max(confidences[i] - confidences[i+1] for i in range(len(confidences)-1))
            if max_drop > thresholds['confidence_drop']:
                signals.append('confidence_drop')
        
        # Signal 3: Contradiction ratio - use metadata if available
        metadata = trace.get('metadata', {})
        inconsistent_turns = metadata.get('inconsistent_reasoning_turns', 0)
        total_ai_turns = metadata.get('ai_turns', len(ai_turns))
        contradiction_ratio = inconsistent_turns / max(total_ai_turns, 1)
        if contradiction_ratio > thresholds['contradiction_ratio']:
            signals.append('contradictions')
            
        # Signal 4: Loop similarity - check for repeated AI responses
        ai_responses = [turn.get('content', '') for turn in ai_turns]
        if len(ai_responses) > 1:
            similarities = []
            for i in range(len(ai_responses)-1):
                # Simple similarity check
                resp1_words = set(ai_responses[i].split())
                resp2_words = set(ai_responses[i+1].split())
                if len(resp1_words) == 0 or len(resp2_words) == 0:
                    similarity = 0
                else:
                    similarity = len(resp1_words & resp2_words) / max(len(resp1_words), len(resp2_words))
                similarities.append(similarity)
            if similarities and max(similarities) > thresholds['loop_similarity']:
                signals.append('loops')
                
        # Signal 5: Growth rate (token growth)
        total_tokens = metadata.get('total_tokens', 0)
        total_turns = metadata.get('total_turns', len(turns))
        if total_turns > 1:
            avg_tokens_per_turn = total_tokens / total_turns
            # Simple heuristic: if average tokens per turn is very high, it might indicate runaway generation
            if avg_tokens_per_turn > thresholds['growth_rate'] * 100:
                signals.append('token_growth')
        
        # Apply min_signals threshold
        prediction = len(signals) >= thresholds['min_signals']
        predictions.append((ground_truth, prediction))
    
    return predictions

def compute_metrics(predictions):
    """Compute F1, precision, recall from predictions."""
    # Calculate confusion matrix components
    tp = sum(1 for true, pred in predictions if true and pred)
    fp = sum(1 for true, pred in predictions if not true and pred)
    fn = sum(1 for true, pred in predictions if true and not pred)
    tn = sum(1 for true, pred in predictions if not true and not pred)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tpr': tpr,
        'fpr': fpr,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def main():
    """Main tuning function."""
    # Paths
    train_path = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper/data/unified/splits/train.json")
    results_dir = Path("/Users/gp/.openclaw/workspace/projects/grey-failures-paper/results")
    results_path = results_dir / "tuned-thresholds.json"
    
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading training data...")
    with open(train_path, 'r') as f:
        data = json.load(f)
    
    # Use first 100 traces for quick testing
    train_data = data['traces'][:100]
    print(f"Using first {len(train_data)} training examples for testing")
    
    # Show data stats
    grey_failures = sum(1 for trace in train_data if trace.get('success', False) and not trace.get('healthy', True))
    total = len(train_data)
    print(f"Grey failures: {grey_failures}/{total} ({100*grey_failures/total:.1f}%)")
    
    if grey_failures == 0:
        print("⚠️  No grey failures found in sample! Let me check success/healthy distribution...")
        success_count = sum(1 for trace in train_data if trace.get('success', False))
        healthy_count = sum(1 for trace in train_data if trace.get('healthy', True))
        print(f"Success: {success_count}/{total}, Healthy: {healthy_count}/{total}")
        
        # Try alternative definition: maybe unhealthy traces are the failures
        alt_failures = sum(1 for trace in train_data if not trace.get('healthy', True))
        print(f"Alternative (unhealthy traces): {alt_failures}/{total} ({100*alt_failures/total:.1f}%)")
    
    print("\nTesting one threshold combination...")
    test_thresholds = {
        "confidence_threshold": 0.5,
        "confidence_drop": 0.2,
        "contradiction_ratio": 0.15,
        "loop_similarity": 0.8,
        "growth_rate": 2.0,
        "min_signals": 1
    }
    
    predictions = apply_detector(train_data, test_thresholds)
    metrics = compute_metrics(predictions)
    print(f"Test metrics: {metrics}")
    
    # For now, just save this test result
    results = {
        "best_thresholds": test_thresholds,
        "train_metrics": metrics,
        "search_results": [{"thresholds": test_thresholds, "metrics": metrics}],
        "sample_size": len(train_data),
        "grey_failure_rate": grey_failures / total if total > 0 else 0,
        "note": "Test run with single threshold combination on first 100 traces"
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Test complete!")
    print(f"F1 score: {metrics['f1']:.4f}")
    print(f"Test thresholds: {test_thresholds}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()