#!/usr/bin/env python3
"""
Run baseline anomaly detectors on unified benchmark data.

Implements three sklearn baseline detectors:
- Isolation Forest (IF)
- Local Outlier Factor (LOF) 
- One-Class SVM (OCSVM)

Extracts features and evaluates performance against ground truth.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features(traces: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from traces for anomaly detection.
    
    Features:
    - Mean confidence per trace
    - Confidence variance
    - Turn count
    - Token count (estimated from text length)
    - Response length variance
    
    Args:
        traces: List of trace dictionaries
        
    Returns:
        features: Feature matrix (n_samples, n_features)
        labels: Ground truth labels (1 = healthy, -1 = unhealthy)
    """
    features = []
    labels = []
    
    for trace in traces:
        try:
            # Extract basic info
            is_healthy = trace.get('healthy', False)  # Use 'healthy' field from trace structure
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
            
            # Rough token estimation: ~4 chars per token
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
            # Convert to sklearn convention: 1 = normal (healthy), -1 = anomaly (unhealthy)
            labels.append(1 if is_healthy else -1)
            
        except Exception as e:
            logger.warning(f"Error processing trace: {e}")
            continue
    
    return np.array(features), np.array(labels)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Calculate detection metrics.
    
    Args:
        y_true: True labels (1 = healthy, -1 = unhealthy)
        y_pred: Predicted labels (1 = healthy, -1 = unhealthy) 
        y_scores: Anomaly scores (higher = more anomalous)
        
    Returns:
        Dictionary with TPR, FPR, F1, AUC metrics
    """
    # Convert to binary format for metrics calculation
    # True: 1 = healthy, 0 = unhealthy
    # Pred: 1 = healthy, 0 = unhealthy
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # Calculate metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    
    # F1 score (harmonic mean of precision and recall)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC-ROC (need to flip scores for unhealthy detection)
    try:
        # For AUC, we want to detect unhealthy (0) vs healthy (1)
        # Higher scores should indicate more likely to be unhealthy
        auc = roc_auc_score(1 - y_true_binary, -y_scores)
    except:
        auc = 0.5
    
    return {
        'tpr': float(tpr),
        'fpr': float(fpr), 
        'f1': float(f1),
        'auc': float(auc)
    }

def run_baseline_detectors(features: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Run all three baseline detectors and calculate metrics.
    
    Args:
        features: Feature matrix
        labels: Ground truth labels
        
    Returns:
        Dictionary with results for each detector
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    # 1. Isolation Forest
    logger.info("Running Isolation Forest...")
    try:
        if_model = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
        if_pred = if_model.fit_predict(features_scaled)
        if_scores = if_model.score_samples(features_scaled)
        results['if'] = calculate_metrics(labels, if_pred, if_scores)
        logger.info(f"IF - AUC: {results['if']['auc']:.3f}, F1: {results['if']['f1']:.3f}")
    except Exception as e:
        logger.error(f"Isolation Forest failed: {e}")
        results['if'] = {'tpr': 0.0, 'fpr': 1.0, 'f1': 0.0, 'auc': 0.5}
    
    # 2. Local Outlier Factor
    logger.info("Running Local Outlier Factor...")
    try:
        lof_model = LocalOutlierFactor(contamination='auto', novelty=False)
        lof_pred = lof_model.fit_predict(features_scaled)
        lof_scores = lof_model.negative_outlier_factor_
        results['lof'] = calculate_metrics(labels, lof_pred, lof_scores)
        logger.info(f"LOF - AUC: {results['lof']['auc']:.3f}, F1: {results['lof']['f1']:.3f}")
    except Exception as e:
        logger.error(f"Local Outlier Factor failed: {e}")
        results['lof'] = {'tpr': 0.0, 'fpr': 1.0, 'f1': 0.0, 'auc': 0.5}
    
    # 3. One-Class SVM
    logger.info("Running One-Class SVM...")
    try:
        svm_model = OneClassSVM(gamma='scale', nu=0.1)
        svm_pred = svm_model.fit_predict(features_scaled)
        svm_scores = svm_model.score_samples(features_scaled)
        results['ocsvm'] = calculate_metrics(labels, svm_pred, svm_scores)
        logger.info(f"OCSVM - AUC: {results['ocsvm']['auc']:.3f}, F1: {results['ocsvm']['f1']:.3f}")
    except Exception as e:
        logger.error(f"One-Class SVM failed: {e}")
        results['ocsvm'] = {'tpr': 0.0, 'fpr': 1.0, 'f1': 0.0, 'auc': 0.5}
    
    return results

def process_benchmark_data(data_path: str) -> Dict[str, Any]:
    """
    Load and process all benchmark data.
    
    Args:
        data_path: Path to the unified benchmarks JSON file
        
    Returns:
        Dictionary with results for each benchmark
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data in chunks to handle large file
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {}
    except MemoryError:
        logger.error("File too large to load into memory")
        return {}
    
    all_results = {}
    
    # Get available benchmarks from the data
    benchmarks = []
    if 'stats' in data:
        benchmarks = list(data['stats'].keys())
    
    logger.info(f"Found benchmarks: {benchmarks}")
    
    # Process each benchmark
    for benchmark_name in benchmarks:
        logger.info(f"\nProcessing benchmark: {benchmark_name}")
        
        # Extract traces for this benchmark
        benchmark_traces = []
        if 'traces' in data:
            for trace in data['traces']:
                if trace.get('benchmark') == benchmark_name:
                    benchmark_traces.append(trace)
        
        if not benchmark_traces:
            logger.warning(f"No traces found for benchmark {benchmark_name}")
            continue
            
        logger.info(f"Found {len(benchmark_traces)} traces for {benchmark_name}")
        
        # Extract features
        features, labels = extract_features(benchmark_traces)
        
        if len(features) == 0:
            logger.warning(f"No features extracted for benchmark {benchmark_name}")
            continue
            
        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Healthy traces: {np.sum(labels == 1)}, Unhealthy: {np.sum(labels == -1)}")
        
        # Run baseline detectors
        baseline_results = run_baseline_detectors(features, labels)
        
        # Store results
        all_results[benchmark_name] = {
            'total_traces': len(benchmark_traces),
            'processed_traces': len(features),
            'healthy_traces': int(np.sum(labels == 1)),
            'unhealthy_traces': int(np.sum(labels == -1)),
            'baselines': baseline_results
        }
    
    return all_results

def main():
    """Main execution function."""
    
    # Paths
    workspace = Path("/Users/gp/.openclaw/workspace")
    data_path = workspace / "projects/grey-failures-paper/data/unified/all-benchmarks.json"
    results_dir = workspace / "projects/grey-failures-paper/results"
    results_path = results_dir / "baseline-results.json"
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data file exists
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Process all benchmarks
    logger.info("Starting baseline detector evaluation...")
    results = process_benchmark_data(str(data_path))
    
    if not results:
        logger.error("No results generated")
        return
    
    # Save results
    logger.info(f"Saving results to {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n=== BASELINE RESULTS SUMMARY ===")
    for benchmark_name, benchmark_results in results.items():
        logger.info(f"\n{benchmark_name.upper()}:")
        logger.info(f"  Traces processed: {benchmark_results['processed_traces']}")
        logger.info(f"  Healthy/Unhealthy: {benchmark_results['healthy_traces']}/{benchmark_results['unhealthy_traces']}")
        
        baselines = benchmark_results['baselines']
        logger.info("  Baseline Performance:")
        for detector, metrics in baselines.items():
            logger.info(f"    {detector.upper()}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
    
    logger.info(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()