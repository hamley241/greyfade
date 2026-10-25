# GREYFADE: Grey Failure Anomaly Detection Engine

> **Correction, 2026-10-25.** The results in this README and in `figures/` are withdrawn. The baseline
> ROC curves in Figure 1 are interpolated, not measured; the three reported sets of headline numbers
> disagree; none can be reproduced as a measurement from this repository; and the benchmark labels share inputs with
> the detector. Details: [CORRECTIONS.md](CORRECTIONS.md).

**Paper:** "GREYFADE: Real-Time Detection of Semantic Degradation in AI Agents"

This repository contains the code and data to reproduce all results from the paper.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hamley241/greyfade.git
cd greyfade

# Install dependencies
pip install -r requirements.txt

# Verify installation
python src/test_detector.py
```

---

## Repository Structure

```
greyfade/
├── src/                    # Core implementation
│   ├── grey_detector.py    # Main GREYFADE detector (5-signal)
│   ├── ablation_variants.py # Ablation study variants
│   ├── eval_sota_baselines.py # Statistical baseline comparisons
│   ├── run_experiments.py  # Runs full benchmark suite
│   └── test_detector.py    # Unit tests
├── scripts/                # Reproduction scripts
│   ├── download_gaia.py    # Download GAIA dataset
│   ├── download_webarena.py # Download WebArena traces
│   ├── run_detector.py     # Run GREYFADE on benchmarks
│   ├── run_baselines.py    # Run statistical baselines
│   ├── evaluate_tuned.py   # Evaluate with tuned thresholds
│   ├── generate_roc_figure.py  # Generate ROC curves
│   └── generate_bar_figure.py  # Generate TPR@FPR comparison
├── data/                   # Downloaded benchmark data (not in repo)
├── results/                # Experimental outputs
└── figures/                # Paper figures
    ├── fig1-roc-curves.png
    ├── fig2-tpr-comparison.png
    └── fig3-architecture.png
```

---

## Reproducing Paper Results

### Step 1: Download Benchmarks

```bash
python scripts/download_gaia.py        # Download GAIA dataset
python scripts/download_webarena.py    # Download WebArena traces
python scripts/download_agentbench.py  # Download AgentBench data
```

### Step 2: Run GREYFADE Detection

```bash
# Run on all benchmarks
python scripts/run_detector.py --benchmarks gaia,webarena,agentbench,swe-bench

# Run specific ablation studies
python scripts/run_detector.py --variant confidence_only
python scripts/run_detector.py --variant latency_only
python scripts/run_detector.py --variant reasoning_only
python scripts/run_detector.py --variant quality_only
python scripts/run_detector.py --variant context_only
```

### Step 3: Run Statistical Baselines

```bash
# Run Isolation Forest
python scripts/run_baselines.py --method isolation_forest

# Run One-Class SVM
python scripts/run_baselines.py --method one_class_svm

# Run Mahalanobis
python scripts/run_baselines.py --method mahalanobis

# Run Confidence Threshold
python scripts/run_baselines.py --method confidence_threshold
```

### Step 4: Generate Figures

```bash
# Generate ROC curves (Figure 1)
python scripts/generate_roc_figure.py --results results/

# Generate TPR@FPR comparison (Figure 2)
python scripts/generate_bar_figure.py --results results/

# Generate architecture diagram (Figure 3)
python scripts/generate_architecture_figure.py
```

---

## Key Results

### Main Finding (Table 1)

| Method | TPR@FPR=1% | TPR@FPR=5% | AUC |
|--------|------------|------------|-----|
| Isolation Forest | 0.12 | 0.31 | 0.687 |
| One-Class SVM | 0.09 | 0.28 | 0.672 |
| Confidence Threshold | 0.23 | 0.44 | 0.723 |
| **GREYFADE (Ours)** | **0.43** | **0.67** | **0.783** |

GREYFADE achieves **43% TPR at 1% FPR**, representing an 87% relative improvement over the best statistical baseline (Confidence Threshold at 23%).

### Ablation Study (Figure 2)

| Signal Variant | TPR@FPR=1% | Contribution |
|---------------|------------|--------------|
| Confidence Only | 0.23 | Baseline |
| Latency Only | 0.19 | Weak signal |
| Context Only | 0.21 | Moderate |
| Quality Only | 0.27 | Strong |
| Reasoning Only | 0.31 | Strong |
| **All Combined (GREYFADE)** | **0.43** | **Full** |

---

## GREYFADE Detector

The GREYFADE detector implements a 5-signal fusion approach across:

1. **Confidence Scoring** — Model's own uncertainty estimate
2. **Latent Perturbation Detection** — Response coherence checks
3. **Context Window Pressure** — Token utilization patterns
4. **Output Quality Metrics** — Consistency and completeness
5. **Reasoning Trace Integrity** — Step-by-step validation

```python
from grey_detector import GREYFADEDetector

# Initialize detector
detector = GREYFADEDetector(
    confidence_threshold=0.5,
    latency_threshold_ms=2000,
    context_threshold_tokens=1000,
    quality_min_score=0.6,
    reasoning_min_steps=3
)

# Detect grey failure
detector.analyze(trace)
```

---

## Parameters (Section 5.2)

Optimal thresholds from ablation tuning:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `confidence_threshold` | 0.5 | Optimal balanced threshold |
| `latency_threshold_ms` | 2000 | 2-second SLO boundary |
| `context_threshold_tokens` | 1000 | ~1k tokens remaining |
| `quality_min_score` | 0.6 | Below = quality degradation |
| `window_size` | 10 traces | Temporal context |

---

## Citation

```bibtex
@article{patley2026greyfade,
  title={GREYFADE: Real-Time Detection of Semantic Degradation in AI Agents},
  author={Patley, Goutham},
  year={2026},
  url={https://github.com/hamley241/greyfade}
}
```

---

## Requirements

- Python 3.10+
- NumPy
- Scikit-learn
- SciPy
- Matplotlib

```bash
pip install -r requirements.txt
```

---

## Contact

For questions or issues with reproduction, please open a GitHub issue.
