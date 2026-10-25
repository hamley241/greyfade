# Corrections — 2026-10-25

This note corrects the results presented in this repository as of commit `3d3013f` (the initial
commit, 2026-04-01 UTC). Nothing has been deleted or rewritten; the original files are kept so the
problems below can be checked. **Do not cite the numbers in `README.md` or `figures/` as measured
results.** Every point below was checked against the code at `3d3013f` on 2026-10-25.

## 1. The baseline ROC curves in Figure 1 are interpolated, not measured

`src/generate_roc_figure.py` (`generate_baseline_roc`) takes three TPR values per baseline (at FPR
0.01, 0.05 and 0.10), read from a results file that is not in this repository. It then constructs
five more points: the first from those values (`tpr_20 = min(tpr_10 + (tpr_10 - tpr_05) * 0.8,
0.95)`), the rest by fixed increments of 0.15, 0.10, 0.08 and 0.05, each capped. The code comment
reads "Add some curvature to make it look more realistic". It draws a PCHIP curve through these
points and reports the area under that curve as the baseline AUC in the figure legend.

The Isolation Forest, LOF and OCSVM curves and their AUCs in `figures/fig1-roc-curves.png`
(0.559, 0.329, 0.411) are therefore **illustrations, not measurements**. They should not be
compared with the GREYFADE curve.

## 2. The three places that report results disagree, and none can be reproduced as a measurement

| Source | GREYFADE | Baselines |
|---|---|---|
| `README.md` Table 1 | TPR@1%FPR 0.43, TPR@5%FPR 0.67, AUC 0.783 | IF AUC 0.687, OCSVM 0.672, "Confidence Threshold" 0.723 |
| `figures/fig1-roc-curves.png` | AUC 0.978, TPR 22.5% @1%, 62.0% @5% | IF 0.559, LOF 0.329, OCSVM 0.411 (interpolated, see §1) |
| `figures/fig2-tpr-comparison.png` (values hard-coded in `src/generate_bar_figure.py`) | aggregate TPR@1%FPR 0.077 | best baseline 0.025; annotated "3.3×", while the code comment computes 3.08× |

None of these measurements can be reproduced from committed inputs:
- No data or result files are committed. Figure 1 requires result JSON files that are not in the
  repository. Figure 2 can be redrawn from its hard-coded values, but those values cannot be
  reproduced as measurements.
- `src/run_experiments.py` and `src/eval_sota_baselines.py` import modules that are not in the
  repository (`stats_utils`, `format_results`, `baselines`), so they cannot run.
- The README's `scripts/` directory, `GREYFADEDetector` class and its CLI flags do not exist. The
  detector classes are `GreyFailureDetector` (`src/grey_detector.py`) and the separate text-heuristic
  detector in `src/run_detector.py`.

## 3. The benchmark "degradation" labels share inputs with the detector

In the benchmark track, `src/convert_unified.py` (`is_healthy`) marks a trace unhealthy if the task
failed, if its average confidence is below 0.4, or if its last turns contain phrases such as "error",
"unable to", "early stop" or "loop detected". For GAIA, WebArena and AgentBench traces that
confidence is itself a regular-expression score over the transcript (`estimate_confidence`);
SWE-bench uses supplied confidence values. `src/run_detector.py` scores
overlapping inputs: the same confidence values and some of the same transcript markers. This overlap
is a label-leakage mechanism that risks inflating the reported separability; its size cannot be
measured from the committed artifacts.

The README ablation's signal names (confidence, latency, context, quality, reasoning) belong to the
prototype detector in `src/grey_detector.py`. The included experiment generator and tests exercise
that detector only on synthetic traces.

The unified benchmark turns carry no latency and no per-turn token counts, so the benchmark
detector's latency signal cannot fire and its token signal uses `len(text) // 4` estimates.

## 4. Smaller corrections

- `src/download_gaia.py` does not download the GAIA dataset. It downloads HAL leaderboard agent
  traces on GAIA tasks and matches evaluation labels to traces by position.
- `src/` contains 24 absolute local paths across 17 files, referring to the author's machine. Several
  scripts need them changed to run elsewhere; some can be overridden with command-line arguments.

## Status

The repository is kept as a record. The claims in the README's "Key Results" section are withdrawn
until a version exists that regenerates from committed data, uses labels independent of the detector,
and reports measured baselines.
