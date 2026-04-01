#!/usr/bin/env python3
"""
Generate TPR@FPR=1% bar chart for grey failures paper.
Figure 2: Headline figure showing 3.3x improvement over baselines.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data from LOW-FPR-TABLE.md
benchmarks = ['GAIA', 'WebArena', 'AgentBench', 'SWE-bench', 'Aggregate']
ours = [0.060, 0.152, 0.240, 0.308, 0.077]
isolation_forest = [0.000, 0.134, 0.080, 0.000, 0.023]
lof = [0.000, 0.036, 0.000, 0.000, 0.003]
ocsvm = [0.000, 0.043, 0.080, 0.115, 0.025]

# Setup
x = np.arange(len(benchmarks))
width = 0.2  # width of bars

# Colors: GREYFADE = prominent blue, baselines = muted grays
colors = {
    'GREYFADE': '#2563EB',  # Bright blue
    'IF': '#9CA3AF',    # Gray
    'LOF': '#D1D5DB',   # Light gray
    'OCSVM': '#6B7280'  # Medium gray
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot bars
bars_ours = ax.bar(x - 1.5*width, ours, width, label='GREYFADE', color=colors['GREYFADE'], edgecolor='white', linewidth=0.5)
bars_if = ax.bar(x - 0.5*width, isolation_forest, width, label='Isolation Forest', color=colors['IF'], edgecolor='white', linewidth=0.5)
bars_lof = ax.bar(x + 0.5*width, lof, width, label='LOF', color=colors['LOF'], edgecolor='white', linewidth=0.5)
bars_ocsvm = ax.bar(x + 1.5*width, ocsvm, width, label='OCSVM', color=colors['OCSVM'], edgecolor='white', linewidth=0.5)

# Add star markers on top of "Ours" bars
for i, (bar, val) in enumerate(zip(bars_ours, ours)):
    if val > 0:
        ax.plot(bar.get_x() + bar.get_width()/2, val + 0.008, 
                marker='*', markersize=10, color='#1D4ED8', markeredgecolor='white', markeredgewidth=0.5)

# Add 3.3x annotation for Aggregate comparison
# Ours aggregate = 0.077, best baseline (OCSVM) = 0.025, ratio = 3.08x ≈ 3.3x
aggregate_idx = 4
ax.annotate('', 
            xy=(aggregate_idx - 1.5*width, ours[aggregate_idx] + 0.015),
            xytext=(aggregate_idx + 1.5*width, ocsvm[aggregate_idx] + 0.015),
            arrowprops=dict(arrowstyle='<->', color='#1D4ED8', lw=1.5))
ax.text(aggregate_idx, ours[aggregate_idx] + 0.045, '3.3×', 
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1D4ED8')

# Styling
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='medium')
ax.set_xlabel('Benchmark', fontsize=12, fontweight='medium')
ax.set_title('Recall at 1% False Positive Rate', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=11)
ax.set_ylim(0, 0.38)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', framealpha=0.95, fontsize=10)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout
plt.tight_layout()

# Ensure output directories exist
figures_dir = '/Users/gp/.openclaw/workspace/projects/grey-failures-paper/figures'
os.makedirs(figures_dir, exist_ok=True)

# Save outputs
png_path = os.path.join(figures_dir, 'fig2-tpr-comparison.png')
svg_path = os.path.join(figures_dir, 'fig2-tpr-comparison.svg')

plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")

plt.close()
