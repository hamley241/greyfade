#!/usr/bin/env python3
"""
Generate Architecture Diagram for Grey Failure Detection System
Shows the flow from Agent Execution through 5 Signal Extractors to Aggregation and Alert Decision
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import os

def create_architecture_diagram():
    # Set up the figure with 10x5 inches
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Color scheme
    blue_color = '#4A90E2'  # Blue for our components
    gray_color = '#8E8E93'  # Gray for inputs/outputs
    text_color = '#333333'
    
    # Define box dimensions
    box_width = 1.2
    box_height = 0.4
    small_box_height = 0.3
    
    # Agent Execution (input) - Gray
    agent_box = FancyBboxPatch(
        (0.5, 2.3), box_width, box_height,
        boxstyle="round,pad=0.02",
        facecolor=gray_color,
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(agent_box)
    ax.text(1.1, 2.5, 'Agent\nExecution', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    # Signal Extractors (5 parallel) - Blue
    extractors = [
        'Confidence Monitor\n(linguistic + logit)',
        'Semantic Validator\n(embeddings)',
        'Loop Detector\n(n-gram overlap)',
        'Context Pressure\n(token growth rate)',
        'Latency Tracker\n(statistical deviation)'
    ]
    
    extractor_y_positions = [4.0, 3.2, 2.4, 1.6, 0.8]
    extractor_boxes = []
    
    for i, (extractor, y_pos) in enumerate(zip(extractors, extractor_y_positions)):
        extractor_box = FancyBboxPatch(
            (3.5, y_pos - small_box_height/2), 2.0, small_box_height,
            boxstyle="round,pad=0.02",
            facecolor=blue_color,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(extractor_box)
        extractor_boxes.append((4.5, y_pos))  # Center point for connections
        
        # Split text for better formatting
        lines = extractor.split('\n')
        ax.text(4.5, y_pos, lines[0], ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
        if len(lines) > 1:
            ax.text(4.5, y_pos - 0.08, lines[1], ha='center', va='center', 
                    fontsize=6, color='white')
    
    # Aggregator - Blue
    aggregator_box = FancyBboxPatch(
        (7.0, 2.15), box_width, 0.7,
        boxstyle="round,pad=0.02",
        facecolor=blue_color,
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(aggregator_box)
    ax.text(7.6, 2.5, 'Aggregator\n(min_signals\nthreshold)', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Alert Decision (output) - Gray
    alert_box = FancyBboxPatch(
        (8.8, 2.3), box_width, box_height,
        boxstyle="round,pad=0.02",
        facecolor=gray_color,
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(alert_box)
    ax.text(9.4, 2.5, 'Alert /\nNo Alert', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    
    # Agent Execution to Signal Extractors
    for _, y_pos in extractor_boxes:
        ax.annotate('', xy=(3.5, y_pos), xytext=(1.7, 2.5),
                   arrowprops=arrow_props)
    
    # Signal Extractors to Aggregator
    for x_pos, y_pos in extractor_boxes:
        ax.annotate('', xy=(7.0, 2.5), xytext=(5.5, y_pos),
                   arrowprops=arrow_props)
    
    # Aggregator to Alert Decision
    ax.annotate('', xy=(8.8, 2.5), xytext=(8.2, 2.5),
               arrowprops=arrow_props)
    
    # Add title
    ax.text(5.0, 4.7, 'Grey Failure Detection System Architecture', 
            ha='center', va='center', fontsize=14, fontweight='bold', color=text_color)
    
    # Add subtle grid lines for better visual organization
    ax.axvline(x=2.8, color='lightgray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(x=6.2, color='lightgray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    return fig

def main():
    # Create output directories if they don't exist
    output_dir = '/Users/gp/.openclaw/workspace/projects/grey-failures-paper/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the diagram
    fig = create_architecture_diagram()
    
    # Save as PNG
    png_path = os.path.join(output_dir, 'fig3-architecture.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"PNG saved to: {png_path}")
    
    # Save as SVG
    svg_path = os.path.join(output_dir, 'fig3-architecture.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"SVG saved to: {svg_path}")
    
    # plt.show()  # Commented out to avoid interactive display

if __name__ == "__main__":
    main()