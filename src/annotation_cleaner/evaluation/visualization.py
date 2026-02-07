#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module.

Generates publication-quality figures (PDF) from evaluation metrics.
Based on the style: Times New Roman, STIX fonts, and Scatter plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from utils.logging import get_logger

logger = get_logger("Visualizer")

# ================= STYLE CONFIGURATION =================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  
    "mathtext.fontset": "stix",         
    "axes.linewidth": 1.2,              
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "pdf.fonttype": 42,                
    "ps.fonttype": 42
})

# Color & Marker Palette
PALETTE = {
    'blur': '#3498db', 'jpeg': '#2ecc71', 'marked': '#95a5a6', 
    'mix': '#9b59b6', 'ood': '#f1c40f', 'gac': '#e74c3c'
}
MARKERS = {
    'blur': 'o', 'jpeg': 'o', 'marked': 's', 'mix': 'X', 'ood': 'd', 'gac': '*'
}
ANCHORS_ORDER = ['marked', 'blur', 'jpeg', 'ood', 'mix', 'gac']


def plot_metrics_scatter(config: Dict[str, Any]):
    """
    Reads the metrics CSV and generates a scatter plot (Figure 4).
    """
    logger.info("Generating evaluation figures...")

    # 1. Resolve Paths
    try:
        eval_conf = config["annotation_cleaner"]["evaluate"]
        out_dir = Path(eval_conf["dirs"]["results"])
        csv_path = out_dir / "metrics_details.csv"
        save_path = out_dir / "visualization.pdf"
    except KeyError:
        logger.error("Invalid config structure for visualization.")
        return

    if not csv_path.exists():
        logger.warning(f"Metrics CSV not found at {csv_path}. Skipping plot.")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    existing_anchors = [a for a in ANCHORS_ORDER if a in df['Anchor'].unique()]

    for anchor in existing_anchors:
        subset = df[df['Anchor'] == anchor]
        
        # Formatting Label
        if anchor == 'gac':
            label_text = r'$\mathbf{GAC \ (Ours)}$'
            z_order = 5
            size = 400
        else:
            label_text = anchor.capitalize()
            z_order = 3
            size = 100
        
        color = PALETTE.get(anchor, '#333333')
        marker = MARKERS.get(anchor, 'o')

        ax.scatter(
            subset['Edge_L1'], 
            subset['D_FeatMap'],
            c=color, 
            label=label_text,
            marker=marker,
            s=size,
            edgecolors='black',
            linewidth=0.8,
            alpha=0.9,
            zorder=z_order
        )

    # 4. Styling
    ax.set_xlabel(r'Structural Change ($\text{Edge } L_1$)', fontsize=15)
    ax.set_ylabel(r'Semantic Distance ($\mathcal{D}_{\text{FeatMap}}$)', fontsize=15)

    legend = ax.legend(
        title='Anchor Types', 
        fontsize=12, 
        title_fontsize=13, 
        loc='best', 
        frameon=True, 
        edgecolor='black'
    )
    legend.get_frame().set_linewidth(1.0)

    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # 5. Save
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    logger.info(f"Figure saved to: {save_path}")
    
    # Optional: Close plot to free memory if running in batch
    plt.close(fig)