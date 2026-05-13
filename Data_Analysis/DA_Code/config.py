"""
Configuration and constants for the Data Analysis script.
"""
import os
from pathlib import Path

# Default seeds and variants (from projection_eval.yaml)
DEFAULT_SEEDS = [6, 7, 8, 9, 10]

DEFAULT_PROJECTION_VARIANTS = [
    'dpcc-r',
    'dpcc-r-tightened',
    'dpcc-c',
    'dpcc-c-tightened',
    'dpcc-t',
    'dpcc-t-tightened',
    'diffuser',
    'gradient',
    'gradient-tightened',
    'post_processing',
    'post_processing-tightened',
    'model_free',
    'model_free-tightened',
    'dpcc-c-tightened-dt0p25',
    'dpcc-c-tightened-dt0p5',
    'dpcc-c-tightened-dt2p0',
    'dpcc-c-tightened-dt4p0',
]

DEFAULT_CONSTRAINT_TYPES = [
    'halfspace',
    'obstacles',
    'dynamics',
    'bounds',
]

DEFAULT_HALFSPACE_VARIANTS = [
    'top-right-hard',
    'top-left-hard',
    'both-hard',
]

# Metrics extracted from .npz files
METRICS = [
    'n_success',
    'n_success_and_constraints',
    'n_steps',
    'n_violations',
    'total_violations',
    'avg_time',
    'collision_free_completed',
]

# Plot styling
PLOT_CONFIG = {
    'figsize': (12, 7),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'font_size': 11,
    'title_size': 13,
    'legend_size': 10,
    'colors': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5',
    ]
}

# Metric-specific formatting
METRIC_LABELS = {
    'n_success': 'Goal Success Rate (%)',
    'n_success_and_constraints': 'Goal + Constraint Success Rate (%)',
    'collision_free_completed': 'Collision-Free Rate (%)',
    'n_steps': 'Planning Steps',
    'avg_time': 'Computation Time (ms)',
    'n_violations': 'Avg Violations per Trial',
    'total_violations': 'Total Cumulative Violations',
}

METRIC_TYPES = {
    'n_success': 'percentage',
    'n_success_and_constraints': 'percentage',
    'collision_free_completed': 'percentage',
    'n_steps': 'continuous',
    'avg_time': 'continuous',
    'n_violations': 'continuous',
    'total_violations': 'continuous',
}

# Output folder naming
OUTPUT_FOLDER_PREFIX = 'FM_V3_ODE_Analysis'
