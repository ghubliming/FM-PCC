"""
Configuration for DA Visual Aligning analysis.

Single-seed focus. No geometric constraints (null). Per-rollout arrays from NPZ.
"""
import os

# Multi-seed list kept for reference; only index [0] (ACTIVE_SEED) used by default
DEFAULT_SEEDS = [6, 7, 8, 9, 10]
ACTIVE_SEED   = 6   # single-seed mode default

DEFAULT_PROJECTION_VARIANTS = [
    'diffuser',
    'dpcc-r',
    'dpcc-r-tightened',
    'dpcc-c',
    'dpcc-c-tightened',
    'dpcc-t',
    'dpcc-t-tightened',
    'gradient',
    'gradient-tightened',
    'post_processing',
    'post_processing-tightened',
    'model_free',
    'model_free-tightened',
]

# Per-rollout array metrics (shape: (N_rollouts,))
METRICS = [
    'n_success',                   # (30,) per-rollout success flags
    'success_rate',                # scalar — mean(n_success)
    'mean_dist_per_rollout',       # (30,) final box–target distance (m)
    'n_steps',                     # (30,) steps per rollout
    'avg_time',                    # (30,) seconds/replan
    'max_phys_error_per_rollout',  # (30,) max PD tracking error (m)
    'context_init_xy_dist',        # (30,) init box→target dist (m)
    # --- U_2: context_info extended ---
    'context_final_xy_dist',       # (30,) final box→target dist (m)
    'context_final_box_angle_deg', # (30,) final box angle (deg)
    # --- U_2: constraint_metrics ---
    'exec_n_violated_steps',       # (30,) violated steps count
    'exec_constraint_sat_rate',    # (30,) constraint satisfaction rate
    'exec_zero_violation_rollout', # (30,) boolean flag
    'exec_bounds_viol_count',      # (30,) bounds violation count
    'exec_halfspace_viol_count',   # (30,) halfspace violation count
    'exec_obstacle_viol_count',    # (30,) obstacle violation count
    'exec_max_bounds_viol_m',      # (30,) max bounds violation (m)
    'exec_max_halfspace_viol_m',   # (30,) max halfspace violation (m)
    'exec_max_obstacle_penetration_m', # (30,) max obstacle penetration (m)
    'exec_constraint_margin_mean_m',   # (30,) constraint margin mean (m)
    'exec_first_violation_step',       # (30,) first violation step idx
    'exec_longest_safe_streak',        # (30,) longest safe streak
    'exec_dynamics_consistency_error_mean',  # (30,) dynamics error mean
    'exec_dynamics_consistency_error_max',   # (30,) dynamics error max
    'plan_post_viol_rate_mean',    # (30,) plan post-viol rate mean
    'plan_post_viol_rate_max',     # (30,) plan post-viol rate max
    'plan_n_replan_steps',         # (30,) plan n replan steps
]

# Constraint category — null for now; slot back in when geo constraints added
# CONSTRAINT-DEACTIVATED
CONSTRAINT_TYPES   = []
HALFSPACE_VARIANTS = []

OUTPUT_FOLDER_PREFIX = 'Visual_Aligning_Analysis'

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
    ],
}

METRIC_LABELS = {
    'success_rate':             'Success Rate (%)',
    'n_success':                'Success per Rollout',
    'mean_dist_per_rollout':    'Final Box-Target Distance (m)',
    'n_steps':                  'Steps per Rollout',
    'avg_time':                 'Inference Time / Replan (s)',
    'max_phys_error_per_rollout': 'Max Physical Tracking Error (m)',
    'context_init_xy_dist':     'Init Box-Target Distance (m)',
    # --- U_2: context_info extended ---
    'context_final_xy_dist':    'Final XY Distance (m)',
    'context_final_box_angle_deg': 'Final Box Angle (deg)',
    # --- U_2: constraint_metrics ---
    'exec_n_violated_steps':       'Violated Steps Count',
    'exec_constraint_sat_rate':    'Constraint Satisfaction Rate',
    'exec_zero_violation_rollout': 'Zero-Violation Rollout',
    'exec_bounds_viol_count':      'Bounds Violation Count',
    'exec_halfspace_viol_count':   'Halfspace Violation Count',
    'exec_obstacle_viol_count':    'Obstacle Violation Count',
    'exec_max_bounds_viol_m':      'Max Bounds Violation (m)',
    'exec_max_halfspace_viol_m':   'Max Halfspace Violation (m)',
    'exec_max_obstacle_penetration_m': 'Max Obstacle Penetration (m)',
    'exec_constraint_margin_mean_m':   'Constraint Margin Mean (m)',
    'exec_first_violation_step':       'First Violation Step',
    'exec_longest_safe_streak':        'Longest Safe Streak',
    'exec_dynamics_consistency_error_mean': 'Dynamics Error Mean',
    'exec_dynamics_consistency_error_max':  'Dynamics Error Max',
    'plan_post_viol_rate_mean':    'Plan Post-Violation Rate Mean',
    'plan_post_viol_rate_max':     'Plan Post-Violation Rate Max',
    'plan_n_replan_steps':         'Plan N Replan Steps',
}
