"""
DA_VA_v2 — configuration and constants.

Scope: visual-aligning evaluation trees (Gen7 `fm_visual_aligning_test`, Gen14
`mix_visual_aligning_test`) *and*, because discovery is purely path-shape driven,
the state-only avoiding trees `DA_Code_v3` was built for.

Nothing here is a hard requirement: the loader ingests every key it finds in the
npz generically. The lists below only drive ordering, labels and default filters.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────

# Both spellings of the per-seed results root. `results_train_set` is what
# `--eval-on-train` writes (Gen14 spec §4 L1); the variant names inside it carry
# a `_train_set` suffix which the loader strips (L4).
RESULTS_DIR_NAMES = ('results', 'results_train_set')

TRAIN_SET_SUFFIX = '_train_set'

# Directories under a results root that never hold variant data.
NON_VARIANT_DIRS = ('expert_references', 'all_seeds', 'diagnostics',
                    'config_snapshot', 'plots', 'logs')

# npz files that are never a finished result.
SKIP_NPZ_SUFFIXES = ('.partial.npz',)

# Legacy geo-directory prefix used by the state-only avoiding pipeline
# (`results/halfspace_top-right-hard/`). Stripped so the `geo` axis reads the
# same as it did in DA_Code_v3's `halfspace_variant` column.
GEO_DIR_PREFIXES = ('halfspace_',)

# Geometry directories ending in this are the auto-generated tightened twin.
TIGHTENED_SUFFIX = '-tightened'

# Sentinel for runs with no geometry directory level at all (old flat Gen7 runs,
# `results/{variant}/{variant}.npz`).
GEO_NONE = 'none'


# ──────────────────────────────────────────────────────────────────────────────
# npz keys
# ──────────────────────────────────────────────────────────────────────────────

# Raw traces. Huge (sampled_trajectories_all is ~90% of the file) and never
# needed for CSV work, so they are NEVER touched — np.load is lazy per key, so
# skipping them means they are not even decompressed. Spec §6 item 5.
HEAVY_KEYS = frozenset({
    'obs_all',
    'act_all',
    'sampled_trajectories_all',
    'selected_idx_all',
    'physical_tracking_errors',
})

# Loaded but kept out of the metric tables (config / bookkeeping).
META_KEYS = frozenset({'args', 'complete', 'seed', 'n_rollouts_done'})

# Per-rollout arrays whose values are 0/1 flags — cast to float so means read as
# rates. (npz stores several of these as bool.)
BOOLEAN_METRICS = frozenset({
    'n_success', 'success_strict', 'success_relaxed',
    'constraint_exec_zero_violation', 'collision_free_completed',
    'n_success_and_constraints', 'projection_cb_tripped', 'frozen',
})


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

# The seven DA_Code_v3 core metrics. Three exist verbatim in the visual npz,
# three are derived (see data_loader._derive_metrics), one has no equivalent.
DA_V3_CORE_METRICS = [
    'n_success',
    'n_success_and_constraints',
    'n_steps',
    'n_violations',
    'total_violations',
    'avg_time',
    'collision_free_completed',
]

# Headline metrics: written first in the CSVs, plotted by default.
PRIMARY_METRICS = [
    'n_success',
    'success_relaxed',
    'n_success_and_constraints',
    'collision_free_completed',
    'constraint_exec_sat_rate',
    'n_violations',
    'mean_dist_per_rollout',
    'max_phys_error_per_rollout',
    'n_steps',
    'avg_time',
    'avg_time_ms',
]

# Ranking / Pareto defaults.
ACCURACY_METRIC = 'n_success_and_constraints'
ACCURACY_FALLBACK_METRIC = 'n_success'
TIME_METRIC = 'avg_time_ms'
TIME_FALLBACK_METRIC = 'avg_time'

# Data-quality columns — reported, never ranked on. Spec §5.3.
QUALITY_METRICS = [
    'frozen',
    'projection_cb_tripped',
    'projection_cb_skipped_steps',
]

METRIC_LABELS = {
    'n_success':                        'Goal Success Rate (strict)',
    'success_strict':                   'Goal Success Rate (strict)',
    'success_relaxed':                  'Goal Success Rate (relaxed)',
    'n_success_and_constraints':        'Goal + Constraint Success Rate',
    'collision_free_completed':         'Zero-Violation Rollout Rate',
    'n_violations':                     'Violated Steps per Rollout',
    'total_violations':                 'Cumulative Violation (state-only runs)',
    'constraint_exec_total_viol_count': 'Violation Count (all families)',
    'max_viol_depth_m':                 'Worst Violation Depth (m)',
    'n_steps':                          'Steps per Rollout',
    'avg_time':                         'Inference Time / Replan (s)',
    'avg_time_ms':                      'Inference Time / Replan (ms)',
    'mean_dist_per_rollout':            'Final Box-Target Distance (m)',
    'mean_distance':                    'Final Box-Target Distance (m)',
    'max_phys_error_per_rollout':       'Max Physical Tracking Error (m)',
    'context_init_xy_dist':             'Init Box-Target Distance (m)',
    'context_final_xy_dist':            'Final Box-Target Distance, JSON (m)',
    'context_final_box_angle_deg':      'Final Box Angle (deg)',
    'constraint_exec_sat_rate':         'Constraint Satisfaction Rate',
    'constraint_exec_n_violated_steps': 'Violated Steps Count',
    'constraint_exec_margin_mean_m':    'Constraint Margin Mean (m)',
    'constraint_exec_max_bounds_viol_m':          'Max Bounds Violation (m)',
    'constraint_exec_max_halfspace_viol_m':       'Max Halfspace Violation (m)',
    'constraint_exec_max_obstacle_penetration_m': 'Max Obstacle Penetration (m)',
    'constraint_exec_first_violation_step':  'First Violation Step',
    'constraint_exec_longest_safe_streak':   'Longest Safe Streak',
    'constraint_exec_dyn_err_mean':          'Dynamics Consistency Error (mean)',
    'constraint_exec_dyn_err_max':           'Dynamics Consistency Error (max)',
    'constraint_plan_post_viol_rate_mean':   'Planned Post-Projection Viol Rate (mean)',
    'constraint_plan_post_viol_rate_max':    'Planned Post-Projection Viol Rate (max)',
    'constraint_plan_n_replan_steps':        'Replan Steps',
    'projection_cb_tripped':            'Projector Circuit-Breaker Tripped',
    'projection_cb_skipped_steps':      'Projector Skipped Steps',
    'frozen':                           'D1-Frozen Rollout Rate',
    'contact_first_step':               'First Contact Step',
    'contact_last_step':                'Last Contact Step',
}

# Rendered as percentages by the reporter/visualizer.
PERCENTAGE_METRICS = frozenset({
    'n_success', 'success_strict', 'success_relaxed',
    'n_success_and_constraints', 'collision_free_completed',
    'constraint_exec_sat_rate', 'frozen', 'projection_cb_tripped',
})


# ──────────────────────────────────────────────────────────────────────────────
# Variants
# ──────────────────────────────────────────────────────────────────────────────

# The 19+3 visual-aligning variants from config/visual_aligning_eval.yaml.
# Discovery does not need this list (it scans the filesystem); it only fixes the
# ordering of rows in the CSVs and tables.
VARIANT_ORDER = [
    'diffuser',
    'dpcc-r', 'dpcc-c', 'dpcc-t',
    'dpcc-c-dt0p25', 'dpcc-c-dt0p5', 'dpcc-c-dt2p0', 'dpcc-c-dt4p0',
    'hardflow_new', 'hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t',
    'gradient',
    'post_processing',
    'model_free',
    'geo_free',
    'bounds_free',
    'geo_free-bounds_free',
    'geo_free-model_free',
    'model_free-bounds_free',
    # state-only avoiding names (this tool can read those trees too)
    'dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened',
    'gradient-tightened', 'post_processing-tightened', 'model_free-tightened',
]

# Headline arms for the per-variant comparison table.
MAJOR_VARIANTS = [
    'dpcc-r', 'dpcc-c', 'dpcc-t',
    'hardflow_new-c', 'hardflow_new-r', 'hardflow_new-t',
    'diffuser',
]


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

# Output folder naming is NOT cosmetic. An HTML viewer builds its run dropdown by
# regexing the `analysis_results/` directory listing for a leading prefix, and
# falls back to results_manifest.json only when that listing fetch fails (it does
# not, under `python -m http.server`) — so a run whose folder name misses the
# prefix is invisible in the picker even though every CSV is present.
#   Visualizer_VA_v2/index.html  matches  href="(batch_va2_[^/"]+)   ← this tool
#   Visualizer/index.html        matches  href="(batch_[^/"]+)       ← also matches
# `batch_va2_<timestamp>` therefore lands in both pickers on its own, with no
# symlink or other accommodation.
OUTPUT_FOLDER_PREFIX = 'batch_va2'
VIEWER_LIST_PREFIX = 'batch_va2_'

# `args` fields lifted into run_config.csv when present (best effort).
RUN_CONFIG_FIELDS = [
    'engine', 'diffusion', 'horizon', 'n_diffusion_steps', 'flow_steps',
    'ode_method', 'diffusion_timestep_threshold', 'batch_size', 'mpc_batch_size',
    'if_vision', 'film_version', 'n_contexts', 'seed', 'projection_variant',
    'activation_threshold', 'device', 'savepath', 'diffusion_loadpath',
]

PLOT_CONFIG = {
    'figsize': (12, 7),
    'dpi': 200,
    'style': 'seaborn-v0_8-darkgrid',
    'font_size': 11,
    'title_size': 13,
    'legend_size': 9,
    'colors': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbbd22', '#9edae5',
    ],
}
