"""
Configuration and constants for the Data Analysis script.
"""
import os
from pathlib import Path

# Default seeds and variants (from projection_eval.yaml)
DEFAULT_SEEDS = [6, 7, 8, 9, 10]

# U7: Gen12 HardFlow arm-C variants (in-loop constrained sampling). Same npz schema
# as the DPCC arms, plus extra keys (nlp_solves / nfe / activation_threshold — loaded
# generically by data_loader). Suffix = DPCC-parity candidate selection (-c/-r/-t).
# U5: full DPCC-parity set — {-r,-c,-t} selection x {'', -tightened} geometry, mirroring
# DPCC's dpcc-{r,t,c}[-tightened]. Loaded generically; a candidate that lacks a variant is
# skipped. At mpc==1 the -r/-c/-t within each geometry are identical (selection is a no-op).
HARDFLOW_VARIANTS = [
    'hardflow_new',
    'hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t',
    'hardflow_new-r-tightened', 'hardflow_new-c-tightened', 'hardflow_new-t-tightened',
    # [SolverSwap 2026-08-27] The SLSQP-backend twins. A run on the new default writes
    # `hardflow_sls-*` so it lands BESIDE the `hardflow_new-*` (IPOPT) corpus instead of
    # overwriting it. Discovery here is an explicit allow-list, so an unregistered name is
    # INVISIBLE rather than an error — these lines are what make the swap runs loadable.
    # 🔴 Never pool `hardflow_new-*` with `hardflow_sls-*`: different solver.
    'hardflow_sls',
    'hardflow_sls-r', 'hardflow_sls-c', 'hardflow_sls-t',
    'hardflow_sls-r-tightened', 'hardflow_sls-c-tightened', 'hardflow_sls-t-tightened',
]

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
] + HARDFLOW_VARIANTS   # U7: make hardflow_new* discoverable/loadable

MAJOR_VARIANTS = [
    'dpcc-r', 'dpcc-r-tightened',
    'dpcc-c', 'dpcc-c-tightened',
    'dpcc-t', 'dpcc-t-tightened',
    # U7: headline arm-C lines, so hardflow_new sits next to dpcc-c-tightened in the
    # per-variant comparison table. The batch aggregator/visualizer skip a major that a
    # given candidate lacks, so DPCC-only candidates are unaffected.
    'hardflow_new', 'hardflow_new-c',
    'hardflow_new-c-tightened',   # U5: headline matched-margin arm vs dpcc-c-tightened
    # [SolverSwap] the same headline arms on the SLSQP backend.
    'hardflow_sls', 'hardflow_sls-c', 'hardflow_sls-c-tightened',
]

AUXILIARY_VARIANTS = [v for v in DEFAULT_PROJECTION_VARIANTS if v not in MAJOR_VARIANTS]

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

# U7: HardFlow-only scalar metrics (present in hardflow_new npz; absent -> NaN for DPCC).
# Kept OUT of core METRICS so existing DPCC tables/plots are unchanged; the data_loader
# reads them generically and the reporter surfaces them for hardflow rows only. Use these
# to answer "is the NLP the same?" — same activation_threshold + comparable nlp_solves.
HARDFLOW_METRICS = [
    'nlp_solves',
    'nlp_failures',
    'nfe',
    'activation_threshold',
    'batch_size',
    'flow_steps',
    # [SolverSwap 2026-08-27] WHICH NLP solver arm C used. The npz carries the
    # string as `nlp_backend` and this float twin (1.0 = DPCC scipy SLSQP,
    # 0.0 = the original CasADi IPOPT); generic loaders coerce npz scalars to
    # float, so the twin is the one that survives. Never pool rows that disagree.
    'nlp_backend_slsqp',
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
    # U7 HardFlow-only
    'nlp_solves': 'NLP Solves (per run, sum)',
    'nlp_failures': 'NLP Failures',
    'nfe': 'Network Fn Evals',
    'activation_threshold': 'Activation Threshold (DPCC polarity)',
    'nlp_backend_slsqp': 'NLP Backend (1 = DPCC SLSQP, 0 = IPOPT)',
    'batch_size': 'MPC Candidates',
    'flow_steps': 'Flow Steps (K)',
}

METRIC_TYPES = {
    'n_success': 'percentage',
    'n_success_and_constraints': 'percentage',
    'collision_free_completed': 'percentage',
    'n_steps': 'continuous',
    'avg_time': 'continuous',
    'n_violations': 'continuous',
    'total_violations': 'continuous',
    # U7 HardFlow-only
    'nlp_solves': 'continuous',
    'nlp_failures': 'continuous',
    'nfe': 'continuous',
    'activation_threshold': 'continuous',
    'nlp_backend_slsqp': 'continuous',
    'batch_size': 'continuous',
    'flow_steps': 'continuous',
}

# Output folder naming
OUTPUT_FOLDER_PREFIX = 'FM_V3_ODE_Analysis'
