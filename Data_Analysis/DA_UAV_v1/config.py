"""
DA_UAV_v1 — configuration and constants (Gen15 UAV Mix-ML).

Scope: the closed-loop UAV MuJoCo evaluation trees written by
`mix_uav_test/eval_mix_uav.py` (Gen15, `logs/UAV_MIX/…`) and — because discovery
is purely path-shape driven — the Gen11 UAV trees they descend from
(`FM_v3_uav_test/eval_fm_uav.py`, `logs/UAV_FM/…`), which share the artifact
schema byte-for-byte (`mix_uav_test/eval_artifacts.py` is a copy).

Built on the `DA_VA_v2` template, which is itself built on `DA_Code_v3`. Nothing
in either was modified; this is a third parallel tool.

Three things make UAV a different tool rather than a flag on DA_VA_v2:

  1. **No `results/` path level.** UAV writes `<seed>/<geo_tag>/<variant>/`
     directly (eval_mix_uav.py:1438-1444); the aligning/avoiding family writes
     `<seed>/results/…`. Discovery here accepts both.
  2. **Timing is not in the npz.** `eval_artifacts.save_npz` never persists the
     `timing` group, so `avg_time` — the metric Gen15 exists to measure — lives
     ONLY in `diagnostics/rollout_<r>_stats.json`. The diagnostics scan is
     therefore mandatory here, not an optional extra (see DIAGNOSTICS_REQUIRED).
  3. **K, engine and scene are first-class axes.** The K sweep IS the Gen15
     experiment (PLAN §7.3), and K/engine/controller/threshold are encoded in
     the eval-tag folder name while the scene is encoded in the dataset folder.
     They are parsed out into real columns instead of being left inside an
     opaque candidate name.

Nothing here is a hard requirement: the loader ingests every key it finds in the
npz generically. The lists below only drive ordering, labels and default filters.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────

# Optional per-seed results root. UAV does NOT write one (the geo_tag folder sits
# directly under the seed), but the level is accepted so a tree that grows one —
# or a state-only tree borrowed for comparison — still reads.
RESULTS_DIR_NAMES = ('results', 'results_train_set')

TRAIN_SET_SUFFIX = '_train_set'

# Directories that never hold variant data.
NON_VARIANT_DIRS = ('expert_references', 'all_seeds', 'diagnostics',
                    'config_snapshot', 'plots', 'logs', 'wandb')

# npz files that are never a finished result.
SKIP_NPZ_SUFFIXES = ('.partial.npz',)

# Legacy avoiding-family geo prefix, stripped for label continuity. UAV geo tags
# carry no prefix (`corridor_bounds+dynamics+geo_bounds+halfspace+obstacles`),
# so this is inert on a UAV tree and only matters if one is merged with an old one.
GEO_DIR_PREFIXES = ('halfspace_',)

# UAV tightening is a per-VARIANT modifier (`dpcc-c-tightened`), NOT a geometry
# twin as in visual-aligning — config/uav_projection.yaml enumerates each
# `-tightened` variant explicitly. So the suffix is read off the VARIANT name
# here, and `geo` never carries it.
TIGHTENED_SUFFIX = '-tightened'

# Sentinel for a unit with no geometry directory level at all.
GEO_NONE = 'none'

# `mix_uav/utils/setup.py::snapshot_configs` writes
# `<seed>/config_snapshot_<config>/` and drops a marker `snapshot_<YYYYMMDD_HHMMSS>`
# on every eval launch, never deleting the previous ones — so the newest marker is
# when that seed was last (re)generated. eval_mix_uav.py writes it at the
# eval-tag-aware seed dir (Fix_8), i.e. exactly at `<candidate>/<seed>/`.
SNAPSHOT_DIR_PREFIX = 'config_snapshot'

# Sentinel file the eval drops when the projection circuit breaker opened
# (eval_mix_uav.py Fix_15.3). Its presence is recorded in data_quality.csv.
CB_SENTINEL_NAME = 'PROJECTION_CB_TRIPPED.txt'

# Per-variant run summary. Carries the run-level `timing`, `projection_health`
# and `hardflow` blocks that have no per-rollout array.
RESULTS_JSON_NAME = 'results.json'


# ──────────────────────────────────────────────────────────────────────────────
# Path-encoded axes (the Gen15 experiment lives in the folder names)
# ──────────────────────────────────────────────────────────────────────────────

# Dataset folder → scene:  logs/UAV_MIX/uav-corridor/plans/…
SCENE_DIR_RE = r'^uav-(?P<scene>[a-zA-Z0-9_]+)$'

# Eval-tag folder (the CANDIDATE folder — it holds the seed subdirs):
#   E{engine}_K{flow_steps}_mpc{B}_{controller}_T{threshold}
# `_uav_eval_tag` in mix_uav_test/eval_mix_uav.py. The controller token can itself
# contain underscores (`pid_const_v`, `pid_stopgo`), so it is matched greedily
# between the fixed `mpc{n}_` head and the trailing `_T{thresh}`.
EVAL_TAG_RE = (r'^E(?P<engine>[A-Za-z0-9]+)_K(?P<K>\d+)_mpc(?P<mpc_batch>\d+)_'
               r'(?P<controller>.+)_T(?P<threshold>[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)$')

# Gen11 wrote the same folder WITHOUT the `E{engine}` token. Accepted so a Gen11
# tree can be read side by side with Gen15 (engine then comes from the model dir).
EVAL_TAG_RE_GEN11 = (r'^K(?P<K>\d+)_mpc(?P<mpc_batch>\d+)_'
                     r'(?P<controller>.+)_T(?P<threshold>[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)$')

# Model-identity folder: `mix_uav_<engine>/H{h}_D{diffusion}[_9D][_tokens]`.
# Tokens are registry-driven (`engine_registry.ENGINES[*]['exp_name_tokens']`):
#   dp{f} meanflow data proportion · bb{name} backbone · as/ae alpha schedule ·
#   K{n}  train-time K (the `diffusion` DDPM arm only — its beta schedule is
#         built from K, so it must live in the CHECKPOINT name)
MODEL_PREFIX_RE = r'^mix_uav_(?P<engine>[a-z]+)$'
MODEL_TOKEN_RES = {
    'horizon':        r'(?:^|_)H(?P<v>\d+)(?:_|$)',
    'diffusion_cls':  r'(?:^|_)D(?P<v>[A-Za-z0-9]+)(?:_|$)',
    'obs_dim_tag':    r'(?:^|_)(?P<v>\d+D)(?:_|$)',
    'data_proportion': r'(?:^|_)dp(?P<v>[0-9.]+)(?:_|$)',
    'backbone':       r'(?:^|_)bb(?P<v>[A-Za-z0-9_]+?)(?:_|$)',
    'alpha_init':     r'(?:^|_)as(?P<v>[0-9.]+)(?:_|$)',
    'alpha_end':      r'(?:^|_)ae(?P<v>[0-9.]+)(?:_|$)',
    'train_K':        r'(?:^|_)K(?P<v>\d+)(?:_|$)',
}

# Engine key → readable label (mix_uav/models/engine_registry.py:ENGINES).
ENGINE_LABELS = {
    'fm':        'Flow Matching (Gen11 FMv3ODE)',
    'mf':        'MeanFlow (Gen3v6)',
    'af':        'alpha-Flow (Gen3v7)',
    'diffusion': 'DDPM / DPCC baseline (GaussianDiffusion)',
}

# Ordering for tables/plots. Unknown engines are appended alphabetically.
ENGINE_ORDER = ['diffusion', 'fm', 'mf', 'af']

# mix_uav_test/eval_mix_uav.py:SCENES. `empty` has a RANDOM per-episode
# start→goal the state-only policy is never told, so goal-reaching is ill-defined
# there and its success is stable/safe flight only — never read its `goal_*`
# columns as a policy failure.
SCENES = ['empty', 'corridor', 's_curve', 'pillars']
GOAL_PATH_SCENES = ('corridor', 's_curve', 'pillars')


# ──────────────────────────────────────────────────────────────────────────────
# npz keys
# ──────────────────────────────────────────────────────────────────────────────

# Raw traces. `sampled_trajectories_all` is the MPC candidate fan for every step
# of every rollout and dominates the file; np.load is lazy per key, so never
# touching these means they are not decompressed at all.
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
# rates. `save_npz` already writes these as float, but a hand-built or Gen11 npz
# may carry bool.
BOOLEAN_METRICS = frozenset({
    'n_success', 'success_strict', 'success_relaxed',
    'success_strict_and_constraints', 'success_relaxed_and_constraints',
    'n_success_and_constraints', 'n_success_relaxed_and_constraints',
    'collision_free_completed', 'constraint_collision_free',
    'phys_safe', 'goal_reached', 'goal_crossed_line',
    'projection_cb_tripped',
})

# npz key → canonical DA name. The npz uses the Fix_10 group-prefixed schema
# (`phys_*`, `constraint_*`, `goal_*`, `success_*`); the canonical names below are
# the ones DA_Code_v3 / DA_VA_v2 / both HTML viewers already speak, so a UAV batch
# opens in them unchanged. Both spellings survive into per_rollout_detail.csv —
# the rename ADDS a column, it does not drop the original.
NPZ_RENAMES = {
    'success_strict':                   'n_success',
    'success_strict_and_constraints':   'n_success_and_constraints',
    'success_relaxed_and_constraints':  'n_success_relaxed_and_constraints',
    'constraint_collision_free':        'collision_free_completed',
    'constraint_n_violations':          'n_violations',
    'constraint_total_violations':      'total_violations',
}


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics-JSON fields (rollout_<r>_stats.json)
# ──────────────────────────────────────────────────────────────────────────────

# ⚠️ TIMING LIVES ONLY HERE. `eval_artifacts.save_npz` persists success /
# physical / constraint / goal / projection-health, but NOT the `timing` group —
# so `avg_time`, the axis the whole Gen15 K sweep is about, is unobtainable from
# the npz. Never pass --no-diagnostics-scan on a run whose timing you intend to
# read; the loader warns when timing ends up all-NaN.
DIAGNOSTICS_REQUIRED = True

# canonical name -> path inside the per-rollout JSON (nested Fix_10 schema first,
# flat pre-Fix_10 spelling second where one existed).
JSON_FIELD_PATHS = {
    'fm_ms':               (['timing', 'fm_ms_mean'], ['fm_ms_mean']),
    'fm_ms_p95':           (['timing', 'fm_ms_p95'], ['fm_ms_p95']),
    'proj_ms':             (['timing', 'proj_ms_mean'], ['proj_ms_mean']),
    'avg_time_ms':         (['timing', 'total_ms_mean'], ['total_ms_mean']),
    'total_ms_p95':        (['timing', 'total_ms_p95'], ['total_ms_p95']),
    'over_budget_steps':   (['timing', 'total_over_budget'], ['total_over_budget']),
    'budget_ms':           (['timing', 'budget_ms'], ['budget_ms']),
    'track_err_mean':      (['track_err_mean'],),
    'n_fm_steps':          (['n_fm_steps'],),
    'max_episode_length':  (['max_episode_length'],),
    'projection_cb_trips':      (['projection_health', 'cb_trips'],),
    'projection_backstop_hits': (['projection_health', 'backstop_hits'],),
    'n_proj_steps':             (['projection_health', 'n_proj_steps'],),
    # Present in the JSON and the npz both; read here so a JSON-only unit
    # (variant killed before its final npz write) is still complete.
    'phys_min_z':          (['physical', 'min_z'],),
    'phys_final_z':        (['physical', 'final_z'],),
    'phys_contact_frac':   (['physical', 'contact_frac'],),
    'goal_dist':           (['goal', 'dist'],),
}

# Boolean JSON fields, same lookup, cast to 0/1.
JSON_BOOL_PATHS = {
    'n_success':                        (['success', 'strict'], ['success']),
    'success_relaxed':                  (['success', 'relaxed'],),
    'n_success_and_constraints':        (['success', 'strict_and_constraints'],),
    'n_success_relaxed_and_constraints': (['success', 'relaxed_and_constraints'],),
    'phys_safe':                        (['physical', 'safe'],),
    'collision_free_completed':         (['constraint', 'collision_free'],),
    'goal_reached':                     (['goal', 'reached'],),
    'goal_crossed_line':                (['goal', 'crossed_line'],),
    'projection_cb_tripped':            (['projection_health', 'cb_tripped'],),
}

# Numeric JSON fields that are per-rollout counts, not rates.
JSON_COUNT_PATHS = {
    'n_violations':               (['constraint', 'n_violations'],),
    'total_violations':           (['constraint', 'total_violations'],),
    'projection_cb_skipped_steps': (['projection_health', 'cb_skipped_steps'],),
}

# Categorical per-rollout JSON fields. Not metrics — carried on the wide
# per-rollout table so a single rollout can be traced back to its route.
JSON_LABEL_PATHS = {
    'homotopy':        (['homotopy'],),
    'homotopy_flown':  (['homotopy_flown'],),
    'scene_json':      (['scene'],),
}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

# The seven DA_Code_v3 core metrics. Five exist verbatim (after NPZ_RENAMES),
# `avg_time` is derived from the JSON timing block, none is missing.
DA_V3_CORE_METRICS = [
    'n_success',
    'n_success_and_constraints',
    'n_steps',
    'n_violations',
    'total_violations',
    'avg_time',
    'collision_free_completed',
]

# Headline metrics: written first in the CSVs, plotted by default. Ordered as the
# read actually goes — did it succeed, was it safe, what did it cost.
PRIMARY_METRICS = [
    'n_success',
    'success_relaxed',
    'n_success_and_constraints',
    'n_success_relaxed_and_constraints',
    'collision_free_completed',
    'phys_safe',
    'goal_reached',
    'goal_dist',
    'n_violations',
    'total_violations',
    'n_steps',
    'steps_to_goal',
    'avg_time',
    'avg_time_ms',
    'fm_ms',
    'proj_ms',
    'total_ms_p95',
    'over_budget_frac',
    'track_err_mean',
    'phys_contact_frac',
    'phys_min_z',
]

# Ranking / Pareto defaults.
ACCURACY_METRIC = 'n_success_and_constraints'
ACCURACY_FALLBACK_METRIC = 'n_success'
TIME_METRIC = 'avg_time_ms'
TIME_FALLBACK_METRIC = 'avg_time'

# Data-quality columns — reported, never ranked on.
QUALITY_METRICS = [
    'projection_cb_tripped',
    'projection_cb_skipped_steps',
    'projection_cb_trips',
    'projection_backstop_hits',
]

METRIC_LABELS = {
    'n_success':                        'Success Rate (goal + safe, strict)',
    'success_strict':                   'Success Rate (goal + safe, strict)',
    'success_relaxed':                  'Success Rate (crossed finish line)',
    'n_success_and_constraints':        'Success + Constraint Rate (strict)',
    'n_success_relaxed_and_constraints': 'Success + Constraint Rate (relaxed)',
    'collision_free_completed':         'Zero-Violation Rollout Rate',
    'constraint_collision_free':        'Zero-Violation Rollout Rate (raw npz)',
    'n_violations':                     'Violated Steps per Rollout',
    'total_violations':                 'Cumulative Violation Magnitude',
    'phys_safe':                        'Physically Safe Rate (contact-free + airborne)',
    'phys_contact_frac':                'Contact Fraction of Episode',
    'phys_min_z':                       'Minimum Altitude (m)',
    'phys_final_z':                     'Final Altitude (m)',
    'goal_reached':                     'Goal-Reached Rate',
    'goal_dist':                        'Final Goal Distance (m)',
    'goal_crossed_line':                'Finish-Line Crossed Rate',
    'n_steps':                          'Steps per Episode',
    'steps_to_goal':                    'Steps to Goal (reaching episodes only)',
    'max_episode_length':               'Episode Step Budget',
    'avg_time':                         'Total Time / Replan (s)',
    'avg_time_ms':                      'Total Time / Replan (ms)',
    'fm_ms':                            'Generation Time / Replan (ms)',
    'fm_ms_p95':                        'Generation Time p95 (ms)',
    'proj_ms':                          'Projection Time / Replan (ms)',
    'total_ms_p95':                     'Total Time p95 (ms)',
    'over_budget_steps':                'Steps Over Real-Time Budget',
    'over_budget_frac':                 'Fraction of Steps Over Budget',
    'budget_ms':                        'Real-Time Budget (ms)',
    'track_err_mean':                   'Mean Tracking Error',
    'projection_cb_tripped':            'Projector Circuit-Breaker Tripped',
    'projection_cb_skipped_steps':      'Projector Skipped Steps',
    'projection_cb_trips':              'Projector Circuit-Breaker Trip Count',
    'projection_backstop_hits':         'Projector Cost-Backstop Hits',
    'n_proj_steps':                     'Projected Steps',
    # run-level scalars (results.json summary)
    'nfe_per_plan':                     'Network Evaluations per Plan (measured)',
    'nfe_effective':                    'Network Evaluations per Plan (effective)',
    'nlp_solves_total':                 'HardFlow NLP Solves (total)',
    'nlp_failures_total':               'HardFlow NLP Failures (total)',
    'activation_threshold':             'HardFlow Activation Threshold',
    'init_noise_scale':                 'ODE Init Noise Scale',
}

# Rendered as percentages by the reporter/visualizer.
PERCENTAGE_METRICS = frozenset({
    'n_success', 'success_strict', 'success_relaxed',
    'n_success_and_constraints', 'n_success_relaxed_and_constraints',
    'collision_free_completed', 'constraint_collision_free',
    'phys_safe', 'goal_reached', 'goal_crossed_line',
    'phys_contact_frac', 'over_budget_frac', 'projection_cb_tripped',
})

# Lower is better — used by the reporter/visualizer to orient a comparison.
LOWER_IS_BETTER = frozenset({
    'goal_dist', 'n_violations', 'total_violations', 'n_steps', 'steps_to_goal',
    'avg_time', 'avg_time_ms', 'fm_ms', 'proj_ms', 'total_ms_p95',
    'over_budget_steps', 'over_budget_frac', 'track_err_mean',
    'phys_contact_frac', 'projection_cb_tripped', 'projection_cb_skipped_steps',
    'projection_cb_trips', 'projection_backstop_hits',
    'nfe_per_plan', 'nfe_effective',
})


# ──────────────────────────────────────────────────────────────────────────────
# Rollout masks
# ──────────────────────────────────────────────────────────────────────────────

# The UAV analogue of DA_VA_v2's frozen-rollout mask.
#
# `all`        every rollout.
# `proj_valid` rollouts whose projection circuit breaker never opened.
#
# When the sustained-slowness breaker trips (mix_uav/sampling/projection.py
# Fix_15.2) the rollout ran — partly or wholly — on the UNPROJECTED trajectory.
# Its constraint numbers therefore describe a policy that was not the one the
# variant name claims, and pooling it inflates or deflates every constraint
# aggregate depending on which way the unprojected plan happened to fall. Both
# reductions are always written, so the toggle is a filter in the viewer rather
# than a re-run.
MASK_FLAG_COLUMN = 'projection_cb_tripped'
MASKS = ('all', 'proj_valid')


# ──────────────────────────────────────────────────────────────────────────────
# Variants
# ──────────────────────────────────────────────────────────────────────────────

# config/uav_projection.yaml `projection_variants` (20) + config/uav_mix.py
# `hardflow_variants` (3, Gen15 U2 — declared there and not in the shared yaml so
# a Gen11 job can never try to run a variant it has no code for).
# Discovery does not need this list; it only fixes row ordering.
VARIANT_ORDER = [
    'diffuser',
    'dpcc-r', 'dpcc-r-tightened',
    'dpcc-c', 'dpcc-c-tightened',
    'dpcc-t', 'dpcc-t-tightened',
    'hardflow_new', 'hardflow_new-c', 'hardflow_new-t',
    'gradient', 'gradient-tightened',
    'post_processing', 'post_processing-tightened',
    'model_free', 'model_free-tightened',
    'bounds_free', 'bounds_free-tightened',
    'geo_free',
    'geo_free-bounds_free',
    'geo_free-model_free',
    'model_free-bounds_free', 'model_free-bounds_free-tightened',
]

# Headline arms for the per-variant comparison table.
MAJOR_VARIANTS = [
    'diffuser',
    'dpcc-r', 'dpcc-c', 'dpcc-t',
    'dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened',
    'hardflow_new', 'hardflow_new-c', 'hardflow_new-t',
]

# Variants that run NO projector at all — their constraint columns describe the
# raw generator, which is the point of them, not a failed projection.
UNPROJECTED_VARIANTS = ('diffuser',)


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

# Output folder naming is NOT cosmetic. Each HTML viewer builds its run dropdown
# by regexing the `analysis_results/` directory listing for a leading prefix and
# falls back to results_manifest.json only when that listing fetch fails (it does
# not, under `python -m http.server`) — so a run whose folder name misses the
# prefix is invisible in the picker even though every CSV is present.
#   Visualizer_UAV_v1/index.html  matches  href="(batch_uav_[^/"]+)   ← this tool
#   Visualizer/index.html         matches  href="(batch_[^/"]+)       ← also matches
# `batch_uav_<timestamp>` therefore lands in both pickers with no symlink.
OUTPUT_FOLDER_PREFIX = 'batch_uav'
VIEWER_LIST_PREFIX = 'batch_uav_'

# Native CSV name stem. `uav_units_long.csv` / `uav_aggregated_long.csv`.
CSV_STEM = 'uav'

# `args` fields lifted into run_config.csv when present (best effort). The UAV
# eval pickles `vars(args)` from its own argparse namespace, which is much
# smaller than the aligning family's Parser namespace.
RUN_CONFIG_FIELDS = [
    'engine', 'flow_steps', 'scene', 'seed', 'n_trials', 'goal_radius',
    'max_episode_length', 'epoch', 'projection', 'record', 'device',
    'diffusion', 'horizon', 'n_diffusion_steps', 'batch_size', 'mpc_batch_size',
    'diffusion_timestep_threshold', 'controller', 'cond_mode',
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
