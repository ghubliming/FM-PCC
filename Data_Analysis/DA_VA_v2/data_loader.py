"""
DA_VA_v2 — per-unit data loading.

One "unit" = one (candidate, seed, split, geo, variant) result, i.e. exactly one
npz. The loader turns it into:

  per_rollout : DataFrame, one row per rollout, one column per per-rollout metric
  scalars     : dict of run-level scalars (success_rate, entropy, elapsed_seconds,
                and any engine-specific extras such as HardFlow's nlp_solves)
  quality     : dict of data-quality counters (frozen rollouts, circuit breaker,
                whether the npz claims to be complete)
  run_config  : dict lifted from the npz `args` object

Ingestion is generic — every key in the npz is picked up and classified by shape,
so extra keys future evals add cost nothing here. The heavy raw traces
(`sampled_trajectories_all` & friends) are never read: np.load is lazy per key,
so skipping them means they are not decompressed at all.

Sources:
  npz  — the single source of truth written at variant end
  json — `diagnostics/rollout_*_stats.json`, for pre-npz Gen7 runs or a variant
         that died before its final write
  auto — npz when present, else json (default)
"""

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

from config import BOOLEAN_METRICS, HEAVY_KEYS, META_KEYS, RUN_CONFIG_FIELDS

logger = logging.getLogger(__name__)


class UnitLoader:
    """Load one result unit into tabular form."""

    def __init__(self, source='auto', scan_diagnostics=True, verbose=False):
        self.source = source
        self.scan_diagnostics = scan_diagnostics
        self.verbose = verbose
        self.log = []          # (level, message)
        self.n_loaded = 0
        self.n_failed = 0
        self.n_skipped = 0

    # ── public ────────────────────────────────────────────────────────────────
    def load(self, unit):
        """Return a dict with per_rollout / scalars / quality / run_config, or None."""
        use_json = (self.source == 'json'
                    or (self.source == 'auto' and not unit['npz_path'])
                    or not unit['npz_path'])

        try:
            if use_json:
                loaded = self._load_json(unit)
                source_used = 'json'
            else:
                loaded = self._load_npz(unit)
                source_used = 'npz'
        except Exception as exc:                                  # noqa: BLE001
            self._note('ERROR', f'{_uid(unit)}: FAILED — {exc}')
            self.n_failed += 1
            return None

        if loaded is None:
            self.n_skipped += 1
            return None

        per_rollout, scalars, run_config, quality = loaded

        # Diagnostics JSONs carry three things the npz does not: the D1
        # box-obstacle conflict flag (frozen rollouts, spec §5.3) and the final
        # box pose. Cheap enough to always read, skippable for speed.
        if self.scan_diagnostics and source_used == 'npz':
            extras = self._scan_diagnostics(unit, len(per_rollout))
            for name, values in extras.items():
                if name not in per_rollout.columns:
                    per_rollout[name] = values

        per_rollout = _finalise_frame(per_rollout)
        quality.update(_quality_from_frame(per_rollout))
        quality['source'] = source_used
        quality['n_rollouts'] = int(len(per_rollout))

        self.n_loaded += 1
        if self.verbose:
            self._note('OK', f'{_uid(unit)}: {len(per_rollout)} rollouts, '
                             f'{len(per_rollout.columns)} metrics ({source_used})')
        return {
            'per_rollout': per_rollout,
            'scalars': scalars,
            'run_config': run_config,
            'quality': quality,
        }

    def summary(self):
        return {
            'files_loaded': self.n_loaded,
            'files_failed': self.n_failed,
            'files_skipped': self.n_skipped,
            'log': self.log,
        }

    def save_log(self, path):
        with open(path, 'w') as f:
            f.write('=== DA_VA_v2 loading log ===\n\n')
            f.write(f'Loaded:  {self.n_loaded}\n')
            f.write(f'Failed:  {self.n_failed}\n')
            f.write(f'Skipped: {self.n_skipped}\n\n')
            for level, msg in self.log:
                f.write(f'[{level:7s}] {msg}\n')
        logger.info(f'Loading log written: {path}')

    # ── npz ───────────────────────────────────────────────────────────────────
    def _load_npz(self, unit):
        path = unit['npz_path']
        if not os.path.exists(path):
            self._note('MISSING', f'{_uid(unit)}: npz not found at {path}')
            return None

        columns = {}
        scalars = {}
        run_config = {}
        quality = {'npz_complete': np.nan}

        with np.load(path, allow_pickle=True) as data:
            keys = [k for k in data.files if k not in HEAVY_KEYS]
            n_rollouts = _infer_n_rollouts(data, keys)
            if n_rollouts == 0:
                self._note('EMPTY', f'{_uid(unit)}: no per-rollout arrays in npz')
                return None

            for key in keys:
                try:
                    value = data[key]
                except Exception as exc:                          # noqa: BLE001
                    self._note('WARN', f'{_uid(unit)}: key {key} unreadable — {exc}')
                    continue

                if key == 'args':
                    run_config = _extract_run_config(value)
                    continue
                if key == 'complete':
                    quality['npz_complete'] = _as_float(value)
                    continue
                if key in META_KEYS:
                    continue

                kind, payload = _classify(key, value, n_rollouts)
                if kind == 'column':
                    columns[key] = payload
                elif kind == 'columns':
                    columns.update(payload)
                elif kind == 'scalar':
                    scalars[key] = payload
                else:
                    self._note('SKIP', f'{_uid(unit)}: key {key} shape '
                                       f'{getattr(value, "shape", "?")} not tabular')

        frame = pd.DataFrame(columns)
        frame.insert(0, 'rollout_idx', np.arange(len(frame), dtype=int))
        return frame, scalars, run_config, quality

    # ── json ──────────────────────────────────────────────────────────────────
    def _load_json(self, unit):
        diag_dir = unit['diagnostics_dir']
        if not os.path.isdir(diag_dir):
            self._note('MISSING', f'{_uid(unit)}: no npz and no diagnostics dir')
            return None

        rows = _read_rollout_stats(diag_dir)
        if not rows:
            self._note('MISSING', f'{_uid(unit)}: diagnostics dir holds no rollout stats')
            return None

        # Two rollout-JSON schemas exist and both are still on disk:
        #   nested (Gen14 / late Gen7): success.strict, outcome.*, timing.*,
        #                               context.*, constraint.exec.*
        #   flat   (early Gen7):        success, mean_distance, steps,
        #                               context_info.*, constraint_metrics.exec_*
        # Every field is looked up along both, first hit wins, so a mixed tree
        # loads without a flag. `_PICK` below is (canonical, [path, path, ...]).
        def get(*paths, default=np.nan):
            return np.array([_pick(r, paths, default) for r in rows], dtype=float)

        columns = {
            'rollout_idx': np.array([_dig(r, ['rollout_index'], i)
                                     for i, r in enumerate(rows)], dtype=int),
            'mode_encoding': get(['mode']),
            'n_success': get(['success', 'strict'], ['success']),
            'success_strict': get(['success', 'strict'], ['success']),
            'success_relaxed': get(['success', 'relaxed']),
            'mean_dist_per_rollout': get(['outcome', 'mean_distance'], ['mean_distance']),
            'max_phys_error_per_rollout': get(['outcome', 'max_physical_tracking_error'],
                                              ['max_physical_tracking_error']),
            'n_steps': get(['timing', 'steps'], ['steps']),
            'avg_time': get(['timing', 'avg_inference_time_per_replan'],
                            ['avg_inference_time_per_replan']),
            'context_init_xy_dist': _ctx(rows, 'init_xy_dist'),
            'context_box_angle_deg': _ctx(rows, 'box_init_angle_deg'),
            'context_target_angle_deg': _ctx(rows, 'target_angle_deg'),
            'context_final_xy_dist': _ctx(rows, 'final_xy_dist'),
            'context_final_box_angle_deg': _ctx(rows, 'final_box_angle_deg'),
            'contact_first_step': get(['contact', 'first_step']),
            'contact_last_step': get(['contact', 'last_step']),
            'constraint_exec_n_violated_steps': _cm(rows, ['exec', 'n_violated_steps'],
                                                    'exec_n_violated_steps'),
            'constraint_exec_sat_rate': _cm(rows, ['exec', 'constraint_sat_rate'],
                                            'exec_constraint_sat_rate'),
            'constraint_exec_zero_violation': _cm(rows, ['exec', 'zero_violation_rollout'],
                                                  'exec_zero_violation_rollout'),
            'constraint_exec_bounds_viol_count': _cm(rows, ['exec', 'by_family', 'bounds', 'viol_count'],
                                                     'exec_bounds_viol_count'),
            'constraint_exec_halfspace_viol_count': _cm(rows, ['exec', 'by_family', 'halfspace', 'viol_count'],
                                                        'exec_halfspace_viol_count'),
            'constraint_exec_obstacle_viol_count': _cm(rows, ['exec', 'by_family', 'obstacles', 'viol_count'],
                                                       'exec_obstacle_viol_count'),
            'constraint_exec_max_bounds_viol_m': _cm(rows, ['exec', 'by_family', 'bounds', 'max_viol_m'],
                                                     'exec_max_bounds_viol_m'),
            'constraint_exec_max_halfspace_viol_m': _cm(rows, ['exec', 'by_family', 'halfspace', 'max_viol_m'],
                                                        'exec_max_halfspace_viol_m'),
            'constraint_exec_max_obstacle_penetration_m': _cm(rows, ['exec', 'by_family', 'obstacles', 'max_viol_m'],
                                                              'exec_max_obstacle_penetration_m'),
            'constraint_exec_margin_mean_m': _cm(rows, ['exec', 'margin_mean_m'],
                                                 'exec_constraint_margin_mean_m'),
            'constraint_exec_first_violation_step': _cm(rows, ['exec', 'first_violation_step'],
                                                        'exec_first_violation_step'),
            'constraint_exec_longest_safe_streak': _cm(rows, ['exec', 'longest_safe_streak'],
                                                       'exec_longest_safe_streak'),
            'constraint_exec_dyn_err_mean': _cm(rows, ['exec', 'dynamics_consistency_error', 'mean'],
                                                'exec_dynamics_consistency_error_mean'),
            'constraint_exec_dyn_err_max': _cm(rows, ['exec', 'dynamics_consistency_error', 'max'],
                                               'exec_dynamics_consistency_error_max'),
            'constraint_plan_post_viol_rate_mean': _cm(rows, ['plan', 'post_viol_rate_mean'],
                                                       'plan_post_viol_rate_mean'),
            'constraint_plan_post_viol_rate_max': _cm(rows, ['plan', 'post_viol_rate_max'],
                                                      'plan_post_viol_rate_max'),
            'constraint_plan_n_replan_steps': _cm(rows, ['plan', 'n_replan_steps'],
                                                  'plan_n_replan_steps'),
            'frozen': np.array([1.0 if _conflict(r) else 0.0 for r in rows]),
            'frozen_worst_overlap_m': np.array(
                [_as_float(_dig(_conflict(r) or {}, ['worst_overlap_m'], np.nan))
                 for r in rows], dtype=float),
        }
        for axis, idx in (('x', 0), ('y', 1)):
            for name, field in (('context_box_init_xy', 'box_init_xy'),
                                ('context_target_xy', 'target_xy'),
                                ('context_final_box_xy', 'final_box_xy')):
                columns[f'{name}_{axis}'] = np.array(
                    [_seq_item(_pick(r, ([CONTEXT_NEW, field], [CONTEXT_OLD, field]), None), idx)
                     for r in rows], dtype=float)

        frame = pd.DataFrame(columns)
        scalars = {'success_rate': _safe_nanmean(frame['n_success'])}
        quality = {'npz_complete': np.nan, 'diagnostics_found': 1.0}

        if np.isnan(scalars['success_rate']):
            self._note('WARN', f'{_uid(unit)}: no success field found in the rollout '
                               f'JSONs — unrecognised schema, metrics will be NaN')
        return frame, scalars, {}, quality

    # ── diagnostics side-scan (npz source) ───────────────────────────────────
    def _scan_diagnostics(self, unit, n_rollouts):
        """Harvest the fields that only the per-rollout JSONs carry.

        Chief among them `context.box_obstacle_conflict` — the D1 guard flag that
        marks a rollout the eval froze in place without ever calling the model
        (spec §5.3). Those rollouts report sat_rate 1.0 / zero_violation 1 and
        would otherwise silently inflate every constraint aggregate.
        """
        out = {}
        diag_dir = unit['diagnostics_dir']
        if not os.path.isdir(diag_dir):
            self._note('WARN', f'{_uid(unit)}: no diagnostics dir — frozen-rollout '
                               f'mask unavailable, treated as unfrozen')
            out['frozen'] = np.zeros(n_rollouts)
            out['diagnostics_found'] = np.zeros(n_rollouts)
            return out

        rows = _read_rollout_stats(diag_dir)
        by_index = {}
        for i, row in enumerate(rows):
            by_index[int(_dig(row, ['rollout_index'], i) or i)] = row

        frozen, overlap, final_dist, final_angle, found = [], [], [], [], []
        for i in range(n_rollouts):
            row = by_index.get(i)
            if row is None:
                frozen.append(0.0)
                overlap.append(np.nan)
                final_dist.append(np.nan)
                final_angle.append(np.nan)
                found.append(0.0)
                continue
            conflict = _conflict(row)   # nested `context` or flat `context_info`
            frozen.append(1.0 if conflict else 0.0)
            overlap.append(_as_float(_dig(conflict or {}, ['worst_overlap_m'], np.nan)))
            final_dist.append(_as_float(_pick(
                row, ([CONTEXT_NEW, 'final_xy_dist'], [CONTEXT_OLD, 'final_xy_dist']), np.nan)))
            final_angle.append(_as_float(_pick(
                row, ([CONTEXT_NEW, 'final_box_angle_deg'],
                      [CONTEXT_OLD, 'final_box_angle_deg']), np.nan)))
            found.append(1.0)

        out['frozen'] = np.array(frozen)
        out['frozen_worst_overlap_m'] = np.array(overlap)
        out['context_final_xy_dist'] = np.array(final_dist)
        out['context_final_box_angle_deg'] = np.array(final_angle)
        out['diagnostics_found'] = np.array(found)
        return out

    # ── misc ──────────────────────────────────────────────────────────────────
    def _note(self, level, message):
        self.log.append((level, message))
        if level in ('ERROR',):
            logger.error(message)
        elif level in ('MISSING', 'WARN', 'EMPTY'):
            logger.debug(message)


# ──────────────────────────────────────────────────────────────────────────────
# derived metrics
# ──────────────────────────────────────────────────────────────────────────────

def _finalise_frame(frame):
    """Add the derived columns, then normalise dtypes/order.

    The DA_Code_v3 core metric names are reconstructed from the visual npz keys
    (spec §5.2):

      collision_free_completed   ← constraint_exec_zero_violation
      n_violations               ← constraint_exec_n_violated_steps
      n_success_and_constraints  ← n_success * constraint_exec_zero_violation

    `total_violations` has no equivalent in the visual pipeline — that eval never
    accumulates a running violation sum, it records per-family maxima. It is left
    as NaN where absent (state-only npz files still supply it), and two honest
    stand-ins are provided instead: `constraint_exec_total_viol_count` (sum of the
    three family step-counts) and `max_viol_depth_m` (worst depth over families).
    """
    if frame.empty:
        return frame

    def col(name):
        return frame[name] if name in frame.columns else None

    zero_viol = col('constraint_exec_zero_violation')
    n_success = col('n_success')
    if n_success is None:
        n_success = col('success_strict')
        if n_success is not None:
            frame['n_success'] = n_success

    if 'collision_free_completed' not in frame.columns and zero_viol is not None:
        frame['collision_free_completed'] = zero_viol.astype(float)

    if 'n_violations' not in frame.columns and col('constraint_exec_n_violated_steps') is not None:
        frame['n_violations'] = frame['constraint_exec_n_violated_steps'].astype(float)

    if 'n_success_and_constraints' not in frame.columns:
        if n_success is not None and zero_viol is not None:
            frame['n_success_and_constraints'] = (n_success.astype(float)
                                                  * zero_viol.astype(float))
        elif n_success is not None:
            frame['n_success_and_constraints'] = np.nan

    if 'total_violations' not in frame.columns:
        frame['total_violations'] = np.nan

    family_counts = [c for c in ('constraint_exec_bounds_viol_count',
                                 'constraint_exec_halfspace_viol_count',
                                 'constraint_exec_obstacle_viol_count')
                     if c in frame.columns]
    if family_counts and 'constraint_exec_total_viol_count' not in frame.columns:
        frame['constraint_exec_total_viol_count'] = frame[family_counts].sum(axis=1)

    family_depths = [c for c in ('constraint_exec_max_bounds_viol_m',
                                 'constraint_exec_max_halfspace_viol_m',
                                 'constraint_exec_max_obstacle_penetration_m')
                     if c in frame.columns]
    if family_depths and 'max_viol_depth_m' not in frame.columns:
        frame['max_viol_depth_m'] = frame[family_depths].max(axis=1)

    # `avg_time` is seconds per replan in both the visual and state-only npz;
    # DA_Code_v3 reported milliseconds, so both units are offered.
    if 'avg_time' in frame.columns and 'avg_time_ms' not in frame.columns:
        frame['avg_time_ms'] = frame['avg_time'].astype(float) * 1000.0

    if 'frozen' not in frame.columns:
        frame['frozen'] = 0.0

    for name in frame.columns:
        if name == 'rollout_idx':
            continue
        if name in BOOLEAN_METRICS or frame[name].dtype == bool:
            frame[name] = frame[name].astype(float)

    ordered = ['rollout_idx'] + sorted(c for c in frame.columns if c != 'rollout_idx')
    return frame[ordered]


def _quality_from_frame(frame):
    """Per-unit data-quality counters. Spec §5.3 — reported, never ranked on."""
    out = {}
    n = len(frame)
    out['n_frozen'] = int(frame['frozen'].sum()) if 'frozen' in frame.columns else 0
    out['frozen_rate'] = (out['n_frozen'] / n) if n else np.nan
    if 'projection_cb_tripped' in frame.columns:
        out['n_cb_tripped'] = int(frame['projection_cb_tripped'].sum())
    else:
        out['n_cb_tripped'] = 0
    if 'projection_cb_skipped_steps' in frame.columns:
        out['cb_skipped_steps'] = float(frame['projection_cb_skipped_steps'].sum())
    else:
        out['cb_skipped_steps'] = 0.0
    if 'diagnostics_found' in frame.columns:
        out['n_diagnostics_json'] = int(frame['diagnostics_found'].sum())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# npz value classification
# ──────────────────────────────────────────────────────────────────────────────

def _classify(key, value, n_rollouts):
    """Sort one npz value into a per-rollout column, several columns, or a scalar."""
    array = np.asarray(value)

    if array.dtype == object:
        return 'skip', None

    if array.ndim == 0 or array.size == 1:
        return 'scalar', _as_float(array)

    if array.ndim == 1:
        if array.shape[0] == n_rollouts:
            return 'column', array.astype(float)
        return 'skip', None

    if array.ndim == 2 and array.shape[0] == n_rollouts:
        if array.shape[1] == 1:
            return 'column', array[:, 0].astype(float)
        if array.shape[1] <= 4:
            suffixes = ['x', 'y', 'z', 'w'][: array.shape[1]]
            return 'columns', {f'{key}_{s}': array[:, i].astype(float)
                               for i, s in enumerate(suffixes)}
    return 'skip', None


def _infer_n_rollouts(data, keys):
    """Rollout count, from the outcome arrays first and any 1-D array as fallback."""
    for key in ('n_success', 'success_strict', 'n_steps', 'mean_dist_per_rollout'):
        if key in keys:
            array = np.asarray(data[key])
            if array.ndim >= 1 and array.size > 1:
                return int(array.shape[0])
    best = 0
    for key in keys:
        if key in META_KEYS:
            continue
        try:
            array = np.asarray(data[key])
        except Exception:                                         # noqa: BLE001
            continue
        if array.dtype != object and array.ndim >= 1 and array.size > 1:
            best = max(best, int(array.shape[0]))
    return best


def _extract_run_config(value):
    """Pull the interesting fields out of the pickled `args` namespace."""
    try:
        raw = value.item() if isinstance(value, np.ndarray) else value
    except Exception:                                             # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        raw = getattr(raw, '__dict__', None)
        if not isinstance(raw, dict):
            return {}
    out = {}
    for field in RUN_CONFIG_FIELDS:
        if field in raw:
            out[field] = _stringify(raw[field])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_rollout_stats(diag_dir):
    """All `rollout_*_stats.json` in a diagnostics dir, ordered by rollout index."""
    files = glob.glob(os.path.join(diag_dir, 'rollout_*_stats.json'))

    def index_of(path):
        try:
            return int(os.path.basename(path).split('_')[1])
        except (IndexError, ValueError):
            return 10 ** 9

    rows = []
    for path in sorted(files, key=index_of):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except Exception as exc:                                  # noqa: BLE001
            logger.debug(f'Unreadable rollout stats {path}: {exc}')
    return rows


def _dig(mapping, path, default=None):
    """Nested dict lookup: _dig(d, ['constraint', 'exec', 'sat_rate'])."""
    node = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


# The rollout-JSON schema changed between early Gen7 and Gen14; both are on disk.
CONTEXT_NEW, CONTEXT_OLD = 'context', 'context_info'
CONSTRAINT_NEW, CONSTRAINT_OLD = 'constraint', 'constraint_metrics'

_MISSING = object()


def _pick(row, paths, default=None):
    """First of several candidate paths that actually resolves.

    Lets one extractor read both the nested Gen14 schema and the flat early-Gen7
    one without the caller knowing which file it has.
    """
    for path in paths:
        value = _dig(row, path, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _ctx(rows, field, default=np.nan):
    """A `context` / `context_info` field across every rollout."""
    return np.array(
        [_pick(r, ([CONTEXT_NEW, field], [CONTEXT_OLD, field]), default) for r in rows],
        dtype=float)


def _cm(rows, nested_path, flat_key, default=np.nan):
    """A constraint metric: nested `constraint.exec.…` or flat `constraint_metrics.exec_…`."""
    return np.array(
        [_pick(r, ([CONSTRAINT_NEW] + list(nested_path), [CONSTRAINT_OLD, flat_key]), default)
         for r in rows],
        dtype=float)


def _conflict(row):
    """The D1 box-obstacle conflict block, under either context spelling."""
    value = _pick(row, ([CONTEXT_NEW, 'box_obstacle_conflict'],
                        [CONTEXT_OLD, 'box_obstacle_conflict']), None)
    if isinstance(value, dict):
        return value
    if not value:                       # absent, null, false, 0
        return None
    # A bare truthy flag (e.g. `true`): still a frozen rollout, just no detail.
    # Must stay truthy — an empty dict would read as "not frozen".
    return {'worst_overlap_m': float('nan')}


def _safe_nanmean(values):
    """nanmean without the all-NaN RuntimeWarning."""
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if valid.size else float('nan')


def _seq_item(seq, idx):
    try:
        return float(seq[idx])
    except (TypeError, IndexError, ValueError):
        return np.nan


def _as_float(value):
    try:
        array = np.asarray(value)
        if array.dtype == object:
            return float(array.item())
        return float(array.reshape(-1)[0])
    except Exception:                                             # noqa: BLE001
        return np.nan


def _stringify(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _uid(unit):
    return (f'cand{unit["Candidate"]}/seed{unit["seed"]}/{unit["split"]}/'
            f'{unit["geo"]}/{unit["variant"]}')
