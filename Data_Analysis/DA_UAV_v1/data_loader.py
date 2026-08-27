"""
DA_UAV_v1 — per-unit data loading.

One "unit" = one (candidate, seed, split, geo, variant) result, i.e. exactly one
`<variant>.npz` plus the `diagnostics/` and `results.json` beside it. The loader
turns it into:

  per_rollout : DataFrame, one row per rollout, one column per per-rollout metric
  scalars     : run-level scalars with no per-rollout array (HardFlow's
                nfe_per_plan / nlp_solves, the effective NFE budget, …)
  quality     : data-quality counters (circuit-breaker rollouts, sentinel file,
                whether timing was recoverable at all)
  run_config  : the pickled eval `args` namespace

Two things differ from the DA_VA_v2 loader they are otherwise a copy of:

1. **The diagnostics scan is not optional.** `eval_artifacts.save_npz` writes the
   success / physical / constraint / goal / projection-health groups but NOT the
   `timing` group — so `avg_time`, `fm_ms` and `proj_ms`, the axes Gen15 exists
   to measure (PLAN §7.2), exist ONLY in `diagnostics/rollout_<r>_stats.json`.
   `--no-diagnostics-scan` is still accepted for a quick structural pass, and
   then says loudly that every timing column will be NaN.

2. **Nothing is invented for a projector-free variant.** `diffuser` runs no
   projector, so its constraint columns describe the raw generator. They are
   real numbers (the eval still measures the flown path against the declared
   margins), so unlike the D3IL case in DA_VA_v2 there is nothing to rescue —
   but `has_projector=0` is still recorded per unit so a reader knows why
   `proj_ms` is 0.0 there rather than suspecting a broken timer.

Sources:
  npz  — the per-rollout outcome arrays written at variant end
  json — `diagnostics/rollout_*_stats.json`, which carry EVERYTHING the npz does
         plus timing; used alone when a variant died before its final npz write
  auto — npz when present (+ the JSON side-scan for timing), else json (default)
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from config import (
    BOOLEAN_METRICS,
    DIAGNOSTICS_REQUIRED,
    HEAVY_KEYS,
    JSON_BOOL_PATHS,
    JSON_COUNT_PATHS,
    JSON_FIELD_PATHS,
    JSON_LABEL_PATHS,
    MASK_FLAG_COLUMN,
    META_KEYS,
    NPZ_RENAMES,
    RUN_CONFIG_FIELDS,
    UNPROJECTED_VARIANTS,
)

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
        self.n_no_timing = 0

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

        # The npz has no timing group at all — the JSON side-scan is where
        # avg_time / fm_ms / proj_ms come from, plus the projector's own trip
        # counters. Always worth its cost on a UAV tree.
        if self.scan_diagnostics and source_used == 'npz':
            extras = self._scan_diagnostics(unit, len(per_rollout))
            for name, values in extras.items():
                if name not in per_rollout.columns:
                    per_rollout[name] = values

        # Run-level blocks (timing rollup, projection health, the HardFlow NFE
        # accounting) live in results.json and nowhere else.
        scalars.update(self._read_results_json(unit))

        per_rollout = _finalise_frame(per_rollout, unit)
        quality.update(_quality_from_frame(per_rollout))
        quality['source'] = source_used
        quality['n_rollouts'] = int(len(per_rollout))
        quality['cb_sentinel'] = 1.0 if unit.get('cb_sentinel') else 0.0
        quality['has_projector'] = 0.0 if _is_unprojected(unit['variant']) else 1.0

        if quality.get('timing_missing'):
            self.n_no_timing += 1
            self._note('WARN', f'{_uid(unit)}: no per-rollout timing found — '
                               f'avg_time / fm_ms / proj_ms are NaN. The npz never '
                               f'carries timing; it needs diagnostics/*.json.')

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
            'files_without_timing': self.n_no_timing,
            'log': self.log,
        }

    def save_log(self, path):
        with open(path, 'w') as f:
            f.write('=== DA_UAV_v1 loading log ===\n\n')
            f.write(f'Loaded:          {self.n_loaded}\n')
            f.write(f'Failed:          {self.n_failed}\n')
            f.write(f'Skipped:         {self.n_skipped}\n')
            f.write(f'Without timing:  {self.n_no_timing}\n\n')
            if self.n_no_timing and not self.scan_diagnostics:
                f.write('NOTE: --no-diagnostics-scan was passed. UAV npz files carry NO\n'
                        '      timing group, so every timing metric is NaN by construction.\n\n')
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
        """Build the whole per-rollout frame from `diagnostics/rollout_*_stats.json`.

        The UAV rollout JSON is the Fix_10 grouped schema — `success.strict`,
        `physical.*`, `constraint.*`, `goal.*`, `timing.*`, `projection_health.*`
        — and is a strict superset of the npz. Older flat spellings are looked up
        as a second path per field so a pre-Fix_10 tree still loads.
        """
        diag_dir = unit['diagnostics_dir']
        if not os.path.isdir(diag_dir):
            self._note('MISSING', f'{_uid(unit)}: no npz and no diagnostics dir')
            return None

        rows = _read_rollout_stats(diag_dir)
        if not rows:
            self._note('MISSING', f'{_uid(unit)}: diagnostics dir holds no rollout stats')
            return None

        columns = {
            'rollout_idx': np.array([_dig(r, ['rollout_index'], i)
                                     for i, r in enumerate(rows)], dtype=int),
        }
        for name, paths in JSON_BOOL_PATHS.items():
            columns[name] = np.array([_bool_of(_pick(r, paths, None)) for r in rows],
                                     dtype=float)
        for name, paths in list(JSON_FIELD_PATHS.items()) + list(JSON_COUNT_PATHS.items()):
            columns[name] = np.array([_as_float(_pick(r, paths, np.nan)) for r in rows],
                                     dtype=float)
        labels = {name: [_str_of(_pick(r, paths, '')) for r in rows]
                  for name, paths in JSON_LABEL_PATHS.items()}

        frame = pd.DataFrame(columns)
        for name, values in labels.items():
            frame[name] = values
        scalars = {'success_rate': _safe_nanmean(frame.get('n_success'))}
        quality = {'npz_complete': np.nan, 'diagnostics_found': 1.0}

        if np.isnan(scalars['success_rate']):
            self._note('WARN', f'{_uid(unit)}: no success field found in the rollout '
                               f'JSONs — unrecognised schema, metrics will be NaN')
        return frame, scalars, {}, quality

    # ── diagnostics side-scan (npz source) ───────────────────────────────────
    def _scan_diagnostics(self, unit, n_rollouts):
        """Harvest the fields only the per-rollout JSONs carry.

        Chief among them the whole `timing` group. `eval_artifacts.save_npz`
        persists success/physical/constraint/goal/projection-health and stops
        there, so without this pass a UAV batch has no time axis at all — and the
        time axis is the Gen15 experiment.
        """
        out = {}
        diag_dir = unit['diagnostics_dir']
        wanted = dict(JSON_FIELD_PATHS)
        wanted.update(JSON_COUNT_PATHS)

        if not os.path.isdir(diag_dir):
            self._note('WARN', f'{_uid(unit)}: no diagnostics dir — every timing '
                               f'metric will be NaN (the npz has no timing group)')
            for name in wanted:
                out[name] = np.full(n_rollouts, np.nan)
            out['diagnostics_found'] = np.zeros(n_rollouts)
            return out

        rows = _read_rollout_stats(diag_dir)
        by_index = {}
        for i, row in enumerate(rows):
            by_index[int(_dig(row, ['rollout_index'], i) or i)] = row

        for name, paths in wanted.items():
            out[name] = np.array(
                [_as_float(_pick(by_index.get(i) or {}, paths, np.nan))
                 for i in range(n_rollouts)], dtype=float)
        for name, paths in JSON_LABEL_PATHS.items():
            out[name] = [_str_of(_pick(by_index.get(i) or {}, paths, ''))
                         for i in range(n_rollouts)]
        out['diagnostics_found'] = np.array(
            [1.0 if i in by_index else 0.0 for i in range(n_rollouts)])
        return out

    # ── results.json (run-level blocks) ───────────────────────────────────────
    def _read_results_json(self, unit):
        """The run-level scalars: HardFlow NFE accounting + the timing rollup.

        `nfe_per_plan` is the number to quote when comparing a HardFlow arm with
        a DPCC one: `hardflow_new` evaluates the network twice per ACTIVE ODE step
        (the reference step and the terminal predict), so an arm-C run costs
        K + n_active - 1 network evals while a DPCC arm at K costs K. Comparing
        them "at the same K" is comparing a smaller generation budget on the DPCC
        side (Gen15 U2) — except at K=1, where HFK1 (2026-08-24) made the two
        equal by dropping the terminal lookahead call, whose weight was exactly 0.
        `nfe_effective` below is the normalised figure to put on an x-axis.

        HFK1 also adds `n_genuine`: the number of ODE steps that are ACTIVE and
        NON-TERMINAL, i.e. the only steps where HardFlow's endpoint lookahead,
        damped pull-back and feedback actually run. `n_genuine == 0` means the row
        is Pi_S(Euler sample) — sample-then-project, == DPCC modulo solver — and
        must NOT be read as a HardFlow result. Runs written before 2026-08-24 have
        no such field, so it is DERIVED here from K and activation_threshold; the
        derivation is exact, since the gate is pure arithmetic.
        """
        path = unit.get('results_json')
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path) as f:
                payload = json.load(f) or {}
        except Exception as exc:                                  # noqa: BLE001
            self._note('WARN', f'{_uid(unit)}: unreadable {os.path.basename(path)} — {exc}')
            return {}

        summary = payload.get('summary') or {}
        out = {}
        hardflow = summary.get('hardflow') or {}
        if hardflow.get('is_hardflow'):
            for field in ('nfe_per_plan', 'nlp_solves_total', 'nlp_failures_total',
                          'activation_threshold', 'init_noise_scale', 'nfe_total',
                          # [SolverSwap] float twin of `nlp_backend`; the string stays
                          # in the json for humans and never enters the scalars dict.
                          'nlp_backend_slsqp'):
                if field in hardflow:
                    out[field] = _as_float(hardflow[field])
            out['is_hardflow'] = 1.0
            # HFK1 (2026-08-24) — degeneracy verdict. Prefer what the run recorded;
            # DERIVE it for the whole pre-2026-08-24 corpus, which predates the field.
            if 'n_genuine' in hardflow:
                out['n_active'] = _as_float(hardflow.get('n_active'))
                out['n_genuine'] = _as_float(hardflow['n_genuine'])
            else:
                K = unit.get('K')
                A = out.get('activation_threshold')
                if K not in (None, '') and A is not None and not np.isnan(A):
                    K = int(float(K))
                    n_active = max(K - int((1.0 - float(A)) * K), 1)
                    out['n_active'] = float(n_active)
                    out['n_genuine'] = float(n_active - 1)
            if 'n_genuine' in out:
                out['hf_degenerate'] = 1.0 if out['n_genuine'] == 0 else 0.0
                if out['hf_degenerate']:
                    self._note('WARN',
                               f'{_uid(unit)}: HardFlow arm is DEGENERATE '
                               f'(K={unit.get("K")}, A={out.get("activation_threshold")}, '
                               f'n_genuine=0) — this row is sample-then-project, not '
                               f'HardFlow. Do not label it a HardFlow result. See '
                               f'logs_in_develop/aggregated_hardflow_lowK/')
        else:
            out['is_hardflow'] = 0.0

        # The effective per-plan network-evaluation count: measured where the run
        # measured it (HardFlow), else the eval-tag K budget. One number that can
        # honestly go on a "cost" axis across every arm.
        if 'nfe_per_plan' in out and not np.isnan(out['nfe_per_plan']):
            out['nfe_effective'] = out['nfe_per_plan']
        elif unit.get('K') not in (None, ''):
            out['nfe_effective'] = _as_float(unit['K'])

        health = summary.get('projection_health') or {}
        for field in ('n_tripped_trials', 'total_skipped_steps'):
            if field in health:
                out[f'summary_{field}'] = _as_float(health[field])
        timing = summary.get('timing') or {}
        for field in ('budget_ms', 'total_over_budget'):
            if field in timing:
                out[f'summary_{field}'] = _as_float(timing[field])
        steps = summary.get('steps') or {}
        if 'max_episode_length' in steps:
            out['summary_max_episode_length'] = _as_float(steps['max_episode_length'])
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

def _finalise_frame(frame, unit):
    """Rename the npz group-prefixed keys onto the canonical names, then derive.

    The rename ADDS the canonical column and keeps the raw one, so
    `per_rollout_detail.csv` can still be read against `eval_artifacts.save_npz`
    line by line while every downstream table speaks the DA_Code_v3 vocabulary
    both HTML viewers already know.

    Derived here:
      avg_time       ← avg_time_ms / 1000     (DA_Code_v3 reports seconds/replan)
      steps_to_goal  ← n_steps on reaching episodes only, NaN otherwise
      over_budget_frac ← over_budget_steps / n_steps
    """
    if frame.empty:
        return frame

    for raw, canonical in NPZ_RENAMES.items():
        if raw in frame.columns and canonical not in frame.columns:
            frame[canonical] = frame[raw]

    def col(name):
        return frame[name] if name in frame.columns else None

    # ── timing ────────────────────────────────────────────────────────────────
    # The eval measures milliseconds per replan; DA_Code_v3's `avg_time` is
    # seconds per replan, and both viewers plot that name. Offer both.
    if 'avg_time_ms' in frame.columns and 'avg_time' not in frame.columns:
        frame['avg_time'] = frame['avg_time_ms'].astype(float) / 1000.0

    # ── steps ─────────────────────────────────────────────────────────────────
    # `n_steps` is the episode length under the U_13 fixed budget: it early-stops
    # on goal-reach and runs the FULL budget on a miss. Averaging it over both
    # therefore measures "how many misses were there" as much as "how fast".
    # `steps_to_goal` is the honest time-to-goal — reaching episodes only, which
    # is exactly what the eval's own `steps.to_goal_mean` reports.
    reached = col('goal_reached')
    steps = col('n_steps')
    if reached is not None and steps is not None and 'steps_to_goal' not in frame.columns:
        frame['steps_to_goal'] = steps.astype(float).where(
            reached.astype(float) == 1.0, np.nan)

    over = col('over_budget_steps')
    if over is not None and steps is not None and 'over_budget_frac' not in frame.columns:
        denominator = steps.astype(float).replace(0.0, np.nan)
        frame['over_budget_frac'] = over.astype(float) / denominator

    # ── masking flag ──────────────────────────────────────────────────────────
    # Absent on a tree written before Fix_15.3; treated as "never tripped", which
    # is the truth for those runs — the breaker did not exist to trip.
    if MASK_FLAG_COLUMN not in frame.columns:
        frame[MASK_FLAG_COLUMN] = 0.0

    for name in frame.columns:
        if name == 'rollout_idx':
            continue
        if name in BOOLEAN_METRICS or frame[name].dtype == bool:
            frame[name] = frame[name].astype(float)

    lead = ['rollout_idx', MASK_FLAG_COLUMN]
    ordered = lead + sorted(c for c in frame.columns if c not in lead)
    return frame[ordered]


def _quality_from_frame(frame):
    """Per-unit data-quality counters — reported, never ranked on."""
    out = {}
    n = len(frame)
    if MASK_FLAG_COLUMN in frame.columns:
        out['n_cb_tripped'] = int(frame[MASK_FLAG_COLUMN].fillna(0).sum())
    else:
        out['n_cb_tripped'] = 0
    out['cb_tripped_rate'] = (out['n_cb_tripped'] / n) if n else np.nan
    for column, name in (('projection_cb_skipped_steps', 'cb_skipped_steps'),
                         ('projection_cb_trips', 'cb_trips'),
                         ('projection_backstop_hits', 'backstop_hits')):
        out[name] = (float(frame[column].fillna(0).sum())
                     if column in frame.columns else 0.0)
    if 'diagnostics_found' in frame.columns:
        out['n_diagnostics_json'] = int(frame['diagnostics_found'].fillna(0).sum())
    # The single most important completeness question on a UAV batch.
    timing = frame['avg_time_ms'] if 'avg_time_ms' in frame.columns else None
    out['timing_missing'] = 1.0 if (timing is None or timing.isna().all()) else 0.0
    return out


def _is_unprojected(variant):
    return str(variant) in UNPROJECTED_VARIANTS


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
    for key in ('success_strict', 'n_success', 'n_steps', 'goal_reached'):
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
    try:
        names = os.listdir(diag_dir)
    except OSError:
        return []
    files = [os.path.join(diag_dir, n) for n in names
             if n.startswith('rollout_') and n.endswith('_stats.json')]

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
    """Nested dict lookup: _dig(d, ['timing', 'total_ms_mean'])."""
    node = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


_MISSING = object()


def _pick(row, paths, default=None):
    """First of several candidate paths that actually resolves.

    Lets one extractor read both the nested Fix_10 schema and the flat pre-Fix_10
    spelling without the caller knowing which file it has.
    """
    for path in paths:
        value = _dig(row, path, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _bool_of(value):
    if value is None:
        return np.nan
    if isinstance(value, dict):
        # `success` was a nested group after Fix_10; bool(dict) is always True,
        # which is exactly the bug Fix_12 fixed in the foresight titles. A dict
        # here means the caller asked for a GROUP, not a flag — refuse it.
        return np.nan
    try:
        return 1.0 if bool(value) else 0.0
    except Exception:                                             # noqa: BLE001
        return np.nan


def _safe_nanmean(values):
    """nanmean without the all-NaN RuntimeWarning."""
    if values is None:
        return float('nan')
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if valid.size else float('nan')


def _as_float(value):
    try:
        array = np.asarray(value)
        if array.dtype == object:
            return float(array.item())
        return float(array.reshape(-1)[0])
    except Exception:                                             # noqa: BLE001
        return np.nan


def _str_of(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list, tuple)):
        return ''
    return str(value)


def _stringify(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _uid(unit):
    return (f'cand{unit["Candidate"]}/seed{unit["seed"]}/'
            f'{unit["geo"]}/{unit["variant"]}')


# Referenced in the module docstring; kept importable so a caller can assert the
# contract rather than re-deriving it.
__all__ = ['UnitLoader', 'DIAGNOSTICS_REQUIRED']
