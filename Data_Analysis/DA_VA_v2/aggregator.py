"""
DA_VA_v2 — aggregation.

Everything downstream is a groupby on one master per-rollout table, rather than
the nested seed→variant→constraint→halfspace dicts DA_Code_v3 carried around.
The axes are explicit columns, which is what makes geometry a first-class
dimension instead of a name prefix (Gen14 spec §4 L2).

Tables produced
---------------
per_rollout   Candidate × seed × split × geo × variant × rollout  (wide)
units_long    the above reduced over rollouts                     (long)
agg_long      units_long reduced over seeds                       (long)
scalars_long  run-level scalars that have no per-rollout array    (long)
quality       one row per unit: frozen count, circuit breaker, completeness

The `mask` column carries the frozen-rollout toggle (spec §5.3):
  all       every rollout
  unfrozen  rollouts the D1 box-obstacle guard did NOT freeze
Both are always emitted, so a viewer can switch between them without a re-run.
"""

import logging

import numpy as np
import pandas as pd

from config import (
    ACCURACY_FALLBACK_METRIC,
    ACCURACY_METRIC,
    MAJOR_VARIANTS,
    PRIMARY_METRICS,
    TIME_FALLBACK_METRIC,
    TIME_METRIC,
    VARIANT_ORDER,
)

logger = logging.getLogger(__name__)

ID_COLUMNS = ['Candidate', 'FolderName', 'FullPath', 'seed', 'split',
              'geo', 'geo_base', 'tightened', 'variant', 'variant_raw']

UNIT_KEYS = ['Candidate', 'FolderName', 'FullPath', 'seed', 'split',
             'geo', 'geo_base', 'tightened', 'variant']

AGG_KEYS = ['Candidate', 'FolderName', 'FullPath', 'split',
            'geo', 'geo_base', 'tightened', 'variant']

MASKS = ('all', 'unfrozen')


class Aggregator:
    """Assemble loaded units into the analysis tables."""

    def __init__(self):
        self.per_rollout = pd.DataFrame()
        self.units_long = pd.DataFrame()
        self.agg_long = pd.DataFrame()
        self.scalars_long = pd.DataFrame()
        self.quality = pd.DataFrame()
        self.run_config = pd.DataFrame()
        self.candidate_stats = {}
        self.ranked_candidates = []

    # ── build ─────────────────────────────────────────────────────────────────
    def build(self, loaded_units):
        """
        Args:
            loaded_units: list of (unit_dict, loaded_dict) pairs from UnitLoader.
        """
        logger.info(f'Aggregating {len(loaded_units)} units...')
        self.per_rollout = self._build_per_rollout(loaded_units)
        self.quality = self._build_quality(loaded_units)
        self.run_config = self._build_run_config(loaded_units)
        self.scalars_long = self._build_scalars(loaded_units)
        self.units_long = self._reduce_over_rollouts(self.per_rollout)
        self.agg_long = self._reduce_over_seeds(self.per_rollout)
        self.candidate_stats = self._candidate_stats()
        self.ranked_candidates = self._rank()
        logger.info(f'per_rollout={len(self.per_rollout)} rows, '
                    f'units_long={len(self.units_long)} rows, '
                    f'agg_long={len(self.agg_long)} rows')
        return self

    # ── master per-rollout table ──────────────────────────────────────────────
    @staticmethod
    def _build_per_rollout(loaded_units):
        frames = []
        for unit, loaded in loaded_units:
            frame = loaded['per_rollout'].copy()
            if frame.empty:
                continue
            for key in reversed(ID_COLUMNS):
                frame.insert(0, key, unit[key])
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=ID_COLUMNS + ['rollout_idx'])
        master = pd.concat(frames, ignore_index=True, sort=False)
        master['variant'] = pd.Categorical(
            master['variant'],
            categories=_variant_categories(master['variant'].unique()),
            ordered=True)
        return master

    @staticmethod
    def _build_quality(loaded_units):
        rows = []
        for unit, loaded in loaded_units:
            row = {key: unit[key] for key in UNIT_KEYS}
            row['variant_raw'] = unit['variant_raw']
            row.update(loaded['quality'])
            row['npz_path'] = unit['npz_path'] or ''
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _build_run_config(loaded_units):
        rows = []
        for unit, loaded in loaded_units:
            config = loaded.get('run_config') or {}
            if not config:
                continue
            row = {key: unit[key] for key in UNIT_KEYS}
            row.update(config)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _build_scalars(loaded_units):
        """Run-level scalars (success_rate, entropy, HardFlow's nlp_solves, ...).

        These have no per-rollout array, so they cannot go through the rollout
        table; they are emitted as long rows with n=1 so nothing is silently lost.
        """
        rows = []
        for unit, loaded in loaded_units:
            for metric, value in (loaded.get('scalars') or {}).items():
                row = {key: unit[key] for key in UNIT_KEYS}
                row.update({'metric': metric, 'value': value})
                rows.append(row)
        return pd.DataFrame(rows)

    # ── reductions ────────────────────────────────────────────────────────────
    def _reduce_over_rollouts(self, per_rollout):
        """One row per (unit, mask, metric): mean/std/min/max/n over rollouts."""
        return _reduce(per_rollout, UNIT_KEYS)

    def _reduce_over_seeds(self, per_rollout):
        """One row per (candidate, split, geo, variant, mask, metric).

        Rollouts are pooled across seeds (every rollout weighs the same) and the
        seed count is reported alongside, so a candidate that only ran one seed
        is visibly distinct from one that ran five.
        """
        table = _reduce(per_rollout, AGG_KEYS)
        if table.empty:
            return table
        seeds = (per_rollout.groupby(AGG_KEYS, observed=True)['seed']
                 .nunique().reset_index().rename(columns={'seed': 'n_seeds'}))
        return table.merge(seeds, on=AGG_KEYS, how='left')

    # ── candidate-level summary ───────────────────────────────────────────────
    def _candidate_stats(self):
        """Headline numbers per candidate, for the ranking table and Pareto plot."""
        stats = {}
        if self.agg_long.empty:
            return stats

        table = self.agg_long[self.agg_long['mask'] == 'all']
        for candidate, block in table.groupby('Candidate', observed=True):
            entry = {
                'Candidate': candidate,
                'FolderName': block['FolderName'].iloc[0],
                'FullPath': block['FullPath'].iloc[0],
                'n_variants': int(block['variant'].nunique()),
                'n_geos': int(block['geo'].nunique()),
                'n_seeds': int(block['n_seeds'].max()) if 'n_seeds' in block else np.nan,
            }
            entry['accuracy'] = _metric_mean(block, ACCURACY_METRIC,
                                             ACCURACY_FALLBACK_METRIC)
            entry['accuracy_std'] = _metric_std(block, ACCURACY_METRIC,
                                                ACCURACY_FALLBACK_METRIC)
            entry['time_ms'] = _metric_mean(block, TIME_METRIC, TIME_FALLBACK_METRIC)
            entry['success_rate'] = _metric_mean(block, 'n_success')
            entry['collision_free'] = _metric_mean(block, 'collision_free_completed')
            entry['mean_distance'] = _metric_mean(block, 'mean_dist_per_rollout')
            entry['max_phys_error'] = _metric_mean(block, 'max_phys_error_per_rollout')
            entry['robustness'] = _metric_std(block, ACCURACY_METRIC,
                                              ACCURACY_FALLBACK_METRIC)

            major = block[block['variant'].isin(MAJOR_VARIANTS)]
            entry['major_metrics'] = {}
            for variant, rows in major.groupby('variant', observed=True):
                value = _metric_mean(rows, ACCURACY_METRIC, ACCURACY_FALLBACK_METRIC)
                if not np.isnan(value):
                    entry['major_metrics'][str(variant)] = value
            entry['major_accuracy'] = (float(np.mean(list(entry['major_metrics'].values())))
                                       if entry['major_metrics'] else np.nan)
            stats[candidate] = entry
        return stats

    def _rank(self):
        pairs = [(key, entry.get('accuracy', np.nan))
                 for key, entry in self.candidate_stats.items()]
        pairs = [(k, v) for k, v in pairs if not (v is None or np.isnan(v))]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs

    def print_ranking_summary(self):
        lines = ['=== Cross-candidate ranking ===']
        if not self.ranked_candidates:
            lines.append('  (no rankable candidates)')
        for rank, (candidate, value) in enumerate(self.ranked_candidates, 1):
            entry = self.candidate_stats[candidate]
            lines.append(f'  {rank}. Candidate {candidate} ({entry["FolderName"][:52]}): '
                         f'{value * 100:.1f}% goal+constraint, '
                         f'{entry.get("time_ms", float("nan")):.1f} ms/replan')
        text = '\n'.join(lines)
        logger.info('\n' + text)
        return text


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _reduce(per_rollout, keys):
    """Melt the wide rollout table and reduce it per (keys, mask, metric)."""
    if per_rollout.empty:
        return pd.DataFrame()

    metric_columns = [c for c in per_rollout.columns
                      if c not in ID_COLUMNS and c != 'rollout_idx']
    if not metric_columns:
        return pd.DataFrame()

    long = per_rollout.melt(
        id_vars=[c for c in ID_COLUMNS if c in per_rollout.columns] + ['rollout_idx'],
        value_vars=metric_columns,
        var_name='metric', value_name='value')
    long['value'] = pd.to_numeric(long['value'], errors='coerce')

    frozen = per_rollout[['frozen']] if 'frozen' in per_rollout.columns else None
    if frozen is not None:
        # `melt` repeats the id block once per metric, so the frozen flag tiles
        # the same way; recompute it by merging on the rollout identity instead
        # of relying on row order.
        merge_keys = [c for c in ID_COLUMNS if c in per_rollout.columns] + ['rollout_idx']
        long = long.merge(per_rollout[merge_keys + ['frozen']], on=merge_keys, how='left')
    else:
        long['frozen'] = 0.0

    blocks = []
    for mask in MASKS:
        subset = long if mask == 'all' else long[long['frozen'] != 1.0]
        if subset.empty:
            continue
        grouped = (subset.groupby(keys + ['metric'], observed=True)['value']
                   .agg(['mean', 'std', 'min', 'max', 'count'])
                   .reset_index())
        grouped['mask'] = mask
        blocks.append(grouped)

    if not blocks:
        return pd.DataFrame()

    table = pd.concat(blocks, ignore_index=True)
    table = table.rename(columns={'count': 'n'})

    # Drop all-NaN rows (n == 0). `_build_per_rollout` concatenates units with
    # different column sets, so every unit ends up carrying every OTHER unit's
    # metrics as NaN padding — and the melt then emits one empty row per
    # (group, metric it never measured). Harmless in a single-pipeline run,
    # quadratic in a merged one: a D3IL baseline unit (~20 metrics) next to
    # Gen14 (~50) is mostly padding, and the CSV grew large enough to OOM the
    # HTML viewer's parser. A row with no observations carries no information.
    empty = int((table['n'] == 0).sum())
    if empty:
        table = table[table['n'] > 0].copy()
        logger.info(f'  dropped {empty} all-NaN metric rows '
                    f'(metrics a unit never measured)')

    table['metric_order'] = table['metric'].map(
        {name: i for i, name in enumerate(PRIMARY_METRICS)}).fillna(len(PRIMARY_METRICS))
    table = table.sort_values(keys + ['mask', 'metric_order', 'metric'])
    return table.drop(columns='metric_order').reset_index(drop=True)


def _variant_categories(present):
    """Known variants first, in config order; anything new appended alphabetically."""
    present = [str(v) for v in present]
    ordered = [v for v in VARIANT_ORDER if v in present]
    extra = sorted(v for v in present if v not in ordered)
    return ordered + extra


def _metric_mean(block, metric, fallback=None):
    """Mean of a metric over a block, falling back when it is missing OR all-NaN.

    The all-NaN case is what a projector-free pipeline looks like: the D3IL
    baseline HAS an `n_success_and_constraints` column (`_finalise_frame` creates
    it) but every value is NaN, because there are no constraints to satisfy.
    Without the NaN check such a candidate would rank as NaN and drop out of the
    comparison entirely instead of ranking on plain goal success.
    """
    value = _mean_of(block, metric)
    if np.isnan(value) and fallback:
        value = _mean_of(block, fallback)
    return value


def _mean_of(block, metric):
    rows = block[block['metric'] == metric]
    return float(rows['mean'].mean()) if not rows.empty else float('nan')


def _metric_std(block, metric, fallback=None):
    value = _std_of(block, metric)
    if np.isnan(value) and fallback:
        value = _std_of(block, fallback)
    return value


def _std_of(block, metric):
    rows = block[block['metric'] == metric]
    return float(rows['mean'].std(ddof=0)) if len(rows) > 1 else float('nan')
