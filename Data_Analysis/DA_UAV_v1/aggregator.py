"""
DA_UAV_v1 — aggregation.

Everything downstream is a groupby on one master per-rollout table. The axes are
explicit columns, which is what makes scene / engine / K first-class dimensions
instead of substrings of a folder name.

Tables produced
---------------
per_rollout   Candidate x seed x split x geo x variant x rollout    (wide)
units_long    the above reduced over rollouts                       (long)
agg_long      units_long reduced over seeds                         (long)
k_sweep       agg_long re-grouped on (scene, engine, variant, K)    (long)
scalars_long  run-level scalars that have no per-rollout array      (long)
quality       one row per unit: circuit-breaker counts, completeness

The `mask` column carries the projection-validity toggle:
  all         every rollout
  proj_valid  rollouts whose projection circuit breaker never opened
Both are always emitted, so a viewer can switch between them without a re-run.

The K sweep table
-----------------
`k_sweep` is the only table here with no DA_VA_v2 counterpart, and it is the
point of the tool. PLAN §7.3: every arm runs at K in {1, 2, 5, 10, 20} and the
claim under test is that `mf`/`af` hold success where `fm` collapses at low K,
with the freed wall-clock showing up as fewer circuit-breaker trips. That is a
statement about a curve over K within a (scene, engine, variant) cell, and it
cannot be read off a table whose rows are candidates — the K values are in
different rows with no column tying them together. This one has K as a column.
"""

import logging

import numpy as np
import pandas as pd

from config import (
    ACCURACY_FALLBACK_METRIC,
    ACCURACY_METRIC,
    ENGINE_ORDER,
    MAJOR_VARIANTS,
    MASK_FLAG_COLUMN,
    MASKS,
    PRIMARY_METRICS,
    TIME_FALLBACK_METRIC,
    TIME_METRIC,
    VARIANT_ORDER,
)

logger = logging.getLogger(__name__)

# Path-encoded axes carried on every row (discovery._make_unit puts them there).
AXIS_COLUMNS = ['scene', 'engine', 'K', 'mpc_batch', 'controller', 'threshold',
                'backbone', 'generation']

ID_COLUMNS = (['Candidate', 'FolderName', 'FullPath', 'seed', 'split',
               'geo', 'geo_scene', 'variant', 'variant_base', 'tightened',
               'variant_raw'] + AXIS_COLUMNS)

UNIT_KEYS = (['Candidate', 'FolderName', 'FullPath', 'seed', 'split',
              'geo', 'variant', 'variant_base', 'tightened'] + AXIS_COLUMNS)

AGG_KEYS = (['Candidate', 'FolderName', 'FullPath', 'split',
             'geo', 'variant', 'variant_base', 'tightened'] + AXIS_COLUMNS)

# The K-sweep cell: everything that must be held equal for a K comparison to be
# a K comparison. Seeds are pooled, candidates are not a key (K IS the candidate
# axis here — that is the whole point).
K_SWEEP_KEYS = ['scene', 'engine', 'geo', 'variant', 'split', 'K']


class Aggregator:
    """Assemble loaded units into the analysis tables."""

    def __init__(self):
        self.per_rollout = pd.DataFrame()
        self.units_long = pd.DataFrame()
        self.agg_long = pd.DataFrame()
        self.k_sweep = pd.DataFrame()
        self.hf_flags = pd.DataFrame()
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
        # [HFK1c 2026-08-30] Build the degeneracy lookup BEFORE the wide tables, so every one
        # of them can carry the flag. See `_build_hf_flags`.
        self.hf_flags = self._build_hf_flags(loaded_units)
        self.units_long = _reduce(self.per_rollout, UNIT_KEYS)
        self.agg_long = self._reduce_over_seeds(self.per_rollout)
        self.k_sweep = self._attach_hf_flags(self._build_k_sweep(self.per_rollout),
                                             K_SWEEP_KEYS)
        self.quality = self._attach_hf_flags(self.quality, UNIT_KEYS)
        self.candidate_stats = self._candidate_stats()
        self.ranked_candidates = self._rank()
        logger.info(f'per_rollout={len(self.per_rollout)} rows, '
                    f'units_long={len(self.units_long)} rows, '
                    f'agg_long={len(self.agg_long)} rows, '
                    f'k_sweep={len(self.k_sweep)} rows')
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
                frame.insert(0, key, unit.get(key, ''))
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=ID_COLUMNS + ['rollout_idx'])
        master = pd.concat(frames, ignore_index=True, sort=False)
        master['variant'] = pd.Categorical(
            master['variant'],
            categories=_ordered_categories(master['variant'].unique(), VARIANT_ORDER),
            ordered=True)
        if 'engine' in master.columns:
            master['engine'] = pd.Categorical(
                master['engine'],
                categories=_ordered_categories(master['engine'].unique(), ENGINE_ORDER),
                ordered=True)
        return master

    @staticmethod
    def _build_quality(loaded_units):
        rows = []
        for unit, loaded in loaded_units:
            row = {key: unit.get(key, '') for key in UNIT_KEYS}
            row['variant_raw'] = unit['variant_raw']
            row.update(loaded['quality'])
            row['npz_path'] = unit['npz_path'] or ''
            row['variant_dir'] = unit['variant_dir']
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _build_run_config(loaded_units):
        """One row per unit: the pickled eval `args` PLUS the path-encoded axes.

        Both halves matter and they are worth reading against each other: the
        path says the run was `K4`, the pickle says `flow_steps=4`. A disagreement
        means the folder name is lying about what ran, which is the Gen11 K bug
        (`_uav_eval_tag`'s 🔴 note) in its observable form.
        """
        rows = []
        for unit, loaded in loaded_units:
            config = loaded.get('run_config') or {}
            row = {key: unit.get(key, '') for key in UNIT_KEYS}
            row['path_K'] = unit.get('K')
            row.update(config)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _build_scalars(loaded_units):
        """Run-level scalars (nfe_per_plan, nlp_solves, is_hardflow, ...).

        These have no per-rollout array, so they cannot go through the rollout
        table; they are emitted as long rows with n=1 so nothing is silently lost.
        """
        rows = []
        for unit, loaded in loaded_units:
            for metric, value in (loaded.get('scalars') or {}).items():
                row = {key: unit.get(key, '') for key in UNIT_KEYS}
                row.update({'metric': metric, 'value': value})
                rows.append(row)
        return pd.DataFrame(rows)

    # ── HFK1c (2026-08-30) — the degeneracy flag, on the WIDE tables ──────────
    # 🔴 Why this exists. `hf_degenerate` was already computed per unit by
    # data_loader.py and emitted into scalars_long, so it reached `uav_units_long.csv`
    # — and NOTHING else. Every table a ranking is actually read off
    # (`uav_k_sweep.csv`, `candidates_ranking.csv`, `candidates_per_variant.csv`,
    # `data_quality.csv`) carried no degeneracy column at all. That is the mechanical
    # reason a K=1 row could be promoted to "HardFlow's best result": the flag existed,
    # but not on the path anyone reads. AUDIT_20260830 §3 Gap B.
    #
    # A degenerate row runs NO HardFlow arithmetic (n_genuine == 0) — it is
    # Pi_S(Euler sample) = sample-then-project, == DPCC modulo solver/variable-scope.
    # It must never carry a HardFlow claim, and best-of / win-count / Pareto selections
    # must be computed over non-degenerate rows only.
    #
    # Aggregation rule: MAX over the group. A cell that pools any degenerate unit is
    # flagged — a partially-degenerate cell is not safe to cite either.
    @staticmethod
    def _build_hf_flags(loaded_units):
        """Per-unit degeneracy lookup: UNIT_KEYS + hf_degenerate / hf_n_genuine."""
        rows = []
        for unit, loaded in loaded_units:
            scal = loaded.get('scalars') or {}
            if 'hf_degenerate' not in scal and 'n_genuine' not in scal:
                continue
            row = {key: unit.get(key, '') for key in UNIT_KEYS}
            row['hf_degenerate'] = scal.get('hf_degenerate', np.nan)
            row['hf_n_genuine'] = scal.get('n_genuine', np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def _attach_hf_flags(self, table, keys):
        """Merge `hf_degenerate` / `hf_n_genuine` onto `table`, grouped on `keys`.

        No-op when the batch has no HardFlow units, so non-HF batches are unchanged.
        Non-HardFlow rows get hf_degenerate = 0 (they cannot be degenerate), never NaN,
        so a `== 0` filter selects exactly the citable rows.
        """
        flags = getattr(self, 'hf_flags', None)
        if table is None or table.empty or flags is None or flags.empty:
            return table
        keys = [k for k in keys if k in table.columns and k in flags.columns]
        if not keys:
            return table
        grouped = (flags.groupby(keys, observed=True)[['hf_degenerate', 'hf_n_genuine']]
                   .max().reset_index())
        out = table.merge(grouped, on=keys, how='left')
        # A non-HardFlow row is not degenerate. Only HardFlow rows can be, and those all
        # appear in `flags`, so an unmatched row is a DPCC/diffuser row -> 0.
        out['hf_degenerate'] = out['hf_degenerate'].fillna(0.0)
        n_deg = int((out['hf_degenerate'] > 0).sum())
        if n_deg:
            logger.warning(
                f'  {n_deg} row(s) in this table are DEGENERATE HardFlow (hf_degenerate=1, '
                f'n_genuine=0): sample-then-project, NOT HardFlow. Filter them out before '
                f'any best-of / win-count / Pareto claim. See '
                f'logs_in_develop/aggregated_hardflow_lowK/')
        return out

    # ── reductions ────────────────────────────────────────────────────────────
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

    def _build_k_sweep(self, per_rollout):
        """One row per (scene, engine, geo, variant, split, K, mask, metric).

        Candidates are NOT a key: two candidates differing only in K are the two
        points of the same curve, and two differing in something else (mpc batch,
        controller, threshold, backbone) would be silently pooled here. That last
        risk is real, so the table carries `n_candidates` and `candidates` — a
        cell built from more than one candidate at the same K is flagged in the
        summary rather than quietly averaged away.
        """
        if per_rollout.empty:
            return pd.DataFrame()
        keys = [k for k in K_SWEEP_KEYS if k in per_rollout.columns]
        if 'K' not in keys:
            logger.info('  k_sweep skipped — no K axis in this batch')
            return pd.DataFrame()
        usable = per_rollout[per_rollout['K'].notna()]
        if usable.empty:
            logger.info('  k_sweep skipped — no candidate has a parsable K')
            return pd.DataFrame()

        table = _reduce(usable, keys)
        if table.empty:
            return table
        provenance = (usable.groupby(keys, observed=True)['Candidate']
                      .agg(n_candidates='nunique',
                           candidates=lambda s: ','.join(
                               str(x) for x in sorted(set(s))))
                      .reset_index())
        seeds = (usable.groupby(keys, observed=True)['seed']
                 .nunique().reset_index().rename(columns={'seed': 'n_seeds'}))
        table = table.merge(provenance, on=keys, how='left')
        table = table.merge(seeds, on=keys, how='left')

        mixed = table[table['n_candidates'] > 1]
        if not mixed.empty:
            logger.warning(
                f'  k_sweep: {len(mixed)} cell(s) pool MORE THAN ONE candidate at the '
                f'same K — they differ in an axis this table does not key on '
                f'(mpc batch / controller / threshold / backbone). See the '
                f'`candidates` column in uav_k_sweep.csv before reading those rows.')
        return table.sort_values([k for k in keys if k != 'K'] + ['K', 'mask', 'metric'])

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
            for axis in AXIS_COLUMNS:
                entry[axis] = block[axis].iloc[0] if axis in block.columns else ''
            # [HFK1c 2026-08-30] Carry the degeneracy verdict onto the candidate, so the
            # ranking table can show it. 1 => at least one variant of this candidate is a
            # DEGENERATE HardFlow arm (n_genuine == 0): sample-then-project, NOT HardFlow.
            entry['hf_degenerate'] = self._candidate_hf_degenerate(candidate)
            entry['accuracy'] = _metric_mean(block, ACCURACY_METRIC,
                                             ACCURACY_FALLBACK_METRIC)
            entry['accuracy_std'] = _metric_std(block, ACCURACY_METRIC,
                                                ACCURACY_FALLBACK_METRIC)
            entry['time_ms'] = _metric_mean(block, TIME_METRIC, TIME_FALLBACK_METRIC)
            entry['fm_ms'] = _metric_mean(block, 'fm_ms')
            entry['proj_ms'] = _metric_mean(block, 'proj_ms')
            entry['success_rate'] = _metric_mean(block, 'n_success')
            entry['success_relaxed'] = _metric_mean(block, 'success_relaxed')
            entry['collision_free'] = _metric_mean(block, 'collision_free_completed')
            entry['phys_safe'] = _metric_mean(block, 'phys_safe')
            entry['goal_dist'] = _metric_mean(block, 'goal_dist')
            entry['steps_to_goal'] = _metric_mean(block, 'steps_to_goal')
            entry['over_budget_frac'] = _metric_mean(block, 'over_budget_frac')
            entry['track_err'] = _metric_mean(block, 'track_err_mean')
            entry['cb_tripped'] = _metric_mean(block, MASK_FLAG_COLUMN)
            entry['robustness'] = _metric_std(block, ACCURACY_METRIC,
                                              ACCURACY_FALLBACK_METRIC)
            entry['nfe_effective'] = self._scalar_mean(candidate, 'nfe_effective')

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

    def _candidate_hf_degenerate(self, candidate):
        """Max hf_degenerate over this candidate's units. 0 when it has no HardFlow arm."""
        flags = getattr(self, 'hf_flags', None)
        if flags is None or flags.empty or 'Candidate' not in flags.columns:
            return 0.0
        rows = flags[flags['Candidate'] == candidate]
        if rows.empty:
            return 0.0
        val = rows['hf_degenerate'].max()
        return 0.0 if pd.isna(val) else float(val)

    def _scalar_mean(self, candidate, metric):
        table = self.scalars_long
        if table is None or table.empty or 'metric' not in table.columns:
            return np.nan
        rows = table[(table['Candidate'] == candidate) & (table['metric'] == metric)]
        return float(rows['value'].mean()) if not rows.empty else np.nan

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
            lines.append(f'  {rank}. Candidate {candidate} ({str(entry["FolderName"])[:52]}): '
                         f'{value * 100:.1f}% success+constraint, '
                         f'{entry.get("time_ms", float("nan")):.1f} ms/replan, '
                         f'K={entry.get("K")}')
        text = '\n'.join(lines)
        logger.info('\n' + text)
        return text

    # ── Pareto ────────────────────────────────────────────────────────────────
    def pareto_front(self, accuracy_key='accuracy', cost_key='time_ms'):
        """Candidates not dominated on (accuracy up, cost down).

        The repo's standing definition of "good" is Pareto dominance at equal
        success and constraints — anything else is a trade-off and must be worded
        that way, never "best". Returned so the reporter can say which candidates
        may be called wins and which may not.
        """
        points = [(key, entry.get(accuracy_key, np.nan), entry.get(cost_key, np.nan))
                  for key, entry in self.candidate_stats.items()]
        points = [p for p in points if not (np.isnan(p[1]) or np.isnan(p[2]))]
        front = []
        for key, accuracy, cost in points:
            dominated = any(
                (other_a >= accuracy and other_c <= cost
                 and (other_a > accuracy or other_c < cost))
                for other_key, other_a, other_c in points if other_key != key)
            if not dominated:
                front.append(key)
        return sorted(front)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _reduce(per_rollout, keys):
    """Melt the wide rollout table and reduce it per (keys, mask, metric)."""
    if per_rollout.empty:
        return pd.DataFrame()

    keys = [k for k in keys if k in per_rollout.columns]
    id_columns = [c for c in ID_COLUMNS if c in per_rollout.columns]
    # Label columns are text, not measurements — melting them into `value` would
    # only produce NaN rows after the numeric coercion below.
    label_columns = [c for c in ('homotopy', 'homotopy_flown', 'scene_json')
                     if c in per_rollout.columns]
    metric_columns = [c for c in per_rollout.columns
                      if c not in id_columns + label_columns + ['rollout_idx']]
    if not metric_columns:
        return pd.DataFrame()

    long = per_rollout.melt(
        id_vars=id_columns + ['rollout_idx'],
        value_vars=metric_columns,
        var_name='metric', value_name='value')
    long['value'] = pd.to_numeric(long['value'], errors='coerce')

    if MASK_FLAG_COLUMN in per_rollout.columns:
        # `melt` repeats the id block once per metric, so the flag tiles the same
        # way; recompute it by merging on the rollout identity instead of relying
        # on row order.
        merge_keys = id_columns + ['rollout_idx']
        long = long.merge(per_rollout[merge_keys + [MASK_FLAG_COLUMN]],
                          on=merge_keys, how='left', suffixes=('', '_flag'))
        flag_column = (f'{MASK_FLAG_COLUMN}_flag'
                       if f'{MASK_FLAG_COLUMN}_flag' in long.columns
                       else MASK_FLAG_COLUMN)
    else:
        long[MASK_FLAG_COLUMN] = 0.0
        flag_column = MASK_FLAG_COLUMN

    blocks = []
    for mask in MASKS:
        subset = long if mask == 'all' else long[long[flag_column].fillna(0) != 1.0]
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
    # (group, metric it never measured). Harmless in a single-arm run, quadratic
    # in a merged one, and a row with no observations carries no information.
    empty = int((table['n'] == 0).sum())
    if empty:
        table = table[table['n'] > 0].copy()
        logger.info(f'  dropped {empty} all-NaN metric rows '
                    f'(metrics a unit never measured)')

    table['metric_order'] = table['metric'].map(
        {name: i for i, name in enumerate(PRIMARY_METRICS)}).fillna(len(PRIMARY_METRICS))
    table = table.sort_values(keys + ['mask', 'metric_order', 'metric'])
    return table.drop(columns='metric_order').reset_index(drop=True)


def _ordered_categories(present, order):
    """Known values first, in config order; anything new appended alphabetically."""
    present = [str(v) for v in present]
    ordered = [v for v in order if v in present]
    extra = sorted(v for v in present if v not in ordered)
    return ordered + extra


def _metric_mean(block, metric, fallback=None):
    """Mean of a metric over a block, falling back when it is missing OR all-NaN."""
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
