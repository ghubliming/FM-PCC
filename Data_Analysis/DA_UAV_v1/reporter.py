"""
DA_UAV_v1 — CSV / TXT reporting.

The CSVs are the product; the PNGs are optional garnish. Three families:

native      per_rollout_detail.csv, uav_units_long.csv, uav_aggregated_long.csv,
            uav_k_sweep.csv, data_quality.csv, run_config.csv — the full
            (candidate, seed, split, geo, variant, mask, metric) cube plus the
            scene/engine/K axes, nothing collapsed.

DA_Code_v3  candidates_multidimensional_raw.csv / _aggregated.csv,
compat      candidates_ranking.csv, candidates_detailed.csv — the column names
            `Data_Analysis/Visualizer/index.html` reads, so a UAV batch can be
            opened in the DAv3 page unchanged.

Axis mapping for the DA_Code_v3-compat files (that schema has no scene, engine,
K or split axis of its own):
    halfspace_variant  <-  geo        (the geo_tag — this task's environment axis)
    constraint_type    <-  split      ('test' / 'train')
Scene, engine and K are not dropped: they are already in the candidate's display
name (`corridor|mf|K4|bbunet`, built by discovery.display_name), which is what
that viewer prints for a candidate. The native CSVs carry them as real columns
for `Visualizer_UAV_v1`.
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    ACCURACY_METRIC,
    CSV_STEM,
    LOWER_IS_BETTER,
    MASK_FLAG_COLUMN,
    PERCENTAGE_METRICS,
    PRIMARY_METRICS,
    TIME_METRIC,
)
from discovery import format_snapshot_ts, snapshot_by_seed_str

logger = logging.getLogger(__name__)

AXIS_COLUMNS = ['scene', 'engine', 'K', 'mpc_batch', 'controller', 'threshold',
                'run_tag', 'backbone', 'generation']


class Reporter:
    """Write every output file for one batch run."""

    def __init__(self, aggregator, candidates_info=None, run_meta=None):
        self.agg = aggregator
        self.candidates_info = candidates_info or {}
        self.run_meta = run_meta or {}
        self.written = []

    # ── entry point ───────────────────────────────────────────────────────────
    def save_all(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        self._write(output_dir, 'per_rollout_detail.csv', self._per_rollout_table())
        self._write(output_dir, f'{CSV_STEM}_units_long.csv', self._units_long_table())
        self._write(output_dir, f'{CSV_STEM}_aggregated_long.csv', self._agg_long_table())
        self._write(output_dir, f'{CSV_STEM}_k_sweep.csv', self.agg.k_sweep)
        self._write(output_dir, 'data_quality.csv', self._quality_table())
        self._write(output_dir, 'run_config.csv', self.agg.run_config)
        self._write(output_dir, 'candidate_axes.csv', self._axes_table())

        self._write(output_dir, 'candidates_multidimensional_raw.csv',
                    self._compat_raw_table())
        self._write(output_dir, 'candidates_multidimensional_aggregated.csv',
                    self._compat_aggregated_table())

        self._write(output_dir, 'candidates_ranking.csv', self._ranking_table())
        self._write(output_dir, 'candidates_detailed.csv', self._detailed_table())
        self._write(output_dir, 'candidates_per_variant.csv', self._per_variant_table())

        self.save_summary_txt(os.path.join(output_dir, 'candidates_summary.txt'))
        logger.info(f'{len(self.written)} report files written to {output_dir}')
        return self.written

    # ── native tables ─────────────────────────────────────────────────────────
    def _per_rollout_table(self):
        """Wide, one row per rollout — the table to eyeball a single rollout in."""
        table = self.agg.per_rollout
        if table.empty:
            return table
        lead = ['Candidate', 'FolderName', 'scene', 'engine', 'K', 'seed', 'split',
                'geo', 'variant', 'variant_base', 'tightened', 'variant_raw',
                'rollout_idx', MASK_FLAG_COLUMN, 'homotopy', 'homotopy_flown']
        lead = [c for c in lead if c in table.columns]
        primary = [c for c in PRIMARY_METRICS if c in table.columns and c not in lead]
        rest = sorted(c for c in table.columns if c not in lead + primary)
        return table[lead + primary + rest]

    def _units_long_table(self):
        """Per (unit, mask, metric), plus the run-level scalars as n=1 rows."""
        table = self.agg.units_long
        scalars = self.agg.scalars_long
        if scalars is not None and not scalars.empty:
            extra = scalars.copy()
            extra = extra.rename(columns={'value': 'mean'})
            extra['std'] = np.nan
            extra['min'] = extra['mean']
            extra['max'] = extra['mean']
            extra['n'] = 1
            extra['mask'] = 'all'
            if table is None or table.empty:
                return self._with_seed_snapshot(extra)
            columns = [c for c in table.columns if c in extra.columns]
            table = pd.concat([table, extra[columns]], ignore_index=True, sort=False)
        return self._with_seed_snapshot(table)

    def _agg_long_table(self):
        """Seeds are pooled here, so the stamp is the candidate's NEWEST run.

        This is the column `Visualizer_UAV_v1` reads for its "Last Run" columns;
        `uav_units_long.csv` keeps the per-seed breakdown.
        """
        table = self.agg.agg_long
        if table is None or table.empty:
            return table
        out = table.copy()
        out['LatestSnapshot'] = out['Candidate'].map(self._latest_snapshot_map()).fillna('')
        return out

    def _quality_table(self):
        """One row per unit. Read this before believing any constraint number.

        `n_cb_tripped`   rollouts where the projection circuit breaker opened;
                         those ran (partly) UNPROJECTED, so their constraint
                         metrics describe a policy the variant name does not name.
        `cb_sentinel`    the eval's own PROJECTION_CB_TRIPPED.txt is present.
        `timing_missing` no per-rollout timing was recoverable — every timing
                         metric of this unit is NaN. On UAV that means the
                         diagnostics JSONs are absent, since the npz never
                         carries timing at all.
        `npz_complete`   0 means the npz is a crash-safety partial.
        `hf_degenerate`  1 means this unit is a DEGENERATE HardFlow arm — n_genuine
                         == 0, so NO HardFlow arithmetic ran and the row is
                         Pi_S(Euler sample) = sample-then-project (== DPCC modulo
                         solver/variable-scope). Valid as a solver comparison; never
                         as a HardFlow result. Exclude from best-of / win-count /
                         Pareto claims. See logs_in_develop/aggregated_hardflow_lowK/
        """
        table = self.agg.quality
        if table is None or table.empty:
            return table
        lead = ['Candidate', 'FolderName', 'scene', 'engine', 'K', 'seed', 'split',
                'geo', 'variant', 'hf_degenerate', 'hf_n_genuine',
                'source', 'has_projector', 'n_rollouts',
                'n_cb_tripped', 'cb_tripped_rate', 'cb_skipped_steps', 'cb_trips',
                'backstop_hits', 'cb_sentinel', 'timing_missing',
                'n_diagnostics_json', 'npz_complete']
        lead = [c for c in lead if c in table.columns]
        rest = [c for c in table.columns if c not in lead]
        return table[lead + rest]

    def _axes_table(self):
        """One row per candidate: everything its path encodes, plus coverage.

        The table to open first. It answers "what is actually in this batch" —
        which scenes, which engines, which K values, how many seeds each — before
        any metric is read, and it is the only place the eval-tag folder name is
        printed next to its parsed meaning so a mis-parse is visible.
        """
        rows = []
        for candidate in sorted(self.candidates_info):
            info = self.candidates_info[candidate]
            axes = info.get('axes', {}) or {}
            snapshots = info.get('snapshots') or {}
            rows.append({
                'Candidate': candidate,
                'Display': info.get('custom_name') or info.get('display') or info['name'],
                'Eval_Tag': info['name'],
                'scene': axes.get('scene', ''),
                'engine': axes.get('engine', ''),
                'engine_label': axes.get('engine_label', ''),
                'K': axes.get('K'),
                'mpc_batch': axes.get('mpc_batch'),
                'controller': axes.get('controller', ''),
                'threshold': axes.get('threshold'),
                'run_tag': axes.get('run_tag', ''),
                'backbone': axes.get('backbone', ''),
                'data_proportion': axes.get('data_proportion', ''),
                'alpha_init': axes.get('alpha_init', ''),
                'alpha_end': axes.get('alpha_end', ''),
                'train_K': axes.get('train_K', ''),
                'horizon': axes.get('horizon', ''),
                'diffusion_cls': axes.get('diffusion_cls', ''),
                'model_name': axes.get('model_name', ''),
                'generation': axes.get('generation', ''),
                'Seeds': ','.join(str(s) for s in info.get('seeds', [])),
                'N_Seeds': len(info.get('seeds', [])),
                'Missing_Seeds': str(info.get('missing_seeds', []) or ''),
                'CB_Sentinels': info.get('cb_sentinels', 0),
                'Latest_Snapshot': snapshots.get('latest', '') or '',
                'Snapshot_By_Seed': snapshot_by_seed_str(snapshots.get('per_seed')),
                'Full_Path': info['path'],
            })
        return pd.DataFrame(rows)

    # ── DA_Code_v3-compatible tables ─────────────────────────────────────────
    def _compat_raw_table(self):
        table = self.agg.units_long
        if table is None or table.empty:
            return pd.DataFrame()
        rows = table[table['mask'] == 'all'].copy()
        return pd.DataFrame({
            'Candidate': rows['Candidate'],
            'Folder_Name': rows['FolderName'],
            'Full_Path': rows['FullPath'],
            'Missing_Seeds': rows['Candidate'].map(self._missing_seeds_map()),
            'Latest_Snapshot': self._seed_snapshot_column(rows),
            'seed': rows['seed'],
            'variant': rows['variant'].astype(str),
            'constraint_type': rows['split'],
            'halfspace_variant': _compat_env(rows),
            'metric': rows['metric'],
            'value': rows['mean'],
        })

    def _compat_aggregated_table(self):
        table = self.agg.agg_long
        if table is None or table.empty:
            return pd.DataFrame()
        rows = table[table['mask'] == 'all'].copy()
        return pd.DataFrame({
            'Candidate': rows['Candidate'],
            'Folder_Name': rows['FolderName'],
            'Full_Path': rows['FullPath'],
            'Missing_Seeds': rows['Candidate'].map(self._missing_seeds_map()),
            'Latest_Snapshot': rows['Candidate'].map(self._latest_snapshot_map()).fillna(''),
            'variant': rows['variant'].astype(str),
            'constraint_type': rows['split'],
            'halfspace_variant': _compat_env(rows),
            'metric': rows['metric'],
            'mean': rows['mean'],
            'std': rows['std'],
            'count': rows['n'],
        })

    # ── candidate-level tables ───────────────────────────────────────────────
    def _ranking_table(self):
        front = set(self.agg.pareto_front())
        rows = []
        for rank, (candidate, accuracy) in enumerate(self.agg.ranked_candidates, 1):
            entry = self.agg.candidate_stats[candidate]
            info = self.candidates_info.get(candidate, {})
            rows.append({
                'Rank': rank,
                'Candidate': candidate,
                'Folder': entry['FolderName'],
                'scene': entry.get('scene', ''),
                'engine': entry.get('engine', ''),
                'K': entry.get('K'),
                # [HFK1c 2026-08-30] 1 => this candidate carries a DEGENERATE HardFlow arm
                # (n_genuine == 0): sample-then-project, NOT HardFlow. Never cite such a row
                # as a HardFlow result, and exclude it from best-of / Pareto claims.
                # See logs_in_develop/aggregated_hardflow_lowK/
                'hf_degenerate': entry.get('hf_degenerate', 0.0),
                'NFE_effective': entry.get('nfe_effective', np.nan),
                'Success+Constraint (%)': accuracy * 100,
                'Std (%)': entry.get('accuracy_std', np.nan) * 100,
                'Success (%)': entry.get('success_rate', np.nan) * 100,
                'Success relaxed (%)': entry.get('success_relaxed', np.nan) * 100,
                'CollisionFree (%)': entry.get('collision_free', np.nan) * 100,
                'PhysSafe (%)': entry.get('phys_safe', np.nan) * 100,
                'Time (ms)': entry.get('time_ms', np.nan),
                'Gen (ms)': entry.get('fm_ms', np.nan),
                'Proj (ms)': entry.get('proj_ms', np.nan),
                'OverBudget (frac)': entry.get('over_budget_frac', np.nan),
                'CB tripped (rate)': entry.get('cb_tripped', np.nan),
                'GoalDist_m': entry.get('goal_dist', np.nan),
                'StepsToGoal': entry.get('steps_to_goal', np.nan),
                'TrackErr': entry.get('track_err', np.nan),
                'Pareto': 'FRONT' if candidate in front else '',
                'Seeds': entry.get('n_seeds', np.nan),
                'Missing_Seeds': str(info.get('missing_seeds', []) or ''),
                'Latest_Snapshot': self._latest_snapshot(candidate),
            })
        return pd.DataFrame(rows)

    def _detailed_table(self):
        rows = []
        for candidate in sorted(self.agg.candidate_stats):
            entry = self.agg.candidate_stats[candidate]
            info = self.candidates_info.get(candidate, {})
            snapshots = self._snapshots(candidate)
            row = {
                'Candidate': candidate,
                'Folder_Name': entry['FolderName'],
                'Full_Path': entry['FullPath'],
                'Latest_Snapshot': snapshots.get('latest', '') or '',
                'First_Snapshot': snapshots.get('first', '') or '',
                'Snapshot_Count': snapshots.get('count', 0),
                'Snapshot_By_Seed': snapshot_by_seed_str(snapshots.get('per_seed')),
                'Accuracy': entry.get('accuracy', np.nan),
                'Accuracy_Std': entry.get('accuracy_std', np.nan),
                'Major_Accuracy': entry.get('major_accuracy', np.nan),
                'Time_ms': entry.get('time_ms', np.nan),
                'Gen_ms': entry.get('fm_ms', np.nan),
                'Proj_ms': entry.get('proj_ms', np.nan),
                'NFE_effective': entry.get('nfe_effective', np.nan),
                'CollisionFree': entry.get('collision_free', np.nan),
                'PhysSafe': entry.get('phys_safe', np.nan),
                'GoalDist_m': entry.get('goal_dist', np.nan),
                'N_Variants': entry.get('n_variants', np.nan),
                'N_Geos': entry.get('n_geos', np.nan),
                'N_Seeds': entry.get('n_seeds', np.nan),
                'Missing_Seeds': str(info.get('missing_seeds', []) or ''),
            }
            for axis in AXIS_COLUMNS:
                row[axis] = entry.get(axis, '')
            rows.append(row)
        return pd.DataFrame(rows)

    def _per_variant_table(self):
        """Flat per (candidate, split, geo, variant) headline table, both masks."""
        table = self.agg.agg_long
        if table is None or table.empty:
            return pd.DataFrame()
        wanted = [m for m in PRIMARY_METRICS if m in set(table['metric'])]
        rows = table[table['metric'].isin(wanted)].copy()
        index = [c for c in ['Candidate', 'FolderName', 'scene', 'engine', 'K',
                             'split', 'geo', 'variant', 'tightened', 'mask']
                 if c in rows.columns]
        pivot = rows.pivot_table(index=index, columns='metric', values='mean',
                                 observed=True)
        counts = rows.groupby(index, observed=True)['n'].max()
        pivot['n_rollouts'] = counts
        ordered = [m for m in PRIMARY_METRICS if m in pivot.columns] + ['n_rollouts']
        return pivot[ordered].reset_index()

    # ── text summary ─────────────────────────────────────────────────────────
    def save_summary_txt(self, path):
        lines = [
            '=' * 78,
            'DA_UAV_v1 — GEN15 UAV MIX-ML CROSS-CANDIDATE SUMMARY',
            '=' * 78,
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        ]
        for key, value in self.run_meta.items():
            lines.append(f'{key}: {value}')

        lines += self._batch_shape_lines()
        lines += self._candidate_lines()
        lines += self._ranking_lines()
        lines += self._k_sweep_lines()
        lines += self._quality_lines()
        lines += self._notes_lines()

        with open(path, 'w') as f:
            f.write('\n'.join(str(line) for line in lines))
        self.written.append(path)
        logger.info(f'Summary written: {path}')

    def _batch_shape_lines(self):
        """What is in this batch, before any number is read."""
        lines = ['', 'BATCH SHAPE', '-' * 78]
        axes = self._axes_table()
        if axes.empty:
            lines.append('  (no candidates)')
            return lines
        for column, label in (('scene', 'scenes'), ('engine', 'engines'),
                              ('K', 'K values'), ('controller', 'controllers'),
                              ('generation', 'generations')):
            values = sorted({str(v) for v in axes[column].tolist() if str(v) not in ('', 'None')})
            lines.append(f'  {label:14s} {", ".join(values) if values else "(none parsed)"}')
        lines.append(f'  {"candidates":14s} {len(axes)}')
        unparsed = axes[axes['engine'].astype(str) == '']
        if not unparsed.empty:
            lines.append(f'  WARNING {len(unparsed)} candidate(s) have no parsable eval tag — '
                         f'their scene/engine/K columns are blank:')
            for _, row in unparsed.iterrows():
                lines.append(f'    C{row["Candidate"]}  {row["Eval_Tag"]}')
        return lines

    def _candidate_lines(self):
        lines = ['', 'CANDIDATES', '-' * 78]
        for candidate in sorted(self.candidates_info):
            info = self.candidates_info[candidate]
            axes = info.get('axes', {}) or {}
            display = info.get('custom_name') or info.get('display') or info['name']
            lines.append(f'  {candidate}: {display}')
            lines.append(f'      {info["path"]}')
            lines.append(f'      scene={axes.get("scene") or "?"}  '
                         f'engine={axes.get("engine") or "?"}  K={axes.get("K")}  '
                         f'mpc={axes.get("mpc_batch")}  ctrl={axes.get("controller") or "?"}  '
                         f'T={axes.get("threshold")}')
            lines.append(f'      seeds={info["seeds"]}'
                         + (f'  MISSING={info["missing_seeds"]}'
                            if info.get('missing_seeds') else ''))
            snapshots = info.get('snapshots') or {}
            if snapshots.get('latest'):
                lines.append(
                    f'      last run {format_snapshot_ts(snapshots["latest"])}'
                    f'   ({snapshots["count"]} config snapshot(s) over '
                    f'{snapshots["n_seeds_stamped"]} seed(s))')
                per_seed = snapshots.get('per_seed') or {}
                if len(set(per_seed.values())) > 1:
                    lines.append(f'      per seed: {snapshot_by_seed_str(per_seed)}')
            else:
                lines.append('      last run unknown (no config_snapshot marker)')
        return lines

    def _ranking_lines(self):
        front = set(self.agg.pareto_front())
        lines = ['', 'RANKING (success + constraint, mask=all, all geos pooled)', '-' * 78]
        if not self.agg.ranked_candidates:
            lines.append('  (nothing rankable — check logs/loading.log)')
        for rank, (candidate, accuracy) in enumerate(self.agg.ranked_candidates, 1):
            entry = self.agg.candidate_stats[candidate]
            tag = '   [PARETO FRONT]' if candidate in front else ''
            lines.append(f'  {rank}. Candidate {candidate} — {entry["FolderName"]}{tag}')
            lines.append(f'      success+constraint {accuracy * 100:6.2f}%   '
                         f'success {entry.get("success_rate", np.nan) * 100:6.2f}%   '
                         f'collision-free {entry.get("collision_free", np.nan) * 100:6.2f}%   '
                         f'{entry.get("time_ms", np.nan):8.1f} ms/replan')
            lines.append(f'      gen {entry.get("fm_ms", np.nan):6.1f} ms   '
                         f'proj {entry.get("proj_ms", np.nan):6.1f} ms   '
                         f'over-budget {entry.get("over_budget_frac", np.nan) * 100:5.1f}%   '
                         f'cb-tripped {entry.get("cb_tripped", np.nan) * 100:5.1f}%   '
                         f'NFE~{entry.get("nfe_effective", np.nan):.1f}')
            major = entry.get('major_metrics') or {}
            if major:
                lines.append('      major arms: ' + ', '.join(
                    f'{name} {value * 100:.1f}%' for name, value in sorted(major.items())))
        lines += [
            '',
            '  Pareto note: "FRONT" means not dominated on (success+constraint up,',
            '  ms/replan down) within THIS batch. A candidate off the front that is',
            '  cheaper OR more accurate but not both is a TRADE-OFF, not a loss, and',
            '  neither is "best" — say non-dominated, or name the axis.',
        ]
        return lines

    def _k_sweep_lines(self):
        """The Gen15 headline: success and cost as K falls, per engine.

        Printed as a compact grid because the question is shaped like one — does
        the curve stay flat as K drops (the claim) or fall off a cliff (the null
        result). Rows that pool more than one candidate at the same K are marked,
        since something other than K then differs between the points.
        """
        lines = ['', 'K SWEEP (mask=all, pooled over seeds)', '-' * 78]
        table = self.agg.k_sweep
        if table is None or table.empty:
            lines.append('  (no K axis in this batch — nothing to sweep)')
            return lines
        rows = table[table['mask'] == 'all']
        for metric, label in ((ACCURACY_METRIC, 'success+constraint'),
                              ('n_success', 'success'),
                              (TIME_METRIC, 'ms/replan'),
                              (MASK_FLAG_COLUMN, 'cb-tripped rate')):
            block = rows[rows['metric'] == metric]
            if block.empty:
                continue
            lines.append(f'  {label}:')
            group_keys = [k for k in ('scene', 'engine', 'variant') if k in block.columns]
            for keys, cell in block.groupby(group_keys, observed=True):
                keys = keys if isinstance(keys, tuple) else (keys,)
                ordered = cell.sort_values('K')
                points = '  '.join(
                    f'K{int(r["K"])}={_fmt_metric(metric, r["mean"])}'
                    + ('*' if int(r.get('n_candidates', 1) or 1) > 1 else '')
                    for _, r in ordered.iterrows())
                lines.append(f'    {"/".join(str(k) for k in keys):46s} {points}')
        lines.append('  * = that K cell pools more than one candidate '
                     '(see the `candidates` column in uav_k_sweep.csv)')
        return lines

    def _quality_lines(self):
        lines = ['', 'DATA QUALITY (units with a problem)', '-' * 78]
        quality = self.agg.quality
        if quality is None or quality.empty:
            lines.append('  (no quality rows)')
            return lines
        flags = pd.Series(False, index=quality.index)
        for column in ('n_cb_tripped', 'cb_sentinel', 'timing_missing'):
            if column in quality.columns:
                flags |= quality[column].fillna(0) > 0
        if 'npz_complete' in quality.columns:
            flags |= quality['npz_complete'].fillna(1) == 0
        flagged = quality[flags]
        if flagged.empty:
            lines.append('  none — no circuit-breaker trips, no missing timing, '
                         'no partial npz.')
            return lines
        lines.append(f'  {len(flagged)} of {len(quality)} units flagged:')
        for _, row in flagged.iterrows():
            marks = []
            if row.get('n_cb_tripped', 0):
                marks.append(f'cb_tripped={int(row["n_cb_tripped"])}/'
                             f'{int(row.get("n_rollouts", 0))}')
            if row.get('cb_sentinel', 0):
                marks.append('SENTINEL')
            if row.get('timing_missing', 0):
                marks.append('NO TIMING')
            if row.get('npz_complete') == 0:
                marks.append('PARTIAL NPZ')
            lines.append(
                f'    cand{row["Candidate"]}/seed{row["seed"]}/'
                f'{row["geo"]}/{row["variant"]}: ' + '  '.join(marks))
        return lines

    def _notes_lines(self):
        return ['', 'NOTES', '-' * 78,
                '  mask=all         every rollout',
                '  mask=proj_valid  rollouts whose projection circuit breaker never',
                '    opened. A tripped rollout ran (partly) UNPROJECTED (sustained',
                '    SLSQP slowness, projection.py Fix_15.2), so its constraint',
                '    numbers describe a policy the variant name does not name.',
                '  avg_time / fm_ms / proj_ms come from diagnostics/*.json ONLY —',
                '    eval_artifacts.save_npz never persists the timing group. A unit',
                '    with timing_missing=1 in data_quality.csv has no time axis.',
                '  n_steps early-stops on goal-reach and runs the FULL budget on a',
                '    miss (U_13), so it measures misses as much as speed. Read',
                '    steps_to_goal (reaching episodes only) for time-to-goal.',
                '  scene=empty has a RANDOM per-episode start/goal the state-only',
                '    policy is never told, so its goal_* columns are not a policy',
                '    failure — its success is stable/safe flight only.',
                '  hardflow_new* evaluates the network TWICE per ODE step, so at the',
                '    same K it spends 2x the generation budget of a DPCC arm. Quote',
                '    nfe_effective (uav_units_long.csv, n=1 rows), not K, when the',
                '    two are compared.',
                '  There is NO diffusion-DPCC UAV checkpoint from Gen11 (PLAN §1.5).',
                '    A Gen15 `diffusion`-engine candidate is the first one; without',
                '    it in the batch, the strongest available claim is "vs naive FM',
                '    + DPCC", never "beats DPCC".',
                '=' * 78]

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _missing_seeds_map(self):
        return {key: (str(info.get('missing_seeds')) if info.get('missing_seeds') else '')
                for key, info in self.candidates_info.items()}

    def _snapshots(self, candidate):
        return (self.candidates_info.get(candidate, {}) or {}).get('snapshots') or {}

    def _latest_snapshot(self, candidate):
        return self._snapshots(candidate).get('latest', '') or ''

    def _latest_snapshot_map(self):
        return {key: self._latest_snapshot(key) for key in self.candidates_info}

    def _seed_snapshot_column(self, rows):
        """Per-row stamp for a frame carrying a seed axis.

        Deliberately no fallback to the candidate's `latest`: a seed that was
        never re-run must stay visibly blank instead of borrowing a sibling
        seed's freshness. UAV seeds are separate SLURM jobs, so this is common.
        """
        per_candidate = {key: (self._snapshots(key).get('per_seed') or {})
                         for key in self.candidates_info}

        def stamp(candidate, seed):
            try:
                return per_candidate.get(candidate, {}).get(int(seed), '') or ''
            except (TypeError, ValueError):
                return ''

        return [stamp(c, s) for c, s in zip(rows['Candidate'], rows['seed'])]

    def _with_seed_snapshot(self, table):
        if table is None or table.empty or 'seed' not in table.columns:
            return table
        out = table.copy()
        out['LatestSnapshot'] = self._seed_snapshot_column(out)
        return out

    def _write(self, output_dir, filename, table):
        path = os.path.join(output_dir, filename)
        if table is None or (hasattr(table, 'empty') and table.empty):
            logger.warning(f'{filename}: no rows — writing header-only file')
            pd.DataFrame(table if table is not None else []).to_csv(path, index=False)
        else:
            table.to_csv(path, index=False)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f'{filename}: {len(table)} rows, {size_mb:.1f} MB')
            # The HTML viewers parse these in the browser; a merge of several
            # trees can produce a file their CSV parser cannot hold.
            if size_mb > 50:
                logger.warning(
                    f'{filename} is {size_mb:.0f} MB — the HTML viewer may fail to '
                    f'load it ("out of memory" while tokenizing). Narrow the run '
                    f'with --scenes / --engines / --k / --variants, or point '
                    f'--parent-path at fewer trees.')
        self.written.append(path)


def _compat_env(rows):
    """geo, tagged with the split when it is not the plain test set."""
    geo = rows['geo'].astype(str)
    return np.where(rows['split'] == 'train', geo + '@train_set', geo)


def _fmt_metric(metric, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n/a'
    if metric in PERCENTAGE_METRICS:
        return f'{value * 100:.0f}%'
    return f'{value:.4g}'


def format_metric(metric, value):
    """Human formatting used by the text summary and plot labels."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n/a'
    if metric in PERCENTAGE_METRICS:
        return f'{value * 100:.1f}%'
    arrow = ' (lower better)' if metric in LOWER_IS_BETTER else ''
    return f'{value:.4g}{arrow}'
