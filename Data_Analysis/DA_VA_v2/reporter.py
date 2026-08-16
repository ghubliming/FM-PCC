"""
DA_VA_v2 — CSV / TXT reporting.

The CSVs are the product; the PNGs are optional garnish. Three families:

native      per_rollout_detail.csv, va2_units_long.csv, va2_aggregated_long.csv,
            data_quality.csv, run_config.csv — the full (candidate, seed, split,
            geo, variant, mask, metric) cube, nothing collapsed.

DA_Code_v3  candidates_multidimensional_raw.csv / _aggregated.csv,
compat      candidates_ranking.csv, candidates_detailed.csv — same column names
            the existing `Data_Analysis/Visualizer/index.html` reads, so a
            visual-aligning batch can be opened in it unchanged.

Axis mapping for the DA_Code_v3-compat files (that schema has no geometry or
split axis of its own):
    halfspace_variant  ←  geo, suffixed `@train_set` for --eval-on-train rows
    constraint_type    ←  split ('test' / 'train')
The suffix keeps a train-set run from being pooled with a test run of the same
geometry in a viewer that only knows about `halfspace_variant`.
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import PERCENTAGE_METRICS, PRIMARY_METRICS
from discovery import format_snapshot_ts, snapshot_by_seed_str

logger = logging.getLogger(__name__)


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
        self._write(output_dir, 'va2_units_long.csv', self._units_long_table())
        self._write(output_dir, 'va2_aggregated_long.csv', self._agg_long_table())
        self._write(output_dir, 'data_quality.csv', self._quality_table())
        self._write(output_dir, 'run_config.csv', self.agg.run_config)

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
        lead = ['Candidate', 'FolderName', 'seed', 'split', 'geo', 'geo_base',
                'tightened', 'variant', 'variant_raw', 'rollout_idx', 'frozen']
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

        This is the column `Visualizer_VA_v2` reads for its "Last Run" columns;
        `va2_units_long.csv` keeps the per-seed breakdown.
        """
        table = self.agg.agg_long
        if table is None or table.empty:
            return table
        out = table.copy()
        out['LatestSnapshot'] = out['Candidate'].map(self._latest_snapshot_map()).fillna('')
        return out

    def _quality_table(self):
        """One row per unit. Read this before believing any constraint number.

        `n_frozen`      rollouts the D1 box-obstacle guard held in place without
                        ever calling the model — they report sat_rate 1.0 and
                        inflate constraint aggregates (spec §5.3).
        `n_cb_tripped`  rollouts where the projector's circuit breaker opened;
                        those did not run the policy the variant name claims.
        `npz_complete`  0 means the npz is a crash-safety partial.
        """
        table = self.agg.quality
        if table is None or table.empty:
            return table
        lead = ['Candidate', 'FolderName', 'seed', 'split', 'geo', 'variant',
                'source', 'legacy', 'legacy_kind', 'has_projector',
                'n_rollouts', 'n_frozen', 'frozen_rate',
                'n_cb_tripped', 'cb_skipped_steps', 'npz_complete']
        lead = [c for c in lead if c in table.columns]
        rest = [c for c in table.columns if c not in lead]
        return table[lead + rest]

    # ── DA_Code_v3-compatible tables ─────────────────────────────────────────
    def _compat_raw_table(self):
        table = self.agg.units_long
        if table is None or table.empty:
            return pd.DataFrame()
        rows = table[table['mask'] == 'all'].copy()
        out = pd.DataFrame({
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
        return out

    def _compat_aggregated_table(self):
        table = self.agg.agg_long
        if table is None or table.empty:
            return pd.DataFrame()
        rows = table[table['mask'] == 'all'].copy()
        out = pd.DataFrame({
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
        return out

    # ── candidate-level tables ───────────────────────────────────────────────
    def _ranking_table(self):
        rows = []
        for rank, (candidate, accuracy) in enumerate(self.agg.ranked_candidates, 1):
            entry = self.agg.candidate_stats[candidate]
            info = self.candidates_info.get(candidate, {})
            rows.append({
                'Rank': rank,
                'Candidate': candidate,
                'Folder': entry['FolderName'],
                'Accuracy (%)': accuracy * 100,
                'Accuracy Std (%)': entry.get('accuracy_std', np.nan) * 100,
                'Success (%)': entry.get('success_rate', np.nan) * 100,
                'CollisionFree (%)': entry.get('collision_free', np.nan) * 100,
                'Time (ms)': entry.get('time_ms', np.nan),
                'MeanDist_m': entry.get('mean_distance', np.nan),
                'MaxPhysErr_m': entry.get('max_phys_error', np.nan),
                'Robustness': entry.get('robustness', np.nan),
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
            rows.append({
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
                'MeanDist_m': entry.get('mean_distance', np.nan),
                'CollisionFree': entry.get('collision_free', np.nan),
                'N_Variants': entry.get('n_variants', np.nan),
                'N_Geos': entry.get('n_geos', np.nan),
                'N_Seeds': entry.get('n_seeds', np.nan),
                'Missing_Seeds': str(info.get('missing_seeds', []) or ''),
            })
        return pd.DataFrame(rows)

    def _per_variant_table(self):
        """Flat per (candidate, split, geo, variant) headline table, both masks."""
        table = self.agg.agg_long
        if table is None or table.empty:
            return pd.DataFrame()
        wanted = [m for m in PRIMARY_METRICS if m in set(table['metric'])]
        rows = table[table['metric'].isin(wanted)].copy()
        pivot = rows.pivot_table(
            index=['Candidate', 'FolderName', 'split', 'geo', 'tightened',
                   'variant', 'mask'],
            columns='metric', values='mean', observed=True)
        counts = rows.groupby(
            ['Candidate', 'FolderName', 'split', 'geo', 'tightened', 'variant', 'mask'],
            observed=True)['n'].max()
        pivot['n_rollouts'] = counts
        ordered = [m for m in PRIMARY_METRICS if m in pivot.columns] + ['n_rollouts']
        return pivot[ordered].reset_index()

    # ── text summary ─────────────────────────────────────────────────────────
    def save_summary_txt(self, path):
        lines = [
            '=' * 78,
            'DA_VA_v2 — VISUAL ALIGNING CROSS-CANDIDATE SUMMARY',
            '=' * 78,
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        ]
        for key, value in self.run_meta.items():
            lines.append(f'{key}: {value}')
        lines += ['', 'CANDIDATES', '-' * 78]
        for candidate in sorted(self.candidates_info):
            info = self.candidates_info[candidate]
            display = info.get('custom_name') or info['name']
            lines.append(f'  {candidate}: {display}')
            lines.append(f'      {info["path"]}')
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

        lines += ['', 'RANKING (goal + constraint success, mask=all, all geos pooled)',
                  '-' * 78]
        if not self.agg.ranked_candidates:
            lines.append('  (nothing rankable — check logs/loading.log)')
        for rank, (candidate, accuracy) in enumerate(self.agg.ranked_candidates, 1):
            entry = self.agg.candidate_stats[candidate]
            info = self.candidates_info.get(candidate, {})
            tag = ''
            if info.get('legacy'):
                tag = f'   [LEGACY {info.get("legacy_kind") or "unknown"}]'
            if not info.get('has_projector', True):
                # Its "goal+constraint" number is really plain goal success — the
                # aggregator falls back when the constraint column is all-NaN.
                tag += '   [no projector: ranked on goal success only]'
            lines.append(f'  {rank}. Candidate {candidate} — {entry["FolderName"]}{tag}')
            lines.append(f'      goal+constraint {accuracy * 100:6.2f}%   '
                         f'goal {entry.get("success_rate", np.nan) * 100:6.2f}%   '
                         f'collision-free {entry.get("collision_free", np.nan) * 100:6.2f}%   '
                         f'{entry.get("time_ms", np.nan):8.1f} ms/replan')
            major = entry.get('major_metrics') or {}
            if major:
                lines.append('      major arms: ' + ', '.join(
                    f'{name} {value * 100:.1f}%' for name, value in sorted(major.items())))

        lines += ['', 'DATA QUALITY (units with a problem)', '-' * 78]
        quality = self.agg.quality
        if quality is not None and not quality.empty:
            flags = pd.Series(False, index=quality.index)
            for column in ('n_frozen', 'n_cb_tripped'):
                if column in quality.columns:
                    flags |= quality[column].fillna(0) > 0
            if 'npz_complete' in quality.columns:
                flags |= quality['npz_complete'].fillna(1) == 0
            flagged = quality[flags]
            if flagged.empty:
                lines.append('  none — no frozen rollouts, no circuit-breaker trips.')
            else:
                lines.append(f'  {len(flagged)} of {len(quality)} units flagged:')
                for _, row in flagged.iterrows():
                    lines.append(
                        f'    cand{row["Candidate"]}/seed{row["seed"]}/{row["split"]}/'
                        f'{row["geo"]}/{row["variant"]}: '
                        f'frozen={int(row.get("n_frozen", 0))}/{int(row.get("n_rollouts", 0))} '
                        f'cb_tripped={int(row.get("n_cb_tripped", 0))} '
                        f'complete={row.get("npz_complete")}')

        lines += ['', 'NOTES', '-' * 78,
                  '  LatestSnapshot / Latest_Snapshot  newest config_snapshot marker,',
                  '    i.e. when that folder was last (re)run. Per SEED in the units /',
                  '    *_raw files, newest over all seeds in the aggregated ones.',
                  '  mask=all      every rollout',
                  '  mask=unfrozen D1 box-obstacle-conflict rollouts removed',
                  '  total_violations is NaN for visual runs — that eval records',
                  '    per-family maxima, never a cumulative sum. Use',
                  '    constraint_exec_total_viol_count / max_viol_depth_m instead.',
                  '  legacy=1 units come from a `_`-prefixed bridged tree (e.g. the',
                  '    D3IL baseline via bridge_d3il_va_to_da_va_v2.py), not from a',
                  '    Gen14-shaped eval. has_projector=0 means every constraint_*',
                  '    column is absent BY DESIGN — that pipeline runs no projector,',
                  '    so it can only be compared on goal success / distance / time.',
                  '=' * 78]

        with open(path, 'w') as f:
            f.write('\n'.join(str(line) for line in lines))
        self.written.append(path)
        logger.info(f'Summary written: {path}')

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _missing_seeds_map(self):
        return {key: (str(info.get('missing_seeds')) if info.get('missing_seeds') else '')
                for key, info in self.candidates_info.items()}

    # ── config-snapshot timestamps ("when was this run produced?") ────────────
    # Filled in by discovery.scan_snapshot_timestamps(). A candidates_info built
    # by older code has no 'snapshots' key — everything below then yields '' and
    # the columns come out blank rather than breaking.
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
        seed's freshness.
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
        """Append the per-seed `LatestSnapshot` column to a frame with a seed axis."""
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
                    f'with --candidates / --variants / --geos / --splits, or point '
                    f'--parent-path at fewer trees.')
        self.written.append(path)


def _compat_env(rows):
    """geo, tagged with the split when it is not the plain test set."""
    geo = rows['geo'].astype(str)
    return np.where(rows['split'] == 'train', geo + '@train_set', geo)


def format_metric(metric, value):
    """Human formatting used by the text summary and plot labels."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n/a'
    if metric in PERCENTAGE_METRICS:
        return f'{value * 100:.1f}%'
    return f'{value:.4g}'
