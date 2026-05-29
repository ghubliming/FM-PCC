"""
Batch Reporter for Visual Aligning DA.

Generates cross-candidate summary TXT and CSVs.
No violation/halfspace sections.
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchReporter:
    """Generate reports from batch aggregation results."""

    def __init__(self, candidate_stats, ranked_candidates, candidates_info=None, aggregator=None):
        self.candidate_stats   = candidate_stats
        self.ranked_candidates = ranked_candidates
        self.candidates_info   = candidates_info or {}
        self.aggregator        = aggregator

    def save_all_reports(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.save_candidates_summary_txt(os.path.join(output_dir, 'candidates_summary.txt'))
        self.save_candidates_ranking_csv(os.path.join(output_dir, 'candidates_ranking.csv'))
        self.save_candidates_detailed_csv(os.path.join(output_dir, 'candidates_detailed.csv'))
        self.save_per_variant_breakdown_csv(os.path.join(output_dir, 'candidates_per_variant.csv'))
        self.save_dynamic_csv(os.path.join(output_dir, 'va_candidates_dynamic.csv'))
        logger.info(f'All reports saved to: {output_dir}')

    def save_dynamic_csv(self, output_path):
        """
        Long-format CSV for the HTML visualizer dynamic plots.
        Columns: Candidate, FolderName, FullPath, variant, metric, mean, std, n
        One row per Candidate × variant × metric combination.
        """
        if not self.aggregator:
            logger.warning('No aggregator — cannot save dynamic CSV.')
            return
        all_dfs = []
        for letter in sorted(self.aggregator.candidate_aggregators.keys()):
            agg  = self.aggregator.candidate_aggregators[letter]
            info = self.candidates_info.get(letter, {})
            df   = agg.to_summary_dataframe()
            if df is not None and not df.empty:
                df = df.copy()
                df['Candidate']  = letter
                df['FolderName'] = info.get('name', 'Unknown')
                df['FullPath']   = info.get('path', 'Unknown')
                all_dfs.append(df)
        if all_dfs:
            out = pd.concat(all_dfs, ignore_index=True)
            # Reorder: Candidate first
            cols = ['Candidate', 'FolderName', 'FullPath', 'variant', 'metric', 'mean', 'std', 'n']
            cols = [c for c in cols if c in out.columns] + [c for c in out.columns if c not in cols]
            out[cols].to_csv(output_path, index=False)
            logger.info(f'Dynamic CSV saved: {output_path} ({len(out)} rows)')
        else:
            logger.warning('No data for dynamic CSV.')

    # ------------------------------------------------------------------
    def save_candidates_summary_txt(self, output_path):
        lines = [
            '=' * 70,
            'FM VISUAL ALIGNING — CROSS-CANDIDATE COMPARISON',
            '=' * 70,
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '',
            'CANDIDATES',
            '-' * 70,
        ]
        for letter in sorted(self.candidates_info.keys()):
            info = self.candidates_info[letter]
            lines.append(f'  {letter}: {info.get("name", "Unknown")}')
        lines.append('')

        lines.append('RANKINGS (by success rate)')
        lines.append('-' * 70)
        for rank, (letter, sr) in enumerate(self.ranked_candidates, 1):
            stats = self.candidate_stats[letter]
            md   = stats.get('mean_distance', np.nan)
            at   = stats.get('avg_time', np.nan)
            line = f'  {rank}. Candidate {letter}: {sr*100:.1f}% success'
            if not np.isnan(md):  line += f', {md:.4f}m avg dist'
            if not np.isnan(at):  line += f', {at:.4f}s avg time'
            lines.append(line)
            # Per-variant breakdown
            pv = stats.get('per_variant', {})
            if pv:
                for variant in sorted(pv.keys()):
                    vm = pv[variant]
                    vsr = vm.get('success_rate')
                    vmd = vm.get('mean_distance')
                    if vsr is not None:
                        lines.append(f'      {variant:30s} {vsr*100:5.1f}% | dist={vmd:.4f}m' if vmd is not None else
                                     f'      {variant:30s} {vsr*100:5.1f}%')
        lines += ['', '=' * 70]

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f'Summary TXT saved: {output_path}')

    def save_candidates_ranking_csv(self, output_path):
        rows = []
        for rank, (letter, sr) in enumerate(self.ranked_candidates, 1):
            stats = self.candidate_stats[letter]
            info  = self.candidates_info.get(letter, {})
            rows.append({
                'Rank':         rank,
                'Candidate':    letter,
                'Folder':       info.get('name', 'Unknown'),
                'SuccessRate':  sr,
                'MeanDist_m':   stats.get('mean_distance', ''),
                'AvgTime_s':    stats.get('avg_time', ''),
                'MaxPhysErr_m': stats.get('max_phys_error', ''),
            })
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f'Ranking CSV saved: {output_path}')

    def save_candidates_detailed_csv(self, output_path):
        rows = []
        for letter in sorted(self.candidate_stats.keys()):
            stats = self.candidate_stats[letter]
            info  = self.candidates_info.get(letter, {})
            if 'error' not in stats:
                rows.append({
                    'Candidate':    letter,
                    'FolderName':   info.get('name', 'Unknown'),
                    'FullPath':     info.get('path', 'Unknown'),
                    'SuccessRate':  stats.get('success_rate', ''),
                    'MeanDist_m':   stats.get('mean_distance', ''),
                    'AvgTime_s':    stats.get('avg_time', ''),
                    'MaxPhysErr_m': stats.get('max_phys_error', ''),
                })
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f'Detailed CSV saved: {output_path}')

    def save_per_variant_breakdown_csv(self, output_path):
        """Flat per-variant summary for all candidates."""
        rows = []
        for letter in sorted(self.candidate_stats.keys()):
            stats = self.candidate_stats[letter]
            if 'error' in stats:
                continue
            info = self.candidates_info.get(letter, {})
            for variant, vm in stats.get('per_variant', {}).items():
                rows.append({
                    'Candidate':    letter,
                    'FolderName':   info.get('name', 'Unknown'),
                    'Variant':      variant,
                    'SuccessRate':  vm.get('success_rate', ''),
                    'MeanDist_m':   vm.get('mean_distance', ''),
                    'AvgTime_s':    vm.get('avg_time', ''),
                    'MaxPhysErr_m': vm.get('max_phys_error', ''),
                    'AvgSteps':     vm.get('n_steps', ''),
                })
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f'Per-variant CSV saved: {output_path}')
