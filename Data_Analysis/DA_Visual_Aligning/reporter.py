"""
Reporter for a single Visual Aligning candidate.

Generates summary TXT and CSV. No violation/halfspace sections.
"""
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Reporter:
    """Generate summary report for one candidate."""

    def __init__(self, aggregator):
        self.aggregator = aggregator

    def save_all_reports(self, output_dir, data_loader_summary=None):
        os.makedirs(output_dir, exist_ok=True)
        self.save_summary_txt(os.path.join(output_dir, 'results_summary.txt'), data_loader_summary)
        self.save_per_rollout_csv(os.path.join(output_dir, 'per_rollout_detail.csv'))
        self.save_summary_csv(os.path.join(output_dir, 'results_by_variant.csv'))
        logger.info('Reports saved.')

    # ------------------------------------------------------------------
    def save_summary_txt(self, output_path, data_loader_summary=None):
        summary = self.aggregator.summary
        with open(output_path, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write('FM Visual Aligning — Analysis Summary\n')
            f.write('=' * 80 + '\n\n')
            f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

            if data_loader_summary:
                f.write('--- Data Loading ---\n')
                f.write(f'Source:        {data_loader_summary.get("source", "npz")}\n')
                f.write(f'Files Found:   {data_loader_summary.get("files_found", "?")}\n')
                f.write(f'Files Loaded:  {data_loader_summary.get("files_loaded", "?")}\n')
                f.write(f'Files Failed:  {data_loader_summary.get("files_failed", "?")}\n\n')

            # Success ranking
            ranking = self.aggregator.get_variant_ranking('n_success')
            if not ranking:
                ranking = self.aggregator.get_variant_ranking('success_rate')
            f.write('--- Variant Rankings by Success Rate ---\n')
            for rank, (variant, val) in enumerate(ranking, 1):
                f.write(f'  {rank:2d}. {variant:30s}  {val*100:5.1f}%\n')
            f.write('\n')

            # Key metric table
            f.write('--- Key Metrics per Variant ---\n')
            header = f'{"Variant":30s} {"Succ%":>6s} {"MeanDist":>9s} {"AvgTime":>8s} {"MaxErr":>8s} {"Steps":>7s}\n'
            f.write(header)
            f.write('-' * len(header) + '\n')
            for variant in sorted(summary.keys()):
                m = summary[variant]
                sr  = (m.get('n_success',   m.get('success_rate', {})).get('mean', np.nan)) * 100
                md  = m.get('mean_dist_per_rollout',     {}).get('mean', np.nan)
                at  = m.get('avg_time',                  {}).get('mean', np.nan)
                pe  = m.get('max_phys_error_per_rollout',{}).get('mean', np.nan)
                st  = m.get('n_steps',                   {}).get('mean', np.nan)
                f.write(f'  {variant:28s}  {sr:5.1f}%  {md:8.4f}  {at:7.4f}  {pe:7.4f}  {st:6.1f}\n')
            f.write('\n' + '=' * 80 + '\n')
        logger.info(f'Summary TXT saved: {output_path}')

    # ------------------------------------------------------------------
    def save_per_rollout_csv(self, output_path):
        """Flat CSV: variant | rollout_idx | success | mean_dist | steps | phys_err | context_xy_dist"""
        per = self.aggregator.per_rollout
        rows = []
        for variant in sorted(per.keys()):
            pv    = per[variant]
            n_suc = pv.get('n_success', np.array([]))
            n_st  = pv.get('n_steps',   np.array([]))
            md    = pv.get('mean_dist_per_rollout',      np.array([]))
            pe    = pv.get('max_phys_error_per_rollout', np.array([]))
            cxy   = pv.get('context_init_xy_dist',       np.array([]))
            at    = pv.get('avg_time',                   np.array([]))

            n_rollouts = max(len(a) for a in [n_suc, n_st, md, pe, cxy, at] if len(a) > 0) if any(len(a) > 0 for a in [n_suc, n_st, md, pe, cxy, at]) else 0

            def _get(arr, i):
                return float(arr[i]) if len(arr) > i else np.nan

            for i in range(n_rollouts):
                rows.append({
                    'variant':          variant,
                    'rollout_idx':      i,
                    'success':          int(_get(n_suc, i)),
                    'mean_dist_m':      _get(md, i),
                    'steps':            int(_get(n_st, i)) if not np.isnan(_get(n_st, i)) else np.nan,
                    'phys_err_m':       _get(pe, i),
                    'context_xy_dist_m': _get(cxy, i),
                    'avg_time_s':       _get(at, i),
                })
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f'Per-rollout CSV saved: {output_path} ({len(df)} rows)')

    # ------------------------------------------------------------------
    def save_summary_csv(self, output_path):
        df = self.aggregator.to_summary_dataframe()
        if df is not None and not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f'Summary CSV saved: {output_path}')
        else:
            logger.warning('No summary data to save')
