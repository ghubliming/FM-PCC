"""
Batch Aggregator for Visual Aligning DA.

Aggregates each candidate independently via DataAggregator,
then assembles cross-candidate stats for comparison plots.
"""
import logging
import numpy as np
from aggregator import DataAggregator

logger = logging.getLogger(__name__)


class BatchAggregator:
    """Cross-candidate statistics for visual aligning."""

    def __init__(self):
        self.batch_data          = {}
        self.candidate_aggregators = {}
        self.candidate_stats       = {}
        # MULTI-SEED-DEACTIVATED: ranked_candidates kept as attribute for reporter compat
        self.ranked_candidates     = []

    def aggregate_all_candidates(self, batch_data):
        """
        Args:
            batch_data: {letter: {variant: metrics_dict}} from BatchDataLoader

        Returns:
            {letter: stats_dict}
        """
        logger.info('Aggregating all candidates...')
        self.batch_data = batch_data
        self.candidate_aggregators = {}
        self.candidate_stats = {}

        for letter, candidate_data in sorted(batch_data.items()):
            logger.info(f'Aggregating candidate {letter}...')
            try:
                agg = DataAggregator(candidate_data)
                agg.aggregate_all()
                self.candidate_aggregators[letter] = agg
                self.candidate_stats[letter] = self._extract_stats(agg, letter)
            except Exception as e:
                logger.error(f'Failed candidate {letter}: {e}')
                self.candidate_stats[letter] = {'error': str(e)}

        self.ranked_candidates = self._rank_by_success()
        logger.info(f'Batch aggregation done: {len(self.candidate_stats)} candidates')
        return self.candidate_stats

    # ------------------------------------------------------------------
    def _extract_stats(self, agg, letter):
        """Pull key scalars from a DataAggregator for cross-candidate plotting."""
        stats = {'letter': letter}
        summary = agg.summary

        if not summary:
            return stats

        # Collect per-variant scalar means
        success_rates, mean_dists, avg_times, phys_errors = [], [], [], []

        for variant, metrics in summary.items():
            if 'n_success' in metrics:
                success_rates.append(metrics['n_success']['mean'])
            elif 'success_rate' in metrics:
                success_rates.append(metrics['success_rate']['mean'])
            if 'mean_dist_per_rollout' in metrics:
                mean_dists.append(metrics['mean_dist_per_rollout']['mean'])
            if 'avg_time' in metrics:
                avg_times.append(metrics['avg_time']['mean'])
            if 'max_phys_error_per_rollout' in metrics:
                phys_errors.append(metrics['max_phys_error_per_rollout']['mean'])

        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else None

        stats['success_rate']     = _safe_mean(success_rates)
        stats['mean_distance']    = _safe_mean(mean_dists)
        stats['avg_time']         = _safe_mean(avg_times)
        stats['max_phys_error']   = _safe_mean(phys_errors)

        # Per-variant breakdown for reporter
        stats['per_variant'] = {}
        for variant, metrics in summary.items():
            stats['per_variant'][variant] = {
                'success_rate':   metrics.get('n_success', {}).get('mean') or
                                   metrics.get('success_rate', {}).get('mean'),
                'mean_distance':  metrics.get('mean_dist_per_rollout', {}).get('mean'),
                'avg_time':       metrics.get('avg_time', {}).get('mean'),
                'max_phys_error': metrics.get('max_phys_error_per_rollout', {}).get('mean'),
                'n_steps':        metrics.get('n_steps', {}).get('mean'),
            }

        return stats

    def _rank_by_success(self):
        pairs = []
        for letter, stats in self.candidate_stats.items():
            sr = stats.get('success_rate')
            if sr is not None and 'error' not in stats:
                pairs.append((letter, sr))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def get_candidate_aggregator(self, letter):
        return self.candidate_aggregators.get(letter)

    def print_ranking_summary(self):
        lines = ['=== Candidate Rankings (by success rate) ===']
        for rank, (letter, sr) in enumerate(self.ranked_candidates, 1):
            lines.append(f'  {rank}. Candidate {letter}: {sr*100:.1f}%')
        summary = '\n'.join(lines)
        logger.info('\n' + summary)
        return summary

    # MULTI-SEED-DEACTIVATED: cross-seed ranking removed.
