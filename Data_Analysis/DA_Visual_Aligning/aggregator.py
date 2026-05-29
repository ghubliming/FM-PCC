"""
Data aggregation for Visual Aligning DA.

Two-level output per variant:
  per_rollout — raw (N_rollouts,) arrays
  summary     — mean/std/min/max scalars for each metric
"""
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_ARRAY_METRICS = [
    'n_success', 'n_steps', 'avg_time',
    'mean_dist_per_rollout', 'max_phys_error_per_rollout',
    'context_init_xy_dist',
]

_SCALAR_METRICS = ['success_rate', 'entropy', 'elapsed_seconds', 'seed']


class DataAggregator:
    """Aggregate single-seed per-rollout data for one candidate."""

    def __init__(self, raw_data):
        """
        Args:
            raw_data: {variant: metrics_dict} from DataLoader.load_results()
        """
        self.raw_data   = raw_data
        self.aggregated = {}

    def aggregate_all(self):
        """
        Returns:
            {
                'per_rollout': {variant: {metric: (N,) array}},
                'summary':     {variant: {metric: {mean, std, min, max, n}}},
            }
        """
        logger.info('Aggregating visual aligning data...')
        result = {
            'per_rollout': self._extract_per_rollout(),
            'summary':     self._compute_summary(),
        }
        self.aggregated = result
        logger.info('Aggregation complete')
        return result

    # ------------------------------------------------------------------
    def _extract_per_rollout(self):
        """Return {variant: {metric: (N,) array}} — raw rollout arrays."""
        out = {}
        for variant, metrics in self.raw_data.items():
            out[variant] = {}
            for k, v in metrics.items():
                arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
                if arr.ndim >= 1 and arr.shape[0] >= 1:
                    out[variant][k] = arr
                # scalar / tiny arrays stored as-is under summary, not per_rollout
        return out

    def _compute_summary(self):
        """Return {variant: {metric: {mean, std, min, max, n}}}."""
        out = {}
        for variant, metrics in self.raw_data.items():
            out[variant] = {}
            for k, v in metrics.items():
                arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
                if arr.ndim == 0 or arr.size == 1:
                    out[variant][k] = {'mean': float(arr.flat[0]), 'std': 0.0,
                                       'min': float(arr.flat[0]), 'max': float(arr.flat[0]), 'n': 1}
                elif arr.ndim == 1:
                    valid = arr[~np.isnan(arr.astype(float))]
                    if len(valid) == 0:
                        continue
                    out[variant][k] = {
                        'mean': float(np.mean(valid)),
                        'std':  float(np.std(valid)),
                        'min':  float(np.min(valid)),
                        'max':  float(np.max(valid)),
                        'n':    len(valid),
                    }
                # 2-D arrays (context_box_init_xy etc.) skipped for summary
        return out

    # ------------------------------------------------------------------
    @property
    def per_rollout(self):
        return self.aggregated.get('per_rollout', {})

    @property
    def summary(self):
        return self.aggregated.get('summary', {})

    def get_variant_ranking(self, metric='success_rate', ascending=False):
        """Rank variants by a summary metric. Returns list of (variant, mean) tuples."""
        pairs = []
        for variant, metrics in self.summary.items():
            if metric in metrics:
                pairs.append((variant, metrics[metric]['mean']))
        pairs.sort(key=lambda x: x[1], reverse=not ascending)
        return pairs

    def to_summary_dataframe(self):
        """Flatten summary into a DataFrame: variant × metric with mean/std columns."""
        rows = []
        for variant, metrics in self.summary.items():
            for metric, stats in metrics.items():
                rows.append({'variant': variant, 'metric': metric, **stats})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # CONSTRAINT-DEACTIVATED
    # def aggregate_by_constraint(self): ...
    # def aggregate_by_halfspace(self): ...
    #
    # MULTI-SEED-DEACTIVATED
    # def aggregate_across_seeds(self, seed_data_list): ...
