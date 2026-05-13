"""
Data aggregation module: Aggregates results across seeds and generates statistics.
"""
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataAggregator:
    """Aggregate evaluation results across seeds and compute statistics."""
    
    def __init__(self, raw_data):
        """
        Initialize aggregator with raw data.
        
        Args:
            raw_data: Dict from DataLoader.load_results()
        """
        self.raw_data = raw_data
        self.aggregated = {}
    
    def aggregate_all(self):
        """
        Perform all aggregations.
        
        Returns:
            Dict with different aggregation views
        """
        logger.info('Starting data aggregation...')
        
        result = {
            'by_variant': self.aggregate_by_variant(),
            'by_constraint': self.aggregate_by_constraint(),
            'by_halfspace': self.aggregate_by_halfspace(),
            'detailed': self.create_detailed_dataframe(),
        }
        
        self.aggregated = result
        logger.info('Aggregation complete')
        return result
    
    @property
    def aggregated_by_variant(self):
        return self.aggregated.get('by_variant')
    
    @property
    def aggregated_by_constraint(self):
        return self.aggregated.get('by_constraint')
    
    @property
    def aggregated_by_halfspace(self):
        return self.aggregated.get('by_halfspace')
    
    @property
    def detailed_df(self):
        return self.aggregated.get('detailed')
    
    def aggregate_by_variant(self):
        """
        Aggregate results grouped by projection variant.
        Averages across all seeds and constraint types.
        
        Returns:
            DataFrame with columns: variant, metric, mean, std, min, max, count
        """
        logger.info('Aggregating by variant...')
        rows = []
        
        variant_metrics = defaultdict(lambda: defaultdict(list))
        
        # Collect all metric values per variant
        for seed, variants_dict in self.raw_data.items():
            for variant, constraints_dict in variants_dict.items():
                for constraint, halfspaces_dict in constraints_dict.items():
                    for halfspace, metrics_dict in halfspaces_dict.items():
                        for metric_name, metric_value in metrics_dict.items():
                            # Skip array data, use mean/std values
                            if not metric_name.endswith('_array'):
                                variant_metrics[variant][metric_name].append(metric_value)
        
        # Compute statistics
        for variant in sorted(variant_metrics.keys()):
            for metric_name in sorted(variant_metrics[variant].keys()):
                values = np.array(variant_metrics[variant][metric_name])
                values = values[~np.isnan(values)]  # Remove NaN
                
                if len(values) > 0:
                    rows.append({
                        'variant': variant,
                        'metric': metric_name,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values),
                    })
        
        df = pd.DataFrame(rows)
        logger.info(f'Aggregated {len(df)} variant-metric combinations')
        return df
    
    def aggregate_by_constraint(self):
        """
        Aggregate results grouped by constraint type.
        
        Returns:
            DataFrame with columns: constraint_type, metric, mean, std, count
        """
        logger.info('Aggregating by constraint type...')
        rows = []
        
        constraint_metrics = defaultdict(lambda: defaultdict(list))
        
        # Collect all metric values per constraint
        for seed, variants_dict in self.raw_data.items():
            for variant, constraints_dict in variants_dict.items():
                for constraint, halfspaces_dict in constraints_dict.items():
                    for halfspace, metrics_dict in halfspaces_dict.items():
                        for metric_name, metric_value in metrics_dict.items():
                            if not metric_name.endswith('_array'):
                                constraint_metrics[constraint][metric_name].append(metric_value)
        
        # Compute statistics
        for constraint in sorted(constraint_metrics.keys()):
            for metric_name in sorted(constraint_metrics[constraint].keys()):
                values = np.array(constraint_metrics[constraint][metric_name])
                values = values[~np.isnan(values)]
                
                if len(values) > 0:
                    rows.append({
                        'constraint_type': constraint,
                        'metric': metric_name,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values),
                    })
        
        df = pd.DataFrame(rows)
        logger.info(f'Aggregated {len(df)} constraint-metric combinations')
        return df
    
    def aggregate_by_halfspace(self):
        """
        Aggregate results grouped by halfspace variant.
        
        Returns:
            DataFrame with columns: halfspace_variant, metric, mean, std, count
        """
        logger.info('Aggregating by halfspace variant...')
        rows = []
        
        halfspace_metrics = defaultdict(lambda: defaultdict(list))
        
        # Collect all metric values per halfspace
        for seed, variants_dict in self.raw_data.items():
            for variant, constraints_dict in variants_dict.items():
                for constraint, halfspaces_dict in constraints_dict.items():
                    for halfspace, metrics_dict in halfspaces_dict.items():
                        for metric_name, metric_value in metrics_dict.items():
                            if not metric_name.endswith('_array'):
                                halfspace_metrics[halfspace][metric_name].append(metric_value)
        
        # Compute statistics
        for halfspace in sorted(halfspace_metrics.keys()):
            for metric_name in sorted(halfspace_metrics[halfspace].keys()):
                values = np.array(halfspace_metrics[halfspace][metric_name])
                values = values[~np.isnan(values)]
                
                if len(values) > 0:
                    rows.append({
                        'halfspace_variant': halfspace,
                        'metric': metric_name,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values),
                    })
        
        df = pd.DataFrame(rows)
        logger.info(f'Aggregated {len(df)} halfspace-metric combinations')
        return df
    
    def create_detailed_dataframe(self):
        """
        Create detailed DataFrame with all individual data points.
        
        Returns:
            DataFrame with columns: seed, variant, constraint, halfspace, metric, value
        """
        logger.info('Creating detailed dataframe...')
        rows = []
        
        for seed, variants_dict in self.raw_data.items():
            for variant, constraints_dict in variants_dict.items():
                for constraint, halfspaces_dict in constraints_dict.items():
                    for halfspace, metrics_dict in halfspaces_dict.items():
                        for metric_name, metric_value in metrics_dict.items():
                            if not metric_name.endswith('_array'):
                                rows.append({
                                    'seed': seed,
                                    'variant': variant,
                                    'constraint_type': constraint,
                                    'halfspace_variant': halfspace,
                                    'metric': metric_name,
                                    'value': metric_value,
                                })
        
        df = pd.DataFrame(rows)
        logger.info(f'Created detailed dataframe with {len(df)} rows')
        return df
    
    def create_pivot_table(self, metric_name, index='variant', columns='constraint_type'):
        """
        Create pivot table for a specific metric.
        
        Args:
            metric_name: Metric to pivot
            index: Dimension for rows
            columns: Dimension for columns
        
        Returns:
            Pivot table DataFrame
        """
        if 'detailed' not in self.aggregated or len(self.aggregated['detailed']) == 0:
            logger.warning('Detailed dataframe not available for pivot')
            return pd.DataFrame()
        
        detailed_df = self.aggregated['detailed']
        metric_df = detailed_df[detailed_df['metric'] == metric_name].copy()
        
        if len(metric_df) == 0:
            logger.warning(f'No data found for metric: {metric_name}')
            return pd.DataFrame()
        
        pivot = metric_df.pivot_table(
            values='value',
            index=index,
            columns=columns,
            aggfunc='mean'
        )
        
        return pivot
    
    def get_variant_ranking(self, metric_name='n_success_and_constraints', ascending=False):
        """
        Rank variants by a specific metric.
        
        Args:
            metric_name: Metric to rank by
            ascending: If True, rank in ascending order
        
        Returns:
            DataFrame with variant, mean, std sorted by metric
        """
        if 'by_variant' not in self.aggregated or len(self.aggregated['by_variant']) == 0:
            return pd.DataFrame()
        
        by_variant = self.aggregated['by_variant']
        metric_data = by_variant[by_variant['metric'] == metric_name].copy()
        metric_data = metric_data.sort_values('mean', ascending=ascending)
        
        return metric_data[['variant', 'mean', 'std', 'count']]
