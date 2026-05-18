"""
Batch Aggregator Module (v2)

Aggregates evaluation data for each candidate separately, then creates
cross-candidate comparison views and rankings.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from aggregator import DataAggregator


logger = logging.getLogger(__name__)


class BatchAggregator:
    """
    Aggregate data across multiple candidates and create comparison views.
    
    Reuses v1 DataAggregator for each candidate, then organizes statistics
    by candidate for cross-comparison.
    """
    
    def __init__(self):
        """Initialize batch aggregator."""
        self.batch_data = {}
        self.candidate_aggregators = {}
        self.candidate_stats = {}
        self.ranked_candidates = []
    
    def aggregate_all_candidates(self, batch_data):
        """
        Aggregate data for all candidates.
        
        Args:
            batch_data: Dict from BatchDataLoader with candidate dimension
            
        Returns:
            dict: Candidate statistics {letter: {metric: {mean, std, min, max}}, ...}
        """
        logger.info("Aggregating data for all candidates...")
        
        self.batch_data = batch_data
        self.candidate_aggregators = {}
        self.candidate_stats = {}
        
        # Aggregate each candidate using v1 DataAggregator
        for candidate_idx, (letter, candidate_data) in enumerate(sorted(batch_data.items())):
            logger.info(f"Aggregating Candidate {letter}...")
            
            try:
                # Create v1 aggregator for this candidate
                agg = DataAggregator(candidate_data)
                agg.aggregate_all()
                
                self.candidate_aggregators[letter] = agg
                
                # Extract key metrics for comparison
                stats = self._extract_candidate_stats(agg, letter)
                self.candidate_stats[letter] = stats
                
                logger.info(f"  ✓ Aggregated with accuracy: {stats.get('accuracy', 'N/A'):.2%}" 
                           if isinstance(stats.get('accuracy'), (int, float)) else "  ✓ Aggregated")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to aggregate Candidate {letter}: {str(e)}")
                self.candidate_stats[letter] = {'error': str(e)}
        
        # Create cross-candidate rankings
        self.ranked_candidates = self._create_rankings()
        
        logger.info(f"Batch aggregation complete for {len(self.candidate_stats)} candidates")
        return self.candidate_stats
    
    def _extract_candidate_stats(self, aggregator, letter):
        """
        Extract key statistics from a v1 aggregator for cross-candidate comparison.
        
        Args:
            aggregator: v1 DataAggregator instance
            letter: Candidate letter
            
        Returns:
            dict: Statistics including accuracy, time, robustness
        """
        stats = {
            'letter': letter,
            'raw_aggregator': aggregator
        }
        
        try:
            # Get by-variant aggregation
            variant_agg = aggregator.aggregated_by_variant
            
            if variant_agg is not None and not variant_agg.empty:
                # Calculate global metrics (average across all variants)
                accuracy_metric = 'n_success_and_constraints'
                time_metric = 'avg_time'
                
                # Find rows for our metrics
                accuracy_rows = variant_agg[variant_agg['metric'] == accuracy_metric]
                time_rows = variant_agg[variant_agg['metric'] == time_metric]
                
                if not accuracy_rows.empty:
                    # Average accuracy across variants
                    accuracy = accuracy_rows['mean'].mean()
                    accuracy_std = accuracy_rows['std'].mean()
                    stats['accuracy'] = accuracy
                    stats['accuracy_std'] = accuracy_std
                
                if not time_rows.empty:
                    # Average time across variants
                    time_ms = time_rows['mean'].mean()
                    time_std = time_rows['std'].mean()
                    stats['time_ms'] = time_ms
                    stats['time_std'] = time_std
            
            # Robustness: overall std across seeds
            stats['robustness'] = aggregator.aggregated_by_variant['std'].mean() if aggregator.aggregated_by_variant is not None and not aggregator.aggregated_by_variant.empty else 0
            
        except Exception as e:
            logger.warning(f"Could not extract full stats for {letter}: {str(e)}")
        
        return stats
    
    def _create_rankings(self):
        """
        Create cross-candidate rankings by metric.
        
        Returns:
            list: Sorted list of (letter, accuracy) tuples
        """
        rankings = []
        
        for letter, stats in self.candidate_stats.items():
            if 'accuracy' in stats and 'error' not in stats:
                rankings.append((letter, stats['accuracy']))
        
        # Sort by accuracy (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_candidate_ranking(self, metric='accuracy', ascending=False):
        """
        Get ranking of candidates by a specific metric.
        
        Args:
            metric: Metric name ('accuracy', 'time_ms', 'robustness')
            ascending: Sort ascending if True, descending if False
            
        Returns:
            list: Sorted list of (letter, value) tuples
        """
        rankings = []
        
        for letter, stats in self.candidate_stats.items():
            if metric in stats and 'error' not in stats:
                rankings.append((letter, stats[metric]))
        
        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings
    
    def get_statistics_dataframe(self):
        """
        Get all candidate statistics as a DataFrame.
        
        Returns:
            pd.DataFrame: Rows are candidates, columns are metrics
        """
        rows = []
        
        for letter in sorted(self.candidate_stats.keys()):
            stats = self.candidate_stats[letter]
            if 'error' not in stats:
                rows.append({
                    'Candidate': letter,
                    'Accuracy (%)': stats.get('accuracy', np.nan) * 100,
                    'Accuracy Std': stats.get('accuracy_std', np.nan),
                    'Time (ms)': stats.get('time_ms', np.nan),
                    'Time Std': stats.get('time_std', np.nan),
                    'Robustness': stats.get('robustness', np.nan)
                })
        
        return pd.DataFrame(rows)
    
    def get_candidate_aggregator(self, letter):
        """
        Get the v1 aggregator for a specific candidate.
        
        Useful for accessing detailed per-variant or per-constraint data.
        
        Args:
            letter: Candidate letter
            
        Returns:
            DataAggregator: v1 aggregator instance
        """
        return self.candidate_aggregators.get(letter)
    
    def get_pareto_points(self):
        """
        Get data points for Pareto frontier plot.
        
        Returns:
            dict: {letter: {'accuracy': value, 'time': value}, ...}
        """
        points = {}
        
        for letter, stats in self.candidate_stats.items():
            if 'accuracy' in stats and 'time_ms' in stats and 'error' not in stats:
                points[letter] = {
                    'accuracy': stats['accuracy'] * 100,  # Convert to percentage
                    'time': stats['time_ms'],
                    'robustness': stats.get('robustness', np.nan)
                }
        
        return points
    
    def print_ranking_summary(self):
        """
        Print human-readable ranking summary.
        """
        lines = ["=== Cross-Candidate Rankings ===\n"]
        
        # By Accuracy
        lines.append("By Accuracy (Goal + Constraint Success):")
        accuracy_ranking = self.get_candidate_ranking('accuracy', ascending=False)
        for rank, (letter, value) in enumerate(accuracy_ranking, 1):
            lines.append(f"  {rank}. Candidate {letter}: {value*100:.1f}%")
        
        lines.append("")
        
        # By Time
        lines.append("By Computation Time (faster is better):")
        time_ranking = self.get_candidate_ranking('time_ms', ascending=True)
        for rank, (letter, value) in enumerate(time_ranking, 1):
            lines.append(f"  {rank}. Candidate {letter}: {value:.1f} ms")
        
        lines.append("")
        
        # By Robustness
        lines.append("By Robustness (lower variance is better):")
        robustness_ranking = self.get_candidate_ranking('robustness', ascending=True)
        for rank, (letter, value) in enumerate(robustness_ranking, 1):
            lines.append(f"  {rank}. Candidate {letter}: {value:.4f}")
        
        summary = "\n".join(lines)
        logger.info("\n" + summary)
        return summary


if __name__ == "__main__":
    # Test the batch aggregator
    import sys
    from multi_candidate_discovery import discover_candidates
    from batch_data_loader import BatchDataLoader
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        parent = sys.argv[1]
        
        # Discover candidates
        candidates = discover_candidates(parent)
        
        # Load data
        loader = BatchDataLoader()
        batch_data = loader.load_all_candidates(candidates)
        
        # Aggregate
        aggregator = BatchAggregator()
        stats = aggregator.aggregate_all_candidates(batch_data)
        
        # Print summary
        aggregator.print_ranking_summary()
        
        # Print DataFrame
        print("\n" + "="*60)
        print(aggregator.get_statistics_dataframe())
    else:
        print("Usage: python batch_aggregator.py <parent_path>")
