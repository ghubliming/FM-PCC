"""
Batch Visualizer Module (v2)

Creates cross-candidate comparison plots showing which candidate is best
across various performance metrics.

Generates 5 plots:
1. Candidate Pareto Frontier (accuracy vs time)
2. Candidate Success Rate Comparison (by constraint)
3. Candidate Time Comparison (bar chart)
4. Candidate Robustness (boxplot of seed variability)
5. Candidate × Constraint Heatmap
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib import rcParams


logger = logging.getLogger(__name__)

# Use config for styling
from config import PLOT_CONFIG

# Color scheme for candidates: A=red, B=orange, C=yellow, D=blue, E=green...
CANDIDATE_COLORS = {
    'A': '#e74c3c',      # red
    'B': '#f39c12',      # orange
    'C': '#f1c40f',      # yellow
    'D': '#3498db',      # blue
    'E': '#2ecc71',      # green
    'F': '#9b59b6',      # purple
    'G': '#1abc9c',      # teal
    'H': '#e67e22',      # dark orange
    'I': '#34495e',      # dark blue
    'J': '#c0392b',      # dark red
}


class BatchVisualizer:
    """
    Generate cross-candidate comparison plots.
    """
    
    def __init__(self, candidate_stats, candidate_aggregators):
        """
        Initialize batch visualizer.
        
        Args:
            candidate_stats: Dict from BatchAggregator with statistics
            candidate_aggregators: Dict of v1 DataAggregator instances
        """
        self.candidate_stats = candidate_stats
        self.candidate_aggregators = candidate_aggregators
        
        # Apply plot config
        try:
            plt.style.use(PLOT_CONFIG.get('style', 'seaborn-v0_8-darkgrid'))
        except:
            pass  # Style may not exist in all matplotlib versions
        
        rcParams['figure.figsize'] = PLOT_CONFIG.get('figsize', (12, 7))
        rcParams['savefig.dpi'] = PLOT_CONFIG.get('dpi', 300)
        rcParams['font.size'] = PLOT_CONFIG.get('font_size', 11)
        rcParams['axes.titlesize'] = PLOT_CONFIG.get('title_size', 13)
        rcParams['legend.fontsize'] = PLOT_CONFIG.get('legend_size', 10)
    
    def _get_candidate_color(self, letter):
        """Get color for a candidate letter."""
        return CANDIDATE_COLORS.get(letter, '#95a5a6')
    
    def plot_candidate_pareto_frontier(self, output_dir, show=False):
        """
        Plot Pareto frontier: accuracy vs time for all candidates.
        
        X-axis: Computation time (ms) - lower is better
        Y-axis: Accuracy (%) - higher is better
        Each point = one candidate, colored and annotated
        
        Args:
            output_dir: Directory to save plot
            show: Display plot if True
        """
        logger.info("Generating Pareto frontier plot...")
        
        points = {}
        for letter, stats in self.candidate_stats.items():
            if 'accuracy' in stats and 'time_ms' in stats and 'error' not in stats:
                points[letter] = {
                    'accuracy': stats['accuracy'] * 100,
                    'time': stats['time_ms']
                }
        
        if not points:
            logger.warning("No valid data for Pareto frontier plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot points
        for letter, point in sorted(points.items()):
            ax.scatter(
                point['time'],
                point['accuracy'],
                s=500,
                alpha=0.7,
                color=self._get_candidate_color(letter),
                edgecolors='black',
                linewidth=2,
                label=f'Candidate {letter}'
            )
            
            # Annotate with letter
            ax.annotate(
                letter,
                (point['time'], point['accuracy']),
                fontsize=14,
                fontweight='bold',
                ha='center',
                va='center',
                color='white'
            )
        
        ax.set_xlabel('Computation Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Candidate Pareto Frontier: Accuracy vs Time', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        output_path = f"{output_dir}/00_candidate_pareto_frontier.png"
        plt.savefig(output_path, dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_candidate_success_comparison(self, output_dir, show=False):
        """
        Bar chart comparing success rates across candidates.
        
        Shows accuracy for each candidate side-by-side.
        
        Args:
            output_dir: Directory to save plot
            show: Display plot if True
        """
        logger.info("Generating success rate comparison plot...")
        
        data = []
        candidates_list = []
        
        for letter in sorted(self.candidate_stats.keys()):
            stats = self.candidate_stats[letter]
            if 'accuracy' in stats and 'error' not in stats:
                data.append(stats['accuracy'] * 100)
                candidates_list.append(f"Candidate {letter}")
        
        if not data:
            logger.warning("No valid data for success rate plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [self._get_candidate_color(c.split()[-1]) for c in candidates_list]
        bars = ax.bar(range(len(data)), data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Candidate Success Rate Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(candidates_list)))
        ax.set_xticklabels(candidates_list)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = f"{output_dir}/01_candidate_success_comparison.png"
        plt.savefig(output_path, dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_candidate_time_comparison(self, output_dir, show=False):
        """
        Bar chart comparing computation time across candidates.
        
        Shows time for each candidate - lower is better.
        
        Args:
            output_dir: Directory to save plot
            show: Display plot if True
        """
        logger.info("Generating time comparison plot...")
        
        data = []
        candidates_list = []
        
        for letter in sorted(self.candidate_stats.keys()):
            stats = self.candidate_stats[letter]
            if 'time_ms' in stats and 'error' not in stats:
                data.append(stats['time_ms'])
                candidates_list.append(f"Candidate {letter}")
        
        if not data:
            logger.warning("No valid data for time comparison plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [self._get_candidate_color(c.split()[-1]) for c in candidates_list]
        bars = ax.bar(range(len(data)), data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}ms',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Computation Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Candidate Computation Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(candidates_list)))
        ax.set_xticklabels(candidates_list)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = f"{output_dir}/02_candidate_time_comparison.png"
        plt.savefig(output_path, dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_candidate_robustness_boxplot(self, output_dir, show=False):
        """
        Boxplot showing seed variability per candidate.
        
        Tight boxes indicate reproducible results across seeds.
        Wide boxes indicate sensitivity to seed variation.
        
        Args:
            output_dir: Directory to save plot
            show: Display plot if True
        """
        logger.info("Generating robustness boxplot...")
        
        # Collect per-seed accuracies for each candidate
        box_data = []
        labels = []
        colors_list = []
        
        for letter in sorted(self.candidate_aggregators.keys()):
            agg = self.candidate_aggregators[letter]
            
            try:
                # Extract per-seed accuracy data
                detailed = agg.detailed_df
                if detailed is not None and not detailed.empty:
                    # Get accuracy by seed
                    seeds_accuracy = []
                    for seed in detailed['seed'].unique():
                        seed_data = detailed[detailed['seed'] == seed]
                        acc = seed_data['value'].mean() if len(seed_data) > 0 else np.nan
                        if not np.isnan(acc):
                            seeds_accuracy.append(acc * 100)
                    
                    if seeds_accuracy:
                        box_data.append(seeds_accuracy)
                        labels.append(f"Candidate {letter}")
                        colors_list.append(self._get_candidate_color(letter))
            except Exception as e:
                logger.warning(f"Could not extract robustness for {letter}: {str(e)}")
        
        if not box_data:
            logger.warning("No valid data for robustness plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Candidate Robustness: Seed Variability', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = f"{output_dir}/03_candidate_robustness_boxplot.png"
        plt.savefig(output_path, dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_candidate_constraint_heatmap(self, output_dir, show=False):
        """
        Heatmap showing success rate per candidate × constraint type.
        
        Rows = candidates, Columns = constraint types
        Cell color intensity = success rate
        
        Args:
            output_dir: Directory to save plot
            show: Display plot if True
        """
        logger.info("Generating constraint heatmap...")
        
        # Build matrix: candidate × constraint
        candidates_sorted = sorted(self.candidate_aggregators.keys())
        constraint_types = ['halfspace', 'obstacles', 'dynamics', 'bounds']
        
        heatmap_data = np.zeros((len(candidates_sorted), len(constraint_types)))
        
        for i, letter in enumerate(candidates_sorted):
            agg = self.candidate_aggregators[letter]
            
            try:
                by_constraint = agg.aggregated_by_constraint
                if by_constraint is not None and not by_constraint.empty:
                    for j, constraint in enumerate(constraint_types):
                        # Filter by constraint_type column
                        constraint_data = by_constraint[by_constraint['constraint_type'] == constraint]
                        if not constraint_data.empty:
                            # Average accuracy for this constraint
                            heatmap_data[i, j] = constraint_data[constraint_data['metric'] == 'n_success_and_constraints']['mean'].mean()
            except Exception as e:
                logger.warning(f"Could not get constraint data for {letter}: {str(e)}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(constraint_types)))
        ax.set_yticks(range(len(candidates_sorted)))
        ax.set_xticklabels(constraint_types)
        ax.set_yticklabels([f"Candidate {l}" for l in candidates_sorted])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate', fontweight='bold')
        
        # Add text annotations
        for i in range(len(candidates_sorted)):
            for j in range(len(constraint_types)):
                text = ax.text(j, i, f'{heatmap_data[i, j]*100:.0f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Cross-Candidate Performance: Constraint Type Breakdown', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = f"{output_dir}/04_candidate_constraint_heatmap.png"
        plt.savefig(output_path, dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_all(self, output_dir, show=False):
        """
        Generate all comparison plots.
        
        Args:
            output_dir: Directory to save all plots
            show: Display plots if True
        """
        logger.info("Generating all cross-candidate comparison plots...")
        
        self.plot_candidate_pareto_frontier(output_dir, show=show)
        self.plot_candidate_success_comparison(output_dir, show=show)
        self.plot_candidate_time_comparison(output_dir, show=show)
        self.plot_candidate_robustness_boxplot(output_dir, show=show)
        self.plot_candidate_constraint_heatmap(output_dir, show=show)
        
        logger.info("All plots generated successfully")


if __name__ == "__main__":
    # Test the batch visualizer
    import sys
    from multi_candidate_discovery import discover_candidates
    from batch_data_loader import BatchDataLoader
    from batch_aggregator import BatchAggregator
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        parent = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "./batch_plots"
        
        import os
        os.makedirs(output, exist_ok=True)
        
        # Discover, load, aggregate
        candidates = discover_candidates(parent)
        loader = BatchDataLoader()
        batch_data = loader.load_all_candidates(candidates)
        aggregator = BatchAggregator()
        stats = aggregator.aggregate_all_candidates(batch_data)
        
        # Visualize
        viz = BatchVisualizer(stats, aggregator.candidate_aggregators)
        viz.plot_all(output, show=False)
        
        print(f"\nPlots saved to: {output}")
    else:
        print("Usage: python batch_visualizer.py <parent_path> [output_dir]")
