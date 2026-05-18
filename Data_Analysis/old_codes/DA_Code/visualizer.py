"""
Visualization module: Creates plots and figures from aggregated data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from config import PLOT_CONFIG, METRIC_LABELS

logger = logging.getLogger(__name__)


class DataVisualizer:
    """Create publication-quality plots from evaluation data."""
    
    def __init__(self, aggregator):
        """
        Initialize visualizer.
        
        Args:
            aggregator: DataAggregator instance
        """
        self.aggregator = aggregator
        self.plots_created = 0
        
        # Set matplotlib style
        plt.style.use(PLOT_CONFIG['style'])
        plt.rcParams.update({
            'font.size': PLOT_CONFIG['font_size'],
            'figure.dpi': PLOT_CONFIG['dpi'],
        })
    
    def plot_variant_comparison(self, metric_name, output_dir, top_n=None):
        """
        Bar plot comparing all variants for a metric.
        
        Args:
            metric_name: Metric to plot
            output_dir: Directory to save plot
            top_n: If set, only plot top N variants
        """
        logger.info(f'Creating variant comparison plot for {metric_name}...')
        
        ranking = self.aggregator.get_variant_ranking(metric_name, ascending=False)
        if len(ranking) == 0:
            logger.warning(f'No data for variant comparison: {metric_name}')
            return
        
        if top_n:
            ranking = ranking.head(top_n)
        
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        
        x_pos = np.arange(len(ranking))
        colors = PLOT_CONFIG['colors'][:len(ranking)]
        
        bars = ax.bar(x_pos, ranking['mean'], yerr=ranking['std'], 
                      capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Projection Variant', fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name), 
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_title(f'Variant Comparison: {METRIC_LABELS.get(metric_name, metric_name)}',
                    fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ranking['variant'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        safe_metric = metric_name.replace('/', '_').replace(' ', '_')
        filename = f'01_variants_{safe_metric}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f'Saved: {filename}')
        self.plots_created += 1
        plt.close()
    
    def plot_constraint_comparison(self, metric_name, output_dir):
        """
        Bar plot comparing constraint types for a metric.
        
        Args:
            metric_name: Metric to plot
            output_dir: Directory to save plot
        """
        logger.info(f'Creating constraint comparison plot for {metric_name}...')
        
        by_constraint = self.aggregator.aggregated.get('by_constraint', pd.DataFrame())
        if len(by_constraint) == 0:
            logger.warning('No constraint aggregation data available')
            return
        
        metric_data = by_constraint[by_constraint['metric'] == metric_name].copy()
        if len(metric_data) == 0:
            logger.warning(f'No data for constraint comparison: {metric_name}')
            return
        
        metric_data = metric_data.sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(metric_data))
        colors = PLOT_CONFIG['colors'][:len(metric_data)]
        
        bars = ax.bar(x_pos, metric_data['mean'], yerr=metric_data['std'],
                      capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Constraint Type', fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name),
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_title(f'Constraint Comparison: {METRIC_LABELS.get(metric_name, metric_name)}',
                    fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_data['constraint_type'], rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        safe_metric = metric_name.replace('/', '_').replace(' ', '_')
        filename = f'02_constraints_{safe_metric}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f'Saved: {filename}')
        self.plots_created += 1
        plt.close()
    
    def plot_heatmap_variant_vs_constraint(self, metric_name, output_dir):
        """
        Heatmap of variant × constraint performance.
        
        Args:
            metric_name: Metric to plot
            output_dir: Directory to save plot
        """
        logger.info(f'Creating heatmap for {metric_name}...')
        
        pivot = self.aggregator.create_pivot_table(metric_name, 
                                                   index='variant', 
                                                   columns='constraint_type')
        
        if len(pivot) == 0:
            logger.warning(f'No pivot data for heatmap: {metric_name}')
            return
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index, fontsize=9)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(METRIC_LABELS.get(metric_name, metric_name),
                      rotation=270, labelpad=20)
        
        ax.set_title(f'Heatmap: {METRIC_LABELS.get(metric_name, metric_name)}\n(Variant × Constraint Type)',
                    fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not pd.isna(value):
                    text = ax.text(j, i, f'{value:.2f}',
                                 ha='center', va='center', color='black', fontsize=8)
        
        plt.tight_layout()
        
        safe_metric = metric_name.replace('/', '_').replace(' ', '_')
        filename = f'03_heatmap_variant_constraint_{safe_metric}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f'Saved: {filename}')
        self.plots_created += 1
        plt.close()
    
    def plot_efficiency_scatter(self, metric_y='n_success_and_constraints', 
                               metric_x='avg_time', output_dir='.'):
        """
        Scatter plot of efficiency (e.g., success vs time).
        
        Args:
            metric_y: Metric for Y-axis
            metric_x: Metric for X-axis (usually time)
            output_dir: Directory to save plot
        """
        logger.info(f'Creating efficiency scatter plot ({metric_y} vs {metric_x})...')
        
        detailed = self.aggregator.aggregated.get('detailed', pd.DataFrame())
        if len(detailed) == 0:
            logger.warning('No detailed data for scatter plot')
            return
        
        # Filter for specific metrics
        data_y = detailed[detailed['metric'] == metric_y][['variant', 'value']].copy()
        data_y.columns = ['variant', 'value_y']
        data_x = detailed[detailed['metric'] == metric_x][['variant', 'value']].copy()
        data_x.columns = ['variant', 'value_x']
        
        # Merge and aggregate
        merged = data_y.merge(data_x, on='variant')
        avg_data = merged.groupby('variant').agg({
            'value_y': 'mean',
            'value_x': 'mean'
        }).reset_index()
        
        if len(avg_data) < 2:
            logger.warning('Not enough data for scatter plot')
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        scatter = ax.scatter(avg_data['value_x'], avg_data['value_y'],
                           s=200, alpha=0.7, c=range(len(avg_data)),
                           cmap='viridis', edgecolors='black', linewidth=1)
        
        # Annotate with variant names
        for idx, row in avg_data.iterrows():
            ax.annotate(row['variant'], 
                       (row['value_x'], row['value_y']),
                       fontsize=8, alpha=0.8, ha='center')
        
        ax.set_xlabel(METRIC_LABELS.get(metric_x, metric_x),
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_ylabel(METRIC_LABELS.get(metric_y, metric_y),
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_title(f'Efficiency: {METRIC_LABELS.get(metric_y, metric_y)} vs {METRIC_LABELS.get(metric_x, metric_x)}',
                    fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'04_efficiency_{metric_y}_vs_{metric_x}.png'
        filename = filename.replace('/', '_').replace(' ', '_')
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f'Saved: {filename}')
        self.plots_created += 1
        plt.close()
    
    def plot_seed_variability_boxplot(self, metric_name, output_dir, top_n=None):
        """
        Box plot showing per-seed variability for each variant.
        
        Args:
            metric_name: Metric to plot
            output_dir: Directory to save plot
            top_n: If set, only plot top N variants
        """
        logger.info(f'Creating seed variability boxplot for {metric_name}...')
        
        detailed = self.aggregator.aggregated.get('detailed', pd.DataFrame())
        if len(detailed) == 0:
            logger.warning('No detailed data for boxplot')
            return
        
        metric_data = detailed[detailed['metric'] == metric_name].copy()
        if len(metric_data) == 0:
            logger.warning(f'No data for boxplot: {metric_name}')
            return
        
        # Get top variants
        top_variants = (metric_data.groupby('variant')['value'].mean()
                       .sort_values(ascending=False))
        if top_n:
            top_variants = top_variants.head(top_n)
        
        metric_data = metric_data[metric_data['variant'].isin(top_variants.index)]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Prepare data for boxplot
        variants = sorted(metric_data['variant'].unique())
        data_by_variant = [metric_data[metric_data['variant'] == v]['value'].values 
                          for v in variants]
        
        bp = ax.boxplot(data_by_variant, labels=variants, patch_artist=True)
        
        # Color boxes
        colors = PLOT_CONFIG['colors']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Projection Variant', fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name),
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_title(f'Seed Variability: {METRIC_LABELS.get(metric_name, metric_name)}',
                    fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_xticklabels(variants, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        safe_metric = metric_name.replace('/', '_').replace(' ', '_')
        filename = f'05_boxplot_seeds_{safe_metric}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f'Saved: {filename}')
        self.plots_created += 1
        plt.close()
    
    def plot_pareto_frontier(self, accuracy_metric='n_success_and_constraints', 
                            time_metric='avg_time', output_dir='.'):
        """
        Plot accuracy vs. time Pareto frontier.
        Highlights major variants (dpcc-c/r/t) and shows baseline comparisons.
        
        Args:
            accuracy_metric: Metric for Y-axis (higher is better)
            time_metric: Metric for X-axis (lower is better)
            output_dir: Directory to save plot
        """
        logger.info('Creating Pareto frontier plot (accuracy vs. time)...')
        
        detailed = self.aggregator.aggregated.get('detailed', pd.DataFrame())
        if len(detailed) == 0:
            logger.warning('No detailed data for Pareto plot')
            return
        
        # Extract accuracy and time data
        accuracy_data = detailed[detailed['metric'] == accuracy_metric][['variant', 'value']].copy()
        accuracy_data.columns = ['variant', 'accuracy']
        time_data = detailed[detailed['metric'] == time_metric][['variant', 'value']].copy()
        time_data.columns = ['variant', 'time']
        
        # Merge and aggregate
        merged = accuracy_data.merge(time_data, on='variant')
        pareto_data = merged.groupby('variant').agg({
            'accuracy': 'mean',
            'time': 'mean'
        }).reset_index()
        
        if len(pareto_data) < 2:
            logger.warning('Not enough data for Pareto plot')
            return
        
        # Categorize variants
        def categorize_variant(variant):
            if 'dpcc-c' in variant:
                return 'dpcc-c (main)', 0
            elif 'dpcc-r' in variant:
                return 'dpcc-r (main)', 1
            elif 'dpcc-t' in variant:
                return 'dpcc-t (main)', 2
            elif variant in ['diffuser', 'gradient', 'model_free', 'post_processing']:
                return 'Baseline', 3
            else:
                return 'Other', 4
        
        pareto_data['category'] = pareto_data['variant'].apply(lambda x: categorize_variant(x)[0])
        pareto_data['cat_order'] = pareto_data['variant'].apply(lambda x: categorize_variant(x)[1])
        
        # Define colors
        color_map = {
            'dpcc-c (main)': '#d62728',  # Red
            'dpcc-r (main)': '#ff7f0e',  # Orange
            'dpcc-t (main)': '#ffdd57',  # Yellow
            'Baseline': '#1f77b4',       # Blue
            'Other': '#2ca02c'           # Green
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot points by category
        for category in ['Baseline', 'Other', 'dpcc-c (main)', 'dpcc-r (main)', 'dpcc-t (main)']:
            cat_data = pareto_data[pareto_data['category'] == category]
            ax.scatter(cat_data['time'], cat_data['accuracy'],
                      s=300, alpha=0.7, color=color_map[category],
                      edgecolors='black', linewidth=2, label=category, zorder=3)
        
        # Annotate with variant names
        for idx, row in pareto_data.iterrows():
            ax.annotate(row['variant'], 
                       (row['time'], row['accuracy']),
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                       zorder=4)
        
        ax.set_xlabel(f'{METRIC_LABELS.get(time_metric, time_metric)} (lower is better)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_ylabel(f'{METRIC_LABELS.get(accuracy_metric, accuracy_metric)} (higher is better)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.set_title('Accuracy-Time Pareto Frontier',
                    fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(alpha=0.3, zorder=0)
        ax.legend(loc='best', fontsize=PLOT_CONFIG['legend_size'])
        
        plt.tight_layout()
        
        filename = f'00_pareto_frontier_accuracy_vs_time.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f'Saved: {filename}')
        self.plots_created += 1
        plt.close()
    
    def plot_all_key_metrics(self, output_dir):
        """
        Generate all key comparison plots.
        
        Args:
            output_dir: Directory to save plots
        """
        logger.info('Creating all key metric plots...')
        
        # Pareto frontier (thesis-focused)
        try:
            self.plot_pareto_frontier('n_success_and_constraints', 'avg_time', output_dir)
        except Exception as e:
            logger.error(f'Failed to create Pareto frontier plot: {str(e)}')
        
        key_metrics = [
            'n_success',
            'n_success_and_constraints',
            'collision_free_completed',
            'avg_time',
            'n_violations',
        ]
        
        for metric in key_metrics:
            try:
                self.plot_variant_comparison(metric, output_dir, top_n=None)
                self.plot_constraint_comparison(metric, output_dir)
                self.plot_heatmap_variant_vs_constraint(metric, output_dir)
                self.plot_seed_variability_boxplot(metric, output_dir, top_n=10)
            except Exception as e:
                logger.error(f'Failed to create plots for {metric}: {str(e)}')
        
        # Efficiency plots
        try:
            self.plot_efficiency_scatter('n_success_and_constraints', 'avg_time', output_dir)
        except Exception as e:
            logger.error(f'Failed to create efficiency plot: {str(e)}')
        
        logger.info(f'Total plots created: {self.plots_created}')
