import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from matplotlib import rcParams
import json


logger = logging.getLogger(__name__)

# Use config for styling
from config import PLOT_CONFIG, MAJOR_VARIANTS, AUXILIARY_VARIANTS

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
        Plot two Pareto frontiers: Standard and Tightened.
        """
        # 1. Standard Plot
        self._generate_pareto_subgroup(
            output_dir, "00a_candidate_pareto_frontier_standard", 
            "accuracy_std_group", "time_ms_std_group", 
            "Standard DPCC (Avg: r, c, t)", show
        )
        
        # 2. Tightened Plot
        self._generate_pareto_subgroup(
            output_dir, "00b_candidate_pareto_frontier_tightened", 
            "accuracy_tight_group", "time_ms_tight_group", 
            "Tightened DPCC (Avg: r-t, c-t, t-t)", show
        )

    def _generate_pareto_subgroup(self, output_dir, filename, acc_key, time_key, title_suffix, show):
        logger.info(f"Generating Pareto frontier plot: {filename}...")
        
        points = {}
        for letter, stats in self.candidate_stats.items():
            acc = stats.get(acc_key)
            time = stats.get(time_key)
            if acc is not None and time is not None:
                points[letter] = {'accuracy': acc * 100, 'time': time}
        
        if not points:
            logger.warning(f"No valid data for {filename}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        for letter, point in sorted(points.items()):
            ax.scatter(point['time'], point['accuracy'], s=500, alpha=0.7, 
                      color=self._get_candidate_color(letter), edgecolors='black', linewidth=2, label=letter)
            ax.annotate(letter, (point['time'], point['accuracy']), fontsize=14, fontweight='bold', ha='center', va='center', color='white')
        
        ax.set_xlabel('Computation Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Cross-Candidate Pareto Frontier: {title_suffix}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        output_path = f"{output_dir}/{filename}.png"
        plt.savefig(output_path, dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        if show: plt.show()
        plt.close()
    
    def plot_candidate_success_comparison(self, output_dir, show=False):
        """
        Bar chart comparing success rates across candidates.
        """
        # 1. Standard
        self._generate_bar_comparison(
            output_dir, "01a_candidate_success_standard",
            "accuracy_std_group", "Success Rate (%)", 
            "Success Comparison: Standard DPCC", show, is_percentage=True
        )
        # 2. Tightened
        self._generate_bar_comparison(
            output_dir, "01b_candidate_success_tightened",
            "accuracy_tight_group", "Success Rate (%)", 
            "Success Comparison: Tightened DPCC", show, is_percentage=True
        )

    def plot_candidate_time_comparison(self, output_dir, show=False):
        """
        Bar chart comparing computation time across candidates.
        """
        # 1. Standard
        self._generate_bar_comparison(
            output_dir, "02a_candidate_time_standard",
            "time_ms_std_group", "Computation Time (ms)", 
            "Time Comparison: Standard DPCC", show, is_percentage=False
        )
        # 2. Tightened
        self._generate_bar_comparison(
            output_dir, "02b_candidate_time_tightened",
            "time_ms_tight_group", "Computation Time (ms)", 
            "Time Comparison: Tightened DPCC", show, is_percentage=False
        )

    def _generate_bar_comparison(self, output_dir, filename, key, ylabel, title, show, is_percentage=False):
        logger.info(f"Generating bar comparison plot: {filename}...")
        data = []
        labels = []
        for letter in sorted(self.candidate_stats.keys()):
            val = self.candidate_stats[letter].get(key)
            if val is not None:
                data.append(val * 100 if is_percentage else val)
                labels.append(letter)
        
        if not data: return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [self._get_candidate_color(l) for l in labels]
        bars = ax.bar(range(len(data)), data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                    f'{val:.1f}%' if is_percentage else f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        if is_percentage: ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}.png", dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        if show: plt.show()
        plt.close()
    
    def plot_candidate_robustness_boxplot(self, output_dir, show=False):
        """
        Boxplot showing seed variability per candidate.
        """
        # 1. Standard
        self._generate_robustness_subgroup(
            output_dir, "03a_candidate_robustness_standard", 
            ['dpcc-r', 'dpcc-c', 'dpcc-t'], 
            "Robustness: Standard DPCC (Seed Variability)", show
        )
        # 2. Tightened
        self._generate_robustness_subgroup(
            output_dir, "03b_candidate_robustness_tightened", 
            ['dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened'], 
            "Robustness: Tightened DPCC (Seed Variability)", show
        )

    def _generate_robustness_subgroup(self, output_dir, filename, variants, title, show):
        logger.info(f"Generating robustness plot: {filename}...")
        box_data = []
        labels = []
        colors_list = []
        
        for letter in sorted(self.candidate_aggregators.keys()):
            agg = self.candidate_aggregators[letter]
            try:
                detailed = agg.detailed_df
                if detailed is not None and not detailed.empty:
                    # Filter for specific variants
                    subset = detailed[detailed['variant'].isin(variants)]
                    if subset.empty: continue
                    
                    seeds_accuracy = []
                    for seed in subset['seed'].unique():
                        seed_data = subset[subset['seed'] == seed]
                        acc = seed_data['value'].mean() if len(seed_data) > 0 else np.nan
                        if not np.isnan(acc):
                            seeds_accuracy.append(acc * 100)
                    
                    if seeds_accuracy:
                        box_data.append(seeds_accuracy)
                        labels.append(letter)
                        colors_list.append(self._get_candidate_color(letter))
            except Exception: pass
            
        if not box_data: return
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}.png", dpi=PLOT_CONFIG.get('dpi', 300))
        if show: plt.show()
        plt.close()

    def plot_candidate_constraint_heatmap(self, output_dir, show=False):
        """
        Heatmap showing success rate per candidate × constraint type.
        """
        # 1. Standard
        self._generate_heatmap_subgroup(
            output_dir, "04a_candidate_heatmap_standard",
            ['dpcc-r', 'dpcc-c', 'dpcc-t'],
            "Constraint Performance: Standard DPCC", show
        )
        # 2. Tightened
        self._generate_heatmap_subgroup(
            output_dir, "04b_candidate_heatmap_tightened",
            ['dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened'],
            "Constraint Performance: Tightened DPCC", show
        )

    def _generate_heatmap_subgroup(self, output_dir, filename, variants, title, show):
        logger.info(f"Generating heatmap: {filename}...")
        candidates_sorted = sorted(self.candidate_aggregators.keys())
        constraint_types = ['halfspace', 'obstacles', 'dynamics', 'bounds']
        heatmap_data = np.zeros((len(candidates_sorted), len(constraint_types)))
        
        for i, letter in enumerate(candidates_sorted):
            agg = self.candidate_aggregators[letter]
            try:
                detailed = agg.detailed_df
                if detailed is not None and not detailed.empty:
                    subset = detailed[detailed['variant'].isin(variants)]
                    for j, constraint in enumerate(constraint_types):
                        c_data = subset[subset['constraint_type'] == constraint]
                        if not c_data.empty:
                            heatmap_data[i, j] = c_data['value'].mean()
            except Exception: pass

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(constraint_types))); ax.set_yticks(range(len(candidates_sorted)))
        ax.set_xticklabels(constraint_types); ax.set_yticklabels(candidates_sorted)
        cbar = plt.colorbar(im, ax=ax); cbar.set_label('Success Rate', fontweight='bold')
        for i in range(len(candidates_sorted)):
            for j in range(len(constraint_types)):
                ax.text(j, i, f'{heatmap_data[i, j]*100:.0f}%', ha="center", va="center", fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}.png", dpi=PLOT_CONFIG.get('dpi', 300))
        if show: plt.show()
        plt.close()

    def plot_matrix_analysis(self, output_dir, show=False):
        """
        Generate a hierarchical matrix of plots organized by test type and candidate.
        """
        logger.info("Generating multidimensional matrix analysis...")
        
        # Gather all data
        all_dfs = []
        for letter, agg in self.candidate_aggregators.items():
            df = agg.detailed_df
            if df is not None and not df.empty:
                df = df.copy()
                df['Candidate'] = letter
                all_dfs.append(df)
        
        if not all_dfs:
            logger.warning("No data for matrix analysis")
            return
            
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # 1. Create Folder Structure
        # plots/by_test_env/both_hard/...
        # plots/by_candidate/A/...
        # plots/matrices/...
        
        base_plots = os.path.join(output_dir, "hierarchical_analysis")
        env_dir = os.path.join(base_plots, "by_test_env")
        cand_dir = os.path.join(base_plots, "by_candidate")
        matrix_dir = os.path.join(base_plots, "matrices")
        
        for d in [env_dir, cand_dir, matrix_dir]:
            os.makedirs(d, exist_ok=True)
            
        test_types = full_df['halfspace_variant'].unique()
        candidate_letters = sorted(full_df['Candidate'].unique())
        major_metrics = ['n_success_and_constraints', 'avg_time']
        
        # --- PHASE A: Analysis by Test Type ---
        for test in test_types:
            test_safe = str(test).replace(' ', '_').replace('/', '_')
            test_folder = os.path.join(env_dir, test_safe)
            os.makedirs(test_folder, exist_ok=True)
            
            test_df = full_df[full_df['halfspace_variant'] == test]
            
            for metric in major_metrics:
                metric_data = test_df[test_df['metric'] == metric]
                if metric_data.empty: continue
                
                # Separate Major vs Auxiliary into distinct plot files
                for group_name, v_list in [('MAJOR', MAJOR_VARIANTS), ('AUX', AUXILIARY_VARIANTS)]:
                    subset = metric_data[metric_data['variant'].isin(v_list)]
                    if subset.empty: continue
                    
                    fig, ax = plt.subplots(figsize=(12, 7))
                    pivot = subset.groupby(['Candidate', 'variant'])['value'].mean().unstack()
                    
                    # Ensure Major variants are plotted in a specific order if possible
                    cols = [v for v in v_list if v in pivot.columns]
                    pivot[cols].plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.9)
                    
                    ax.set_title(f"Env: {test} | {group_name} Variants | {metric}", fontsize=14, fontweight='bold')
                    ax.set_ylabel(metric)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.legend(title=f"{group_name} Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(test_folder, f"{group_name}_comp_{metric}.png"), bbox_inches='tight')
                    plt.close()

        # --- PHASE B: Analysis by Candidate ---
        for cand in candidate_letters:
            cand_folder = os.path.join(cand_dir, f"Candidate_{cand}")
            os.makedirs(cand_folder, exist_ok=True)
            
            cand_df = full_df[full_df['Candidate'] == cand]
            
            for metric in major_metrics:
                metric_data = cand_df[cand_df['metric'] == metric]
                if metric_data.empty: continue
                
                for group_name, v_list in [('MAJOR', MAJOR_VARIANTS), ('AUX', AUXILIARY_VARIANTS)]:
                    subset = metric_data[metric_data['variant'].isin(v_list)]
                    if subset.empty: continue
                    
                    fig, ax = plt.subplots(figsize=(14, 8))
                    pivot = subset.groupby(['halfspace_variant', 'variant'])['value'].mean().unstack()
                    
                    cols = [v for v in v_list if v in pivot.columns]
                    pivot[cols].plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.9)
                    
                    ax.set_title(f"Candidate {cand} | {group_name} Behavior | {metric}", fontsize=14, fontweight='bold')
                    ax.set_ylabel(metric)
                    ax.set_xlabel("Test Environment")
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.legend(title=f"{group_name} Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(cand_folder, f"{group_name}_cross_env_{metric}.png"), bbox_inches='tight')
                    plt.close()

        # --- PHASE C: Metric Matrices ---
        # Global Heatmaps
        for variant in MAJOR_VARIANTS:
            v_df = full_df[(full_df['variant'] == variant) & (full_df['metric'] == 'n_success_and_constraints')]
            if v_df.empty: continue
            
            matrix = v_df.groupby(['Candidate', 'halfspace_variant'])['value'].mean().unstack()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(matrix.values, cmap='RdYlGn', vmin=0, vmax=1)
            
            ax.set_xticks(np.arange(len(matrix.columns)))
            ax.set_yticks(np.arange(len(matrix.index)))
            ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(matrix.index)
            
            for i in range(len(matrix.index)):
                for j in range(len(matrix.columns)):
                    ax.text(j, i, f"{matrix.values[i, j]*100:.1f}%", ha="center", va="center", fontweight='bold', fontsize=10)
            
            ax.set_title(f"Success Rate Matrix: Candidate vs Env (Variant: {variant})", fontsize=15, fontweight='bold')
            plt.colorbar(im, ax=ax, label="Goal + Constraint Success Rate")
            
            plt.tight_layout()
            plt.savefig(os.path.join(matrix_dir, f"matrix_success_{variant}.png"), bbox_inches='tight')
            plt.close()
        
        print(f"[ DA ] Matrix analysis complete. Saved to: {output_dir}")

    def plot_all(self, output_dir, show=False):
        """
        Generate all comparison plots including hierarchical matrix analysis.
        """
        logger.info("Generating all cross-candidate comparison plots...")
        
        # Original v2 plots
        self.plot_candidate_pareto_frontier(output_dir, show=show)
        self.plot_candidate_success_comparison(output_dir, show=show)
        self.plot_candidate_time_comparison(output_dir, show=show)
        self.plot_candidate_robustness_boxplot(output_dir, show=show)
        self.plot_candidate_constraint_heatmap(output_dir, show=show)
        
        # New Multidimensional Matrix Analysis
        self.plot_matrix_analysis(output_dir, show=show)
        
        logger.info("All plots generated successfully")


if __name__ == "__main__":
    import sys
    from multi_candidate_discovery import discover_candidates
    from batch_data_loader import BatchDataLoader
    from batch_aggregator import BatchAggregator
    
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    if len(sys.argv) > 1:
        parent = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "./batch_analysis_v3"
        os.makedirs(output, exist_ok=True)
        
        # 1. Discover
        candidates = discover_candidates(parent)
        # 2. Load
        loader = BatchDataLoader()
        batch_data = loader.load_all_candidates(candidates)
        # 3. Aggregate
        aggregator = BatchAggregator()
        stats = aggregator.aggregate_all_candidates(batch_data)
        
        # 4. Visualize
        viz = BatchVisualizer(stats, aggregator.candidate_aggregators)
        viz.plot_all(output, show=False)
        
        print("\n" + "="*80)
        print(" ANALYSIS COMPLETE")
        print("="*80)
        print(f" Output Directory: {os.path.abspath(output)}")
        print(f" Interactive Dashboard: {os.path.join(os.path.abspath(output), 'dashboard.html')}")
        print("="*80)
    else:
        print("Usage: python batch_visualizer.py <batch_log_dir> [output_dir]")
