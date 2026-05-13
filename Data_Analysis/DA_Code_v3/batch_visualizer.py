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
                label=letter
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
                candidates_list.append(letter)
        
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
                candidates_list.append(letter)
        
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
                        labels.append(letter)
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
        ax.set_yticklabels(candidates_sorted)
        
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

        self._generate_dashboard_html(output_dir, test_types, candidate_letters, major_metrics)

    def _generate_dashboard_html(self, output_dir, test_types, candidates, metrics):
        """Generate a scientific, clean, and minimal HTML dashboard (Cold Design)."""
        # Centralized visualizer path
        viz_root = os.path.join(os.path.dirname(os.path.dirname(output_dir)), "Visualizer")
        os.makedirs(viz_root, exist_ok=True)
        html_path = os.path.join(viz_root, "dashboard.html")
        
        # Calculate relative path from Visualizer/ to the output hierarchical analysis
        # Viz is at Data_Analysis/Visualizer
        # Plots are at Data_Analysis/analysis_results/batch_v3_.../hierarchical_analysis
        rel_to_plots = os.path.relpath(os.path.join(output_dir, "hierarchical_analysis"), viz_root)
        rel_to_base = os.path.relpath(output_dir, viz_root)
        
        config_data = {
            'tests': [str(t).replace(' ', '_').replace('/', '_') for t in test_types],
            'candidates': [f"Candidate_{c}" for c in candidates],
            'metrics': metrics,
            'major_variants': MAJOR_VARIANTS,
            'base_path': rel_to_plots + "/",
            'summary_path': rel_to_base + "/"
        }
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scientific DA Visualizer</title>
    <style>
        body {{ font-family: monospace; background: #ffffff; color: #000000; margin: 0; padding: 10px; display: flex; flex-direction: column; height: 100vh; }}
        header {{ border-bottom: 2px solid #000; padding: 5px 0; margin-bottom: 10px; }}
        .container {{ display: flex; flex: 1; overflow: hidden; gap: 20px; }}
        .sidebar {{ width: 250px; border-right: 1px solid #ccc; padding-right: 15px; display: flex; flex-direction: column; gap: 15px; }}
        .viewer {{ flex: 1; overflow-y: auto; border: 1px solid #eee; padding: 10px; text-align: center; }}
        .control {{ display: flex; flex-direction: column; gap: 5px; }}
        label {{ font-weight: bold; font-size: 12px; color: #666; }}
        select {{ font-family: monospace; padding: 5px; border: 1px solid #000; }}
        img {{ max-width: 100%; border: 1px solid #000; }}
        .meta {{ font-size: 11px; margin-top: auto; color: #999; border-top: 1px solid #eee; padding-top: 10px; }}
        h3 {{ margin: 0 0 10px 0; font-size: 16px; text-decoration: underline; }}
    </style>
</head>
<body>
    <header>
        [ FM-PCC DATA ANALYSIS ] | BATCH: {os.path.basename(output_dir)}
    </header>

    <div class="container">
        <div class="sidebar">
            <h3>Controls</h3>
            
            <div class="control">
                <label>VIEW_MODE</label>
                <select id="mode" onchange="updateUI()">
                    <option value="env">BY_ENVIRONMENT</option>
                    <option value="cand">BY_CANDIDATE</option>
                    <option value="matrix">GLOBAL_MATRIX</option>
                </select>
            </div>

            <div id="envControls" class="control">
                <label>ENVIRONMENT</label>
                <select id="testSelect" onchange="update()">
                    { "".join([f'<option value="{t}">{t}</option>' for t in config_data['tests']]) }
                </select>
            </div>

            <div id="candControls" class="control" style="display:none">
                <label>CANDIDATE</label>
                <select id="candSelect" onchange="update()">
                    { "".join([f'<option value="{c}">{c}</option>' for c in config_data['candidates']]) }
                </select>
            </div>

            <div id="metricControls" class="control">
                <label>METRIC</label>
                <select id="metricSelect" onchange="update()">
                    <option value="n_success_and_constraints">SUCCESS_RATE</option>
                    <option value="avg_time">COMP_TIME</option>
                </select>
            </div>

            <div id="variantControls" class="control" style="display:none">
                <label>VARIANT</label>
                <select id="variantSelect" onchange="update()">
                    { "".join([f'<option value="{v}">{v}</option>' for v in config_data['major_variants']]) }
                </select>
            </div>

            <div class="meta">
                Status: ATTACHED<br>
                Source: {output_dir}
            </div>
        </div>

        <div class="viewer">
            <h3 id="plotLabel">IMAGE_PREVIEW</h3>
            <img id="display" src="" alt="NO_IMAGE_LOADED">
            <br><br>
            <label>PARETO_REFERENCE</label><br>
            <img src="{config_data['summary_path']}00_candidate_pareto_frontier.png" style="max-width: 400px; opacity: 0.5;">
        </div>
    </div>

    <script>
        const base = "{config_data['base_path']}";
        
        function updateUI() {{
            const m = document.getElementById('mode').value;
            document.getElementById('envControls').style.display = (m === 'env') ? 'flex' : 'none';
            document.getElementById('candControls').style.display = (m === 'cand') ? 'flex' : 'none';
            document.getElementById('metricControls').style.display = (m !== 'matrix') ? 'flex' : 'none';
            document.getElementById('variantControls').style.display = (m === 'matrix') ? 'flex' : 'none';
            update();
        }}

        function update() {{
            const mode = document.getElementById('mode').value;
            const test = document.getElementById('testSelect').value;
            const cand = document.getElementById('candSelect').value;
            const metric = document.getElementById('metricSelect').value;
            const variant = document.getElementById('variantSelect').value;
            
            let src = "";
            if (mode === 'env') {{
                src = `${{base}}by_test_env/${{test}}/MAJOR_comp_${{metric}}.png`;
            }} else if (mode === 'cand') {{
                src = `${{base}}by_candidate/${{cand}}/MAJOR_cross_env_${{metric}}.png`;
            }} else if (mode === 'matrix') {{
                src = `${{base}}matrices/matrix_success_${{variant}}.png`;
            }}

            document.getElementById('display').src = src;
            document.getElementById('plotLabel').innerText = `PLOT: ${{mode.toUpperCase()}}_MAJOR_VIEW`;
        }}

        updateUI();
    </script>
</body>
</html>
"""
        with open(html_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Scientific Dashboard generated: {html_path}")
        with open(html_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Interactive Dashboard generated: {html_path}")

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
