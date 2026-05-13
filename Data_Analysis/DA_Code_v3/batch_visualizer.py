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
        # "both hard folder then inside is dpcc C plots for all tests"
        for test in test_types:
            test_safe = str(test).replace(' ', '_').replace('/', '_')
            test_folder = os.path.join(env_dir, test_safe)
            os.makedirs(test_folder, exist_ok=True)
            
            test_df = full_df[full_df['halfspace_variant'] == test]
            
            # Plot Major metrics (comparing all candidates)
            for metric in major_metrics:
                metric_data = test_df[test_df['metric'] == metric]
                if metric_data.empty: continue
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Separate Major variants for clear comparison
                for variant_group, variants_list in [('MAJOR', MAJOR_VARIANTS), ('AUX', AUXILIARY_VARIANTS)]:
                    subset = metric_data[metric_data['variant'].isin(variants_list)]
                    if subset.empty: continue
                    
                    pivot = subset.groupby(['Candidate', 'variant'])['value'].mean().unstack()
                    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.8 if variant_group == 'MAJOR' else 0.4)
                
                ax.set_title(f"Environment: {test} | Metric: {metric}", fontsize=14, fontweight='bold')
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(title="Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.savefig(os.path.join(test_folder, f"candidate_comp_{metric}.png"), bbox_inches='tight')
                plt.close()

        # --- PHASE B: Analysis by Candidate ---
        # "A tests what is its behaviors across diffrent env"
        for cand in candidate_letters:
            cand_folder = os.path.join(cand_dir, f"Candidate_{cand}")
            os.makedirs(cand_folder, exist_ok=True)
            
            cand_df = full_df[full_df['Candidate'] == cand]
            
            for metric in major_metrics:
                metric_data = cand_df[cand_df['metric'] == metric]
                if metric_data.empty: continue
                
                # Plot performance across ALL test types for this candidate
                fig, ax = plt.subplots(figsize=(16, 9))
                
                pivot = metric_data.groupby(['halfspace_variant', 'variant'])['value'].mean().unstack()
                # Focus on Major Variants first
                major_pivot = pivot[[v for v in MAJOR_VARIANTS if v in pivot.columns]]
                major_pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.9)
                
                ax.set_title(f"Candidate {cand} Performance across Environments | {metric}", fontsize=16, fontweight='bold')
                ax.set_ylabel(metric)
                ax.set_xlabel("Test Environment")
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(title="Major Variants", bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.savefig(os.path.join(cand_folder, f"cross_env_{metric}.png"), bbox_inches='tight')
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
        """Generate a high-end interactive dashboard (Premium Design)."""
        html_path = os.path.join(output_dir, "dashboard.html")
        
        config_data = {
            'tests': [str(t).replace(' ', '_').replace('/', '_') for t in test_types],
            'candidates': [f"Candidate_{c}" for c in candidates],
            'metrics': metrics,
            'major_variants': MAJOR_VARIANTS,
            'base_path': "hierarchical_analysis/"
        }
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FM-PCC Premium Analysis Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a; --card: #1e293b; --accent: #3b82f6; --text: #f8fafc;
            --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
        }}
        body {{ font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; display: flex; height: 100vh; overflow: hidden; }}
        
        /* Sidebar */
        nav {{ width: 300px; background: #020617; padding: 2rem; border-right: 1px solid #1e293b; display: flex; flex-direction: column; gap: 1.5rem; }}
        h1 {{ font-size: 1.25rem; font-weight: 800; color: var(--accent); margin-bottom: 1rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
        
        .control-group {{ display: flex; flex-direction: column; gap: 0.5rem; }}
        label {{ font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
        select {{ background: var(--card); color: white; border: 1px solid #334155; padding: 0.75rem; border-radius: 0.5rem; outline: none; cursor: pointer; transition: all 0.2s; }}
        select:hover {{ border-color: var(--accent); }}

        /* Main Content */
        main {{ flex: 1; padding: 3rem; overflow-y: auto; background: radial-gradient(circle at top right, #1e1b4b, transparent); }}
        .header-meta {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }}
        .badge {{ background: #1e293b; padding: 0.5rem 1rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; border: 1px solid #334155; }}

        .plot-container {{ background: var(--card); border-radius: 1.5rem; padding: 2rem; border: 1px solid #334155; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); position: relative; }}
        img {{ width: 100%; border-radius: 0.75rem; display: block; filter: drop-shadow(0 10px 8px rgb(0 0 0 / 0.1)); }}
        
        .loading-overlay {{ position: absolute; inset: 0; background: var(--card); display: flex; align-items: center; justify-content: center; border-radius: 1.5rem; opacity: 0; pointer-events: none; transition: 0.3s; }}
        .loading-overlay.active {{ opacity: 0.8; }}

        .footer-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem; }}
        .stat-card {{ background: var(--card); padding: 1.25rem; border-radius: 1rem; border: 1px solid #334155; }}
        .stat-val {{ font-size: 1.5rem; font-weight: 800; color: var(--accent); }}
        .stat-label {{ font-size: 0.75rem; color: #94a3b8; }}
    </style>
</head>
<body>
    <nav>
        <h1>FM-PCC ANALYTICS</h1>
        
        <div class="control-group">
            <label>View Perspective</label>
            <select id="mode" onchange="updateUI()">
                <option value="env">Environment Comparison</option>
                <option value="cand">Candidate Trajectories</option>
                <option value="matrix">Global Performance Matrix</option>
            </select>
        </div>

        <div id="envControls" class="control-group">
            <label>Target Environment</label>
            <select id="testSelect" onchange="update()">
                { "".join([f'<option value="{t}">{t.replace("_", " ")}</option>' for t in config_data['tests']]) }
            </select>
        </div>

        <div id="candControls" class="control-group" style="display:none">
            <label>Target Candidate</label>
            <select id="candSelect" onchange="update()">
                { "".join([f'<option value="{c}">{c.replace("_", " ")}</option>' for c in config_data['candidates']]) }
            </select>
        </div>

        <div id="metricControls" class="control-group">
            <label>Primary Metric</label>
            <select id="metricSelect" onchange="update()">
                <option value="n_success_and_constraints">Goal + Constraint Success</option>
                <option value="avg_time">Computation Time (ms)</option>
            </select>
        </div>

        <div id="variantControls" class="control-group" style="display:none">
            <label>Target Variant</label>
            <select id="variantSelect" onchange="update()">
                { "".join([f'<option value="{v}">{v}</option>' for v in config_data['major_variants']]) }
            </select>
        </div>
    </nav>

    <main>
        <div class="header-meta">
            <div>
                <h2 id="title" style="margin:0; font-size: 2rem;">Analysis Overview</h2>
                <p id="subtitle" style="color: #94a3b8; margin: 0.5rem 0 0;">Aggregated metrics across multi-seed experiments</p>
            </div>
            <div class="badge">Session: 2026.05.13</div>
        </div>

        <div class="plot-container">
            <div id="loader" class="loading-overlay"><span>Loading Visualization...</span></div>
            <img id="display" src="" alt="Plot View">
        </div>

        <div class="footer-grid">
            <div class="stat-card"><div class="stat-label">TOTAL SEEDS</div><div class="stat-val">5</div></div>
            <div class="stat-card"><div class="stat-label">CANDIDATES</div><div class="stat-val">{len(candidates)}</div></div>
            <div class="stat-card"><div class="stat-label">ENVIRONMENTS</div><div class="stat-val">{len(test_types)}</div></div>
            <div class="stat-card"><div class="stat-label">STATUS</div><div class="stat-val" style="color: var(--success);">READY</div></div>
        </div>
    </main>

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
            const loader = document.getElementById('loader');
            loader.classList.add('active');
            
            const mode = document.getElementById('mode').value;
            const test = document.getElementById('testSelect').value;
            const cand = document.getElementById('candSelect').value;
            const metric = document.getElementById('metricSelect').value;
            const variant = document.getElementById('variantSelect').value;
            
            let src = "";
            let tText = "";
            let sText = "";

            if (mode === 'env') {{
                src = `${{base}}by_test_env/${{test}}/candidate_comp_${{metric}}.png`;
                tText = `Environment Analysis: ${{test.replace('_', ' ')}}`;
                sText = `Comparing all candidates on ${{metric.replace('_', ' ')}}`;
            }} else if (mode === 'cand') {{
                src = `${{base}}by_candidate/${{cand}}/cross_env_${{metric}}.png`;
                tText = `Performance Profile: ${{cand.replace('_', ' ')}}`;
                sText = `Behavior across all environments on ${{metric.replace('_', ' ')}}`;
            }} else if (mode === 'matrix') {{
                src = `${{base}}matrices/matrix_success_${{variant}}.png`;
                tText = `Global Success Matrix: ${{variant}}`;
                sText = `Cross-environment vs Cross-candidate success density`;
            }}

            const img = document.getElementById('display');
            img.onload = () => loader.classList.remove('active');
            img.src = src;
            document.getElementById('title').innerText = tText;
            document.getElementById('subtitle').innerText = sText;
        }}

        updateUI();
    </script>
</body>
</html>
"""
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
