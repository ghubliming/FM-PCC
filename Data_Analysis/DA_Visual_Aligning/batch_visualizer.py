"""
Batch Visualizer for Visual Aligning DA.

Generates cross-candidate comparison plots (00a–04b flat PNGs).
No hierarchical analysis.
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from config import PLOT_CONFIG

logger = logging.getLogger(__name__)

CANDIDATE_COLORS = {
    1: '#e74c3c', 2: '#f39c12', 3: '#f1c40f', 4: '#3498db',
    5: '#2ecc71', 6: '#9b59b6', 7: '#1abc9c', 8: '#e67e22',
    9: '#34495e', 10: '#c0392b',
}


class BatchVisualizer:
    """Cross-candidate comparison plots for visual aligning."""

    def __init__(self, candidate_stats, candidate_aggregators):
        self.candidate_stats       = candidate_stats
        self.candidate_aggregators = candidate_aggregators
        try:
            plt.style.use(PLOT_CONFIG.get('style', 'seaborn-v0_8-darkgrid'))
        except Exception:
            pass
        rcParams['figure.figsize'] = PLOT_CONFIG.get('figsize', (12, 7))
        rcParams['savefig.dpi']    = PLOT_CONFIG.get('dpi', 300)
        rcParams['font.size']      = PLOT_CONFIG.get('font_size', 11)

    def _color(self, letter):
        return CANDIDATE_COLORS.get(letter, '#95a5a6')

    def _save(self, fig, output_dir, filename):
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=PLOT_CONFIG.get('dpi', 300), bbox_inches='tight')
        plt.close(fig)
        logger.info(f'Saved: {filename}')

    # ------------------------------------------------------------------
    # 00a — Pareto: success_rate vs mean_distance (aggregate avg per candidate)
    # ------------------------------------------------------------------
    def plot_candidate_pareto_frontier(self, output_dir, show=False):
        logger.info('Generating 00a Pareto success vs distance...')
        pts = {}
        for letter, stats in self.candidate_stats.items():
            sr = stats.get('success_rate')
            md = stats.get('mean_distance')
            if sr is not None and md is not None:
                pts[letter] = (sr * 100, md)
        if not pts:
            return
        fig, ax = plt.subplots(figsize=(10, 7))
        for letter, (sr, md) in sorted(pts.items()):
            c = self._color(letter)
            ax.scatter(md, sr, s=400, color=c, edgecolors='black', linewidth=1.5, zorder=3)
            ax.annotate(letter, (md, sr), fontsize=12, fontweight='bold', ha='center', va='center', color='white', zorder=4)
        ax.set_xlabel('Mean Final Distance (m) — lower is better', fontweight='bold')
        ax.set_ylabel('Success Rate (%) — higher is better', fontweight='bold')
        ax.set_title('Cross-Candidate Pareto: Success Rate vs Distance',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '00a_pareto.png')

    # ------------------------------------------------------------------
    # 00b — Pareto: success_rate vs avg_time
    # ------------------------------------------------------------------
    def plot_candidate_pareto_time(self, output_dir, show=False):
        logger.info('Generating 00b Pareto success vs time...')
        pts = {}
        for letter, stats in self.candidate_stats.items():
            sr = stats.get('success_rate')
            at = stats.get('avg_time')
            if sr is not None and at is not None:
                pts[letter] = (sr * 100, at)
        if not pts:
            return
        fig, ax = plt.subplots(figsize=(10, 7))
        for letter, (sr, at) in sorted(pts.items()):
            c = self._color(letter)
            ax.scatter(at, sr, s=400, color=c, edgecolors='black', linewidth=1.5, zorder=3)
            ax.annotate(letter, (at, sr), fontsize=12, fontweight='bold', ha='center', va='center', color='white', zorder=4)
        ax.set_xlabel('Avg Inference Time / Replan (s) — lower is better', fontweight='bold')
        ax.set_ylabel('Success Rate (%) — higher is better', fontweight='bold')
        ax.set_title('Cross-Candidate Pareto: Success Rate vs Time',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '00b_pareto.png')

    # ------------------------------------------------------------------
    # 01a — Bar: success_rate per candidate
    # ------------------------------------------------------------------
    def plot_candidate_success_comparison(self, output_dir, show=False):
        logger.info('Generating 01a success bar...')
        letters = sorted(self.candidate_stats)
        vals    = [self.candidate_stats[l].get('success_rate', 0) * 100 for l in letters]
        colors  = [self._color(l) for l in letters]
        fig, ax = plt.subplots(figsize=(max(8, len(letters) * 1.2), 6))
        bars = ax.bar(range(len(letters)), vals, color=colors, edgecolor='black', alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels(letters)
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.set_title('Cross-Candidate Success Rate Comparison',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '01a_success.png')

    # ------------------------------------------------------------------
    # 01b — Heatmap: variant × rollout, stacked by candidate
    # ------------------------------------------------------------------
    def plot_candidate_success_heatmap(self, output_dir, show=False):
        logger.info('Generating 01b success heatmap (per rollout)...')
        # One subplot per candidate
        letters = sorted(self.candidate_aggregators)
        if not letters:
            return
        n = len(letters)
        fig, axes = plt.subplots(1, n, figsize=(max(12, n * 5), 7), squeeze=False)
        axes = axes[0]
        for ax, letter in zip(axes, letters):
            agg = self.candidate_aggregators[letter]
            per = agg.per_rollout
            variants = sorted(v for v in per if 'n_success' in per[v])
            if not variants:
                ax.set_title(f'Candidate {letter} — no data')
                continue
            max_n = max(len(per[v]['n_success']) for v in variants)
            mat = np.full((len(variants), max_n), np.nan)
            for i, v in enumerate(variants):
                arr = per[v]['n_success']
                mat[i, :len(arr)] = arr.astype(float)
            im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax.set_yticks(np.arange(len(variants)))
            ax.set_yticklabels(variants, fontsize=6)
            ax.set_xlabel('Rollout', fontsize=8)
            ax.set_title(f'Cand {letter}', fontsize=9, fontweight='bold')
        fig.suptitle('Per-Rollout Success Heatmap (green=success)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        plt.tight_layout()
        self._save(fig, output_dir, '01b_success_rollouts.png')

    # ------------------------------------------------------------------
    # 02a — Bar: mean distance per candidate
    # ------------------------------------------------------------------
    def plot_candidate_distance_comparison(self, output_dir, show=False):
        logger.info('Generating 02a distance bar...')
        letters = sorted(self.candidate_stats)
        vals    = [self.candidate_stats[l].get('mean_distance', np.nan) for l in letters]
        colors  = [self._color(l) for l in letters]
        fig, ax = plt.subplots(figsize=(max(8, len(letters) * 1.2), 6))
        ax.bar(range(len(letters)), vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels(letters)
        ax.set_ylabel('Mean Final Distance (m)', fontweight='bold')
        ax.set_title('Cross-Candidate Mean Final Distance',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '02a_mean_distance.png')

    # ------------------------------------------------------------------
    # 02b — Boxplot: distance distribution per candidate
    # ------------------------------------------------------------------
    def plot_candidate_distance_boxplot(self, output_dir, show=False):
        logger.info('Generating 02b distance boxplot...')
        letters = sorted(self.candidate_aggregators)
        data, labels, colors = [], [], []
        for letter in letters:
            agg = self.candidate_aggregators[letter]
            per = agg.per_rollout
            all_dists = []
            for v in per:
                if 'mean_dist_per_rollout' in per[v]:
                    all_dists.extend(per[v]['mean_dist_per_rollout'].tolist())
            if all_dists:
                data.append(all_dists)
                labels.append(letter)
                colors.append(self._color(letter))
        if not data:
            return
        fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.2), 6))
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_ylabel('Final Distance (m)', fontweight='bold')
        ax.set_title('Cross-Candidate Distance Distribution (all rollouts)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '02b_distance_rollouts.png')

    # ------------------------------------------------------------------
    # 03a — Bar: avg max tracking error
    # ------------------------------------------------------------------
    def plot_candidate_tracking_error(self, output_dir, show=False):
        logger.info('Generating 03a tracking error bar...')
        letters = sorted(self.candidate_stats)
        vals    = [self.candidate_stats[l].get('max_phys_error', np.nan) for l in letters]
        colors  = [self._color(l) for l in letters]
        fig, ax = plt.subplots(figsize=(max(8, len(letters) * 1.2), 6))
        ax.bar(range(len(letters)), vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels(letters)
        ax.set_ylabel('Avg Max Physical Tracking Error (m)', fontweight='bold')
        ax.set_title('Cross-Candidate Avg PD Tracking Error',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '03a_tracking_error.png')

    # ------------------------------------------------------------------
    # 03b — Bar: avg steps per candidate (placeholder at batch level)
    # ------------------------------------------------------------------
    def plot_candidate_steps(self, output_dir, show=False):
        logger.info('Generating 03b steps bar...')
        letters, vals, colors = [], [], []
        for letter in sorted(self.candidate_aggregators):
            agg = self.candidate_aggregators[letter]
            steps_means = [
                agg.summary.get(v, {}).get('n_steps', {}).get('mean', np.nan)
                for v in agg.summary
                if 'n_steps' in agg.summary.get(v, {})
            ]
            if steps_means:
                letters.append(letter)
                vals.append(float(np.mean(steps_means)))
                colors.append(self._color(letter))
        if not letters:
            return
        fig, ax = plt.subplots(figsize=(max(8, len(letters) * 1.2), 6))
        ax.bar(range(len(letters)), vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels(letters)
        ax.set_ylabel('Avg Steps per Rollout', fontweight='bold')
        ax.set_title('Cross-Candidate Avg Steps',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '03b_steps.png')

    # ------------------------------------------------------------------
    # 04a — Bar: avg time per candidate
    # ------------------------------------------------------------------
    def plot_candidate_time_comparison(self, output_dir, show=False):
        logger.info('Generating 04a time bar...')
        letters = sorted(self.candidate_stats)
        vals    = [self.candidate_stats[l].get('avg_time', np.nan) for l in letters]
        colors  = [self._color(l) for l in letters]
        fig, ax = plt.subplots(figsize=(max(8, len(letters) * 1.2), 6))
        ax.bar(range(len(letters)), vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels(letters)
        ax.set_ylabel('Avg Inference Time / Replan (s)', fontweight='bold')
        ax.set_title('Cross-Candidate Avg Inference Time',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '04a_time.png')

    # ------------------------------------------------------------------
    # 04b — Scatter: context_init_xy_dist vs mean_distance per candidate
    # ------------------------------------------------------------------
    def plot_candidate_context_scatter(self, output_dir, show=False):
        logger.info('Generating 04b context scatter...')
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        handles = []
        for letter in sorted(self.candidate_aggregators):
            agg = self.candidate_aggregators[letter]
            per = agg.per_rollout
            x_all, y_all = [], []
            for v in per:
                if 'context_init_xy_dist' in per[v] and 'mean_dist_per_rollout' in per[v]:
                    x_all.extend(per[v]['context_init_xy_dist'].tolist())
                    y_all.extend(per[v]['mean_dist_per_rollout'].tolist())
            if x_all:
                c = self._color(letter)
                ax.scatter(x_all, y_all, s=25, color=c, alpha=0.5, edgecolors='none', label=letter)
                handles.append(mpatches.Patch(color=c, label=f'Candidate {letter}'))
        ax.set_xlabel('Init Box-Target Distance (m) — scene difficulty', fontweight='bold')
        ax.set_ylabel('Final Distance (m)', fontweight='bold')
        ax.set_title('Scene Difficulty vs Final Distance (all rollouts)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.legend(handles=handles, loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '04b_context_scatter.png')

    # ------------------------------------------------------------------
    # plot_all — flat 00a–04b only
    # ------------------------------------------------------------------
    def plot_all(self, output_dir, show=False):
        os.makedirs(output_dir, exist_ok=True)
        logger.info('Generating all cross-candidate comparison plots...')
        for fn, label in [
            (self.plot_candidate_pareto_frontier,     '00a'),
            (self.plot_candidate_pareto_time,         '00b'),
            (self.plot_candidate_success_comparison,  '01a'),
            (self.plot_candidate_success_heatmap,     '01b'),
            (self.plot_candidate_distance_comparison, '02a'),
            (self.plot_candidate_distance_boxplot,    '02b'),
            (self.plot_candidate_tracking_error,      '03a'),
            (self.plot_candidate_steps,               '03b'),
            (self.plot_candidate_time_comparison,     '04a'),
            (self.plot_candidate_context_scatter,     '04b'),
        ]:
            try:
                fn(output_dir, show=show)
            except Exception as e:
                logger.error(f'{label} failed: {e}')
        # self.plot_matrix_analysis(output_dir, show=show)  # HIERARCHICAL-DEACTIVATED

    # HIERARCHICAL-DEACTIVATED
    def plot_matrix_analysis(self, output_dir, show=False):
        pass
