"""
Visualizer for a single Visual Aligning candidate.

Flat PNG outputs only (00a–04b). Hierarchical analysis deactivated.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from config import PLOT_CONFIG, METRIC_LABELS

logger = logging.getLogger(__name__)


class DataVisualizer:
    """Create per-candidate flat PNGs from aggregated data."""

    def __init__(self, aggregator):
        self.aggregator   = aggregator
        self.plots_created = 0
        try:
            plt.style.use(PLOT_CONFIG['style'])
        except Exception:
            pass
        plt.rcParams.update({
            'font.size': PLOT_CONFIG['font_size'],
            'figure.dpi': PLOT_CONFIG['dpi'],
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _colors(self, n):
        return PLOT_CONFIG['colors'][:n]

    def _save(self, fig, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close(fig)
        self.plots_created += 1
        logger.info(f'Saved: {filename}')

    def _variants_ordered(self, require_metric=None):
        """Return sorted variant list, optionally filtered to those with a metric."""
        variants = sorted(self.aggregator.summary.keys())
        if require_metric:
            variants = [v for v in variants if require_metric in self.aggregator.summary.get(v, {})]
        return variants

    # ------------------------------------------------------------------
    # 00a — Pareto: success_rate vs mean_distance
    # ------------------------------------------------------------------
    def plot_pareto_success_distance(self, output_dir):
        logger.info('Creating 00a Pareto success vs distance...')
        summary = self.aggregator.summary
        pts = {}
        for variant, metrics in summary.items():
            sr = metrics.get('n_success', metrics.get('success_rate', {})).get('mean')
            md = metrics.get('mean_dist_per_rollout', {}).get('mean')
            if sr is not None and md is not None:
                pts[variant] = (sr, md)
        if not pts:
            logger.warning('00a: no data')
            return
        variants = sorted(pts.keys())
        colors   = self._colors(len(variants))
        fig, ax  = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        for (var, (sr, md)), c in zip(pts.items(), colors):
            ax.scatter(md, sr * 100, s=250, color=c, edgecolors='black', linewidth=1.2, zorder=3)
            ax.annotate(var, (md, sr * 100), fontsize=7, ha='center', va='bottom')
        ax.set_xlabel('Mean Final Distance (m) — lower is better', fontweight='bold')
        ax.set_ylabel('Success Rate (%) — higher is better', fontweight='bold')
        ax.set_title('Pareto: Success Rate vs Final Distance', fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '00a_pareto.png')

    # ------------------------------------------------------------------
    # 00b — Pareto: success_rate vs avg_time
    # ------------------------------------------------------------------
    def plot_pareto_success_time(self, output_dir):
        logger.info('Creating 00b Pareto success vs time...')
        summary = self.aggregator.summary
        pts = {}
        for variant, metrics in summary.items():
            sr = metrics.get('n_success', metrics.get('success_rate', {})).get('mean')
            at = metrics.get('avg_time', {}).get('mean')
            if sr is not None and at is not None:
                pts[variant] = (sr, at)
        if not pts:
            return
        variants = sorted(pts.keys())
        colors   = self._colors(len(variants))
        fig, ax  = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        for (var, (sr, at)), c in zip(pts.items(), colors):
            ax.scatter(at, sr * 100, s=250, color=c, edgecolors='black', linewidth=1.2, zorder=3)
            ax.annotate(var, (at, sr * 100), fontsize=7, ha='center', va='bottom')
        ax.set_xlabel('Avg Inference Time/Replan (s) — lower is better', fontweight='bold')
        ax.set_ylabel('Success Rate (%) — higher is better', fontweight='bold')
        ax.set_title('Pareto: Success Rate vs Inference Time', fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '00b_pareto.png')

    # ------------------------------------------------------------------
    # 01a — Bar: success_rate per variant
    # ------------------------------------------------------------------
    def plot_success_bar(self, output_dir):
        logger.info('Creating 01a success bar...')
        variants = self._variants_ordered(require_metric='n_success')
        if not variants:
            variants = self._variants_ordered(require_metric='success_rate')
        if not variants:
            return
        means = []
        for v in variants:
            m = self.aggregator.summary[v]
            sr = m.get('n_success', m.get('success_rate', {})).get('mean', 0.0)
            means.append(sr * 100)
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        x = np.arange(len(variants))
        bars = ax.bar(x, means, color=self._colors(len(variants)), edgecolor='black', alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=40, ha='right')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.set_title('Success Rate per Variant', fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '01a_success.png')

    # ------------------------------------------------------------------
    # 01b — Heatmap: variant × rollout_idx coloured by success
    # ------------------------------------------------------------------
    def plot_success_rollout_heatmap(self, output_dir):
        logger.info('Creating 01b success rollout heatmap...')
        per_rollout = self.aggregator.per_rollout
        variants = sorted(v for v in per_rollout if 'n_success' in per_rollout[v])
        if not variants:
            return
        # Build matrix
        max_n = max(len(per_rollout[v]['n_success']) for v in variants)
        mat = np.full((len(variants), max_n), np.nan)
        for i, v in enumerate(variants):
            arr = per_rollout[v]['n_success']
            mat[i, :len(arr)] = arr.astype(float)
        fig, ax = plt.subplots(figsize=(max(12, max_n * 0.4), max(5, len(variants) * 0.5)))
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks(np.arange(len(variants)))
        ax.set_yticklabels(variants, fontsize=8)
        ax.set_xlabel('Rollout Index', fontweight='bold')
        ax.set_title('Per-Rollout Success (green=success, red=fail)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        plt.colorbar(im, ax=ax, label='Success')
        plt.tight_layout()
        self._save(fig, output_dir, '01b_success_rollouts.png')

    # ------------------------------------------------------------------
    # 02a — Bar: mean_dist_per_rollout (mean across rollouts)
    # ------------------------------------------------------------------
    def plot_distance_bar(self, output_dir):
        logger.info('Creating 02a distance bar...')
        summary  = self.aggregator.summary
        variants = sorted(v for v in summary if 'mean_dist_per_rollout' in summary[v])
        if not variants:
            return
        means = [summary[v]['mean_dist_per_rollout']['mean'] for v in variants]
        stds  = [summary[v]['mean_dist_per_rollout']['std']  for v in variants]
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=stds, capsize=4, color=self._colors(len(variants)),
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=40, ha='right')
        ax.set_ylabel('Mean Final Distance (m) — lower better', fontweight='bold')
        ax.set_title('Mean Final Box-Target Distance per Variant',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '02a_mean_distance.png')

    # ------------------------------------------------------------------
    # 02b — Boxplot: distance distribution per variant (30 rollouts)
    # ------------------------------------------------------------------
    def plot_distance_boxplot(self, output_dir):
        logger.info('Creating 02b distance boxplot...')
        per_rollout = self.aggregator.per_rollout
        variants = sorted(v for v in per_rollout if 'mean_dist_per_rollout' in per_rollout[v])
        if not variants:
            return
        data    = [per_rollout[v]['mean_dist_per_rollout'] for v in variants]
        colors  = self._colors(len(variants))
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        bp = ax.boxplot(data, labels=variants, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticklabels(variants, rotation=40, ha='right')
        ax.set_ylabel('Final Distance (m)', fontweight='bold')
        ax.set_title('Distance Distribution per Variant (30 rollouts)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '02b_distance_rollouts.png')

    # ------------------------------------------------------------------
    # 03a — Bar: avg max_phys_error
    # ------------------------------------------------------------------
    def plot_tracking_error_bar(self, output_dir):
        logger.info('Creating 03a tracking error bar...')
        summary  = self.aggregator.summary
        variants = sorted(v for v in summary if 'max_phys_error_per_rollout' in summary[v])
        if not variants:
            return
        means = [summary[v]['max_phys_error_per_rollout']['mean'] for v in variants]
        stds  = [summary[v]['max_phys_error_per_rollout']['std']  for v in variants]
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=stds, capsize=4, color=self._colors(len(variants)),
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=40, ha='right')
        ax.set_ylabel('Avg Max Physical Tracking Error (m)', fontweight='bold')
        ax.set_title('Avg Max PD Tracking Error per Variant',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '03a_tracking_error.png')

    # ------------------------------------------------------------------
    # 03b — Bar: avg steps
    # ------------------------------------------------------------------
    def plot_steps_bar(self, output_dir):
        logger.info('Creating 03b steps bar...')
        summary  = self.aggregator.summary
        variants = sorted(v for v in summary if 'n_steps' in summary[v])
        if not variants:
            return
        means = [summary[v]['n_steps']['mean'] for v in variants]
        stds  = [summary[v]['n_steps']['std']  for v in variants]
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=stds, capsize=4, color=self._colors(len(variants)),
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=40, ha='right')
        ax.set_ylabel('Avg Steps per Rollout', fontweight='bold')
        ax.set_title('Avg Planning Steps per Variant',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '03b_steps.png')

    # ------------------------------------------------------------------
    # 04a — Bar: avg inference time/replan
    # ------------------------------------------------------------------
    def plot_time_bar(self, output_dir):
        logger.info('Creating 04a time bar...')
        summary  = self.aggregator.summary
        variants = sorted(v for v in summary if 'avg_time' in summary[v])
        if not variants:
            return
        means = [summary[v]['avg_time']['mean'] for v in variants]
        stds  = [summary[v]['avg_time']['std']  for v in variants]
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=stds, capsize=4, color=self._colors(len(variants)),
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=40, ha='right')
        ax.set_ylabel('Avg Inference Time / Replan (s)', fontweight='bold')
        ax.set_title('Avg Inference Time per Replan per Variant',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '04a_time.png')

    # ------------------------------------------------------------------
    # 04b — Scatter: context_init_xy_dist vs mean_distance, coloured by variant
    # ------------------------------------------------------------------
    def plot_context_scatter(self, output_dir):
        logger.info('Creating 04b context scatter...')
        per_rollout = self.aggregator.per_rollout
        variants = sorted(
            v for v in per_rollout
            if 'context_init_xy_dist' in per_rollout[v]
            and 'mean_dist_per_rollout' in per_rollout[v]
        )
        if not variants:
            return
        colors  = self._colors(len(variants))
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        handles = []
        for var, c in zip(variants, colors):
            x_arr = per_rollout[var]['context_init_xy_dist']
            y_arr = per_rollout[var]['mean_dist_per_rollout']
            ax.scatter(x_arr, y_arr, s=40, color=c, alpha=0.6, edgecolors='none', label=var)
            patch = mpatches.Patch(color=c, label=var)
            handles.append(patch)
        ax.set_xlabel('Init Box-Target Distance (m) — scene difficulty', fontweight='bold')
        ax.set_ylabel('Final Distance (m)', fontweight='bold')
        ax.set_title('Init Scene Difficulty vs Final Distance (per rollout)',
                     fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
        ax.legend(handles=handles, loc='best', fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, output_dir, '04b_context_scatter.png')

    # ------------------------------------------------------------------
    # plot_all — flat 00a–04b only
    # ------------------------------------------------------------------
    def plot_all(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        try: self.plot_pareto_success_distance(output_dir)
        except Exception as e: logger.error(f'00a failed: {e}')
        try: self.plot_pareto_success_time(output_dir)
        except Exception as e: logger.error(f'00b failed: {e}')
        try: self.plot_success_bar(output_dir)
        except Exception as e: logger.error(f'01a failed: {e}')
        try: self.plot_success_rollout_heatmap(output_dir)
        except Exception as e: logger.error(f'01b failed: {e}')
        try: self.plot_distance_bar(output_dir)
        except Exception as e: logger.error(f'02a failed: {e}')
        try: self.plot_distance_boxplot(output_dir)
        except Exception as e: logger.error(f'02b failed: {e}')
        try: self.plot_tracking_error_bar(output_dir)
        except Exception as e: logger.error(f'03a failed: {e}')
        try: self.plot_steps_bar(output_dir)
        except Exception as e: logger.error(f'03b failed: {e}')
        try: self.plot_time_bar(output_dir)
        except Exception as e: logger.error(f'04a failed: {e}')
        try: self.plot_context_scatter(output_dir)
        except Exception as e: logger.error(f'04b failed: {e}')
        logger.info(f'Total plots: {self.plots_created}')

    # HIERARCHICAL-DEACTIVATED
    def plot_matrix_analysis(self, output_dir, show=False):
        pass  # HIERARCHICAL-DEACTIVATED — kept as empty stub
