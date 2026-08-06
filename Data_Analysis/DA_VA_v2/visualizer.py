"""
DA_VA_v2 — optional plots.

Opt-in (`--plots`), and off by default: the CSVs plus the HTML visualizer are the
intended path, and matplotlib is the slowest part of a batch run by a wide margin.
Kept small on purpose — five plots that answer the questions the summary table
cannot, all driven off `aggregator.agg_long` so they cannot disagree with the CSVs.

Every figure is written with mask='all'; the unfrozen view lives in the CSVs.
"""

import logging
import os

import numpy as np

from config import (
    ACCURACY_FALLBACK_METRIC,
    ACCURACY_METRIC,
    METRIC_LABELS,
    PERCENTAGE_METRICS,
    PLOT_CONFIG,
    TIME_METRIC,
)

logger = logging.getLogger(__name__)


class Visualizer:
    """Render the small default plot set into `<output>/plots/`."""

    def __init__(self, aggregator):
        self.agg = aggregator
        self._plt = None

    # ── entry point ───────────────────────────────────────────────────────────
    def plot_all(self, plots_dir):
        plt = self._pyplot()
        if plt is None:
            return []
        os.makedirs(plots_dir, exist_ok=True)
        table = self.agg.agg_long
        if table is None or table.empty:
            logger.warning('Nothing to plot — aggregated table is empty.')
            return []
        table = table[table['mask'] == 'all']

        written = []
        written += self._bar_per_variant(table, plots_dir, ACCURACY_METRIC)
        written += self._bar_per_variant(table, plots_dir, 'n_success')
        written += self._bar_per_variant(table, plots_dir, 'collision_free_completed')
        written += self._pareto(table, plots_dir)
        written += self._quality_bar(plots_dir)
        logger.info(f'{len(written)} plots written to {plots_dir}')
        return written

    # ── plots ─────────────────────────────────────────────────────────────────
    def _bar_per_variant(self, table, plots_dir, metric):
        """Grouped bars: variant on x, one bar per candidate, one figure per geo."""
        plt = self._pyplot()
        rows = table[table['metric'] == metric]
        if rows.empty and metric == ACCURACY_METRIC:
            rows = table[table['metric'] == ACCURACY_FALLBACK_METRIC]
            metric = ACCURACY_FALLBACK_METRIC
        if rows.empty:
            return []

        written = []
        for (split, geo), block in rows.groupby(['split', 'geo'], observed=True):
            pivot = block.pivot_table(index='variant', columns='Candidate',
                                      values='mean', observed=True)
            errors = block.pivot_table(index='variant', columns='Candidate',
                                       values='std', observed=True)
            if pivot.empty:
                continue
            scale = 100.0 if metric in PERCENTAGE_METRICS else 1.0
            variants = list(pivot.index)
            candidates = list(pivot.columns)
            width = 0.8 / max(len(candidates), 1)
            positions = np.arange(len(variants))

            fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
            for i, candidate in enumerate(candidates):
                ax.bar(positions + i * width,
                       pivot[candidate].to_numpy() * scale,
                       width=width,
                       yerr=(errors[candidate].to_numpy() * scale
                             if candidate in errors.columns else None),
                       capsize=2,
                       label=f'C{candidate}',
                       color=PLOT_CONFIG['colors'][i % len(PLOT_CONFIG['colors'])])
            ax.set_xticks(positions + width * (len(candidates) - 1) / 2)
            ax.set_xticklabels([str(v) for v in variants], rotation=45, ha='right')
            ax.set_ylabel(_label(metric) + (' (%)' if scale == 100 else ''))
            ax.set_title(f'{_label(metric)} — geo={geo}, split={split}',
                         fontsize=PLOT_CONFIG['title_size'])
            ax.legend(fontsize=PLOT_CONFIG['legend_size'], ncol=2)
            fig.tight_layout()
            path = os.path.join(
                plots_dir, f'01_variants_{metric}_{_slug(split)}_{_slug(geo)}.png')
            fig.savefig(path, dpi=PLOT_CONFIG['dpi'])
            plt.close(fig)
            written.append(path)
        return written

    def _pareto(self, table, plots_dir):
        """Accuracy vs replan time, one point per candidate."""
        plt = self._pyplot()
        stats = self.agg.candidate_stats
        points = {key: entry for key, entry in stats.items()
                  if not np.isnan(entry.get('accuracy', np.nan))
                  and not np.isnan(entry.get('time_ms', np.nan))}
        if not points:
            return []

        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        for i, (key, entry) in enumerate(sorted(points.items())):
            ax.scatter(entry['time_ms'], entry['accuracy'] * 100, s=140,
                       color=PLOT_CONFIG['colors'][i % len(PLOT_CONFIG['colors'])],
                       label=f'C{key}: {entry["FolderName"][:40]}')
            ax.annotate(f'C{key}', (entry['time_ms'], entry['accuracy'] * 100),
                        textcoords='offset points', xytext=(6, 6), fontsize=9)
        ax.set_xlabel(_label(TIME_METRIC))
        ax.set_ylabel(_label(ACCURACY_METRIC) + ' (%)')
        ax.set_title('Accuracy vs computation time', fontsize=PLOT_CONFIG['title_size'])
        ax.legend(fontsize=PLOT_CONFIG['legend_size'], loc='best')
        fig.tight_layout()
        path = os.path.join(plots_dir, '00_pareto_accuracy_vs_time.png')
        fig.savefig(path, dpi=PLOT_CONFIG['dpi'])
        plt.close(fig)
        return [path]

    def _quality_bar(self, plots_dir):
        """Frozen rollouts and circuit-breaker trips per unit — the caveat plot."""
        plt = self._pyplot()
        quality = self.agg.quality
        if quality is None or quality.empty:
            return []
        flagged = quality[(quality.get('n_frozen', 0) > 0)
                          | (quality.get('n_cb_tripped', 0) > 0)] \
            if 'n_frozen' in quality.columns else quality.iloc[0:0]
        if flagged.empty:
            return []

        labels = [f'C{row["Candidate"]}/{row["geo"]}/{row["variant"]}'
                  for _, row in flagged.iterrows()]
        positions = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.35), 6))
        ax.bar(positions - 0.2, flagged['n_frozen'].to_numpy(), width=0.4,
               label='frozen rollouts', color=PLOT_CONFIG['colors'][3])
        ax.bar(positions + 0.2, flagged['n_cb_tripped'].to_numpy(), width=0.4,
               label='circuit-breaker rollouts', color=PLOT_CONFIG['colors'][7])
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel('rollouts')
        ax.set_title('Data quality — rollouts that did not run the stated policy',
                     fontsize=PLOT_CONFIG['title_size'])
        ax.legend(fontsize=PLOT_CONFIG['legend_size'])
        fig.tight_layout()
        path = os.path.join(plots_dir, '09_data_quality.png')
        fig.savefig(path, dpi=PLOT_CONFIG['dpi'])
        plt.close(fig)
        return [path]

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _pyplot(self):
        if self._plt is not None:
            return self._plt
        try:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            try:
                plt.style.use(PLOT_CONFIG['style'])
            except Exception:                                     # noqa: BLE001
                pass
            plt.rcParams['font.size'] = PLOT_CONFIG['font_size']
            self._plt = plt
        except Exception as exc:                                  # noqa: BLE001
            logger.warning(f'matplotlib unavailable — plots skipped ({exc})')
            self._plt = None
        return self._plt


def _label(metric):
    return METRIC_LABELS.get(metric, metric)


def _slug(text):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(text))
