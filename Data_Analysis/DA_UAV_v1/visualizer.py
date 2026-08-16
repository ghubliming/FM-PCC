"""
DA_UAV_v1 — optional plots.

Opt-in (`--plots`), off by default: the CSVs plus the HTML visualizer are the
intended path and matplotlib is the slowest stage of a batch run by a wide
margin. Kept small on purpose — six figures that answer questions the summary
table cannot, all driven off `aggregator.agg_long` / `aggregator.k_sweep` so
they cannot disagree with the CSVs.

The K-sweep figures (`00_*`) are the ones this tool exists for; the rest are the
usual per-variant bars carried over from DA_VA_v2.

Every figure is written with mask='all'; the proj_valid view lives in the CSVs.
"""

import logging
import os

import numpy as np

from config import (
    ACCURACY_FALLBACK_METRIC,
    ACCURACY_METRIC,
    MASK_FLAG_COLUMN,
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
        written += self._k_sweep(plots_dir, ACCURACY_METRIC, '00_k_sweep_accuracy')
        written += self._k_sweep(plots_dir, TIME_METRIC, '00_k_sweep_time')
        written += self._k_sweep(plots_dir, MASK_FLAG_COLUMN, '00_k_sweep_cb_tripped')
        written += self._pareto(plots_dir)
        written += self._bar_per_variant(table, plots_dir, ACCURACY_METRIC)
        written += self._bar_per_variant(table, plots_dir, 'n_success')
        written += self._time_split(table, plots_dir)
        written += self._quality_bar(plots_dir)
        logger.info(f'{len(written)} plots written to {plots_dir}')
        return written

    # ── the Gen15 plots ───────────────────────────────────────────────────────
    def _k_sweep(self, plots_dir, metric, stem):
        """`metric` vs K, one line per (engine, variant), one figure per scene+geo.

        This is the shape of the Gen15 claim (PLAN §7.3): mf/af hold success as K
        falls where fm collapses, and the released wall-clock shows up as fewer
        circuit-breaker trips. A curve per engine is the only rendering in which
        "holds" and "collapses" are visible at all.
        """
        plt = self._pyplot()
        table = self.agg.k_sweep
        if table is None or table.empty:
            return []
        rows = table[(table['mask'] == 'all') & (table['metric'] == metric)]
        if rows.empty:
            return []

        written = []
        group_keys = [k for k in ('scene', 'geo') if k in rows.columns]
        for keys, block in rows.groupby(group_keys, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
            scale = 100.0 if metric in PERCENTAGE_METRICS else 1.0
            drawn = 0
            line_keys = [k for k in ('engine', 'variant') if k in block.columns]
            for i, (line, cell) in enumerate(block.groupby(line_keys, observed=True)):
                line = line if isinstance(line, tuple) else (line,)
                ordered = cell.sort_values('K')
                if ordered['K'].nunique() < 2:
                    continue     # a single point is not a sweep
                ax.errorbar(
                    ordered['K'].to_numpy(dtype=float),
                    ordered['mean'].to_numpy(dtype=float) * scale,
                    yerr=(ordered['std'].to_numpy(dtype=float) * scale
                          if 'std' in ordered.columns else None),
                    marker='o', capsize=3, lw=1.6,
                    label='/'.join(str(x) for x in line),
                    color=PLOT_CONFIG['colors'][i % len(PLOT_CONFIG['colors'])])
                drawn += 1
            if not drawn:
                plt.close(fig)
                continue
            ax.set_xscale('log', base=2)
            ax.set_xlabel('K  (NFE budget per plan)')
            ax.set_ylabel(_label(metric) + (' (%)' if scale == 100 else ''))
            ax.set_title(f'{_label(metric)} vs K — {" / ".join(str(k) for k in keys)}',
                         fontsize=PLOT_CONFIG['title_size'])
            ax.legend(fontsize=PLOT_CONFIG['legend_size'], ncol=2)
            fig.tight_layout()
            path = os.path.join(plots_dir,
                                f'{stem}_{"_".join(_slug(k) for k in keys)}.png')
            fig.savefig(path, dpi=PLOT_CONFIG['dpi'])
            plt.close(fig)
            written.append(path)
        return written

    def _pareto(self, plots_dir):
        """Accuracy vs replan time, one point per candidate, front joined by a line.

        The front is drawn because "good" here means Pareto-dominant, and a
        scatter alone invites reading the top-left-most point as the winner even
        when nothing dominates it.
        """
        plt = self._pyplot()
        stats = self.agg.candidate_stats
        points = {key: entry for key, entry in stats.items()
                  if not np.isnan(entry.get('accuracy', np.nan))
                  and not np.isnan(entry.get('time_ms', np.nan))}
        if not points:
            return []
        front = set(self.agg.pareto_front())

        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        for i, (key, entry) in enumerate(sorted(points.items())):
            ax.scatter(entry['time_ms'], entry['accuracy'] * 100,
                       s=180 if key in front else 110,
                       marker='*' if key in front else 'o',
                       color=PLOT_CONFIG['colors'][i % len(PLOT_CONFIG['colors'])],
                       edgecolors='black' if key in front else 'none',
                       linewidths=1.0,
                       label=f'C{key}: {str(entry["FolderName"])[:44]}')
            ax.annotate(f'C{key}', (entry['time_ms'], entry['accuracy'] * 100),
                        textcoords='offset points', xytext=(6, 6), fontsize=9)
        if front:
            ordered = sorted((points[k]['time_ms'], points[k]['accuracy'] * 100)
                             for k in front)
            ax.plot([p[0] for p in ordered], [p[1] for p in ordered],
                    ls='--', lw=1.0, color='#555555', zorder=0,
                    label='Pareto front')
        ax.set_xlabel(_label(TIME_METRIC))
        ax.set_ylabel(_label(ACCURACY_METRIC) + ' (%)')
        ax.set_title('Success+constraint vs computation time  (star = Pareto front)',
                     fontsize=PLOT_CONFIG['title_size'])
        ax.legend(fontsize=PLOT_CONFIG['legend_size'], loc='best', ncol=2)
        fig.tight_layout()
        path = os.path.join(plots_dir, '01_pareto_accuracy_vs_time.png')
        fig.savefig(path, dpi=PLOT_CONFIG['dpi'])
        plt.close(fig)
        return [path]

    def _time_split(self, table, plots_dir):
        """Where the 30 ms went: generation vs projection, stacked per variant.

        The Gen15 mechanism is that cheaper generation RELEASES budget to the
        projector. That is a statement about the split of the per-plan wall
        clock, not about its total, so the total-only bar cannot show it.
        """
        plt = self._pyplot()
        parts = ('fm_ms', 'proj_ms')
        rows = table[table['metric'].isin(parts)]
        if rows.empty:
            return []
        written = []
        for (split, geo), block in rows.groupby(['split', 'geo'], observed=True):
            pivot = block.pivot_table(index='variant', columns='metric',
                                      values='mean', observed=True)
            if pivot.empty:
                continue
            variants = list(pivot.index)
            positions = np.arange(len(variants))
            fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
            bottom = np.zeros(len(variants))
            for i, part in enumerate(parts):
                if part not in pivot.columns:
                    continue
                values = pivot[part].fillna(0).to_numpy(dtype=float)
                ax.bar(positions, values, bottom=bottom, width=0.7,
                       label=_label(part),
                       color=PLOT_CONFIG['colors'][i * 3 % len(PLOT_CONFIG['colors'])])
                bottom += values
            budget = table[table['metric'] == 'budget_ms']['mean'].mean()
            if budget and not np.isnan(budget):
                ax.axhline(budget, ls='--', color='crimson', lw=1.3,
                           label=f'real-time budget {budget:.0f} ms')
            ax.set_xticks(positions)
            ax.set_xticklabels([str(v) for v in variants], rotation=45, ha='right')
            ax.set_ylabel('ms per replan')
            ax.set_title(f'Per-plan wall clock split — geo={geo}, split={split}',
                         fontsize=PLOT_CONFIG['title_size'])
            ax.legend(fontsize=PLOT_CONFIG['legend_size'])
            fig.tight_layout()
            path = os.path.join(plots_dir,
                                f'03_time_split_{_slug(split)}_{_slug(geo)}.png')
            fig.savefig(path, dpi=PLOT_CONFIG['dpi'])
            plt.close(fig)
            written.append(path)
        return written

    # ── carried over from DA_VA_v2 ────────────────────────────────────────────
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
                plots_dir, f'02_variants_{metric}_{_slug(split)}_{_slug(geo)}.png')
            fig.savefig(path, dpi=PLOT_CONFIG['dpi'])
            plt.close(fig)
            written.append(path)
        return written

    def _quality_bar(self, plots_dir):
        """Circuit-breaker trips and missing timing per unit — the caveat plot."""
        plt = self._pyplot()
        quality = self.agg.quality
        if quality is None or quality.empty or 'n_cb_tripped' not in quality.columns:
            return []
        flagged = quality[(quality['n_cb_tripped'].fillna(0) > 0)
                          | (quality.get('timing_missing', 0) > 0)]
        if flagged.empty:
            return []

        labels = [f'C{row["Candidate"]}/{row["variant"]}'
                  for _, row in flagged.iterrows()]
        positions = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.35), 6))
        ax.bar(positions - 0.2, flagged['n_cb_tripped'].fillna(0).to_numpy(), width=0.4,
               label='circuit-breaker rollouts', color=PLOT_CONFIG['colors'][3])
        if 'n_rollouts' in flagged.columns:
            ax.bar(positions + 0.2, flagged['n_rollouts'].fillna(0).to_numpy(), width=0.4,
                   label='rollouts in unit', color=PLOT_CONFIG['colors'][7], alpha=0.5)
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
