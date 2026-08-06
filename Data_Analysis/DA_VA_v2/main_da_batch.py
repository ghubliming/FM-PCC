"""
DA_VA_v2 — multi-candidate batch analysis CLI.

Reads Gen14 (`mix_visual_aligning_test`) and Gen7 (`fm_visual_aligning_test`)
evaluation trees — and, since discovery is path-shape driven, the state-only
avoiding trees DA_Code_v3 was written for — and emits the CSV cube plus optional
plots.

Usage
-----
  # one engine tree
  python main_da_batch.py --parent-path logs/aligning-d3il-visual/plans

  # four-arm Gen14 comparison (comma-separated parents are merged into one run)
  python main_da_batch.py \\
      --parent-path logs/aligning-d3il-visual/plans,logs/visual-aligning-dpcc/plans \\
      --output-path Data_Analysis/analysis_results/va2_$(date +%Y%m%d_%H%M%S)

  # narrow it down
  python main_da_batch.py --parent-path ... --geos combined_5 --splits test
  python main_da_batch.py --parent-path ... --variants dpcc-c,dpcc-r,diffuser
  python main_da_batch.py --parent-path ... --candidates 1,3 --candidate-names "fm,af"

  # PNGs (off by default — the CSVs are the product)
  python main_da_batch.py --parent-path ... --plots
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from aggregator import Aggregator
from data_loader import UnitLoader
from discovery import (
    assign_custom_names,
    discover_candidates,
    discover_units,
    filter_candidates,
    get_candidate_summary,
    write_discovery_manifest,
)
from reporter import Reporter
from utils import (
    create_output_directory,
    parse_int_list,
    parse_str_list,
    prepare_output_directory,
    setup_logger,
    update_results_manifest,
)

DEFAULT_OUTPUT = './va2_batch_analysis'


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='DA_VA_v2 — visual-aligning multi-candidate batch analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('--parent-path', required=True,
                        help='Parent directory (or comma-separated list) holding '
                             'candidate folders. Candidates from several trees are '
                             'merged into one comparison run.')
    parser.add_argument('--output-path', default=DEFAULT_OUTPUT,
                        help=f'Output directory (default: {DEFAULT_OUTPUT}, which '
                             f'gets a timestamped subfolder)')
    parser.add_argument('--candidates', default=None,
                        help='Comma-separated candidate indices to keep (e.g. "1,3")')
    parser.add_argument('--candidate-names', default=None,
                        help='Comma-separated display names, in candidate index order')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seeds (default: every seed found)')
    parser.add_argument('--variants', default=None,
                        help='Comma-separated variant names, `_train_set` suffix '
                             'already stripped (default: all)')
    parser.add_argument('--geos', default=None,
                        help='Comma-separated geometry names, e.g. '
                             '"combined_5,combined_5-tightened" (default: all)')
    parser.add_argument('--splits', default=None,
                        help='Comma-separated splits: test, train (default: both)')
    parser.add_argument('--source', choices=['auto', 'npz', 'json'], default='auto',
                        help='Where per-rollout data comes from. auto = npz when '
                             'present, else diagnostics JSONs (pre-npz Gen7 runs).')
    parser.add_argument('--no-diagnostics-scan', action='store_true',
                        help='Skip reading diagnostics/rollout_*_stats.json. Faster, '
                             'but the D1 frozen-rollout mask and the JSON-only final '
                             'pose columns are then unavailable.')
    parser.add_argument('--plots', action='store_true',
                        help='Also render the default PNG set (slow; off by default)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Accepted for symmetry with DA_Code_v3; plots are '
                             'already off unless --plots is given.')
    parser.add_argument('--verbose', action='store_true', help='Debug logging')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    if args.output_path == DEFAULT_OUTPUT:
        output_dir, _ = create_output_directory(args.output_path,
                                                return_timestamp=True)
    else:
        output_dir = prepare_output_directory(args.output_path)

    manifest = update_results_manifest(output_dir)

    logger = setup_logger('DA_VA_v2',
                          os.path.join(output_dir, 'logs', 'analysis.log'),
                          level=logging.DEBUG if args.verbose else logging.INFO)
    started = datetime.now()
    logger.info('=' * 70)
    logger.info('DA_VA_v2 — visual-aligning batch analysis')
    logger.info('=' * 70)
    logger.info(f'Start:  {started.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Parent: {args.parent_path}')
    logger.info(f'Output: {output_dir}')
    if manifest:
        logger.info(f'Manifest: {manifest}')

    try:
        # ── 1. discover candidates ───────────────────────────────────────────
        logger.info('[1/5] Discovering candidates')
        seeds = parse_int_list(args.seeds)
        candidates = discover_candidates(args.parent_path, seed_list=seeds)
        if not candidates:
            logger.error('No candidates found. Nothing to do.')
            return 1
        if args.candidates:
            candidates = filter_candidates(candidates, args.candidates)
        if args.candidate_names:
            candidates = assign_custom_names(candidates, args.candidate_names)
        if not candidates:
            logger.error('All candidates filtered out. Nothing to do.')
            return 1
        logger.info('\n' + get_candidate_summary(candidates))

        # ── 2. enumerate result units ────────────────────────────────────────
        logger.info('[2/5] Enumerating result units')
        variants = parse_str_list(args.variants)
        geos = parse_str_list(args.geos)
        splits = parse_str_list(args.splits)

        units = []
        for index, info in sorted(candidates.items()):
            found = discover_units(index, info, seeds=seeds, splits=splits,
                                   geos=geos, variants=variants)
            logger.info(f'  Candidate {index} ({info["name"][:60]}): '
                        f'{len(found)} units')
            units.extend(found)
        if not units:
            logger.error('No result units matched the filters. Nothing to do.')
            return 1
        write_discovery_manifest(
            os.path.join(output_dir, 'logs', 'discovery_manifest.json'),
            candidates, units)

        # ── 3. load ──────────────────────────────────────────────────────────
        logger.info(f'[3/5] Loading {len(units)} units')
        loader = UnitLoader(source=args.source,
                            scan_diagnostics=not args.no_diagnostics_scan,
                            verbose=args.verbose)
        loaded = []
        for position, unit in enumerate(units, 1):
            result = loader.load(unit)
            if result is not None:
                loaded.append((unit, result))
            if position % 25 == 0 or position == len(units):
                logger.info(f'  {position}/{len(units)} units processed')
        loader.save_log(os.path.join(output_dir, 'logs', 'loading.log'))
        summary = loader.summary()
        logger.info(f'  loaded={summary["files_loaded"]} '
                    f'failed={summary["files_failed"]} '
                    f'skipped={summary["files_skipped"]}')
        if not loaded:
            logger.error('Every unit failed to load — see logs/loading.log.')
            return 1

        # ── 4. aggregate ─────────────────────────────────────────────────────
        logger.info('[4/5] Aggregating')
        aggregator = Aggregator().build(loaded)
        aggregator.print_ranking_summary()

        # ── 5. report (+ optional plots) ─────────────────────────────────────
        logger.info('[5/5] Writing reports')
        reporter = Reporter(
            aggregator,
            candidates_info=candidates,
            run_meta={
                'parent_path': args.parent_path,
                'source': args.source,
                'diagnostics_scan': not args.no_diagnostics_scan,
                'units_loaded': summary['files_loaded'],
                'units_failed': summary['files_failed'],
            })
        reporter.save_all(output_dir)

        if args.plots and not args.no_plots:
            logger.info('Rendering plots')
            from visualizer import Visualizer
            Visualizer(aggregator).plot_all(os.path.join(output_dir, 'plots'))

        update_results_manifest(output_dir)

        elapsed = (datetime.now() - started).total_seconds()
        logger.info('=' * 70)
        logger.info(f'COMPLETE in {elapsed:.1f}s — {output_dir}')
        logger.info('  per_rollout_detail.csv               every rollout, wide')
        logger.info('  va2_units_long.csv                   per seed/geo/variant/mask')
        logger.info('  va2_aggregated_long.csv              pooled over seeds')
        logger.info('  candidates_multidimensional_*.csv    DA_Code_v3 visualizer')
        logger.info('  va_candidates_dynamic.csv            VA visualizer')
        logger.info('  data_quality.csv                     frozen / circuit breaker')
        logger.info('=' * 70)
        return 0

    except Exception as exc:                                      # noqa: BLE001
        logger.exception(f'Fatal error: {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
