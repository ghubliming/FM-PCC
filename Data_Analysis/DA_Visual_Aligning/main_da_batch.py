"""
DA Visual Aligning — Batch Analysis CLI

Discovers candidate model folders, loads per-rollout data, aggregates,
generates flat PNGs (00a–04b) and reports.

Usage:
    python main_da_batch.py --parent-path logs/visual-aligning-dpcc/plans
    python main_da_batch.py --parent-path ... --seed 6
    python main_da_batch.py --parent-path ... --source json   # pre-U10.2 runs
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime

from multi_candidate_discovery import discover_candidates_recursive, filter_candidates, assign_custom_names, get_candidate_summary
from batch_data_loader import BatchDataLoader
from batch_aggregator import BatchAggregator
from batch_visualizer import BatchVisualizer
from batch_reporter import BatchReporter
from utils import setup_logger, create_output_directory


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Visual Aligning DA — Multi-Candidate Batch Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_da_batch.py --parent-path logs/visual-aligning-dpcc/plans
  python main_da_batch.py --parent-path ... --seed 7 --source npz
  python main_da_batch.py --parent-path ... --source json   # pre-U10.2 runs
  python main_da_batch.py --parent-path ... --candidates A,C --no-plots
        """
    )
    parser.add_argument('--parent-path', required=True,
                        help='Parent directory containing candidate subfolders')
    parser.add_argument('--output-path', default='./va_batch_analysis',
                        help='Output directory (default: ./va_batch_analysis)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed number to load (default: ACTIVE_SEED=6)')
    parser.add_argument('--source', choices=['npz', 'json'], default='npz',
                        help='Data source: npz (default, U10.2+) or json (pre-U10.2)')
    parser.add_argument('--candidates', default=None,
                        help='Comma-separated candidate letters to analyse (default: all)')
    parser.add_argument('--candidate-names', default=None,
                        help='Comma-separated custom names (e.g. "fm_v1,fm_v2")')
    parser.add_argument('--variants', default=None,
                        help='Comma-separated variant names (default: all)')
    parser.add_argument('--geo-variant', default=None,
                        help=('Geometric constraint subfolder name to step into '
                              'before discovering projection variants '
                              '(e.g. "combined_5"). '
                              'Use when eval was run with geo_constraint_variants in the YAML. '
                              'Omit for old flat-schema runs (no geo layer).'))
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation (faster)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    return parser.parse_args()


def main():
    args = parse_arguments()

    output_dir = args.output_path
    if output_dir == './va_batch_analysis':
        output_dir, _ = create_output_directory(output_dir, 'VA_BATCH', return_timestamp=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    # Manifest for HTML visualizer
    try:
        abs_output    = os.path.abspath(output_dir)
        parent_results = os.path.dirname(abs_output)
        if os.path.exists(parent_results):
            batches = [d for d in os.listdir(parent_results)
                       if os.path.isdir(os.path.join(parent_results, d))
                       and d not in ('plots', 'logs')]
            manifest_path = os.path.join(parent_results, 'results_manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump({'batches': sorted(batches, reverse=True)}, f, indent=2)
            print(f'[ DA ] Updated manifest: {manifest_path}')
    except Exception as e:
        print(f'[ DA ] Warning: manifest update failed: {e}')

    log_file = os.path.join(output_dir, 'logs', 'batch_analysis.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger('DA_VA_Batch', log_file,
                          level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info('=' * 70)
    logger.info('FM Visual Aligning — Multi-Candidate Batch Analysis')
    logger.info('=' * 70)
    logger.info(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Source: {args.source}  Seed: {args.seed}')
    logger.info(f'Output: {output_dir}')

    try:
        # ----------------------------------------------------------
        # Phase 1: Discover candidates
        # ----------------------------------------------------------
        logger.info('[1/5] Discovering candidates...')
        from config import ACTIVE_SEED
        seed = args.seed if args.seed is not None else ACTIVE_SEED
        candidates = discover_candidates_recursive(args.parent_path, seed_list=[seed], max_depth=10)
        if not candidates:
            logger.error('No candidates found. Exiting.')
            return 1
        logger.info(get_candidate_summary(candidates))

        # ----------------------------------------------------------
        # Phase 2: Filter / rename
        # ----------------------------------------------------------
        if args.candidates:
            logger.info('[2A/5] Filtering candidates...')
            candidates = filter_candidates(candidates, args.candidates)
        if args.candidate_names:
            logger.info('[2B/5] Assigning custom names...')
            candidates = assign_custom_names(candidates, args.candidate_names)

        # ----------------------------------------------------------
        # Phase 3: Load data
        # ----------------------------------------------------------
        logger.info('[3/5] Loading data...')
        variants = [v.strip() for v in args.variants.split(',')] if args.variants else None
        loader   = BatchDataLoader(verbose=args.verbose, source=args.source,
                                   geo_variant=args.geo_variant)
        batch_data = loader.load_all_candidates(candidates, variants=variants, seed=seed)
        loader.save_batch_loading_log(os.path.join(output_dir, 'logs', 'batch_loading.log'))

        # ----------------------------------------------------------
        # Phase 4: Aggregate
        # ----------------------------------------------------------
        logger.info('[4/5] Aggregating...')
        aggregator = BatchAggregator()
        candidate_stats = aggregator.aggregate_all_candidates(batch_data)
        aggregator.print_ranking_summary()

        # ----------------------------------------------------------
        # Phase 5A: Plots
        # ----------------------------------------------------------
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        if not args.no_plots:
            logger.info('[5A/5] Generating plots...')
            viz = BatchVisualizer(candidate_stats, aggregator.candidate_aggregators)
            viz.plot_all(plots_dir, show=False)
        else:
            logger.info('[5A/5] Skipping plots (--no-plots)')

        # ----------------------------------------------------------
        # Phase 5B: Reports
        # ----------------------------------------------------------
        logger.info('[5B/5] Generating reports...')
        reporter = BatchReporter(candidate_stats, aggregator.ranked_candidates,
                                 candidates_info=candidates, aggregator=aggregator)
        reporter.save_all_reports(output_dir)

        logger.info('=' * 70)
        logger.info('COMPLETE')
        logger.info(f'Output: {output_dir}')
        logger.info('=' * 70)
        return 0

    except Exception as e:
        logger.exception(f'Fatal error: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
