"""
DA Visual Aligning — Single Candidate Analysis CLI

Loads per-rollout data for one experiment folder, aggregates,
generates flat PNGs (00a–04b) and reports.

Usage:
    python main_da.py --input-path logs/visual-aligning-dpcc/plans/flow_matching_v3_imeanflow
    python main_da.py --input-path ... --seed 6
    python main_da.py --input-path ... --source json   # pre-U10.2 runs
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from config import DEFAULT_PROJECTION_VARIANTS, ACTIVE_SEED, OUTPUT_FOLDER_PREFIX
from utils import setup_logger, create_output_directory
from data_loader import DataLoader
from aggregator import DataAggregator
from visualizer import DataVisualizer
from reporter import Reporter


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Visual Aligning DA — Single Candidate Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_da.py --input-path logs/visual-aligning-dpcc/plans/my_model
  python main_da.py --input-path ... --seed 7
  python main_da.py --input-path ... --source json   # pre-U10.2 runs
  python main_da.py --input-path ... --variants diffuser,dpcc-c,dpcc-r
        """
    )
    parser.add_argument('--input-path', required=True,
                        help='Path to model experiment folder (contains seed subfolders)')
    parser.add_argument('--output-path', default='./va_single_analysis',
                        help='Base output directory (default: ./va_single_analysis)')
    parser.add_argument('--seed', type=int, default=None,
                        help=f'Seed to load (default: ACTIVE_SEED={ACTIVE_SEED})')
    parser.add_argument('--source', choices=['npz', 'json'], default='npz',
                        help='Data source: npz (default, U10.2+) or json (pre-U10.2)')
    parser.add_argument('--variants', default=None,
                        help='Comma-separated variant names (default: all in results folder)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation (faster)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    return parser.parse_args()


def main():
    args = parse_arguments()

    output_dir = create_output_directory(args.output_path, OUTPUT_FOLDER_PREFIX)
    logs_dir   = os.path.join(output_dir, 'logs')
    plots_dir  = os.path.join(output_dir, 'plots')

    logger = setup_logger(
        'DA_VA_Single',
        os.path.join(logs_dir, 'analysis.log'),
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    logger.info('=' * 80)
    logger.info('FM Visual Aligning — Single Candidate Analysis')
    logger.info('=' * 80)
    logger.info(f'Input:  {args.input_path}')
    logger.info(f'Source: {args.source}')
    logger.info(f'Output: {output_dir}')

    seed = args.seed if args.seed is not None else ACTIVE_SEED
    variants = [v.strip() for v in args.variants.split(',')] if args.variants else None

    # ------------------------------------------------------------------
    # Phase 1: Load
    # ------------------------------------------------------------------
    logger.info('--- PHASE 1: LOADING ---')
    loader   = DataLoader(verbose=args.verbose, source=args.source)
    raw_data = loader.load_results(args.input_path, seed=seed, variants=variants)
    loader.save_loading_log(os.path.join(logs_dir, 'data_loading.log'))
    summary  = loader.get_loading_summary()
    logger.info(f'Loaded {summary["files_loaded"]}/{summary["files_found"]}')

    if summary['files_loaded'] == 0:
        logger.error('No data loaded. Aborting.')
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Aggregate
    # ------------------------------------------------------------------
    logger.info('--- PHASE 2: AGGREGATION ---')
    aggregator = DataAggregator(raw_data)
    aggregator.aggregate_all()

    # ------------------------------------------------------------------
    # Phase 3: Report
    # ------------------------------------------------------------------
    logger.info('--- PHASE 3: REPORTING ---')
    reporter = Reporter(aggregator)
    reporter.save_all_reports(output_dir, {**summary, 'source': args.source})

    # ------------------------------------------------------------------
    # Phase 4: Visualize
    # ------------------------------------------------------------------
    if not args.no_plots:
        logger.info('--- PHASE 4: VISUALIZATION ---')
        try:
            viz = DataVisualizer(aggregator)
            viz.plot_all(plots_dir)
            logger.info(f'Plots created: {viz.plots_created}')
        except Exception as e:
            logger.error(f'Visualization failed: {e}', exc_info=True)
    else:
        logger.info('Skipping visualization (--no-plots)')

    logger.info('=' * 80)
    logger.info('COMPLETE')
    logger.info(f'Output: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f'\nFatal error: {e}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
