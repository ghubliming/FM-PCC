#!/usr/bin/env python3
"""
Main Data Analysis Script for FM v3 ODE-Selectable Evaluation Results.

Orchestrates data loading, aggregation, visualization, and reporting.

Usage:
    python main_da.py --input-path /path/to/eval/results --output-path /path/to/output
    python main_da.py --input-path /workspaces/FM-PCC/FM_v3_ode_selectable_test --output-path ./analysis_output
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Import modules
from config import (
    DEFAULT_SEEDS, DEFAULT_PROJECTION_VARIANTS, DEFAULT_CONSTRAINT_TYPES,
    DEFAULT_HALFSPACE_VARIANTS, OUTPUT_FOLDER_PREFIX
)
from utils import setup_logger, create_output_directory
from data_loader import DataLoader
from aggregator import DataAggregator
from visualizer import DataVisualizer
from reporter import Reporter


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Data Analysis for FM v3 ODE-Selectable Evaluation Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_da.py --input-path /path/to/results
  python main_da.py --input-path /path/to/results --output-path ./my_output
  python main_da.py --input-path /path/to/results --seeds 6,7,8,9,10
  python main_da.py --input-path /path/to/results --variants dpcc-c,dpcc-r,diffuser
        """
    )
    
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to evaluation results directory (contains seed folders)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='./fm_v3_analysis_output',
        help='Base output directory (default: ./fm_v3_analysis_output)'
    )
    
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='Comma-separated seed numbers (default: 6,7,8,9,10)'
    )
    
    parser.add_argument(
        '--variants',
        type=str,
        default=None,
        help='Comma-separated projection variant names (default: all variants)'
    )
    
    parser.add_argument(
        '--constraint-types',
        type=str,
        default=None,
        help='Comma-separated constraint types (default: halfspace,obstacles,dynamics,bounds)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation (faster)'
    )
    
    return parser.parse_args()


def parse_comma_separated(value, default):
    """Parse comma-separated string to list."""
    if value is None:
        return default
    return [v.strip() for v in value.split(',')]


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory with timestamp
    output_dir = create_output_directory(args.output_path, OUTPUT_FOLDER_PREFIX)
    logs_dir = os.path.join(output_dir, 'logs')
    plots_dir = os.path.join(output_dir, 'plots')
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(
        'DataAnalysis',
        os.path.join(logs_dir, 'analysis.log'),
        level=log_level
    )
    
    logger.info('=' * 80)
    logger.info('FM v3 ODE-Selectable Evaluation Analysis')
    logger.info('=' * 80)
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Input path: {args.input_path}')
    
    # Parse parameters
    seeds = parse_comma_separated(args.seeds, DEFAULT_SEEDS)
    variants = parse_comma_separated(args.variants, DEFAULT_PROJECTION_VARIANTS)
    constraint_types = parse_comma_separated(args.constraint_types, DEFAULT_CONSTRAINT_TYPES)
    halfspace_variants = DEFAULT_HALFSPACE_VARIANTS
    
    logger.info(f'Seeds: {seeds}')
    logger.info(f'Variants: {len(variants)} variants')
    logger.info(f'Constraint types: {constraint_types}')
    logger.info(f'Halfspace variants: {halfspace_variants}')
    
    # ==================== PHASE 1: DATA LOADING ====================
    logger.info('\n--- PHASE 1: DATA LOADING ---')
    loader = DataLoader()
    raw_data = loader.load_results(
        args.input_path,
        seeds,
        variants,
        constraint_types,
        halfspace_variants
    )
    
    # Save loading log
    loader.save_loading_log(os.path.join(logs_dir, 'data_loading.log'))
    loading_summary = loader.get_loading_summary()
    logger.info(f'Loading Summary: {loading_summary["files_loaded"]}/{loading_summary["files_found"]} files loaded')
    
    if loading_summary['files_loaded'] == 0:
        logger.error('No data files loaded! Aborting.')
        sys.exit(1)
    
    # ==================== PHASE 2: DATA AGGREGATION ====================
    logger.info('\n--- PHASE 2: DATA AGGREGATION ---')
    aggregator = DataAggregator(raw_data)
    aggregated = aggregator.aggregate_all()
    logger.info('Aggregation complete')
    
    # ==================== PHASE 3: REPORTING ====================
    logger.info('\n--- PHASE 3: REPORTING ---')
    reporter = Reporter(aggregator)
    reporter.save_all_reports(output_dir, loading_summary)
    logger.info('Reports saved')
    
    # ==================== PHASE 4: VISUALIZATION ====================
    if not args.no_plots:
        logger.info('\n--- PHASE 4: VISUALIZATION ---')
        try:
            visualizer = DataVisualizer(aggregator)
            
            # Create all key metric plots (includes Pareto frontier)
            logger.info('Creating metric plots (including Pareto frontier)...')
            visualizer.plot_all_key_metrics(plots_dir)
            
            logger.info(f'Visualization complete ({visualizer.plots_created} plots created)')
        except Exception as e:
            logger.error(f'Visualization failed: {str(e)}', exc_info=True)
    else:
        logger.info('Skipping visualization (--no-plots flag)')
    
    # ==================== COMPLETION ====================
    logger.info('\n' + '=' * 80)
    logger.info('ANALYSIS COMPLETE')
    logger.info('=' * 80)
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Plots: {plots_dir}')
    logger.info(f'Reports: {output_dir}')
    logger.info(f'Logs: {logs_dir}')
    logger.info('\nKey files:')
    logger.info(f'  - results_summary.txt')
    logger.info(f'  - results_by_variant.csv')
    logger.info(f'  - results_by_constraint.csv')
    logger.info(f'  - detailed_results.csv')
    logger.info(f'  - plots/00_pareto_frontier_accuracy_vs_time.png (THESIS MAIN)')
    logger.info('\n' + '=' * 80 + '\n')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nInterrupted by user.')
        sys.exit(1)
    except Exception as e:
        print(f'\n\nFatal error: {str(e)}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
