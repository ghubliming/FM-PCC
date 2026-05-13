"""
DA Tool v2: Multi-Candidate Batch Analysis CLI

Main entry point for cross-candidate comparison analysis.
Automatically discovers experimental candidates, loads & aggregates their results,
generates comparison plots and rankings.

Usage:
    python main_da_batch.py --parent-path logs/avoiding-d3il/plans
    python main_da_batch.py --parent-path logs/avoiding-d3il/plans --candidates A,C,E
    python main_da_batch.py --parent-path logs/avoiding-d3il/plans --candidate-names "aw=1,aw=10,dpcc"
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Import v2 modules
from multi_candidate_discovery import discover_candidates_recursive, filter_candidates, assign_custom_names, get_candidate_summary
from batch_data_loader import BatchDataLoader
from batch_aggregator import BatchAggregator
from batch_visualizer import BatchVisualizer
from batch_reporter import BatchReporter
from utils import setup_logger, create_output_directory


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FM v3 ODE-Selectable: Multi-Candidate Batch Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all candidates in parent directory
  python main_da_batch.py --parent-path logs/avoiding-d3il/plans
  
  # Select specific candidates
  python main_da_batch.py --parent-path logs/avoiding-d3il/plans --candidates A,C,E
  
  # Use custom names instead of auto A/B/C
  python main_da_batch.py --parent-path logs/avoiding-d3il/plans \\
    --candidate-names "aw=1,aw=5,aw=10,dpcc-baseline"
  
  # Skip plot generation for speed
  python main_da_batch.py --parent-path logs/avoiding-d3il/plans --no-plots
  
  # Debug mode
  python main_da_batch.py --parent-path logs/avoiding-d3il/plans --verbose
        """
    )
    
    parser.add_argument(
        '--parent-path',
        required=True,
        help='Parent directory containing candidate subfolders'
    )
    
    parser.add_argument(
        '--output-path',
        default='Data_Analysis/analysis_results',
        help='Output directory for results (default: Data_Analysis/analysis_results)'
    )
    
    parser.add_argument(
        '--candidates',
        default=None,
        help='Comma-separated candidate letters to analyze (e.g., "A,C,E"). Default: all'
    )
    
    parser.add_argument(
        '--candidate-names',
        default=None,
        help='Comma-separated custom names for candidates (e.g., "aw=1,aw=10,dpcc")'
    )
    
    parser.add_argument(
        '--seeds',
        default=None,
        help='Comma-separated seed numbers (default: 6,7,8,9,10)'
    )
    
    parser.add_argument(
        '--variants',
        default=None,
        help='Comma-separated variant names to analyze (default: all)'
    )
    
    parser.add_argument(
        '--constraint-types',
        default=None,
        help='Comma-separated constraint types (default: all)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation (faster)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main batch analysis pipeline."""
    args = parse_arguments()
    
    # Setup output directory
    output_base = args.output_path
    output_dir, output_timestamp = create_output_directory(
        output_base,
        'FM_V3_BATCH',
        return_timestamp=True,
    )
    
    # Setup logging
    log_file = os.path.join(output_dir, 'logs', 'batch_analysis.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = setup_logger(
        'DA_Batch_v2',
        log_file,
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    logger.info("="*70)
    logger.info("FM v3 ODE-Selectable: Multi-Candidate Batch Analysis (v2)")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    try:
        # ============================================================================
        # PHASE 1: Discover Candidates
        # ============================================================================
        logger.info("[PHASE 1/5] AUTO-DISCOVERING CANDIDATES")
        logger.info("-" * 70)
        
        candidates = discover_candidates_recursive(args.parent_path, max_depth=10)
        
        if not candidates:
            logger.error("No candidates found. Exiting.")
            return 1
        
        logger.info(get_candidate_summary(candidates))
        
        # ============================================================================
        # PHASE 2: Filter Candidates (if specified)
        # ============================================================================
        if args.candidates:
            logger.info("[PHASE 2A/5] FILTERING CANDIDATES")
            logger.info("-" * 70)
            candidates = filter_candidates(candidates, args.candidates)
            logger.info(f"Filtered to: {list(candidates.keys())}")
            logger.info("")
        
        # ============================================================================
        # PHASE 2B: Assign Custom Names (if specified)
        # ============================================================================
        if args.candidate_names:
            logger.info("[PHASE 2B/5] ASSIGNING CUSTOM NAMES")
            logger.info("-" * 70)
            candidates = assign_custom_names(candidates, args.candidate_names)
            logger.info(f"Custom names assigned")
            logger.info("")
        
        # ============================================================================
        # PHASE 3: Load Data
        # ============================================================================
        logger.info("[PHASE 3/5] BATCH LOADING DATA")
        logger.info("-" * 70)
        
        # Parse optional arguments
        seeds = None
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
        
        variants = None
        if args.variants:
            variants = [v.strip() for v in args.variants.split(',')]
        
        constraint_types = None
        if args.constraint_types:
            constraint_types = [c.strip() for c in args.constraint_types.split(',')]
        
        loader = BatchDataLoader(verbose=args.verbose)
        batch_data = loader.load_all_candidates(
            candidates,
            variants=variants,
            constraint_types=constraint_types
        )
        
        # Save loading log
        loader.save_batch_loading_log(
            os.path.join(output_dir, 'logs', 'batch_loading.log')
        )
        logger.info("")
        
        # ============================================================================
        # PHASE 4: Aggregate Data
        # ============================================================================
        logger.info("[PHASE 4/5] BATCH AGGREGATION")
        logger.info("-" * 70)
        
        aggregator = BatchAggregator()
        candidate_stats = aggregator.aggregate_all_candidates(batch_data)
        
        # Print ranking summary
        aggregator.print_ranking_summary()
        logger.info("")
        
        # ============================================================================
        # PHASE 5A: Generate Visualizations
        # ============================================================================
        if not args.no_plots:
            logger.info("[PHASE 5A/5] GENERATING COMPARISON PLOTS")
            logger.info("-" * 70)
            
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            visualizer = BatchVisualizer(candidate_stats, aggregator.candidate_aggregators)
            visualizer.plot_all(plots_dir, show=False)
            
            logger.info("")
        
        # ============================================================================
        # PHASE 5B: Generate Reports
        # ============================================================================
        logger.info("[PHASE 5B/5] GENERATING REPORTS")
        logger.info("-" * 70)
        
        reporter = BatchReporter(
            candidate_stats,
            aggregator.ranked_candidates,
            candidates_info=candidates
        )
        reporter.save_all_reports(output_dir)
        
        logger.info("")
        
        # ============================================================================
        # Summary
        # ============================================================================
        logger.info("="*70)
        logger.info("BATCH ANALYSIS COMPLETE")
        logger.info("="*70)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("")
        logger.info("Key outputs:")
        logger.info(f"  - Plots: {os.path.join(output_dir, 'plots')}")
        logger.info(f"  - Summary: {os.path.join(output_dir, 'candidates_summary.txt')}")
        logger.info(f"  - Rankings: {os.path.join(output_dir, 'candidates_ranking.csv')}")
        logger.info(f"  - Logs: {os.path.join(output_dir, 'logs')}")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during batch analysis: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
