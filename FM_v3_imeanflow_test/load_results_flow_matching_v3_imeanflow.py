#!/usr/bin/env python3
"""
Load and display iMeanFlow evaluation results.

Usage:
    python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py --results-dir logs/avoiding-d3il/flow_matching_v3_imeanflow/evaluation_results/imf
"""

import argparse
import json
import os
from pathlib import Path


def load_and_display_results(results_dir='logs/avoiding-d3il/flow_matching_v3_imeanflow/evaluation_results/imf'):
    """Load evaluation results from eval_results.json and print a summary table."""
    results_file = os.path.join(results_dir, 'eval_results.json')

    if not os.path.exists(results_file):
        print(f'[ load ] ERROR: No results file found: {results_file}')
        return False

    with open(results_file, 'r') as f:
        results = json.load(f)

    print()
    print('=' * 80)
    print('iMeanFlow (iMF-PCC) Evaluation Results')
    print('=' * 80)
    print()
    print('Per-Seed Results:')
    print('-' * 80)
    print(f"{'Seed':>6s} {'MSE Error':>14s} {'Std Dev':>14s} {'Num Samples':>12s}")
    print('-' * 80)

    mse_errors = []
    sorted_keys = sorted(results.keys(), key=lambda value: int(value) if str(value).isdigit() else 999)

    for seed_key in sorted_keys:
        result = results[seed_key]
        seed = result.get('seed', seed_key)
        mse_error = float(result.get('mse_error', 0.0))
        mse_std = float(result.get('mse_std', 0.0))
        num_samples = int(result.get('num_samples', 0))

        print(f'{int(seed):6d} {mse_error:14.6f} {mse_std:14.6f} {num_samples:12d}')
        mse_errors.append(mse_error)

    print('-' * 80)

    if mse_errors:
        mean_mse = sum(mse_errors) / len(mse_errors)
        variance = sum((value - mean_mse) ** 2 for value in mse_errors) / len(mse_errors)
        print(f"{'MEAN':>6s} {mean_mse:14.6f} {variance ** 0.5:14.6f} {len(mse_errors):12d}")

    print('-' * 80)
    print()
    print('=' * 80)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Load iMF evaluation results')
    parser.add_argument('--results-dir', type=str, default='logs/avoiding-d3il/flow_matching_v3_imeanflow/evaluation_results/imf', help='Results directory.')
    args = parser.parse_args()

    print('[ load ] iMeanFlow Results Loader')
    print(f'[ load ] Loading from: {args.results_dir}')

    success = load_and_display_results(args.results_dir)

    if success:
        print('[ load ] ✓ Complete')
    else:
        print('[ load ] ✗ Failed to load results')


if __name__ == '__main__':
    main()
