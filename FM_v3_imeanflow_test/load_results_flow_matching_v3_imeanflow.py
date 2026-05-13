#!/usr/bin/env python3
"""
Load and display iMeanFlow evaluation results.

Simple results aggregation + formatting (standard FM-PCC pattern).

Usage:
    python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py --results-dir evaluation_results/imf
"""

import argparse
import json
import os
from pathlib import Path


def load_and_display_results(results_dir='evaluation_results'):
    """Load and display evaluation results in table format."""
    results_file = os.path.join(results_dir, 'eval_results.json')
    
    if not os.path.exists(results_file):
        print(f"[ load ] ERROR: No results file found: {results_file}")
        return False
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print()
    print("=" * 80)
    print("iMeanFlow (iMF-PCC) Evaluation Results")
    print("=" * 80)
    print()
    print("Per-Seed Results:")
    print("-" * 80)
    print(f"{'Seed':>6s} {'MSE Error':>14s} {'Std Dev':>14s} {'Num Samples':>12s}")
    print("-" * 80)
    
    mse_errors = []
    
    # Sort by seed number
    sorted_keys = sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else 999)
    
    for seed_key in sorted_keys:
        result = results[seed_key]
        seed = result.get('seed', seed_key)
        mse = result.get('mse_error', 0.0)
        std = result.get('mse_std', 0.0)
        num_samples = result.get('num_samples', 0)
        
        print(f"{int(seed):6d} {mse:14.6f} {std:14.6f} {num_samples:12d}")
        mse_errors.append(mse)
    
    print("-" * 80)
    
    # Summary statistics
    if mse_errors:
        mean_mse = sum(mse_errors) / len(mse_errors)
        var_mse = sum((x - mean_mse) ** 2 for x in mse_errors) / len(mse_errors)
        std_mse = var_mse ** 0.5
        
        print(f"{'MEAN':>6s} {mean_mse:14.6f} {std_mse:14.6f} {len(mse_errors):12d}")
    
    print("-" * 80)
    print()
    print("=" * 80)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Load iMF evaluation results')
    parser.add_argument('--results-dir', type=str, default='evaluation_results', help='Results directory.')
    args = parser.parse_args()
    
    print(f"[ load ] iMeanFlow Results Loader")
    print(f"[ load ] Loading from: {args.results_dir}")
    
    success = load_and_display_results(args.results_dir)
    
    if success:
        print("[ load ] ✓ Complete")
    else:
        print("[ load ] ✗ Failed to load results")
                # Collect metric values
                error_key = f'{variant}_trajectory_error'
                length_key = f'{variant}_trajectory_length_list'
                smooth_key = f'{variant}_smoothness_list'
                
                if error_key in data:
                    variant_data['trajectory_error'].append(float(data[error_key]))
                
                if length_key in data:
                    lengths = data[length_key]
                    if hasattr(lengths, '__len__'):
                        variant_data['path_length'].append(float(np.mean(lengths)))
                
                if smooth_key in data:
                    smoothness = data[smooth_key]
                    if hasattr(smoothness, '__len__'):
                        variant_data['smoothness'].append(float(np.mean(smoothness)))
                
                variant_data['seeds'].append(seed)
            
            # Compute statistics
            self.summary[variant] = {
                'num_seeds': len(variant_data['seeds']),
                'seeds_evaluated': variant_data['seeds'],
            }
            
            if variant_data['trajectory_error']:
                errors = np.array(variant_data['trajectory_error'])
                self.summary[variant]['trajectory_error_mean'] = float(np.mean(errors))
                self.summary[variant]['trajectory_error_std'] = float(np.std(errors))
                self.summary[variant]['trajectory_error_min'] = float(np.min(errors))
                self.summary[variant]['trajectory_error_max'] = float(np.max(errors))
            
            if variant_data['path_length']:
                lengths = np.array(variant_data['path_length'])
                self.summary[variant]['path_length_mean'] = float(np.mean(lengths))
                self.summary[variant]['path_length_std'] = float(np.std(lengths))
            
            if variant_data['smoothness']:
                smoothness = np.array(variant_data['smoothness'])
                self.summary[variant]['smoothness_mean'] = float(np.mean(smoothness))
                self.summary[variant]['smoothness_std'] = float(np.std(smoothness))
        
        print(f"✓ Aggregated {len(self.summary)} variants across {len(self.all_results)} seeds")
        return self.summary
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("AGGREGATED RESULTS SUMMARY")
        print("=" * 80)
        
        # Create DataFrame for display
        rows = []
        for variant, stats in sorted(self.summary.items()):
            row = {
                'Variant': variant,
                'Seeds': stats.get('num_seeds', '?'),
                'Traj Error (μ)': f"{stats.get('trajectory_error_mean', np.nan):.4f}",
                'Traj Error (σ)': f"{stats.get('trajectory_error_std', np.nan):.4f}",
                'Path Length (μ)': f"{stats.get('path_length_mean', np.nan):.4f}",
                'Smoothness (μ)': f"{stats.get('smoothness_mean', np.nan):.4f}",
            }
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
            print()
        else:
            print("⚠ No results to display")
    
    def save_csv_report(self, output_path: Path = None):
        """Save results as CSV report."""
        if output_path is None:
            output_path = self.results_dir / 'results_summary.csv'
        
        rows = []
        for variant, stats in sorted(self.summary.items()):
            row = {
                'variant': variant,
                'num_seeds': stats.get('num_seeds', 0),
                'trajectory_error_mean': stats.get('trajectory_error_mean', np.nan),
                'trajectory_error_std': stats.get('trajectory_error_std', np.nan),
                'trajectory_error_min': stats.get('trajectory_error_min', np.nan),
                'trajectory_error_max': stats.get('trajectory_error_max', np.nan),
                'path_length_mean': stats.get('path_length_mean', np.nan),
                'path_length_std': stats.get('path_length_std', np.nan),
                'smoothness_mean': stats.get('smoothness_mean', np.nan),
                'smoothness_std': stats.get('smoothness_std', np.nan),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"✓ CSV report saved to {output_path}")
        
        return df
    
    def save_json_report(self, output_path: Path = None):
        """Save results as JSON report."""
        if output_path is None:
            output_path = self.results_dir / 'results_summary.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_seeds': len(self.all_results),
            'summary': self.summary,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ JSON report saved to {output_path}")
    
    def plot_comparison(self, output_dir: Path = None):
        """Generate comparison plots."""
        if not HAS_MATPLOTLIB:
            print("⚠ Matplotlib not available, skipping plots")
            return
        
        if output_dir is None:
            output_dir = self.results_dir / 'plots'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        variants = list(self.summary.keys())
        errors = [
            self.summary[v].get('trajectory_error_mean', np.nan)
            for v in variants
        ]
        error_stds = [
            self.summary[v].get('trajectory_error_std', np.nan)
            for v in variants
        ]
        
        # Plot 1: Trajectory Error Comparison
        if errors:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(variants))
            plt.bar(x, errors, yerr=error_stds, capsize=5, alpha=0.7, color='steelblue')
            plt.xlabel('Variant', fontsize=12)
            plt.ylabel('Trajectory Error', fontsize=12)
            plt.title('iMeanFlow: Trajectory Error by Variant', fontsize=14, fontweight='bold')
            plt.xticks(x, variants, rotation=45, ha='right')
            plt.tight_layout()
            
            plot_path = output_dir / 'trajectory_error_comparison.png'
            plt.savefig(plot_path, dpi=150)
            print(f"✓ Saved plot to {plot_path}")
            plt.close()
        
        # Plot 2: Path Length Comparison
        path_lengths = [
            self.summary[v].get('path_length_mean', np.nan)
            for v in variants
        ]
        
        if any(~np.isnan(path_lengths)):
            plt.figure(figsize=(10, 6))
            x = np.arange(len(variants))
            plt.bar(x, path_lengths, alpha=0.7, color='coral')
            plt.xlabel('Variant', fontsize=12)
            plt.ylabel('Path Length', fontsize=12)
            plt.title('iMeanFlow: Path Length by Variant', fontsize=14, fontweight='bold')
            plt.xticks(x, variants, rotation=45, ha='right')
            plt.tight_layout()
            
            plot_path = output_dir / 'path_length_comparison.png'
            plt.savefig(plot_path, dpi=150)
            print(f"✓ Saved plot to {plot_path}")
            plt.close()
        
        # Plot 3: Smoothness Comparison
        smoothness = [
            self.summary[v].get('smoothness_mean', np.nan)
            for v in variants
        ]
        
        if any(~np.isnan(smoothness)):
            plt.figure(figsize=(10, 6))
            x = np.arange(len(variants))
            plt.bar(x, smoothness, alpha=0.7, color='lightgreen')
            plt.xlabel('Variant', fontsize=12)
            plt.ylabel('Smoothness', fontsize=12)
            plt.title('iMeanFlow: Trajectory Smoothness by Variant', fontsize=14, fontweight='bold')
            plt.xticks(x, variants, rotation=45, ha='right')
            plt.tight_layout()
            
            plot_path = output_dir / 'smoothness_comparison.png'
            plt.savefig(plot_path, dpi=150)
            print(f"✓ Saved plot to {plot_path}")
            plt.close()
    
    def run(self, save_plots: bool = True):
        """Run full analysis pipeline."""
        print("=" * 80)
        print("iMeanFlow Results Analysis")
        print("=" * 80)
        print()
        
        # Load results
        if not self.load_results():
            return False
        
        # Aggregate
        self.aggregate_results()
        
        # Print summary
        self.print_summary()
        
        # Save reports
        self.save_csv_report()
        self.save_json_report()
        
        # Plot comparisons
        if save_plots:
            self.plot_comparison()
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        
        return True


def main():
    """Main results loading script."""
    parser = argparse.ArgumentParser(description='Load and aggregate iMF results')
    parser.add_argument('--results-dir', type=str, default='evaluation_results',
                       help='Results directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    loader = ImfResultsLoader(results_dir=Path(args.results_dir))
    success = loader.run(save_plots=not args.no_plots)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
