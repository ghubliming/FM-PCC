"""
Reporting module: Generates summary reports and CSV outputs.
"""
import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Reporter:
    """Generate summary reports and CSV exports."""
    
    def __init__(self, aggregator):
        """
        Initialize reporter.
        
        Args:
            aggregator: DataAggregator instance
        """
        self.aggregator = aggregator
    
    def save_all_reports(self, output_dir, data_loader_summary=None):
        """
        Generate and save all reports.
        
        Args:
            output_dir: Directory to save reports
            data_loader_summary: Optional dict with loading statistics
        """
        logger.info('Generating all reports...')
        
        self.save_summary_txt(os.path.join(output_dir, 'results_summary.txt'),
                            data_loader_summary)
        self.save_thesis_ranking(os.path.join(output_dir, 'thesis_ranking.txt'))
        self.save_summary_csv(os.path.join(output_dir, 'results_by_variant.csv'))
        self.save_constraint_csv(os.path.join(output_dir, 'results_by_constraint.csv'))
        self.save_halfspace_csv(os.path.join(output_dir, 'results_by_halfspace.csv'))
        self.save_detailed_csv(os.path.join(output_dir, 'detailed_results.csv'))
        
        logger.info('Reports saved successfully')
    
    def save_summary_txt(self, output_path, data_loader_summary=None):
        """
        Save human-readable summary report.
        
        Args:
            output_path: Path to output file
            data_loader_summary: Optional dict with loading statistics
        """
        logger.info(f'Saving summary report to {output_path}...')
        
        with open(output_path, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write('FM v3 ODE-Selectable Evaluation Analysis Summary\n')
            f.write('=' * 80 + '\n\n')
            
            f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # Data loading summary
            if data_loader_summary:
                f.write('--- Data Loading Summary ---\n')
                f.write(f'Files Found: {data_loader_summary["files_found"]}\n')
                f.write(f'Files Loaded: {data_loader_summary["files_loaded"]}\n')
                f.write(f'Files Failed: {data_loader_summary["files_failed"]}\n')
                f.write(f'Success Rate: {100 * data_loader_summary["success_rate"]:.1f}%\n\n')
            
            # Variant ranking - top performers
            f.write('--- Top 10 Variants by Goal + Constraint Success ---\n')
            ranking = self.aggregator.get_variant_ranking('n_success_and_constraints', ascending=False)
            if len(ranking) > 0:
                for idx, row in ranking.head(10).iterrows():
                    f.write(f'{idx+1:2d}. {row["variant"]:30s} | Mean: {row["mean"]:6.3f} '
                           f'(±{row["std"]:5.3f}) | N={int(row["count"])}\n')
            else:
                f.write('No data available\n')
            
            f.write('\n--- Top 10 Variants by Success Rate (Goal Only) ---\n')
            ranking_goal = self.aggregator.get_variant_ranking('n_success', ascending=False)
            if len(ranking_goal) > 0:
                for idx, row in ranking_goal.head(10).iterrows():
                    f.write(f'{idx+1:2d}. {row["variant"]:30s} | Mean: {row["mean"]:6.3f} '
                           f'(±{row["std"]:5.3f}) | N={int(row["count"])}\n')
            else:
                f.write('No data available\n')
            
            # Performance by constraint type
            f.write('\n--- Average Performance by Constraint Type ---\n')
            by_constraint = self.aggregator.aggregated.get('by_constraint', pd.DataFrame())
            if len(by_constraint) > 0:
                for constraint in sorted(by_constraint['constraint_type'].unique()):
                    constraint_data = by_constraint[by_constraint['constraint_type'] == constraint]
                    # Find n_success_and_constraints metric
                    metric_data = constraint_data[constraint_data['metric'] == 'n_success_and_constraints']
                    if len(metric_data) > 0:
                        mean = metric_data['mean'].values[0]
                        std = metric_data['std'].values[0]
                        f.write(f'  {constraint:15s}: Mean={mean:6.3f} ±{std:5.3f}\n')
            
            # Performance by halfspace variant
            f.write('\n--- Average Performance by Halfspace Variant ---\n')
            by_halfspace = self.aggregator.aggregated.get('by_halfspace', pd.DataFrame())
            if len(by_halfspace) > 0:
                for halfspace in sorted(by_halfspace['halfspace_variant'].unique()):
                    halfspace_data = by_halfspace[by_halfspace['halfspace_variant'] == halfspace]
                    metric_data = halfspace_data[halfspace_data['metric'] == 'n_success_and_constraints']
                    if len(metric_data) > 0:
                        mean = metric_data['mean'].values[0]
                        std = metric_data['std'].values[0]
                        f.write(f'  {halfspace:20s}: Mean={mean:6.3f} ±{std:5.3f}\n')
            
            # Overall statistics
            f.write('\n--- Overall Statistics (All Variants, All Constraints) ---\n')
            by_variant = self.aggregator.aggregated.get('by_variant', pd.DataFrame())
            if len(by_variant) > 0:
                for metric in ['n_success', 'n_success_and_constraints', 'avg_time', 'n_violations']:
                    metric_data = by_variant[by_variant['metric'] == metric]
                    if len(metric_data) > 0:
                        overall_mean = metric_data['mean'].mean()
                        overall_std = metric_data['std'].mean()
                        f.write(f'  {metric:30s}: Mean={overall_mean:10.3f} ±{overall_std:8.3f}\n')
            
            f.write('\n' + '=' * 80 + '\n')
        
        logger.info(f'Summary report saved to {output_path}')
    
    def save_thesis_ranking(self, output_path):
        """
        Save thesis-focused ranking: accuracy vs. time with variant categorization.
        
        Args:
            output_path: Path to output file
        """
        logger.info(f'Saving thesis ranking to {output_path}...')
        
        detailed = self.aggregator.aggregated.get('detailed', pd.DataFrame())
        if len(detailed) == 0:
            logger.warning('No detailed data for thesis ranking')
            return
        
        # Extract accuracy and time
        accuracy_data = detailed[detailed['metric'] == 'n_success_and_constraints'].copy()
        time_data = detailed[detailed['metric'] == 'avg_time'].copy()
        
        merged = accuracy_data.merge(
            time_data[['variant', 'value']],
            on='variant',
            suffixes=('_acc', '_time')
        )
        
        # Aggregate by variant
        ranking = merged.groupby('variant').agg({
            'value_acc': ['mean', 'std'],
            'value_time': 'mean'
        }).reset_index()
        ranking.columns = ['variant', 'accuracy', 'accuracy_std', 'time']
        
        # Categorize variants
        def categorize(variant):
            if 'dpcc-c' in variant:
                return 'PRIMARY-dpcc-c'
            elif 'dpcc-r' in variant:
                return 'PRIMARY-dpcc-r'
            elif 'dpcc-t' in variant:
                return 'PRIMARY-dpcc-t'
            elif variant in ['diffuser', 'gradient', 'model_free', 'post_processing']:
                return 'BASELINE-ML'
            else:
                return 'OTHER'
        
        ranking['category'] = ranking['variant'].apply(categorize)
        
        # Sort by accuracy (descending), then time (ascending)
        ranking = ranking.sort_values(
            by=['accuracy', 'time'],
            ascending=[False, True]
        ).reset_index(drop=True)
        
        with open(output_path, 'w') as f:
            f.write('=' * 100 + '\n')
            f.write('THESIS RANKING: Accuracy-Time Pareto Frontier Analysis\n')
            f.write('=' * 100 + '\n\n')
            
            f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            f.write('PRIMARY VARIANTS (dpcc-c, dpcc-r, dpcc-t) vs BASELINE\n')
            f.write('-' * 100 + '\n\n')
            
            f.write(f'{"Rank":>4s} | {"Category":20s} | {"Variant":30s} | {"Accuracy (%)":>15s} | {"Time (ms)":>12s}\n')
            f.write('-' * 100 + '\n')
            
            for idx, row in ranking.iterrows():
                rank = idx + 1
                category = row['category']
                variant = row['variant']
                acc_str = f"{row['accuracy']:.2f} ±{row['accuracy_std']:.2f}"
                time_str = f"{row['time']:.2f}"
                
                f.write(f'{rank:4d} | {category:20s} | {variant:30s} | {acc_str:>15s} | {time_str:>12s}\n')
            
            f.write('\n' + '=' * 100 + '\n')
            f.write('CATEGORY LEGEND\n')
            f.write('=' * 100 + '\n')
            f.write('PRIMARY-dpcc-c : Constrained DPCC (main entry for thesis)\n')
            f.write('PRIMARY-dpcc-r : Relaxed DPCC (main entry for thesis)\n')
            f.write('PRIMARY-dpcc-t : Tight DPCC (main entry for thesis)\n')
            f.write('BASELINE-ML    : Raw machine learning baseline (diffuser, gradient, etc.)\n')
            f.write('OTHER          : Variant methods\n\n')
            
            # Summary statistics
            primary_dpcc_c = ranking[ranking['category'] == 'PRIMARY-dpcc-c']
            primary_dpcc_r = ranking[ranking['category'] == 'PRIMARY-dpcc-r']
            primary_dpcc_t = ranking[ranking['category'] == 'PRIMARY-dpcc-t']
            baseline = ranking[ranking['category'] == 'BASELINE-ML']
            
            f.write('=' * 100 + '\n')
            f.write('SUMMARY STATISTICS\n')
            f.write('=' * 100 + '\n\n')
            
            if len(primary_dpcc_c) > 0:
                best_c = primary_dpcc_c.iloc[0]
                f.write(f'Best dpcc-c:  {best_c["variant"]:30s} | Acc: {best_c["accuracy"]:.2f}% | Time: {best_c["time"]:.2f}ms\n')
            
            if len(primary_dpcc_r) > 0:
                best_r = primary_dpcc_r.iloc[0]
                f.write(f'Best dpcc-r:  {best_r["variant"]:30s} | Acc: {best_r["accuracy"]:.2f}% | Time: {best_r["time"]:.2f}ms\n')
            
            if len(primary_dpcc_t) > 0:
                best_t = primary_dpcc_t.iloc[0]
                f.write(f'Best dpcc-t:  {best_t["variant"]:30s} | Acc: {best_t["accuracy"]:.2f}% | Time: {best_t["time"]:.2f}ms\n')
            
            if len(baseline) > 0:
                best_baseline = baseline.iloc[0]
                f.write(f'Best baseline: {best_baseline["variant"]:30s} | Acc: {best_baseline["accuracy"]:.2f}% | Time: {best_baseline["time"]:.2f}ms\n')
            
            f.write('\n' + '=' * 100 + '\n')
        
        logger.info(f'Thesis ranking saved to {output_path}')
    
    def save_summary_csv(self, output_path):
        """
        Save variant-level summary as CSV.
        
        Args:
            output_path: Path to output file
        """
        logger.info(f'Saving variant summary CSV to {output_path}...')
        
        by_variant = self.aggregator.aggregated.get('by_variant', pd.DataFrame())
        if len(by_variant) > 0:
            by_variant.to_csv(output_path, index=False)
            logger.info(f'Saved {len(by_variant)} rows to {output_path}')
        else:
            logger.warning('No variant data to save')
    
    def save_constraint_csv(self, output_path):
        """
        Save constraint-level summary as CSV.
        
        Args:
            output_path: Path to output file
        """
        logger.info(f'Saving constraint summary CSV to {output_path}...')
        
        by_constraint = self.aggregator.aggregated.get('by_constraint', pd.DataFrame())
        if len(by_constraint) > 0:
            by_constraint.to_csv(output_path, index=False)
            logger.info(f'Saved {len(by_constraint)} rows to {output_path}')
        else:
            logger.warning('No constraint data to save')
    
    def save_halfspace_csv(self, output_path):
        """
        Save halfspace-level summary as CSV.
        
        Args:
            output_path: Path to output file
        """
        logger.info(f'Saving halfspace summary CSV to {output_path}...')
        
        by_halfspace = self.aggregator.aggregated.get('by_halfspace', pd.DataFrame())
        if len(by_halfspace) > 0:
            by_halfspace.to_csv(output_path, index=False)
            logger.info(f'Saved {len(by_halfspace)} rows to {output_path}')
        else:
            logger.warning('No halfspace data to save')
    
    def save_detailed_csv(self, output_path):
        """
        Save full detailed results as CSV.
        
        Args:
            output_path: Path to output file
        """
        logger.info(f'Saving detailed results CSV to {output_path}...')
        
        detailed = self.aggregator.aggregated.get('detailed', pd.DataFrame())
        if len(detailed) > 0:
            detailed.to_csv(output_path, index=False)
            logger.info(f'Saved {len(detailed)} rows to {output_path}')
        else:
            logger.warning('No detailed data to save')
