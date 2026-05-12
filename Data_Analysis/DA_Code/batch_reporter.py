"""
Batch Reporter Module (v2)

Generates human-readable reports and machine-readable export files
for cross-candidate comparison results.
"""

import logging
import os
import pandas as pd
from typing import Dict


logger = logging.getLogger(__name__)


class BatchReporter:
    """
    Generate reports from batch aggregation results.
    """
    
    def __init__(self, candidate_stats, ranked_candidates, candidates_info=None):
        """
        Initialize batch reporter.
        
        Args:
            candidate_stats: Dict from BatchAggregator
            ranked_candidates: Ranked list from BatchAggregator
            candidates_info: Original candidate info dict
        """
        self.candidate_stats = candidate_stats
        self.ranked_candidates = ranked_candidates
        self.candidates_info = candidates_info or {}
    
    def save_candidates_summary_txt(self, output_path):
        """
        Save human-readable summary of cross-candidate comparison.
        
        Args:
            output_path: Path to save summary
        """
        lines = [
            "=" * 70,
            "CROSS-CANDIDATE COMPARISON SUMMARY",
            "=" * 70,
            ""
        ]
        
        # Candidates discovered
        lines.append("CANDIDATES DISCOVERED")
        lines.append("-" * 70)
        
        for letter in sorted(self.candidates_info.keys()):
            info = self.candidates_info[letter]
            lines.append(f"  {letter}: {info.get('name', 'Unknown')}")
        
        lines.append("")
        
        # Rankings
        lines.append("RANKINGS BY ACCURACY (Goal + Constraint Success)")
        lines.append("-" * 70)
        
        for rank, (letter, accuracy) in enumerate(self.ranked_candidates, 1):
            stats = self.candidate_stats[letter]
            time_ms = stats.get('time_ms', np.nan)
            accuracy_std = stats.get('accuracy_std', np.nan)
            
            line = f"  {rank}. Candidate {letter}: {accuracy*100:.2f}%"
            if not np.isnan(accuracy_std):
                line += f" (±{accuracy_std*100:.2f}%)"
            if not np.isnan(time_ms):
                line += f", Time: {time_ms:.1f}ms"
            
            lines.append(line)
        
        lines.append("")
        
        # Best candidates
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        
        if self.ranked_candidates:
            best_letter, best_accuracy = self.ranked_candidates[0]
            lines.append(f"  Overall Best: Candidate {best_letter}")
            lines.append(f"  Accuracy: {best_accuracy*100:.2f}%")
            
            # Find fastest
            fastest = min(
                self.candidate_stats.items(),
                key=lambda x: x[1].get('time_ms', float('inf'))
            )
            if fastest[1].get('time_ms', None):
                lines.append(f"  Fastest: Candidate {fastest[0]} ({fastest[1]['time_ms']:.1f}ms)")
        
        lines.append("")
        lines.append("=" * 70)
        
        content = "\n".join(lines)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved summary: {output_path}")
        return content
    
    def save_candidates_ranking_csv(self, output_path):
        """
        Save candidate rankings as CSV for Excel import.
        
        Args:
            output_path: Path to save CSV
        """
        rows = []
        
        for rank, (letter, accuracy) in enumerate(self.ranked_candidates, 1):
            stats = self.candidate_stats[letter]
            info = self.candidates_info.get(letter, {})
            
            row = {
                'Rank': rank,
                'Candidate': letter,
                'Folder': info.get('name', 'Unknown'),
                'Accuracy (%)': accuracy * 100,
                'Accuracy Std (%)': stats.get('accuracy_std', '') * 100 if stats.get('accuracy_std') else '',
                'Time (ms)': stats.get('time_ms', ''),
                'Time Std (ms)': stats.get('time_std', ''),
                'Robustness': stats.get('robustness', '')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved ranking CSV: {output_path}")
    
    def save_candidates_detailed_csv(self, output_path):
        """
        Save detailed comparison table as CSV.
        
        Args:
            output_path: Path to save CSV
        """
        rows = []
        
        for letter in sorted(self.candidate_stats.keys()):
            stats = self.candidate_stats[letter]
            info = self.candidates_info.get(letter, {})
            
            if 'error' not in stats:
                row = {
                    'Candidate_Letter': letter,
                    'Folder_Name': info.get('name', 'Unknown'),
                    'Full_Path': info.get('path', 'Unknown'),
                    'Accuracy': stats.get('accuracy', ''),
                    'Accuracy_Std': stats.get('accuracy_std', ''),
                    'Time_ms': stats.get('time_ms', ''),
                    'Time_Std': stats.get('time_std', ''),
                    'Robustness_Score': stats.get('robustness', '')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved detailed CSV: {output_path}")
    
    def save_all_reports(self, output_dir):
        """
        Generate all reporting outputs.
        
        Args:
            output_dir: Directory to save all reports
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Text summary
        self.save_candidates_summary_txt(
            os.path.join(output_dir, 'candidates_summary.txt')
        )
        
        # CSV files
        self.save_candidates_ranking_csv(
            os.path.join(output_dir, 'candidates_ranking.csv')
        )
        
        self.save_candidates_detailed_csv(
            os.path.join(output_dir, 'candidates_detailed.csv')
        )
        
        logger.info(f"All reports saved to: {output_dir}")


if __name__ == "__main__":
    import numpy as np
    
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    candidate_stats = {
        'A': {'accuracy': 0.873, 'accuracy_std': 0.021, 'time_ms': 42.1, 'time_std': 2.3, 'robustness': 0.015},
        'B': {'accuracy': 0.842, 'accuracy_std': 0.028, 'time_ms': 38.2, 'time_std': 1.9, 'robustness': 0.020},
        'C': {'accuracy': 0.895, 'accuracy_std': 0.019, 'time_ms': 45.2, 'time_std': 3.1, 'robustness': 0.018},
    }
    
    ranked = [('C', 0.895), ('A', 0.873), ('B', 0.842)]
    
    candidates_info = {
        'A': {'name': 'diffusion_H8_K20_aw10', 'path': '/path/to/A'},
        'B': {'name': 'diffusion_H8_K10_aw10', 'path': '/path/to/B'},
        'C': {'name': 'diffusion_H8_K20_T1', 'path': '/path/to/C'},
    }
    
    reporter = BatchReporter(candidate_stats, ranked, candidates_info)
    reporter.save_all_reports('./test_reports')
    
    print("\nTest reports generated in ./test_reports")
