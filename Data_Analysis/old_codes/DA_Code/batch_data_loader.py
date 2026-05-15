"""
Batch Data Loader Module (v2)

Loads evaluation data for multiple candidate folders and aggregates them
into a unified data structure with candidate dimension.
"""

import os
import logging
from typing import Dict, List, Optional
from data_loader import DataLoader


logger = logging.getLogger(__name__)


class BatchDataLoader:
    """
    Load evaluation data for multiple candidate folders.
    
    Reuses v1 DataLoader for each candidate, then organizes results
    by candidate letter.
    """
    
    def __init__(self, verbose=False):
        """
        Initialize batch loader.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.batch_data = {}
        self.loading_summary = {}
    
    def load_all_candidates(
        self,
        candidates_dict,
        variants=None,
        constraint_types=None,
        halfspace_variants=None
    ):
        """
        Load data for all candidates.
        
        Args:
            candidates_dict: Dict from discover_candidates() with letter → {path, name, seeds}
            variants: List of projection variants to load (default: all)
            constraint_types: List of constraint types to load (default: all)
            halfspace_variants: List of halfspace variants to load (default: all)
            
        Returns:
            dict: Batch data organized by candidate
            {
                'A': {seed: {variant: {constraint: {halfspace: metrics}}}},
                'B': {...},
                'C': {...}
            }
        """
        # Use defaults from config if not provided
        if variants is None:
            from config import DEFAULT_PROJECTION_VARIANTS
            variants = DEFAULT_PROJECTION_VARIANTS
        
        if constraint_types is None:
            from config import DEFAULT_CONSTRAINT_TYPES
            constraint_types = DEFAULT_CONSTRAINT_TYPES
        
        if halfspace_variants is None:
            from config import DEFAULT_HALFSPACE_VARIANTS
            halfspace_variants = DEFAULT_HALFSPACE_VARIANTS
        
        total_candidates = len(candidates_dict)
        logger.info(f"Loading {total_candidates} candidates...")
        
        self.batch_data = {}
        self.loading_summary = {}
        
        # Load each candidate
        for candidate_idx, (letter, info) in enumerate(sorted(candidates_dict.items())):
            logger.info(f"[{candidate_idx + 1}/{total_candidates}] Loading Candidate {letter}: {info['name']}")
            
            try:
                # Use v1 DataLoader for this candidate
                loader = DataLoader(verbose=self.verbose)
                candidate_data = loader.load_results(
                    root_path=info['path'],
                    seeds=info['seeds'],
                    variants=variants,
                    constraint_types=constraint_types,
                    halfspace_variants=halfspace_variants
                )
                
                self.batch_data[letter] = candidate_data
                self.loading_summary[letter] = loader.get_loading_summary()
                
                logger.info(
                    f"  ✓ Candidate {letter} loaded: "
                    f"{self.loading_summary[letter]['files_loaded']} files"
                )
                
            except Exception as e:
                logger.error(f"  ✗ Failed to load Candidate {letter}: {str(e)}")
                self.loading_summary[letter] = {
                    'files_found': 0,
                    'files_loaded': 0,
                    'files_failed': 0,
                    'error': str(e)
                }
        
        logger.info(f"Batch loading complete: {len(self.batch_data)} candidates loaded")
        return self.batch_data
    
    def get_batch_summary(self):
        """
        Get loading summary for all candidates.
        
        Returns:
            dict: Summary with files loaded per candidate
        """
        return self.loading_summary
    
    def save_batch_loading_log(self, output_path):
        """
        Save detailed loading log for all candidates.
        
        Args:
            output_path: Path to save log file
        """
        lines = ["=== Batch Loading Summary ===\n"]
        
        for letter in sorted(self.loading_summary.keys()):
            summary = self.loading_summary[letter]
            lines.append(f"Candidate {letter}:")
            lines.append(f"  Files Found: {summary.get('files_found', 0)}")
            lines.append(f"  Files Loaded: {summary.get('files_loaded', 0)}")
            lines.append(f"  Files Failed: {summary.get('files_failed', 0)}")
            
            if 'error' in summary:
                lines.append(f"  Error: {summary['error']}")
            
            lines.append("")
        
        # Summary statistics
        total_loaded = sum(s.get('files_loaded', 0) for s in self.loading_summary.values())
        total_failed = sum(s.get('files_failed', 0) for s in self.loading_summary.values())
        
        lines.append("=== Total ===")
        lines.append(f"Total Files Loaded: {total_loaded}")
        lines.append(f"Total Files Failed: {total_failed}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Batch loading log saved: {output_path}")
    
    def get_data_for_candidate(self, letter):
        """
        Get raw data dict for a specific candidate.
        
        Args:
            letter: Candidate letter (e.g., 'A', 'B')
            
        Returns:
            dict: Candidate's data structure
        """
        return self.batch_data.get(letter, {})
    
    def get_all_data(self):
        """
        Get all batch data.
        
        Returns:
            dict: Full batch_data dict
        """
        return self.batch_data
    
    def validate_data(self):
        """
        Validate that all candidates have data.
        
        Returns:
            dict: Validation results {letter: bool, ...}
        """
        results = {}
        for letter, data in self.batch_data.items():
            has_data = len(data) > 0
            results[letter] = has_data
            
            status = "✓" if has_data else "✗"
            logger.info(f"Candidate {letter}: {status}")
        
        return results


if __name__ == "__main__":
    # Test the batch loader
    import sys
    from multi_candidate_discovery import discover_candidates
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        parent = sys.argv[1]
        candidates = discover_candidates(parent)
        
        loader = BatchDataLoader(verbose=True)
        batch_data = loader.load_all_candidates(candidates)
        
        print("\n" + "="*50)
        print("Batch Loading Complete")
        print("="*50)
        for letter in sorted(batch_data.keys()):
            print(f"Candidate {letter}: {len(batch_data[letter])} seeds")
    else:
        print("Usage: python batch_data_loader.py <parent_path>")
