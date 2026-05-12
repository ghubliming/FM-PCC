"""
Data loader module: Discovers and loads .npz evaluation result files.
"""
import os
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataLoader:
    """Load evaluation data from .npz files organized by seed/variant/constraint."""
    
    def __init__(self):
        self.data = {}  # {seed: {variant: {constraint: {halfspace: metrics_dict}}}}
        self.files_found = 0
        self.files_loaded = 0
        self.files_failed = 0
        self.loading_log = []
    
    def load_results(self, root_path, seeds, variants, constraint_types, halfspace_variants):
        """
        Load all .npz result files from directory structure.
        
        Expected structure:
        root_path/
          {seed}/
            results/
              halfspace_{halfspace_variant}/
                {variant}.npz
        
        Args:
            root_path: Root directory containing results
            seeds: List of seed numbers
            variants: List of projection variant names
            constraint_types: List of constraint types
            halfspace_variants: List of halfspace variant names
        
        Returns:
            Dict of loaded data: {seed: {variant: {constraint: {halfspace: data}}}}
        """
        if not os.path.exists(root_path):
            logger.error(f'Root path does not exist: {root_path}')
            return {}
        
        logger.info(f'Starting data loading from: {root_path}')
        logger.info(f'Seeds: {seeds}')
        logger.info(f'Variants: {len(variants)}')
        logger.info(f'Constraint types: {constraint_types}')
        
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        
        for seed in seeds:
            seed_path = os.path.join(root_path, str(seed), 'results')
            
            if not os.path.exists(seed_path):
                msg = f'Seed {seed}: results directory not found at {seed_path}'
                logger.warning(msg)
                self.loading_log.append(('WARNING', msg))
                continue
            
            for halfspace_variant in halfspace_variants:
                halfspace_path = os.path.join(seed_path, f'halfspace_{halfspace_variant}')
                
                if not os.path.exists(halfspace_path):
                    msg = f'Seed {seed}/{halfspace_variant}: directory not found'
                    logger.debug(msg)
                    self.loading_log.append(('DEBUG', msg))
                    continue
                
                for variant in variants:
                    npz_file = os.path.join(halfspace_path, f'{variant}.npz')
                    self.files_found += 1
                    
                    if not os.path.exists(npz_file):
                        msg = f'Seed {seed}/{halfspace_variant}/{variant}.npz: NOT FOUND'
                        logger.debug(msg)
                        self.loading_log.append(('MISSING', msg))
                        self.files_failed += 1
                        continue
                    
                    try:
                        data_dict = self._load_npz_file(npz_file)
                        
                        # Store under 'halfspace' and other constraint types
                        for constraint in constraint_types:
                            self.data[seed][variant][constraint][halfspace_variant] = data_dict.copy()
                        
                        self.files_loaded += 1
                        logger.debug(f'Loaded: seed={seed}, variant={variant}, halfspace={halfspace_variant}')
                        
                    except Exception as e:
                        msg = f'Seed {seed}/{halfspace_variant}/{variant}.npz: FAILED - {str(e)}'
                        logger.error(msg)
                        self.loading_log.append(('ERROR', msg))
                        self.files_failed += 1
        
        logger.info(f'Loading complete. Loaded: {self.files_loaded}, Failed: {self.files_failed}, Total: {self.files_found}')
        return dict(self.data)
    
    def _load_npz_file(self, npz_file):
        """
        Load single .npz file and extract metrics.
        
        Args:
            npz_file: Path to .npz file
        
        Returns:
            Dict with extracted metrics
        """
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            metrics_dict = {}
            
            # Extract all available metrics
            for key in data.files:
                try:
                    value = data[key]
                    # Handle numpy arrays and scalars
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            metrics_dict[key] = float(value.item())
                        else:
                            # For arrays, store mean/std
                            metrics_dict[f'{key}_array'] = value
                            metrics_dict[f'{key}_mean'] = float(np.mean(value))
                            metrics_dict[f'{key}_std'] = float(np.std(value))
                    else:
                        metrics_dict[key] = float(value)
                except (ValueError, TypeError) as e:
                    logger.debug(f'Could not convert key {key}: {str(e)}')
            
            return metrics_dict
        
        except Exception as e:
            logger.error(f'Failed to load {npz_file}: {str(e)}')
            raise
    
    def get_loading_summary(self):
        """
        Get summary of loading process.
        
        Returns:
            Dict with loading statistics
        """
        return {
            'files_found': self.files_found,
            'files_loaded': self.files_loaded,
            'files_failed': self.files_failed,
            'success_rate': self.files_loaded / max(self.files_found, 1),
            'loading_log': self.loading_log,
        }
    
    def save_loading_log(self, output_path):
        """
        Save detailed loading log to file.
        
        Args:
            output_path: Path to output log file
        """
        with open(output_path, 'w') as f:
            f.write('=== Data Loading Log ===\n\n')
            f.write(f'Files Found: {self.files_found}\n')
            f.write(f'Files Loaded: {self.files_loaded}\n')
            f.write(f'Files Failed: {self.files_failed}\n')
            f.write(f'Success Rate: {100 * self.files_loaded / max(self.files_found, 1):.1f}%\n\n')
            f.write('=== Detailed Log ===\n')
            
            for level, msg in self.loading_log:
                f.write(f'[{level:7s}] {msg}\n')
        
        logger.info(f'Loading log saved to: {output_path}')
