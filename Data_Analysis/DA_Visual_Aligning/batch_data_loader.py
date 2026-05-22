"""
Batch Data Loader for Visual Aligning DA.

Loads one candidate at a time using the single-seed DataLoader,
then organises by candidate letter.
"""
import os
import logging
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class BatchDataLoader:
    """Load evaluation data for multiple candidate folders (single seed each)."""

    def __init__(self, verbose=False, source='npz'):
        self.verbose  = verbose
        self.source   = source      # passed to each DataLoader
        self.batch_data      = {}   # {letter: {variant: metrics_dict}}
        self.loading_summary = {}

    def load_all_candidates(self, candidates_dict, variants=None, seed=None):
        """
        Load data for all candidates.

        Args:
            candidates_dict: from discover_candidates() — letter → {path, name, seeds}
            variants:        list of variants (default: auto-discover per candidate)
            seed:            seed number (default: ACTIVE_SEED)

        Returns:
            {letter: {variant: metrics_dict}}
        """
        from config import ACTIVE_SEED
        if seed is None:
            seed = ACTIVE_SEED
        # Pass variants=None through to DataLoader so each candidate auto-discovers
        # its own variant folders via os.listdir.  Only override when caller explicitly
        # passes a list.  Using DEFAULT_PROJECTION_VARIANTS here caused 12 "MISSING"
        # failures for runs that only have 'diffuser', making JSON mode appear broken.

        total = len(candidates_dict)
        logger.info(f'Loading {total} candidates, seed={seed}, source={self.source}')

        self.batch_data      = {}
        self.loading_summary = {}

        for idx, (letter, info) in enumerate(sorted(candidates_dict.items())):
            logger.info(f'[{idx+1}/{total}] Candidate {letter}: {info["name"]}')
            try:
                loader = DataLoader(verbose=self.verbose, source=self.source)
                candidate_data = loader.load_results(
                    root_path=info['path'],
                    seed=seed,
                    variants=variants,
                )
                self.batch_data[letter]      = candidate_data
                self.loading_summary[letter] = loader.get_loading_summary()
                n = self.loading_summary[letter]['files_loaded']
                logger.info(f'  OK: {n} variants loaded')
            except Exception as e:
                logger.error(f'  FAIL Candidate {letter}: {e}')
                self.loading_summary[letter] = {
                    'files_found': 0, 'files_loaded': 0, 'files_failed': 0, 'error': str(e)
                }

        logger.info(f'Batch loading complete: {len(self.batch_data)} candidates')
        return self.batch_data

    def get_batch_summary(self):
        return self.loading_summary

    def save_batch_loading_log(self, output_path):
        lines = ['=== Batch Loading Summary ===\n']
        for letter in sorted(self.loading_summary.keys()):
            s = self.loading_summary[letter]
            lines += [
                f'Candidate {letter}:',
                f'  Files Found:  {s.get("files_found", 0)}',
                f'  Files Loaded: {s.get("files_loaded", 0)}',
                f'  Files Failed: {s.get("files_failed", 0)}',
            ]
            if 'error' in s:
                lines.append(f'  Error: {s["error"]}')
            lines.append('')
        total_loaded = sum(s.get('files_loaded', 0) for s in self.loading_summary.values())
        total_failed = sum(s.get('files_failed', 0) for s in self.loading_summary.values())
        lines += ['=== Total ===', f'Total Loaded: {total_loaded}', f'Total Failed: {total_failed}']
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f'Batch loading log saved: {output_path}')

    def get_data_for_candidate(self, letter):
        return self.batch_data.get(letter, {})

    def get_all_data(self):
        return self.batch_data

    # MULTI-SEED-DEACTIVATED: multi-seed ranking/averaging removed.
    # When re-activating, average metrics across seeds before returning.
