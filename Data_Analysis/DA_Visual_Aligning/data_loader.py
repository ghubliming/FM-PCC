"""
Data loader for Visual Aligning DA.

Single-seed, no halfspace/constraint folders.
Supports two sources:
  --source npz  (default) — loads {seed}/results/{variant}/{variant}.npz
  --source json           — reconstructs per-rollout arrays from
                            {seed}/results/{variant}/diagnostics/rollout_*_stats.json
"""
import os
import glob
import json
import logging
import numpy as np
from collections import defaultdict
from config import METRICS, ACTIVE_SEED

logger = logging.getLogger(__name__)


class DataLoader:
    """Load per-rollout evaluation data for a single seed."""

    def __init__(self, verbose: bool = False, source: str = 'npz'):
        self.verbose    = verbose
        self.source     = source   # 'npz' or 'json'
        self.data       = {}       # {variant: metrics_dict}
        self.files_found  = 0
        self.files_loaded = 0
        self.files_failed = 0
        self.loading_log  = []

    # ------------------------------------------------------------------
    def load_results(self, root_path, seed=None, variants=None):
        """
        Load result files for one seed.

        Args:
            root_path: path to model_exp_name folder (contains seed subfolders)
            seed:      seed number (default: ACTIVE_SEED)
            variants:  list of variant names (default: all discovered)

        Returns:
            {variant: metrics_dict} where array values have shape (N_rollouts,)
        """
        if seed is None:
            seed = ACTIVE_SEED

        if not os.path.exists(root_path):
            logger.error(f'Root path does not exist: {root_path}')
            return {}

        seed_results_path = os.path.join(root_path, str(seed), 'results')
        if not os.path.exists(seed_results_path):
            logger.warning(f'Results dir not found: {seed_results_path}')
            return {}

        # Auto-discover variants if not specified
        if variants is None:
            variants = sorted(
                d for d in os.listdir(seed_results_path)
                if os.path.isdir(os.path.join(seed_results_path, d))
            )

        logger.info(f'Loading seed={seed}, source={self.source}, variants={len(variants)}')

        self.data = {}

        for variant in variants:
            try:
                if self.source == 'npz':
                    metrics = self._load_npz(seed_results_path, variant)
                else:
                    metrics = self._load_json(seed_results_path, variant)

                if metrics is not None:
                    self.data[variant] = metrics
                    self.files_loaded += 1
                else:
                    self.files_failed += 1
            except Exception as e:
                msg = f'variant={variant}: FAILED — {e}'
                logger.error(msg)
                self.loading_log.append(('ERROR', msg))
                self.files_failed += 1

        logger.info(f'Loaded {self.files_loaded}, failed {self.files_failed}')
        return dict(self.data)

    # ------------------------------------------------------------------
    def _load_npz(self, seed_results_path, variant):
        """Load per-rollout arrays from {variant}/{variant}.npz."""
        npz_path = os.path.join(seed_results_path, variant, f'{variant}.npz')
        self.files_found += 1

        if not os.path.exists(npz_path):
            msg = f'{variant}: NPZ not found at {npz_path}'
            logger.warning(msg)
            self.loading_log.append(('MISSING', msg))
            return None

        data = np.load(npz_path, allow_pickle=True)
        metrics = {k: data[k] for k in data.files}
        logger.debug(f'Loaded NPZ: {variant} — keys: {list(metrics.keys())}')
        return metrics

    # ------------------------------------------------------------------
    def _load_json(self, seed_results_path, variant):
        """
        Reconstruct per-rollout arrays from diagnostics/rollout_N_stats.json files.

        JSON field mapping:
          success                    → n_success
          steps                      → n_steps
          avg_inference_time_per_replan → avg_time
          mean_distance              → mean_dist_per_rollout
          max_physical_tracking_error → max_phys_error_per_rollout
          context_info.init_xy_dist  → context_init_xy_dist
          context_info.box_init_xy   → context_box_init_xy
          context_info.target_xy     → context_target_xy
          context_info.box_init_angle_deg   → context_box_angle_deg
          context_info.target_angle_deg     → context_target_angle_deg
        """
        diag_path = os.path.join(seed_results_path, variant, 'diagnostics')
        self.files_found += 1

        if not os.path.exists(diag_path):
            msg = f'{variant}: diagnostics folder not found at {diag_path}'
            logger.warning(msg)
            self.loading_log.append(('MISSING', msg))
            return None

        pattern = os.path.join(diag_path, 'rollout_*_stats.json')
        json_files = sorted(glob.glob(pattern))

        if not json_files:
            msg = f'{variant}: no rollout JSON files in {diag_path}'
            logger.warning(msg)
            self.loading_log.append(('MISSING', msg))
            return None

        # Sort by rollout index
        def _rollout_idx(path):
            base = os.path.basename(path)  # rollout_7_stats.json
            try:
                return int(base.split('_')[1])
            except Exception:
                return 0

        json_files = sorted(json_files, key=_rollout_idx)

        rows = []
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    r = json.load(f)
                rows.append(r)
            except Exception as e:
                logger.warning(f'Failed to load {jf}: {e}')

        if not rows:
            return None

        def _arr(key, default=0.0):
            return np.array([r.get(key, default) for r in rows], dtype=np.float32)

        def _ctx(field, default=0.0):
            return np.array(
                [r.get('context_info', {}).get(field, default) for r in rows],
                dtype=np.float32
            )

        n_success = _arr('success', 0)
        metrics = {
            'n_success':                 n_success,
            'success_rate':              float(np.mean(n_success)),
            'n_steps':                   _arr('steps', 0),
            'avg_time':                  _arr('avg_inference_time_per_replan', 0.0),
            'mean_dist_per_rollout':     _arr('mean_distance', 0.0),
            'max_phys_error_per_rollout': _arr('max_physical_tracking_error', 0.0),
            'context_init_xy_dist':      _ctx('init_xy_dist', 0.0),
            'context_box_angle_deg':     _ctx('box_init_angle_deg', 0.0),
            'context_target_angle_deg':  _ctx('target_angle_deg', 0.0),
            # 2-D arrays
            'context_box_init_xy': np.array(
                [r.get('context_info', {}).get('box_init_xy', [0.0, 0.0]) for r in rows],
                dtype=np.float32
            ),
            'context_target_xy': np.array(
                [r.get('context_info', {}).get('target_xy', [0.0, 0.0]) for r in rows],
                dtype=np.float32
            ),
        }
        logger.debug(f'Loaded JSON: {variant} — {len(rows)} rollouts')
        return metrics

    # ------------------------------------------------------------------
    def get_loading_summary(self):
        return {
            'files_found':  self.files_found,
            'files_loaded': self.files_loaded,
            'files_failed': self.files_failed,
            'success_rate': self.files_loaded / max(self.files_found, 1),
            'loading_log':  self.loading_log,
        }

    def save_loading_log(self, output_path):
        with open(output_path, 'w') as f:
            f.write('=== Data Loading Log ===\n\n')
            f.write(f'Source:        {self.source}\n')
            f.write(f'Files Found:   {self.files_found}\n')
            f.write(f'Files Loaded:  {self.files_loaded}\n')
            f.write(f'Files Failed:  {self.files_failed}\n')
            f.write(f'Success Rate:  {100 * self.files_loaded / max(self.files_found, 1):.1f}%\n\n')
            f.write('=== Detailed Log ===\n')
            for level, msg in self.loading_log:
                f.write(f'[{level:7s}] {msg}\n')
        logger.info(f'Loading log saved to: {output_path}')

    # ------------------------------------------------------------------
    # MULTI-SEED-DEACTIVATED: original multi-seed loop kept below for reference.
    # To re-activate: remove comment markers, restore seeds parameter in load_results().
    #
    # def load_results_multi_seed(self, root_path, seeds, variants, ...):
    #     self.data = defaultdict(dict)   # {seed: {variant: metrics}}
    #     for seed in seeds:
    #         seed_path = os.path.join(root_path, str(seed), 'results')
    #         ...
