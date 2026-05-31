"""One-shot regenerator for stale model_config.pkl / model_config.json.

Context: Fix-18 patched the training scripts so VisualUNet builds with the
correct 23-D transition_dim for non-visual aligning runs. But the legacy
utils.Config.save() silently SKIPPED overwriting any existing pkl on disk,
so a previously-saved 9-D model_config.pkl from a pre-Fix-18 crashed run
remained in place. State dicts on disk are correct 23-D (from successful
post-Fix-18 training); only the config artifact is stale, causing eval
to fail with a shape mismatch.

This script rebuilds the config artifact in-place WITHOUT retraining.
After running, eval will succeed against the existing state dict.

This is a one-off cleanup. The underlying utils.Config bug has been
patched (see STALE_CONFIG_PATCH.md), so future training runs will not
recreate this issue.

Usage on the cluster (from repo root):

    # Auto-detect: regenerates configs for both DPCC and FM non-visual
    # checkpoints if found at canonical paths
    python logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/\
fix_18_nonvisual_step1/regen_stale_model_config.py

    # Or explicit path:
    python .../regen_stale_model_config.py --checkpoint-dir \
        logs/aligning-d3il-visual/visual_aligning_dpcc/H8_K1_..._VFalse_steps900_bs64/6

    # Dry run (show what would be written, don't touch disk):
    python .../regen_stale_model_config.py --dry-run
"""

import argparse
import importlib
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
REPO = _THIS.parents[4]  # logs_in_develop/Gen7.../New.../fix_18.../<this> → repo
sys.path.insert(0, str(REPO))


# ── Canonical post-Fix-18 args for non-visual variants ────────────────────────
def build_dpcc_nonvisual_args():
    """Mirrors the args that train_visual_aligning_dpcc.py produces AFTER
    Fix-18 mutation for the non-visual K=1 run."""
    return Namespace(
        config='config.aligning-d3il-visual',
        seed=6,
        dataset='aligning-d3il-visual',
        model='diffuser_visual_aligning.models.visual_unet.VisualUNet',
        diffusion='diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        action_dim=3,
        obs_dim=20,             # ← FIX-18 override: 6 → 20 for non-visual
        if_vision=False,
        horizon=8,
        n_diffusion_steps=1,
        action_weight=10,
        loss_type='l2',
        dim=32,
        dim_mults=(1, 2, 4, 8),
        condition_dropout=0.1,
        returns_condition=False,
        max_path_length=900,
        logbase='logs',
        prefix='visual_aligning_dpcc/',
        exp_name=('visual_aligning_dpcc/H8_K1_Ddiffuser_visual_aligning.models.'
                  'visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VFalse_'
                  'steps900_bs64'),
        batch_size=64,
        learning_rate=0.0002,
        ema_decay=0.995,
        n_steps_per_epoch=1000,
        n_train_steps=100000.0,
        gradient_accumulate_every=2,
        train_test_split=0.9,
        device='cuda',
    )


def build_fm_nonvisual_args():
    """Mirrors the args that train_fm_visual_aligning.py produces AFTER
    Fix-18 mutation for the non-visual K=1 / FM run."""
    return Namespace(
        config='config.aligning-d3il-visual',
        seed=6,
        dataset='aligning-d3il-visual',
        model='fm_visual_aligning.models.visual_unet.VisualUNet',
        diffusion='fm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching',
        action_dim=3,
        obs_dim=20,             # ← FIX-18 override
        if_vision=False,
        horizon=8,
        time_beta_alpha_v3=1.5,
        time_beta_beta_v3=1.0,
        action_weight=1,
        loss_type='l2',
        dim=32,
        dim_mults=(1, 2, 4, 8),
        condition_dropout=0.1,
        returns_condition=False,
        max_path_length=900,
        logbase='logs',
        prefix='fm_visual_aligning/',
        exp_name=('fm_visual_aligning/H8_Dfm_visual_aligning.models.'
                  'visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VFalse_'
                  'steps900_bs64'),
        batch_size=64,
        learning_rate=0.0002,
        ema_decay=0.995,
        n_steps_per_epoch=1000,
        n_train_steps=100000.0,
        gradient_accumulate_every=2,
        train_test_split=0.9,
        device='cuda',
    )


CANONICAL_TARGETS = [
    # (label, checkpoint_dir_relative_to_repo, args_builder, model_path, utils_module)
    (
        'DPCC non-visual K=1',
        'logs/aligning-d3il-visual/visual_aligning_dpcc/'
        'H8_K1_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.'
        'VisualGaussianDiffusion_aw10_VFalse_steps900_bs64/6',
        build_dpcc_nonvisual_args,
        'diffuser_visual_aligning.models.visual_unet.VisualUNet',
        'diffuser_visual_aligning.utils.config',
    ),
    (
        'FM non-visual',
        'logs/aligning-d3il-visual/fm_visual_aligning/'
        'H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.'
        'VisualFlowMatching_a1.5_b1.0_aw1_VFalse_steps900_bs64/6',
        build_fm_nonvisual_args,
        'fm_visual_aligning.models.visual_unet.VisualUNet',
        'fm_visual_aligning.utils.config',
    ),
]


# ── Core regen ────────────────────────────────────────────────────────────────
def regen_one(checkpoint_dir: Path, args_builder, model_path: str,
              utils_module_name: str, dry_run: bool, label: str = '') -> bool:
    """Rebuild model_config.pkl in `checkpoint_dir`. Returns True if it was
    (or would be) written."""
    pkl = checkpoint_dir / 'model_config.pkl'
    js  = checkpoint_dir / 'model_config.json'

    if not checkpoint_dir.is_dir():
        print(f'[ regen ] SKIP {label}: dir does not exist: {checkpoint_dir}')
        return False

    # Inspect existing pkl (if any) for the obs_dim — tells us whether it's
    # stale or already fixed.
    existing_obs_dim = None
    if pkl.exists():
        try:
            with open(pkl, 'rb') as f:
                existing_cfg = pickle.load(f)
            existing_args = existing_cfg._dict.get('config')
            existing_obs_dim = getattr(existing_args, 'obs_dim', None)
        except Exception as exc:
            print(f'[ regen ] WARN: could not introspect existing pkl '
                  f'({exc}); will overwrite anyway')

    target_args = args_builder()
    target_obs_dim = target_args.obs_dim

    print(f'[ regen ] === {label} ===')
    print(f'[ regen ]   checkpoint dir : {checkpoint_dir}')
    print(f'[ regen ]   existing obs_dim in pkl : {existing_obs_dim}')
    print(f'[ regen ]   target   obs_dim (post-Fix-18) : {target_obs_dim}')

    if existing_obs_dim == target_obs_dim:
        print(f'[ regen ]   already correct — nothing to do')
        return False

    if dry_run:
        print(f'[ regen ]   DRY-RUN: would write fresh pkl + json '
              f'(obs_dim {existing_obs_dim} → {target_obs_dim})')
        return True

    # Import the right utils.Config so the pickle is binary-compatible with
    # what the eval script expects to unpickle.
    utils_mod = importlib.import_module(utils_module_name)
    Config = utils_mod.Config

    # Construct + save. utils.Config.save() is now overwrite-always after the
    # bundled patch (so this would work even if pkl already exists), but we
    # still defensively remove the old files first to avoid any json
    # resume-numbering side-effects.
    if pkl.exists(): pkl.unlink()
    if js.exists():  js.unlink()

    cfg = Config(
        model_path,
        savepath=str(pkl),
        config=target_args,
    )
    print(f'[ regen ]   wrote {pkl}')
    print(f'[ regen ]   wrote {js}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint-dir', type=str, default=None,
                   help='Explicit path to a checkpoint dir. If omitted, '
                        'auto-detect both DPCC and FM canonical targets.')
    p.add_argument('--variant', choices=['dpcc', 'fm'], default=None,
                   help='Only used with --checkpoint-dir to disambiguate '
                        'which args builder to use. Auto-detected from path '
                        'substrings if not given.')
    p.add_argument('--dry-run', action='store_true',
                   help='Show what would be written without touching disk.')
    args = p.parse_args()

    print(f'[ regen ] repo root: {REPO}')
    print(f'[ regen ] dry-run  : {args.dry_run}')

    if args.checkpoint_dir is None:
        # Auto-detect: try both canonical paths
        any_done = False
        for label, rel, builder, model_path, utils_mod in CANONICAL_TARGETS:
            done = regen_one(REPO / rel, builder, model_path, utils_mod,
                             args.dry_run, label)
            any_done = any_done or done
        if not any_done:
            print('[ regen ] No stale configs found at canonical paths. '
                  'Pass --checkpoint-dir to target a custom path.')
        return

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt

    if args.variant is None:
        path_str = str(ckpt).lower()
        if 'fm_visual_aligning' in path_str:
            args.variant = 'fm'
        elif 'visual_aligning_dpcc' in path_str:
            args.variant = 'dpcc'
        else:
            print('[ regen ] ERROR: could not auto-detect variant from path. '
                  'Pass --variant dpcc or --variant fm explicitly.')
            sys.exit(2)

    matches = [t for t in CANONICAL_TARGETS
               if (args.variant == 'dpcc' and 'visual_aligning_dpcc' in t[1])
               or (args.variant == 'fm'   and 'fm_visual_aligning'   in t[1])]
    _, _, builder, model_path, utils_mod = matches[0]
    regen_one(ckpt, builder, model_path, utils_mod, args.dry_run,
              f'{args.variant.upper()} (explicit path)')


if __name__ == '__main__':
    main()
