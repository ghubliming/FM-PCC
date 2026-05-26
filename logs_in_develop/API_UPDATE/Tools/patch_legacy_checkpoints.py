#!/usr/bin/env python3
"""
patch_legacy_checkpoints.py — Patch one FM-PCC checkpoint folder after the API rename.

Give the path to an experiment folder. The script checks the folder's basename for
old class tokens. If none are found the script exits immediately without touching
anything. If legacy tokens are found it patches all pkl/json config files inside,
then renames the folder to the corrected name.

Usage:
    python patch_legacy_checkpoints.py --path <folder_path> [--dry-run] [--backup]
"""

import argparse
import io
import os
import pickle
import shutil
import sys

# ── Token remap ────────────────────────────────────────────────────────────
# Applied as plain string replacement on the folder basename.
# Longer / more-specific patterns listed first to avoid partial matches.
TOKEN_REMAP = [
    ('fm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
     'fm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching'),
    ('fm_visual_aligning.models.diffusion.GaussianDiffusion',
     'fm_visual_aligning.models.diffusion.FlowMatchingODE'),
    ('flow_matcher_v3_drifting.models.diffusion.FlowMatchingODE',
     'flow_matcher_v3_drifting.models.diffusion.FlowMatchingDrifting'),
    ('flow_matcher_v3_drifting.models.diffusion.GaussianDiffusion',
     'flow_matcher_v3_drifting.models.diffusion.FlowMatchingDrifting'),
    ('flow_matcher_v3_imeanflow.models.diffusion.FlowMatchingODE',
     'flow_matcher_v3_imeanflow.models.diffusion.FlowMatchingIMF'),
    ('flow_matcher_v3_imeanflow.models.diffusion.GaussianDiffusion',
     'flow_matcher_v3_imeanflow.models.diffusion.FlowMatchingIMF'),
    ('flow_matcher_v3_imeanflow.models.imf_diffusion.iMFDiffusion',
     'flow_matcher_v3_imeanflow.models.imf_diffusion.iMeanFlowODE'),
    ('flow_matcher_v3_ode_selectable.models.diffusion.GaussianDiffusion',
     'flow_matcher_v3_ode_selectable.models.diffusion.FlowMatchingODE'),
    # Short-form fallbacks
    ('models.visual_gaussian_diffusion.VisualGaussianDiffusion',
     'models.visual_gaussian_diffusion.VisualFlowMatching'),
    ('models.imf_diffusion.iMFDiffusion', 'models.imf_diffusion.iMeanFlowODE'),
    ('VisualGaussianDiffusion', 'VisualFlowMatching'),
    ('iMFDiffusion',            'iMeanFlowODE'),
]

OLD_TOKENS = [old for old, _ in TOKEN_REMAP]

# ── Class remap for RemapUnpickler ─────────────────────────────────────────
CLASS_REMAP = {
    ('fm_visual_aligning.models.visual_gaussian_diffusion', 'VisualGaussianDiffusion'):
        ('fm_visual_aligning.models.visual_gaussian_diffusion', 'VisualFlowMatching'),
    ('fm_visual_aligning.models.diffusion', 'GaussianDiffusion'):
        ('fm_visual_aligning.models.diffusion', 'FlowMatchingODE'),
    ('flow_matcher_v3_ode_selectable.models.diffusion', 'GaussianDiffusion'):
        ('flow_matcher_v3_ode_selectable.models.diffusion', 'FlowMatchingODE'),
    ('flow_matcher_v3_drifting.models.diffusion', 'GaussianDiffusion'):
        ('flow_matcher_v3_drifting.models.diffusion', 'FlowMatchingDrifting'),
    ('flow_matcher_v3_drifting.models.diffusion', 'FlowMatchingODE'):
        ('flow_matcher_v3_drifting.models.diffusion', 'FlowMatchingDrifting'),
    ('flow_matcher_v3_imeanflow.models.diffusion', 'GaussianDiffusion'):
        ('flow_matcher_v3_imeanflow.models.diffusion', 'FlowMatchingIMF'),
    ('flow_matcher_v3_imeanflow.models.diffusion', 'FlowMatchingODE'):
        ('flow_matcher_v3_imeanflow.models.diffusion', 'FlowMatchingIMF'),
    ('flow_matcher_v3_imeanflow.models.imf_diffusion', 'iMFDiffusion'):
        ('flow_matcher_v3_imeanflow.models.imf_diffusion', 'iMeanFlowODE'),
}


class RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        mapped = CLASS_REMAP.get((module, name))
        if mapped:
            module, name = mapped
        return super().find_class(module, name)


def is_legacy(basename):
    """Return True if the folder basename contains any old class token."""
    return any(tok in basename for tok in OLD_TOKENS)


def apply_token_remap(s):
    for old, new in TOKEN_REMAP:
        s = s.replace(old, new)
    return s


def patch_pkl(path, dry_run=False, backup=False):
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        obj     = RemapUnpickler(io.BytesIO(raw)).load()
        new_raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f'  [pkl] ERROR: {os.path.basename(path)}: {e}')
        return False
    if new_raw == raw:
        print(f'  [pkl] unchanged: {os.path.basename(path)}')
        return False
    print(f'  [pkl] {"(dry) " if dry_run else ""}patching: {os.path.basename(path)}')
    if not dry_run:
        if backup:
            shutil.copy2(path, path + '.bak')
        with open(path, 'wb') as f:
            f.write(new_raw)
    return True


def patch_json(path, dry_run=False):
    with open(path, 'r') as f:
        text = f.read()
    new_text = apply_token_remap(text)
    if new_text == text:
        print(f'  [json] unchanged: {os.path.basename(path)}')
        return False
    print(f'  [json] {"(dry) " if dry_run else ""}patching: {os.path.basename(path)}')
    if not dry_run:
        with open(path, 'w') as f:
            f.write(new_text)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backup',  action='store_true')
    args = parser.parse_args()

    folder   = os.path.abspath(args.path)
    basename = os.path.basename(folder)
    parent   = os.path.dirname(folder)

    print(f'[ patch ] Path     : {folder}')

    # ── Guard: only act on legacy folders ─────────────────────────────────
    if not os.path.isdir(folder):
        print(f'[ patch ] ERROR: path does not exist or is not a directory.')
        sys.exit(1)

    if not is_legacy(basename):
        print(f'[ patch ] Folder name contains no legacy tokens — nothing to do.')
        sys.exit(0)

    new_basename = apply_token_remap(basename)
    new_folder   = os.path.join(parent, new_basename)
    print(f'[ patch ] New name : {new_basename}')
    print(f'[ patch ] Dry-run  : {args.dry_run}')
    print()

    # ── Patch pkl / json inside ────────────────────────────────────────────
    n_patched = 0
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.endswith('.pkl') and fname != 'losses.pkl':
                if patch_pkl(fpath, dry_run=args.dry_run, backup=args.backup):
                    n_patched += 1
            elif fname == 'args.json':
                if patch_json(fpath, dry_run=args.dry_run):
                    n_patched += 1

    print()
    print(f'[ patch ] {n_patched} file(s) patched.')

    # ── Rename folder ──────────────────────────────────────────────────────
    if os.path.exists(new_folder):
        print(f'[ patch ] WARNING: rename target already exists — skipping rename.')
        print(f'          {new_folder}')
    else:
        print(f'[ patch ] {"(dry) " if args.dry_run else ""}rename:')
        print(f'          {basename}')
        print(f'          → {new_basename}')
        if not args.dry_run:
            os.rename(folder, new_folder)


if __name__ == '__main__':
    main()
