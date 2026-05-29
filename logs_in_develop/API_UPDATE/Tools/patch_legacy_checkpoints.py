#!/usr/bin/env python3
"""
patch_legacy_checkpoints.py — Patch one FM-PCC checkpoint folder after the API rename.

Scan-then-fix flow:
  1. SCAN  — check folder name, every .pkl, every args.json for legacy tokens.
  2. REPORT — print exactly what is legacy and what is already clean.
  3. FIX   — patch only the items that are actually legacy; leave clean items untouched.

Safe to point at an already-correct path — if nothing is legacy, nothing is written.

Usage:
    python patch_legacy_checkpoints.py --path <folder> [--dry-run] [--backup]
"""

import argparse
import io
import os
import pickle
import shutil
import sys

# ── Token remap (folder name / json string replacement) ───────────────────
# Longer / more-specific patterns first to avoid partial matches.
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remapped_count = 0

    def find_class(self, module, name):
        mapped = CLASS_REMAP.get((module, name))
        if mapped:
            module, name = mapped
            self.remapped_count += 1
        return super().find_class(module, name)


def apply_token_remap(s):
    for old, new in TOKEN_REMAP:
        s = s.replace(old, new)
    return s


def has_legacy_token(s):
    return any(tok in s for tok in OLD_TOKENS)


# ── Scan helpers ───────────────────────────────────────────────────────────

def scan_pkl(path):
    """Return (is_legacy, raw_bytes, remapped_bytes_or_None, error_str_or_None)."""
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        unpickler = RemapUnpickler(io.BytesIO(raw))
        obj = unpickler.load()
        new_raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        return False, raw, None, str(e)
    legacy = (unpickler.remapped_count > 0)
    return legacy, raw, new_raw if legacy else None, None


def scan_json(path):
    """Return (is_legacy, original_text, new_text_or_None)."""
    with open(path, 'r') as f:
        text = f.read()
    new_text = apply_token_remap(text)
    legacy = (new_text != text)
    return legacy, text, new_text if legacy else None


# ── Fix helpers ────────────────────────────────────────────────────────────

def fix_pkl(path, new_raw, dry_run=False, backup=False):
    if not dry_run:
        if backup:
            shutil.copy2(path, path + '.bak')
        with open(path, 'wb') as f:
            f.write(new_raw)


def fix_json(path, new_text, dry_run=False):
    if not dry_run:
        with open(path, 'w') as f:
            f.write(new_text)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',    required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backup',  action='store_true')
    args = parser.parse_args()

    folder   = os.path.abspath(args.path)
    basename = os.path.basename(folder)
    parent   = os.path.dirname(folder)

    if not os.path.isdir(folder):
        print(f'[ patch ] ERROR: path does not exist or is not a directory.')
        sys.exit(1)

    print(f'[ patch ] Path    : {folder}')
    print(f'[ patch ] Dry-run : {args.dry_run}')
    print()

    # ── 1. SCAN ────────────────────────────────────────────────────────────
    legacy_folder = has_legacy_token(basename)
    legacy_pkls   = {}   # path → new_raw
    legacy_jsons  = {}   # path → new_text
    errors        = {}   # path → error string

    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            rel   = os.path.relpath(fpath, folder)

            if fname.endswith('.pkl') and fname != 'losses.pkl':
                is_leg, _, new_raw, err = scan_pkl(fpath)
                if err:
                    errors[rel] = err
                elif is_leg:
                    legacy_pkls[fpath] = new_raw

            elif fname == 'args.json':
                is_leg, _, new_text = scan_json(fpath)
                if is_leg:
                    legacy_jsons[fpath] = new_text

    # ── 2. REPORT ──────────────────────────────────────────────────────────
    any_legacy = legacy_folder or legacy_pkls or legacy_jsons

    print('[ scan ] Folder name  :', 'LEGACY' if legacy_folder else 'clean')
    for fpath in sorted(legacy_pkls):
        print(f'[ scan ] pkl   LEGACY  : {os.path.relpath(fpath, folder)}')
    for fpath in sorted(legacy_jsons):
        print(f'[ scan ] json  LEGACY  : {os.path.relpath(fpath, folder)}')
    for rel, err in sorted(errors.items()):
        print(f'[ scan ] pkl   ERROR   : {rel}  ({err})')

    if not any_legacy and not errors:
        print()
        print('[ patch ] Everything is already up to date — nothing to do.')
        sys.exit(0)

    print()

    # ── 3. FIX ─────────────────────────────────────────────────────────────
    n_fixed = 0

    for fpath, new_raw in sorted(legacy_pkls.items()):
        rel = os.path.relpath(fpath, folder)
        print(f'  [pkl]  {"(dry) " if args.dry_run else ""}fixing: {rel}')
        fix_pkl(fpath, new_raw, dry_run=args.dry_run, backup=args.backup)
        n_fixed += 1

    for fpath, new_text in sorted(legacy_jsons.items()):
        rel = os.path.relpath(fpath, folder)
        print(f'  [json] {"(dry) " if args.dry_run else ""}fixing: {rel}')
        fix_json(fpath, new_text, dry_run=args.dry_run)
        n_fixed += 1

    if legacy_folder:
        new_basename = apply_token_remap(basename)
        new_folder   = os.path.join(parent, new_basename)
        if os.path.exists(new_folder):
            print(f'  [dir]  WARNING: rename target already exists — skipping.')
            print(f'         {new_folder}')
        else:
            print(f'  [dir]  {"(dry) " if args.dry_run else ""}rename:')
            print(f'         {basename}')
            print(f'         → {new_basename}')
            if not args.dry_run:
                os.rename(folder, new_folder)
            n_fixed += 1

    print()
    print(f'[ patch ] {"(dry) " if args.dry_run else ""}{n_fixed} item(s) fixed.')


if __name__ == '__main__':
    main()
