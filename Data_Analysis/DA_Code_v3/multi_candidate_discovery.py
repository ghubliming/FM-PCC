"""
Multi-Candidate Discovery Module (v2)

Auto-discovers experimental candidate folders and assigns numeric indices (1, 2, 3...).
A candidate is any subfolder that contains the required seed directories.
"""

import os
import logging
import re
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Config-snapshot timestamps — "when was this folder last run?"
# ============================================================================
# `diffuser/utils/setup.py::snapshot_configs` copies the config into
# `<savepath>/config_snapshot_<config>/` and then drops a marker file named
# `snapshot_<YYYYMMDD_HHMMSS>` next to it, once per Parser() call. For a
# `.../plans/<exp>/<seed>/` folder that is one marker per EVAL LAUNCH, so:
#   newest marker  = when the results in this folder were last (re)generated
#   marker count   = how many times the folder has been written into
# Re-running an eval never deletes the older markers, which is exactly what
# makes them an audit trail rather than a single mtime.
SNAPSHOT_DIR_PREFIX = 'config_snapshot'
_SNAPSHOT_FILE_RE = re.compile(r'^snapshot_(\d{8}_\d{6})$')


def _snapshot_stamps_in_seed(seed_dir):
    """Every `snapshot_<ts>` stamp under `<seed_dir>/config_snapshot*/`."""
    stamps = []
    try:
        entries = sorted(os.listdir(seed_dir))
    except OSError:
        return stamps
    for entry in entries:
        if not entry.startswith(SNAPSHOT_DIR_PREFIX):
            continue
        snapshot_dir = os.path.join(seed_dir, entry)
        if not os.path.isdir(snapshot_dir):
            continue
        try:
            names = os.listdir(snapshot_dir)
        except OSError:
            continue
        for name in names:
            match = _SNAPSHOT_FILE_RE.match(name)
            if match:
                stamps.append(match.group(1))
    return stamps


def scan_snapshot_timestamps(candidate_path, seeds=None):
    """Collect the config-snapshot timestamps of one candidate folder.

    Args:
        candidate_path: candidate folder (the one holding the seed subdirs)
        seeds: seeds to look at (None = every numeric subdirectory)

    Returns:
        dict:
            latest    newest stamp over all seeds, 'YYYYMMDD_HHMMSS' ('' if none)
            first     oldest stamp over all seeds ('' if none)
            count     total number of stamp files found
            per_seed  {seed:int -> newest stamp of that seed}
            n_seeds_stamped  how many seeds carried at least one stamp

    Never raises: a tree with no snapshots at all (pre-snapshot runs, or a
    hand-assembled folder) simply reports empty strings and count 0.
    """
    if seeds is None:
        try:
            seeds = sorted(int(e) for e in os.listdir(candidate_path)
                           if e.isdigit()
                           and os.path.isdir(os.path.join(candidate_path, e)))
        except OSError:
            seeds = []

    per_seed = {}
    all_stamps = []
    for seed in seeds:
        stamps = _snapshot_stamps_in_seed(os.path.join(candidate_path, str(seed)))
        if not stamps:
            continue
        all_stamps.extend(stamps)
        per_seed[int(seed)] = max(stamps)

    return {
        'latest': max(all_stamps) if all_stamps else '',
        'first': min(all_stamps) if all_stamps else '',
        'count': len(all_stamps),
        'per_seed': per_seed,
        'n_seeds_stamped': len(per_seed),
    }


def format_snapshot_ts(stamp):
    """'20260506_034806' -> '2026-05-06 03:48:06'. Anything else passes through."""
    text = str(stamp or '').strip()
    if len(text) == 15 and text[8] == '_' and text.replace('_', '').isdigit():
        return (f'{text[0:4]}-{text[4:6]}-{text[6:8]} '
                f'{text[9:11]}:{text[11:13]}:{text[13:15]}')
    return text


def snapshot_by_seed_str(per_seed):
    """'6:20260506_034806 | 7:20260506_041233' — the per-seed audit trail.

    Worth a column of its own: seeds of the same candidate are often launched
    by separate jobs, so one stale seed hides behind a fresh `latest`.
    """
    if not per_seed:
        return ''
    return ' | '.join(f'{seed}:{per_seed[seed]}' for seed in sorted(per_seed))


def get_existing_seeds(seed_list, folder_path):
    """
    Check which required seed directories exist in a folder.
    
    Args:
        seed_list: List of seed numbers (e.g., [6, 7, 8, 9, 10])
        folder_path: Path to check for seed directories
        
    Returns:
        list: List of seeds that exist
    """
    existing = []
    if not os.path.isdir(folder_path):
        return existing
    
    for seed in seed_list:
        seed_dir = os.path.join(folder_path, str(seed))
        if os.path.isdir(seed_dir):
            existing.append(seed)
    
    return existing


def discover_candidates(parent_path, seed_list=None):
    """
    Auto-discover candidate folders in a parent directory.
    
    A candidate is any direct subfolder containing all required seeds.
    Candidates are assigned letters: A, B, C, D, E...
    
    Args:
        parent_path: Parent directory path to scan
        seed_list: List of seed numbers to require (default: [6, 7, 8, 9, 10])
        
    Returns:
        dict: Mapping of candidate letters to folder info
        {
            'A': {
                'path': '/full/path/to/folder',
                'name': 'folder_name',
                'seeds': [6, 7, 8, 9, 10]
            },
            'B': {...},
            ...
        }
        
    Raises:
        ValueError: If parent_path doesn't exist
    """
    if seed_list is None:
        seed_list = [6, 7, 8, 9, 10]
    
    if not os.path.isdir(parent_path):
        raise ValueError(f"Parent path does not exist: {parent_path}")
    
    candidates = {}
    letter_index = 0
    
    # Scan immediate subfolders, sorted alphabetically for reproducibility
    subfolders = sorted(os.listdir(parent_path))
    
    for subfolder_name in subfolders:
        subfolder_path = os.path.join(parent_path, subfolder_name)
        
        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Skip hidden folders
        if subfolder_name.startswith('.'):
            continue
        
        # Check if this folder contains required seeds
        existing_seeds = get_existing_seeds(seed_list, subfolder_path)
        if existing_seeds:
            cand_idx = letter_index + 1
            missing_seeds = [s for s in seed_list if s not in existing_seeds]

            if missing_seeds:
                logger.warning(f"Candidate {cand_idx} ({subfolder_name}) is MISSING seeds: {missing_seeds}")

            candidates[cand_idx] = {
                'path': os.path.abspath(subfolder_path),
                'name': subfolder_name,
                'seeds': existing_seeds,
                'missing_seeds': missing_seeds,
                'snapshots': scan_snapshot_timestamps(subfolder_path, existing_seeds)
            }

            logger.info(f"Candidate {cand_idx}: {subfolder_name}")
            letter_index += 1
    
    if not candidates:
        logger.warning(f"No candidates found in {parent_path}")
    else:
        logger.info(f"Total candidates discovered: {len(candidates)}")
    
    return candidates


def discover_candidates_recursive(parent_path, seed_list=None, max_depth=3):
    """
    Recursively discover candidate folders up to max_depth levels deep.
    
    Useful when experimental folders are nested deeper than one level.
    
    Args:
        parent_path: Root directory to search
        seed_list: List of seed numbers to require
        max_depth: Maximum directory depth to search
        
    Returns:
        dict: Same format as discover_candidates()
    """
    if seed_list is None:
        seed_list = [6, 7, 8, 9, 10]
    
    candidates = {}
    letter_index = 0
    
    def _search_recursive(current_path, depth):
        nonlocal letter_index
        
        if depth > max_depth or not os.path.isdir(current_path):
            return
        
        try:
            entries = os.listdir(current_path)
        except PermissionError:
            return
        
        for entry in sorted(entries):
            if entry.startswith('.'):
                continue
            
            entry_path = os.path.join(current_path, entry)
            
            if not os.path.isdir(entry_path):
                continue
            
            # Check if this is a candidate
            existing_seeds = get_existing_seeds(seed_list, entry_path)
            if existing_seeds:
                cand_idx = letter_index + 1
                missing_seeds = [s for s in seed_list if s not in existing_seeds]

                if missing_seeds:
                    logger.warning(f"Candidate {cand_idx} ({entry}) is MISSING seeds: {missing_seeds}")

                candidates[cand_idx] = {
                    'path': os.path.abspath(entry_path),
                    'name': entry,
                    'seeds': existing_seeds,
                    'missing_seeds': missing_seeds,
                    'snapshots': scan_snapshot_timestamps(entry_path, existing_seeds)
                }
                logger.info(f"Candidate {cand_idx}: {entry}")
                letter_index += 1
            else:
                # Recurse deeper
                _search_recursive(entry_path, depth + 1)
    
    _search_recursive(parent_path, 1)
    
    if not candidates:
        logger.warning(f"No candidates found in {parent_path}")
    else:
        logger.info(f"Total candidates discovered: {len(candidates)}")
    
    return candidates


def filter_candidates(candidates, selected_letters):
    """
    Filter candidates by letter selection.
    
    Args:
        candidates: Full candidate dict from discover_candidates()
        selected_letters: String like "A,C,E" or list ['A', 'C', 'E']
        
    Returns:
        dict: Filtered candidates dict
    """
    if isinstance(selected_letters, str):
        selected_letters = [l.strip().upper() for l in selected_letters.split(',')]
    else:
        selected_letters = [l.upper() for l in selected_letters]
    
    filtered = {k: v for k, v in candidates.items() if k in selected_letters}
    
    logger.info(f"Filtered candidates: {list(filtered.keys())}")
    return filtered


def assign_custom_names(candidates, custom_names):
    """
    Assign custom names to candidates instead of auto-discovered names.
    
    Args:
        candidates: Full candidate dict from discover_candidates()
        custom_names: String like "name1,name2,name3" or list of names
        
    Returns:
        dict: Candidates dict with updated names
    """
    if isinstance(custom_names, str):
        names = [n.strip() for n in custom_names.split(',')]
    else:
        names = custom_names
    
    candidate_letters = sorted(candidates.keys())
    
    if len(names) != len(candidate_letters):
        logger.warning(
            f"Number of names ({len(names)}) doesn't match candidates ({len(candidate_letters)}). "
            f"Using auto names."
        )
        return candidates
    
    updated = {}
    for letter, name in zip(candidate_letters, names):
        updated[letter] = candidates[letter].copy()
        updated[letter]['custom_name'] = name
    
    logger.info(f"Applied custom names: {names}")
    return updated


def get_candidate_summary(candidates):
    """
    Generate human-readable summary of discovered candidates.
    
    Args:
        candidates: Candidate dict from discover_candidates()
        
    Returns:
        str: Formatted summary text
    """
    lines = [
        "=== Candidates Discovered ===",
        f"Total: {len(candidates)}",
        ""
    ]
    
    for letter in sorted(candidates.keys()):
        info = candidates[letter]
        lines.append(f"  {letter}: {info['name']}")
        lines.append(f"      Path: {info['path']}")
        lines.append(f"      Seeds Found: {info['seeds']}")
        snapshots = info.get('snapshots') or {}
        if snapshots.get('latest'):
            lines.append(f"      Last Run: {format_snapshot_ts(snapshots['latest'])}"
                         f"  ({snapshots['count']} config snapshot(s) over "
                         f"{snapshots['n_seeds_stamped']} seed(s))")
        if info.get('missing_seeds'):
            lines.append(f"      WARNING: Missing Seeds {info['missing_seeds']}")
        if 'custom_name' in info:
            lines.append(f"      Display: {info['custom_name']}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the discovery module
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        parent = sys.argv[1]
        candidates = discover_candidates(parent)
        print(get_candidate_summary(candidates))
    else:
        print("Usage: python multi_candidate_discovery.py <parent_path>")
