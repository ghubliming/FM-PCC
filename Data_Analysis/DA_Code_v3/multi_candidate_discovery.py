"""
Multi-Candidate Discovery Module (v2)

Auto-discovers experimental candidate folders and assigns letter codes (A, B, C, D, E...).
A candidate is any subfolder that contains the required seed directories.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


def has_seeds(seed_list, folder_path):
    """
    Check if a folder contains all required seed directories.
    
    Args:
        seed_list: List of seed numbers (e.g., [6, 7, 8, 9, 10])
        folder_path: Path to check for seed directories
        
    Returns:
        bool: True if all seeds exist as subdirectories, False otherwise
    """
    if not os.path.isdir(folder_path):
        return False
    
    for seed in seed_list:
        seed_dir = os.path.join(folder_path, str(seed))
        if not os.path.isdir(seed_dir):
            return False
    
    return True


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
        
        # Check if this folder contains all required seeds
        if has_seeds(seed_list, subfolder_path):
            # Assign letter code
            letter_code = chr(ord('A') + letter_index)
            
            candidates[letter_code] = {
                'path': os.path.abspath(subfolder_path),
                'name': subfolder_name,
                'seeds': seed_list.copy()
            }
            
            logger.info(f"Candidate {letter_code}: {subfolder_name}")
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
            if has_seeds(seed_list, entry_path):
                letter_code = chr(ord('A') + letter_index)
                candidates[letter_code] = {
                    'path': os.path.abspath(entry_path),
                    'name': entry,
                    'seeds': seed_list.copy()
                }
                logger.info(f"Candidate {letter_code}: {entry}")
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
        lines.append(f"      Seeds: {info['seeds']}")
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
