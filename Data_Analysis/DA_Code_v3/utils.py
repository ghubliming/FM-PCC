"""
Utility functions for data loading, logging, and file operations.
"""
import os
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def create_output_directory(base_path, prefix='FM_V3_ODE_Analysis', return_timestamp=False):
    """
    Create output directory with timestamp.
    
    Args:
        base_path: Base directory for output
        prefix: Prefix for folder name
    
    Returns:
        Path to created directory, or (path, timestamp) when return_timestamp is True
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_path, f'{timestamp}_{prefix}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    if return_timestamp:
        return output_dir, timestamp

    return output_dir


def discover_seed_folders(root_path, expected_seeds=None):
    """
    Discover all seed folders in a results directory.
    
    Args:
        root_path: Root results directory
        expected_seeds: List of expected seed numbers (optional)
    
    Returns:
        List of seed numbers found
    """
    seeds_found = []
    
    if not os.path.exists(root_path):
        return seeds_found
    
    for folder in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, folder)):
            try:
                seed = int(folder)
                if expected_seeds is None or seed in expected_seeds:
                    seeds_found.append(seed)
            except ValueError:
                pass  # Not a seed number
    
    return sorted(seeds_found)


def discover_files_in_structure(root_path, pattern='*.npz'):
    """
    Recursively find files matching pattern in tree structure.
    
    Args:
        root_path: Root directory
        pattern: File pattern (e.g., '*.npz')
    
    Returns:
        List of file paths
    """
    from pathlib import Path
    
    if not os.path.exists(root_path):
        return []
    
    files = list(Path(root_path).rglob(pattern))
    return [str(f) for f in files]


def ensure_numeric(value, default=0):
    """
    Ensure value is numeric (float), return default if invalid.
    
    Args:
        value: Input value
        default: Default value if conversion fails
    
    Returns:
        Numeric value
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, return default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is 0 or division fails
    
    Returns:
        Result or default
    """
    try:
        if denominator == 0 or denominator is None:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default
