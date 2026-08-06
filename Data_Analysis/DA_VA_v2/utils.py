"""DA_VA_v2 — logging, output directories, HTML-visualizer manifest."""

import json
import logging
import os
from datetime import datetime

from config import OUTPUT_FOLDER_PREFIX, VIEWER_ALIAS_PREFIX, VIEWER_LIST_PREFIX


def setup_logger(name, log_file=None, level=logging.INFO):
    """Console + file logger. Handlers are reset so a re-run does not double-log."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # The pipeline modules log through their own module loggers; route those to
    # the same handlers so `logs/analysis.log` is the complete record.
    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in logger.handlers:
        root.addHandler(handler)

    return logger


def create_output_directory(base_path, prefix=OUTPUT_FOLDER_PREFIX,
                            return_timestamp=False):
    """`<base>/<prefix>_<timestamp>/` with plots/ and logs/ subdirectories.

    Prefix first, not timestamp first: the visualizers match the run folder by
    leading prefix, so `20260806_..._batch_va2` would never be listed.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_path, f'{prefix}_{timestamp}')
    prepare_output_directory(output_dir)
    if return_timestamp:
        return output_dir, timestamp
    return output_dir


def prepare_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    return output_dir


def create_viewer_alias(output_dir):
    """Drop a `va_batch_*` symlink beside the run folder.

    The two HTML visualizers pick runs out of the `analysis_results/` directory
    listing by leading prefix and disagree about it — `batch_` for
    `Visualizer/index.html`, `va_batch_` for `Visualizer_Visual_Aligning/`. One
    folder name cannot satisfy both, so the real directory is named for the
    former and this alias makes it visible in the latter. Same files, one copy.

    Non-fatal: a filesystem without symlinks just loses the alias.
    """
    logger = logging.getLogger(__name__)
    absolute = os.path.abspath(output_dir)
    parent = os.path.dirname(absolute)
    name = os.path.basename(absolute)

    if name.startswith(VIEWER_ALIAS_PREFIX):
        return None                      # already the VA-visible spelling
    if name.startswith(VIEWER_LIST_PREFIX):
        alias_name = VIEWER_ALIAS_PREFIX + name[len(VIEWER_LIST_PREFIX):]
    else:
        alias_name = VIEWER_ALIAS_PREFIX + name
        logger.warning(
            f"Output folder '{name}' does not start with '{VIEWER_LIST_PREFIX}' — it "
            f"will NOT appear in Visualizer/index.html's run list (that page regexes "
            f"the directory listing for that prefix).")

    alias = os.path.join(parent, alias_name)
    try:
        if os.path.islink(alias):
            if os.path.realpath(alias) == absolute:
                return alias
            os.unlink(alias)
        elif os.path.exists(alias):
            logger.warning(f'Viewer alias path already exists and is not a symlink: {alias}')
            return None
        os.symlink(name, alias)          # relative target — survives a repo move
        logger.info(f'Viewer alias: {alias_name} -> {name}')
        return alias
    except OSError as exc:
        logger.warning(f'Could not create viewer alias {alias_name}: {exc}')
        return None


def update_results_manifest(output_dir):
    """Refresh `results_manifest.json` next to the batch folder.

    The HTML visualizers read it to populate their run dropdown; a static file
    server gives no directory listing, so without this a new batch is invisible.
    """
    try:
        absolute = os.path.abspath(output_dir)
        parent = os.path.dirname(absolute)
        if not os.path.isdir(parent):
            return None
        batches = [d for d in os.listdir(parent)
                   if os.path.isdir(os.path.join(parent, d))
                   and d not in ('plots', 'logs')]
        manifest = os.path.join(parent, 'results_manifest.json')
        with open(manifest, 'w') as f:
            json.dump({'batches': sorted(batches, reverse=True)}, f, indent=2)
        return manifest
    except Exception as exc:                                      # noqa: BLE001
        logging.getLogger(__name__).warning(f'Manifest update failed: {exc}')
        return None


def parse_int_list(text):
    if not text:
        return None
    return [int(part.strip()) for part in text.split(',') if part.strip()]


def parse_str_list(text):
    if not text:
        return None
    return [part.strip() for part in text.split(',') if part.strip()]
