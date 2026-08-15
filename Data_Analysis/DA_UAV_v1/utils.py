"""DA_UAV_v1 — logging, output directories, HTML-visualizer manifest."""

import json
import logging
import os
from datetime import datetime

from config import OUTPUT_FOLDER_PREFIX, VIEWER_LIST_PREFIX


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
    leading prefix, so `20260815_..._batch_uav` would never be listed.
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


def check_viewer_visibility(output_dir):
    """Warn when the run folder will not show up in the viewer's run picker.

    `Visualizer_UAV_v1/index.html` regexes the `analysis_results/` directory
    listing for `batch_uav_*`; a differently-named `--output-path` still works
    but has to be opened through the viewer's CUSTOM_PATH box.
    """
    name = os.path.basename(os.path.abspath(output_dir))
    if name.startswith(VIEWER_LIST_PREFIX):
        return True
    logging.getLogger(__name__).warning(
        f"Output folder '{name}' does not start with '{VIEWER_LIST_PREFIX}' — it will "
        f"not appear in the viewer's QUICK_LIST; open it via CUSTOM_PATH instead.")
    return False


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
