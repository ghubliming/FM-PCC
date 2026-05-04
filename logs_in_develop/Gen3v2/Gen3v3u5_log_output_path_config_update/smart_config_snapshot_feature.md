# Feature Report: Smart Config Snapshot

**Date**: 2026-05-04
**Category**: Traceability / Audit Infrastructure
**Status**: Implemented / Upgraded (Force Overwrite)

## Overview
The "Smart Config Snapshot" is an automated archiving feature integrated into the `Parser` utility. It ensures that every experiment (training or evaluation) contains a permanent record of the exact code and settings used, solving the problem of "lost context" when configuration files are modified over time.

## How it Works
The feature is injected into the `Parser.mkdir()` method. When a run initializes or resumes:
1.  **Discovery**: It dynamically identifies the file path of the Python configuration module (`args.config`).
2.  **Archiving**: It creates a dedicated subfolder named `config_snapshot_{config_name}/` inside the run's `savepath`.
3.  **Capture (Force Overwrite)**: It copies the primary `.py` config and any associated `.yaml` evaluation configs (e.g., `projection_eval.yaml`) into this subfolder. **Crucially, it overwrites these files on every run** to ensure the snapshot always matches the current code execution.
4.  **Verification**: After copying is complete, it drops a `snapshot_YYYYMMDD_HHMMSS` marker file into the folder to provide a chronological log of when the snapshot was last updated.

## Key Benefits
-   **Traceability**: Every run is now self-documenting. You no longer need to guess which version of `avoiding-d3il.py` was used for a run from last week.
-   **Comment Preservation**: Unlike `args.json` which only saves values, the snapshot saves your Python comments, human notes, and logic.
-   **YAML Backups**: Since evaluation YAMLs are often shared, this provides a per-run backup of the exact evaluation parameters used.
-   **Live Mirroring**: By overwriting on every run, the snapshot folder is always a 1:1 mirror of the code that generated the current results (matching the behavior of `.npz` and `.png` files).

## Implementation Details
The logic was added to the `Parser` class in the following core utility files:
-   `diffuser/utils/setup.py`
-   `flow_matcher_v3_ode_selectable/utils/setup.py`

## Folder Structure Example
```text
logs/avoiding-d3il/.../seed_0/
├── args.json
├── model_config.pkl
├── config_snapshot_avoiding-d3il/  <-- NEW
│   ├── snapshot_20260504_160800    <-- NEW (Time Marker)
│   ├── avoiding-d3il.py            <-- (Overwritten each run)
│   └── projection_eval.yaml        <-- (Overwritten each run)
├── state_best.pt
└── ...
```

## Maintenance
The feature is designed to be "invisible" and robust. It uses `importlib` for path resolution, so it will continue to work even if you rename or move your configuration files.
