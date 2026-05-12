# 05 Gen5: Visual Path Robustness Hotfix (12 May)

Date: 2026-05-12
Status: Completed
Related Traceback: `FileNotFoundError` on cluster run (`train_files.pkl` not found)

---

## 1) Problem Description

During a remote training run on a SLURM cluster, the visual training script failed with the following error:

```
FileNotFoundError: [Errno 2] No such file or directory: '/u/home/llim/FMPCC/FM-PCC/d3il/environments/dataset/data/aligning/train_files.pkl'
```

### Analysis
- **CWD**: The script was running from `/data/home/llim/FMPCC/FM-PCC/`.
- **Import Origin**: The traceback showed that `d3il` modules were being imported from `/u/home/llim/FMPCC/FM-PCC/d3il/`.
- **Cause**: On certain clusters, `/u/home` and `/data/home` may be different mounts or aliases. If a version of the code is installed in the user's home directory (e.g., via `pip install -e`), the `PYTHONPATH` might prioritize the stale version in `/u/home` which lacks the newly added data files or sub-folders.
- **Mechanism**: The `d3il` dataset class uses `agents.utils.sim_path.sim_framework_path()`, which calculates the absolute path to the framework root based on the `__file__` location of the `sim_path.py` script. If the script is loaded from `/u/home`, it will search for data in `/u/home`, ignoring the local data in `/data/home`.

---

## 2) Solution: Path Robustness Implementation

To ensure the visual pipeline always uses the local, vendored version of `d3il` and the current workspace, the following changes were implemented:

### 2.1 PYTHONPATH Prioritization
In both `train_ddpm_encdec_vision.py` and `eval_ddpm_encdec_vision.py`, `sys.path` is now explicitly modified at the very top of the script:

```python
# Ensure the current directory and vendored d3il are in the path
# This prevents picking up stale versions from other locations (e.g. /u/home vs /data/home)
import sys
import os
current_dir = os.path.abspath(os.path.curdir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
d3il_path = os.path.join(current_dir, 'd3il')
if d3il_path not in sys.path:
    sys.path.insert(0, d3il_path)
```

### 2.2 Dataset Path Standardization
The `data_directory` argument for the `Aligning_Img_Dataset` was standardized to a relative path that is compatible with D3IL's internal path resolution logic.

```python
# train_ddpm_encdec_vision.py
train_data_rel_path = 'environments/dataset/data/aligning/train_files.pkl'
dataset = Aligning_Img_Dataset(
    data_directory=train_data_rel_path,
    ...
)
```

---

## 3) Verification

A debug script `scratch/debug_path.py` was executed in the workspace to verify that `sim_framework_path` correctly resolves to the absolute path within the current `FM-PCC` directory when `d3il` is correctly placed in the path.

**Output:**
```
Rel path: environments/dataset/data/aligning/train_files.pkl
Abs path: /workspaces/FM-PCC/d3il/environments/dataset/data/aligning/train_files.pkl
Exists: True
```

---

## 4) Conclusion
The visual pipeline is now hardened against environment mount variations. It will consistently prioritize the local codebase and data files provided in the `FM-PCC` repository.
