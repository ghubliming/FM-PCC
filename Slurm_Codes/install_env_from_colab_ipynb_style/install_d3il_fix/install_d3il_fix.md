# D3IL Real Installation Fix Guide

## 1. Status After Standard "Colab-Style" Install
After following the `remote_setup_guide.md` exactly:
- **D3IL Code**: Installed (linked in `site-packages`).
- **Simulator**: Working (Avoiding task runs).
- **Vision/Diffusion**: **BROKEN**.
- **Reason**: The standard install only follows `setup.py`, which is empty. It misses the hidden dependencies in `d3il/install.sh`.

### Current Error Status
```bash
(FMPCC) llim@vmknoll81:~$ python -c "import omegaconf"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'omegaconf'
```

---

## 2. Step-by-Step "Real" Fix

### Step 1: Sync Workspace
Push the updated `requirements.txt` (which now contains all D3IL hidden pkgs) to your cluster.

### Step 2: Activate and Update
Run these commands exactly:

```bash
llim@vmknoll81:~$ cd ~/FMPCC/FM-PCC
llim@vmknoll81:~/FMPCC/FM-PCC$ conda activate FMPCC
(FMPCC) llim@vmknoll81:~/FMPCC/FM-PCC$ pip install -r requirements.txt
```

### Step 3: Verify "Real" D3IL Installation
Confirm that the vision/diffusion dependencies are now physically present:

```bash
(FMPCC) llim@vmknoll81:~/FMPCC/FM-PCC$ python -c "import omegaconf; import hydra; import open3d; import addict; print('--- D3IL FULL INSTALL: SUCCESS ---')"
```

---

## 3. Why This Works
The `requirements.txt` has been manually patched to include the missing D3IL core:
- `hydra-core==1.1.1` & `omegaconf==2.1.1` (Config Engines)
- `addict`, `pandas`, `plyfile` (Data Engines)
- `open3d` (Vision/Pointcloud Engine)
- `torchsde`, `torchdiffeq` (Diffusion Engine)
