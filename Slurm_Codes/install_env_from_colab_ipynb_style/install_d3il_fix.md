# D3IL Installation Hotfix: Restoring Visual Dependencies

This document explains the discrepancy between the standard D3IL installation and the "Colab-style" remote installer, and how to properly complete the installation for **Visual/Diffusion** tasks.

---

## 1. The Previous State (Why it "Partially" Worked)
In the original `remote_setup_guide.md` and the SLURM Colab installer, the D3IL installation was performed as follows:

```bash
# From remote_setup_guide.md Step 4 & 5
pip install -e d3il/environments/d3il
pip install -e d3il/environments/d3il/envs/gym_avoiding_env
pip install -r requirements.txt
```

**Outcome:**
- ✅ **Avoiding Task**: Works perfectly because it only uses the simulation physics.
- ❌ **Visual Tasks**: Crashes with `ModuleNotFoundError: No module named 'omegaconf'`.

## 2. What Went Wrong
The D3IL repository uses a "fragmented" dependency structure:
1.  **Empty `setup.py`**: The D3IL core package has a `setup.py` that lists **zero** dependencies. Thus, `pip install -e` installs the code but none of the libraries needed to run it.
2.  **Manual install.sh**: The original `d3il/install.sh` has a manual line `pip install hydra-core==1.1.1` buried in the script.
3.  **Installer Gap**: The remote/SLURM installer trusted the `setup.py` and the `requirements.txt`, both of which were missing the Hydra/OmegaConf libraries required for D3IL's Vision Encoders and Diffusion Core.

## 3. The "Real" Installation Fix
To fully install D3IL support for Vision-based Gen 5 models, you must ensure the underlying configuration engines are present.

### Option A: Manual Fix (On Cluster)
If you have an existing `FMPCC` environment, run:
```bash
pip install omegaconf==2.1.1 hydra-core==1.1.1
```

### Option B: The "Gen 5" Way
I have updated the project-level `requirements.txt` to include these missing libraries. For any new installation, simply running the standard requirements step will now work:
```bash
pip install -r requirements.txt
```

---

## 4. Verification
After the fix, you should be able to run this without error:
```bash
python -c "import omegaconf; import hydra; print('D3IL Vision Dependencies: READY')"
```
