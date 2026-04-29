# FM-PCC Remote SLURM Setup Guide

This guide outlines the "Colab-style" robust installation process for setting up FM-PCC on a remote SLURM cluster via SSH.

---

## 1. Prerequisites: Install Miniconda

---

## 2. Git & Workspace Setup (SSH Mode)
To push/pull without passwords, use SSH keys.

### 2.1 Generate SSH Key
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub  # Copy this to GitHub Settings -> SSH Keys
```

### 2.2 Clone Repository
Clone into the standard `~/FMPCC` root used by the project scripts:
```bash
mkdir -p ~/FMPCC && cd ~/FMPCC
git clone --recurse-submodules git@github.com:ghubliming/FM-PCC.git
```

---

## 3. Environment Creation
Create the dedicated Conda environment:
```bash
conda create -n FMPCC python=3.10 -y
conda activate FMPCC
```

---

## 4. Install D3IL (Editable Packages)
These must be installed in "editable" mode so the code can find the physics engine.
```bash
cd ~/FMPCC/FM-PCC
pip install -e d3il/environments/d3il
pip install -e d3il/environments/d3il/envs/gym_avoiding_env
```

---

## 5. Install Requirements (The "Colab Way")
Install all external dependencies from `requirements.txt`.

> **IMPORTANT**: This step ensures `numpy` is pinned to `1.26.4` to avoid breaking MuJoCo compatibility.

```bash
cd ~/FMPCC/FM-PCC
pip install -r requirements.txt
```

---

## 6. Runtime Environment Variables
These are required for **headless rendering** on remote clusters and for Python to find the project modules. 

> [!TIP]
> For a clean and reproducible setup, these should be included **directly inside your SLURM job script** (like `Slurm_Codes/train_fmpcc_job.sh`). This ensures the environment is perfectly set up on every compute node.

**The Mandatory Exports are:**
```bash
# Project Paths
export FMPCC="$HOME/FMPCC/FM-PCC"
export D3IL_ROOT="$FMPCC/d3il"
export GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$GYM_AV:$PYTHONPATH"

# Headless Rendering (Cluster Mode)
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"

# W&B Cleanup
unset WANDB_SERVICE
unset WANDB__SERVICE
```

---

## 7. W&B API Login (Persistent Mode)
Store your key in a file so scripts can log in automatically without interaction.

```bash
# Save your key
echo "YOUR_API_KEY_HERE" > ~/FMPCC/.wandb_api_key

# Login once to verify
export WANDB_API_KEY=$(cat ~/FMPCC/.wandb_api_key)
wandb login --relogin $WANDB_API_KEY
```

---

## 8. Dataset Preparation
Ensure your `dataset.zip` is ready 
```

---

## 9. Running a Job
Use the provided SLURM script to submit training:
```bash
sbatch Slurm_Codes/.sh
```
