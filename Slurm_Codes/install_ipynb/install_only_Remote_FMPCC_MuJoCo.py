# %% [markdown]
# # FM-PCC Remote Workflow
# 
# ---
# 
# ## Run Order
# 1. Setup Workspace
# 2. Clone/Update FM-PCC
# 3. Install Miniconda + Create `FMPCC` env
# 4. Install D3IL
# 5. Install requirements with pinned compatibility
# 6. Set runtime env variables
# 7. Optional W&B relogin
# 8. Verify dependencies
# 9. Prepare dataset
# 10. Smoke test
# 
# 
# ## Remember to clear cell outputs before run!
# 


# %% [code]
# @title Notebook Version Marker
from datetime import datetime
import pytz

# Change 'UTC' to your local timezone if preferred (e.g., 'US/Eastern', 'Asia/Shanghai')
TIMEZONE = 'UTC'

now = datetime.now(pytz.timezone(TIMEZONE))
version_mark = now.strftime("%Y.%m.%d_%H%M%S")

print("=" * 40)
print(f"SESSION VERSION: {version_mark}")
print(f"START TIME     : {now.strftime('%A, %b %d, %Y - %I:%M:%S %p %Z')}")
print("=" * 40)
print("Raw Traj Test (Ben.V4) II - more traj - init points test.")
print("Raw Traj Test (Ben.V4) II - more traj - init points test. Batch 20 test done, init test failed, colab timeout")

# %% [markdown]
# ## 1) Setup Workspace

# %% [code]
import os

# Using ~/FMPCC as the root directory on the remote machine
FMPCC_ROOT = os.path.expanduser('~/FMPCC')
REPO = os.path.join(FMPCC_ROOT, 'FM-PCC')
os.makedirs(FMPCC_ROOT, exist_ok=True)

print('Repo path:', REPO)



# %% [markdown]
# ## 2) Clone or Update FM-PCC
# 


# %% [code]
# @title Git Repository Sync
# @markdown Set this to 1 to replace any edited GitHub files with the latest versions.
# @markdown Your new files/notes will NOT be deleted.
OVERWRITE_LOCAL_CHANGES = "1" # @param [0, 1]
UPDATE_REPO = 1 # @param [0, 1]

import os
os.environ['OVERWRITE_LOCAL_CHANGES'] = str(OVERWRITE_LOCAL_CHANGES)
os.environ['UPDATE_REPO'] = str(UPDATE_REPO)

# %% [code]
%%bash
set -e

ROOT="$HOME/FMPCC"
REPO="$ROOT/FM-PCC"
BRANCH="main" # You can change this if you ever decide to switch branches

mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d "$REPO/.git" ]; then
  echo "Cloning fresh repository..."
  git clone --recurse-submodules https://github.com/ghubliming/FM-PCC.git
else
  cd "$REPO"

  if [ "$UPDATE_REPO" != "1" ]; then
    echo "Repo update disabled. Using existing local checkout."
  else
    echo "Fetching latest updates from GitHub..."
    git fetch origin "$BRANCH"

    if [ "$OVERWRITE_LOCAL_CHANGES" = "1" ]; then
      echo "OVERWRITE_LOCAL_CHANGES=1: Resetting tracked files to match remote."
      # This command ONLY affects files that exist in the Git repo.
      # It does NOT touch your new notes or results.
      git reset --hard "origin/$BRANCH"
      git submodule update --init --recursive
    else
      # Check if there are any local edits to GitHub files
      CHANGED_FILES="$(git status --porcelain | grep '^ M' || true)"
      if [ -n "$CHANGED_FILES" ]; then
        echo "WARNING: Local edits detected in GitHub files. Skipping update to protect them."
        echo "Set OVERWRITE_LOCAL_CHANGES=1 to replace these files with the remote version."
      else
        echo "No conflicts found. Updating..."
        git merge "origin/$BRANCH"
        git submodule update --init --recursive
      fi
    fi
  fi
fi

echo "------------------------------------------------"
echo "Repo ready: $REPO"
echo "Current Branch: $(git branch --show-current)"
echo "------------------------------------------------"

# This part specifically shows you what is yours (notes/new files)
echo "Your Local Notes & New Files (Not in Git):"
UNTRACKED="$(git ls-files --others --exclude-standard)"
if [ -z "$UNTRACKED" ]; then
  echo "  (None)"
else
  echo "$UNTRACKED"
fi

# %% [markdown]
# %% [markdown]
# ## 3) Install Miniconda and Create Env
# 
# Keeps Python pinned to 3.10 for compatibility with project dependencies.
# 


# %% [code]
%%bash
set -e

ROOT="$HOME/FMPCC"
PERSIST_CONDA="$ROOT/miniconda3"
RUNTIME_CONDA="$HOME/FMPCC/miniconda3"
CONDA_BIN="$PERSIST_CONDA/bin/conda"
LOCAL_FALLBACK_CONDA="$HOME/FMPCC/miniconda3_runtime"
CONDA_SNAPSHOT_DIR="$ROOT/cache"
CONDA_SNAPSHOT="$CONDA_SNAPSHOT_DIR/fmpcc_conda_env.tar.gz"
FORCE_REINSTALL_CONDA="${FORCE_REINSTALL_CONDA:-0}"
REFRESH_CONDA_SNAPSHOT="${REFRESH_CONDA_SNAPSHOT:-0}"

# Ensure the runtime path points to the persistent conda directory.
if [ -L "$RUNTIME_CONDA" ]; then
  LINK_TARGET="$(readlink -f "$RUNTIME_CONDA" || true)"
  if [ "$LINK_TARGET" != "$PERSIST_CONDA" ]; then
    rm -f "$RUNTIME_CONDA"
    ln -s "$PERSIST_CONDA" "$RUNTIME_CONDA"
  fi
elif [ ! -e "$RUNTIME_CONDA" ]; then
  ln -s "$PERSIST_CONDA" "$RUNTIME_CONDA"
fi

if [ "$FORCE_REINSTALL_CONDA" = "1" ]; then
  echo "FORCE_REINSTALL_CONDA=1 -> reinstalling Miniconda"
  rm -rf "$PERSIST_CONDA"
fi

if [ ! -d "$PERSIST_CONDA" ]; then
  echo "First run detected: no persistent conda directory yet"
fi

# 1) Check if conda exists and works.
NEED_CONDA_INSTALL=0
if [ -x "$CONDA_BIN" ]; then
  chmod +x "$PERSIST_CONDA/bin/python" "$PERSIST_CONDA/bin/conda" || true
  if ! "$CONDA_BIN" --version >/tmp/conda_check.log 2>&1; then
    if grep -qi "Permission denied" /tmp/conda_check.log; then
      echo "Drive conda is not executable (permission denied). Switching to local runtime conda."
      if [ -L "$RUNTIME_CONDA" ] || [ -e "$RUNTIME_CONDA" ]; then
        rm -rf "$RUNTIME_CONDA"
      fi
      ln -s "$LOCAL_FALLBACK_CONDA" "$RUNTIME_CONDA"
      PERSIST_CONDA="$LOCAL_FALLBACK_CONDA"
      CONDA_BIN="$PERSIST_CONDA/bin/conda"
      if [ -x "$CONDA_BIN" ]; then
        echo "Reusing existing local runtime conda."
      else
        echo "No local runtime conda found yet; it will be installed now."
        NEED_CONDA_INSTALL=1
      fi
    else
      echo "Conda exists but failed health check -> reinstalling"
      rm -rf "$PERSIST_CONDA"
      NEED_CONDA_INSTALL=1
    fi
  else
    echo "Conda exists and passed health check -> skip reinstall"
  fi
else
  NEED_CONDA_INSTALL=1
fi

# 2) If there is a problem, reinstall.
if [ "$NEED_CONDA_INSTALL" = "1" ]; then
  if [ -f "$CONDA_SNAPSHOT" ]; then
    echo "Restoring conda snapshot: $CONDA_SNAPSHOT"
    rm -rf "$PERSIST_CONDA"
    mkdir -p "$PERSIST_CONDA"
    if ! tar -xzf "$CONDA_SNAPSHOT" -C "$PERSIST_CONDA" --strip-components=1; then
      echo "Snapshot restore failed -> will run installer fallback"
      rm -rf "$PERSIST_CONDA"
    fi
  fi

  if [ ! -x "$CONDA_BIN" ] || ! "$CONDA_BIN" --version >/dev/null 2>&1; then
    echo "Conda snapshot missing/broken -> installing Miniconda"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /content/miniconda.sh
    bash /content/miniconda.sh -b -p "$PERSIST_CONDA" -u
  else
    echo "Conda restored from snapshot"
  fi
fi

"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if ! "$CONDA_BIN" env list | grep -q "^FMPCC "; then
  "$CONDA_BIN" create -n FMPCC python=3.10 -y -q
fi

if [ ! -x "$PERSIST_CONDA/envs/FMPCC/bin/python" ] || [ ! -x "$PERSIST_CONDA/envs/FMPCC/bin/pip" ]; then
  "$CONDA_BIN" remove -n FMPCC --all -y || true
  "$CONDA_BIN" create -n FMPCC python=3.10 -y -q
fi

# If env binaries on Drive are not executable, switch to local runtime conda and rebuild env there.
if ! "$PERSIST_CONDA/envs/FMPCC/bin/python" -V >/tmp/env_python_check.log 2>&1; then
  if grep -qi "Permission denied" /tmp/env_python_check.log; then
    echo "FMPCC env python is not executable on current path. Switching env to local runtime conda."
    if [ -L "$RUNTIME_CONDA" ] || [ -e "$RUNTIME_CONDA" ]; then
      rm -rf "$RUNTIME_CONDA"
    fi
    ln -s "$LOCAL_FALLBACK_CONDA" "$RUNTIME_CONDA"
    PERSIST_CONDA="$LOCAL_FALLBACK_CONDA"
    CONDA_BIN="$PERSIST_CONDA/bin/conda"

    if [ ! -x "$CONDA_BIN" ]; then
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /content/miniconda.sh
      bash /content/miniconda.sh -b -p "$PERSIST_CONDA" -u
    fi

    "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    "$CONDA_BIN" remove -n FMPCC --all -y || true
    "$CONDA_BIN" create -n FMPCC python=3.10 -y -q
  else
    cat /tmp/env_python_check.log
    exit 1
  fi
fi

"$PERSIST_CONDA/envs/FMPCC/bin/python" -V
"$PERSIST_CONDA/envs/FMPCC/bin/pip" --version

# Save a conda snapshot for fast restore on future Colab runtimes.
mkdir -p "$CONDA_SNAPSHOT_DIR"
if [ ! -f "$CONDA_SNAPSHOT" ] || [ "$REFRESH_CONDA_SNAPSHOT" = "1" ]; then
  echo "Creating conda snapshot: $CONDA_SNAPSHOT"
  if ! tar -czf "$CONDA_SNAPSHOT" -C "$PERSIST_CONDA" .; then
    echo "Snapshot creation failed -> continuing without cache update"
  fi
else
  echo "Conda snapshot exists -> skipping snapshot refresh"
fi



# %% [markdown]
# ## 4) Install D3IL (Install Once + Verify)
# 
# Uses editable installs for both D3IL core and `gym_avoiding_env`, but skips reinstall when editable links already exist.
# 


# %% [markdown]
# ```
# %%bash
# set -e
# 
# PIP="$HOME/FMPCC/miniconda3/envs/FMPCC/bin/pip"
# REPO="$HOME/FMPCC/FM-PCC"
# D3IL="$REPO/d3il"
# 
# if [ ! -d "$D3IL/.git" ]; then
#   echo "d3il missing/incomplete -> recloning"
#   rm -rf "$D3IL"
#   git clone https://github.com/ALRhub/d3il.git "$D3IL"
# fi
# 
# if "$PIP" freeze | grep -Fq "d3il/environments/d3il" && "$PIP" freeze | grep -Fq "d3il/envs/gym_avoiding_env"; then
#   echo "D3IL editable installs already present; skipping reinstall"
# else
#   "$PIP" install -e "$D3IL/environments/d3il"
#   "$PIP" install -e "$D3IL/environments/d3il/envs/gym_avoiding_env"
# fi
# 
# echo "D3IL installed"
# ```


# %% [markdown]
# ### After Gen4 Update, the D3IL is in FM-PCC Repo, just need install now


# %% [code]
%%bash
set -e

PIP="$HOME/FMPCC/miniconda3/envs/FMPCC/bin/pip"
REPO="$HOME/FMPCC/FM-PCC"
D3IL="$REPO/d3il"

if [ ! -d "$D3IL" ]; then
  echo "ERROR: d3il directory not found at $D3IL"
  exit 1
fi

if "$PIP" freeze | grep -Fq "d3il/environments/d3il" && "$PIP" freeze | grep -Fq "d3il/envs/gym_avoiding_env"; then
  echo "D3IL editable installs already present; skipping reinstall"
else
  "$PIP" install -e "$D3IL/environments/d3il"
  "$PIP" install -e "$D3IL/environments/d3il/envs/gym_avoiding_env"
fi

echo "D3IL installed"

# %% [markdown]
# ## 5) Install Requirements (Install Once + Verify)
# 
# Runs validation first and only installs when the environment is missing or inconsistent.
# 


# %% [code]
%%bash
set -e

REPO="$HOME/FMPCC/FM-PCC"
PIP="$HOME/FMPCC/miniconda3/envs/FMPCC/bin/pip"
PY="$HOME/FMPCC/miniconda3/envs/FMPCC/bin/python"

cd "$REPO"

"$PY" - <<'PY'
import importlib
import sys

pkgs = [
    'torch', 'numpy', 'scipy', 'gym', 'gymnasium', 'gymnasium_robotics',
    'minari', 'wandb', 'mujoco', 'diffusers', 'transformers'
]

ok = True
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception as e:
        ok = False

import numpy
if int(numpy.__version__.split('.')[0]) >= 2:
    ok = False

sys.exit(0 if ok else 2)
PY

if [ $? -ne 0 ]; then
  echo "Package validation failed; installing requirements"
  "$PIP" install -r requirements.txt
  "$PIP" check || true
else
  echo "Package validation passed; skipping reinstall"
fi

# Quick sanity check
"$PY" - <<'PY'
import numpy, torch
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("python:", __import__("sys").executable)
PY




# %% [markdown]
# ## 6) Runtime Environment Variables
# 
# Includes W&B malformed service cleanup and Colab rendering settings.
# 


# %% [code]
import os

FMPCC = '$HOME/FMPCC/FM-PCC'
D3IL_ROOT = f'{FMPCC}/d3il'
GYM_AV = f'{D3IL_ROOT}/environments/d3il/envs/gym_avoiding_env'

existing_pp = os.environ.get('PYTHONPATH', '')
parts = [FMPCC, D3IL_ROOT, GYM_AV]
if existing_pp:
    parts.append(existing_pp)

os.environ['FMPCC'] = FMPCC
os.environ['PYTHONPATH'] = ':'.join(parts)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MPLBACKEND'] = 'agg'

for key in ('WANDB_SERVICE', 'WANDB__SERVICE'):
    os.environ.pop(key, None)

os.chdir(FMPCC)
print('cwd:', os.getcwd())
print('PYTHONPATH:', os.environ['PYTHONPATH'])



# %% [markdown]
# ## 7) Optional W&B Login
# 


# %% [code]
import os
from pathlib import Path
import wandb

KEY_FILE = Path('$HOME/FMPCC/.wandb_api_key')

if not KEY_FILE.exists():
    raise FileNotFoundError(
        f'Missing W&B key file: {KEY_FILE}. Create it with your API key on one line.'
    )

api_key = KEY_FILE.read_text(encoding='utf-8').strip()
if not api_key:
    raise ValueError(f'W&B key file is empty: {KEY_FILE}')

for k in ('WANDB_SERVICE', 'WANDB__SERVICE'):
    os.environ.pop(k, None)

wandb.finish()
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_API_KEY'] = api_key

wandb.login(key=api_key, relogin=True)

print('W&B mode:', os.environ.get('WANDB_MODE'))
print('W&B key file:', KEY_FILE)
print('W&B key loaded:', f'***{api_key[-4:]}' if len(api_key) >= 4 else '***')



# %% [markdown]
# ## 8) Full Verification
# 
# Validates import chain with the exact env interpreter used for training.
# 


# %% [code]
%%bash
set -e

$HOME/FMPCC/miniconda3/envs/FMPCC/bin/python - <<'PY'
import importlib
import sys

pkgs = [
    'torch', 'numpy', 'scipy', 'gym', 'gymnasium', 'gymnasium_robotics',
    'minari', 'wandb', 'mujoco', 'diffusers', 'transformers'
]

ok = True
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', 'unknown')
        print(f'{p:20s} {v}')
    except Exception as e:
        ok = False
        print(f'{p:20s} NOT IMPORTABLE ({type(e).__name__}: {e})')

import numpy, torch
print('numpy pinned:', numpy.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))

major = int(numpy.__version__.split('.')[0])
if major >= 2:
    ok = False
    print('ERROR: numpy 2.x detected, expected 1.26.4 for this workflow')

if not ok:
    sys.exit(2)
PY



# %% [markdown]
# ## 9) Dataset Preparation (Avoiding)
# 
# ### Option A: Use existing zip from old DPCC path
# 
# Warning! : It searches the ~15Gb Zip in the DPCC Path, not this FMPCC Path!
# 
# 
# This exits quickly if avoiding data already exists.
# 


# %% [code]
%%bash
set -e

REPO="$HOME/FMPCC/FM-PCC"
AVOIDING_DATA="$REPO/d3il/environments/dataset/data/avoiding/data"
DATA_ZIP="$HOME/FMPCC/DPCC/d3il/environments/dataset/data/dataset.zip"

if [ -d "$AVOIDING_DATA" ] && [ "$(ls -A "$AVOIDING_DATA")" ]; then
  echo "avoiding data already present: $(ls "$AVOIDING_DATA" | wc -l) files"
  exit 0
fi

if [ ! -f "$DATA_ZIP" ]; then
  echo "dataset zip not found: $DATA_ZIP"
  echo "Skip this cell and use Option B below if needed."
  exit 1
fi

TMP="/tmp/avoiding_tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
unzip -q "$DATA_ZIP" "avoiding/*" -d "$TMP"
mkdir -p "$REPO/d3il/environments/dataset/data/avoiding"
cp -r "$TMP/avoiding/." "$REPO/d3il/environments/dataset/data/avoiding/"
rm -rf "$TMP"

echo "avoiding dataset ready: $(ls "$AVOIDING_DATA" | wc -l) files"



# %% [markdown]
# ### Option B: Download full D3IL dataset zip with gdown (only if Option A unavailable)
# 


# %% [markdown]
# ```
# %%bash
# set -e
# 
# REPO="$HOME/FMPCC/FM-PCC"
# DATA_DIR="$REPO/d3il/environments/dataset/data"
# ZIP_FILE="$DATA_DIR/dataset.zip"
# 
# if [ -f "$ZIP_FILE" ]; then
#   echo "zip already exists: $ZIP_FILE"
#   exit 0
# fi
# 
# $HOME/FMPCC/miniconda3/envs/FMPCC/bin/pip install gdown -q
# $HOME/FMPCC/miniconda3/envs/FMPCC/bin/python -m gdown \
#   "https://drive.google.com/uc?id=1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8" \
#   -O "$ZIP_FILE"
# 
# echo "downloaded zip: $ZIP_FILE"
# 


# %% [markdown]
# ## 10) Smoke Test Train
# 
# Short check before full run.
# 


