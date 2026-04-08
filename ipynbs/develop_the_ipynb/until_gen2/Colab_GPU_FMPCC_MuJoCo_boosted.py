# %% [markdown]
# # FM-PCC Colab Workflow
#
# ---
#
# ## Run Order
# 1. Mount Drive
# 2. Clone/Update FM-PCC
# 3. Boost startup cache wiring
# 4. Install Miniconda + Create `FMPCC` env
# 5. Install D3IL
# 6. Install requirements with pinned compatibility
# 7. Set runtime env variables
# 8. Optional W&B relogin
# 9. Verify dependencies
# 10. Prepare dataset
# 11. Smoke test
# 12. Full train
# 13. Eval + visualization
# 14. Archive logs

# %% [markdown]
# ## 1) Mount Google Drive

# %%
import os
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

FMPCC_ROOT = '/content/drive/MyDrive/FMPCC'
REPO = f'{FMPCC_ROOT}/FM-PCC'
os.makedirs(FMPCC_ROOT, exist_ok=True)

print('Drive mounted')
print('Repo path:', REPO)

# %% [markdown]
# ## 2) Clone or Update FM-PCC

# %%
%%bash
set -e

ROOT="/content/drive/MyDrive/FMPCC"
REPO="$ROOT/FM-PCC"
UPDATE_REPO="${UPDATE_REPO:-1}"
OVERWRITE_LOCAL_CHANGES="${OVERWRITE_LOCAL_CHANGES:-0}"

mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d "$REPO/.git" ]; then
  git clone --recurse-submodules https://github.com/ghubliming/FM-PCC.git
else
  cd "$REPO"
  if [ "$UPDATE_REPO" != "1" ]; then
    echo "Repo update disabled (UPDATE_REPO=$UPDATE_REPO). Using existing local checkout."
    echo "Repo ready: $REPO"
    exit 0
  fi

  CHANGED_FILES="$(git status --porcelain)"
  if [ -n "$CHANGED_FILES" ]; then
    echo "WARNING: Local changes detected in repo."
    echo "Changed files:"
    echo "$CHANGED_FILES"

    if [ "$OVERWRITE_LOCAL_CHANGES" = "1" ]; then
      echo "OVERWRITE_LOCAL_CHANGES=1 -> discarding local changes and untracked files"
      git reset --hard HEAD
      git clean -fd
    else
      echo "Skipping git pull to protect local changes."
      echo "Set OVERWRITE_LOCAL_CHANGES=1 to force overwrite on next run."
      echo "Repo ready: $REPO"
      exit 0
    fi
  fi

  git fetch origin
  git reset --hard origin/main
  git submodule update --init --recursive
fi

echo "Repo ready: $REPO"

# %% [markdown]
# ## 3) Boost Startup Cache Wiring
#
# Keeps the original `/content/miniconda3` path logic, but maps it to Drive so restarts can reuse the same conda env and pip cache.

# %%
%%bash
set -e

ROOT="/content/drive/MyDrive/FMPCC"
PERSIST_CONDA="$ROOT/miniconda3"
RUNTIME_CONDA="/content/miniconda3"
PIP_CACHE="$ROOT/.pip-cache"

mkdir -p "$ROOT" "$PIP_CACHE"

if [ -L "$RUNTIME_CONDA" ]; then
  rm -f "$RUNTIME_CONDA"
elif [ -d "$RUNTIME_CONDA" ]; then
  rm -rf "$RUNTIME_CONDA"
fi

ln -s "$PERSIST_CONDA" "$RUNTIME_CONDA"

echo "Runtime conda path mapped to: $PERSIST_CONDA"
echo "Persistent pip cache path: $PIP_CACHE"

# %% [markdown]
# ## 4) Install Miniconda and Create Env
#
# Keeps Python pinned to 3.10 for compatibility with project dependencies.

# %%
%%bash
set -e

ROOT="/content/drive/MyDrive/FMPCC"
PERSIST_CONDA="$ROOT/miniconda3"
RUNTIME_CONDA="/content/miniconda3"
CONDA_BIN="$PERSIST_CONDA/bin/conda"
LOCAL_FALLBACK_CONDA="/content/miniconda3_runtime"
CONDA_SNAPSHOT_DIR="$ROOT/cache"
CONDA_SNAPSHOT="$CONDA_SNAPSHOT_DIR/fmpcc_conda_env.tar.gz"
FORCE_REINSTALL_CONDA="${FORCE_REINSTALL_CONDA:-0}"
REFRESH_CONDA_SNAPSHOT="${REFRESH_CONDA_SNAPSHOT:-0}"
REBUILD_CONDA_SNAPSHOT=0

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
  REBUILD_CONDA_SNAPSHOT=1
fi

if [ -f "$CONDA_SNAPSHOT" ] && ! tar -tzf "$CONDA_SNAPSHOT" >/dev/null 2>&1; then
  echo "Conda snapshot is corrupted -> removing and rebuilding"
  rm -f "$CONDA_SNAPSHOT"
  REBUILD_CONDA_SNAPSHOT=1
fi

if [ ! -f "$CONDA_SNAPSHOT" ]; then
  REBUILD_CONDA_SNAPSHOT=1
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
      rm -f "$CONDA_SNAPSHOT"
      REBUILD_CONDA_SNAPSHOT=1
    fi
  fi

  if [ ! -x "$CONDA_BIN" ] || ! "$CONDA_BIN" --version >/dev/null 2>&1; then
    echo "Conda snapshot missing/broken -> installing Miniconda"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /content/miniconda.sh
    bash /content/miniconda.sh -b -p "$PERSIST_CONDA" -u
    REBUILD_CONDA_SNAPSHOT=1
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
if [ "$REBUILD_CONDA_SNAPSHOT" = "1" ] || [ ! -f "$CONDA_SNAPSHOT" ] || [ "$REFRESH_CONDA_SNAPSHOT" = "1" ]; then
  echo "Creating conda snapshot: $CONDA_SNAPSHOT"
  if ! tar -czf "$CONDA_SNAPSHOT" -C "$PERSIST_CONDA" .; then
    echo "Snapshot creation failed -> continuing without cache update"
  fi
else
  echo "Conda snapshot exists -> skipping snapshot refresh"
fi

# %% [markdown]
# ## 5) Install D3IL (Install Once + Verify)
#
# Uses editable installs for both D3IL core and `gym_avoiding_env`, but skips reinstall when editable links already exist.

# %%
%%bash
set -e

PIP="/content/miniconda3/envs/FMPCC/bin/pip"
REPO="/content/drive/MyDrive/FMPCC/FM-PCC"
D3IL="$REPO/d3il"

if [ ! -d "$D3IL/.git" ]; then
  echo "d3il missing/incomplete -> recloning"
  rm -rf "$D3IL"
  git clone https://github.com/ALRhub/d3il.git "$D3IL"
fi

if "$PIP" freeze | grep -Fq "d3il/environments/d3il" && "$PIP" freeze | grep -Fq "d3il/envs/gym_avoiding_env"; then
  echo "D3IL editable installs already present; skipping reinstall"
else
  "$PIP" install -e "$D3IL/environments/d3il"
  "$PIP" install -e "$D3IL/environments/d3il/envs/gym_avoiding_env"
fi

echo "D3IL installed"

# %% [markdown]
# ## 6) Install Requirements (Install Once + Verify)
#
# Runs validation first and only installs when the environment is missing or inconsistent.

# %%
%%bash
set -e

REPO="/content/drive/MyDrive/FMPCC/FM-PCC"
PIP="/content/miniconda3/envs/FMPCC/bin/pip"
PY="/content/miniconda3/envs/FMPCC/bin/python"
PIP_CACHE="/content/drive/MyDrive/FMPCC/.pip-cache"
CACHE_DIR="/content/drive/MyDrive/FMPCC/cache"
REQ_FILE="$REPO/requirements.txt"
REQ_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
WHEELHOUSE="$CACHE_DIR/wheelhouse_$REQ_HASH"
REQ_STAMP="/content/miniconda3/envs/FMPCC/.fmpcc_requirements_hash"
FORCE_REINSTALL_REQS="${FORCE_REINSTALL_REQS:-0}"
REBUILD_WHEELHOUSE=0

mkdir -p "$PIP_CACHE" "$CACHE_DIR"
cd "$REPO"

if [ ! -d "$WHEELHOUSE" ]; then
  REBUILD_WHEELHOUSE=1
else
  WHEEL_COUNT_EXISTING="$(find "$WHEELHOUSE" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) | wc -l)"
  if [ "$WHEEL_COUNT_EXISTING" -eq 0 ]; then
    REBUILD_WHEELHOUSE=1
  fi
fi

if [ "$FORCE_REINSTALL_REQS" = "1" ]; then
  echo "FORCE_REINSTALL_REQS=1 -> forcing requirements reinstall"
  REQ_VALID=0
elif "$PY" - <<'PY'
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
        print(f'Missing/broken package: {p} ({type(e).__name__}: {e})')

import numpy
if int(numpy.__version__.split('.')[0]) >= 2:
    ok = False
    print(f'Invalid numpy version: {numpy.__version__}; expected 1.x for this workflow')

sys.exit(0 if ok else 2)
PY
then
  REQ_VALID=1
else
  REQ_VALID=0
fi

if [ "$REQ_VALID" = "1" ]; then
  if [ "$REBUILD_WHEELHOUSE" = "1" ]; then
    echo "First run or broken wheelhouse detected -> creating wheel cache"
    rm -rf "$WHEELHOUSE"
    mkdir -p "$WHEELHOUSE"
    PIP_CACHE_DIR="$PIP_CACHE" "$PIP" download -r requirements.txt -d "$WHEELHOUSE" || true
  fi

  if [ -f "$REQ_STAMP" ] && grep -Fxq "$REQ_HASH" "$REQ_STAMP"; then
    echo "Package validation passed and requirements hash unchanged; skipping reinstall"
  else
    echo "Package validation passed but requirements hash stamp missing/changed; updating stamp only"
    echo "$REQ_HASH" > "$REQ_STAMP"
  fi
else
  echo "Package validation failed; installing requirements"

  # Build or reuse a Drive-backed wheelhouse for faster future reinstalls.
  if [ "$REBUILD_WHEELHOUSE" = "1" ] || [ ! -d "$WHEELHOUSE" ] || [ -z "$(ls -A "$WHEELHOUSE" 2>/dev/null || true)" ]; then
    echo "Building wheelhouse cache at: $WHEELHOUSE"
    rm -rf "$WHEELHOUSE"
    mkdir -p "$WHEELHOUSE"
    PIP_CACHE_DIR="$PIP_CACHE" "$PIP" download -r requirements.txt -d "$WHEELHOUSE" || true
  else
    echo "Wheelhouse cache found: $WHEELHOUSE"
  fi

  WHEEL_COUNT="$(find "$WHEELHOUSE" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) | wc -l)"
  if [ "$WHEEL_COUNT" -gt 0 ] && PIP_CACHE_DIR="$PIP_CACHE" "$PIP" install --no-index --find-links "$WHEELHOUSE" -r requirements.txt; then
    echo "Offline wheelhouse install succeeded"
  else
    echo "First run or incomplete wheelhouse -> running online install"
    rm -rf "$WHEELHOUSE"
    mkdir -p "$WHEELHOUSE"
    PIP_CACHE_DIR="$PIP_CACHE" "$PIP" install -r requirements.txt

    # Backfill wheelhouse after a successful online install for faster future runs.
    PIP_CACHE_DIR="$PIP_CACHE" "$PIP" download -r requirements.txt -d "$WHEELHOUSE" || true
  fi

  # pip check may report known non-fatal issues on Colab (platform/extra metadata).
  PIP_CHECK_OUT="$("$PIP" check 2>&1 || true)"
  echo "$PIP_CHECK_OUT"

  UNEXPECTED_PIP_CHECK="$(echo "$PIP_CHECK_OUT" | grep -Ev '(^$|gurobipy .* is not supported on this platform|WARNING: typer .* does not provide the extra .all.)' || true)"
  if [ -n "$UNEXPECTED_PIP_CHECK" ]; then
    echo "Unexpected pip check issues found:"
    echo "$UNEXPECTED_PIP_CHECK"
    exit 1
  else
    echo "Only known non-fatal pip check warnings detected; continuing"
  fi

  echo "$REQ_HASH" > "$REQ_STAMP"
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
# ## 7) Runtime Environment Variables
#
# Includes W&B malformed service cleanup and Colab rendering settings.

# %%
import os

FMPCC = '/content/drive/MyDrive/FMPCC/FM-PCC'
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
# ## 8) Optional W&B Login

# %%
import os
from pathlib import Path
import wandb

KEY_FILE = Path('/content/drive/MyDrive/FMPCC/.wandb_api_key')

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
# ## 9) Full Verification
#
# Validates import chain with the exact env interpreter used for training.

# %%
%%bash
set -e

/content/miniconda3/envs/FMPCC/bin/python - <<'PY'
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
# ## 10) Dataset Preparation (Avoiding)
#
# ### Option A: Use existing zip from old DPCC path
#
# Warning: This searches for the ~15 GB zip in the DPCC path, not in this FMPCC path.
#
#
# This exits quickly if avoiding data already exists.


# %%
%%bash
set -e

REPO="/content/drive/MyDrive/FMPCC/FM-PCC"
AVOIDING_DATA="$REPO/d3il/environments/dataset/data/avoiding/data"
DATA_ZIP="/content/drive/MyDrive/DPCC/dpcc/d3il/environments/dataset/data/dataset.zip"

if [ -d "$AVOIDING_DATA" ] && [ "$(ls -A "$AVOIDING_DATA")" ]; then
  echo "avoiding data already present: $(ls "$AVOIDING_DATA" | wc -l) files"
  exit 0
fi

if [ ! -f "$DATA_ZIP" ]; then
  echo "dataset zip not found: $DATA_ZIP"
  echo "Skip this cell and use Option B below if needed."
  exit 1
fi

TMP="/content/avoiding_tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
unzip -q "$DATA_ZIP" "avoiding/*" -d "$TMP"
mkdir -p "$REPO/d3il/environments/dataset/data/avoiding"
cp -r "$TMP/avoiding/." "$REPO/d3il/environments/dataset/data/avoiding/"
rm -rf "$TMP"

echo "avoiding dataset ready: $(ls "$AVOIDING_DATA" | wc -l) files"

# %% [markdown]
# ### Option B: Download full D3IL dataset zip with gdown (only if Option A unavailable)

# %% [markdown]
# ```bash
# set -e
# 
# REPO="/content/drive/MyDrive/FMPCC/FM-PCC"
# DATA_DIR="$REPO/d3il/environments/dataset/data"
# ZIP_FILE="$DATA_DIR/dataset.zip"
# 
# if [ -f "$ZIP_FILE" ]; then
#   echo "zip already exists: $ZIP_FILE"
#   exit 0
# fi
# 
# /content/miniconda3/envs/FMPCC/bin/pip install gdown -q
# /content/miniconda3/envs/FMPCC/bin/python -m gdown \
#   "https://drive.google.com/uc?id=1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8" \
#   -O "$ZIP_FILE"
# 
# echo "downloaded zip: $ZIP_FILE"
# ```

# %% [markdown]
# ## 11) Smoke Test Train
#
# Short check before full run.

# %%


# %% [markdown]
# ## 12) Full Train
#
# Real-time streaming via `!python`.

# %%
!/content/miniconda3/envs/FMPCC/bin/python scripts/train.py --seeds 6 --num-seeds 1 --use-wandb --wandb-project FMPCC

# %% [markdown]
# ## 13) Resume Training (Optional)

# %%
!/content/miniconda3/envs/FMPCC/bin/python scripts/train.py --seeds 6 --use-wandb --auto-resume --wandb-project FMPCC

# %% [markdown]
# ## 14) Evaluation and Results

# %% [markdown]
# Remember to edit the YAML in /config to choose the seeds and set write_to_file: True.
# %%
!/content/miniconda3/envs/FMPCC/bin/python scripts/eval.py

# %% [markdown]
# ### Load Results
#
# If the process crashes, update the YAML to resume from the crash point.
#
# Save path: logs/avoiding-d3il/plans/H8_K20_Dmodels.GaussianDiffusion/0/results/

# %%
!/content/miniconda3/envs/FMPCC/bin/python scripts/load_results.py

# %% [markdown]
# ## 15) Visualization

# %%
!/content/miniconda3/envs/FMPCC/bin/python scripts/visualize_data_constraints.py

# %% [markdown]
# ## 16) Archive Logs to Drive

# %%
%%bash
set -e

REPO="/content/drive/MyDrive/FMPCC/FM-PCC"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/content/drive/MyDrive/FMPCC/runs_snapshot/$STAMP"

mkdir -p "$OUT"
if [ -d "$REPO/logs" ]; then
  cp -r "$REPO/logs" "$OUT/"
fi
if [ -d "$REPO/wandb" ]; then
  cp -r "$REPO/wandb" "$OUT/"
fi

echo "snapshot saved: $OUT"
