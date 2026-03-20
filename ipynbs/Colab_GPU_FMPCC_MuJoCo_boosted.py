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

mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d "$REPO/.git" ]; then
  git clone --recurse-submodules https://github.com/ghubliming/FM-PCC.git
else
  cd "$REPO"
  git pull --ff-only
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

if [ ! -x "/content/miniconda3/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /content/miniconda.sh
  bash /content/miniconda.sh -b -p /content/miniconda3 -u
fi

/content/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/content/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if ! /content/miniconda3/bin/conda env list | grep -q "^FMPCC "; then
  /content/miniconda3/bin/conda create -n FMPCC python=3.10 -y -q
fi

/content/miniconda3/envs/FMPCC/bin/python -V
/content/miniconda3/envs/FMPCC/bin/pip --version

# %% [markdown]
# ## 5) Install D3IL (Critical Original Logic)
#
# Uses editable installs for both D3IL core and `gym_avoiding_env`.

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

"$PIP" install -e "$D3IL/environments/d3il"
"$PIP" install -e "$D3IL/environments/d3il/envs/gym_avoiding_env"

echo "D3IL installed"

# %% [markdown]
# ## 6) Install Requirements (Stable, Pinned)
#
# Adds stamp-based skip logic and persistent pip cache to avoid reinstalling on every restart.

# %%
%%bash
set -e

REPO="/content/drive/MyDrive/FMPCC/FM-PCC"
PIP="/content/miniconda3/envs/FMPCC/bin/pip"
PY="/content/miniconda3/envs/FMPCC/bin/python"
PIP_CACHE="/content/drive/MyDrive/FMPCC/.pip-cache"
STAMP="/content/drive/MyDrive/FMPCC/.requirements.sha256"

mkdir -p "$PIP_CACHE"
cd "$REPO"

REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$REQ_HASH" ]; then
  echo "requirements hash unchanged; running import check and skipping reinstall"
  "$PY" - <<'PY'
import importlib
pkgs = [
    'torch', 'numpy', 'scipy', 'gym', 'gymnasium', 'gymnasium_robotics',
    'minari', 'wandb', 'mujoco', 'diffusers', 'transformers'
]
for p in pkgs:
    importlib.import_module(p)
print('Import check passed, reuse environment')
PY
else
  echo "requirements changed or stamp missing; installing dependencies"
  PIP_CACHE_DIR="$PIP_CACHE" "$PIP" install -r requirements.txt
  echo "$REQ_HASH" > "$STAMP"
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
# Keeps your original logic. This exits quickly if avoiding data already exists.

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

# %%
%%bash
set -e

REPO="/content/drive/MyDrive/FMPCC/FM-PCC"
DATA_DIR="$REPO/d3il/environments/dataset/data"
ZIP_FILE="$DATA_DIR/dataset.zip"

if [ -f "$ZIP_FILE" ]; then
  echo "zip already exists: $ZIP_FILE"
  exit 0
fi

/content/miniconda3/envs/FMPCC/bin/pip install gdown -q
/content/miniconda3/envs/FMPCC/bin/python -m gdown \
  "https://drive.google.com/uc?id=1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8" \
  -O "$ZIP_FILE"

echo "downloaded zip: $ZIP_FILE"

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
# Remember to edit the yaml in config to choose seeds

# %%
!/content/miniconda3/envs/FMPCC/bin/python scripts/eval.py
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
