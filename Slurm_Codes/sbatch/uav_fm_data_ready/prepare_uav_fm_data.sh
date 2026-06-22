#!/bin/bash
#SBATCH --job-name=uav_fm_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 Aux-1 — post-collect data prep, batched (no login-node steps).
# Runs AFTER the E4 collect jobs have produced logs/uav_expert_data/<scene>/.
# Curates raw → flat data/uav_fm/v1/, then verifies the manifest so a bad/empty
# dataset fails HERE (cheap) instead of mid-training.
#
# Usage (submit from repo root via submit.sh):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/prepare_uav_fm_data.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/prepare_uav_fm_data.sh "empty corridor s_curve pillars" data/uav_fm/v1
#
# Args:
#   $1 = scenes (space-separated, quoted)  [default: "empty corridor s_curve pillars"]
#   $2 = out dir                            [default: data/uav_fm/v1]
#   $3 = pillars-src (raw dir override)     [optional]
set -e

SCENES="${1:-empty corridor s_curve pillars}"
OUT="${2:-data/uav_fm/v1}"
PILLARS_SRC="${3:-}"

echo "================================================================================"
echo "UAV-FM DATA PREPARE (Aux-1)  $(date)  | JOB $SLURM_JOB_ID | NODE $(hostname)"
echo "scenes=[$SCENES]  out=$OUT  pillars_src=${PILLARS_SRC:-<default>}"
echo "================================================================================"
function on_exit { echo "JOB END: $(date)"; }
trap on_exit EXIT

# Resolve repo root (same marker logic as collect.sh).
MARKER="d3il/environments/d3il/models/mj/robot/quadrotor/scenes"
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO="$SLURM_SUBMIT_DIR"
    while [ "$REPO" != "/" ] && [ ! -d "$REPO/$MARKER" ]; do REPO="$(dirname "$REPO")"; done
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
[ -d "$REPO/$MARKER" ] || { echo "[ prepare ] ERROR: repo root not found."; exit 1; }
cd "$REPO"
echo "[ prepare ] Repo: $REPO"

source activate FMPCC 2>/dev/null || conda activate FMPCC

# 1) Curate raw E4 tree → flat data/uav_fm/v1/<scene>/*.pkl (+ manifest.json).
PILLARS_ARG=""
[ -n "$PILLARS_SRC" ] && PILLARS_ARG="--pillars-src $PILLARS_SRC"
echo "[ prepare ] curating …"
python uav_expert_data_collect/curate_dataset.py --scenes $SCENES --out "$OUT" $PILLARS_ARG

# 2) Verify: manifest exists and every scene has > 0 flat episodes (fail loudly otherwise).
echo "[ prepare ] verifying $OUT …"
python - "$OUT" $SCENES <<'PY'
import json, os, sys, glob
out = sys.argv[1]; scenes = sys.argv[2:]
mani = os.path.join(out, 'manifest.json')
assert os.path.exists(mani), f'NOT READY: missing {mani}'
m = json.load(open(mani))
bad = []
for s in scenes:
    n = len(glob.glob(os.path.join(out, s, '*.pkl')))
    rec = m.get('scenes', {}).get(s, {}).get('n_episodes', 0)
    flag = 'OK' if n > 0 else 'EMPTY'
    if n == 0: bad.append(s)
    print(f'  {s:9s} flat_pkls={n:4d}  manifest={rec:4d}  [{flag}]')
if bad:
    print(f'NOT READY: scenes with zero episodes: {bad}'); sys.exit(2)
print('DATASET READY ✓  → submit Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh (per-scene, see U2)')
PY

echo "[ prepare ] done."
