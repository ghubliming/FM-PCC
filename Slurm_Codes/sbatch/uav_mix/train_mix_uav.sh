#!/bin/bash
#SBATCH --job-name=uav_mix_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
set -e

# ---- Args: $1=engine (fm|mf|af, def fm)  $2=scene (all|empty|corridor|s_curve|pillars, def all)
#            $3=seeds (quoted, space-sep, def "6") ----
# Gen15: `engine` is the ML objective — fm (Gen11 FM) | mf (Gen3v6 MeanFlow) |
# af (Gen3v7 alpha-Flow) | diffusion (U3: the DPCC DDPM baseline). It selects the config block
# AND the checkpoint tree, so the arms never collide. Seeds loop INSIDE one job allocation —
# never one sbatch job per seed. If you add seeds, bump --time proportionally.
ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af|diffusion) ;; *) echo "[ ERROR ] engine must be fm|mf|af|diffusion (got '$ENGINE')"; exit 1 ;; esac
SCENE="${2:-all}"
# Default single seed=6 for testing. For the full multi-seed run pass "6 7 8 9 10".
SEEDS="${3:-6}"

# ── Gen15 U6 (2026-09-03) — the alpha-Flow arm's three knobs ──────────────────────────────
#   UAV_MIX_BONE_AF        unet (DEFAULT since U6) | sit | dit
#                          🔴 THE DEFAULT CHANGED. It was a hard 'sit' (~9.4 M), which is NOT
#                          parameter-matched to the fm/mf 4.0 M U-Net — so every pre-U6 af row
#                          moves objective, backbone and param count together. 'sit' is kept,
#                          not deleted: pass UAV_MIX_BONE_AF=sit to reach it.
#                          CHECKPOINT-PATH KEY ('_bb<val>'): each bone has its own tree, so
#                          nothing is overwritten — but a default af EVAL will fail on a
#                          missing checkpoint until the U-Net arm has been TRAINED.
#   UAV_MIX_AF_ALPHA_END   terminal alpha (default 0.0). At 0.0 the sigmoid+clamp snap alpha to
#                          EXACTLY 0 from ~71.2% of the budget on and af_diffusion.py:568 runs
#                          Gen3v6's MeanFlow target — i.e. the arm DEPLOYS A MEANFLOW MODEL.
#                          >0 floors alpha so the bootstrap trains the final weights.
#                          CHECKPOINT-PATH KEY ('_ae<val>').
#   UAV_MIX_EPOCH          best (default) | latest | <step>.  EVAL-ONLY, no retrain.
#                          RESULTS-PATH KEY ('_EP<sel>' in the eval-params folder).
#                          🔴 On the af arm 'best' is chosen on an alpha-weighted test_loss and
#                          therefore prefers a MID-CURRICULUM checkpoint: pairing
#                          UAV_MIX_AF_ALPHA_END with 'best' floors alpha and then discards the
#                          model the floor produced. Use 'latest'.
export UAV_MIX_BONE_AF="${UAV_MIX_BONE_AF:-}"
export UAV_MIX_AF_ALPHA_END="${UAV_MIX_AF_ALPHA_END:-}"
export UAV_MIX_EPOCH="${UAV_MIX_EPOCH:-}"
if [ -n "$UAV_MIX_BONE_AF" ]; then
    case "$UAV_MIX_BONE_AF" in
        unet|sit|dit) ;;
        *) echo "[ ERROR ] UAV_MIX_BONE_AF='$UAV_MIX_BONE_AF' must be unet|sit|dit"
           echo "          ('mf_dit' belongs to the mf arm -- a different class.)"; exit 1 ;;
    esac
    if [ "$ENGINE" != "af" ]; then
        echo "[ ERROR ] UAV_MIX_BONE_AF is set but engine='$ENGINE'. It applies to the af arm only."
        exit 1
    fi
fi
if [ -n "$UAV_MIX_AF_ALPHA_END" ] && [ "$ENGINE" != "af" ]; then
    echo "[ ERROR ] UAV_MIX_AF_ALPHA_END is set but engine='$ENGINE'. af arm only."; exit 1
fi
if [ -n "$UAV_MIX_EPOCH" ]; then
    case "$UAV_MIX_EPOCH" in
        best|latest) ;;
        ''|*[!0-9]*) echo "[ ERROR ] UAV_MIX_EPOCH='$UAV_MIX_EPOCH' must be best|latest|<step>"; exit 1 ;;
    esac
fi
if [ "$ENGINE" = "af" ]; then
    echo "[ U6 ] af bone      = ${UAV_MIX_BONE_AF:-unet (U6 default; was sit)}"
    echo "[ U6 ] af_alpha_end = ${UAV_MIX_AF_ALPHA_END:-0.0}  $([ -z "$UAV_MIX_AF_ALPHA_END" ] && echo '⚠ ends on the MeanFlow target -- set >0 to train alpha-Flow proper')"
fi
echo "[ U6 ] checkpoint   = ${UAV_MIX_EPOCH:-best (default; no _EP fragment)}"

# ---- Pro-logging ----
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)  |  $SLURM_JOB_NAME  |  ID $SLURM_JOB_ID  |  NODE $(hostname)"
echo "ENGINE: $ENGINE   SCENE: $SCENE   SEEDS: $SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"
function on_exit { echo "JOB END: $(date)"; }
trap on_exit EXIT

# ---- Environment ----
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# UAV FM needs only the repo on the path — NO D3IL (the UAV data loader has no d3il dep).
export FMPCC="$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"
# Headless MuJoCo (not used by train, harmless; kept for parity with eval).
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# GPU-leak guard (same as all FMPCC jobs): pin EGL to the Slurm-allocated GPU and abort if they diverge.
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

# W&B Login (key file on the cluster) — enables --use-wandb below.
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

cd "$REPO"
for seed in $SEEDS; do
    echo "--------------------------------------------------------------------------------"
    echo "[ uav_mix_train ] engine=$ENGINE scene=$SCENE seed=$seed  $(date)"
    echo "[ uav_mix_train ] python mix_uav_test/train_mix_uav.py --engine $ENGINE --scene $SCENE --seed $seed --use-wandb --wandb-project FM-PCC-uav-mix --wandb-group uav-$ENGINE-$SCENE"
    python mix_uav_test/train_mix_uav.py --engine "$ENGINE" --scene "$SCENE" --seed "$seed" \
        --use-wandb --wandb-project FM-PCC-uav-mix --wandb-group "uav-$ENGINE-$SCENE"
done
echo "Job completed successfully. Trained engine=$ENGINE scene=$SCENE for seeds=[$SEEDS]"
