#!/bin/bash
#SBATCH --job-name=train_mix_visual_aligning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student

set -e

# Logging setup
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB NAME:  $SLURM_JOB_NAME"
echo "JOB ID:    $SLURM_JOB_ID"
echo "NODE:      $(hostname)"
echo "GPU INFO:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU info available)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"

# Setup Workspace Paths
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$D3IL_ENV_ROOT:$PYTHONPATH"

# Headless rendering
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

# W&B Login
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

cd "$REPO"

# ─── Gen14: pick the ML engine arm ──────────────────────────────────────
# $1 = engine  (diffusion | fm | mf | af)   default: fm  (the Gen7 reference arm)
# $2 = seed(s) (optional)              default: $MIX_SEEDS (6 7 8 9 10)
#
#   sbatch train_mix_visual_aligning.sh mf          # MeanFlow arm, all default seeds
#   sbatch train_mix_visual_aligning.sh af 7        # alpha-Flow arm, seed 7 only
#   sbatch train_mix_visual_aligning.sh mf "6 7"    # two seeds, SEQUENTIALLY in this one job
#
# Each arm writes to its OWN checkpoint tree (mix_visual_aligning_<engine>/...), so the
# four can train concurrently without touching each other or the Gen6V4/Gen7 originals.
#
# ⚠ SEQUENTIAL vs FAN-OUT. Seeds listed here run one after another INSIDE this single job,
#   against a 24 h wall. Visual aligning trains a ResNet-18 pair alongside the U-Net, so one
#   seed at 1e5 steps is already a large fraction of that wall — 5 sequential seeds will not
#   fit. Use mix_visual_aligning_pipeline.sh instead: it submits ONE JOB PER SEED, so each
#   seed gets its own 24 h budget and they run in parallel. Pass a list here only when you
#   deliberately want them serialised (e.g. to hold a single GPU allocation).
ENGINE="${1:-fm}"
SEEDS="${2:-${MIX_SEEDS:-6 7 8 9 10}}"
case "$ENGINE" in
    diffusion|fm|mf|af) ;;
    ddpm) ENGINE=diffusion; echo "[ engine ] NOTE: 'ddpm' is a deprecated alias for 'diffusion' (Gen14 U5)" ;;
    *) echo "[ train ] ERROR: unknown engine '$ENGINE' (want: diffusion | fm | mf | af)"; exit 1 ;;
esac
echo "[ train ] engine=$ENGINE  seeds='$SEEDS'"

# ── Gen14 Fix_9 ── FiLM backbone. Each Gen14 arm has its own knob, MIX_FILM_MODE_<ENGINE>,
# with a bare MIX_FILM_MODE as the all-arms fallback (default v1). Accepted either way:
#   MIX_FILM_MODE_MF=v2 ./Slurm_Codes/submit.sh <this script> mf 6
#   MIX_FILM_MODE=v2    ./Slurm_Codes/submit.sh <this script> mf 6
#
# 🔴 ARCHITECTURE key: v1 and v2 train into separate '..._film{mode}_E<arm>' trees and their
# state_dicts are NOT interchangeable.
#
# NARROWING. This job trains exactly one arm, so the mode is resolved here and re-published
# as the ARM-SPECIFIC variable, then the broadcast form is unset. Without the unset, a bare
# MIX_FILM_MODE inherited through --export=ALL would also be visible to the other three arm
# blocks when the config module imports — harmless today (only $ENGINE's block is consumed)
# but it would make the config's own resolution disagree with what this job is doing, and
# that is exactly the kind of latent disagreement this generation keeps getting bitten by.
ENGINE_UC=$(echo "$ENGINE" | tr '[:lower:]' '[:upper:]')
eval "FILM_MODE=\${MIX_FILM_MODE_${ENGINE_UC}:-\${MIX_FILM_MODE:-v1}}"
case "$FILM_MODE" in
    v1|v2) ;;
    *) echo "[ train ] ERROR: FiLM mode '$FILM_MODE' is not known (want: v1 | v2)"; exit 1 ;;
esac
unset MIX_FILM_MODE
export "MIX_FILM_MODE_${ENGINE_UC}=$FILM_MODE"
echo "[ train ] film_mode = $FILM_MODE  (MIX_FILM_MODE_${ENGINE_UC}; unset -> v1)"

# ── Gen14 U8 ── ML BONE (generative backbone). Same knob shape as MIX_FILM_MODE:
# MIX_BONE_<ENGINE> for one arm, bare MIX_BONE for all two-time arms, default 'unet'.
#   MIX_BONE_MF=mf_dit ./Slurm_Codes/submit.sh <this script> mf 6
#   MIX_BONE=dit       ./Slurm_Codes/submit.sh <this script> mf 6     # moves mf AND af
#
#   unet    VisualUNetTwoTime      — the Gen14 baseline (FiLM v1/v2)
#   mf_dit  official MeanFlow DiT  — mf arm only
#   sit     alpha-Flow SiT         — af arm only
#   dit     iMF RoPE DiT           — both arms
#
# On every transformer bone the 128-D visual latent enters as ONE PREPENDED TOKEN.
# 🔴 ARCHITECTURE + PATH key: a DiT trains into '..._B{bone}_E<arm>' and carries NO
# '_film..' fragment (FiLM is a U-Net concept). state_dicts are NOT interchangeable across
# bones. The U-Net path is unchanged from pre-U8, so existing checkpoints are untouched.
# See logs_in_develop/Gen14/U8/.
#
# Same NARROWING as MIX_FILM_MODE: resolve for THIS arm, re-publish arm-specifically, drop
# the broadcast form so the other arm blocks resolve 'unet' on import.
eval "ML_BONE=\${MIX_BONE_${ENGINE_UC}:-\${MIX_BONE:-unet}}"
case "$ENGINE" in
    mf) VALID_BONES="unet mf_dit dit" ;;
    af) VALID_BONES="unet sit dit" ;;
    *)  VALID_BONES="unet" ;;
esac
if ! echo " $VALID_BONES " | grep -q " $ML_BONE "; then
    echo "[ train ] ERROR: ml_bone '$ML_BONE' is not valid for engine '$ENGINE' (want: $VALID_BONES)"
    exit 1
fi
unset MIX_BONE
export "MIX_BONE_${ENGINE_UC}=$ML_BONE"
if [ "$ML_BONE" = "unet" ]; then
    echo "[ train ] ml_bone = unet (baseline; film_mode=$FILM_MODE applies)"
else
    echo "[ train ] ml_bone = $ML_BONE  -- VisualDiTTwoTime, visual latent as ONE TOKEN;"
    echo "[ train ]           film_mode is N/A on this bone and is absent from the path."
fi
if [ "$(echo $SEEDS | wc -w)" -gt 1 ]; then
    echo "[ train ] WARNING: $(echo $SEEDS | wc -w) seeds will run SEQUENTIALLY in this one job"
    echo "[ train ]          against the 24 h wall. Prefer mix_visual_aligning_pipeline.sh,"
    echo "[ train ]          which submits one job per seed."
fi

# ── Resume + checkpoint cadence + training budget ────────────────────────────
# Three env knobs, all opt-in, all no-ops when unset:
#
#   MIX_AUTO_RESUME=1      pick up the newest state_<step>.pt in the savepath and continue
#                          from it. Restores step, EMA, Adam moments AND the cosine LR
#                          schedule exactly (_restore_optimizer_state), so a requeued run is
#                          not a warm restart -- it is the same run. Ignores state_0.pt.
#   MIX_SAVE_EVERY=<steps> checkpoint cadence. Default n_train_steps // 5 = five saves for a
#                          whole run, which is what let job 24838 lose 84k steps to a wall
#                          -clock kill. On a 24 h visual run use ~5000.
#   MIX_TRAIN_STEPS=<n>    training budget. 🔴 This is a PATH KEY: anything below the full
#                          1e5 stamps '_TB<pct>pct' onto the checkpoint folder so a short run
#                          can never be mistaken for -- or overwrite -- a full one. The EVAL
#                          must see the same value or its diffusion_loadpath will point at
#                          the full-budget directory; the pipeline exports it for you.
#
#   MIX_TRAIN_STEPS=50000 MIX_SAVE_EVERY=5000 ./Slurm_Codes/submit.sh <this script> mf 6
if [ -n "$MIX_AUTO_RESUME" ]; then echo "[ train ] auto-resume: ON"; fi
if [ -n "$MIX_SAVE_EVERY" ]; then echo "[ train ] save_freq  = $MIX_SAVE_EVERY steps"; fi
if [ -n "$MIX_TRAIN_STEPS" ]; then
    echo "[ train ] budget     = $MIX_TRAIN_STEPS steps (PATH KEY -> _TB<pct>pct suffix)"
fi

# $SEEDS is intentionally unquoted: it must word-split into separate --seeds arguments.
# The ${VAR:+...} forms vanish entirely when the variable is unset.
python mix_visual_aligning_test/train_mix_visual_aligning.py \
    --engine "$ENGINE" \
    --seeds $SEEDS \
    ${MIX_AUTO_RESUME:+--auto-resume} \
    ${MIX_SAVE_EVERY:+--save-every $MIX_SAVE_EVERY} \
    --use-wandb \
    --wandb-project FM-PCC-visual-aligning-gen14

echo "Job completed successfully."
