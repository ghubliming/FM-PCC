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
# -- Gen14 U9 -- PERCEPTION-FIRST knobs. Same narrowing shape as MIX_FILM_MODE/MIX_BONE:
# MIX_<KNOB>_<ENGINE> for one arm, bare MIX_<KNOB> for all two-time arms. All three default
# to the pre-U9 value, and at the defaults the checkpoint path is character-for-character
# the U8 one -- nothing existing is orphaned.
#
#   MIX_VIS_PRETRAINED=1     ImageNet init of the dual ResNet-18   (default 0 = random)
#   MIX_VIS_LR_SCALE=0.1     encoder LR = train_lr * scale         (default 1.0; 0.0 = frozen)
#   MIX_VIS_COND=adaln       latent into adaLN's `c` not the seq   (default token = U8)
#
# The headline U9 run (PLAN Gen14/U9 R1):
#   MIX_BONE_MF=mf_dit MIX_VIS_PRETRAINED=1 MIX_VIS_LR_SCALE=0.1 MIX_VIS_COND=adaln \
#     ./Slurm_Codes/submit.sh <this script> mf 6
#
# 🔴 MIX_VIS_PRETRAINED=1 downloads ImageNet weights into ~/.cache/torch/hub/checkpoints/ and
#    COMPUTE NODES HAVE NO INTERNET. Pre-fetch once on the login node:
#      python -c "import torchvision as tv; tv.models.resnet18(pretrained=True)"
#    Gate G-B11 turns a silent fallback to random weights into a loud failure.
eval "VIS_PRETRAINED=\${MIX_VIS_PRETRAINED_${ENGINE_UC}:-\${MIX_VIS_PRETRAINED:-0}}"
eval "VIS_LR_SCALE=\${MIX_VIS_LR_SCALE_${ENGINE_UC}:-\${MIX_VIS_LR_SCALE:-1.0}}"
eval "VIS_COND=\${MIX_VIS_COND_${ENGINE_UC}:-\${MIX_VIS_COND:-token}}"
case "$VIS_PRETRAINED" in 0|1|true|false) ;; *) echo "[ train ] ERROR: MIX_VIS_PRETRAINED='$VIS_PRETRAINED' (want 0|1)"; exit 1 ;; esac
case "$VIS_COND" in token|adaln|both) ;; *) echo "[ train ] ERROR: MIX_VIS_COND='$VIS_COND' (want token|adaln|both)"; exit 1 ;; esac
if [ "$VIS_COND" != "token" ] && [ "$ML_BONE" != "mf_dit" ] && [ "$ML_BONE" != "sit" ]; then
    echo "[ train ] ERROR: MIX_VIS_COND='$VIS_COND' needs an adaLN bone (mf_dit|sit), got '$ML_BONE'."
    echo "[ train ]        The RoPE bones have no adaLN pathway; the knob would be silently ignored."
    exit 1
fi
unset MIX_VIS_PRETRAINED MIX_VIS_LR_SCALE MIX_VIS_COND
export "MIX_VIS_PRETRAINED_${ENGINE_UC}=$VIS_PRETRAINED"
export "MIX_VIS_LR_SCALE_${ENGINE_UC}=$VIS_LR_SCALE"
export "MIX_VIS_COND_${ENGINE_UC}=$VIS_COND"
if [ "$VIS_PRETRAINED" = "0" ] && [ "$VIS_LR_SCALE" = "1.0" ] && [ "$VIS_COND" = "token" ]; then
    echo "[ train ] U9 knobs = ALL DEFAULT (pre-U9 behaviour; path unchanged)"
else
    echo "[ train ] U9: vis_pretrained=$VIS_PRETRAINED  vis_lr_scale=$VIS_LR_SCALE  vis_cond_mode=$VIS_COND"
fi

# ── Gen14 U10 ── alpha-FLOW SCHEDULE (af arm only). The knob that decides whether this
# arm actually trains the alpha-Flow target, or anneals onto MeanFlow's before it finishes.
#
#   MIX_AF_ALPHA_SCHED   constant | step | linear | exponential | log | sigmoid  (default sigmoid)
#   MIX_AF_ALPHA_INIT    alpha at step 0                                          (default 1.0 = pure FM)
#   MIX_AF_ALPHA_END     alpha at the end of the anneal                           (default 0.0 = MeanFlow)
#   MIX_AF_ALPHA_CLAMP   snap-to-endpoint threshold                               (default 0.005)
#   MIX_AF_ALPHA_GAMMA   sigmoid steepness                                        (default 25.0)
#
# 🔴 WHY YOU PROBABLY WANT THIS. At the shipped defaults the sigmoid + clamp force alpha to
# EXACTLY 0 from ~71.2 % of the budget onward, so the last ~28.8 % of training runs the
# MeanFlow target, not alpha-Flow's. Gen14 U5 measured the cost: test raw_mse_u 2.657 at
# step 70 k (alpha 0.0067) -> 8.504 at step 72 k (alpha 0), a 2.9x jump that never recovers.
# See logs_in_develop/Gen14/U5/DA_20260804_mf_af_visual_aligning_first_run.md §3.
#
# 🔴 PATH KEY. A non-default value stamps '_AF<tag>' onto the checkpoint folder, the plans/
# results folder and the eval's diffusion_loadpath (config: _mix_af_alpha_keys). At the
# defaults NOTHING is stamped, so every existing af tree keeps its exact current name.
# This is what stops a re-tuned run from silently overwriting the alpha->0 checkpoint —
# 'af_alpha_scheduler' alone would NOT have, since 'sigmoid' is unchanged by MIX_AF_ALPHA_END.
#
#   # alpha held where U5 found af's best model — the arm alpha-Flow was meant to be:
#   MIX_AF_ALPHA_SCHED=constant MIX_AF_ALPHA_INIT=0.05 MIX_AF_ALPHA_END=0.05 \
#     ./Slurm_Codes/submit.sh <this script> af 6
#
#   # keep upstream's anneal, just never reach 0 (tests "is the clamp the bug?"):
#   MIX_AF_ALPHA_END=0.02 MIX_AF_ALPHA_CLAMP=1e-4 \
#     ./Slurm_Codes/submit.sh <this script> af 6
if [ "$ENGINE" = "af" ]; then
    AF_A_SET=""
    for v in MIX_AF_ALPHA_SCHED MIX_AF_ALPHA_INIT MIX_AF_ALPHA_END MIX_AF_ALPHA_CLAMP MIX_AF_ALPHA_GAMMA; do
        eval "val=\${$v:-}"
        if [ -n "$val" ]; then AF_A_SET="$AF_A_SET $v=$val"; fi
    done
    if [ -n "$AF_A_SET" ]; then
        echo "[ train ] alpha schedule OVERRIDE:$AF_A_SET"
        echo "[ train ]   -> checkpoint + results dirs gain an '_AF<tag>' fragment (config validates and builds it)"
    else
        echo "[ train ] alpha schedule = SHIPPED DEFAULT (sigmoid 1.0 -> 0.0, clamp 0.005)"
        echo "[ train ]   ⚠  alpha snaps to EXACTLY 0 at ~71.2% of the budget: the last ~28.8% of"
        echo "[ train ]      this run trains the MEANFLOW target, not alpha-Flow's. Gen14 U5 §3"
        echo "[ train ]      measured that as a 2.9x jump in test raw_mse_u. Set MIX_AF_ALPHA_*"
        echo "[ train ]      if you meant to train alpha-Flow at its own operating point."
    fi
elif [ -n "${MIX_AF_ALPHA_SCHED:-}${MIX_AF_ALPHA_INIT:-}${MIX_AF_ALPHA_END:-}${MIX_AF_ALPHA_CLAMP:-}${MIX_AF_ALPHA_GAMMA:-}" ]; then
    echo "[ train ] ERROR: MIX_AF_ALPHA_* is set but engine='$ENGINE'. These knobs exist only on the af arm."
    exit 1
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
#                          not a warm restart -- it is the same run. Ignores state_0.pt, and
#                          falls back to state_best.pt when no numbered checkpoint survives.
#   MIX_RESUME_FROM=<x>    explicit resume target: a step number, or the literal 'best'.
#                          Overrides the auto-resume choice. Use 'best' when the periodic
#                          saves are gone -- state_best.pt is a full checkpoint, not just
#                          weights, so it resumes exactly like a numbered one.
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
if [ -n "$MIX_AUTO_RESUME" ]; then echo "[ train ] auto-resume: ON (numbered saves first, then state_best.pt)"; fi
if [ -n "$MIX_RESUME_FROM" ]; then echo "[ train ] resume target = $MIX_RESUME_FROM (explicit)"; fi
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
    ${MIX_RESUME_FROM:+--resume-step $MIX_RESUME_FROM} \
    ${MIX_SAVE_EVERY:+--save-every $MIX_SAVE_EVERY} \
    --use-wandb \
    --wandb-project FM-PCC-visual-aligning-gen14

echo "Job completed successfully."
