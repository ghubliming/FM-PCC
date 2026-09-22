#!/bin/bash
#SBATCH --job-name=u18_turbo_gif    # Gen15 U18 — Mode T smoke run WITH overhead GIFs (needs a GPU for the EGL context)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student

# Same job as turbo.sh, plus a GPU so MuJoCo can render. Everything else (MODE, GO, REPLAYS, SCALE, ...) is the
# same environment interface; GIF defaults to 3 episodes per cell here (0 = no GIF, use turbo.sh instead).
#     GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo_gif.sh            # pilot + GIFs
#     GO=1 GIF=5 REPLAYS=settle ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo_gif.sh
# GIFs: <out-root>/…/<seed>/results/halfspace_<geo>/diagnostics/<variant>/rollout_<i>.gif (160 px, camera 8 m above
# the drone looking straight down, 5 sim-fps, capped at 900 frames; the frame text = step, z, |v|, tilt, CONTACT).
export GIF="${GIF:-3}"
exec bash "$SLURM_SUBMIT_DIR/Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh"
