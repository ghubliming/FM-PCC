#!/bin/bash
#SBATCH --job-name=build_mjpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-1-student
#SBATCH --output=Slurm_Codes/logs/build_mjpc_%j.log
set -e

FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
TARGET="$REPO/third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server"

# Build source goes to scratch — NEVER inside third_party/
BUILD_DIR="/tmp/mjpc_build_${SLURM_JOB_ID}"

echo "========================================================"
echo "BUILD START: $(date)  |  Job $SLURM_JOB_ID  |  $(hostname)"
echo "Build dir : $BUILD_DIR"
echo "Binary target: $TARGET"
echo "========================================================"

# ── Isolated build env (never touches FMPCC env) ─────────────────────────────
CONDA_DIR="$HOME/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda create -n _mjpc_build -c conda-forge \
    cmake c-compiler cxx-compiler ninja zlib \
    xorg-libx11 xorg-libxinerama xorg-libxcursor xorg-libxrandr xorg-libxi xorg-libxxf86vm \
    xorg-xproto xorg-randrproto xorg-xineramaproto xorg-inputproto \
    xorg-xf86vidmodeproto xorg-renderproto xorg-fixesproto xorg-kbproto \
    mesalib \
    -y
conda activate _mjpc_build

# ── Clone to scratch ─────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"
git clone https://github.com/google-deepmind/mujoco_mpc.git "$BUILD_DIR/mujoco_mpc"
cd "$BUILD_DIR/mujoco_mpc"

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir build && cd build
cmake .. -DMJPC_BUILD_GRPC_SERVICE=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DZLIB_ROOT="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DMUJOCO_BUILD_SIMULATE=OFF \
    -DMUJOCO_BUILD_EXAMPLES=OFF
cmake --build . --target agent_server -j${SLURM_CPUS_PER_TASK}

# ── Deploy binary only ────────────────────────────────────────────────────────
cp bin/agent_server "$TARGET"
chmod +x "$TARGET"

# ── Cleanup scratch + temp env ────────────────────────────────────────────────
cd "$REPO"
conda deactivate
conda env remove -n _mjpc_build -y
rm -rf "$BUILD_DIR"

echo "========================================================"
echo "BUILD COMPLETE: $(date)"
echo "Binary at: $TARGET"
echo "========================================================"
