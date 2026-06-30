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
    cmake c-compiler cxx-compiler ninja zlib patchelf \
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

# ── SMOKE TEST 1: run the binary in its NATIVE build env (conda active) ───────
# This is the "golden" environment — exactly what it was compiled against.
# If it segfaults HERE, the build itself is broken (not a deployment problem).
echo "════════════════════════════════════════════════════════════════════════"
echo "[ SMOKE-1 ] Run freshly-built binary inside conda env (golden environment)"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
set +e
timeout 5 ./bin/agent_server --mjpc_port=19996
echo "[ SMOKE-1 ] exit code = $?  (124=timeout=GOOD: it stayed up; 139=SIGSEGV=build broken)"
set -e
echo "════════════════════════════════════════════════════════════════════════"

# ── Deploy binary + ALL conda runtime libs (env is deleted after this step) ──
LIB_DIR="$(dirname "$TARGET")"
cp bin/agent_server "$TARGET"
chmod +x "$TARGET"

# libmujoco itself
find . -name "libmujoco.so*" ! -type l | xargs -I{} cp {} "$LIB_DIR/"

# GCC runtime libs — conda puts these in an arch-specific subdir, NOT $CONDA_PREFIX/lib/,
# so ldd alone misses them. Copy explicitly.
find "$CONDA_PREFIX" -name "libstdc++.so*" ! -type l -exec cp {} "$LIB_DIR/" \; -print
find "$CONDA_PREFIX" -name "libgcc_s.so*"  ! -type l -exec cp {} "$LIB_DIR/" \; -print

# Any other conda libs agent_server links (libGL, libX11, etc. — gone when env removed)
ldd bin/agent_server \
    | awk -v prefix="$CONDA_PREFIX" '$3 ~ prefix {print $3}' \
    | while read lib; do
        [ -f "$lib" ] && cp "$lib" "$LIB_DIR/" && echo "  + $(basename $lib)"
      done

echo "Libs deployed to $LIB_DIR:"
ls "$LIB_DIR"/*.so* 2>/dev/null | xargs -I{} basename {}

# ── Make the binary RELOCATABLE: RPATH=$ORIGIN ───────────────────────────────
# ROOT CAUSE (proven by job 22271 SMOKE-2 ldd): the binary's baked-in DT_RPATH
# points at the build dir (/tmp/mjpc_build_*) + conda env. RPATH is searched
# BEFORE LD_LIBRARY_PATH, so even with LD_LIBRARY_PATH set, libs resolved to the
# build dir — which is deleted after this job → dead paths → SIGSEGV in eval.
# Fix: rewrite RPATH to $ORIGIN so the binary always finds the libs we copied
# next to it, regardless of LD_LIBRARY_PATH or where the build happened.
echo "[ RPATH ] before: $(patchelf --print-rpath "$TARGET")"
patchelf --set-rpath '$ORIGIN' "$TARGET"
patchelf --set-rpath '$ORIGIN' "$LIB_DIR/libmujoco.so.3.2.3"
echo "[ RPATH ] after : $(patchelf --print-rpath "$TARGET")"

# ── Cleanup scratch + temp env (do this BEFORE the smoke test so the test ────
# faithfully reproduces the eval condition: build dir + conda env GONE) ───────
cd "$REPO"
conda deactivate
conda env remove -n _mjpc_build -y
rm -rf "$BUILD_DIR"

# ── SMOKE TEST 2: deployed binary, FULLY self-contained (the eval condition) ─
# Build dir and conda env are now DELETED. No LD_LIBRARY_PATH at all — if the
# $ORIGIN RPATH works, the binary finds its co-located libs on its own.
# 124=timeout=GOOD (stayed up, fix works); 139=SIGSEGV=still broken.
echo "════════════════════════════════════════════════════════════════════════"
echo "[ SMOKE-2 ] Deployed binary, build dir + conda env DELETED, no LD_LIBRARY_PATH"
set +e
env -i HOME="$HOME" PATH="/usr/bin:/bin" timeout 5 "$TARGET" --mjpc_port=19995
echo "[ SMOKE-2 ] exit code = $?  (124=GOOD: self-contained & runs; 139=still broken)"
echo "[ SMOKE-2 ] ldd of deployed binary (should resolve everything next to it):"
env -i ldd "$TARGET" 2>&1
set -e
echo "════════════════════════════════════════════════════════════════════════"

echo "========================================================"
echo "BUILD COMPLETE: $(date)"
echo "Binary at: $TARGET"
echo "========================================================"
