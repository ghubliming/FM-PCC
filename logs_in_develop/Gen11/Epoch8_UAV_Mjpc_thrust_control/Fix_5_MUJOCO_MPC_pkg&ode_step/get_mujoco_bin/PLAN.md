# Build `agent_server` on Cluster — Safe Plan

**Only needed when `controller='mjpc'`. Current default is `pid_stopgo` — not blocking.**

---

## The One Risk to Avoid

`third_party/mujoco_mpc/mujoco_mpc/` already exists (our bundled Python package).
NEVER clone or run cmake inside `third_party/`. Build source goes to a scratch temp dir;
only the final binary is copied into the repo.

---

## SBATCH Script

Save as `Slurm_Codes/sbatch/uav_fm/build_mjpc_agent_server.sh`:

```bash
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
conda create -n _mjpc_build -c conda-forge cmake c-compiler cxx-compiler ninja -y
conda activate _mjpc_build

# ── Clone to scratch ─────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"
git clone https://github.com/google-deepmind/mujoco_mpc.git "$BUILD_DIR/mujoco_mpc"
cd "$BUILD_DIR/mujoco_mpc"

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir build && cd build
cmake .. -DMJPC_BUILD_GRPC_SERVICE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target agent_server -j${SLURM_CPUS_PER_TASK}

# ── Deploy binary only ────────────────────────────────────────────────────────
cp bin/agent_server "$TARGET"
chmod +x "$TARGET"

# ── Cleanup scratch + temp env ───────────────────────────────────────────────
cd "$REPO"
conda deactivate
conda env remove -n _mjpc_build -y
rm -rf "$BUILD_DIR"

echo "========================================================"
echo "BUILD COMPLETE: $(date)"
echo "Binary at: $TARGET"
echo "========================================================"
```

## Submit

```bash
sbatch Slurm_Codes/sbatch/uav_fm/build_mjpc_agent_server.sh
```

## Monitor

```bash
tail -f Slurm_Codes/logs/build_mjpc_<JOB_ID>.log
```

## Verify

```bash
ls -lh third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server
file third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server   # should say ELF 64-bit
```

---

## After Build Finishes — What To Do

### 1. Confirm binary is real
```bash
ls -lh third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server
# expect: ~50–150 MB ELF binary
file third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server
# expect: ELF 64-bit LSB executable, x86-64
```

### 2. Switch config to mjpc controller
In `config/uav.py` plan block:
```python
'controller': 'mjpc',
```

### 3. Run eval
```bash
sbatch Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh pillars 6
```

---

## What To Expect If Binary Works

When `controller='mjpc'` and the binary is in place:

- Log will print: `[ MJPCTracker ] ...` on init — no `RuntimeError` about missing binary
- `agent_server` spawns as a background gRPC subprocess (visible in `ps aux` during the job)
- Each FM waypoint is handed to MJPC which optimizes thrust over a short horizon
- The eval output folder will be `K20_mpc4_mjpc_T0.5/` (controller name in path)
- Success rate on pillars should be higher than `pid_stopgo` if MJPC tracking is tighter

If the binary is wrong architecture or corrupt:
- Python will throw `OSError` or `subprocess.CalledProcessError` when spawning `agent_server`
- No data loss — just a clear error at the start of the first rollout

---

## Notes

- Build takes ~1–3 h (gRPC downloads and compiles ~200 MB of C++ from source via CMake FetchContent)
- `third_party/` is never touched during build — only the final binary is written there
- The binary is NOT committed to git (too large, arch-specific) — must be rebuilt after cluster wipe
