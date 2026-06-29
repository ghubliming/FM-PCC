# MuJoCo MPC Build Plan for SLURM

Because building the MuJoCo MPC `agent_server` with gRPC can be resource-intensive, submitting it as a SLURM job is the smartest way to do it. It will run in the background on a compute node, utilize multiple CPU cores for a fast compile, and avoid locking up the login node.

This plan uses the **Isolated Conda Build** strategy to prevent any changes to your cluster's module environment.

## The Strategy
1. **Submit an SBATCH script** that requests 8 CPUs and 16GB of RAM.
2. **Create a temporary Conda environment** (`temp_mjpc_build`) populated with `cmake`, `gcc/g++` from `conda-forge`, and `ninja`.
3. **Download and Build** the `agent_server` binary from source using the temporary compilers.
4. **Deploy** the compiled binary directly into your `FM-PCC` repository at the correct location.
5. **Auto-Cleanup** by automatically deleting the temporary Conda environment when the build finishes.

---

## Step 1: Create the SBATCH Build Script
Create a new file in your cluster workspace named `build_mjpc.sh` (e.g. in your `Slurm_Codes/sbatch/` folder) and paste the following code:

```bash
#!/bin/bash
#SBATCH --job-name=build_mjpc
#SBATCH --output=build_mjpc_%j.log
#SBATCH --error=build_mjpc_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --partition=compute  # <-- Adjust this to your cluster's partition name!

# NOTE: Adjust the PROJECT_ROOT to the absolute path of your FM-PCC repo on the cluster
PROJECT_ROOT="/path/to/your/FM-PCC"
MUJOCO_MPC_DIR="$PROJECT_ROOT/third_party/mujoco_mpc/mujoco_mpc"
TARGET_BIN_DIR="$MUJOCO_MPC_DIR/mjpc"

echo "=========================================="
echo "Starting MuJoCo MPC SLURM Build"
echo "Time: $(date)"
echo "=========================================="

# 1. Initialize Conda (Assuming Conda is set up in your ~/.bashrc)
source ~/.bashrc

# 2. Create the temporary build environment
echo "[1/5] Creating temporary Conda build environment..."
conda create -n temp_mjpc_build -c conda-forge cmake c-compiler cxx-compiler ninja -y

# 3. Activate the environment
# Note: In shell scripts, 'conda activate' sometimes requires 'source activate'
source activate temp_mjpc_build

# 4. Clone mujoco_mpc if not already present
echo "[2/5] Preparing source directory..."
mkdir -p "$PROJECT_ROOT/third_party/mujoco_mpc"
cd "$PROJECT_ROOT/third_party/mujoco_mpc"

if [ ! -d "mujoco_mpc" ]; then
    echo "Cloning mujoco_mpc repository..."
    git clone https://github.com/google-deepmind/mujoco_mpc.git
fi

cd mujoco_mpc

# 5. Build the agent_server
echo "[3/5] Configuring CMake and building..."
# Clean previous build if it exists
rm -rf build
mkdir build && cd build

# Configure CMake with gRPC enabled, using Ninja for fast compilation
cmake .. -DMJPC_BUILD_GRPC_SERVICE=ON -DCMAKE_BUILD_TYPE=Release -G Ninja

# Compile using all allocated SBATCH cores
cmake --build . --target agent_server -j$SLURM_CPUS_PER_TASK

# 6. Deploy the Binary
echo "[4/5] Deploying binary to target location..."
mkdir -p "$TARGET_BIN_DIR"
cp bin/agent_server "$TARGET_BIN_DIR/agent_server"
chmod +x "$TARGET_BIN_DIR/agent_server"

# 7. Clean up
echo "[5/5] Cleaning up..."
cd "$PROJECT_ROOT"
source deactivate
conda env remove -n temp_mjpc_build -y

echo "=========================================="
echo "Build Complete! Binary deployed to:"
echo "$TARGET_BIN_DIR/agent_server"
echo "Time: $(date)"
echo "=========================================="
```

## Step 2: Customize the Script
Before submitting, you must change two variables at the top of the script:
1. `PROJECT_ROOT`: Change this to the absolute path of your `FM-PCC` directory on the SLURM cluster.
2. `#SBATCH --partition=compute`: Change `compute` to whatever partition/queue you normally use for standard CPU jobs.

## Step 3: Submit the Job
Once the script is saved and customized on the cluster, submit it using:
```bash
sbatch build_mjpc.sh
```

## Step 4: Monitor the Build
You can monitor the progress by looking at the generated log file:
```bash
tail -f build_mjpc_<JOB_ID>.log
```
The job will take approximately 10 to 15 minutes, primarily because it has to download and statically link the large `gRPC` and `MuJoCo` libraries. Once it says "Build Complete!", you are ready to run your evaluations!
