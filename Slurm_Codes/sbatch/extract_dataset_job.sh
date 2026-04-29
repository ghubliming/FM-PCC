#!/bin/bash
#SBATCH --job-name=extract_data      # Job name
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=16          # Heavy CPU for fast unzipping
#SBATCH --mem=32G                    # Enough RAM for file caching
#SBATCH --time=02:00:00             # 2 hour limit (usually takes 15-30 mins)
#SBATCH --partition=gpu-1-student   # Use a powerful node

# Exit on error
set -e

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
TARGET_TASK="avoiding"   # <-- CHANGE THIS: avoiding / aligning / stacking / etc.

# Paths
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
DATASET_STORAGE="$FMPCC_ROOT/d3il_data" # Storage for the 15GB ZIP
ZIP_FILE="$DATASET_STORAGE/dataset.zip"
TARGET_DATA_ROOT="$REPO/d3il/environments/dataset/data"

# Conda Setup
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

# ------------------------------------------------------------------------------
# PRO-LOGGING SETUP
# ------------------------------------------------------------------------------
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "================================================================================"
echo "DATASET EXTRACTION START: $(date)"
echo "TARGET FOLDER:           $TARGET_TASK"
echo "ZIP SOURCE:              $ZIP_FILE"
echo "DESTINATION:             $TARGET_DATA_ROOT/$TARGET_TASK"
echo "================================================================================"

# 1) Initialize Conda (needed for gdown)
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# 2) Check for dataset.zip
if [ ! -f "$ZIP_FILE" ]; then
    echo "✗ ERROR: Zip file not found at: $ZIP_FILE"
    echo "  Please ensure 'dataset.zip' is placed in $DATASET_STORAGE before running this job."
    exit 1
else
    echo "✅ Zip file found: $(du -sh $ZIP_FILE | cut -f1)"
fi

# 3) Check if already extracted
FINAL_PATH="$TARGET_DATA_ROOT/$TARGET_TASK/data"
if [ -d "$FINAL_PATH" ] && [ "$(ls -A $FINAL_PATH)" ]; then
    echo "✅ Dataset already extracted at: $FINAL_PATH"
    echo "   File count: $(ls $FINAL_PATH | wc -l)"
    echo "   Skipping extraction."
    exit 0
fi

# 4) Extract ONLY the specific task (using Python because 'unzip' is missing)
echo "🚀 Extracting '$TARGET_TASK/' from ZIP (Python zipfile mode)..."
mkdir -p "$TARGET_DATA_ROOT"

python -c "
import zipfile, os
zip_path = '$ZIP_FILE'
target_dir = '$TARGET_DATA_ROOT'
prefix = '$TARGET_TASK/'
print(f'Opening {zip_path}...')
with zipfile.ZipFile(zip_path, 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith(prefix)]
    print(f'Extracting {len(members)} files to {target_dir}...')
    zf.extractall(target_dir, members)
"

echo "✅ Extraction complete!"

# 5) Verification
echo "------------------------------------------------"
echo "VERIFICATION"
echo "------------------------------------------------"
if [ -d "$FINAL_PATH" ]; then
    echo "✓ Folder found: $FINAL_PATH"
    echo "✓ Files:        $(find $FINAL_PATH -maxdepth 1 | wc -l)"
    echo "✓ Sample:       $(ls $FINAL_PATH | head -3)"
else
    echo "✗ ERROR: Extraction failed or folder structure unexpected."
    echo "Actual folders in target data root:"
    ls -R "$TARGET_DATA_ROOT" | head -20
    exit 1
fi

echo "================================================================================"
echo "FINISHED AT: $(date)"
echo "================================================================================"
