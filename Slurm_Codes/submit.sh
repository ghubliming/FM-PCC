#!/bin/bash

# FM-PCC SLURM Submission Wrapper
# Usage: ./submit.sh Slurm_Codes/sbatch/your_script.sh

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_slurm_script>"
    exit 1
fi

SCRIPT_PATH=$1
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
JOB_NAME="${SCRIPT_NAME%.*}"

# 1. Generate Date and Time
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H%M%S)

# 2. Create the Date-based Log Directory
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"

# 3. Define the Log Path
# We use %x (job name) and %j (job id) provided by SLURM
# and the $TIME from our local clock.
LOG_FILE="$LOG_DIR/%x_%j_$TIME.log"

echo "------------------------------------------------"
echo "🚀 Submitting Job: $JOB_NAME"
echo "📅 Date:           $DATE"
echo "📁 Log Directory:  $LOG_DIR"
echo "------------------------------------------------"

# 4. Submit to SLURM
# We override the output and error paths to point to the same file
sbatch --job-name="$JOB_NAME" \
       --output="$LOG_FILE" \
       --error="$LOG_FILE" \
       "$SCRIPT_PATH"
