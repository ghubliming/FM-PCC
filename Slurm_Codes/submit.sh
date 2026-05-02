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
TIME=$(date +%H_%M_%S)

# 2. Create the Date-based Log Directory
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"

# 3. Define the Log Path
# Convention: HH_MM_SS_JOBNAME_JOBID.log
LOG_FILE="$LOG_DIR/${TIME}_%x_%j.log"

echo "------------------------------------------------"
echo "🚀 Submitting Job: $JOB_NAME"
echo "📅 Date:           $DATE"
echo "📁 Log Directory:  $LOG_DIR"
echo "------------------------------------------------"

# 4. Submit to SLURM
# We use --parsable to easily capture the Job ID
# We export SUBMIT_TIME and SUBMIT_DATE to unify logs for pipelines
SBATCH_OUT=$(sbatch --parsable \
       --job-name="$JOB_NAME" \
       --output="$LOG_FILE" \
       --error="$LOG_FILE" \
       --export=ALL,SUBMIT_TIME=$TIME,SUBMIT_DATE=$DATE \
       "$SCRIPT_PATH")

if [ $? -eq 0 ]; then
    JOB_ID=$SBATCH_OUT
    echo "✅ Submission Successful!"
    echo "🆔 Job ID:         $JOB_ID"
    echo "------------------------------------------------"
else
    echo "❌ Submission Failed!"
    exit 1
fi
