#!/bin/bash

# FM-PCC SLURM Dependent Submission Wrapper
# Usage: ./submit_after.sh <dependency_job_id> <path_to_slurm_script> [script_arg1] ...

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dependency_job_id> <path_to_slurm_script> [script_arg1] [script_arg2] ..."
    echo "Example: $0 24069 Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh"
    exit 1
fi

DEPENDENCY_JOB_ID=$1
shift
SCRIPT_PATH=$1
shift          # remaining args ($2, $3, ...) are passed to the job script
SCRIPT_ARGS=("$@")
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
echo "🚀 Submitting Dependent Job: $JOB_NAME"
echo "⏳ Waiting for Job ID: $DEPENDENCY_JOB_ID (afterok)"
echo "📅 Date:               $DATE"
echo "📁 Log Directory:      $LOG_DIR"
echo "------------------------------------------------"

# 4. Submit to SLURM with dependency
# We use --parsable to easily capture the Job ID
# We export SUBMIT_TIME and SUBMIT_DATE to unify logs for pipelines
SBATCH_OUT=$(sbatch --parsable \
       --dependency="afterok:$DEPENDENCY_JOB_ID" \
       --job-name="$JOB_NAME" \
       --output="$LOG_FILE" \
       --error="$LOG_FILE" \
       --export=ALL,SUBMIT_TIME=$TIME,SUBMIT_DATE=$DATE \
       "$SCRIPT_PATH" "${SCRIPT_ARGS[@]}")

if [ $? -eq 0 ]; then
    JOB_ID=$SBATCH_OUT
    echo "✅ Submission Successful!"
    echo "🆔 Job ID:             $JOB_ID"
    echo "------------------------------------------------"
else
    echo "❌ Submission Failed!"
    exit 1
fi
