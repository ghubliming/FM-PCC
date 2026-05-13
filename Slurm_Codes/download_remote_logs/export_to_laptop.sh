#!/bin/bash
#SBATCH --job-name=export_logs
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=export_logs_%j.out

# ==============================================================================
# CONFIGURATION - PLEASE EDIT THESE BEFORE RUNNING
# ==============================================================================
LAPTOP_USER="your_username"        # Your laptop's local username
LAPTOP_IP="your.laptop.ip.here"    # Your laptop's IP address
DEST_PATH="~/Downloads/FMPCC_Logs" # Target folder on your laptop
# ==============================================================================

REPO_ROOT="$HOME/FMPCC/FM-PCC"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="fmpcc_logs_${TIMESTAMP}.tar.gz"

echo "[ cluster ] Moving to: $REPO_ROOT"
cd "$REPO_ROOT"

# Ensure destination exists on server for the temporary zip
mkdir -p export_tmp

echo "[ cluster ] Archiving logs... (using compute node to avoid login lag)"
# We zip into a temporary folder to avoid permissions issues
tar -czf "export_tmp/$ARCHIVE_NAME" logs/

echo "[ cluster ] Archive created: export_tmp/$ARCHIVE_NAME"
echo "[ cluster ] Starting rsync push to $LAPTOP_IP..."

# Push the single large file to the laptop
rsync -avzP "export_tmp/$ARCHIVE_NAME" "${LAPTOP_USER}@${LAPTOP_IP}:${DEST_PATH}"

if [ $? -eq 0 ]; then
    echo "[ cluster ] SUCCESS: Logs transferred to laptop."
    echo "[ cluster ] Cleaning up temporary archive..."
    rm "export_tmp/$ARCHIVE_NAME"
else
    echo "[ cluster ] ERROR: Transfer failed."
    echo "[ cluster ] 1. Is your laptop's SSH enabled (Remote Login)?"
    echo "[ cluster ] 2. Is your IP ($LAPTOP_IP) correct and reachable?"
fi
