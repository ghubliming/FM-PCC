#!/bin/bash
set -euo pipefail

# Define directories to search
TARGET_DIRS=(
    "/u/home/llim/FMPCC/FM-PCC/Data_Analysis/analysis_results_checkpoint"
    "/u/home/llim/FMPCC/FM-PCC/Data_Analysis/analysis_results"
)

echo "Searching for files larger than 100 MB..."

for dir in "${TARGET_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Directory not found, skipping: $dir"
        continue
    fi

    echo "--- Scanning: $dir ---"
    
    # Find files >100M that are not already .gz, .zip, or .tar files
    find "$dir" -type f -size +100M ! -name "*.gz" ! -name "*.zip" ! -name "*.tar*" | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "Compressing ($size): $file"
        
        # Compress in-place (replaces filename.ext with filename.ext.gz)
        gzip -v "$file"
    done
done

echo "Done! All files over 100 MB have been compressed."