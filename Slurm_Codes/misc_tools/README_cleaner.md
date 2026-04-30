# Log Cleaner Guide

This tool removes the "insane" amount of `tqdm` progress bar updates (0% to 99%) from your existing Slurm log files, keeping only the final 100% summary lines and all other important output (errors, prints, etc.).

## Location
Script: `Slurm_Codes/misc_tools/clean_logs.py`

## How to Use

Run the script and point it to any `.log` file that is cluttered with progress bars:

```bash
python Slurm_Codes/misc_tools/clean_logs.py your_polluted_file.log
```

## Outcome
- **Preserves Original**: The original log file is not touched.
- **New Cleaned File**: A new file named `your_polluted_file_cleaned.log` is created.
- **Surgical Cleaning**: It deletes only the redundant progress refreshes. 100% completion lines and all crash/error logs are **kept**.
