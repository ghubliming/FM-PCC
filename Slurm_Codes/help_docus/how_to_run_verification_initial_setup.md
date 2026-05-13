# How to Run FM-PCC Environment Verification

This guide explains how to submit the environment audit job to your SLURM cluster to ensure your setup is perfect.

---

## 1. Login to SLURM
SSH into your remote cluster login node:
```bash
ssh llim@vmknoll81
```

## 2. Navigate to the Repository
Move into your project folder:
```bash
cd ~/FMPCC/FM-PCC
```

## 3. Submit the Job
You **do not** need to activate the Conda environment manually on the login node. The `.sh` script handles the activation automatically inside the compute node.

Submit the verification job using the new **Pro-Logging Wrapper**:
```bash
# Make the wrapper executable (one time only)
chmod +x Slurm_Codes/submit.sh

# Submit the job
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/verify_env_job.sh
```

## 4. Monitor the Job Status
Check if your job is running or pending:
```bash
squeue -u llim
```
*   `PD`: Pending (waiting for a GPU)
*   `R`: Running
*   `CG`: Completing

## 5. Check the Results
Your logs are now organized by **Date**. You can find them in `Slurm_Codes/logs/YYYY-MM-DD/`.

### The Easy Way (Monitor Live)
To see the results of your **most recent** job instantly without searching for the filename:
```bash
tail -f Slurm_Codes/logs/latest.log
```

### The Organized Way
To see the full history:
```bash
# Example (replace DATE with today's date)
cat Slurm_Codes/logs/2026-04-29/fmpcc_verify_19739_133500.log
```

### What to look for:
At the bottom of the log, you should see:
```text
✅ VERIFICATION SUCCESSFUL: Environment is ready for training.
```

If you see a `❌ VERIFICATION FAILED`, the log will tell you exactly which package is missing or if the Numpy version is wrong.
