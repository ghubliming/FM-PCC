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

Submit the verification job using `sbatch`:
```bash
sbatch Slurm_Codes/sbatch/verify_env_job.sh
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
Once the job is finished (it disappears from `squeue`), look for the log file in the `Slurm_Codes/sbatch/logs/` directory.

The file will be named `verify_<JOB_ID>.out`. Open it to see the audit results:
```bash
# Example (replace <JOB_ID> with your actual ID)
cat Slurm_Codes/sbatch/logs/verify_12345.out
```

### What to look for:
At the bottom of the log, you should see:
```text
✅ VERIFICATION SUCCESSFUL: Environment is ready for training.
```

If you see a `❌ VERIFICATION FAILED`, the log will tell you exactly which package is missing or if the Numpy version is wrong.
