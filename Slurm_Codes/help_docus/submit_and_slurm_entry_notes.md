# Common Commands

## let submit.sh runable
cd FMPCC/FM-PCC
chmod +x Slurm_Codes/submit.sh

## Check the status
squeue -u llim
squeue -o "%.10i %.10P %.30j %.10u %.2t %.10M %.10D %R"

## shutdown sbatch
scancel XXXXX

scancel -u $(whoami)
llim

## Verfication
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/verify_env_job.sh

## To Start Training
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/train_fmv3_ode_job.sh

## To Start Evaluation
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_fmv3_ode_job.sh

## To monitor whichever one you started most recently
tail -f Slurm_Codes/logs/latest.log

---
# Cheat Sheet

For managing and monitoring Slurm jobs, here are the "Must-Know" commands:

## 1. Monitoring Your Jobs
*   **`squeue -u $USER`**: Shows only **your** active and pending jobs.
*   **`squeue`**: Shows the entire cluster queue (all users).
*   **`scontrol show job <JOB_ID>`**: Provides very detailed info on a specific job (where it's running, why it's pending, etc.).

## 2. Checking Cluster Resources (Availability)
*   **`sinfo`**: Shows the status of partitions (nodes available, idle, or down).
*   **`sinfo -O "Partition,NodeList,Available,CPUs,Memory,Gres"`**: A more detailed view to see exactly which GPUs/Memory are free.

### To see the raw detailed capacity:
```bash
scontrol show node i6-gpu-1
```

### To get a clean "At a Glance" summary of Free vs Total:
Run this to see exactly how many CPUs and how much Memory/GPU are actually available:
```bash
scontrol show node i6-gpu-1 | grep -E "CfgTRES|AllocTRES"
```

**What to look for in the output:**
*   **CfgTRES**: The **Total** capacity of the node (what it has in total).
*   **AllocTRES**: What is **currently used** by other jobs.
*   **The Difference**: Subtract Alloc from Cfg to find your "Max Capacity" for new jobs.

## 3. Usage & Quotas (Depending on Cluster Setup)
*   **`sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS`**: Shows how much memory and time a **finished** or running job actually used.
*   **`sshare -u $USER`**: Shows your current "Fair Share" priority and usage compared to others.

## 4. Controlling Jobs
*   **`scancel <JOB_ID>`**: Kills a specific job.
*   **`scancel -u $USER`**: Kills **all** of your jobs at once.
