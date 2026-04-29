# Common Commands

## let submit.sh runable
chmod +x Slurm_Codes/submit.sh

## Check the status
squeue -u llim

## shutdown sbatch
scancel XXXXX

# Verfication
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/verify_env_job.sh

# To Start Training
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/train_fmv3_ode_job.sh

# To Start Evaluation
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_fmv3_ode_job.sh

# To monitor whichever one you started most recently
tail -f Slurm_Codes/logs/latest.log
