# To Start Training
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/train_fmv3_ode_job.sh

# To Start Evaluation
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_fmv3_ode_job.sh

# To monitor whichever one you started most recently
tail -f Slurm_Codes/logs/latest.log
