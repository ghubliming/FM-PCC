# Example 
#!/bin/bash
#SBATCH --job-name=fm_training        # The name of your job
#SBATCH --partition=gpu-1-student     # The specific queue you have access to
#SBATCH --nodes=1                     # How many machines you need (always 1)
#SBATCH --gres=gpu:1                  # How many GPUs you need (request 1)
#SBATCH --cpus-per-task=4             # How many CPU cores you need for data loading
#SBATCH --mem=16G                     # How much RAM you need
#SBATCH --time=12:00:00               # Max time limit (Hours:Minutes:Seconds)
#SBATCH --output=logs/job_%j.out      # Where to save your python print() statements

# 1. Initialize your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fm_pcc

# 2. Navigate to your code folder (change this to your actual folder name)
cd ~/your_project_folder

# 3. Run your python script
python main.py