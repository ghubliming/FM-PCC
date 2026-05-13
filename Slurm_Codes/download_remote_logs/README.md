# Remote Log Export Utility

This utility allows you to download the massive `logs/` directory from the cluster to your local laptop without crashing the login node. It uses a SLURM compute job to handle the heavy compression and transfer in the background.

## Prerequisites

### 1. Enable SSH on your Laptop
Your laptop must be reachable from the cluster via SSH.
- **macOS**: Go to `System Settings` -> `General` -> `Sharing` -> Enable `Remote Login`.
- **Linux**: Ensure `openssh-server` is installed and the service is running (`sudo systemctl enable --now ssh`).
- **Windows (WSL2)**: Ensure your WSL instance has an SSH server running.

### 2. Identify your IP
Find your laptop's IP address (use `ifconfig` or `ip addr`).
> [!NOTE]
> If you are on a VPN or a restricted network, you may need to use an SSH tunnel or your university's internal IP.

## Usage

1. **Configure the script**:
   Open `export_to_laptop.sh` and update these lines:
   ```bash
   LAPTOP_USER="your_username"        # Your laptop's login name
   LAPTOP_IP="your.laptop.ip.here"    # Your laptop's IP address
   DEST_PATH="~/Downloads/FMPCC_Logs" # Target folder on your laptop
   ```

2. **Run the export job**:
   From the project root on the cluster:
   ```bash
   sbatch Slurm_Codes/download_remote_logs/export_to_laptop.sh
   ```

3. **Check Progress**:
   You can monitor the compression and transfer by checking the SLURM output file:
   ```bash
   tail -f export_logs_<JOB_ID>.out
   ```

## Why this is better than standard `scp`:
1. **No Login Lag**: Compressing millions of small files is CPU-intensive. This script runs on a compute node, keeping the login node fast for other users.
2. **Single Stream**: Transferring one `.tar.gz` is significantly faster than transferring individual files.
3. **Background Persistence**: You can close your terminal; the job will continue running on the cluster until the transfer is complete.
