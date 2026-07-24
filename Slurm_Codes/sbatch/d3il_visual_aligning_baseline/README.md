# D3IL Visual Aligning Baseline — Sbatch Scripts

| Script | Role | Entry point? |
|---|---|---|
| `train_d3il_baseline.sh` | Compute job — train one agent/seed (24h, GPU) | No |
| `eval_d3il_baseline.sh` | Compute job — eval one seed, writes success+entropy JSON (24h, GPU) | No |
| `pipeline_d3il_baseline.sh` | Orchestrator — submits train→eval (`afterok`) for **one seed** (10min) | Single seed |
| `run_all_seeds_d3il_baseline.sh` | Orchestrator — submits train→eval pairs for **all 6 seeds** (10min) | **Normal use** |

## Normal Usage

```bash
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/run_all_seeds_d3il_baseline.sh
```

## Single Seed (debug / rerun one seed)

```bash
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh ddpm_encdec_vision 42 200 all paper
```
