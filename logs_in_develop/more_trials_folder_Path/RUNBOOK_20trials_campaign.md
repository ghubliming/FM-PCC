# RUNBOOK — 20-trial campaign on Slurm (`FMPCC_RUN_MSG`)

**Date:** 2026-08-13
**Patch it depends on:** [`CHANGELOG_custom_msg_path_token.md`](CHANGELOG_custom_msg_path_token.md)
**Plan:** [`CHECKLIST_custom_msg_path_token.md`](CHECKLIST_custom_msg_path_token.md)

**Phasing (user, 2026-08-13): run MeanFlow only for now — §3. The other arms are §5, later.**
The tag is per-job, so arms run weeks apart still land in the same `_msg20trials` namespace and
compare fine. Just keep `FMPCC_RUN_MSG` spelled identically every time.

Everything here runs on the cluster (i6-gpu-1). Nothing in this repo's container can execute it.

---

## 0. Ship the patch

```bash
# local
git add config/avoiding-d3il.py config/aligning-d3il-visual.py config/avoiding-d3il-visual.py \
        logs_in_develop/more_trials_folder_Path
git commit -m "(DA 20-trial campaign) feat: custom_msg results-path token for plan blocks"
git push

# cluster
cd ~/FMPCC/FM-PCC && git pull
```

---

## 1. Campaign config edits (NOT part of the patch — do these by hand)

### a) Trial budget — `config/meanflow_projection_eval.yaml`
```yaml
n_trials: 20        # was 2
```
(For §5 later, the same edit in `config/projection_eval.yaml` and
`config/alphaflow_projection_eval.yaml`.)

### b) Trim `projection_variants` in that same yaml
This is what pays for the 10× trials. Candidates to drop: the Table-2 rows
`dpcc-c-tightened-dt0p25 / dt0p5 / dt2p0 / dt4p0`, plus any arm already ruled out.
Keep the headline set you actually want in the final table.

### c) `mf_unet` — point the plan block at the UNet checkpoint
`config/avoiding-d3il.py:1430` (block `plan_fm_v3_meanflow`):
```python
'imf_backbone': 'unet',      # was 'mf_dit'
```
The `_bb{imf_backbone}` token in `diffusion_loadpath` then resolves to the existing
`flow_matching_v3_meanflow/…_bbunet_tslogit_normal_dp0.5` checkpoint (the one behind
`DA_20260811_MF_UNet32_full5seeds_avoiding.md`). `freq_dim: 32` is already the default — no
second edit. **`af_sit` needs no edit** (`'sit'` is already the default at `:1538`).

### d) ⚠️ FMv3ODE K knob — §5 only, still missing
`plan_fm_v3_ode_selectable` hard-codes `'flow_steps_v3': 10` and the eval has no `--flow-steps`
(`Parser.add_extras` is commented out at `diffuser/utils/setup.py:76`). Until that is added,
K ∈ {1,2,5,10,20} for FMv3ODE means hand-editing `config/avoiding-d3il.py:1210` between submits.
Does not block §3.

---

## 2. Set the message

```bash
export FMPCC_RUN_MSG=20trials
```

- One export per cluster shell; `submit.sh:42` passes `--export=ALL`, so it reaches every job.
- **Same spelling for every arm**, including the §5 arms run later — DA groups on this string.
- 🔴 **Do NOT put it in `~/.bashrc`.** It is global to the job and would silently tag every
  unrelated eval you run afterwards.
- Sanitizer: `[A-Za-z0-9._-]` kept, everything else → `-`, capped at 40 chars. `20trials` is clean.

---

## 3. ⭐ PHASE 1 — MeanFlow only

### 3.1 Smoke test on one seed first

```bash
export FMPCC_RUN_MSG=20trials
MF_FLOW_STEPS="1" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh --seed 6
```

Three lines to check in the log:

| line | expect |
|---|---|
| `[ config/avoiding-d3il ] custom_msg="20trials" -> …` | present |
| `[ utils/setup ] Made savepath: …/H8_K1_…_msg20trials/6` | tag on the **results** leaf |
| the model-load line, `…/flow_matching_v3_meanflow/H8_…_bbunet_…` | **NO `_msg`** — checkpoints must not move |

Also confirm `custom_msg` appears in that run's `args.json`. Then **time the job** and scale.

### 3.2 Budget

`eval_meanflow.sh` loops `MF_FLOW_STEPS="1 2 5 10 20"` inside ONE job. At `n_trials: 20` that will
not fit `--time=24:00:00`: eval 24416 was 4 h 17 m for 4 seeds at `n_trials: 2`, so the full grid
lands near 50 h. **One K per job:**

```bash
for K in 1 2 5 10 20; do
  MF_FLOW_STEPS="$K" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh
done
```

If a single K still overruns, fan out per seed as well:
```bash
for K in 1 2 5 10 20; do for S in 6 7 8 9 10; do
  MF_FLOW_STEPS="$K" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh --seed $S
done; done
```

### 3.3 HardFlow arm (optional, same phase)

```bash
for K in 1 2 5 10 20; do
  HFFM_FLOW_STEPS="$K" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
done
```
`HFFM_FLOW_STEPS` forces single-K on that script. Hold `DPCC_THRESHOLD` at 0.5 as the fixed arm-B
reference; arm-B and arm-C thresholds are independent by design.

### 3.4 load_results — **same env** 🔴

```bash
export FMPCC_RUN_MSG=20trials     # re-export if this is a fresh shell
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/load_results_meanflow.sh
```

It rebuilds the path through the same Parser. **Without the export it reads the OLD 2-trial folder
and reports those numbers with no error** — the quietest failure mode in this plan.

---

## 4. DA (works after any phase, partial data is fine)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/DA/run_da_batch_v3.sh
```

No flags, no code change. `--parent-path logs/avoiding-d3il/plans` + `max_depth=10`
(`main_da_batch.py:182`) finds the new folders; the `_msg20trials` suffix in the candidate name
separates them from their 2-trial siblings in every table and plot. The aggregator is trial-count
agnostic (`data_loader.py:163`, `float(np.mean(value))`), so 2-trial and 20-trial candidates can
sit in one report — read the suffix before comparing.

In the HTML viewer, select only the `_msg20trials` candidates for the final table.

---

## 5. PHASE 2 — the other arms (later)

Same `export FMPCC_RUN_MSG=20trials`, same `n_trials: 20` + variant trim in each arm's yaml.

```bash
# α-Flow (af_sit — no config edit needed)
for K in 1 2 5 10 20; do
  AF_FLOW_STEPS="$K" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh
done
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/load_results_alphaflow.sh

# DPCC K20 / aw10 baseline — the pinned Target
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_dpcc_job.sh
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/load_results_dpcc_job.sh

# FMv3ODE — see §1d first
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_fmv3_ode_job.sh
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/load_results_fmv3_job.sh
```

---

## 6. Gotchas

| # | risk |
|---|---|
| 1 | **Fresh shell = no `FMPCC_RUN_MSG`.** Untagged results land in the OLD folder and overwrite 2-trial data. Re-export in every new session, and check the `custom_msg=` line at the top of each log. |
| 2 | **`load_results` without the export** silently reports the old numbers (§3.4). |
| 3 | **npz ~10× bigger** — per-trial `obs_all`/`act_all`/`sampled_trajectories_all` (`eval_*.py:254`). Commit `567af3d7` already had to harden DA auto-scan against OOM; watch memory on the first 20-trial scan. |
| 4 | **`--time` is already at the 24 h cap** on the eval sbatch scripts. Split by K (and seed) rather than raising it. |
| 5 | **Backbone edit is an identity key.** After §1c, `imf_backbone` must be `'unet'` in the plan block *and* match the checkpoint, or the state_dict load fails loudly (that failure is fine — it's the silent ones that hurt). |
| 6 | Mixing a partially-trimmed `projection_variants` between arms makes the final table ragged. Decide the kept set once, before §3. |

---

## 7. Rollback

Patch is a no-op with the env unset. To remove entirely:
```bash
git checkout -- config/avoiding-d3il.py config/aligning-d3il-visual.py config/avoiding-d3il-visual.py
```
Results already written under `_msg20trials` folders are untouched by that and stay readable by DA.
