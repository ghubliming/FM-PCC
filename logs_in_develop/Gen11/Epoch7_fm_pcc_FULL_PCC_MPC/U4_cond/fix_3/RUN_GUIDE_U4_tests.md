# U4 Test Run Guide (SLURM)

**As of**: fix_3 (2026-06-28)
**Submit pattern**: `./Slurm_Codes/submit.sh <script.sh> [arg1 arg2 ...]`
**Eval script args**: `$1=scene  $2=seeds  $3=n_trials  $4=projection  $5=record`
**Train script args**: `$1=scene  $2=seeds`

All config edits happen **here in Docker**, then `git push` → `git pull` on the cluster before submitting.

---

## One-time setup — rename pre-U4 checkpoint folders (on cluster)

`cond_mode` was added to `args_to_watch`, which appends `_condp_des` to every checkpoint folder.
Run these `mv` commands **on the cluster** for each scene you have a checkpoint for:

```bash
# corridor
mv logs/UAV_FM/uav-corridor/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE \
   logs/UAV_FM/uav-corridor/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condp_des

# s_curve
mv logs/UAV_FM/uav-s_curve/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE \
   logs/UAV_FM/uav-s_curve/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condp_des

# pillars
mv logs/UAV_FM/uav-pillars/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE \
   logs/UAV_FM/uav-pillars/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condp_des

# all (if pooled checkpoint exists)
mv logs/UAV_FM/uav-all/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE \
   logs/UAV_FM/uav-all/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condp_des
```

Do once. No retrain needed.

---

## Phase 0 — Re-anchor (NO retrain)

`cond_mode = 'p_des'` stays. Only `reanchor_alpha` in the plan block changes.
Each test: edit `config/uav.py` → git sync → submit eval.

---

### Test 0a — Baseline alpha = 0.0 (confirm existing behavior still works)

In `config/uav.py` → `plan_flow_matching_v3_uav`:
```python
'reanchor_alpha': 0.0,
```

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"
```

Output folder: `logs/UAV_FM/uav-corridor/plans/.../<variant>/`
(no `_reanchor` tag when alpha=0.0 — the baseline, nothing appended)

---

### Test 0b — Re-anchor alpha = 0.5

In `config/uav.py` → `plan_flow_matching_v3_uav`:
```python
'reanchor_alpha': 0.5,
```

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"
```

Output folder: `logs/UAV_FM/uav-corridor/plans/.../<variant>_reanchor0.5/`

---

### Test 0c — Re-anchor alpha = 1.0 (hard reset every step)

In `config/uav.py` → `plan_flow_matching_v3_uav`:
```python
'reanchor_alpha': 1.0,
```

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"
```

Output folder: `logs/UAV_FM/uav-corridor/plans/.../<variant>_reanchor1.0/`

---

All three phase 0 results land in **separate folders** — submit in any order, no overwrite.
Check `results.json` → `track_err_mean`, `success_rate`, `total_over_budget` across 0a/0b/0c.
If 0c stops the corridor crash → phase 1 is worth the retrain cost.

---

## Phase 1 — Plan-in-real-position (retrain required)

Two config edits, then train → eval.

### Step 1 — Edit config/uav.py (both blocks)

Training block (`flow_matching_v3_uav`):
```python
'cond_mode': 'real_p',   # was: 'p_des'
```

Plan block (`plan_flow_matching_v3_uav`):
```python
'cond_mode': 'real_p',   # was: 'p_des'
'lead_gain': 1.0,        # start here; increase if drone under-reaches goals
```

`git add config/uav.py && git push` → `git pull` on the cluster.

### Step 2 — Retrain

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh corridor "6"
```

New checkpoint at: `logs/UAV_FM/uav-corridor/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condreal_p/6/`
(fully isolated — `p_des` checkpoint untouched)

### Step 3 — Eval (lead_gain = 1.0)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"
```

Output: `logs/UAV_FM/uav-corridor/plans/.../<variant>/` (no tag = lead=1.0 default)

### Step 4 — Eval lead_gain = 1.5 (if under-reaches goals)

In `config/uav.py` → `plan_flow_matching_v3_uav`:
```python
'lead_gain': 1.5,
```

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"
```

Output: `logs/UAV_FM/uav-corridor/plans/.../<variant>_lead1.5/`

### Or: train + eval as one chained pipeline

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh corridor 6 20
```

Eval runs automatically after train (SLURM `afterok` dependency).

---

## Quick reference

| What to change | File | Key |
|---|---|---|
| Phase 0 alpha sweep | `config/uav.py` → `plan_flow_matching_v3_uav` | `'reanchor_alpha'` |
| Phase 1 mode switch | `config/uav.py` → **both** blocks | `'cond_mode'` |
| Phase 1 lead tuning | `config/uav.py` → `plan_flow_matching_v3_uav` | `'lead_gain'` |
| Default seed | `config/uav_projection.yaml` | `seed:` |
| Default n_trials | `config/uav_projection.yaml` | `n_trials:` |

**n_trials source priority**: yaml (`config/uav_projection.yaml`) is the default; passing `$3` to the submit command overrides it:
```bash
# uses yaml n_trials (e.g. 10 if that's what the yaml says)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"

# CLI override: forces 5 trials regardless of yaml
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6" 5
#                                                                    scene  seeds n_trials
```
