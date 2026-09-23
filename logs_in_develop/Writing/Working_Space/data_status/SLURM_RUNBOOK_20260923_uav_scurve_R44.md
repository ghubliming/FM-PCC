# SLURM RUNBOOK 23-09 — UAV-s-curve R44: raw generative grid first (A); projection (B) and controllers (C) after the author's pick

**Companion of [`PENDING_20260923_uav_scurve_R44_raw_first.md`](PENDING_20260923_uav_scurve_R44_raw_first.md)** (the spec; this
file is the "how"). Cluster i6-gpu-1. One driver for all three phases:
**`Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh`**, modes `plan` (default) · `submit` · `status` · `export`. Only
`submit` touches Slurm. Nothing here was run from the container; the driver was dry-run locally against a sandbox with a
fake `sbatch`/`squeue` and real 05–06-09 job logs (submit args, env isolation, status checks, export all verified).
Replaces the s-curve commands of the 22-09 runbook (R31) and of the 23-09 corridor/pillars runbook §3b (R42).

## 0 · Before the first job

| step | what | check |
| :-- | :-- | :-- |
| 0.1 | get the driver to the cluster. `Slurm_Codes/temp_bash/` is gitignored: `git add -f Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh`, commit, `git pull` on the cluster (the other p23 drivers are tracked the same way), or copy it by hand | `ls Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh` |
| 0.2 | `bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh` (plan, submits nothing) | `pre-flight passed`; four `[ ok ]` checkpoint lines (mf, af, fm, diffusion at seed 6); `marker 'p23scgrid': 0 result folder(s)`; `10 job(s) would be submitted, 0 skipped` |

The pre-flight refuses to run if: `s_curve_hg` is missing from `config/uav_projection.yaml`; the threshold is not 0.5; the
default controller in `config/uav_mix.py` is not `pid_stopgo`; the plan-block K is not 20; the eval does not read
`UAV_MIX_VARIANTS` / `UAV_MIX_GEO_VARIANTS` / `FMPCC_UAV_EVAL_TAG` / `UAV_MIX_CONTROLLER`; or `eval_mix_uav.sh` lost its
GPU/EGL isolation block. Phase B also checks the HardFlow guard and the `hardflow_new-{r,c,t}` registration; Phase C checks
for the `FMPCC_mjx` conda env.

## 1 · The marker (unique per phase, carried three ways)

| phase | tag | results folder (under `logs/UAV_MIX/uav-s_curve/plans/mix_uav_<e>/<train>/`) | job name = log file stem |
| :-- | :-- | :-- | :-- |
| A | **`p23scgrid`** | `E<e>_K<k>_mpc4_pid_stopgo_T0.5[_EPlatest]_p23scgrid/6/s_curve_hg_bounds+dynamics+geo_bounds+halfspace+obstacles/diffuser/` | `p23scgrid_<cell>_<e>_K<k>` |
| B | **`p23scproj`** | `E<e>_K<k>_mpc4_pid_stopgo_T0.5[_EPlatest]_p23scproj/6/s_curve_hg_…/<variant>/` | `p23scproj_<cell>_<e>_K<k>` |
| C | **`p23scmjpc`** | `E<e>_K<k>_mpc4_mjpc_T0.5[_EPlatest]_p23scmjpc/6/s_curve_hg_…/<variant>/` | `p23scmjpc_<cell>_<e>_K<k>` |

- `_EPlatest` appears on CI-MeanFM (`af`) only. Endpoint rows are written as `hardflow_sls-…` (input name `hardflow_new-…`).
- Job logs: `Slurm_Codes/logs/<date>/<time>_<job name>_<id>.log`, so `squeue` and `ls` show the marker directly. Every
  submission also appends a line to the ledger `Slurm_Codes/logs/<date>/R44_<tag>_jobids.tsv`.
- The old tags (`u7hg`, `u18sc`, `u6unet_ae02`, …) are refused as `TAG=`. Nothing new is pooled with them.
- `submit` skips a cell that is already in `squeue` or whose `results.json` already exist under the marker, so running it
  twice is safe. `FORCE=1` re-flies such a cell and overwrites it.

**Why the driver calls `sbatch` itself.** It makes the same per-K call as `eval_k_sweep.sh`: `sbatch --parsable`, 24 h, the
`submit.sh` log-path convention, then `Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh <e> s_curve 6 10 fm_only none <K>`. The
only difference is `--job-name=<marker>`. `eval_k_sweep.sh` would name every child `uav_mix_eval`, which is also the
name of every corridor-v3 job in the queue. The job script, its EGL isolation, the conda-env selection and the eval are
the tracked files, unchanged.

**Fixed environment on every job.** Every other knob the eval reads is unset, so nothing exported in the login shell can
leak in; the sandbox test exported `UAV_MIX_CONTROLLER=mjpc FMPCC_SAFE_EPS_FRAC=1.0 SBATCH_DEPENDENCY=…`, and none reached a job.
`FMPCC_UAV_EVAL_TAG=<tag> UAV_MIX_GEO_VARIANTS=s_curve_hg UAV_MIX_VARIANTS=<cell> FMPCC_SAFE_EPS_MODE=scaled
FMPCC_SAFE_EPS_FRAC=1e-3`; `af` adds `UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest`; C adds
`UAV_MIX_CONTROLLER=mjpc`. Diffusion gets no K argument (its K is the training budget; the plan block labels the folder `K20`,
the log prints `K: <plan block>`, as in job 25612).

**SAFE_EPS `scaled` / `1e-3` is what the existing tables used.** Every s-curve job on record ran with it. This includes
the tightened cells the spec (§3) named: FM K2 `u7hg` (job 25424, which carried `dpcc-{r,c,t}-tightened`) and diffusion
`u7hg` (job 25612, `dpcc-{c,t}-tightened`). The same holds for jobs 25422, 25441–25443, 25500, 25502 and 25908–25910. It
is not the corridor's `1.0`. The driver pins it in all three phases.

## 2 · Phase A — the raw grid (Table 6.14 `tab:uav-scurve`) — RUN NOW

```bash
bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh             # plan
bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh submit      # the ten cells, one GPU job each
bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh status      # any time: queue, log checks, results preview
bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh export      # when status says "all 10 cell(s) complete"
# only the four required cells:   CELLS=required bash … submit        (= A1 A3 A6 A7)
```

**Default = all ten cells.** Author, 23-09: *"we will finish together all the scurve runs for thesis."* One tag then covers
the whole table. The six recommended re-flies add about 1 GPU-h.

| cell | model | K | job name | checkpoint expected in the log | planner time for 10 flights (at most) |
| :-- | :-- | --: | :-- | :-- | --: |
| A1 | MeanFM `mf` | 1 | `p23scgrid_A1_mf_K1` | `state_best.pt` (step 95000) | ~1.5 min |
| A2 | MeanFM | 2 | `p23scgrid_A2_mf_K2` | same | ~3 min |
| A4 | CI-MeanFM `af` (U-Net, α_end 0.2, latest) | 1 | `p23scgrid_A4_af_K1` | `state_100000.pt` | ~1.5 min |
| A5 | CI-MeanFM | 2 | `p23scgrid_A5_af_K2` | same | ~3 min |
| A7 | FM `fm` | 1 | `p23scgrid_A7_fm_K1` | `state_best.pt` (step 91000) | ~1.5 min |
| A8 | FM | 2 | `p23scgrid_A8_fm_K2` | same | ~3 min |
| A3 | MeanFM | 20 | `p23scgrid_A3_mf_K20` | step 95000 | ~25 min |
| A6 | CI-MeanFM | 20 | `p23scgrid_A6_af_K20` | step 100000 | ~25 min |
| A9 | FM | 20 | `p23scgrid_A9_fm_K20` | step 91000 | ~25 min |
| A10 | diffusion | 20 | `p23scgrid_A10_diffusion_K20` | `state_best.pt` (step 91000) | ~25 min |

The expected checkpoints are the ones the `u7hg` / `u18sc` / `u6unet_ae02` jobs loaded (their logs). **CI-MeanFM
`EPlatest_u6unet_ae02` (jobs 25441–25443) and `EPlatest_u7hg` (25634, 25637) loaded the same file:** `state_100000.pt` from
`mix_uav_af/H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet/6`. `status` prints the step each new job loaded. If a
step differs from this table, the checkpoint changed after 13-09 (the 5-seed campaign), and the old rows are a different
model.

Planner time is 10 × 871 steps × the ms/step of Table 6.14 (9 / 18 / ~170 ms). Aborted flights end earlier. Loading the
model and the simulator add a few minutes per job. Every job still requests 24 h (standing UAV rule). The jobs queue beside
the corridor-v3 waves (`AssocGrpGRES`) and start as GPUs free up.

**What `status` checks per cell** (the spec's "check each child log before counting it"). Each check prints `ok` or
`XX`:

- tag `p23scgrid` with `SAFE_EPS_MODE=scaled SAFE_EPS_FRAC=1e-3`;
- the header line `ENGINE: <e> SCENE: s_curve SEEDS: 6 N_TRIALS: 10 K: <k>`;
- `[ U11 ] geo variants = s_curve_hg`;
- `variants=['diffuser']`, exactly;
- controller `pid_stopgo` with the FMPCC env;
- CI-MeanFM only: bone `unet`, α_end 0.2 and checkpoint `latest`; the other models: checkpoint `best`;
- the job finished (`Job completed successfully`); a time-limit, cancel, traceback or `[ ERROR ]` is flagged;
- `results.json` present with `n_trials 10`, previewed as success x/10 · S&C x/10 · distance · aborted x/10 · ms/step ·
  circuit-breaker trips.

The preview is for watching the run only. The DA is made from the exported folders. At K ≤ 2 the log contains an
`[hardflow][BLOCKED]` line: the eval drops the endpoint arm it would otherwise add. That is expected in Phase A.

**Export and download.**

```bash
bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh export
#   -> export_tmp/R44_p23scgrid_<stamp>.tar.gz : the 10 result folders (results.json, npz, png, diagnostics/),
#      the 10 job logs, the ledger, and any DA_UAV_v1 batch folder built from this tag
# laptop:  scp <cluster>:~/FMPCC/FM-PCC/export_tmp/R44_p23scgrid_<stamp>.tar.gz temp/<dd-mm>/  &&  tar -xzf …
```

Optional, and recommended for provenance: the thesis cites a `batch_uav_*` folder, so run the official batch on this tag
before exporting. `export` then includes it automatically.

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/DA/run_da_batch_uav.sh \
  "$(find logs/UAV_MIX/uav-s_curve/plans -mindepth 3 -maxdepth 3 -type d -name 'E*_p23scgrid' | paste -sd, -)"
```

## 3 · After Phase A — the DA, then the author's pick (spec §2); nothing further is submitted before that

1. The author passes the unpacked folder to Claude.
2. Claude writes the preliminary DA to `Data_Analysis/DA_in_Paper/analysis/`:
   - the ten cells of Table 6.14 from `results.json` (or from the batch CSV, if it was run);
   - DA_UAV_v1's gates checked by hand: `projection_health`, `divergence`, n = 10 per cell;
   - the old-versus-new comparison for the six re-flown cells;
   - the selection rule's answer: the configuration whose plans cross the finish line on the most flights.

   It then notifies v3 through `cross_draft/to_v3/` and `INBOX.md`.

   The raw folders are sufficient without the batch. Checked on 23-09 on the MeanFM K10 `u7hg` cell, where `results.json`
   and the 15-09 batch's `per_rollout_detail.csv` agree on all ten flights: `success.relaxed` = `success_relaxed`,
   `goal.dist` = `goal_dist`, `divergence.aborted` = `divergence_aborted`, `timing.total_ms_mean` = `avg_time_ms`.
3. The author names **engine and K** for Phase B. Rule of spec §2: at K = 20 both projection methods run. At K = 1 or 2 the
   choice is between (i) that model at K = 20 with both methods and (ii) per-step only at the winning budget, with the
   endpoint rows printed as "no guiding step".

## 4 · Phase B — projection on the pick (Table 6.15 `tab:uav-scurve-projection`) — AFTER §3

```bash
PHASE=B SC_ENGINE=<mf|af|fm|diffusion> SC_K=<1|2|20> bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh          # plan
PHASE=B SC_ENGINE=… SC_K=… bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh submit   # then status / export with the same prefix
```
Without `SC_ENGINE` and `SC_K` the driver stops at a `[GATE]` line. K must be 1, 2 or 20; diffusion must be K = 20.

| pick | jobs (tag `p23scproj`) | wall per job |
| :-- | :-- | --: |
| flow model, K = 20 | **B0** `diffuser` · **Br** `dpcc-r-tightened,hardflow_new-r-tightened` · **Bc** (…`-c-`…) · **Bt** (…`-t-`…) | B0 ~0.5 h; each pair **~10–20 h ⚠** |
| flow model, K = 1 or 2 | **B1** `diffuser,dpcc-r-tightened,dpcc-c-tightened,dpcc-t-tightened` (no guiding step → no endpoint rows) | ~1.5 h |
| diffusion, K = 20 | **B0** `diffuser` · **Br** / **Bc** / **Bt**, one per-step rule each (no velocity field → no endpoint rows) | up to 24 h each |

⚠ **Why a pair shares one job.** The eval refuses a job that holds HardFlow variants and no `dpcc-*` row
(`eval_mix_uav.py`, the `UAV_MIX_VARIANTS` guard). Each endpoint rule therefore rides with its per-step twin, and every cell
still runs exactly once. Spec §2 estimates the cost at K = 20 as per-step 7–11 h plus endpoint 3–9 h, so a pair takes
10–20 h. That is inside the 24 h cap, but not by much in the worst case. If a pair job is killed by the time limit, the
second variant is left partial (n < 10; `status` flags it). The re-run is `FORCE=1 CELLS=<id>`. **Flagged: expected
> 12 h.**

`status` also checks the endpoint rows: no `[hardflow][BLOCKED]` and no `DEGENERATE` (at K = 20 and A = 0.5 there are nine
guiding steps).

## 5 · Phase C — the same plans under MuJoCo MPC (Table 6.16 `tab:uav-controller`) — AFTER PHASE B

```bash
PHASE=C SC_ENGINE=<as B> SC_K=<as B> SC_RULE=dpcc-<r|c|t>-tightened bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh [submit]
```

`SC_RULE` is the per-step rule with the most crossings in Phase B. The jobs are **C0** `diffuser` and **C1** `<SC_RULE>`,
with `UAV_MIX_CONTROLLER=mjpc`; `eval_mix_uav.sh` then activates the `FMPCC_mjx` env, which the pre-flight checks. MuJoCo
MPC adds about 126 ms per control step (Table 6.17): about 40 min for C0 at K = 20, and C1 costs its Phase B twin plus that.
The cascaded-geometric rows of Table 6.16 are the matching A and B cells.

## 6 · Submission record (fill in; the ledger `Slurm_Codes/logs/<date>/R44_<tag>_jobids.tsv` has the ids)

| date | phase / cells | job ids | outcome |
| :-- | :-- | :-- | :-- |
| 2026-09-23 21:26 | **A · all ten cells** (`p23scgrid`), `bash … submit` on the cluster at git `0df6df71` (code dirs identical to the checked tree); pre-flight passed, four checkpoints `[ ok ]`, marker fresh (0 folders) | **26167** A1 mf K1 · **26168** A2 mf K2 · **26169** A4 af K1 · **26170** A5 af K2 · **26171** A7 fm K1 · **26172** A8 fm K2 · **26173** A3 mf K20 · **26174** A6 af K20 · **26175** A9 fm K20 · **26176** A10 diffusion K20 | submitted; logs `Slurm_Codes/logs/2026-09-23/21_26_50_p23scgrid_<cell>_<e>_K<k>_<id>.log`, ledger `…/2026-09-23/R44_p23scgrid_jobids.tsv` |
| | A · DA_UAV_v1 batch on `p23scgrid` (optional) | | |
| | B · (engine, K named by the author) | | |
| | C · (rule from B) | | |

Claude (Opus 5.5, Claude Code) · 2026-09-23 · driver written and dry-run in a sandbox; nothing submitted.
