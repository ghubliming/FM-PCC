# SLURM RUNBOOK — 2026-09-20 · remaining thesis runs

Source of scope:
[`PENDING_20260920_all_lacking_runs.md`](PENDING_20260920_all_lacking_runs.md). This runbook replaces
groups G/H/I of the 2026-09-18 runbook for future submissions. In particular, its old group I must not
be used as written: one monolithic five-seed diffusion job can exceed 24 hours, as the K20 campaign did.

This document plans the wave only. No job has been submitted and no executable 2026-09-20 driver has
been written yet. The eventual driver should live under `Slurm_Codes/temp_bash/`, default to PLAN mode,
and submit only through `Slurm_Codes/submit.sh` (or reproduce its log routing inside a dependency-only
orchestrator). It must patch one-off experiment dictionaries in memory rather than leave shared YAML or
Python configuration changed while queued jobs start.

## 1 · Decision and order

Run the visible thesis blockers first. The default wave is **R2 + R25 + R26 + R18 + the diffusion-K20
repair**. Keep R7/R14, R9 and R15 opt-in. Do not spend compute on R16/R17/R27 or `pillars_xl` until the
default wave has landed and the draft is rebuilt.

| phase | owner | action | training | planned Slurm children | dependency |
| :-- | :-- | :-- | :-- | --: | :-- |
| A | R2 | aligning diffusion K20, seed 6, ten training contexts, tightened geometry, unprojected + per-step $r/c/t$ | no | 1 | none |
| B | R25 | CI-MeanFM U-Net, $\alpha_{\rm end}=0.2$, K1/K2, seeds 6–10, three geometries, 20 episodes, tightened $r/c/t$ | no | 2 (one per K) | none |
| C1 | R26 | diffusion K2 training, seeds 7–10, same settings as job 25965 | **four** | 4 (one per seed) | none |
| C2 | R26 | diffusion K2, DPCC protocol, seeds 6–10, tightened $r/c/t$ | no | 3 (one per geometry) | `afterok` all C1 jobs |
| C3 | R26 + R18 | diffusion K2, extended protocol, seeds 6–10, 20 episodes, tightened $r/c/t$ | no | 3 (one per geometry) | `afterok` all C1 jobs |
| D | R18 | diffusion K1/K10, extended protocol, seeds 6–10, 20 episodes, tightened $r/c/t$ | no | 6 (K × geometry) | none |
| E | D10a/table repair | diffusion K20, seed 6, `both-hard`, 20 episodes, `dpcc-t-tightened` only | no | 1 | none |

Default total: **20 jobs**, of which four train. A, B, D and E may run while C1 trains. Release C2/C3
only if all four training jobs exit zero and all four checkpoint folders pass identity checks.

Why shard by geometry and K: the previous K20 extended job reached the 24-hour limit part-way through
seed 6 `both-hard`. Small leaves are resumable and keep each failure local. Never combine training and
evaluation inline in one allocation.

### Time estimate and deadline cut

These are compute-time estimates, excluding scheduler wait and download/DA time. They use job 25965
(2 h 41 m per diffusion training), job 24515 (10 h 20 m for 1,950 CI-MeanFM episodes), job 25247
(about 4 h for 38 alignment items × ten contexts), and the truncated 24-hour K20 diffusion campaign.
The diffusion extended estimates deliberately use the slower K20 campaign rate rather than scaling the
17-minute, two-episode K2 panel job optimistically.

| phase | estimated GPU time | likely elapsed if its shards start together |
| :-- | --: | --: |
| A · R2 | 0.5–1.0 h | 0.5–1.0 h |
| B · R25 | 9–11 h total | 4.5–6 h (two K jobs) |
| C1 · four K2 trainings | 10.7 h total | 2.7–5.5 h (four/two GPUs) |
| C2 · K2, two-episode protocol | 0.3–0.7 h total | under 0.5 h |
| C3 · K2 extended | 7–9 h total | 2.5–3.5 h (three geometries) |
| D · K1/K10 extended | 15–18 h total | 3–6 h (six shards, subject to slots) |
| E · K20 repair | 0.1–0.3 h | 0.1–0.3 h |

**Full A–E: about 45–50 GPU-hours.** Best-case elapsed time with four continuously available GPUs is
roughly 12–15 hours; with two GPUs, roughly 24–28 hours; with one effective GPU slot, about two days.
Queueing can dominate all three estimates.

If time is short, submit in this order:

1. **E + A** — about one GPU-hour total; removes the known K20 coverage defect and closes the visible
   alignment baseline gap.
2. **B** — about ten GPU-hours; fills both explicit CI-MeanFM rows without training.
3. **C1 + C2 only** — about eleven GPU-hours; fills diffusion K2 in the DPCC-protocol table. Defer C3.
4. **C3**, then **D** — extended diffusion rows are the first work to cut.

Thus the recommended deadline wave is **A + B + E** (about 10–12 GPU-hours). Add **C1 + C2** only if
there is another day or several GPUs available. Do not start D merely to obtain partial geometries.

## 2 · Pre-flight gates

Run these on the cluster before PLAN and again immediately before SUBMIT:

```bash
cd ~/FMPCC/FM-PCC
git rev-parse --short HEAD
git status --short
df -h ~/FMPCC ~/FMPCC/FM-PCC/logs
squeue -u "$USER"
```

Then establish the following:

1. **Revision identity.** Record the exact revision in the run record. The local working tree contains
   uncommitted thesis/DA work; do not assume the cluster has it.
2. **Disk.** Job 25965 started with 43 GB free. Four new diffusion checkpoints are the only material
   storage growth in this wave. Do not submit C1 without checking the live value.
3. **R2 contexts = 10.** `config/visual_aligning_eval.yaml` has said `n_contexts: 3` since May, but the
   thesis corpus was produced with an untracked cluster edit and its raw logs/data contain ten contexts
   (documented in `DA_20260901_Gen14_flagship_K20_T0.2_dpcc_vs_hardflow.md`, provenance gap). The new
   job must inject **10 in memory** and echo it. Do not run R2 with the committed value 3.
4. **Checkpoint identities.** R25 must find U-Net, `ae0.2`, `latest` checkpoints for seeds 6–10. C1
   seed 6 must already exist at `diffusion/H8_K2_Dmodels.GaussianDiffusion_aw10/6`; D must find K1 and
   K10 checkpoints for every seed; E must find the K20 seed-6 checkpoint.
5. **No shared-config race.** `n_trials`, seeds, geometries and variants must all be injected by the
   generated job wrapper. Do not perform the old `n_trials: 2 -> 20 -> 2` round trip in
   `config/projection_eval.yaml`.
6. **Exact arms only.** For the headline rows evaluate `dpcc-r-tightened`, `dpcc-c-tightened` and
   `dpcc-t-tightened`. Do not spend the wave on the ten unrelated variants in the stock YAML.
7. **GPU/EGL isolation.** Every generated GPU script retains the standard
   `CUDA_DEVICE_ORDER` / allocated `MUJOCO_EGL_DEVICE_ID` block. No hard-coded device index.

## 3 · Proposed aggregate driver

The implementation should expose this interface:

```bash
bash Slurm_Codes/temp_bash/pipeline_20260920_all_lacking_runs.sh
bash Slurm_Codes/temp_bash/pipeline_20260920_all_lacking_runs.sh submit

WAVE="A B D E" bash Slurm_Codes/temp_bash/pipeline_20260920_all_lacking_runs.sh
WAVE="A B D E" bash Slurm_Codes/temp_bash/pipeline_20260920_all_lacking_runs.sh submit
WAVE="C"       bash Slurm_Codes/temp_bash/pipeline_20260920_all_lacking_runs.sh submit
```

PLAN is the default and must print, without submitting: experiment block, model folder, checkpoint,
seed set, K, geometry, variants, episode count, result tag, requested time and dependency graph. `submit`
must print every returned job ID. Recommended tags are `paper20ext` for newly restricted extended rows,
`paper20dpcc` for the K2 two-episode rows, and the existing `20trials` namespace only for phase E, which
repairs an incomplete leaf in that campaign.

The wrapper patches the imported config dictionary before running the stock evaluator through `runpy`,
following the proven job-25965/25966 Option-A pattern. It must not patch model code. Generated wrappers
and sbatch files are read at job start, so retain them until the full wave ends.

### A · R2 alignment baseline

Use `mix_visual_aligning_test/eval_mix_visual_aligning.py`, engine `diffusion`, seed 6, training split,
K20 checkpoint, `n_contexts=10`, `active_geo_variants=['combined_5']`, and restrict the emitted work to
the tightened sibling with `diffuser` plus `dpcc-r/c/t`. If the evaluator cannot run only the tightened
sibling, running its untightened companion is acceptable; the tightened four cells are the deliverable.

Identity log:

```text
engine=diffusion  seed=6  n_contexts=10  eval_on_train=True
K=20  geometry=combined_5-tightened
variants=diffuser,dpcc-r,dpcc-c,dpcc-t
```

Reject the result if it contains only three contexts or uses the held-out split.

### B · R25 CI-MeanFM extended rows

Two jobs, K1 and K2. Both use `AF_SEEDS='6 7 8 9 10'`, `AF_BONE=unet`, `AF_ALPHA_END=0.2`,
`AF_EPOCH=latest`, `AF_NTRIALS=20`, `FMPCC_MPC_BATCH=4`, three halfspaces and only the three tightened
DPCC rules. The current evaluator already supports the seed/trial/checkpoint overrides; the wrapper is
needed only to narrow geometries/variants and give the run a collision-proof tag.

Each job is 900 episodes (5 seeds × 3 geometries × 20 × 3 rules), not the stock 3,900-episode sweep.
Keep K1 and K2 separate so either can be resumed independently.

### C · R26 diffusion K2

C1 uses four independent copies of the job-25965 training wrapper, seeds 7, 8, 9 and 10. Each patches
`n_diffusion_steps=2` in memory, trains one seed, requests about twice job 25965's measured 2 h 41 m
(use 06:00:00), and writes the normal `H8_K2_..._aw10/<seed>` folder. A four-seed inline job would have
an expected duration near eleven hours and no comfortable 2× margin under the 24-hour ceiling.

C2 and C3 depend on **all four** training IDs. Both include seed 6, so the table becomes a genuine
five-seed row. C2 uses two episodes; C3 uses twenty. Each is split into one job per geometry and runs
only tightened $r/c/t$. A failed C1 job must keep every C2/C3 job held or cancelled; never bypass the
dependency with a missing seed.

### D · R18 diffusion K1/K10 extended rows

Six evaluation jobs: `(K1, K10) × (top-right-hard, top-left-hard, both-hard)`. Each runs seeds 6–10,
20 episodes and tightened $r/c/t$ only. These checkpoints already exist. Do not use the 2026-09-18
group-I command: it launches the full variant/geometry sweep in one allocation and repeats the known
truncation pattern.

### E · repair the truncated K20 leaf

One minimal job: K20, seed 6, `both-hard`, 20 episodes, `dpcc-t-tightened` only, same checkpoint and
`_msg20trials` result identity as the incomplete campaign. It closes both remaining consequences of the
old truncation:

- supplies `dpcc-t-tightened.npz` for D10a's executed-path figure;
- restores the third geometry to the diffusion K20 temporal-consistency table cell.

The raw-plan PNG is already closed from the unprojected sibling campaign and is not a reason to widen
this repair.

## 4 · Optional wave, after the default data lands

| group | owner | recommendation |
| :-- | :-- | :-- |
| F | R7 | MeanFlow U-Net at the endpoint budget floor, seeds 6–10 × three geometries × 20 episodes, both projectors at candidate fan 4. Preserve the published threshold/budget identity and give the rerun a new tag. |
| G | R14 | Candidate-matched ladder with `FMPCC_MPC_BATCH=4` and `HFFM_BATCH=4`. Run the genuine endpoint budgets separately from K1/K2, which are degenerate at A=0.5 and may be controls only. Lock the precise K/threshold grid before implementing; do not silently reinterpret the current table. |
| H | R9 | Requires an authorised code change first: remove the `0.5 *` sampler factor in the three FM implementations and add an explicit result-path token. Start with aligning FM K20 and corridor FM K3/K5; add the avoiding control. No retraining. Do not run untagged or overwrite old FM cells. |
| I | R15 | Re-run the exact controller comparison: MeanFlow U-Net, s-curve, K10, seed 6, controller `mjpc`, variants `diffuser`, `dpcc-r`, `hardflow_sls-r`, **10 flights**, tag distinct from `u7hg`. Set `UAV_EVAL_HOURS=24` as required for every UAV job. |

R16 and R17 are optional thesis strengthening, not closure work. R27 conflicts with the current UAV
engine registry (`ddpm` exposes no velocity field and `supports_hardflow=False`); it is not executable
without a method/code decision and therefore is not a submission in this runbook. R23 stays dormant
while UAV-pillars is absent from Results.

## 5 · Non-run follow-up

These are not Slurm model runs and should not delay A–E:

- D10a: after E finishes, fetch the repaired `dpcc-t-tightened.npz`; the other 20/21 cells are already
  staged. Use projected variants only—the unprojected avoiding NPZ contains scalars, not executed paths.
- D10b: data already exists; build the figure locally.
- D10c: requires an authorised logging change, then a narrow MeanFlow K20 re-evaluation for unprojected,
  per-step and endpoint variants. `obs_all` currently contains commanded/actual TCP, not box pose.
- D2: settle the orientation convention from existing columns.
- D8/D9: stage accounting/provenance (`sacct`, complete logs, job IDs, revisions, checkpoint tags).

## 6 · Completion checks

A job is not complete merely because Slurm says `COMPLETED`. For each leaf verify:

1. the log echoes the intended checkpoint, K, seed(s), context/episode count, geometry and exact variant
   subset;
2. every expected `results.json`/NPZ exists and loads, with `n=2` or `n=20` as requested;
3. no cell is pooled across result tags, protocols, candidate counts or incomplete seeds;
4. C contains five distinct training seeds and three geometries before entering either avoiding table;
5. E changes the K20 temporal-consistency cell from `geos=2` to `geos=3` and provides `obs_all` for 20
   episodes;
6. the result downloader and DA batch preserve the new tags and full paths;
7. the draft tables/figures are rebuilt from real result data, never console logs.

Record job IDs, git revision, checkpoint identities, start/end times and failures in this file after
submission. Do not update `MASTER_TEST_HISTORY.md` unless explicitly instructed.

## 7 · Submission record

| phase | job IDs | revision | state / verification |
| :-- | :-- | :-- | :-- |
| A · R2 | — | — | planned, not submitted |
| B · R25 | — | — | planned, not submitted |
| C1 · R26 train | — | — | planned, not submitted |
| C2 · R26 DPCC protocol | — | — | planned, dependent on C1 |
| C3 · R26 extended | — | — | planned, dependent on C1 |
| D · R18 | — | — | planned, not submitted |
| E · K20 repair | — | — | planned, not submitted |
