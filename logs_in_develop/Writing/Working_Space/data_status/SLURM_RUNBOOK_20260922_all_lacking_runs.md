# SLURM RUNBOOK — 2026-09-22 · the remaining thesis runs, most urgent first

Source of scope: [`PENDING_20260922_all_lacking_runs.md`](PENDING_20260922_all_lacking_runs.md) §0 (the
open list, by priority) and §16–§18. **This runbook supersedes
[`SLURM_RUNBOOK_20260920_all_lacking_runs.md`](SLURM_RUNBOOK_20260920_all_lacking_runs.md)**, which was never
run and whose phases B (R25), C3/D (R18) and E (the K20 `both-hard` repair) the author has since struck.
It also retires groups G/H/I of the 2026-09-18 driver.

Driver: `Slurm_Codes/temp_bash/pipeline_20260922_all_lacking_runs.sh` (gitignored — copy to the remote by
hand). PLAN is the default and submits nothing. Every job goes through `Slurm_Codes/submit.sh`; nothing
here edits a tracked config.

**Ownership boundary:** the driver submits and validates. It never downloads, copies or deletes results.

---

## 0 · The author's feedbacks, and what each one changed here

| feedback (where recorded) | effect on this runbook |
| :-- | :-- |
| *"20 episodes for seeds 6–10 … extreme highly cost, remove them"* (v3.52, §10) | R25, R18 and every extended-protocol (20-episode) evaluation are **out**. No group runs `n_trials: 20`; the driver refuses to run B/H unless the yaml says 2. |
| The twenty-episode campaign is quarantined; §6.1 is reported at DPCC's own protocol (v3.55, §12) | The K20 `both-hard` repair (old phase E) is **not** submitted. Everything avoiding-side is 5 seeds × 3 geometries × 2 episodes. |
| R26 **reopened**: five-seed diffusion $\nfe=2$ is visibly pending in Tables 6.1/6.2 (v3.57) | Group B: four trainings (seeds 7–10, one job each, the job-25965 wrapper pattern) and one dependent five-seed evaluation, tag `_msgdpccproto` — the same namespace as the FM and CI-MeanFM protocol rows. |
| A budget enters the avoiding tables only with the full DPCC protocol (v3.60, §13) | B2 evaluates **all five seeds** in one job (seed 6 is re-evaluated under the new tag so the row is one corpus). R28 (FM at 5/10) is offered as opt-in group H because it is *not owed*. |
| R23 pillars: 53/54 projected cells at 0.00; **no further pillars compute** until four `results.json` are read; diffusion pillars projection closed by decision (2026-09-21, §16; pillars runbook §3e) | **No pillars group exists in this driver.** Not the $\nfe=3$ rung, not the diffusion remainder, not `pillars_xxl`. The driver prints the gate as a reminder. |
| The corridor ladder is $\{1,3,5\}$ because endpoint projection has its first real guiding step at $\nfe=3$ (§15/§16) | The s-curve grid (R31) is run at $\nfe\in\{1,3,5\}$ unprojected and $\{3,5\}$ projected — the same ladder — so the two quadrotor scenes read on one axis. |
| *"the S_curve feels still ridiculous … build the table first then I will fill in slowly, mark the missing lines"* (v3.61, §18) | Groups D and E fill exactly the \emph{pending} cells of `tab:uav-scurve` and `tab:uav-scurve-projection`, one child job per cell, and nothing that already exists is re-run. |
| Only the corrected endpoint run (`u18sc`) counts for the projected s-curve grid; its per-step arm was plain `dpcc-t` (§18) | Group E uses `dpcc-t` and tags `u18sc`. `R31_PERSTEP=dpcc-t-tightened` switches to the corridor convention — **decide before submitting E**. |
| The corridor diffusion row is the only baseline row not under plain `dpcc-t-tightened` (§15, R30) | Group C: one 12-flight diffusion job, variant `dpcc-t-tightened`, tag `u17cv2`. New cell; nothing overwritten. |
| Every alignment row is ten contexts and seed 6; the diffusion baseline has no tightened cell (§14, R2) | Group A injects `n_contexts = 10` **in memory** and runs `combined_5` with its auto-generated tightened twin. Results are tagged `_msglr22` so the published seed-6 corpus on the cluster is never overwritten. |
| R7/R14 need a locked grid; R9 needs a code change; D10c/D11/D12 are logging or rendering changes | None of them is in the driver. Listed under §2 so they are not mistaken for forgotten. |

---

## 1 · Decision and order

Groups are lettered in ledger priority. **Default wave = A B C D E** (no training except B1). F adds one
visual-aligning training. G and H are opt-in.

| group | ledger | priority | what | training | Slurm children | dependency | ~GPU time |
| :-- | :-- | :-- | :-- | :-- | --: | :-- | --: |
| **A** | R2 | 🔴 | D3IL-aligning diffusion $\nfe=20$, seed 6, **10 contexts**, `combined_5` + tightened twin, `diffuser`+`dpcc-r/c/t` | no | 1 | none | ~1.5 h |
| **B1** | R26 | 🟡 | D3IL-avoiding diffusion $\nfe=2$ **training**, seeds 7, 8, 9, 10, one job each | **4** | 4 | none | ~11 h total (2 h 41 m each) |
| **B2** | R26 | 🟡 | diffusion $\nfe=2$ evaluation, seeds 6–10, 3 geometries, 2 episodes, the stock 17-variant list, tag `_msgdpccproto` | no | 1 | `afterok` all of B1 | ~1.5 h |
| **C** | R30 | 🟢 | UAV-corridor diffusion under `dpcc-t-tightened`, 12 flights, tag `u17cv2` | no | 1 | none | ~1 h |
| **D** | R31 | 🟡 | UAV-s-curve **unprojected**: mf $\nfe=1,3,5$ · af $3$ · fm $1,3,5$, `diffuser`, 10 flights, tag `u7hg` | no | 7 | none | ~1 h total |
| **E** | R31 | 🟡 | UAV-s-curve **projected**: mf $3,5$ · af $3$ · fm $3,5$ × {`diffuser`,`dpcc-t`,`hardflow_new`,`-r`,`-c`,`-t`}, 10 flights, tag `u18sc` | no | 5 | none | ~6 h **per cell** → ~30 h |
| **F1/F2** | R16 | 🟡 | D3IL-aligning fm and af at $\nfe=10$, $T=0.4$ (the budget the existing mf $\nfe=10$ row used), 10 contexts | no | 2 | none | ~1 h each |
| **F3/F4** | R16 | 🟡 | D3IL-aligning diffusion at $\nfe=10$: **train** (`n_diffusion_steps=10`, seed 6), then evaluate | **1** | 2 | F4 `afterok` F3 | ~3 h + 1 h |
| **G** | R15 | 🟡 opt-in | UAV-s-curve MuJoCo-MPC controller at **10 flights**, mf $\nfe=10$, `diffuser`,`dpcc-r`,`hardflow_new-r`, tag `u19mjpc` | no | 1 | none | 30 min unproj. / ~6 h |
| **H** | R28 | 🟡 opt-in, **not owed** | D3IL-avoiding FM at $\nfe=5,10$, seeds 6–10, 2 episodes, tag `_msgdpccproto` | no | 1 | none | ~2 h |

Why B goes second although it is not the 🔴 item: it is the long pole (four trainings before one eval),
so it should enter the queue early; A is a single one-hour job and can run beside it.

Recommended submission if time is short:

1. `WAVE="A B"` — the blocker and the long pole (≈ 14 GPU-h, 6 jobs).
2. `WAVE="C D"` — two cheap groups that close stated gaps (≈ 2 GPU-h, 8 children).
3. `WAVE="E"` — only after the `R31_PERSTEP` choice is confirmed (≈ 30 GPU-h, 5 children).
4. `WAVE="F"` — one training plus three evaluations (≈ 6 GPU-h).
5. `G`, `H` — only with spare capacity.

Never combine training and evaluation in one allocation; B2 and F4 are separate jobs held on `afterok`.

## 2 · Deliberately not in this wave

| item | why |
| :-- | :-- |
| **R23** UAV-pillars ($\nfe=3$ rung, diffusion remainder, `pillars_xxl`) | 🔴 gate: read the four `results.json` named in [`SLURM_RUNBOOK_20260919_pillars_enlarged.md` §3e](SLURM_RUNBOOK_20260919_pillars_enlarged.md) first. If the goal-radius hypothesis holds, the wave is re-scored or re-run either way. Diffusion pillars projection is closed by decision. |
| **R25, R18**, the K20 `both-hard` repair | struck by the author (v3.52, v3.55). |
| **R7 / R14** | the K / threshold / candidate grid must be locked by the author before a driver is written; a silent reinterpretation of `tab:hf-ladder` is worse than the confound. |
| **R9** | needs an authorised code change (remove the `0.5 *` prior scale in the FM samplers, add a path token). Not a driver knob. |
| **R27** | the diffusion engine exposes no velocity field (`supports_hardflow=False`) — a model limit, not a run. |
| **R17** | run-to-run variation; optional. Note that group A's four untightened cells are a free repeat of the published diffusion $\nfe=20$ `combined_5` cells under a new tag, which already gives one such measurement. |
| **D10c / D11 / D12** | logging, timing instrumentation and rendering code; not model runs. |
| **D8 / D9** | log analysis, no GPU. |

## 3 · One-command submission

```bash
bash Slurm_Codes/temp_bash/pipeline_20260922_all_lacking_runs.sh            # PLAN — prints every job, submits nothing
bash Slurm_Codes/temp_bash/pipeline_20260922_all_lacking_runs.sh submit     # default WAVE="A B C D E"

WAVE="A B"  bash Slurm_Codes/temp_bash/pipeline_20260922_all_lacking_runs.sh submit
WAVE="F"    bash Slurm_Codes/temp_bash/pipeline_20260922_all_lacking_runs.sh submit
WAVE="G H"  bash Slurm_Codes/temp_bash/pipeline_20260922_all_lacking_runs.sh submit
```

**Always run PLAN first.** It writes the throwaway wrappers and job files (`Slurm_Codes/temp_bash/_lr22_*`)
so they can be read before anything is submitted, and prints the exact variant list, tag, seed set,
context/episode count and dependency of every job. `WAVE` is used instead of `GROUPS` on purpose —
`GROUPS` is a bash built-in.

Knobs (all optional): `ALIGN_TAG` (default `lr22`), `ALIGN_NCTX` (10), `ALIGN_SEED` (6), `R16_T` (0.4),
`R26_TRAIN_SEEDS` ("7 8 9 10"), `R26_TAG` (`dpccproto`), `R31_PERSTEP` (`dpcc-t`), `R30_TAG` (`u17cv2`),
`R31U_TAG` (`u7hg`), `R31P_TAG` (`u18sc`), `R15_TAG` (`u19mjpc`), `R28_KS` ("5 10"), `SCURVE_NTRIALS` (10),
`CORRIDOR_NTRIALS` (12), `RECORD` (`none`), `AVOID_TRAIN_HOURS` (6), `ALIGN_HOURS` (12), `FORCE_DISK=1`.

## 4 · Pre-flight gates

The driver runs these itself and aborts on failure; repeat by hand before SUBMIT:

```bash
cd ~/FMPCC/FM-PCC
git rev-parse --short HEAD; git status --short | wc -l
df -h ~/FMPCC ~/FMPCC/FM-PCC/logs
squeue -u "$USER"
```

1. **Revision.** Record the cluster revision in §7. The local tree carries uncommitted thesis/DA work
   that the cluster does not need for these jobs, but the pillars fixes and the switched-wall fix must be
   present (`update_constraint_list`, the composed-toggle allow-list — both grepped by the driver).
2. **Protocol file.** `config/projection_eval.yaml` must say `n_trials: 2` and `seeds: [6,7,8,9,10]`
   for B2 and H. The driver refuses otherwise, and B2 re-checks at job start. **Do not** flip it to 20 for
   anything — the extended protocol is retired.
3. **Contexts.** `config/visual_aligning_eval.yaml` still says `n_contexts: 3`. Every aligning job here
   injects 10 in memory through `_lr22_align_eval.py` and echoes it. Reject any result whose log does not
   show `n_contexts 3 -> 10`.
4. **Disk.** B1 writes four avoiding checkpoints (~430 MiB each); F3 writes one visual-aligning checkpoint.
   The driver warns under 4 GiB and refuses to submit B/F without `FORCE_DISK=1`.
5. **Checkpoints.** B1 skips a seed whose `H8_K2_…_aw10/<seed>` folder already holds `state_*.pt` and
   refuses to retrain over one; B2 verifies all five at job start. F4 is held on F3.
6. **No pillars.** There is no pillars group. If a pillars job is wanted, the gate in §2 applies first.
7. **GPU/EGL isolation.** Every generated job keeps the `CUDA_DEVICE_ORDER` / `MUJOCO_EGL_DEVICE_ID` block.
   UAV jobs run with `UAV_EVAL_HOURS=24`.

## 5 · Per-group specification and identity checks

### A · R2 — the tightened alignment baseline

Job `_lr22_A_r2_align_diffusion_K20.sh` → `_lr22_align_eval.py`: engine `diffusion`, seed 6,
`--eval-on-train`, `n_contexts=10`, `projection_variants=[diffuser, dpcc-r, dpcc-c, dpcc-t]`,
`active_geo_variants=[combined_5]`, tag `_msglr22`. The geo loop generates `combined_5-tightened` itself,
so the job produces 8 items; the four tightened cells are the deliverable.

Log must show:

```text
[ lr22 ] yaml n_contexts 3 -> 10
[ lr22 ] yaml projection_variants 17 entries -> ['diffuser', 'dpcc-r', 'dpcc-c', 'dpcc-t']
[ lr22 ] yaml active_geo_variants ['combined_5'] -> ['combined_5']
[ eval ] engine=diffusion ... config block: plan_mix_visual_aligning_diffusion
[ eval ] >>> item 8/8: geo=combined_5-tightened  variant=dpcc-t  tightened=True
```

Reject if it shows three contexts, the held-out split, or a `results/` root instead of `results_train_set/`.

### B · R26 — diffusion $\nfe=2$, five seeds

B1 (`_lr22_B1_r26_avoid_train_K2_s<seed>.sh` → `_lr22_avoid_train.py`): `scripts/train.py --seed <s>`
with `base['diffusion']['n_diffusion_steps']=2` patched in memory, exactly as job 25965. Six-hour limit
(2× the measured 2 h 41 m). Must print `TRAIN n_diffusion_steps -> 2` and end with a listing of
`logs/avoiding-d3il/diffusion/H8_K2_Dmodels.GaussianDiffusion_aw10/<seed>/`.

B2 (`_lr22_B2_r26_avoid_eval_K2_5seeds.sh` → `_lr22_avoid_eval.py`): `scripts/eval.py` with the plan
block's `n_diffusion_steps=2`, no `--seed` (the yaml's five), `FMPCC_RUN_MSG=dpccproto`,
`FMPCC_MPC_BATCH=4`. Must print `EVAL n_diffusion_steps -> 2`, five `seed <s>: …/state_<step>.pt` lines
and results folders ending `_msgdpccproto/<seed>`. The stock 17-variant list runs; only `diffuser` and
the three tightened rules enter the tables.

A failed B1 keeps B2 held (`afterok`); never bypass the dependency with a missing seed.

### C · R30 — corridor diffusion under the flow rows' variant

`eval_mix_uav.sh diffusion corridor 6 12 fm_only none ""` with `UAV_MIX_GEO_VARIANTS=corridor_v2_slide`,
`UAV_MIX_VARIANTS=dpcc-t-tightened`, `FMPCC_SAFE_EPS_FRAC=1.0`, `FMPCC_SAFE_EPS_MODE=scaled`, tag
`u17cv2`. Log must show `corridor_v2_slide`, engine `diffusion`, exactly one variant, results path ending
`…_u17cv2`. Expected outcome: still $0/12$ strict success (the goal-line criterion), but now matched.

### D · R31 unprojected — seven s-curve cells

`eval_k_sweep.sh <engine> s_curve 6 "<K list>" 10 fm_only none`, `UAV_MIX_GEO_VARIANTS=s_curve_hg`,
`UAV_MIX_VARIANTS=diffuser`, tag `u7hg`. K lists: mf `1 3 5`, af `3`, fm `1 3 5`. Seven children. Existing
cells (mf 2/10, af 1/2/5, fm 2/20) are not re-run. Controller is the config default (`pid_stopgo`), which
is what the existing `u7hg` rows used — do not set `UAV_MIX_CONTROLLER`.

### E · R31 projected — five s-curve cells

Same entrypoint, K lists mf `3 5`, af `3`, fm `3 5`, variants
`diffuser,dpcc-t,hardflow_new,hardflow_new-r,hardflow_new-c,hardflow_new-t`, tag `u18sc`. One child per
cell, ~6 h each (per-step `dpcc-t` at ~1.4 s/step dominates). Log must show `s_curve_hg`, `u18sc`,
`update_constraint_list` in the sampler path, and **no** `[hardflow][BLOCKED]` (K ≥ 3). `diffuser` is
included so the `u18sc` tree carries its own unprojected row beside the projected ones, as the existing
af $\nfe=5$ cell does; it duplicates a D cell by a few minutes and that is deliberate.

🟠 **Open before submitting E:** `dpcc-t` (follows `u18sc`, default) or `dpcc-t-tightened` (mirrors the
corridor). The grid's one filled projected cell is plain `dpcc-t`.

### F · R16 — the $\nfe=10$ alignment rows

F1/F2 (`_lr22_F1_r16_align_fm_K10.sh`, `_lr22_F2_r16_align_af_K10.sh`): `--flow-steps 10
--proj-threshold 0.4`, ten contexts, same variants/geo/tag as A; F2 adds `MIX_AF_ALPHA_END=0.2`,
`--epoch latest`, U-Net. Logs must show `--flow-steps: flow_steps_v3 <old> -> 10` and `T = 0.4`.

F3 (`_lr22_F3_r16_align_train_diffusion_K10.sh` → `_lr22_align_train.py`): trains the diffusion arm with
`n_diffusion_steps=10` patched on both the train and plan blocks; the checkpoint folder must carry
`_K10_`. F4 evaluates it with the same patch (`LR_DIFF_K=10`), held on F3. The existing $\nfe=20$
checkpoint is untouched.

### G · R15 (opt-in) and H · R28 (opt-in)

G: `eval_k_sweep.sh mf s_curve 6 "10" 10`, `UAV_MIX_CONTROLLER=mjpc`, variants
`diffuser,dpcc-r,hardflow_new-r`, tag `u19mjpc`. The controller is a results-path key, so it cannot
collide with the PID rows. Needs the `FMPCC_mjx` env on the node.

H: `FMV3_FLOW_STEPS="5 10" FMPCC_RUN_MSG=dpccproto eval_fmv3_ode_job.sh` — seeds 6–10 from the yaml,
2 episodes. Log must show `NFE budgets to evaluate: 5 10` and `_msgdpccproto`. Not owed; do not let it
delay A–F.

## 6 · Completion checks

A job is not complete because Slurm says `COMPLETED`. For each leaf verify:

1. the log echoes the intended checkpoint, K, seed(s), context/episode count, geometry, tag and the exact
   variant subset (the `[ lr22 ]` identity line for the wrapped jobs);
2. every expected `results.json` / NPZ exists and loads, with ten contexts (aligning), two episodes
   (avoiding) or ten/twelve flights (UAV);
3. no cell is pooled across tags, protocols or candidate counts — `_msglr22` and `_msgdpccproto` are new
   keys the DA readers must register; `u7hg`, `u18sc`, `u17cv2` pool with their existing trees;
4. B contains five distinct seeds and three geometries before the row enters Tables 6.1/6.2;
5. D and E fill only \emph{pending} cells of the s-curve grid; existing cells keep their values;
6. the draft tables/figures are rebuilt from real result data, never console logs.

Record job IDs, revision, start/end times and failures in §7. Do not touch `MASTER_TEST_HISTORY.md`
unless told to.

## 7 · Submission record

| group | job IDs | revision | state / verification |
| :-- | :-- | :-- | :-- |
| A · R2 | — | — | planned, not submitted |
| B1 · R26 train ×4 | — | — | planned, not submitted |
| B2 · R26 eval | — | — | planned, dependent on B1 |
| C · R30 | — | — | planned, not submitted |
| D · R31 unprojected ×7 | — | — | planned, not submitted |
| E · R31 projected ×5 | — | — | planned; `R31_PERSTEP` to confirm |
| F1/F2 · R16 fm/af | — | — | planned, not submitted |
| F3/F4 · R16 diffusion train → eval | — | — | planned, not submitted |
| G · R15 | — | — | opt-in, not submitted |
| H · R28 | — | — | opt-in, not submitted |
| Required result folders downloaded | **human only** | — | `XXX` |
| DA readers updated for `_msglr22`, `_msgdpccproto` (diffusion K2), new s-curve cells | human/local | — | `XXX` |
