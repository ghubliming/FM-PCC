# SLURM RUNBOOK — 2026-09-22 · the remaining thesis runs, most urgent first

> **23-09 (v3.75): every UAV-s-curve group of this runbook (D, E: R31; G: R15) is struck. The s-curve runs are R44 in [`PENDING_20260923_uav_scurve_R44_raw_first.md`](PENDING_20260923_uav_scurve_R44_raw_first.md).**

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
| ~~**C**~~ | ~~R30~~ | ⛔ | **superseded 23-09** — the corridor is re-run as corridor v3 (R33) and pillars v2 is R39: `SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md` | — | — | — | — |
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
2. `WAVE="D"` — one cheap group that closes a stated gap (C is superseded, see the 23-09 runbook).
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
`CORRIDOR_NTRIALS` (12), `RECORD` (`none`), `AVOID_TRAIN_HOURS` (12), `AVOID_EVAL_HOURS` (12), `ALIGN_HOURS` (12), `FORCE_DISK=1`.

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
with `base['diffusion']['n_diffusion_steps']=2` patched in memory, exactly as job 25965. Twelve-hour limit (author: every job in the 12–24 h window; job 25965 took 2 h 41 m)
(2× the measured 2 h 41 m). Must print `TRAIN n_diffusion_steps -> 2` and end with a listing of
`logs/avoiding-d3il/diffusion/H8_K2_Dmodels.GaussianDiffusion_aw10/<seed>/`.

B2 (`_lr22_B2_r26_avoid_eval_K2_5seeds.sh` → `_lr22_avoid_eval.py`): `scripts/eval.py` with the plan
block's `n_diffusion_steps=2`, no `--seed` (the yaml's five), `FMPCC_RUN_MSG=dpccproto`,
`FMPCC_MPC_BATCH=4`. Must print `EVAL n_diffusion_steps -> 2`, five `seed <s>: …/state_<step>.pt` lines
and results folders ending `_msgdpccproto/<seed>`. The stock 17-variant list runs; only `diffuser` and
the three tightened rules enter the tables.

A failed B1 keeps B2 held (`afterok`); never bypass the dependency with a missing seed.

### C · R30 — corridor diffusion under the flow rows' variant — ⛔ SUPERSEDED 23-09 (see `SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md`; do not submit)

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

## 6b · 23-09 · the MUST-NEED wave: R16 + R36 (✅ complete 24-09: jobs 26181–26186; DA [`DA_20260924_R16_R36_must_need.md`](../../../../Data_Analysis/DA_in_Paper/analysis/DA_20260924_R16_R36_must_need.md))

Author, 23-09: *"prepare tempbash/command for the runs that must need runs as showing in the Pending md,
and mark them scheduled. optional NO. first clean all the must need runs."* After R2, R37, R26 and R35
closed, the NOW table's must-need items outside the quadrotor scenes (corridor R33, s-curve R44 and
pillars R39 live in their own files) are **R16 and R36**. R43 is optional and not included.

**Checked against the corpus first** (R37a lesson): none of the six cells exists. R16 — no FM K2, FM K10
or CI-MeanFM K10 folder in `batch_va2_20260923_210100`. R36 — no CI-MeanFM K3, FM K3 or MeanFM K10 `B4`
cell in the TenpK2D avoiding batch; MeanFM K10 exists only as `B1`, as Table 6.3 says.

**Driver:** `Slurm_Codes/temp_bash/pipeline_20260923_must_need.sh` — one self-contained file; it writes its
`_mn23_*` wrappers, sbatch files and pruned yamls itself. PLAN by default.

```bash
bash Slurm_Codes/temp_bash/pipeline_20260923_must_need.sh            # PLAN
bash Slurm_Codes/temp_bash/pipeline_20260923_must_need.sh submit     # all six, serial chain
ANCHOR=<jobid> bash … submit    # first job waits for <jobid>;  SERIAL=0 for no chain
```

| job | ledger | table | cell | how it is matched to the printed rows |
| :-- | :-- | :-- | :-- | :-- |
| R16a | R16 | 6.5 | D3IL-aligning FM K2, unprojected | `diffuser` on `combined_5`, ten contexts, η 0.5 names the folder (as the other K2 rows), config-default checkpoint as the FM rows, tag `_msgR16` |
| R16b | R16 | 6.5 | D3IL-aligning FM K10, unprojected | same, η 0.4 (as MeanFM K10) |
| R16c | R16 | 6.5 | D3IL-aligning CI-MeanFM K10, unprojected | `MIX_AF_ALPHA_END=0.2` → checkpoint `…_afschsigmoid_AFAFend0p2`, `--epoch latest`, as the printed CI-MeanFM rows |
| R36a 🔄 v3.77 | R36 | 6.3 | D3IL-avoiding CI-MeanFM K3, **seeds 7–10** | `eval_alphaflow_hardflow.sh`, `AF_BONE=unet AF_ALPHA_END=0.2 AF_EPOCH=latest`, `--flow-steps 3`, activation 1.0 (two guiding steps, as MeanFM K3), `HFFM_BATCH=4 FMPCC_MPC_BATCH=4` |
| R36b 🔄 v3.77 | R36 | 6.3 | D3IL-avoiding FM K3, **seeds 7–10** | `eval_fmv3_hardflow_job.sh`, `HFFM_FLOW_STEPS=3`, activation 1.0, both fans 4 |
| R36c | R36 | 6.3 | D3IL-avoiding MeanFM K10, four candidates | `eval_meanflow_hardflow.sh`, `MF_FLOW_STEPS=10 MF_BACKBONE=unet MF_HORIZON=8` (job 25444's knobs), activation 0.5 (four guiding steps), both fans 4 |

**🔄 Updated v3.77 (23-09):** R36a and R36b run on **seeds 7–10**, the seeds of the printed MeanFM K3 row of Table 6.3
(job 25444, which does not contain seed 6), so the K3 block is paired on one seed set; R36c stays at seed 6, the seed of
the printed K10 rows. The driver pins the lists per pruned yaml (`SEEDS_K3`, default `7, 8, 9, 10`; tested on scratch
copies: the alphaflow and hardflow copies read `seeds: [7, 8, 9, 10]`, the meanflow copy `seeds: [6]`). Before: every
job at seed 6. R36 runs three geometries × two episodes per seed, with `dpcc-{r,c,t}-tightened` against
`hardflow_new-{r,c,t}-tightened` (folders land as `hardflow_sls-*`), tag `_msgR36`. Seeds, episodes and
variants come from a **pruned copy** of each entrypoint's own yaml, passed as `--config` — the hook all three
evals document. The copies were parsed and diffed: only `projection_variants` differs from the shared yaml,
which is never edited.

**Verified before handing over:** syntax of the driver and every generated file; the pruned yamls parse
and change only the variant list; and a dry run against a stand-in `sbatch` showed every job submitted
with an explicit `--dependency=afterok:<previous>` and the intended environment: fans 4/4, the activation
threshold, K, `AF_BONE/AF_ALPHA_END/AF_EPOCH = unet/0.2/latest` on R36a only, and the tag.

**✅ Re-verified 23-09 after the v3.77 edit — ready to submit.** Driver syntax, PLAN run, every generated sbatch
file and the wrapper pass; the pruned yamls parse and differ from the shared ones only in `projection_variants`
(and `seeds` for the two K3 copies: `[7, 8, 9, 10]`; MeanFM K10 `[6]`). **Checkpoints exist for every new seed**,
read from the corpus: CI-MeanFM `…bbunet…ae0.2…` seeds 6–10 (trained by 25879); FM seeds 6–10 under
`flow_matching_v3_ode_selectable/…aw10`, which is exactly the load path of `plan_fm_v3_hardflow` (config/avoiding-d3il.py,
"copied from plan_fm_v3_ode_selectable"); MeanFM `…bbunet…dp0.5` seeds 6–10, so R36c's seed 6 loads although job 25444
did not evaluate it. A dry run against a stand-in `sbatch` submitted all six with an explicit `afterok` chain and the
intended environment (fans 4/4, activation 1.0/1.0/0.5, K 3/3/10, `AF_* = unet/0.2/latest` on R36a only, tag `R36`).

**Cost:** about 2–5 h in total even fully serial (R36a/b four seeds each since v3.77, ~40 min–2 h per cell). R16 jobs get 6 h; R36 keeps its
entrypoints' 24 h.

**Identity checks in the logs:** R16 — `[ mn23 ] identity: … n_contexts=10 … tag=_msgR16` (R16c also
`af_alpha_end=0.2, epoch=latest`). R36 — `[ hardflow ] HFFM_BATCH=4 … FMPCC_MPC_BATCH=4 …
HFFM_ACT_THRESHOLD=<1.0|0.5>`, `[ eval ] config: …_mn23_R36_<…>.yaml`, a savepath ending `_msgR36`,
`hf_n_genuine` = 2 at K3 and 4 at K10, and no `[hardflow][BLOCKED]`; R36a/b show four seed blocks (7, 8, 9, 10), R36c seed 6 only.

## 7 · Submission record

### 🔴 23-09 · two defects in this wave, and what survives of the driver

**1 · The B2 dependency never attached (my error).** `submit_file()` passed the dependency as the
environment variable `SBATCH_DEPENDENCY` through `submit.sh` instead of the explicit `--dependency=`
flag. Job **26056** was therefore queued with reason `QOSMaxCpuPerUserLimit`, never `(Dependency)`, and
**started while its training job 26055 was still running**. Its own pre-flight only asked whether *any*
`state_*.pt` existed, and training writes one every 1000 steps, so a half-trained seed 10 passed the
check. The author cancelled it after 45 min. **Re-submitted as 26112** with
`./Slurm_Codes/submit_after.sh 26055 …`, the repo's own dependent wrapper, which uses the explicit flag.
**Rule for every future driver: dependencies go through `Slurm_Codes/submit_after.sh`, never through an
environment variable, and a checkpoint guard compares the step number against a completed reference
seed rather than testing for existence.** Both fixes are in the local driver; the cluster copy predates them.

**2 · 🟠 Job 26051 (A · R2) ran at the wrong projection threshold.** It took η from the shared yaml
(`diffusion_timestep_threshold: 0.5`) because group A passed no `LR_T`, and wrote
`…/H8_K20_**T0.5**_…_msglr22/6`. Table 6.8 is **η = 0.2** (v3.66: "the operating point only, K = 20,
η = 0.2"), and its three flow rows are all keyed `_T0.2_`. The cells 26051 produced are tightened, K20,
`dpcc-r/c/t`, ten contexts — everything except the threshold. **R2 is therefore not closed by 26051
unless the author decides the diffusion row keeps DPCC's native η = 0.5.** A re-run at η = 0.2 costs
less than the original (5 projected steps per replan instead of 11): ~4–5 h. Decision needed.

### 23-09 · the RED wave, submitted as one serial chain

> ✅ **ALL FOUR LINKS COMPLETE 23-09, 18:11 UTC; DA done** on `batch_va2_20260923_210100`: [`DA_20260923_R2fix_R37_aligning.md`](../../../../Data_Analysis/DA_in_Paper/analysis/DA_20260923_R2fix_R37_aligning.md). R2 and R37 are closed.
> `26112 → 26113 (R2fix) → 26114 (R37a) → 26115 (R37b)`, every link
> `afterok` on the one before it, all through `submit_after.sh`. Anchor 26112 was `PENDING` at
> submission. The cluster yaml reads `n_contexts: 10`, so the in-memory injection is a no-op
> safeguard there.

Author, 23-09: *"set the dependency chain, the V_A rerun after 26112 then the other red alert run
after the V_A run"*, and *"others we will run later, but mark/remember in the pending md/runbook"*.

```
26112  B2 · R26 diffusion K2 five-seed eval        [already queued, afterok:26055]
  └─►  R2fix   aligning diffusion K20 @ eta 0.2     ~4.5 h   tag _msgR2fix
         └─►  R37a  MeanFM K2   eta 0.5, per-step   ~0.5 h   tag _msgR37
                └─►  R37b  MeanFM K100 eta 0.1, per-step + endpoint  ~5.6 h   tag _msgR37
```

**ONE file to copy to the cluster:** `Slurm_Codes/temp_bash/pipeline_20260923_red_wave.sh`. It
writes every wrapper and sbatch file it needs (`_rw23_*`) at run time, exactly as the 22-09
driver did, so nothing is referenced by filename and the driver can be renamed freely. An
earlier three-file version (a chain plus two link scripts) was deleted: the links were looked up
by name, and the first submission attempt failed on the cluster because only the chain had been
copied across. Author: *"they are created from the last temp bash? can you just reuse redo the
same way? dont find it. it is unreliable."*

```bash
bash Slurm_Codes/temp_bash/pipeline_20260923_red_wave.sh                  # PLAN
ANCHOR=26112 bash Slurm_Codes/temp_bash/pipeline_20260923_red_wave.sh submit
ANCHOR=none  bash ... submit      # if 26112 has already finished and left the queue
```

Every link is submitted through `Slurm_Codes/submit_after.sh`, never the environment variable.
~11 h of compute after 26112.

| tag | what it marks |
| :-- | :-- |
| `_msgR2fix` | the eta = 0.2 diffusion row of Table 6.8. 26051's `_msglr22` eta = 0.5 folders are left alone |
| `_msgR37` | the tightened threshold-ladder cells of Table 6.6 |

#### 🔴 R37c does not fit the cluster and is NOT queued

> **Author's decision, 23-09 (v3.70): R37c is struck — "clearly not possible, mark impossible".** Nothing to run.
> The thesis (§6.2.2, Table 6.6) prints the K10/K20 tightened cells, marks K100 η0.5 *not run* and prices it by the
> rule of thumb (solves per control step × per-solve cost from the K10/K20 cells); R37a/b stay queued as submitted.

The third pair of the ladder, K = 100 at eta = 0.5, costs far more than the ledger's "~2 h"
estimate for all of R37. From the ledger's own measurement (§14): 50 guiding steps, **15,218 ms
per control step**. An alignment cell is 400 steps x 10 contexts:

| pair | ms/step | one item | items (plain + tightened) | job |
| :-- | --: | --: | --: | --: |
| K2 eta0.5 per-step | ~191 | 0.2 h | 2 | **0.5 h** |
| K100 eta0.1 per-step | 1,195 | 1.3 h | 2 | 2.7 h |
| K100 eta0.1 endpoint | 1,309 | 1.5 h | 2 | 2.9 h |
| **K100 eta0.5, either arm** | **15,218** | **16.9 h** | **2** | **🔴 33.8 h** |

The geo loop always runs the plain geometry beside the tightened twin, and there is no
tightened-only switch, so even a single variant is two items. 33.8 h is over the 24 h cap.
The driver refuses it unless `WAVE="… R37c" ALLOW_OVERCAP=1`. Three ways out, all the author's call:

1. **drop it** — §14 already carries the eta0.5-vs-eta0.1 cost evidence at K = 100 (12.7x the
   cost for 6 mm of final distance), which is the argument §6.2.2 actually makes;
2. **authorise skipping the plain twin** (a change to the geo loop) — each variant then fits one
   24 h job at ~17 h;
3. **fewer contexts for that pair only** — breaks the ten-context protocol.

### 🟡 Queued for the next parts, not submitted now

**R41 (v3.69, 23-09) — the step-budget grid, evaluation only.** Same driver pattern as the 17-09 protocol wave
(`_msgdpccproto`, `diffuser` + `dpcc-r/c/t` tightened, mpc4, T0.5, 2 episodes per geometry, seeds 6–10): MeanFM K20;
CI-MeanFM α0.2 K5, K10, K20; FM K5, K10. ~3 GPU-hours. Diffusion K5 (five trainings) only if the author asks. Ledger §28.


Marked here so they are not lost. One part at a time, as agreed.

| next | ID | what | ~GPU | entrypoint |
| :-- | :-- | :-- | --: | :-- |
| part 2 | **R16** | aligning unprojected: CI-MeanFM K10; FM K2, K10 | ~1 h | same aligning eval; three cells, `diffuser` only |
| part 2 | **R35** | aligning endpoint on CI-MeanFM K20, tightened, r/c/t | ~1 h | same eval + `HFFM_VARIANTS` |
| part 3 | **R36** | avoiding matched 4-candidate: CI-MeanFM + FM at K3, MeanFM K10, seed 6 | ~2 h | `eval_dpcc_job.sh` family, `FMPCC_MPC_BATCH=4` |
| separate | R33, R39, R40 | corridor v3, pillars v2, s-curve caveat | ~2 GPU-days | the **23-09** file and its own runbook — not this ledger |

R16 and R35 are both the aligning entrypoint at seed 6 and should be one script, built after
the red wave lands so the spec cannot shift under it.

### Which groups of this driver are still alive (23-09 NOW table)

| group | status |
| :-- | :-- |
| A · R2 | ⚠ alive but **must pass η = 0.2**; 26051's T0.5 output is a different cell |
| B · R26 | ✅ running: trainings 26052–26055, eval **26112** on `afterok:26055` |
| C · R30 | — | — | ⛔ superseded 23-09 — the corridor is R33 in the 23-09 file |
| D / E · R31 | ⛔ struck at v3.68 — the s-curve K ladder is no longer printed |
| F · R16 | ⚠ rework: R16 is now three unprojected cells (CI-MeanFM K10; FM K2, K10). F3/F4, the diffusion K10 training, is struck — Table 6.5 keeps diffusion at K20 only |
| G · R15 | — | — | ⛔ struck 23-09 — the MJPC caveat is a demo |
| H · R28 | — | — | ⛔ struck 23-09 — not owed |

**Still owed in THIS ledger** (R33/R39/R40 belong to the 23-09 file): R37 🔴, R2 🔴, R16 🟡, R35 🟡, R36 🟡.


**2026-09-22 submit, `WAVE="A B"`:** PLAN then SUBMIT from the cluster copy `Slurm_Codes/temp_bash/22-09-pending.sh`; pre-flight passed at revision 999152f1 (38 dirty paths, docs only), 174 GB free. Six jobs accepted in order A → B1 ×4 → B2. Groups C, D, E, F, G, H remain unsubmitted.

| group | job IDs | revision | state / verification |
| :-- | :-- | :-- | :-- |
| R16a / R16b / R16c · R16 | **26181** · **26182** · **26183** | 5092e056 | ✅ **COMPLETE 24-09, verified** (logs + DA), serial chain: 26181 (FM K2, no dependency) → 26182 (FM K10, `afterok:26181`) → 26183 (CI-MeanFM K10, `afterok:26182`); seed 6, tag `_msgR16`; Slurm names `_mn23_R16a_fm_K2`, `_mn23_R16b_fm_K10`, `_mn23_R16c_af_K10` |
| R36a / R36b / R36c · R36 | **26184** · **26185** · **26186** | 5092e056 | ✅ **COMPLETE 24-09, verified** (logs + DA; FM data under `K3_thres1_mpc4_n2_msgR36`, its parent folder is misnamed K10 by the evaluator), continuing the chain: 26184 (CI-MeanFM K3, seeds 7–10, `afterok:26183`) → 26185 (FM K3, seeds 7–10, `afterok:26184`) → 26186 (MeanFM K10, seed 6, `afterok:26185`); tag `_msgR36`; Slurm names are the entrypoints' own: `eval_alphaflow_hardflow`, `eval_fmv3_hardflow_job`, `eval_meanflow_hardflow` |
| A · R2 | **26051** | 999152f1 | ⚠ **η = 0.5, not the 0.2 Table 6.8 needs** (see above). RUNNING since 07:57 UTC; identity lines correct (`n_contexts 10 -> 10`, `combined_5` + twin, 4 variants, `_msglr22`). **Measured: ~18 min per unprojected item, ~75 min per projected item → ~8 h for the 8 items**, not the ledger's 1.5 h (the 265–450 ms/step figure is per replan, not per rollout wall time). Expected end ~16:00 UTC, inside the 12 h limit |
| B1 · R26 train ×4 | **26052** s7 · **26053** s8 · **26054** s9 · **26055** s10 | 999152f1 | submitted 2026-09-22, 6 h limit each (the cluster copy predates the 12 h default; 2.2× the measured 2 h 41 m) |
| B2 · R26 eval | ~~26056~~ → **26112** | 999152f1 | ✅ **COMPLETE 2026-09-23.** 5 seeds × 3 geometries × 2 episodes, 13 variants, no missing seeds. R26 is **data-complete**; preliminary DA in [`DA_20260923_diffusion_K2_five_seeds.md`](../../../../Data_Analysis/DA_in_Paper/analysis/DA_20260923_diffusion_K2_five_seeds.md). 26056 had run unchained and was cancelled (see above); 26112 was re-submitted via `submit_after.sh 26055` |
| **R2fix · R2** | **26113** | 1e8e707d | ✅ **COMPLETE, verified 23-09** (07:25–12:31 UTC). Log shows `eta 0.2 (source: cli --proj-threshold)`, savepath `H8_K20_T0.2_…_msgR2fix/6`, 10 contexts, 8/8 items incl. the tightened `dpcc-r/c/t`. Needs the V_A DA for numbers |
| **R37a · R37** | **26114** | 1e8e707d | ✅ **COMPLETE, verified 23-09** (12:31–12:48 UTC). K2, η 0.5, savepath `H8_K2_Meuler_T0.5_…_msgR37/6`, 2/2 items (plain + tightened `dpcc-r`) |
| **R37b · R37** | **26115** | 1e8e707d | ✅ **COMPLETE, verified 23-09** (12:48–18:11 UTC). K100, η 0.1, savepath `H8_K100_Meuler_T0.1_…_msgR37/6`, 4/4 items (`dpcc-r` + `hardflow_new-r`, plain + tightened). 🟠 the endpoint arm logged a non-converged SLSQP solve (`[hardflow][NLP-FAILURE]` at τ 0.910, both geometries); the total is not in the log or the DA CSV |
| ~~R37c · R37~~ | — | — | ⛔ **struck by the author, 23-09 (v3.70):** *"clearly not possible, mark impossible"* — ~33.8 h per variant against a 24 h cap; the thesis prices the pair by rule of thumb instead |

🟠 **Queue note, 2026-09-22 13:15 UTC:** the account runs under `QOSMaxCpuPerUserLimit` — two 8-CPU jobs at a time. 26051 (5 h 18 m) and 26053 (s8, 1 h 06 m) were running; 26054/26055/26056 pending on the QOS. Chained on two slots the wave ends ~20:30 UTC. If other runs are more urgent, `scontrol hold 26054 26055 26056` (nothing lost, dependency intact) and `release` later; never cancel s9/s10 alone, that strands B2 as `DependencyNeverSatisfied`.
| C · R30 | — | — | planned, not submitted |
| D · R31 unprojected ×7 | — | — | ⛔ struck v3.68 — the s-curve K ladder is no longer printed |
| E · R31 projected ×5 | — | — | ⛔ struck v3.68 — same |
| F1/F2 · R16 fm/af | — | — | ⚠ superseded — R16 is now three unprojected cells; queued as part 2 |
| F3/F4 · R16 diffusion train → eval | — | — | ⛔ struck v3.63 — Table 6.5 keeps diffusion at K20 only |
| G · R15 | — | — | opt-in, not submitted |
| H · R28 | — | — | opt-in, not submitted |
| Required result folders downloaded | **human only** | — | `XXX` |
| DA readers updated for `_msglr22`, `_msgdpccproto` (diffusion K2), new s-curve cells | human/local | — | `XXX` |
