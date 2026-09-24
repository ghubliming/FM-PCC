# PENDING 23-09 — UAV-s-curve: the raw generative grid first, projection afterwards (R44)

> **Status 2026-09-24 (v3.92): R44 COMPLETE and in the draft.**
> - C1 is job 26204. Table 6.16 is complete, and Figure 6.9 is rebuilt on FM nfe 1 and exported.
> - The s-curve conclusion is final. The finish line at the end of the walls (x = 3.0 m) changes 0 of 160 flights.
> - Nothing is open. The setpoint-bound per-step run (DA §7) stays optional and is the author's call.

> **Status 2026-09-24 (v3.91):** R44b (job 26195) and C0 (job 26196) are in the draft.
> - Table 6.15 is complete. Table 6.16 has three rows; the per-step rule is random, chosen by success, then aborts,
>   then violating steps. Table 6.17 is on FM nfe 1, ten flights against ten.
> - Analysis of record: `DA_20260924_scurve_R44bc_projection_controller.md`.
> - **Open:** C1 (per-step random under MuJoCo MPC) and the rebuilt Figure 6.9. The optional setpoint-bound per-step
>   run (DA §7) is the author's call.

**Author, 2026-09-23 (review of the v3.74 bundle).** "we need the K1,2,20 for af/mf/FM runs, K20 diffu not … it is pure NN
output, it should be quick, build the table, mark as pending" · "why … the weird K10,5 runs? … rebuild in my way and I will
run the slurm" · "Table 6.15 feels so random. why K2, the HF is even not on in K2. we may choose first the raw NN output then
decide on which model run the projection … build a NEW s_curve pending md, first run the raw NN then after decide run the
projection. I will delegate another agent to do the runs."

**This file is the only live list of s-curve runs.** It replaces R42 of
`PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md` §4 with its runbook section
`SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md` §3b (v3.71), and R15, R29, R31, R38 and R40 of
`PENDING_20260922_all_lacking_runs.md`. None of those is run.

> **Driver and runbook (23-09, Claude):** `Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh` (`plan` · `submit` ·
> `status` · `export`; phases A/B/C) with [`SLURM_RUNBOOK_20260923_uav_scurve_R44.md`](SLURM_RUNBOOK_20260923_uav_scurve_R44.md).
> Same cells and environment as below. Every job carries its tag as the job name as well (`p23scgrid_A1_mf_K1`, …), so
> `squeue`, the log files and the results folders all show the marker. Phase A submits all ten cells by default
> (`CELLS=required` = A1 A3 A6 A7). B and C stop at a gate until `SC_ENGINE` / `SC_K` (/ `SC_RULE`) are given.
> **Phase A SUBMITTED 2026-09-23 21:26, all ten cells: jobs 26167–26176** (runbook §6). B and C are not submitted.
> **✅ Phase A DONE (24-09):** every check passes, and the six old cells reproduce flight for flight. **Selection: FM at
> nfe = 1 (9/10 crossing).** DA `DA_in_Paper/analysis/DA_20260924_scurve_R44a_raw_grid.md`; v3 notified
> (`cross_draft/to_v3/FROM_DA_20260924_scurve_R44a_ready.md`). **Author picked option (ii), FM nfe 1: B1 submitted 24-09 11:27 (job 26195), C0 (job 26196) beside it; C1 after B1** (runbook §6).
> **24-09: B1 and C0 done and analysed** (`DA_in_Paper/analysis/DA_20260924_scurve_R44bc_projection_controller.md`; v3 notified). **C1 = `dpcc-r-tightened` under MuJoCo MPC — job 26204, ✅ done.** 
> **✅ R44 COMPLETE (24-09): every s-curve cell of Ch 6 is flown and analysed** (Tables 6.14–6.16, Fig. 6.9 rebuilt, Table 6.17 values on the selected configuration) — `DA_in_Paper/analysis/DA_20260924_scurve_R44bc_projection_controller.md`; v3 notified. Nothing further is required; the setpoint-bound projection run (DA §7) stays optional, the author's call. Optional, author's call: a setpoint-bound per-step run (`-bounds_free-pdes-tightened`), DA §7.
> **✅ Phase A read into the draft at v3.80 (24-09):** Table 6.15 `tab:uav-scurve` complete from tag `p23scgrid` (FM K1 marked as the
> selection); Table 6.16 `tab:uav-scurve-projection` prints the none row and the endpoint line ("no guiding step at K = 1"), the three per-step rows
> *pending (R44b)*; Table 6.17 `tab:uav-controller` prints the cascaded-none cell, the other three *pending (R44b/c)*; Ch 5 protocol now says
> per-step only for the selected configuration. **Open: B1 and C0/C1 only.** (Numbers as built at v3.80 — the corridor gained a
> table, so the s-curve tables are 6.15/6.16/6.17; the spec below keeps the v3.75 numbers 6.14/6.15/6.16.)

## 0 · Rules

1. **Three phases, in order.** Phase A runs now. **Phases B and C are not submitted until the author has read Phase A and
   named the model and the budget** (§2).
2. Scene `s_curve`, geometry `s_curve_hg`, seed 6, **ten flights per cell** (one route), `record=none`.
3. Controller: the config default in Phases A and B — the cascaded geometric controller, run tag `pid_stopgo`; do **not**
   set `UAV_MIX_CONTROLLER`. Phase C sets `UAV_MIX_CONTROLLER=mjpc`.
4. **Grid: MeanFM, CI-MeanFM (α_end 0.2, U-Net, latest epoch) and FM at K ∈ {1, 2, 20}; diffusion at K = 20 only**, the
   budget it was trained at. **No K = 5 or 10.**
5. **New tags, never pooled with `u7hg`, `u18sc` or `EPlatest_u6unet_ae02`:** `p23scgrid` (A), `p23scproj` (B),
   `p23scmjpc` (C), set through `FMPCC_UAV_EVAL_TAG`.
6. Tightened only in B and C; `-tightened` is always the last token of a variant.

## 1 · Phase A — the raw generative grid (Ch 6, Table 6.14 `tab:uav-scurve`) — RUN NOW

Unprojected (`UAV_MIX_VARIANTS=diffuser`). The chapter prints the four missing cells as *pending (R44a)*; the six that
exist are printed from the old tags until Phase A replaces them.

| # | model | K | today | run |
| :-- | :-- | --: | :-- | :-- |
| A1 | MeanFM (`mf`) | 1 | missing | **required** |
| A2 | MeanFM | 2 | `u7hg`: 0/10 cross, 9/10 aborted | re-fly (recommended) |
| A3 | MeanFM | 20 | missing | **required** |
| A4 | CI-MeanFM (`af`) | 1 | `EPlatest_u6unet_ae02`: 6/10 cross | re-fly (recommended) |
| A5 | CI-MeanFM | 2 | `EPlatest_u6unet_ae02`: 1/10 cross | re-fly (recommended) |
| A6 | CI-MeanFM | 20 | missing | **required** |
| A7 | FM (`fm`) | 1 | missing | **required** |
| A8 | FM | 2 | `u7hg`: 7/10 cross | re-fly (recommended) |
| A9 | FM | 20 | `u7hg`: 6/10 cross | re-fly (recommended) |
| A10 | diffusion | 20 | `u7hg`: 0/10 cross, 8/10 aborted | re-fly (recommended) |

**Result (tag `p23scgrid`, batch `batch_uav_20260924_081422`), success / aborted of ten; S&C 0/10 in every cell:**

| # | model | K | success | aborted | distance [m] | ms/step |
| :-- | :-- | --: | --: | --: | --: | --: |
| A1 | MeanFM | 1 | **3/10** | 7/10 | 2.023 | 9.2 |
| A2 | MeanFM | 2 | 0/10 | 9/10 | 2.515 | 17.9 |
| A3 | MeanFM | 20 | **0/10** | 10/10 | 2.772 | 176.4 |
| A4 | CI-MeanFM | 1 | 6/10 | 4/10 | 1.306 | 9.3 |
| A5 | CI-MeanFM | 2 | 1/10 | 7/10 | 2.082 | 18.0 |
| A6 | CI-MeanFM | 20 | **1/10** | 9/10 | 2.511 | 180.4 |
| A7 | FM | 1 | **9/10** | 1/10 | 0.571 | 8.9 |
| A8 | FM | 2 | 7/10 | 3/10 | 1.110 | 17.2 |
| A9 | FM | 20 | 6/10 | 4/10 | 1.193 | 168.3 |
| A10 | diffusion | 20 | 0/10 | 8/10 | 2.432 | 174.3 |

Bold: the four cells that were missing. The re-flown six are identical, flight for flight, to their old tags: the
evaluation is deterministic for a fixed configuration.

**Required: A1, A3, A6, A7.** Recommended, the author's call: the other six under the same tag, so that the whole table
comes from one tag. *Checkpoint question, answered from the job logs (23-09):* `EPlatest_u6unet_ae02` (jobs 25441–25443)
and `EPlatest_u7hg` (25634, 25637) loaded the **same file**, `mix_uav_af/…AlphaFlowODE_9D_as1_ae0.2_bbunet/6/state_100000.pt`.
The re-fly confirms that it is still the file on disk; `status` prints the step each job loaded. An
unprojected flight costs the planner time only (9–174 ms per control step, at most 871 steps): about 3 min per cell at
K = 1, 5 min at K = 2, 25 min at K = 20. Required cells ≈ 1 GPU-h, the full grid ≈ 2 GPU-h, far less wall time in parallel.

```bash
# from the repo root; the sweep submits one child job per K and exits
export FMPCC_UAV_EVAL_TAG=p23scgrid UAV_MIX_GEO_VARIANTS=s_curve_hg UAV_MIX_VARIANTS=diffuser
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf s_curve "6" "1 20" 10 fm_only none   # A1 A3   ("1 2 20" adds A2)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh fm s_curve "6" "1"    10 fm_only none   # A7      ("1 2 20" adds A8 A9)
UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest \
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af s_curve "6" "20"   10 fm_only none   # A6      ("1 2 20" adds A4 A5)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh diffusion s_curve "6" "20" 10 fm_only none   # A10 (recommended)
```

The four lines above are the spec's command shape. **Submit with the driver instead** (runbook §2): same cells and
environment, plus `FMPCC_SAFE_EPS_MODE=scaled FMPCC_SAFE_EPS_FRAC=1e-3` pinned and `UAV_EVAL_HOURS` replaced by an explicit
`--time=24:00:00`. The driver's `status` mode runs the checks below. Its `export` mode packs only this tag's result
folders and job logs, instead of the whole `logs/` tree that `export_to_laptop.sh` archives.

**Check each child log before counting it:** `s_curve_hg`, tag `p23scgrid`, exactly one variant `diffuser`, `n_trials 10`,
the intended K and engine (CI-MeanFM: `unet`, α_end 0.2, epoch latest), controller `pid_stopgo`, and ten rollouts written.
Download with `Slurm_Codes/download_remote_logs/export_to_laptop.sh`; do not pool with older tags.

**What the DA produces for the table** (per cell, from `per_rollout_detail.csv`): Success = `success_relaxed` (x/10),
S&C = `success_relaxed_and_constraints` (x/10), Distance = mean `goal_dist`, Aborted = `divergence_aborted` (x/10),
ms/step = mean `avg_time_ms`. `DA_in_Paper/analysis/uav_results.py` filters the s-curve block on tag `u18sc` (l.81):
extend it to `p23sc*` before reading.

## 2 · After Phase A — the author decides; nothing further is submitted before that

The chapter's rule: the projection is flown on the model and budget whose plans cross the finish line on the most flights
(an aborted flight never crosses). Two facts bound the budget at which the two projection methods can be compared:

- **Endpoint projection needs guiding steps.** At the default activation threshold 0.5 the evaluation drops the endpoint
  (HardFlow) variants at K ≤ 2 as degenerate (`HF_DEGENERATE_SKIPPED.txt`; header of `eval_k_sweep.sh`), and K = 3–4 has
  a single guiding step. **In this grid only K = 20 has guiding steps (nine).**
- Hence: if the winner is at K = 20, Phase B runs there with both methods. If the winner is at K = 1 or 2, the author
  chooses between (i) Phase B on that model at K = 20, both methods, or (ii) per-step projection only, at the winning
  budget, with the endpoint rows printed as "no guiding step".

### 2a · The pick after Phase A (Claude's pitch, 24-09) — the author decides

The winner is **FM at K = 1** (9/10), so the rule above applies: K = 1 has no guiding step. Both options, costed:

| | (ii) **FM K = 1, per-step only — recommended** | (i) FM K = 20, both methods |
| :-- | :-- | :-- |
| follows the chapter's rule (most crossings) | yes (9/10) | no (6/10; chosen because K = 20 has nine guiding steps) |
| Table 6.15 | none + per-step r/c/t; endpoint rows "no guiding step" | none (= A9) + per-step r/c/t + endpoint r/c/t |
| Table 6.16 | on FM K = 1 | on FM K = 20 |
| jobs | B1 (1 job) + C0 + C1 | Br, Bc, Bt (3 pair jobs, 10–20 h each, flagged over 12 h) + C0 + C1 (~12 h) |
| GPU / wall | **~2–2.5 GPU-h / ~2 h** | ~40–70 GPU-h / 1–2 days |
| expected, from cells already on disk | per-step: FM K = 2 tightened `u7hg` crossed 3, 5, 5 of 10, S&C 0 | per-step: 0–2/10 crossing, 8–10 inverted at 3.2–4.7 s per step; endpoint: 2–8/10; S&C 0 (plain sets, `u7hg` / `u18sc`) |

**Why (ii).**
- S&C is expected to be 0 under both options: flights track at 0.30 m against the route's 0.12 m clearance. Table 6.15
  therefore cannot separate the two projection methods on this scene in the column that matters.
- The endpoint-versus-per-step comparison is already carried by D3IL-avoiding, D3IL-aligning and corridor v3 (where FM
  K = 20 endpoint is the hump's best, 8/10).
- The scene's role in the thesis is the controller caveat. Table 6.16 on the winner asks exactly that question: does
  MuJoCo MPC turn 9/10 crossings into violation-free crossings?

**Commands for (ii).** B1 and C0 are independent, so submit them together:

```bash
PHASE=B SC_ENGINE=fm SC_K=1 bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh submit                     # B1: diffuser + dpcc-r/c/t-tightened, ~1 h
PHASE=C SC_ENGINE=fm SC_K=1 SC_RULE=dpcc-t-tightened CELLS=C0 bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh submit   # C0: diffuser under MuJoCo MPC (SC_RULE unused by C0)
# after B1: the per-step rule with the most crossings (ties: fewer aborted, then fewer violating steps, then distance)
PHASE=C SC_ENGINE=fm SC_K=1 SC_RULE=<rule> CELLS=C1 bash Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh submit
# after C1: one download for all three tags
bash Slurm_Codes/temp_bash/fetch_20260924_R44_scurve.sh
```

For (i): `PHASE=B SC_ENGINE=fm SC_K=20 CELLS="Br Bc Bt" … submit`. B0 is skipped because it would re-fly A9, which is
deterministic. Then C with the same K. Not proposed:
- re-running any existing cell;
- FM at unit initial noise (FLAW §B, R9-style: needs an authorised sampler change and a run-name token; ~20 min GPU).
  It would tell whether FM's lead here depends on the half-scale prior.

Cost of Phase B at K = 20, from the plain-set cells of 19-09 on this scene: per-step 3.2–4.7 s per control step, endpoint
1.1–3.7 s → **7–11 h per per-step cell and 3–9 h per endpoint cell** (ten flights, up to 871 steps); one job per cell stays
under the 24 h cap. At K = 1–2 per-step costs 0.14–0.17 s per step, about 20 min per cell.

## 3 · Phase B — projection on the selected model (Ch 6, Table 6.15 `tab:uav-scurve-projection`) — AFTER §2

- Variants: `diffuser` (the tag's own unprojected row), per-step `dpcc-r-tightened`, `dpcc-c-tightened`,
  `dpcc-t-tightened`, and **only where K ≥ 3** endpoint `hardflow_new-r-tightened`, `hardflow_new-c-tightened`,
  `hardflow_new-t-tightened` (`hardflow_new` is the input name; the eval writes the folders as `hardflow_sls-…`).
- Tag `p23scproj`. `FMPCC_SAFE_EPS_MODE` / `FMPCC_SAFE_EPS_FRAC`: use the values printed in the logs of the existing
  tightened s-curve cells (`u7hg`, FM K2 `dpcc-*-tightened`) and record them; do not rely on the sweep's defaults unseen.
  **Recorded (23-09): `scaled` / `1e-3`.** Job 25424 (FM K2 `u7hg`, which ran `dpcc-{r,c,t}-tightened`) and job 25612
  (diffusion `u7hg`, `dpcc-{c,t}-tightened`) both print `SAFE_EPS_MODE=scaled SAFE_EPS_FRAC=1e-3`, as does every other
  s-curve job on record. This is not the corridor's `1.0`. The driver pins it in A, B and C.
- **Job layout (driver).** The eval refuses a job that holds HardFlow variants and no `dpcc-*` row. At K = 20 each endpoint
  rule therefore rides with its per-step twin (`dpcc-X-tightened,hardflow_new-X-tightened`, about 10–20 h per job; flagged
  as over 12 h), plus one `diffuser` job. At K ≤ 2 there is one job with the three per-step rules. Diffusion gets one job
  per per-step rule. Runbook §4.
- Command shape as Phase A with `FMPCC_UAV_EVAL_TAG=p23scproj` and the variant list above, engine and K as decided.
- The log must show the variants exactly, and for the endpoint arm **no** `[hardflow][BLOCKED]` or `DEGENERATE`.

## 4 · Phase C — the same plans under two controllers (Ch 6, Table 6.16 `tab:uav-controller`) — AFTER PHASE B

> **24-09, checked against v3 Ch 6 as of v3.90.** The author asked for 10 vs 10 flights on the best projected setup
> under both controllers. **The only run the s-curve section still lacks is C1**: FM nfe 1, the winning per-step rule of
> B1, under MuJoCo MPC, ten flights. That rule's B1 cell is the cascaded-geometric side of the 10-vs-10.
>
> | item | status |
> | :-- | :-- |
> | Table 6.15 | B1, job 26195, ✅ flown; data in the fetch |
> | Table 6.16 | none/MPC is C0, job 26196, ✅; per-step/cascaded comes from B1; **per-step/MPC is C1, still to run** |
> | Fig. 6.9 (marked `\outdated`) | no run: rebuilt from the fetched npz, FM nfe 1 under both controllers, ten flights each |
> | Table 6.17 | no run: can be recomputed 10 vs 10 on FM nfe 1 from the same runs if the writing agent wants |
>
> **Rule choice.** The job log (run status only) shows r, c and t tied at 5/10 crossings. The pick is therefore made from
> the fetched `results.json`: most crossings, then fewer aborted, then fewer violating steps (the chapter's constraint
> column since v3.84), then smaller distance.

- `UAV_MIX_CONTROLLER=mjpc`, variants `diffuser` and the per-step rule with the most crossings in Phase B, tag
  `p23scmjpc`, ten flights each. Needs the `FMPCC_mjx` conda env on the node (`mix_uav_test/eval_mix_uav.py:740`).
- The cascaded-geometric rows of the table are the matching Phase A and B cells; nothing else is re-run.
- Cost: MuJoCo MPC adds about 126 ms per control step (Table 6.17) — ~40 min for the unprojected cell at K = 20; the
  projected cell costs what its Phase B twin cost, plus that.

## 5 · What stays as it is

Figure 6.9 and Table 6.17 are the MeanFM K = 10 pilot (ten flights against three) and are not re-run. The appendix section
`app:uav-scurve-budgets` keeps the pilot budgets (MeanFM K = 10, CI-MeanFM K = 5) for the record.

## 6 · Cost summary

| phase | cells | GPU |
| :-- | --: | --: |
| A required (A1 A3 A6 A7) | 4 | ~1 h |
| A full grid | 10 | ~2 h |
| B at K = 1–2 (per-step only) | 3 (+1 unprojected) | ~1 h |
| B at K = 20 (both methods) | 6 (+1 unprojected) | ~40–60 h, six parallel jobs of ≤ 11 h |
| C | 2 | ~1 h (K ≤ 2) to ~12 h (K = 20, projected) |

Claude (Opus 5.5, Claude Code) · 2026-09-23 · specification only; nothing submitted. The chapter cells it fills are
already in `v3/chapters/06_results.tex` as *pending (R44a/b/c)*.
