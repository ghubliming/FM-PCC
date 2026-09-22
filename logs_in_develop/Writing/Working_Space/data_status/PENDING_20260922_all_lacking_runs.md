# PENDING — 2026-09-22 · complete current list of missing runs and artefacts

**Renamed from `PENDING_20260920_all_lacking_runs.md` on 2026-09-22.** Same file, same history, nothing
removed; only the date in the name moved forward. Older changelogs (v3.44–v3.59) still cite the previous
name and were deliberately left alone, because they are a record of what was true when they were written.

**Updated for thesis v3.60 (2026-09-21/22).** This checklist answers *"what is still missing?"* on its own.
The historical ledger remains
[`PENDING_20260916_missing_data_and_analyses.md`](PENDING_20260916_missing_data_and_analyses.md);
the executable groups remain in the two `SLURM_RUNBOOK` files.
The reopened R26 row below supersedes older historical notes in this file that call it retired.

**Nothing in this file is ever deleted.** An item that stops being needed is struck through and marked
with the reason and the thesis version that retired it (§11, §12 and the table below), so that a later
reader can tell *"we decided against this"* from *"we forgot this"*.

---

## 0 · State of play, 2026-09-22 — read this first

v3.60 censused every corpus the thesis draws on. The scoreboard below is what came out of it. **Three
items were closed without a run**, because the data already existed and had simply never been published;
**three new items were opened**; and the pillars campaign is restated here with a corrected budget ladder.

### Closed by the census, no run needed

| what | where it landed | version |
| :-- | :-- | :-- |
| MeanFM $\nfe=5$ and $\nfe=10$ on D3IL-avoiding | promoted into Tables 6.1 and 6.2 | v3.60 |
| Diffusion $\nfe=10$ on D3IL-avoiding | promoted out of the appendix into Tables 6.1 and 6.2 | v3.60 |
| CI-MeanFM and FM at $\nfe=100$ on D3IL-aligning | added to Table 6.5 | v3.60 |
| CI-MeanFM $\nfe=2$ on UAV-s-curve | made `tab:uav-scurve` a matched comparison | v3.60 |
| The activation-threshold evidence ($\eta=0.5$ vs $0.1$ at $\nfe=100$) | new discussion in §6.2.2 | v3.60 |

### The open list, by priority

Runtime is a **rough GPU-time estimate**, derived as described under the table. It is compute, not
wall-clock: queue time and the 24 h `--time` cap are not in it.

| ID | Priority | One line | Training? | ~GPU time |
| :-- | :-- | :-- | :-- | --: |
| **R23** | ⛔ **ABANDONED 2026-09-22 → FALLBACK to `pillars_hg`** | UAV-pillars at `pillars_xl` — **Gen15 U17 abandoned; the thesis restores the withheld `pillars_hg` section with its caveat** (`Gen15/U17/CLOSURE_20260922_U17_abandoned.md`) (Gen15 U17, Slurm jobs up to 25995, author, 2026-09-22). Coverage, cost and whether anything is still new are filled in by the author when the wave ends; see §16. **No new pillars run is listed until then.** 🔴 **First read 2026-09-21: 10/12 children complete, 2 running, diffusion walled at 2/7 (✅ diffusion arm CLOSED BY DECISION — reported unprojected only, K=20 projection is ~1 GPU-day per variant and the `c` rules exceed 24 h; see §16) — and 53 of 54 finished projected cells read `success = 0.00`. Read [`SLURM_RUNBOOK_20260919_pillars_enlarged.md` §3e](SLURM_RUNBOOK_20260919_pillars_enlarged.md) before scheduling anything on this scene** | no | ~5 GPU-days spent, see §16 |
| **R2** | 🔴 | D3IL-aligning: the tightened projected diffusion baseline — blocks the four-model pre/post-projection merge | no | ~1.5 h |
| **R26** | 🟡 | D3IL-avoiding: five-seed diffusion $\nfe=2$ | **4 trainings** | ~11 h (training) + 15 min |
| **R16** | 🟡 **widened at v3.62** | D3IL-aligning, the full unprojected ladder $\nfe\in\{1,2,10,20,100\}$: the **eleven** cells `tab:va-models` now prints as *pending* — all four models at $\nfe=1$; FM and diffusion at $2$; CI-MeanFM, FM and diffusion at $10$ (§20) | **3 trainings** (diffusion at chain lengths 1, 2, 10) | ~9 h (training) + ~3 h (11 evaluations) |
| **R28** | 🟡 | D3IL-avoiding: FM at $\nfe=5$ (seeds 7–10) and $\nfe=10$ ($T{=}0.5$, five seeds) — **not owed**, nothing claims those cells | no | ~2 h |
| **R7 / R14** | 🟡 | D3IL-avoiding: candidate-matched endpoint vs per-step projection, 5 seeds × 20 episodes | no | ~6.5 h |
| **R9** | 🟡 | FM with unit-scale initial noise, all three environments | no | ~3 h |
| **R15** | 🟡 | UAV-s-curve: MuJoCo MPC at ten flights instead of three | no | 30 min unprojected / ~6.5 h projected |
| **R30** | 🟢 | UAV-corridor: diffusion under plain `dpcc-t-tightened` — removes the last confound in Table 6.8 | no | ~1 h |
| **R31** | 🟡 | UAV-s-curve: **the matched grid** — unprojected at $\nfe\in\{1,2,3,5\}$ for the three flow models (7 of 12 cells missing), and the corridor-shaped projection grid at $\nfe\in\{3,5\}$ (5 of 6 cells missing). Opened at v3.61, §18; the draft prints the grid with the missing cells marked \emph{pending} | no | ~1 h unprojected + ~30 h projected |
| ~~**R29**~~ | ⬜ **superseded by R31** (v3.61) | UAV-s-curve: matched-budget switched-wall re-run — the matched grid of R31 contains it | — | — |
| D10c | 🟡 | Box-pose logging, then re-evaluation of MeanFM $\nfe=20$ | no | ~1 h + a logging change |
| D11 | 🟢 | Matched full executed-step timing on D3IL-avoiding, for a total-time frontier | no | ~3 h + instrumentation |
| D8 / D9 | 🟡 | Compute totals and SLURM/git provenance for the Reproducibility appendix | no | **no GPU** — log analysis |
| D12 | 🟢 | Print-resolution MuJoCo renders for the two 320×180 stills | no | minutes (needs `mujoco`, cluster) |

### Optional — open, but nothing in the draft waits on them

| ID | Missing work | ~GPU time |
| :-- | :-- | --: |
| **R27** | Diffusion baseline under endpoint projection on UAV-corridor at $\nfe=20$, four rules × 12 flights | ~3.5 h |
| **R17** | Repeat the projected alignment configurations, to measure run-to-run variation | ~2 h |
| ~~**R16 extension**~~ | ⬜ **absorbed into R16 at v3.62** (FM at $\nfe=2$ is one of its eleven cells) | — |

### How the estimates were made

Each is `measured ms per control step × steps per episode × episodes × variants`, using per-step times
already recorded in the corpora rather than job wall-times, which are not logged per cell.

- **UAV**: planner time from `tab:uav-corridor` ($29$, $99$, $144$, $625$\,ms at $\nfe=1,3,5,20$), plus a
  measured **$66$\,ms/step controller-and-simulator overhead** — the difference between the $94.9$\,ms
  full executed step in `tab:uav-controller` and the $29$\,ms planner at the same budget. Episode limits
  $396 / 634 / 871$ from `tab:eval`.
- **D3IL-aligning**: $400$-step limit, ten contexts, projected times of $265$–$450$\,ms/step from
  `tab:va-projection-models`.
- **D3IL-avoiding**: episodes are short (~$65$ control steps), so evaluation there is cheap; its cost is
  dominated by the variant count (~13 per cell).
- **Training** is the one number taken from a real job: diffusion $\nfe=2$, job 25965, **2 h 41 m on one
  A5000**. Diffusion training cost is roughly independent of the budget, so other diffusion trainings are
  estimated at the same figure.

🟠 **Treat these as ±50 %.** They ignore queue time, assume no early episode termination except where
stated, and assume one flight at a time on one GPU. **R23 carries no estimate**: the campaign is
already running, and the earlier figure derived here (~45 h) was built on UAV-corridor per-step times,
which the pillars runbook's own measurements do not support. It is withdrawn, not corrected.

**Struck, not deleted:** R25, R18 (ruled out on cost, v3.52 — §11); the $\nfe=20$ `both-hard` re-run
(not owed, v3.55 — §12); D2, the diffusion `both-hard` artefact, the two-episode rollout fetch for
`fig:avoiding-paths` (§11, §12). R11, R12, R13, R21, R22, R24 are retired or superseded (§2, §3).

## 1 · Runs that close a visible gap in the current thesis

| ID | Priority | Missing work | Training? | Draft effect / blocker |
| :-- | :-- | :-- | :-- | :-- |
| **R2** | 🔴 | D3IL-aligning: tightened diffusion baseline at $\nfe=20$ on the ten thesis contexts | no | closes the baseline row in `sec:res:aligning:projection`; first resolve why the evaluation config says `n_contexts: 3` |
| **R23** | ⛔ **ABANDONED 2026-09-22 → fallback to `pillars_hg`** (Gen15 U17, jobs 25970–25995 complete, all-zero grid; `Gen15/U17/CLOSURE_20260922_U17_abandoned.md`) | UAV-pillars at `pillars_xl`: **no longer a run.** v3.63 restored the withheld `pillars_hg` section with its caveat (§21); no `pillars_xl` number is in the draft | — | — |
| ~~**R25**~~ | ⬜ **NO LONGER NEEDED — ruled out on cost by the author, v3.52.** The rows it filled are removed from `tab:state-headline`; §6.1.2.7 answers the question from existing data | D3IL-avoiding: CI-MeanFM, $\alpha_{\mathrm{end}}=0.2$, $\nfe=1,2$, all three rules, **20 episodes per seed and geometry**, seeds 6--10 | no | ~~fills the two pending rows of `tab:state-headline`~~ — those rows are removed (§10); CI-MeanFM is reported at seed 6 in `tab:avoiding-budget20`. Would refine, not open, a claim |
| **R26** | 🟡 **REOPENED by the author at v3.57:** five-seed diffusion $\nfe=2$ is visibly pending in Tables 6.1 and 6.2 | D3IL-avoiding: train diffusion $\nfe=2$ seeds 7--10, then evaluate five seeds at two episodes per geometry under unprojected and three projected rules | **four trainings** | fills the pending cells in `tab:avoiding-raw-models` and `tab:avoiding-dpcc-protocol`; seed 6 alone cannot be pooled into a five-seed row |
| **R16** | 🟡 **widened at v3.62** | D3IL-aligning, unprojected, untightened set, ten contexts: MeanFM $\nfe=1$; CI-MeanFM ($\alpha_{\mathrm{end}}=0.2$) $\nfe=1,10$; FM $\nfe=1,2,10$; diffusion $\nfe=1,2,10$ — each diffusion budget needs a checkpoint trained at that noise-schedule length | diffusion: **three trainings**; flow models: evaluation only | fills the eleven *pending* rows of `tab:va-models` (v3.62 prints the whole ladder in the main text; the appendix copy is gone); MeanFM $\nfe=2,10$ and every $\nfe=20,100$ cell already exist |
| ~~**R18**~~ | ⬜ **NO LONGER NEEDED — ruled out on cost by the author, v3.52.** `tab:state-headline` no longer advertises the below-native-budget rows | D3IL-avoiding: extended evaluation of diffusion at $\nfe=1,10$; add $\nfe=2$ after R26 | no for 1/10 | ~~fills the below-native-budget rows absent from `tab:state-headline`~~ — no longer advertised there (§10) |
| **R7** | 🟡 | D3IL-avoiding: candidate-matched endpoint versus per-step projection, five seeds $\times$ twenty episodes | no | replaces the provisional four-seed, two-episode result |
| **R14** | 🟡 | D3IL-avoiding: repeat the endpoint budget ladder with the same candidate count for both projectors | no | removes the confound from `tab:hf-ladder` |
| **R9** | 🟡 | re-evaluate FM with unit-scale initial noise in D3IL-avoiding, D3IL-aligning and UAV-corridor | no | tests the known prior-scale flaw; alignment and UAV conclusions may move |
| **R15** | 🟡 | UAV-s-curve: MuJoCo MPC for ten flights instead of three | no | strengthens `tab:uav-controller` |

**R25 exact scope.** Evaluate seeds 6--10, three geometries, twenty episodes per seed and geometry,
tightened constraints, four candidates, rules $r$, $c$ and $t$, at $\nfe=1,2$. No training is needed.

**R26 exact scope.** Seed 6 exists (training job 25965, evaluation job 25966). Because diffusion's
denoising-step count is fixed at training time, seeds 7--10 require four new trainings. Evaluate all
five seeds under the same rules and protocol as `tab:avoiding-dpcc-protocol`, then at twenty episodes
if the extended comparison is to be complete.

## 2 · Optional runs

| ID | Missing work | Value |
| :-- | :-- | :-- |
| ~~**R16 extension**~~ | ⬜ absorbed into R16 at v3.62 | — |
| **R17** | repeat the projected alignment configurations | measures run-to-run variation |
| **R27** | diffusion baseline under endpoint projection on UAV-corridor at $\nfe=20$, single/$r$/$c$/$t$, twelve flights each | endpoint projection has never been evaluated on diffusion in any environment |

R11 is retired: v3.47 removes the subsection it was meant to support. R24 is closed: all eight
raw-plan panels are present.

## 3 · Prepared work now represented in the draft

**R23 — UAV-pillars at `pillars_xl`.** The enlarged geometry, driver and identity check exist; the
campaign is running (Gen15 U17, Slurm jobs up to 25995) and its coverage is not yet censused here. Since v3.56 the scene has a visible section, table templates and
figure specification in Results. Its old `pillars_hg` numbers remain withheld and must not fill those
slots. The `u7xlchk` verification is one unprojected cell, not the campaign. R12, R13, R21 and R22
remain superseded by R23. The rollout artefacts for the selected completed cells must be transferred
and drawn as `fig_uav_pillars_xl_paths` only after their outcomes are checked.

## 4 · Transfers or logging changes, not model runs

| ID | Work | What is needed |
| :-- | :-- | :-- |
| ~~**D10a**~~ | ~~D3IL-avoiding executed paths~~ | ✅ **CLOSED at v3.50.** 20 of 21 cells staged; `extract/exec_paths.py` → `data/exec_paths.json` → `fig_avoiding_paths`, now `fig:avoiding-paths` in §6.1.2.2. Drawn on `top-right-hard`. **Superseded at v3.53:** the author replaced the four-model panel set with the two average-velocity models at $\nfe=1$ and $\nfe=2$ (all four cells present in the same drop), so the figure no longer needs the diffusion `both-hard` artefact at all |
| ~~**D10d**~~ | ~~`fig:raw-plans`, the diffusion baseline at its own budget $\nfe=20$~~ | ✅ **CLOSED at v3.49.** The panel was fetched from the complete sibling campaign `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` (seed 6, `both-hard`) — the run this row originally named died mid-variant and never wrote the dashboard. `fig_raw_plans_diffusion_K20` is in the store, exported to `v3/figures/`, and wired into `fig:raw-plans` as the $\nfe=20$ row. **The draft now carries no `\todofigure` at all.** Details in §8 below |
| ~~**D10b**~~ | ~~D3IL-aligning end-effector paths~~ | ✅ **CLOSED at v3.50.** The 09-19 drop already carried them: `obs_all` is `(10, 400, 6)` at `combined_5-tightened`, seed 6, for all three variants. Built as `fig_aligning_paths`, now `fig:aligning-paths` in §6.2; its three panel counts reproduce the $\nfe=20$ rows of `tab:va-projection` (2/10, 9/10, 10/10) |
| **D10c** | D3IL-aligning box paths | add per-step box-pose logging, then re-evaluate MeanFM $\nfe=20$ for unprojected, per-step and endpoint variants |
| **D11** | Matched full executed-step timing for the final cost comparisons | Record elapsed time per executed control step around **planner + low-level controller + MuJoCo step** for every cell used in the D3IL-avoiding frontier (and, if extended, the projected alignment/corridor comparisons). Existing `avg_time`/`avg_time_ms` prices only generation, projection and selection. The UAV-s-curve controller subset already has full elapsed-step timing in `tab:uav-controller`, but does not supply a matched cross-model frontier. Build a total-time Pareto plot only after the matched measurements exist. Added by v3.59, 2026-09-21. |
| **D12** | Print-resolution MuJoCo stills of the D3IL-avoiding start and finish poses | Figure `fig:env-avoiding` pairs authentic MuJoCo frames 0 (start) and 200 (end effector beyond the green goal line), both from the first tile of the tracked D3IL montage; it shows no trajectory plot. Each frame is only 320×180 pixels. For print-resolution replacements, write/run dedicated MuJoCo rendering code on the cluster at the recorded start and completed poses, preserving the camera and scene; then replace `fig_render_avoiding_start` and `fig_render_avoiding` in the DA figure store and export to v3. Local Python environments lack `mujoco`; no synthetic pose or trajectory plot should stand in for the simulator views. Updated in the v3.59 continuation, 2026-09-21. |
| ~~**D2**~~ | ~~alignment final-orientation error~~ | ⬜ **NO LONGER NEEDED — the metric is out of the thesis.** v3.51 removed the orientation bullet from §5.4.2 and the scoring row of `tab:va-vs-d3il`; v3.53 removed the last `\hole` in §6.2. Author: *"we didnt use it ... we dont care the angle"*. The recorded `context_final_box_angle_deg` column uses a convention that differs between runs, which is stated once in the §5.4.2 `\srcnote` and nowhere else |
| **D8** | total compute | derive GPU-hours from `sacct` / complete logs |
| **D9** | corpus provenance | add job IDs, git revisions and checkpoint tags to `tab:corpora` |

Plan smoothness is deliberately **not pending**. The author rejected it at v3.47; its `\hole` and
the old D1 item are removed.

## 5 · Decided against or not runnable

| Item | Decision |
| :-- | :-- |
| more seeds for alignment or UAV | closed for storage and compute cost; seed 6 is final |
| 50--60 alignment contexts | retired; ten enumerated contexts are this thesis's protocol |
| endpoint projection at $\nfe=1$ | no guiding step exists |
| UAV-s-curve per-step projection under $r,c$ | post-fix cells already fail through controller divergence at prohibitive cost |
| diffusion at $\nfe=5$ on D3IL-avoiding | no checkpoint and no main-text comparison needs it |
| FM at $\nfe=5,10$ in the DPCC-protocol table | above the locked budgets; budget behaviour remains in the frontier/ladder |

## 6 · Coverage audit

| Draft absence | Owner |
| :-- | :-- |
| ~~`tab:avoiding-dpcc-protocol`, diffusion $\nfe=2$~~ | ~~R26~~ — ⬜ not an absence since v3.52: the row is gone and the caption says why |
| ~~`tab:state-headline`, CI-MeanFM $\nfe=1,2$~~ | ~~R25~~ — ⬜ not an absence since v3.52; CI-MeanFM is reported at seed 6 in `tab:avoiding-budget20` |
| ~~`tab:state-headline`, diffusion below $\nfe=20$~~ | ~~R18 + R26~~ — ⬜ not an absence since v3.52 |
| endpoint comparison on D3IL-avoiding | R7, R14 |
| tightened alignment diffusion | R2 |
| alignment path figure of the **box** (the end-effector one is drawn) | D10c |
| ~~alignment end-effector path figure~~ | ~~D10b~~ — ✅ drawn at v3.50 |
| ~~alignment final-orientation error~~ | ~~D2~~ — ⬜ metric removed from the thesis at v3.51/v3.53 |
| ~~avoiding executed-path figure~~ | ~~D10a~~ — ✅ drawn at v3.50; **re-scoped at v3.53** to MeanFM and CI-MeanFM at $\nfe=1,2$, so nothing waits |
| ~~`fig:raw-plans`, diffusion $\nfe=20$ panel~~ | ~~D10d~~ — ✅ filled at v3.49 |
| `tab:state-headline`, **every $\nfe=20$ row** is short of seeds or geometries | 🟡 **disclosed, not pending.** v3.52 counted the whole $\nfe=20$ census (§10) into a `\guard` beside the table; the re-run stays ruled out on cost |
| FM prior-scale flaw | R9 |
| UAV controller at $n=3$ | R15 |
| appendix compute / provenance | D8, D9 |
| full executed-step time Pareto comparison for avoiding (optional matched extension to alignment/corridor) | D11 — existing frontiers price planner time only; the UAV-s-curve controller subset has full-step timing but is not a matched model grid |
| print-resolution simulator stills for `fig:env-avoiding` | D12 — both authentic start and end frames are 320×180; high-resolution renders require MuJoCo on the cluster |
| UAV-pillars absent from Results | R23, only if it returns |

Checked against active `\hole`, `\provisional`, `\guard` and the missing table rows in
`v3/chapters/05_setup.tex`, `06_results.tex` and `09_appendix.tex` at v3.53 (`\hole` 22 → 16 over
v3.51--v3.53). There is no
`\todofigure` left: every figure the draft references exists.

Chapter 8 re-entered the build at v3.49 (author instruction) and adds no data item — it restates
Chapter 6 and cites no number Chapter 6 does not carry.

---

## 7 · DA note, 2026-09-20 — section 4 checked against the artefacts on disk

Added by the figure pipeline, not the draft. The run/eval items in sections 1–3 are untouched; only
the transfer items were verified against what the 09-19 drop and the 09-19 log tree actually hold.

| ID | status after checking | evidence |
| :-- | :-- | :-- |
| **D10d** | ⏳ still a fetch, and it is the only one that blocks a `\todofigure` | run folder present in the 09-19 tree, 562 files / 284.4 MiB; the wanted `.png` is one of them |
| **D10a** | ⏳ still a fetch; 4 of its 24 leaves are already local | `dpcc-t-tightened.npz` carries `obs_all` as a `(20,)` object array of per-episode paths |
| **D10b** | ✅ **already satisfied — no download needed** | the 09-19 drop holds `obs_all` of shape `(10, 400, 6)` at `combined_5-tightened`, seed 6, for `dpcc-r`, `hardflow_sls-r` and `diffuser`. What is missing is the *builder*, not the data |
| **D10c** | ⏳ unchanged, and now confirmed by inspection | the evaluation assembles `obs_all` itself as 6-D `[des_c_pos(0:3) \| c_pos(3:6)]` — commanded TCP and actual TCP (`eval_mix_visual_aligning.py:1054`, and the env's `robot_state()` returns `tcp_pos` alone). The box pose is nowhere in it. The logging change is genuinely required |

**One trap worth carrying into D10a.** `obs_all` is variant-dependent in the avoiding runs: the
projected variants carry the executed path, but the unprojected `diffuser.npz` of the same leaf carries
**scalars only**. The executed-path figure can therefore only be built from a projected variant — which
is what D10a already specifies, but the opposite assumption has been made in this repo before.

**Stager:** `Slurm_Codes/temp_bash/fetch_20260920_v3_figure_artefacts_wave3.sh`, groups `F6` (D10d),
`F7` (D10a), plus opt-in `F8` (D10b, widen only) and `F9` (D8/D9). Recorded in
`Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md`, 2026-09-20
section.

---

## 8 · CORRECTION to section 4, 2026-09-20 — D10d is **not** a download

The wave-3 PLAN ran on the cluster: 20 of 22 cells resolved (40 files, 35.0 MiB). The two that did not
are the same leaf, and it is **truncated, not missing**:

```
logs/avoiding-d3il/plans/diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10/
    H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials/6/results/halfspace_both-hard/
```

That folder holds `dpcc-r`, `dpcc-r-tightened`, `dpcc-c`, `dpcc-c-tightened` and `dpcc-t` — and then
stops. `eval_dpcc-t-tightened.log` is 109 bytes with no artefacts beside it, and 12:32 on 2026-08-18 is
the newest mtime of the whole run folder, so the job ended mid-variant. A complete leaf holds thirteen.
The batch CSV shows the same five-of-thirteen for that cell, so the corpus and the disk agree.

| ID | section 4 said | corrected |
| :-- | :-- | :-- |
| **D10d** | download one dashboard `.png` | ❌ **the `.png` was never written.** Needs an evaluation re-run |
| **D10a** | download `.npz` | ✅ for 20 of 21 cells · ❌ the diffusion `both-hard` cell needs the same re-run |

### This also touches a published table cell

`avoiding_table_spread.py` prints `seeds=` and `geos=` for every published cell. Of the 48 it prints,
exactly one is not `geos=3`:

```
Diffusion  K=20  t  S&C=0.960 +/- 0.045  steps=82.8 +/- 14.7  ms=564.3 +/- 29.5  seeds=5 geos=2
```

`dpcc-t-tightened` is the variant the job died on, so **both-hard contributes nothing** to the
extended-protocol diffusion K20 per-step-tightened row. The `r` and `c` rows are unaffected — the job
got through `dpcc-c-tightened` nineteen minutes before it stopped.

This is a coverage asymmetry, **not** a flattered baseline: on `n_success_and_constraints` both-hard is
not the harder geometry — every model scores 1.000 there under the tightened-c rule.

### Wider coverage of the 20-trials corpus, for the record

| run | both-hard | top-left | top-right |
| :-- | --: | --: | --: |
| Diffusion K20 | **1** (partial) | 5 | 5 |
| FM K20 | **3** | 5 | 5 |
| MeanFM K20 | **0** | 3 | 5 |
| all K1 / K2 / K5 / K10 runs | 5 | 5 | 5 |

MeanFM K20 and FM K20 are outside the budgets §1 locks into the tables, but FM K20 **is** a table row,
and MeanFM K20 feeds `fig:k-ladder`. Worth knowing before either is quoted as "five seeds and three
geometries".

### The re-run that closes all three

Diffusion K20, 20-episode protocol, seed 6, geometry `both-hard`, the eight variants from
`dpcc-t-tightened` onward. That single cell closes the last `\todofigure` of `fig:raw-plans`, fills the
diffusion row of the new avoiding executed-path figure, and repairs the `geos=2` cell. Extending it to
seeds 7–10 at `both-hard` would make the diffusion K20 row a genuine 5x3 like every other row — larger
job, separate decision.

### §8 update, same day — D10d is CLOSED, D10a is not

**D10d ✅ closed by fetch after all**, just not from the run the request named. The dashboard exists in
the complete sibling campaign `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5`, seed 6, `both-hard`.
`thres0.5` is a *projection* threshold and `diffuser` is the *unprojected* arm, so the projector never
ran and the plans are the model's own. `fig_raw_plans_diffusion_K20` is in the store; `sources.PLANNED`
is now **empty**.

Note for the next reader: the request's crop spec was wrong. It says to use the twenty-episode box with
the `(403,400)` resize; that file is $3000\times1000$, the *two*-episode layout, and takes the 08-19 box
with no resize. Measured from the grid bands rather than assumed.

**D10a ⏳ still open for exactly one cell.** 20 of 21 executed-path cells are staged and validated (all
load, 20 episodes each, S&C matching the tables). The diffusion K20 `both-hard` cell needs
`dpcc-t-tightened.npz`, which has no image substitute — so that row of the executed-path figure, and the
`geos=2` cell in Table 6.2, both still wait on the re-run described above.

### §8 update, v3.49 — what the draft now says about it

The panel is in `fig:raw-plans`. Its provenance is stated in the figure's `\guard`: a second evaluation
campaign of the same checkpoint, scene, seed and geometry at a different projection threshold, on the
unprojected arm, so no projector ran in either campaign. The `geos=2` cell is disclosed in the
`$^{\dagger}$` footnote of `tab:state-headline`, which now says that the diffusion row under temporal
consistency is a mean over two geometries and that it is the only such cell in either avoiding table.
Neither is a claim change; both are disclosures. The re-run remains the way to remove them.

---

## 9 · v3.50 — the two executed-path figures are built

The `\hole`s that asked for a counterpart of `fig:uav-corridor-paths` on the two manipulator
benchmarks are closed. Both figures read `DA_in_Paper/data/exec_paths.json`, written by the new
`plotting/extract/exec_paths.py`; the constraint set under each path is drawn by the same code as the
Chapter 5 environment figures, so it is the geometry the projector was given rather than a redrawing.

| figure | cells | what it shows |
| :-- | :-- | :-- |
| `fig:avoiding-paths` §6.1.2.2 | 4 × 20 episodes, `top-right-hard`, `dpcc-t-tightened`, seed 6 | no episode of any model touches the excluded region; 19/17/17/20 of 20 reach the goal |
| `fig:aligning-paths` §6.2 | 3 × 10 contexts, `combined_5-tightened`, seed 6 | 2/10, 9/10, 10/10 violation-free — the $\nfe=20$ rows of `tab:va-projection`, reproduced |

**What is still open, and is now the only path-figure gap: D10c.** The alignment figure draws the
**end effector**, because the evaluation logs no box pose. That is the right quantity — the constraint
set applies to the planned end-effector position — but the quadrotor figures draw the constrained body,
and the equivalent here is the box. Stated in the figure's `\guard`, not left implied.

## 10 · v3.52 — the extended grid is closed by decision, and what the $\nfe=20$ cells really cover

**Author instruction, 2026-09-20:** *"pending: 20 episodes for seeds 6–10 — THIS I THINKED will be
extreme highly cost, remove them, and turn into only check the FLAGSHIP ones 20 vs 2."*

### The decision

**R25, R26 and R18 are ruled out.** Together they are four diffusion training runs plus extended
evaluations of every remaining budget × model × rule cell, and the question they were opened for —
*does a larger step budget buy the flow-based models anything once the projector is in the loop?* — is
answered by evaluations that already exist. The pending rows are gone from
`tab:avoiding-dpcc-protocol` and `tab:state-headline`; the new
§6.1.2.7 `sec:res:avoiding:budget20` and `tab:avoiding-budget20` answer the question with a
**within-model, matched-cell** comparison of $\nfe=2$ against $\nfe=20$ at the extended protocol, for
all three flow-based models, against the baseline of record.

Result: none of the three improves at twenty (S&C $1.000\to0.970$ for FM, $0.990\to0.955$ for MeanFM,
$1.000\to0.950$ for CI-MeanFM) and the time per action rises by $17$–$36\times$. If any of R25/R26/R18
is ever run, it refines a number; it does not open a question.

### 🔴 New finding — the $\nfe=20$ extended campaigns are demonstrably truncated

Counted cell by cell from the corpus of record
(`19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703`, via
[`avoiding_unprojected_and_budget20.py`](../../../../Data_Analysis/DA_in_Paper/analysis/avoiding_unprojected_and_budget20.py)
block C). Every one of them is short, and they are the slowest cells in the campaign:

| cell (`_msg20trials`, 20 episodes/seed) | unprojected | $r$ | $c$ | $t$ |
| :-- | :-- | :-- | :-- | :-- |
| **Diffusion $\nfe=20$** | top-left 5, top-right 5, **both-hard none** | both-hard **seed 6 only** | both-hard **seed 6 only** | **no both-hard** |
| **MeanFM $\nfe=20$** | top-left 3, top-right 5, **no both-hard** | top-left **2 of 5**, top-right 5, **no both-hard** | same | same |
| **FM $\nfe=20$** | both-hard 2 of 5 | both-hard **3 of 5** | both-hard **3 of 5** | both-hard **2 of 5** |
| *(for contrast)* every $\nfe\in\{1,2\}$ cell | complete | complete | complete | complete |

The author's suspicion was right: the baseline row of `tab:state-headline` is **five seeds on two
geometries plus one seed on the third**, not five seeds × three geometries. v3.52 marks it in the
caption and counts it in a `\guard`; `tab:avoiding-budget20` matches each budget pair to the cells both
cover so the within-model comparison is not contaminated by the ragged coverage.

**Not a new run request.** The author has ruled the re-run out; this section records what the numbers
rest on so that the draft states it rather than implying full coverage.

## 11 · v3.53 — what this version struck from the list

Nothing here was run. Each item stopped being needed because the draft changed.

| item | struck because | v3 version |
| :-- | :-- | :-- |
| **D2** — alignment final-orientation error | the metric is out of the thesis: §5.4.2 no longer defines it, `tab:va-vs-d3il` no longer scores on it, §6.2's `\hole` is gone. Author: *"we didnt use it, I belive so, so dont mention in the Chap5 … we dont care the angle"* | v3.51, v3.53 |
| **the diffusion `both-hard` artefact** (last remnant of D10a) | `fig:avoiding-paths` was re-scoped to the two average-velocity models at $\nfe=1$ and $\nfe=2$, and all four of those cells are already in the 20-09 drop. The figure needs nothing | v3.53 |
| **R25, R26, R18** | ruled out on cost by the author; §6.1.2.7 answers their question from existing data | v3.52 |
| **the K20 `both-hard` re-run**, as a *pending* item | the coverage is now counted in full beside the table instead of being repaired (§10). It would improve a number; it is not owed | v3.52 |

**What remains genuinely open** is unchanged: R2 (tightened alignment diffusion baseline), D10c (box-pose
logging then re-evaluation), R7/R14 (candidate-matched endpoint projection on D3IL-avoiding), R9 (FM
prior scale), R15 (MuJoCo MPC at $n=10$), D8/D9 (compute and provenance for the appendix), and R16/R17/R27
as optional.

## 12 · v3.55 — the twenty-episode evaluation is quarantined, not repaired

**Author instruction, 2026-09-21:** the twenty-episode campaign is suspicious, so §6.1 is reported at
**DPCC's own protocol** (5 seeds × 2 episodes) throughout, and everything that rests on twenty episodes
is separated into one subsection at the end of §6.1, ahead of the conclusion.

This changes what the open runs are worth, without adding any.

| item | status after v3.55 |
| :-- | :-- |
| the K20 `both-hard` re-run | still ⬜ **not owed**. §6.1 no longer leans on any $\nfe=20$ twenty-episode cell; `tab:coverage-k20` prints the census in the quarantined subsection |
| **R25, R26, R18** | unchanged: ⬜ ruled out on cost (§10) |
| ~~a 2-episode rollout fetch for `fig:avoiding-paths`~~ | ⬜ **not needed.** Episode *i* of an evaluation is fixed by *i* (`torch.manual_seed(i)`, `env_seed = i`, `scripts/eval.py:297--299`), so the first two episodes of the staged 20-episode artefact **are** the two the DPCC protocol runs. The figure is rebuilt from them |

🟠 **One real gap this exposed, and it is small.** `fig:avoiding-paths` draws seed 6 only, because only
seed 6's rollout artefacts were staged (D10a). At the DPCC protocol a cell is 5 seeds × 2 episodes, so
the figure shows 2 of the 30 episodes the row beside it averages. Fetching the other four seeds'
`dpcc-t-tightened.npz` for MeanFM and CI-MeanFM at $\nfe=1,2$ on `top-right-hard` — **8 files, a
download, no run** — would make the figure the whole protocol. Stated in the figure's `\guard` either
way.

## 13 · v3.60 — the K=5 / K=10 census, and one gap it opened

**Author instruction, 2026-09-21:** a budget goes into the main D3IL-avoiding tables if and only if it
carries the full protocol of \ac{DPCC} (5 seeds × 3 geometries × 2 episodes). Everything was censused
against `19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703`.

| model | $\nfe$ | coverage | outcome |
| :-- | :-- | :-- | :-- |
| MeanFM | 5 | ✅ 5 seeds × 3 geos × 2 eps | promoted into Tables 6.1 and 6.2 |
| MeanFM | 10 | ✅ 5 seeds × 3 geos × 2 eps | promoted into Tables 6.1 and 6.2 |
| Diffusion | 10 | ✅ 5 seeds × 3 geos × 2 eps | promoted out of Appendix B into Tables 6.1 and 6.2 |
| **FM** | **5** | ❌ **seed 6 only** at $T{=}0.5$ | **R28, below** |
| **FM** | **10** | ❌ **absent** at $T{=}0.5$ (only $T{=}0.05$, $T{=}0.1$, seed 6) | **R28, below** |

🟠 **A promotion trap, recorded so it is not walked into.** FM *does* have a complete five-seed
$\nfe=5$ result — in `H8_K5_..._FlowMatchingODE_msg20trials`, the **twenty-episode** campaign. Its
success rates carry denominators up to 20 where every protocol cell carries 2. It must not be pooled
with, or promoted into, a protocol table.

### New item

| ID | Priority | Missing work | Training? | Draft effect |
| :-- | :-- | :-- | :-- | :-- |
| **R28** | 🟡 | D3IL-avoiding: instantaneous-velocity matching (FM) at $\nfe=5$ on seeds 7--10, and at $\nfe=10$ at the standard $T{=}0.5$ on all five seeds, two episodes per seed and geometry, unprojected plus the three tightened rules | no | would complete the FM column of `tab:avoiding-raw-models` and `tab:avoiding-dpcc-protocol` at the two intermediate budgets MeanFM and diffusion already carry, and would let `fig:avoiding-tradeoff` / `fig:k-ladder` draw FM's mid-budget shape. **Not owed:** nothing in the draft claims those cells, and FM is operated at $\nfe\in\{1,2\}$ |

### Coverage confirmed for the two endpoint-projection studies (v3.60)

Both were censused because the author asked whether they are single-seed. Neither is.

- `fig:projector-cost` — `temp/0609/I/batch_avoiding_combined_20260906_125724`, $\nfe=3$ and $5$,
  `A1_B4`: **4 seeds (7, 8, 9, 10) × 3 geometries × 2 episodes = 24 episodes**, both methods on the same
  four. 🟠 **Seed 6 is absent**, although the run tag reads `..._mfunet_s6`. R7 (five seeds × twenty
  episodes) remains open and unchanged.
- `tab:hf-ladder` — `temp/1108/Revised_2/batch_avoiding_combined_20260811_221322`: **5 seeds (6--10) ×
  3 geometries × 2 episodes = 30 episodes**. The caption previously said "15 episodes per cell"; 15 is
  the seed--geometry cell count. Its per-budget numbers are now published as
  `tab:app:hf-ladder-detail`. R14 (candidate-matched ladder) remains open: endpoint ran at one candidate
  and per-step at four, so the time columns of that corpus stay unusable.

## 14 · v3.60 — the D3IL-aligning census

Every alignment cell was censused against
`analysis_results_checkpoint/15-09/batch_va2_20260915_100754`, keyed on the folder name (in which the
`T` token is the **activation threshold**, not the transport time).

**Every alignment result in this thesis is training seed 6.** There is no multi-seed alignment data of
any kind; the spread printed in those tables is across contexts, as `sec:setup:metrics:spread` states.

### What exists, unprojected, on the ten shared contexts

| $\nfe$ | MeanFM | CI-MeanFM | FM | Diffusion |
| :-- | :-- | :-- | :-- | :-- |
| 1 | ❌ | ❌ | ❌ | ❌ *(row added at v3.62 — all four are R16)* |
| 2 | ✅ | ✅ | ❌ (**R16**) | ❌ (**R16**) |
| 10 | ✅ | ❌ (**R16**) | ❌ (**R16**) | ❌ (**R16**) |
| 20 | ✅ | ✅ | ✅ | ✅ |
| 100 | ✅ | ✅ **newly published, v3.60** | ✅ **newly published, v3.60** | ✅ |

🆕 **Two cells were on disk and not in the draft.** CI-MeanFM at $\nfe=100$ (median $0.1791$\,m,
$902.2$\,ms) and FM at $\nfe=100$ (median $0.4247$\,m, $1425.0$\,ms) existed in the corpus and were
missing from `tab:va-models`. Both are now printed, which makes $\nfe=100$ the second budget at which
all four models are measured. No run was needed.

### Projected cells, tightened set — what blocks the four-model merge

| model | per-step (`dpcc-*`) | endpoint (`hardflow_sls-*`) |
| :-- | :-- | :-- |
| MeanFM | ✅ all three rules | ✅ all three rules |
| CI-MeanFM | ✅ | ✅ |
| FM | ✅ | ✅ |
| **Diffusion** | ❌ **none** | ❌ **none** |

**R2 confirmed and sharpened.** The diffusion baseline has **no projected cell on the tightened
alignment set under either projector**, so the pre-projection/post-projection comparison cannot be run
across all four models. It is complete for the three flow-based models. Until R2 lands,
`tab:va-projection-models` is the full merge the data allows, and `sec:res:aligning:projection` is
stated on the flow-based models rather than against the baseline.

### 🆕 Threshold evidence found while censusing — no run needed

Analytic average-velocity matching at $\nfe=100$ was evaluated at **two activation thresholds**,
$\eta=0.5$ and $\eta=0.1$, same model, constraints and selection rule:

| $\eta$ | guiding steps | ms per control step | median final distance |
| :-- | :-- | :-- | :-- |
| 0.5 | 50 | $15{,}218 \pm 3{,}885$ | $0.1816$\,m |
| 0.1 | 10 | $1{,}195 \pm 180$ | $0.1876$\,m |

A factor of $12.7$ in cost for $6$\,mm of distance. This is now the measured justification for lowering
$\eta$ with the budget in `sec:res:aligning:projection`, where the argument was previously made from the
arithmetic alone.

## 15 · v3.60 — the quadrotor census

### UAV-corridor: why the ladder is $\{1,3,5\}$

Verified, not guessed. At the scene's activation threshold $\eta=0.5$ the projector acts on the last
$\lceil \eta\nfe \rceil$ sampling steps, so $\nfe=1$ and $\nfe=2$ leave only the terminal step and
endpoint projection is degenerate there. **$\nfe=3$ is the smallest budget at which endpoint projection
has a real guiding step**, and the projector comparison needs it. The `u17cv2` campaign is matched on it:
`mf|K1,K3,K5`, `af|K1,K3,K5`, `fm|K1,K3,K5`, `diffusion|K20`.

🟠 **UAV-pillars is currently planned at $\{1,2,5\}$ (R23), which does not match.** Recommendation:
move R23 to $\{1,3,5\}$. At $\nfe=2$ the pillars scene could never support the same projector comparison
the corridor carries, and the two scenes could not be read on one ladder.

### UAV-corridor: the diffusion row's variant is not the flow rows' variant

The three flow rows of `tab:uav-corridor` are `dpcc-t-tightened`. The diffusion row is
`dpcc-t-bounds_free-pdes-tightened` — same projector and geometry, action bound released, plan anchored
on the commanded rather than the measured position. It is the **only** projected corridor variant the
baseline was evaluated under. Now disclosed in a `\guard`. A matched `dpcc-t-tightened` diffusion corridor
cell would remove the last confound in that table.

### UAV-corridor: why the baseline scores $0/12$ passed while flying the corridor

Not a scoring bug. `success_relaxed = crossed_line and safe`; the finish line is the plane through the
expert endpoint **normal to that route's final approach heading**, and the routes turn downward on exit.
Projected, the baseline holds its lane, ends between $x=2.66$ and $2.79$ (inside the passing flights'
range) but on the near side of the line, runs the full 396-step limit every flight and ends $0.51$\,m from
the goal. `phys_safe = 1`, `constraint_collision_free = 1`. The figure encoding was the real defect and is
fixed (dotted stroke, enlarged hollow end mark).

### UAV-s-curve: budgets are ragged, and one table could be matched

| campaign | mf | af | fm | diffusion |
| :-- | :-- | :-- | :-- | :-- |
| `@u7hg` (brake-to-rest, unprojected) | K2, K10 | K5 | K2, K20 | K20 |
| `@u6unet_ae02` | — | K1, K2, K5 | — | — |
| `@u18sc` (switched-wall re-run) | K10 | K5 | K20 | — |

- ✅ **`tab:uav-scurve` is now matched at $\nfe=2$** for all three flow models — the `af|K2` cell existed
  and was not published. (`af|K5` appears in both `@u7hg` and `@u6unet_ae02` with identical numbers, which
  is what makes the two tags safe to read together.) The other budgets are kept in a separate block
  labelled as not a comparison.
- ❌ **`tab:uav-scurve-aborts` cannot be matched**: the `@u18sc` re-run has exactly one budget per model
  ($20$, $10$, $5$). Every cell is $0.00$ on both outcome columns, so nothing is ranked, but the table now
  says so outright.

### New item

| ID | Priority | Missing work | Training? | Draft effect |
| :-- | :-- | :-- | :-- | :-- |
| **R29** | 🟡 | UAV-s-curve: repeat the switched-wall endpoint-projection re-run (`u18sc`) at **one shared budget** across MeanFM, CI-MeanFM and FM, ten flights per variant | no | makes `tab:uav-scurve-aborts` a comparison instead of three per-model records. **Low value while every cell is $0.00$** — the scene's failure is the tracking controller, not the projector |
| **R30** | 🟢 | UAV-corridor: the diffusion baseline under plain `dpcc-t-tightened`, twelve flights | no | removes the `bounds_free`/`pdes` confound from the one baseline row of `tab:uav-corridor` |

---

## 16 · 2026-09-22 — UAV-pillars (R23), restated here with a corrected budget ladder

> ⛔ **SUPERSEDED 2026-09-22, later the same day.** The campaign this section specified ran to completion and is **abandoned** (all-zero grid, every projector takes the forbidden centre lane). Nothing below is to be run. See §21 and the closure record.

**This section is the live specification for the pillars campaign.** It supersedes the budget list in the
older pillars PENDING/RUNBOOK files for the purposes of the thesis; those files are kept as history and
are not to be run from.

### 🔴 The change: $\{1,2,5\} \rightarrow \{1,3,5\}$

R23 was specified at $\nfe\in\{1,2,5\}$, matching neither UAV-corridor nor the projector study. The reason
the corridor uses $\{1,3,5\}$ is mechanical, and it applies to pillars unchanged:

> At the scene's activation threshold $\eta=0.5$ the projector acts on the last $\lceil \eta\nfe \rceil$
> sampling steps. At $\nfe=1$ and $\nfe=2$ that is the terminal step alone, where endpoint projection
> degenerates into projection after sampling and has nothing to guide. **$\nfe=3$ is the smallest budget
> at which endpoint projection does any of its own arithmetic.**

So a pillars campaign at $\nfe=2$ can never support the endpoint-versus-per-step comparison that
UAV-corridor carries, and the two scenes cannot be read on one ladder. Running $\{1,3,5\}$ costs exactly
the same as $\{1,2,5\}$ — three budgets either way — and buys the projector comparison.

### The run matrix

| axis | values | count |
| :-- | :-- | :-- |
| geometry | `pillars_xl` (keep-out radius $0.35$\,m; the MJCF pillar stays at $0.12$) | 1 |
| seed | 6 | 1 |
| model × budget | MeanFM, CI-MeanFM, FM at $\nfe\in\{1,3,5\}$; diffusion at $\nfe=20$ | 10 cells |
| projection variant | unprojected, plus per-step and endpoint under $r$, $c$, $t$, tightened and untightened | as U17 |
| flights per cell | 10 (the scene's four homotopy classes, balanced as far as 10 allows) | — |

> ⏱ **TIME RULE for any per-step projection at $\nfe=20$ on the quadrotor — learned on 25951, applies to
> every future row of this kind (corridor, s-curve, pillars alike).** Projection cost scales with the
> budget: measured `proj_ms` per control step is 96–265 ms at $\nfe\le2$, 2.3–3.4 s at $\nfe=5$ and
> **10.4 s at $\nfe=20$**. At ten flights of ~424–634 steps that is **~18.5 h for one `r` variant**, and
> the `c`/`t` rules run 1.2–1.3× slower, i.e. **over the 24 h cap on their own**. Therefore: never more
> than one $\nfe=20$ per-step variant per job; budget `c`/`t` rules as *not fitting* unless `n_trials` is
> cut, which breaks the ten-flight protocol; the eval cannot shard a variant across jobs (episode ids
> are `10_000 + i`). Cost any such line item at ~1 GPU-day per variant before listing it as a run.

Diffusion is at $\nfe=20$ only: its step count is fixed when the noise schedule is discretised at
training, so each extra budget is a training run, and no cheaper pillars checkpoint exists.

### 🟠 Status, 2026-09-22 — the campaign is running; nothing here is a new run yet

**Author, 2026-09-22:** the pillars campaign is on the cluster under Gen15 U17, Slurm jobs up to
**25995**, and is close to complete. The author updates the real Slurm status when all pillars jobs
end. Until then this ledger lists **no** pillars run as new. Everything below the run matrix is the
target specification, not a work order.

| question | answer | who settles it |
| :-- | :-- | :-- |
| Which budget ladder is the running wave on, $\{1,2,5\}$ or $\{1,3,5\}$? | ✅ **$\{1,2,5\}$** — `Gen15_U17.sh` groups A (K=1,2), B/C (K=5). So by this section's own logic the **$\nfe=3$ rung is the one genuinely new run** (3 cells: mf/af/fm at K=3, eleven variants). 🔴 **Do not schedule it yet** — see the first-read row below | settled 2026-09-21 from the driver and the child logs |
| Does the diffusion $\nfe=20$ arm complete inside the 24 h cap? | ❌ **No — and ✅ CLOSED BY DECISION 2026-09-21: it will not be re-run.** Job 25951 was `CANCELLED … DUE TO TIME LIMIT` with `diffuser` and `dpcc-r` delivered (`dpcc-r` alone 67,033 s = **18.6 h**; `proj_ms` **10.4 s per control step**, against 2.3–3.4 s for the flow models at $\nfe=5$). Completing the six per-step variants is ~118–130 h ≈ 5 GPU-days, one variant per 24 h job at best, and the `c` rules (~24–26 h projected) do not fit at all at ten flights. Endpoint projection is unavailable to this arm by construction. **The thesis reports the diffusion baseline on pillars from its raw network output only** (`success = 0.000`, `relaxed = 1.000`, `safe = 1.000`, `track_err = 0.319`) and states the time limit; the delivered `dpcc-r` cell is kept on disk, not printed. Full table: runbook §3d | decided 2026-09-21 |
| Which projection variants ran per cell, and at what geometry tag? | ✅ geometry tag `pillars_xl_bounds+dynamics+geo_bounds+obstacles`, eval tag `u7xl`, seed 6, 10 flights. **K=1, 2 (all three flow models):** `diffuser`, `dpcc-{r,c,t}` ± tightened = 7 (HardFlow is degenerate there and dropped by the eval). **K=5:** the same 7 plus `hardflow_new{,-r,-c,-t}` = 11, split over two jobs. **Diffusion K=20:** `diffuser`, `dpcc-r` only (see above) | settled 2026-09-21 from the child logs; `results.json` census still to do |
| GPU cost | wall-times, attempt 2 (each child on one GPU): K=1 ≈ 1.1–4 h, K=2 ≈ 1.4–9 h, K=5 half-1 (5 per-step) ≈ 9–24 h, K=5 half-2 (2 per-step + 4 HardFlow) ≈ 13 h, diffusion 24 h (walled). **≈ 5 GPU-days for the flow models, plus ~4 more GPU-days if the diffusion remainder is run.** Per control step: `proj_ms` K=1 122–265 ms, K=2 96–162 ms, K=5 per-step 2.3–3.4 s, HardFlow K=5 77–313 ms, diffusion K=20 10.4 s | settled 2026-09-21 from `JOB START`/`JOB END` and the `TIMING` lines |

**What is fixed regardless of the answers.** The old `pillars_hg` wave (13 cells, $\{1,2,5\}$ + diffusion
$\nfe=20$) is withdrawn from the thesis: that geometry enforced exactly the clearance its own
demonstration generator kept, so every demonstrated route already satisfied it and the projector had
nothing to repair. No `pillars_hg` number may fill a thesis slot. The one `u7xlchk` cell (`mf K5`,
unprojected, seed 6, 42 rows) was the pre-flight check, not the campaign. Checkpoints, demonstrations
and driver are reused; only the constraint changed, so **no training** is involved either way.

🔴 **First read of the delivered wave, 2026-09-21 — read before scheduling the $\nfe=3$ rung or the
diffusion remainder.** 10 of 12 flow-model children are complete, 2 (fm and af, K=5 endpoint half) are
running and on track; no crash, every finished cell has all its variants. But the result is a **floor**:
of the 54 finished *projected* cells, **53 read `success = 0.00` and one reads 0.10** (S&C ≤ success, so
those are S&C 0.00). The unprojected rows reproduce their `pillars_hg` values (mf/fm 0.90 at K=5). At
`pillars_hg` the same K=5 cells read S&C 0.80–1.00. So the scene went from "every plan feasible,
projector idle, models unordered" to "projector repairs every plan into one that never reaches the goal,
models unordered" — and **neither regime orders the four models**, which is the only reason the section
exists. At $\nfe\le2$ the projected signature is uniform: `success_relaxed=1.00, safe=1.00,
goal_reached=0.00` — line crossed, contact-free, **goal missed laterally by more than 0.30 m**. The
leading hypothesis is a metric/geometry mismatch: the goal sits at $(3.2, \pm1.11)$ with radius 0.30,
the constraint forces the lane out to $|y|\ge1.26$, nothing brings it back, and the measured tracking
error (0.34–0.45 m) exceeds the 0.15 m of radius left. If so the projected flights may be
constraint-satisfying and fail only the goal test — fixable at evaluation time, but then the wave is
re-scored or re-run either way. The alternative (non-converged SLSQP iterates being flown — 25993 logs
`[hardflow][NLP-FAILURE]` on every HardFlow variant) is not excluded. **Four `results.json` files
separate the two**; they are named in the runbook §3e. Until they are read, **no further pillars compute**
— the $\nfe=3$ rung, the diffusion remainder and `pillars_xxl` all wait, and `pillars_xxl` is off the
table in either outcome (the scene is too hard for the metric, not too easy).

⛔ **VERDICT, 2026-09-22 — the batch is in, and U17 is ABANDONED.** Author decision the same day: **fall back to `pillars_hg`** — v3 restores `v3/withheld/20260918_uav_pillars_section.tex` (three tables, flown-path figure) with the caveat that the unprojected flows are already feasible there, and removes the `pillars_xl` templates. `pillars_hg` is therefore **no longer withdrawn**; the "What must not happen" bullet below about `pillars_hg` is superseded. Closure: `Gen15/U17/CLOSURE_20260922_U17_abandoned.md`. 94 cells × 10 flights: **S&C = 0.00 and collision-free =
0.00 in every cell, unprojected rows included.** Projection adds violating steps rather than removing
them: every projector routes the plan into the **centre corridor between the pillar rows** — physically
open, closed by the enforced keep-out by only 6 cm — contact-free and across the finish line (relaxed
success 1.00) but ~1 m from the goal point and 0.4 m inside the keep-out at every column; per-step at
$\nfe=5$ additionally hits pillars. The goal-radius hypothesis of the first read is refuted; the local
projectors prefer the forbidden lane to the 0.15 m outer detour, and the scene ranks nothing. **`tab:uav-pillars`,
`tab:uav-pillars-projection` and `fig_uav_pillars_xl_paths` are not filled and the section stays
withheld.** The $\nfe=3$ rung, the diffusion remainder and `pillars_xxl` are all closed. Analysis of
record: `DA_in_Paper/analysis/DA_20260922_pillars_xl_wave.md`; runbook §3f.

~~**Uncertain, therefore not marked as a run:** the $\nfe=3$ rung.~~ The mechanical case for it stands
(§15 and the block above), but whether it is *already* in the running wave is the first question in the
table. **Do not schedule it before that is answered.**

### What it fills in the draft

- `tab:uav-pillars` — the generative-model table, currently a visible `\hole`. **Its `\hole` text still
  says $\{1,2,5\}$; set it to whatever ladder the delivered wave actually carries**
  (`06_results.tex`, `sec:res:uav:pillars`).
- `tab:uav-pillars-projection` — the per-step versus endpoint table, which is the reason $\nfe=3$ is in
  the ladder at all.
- `fig_uav_pillars_xl_paths` — the executed-path figure, currently a `\todofigure`. Needs the rollout
  artefacts transferred, and their outcomes checked before anything is drawn.

### What must not happen

- ~~🔴 **No `pillars_hg` number may fill these slots.**~~ **SUPERSEDED 2026-09-22 — `pillars_hg` is the fallback and is restored, with its caveat in the prose.** Original text kept for the record: The old geometry enforced exactly the clearance its
  own demonstration generator was built to keep, so every demonstrated route satisfied it and the scene
  measured nothing. It is withdrawn from the thesis.
- 🔴 **The `u7xlchk` verification is one unprojected cell, not the campaign.** It confirms the enlarged
  geometry is wired correctly and nothing more.
- 🔴 **The tightening knob is not the enlargement.** `enlarge_constraints` stays at $0.025$; the enlarged
  keep-out radius lives in the entry's own `radius`
  (`config/uav_projection.yaml:876`). The two must stay separable, as they are in Chapter 5.

---

## 17 · 2026-09-22 — new items opened by the v3.60 census

| ID | Priority | Scope | Training? | Why it is or is not owed |
| :-- | :-- | :-- | :-- | :-- |
| **R28** | 🟡 | D3IL-avoiding: FM at $\nfe=5$ on seeds 7--10, and FM at $\nfe=10$ at the standard $T{=}0.5$ on all five seeds; two episodes per seed and geometry, unprojected plus the three tightened rules | no | would complete the FM column at the two intermediate budgets MeanFM and diffusion already carry. **Not owed** — nothing in the draft claims those cells, and FM is operated at $\nfe\in\{1,2\}$ |
| ~~**R29**~~ | ⬜ **superseded by R31, v3.61** | ~~UAV-s-curve: repeat the switched-wall endpoint-projection re-run at one shared budget~~ | no | the matched grid of §18 contains this run; `tab:uav-scurve-aborts` itself moved to the appendix as `tab:app:uav-scurve-aborts` |
| **R30** | 🟢 | UAV-corridor: the diffusion baseline under plain `dpcc-t-tightened`, twelve flights | no | removes the `bounds_free`/`pdes` confound from the one baseline row of `tab:uav-corridor`. The result is not expected to move — the baseline fails that scene's goal criterion under the variant it was run with — but the row would then be matched to the flow rows |

**R23 is running, so it is not on the affordability list.** Of the items that are, R2 and R30 are the
only ones that close a gap the draft states out loud; every other open item refines a number the draft
already prints, or removes a caveat it already carries.


---

## 18 · v3.61 — UAV-s-curve is printed as a matched grid, and the grid is mostly empty (R31)

**Author, 2026-09-22:** *"the S_curve feels still ridiculous, we are lacking much of the proper runs
right? … build the table first then I will fill in slowly, mark the missing lines."* Agreed, and done: the
scene's two main-text tables now have the corridor's shape — one budget ladder for all three flow-based
models, diffusion at its own budget — and every cell that has not been flown is printed as
\emph{pending}. The ragged per-model records (FM 20, MeanFM 10, CI-MeanFM 5) moved to
`app:uav-scurve-budgets` and are **not** used to fill grid cells at another budget.

Census, 2026-09-22, from `per_rollout_detail.csv` of `batch_uav_20260915_100816` and
`batch_uav_20260919_111701`, geometry `s_curve_hg_*`, seed 6, ten flights per cell:

### `tab:uav-scurve` — unprojected, $\nfe\in\{1,2,3,5\}$ + diffusion 20

| model | 1 | 2 | 3 | 5 | source of the filled cells |
| :-- | :-- | :-- | :-- | :-- | :-- |
| MeanFM | ⬜ pending | ✅ | ⬜ pending | ⬜ pending | `u7hg` |
| CI-MeanFM | ✅ | ✅ | ⬜ pending | ✅ | `u6unet_ae02` (1, 2), `u18sc` (5) |
| FM | ⬜ pending | ✅ | ⬜ pending | ⬜ pending | `u7hg` |
| Diffusion | — | — | — | ✅ at $\nfe=20$ | `u7hg` |

$\nfe=2$ is kept beside the corridor ladder $\{1,3,5\}$ because it is the one budget at which all three
flow models already exist; drop it if the scene should mirror the corridor exactly.

### `tab:uav-scurve-projection` — post-correction, $\nfe\in\{3,5\}$, per-step $t$ + endpoint single/$r$/$c$/$t$

| model | 3 | 5 |
| :-- | :-- | :-- |
| MeanFM | ⬜ pending (5 variants) | ⬜ pending (5 variants) |
| CI-MeanFM | ⬜ pending (5 variants) | ✅ `u18sc` |
| FM | ⬜ pending (5 variants) | ⬜ pending (5 variants) |
| Diffusion, per-step $t$ at $\nfe=20$ | — | ✅ `u7hg` (pre-correction; per-step is not what the correction touched) |

🟠 **Only the corrected endpoint run (`u18sc`) counts for the projected grid.** The `u7hg` and
`u6unet_ae02` campaigns carry `hardflow_sls*` cells at CI-MeanFM 5, FM 20 and MeanFM 10 from **before**
the switched-wall correction; they are not printed anywhere. Per-step cells from those campaigns (e.g.
CI-MeanFM 5 `dpcc-t`: 5/10 aborted in `u7hg`, 7/10 in `u18sc`) differ between runs by sampling, and the
grid takes the `u18sc` value where one exists.

### R31 — exact scope

| ID | Priority | Missing work | Training? | Draft effect |
| :-- | :-- | :-- | :-- | :-- |
| **R31** | 🟡 | **Unprojected:** MeanFM at $\nfe=1,3,5$; CI-MeanFM at $\nfe=3$; FM at $\nfe=1,3,5$ — 7 cells × 10 flights, brake-to-rest, `s_curve_hg`, seed 6. **Projected, post-correction:** MeanFM at $\nfe=3,5$; CI-MeanFM at $\nfe=3$; FM at $\nfe=3,5$ — 5 cells × {`dpcc-t`, `hardflow_sls`, `hardflow_sls-r`, `hardflow_sls-c`, `hardflow_sls-t`} × 10 flights | no | fills the 7 + 5 \emph{pending} cells of `tab:uav-scurve` and `tab:uav-scurve-projection`. Expected outcome: S\&C stays $0.00$ (the scene fails in the controller), so what it buys is a matched abort/pass/cost comparison, not a ranking |

**Rough cost, same convention as §0 (±50 %):** unprojected ~1 h in total (9–45 ms/step, 871-step limit);
projected ~6 h per cell — per-step $t$ at ~1.4 s/step dominates — so ~30 h for the five cells, split
one job per cell. **The tightened/untightened choice for the per-step arm is open**: the `u18sc` re-run
used plain `dpcc-t`, the corridor uses `dpcc-t-tightened`; the grid follows `u18sc` so its one filled
cell is comparable, and this should be confirmed before the jobs are written.

---

## 19 · 2026-09-22 — an executable driver exists for the affordable items

[`SLURM_RUNBOOK_20260922_all_lacking_runs.md`](SLURM_RUNBOOK_20260922_all_lacking_runs.md) and
`Slurm_Codes/temp_bash/pipeline_20260922_all_lacking_runs.sh` (PLAN by default). Groups in ledger
order: **A** R2 · **B** R26 (four trainings → one five-seed eval) · **C** R30 · **D/E** R31 unprojected /
projected · **F** R16 (fm, af evals + diffusion $\nfe=10$ train → eval) · **G** R15 and **H** R28 opt-in.
Nothing here changes a decision above: R23 has **no** group (§16 gate), R25/R18 stay struck, R7/R14/R9
stay unwritten until their prerequisites are settled. One choice is left open for E — plain `dpcc-t`
(as `u18sc`) or `dpcc-t-tightened` (as the corridor) — see the runbook §5 E.

## 20 · v3.62 — Table 6.5 prints the whole budget ladder, and eleven of its twenty cells are empty (R16 widened)

The author asked for $\nfe=1$, $2$ and $10$ back in `tab:va-models` — *"just showing K20 is not enough … build THE
TABLE and mark lack for missing data"*. The table now prints the ladder $\{1, 2, 10, 20, 100\}$ for all four
models with *pending* in every unevaluated cell, and the v3.61 appendix copy (`tab:app:va-models-full`,
`app:aligning-budgets`) is removed as redundant. Corpus: `15-09/batch_va2_20260915_100754`, geometry
`combined_5`, variant `diffuser`, keyed on the `H8_K<n>_…` folder prefix.

### Census, unprojected, ten shared contexts

| $\nfe$ | MeanFM | CI-MeanFM ($\alpha_{\mathrm{end}}=0.2$) | FM | Diffusion |
| :-- | :-- | :-- | :-- | :-- |
| 1 | ⬜ pending | ⬜ pending | ⬜ pending | ⬜ pending (checkpoint) |
| 2 | ✅ 0.2779 (39 %) | ✅ 0.3738 (17 %) | ⬜ pending | ⬜ pending (checkpoint) |
| 10 | ✅ 0.1194 (74 %) | ⬜ pending | ⬜ pending | ⬜ pending (checkpoint) |
| 20 | ✅ 0.0741 (84 %) | ✅ 0.3897 (14 %) | ✅ 0.4085 (10 %) | ✅ 0.4619 (−2 %) |
| 100 | ✅ 0.0673 (85 %) | ✅ 0.1791 (60 %) | ✅ 0.4247 (6 %) | ✅ 0.1849 (59 %) |

**No model has an $\nfe=1$ cell on this task** — the corpus has no `H8_K1_` folder for `combined_5/diffuser`.
The $\nfe=1$ row is the one the author did not name explicitly ("1,2,10 or pick whatever you think we need");
it is printed because MeanFM's one-evaluation regime is the thesis's central claim on the other two scenes.

### R16, widened — exact scope

| arm | cells | needs |
| :-- | :-- | :-- |
| flow models, evaluation only | MeanFM 1 · CI-MeanFM 1, 10 · FM 1, 2, 10 | the existing seed-6 checkpoints; ~20 min per cell at the recorded ms/step × 10 contexts |
| diffusion | 1, 2, 10 | **one training per chain length** (the K=20 and K=100 checkpoints are separate models already); then three evaluations |

Cost: ~3 h per diffusion training (the v3.60 figure for one), so ~9 h training, plus ~3 h for the eleven
evaluations. ±50 %. The two `R16 extension` rows (FM at $\nfe=2$) are absorbed.

### Also done in this pass, no run needed

`fig_aligning_tradeoff` had **two table rows missing**: the CI-MeanFM and FM cells at $\nfe=100$ that v3.60
added to the table were never added to the figure's cell registry (`sources.ALIGNING_CELLS/REPORTED`). Both
reproduce the table exactly from the corpus (0.1791 m / 902.2 ms; 0.4247 m / 1425.0 ms) and are now drawn.
The frontier is unchanged: both are dominated by MeanFM at $\nfe=20$.

## 21 · v3.63 — UAV-pillars falls back to `pillars_hg`; the enlarged campaign (R23) is abandoned

**Trigger:** `cross_draft/to_v3/FROM_DA_20260922_pillars_u17_abandoned_fallback_hg.md`, on the author's decision.
The `pillars_xl` wave (94 cells × 10 flights, jobs 25970–25995) is **S&C 0.00 and collision-free 0.00 in every
cell, unprojected rows included**: the enlarged keep-out closes the channel between the pillar rows by
centimetres only, and every projector routes into it. Post-mortem `DA_in_Paper/analysis/DA_20260922_pillars_xl_wave.md`.

**What v3.63 did in the draft**

| where | change |
| :-- | :-- |
| `06_results.tex` §6.3.1.2 | the withheld `pillars_hg` section is **back verbatim** (prose, `tab:uav-pillars`, `tab:uav-pillars-endpoint`, `tab:uav-pillars-best`, `fig:uav-pillars-paths`), with the v3.62 *success* wording, its caveat as the first paragraph, and one pointer to the abandoned attempt; the v3.56 `pillars_xl` templates, `\hole`s and `	odofigure` are gone |
| `06_results.tex` §6.3 intro, cost-frontier paragraph, §6.3.3 projection, §6.3.5 conclusion | read the `pillars_hg` result again (the projection paragraph's 43/180, 89/180, 1611 ms and 0/120, 41/120, 104 ms **re-verified against the 15-09 corpus today**) |
| `05_setup.tex` §5.2.3, `tab:uav-scenes`, its `\srcnote`, `fig:constraints-uav` and `fig:expert-uav` captions, `tab:protocol` | back to the 0.12 m pillar; a pillars row in the protocol table |
| `05_setup.tex` §5.3.3 (expert data) | **new paragraph, author's ask:** the enlargement is described and *why it is not reported* — none of the demonstrated routes satisfied the enlarged set; in evaluation the flights crossed the finish line (success) while violating at every column, so nothing was collision-free and the scene ranked nothing |
| figures | `sources.UAV_CONSTRAINTS` pillars back to r = 0.12; `expert_paths.json` re-extracted (pillars 4/4 clean); `fig_constraints_uav`, `fig_expert_uav`, `fig_uav_pillars_paths` rebuilt and exported |

**Not done, and why.** §6.4 *Comparison across Environments* is 🔒 locked by the author (v3.60): its three summary
tables still have **no UAV-pillars row** and its first sentence still says "each of the three environments that
yield a result". The rows to add when the section is unlocked, in the v3.62 wording, are in the v3.63 changelog.

**Runs.** None new. The restored section carries its old `\hole` (MeanFM and FM at $
fe=1$ on `pillars_hg`);
R12/R13/R21/R22 stay ⏸ superseded unless the author reopens them. R23 is closed. R2, R30, R31, R16 unchanged.
