# PENDING — 2026-09-20 · complete current list of missing runs and artefacts

**Updated for thesis v3.53.** This compact checklist supersedes the earlier version of this file,
which listed only newly discovered gaps and therefore could not answer “what is still missing?” alone.
The historical ledger remains
[`PENDING_20260916_missing_data_and_analyses.md`](PENDING_20260916_missing_data_and_analyses.md);
the executable groups remain in the two `SLURM_RUNBOOK` files. This file lists every item open now.

## 1 · Runs that close a visible gap in the current thesis

| ID | Priority | Missing work | Training? | Draft effect / blocker |
| :-- | :-- | :-- | :-- | :-- |
| **R2** | 🔴 | D3IL-aligning: tightened diffusion baseline at $\nfe=20$ on the ten thesis contexts | no | closes the baseline row in `sec:res:aligning:projection`; first resolve why the evaluation config says `n_contexts: 3` |
| ~~**R25**~~ | ⬜ **NO LONGER NEEDED — ruled out on cost by the author, v3.52.** The rows it filled are removed from `tab:state-headline`; §6.1.2.7 answers the question from existing data | D3IL-avoiding: CI-MeanFM, $\alpha_{\mathrm{end}}=0.2$, $\nfe=1,2$, all three rules, **20 episodes per seed and geometry**, seeds 6--10 | no | ~~fills the two pending rows of `tab:state-headline`~~ — those rows are removed (§10); CI-MeanFM is reported at seed 6 in `tab:avoiding-budget20`. Would refine, not open, a claim |
| ~~**R26**~~ | ⬜ **NO LONGER NEEDED — ruled out on cost by the author, v3.52** (four training runs). The row is removed from `tab:avoiding-dpcc-protocol`; its caption now states why the baseline has no $\nfe=2$ row | D3IL-avoiding: diffusion $\nfe=2$, training seeds 7--10, then DPCC-protocol and extended evaluations | **four trainings** | ~~fills the explicit $\nfe=2$ row of `tab:avoiding-dpcc-protocol`~~ — row removed (§10); the caption now states why the baseline has no $\nfe=2$ row |
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
| **R16** | D3IL-aligning budget ladder: FM at $\nfe=2,10$ and CI-MeanFM at $\nfe=10$ | tests whether the MeanFM budget trend extends across objectives |
| **R17** | repeat the projected alignment configurations | measures run-to-run variation |
| **R27** | diffusion baseline under endpoint projection on UAV-corridor at $\nfe=20$, single/$r$/$c$/$t$, twelve flights each | endpoint projection has never been evaluated on diffusion in any environment |

R11 is retired: v3.47 removes the subsection it was meant to support. R24 is closed: all eight
raw-plan panels are present.

## 3 · Prepared but outside the current draft

**R23 — UAV-pillars at `pillars_xl`.** The enlarged geometry, driver and identity check exist, but
the scene is absent from Results. Run it only if UAV-pillars returns. R12, R13, R21 and R22 remain
superseded with it.

## 4 · Transfers or logging changes, not model runs

| ID | Work | What is needed |
| :-- | :-- | :-- |
| ~~**D10a**~~ | ~~D3IL-avoiding executed paths~~ | ✅ **CLOSED at v3.50.** 20 of 21 cells staged; `extract/exec_paths.py` → `data/exec_paths.json` → `fig_avoiding_paths`, now `fig:avoiding-paths` in §6.1.2.2. Drawn on `top-right-hard`. **Superseded at v3.53:** the author replaced the four-model panel set with the two average-velocity models at $\nfe=1$ and $\nfe=2$ (all four cells present in the same drop), so the figure no longer needs the diffusion `both-hard` artefact at all |
| ~~**D10d**~~ | ~~`fig:raw-plans`, the diffusion baseline at its own budget $\nfe=20$~~ | ✅ **CLOSED at v3.49.** The panel was fetched from the complete sibling campaign `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` (seed 6, `both-hard`) — the run this row originally named died mid-variant and never wrote the dashboard. `fig_raw_plans_diffusion_K20` is in the store, exported to `v3/figures/`, and wired into `fig:raw-plans` as the $\nfe=20$ row. **The draft now carries no `\todofigure` at all.** Details in §8 below |
| ~~**D10b**~~ | ~~D3IL-aligning end-effector paths~~ | ✅ **CLOSED at v3.50.** The 09-19 drop already carried them: `obs_all` is `(10, 400, 6)` at `combined_5-tightened`, seed 6, for all three variants. Built as `fig_aligning_paths`, now `fig:aligning-paths` in §6.2; its three panel counts reproduce the $\nfe=20$ rows of `tab:va-projection` (2/10, 9/10, 10/10) |
| **D10c** | D3IL-aligning box paths | add per-step box-pose logging, then re-evaluate MeanFM $\nfe=20$ for unprojected, per-step and endpoint variants |
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
