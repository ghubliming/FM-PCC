# DA — 2026-09-23 · D3IL-aligning: the diffusion per-step row (R2) and the threshold ladder (R37)

**Preliminary, for the thesis agent.** Every number below reproduces with the conventions of
[`va_results.py`](va_results.py), and the new batch reproduces every cell the draft already prints to
the last digit (§2). Both ledger items are **data-complete**. Five things the agent should read before
filling the tables are collected in §6.

**Ledger items:** R2 (Table 6.8, `tab:va-projection-models`, the *pending (R2)* diffusion row) and R37
(Table 6.6, `tab:va-threshold`, the $\nfe=2$ and $\nfe=100$, $\eta=0.1$ rows) of
[`PENDING_20260922_all_lacking_runs.md`](../../../logs_in_develop/Writing/Working_Space/data_status/PENDING_20260922_all_lacking_runs.md).

**Corpus:** `temp/23-09/batch_va2_20260923_210100/per_rollout_detail.csv` (DA run
`run_da_batch_va_v2.sh`, zero arguments, 23-09).
**Runs:** 26113 (R2fix), 26114 (R37a), 26115 (R37b), chained after 26112 via `submit_after.sh`, driver
`pipeline_20260923_red_wave.sh`. All three logs verified: η, K, tag, ten contexts and every item as
intended ([runbook §7](../../../logs_in_develop/Writing/Working_Space/data_status/SLURM_RUNBOOK_20260922_all_lacking_runs.md)).
**Protocol:** seed 6, the ten thesis contexts (fingerprints of MeanFM $\nfe=20$ `combined_5` `diffuser`,
as `va_results.py` defines them), training split, **tightened geometry `combined_5-tightened`**
(all-tightened rule, v3.67), four candidates.
**Columns, verbatim from `va_results.py`:** violation-free = `constraint_exec_zero_violation == 1`;
violating steps = mean `n_violations`; moved = final distance differs from initial by ≥ 1e-6; median and
mean ± sd over the per-context final box-to-target distance; ms/step = mean ± sd over contexts of
`avg_time_ms`; the percent is the median's reduction from the start distance 0.4530 m.

---

## 1 · Verdict

| | as expected? |
| :-- | :-- |
| the runs did what they were asked | ✅ η = 0.2 landed in R2fix's path (`H8_K20_T0.2_…_msgR2fix`); K and η landed in both R37 paths |
| the batch is consistent with the corpus of record | ✅ every printed cell of Tables 6.6, 6.7 and 6.8 reproduces exactly (§2) |
| η is the only difference between R2fix and the discarded 26051 | ✅ the unprojected rows are identical; every projected cell is cheaper at 0.2 (§5.1) |
| R37b agrees with the untightened K100 cells measured before | ✅ within 3 % on time, same clean counts, same medians to 3 mm (§5.2) |
| the diffusion baseline is weak under projection on this task | ✅ **expected** — the withdrawn untightened cell read 2/10 clean; the tightened one reads 1–4/10 |
| **anything wrong?** | 🟠 no pipeline fault; five reading notes in §6, one of them a **ledger error** (R37a was not needed) |

## 2 · Validation — the new batch reproduces the draft

Recomputed from the 23-09 batch, against what the draft prints:

| cell (tightened) | draft | recomputed |
| :-- | :-- | :-- |
| MeanFM $\nfe=20$ per-step $r$ | 9/10 · 2.7 · 8/10 · 0.179 · 0.200 ± 0.150 · 265.9 ± 101.9 | identical |
| MeanFM $\nfe=20$ endpoint $r$ / $c$ / $t$ | 10/10 · 10/10 · 9/10, 275.3 / 282.3 / 301.5 ms | identical |
| FM $\nfe=20$ per-step $r$ | 5/10 · 14.3 · 3/10 · 0.425 · 409.7 ± 147.9 | identical |
| Table 6.6 $\nfe=10$ row | 89.7 · 517.5 · 0.434 · 7/10 · 192.5 · 0.242 · 10/10 | identical |
| Table 6.6 $\nfe=20$ unprojected | 172.5 | identical |

The new cells therefore enter tables whose other rows do not move.

## 3 · R2 — the diffusion per-step row of Table 6.8

Diffusion $\nfe=20$, $\eta=0.2$, tightened, per-step projection. Folder
`H8_K20_T0.2_Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_VTrue_mpc4_filmv1_Ediffusion_msgR2fix`.

| rule | violation-free | violating steps | moved | median [m] | mean [m] | ms/step |
| :-- | --: | --: | --: | :-- | :-- | :-- |
| $r$ | 4/10 | 14.8 | 3/10 | 0.462 (**−2 %**) | 0.397 ± 0.160 | 809.2 ± 756.0 |
| $c$ | 1/10 | 35.0 | 0/10 | 0.456 (**−1 %**) | 0.453 ± 0.037 | 1069.7 ± 848.6 |
| $t$ | 1/10 | 41.1 | 1/10 | 0.462 (**−2 %**) | 0.456 ± 0.039 | 766.0 ± 284.5 |

For reference, its unprojected plan on the same item: 4/10 · 14.2 · 6/10 · 0.473 · 271.6 ± 95.4 ms.
The endpoint row stays a dash: endpoint projection is not defined for the diffusion sampler.

**How it reads against the rest of Table 6.8.** On the operating point the baseline's best per-step
rule is $r$. MeanFM under the same rule — 9/10 clean, 0.179 m, 265.9 ms — has more clean contexts,
a smaller final distance and a third of the time per step, so it **Pareto-dominates the baseline's best
row**. The baseline's $r$ row is level with FM and CI-MeanFM on clean contexts (4–6/10) and costs about
twice their time. Its $c$ and $t$ rows are the lowest clean counts in the table. It moves the box in at
most three contexts, and its median ends slightly farther from the target than where the box started —
the percent is negative, which the table's column has not had to print before.

## 4 · R37 — the two missing rows of Table 6.6

MeanFM, random rule, tightened. Guiding steps $\lceil \eta\nfe \rceil - 1$.

| $\nfe$ | $\eta$ | guiding | ms/step unproj. | per-step ms/step | median [m] | clean | endpoint ms/step | median [m] | clean |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **2** | 0.5 | 0 | **24.6** ± 8.6 | **46.2** ± 18.4 | **0.275** | **8/10** | — no guiding step | | |
| 10 | 0.4 | 3 | 89.7 | 517.5 | 0.434 | 7/10 | 192.5 | 0.242 | 10/10 |
| 20 | 0.2 | 3 | 172.5 | 265.9 | 0.179 | 9/10 | 275.3 | 0.197 | 10/10 |
| **100** | 0.1 | 9 | **924.6** ± 1.6 | **1112.0** ± 477.9 | **0.248** | **8/10** | **1107.0** ± 410.7 | **0.153** | **8/10** |

Bold cells are the new ones. Sources:

- **$\nfe=2$ row — use the cell Table 6.7 already prints**, not R37a. Table 6.6 fills its $\nfe=10$ and
  $20$ rows from "the random-rule cells of `tab:va-projection`", and `tab:va-projection` already carries
  the $\nfe=2$ tightened per-step $r$ cell (8/10 · 0.275 · 46.2 ± 18.4, unprojected 24.6 ± 8.6), from
  folder `H8_K2_Meuler_T0.5_…VisualMeanFlow_VTrue_mpc4_filmv1_Emf` of the 15-09 corpus. Taking R37a
  instead would print two different numbers for one configuration in adjacent tables. See §6.5.
- **$\nfe=100$ projected cells — R37b**, folder `H8_K100_Meuler_T0.1_…VisualMeanFlow_…_Emf_msgR37`.
- **$\nfe=100$ unprojected — 924.6 ± 1.6 ms** from the untagged `H8_K100_Meuler_T0.1_…_Emf` cell, the
  only $\nfe=100$, $\eta=0.1$ unprojected measurement. R37b ran no unprojected arm. η does not enter an
  unprojected plan, and that cell's geometry is the plain one, which does not enter an unprojected plan's
  cost either. The $\eta=0.5$ unprojected cell at the same budget reads 892.4 ± 1.8 ms, so the choice
  moves the number by 3.5 %.

**How the ladder reads with the new row.** The projected step costs 1.9×, 5.8×, 1.5× and 1.2× the
unprojected step at $\nfe = 2, 10, 20, 100$ under per-step projection. At $\nfe=100$ the projector adds
about 190 ms to a 925 ms step, so the compressed threshold does what the section says it does: nine
guiding steps at a hundred evaluations cost a fifth more than no projection at all. **What changes in
the prose:** endpoint projection is **not** 10/10 clean at $\nfe=100$ — it is 8/10, the same as
per-step — while its median is the best of the ladder (0.153 m). The draft's sentence *"Endpoint
projection keeps ten of ten contexts clean at both budgets"* is still true of the two budgets it names,
but it does not extend to the third.

## 5 · The checks

### 5.1 · η is the only difference between R2fix and 26051

Same checkpoint, same ten contexts, same tightened item order; 26051 took η = 0.5 from the yaml.

| variant | η = 0.2 (R2fix) clean · viol · median · ms | η = 0.5 (26051) clean · viol · median · ms |
| :-- | :-- | :-- |
| `diffuser` | 4/10 · 14.2 · 0.473 · 271.6 | 4/10 · 14.2 · 0.473 · 263.2 |
| `dpcc-r` | 4/10 · 14.8 · 0.462 · 809.2 | 3/10 · 20.0 · 0.462 · 1082.7 |
| `dpcc-c` | 1/10 · 35.0 · 0.456 · 1069.7 | 3/10 · 35.1 · 0.456 · 1273.3 |
| `dpcc-t` | 1/10 · 41.1 · 0.462 · 766.0 | 1/10 · 39.6 · 0.462 · 1447.3 |

The unprojected rows are identical, so the two runs sampled the same plans before projection. Every
projected cell is cheaper at η = 0.2 (5 projected denoising steps instead of 11), by 16–47 %. The
outcomes are of the same size, so the fix changes the cost the baseline is charged, not its verdict.

### 5.2 · R37b reproduces the untightened $\nfe=100$ cells measured before

R37b also produced the plain-geometry twin, which the corpus already held from an earlier run:

| plain geometry | R37b (23-09) | earlier run | ledger quote |
| :-- | :-- | :-- | :-- |
| per-step $r$ | 2/10 · 0.191 m · 1163.6 ms | 2/10 · 0.188 m · 1194.8 ms | §14: 0.1876 m, 1,195 ms |
| endpoint $r$ | 6/10 · 0.108 m · 1285.2 ms | 6/10 · 0.108 m · 1308.9 ms | §23: 0.108 m, 1309 ms, 6 clean |

Same clean counts, medians within 3 mm, times within 3 %. The tightened cells come from the same job.

### 5.3 · A first measurement of run-to-run variation (the old R17)

R37a re-ran a cell that already existed (see §6.5), which makes it a replicate:

| $\nfe=2$ tightened per-step $r$ | clean | violating steps | median | ms/step |
| :-- | --: | --: | --: | --: |
| 15-09 corpus (Table 6.7) | 8/10 | 11.6 | 0.275 | 46.2 |
| R37a, 23-09 | 9/10 | 8.4 | 0.243 | 46.4 |

Time reproduces to 0.5 %. Outcomes move by one context and 3 cm, because the projected rollouts are not
bit-identical between runs: the random state an item sees depends on its position in the job. That is
the size of run-to-run variation for one projected alignment cell, and it is worth one sentence where
the chapter compares cells that differ by a single context.

## 6 · Five things to read before filling the tables

**6.1 · 🟠 On the diffusion arm the projector trades geometric violations for action-bound violations.**
Violating steps summed over the ten contexts, by constraint family, tightened:

| diffusion $\nfe=20$, η 0.2 | action bound | halfspace | obstacle |
| :-- | --: | --: | --: |
| unprojected | 64 | 52 | 56 |
| per-step $r$ | 145 | 0 | 3 |
| per-step $c$ | 348 | 0 | 2 |
| per-step $t$ | 408 | 0 | 3 |

The projector does its geometric job — halfspace and obstacle violations fall to almost zero — while
the executed steps that exceed the action-magnitude bound (`bounds`, the dataset-derived action range,
`action_bounds: 'auto'`) grow to two to six times the unprojected count. That is why the clean count is
low even though the plan is geometrically repaired. The MeanFM rows do not show it: R37b's per-step
failures are 7 action-bound and 26 halfspace steps over two contexts. **Why the action bound gives on
this arm is not established:** the eval records no solve counts for the diffusion arm, so a
non-converged solve cannot be told apart from an executed action that differs from the projected one.
State the family; do not state the cause.

**6.2 · 🟠 On the tightened geometry, one of the ten contexts is never attempted.** Context 8 (box at
0.534, −0.133) does not conflict with the plain obstacle, but once the obstacle is enlarged by the
tightening margin (δ = 0.03 m) the box overlaps it by 0.0193 m against a 0.005 m tolerance. The eval's
box/obstacle guard then aborts the rollout, holds position and logs *"EXCLUDE FROM METRICS"* — in
R2fix's log it fires on the four tightened items only. `va_results.py` does not exclude it, so on
`combined_5-tightened` it scores **clean and unmoved in all 205 of its rollouts across the FiLM-v1 cells
of this corpus**, while on the plain geometry it is attempted and the box moves in 189 of 247. Every
"x/10" in the **tightened** alignment tables (6.6, 6.7, 6.8) therefore means "x − 1 of nine attempted
contexts, plus one held"; Table 6.5, which is unprojected on the plain geometry, is unaffected.
Comparisons between rows are unaffected because every tightened row carries it. It matters for this row
in one place: the diffusion $c$ and $t$ rules' single clean context **is** the held one, so on the nine
attempted contexts they are 0/9. The convention predates these runs; whether to state it is the author's
call.

**6.3 · 🟠 Endpoint projection at $\nfe=100$, $\eta=0.1$ is 8/10, and its solver did not always
converge.** Both non-clean contexts carry small action-bound violations (3 and 6 steps). The job log
records a non-converged SLSQP solve at τ = 0.910 on both geometries (`[hardflow][NLP-FAILURE]`), after
which failures are counted silently. That count is not in the per-rollout CSV (the circuit-breaker
columns `projection_cb_tripped` and `projection_cb_skipped_steps` are zero everywhere), so whether the two
non-clean contexts are the non-converged ones cannot be read from the corpus. Report the 8/10; do not
attribute it.

**6.4 · The diffusion medians have a negative percent.** The box ends farther from the target than it
started (0.456–0.462 m against 0.4530 m). The column prints "(−2 %)", not "(2 %)".

**6.5 · 🔴 Ledger error: R37a was not needed.** §24 of the ledger listed the $\nfe=2$ tightened
per-step pair as new, but Table 6.7 already printed it from the 15-09 corpus. The run was not wasted — it
is the replicate of §5.3 — but Table 6.6's $\nfe=2$ row should cite Table 6.7's cell, as its $\nfe=10$
and $20$ rows already do.

## 7 · What this closes

| | |
| :-- | :-- |
| **R2** | ✅ data-complete — Table 6.8's diffusion per-step row, §3. The four-model before/after-projection comparison of §6.2 can now be made; the diffusion endpoint cell stays a dash for good |
| **R37** | ✅ data-complete — Table 6.6's $\nfe=2$ row (from Table 6.7) and $\nfe=100$, $\eta=0.1$ row (R37b), §4. $\nfe=100$, $\eta=0.5$ stays struck (author, 23-09) |
| draft text to revisit | the *"no tightened per-step cell"* guard and prose around Table 6.8 (§3); *"ten of ten contexts clean at both budgets"* (true, but not at the third, §4); anything reading a one-context difference between alignment cells (§5.3) |
| not changed | every other cell of Tables 6.6, 6.7 and 6.8 (§2) |

## 8 · Reproducing

Scratch script, conventions of `va_results.py` unchanged (not yet added to this folder; the permanent
home is new `CELLS` keys in `va_results.py`, a code edit that waits for a go-ahead):

```
corpus    temp/23-09/batch_va2_20260923_210100/per_rollout_detail.csv
contexts  fingerprints (box x,y; target x,y, 3 dp) of H8_K20_Meuler_T0.2_…VisualMeanFlow_VTrue_mpc4_filmv1_Emf, combined_5, diffuser
geometry  combined_5-tightened
R2        H8_K20_T0.2_Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_VTrue_mpc4_filmv1_Ediffusion_msgR2fix
          variants dpcc-r, dpcc-c, dpcc-t   (diffuser for the unprojected reference)
R37 K2    H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_VTrue_mpc4_filmv1_Emf        (Table 6.7's cell)
          …_filmv1_Emf_msgR37                                                                                          (R37a, replicate)
R37 K100  H8_K100_Meuler_T0.1_…VisualMeanFlow_VTrue_mpc4_filmv1_Emf_msgR37   variants dpcc-r, hardflow_sls-r
          H8_K100_Meuler_T0.1_…VisualMeanFlow_VTrue_mpc4_filmv1_Emf, combined_5, diffuser   (unprojected ms)
```

---

## 9 · Addendum — R35 is already in the corpus; no run is needed

Checked 23-09 against the same batch, before any R35 job was written. The ledger (§23) records Table 6.8's
CI-MeanFM endpoint block as *never run*. It was run: the untagged folder
`H8_K20_Meuler_T0.2_…VisualAlphaFlow_VTrue_mpc4_filmv1_Eaf_EPlatest` carries `hardflow_sls-r/c/t` on
`combined_5-tightened`, ten contexts each. The census looked only at the tagged `_msgafon02_s6` run,
which has no endpoint arm.

**Same model, same checkpoint.** Both folders sit under
`…_Eaf_tslogit_normal_afschsigmoid_AFAFend0p2` (α floor 0.2, the thesis's CI-MeanFM) and both deploy the
latest checkpoint (`EPlatest`). They are two runs of one configuration, and their per-step blocks agree:

| per-step, tightened | clean · moved · median | violating steps | ms/step |
| :-- | :-- | :-- | :-- |
| $r$ — printed (`_msgafon02_s6`) / untagged | 4/10 · 8/10 · 0.429, both | 44.6 / 45.5 | 415.8 / 420.3 |
| $c$ | 5/10 · 5/10 · 0.443, both | 10.7 / 10.7 | 311.4 / 311.4 |
| $t$ | 6/10 · 7/10 · 0.422, both | 23.2 / 22.9 | 320.5 / 319.9 |

**The block Table 6.8 is waiting for** — CI-MeanFM, $\nfe=20$, $\eta=0.2$, tightened, endpoint projection:

| rule | violation-free | violating steps | moved | median [m] | mean [m] | ms/step |
| :-- | --: | --: | --: | :-- | :-- | :-- |
| $r$ | 8/10 | 2.7 | 7/10 | 0.425 (6 %) | 0.333 ± 0.183 | 294.2 ± 113.3 |
| $c$ | 6/10 | 10.1 | 4/10 | 0.456 (−1 %) | 0.440 ± 0.076 | 314.8 ± 119.6 |
| $t$ | 5/10 | 8.6 | 6/10 | 0.434 (4 %) | 0.368 ± 0.124 | 344.8 ± 164.6 |

Two ways to print it, the author's choice: (a) keep the printed per-step block and add this endpoint
block, disclosing that the two come from two runs of the same checkpoint; or (b) take both blocks from
the untagged run, which makes the comparison within one run and moves only the $r$ and $t$ violating
steps and times by the amounts above. (b) is the cleaner read of a projector comparison.

**What it changes in the prose around Table 6.8.** The draft counts six comparisons of the two projectors
on the same model, budget and rule, endpoint ahead in five. With CI-MeanFM there are nine: endpoint keeps
more contexts clean in seven (CI-MeanFM $r$ 8 against 4, $c$ 6 against 5) and per-step in two, the new
exception being CI-MeanFM under temporal consistency (6 against 5), by one context — the same rule and
margin as the FM exception. Endpoint is also cheaper for CI-MeanFM under $r$ (294 against 416 ms) and
dearer under $t$ (345 against 321 ms). The §6.2 caveat about held context 8 applies here too.

