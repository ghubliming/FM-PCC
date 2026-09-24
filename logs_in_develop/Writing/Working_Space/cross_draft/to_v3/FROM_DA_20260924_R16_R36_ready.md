# TO v3 — R16 and R36 are ready: Table 6.5's three *pending* rows and Table 6.3's three *lacking* cells

**2026-09-24 · from the DA side.** Nothing in `v3/` touched. Analysis of record:
`Data_Analysis/DA_in_Paper/analysis/DA_20260924_R16_R36_must_need.md` (INDEX row added). Jobs 26181–26186, all
verified. Both new batches reproduce every cell Tables 6.3 and 6.5 already print, so no existing row moves.
After this, the draft's only open data cells are the quadrotor ones (R33 corridor, R44 s-curve).

## Table 6.5 `tab:va-models` — the three pending rows (median · mean · unmoved · ms/step)

| row | values |
| :-- | :-- |
| FM, $\nfe=2$ | 0.4257 (6 %) · $0.375 \pm 0.133$ · 3/10 · $32.5 \pm 2.5$ |
| CI-MeanFM, $\alpha_{\mathrm{end}}=0.2$, $\nfe=10$ | 0.4247 (6 %) · $0.399 \pm 0.094$ · 2/10 · $98.5 \pm 4.0$ |
| FM, $\nfe=10$ | 0.4224 (7 %) · $0.415 \pm 0.085$ · 4/10 · $145.1 \pm 2.0$ |

## Table 6.3 `tab:avoiding-projectors` — the three lacking cells (S&C · steps · ms/step)

| row | seeds | per-step $t$ | endpoint |
| :-- | :-- | :-- | :-- |
| CI-MeanFM, $\alpha_{\mathrm{end}}=0.2$, $\nfe=3$ | 4 (7–10) | 0.958 · 60.0 · 146 | **1.000** · 60.0 · **76.1** |
| FM, $\nfe=3$ | 4 (7–10) | 1.000 · 61.2 · **64.1** | 1.000 · 60.8 · 71.5 |
| MeanFM, $\nfe=10$ | 1 (6) | 1.000 · 64.2 · 362 | 1.000 · 64.8 · **176** |

Other rules for the prose (S&C per-step against endpoint): CI-MeanFM K3 $r$ 0.875/0.875, $c$ 0.792/1.000;
FM K3 1.000/1.000 under both; MeanFM K10 $r$ **0.500**/1.000, $c$ 0.833/0.833. Endpoint paths lengthen under
$c$ on every model (78–97 steps).

## Prose to revisit

1. **Table 6.5:** FM closes 6–10 % at every budget from 2 to 100 — flat. CI-MeanFM is not monotone
   (17 → 6 → 14 → 60 %); at $\nfe=10$ it sits at FM's level. MeanFM leads every budget; at $\nfe=10$ it dominates
   FM (lower distance, lower time) and matches CI-MeanFM's cost (99.2 vs 98.5 ms), so "at the same cost", not
   "dominates", against CI-MeanFM.
2. **Table 6.3, K3:** "on the one model that has the matched cell" is now three models, paired on seeds 7–10.
   Endpoint dominates per-step on both average-velocity models (better S&C, level steps, half the time). **On FM
   it does not** — both reach 1.000 and per-step is the cheaper projector (64.1 vs 71.5 ms). The "factor of two …
   it is the projector" sentence holds for the average-velocity models only.
3. **Table 6.3, K10:** "The analytic average-velocity model has no four-candidate endpoint run … and is lacking"
   is now filled: endpoint is level or ahead and cheaper on all three models (MeanFM 176 vs 362 ms).
4. 🔴 **Pre-existing, found in validation:** the printed **CI-MeanFM K10 row is a twenty-episode cell** (one
   geometry reads 0.95 = 19/20; ten times the solves of the FM K10 cell). The caption's "three geometries × two
   episodes" and the guard's "six episodes per rule" are wrong for that row; the other two K10 rows are two
   episodes. The author decides: disclose, or re-run that one cell at two episodes (~15 min).
5. Per-step time depends on the evaluator (FM 64 ms against 146–148 ms at K3); the within-model comparison the
   table makes is unaffected.
6. Side note: Table 6.5's CI-MeanFM K100 row recomputes as **unmoved 1/10**, not the printed 0/10 (every other
   value matches).
