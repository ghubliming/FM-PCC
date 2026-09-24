# DA 2026-09-24 · UAV-corridor v3 (R33) — every Chapter 6 corridor item (complete)

> **R45a (2026-09-24) — read beside:** [`DA_20260924_corridor_v3_R45a_clear_line.md`](DA_20260924_corridor_v3_R45a_clear_line.md)
> re-scores this corpus with the finish line moved to the end of the corridor (x′ = 2.0 m) and prints both rules. Everything
> below is the **original rule** and stays correct as such. Under x′ only the baseline's six projected cells change in S&C
> (tilt r/c/t 0/0/0 → 3/8/3, hump 3/3/6 → 10/9/9), and with them the baseline rows of §3.2–§3.3, the frontier (§3.4) and the
> conclusion facts (§3.7): on the tilt the baseline per-step c (8/10, 666 ms) becomes the only configuration with S&C > 0; on
> the hump FM $\nfe=20$ endpoint single (8/10, 291 ms) and the baseline per-step r (10/10, 654 ms) are a trade-off.

**Status: complete — all 136 cells, tables and flown paths.** All 62 corridor jobs finished every variant (130 cells × 12
flights, 6 × 10; no failure, no rerun). Tables and frontier: the DA_UAV_v1 batch of the full corpus,
`temp/23-09-FULL/24-09-1000/batch_uav_20260924_081422/per_rollout_detail.csv`. Flown paths: the npz of every cell in
`temp/23-09-Corridor-TEMP/plans` (the 23-09 copy completed with `p23cv3_paths_20260924_103108.tar.gz`, §2). Computed by
[`corridor_v3_grid.py`](corridor_v3_grid.py). Supersedes [`DA_20260923_corridor_v3_preliminary.md`](DA_20260923_corridor_v3_preliminary.md).
Chapter 6 (v3 draft) is **not** edited; the writing agent transcribes from §3.

Scenes: `corridor_v3_tilt` = the v2 slide leaned −60° about z_ref 1.11 m (tag `p23cv3t`), the corridor v3;
`corridor_v3_ablation_hump` = the x–z roof, H 1.10 m (tag `p23cv3ah`), the ablation. Protocol: seed 6, **the first ten flights**
of each cell (routes L, C, R cycling, 4/3/3); trial *i* is seeded by its index, so the first ten of a 12-flight cell are the
flights a 10-flight cell flies. Model names as in the chapter; ★ = the diffusion baseline.

---

## 1 · Verdict

No failed run, no code bug, no wrong setup. Two structural results decide what the scene can show:

1. **Tilt (the corridor v3): S&C is 0/10 in all 68 cells.** Every projected flow flight — 53 cells, 530 flights — crosses the
   finish line and descends under the leaned plane (median 0.12–0.37 m below its unprojected twin), and every one leaves a
   residue of 1.4–11 violating steps in the last quarter-metre of the plane's window (x ∈ [1.75, 2.00]). At the first violating
   step the commanded setpoint is already past the window end (x = 2.27–2.52 m) and 0.49–0.61 m ahead of the vehicle; the
   setpoint is feasible at 100 % of the violating steps. The residue shrinks with the budget (depth 8.8–11.9 cm at $\nfe \le 2$,
   1.0–2.2 cm at $\nfe=20$) and never reaches zero. The baseline's projected flights reach the line on 0–3 of 10 flights (the
   others run out of steps 0.2–0.3 m short) and are violation-free on 3–8 of 10.
2. **Hump (the ablation): two configurations succeed violation-free** — FM at $\nfe=20$ with endpoint projection (single 8/10
   at 291 ms, r 6/10, c 6/10, t 4/10) and the baseline with per-step projection (t 6/10 at 626 ms, r 3/10, c 3/10). Every flow
   model under per-step projection at $\nfe \le 2$ stops on the roof before the apex (18/18 cells, 0 of 180 flights cross); from
   $\nfe=3$ they cross 10/10 with a residue at the apex that shrinks with the budget, 3.5 cm deep at most (setpoint feasible 100 %).

FM $\nfe=20$ endpoint is ahead of the baseline on the hump on both counts: 8 against 6 violation-free successes at 291 against
626 ms per action. The three flow models do not separate from one another on either constraint: at every matched budget and
projector their mean violating steps per flight are within 0.9 of each other, and their altitude changes coincide.

## 2 · Checks

| check | result |
| :-- | :-- |
| job logs: all 62 `uav_mix_eval` children of the chain, incl. the four C5 jobs 26163–26166 | **every job finished every requested variant**; no traceback / timeout / cancellation; 68 cells per scene |
| batch coverage (`per_rollout_detail.csv`, `data_quality.csv`) | 136 cells, 1 620 flights (130 × 12, 6 × 10); source npz for all; 0 projection cut-off trips; timing present everywhere; HardFlow guiding steps 1 ($\nfe=3$), 2 (5), 9 (20), degenerate 0 |
| fetch `p23cv3_paths_20260924_103108.tar.gz` (52 cells, 96 MB) | 142 files, **all md5 checksums verify** (staging and in place); 0 empty files; the 52 files that were already readable locally are byte-identical to the fetched ones |
| batch CSV vs npz, every flight of every cell | identical (6 490 values compared before the fetch; the three tables are unchanged after it) |
| geometry each run used (24 eval-folder snapshots) vs repo yaml | 0 differences |
| code across the 5 git revisions the waves used | identical for the eval, projector, HardFlow, configs, sbatch, scenes |
| violating steps re-scored from every flown path with the eval's scorer | **1 360 flights, 0 disagree** |
| divergence aborts | 0 |

## 3 · The Chapter 6 corridor items

Counts of 10 flights; mean ± sample standard deviation over the ten flights.

### 3.1 `tab:uav-corridor-raw` — before projection

| block | model | nfe | success | violation-free | S&C | violating steps | ms/step |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| tilt | MeanFM | 1 | 10/10 | 0/10 | 0/10 | 91.3 ± 34.8 | 9.4 ± 0.4 |
| tilt | MeanFM | 2 | 10/10 | 0/10 | 0/10 | 92.9 ± 30.2 | 18.1 ± 0.4 |
| tilt | MeanFM | 3 | 10/10 | 0/10 | 0/10 | 92.0 ± 30.4 | 27.0 ± 0.4 |
| tilt | CI-MeanFM | 1 | 10/10 | 0/10 | 0/10 | 90.0 ± 32.6 | 9.3 ± 0.4 |
| tilt | CI-MeanFM | 2 | 10/10 | 0/10 | 0/10 | 93.8 ± 30.0 | 18.4 ± 0.4 |
| tilt | CI-MeanFM | 3 | 10/10 | 0/10 | 0/10 | 93.1 ± 30.1 | 27.3 ± 0.4 |
| tilt | FM | 1 | 10/10 | 0/10 | 0/10 | 74.5 ± 25.6 | 9.1 ± 0.4 |
| tilt | FM | 2 | 10/10 | 0/10 | 0/10 | 83.9 ± 28.4 | 17.7 ± 0.4 |
| tilt | FM | 3 | 10/10 | 0/10 | 0/10 | 87.0 ± 28.3 | 25.9 ± 0.5 |
| tilt | FM | 5 | 10/10 | 0/10 | 0/10 | 89.9 ± 28.4 | 43.4 ± 0.4 |
| tilt | FM | 20 | 10/10 | 0/10 | 0/10 | 94.7 ± 27.0 | 170.8 ± 0.4 |
| tilt | Diffusion ★ | 20 | 10/10 | 0/10 | 0/10 | 99.4 ± 27.0 | 178.1 ± 0.8 |
| hump | MeanFM | 1 | 10/10 | 0/10 | 0/10 | 36.0 ± 8.9 | 9.4 ± 0.4 |
| hump | MeanFM | 2 | 10/10 | 0/10 | 0/10 | 35.6 ± 8.1 | 18.2 ± 0.4 |
| hump | MeanFM | 3 | 10/10 | 0/10 | 0/10 | 35.7 ± 7.8 | 27.2 ± 0.4 |
| hump | CI-MeanFM | 1 | 10/10 | 0/10 | 0/10 | 36.6 ± 8.4 | 9.3 ± 0.4 |
| hump | CI-MeanFM | 2 | 10/10 | 0/10 | 0/10 | 35.6 ± 7.9 | 18.0 ± 0.4 |
| hump | CI-MeanFM | 3 | 10/10 | 0/10 | 0/10 | 35.6 ± 7.6 | 27.1 ± 0.4 |
| hump | FM | 1 | 10/10 | 0/10 | 0/10 | 34.7 ± 6.8 | 9.1 ± 0.4 |
| hump | FM | 2 | 10/10 | 0/10 | 0/10 | 32.6 ± 7.8 | 17.5 ± 0.3 |
| hump | FM | 3 | 10/10 | 0/10 | 0/10 | 32.6 ± 8.0 | 26.3 ± 0.4 |
| hump | FM | 5 | 10/10 | 0/10 | 0/10 | 32.0 ± 7.9 | 43.6 ± 0.4 |
| hump | FM | 20 | 10/10 | 0/10 | 0/10 | 31.3 ± 7.6 | 172.3 ± 2.6 |
| hump | Diffusion ★ | 20 | 10/10 | 0/10 | 0/10 | 28.6 ± 7.7 | 178.2 ± 2.7 |

**Reading (hole "Result sentence, before projection").** Every model at every budget crosses the finish line on every flight, and
no flight is violation-free on either constraint: the tilt is crossed on 74.5–99.4 control steps per flight, the hump on
28.6–36.6. The flow models sit within one standard deviation of each other (tilt 74.5–94.7, hump 31.3–36.6); the baseline has the
most violating steps on the tilt (99.4) and the fewest on the hump (28.6). Time per action unprojected: 9.1–9.4 ms at $\nfe=1$,
17.5–18.4 at 2, 25.9–27.3 at 3, 43.4–43.6 at 5, 170.8–172.3 at 20, the baseline 178.1–178.2. Because every model reaches the goal
unprojected, no before-projection frontier is drawn (the chapter's rule).

### 3.2 `tab:uav-corridor` — after projection, the best rule per projector

Best rule = most S&C; ties (almost everywhere, since S&C is 0) → fewer mean violating steps → lower ms/step.

| block | model | nfe | per-step S&C | rule | ms/step | endpoint S&C | rule | ms/step |
| :-- | :-- | --: | --: | :-- | --: | --: | :-- | --: |
| tilt | MeanFM | 1 | 0/10 | c | 27.4 | — | — | — |
| tilt | MeanFM | 2 | 0/10 | c | 36.4 | — | — | — |
| tilt | MeanFM | 3 | 0/10 | c | 93.6 | 0/10 | c | 73.5 |
| tilt | CI-MeanFM | 1 | 0/10 | c | 27.9 | — | — | — |
| tilt | CI-MeanFM | 2 | 0/10 | c | 36.1 | — | — | — |
| tilt | CI-MeanFM | 3 | 0/10 | c | 93.4 | 0/10 | c | 72.7 |
| tilt | FM | 1 | 0/10 | c | 27.6 | — | — | — |
| tilt | FM | 2 | 0/10 | c | 34.7 | — | — | — |
| tilt | FM | 3 | 0/10 | c | 79.2 | 0/10 | c | 71.6 |
| tilt | FM | 5 | 0/10 | c | 116.9 | 0/10 | c | 115.2 |
| tilt | FM | 20 | 0/10 | c | 360.4 | 0/10 | c | 416.9 |
| tilt | Diffusion ★ | 20 | 0/10 | c | 666.3 | — | — | — |
| hump | MeanFM | 1 | 0/10 | c | 26.4 | — | — | — |
| hump | MeanFM | 2 | 0/10 | r | 35.0 | — | — | — |
| hump | MeanFM | 3 | 0/10 | c | 91.1 | 0/10 | c | 72.1 |
| hump | CI-MeanFM | 1 | 0/10 | t | 26.6 | — | — | — |
| hump | CI-MeanFM | 2 | 0/10 | r | 35.4 | — | — | — |
| hump | CI-MeanFM | 3 | 0/10 | c | 89.2 | 0/10 | c | 71.9 |
| hump | FM | 1 | 0/10 | c | 26.2 | — | — | — |
| hump | FM | 2 | 0/10 | r | 34.5 | — | — | — |
| hump | FM | 3 | 0/10 | c | 77.1 | 0/10 | c | 71.9 |
| hump | FM | 5 | 0/10 | c | 114.1 | 0/10 | c | 112.4 |
| hump | FM | 20 | 0/10 | c | 355.2 | 8/10 | single | 291.0 |
| hump | Diffusion ★ | 20 | 6/10 | t | 625.9 | — | — | — |

**Reading (hole "Result sentence, after projection").** On the tilt no configuration succeeds violation-free, at any budget,
under either projector. On the hump the violation-free successes are FM $\nfe=20$ under endpoint projection (8/10, 291 ms) and the
baseline under per-step projection (6/10, 626 ms); every flow model under per-step projection scores 0/10 — stopped on the roof
at $\nfe \le 2$, a residue at the apex from $\nfe=3$.

**Budget paragraph facts.** The residue falls monotonically with the budget on both constraints and both projectors:

| mean violating steps per flight | $\nfe=1$ | 2 | 3 | 5 | 20 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| tilt, per-step (all rules) | 10.2–10.9 | 9.8–11.1 | 5.0–7.5 | 4.4–5.4 | 2.1–3.2 (FM); baseline 0.8–6.1 |
| tilt, endpoint (all rules) | — | — | 6.9–9.2 | 4.9–6.3 | 1.4–2.5 |
| hump, per-step (all rules) | 0 (stopped on the roof) | 0 (stopped; 17.9 / 19.8 in the two wall-drift cells) | 4.6–6.5 | 3.9–4.3 | 1.8–2.2 (FM); baseline 0.0–0.4 |
| hump, endpoint (all rules) | — | — | 8.0–10.4 | 3.9–5.6 | 0.2–0.7 |

| deepest violating step per cell (m) | $\nfe \le 2$ | 3 | 5 | 20 |
| :-- | :-- | :-- | :-- | :-- |
| tilt, per-step | 0.088–0.119 | 0.033–0.069 | 0.034–0.040 | 0.013–0.022 (FM); baseline 0.013–0.071 |
| tilt, endpoint | — | 0.055–0.083 | 0.037–0.048 | 0.010–0.013 |
| hump, per-step | — (no violation, stopped) | 0.023–0.030 | 0.017–0.019 | 0.008–0.011 (FM); baseline 0.013 |
| hump, endpoint | — | 0.017–0.035 | 0.016–0.022 | 0.001–0.006 |

Price per action (ms): per-step 26–29 ($\nfe=1$), 35–43 (2), 77–94 (3; FM 77–79, MeanFM and CI-MeanFM 89–94), 112–118 (5),
354–360 (20), the baseline 623–680; endpoint 44–74 (3), 73–115 (5), 289–425 (20). The baseline's projected flights take
377–396 control steps.

### 3.3 `tab:uav-corridor-projection` — S&C under every rule ($\nfe \ge 3$ in the chapter; $\nfe = 1, 2$ added)

| block | model | nfe | per-step r | c | t | endpoint single | r | c | t |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| tilt | MeanFM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | MeanFM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | MeanFM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | CI-MeanFM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | CI-MeanFM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | CI-MeanFM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | FM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | FM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | FM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | FM | 5 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | FM | 20 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | Diffusion ★ | 20 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | MeanFM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | MeanFM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | MeanFM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hump | CI-MeanFM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | CI-MeanFM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | CI-MeanFM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hump | FM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | FM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | FM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hump | FM | 5 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hump | FM | 20 | 0/10 | 0/10 | 0/10 | 8/10 | 6/10 | 6/10 | 4/10 |
| hump | Diffusion ★ | 20 | 3/10 | 3/10 | 6/10 | — | — | — | — |

**Reading (hole "Corridor result sentence, per-step against endpoint").** At $\nfe=3$ and 5 the two projectors are level on S&C
(0/10) on both constraints, endpoint projection at the same or lower price (single 44–74 ms vs per-step 77–94 at $\nfe=3$; 73–115
vs 112–118 at 5), with more violating steps at $\nfe=3$ (tilt 6.9–9.2 vs 5.0–7.5, hump 8.0–10.4 vs 4.6–6.5) and about one more
at 5. At $\nfe=20$ endpoint projection leaves the fewest and shallowest violating steps on the tilt (1.4–2.5 steps, ≤ 1.3 cm, vs
2.1–3.2 steps, ≤ 2.2 cm) and is the only flow configuration that succeeds violation-free on the hump (4–8/10). The endpoint rule
order on the hump at $\nfe=20$: single 8 > r 6 = c 6 > t 4. Within a projector the rules differ by at most two violating steps.

### 3.4 `fig:uav-corridor-tradeoff` — the frontier

| block | eligible (S&C > 0) of the best-rule points | frontier |
| :-- | :-- | :-- |
| tilt | 0 of 17 | none |
| hump | 2 of 17 (FM $\nfe=20$ endpoint single 8/10 at 291 ms; baseline per-step t 6/10 at 626 ms) | **FM $\nfe=20$, endpoint, single** — 8/10 at 291 ms; the baseline point is dominated (fewer successes at twice the time) |

### 3.5 `fig:uav-corridor-paths` — facts for the paths paragraph

- **Unprojected**, both constraints: every flight follows its route and crosses the line; the body enters the leaned plane from
  x ≈ −1.3 at the earliest to the window end, and the roof over the apex region (x ∈ [−0.5, 0.5]); deepest step per cell 0.37–0.48 m.
- **Projected, tilt, every flow model**: sideways and down under the plane with a few centimetres of clearance, then a clip of the
  plane in the last 0.25 m of the window while the setpoint is already past it (diagnostic figure (a),
  [`DA_20260923_corridor_v3_preliminary/fig_diag_tilt_residue_hump_stall.png`](DA_20260923_corridor_v3_preliminary/fig_diag_tilt_residue_hump_stall.png));
  10/10 across the line at every budget.
- **Projected, hump**: at $\nfe \le 2$ onto the roof's limit and stopped 0.02–0.17 m before the apex at 1.39–1.50 m (figure (b)),
  every flow model alike; two cells (MeanFM $\nfe=2$ t, CI-MeanFM $\nfe=2$ t) have 3 flights each that get over the apex and then
  drift sideways into the corridor wall. From $\nfe=3$ over the apex and back toward the route altitude, 10/10 across the line.
- **Baseline, projected**: descends and climbs like the flow models but flies slower; the flights that do not cross (tilt 7–10 of
  10, hump 3–7 of 10) pass the constraint and run out of steps 0.16–0.39 m before the line (final x 2.41–2.64).
- **The budget at which a model stops violating** the constraint: never on the tilt; on the hump FM at $\nfe=20$ with endpoint
  projection (and the baseline at its own budget). **The budget at which it stops reaching the end**: $\nfe \le 2$ under per-step
  projection on the hump, all three flow models; never on the tilt.

### 3.6 `fig:uav-corridor-altitude` — the altitude numbers

Tilt = minimum z over x ∈ [0.5, 2.0]; hump = maximum z over x ∈ [−0.5, 0.5]; Δz = projected − unprojected, same trial index;
per cell the median over its ten flights (full per-cell list in Appendix B).

| median Δz per cell (m) | MeanFM | CI-MeanFM | FM | Diffusion ★ |
| :-- | :-- | :-- | :-- | :-- |
| tilt (descent) | −0.13 … −0.26 | −0.12 … −0.26 | −0.19 … −0.37 | −0.22 … −0.30 |
| hump (climb) | +0.33 … +0.39 | +0.33 … +0.39 | +0.27 … +0.36 | +0.32 … +0.34 |

**Reading (hole "Altitude numbers per constraint").** Unprojected, every model flies level in a 0.83–1.27 m band. Projected,
every flight of every cell moves altitude the way its constraint asks: under the tilt it descends by a median 0.12–0.37 m — more
with the budget (MeanFM / CI-MeanFM 0.12–0.19 at $\nfe \le 2$, 0.21–0.26 at 3; FM 0.19–0.25 at $\nfe \le 2$, 0.35–0.37 at 20) —
and over the hump it climbs by a median 0.27–0.39 m (single flights up to 0.60 m), at every budget and under both projectors.
The pilots' values (−0.15 to −0.29, +0.30 to +0.40) sit inside these ranges.

### 3.7 The conclusion line and the §6.4 rows

- **`sec:res:uav:conclusion`, UAV-corridor** (facts to phrase): tilt — every flow model at every budget flies the descent and
  crosses the line, none violation-free (a hand-over residue of 1.4–11 steps at the window end, the setpoint clean throughout); the
  baseline reaches the line on at most 3 of 10 flights. Hump — FM at $\nfe=20$ with endpoint projection succeeds violation-free
  8/10 at 291 ms, ahead of the baseline's best, 6/10 at 626 ms; per-step projection of the flow models stops on the roof at
  $\nfe \le 2$. The hole's expected shape ("average-velocity models ahead of instantaneous-velocity matching on the constraint")
  is **not** in the data: at every matched budget and projector the mean violating steps of MeanFM, CI-MeanFM and FM are within
  0.9 of each other (e.g. tilt per-step $\nfe=3$: 7.1 / 6.2 / 6.4).
- **§6.4 rows quoting the v2 corridor** (`tab:summary-models` "MeanFM, CI-MeanFM (12/12)", `tab:summary-projection` "per-step at
  K3 … endpoint at most 7 of 12", `tab:summary-combinations` "MeanFM K3 per-step t, 12/12") — **none holds on v3.** On the
  numbers above: the best corridor configuration is FM $\nfe=20$ endpoint single on the hump (8/10, 291 ms); on the tilt no
  combination succeeds violation-free. (The chapter's 🔒 applies — author's call.)

## 4 · The two structural results

### 4.1 Tilt: the window-end hand-over residue
Every violating step of every projected cell lies at x ∈ [1.75, 2.00]; depths as in §3.2; setpoint feasible at 100 %. At the first
violating step of each of the 530 projected flow flights the setpoint is at x = 2.27–2.52 m (median 2.38), 0.49–0.61 m ahead of
the vehicle (median 0.55). The gate `x_active: [-2.0, 2.0]` reads the setpoint's x under `-pdes` (U16 fix); once the setpoint is
past x = 2 the plan is no longer bound by the plane and returns toward the route, and the vehicle, half a metre behind, follows it
into the plane.

### 4.2 Hump: the stop at $\nfe \le 2$
18/18 per-step cells at $\nfe=1, 2$: 0 of 180 flights cross, every flight on the 396-step cap, stopped on the tightened planning
limit (1.39–1.50 m) 0.02–0.17 m before the apex. Causes acting together: plans are clipped to the demonstrated altitude box
(0.90–1.30 m; `LimitsNormalizer.unnormalize`) — on the box top 55–61 % of the time — while the apex needs 1.485 m (1.515
tightened); the minimum-norm correction is taken in normalised coordinates, where a metre of climb costs ≈ 200 metres of retreat
along x; the roof's rising row is applied to the whole horizon while the setpoint's x < 0. The two 7/10 cells (MeanFM $\nfe=2$ t,
CI-MeanFM $\nfe=2$ t): 3 flights each clear the roof, then drift sideways into the corridor wall (y −0.73 to −0.97 against the
±0.64 limit, several also into the wall-end caps; 0–3 roof steps) and end 0.95–1.79 m from the goal — sideways drift at an
altitude no demonstration flew.

### 4.3 Options for the author (none applied)

| option | what | cost |
| :-- | :-- | :-- |
| A · report as is | Chapter 6 reads the corridor on S&C (0 on the tilt) plus violating steps, depth and descent/climb, with the hand-over residue and the stop named | none |
| B · a secondary tilt metric | violation-free **while the setpoint is inside the window** (the residue after the hand-over excluded), from the stored npz — no run | minutes |
| C · re-fly the tilt with the gate on the vehicle's x (or the window extended to x = 2.5) | config/code change; tilt waves again | ~1 GPU-day |
| D · hump: per-horizon-point row selection | code change in `setup_dpcc_projector`; hump re-run | code + runs |

## 5 · Reproduce

```bash
python3.14 Data_Analysis/DA_in_Paper/analysis/corridor_v3_grid.py --json cells.json      # defaults: the batch above + temp/23-09-Corridor-TEMP/plans
python3.14 Data_Analysis/DA_in_Paper/analysis/DA_20260923_corridor_v3_preliminary/make_diag_fig.py
```

---

## Appendix A · Every projected cell: outcome, where the residue is, setpoint, plan box

| block | model | nfe | variant | success | viol-free | S&C | strict | viol. steps | steps | residue x (5–95 %) | max depth m | setpoint clean | plan on box top / bottom |
| :-- | :-- | --: | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- |
| hump | MeanFM | 1 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 36.0 ± 8.9 | 270 | [-0.53, +0.56] | 0.476 | 0.69 | 0.00 / 0.07 |
| hump | MeanFM | 1 | dpcc-c | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.60 / 0.00 |
| hump | MeanFM | 1 | dpcc-r | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.59 / 0.00 |
| hump | MeanFM | 1 | dpcc-t | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.61 / 0.00 |
| hump | MeanFM | 2 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 35.6 ± 8.1 | 271 | [-0.52, +0.53] | 0.458 | 0.70 | 0.00 / 0.03 |
| hump | MeanFM | 2 | dpcc-c | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.60 / 0.00 |
| hump | MeanFM | 2 | dpcc-r | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.59 / 0.00 |
| hump | MeanFM | 2 | dpcc-t | 0/10 | 7/10 | 0/10 | 0/10 | 17.9 ± 30.0 | 396 | [+0.33, +1.99] | 0.556 | 0.10 | 0.55 / 0.01 |
| hump | MeanFM | 3 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 35.7 ± 7.8 | 270 | [-0.51, +0.55] | 0.460 | 0.70 | 0.00 / 0.03 |
| hump | MeanFM | 3 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 4.9 ± 0.9 | 310 | [-0.03, +0.05] | 0.023 | 1.00 | 0.22 / 0.00 |
| hump | MeanFM | 3 | dpcc-r | 10/10 | 0/10 | 0/10 | 10/10 | 5.9 ± 0.7 | 312 | [-0.03, +0.06] | 0.029 | 1.00 | 0.21 / 0.00 |
| hump | MeanFM | 3 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 6.2 ± 0.6 | 314 | [-0.03, +0.06] | 0.030 | 1.00 | 0.20 / 0.00 |
| hump | MeanFM | 3 | hf | 10/10 | 0/10 | 0/10 | 10/10 | 10.1 ± 2.1 | 328 | [-0.01, +0.13] | 0.029 | 1.00 | 0.21 / 0.00 |
| hump | MeanFM | 3 | hf-c | 10/10 | 0/10 | 0/10 | 10/10 | 8.0 ± 1.5 | 311 | [-0.03, +0.09] | 0.033 | 1.00 | 0.18 / 0.00 |
| hump | MeanFM | 3 | hf-r | 10/10 | 0/10 | 0/10 | 10/10 | 8.6 ± 2.5 | 330 | [-0.01, +0.11] | 0.028 | 1.00 | 0.21 / 0.00 |
| hump | MeanFM | 3 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 10.0 ± 0.8 | 318 | [-0.02, +0.11] | 0.034 | 1.00 | 0.18 / 0.00 |
| hump | CI-MeanFM | 1 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 36.6 ± 8.4 | 266 | [-0.53, +0.56] | 0.476 | 0.68 | 0.00 / 0.05 |
| hump | CI-MeanFM | 1 | dpcc-c | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.61 / 0.00 |
| hump | CI-MeanFM | 1 | dpcc-r | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.60 / 0.00 |
| hump | CI-MeanFM | 1 | dpcc-t | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.61 / 0.00 |
| hump | CI-MeanFM | 2 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 35.6 ± 7.9 | 270 | [-0.52, +0.54] | 0.461 | 0.70 | 0.00 / 0.04 |
| hump | CI-MeanFM | 2 | dpcc-c | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.60 / 0.00 |
| hump | CI-MeanFM | 2 | dpcc-r | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.60 / 0.00 |
| hump | CI-MeanFM | 2 | dpcc-t | 0/10 | 7/10 | 0/10 | 0/10 | 19.8 ± 32.1 | 396 | [+0.42, +2.07] | 0.275 | 0.24 | 0.55 / 0.01 |
| hump | CI-MeanFM | 3 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 35.6 ± 7.6 | 269 | [-0.52, +0.54] | 0.459 | 0.70 | 0.00 / 0.04 |
| hump | CI-MeanFM | 3 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 4.6 ± 0.8 | 307 | [-0.02, +0.05] | 0.029 | 1.00 | 0.22 / 0.00 |
| hump | CI-MeanFM | 3 | dpcc-r | 10/10 | 0/10 | 0/10 | 10/10 | 5.0 ± 1.1 | 310 | [-0.02, +0.05] | 0.027 | 1.00 | 0.21 / 0.00 |
| hump | CI-MeanFM | 3 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 6.5 ± 0.5 | 312 | [-0.04, +0.06] | 0.029 | 1.00 | 0.20 / 0.00 |
| hump | CI-MeanFM | 3 | hf | 10/10 | 0/10 | 0/10 | 10/10 | 10.4 ± 1.8 | 324 | [-0.01, +0.13] | 0.022 | 1.00 | 0.20 / 0.00 |
| hump | CI-MeanFM | 3 | hf-c | 10/10 | 0/10 | 0/10 | 10/10 | 8.0 ± 1.2 | 308 | [-0.03, +0.09] | 0.028 | 1.00 | 0.18 / 0.00 |
| hump | CI-MeanFM | 3 | hf-r | 10/10 | 0/10 | 0/10 | 10/10 | 9.8 ± 1.9 | 324 | [-0.01, +0.12] | 0.026 | 1.00 | 0.20 / 0.00 |
| hump | CI-MeanFM | 3 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 10.1 ± 1.2 | 315 | [-0.02, +0.12] | 0.035 | 1.00 | 0.18 / 0.00 |
| hump | FM | 1 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 34.7 ± 6.8 | 273 | [-0.53, +0.54] | 0.449 | 0.71 | 0.00 / 0.00 |
| hump | FM | 1 | dpcc-c | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.58 / 0.00 |
| hump | FM | 1 | dpcc-r | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.57 / 0.00 |
| hump | FM | 1 | dpcc-t | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.60 / 0.00 |
| hump | FM | 2 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 32.6 ± 7.8 | 273 | [-0.50, +0.50] | 0.447 | 0.75 | 0.00 / 0.00 |
| hump | FM | 2 | dpcc-c | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.59 / 0.00 |
| hump | FM | 2 | dpcc-r | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.58 / 0.00 |
| hump | FM | 2 | dpcc-t | 0/10 | 10/10 | 0/10 | 0/10 | 0.0 ± 0.0 | 396 | — | — | — | 0.61 / 0.00 |
| hump | FM | 3 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 32.6 ± 8.0 | 274 | [-0.50, +0.50] | 0.439 | 0.75 | 0.00 / 0.00 |
| hump | FM | 3 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 5.6 ± 0.5 | 301 | [-0.03, +0.05] | 0.025 | 1.00 | 0.23 / 0.00 |
| hump | FM | 3 | dpcc-r | 10/10 | 0/10 | 0/10 | 10/10 | 5.9 ± 0.3 | 302 | [-0.02, +0.06] | 0.025 | 1.00 | 0.21 / 0.00 |
| hump | FM | 3 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 5.8 ± 0.4 | 294 | [-0.03, +0.06] | 0.028 | 1.00 | 0.21 / 0.00 |
| hump | FM | 3 | hf | 10/10 | 0/10 | 0/10 | 10/10 | 10.1 ± 1.4 | 338 | [-0.01, +0.11] | 0.018 | 1.00 | 0.21 / 0.00 |
| hump | FM | 3 | hf-c | 10/10 | 0/10 | 0/10 | 10/10 | 8.3 ± 1.1 | 326 | [-0.02, +0.09] | 0.022 | 1.00 | 0.19 / 0.00 |
| hump | FM | 3 | hf-r | 10/10 | 0/10 | 0/10 | 10/10 | 9.3 ± 1.3 | 338 | [-0.01, +0.10] | 0.017 | 1.00 | 0.21 / 0.00 |
| hump | FM | 3 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 9.8 ± 0.8 | 315 | [-0.02, +0.11] | 0.027 | 1.00 | 0.20 / 0.00 |
| hump | FM | 5 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 32.0 ± 7.9 | 272 | [-0.51, +0.49] | 0.433 | 0.76 | 0.00 / 0.00 |
| hump | FM | 5 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 3.9 ± 0.7 | 303 | [-0.02, +0.03] | 0.019 | 1.00 | 0.30 / 0.00 |
| hump | FM | 5 | dpcc-r | 10/10 | 0/10 | 0/10 | 10/10 | 4.2 ± 0.4 | 303 | [-0.02, +0.03] | 0.017 | 1.00 | 0.28 / 0.00 |
| hump | FM | 5 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 4.3 ± 0.5 | 295 | [-0.02, +0.04] | 0.017 | 1.00 | 0.29 / 0.00 |
| hump | FM | 5 | hf | 10/10 | 0/10 | 0/10 | 10/10 | 4.9 ± 0.6 | 321 | [-0.02, +0.04] | 0.020 | 1.00 | 0.24 / 0.00 |
| hump | FM | 5 | hf-c | 10/10 | 0/10 | 0/10 | 10/10 | 3.9 ± 0.3 | 318 | [-0.02, +0.03] | 0.016 | 1.00 | 0.26 / 0.00 |
| hump | FM | 5 | hf-r | 10/10 | 0/10 | 0/10 | 10/10 | 4.8 ± 0.6 | 320 | [-0.02, +0.04] | 0.019 | 1.00 | 0.24 / 0.00 |
| hump | FM | 5 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 5.6 ± 0.5 | 309 | [-0.03, +0.05] | 0.022 | 1.00 | 0.24 / 0.00 |
| hump | FM | 20 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 31.3 ± 7.6 | 272 | [-0.48, +0.48] | 0.417 | 0.77 | 0.00 / 0.00 |
| hump | FM | 20 | dpcc-c | 10/10 | 0/10 | 0/10 | 8/10 | 1.8 ± 0.4 | 326 | [-0.01, +0.01] | 0.008 | 1.00 | 0.48 / 0.00 |
| hump | FM | 20 | dpcc-r | 10/10 | 0/10 | 0/10 | 8/10 | 2.2 ± 0.8 | 328 | [-0.01, +0.02] | 0.010 | 1.00 | 0.46 / 0.00 |
| hump | FM | 20 | dpcc-t | 10/10 | 0/10 | 0/10 | 8/10 | 2.2 ± 0.4 | 321 | [-0.01, +0.01] | 0.011 | 1.00 | 0.47 / 0.00 |
| hump | FM | 20 | hf | 10/10 | 8/10 | 8/10 | 7/10 | 0.2 ± 0.4 | 362 | [+0.00, +0.00] | 0.002 | 1.00 | 0.53 / 0.00 |
| hump | FM | 20 | hf-c | 10/10 | 6/10 | 6/10 | 5/10 | 0.4 ± 0.5 | 367 | [-0.00, +0.00] | 0.002 | 1.00 | 0.54 / 0.00 |
| hump | FM | 20 | hf-r | 10/10 | 6/10 | 6/10 | 7/10 | 0.5 ± 0.7 | 363 | [-0.00, +0.01] | 0.001 | 1.00 | 0.53 / 0.00 |
| hump | FM | 20 | hf-t | 10/10 | 4/10 | 4/10 | 8/10 | 0.7 ± 0.7 | 348 | [-0.00, +0.01] | 0.006 | 1.00 | 0.54 / 0.00 |
| hump | Diffusion ★ | 20 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 28.6 ± 7.7 | 265 | [-0.47, +0.44] | 0.396 | 0.82 | 0.00 / 0.00 |
| hump | Diffusion ★ | 20 | dpcc-c | 3/10 | 10/10 | 3/10 | 3/10 | 0.0 ± 0.0 | 391 | — | — | — | 0.53 / 0.00 |
| hump | Diffusion ★ | 20 | dpcc-r | 3/10 | 10/10 | 3/10 | 1/10 | 0.0 ± 0.0 | 394 | — | — | — | 0.52 / 0.00 |
| hump | Diffusion ★ | 20 | dpcc-t | 7/10 | 9/10 | 6/10 | 5/10 | 0.4 ± 1.3 | 385 | [-0.01, +0.03] | 0.013 | 1.00 | 0.47 / 0.00 |
| tilt | MeanFM | 1 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 91.3 ± 34.8 | 270 | [-1.19, +1.89] | 0.400 | 0.27 | 0.00 / 0.07 |
| tilt | MeanFM | 1 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 10.2 ± 1.1 | 286 | [+1.78, +1.99] | 0.093 | 1.00 | 0.00 / 0.09 |
| tilt | MeanFM | 1 | dpcc-r | 10/10 | 0/10 | 0/10 | 8/10 | 10.4 ± 1.1 | 315 | [+1.78, +1.99] | 0.092 | 1.00 | 0.00 / 0.00 |
| tilt | MeanFM | 1 | dpcc-t | 10/10 | 0/10 | 0/10 | 9/10 | 10.3 ± 1.1 | 300 | [+1.77, +1.98] | 0.100 | 1.00 | 0.00 / 0.01 |
| tilt | MeanFM | 2 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 92.9 ± 30.2 | 271 | [-1.16, +1.88] | 0.401 | 0.27 | 0.00 / 0.03 |
| tilt | MeanFM | 2 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 10.3 ± 0.9 | 287 | [+1.77, +1.99] | 0.091 | 1.00 | 0.00 / 0.09 |
| tilt | MeanFM | 2 | dpcc-r | 10/10 | 0/10 | 0/10 | 5/10 | 11.0 ± 1.2 | 354 | [+1.77, +1.99] | 0.093 | 1.00 | 0.00 / 0.00 |
| tilt | MeanFM | 2 | dpcc-t | 10/10 | 0/10 | 0/10 | 9/10 | 10.6 ± 0.8 | 306 | [+1.77, +1.99] | 0.098 | 1.00 | 0.00 / 0.03 |
| tilt | MeanFM | 3 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 92.0 ± 30.4 | 270 | [-1.16, +1.88] | 0.405 | 0.27 | 0.00 / 0.03 |
| tilt | MeanFM | 3 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 6.6 ± 0.8 | 287 | [+1.84, +1.99] | 0.052 | 1.00 | 0.00 / 0.21 |
| tilt | MeanFM | 3 | dpcc-r | 10/10 | 0/10 | 0/10 | 10/10 | 7.5 ± 0.8 | 286 | [+1.82, +1.99] | 0.069 | 1.00 | 0.00 / 0.19 |
| tilt | MeanFM | 3 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 7.2 ± 1.2 | 292 | [+1.84, +1.99] | 0.054 | 1.00 | 0.00 / 0.20 |
| tilt | MeanFM | 3 | hf | 10/10 | 0/10 | 0/10 | 10/10 | 9.2 ± 1.4 | 286 | [+1.79, +1.99] | 0.080 | 1.00 | 0.00 / 0.14 |
| tilt | MeanFM | 3 | hf-c | 10/10 | 0/10 | 0/10 | 10/10 | 7.6 ± 1.3 | 285 | [+1.82, +1.99] | 0.062 | 1.00 | 0.00 / 0.17 |
| tilt | MeanFM | 3 | hf-r | 10/10 | 0/10 | 0/10 | 10/10 | 8.5 ± 0.8 | 288 | [+1.81, +1.98] | 0.069 | 1.00 | 0.00 / 0.14 |
| tilt | MeanFM | 3 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 9.0 ± 1.2 | 293 | [+1.81, +2.00] | 0.074 | 1.00 | 0.00 / 0.14 |
| tilt | CI-MeanFM | 1 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 90.0 ± 32.6 | 266 | [-1.14, +1.88] | 0.400 | 0.28 | 0.00 / 0.05 |
| tilt | CI-MeanFM | 1 | dpcc-c | 10/10 | 0/10 | 0/10 | 8/10 | 10.3 ± 0.9 | 305 | [+1.77, +1.99] | 0.093 | 1.00 | 0.00 / 0.07 |
| tilt | CI-MeanFM | 1 | dpcc-r | 10/10 | 0/10 | 0/10 | 9/10 | 10.5 ± 1.2 | 307 | [+1.78, +1.99] | 0.092 | 1.00 | 0.00 / 0.00 |
| tilt | CI-MeanFM | 1 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 10.3 ± 1.3 | 293 | [+1.77, +1.98] | 0.100 | 1.00 | 0.00 / 0.02 |
| tilt | CI-MeanFM | 2 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 93.8 ± 30.0 | 270 | [-1.19, +1.88] | 0.414 | 0.27 | 0.00 / 0.04 |
| tilt | CI-MeanFM | 2 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 10.4 ± 0.7 | 287 | [+1.77, +1.99] | 0.089 | 1.00 | 0.00 / 0.10 |
| tilt | CI-MeanFM | 2 | dpcc-r | 10/10 | 0/10 | 0/10 | 6/10 | 10.5 ± 1.4 | 339 | [+1.76, +1.98] | 0.088 | 1.00 | 0.00 / 0.00 |
| tilt | CI-MeanFM | 2 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 10.6 ± 1.4 | 301 | [+1.75, +1.99] | 0.102 | 1.00 | 0.00 / 0.03 |
| tilt | CI-MeanFM | 3 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 93.1 ± 30.1 | 269 | [-1.18, +1.88] | 0.412 | 0.27 | 0.00 / 0.04 |
| tilt | CI-MeanFM | 3 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 5.0 ± 0.8 | 286 | [+1.87, +1.98] | 0.033 | 1.00 | 0.00 / 0.22 |
| tilt | CI-MeanFM | 3 | dpcc-r | 10/10 | 0/10 | 0/10 | 10/10 | 6.6 ± 1.0 | 284 | [+1.84, +1.99] | 0.053 | 1.00 | 0.00 / 0.20 |
| tilt | CI-MeanFM | 3 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 7.0 ± 1.2 | 289 | [+1.84, +1.99] | 0.064 | 1.00 | 0.00 / 0.21 |
| tilt | CI-MeanFM | 3 | hf | 10/10 | 0/10 | 0/10 | 10/10 | 8.8 ± 0.8 | 284 | [+1.80, +1.99] | 0.072 | 1.00 | 0.00 / 0.15 |
| tilt | CI-MeanFM | 3 | hf-c | 10/10 | 0/10 | 0/10 | 10/10 | 7.1 ± 1.0 | 283 | [+1.83, +1.99] | 0.055 | 1.00 | 0.00 / 0.18 |
| tilt | CI-MeanFM | 3 | hf-r | 10/10 | 0/10 | 0/10 | 10/10 | 9.0 ± 0.9 | 285 | [+1.80, +1.99] | 0.077 | 1.00 | 0.00 / 0.13 |
| tilt | CI-MeanFM | 3 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 8.7 ± 1.5 | 291 | [+1.81, +1.99] | 0.083 | 1.00 | 0.00 / 0.14 |
| tilt | FM | 1 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 74.5 ± 25.6 | 273 | [-0.74, +1.90] | 0.369 | 0.33 | 0.00 / 0.00 |
| tilt | FM | 1 | dpcc-c | 10/10 | 0/10 | 0/10 | 10/10 | 10.2 ± 1.5 | 287 | [+1.75, +1.99] | 0.119 | 1.00 | 0.00 / 0.11 |
| tilt | FM | 1 | dpcc-r | 10/10 | 0/10 | 0/10 | 8/10 | 10.2 ± 0.8 | 309 | [+1.76, +1.99] | 0.111 | 1.00 | 0.00 / 0.08 |
| tilt | FM | 1 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 10.9 ± 1.0 | 284 | [+1.75, +1.98] | 0.115 | 1.00 | 0.00 / 0.14 |
| tilt | FM | 2 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 83.9 ± 28.4 | 273 | [-1.04, +1.89] | 0.411 | 0.30 | 0.00 / 0.00 |
| tilt | FM | 2 | dpcc-c | 10/10 | 0/10 | 0/10 | 9/10 | 9.8 ± 1.4 | 296 | [+1.77, +1.99] | 0.109 | 1.00 | 0.00 / 0.10 |
| tilt | FM | 2 | dpcc-r | 10/10 | 0/10 | 0/10 | 6/10 | 10.4 ± 1.1 | 331 | [+1.76, +1.99] | 0.109 | 1.00 | 0.00 / 0.06 |
| tilt | FM | 2 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 11.1 ± 0.9 | 276 | [+1.76, +1.99] | 0.118 | 1.00 | 0.00 / 0.13 |
| tilt | FM | 3 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 87.0 ± 28.3 | 274 | [-1.08, +1.89] | 0.412 | 0.29 | 0.00 / 0.00 |
| tilt | FM | 3 | dpcc-c | 10/10 | 0/10 | 0/10 | 9/10 | 5.8 ± 1.5 | 289 | [+1.85, +1.99] | 0.048 | 1.00 | 0.00 / 0.20 |
| tilt | FM | 3 | dpcc-r | 10/10 | 0/10 | 0/10 | 10/10 | 6.7 ± 1.5 | 276 | [+1.82, +1.99] | 0.059 | 1.00 | 0.00 / 0.20 |
| tilt | FM | 3 | dpcc-t | 10/10 | 0/10 | 0/10 | 10/10 | 6.7 ± 1.2 | 271 | [+1.83, +1.99] | 0.055 | 1.00 | 0.00 / 0.20 |
| tilt | FM | 3 | hf | 10/10 | 0/10 | 0/10 | 10/10 | 8.0 ± 1.6 | 284 | [+1.80, +1.99] | 0.077 | 1.00 | 0.00 / 0.14 |
| tilt | FM | 3 | hf-c | 10/10 | 0/10 | 0/10 | 10/10 | 6.9 ± 1.8 | 288 | [+1.81, +1.98] | 0.072 | 1.00 | 0.00 / 0.15 |
| tilt | FM | 3 | hf-r | 10/10 | 0/10 | 0/10 | 10/10 | 8.2 ± 1.5 | 286 | [+1.81, +1.99] | 0.076 | 1.00 | 0.00 / 0.14 |
| tilt | FM | 3 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 8.2 ± 1.4 | 277 | [+1.81, +1.99] | 0.079 | 1.00 | 0.00 / 0.17 |
| tilt | FM | 5 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 89.9 ± 28.4 | 272 | [-1.13, +1.88] | 0.420 | 0.28 | 0.00 / 0.00 |
| tilt | FM | 5 | dpcc-c | 10/10 | 0/10 | 0/10 | 8/10 | 4.4 ± 1.5 | 303 | [+1.87, +1.99] | 0.034 | 1.00 | 0.00 / 0.23 |
| tilt | FM | 5 | dpcc-r | 10/10 | 0/10 | 0/10 | 8/10 | 5.4 ± 1.4 | 302 | [+1.86, +2.00] | 0.040 | 1.00 | 0.00 / 0.22 |
| tilt | FM | 5 | dpcc-t | 10/10 | 0/10 | 0/10 | 8/10 | 5.3 ± 1.3 | 298 | [+1.87, +1.99] | 0.039 | 1.00 | 0.00 / 0.23 |
| tilt | FM | 5 | hf | 10/10 | 0/10 | 0/10 | 8/10 | 6.0 ± 1.7 | 307 | [+1.84, +1.99] | 0.048 | 1.00 | 0.00 / 0.20 |
| tilt | FM | 5 | hf-c | 10/10 | 0/10 | 0/10 | 8/10 | 4.9 ± 1.9 | 309 | [+1.85, +1.99] | 0.037 | 1.00 | 0.00 / 0.21 |
| tilt | FM | 5 | hf-r | 10/10 | 0/10 | 0/10 | 8/10 | 6.0 ± 1.4 | 309 | [+1.84, +1.99] | 0.048 | 1.00 | 0.00 / 0.19 |
| tilt | FM | 5 | hf-t | 10/10 | 0/10 | 0/10 | 10/10 | 6.3 ± 1.3 | 284 | [+1.85, +1.99] | 0.048 | 1.00 | 0.00 / 0.22 |
| tilt | FM | 20 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 94.7 ± 27.0 | 272 | [-1.20, +1.88] | 0.439 | 0.27 | 0.00 / 0.00 |
| tilt | FM | 20 | dpcc-c | 10/10 | 0/10 | 0/10 | 6/10 | 2.1 ± 0.7 | 338 | [+1.93, +1.99] | 0.013 | 1.00 | 0.00 / 0.32 |
| tilt | FM | 20 | dpcc-r | 10/10 | 0/10 | 0/10 | 7/10 | 3.0 ± 0.9 | 335 | [+1.92, +2.00] | 0.022 | 1.00 | 0.00 / 0.30 |
| tilt | FM | 20 | dpcc-t | 10/10 | 0/10 | 0/10 | 7/10 | 3.2 ± 0.8 | 331 | [+1.93, +1.99] | 0.016 | 1.00 | 0.00 / 0.30 |
| tilt | FM | 20 | hf | 10/10 | 0/10 | 0/10 | 5/10 | 2.3 ± 0.5 | 354 | [+1.93, +2.00] | 0.010 | 1.00 | 0.00 / 0.44 |
| tilt | FM | 20 | hf-c | 10/10 | 0/10 | 0/10 | 4/10 | 1.4 ± 0.7 | 355 | [+1.95, +2.00] | 0.010 | 1.00 | 0.00 / 0.44 |
| tilt | FM | 20 | hf-r | 10/10 | 0/10 | 0/10 | 4/10 | 2.3 ± 0.9 | 354 | [+1.94, +2.00] | 0.013 | 1.00 | 0.00 / 0.44 |
| tilt | FM | 20 | hf-t | 10/10 | 0/10 | 0/10 | 5/10 | 2.5 ± 0.7 | 343 | [+1.94, +1.99] | 0.012 | 1.00 | 0.00 / 0.45 |
| tilt | Diffusion ★ | 20 | diffuser | 10/10 | 0/10 | 0/10 | 10/10 | 99.4 ± 27.0 | 265 | [-1.30, +1.87] | 0.479 | 0.25 | 0.00 / 0.00 |
| tilt | Diffusion ★ | 20 | dpcc-c | 0/10 | 8/10 | 0/10 | 0/10 | 0.8 ± 1.8 | 396 | [+1.97, +2.00] | 0.013 | 1.00 | 0.00 / 0.00 |
| tilt | Diffusion ★ | 20 | dpcc-r | 2/10 | 3/10 | 0/10 | 1/10 | 3.2 ± 3.3 | 378 | [+1.84, +2.00] | 0.066 | 1.00 | 0.00 / 0.10 |
| tilt | Diffusion ★ | 20 | dpcc-t | 3/10 | 3/10 | 0/10 | 1/10 | 6.1 ± 4.6 | 377 | [+1.77, +1.99] | 0.071 | 1.00 | 0.00 / 0.18 |

## Appendix B · Altitude, per cell and rule

tilt: min z over x ∈ [0.5, 2.0] (the descent); hump: max z over x ∈ [−0.5, 0.5] (the climb). Unprojected band first.

| block | model | nfe | unprojected band (m) | projected variant | Δz per flight: min / median / max (m) |
| :-- | :-- | --: | :-- | :-- | :-- |
| tilt | MeanFM | 1 | 0.84–1.17 | dpcc-r | -0.25 / -0.13 / +0.09  (entered 10/10) |
| tilt | MeanFM | 1 | 0.84–1.17 | dpcc-c | -0.28 / -0.19 / +0.04  (entered 10/10) |
| tilt | MeanFM | 1 | 0.84–1.17 | dpcc-t | -0.23 / -0.14 / +0.06  (entered 10/10) |
| tilt | MeanFM | 2 | 0.87–1.17 | dpcc-r | -0.24 / -0.14 / +0.05  (entered 10/10) |
| tilt | MeanFM | 2 | 0.87–1.17 | dpcc-c | -0.28 / -0.19 / +0.01  (entered 10/10) |
| tilt | MeanFM | 2 | 0.87–1.17 | dpcc-t | -0.23 / -0.16 / +0.05  (entered 10/10) |
| tilt | MeanFM | 3 | 0.87–1.17 | dpcc-r | -0.33 / -0.26 / -0.05  (entered 10/10) |
| tilt | MeanFM | 3 | 0.87–1.17 | dpcc-c | -0.33 / -0.26 / -0.05  (entered 10/10) |
| tilt | MeanFM | 3 | 0.87–1.17 | dpcc-t | -0.33 / -0.26 / -0.05  (entered 10/10) |
| tilt | MeanFM | 3 | 0.87–1.17 | hf | -0.29 / -0.23 / -0.00  (entered 10/10) |
| tilt | MeanFM | 3 | 0.87–1.17 | hf-r | -0.29 / -0.22 / -0.01  (entered 10/10) |
| tilt | MeanFM | 3 | 0.87–1.17 | hf-c | -0.32 / -0.24 / -0.03  (entered 10/10) |
| tilt | MeanFM | 3 | 0.87–1.17 | hf-t | -0.29 / -0.22 / -0.02  (entered 10/10) |
| tilt | CI-MeanFM | 1 | 0.83–1.15 | dpcc-r | -0.23 / -0.12 / +0.11  (entered 10/10) |
| tilt | CI-MeanFM | 1 | 0.83–1.15 | dpcc-c | -0.26 / -0.16 / +0.04  (entered 10/10) |
| tilt | CI-MeanFM | 1 | 0.83–1.15 | dpcc-t | -0.22 / -0.12 / +0.05  (entered 10/10) |
| tilt | CI-MeanFM | 2 | 0.88–1.17 | dpcc-r | -0.24 / -0.13 / +0.04  (entered 10/10) |
| tilt | CI-MeanFM | 2 | 0.88–1.17 | dpcc-c | -0.28 / -0.19 / +0.00  (entered 10/10) |
| tilt | CI-MeanFM | 2 | 0.88–1.17 | dpcc-t | -0.24 / -0.16 / +0.04  (entered 10/10) |
| tilt | CI-MeanFM | 3 | 0.88–1.17 | dpcc-r | -0.34 / -0.26 / -0.05  (entered 10/10) |
| tilt | CI-MeanFM | 3 | 0.88–1.17 | dpcc-c | -0.34 / -0.26 / -0.06  (entered 10/10) |
| tilt | CI-MeanFM | 3 | 0.88–1.17 | dpcc-t | -0.33 / -0.26 / -0.05  (entered 10/10) |
| tilt | CI-MeanFM | 3 | 0.88–1.17 | hf | -0.30 / -0.22 / -0.02  (entered 10/10) |
| tilt | CI-MeanFM | 3 | 0.88–1.17 | hf-r | -0.29 / -0.22 / -0.02  (entered 10/10) |
| tilt | CI-MeanFM | 3 | 0.88–1.17 | hf-c | -0.32 / -0.24 / -0.04  (entered 10/10) |
| tilt | CI-MeanFM | 3 | 0.88–1.17 | hf-t | -0.29 / -0.21 / -0.02  (entered 10/10) |
| tilt | FM | 1 | 0.92–1.17 | dpcc-r | -0.27 / -0.19 / +0.00  (entered 10/10) |
| tilt | FM | 1 | 0.92–1.17 | dpcc-c | -0.29 / -0.21 / +0.00  (entered 10/10) |
| tilt | FM | 1 | 0.92–1.17 | dpcc-t | -0.31 / -0.20 / -0.02  (entered 10/10) |
| tilt | FM | 2 | 0.92–1.22 | dpcc-r | -0.32 / -0.22 / -0.01  (entered 10/10) |
| tilt | FM | 2 | 0.92–1.22 | dpcc-c | -0.34 / -0.25 / -0.01  (entered 10/10) |
| tilt | FM | 2 | 0.92–1.22 | dpcc-t | -0.35 / -0.24 / -0.02  (entered 10/10) |
| tilt | FM | 3 | 0.93–1.22 | dpcc-r | -0.41 / -0.32 / -0.05  (entered 10/10) |
| tilt | FM | 3 | 0.93–1.22 | dpcc-c | -0.41 / -0.32 / -0.05  (entered 10/10) |
| tilt | FM | 3 | 0.93–1.22 | dpcc-t | -0.41 / -0.32 / -0.05  (entered 10/10) |
| tilt | FM | 3 | 0.93–1.22 | hf | -0.37 / -0.28 / -0.04  (entered 10/10) |
| tilt | FM | 3 | 0.93–1.22 | hf-r | -0.37 / -0.28 / -0.03  (entered 10/10) |
| tilt | FM | 3 | 0.93–1.22 | hf-c | -0.38 / -0.29 / -0.04  (entered 10/10) |
| tilt | FM | 3 | 0.93–1.22 | hf-t | -0.38 / -0.30 / -0.04  (entered 10/10) |
| tilt | FM | 5 | 0.94–1.22 | dpcc-r | -0.41 / -0.33 / -0.07  (entered 10/10) |
| tilt | FM | 5 | 0.94–1.22 | dpcc-c | -0.41 / -0.33 / -0.07  (entered 10/10) |
| tilt | FM | 5 | 0.94–1.22 | dpcc-t | -0.41 / -0.33 / -0.07  (entered 10/10) |
| tilt | FM | 5 | 0.94–1.22 | hf | -0.40 / -0.32 / -0.07  (entered 10/10) |
| tilt | FM | 5 | 0.94–1.22 | hf-r | -0.40 / -0.32 / -0.06  (entered 10/10) |
| tilt | FM | 5 | 0.94–1.22 | hf-c | -0.40 / -0.32 / -0.07  (entered 10/10) |
| tilt | FM | 5 | 0.94–1.22 | hf-t | -0.41 / -0.32 / -0.07  (entered 10/10) |
| tilt | FM | 20 | 0.96–1.23 | dpcc-r | -0.42 / -0.35 / -0.12  (entered 10/10) |
| tilt | FM | 20 | 0.96–1.23 | dpcc-c | -0.42 / -0.35 / -0.12  (entered 10/10) |
| tilt | FM | 20 | 0.96–1.23 | dpcc-t | -0.42 / -0.35 / -0.12  (entered 10/10) |
| tilt | FM | 20 | 0.96–1.23 | hf | -0.45 / -0.37 / -0.16  (entered 10/10) |
| tilt | FM | 20 | 0.96–1.23 | hf-r | -0.45 / -0.37 / -0.16  (entered 10/10) |
| tilt | FM | 20 | 0.96–1.23 | hf-c | -0.45 / -0.37 / -0.17  (entered 10/10) |
| tilt | FM | 20 | 0.96–1.23 | hf-t | -0.45 / -0.37 / -0.16  (entered 10/10) |
| tilt | Diffusion ★ | 20 | 0.99–1.27 | dpcc-r | -0.55 / -0.25 / -0.04  (entered 10/10) |
| tilt | Diffusion ★ | 20 | 0.99–1.27 | dpcc-c | -0.32 / -0.22 / -0.04  (entered 10/10) |
| tilt | Diffusion ★ | 20 | 0.99–1.27 | dpcc-t | -0.55 / -0.30 / -0.08  (entered 10/10) |
| hump | MeanFM | 1 | 0.89–1.21 | dpcc-r | +0.24 / +0.33 / +0.55  (entered 10/10) |
| hump | MeanFM | 1 | 0.89–1.21 | dpcc-c | +0.28 / +0.38 / +0.60  (entered 10/10) |
| hump | MeanFM | 1 | 0.89–1.21 | dpcc-t | +0.27 / +0.38 / +0.60  (entered 10/10) |
| hump | MeanFM | 2 | 0.92–1.20 | dpcc-r | +0.25 / +0.33 / +0.53  (entered 10/10) |
| hump | MeanFM | 2 | 0.92–1.20 | dpcc-c | +0.28 / +0.38 / +0.58  (entered 10/10) |
| hump | MeanFM | 2 | 0.92–1.20 | dpcc-t | +0.27 / +0.39 / +0.58  (entered 10/10) |
| hump | MeanFM | 3 | 0.91–1.20 | dpcc-r | +0.26 / +0.35 / +0.55  (entered 10/10) |
| hump | MeanFM | 3 | 0.91–1.20 | dpcc-c | +0.26 / +0.35 / +0.56  (entered 10/10) |
| hump | MeanFM | 3 | 0.91–1.20 | dpcc-t | +0.25 / +0.34 / +0.55  (entered 10/10) |
| hump | MeanFM | 3 | 0.91–1.20 | hf | +0.28 / +0.38 / +0.57  (entered 10/10) |
| hump | MeanFM | 3 | 0.91–1.20 | hf-r | +0.28 / +0.37 / +0.58  (entered 10/10) |
| hump | MeanFM | 3 | 0.91–1.20 | hf-c | +0.28 / +0.35 / +0.56  (entered 10/10) |
| hump | MeanFM | 3 | 0.91–1.20 | hf-t | +0.27 / +0.35 / +0.56  (entered 10/10) |
| hump | CI-MeanFM | 1 | 0.90–1.20 | dpcc-r | +0.25 / +0.33 / +0.54  (entered 10/10) |
| hump | CI-MeanFM | 1 | 0.90–1.20 | dpcc-c | +0.29 / +0.39 / +0.59  (entered 10/10) |
| hump | CI-MeanFM | 1 | 0.90–1.20 | dpcc-t | +0.28 / +0.38 / +0.58  (entered 10/10) |
| hump | CI-MeanFM | 2 | 0.92–1.21 | dpcc-r | +0.25 / +0.33 / +0.52  (entered 10/10) |
| hump | CI-MeanFM | 2 | 0.92–1.21 | dpcc-c | +0.28 / +0.38 / +0.58  (entered 10/10) |
| hump | CI-MeanFM | 2 | 0.92–1.21 | dpcc-t | +0.27 / +0.38 / +0.57  (entered 10/10) |
| hump | CI-MeanFM | 3 | 0.92–1.21 | dpcc-r | +0.25 / +0.36 / +0.56  (entered 10/10) |
| hump | CI-MeanFM | 3 | 0.92–1.21 | dpcc-c | +0.26 / +0.35 / +0.55  (entered 10/10) |
| hump | CI-MeanFM | 3 | 0.92–1.21 | dpcc-t | +0.25 / +0.34 / +0.54  (entered 10/10) |
| hump | CI-MeanFM | 3 | 0.92–1.21 | hf | +0.27 / +0.37 / +0.57  (entered 10/10) |
| hump | CI-MeanFM | 3 | 0.92–1.21 | hf-r | +0.27 / +0.37 / +0.57  (entered 10/10) |
| hump | CI-MeanFM | 3 | 0.92–1.21 | hf-c | +0.27 / +0.35 / +0.56  (entered 10/10) |
| hump | CI-MeanFM | 3 | 0.92–1.21 | hf-t | +0.26 / +0.35 / +0.56  (entered 10/10) |
| hump | FM | 1 | 0.92–1.18 | dpcc-r | +0.22 / +0.32 / +0.47  (entered 10/10) |
| hump | FM | 1 | 0.92–1.18 | dpcc-c | +0.24 / +0.34 / +0.50  (entered 10/10) |
| hump | FM | 1 | 0.92–1.18 | dpcc-t | +0.24 / +0.35 / +0.50  (entered 10/10) |
| hump | FM | 2 | 0.93–1.22 | dpcc-r | +0.18 / +0.27 / +0.46  (entered 10/10) |
| hump | FM | 2 | 0.93–1.22 | dpcc-c | +0.20 / +0.30 / +0.49  (entered 10/10) |
| hump | FM | 2 | 0.93–1.22 | dpcc-t | +0.20 / +0.30 / +0.49  (entered 10/10) |
| hump | FM | 3 | 0.94–1.22 | dpcc-r | +0.24 / +0.34 / +0.53  (entered 10/10) |
| hump | FM | 3 | 0.94–1.22 | dpcc-c | +0.24 / +0.33 / +0.53  (entered 10/10) |
| hump | FM | 3 | 0.94–1.22 | dpcc-t | +0.24 / +0.33 / +0.52  (entered 10/10) |
| hump | FM | 3 | 0.94–1.22 | hf | +0.27 / +0.36 / +0.55  (entered 10/10) |
| hump | FM | 3 | 0.94–1.22 | hf-r | +0.26 / +0.36 / +0.56  (entered 10/10) |
| hump | FM | 3 | 0.94–1.22 | hf-c | +0.26 / +0.35 / +0.55  (entered 10/10) |
| hump | FM | 3 | 0.94–1.22 | hf-t | +0.25 / +0.35 / +0.55  (entered 10/10) |
| hump | FM | 5 | 0.94–1.23 | dpcc-r | +0.24 / +0.33 / +0.52  (entered 10/10) |
| hump | FM | 5 | 0.94–1.23 | dpcc-c | +0.24 / +0.33 / +0.53  (entered 10/10) |
| hump | FM | 5 | 0.94–1.23 | dpcc-t | +0.24 / +0.33 / +0.52  (entered 10/10) |
| hump | FM | 5 | 0.94–1.23 | hf | +0.25 / +0.34 / +0.53  (entered 10/10) |
| hump | FM | 5 | 0.94–1.23 | hf-r | +0.25 / +0.34 / +0.53  (entered 10/10) |
| hump | FM | 5 | 0.94–1.23 | hf-c | +0.25 / +0.34 / +0.53  (entered 10/10) |
| hump | FM | 5 | 0.94–1.23 | hf-t | +0.25 / +0.33 / +0.52  (entered 10/10) |
| hump | FM | 20 | 0.96–1.23 | dpcc-r | +0.24 / +0.33 / +0.52  (entered 10/10) |
| hump | FM | 20 | 0.96–1.23 | dpcc-c | +0.24 / +0.33 / +0.52  (entered 10/10) |
| hump | FM | 20 | 0.96–1.23 | dpcc-t | +0.24 / +0.33 / +0.52  (entered 10/10) |
| hump | FM | 20 | 0.96–1.23 | hf | +0.25 / +0.34 / +0.53  (entered 10/10) |
| hump | FM | 20 | 0.96–1.23 | hf-r | +0.25 / +0.34 / +0.53  (entered 10/10) |
| hump | FM | 20 | 0.96–1.23 | hf-c | +0.25 / +0.34 / +0.53  (entered 10/10) |
| hump | FM | 20 | 0.96–1.23 | hf-t | +0.25 / +0.33 / +0.53  (entered 10/10) |
| hump | Diffusion ★ | 20 | 0.99–1.27 | dpcc-r | +0.24 / +0.34 / +0.52  (entered 10/10) |
| hump | Diffusion ★ | 20 | 0.99–1.27 | dpcc-c | +0.24 / +0.33 / +0.55  (entered 10/10) |
| hump | Diffusion ★ | 20 | 0.99–1.27 | dpcc-t | +0.22 / +0.32 / +0.52  (entered 10/10) |

## Appendix C · Flights that do not cross the finish line

| block | model | nfe | variant | not crossed | final x (median) | final altitude (median, m) | ended on the step cap |
| :-- | :-- | --: | :-- | --: | --: | --: | --: |
| hump | CI-MeanFM | 1 | dpcc-c | 10/10 | -0.02 | 1.49 | 10/10 |
| hump | CI-MeanFM | 1 | dpcc-r | 10/10 | -0.09 | 1.44 | 10/10 |
| hump | CI-MeanFM | 1 | dpcc-t | 10/10 | -0.04 | 1.48 | 10/10 |
| hump | CI-MeanFM | 2 | dpcc-c | 10/10 | -0.02 | 1.49 | 10/10 |
| hump | CI-MeanFM | 2 | dpcc-r | 10/10 | -0.09 | 1.45 | 10/10 |
| hump | CI-MeanFM | 2 | dpcc-t | 10/10 | -0.03 | 1.48 | 10/10 |
| hump | Diffusion ★ | 20 | dpcc-c | 7/10 | +2.41 | 1.25 | 7/7 |
| hump | Diffusion ★ | 20 | dpcc-r | 7/10 | +2.47 | 1.24 | 7/7 |
| hump | Diffusion ★ | 20 | dpcc-t | 3/10 | +2.64 | 1.33 | 3/3 |
| hump | FM | 1 | dpcc-c | 10/10 | -0.13 | 1.42 | 10/10 |
| hump | FM | 1 | dpcc-r | 10/10 | -0.16 | 1.39 | 10/10 |
| hump | FM | 1 | dpcc-t | 10/10 | -0.12 | 1.42 | 10/10 |
| hump | FM | 2 | dpcc-c | 10/10 | -0.13 | 1.42 | 10/10 |
| hump | FM | 2 | dpcc-r | 10/10 | -0.17 | 1.39 | 10/10 |
| hump | FM | 2 | dpcc-t | 10/10 | -0.13 | 1.42 | 10/10 |
| hump | MeanFM | 1 | dpcc-c | 10/10 | -0.02 | 1.50 | 10/10 |
| hump | MeanFM | 1 | dpcc-r | 10/10 | -0.09 | 1.44 | 10/10 |
| hump | MeanFM | 1 | dpcc-t | 10/10 | -0.04 | 1.48 | 10/10 |
| hump | MeanFM | 2 | dpcc-c | 10/10 | -0.03 | 1.49 | 10/10 |
| hump | MeanFM | 2 | dpcc-r | 10/10 | -0.09 | 1.45 | 10/10 |
| hump | MeanFM | 2 | dpcc-t | 10/10 | -0.03 | 1.49 | 10/10 |
| tilt | Diffusion ★ | 20 | dpcc-c | 10/10 | +2.49 | 1.12 | 10/10 |
| tilt | Diffusion ★ | 20 | dpcc-r | 8/10 | +2.55 | 1.14 | 8/8 |
| tilt | Diffusion ★ | 20 | dpcc-t | 7/10 | +2.60 | 1.16 | 7/7 |

---
Claude (Opus 5.5, Claude Code, U19 corridor chat) · 2026-09-24 · computed locally with python3.14 (numpy, yaml); nothing run on
the cluster; the v3 draft was not edited.

---

## Addendum · v3.80 writing pass (2026-09-24, v3 agent) — checks made while transcribing into Chapter 6

Nothing above was changed. Every cell of §3.1–§3.3 was recomputed from `per_rollout_detail.csv` (first ten flights)
and is **equal**. Five wordings were narrowed in the thesis after checking them against the flights:

| DA wording | measured | thesis wording |
| :-- | :-- | :-- |
| §1, §4.1 "every violating step … lies at x ∈ [1.75, 2.00]" | that is the 5–95 % range; all **4 115** violating steps of the 560 projected tilt flights lie in **[1.709, 2.000]**, 99 % at x ≥ 1.75 | "the last 0.3 m of the plane's span, x 1.71–2.00" |
| §1, §3.7 flow models "within 0.9 of each other at every matched budget and projector" | holds for **rule averages** except hump K2 per-step (6.0 / 6.6 / 0.0, the six wall-drift flights); at each model's **best rule** the spread is at most **1.6** (tilt per-step K3: 6.6 / 5.0 / 5.8) | "within two violating steps, each at its best rule" |
| §3.3 "within a projector the rules differ by at most two violating steps" | hump endpoint CI-MeanFM K3 8.0–10.4 (2.4); baseline tilt per-step 0.8–6.1 | not used |
| §3.6 "every flight of every cell moves altitude the way its constraint asks" | tilt **540 / 560** descend (not: MeanFM 8, CI-MeanFM 10, FM 2, each within +0.11 m); hump 560 / 560 climb | the counts |
| §4.2 "the apex needs 1.485 m" | 1.10 + 0.31·√(1 + (1.1/1.5)²) = **1.4844** m | 1.48 m (Ch 5 printed 1.49 by double rounding; fixed at v3.80) |

Also verified from the flights: the hand-over (530 projected flow flights; commanded position at the first violating step
x 2.27–2.52, 0.49–0.61 m ahead; clear of the constraint at 4 014 / 4 014 violating steps; the span check reads
`p_des[0]` under `-pdes`, `eval_mix_uav.py:1830–1833`, and gates the whole plan, `:1483–1484`); the hump stops per flight
(x −0.174 … −0.006, z 1.38–1.51 m, 174 of 180 flights, all on the 396-step cap; the six others have `phys_contact_frac`
0 — wall **violations**, no collision); the baseline's unsuccessful projected flights (none crossed the line, all safe,
all on the cap). For the thesis the diagnostic figure is redrawn in the store as `fig_uav_corridor_side`
(`plotting/extract/corridor_v3_side.py` → `builders/altitude.py`); the figure above stays as it is.
