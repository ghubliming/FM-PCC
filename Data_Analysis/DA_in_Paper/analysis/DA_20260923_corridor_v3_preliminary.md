> **Superseded 2026-09-24 by [`DA_20260924_corridor_v3.md`](DA_20260924_corridor_v3.md)** (complete tables from the full-corpus batch). Kept for the record.

# DA 2026-09-23 · UAV-corridor v3 (R33) — PRELIMINARY: every Chapter 6 corridor item, from the runs that landed

**Status: preliminary.** Built on the 23-09 download `temp/23-09-Corridor-TEMP/plans` (93 of 136 cells readable). Every
number below is computed from the raw result folders (npz + `results.json`) by
[`corridor_v3_grid.py`](corridor_v3_grid.py); nothing is taken from a console log. Cells that are not readable are printed
*fetch* (finished on the cluster, not in this download) or *run* (still running). Re-run the script on the complete corpus
and this document's tables update 1:1. Chapter 6 (v3 draft) is **not** edited; the writing agent transcribes from §3.

Scene definitions: `corridor_v3_tilt` = the v2 slide leaned −60° about z_ref 1.11 m (tag `p23cv3t`); `corridor_v3_ablation_hump`
= the x–z roof, H 1.10 m (tag `p23cv3ah`). Protocol: seed 6, **the first ten flights** of each cell (routes L, C, R cycling,
4/3/3); C1–C4 were flown with 12, C5 with 10, and trial *i* is seeded by its index, so the ten are the same flights in both.

---

## 0 · What this download lacks — to fetch before the final DA

| what | why | size on the cluster |
| :-- | :-- | :-- |
| `logs/UAV_MIX/uav-corridor/plans/mix_uav_af/H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet/Eaf_K{1,2,3}_mpc4_pid_stopgo_T0.5_EPlatest_p23cv3{t,ah}` (6 folders) | the download's copies are **0-byte files** (npz, json and logs alike, all stamped 21:17 — an interrupted copy); 13 tilt + 10 hump CI-MeanFM cells unreadable | 32–62 MiB each |
| `…/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/Emf_K3_mpc4_pid_stopgo_T0.5_p23cv3{t,ah}` (2 folders) | variant folders present but **empty** here; the jobs finished at 13:37 / 15:23 | 59 / 62 MiB |
| `…/mix_uav_diffusion/H8_Dmodels.ddpm_diffusion.GaussianDiffusion_9D_K20/Ediffusion_K20_mpc4_pid_stopgo_T0.5_p23cv3{t,ah}` | tilt `dpcc-c` (job 26163) and hump `dpcc-t` (26164) finished after the copy; hump `dpcc-r` / `dpcc-c` (26165 / 26166) **still running** at 23:21 | 37 / 18 MiB, growing |
| optional: `Data_Analysis/analysis_results/batch_uav_20260923_231532/` | the corridor DA_UAV_v1 batch of 23:15 — the folder attached as "DA-CSV" (`batch_va2_20260923_210100`) is the **visual-aligning** batch and holds no corridor row | ~2 MB |

Diagnostic files (`diagnostics/`, per-rollout SVG/stats) are not needed: timing comes from `results.json`.

| geometry | readable | readable, no timing (CI-MeanFM, json empty) | landed, not in download | running |
| :-- | --: | --: | --: | --: |
| tilt | 44 | 2 | 22 | 0 |
| hump | 47 | 0 | 19 | 2 |

## 1 · Verdict

**No code bug and no wrong setup — the DA was not aborted.** The runs are what they claim to be (§2). The data carry two
**structural results** that shape what Chapter 6 can say about this scene; neither is a malfunction, both are properties of
the evaluation stack the author chose to run as is (22-09, "if it is the model / projector, no need to fix the env").

1. **Tilt (the corridor v3): S&C is 0/10 in every cell, projected or not.** Every projected flow flight crosses the finish
   line (10/10 in every readable cell) and descends under the leaned plane (median −0.13 to −0.37 m, §3.6), but none is
   violation-free: a short residue of 1.4–11 violating steps per flight, **all in x ∈ [1.75, 2.00]**, the last quarter-metre
   before the plane's window ends. At the first violating step the commanded setpoint is already at x = 2.28–2.48 — past
   the window end, so the plane no longer binds the plan — and 0.52–0.57 m ahead of the vehicle; the setpoint is on the
   feasible side at **100 %** of the violating steps. Depth falls with the budget: 9–12 cm at K1/K2, 5–8 cm at K3, 3–5 cm at
   K5, 1–2 cm at K20. This is the pilot's hand-over residue at full scale: the window gate reads the setpoint's x
   (U16 `-pdes` fix), the vehicle lags the setpoint, and at the window end the plan returns toward the route while the
   vehicle is still inside the window. **Consequence:** the tilt's S&C column is all zeros; no tilt configuration is eligible
   for the frontier; the ordering on the tilt can only be read on violating steps, depth and descent.
2. **Hump (the ablation): per-step projection at K1/K2 stalls on the roof; from K3 it crosses with an apex residue; only
   FM K20 endpoint projection is violation-free (4–8/10).** All 16 readable K1/K2 per-step cells cross 0/10: the vehicle
   climbs +0.27–0.39 m (median) onto the drone-centre limit and stops 0.02–0.17 m before the apex, on the step cap, 10/10
   flights. Three causes act together (§4.2): the plan's altitude is clipped to the demonstrated band (the normaliser's
   `unnormalize` clips to [−1, 1]; plans sit on the 1.299 m top **55–61 %** of the time at K1/K2) while the apex needs
   1.485 m (1.515 tightened); the minimum-norm correction is taken in normalised coordinates, where a metre of climb costs
   about 200 times a metre of retreat along x; and the roof's rising row is applied to the whole horizon while the setpoint's x < 0, i.e. to
   horizon points past the apex. From K3 the flights cross 10/10 with 4–10 violating steps at the apex (depth ≤ 2.8 cm,
   setpoint clean 100 %). At FM K20 per-step leaves 1.8–2.2 steps (≤ 1.1 cm); endpoint projection leaves 0.2–0.7 and is the
   only S&C > 0 on the scene: single 8/10, r 6/10, c 6/10, t 4/10.

What the author may want to decide before the writing agent runs (none of this is applied): see §4.3.

## 2 · Checks (all pass)

| check | result |
| :-- | :-- |
| geometry each run used (`config_snapshot_uav_mix/uav_projection.yaml`, 24 snapshots) vs repo `config/uav_projection.yaml` — halfspaces incl. `z_lean` / `plane: xz`, caps, box, inflation, tightening, threshold | **0 differences** |
| code version: the 62 jobs ran on 5 git revisions (`1e8e707d` ×39, `0df6df71` ×8, `6a8cc26f` ×6, `c4f91be8` ×6, `5092e056` ×3) | `git diff` of `mix_uav_test/`, `mix_uav/`, `config/uav_projection.yaml`, `config/uav_mix.py`, the UAV sbatch and the quadrotor scenes between them: **empty** — every wave ran the same code; all contain `_hs_lean` |
| job logs (62): geometry banner `hs=3` (tilt) / `hs=4` (hump), `scene_corridor_v2.xml`, requested variants, n_trials 12 (C1–C4) / 10 (C5), no traceback | **60 complete, 2 running** (26165, 26166); HardFlow "thin" notice on the K3 jobs only (expected, one guiding step) |
| violating steps re-scored from each flown path with the eval's own scorer (`_exec_constraint_violations`, extracted) | **930 flights, 0 disagree** with the stored counts |
| divergence aborts / projection circuit-breaker trips | **0 / 0** over all readable flights |
| setpoint (commanded p_des) at every violating step of every projected flow flight | feasible at **100 %** (tilt and hump); unprojected 25–33 % (tilt), 68–82 % (hump) |
| plan altitude box (stored plans, all cells) | every plan z ∈ [0.9008, 1.2992] m = the demonstrated band; plan y ∈ [−0.12, 0.12] m. The executed setpoint leaves the box through the action increments (FM K3: tilt 0.71–1.24 m, y to −0.29; hump to 1.52 m). The clip is `LimitsNormalizer.unnormalize` (`mix_uav/datasets/normalization.py:209–228`), present since Gen15 init (2026-08-10) and inherited from Diffuser/DPCC — it was active in the v2 corridor results too |

Minor, not affecting any number: the corridor divergence guard still uses the v1 corridor envelope (y ±0.45, z 0.70–1.30)
with 2 m slack — nothing came near it.

## 3 · The Chapter 6 corridor items (v3 `06_results.tex`, `sec:res:uav:corridor` … `sec:res:uav:projection`)

Notation: counts of 10 flights; mean ± sample standard deviation over the ten; *fetch* / *run* as in §0. Model names as in
the chapter (MeanFM, CI-MeanFM α_end 0.2, FM, Diffusion ★).

### 3.1 `tab:uav-corridor-raw` — before projection

| block | model | nfe | success | violation-free | S&C | violating steps | ms/step |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| tilt | MeanFM | 1 | 10/10 | 0/10 | 0/10 | 91.3 ± 34.8 | 9.4 ± 0.4 |
| tilt | MeanFM | 2 | 10/10 | 0/10 | 0/10 | 92.9 ± 30.2 | 18.1 ± 0.4 |
| tilt | MeanFM | 3 | *fetch* | | | | |
| tilt | CI-MeanFM | 1 | 10/10 | 0/10 | 0/10 | 90.0 ± 32.6 | 9.3 ± 0.4 |
| tilt | CI-MeanFM | 2 | *fetch* | | | | |
| tilt | CI-MeanFM | 3 | *fetch* | | | | |
| tilt | FM | 1 | 10/10 | 0/10 | 0/10 | 74.5 ± 25.6 | 9.1 ± 0.4 |
| tilt | FM | 2 | 10/10 | 0/10 | 0/10 | 83.9 ± 28.4 | 17.7 ± 0.4 |
| tilt | FM | 3 | 10/10 | 0/10 | 0/10 | 87.0 ± 28.3 | 25.9 ± 0.5 |
| tilt | FM | 5 | 10/10 | 0/10 | 0/10 | 89.9 ± 28.4 | 43.4 ± 0.4 |
| tilt | FM | 20 | 10/10 | 0/10 | 0/10 | 94.7 ± 27.0 | 170.8 ± 0.4 |
| tilt | Diffusion ★ | 20 | 10/10 | 0/10 | 0/10 | 99.4 ± 27.0 | 178.1 ± 0.8 |
| hump | MeanFM | 1 | 10/10 | 0/10 | 0/10 | 36.0 ± 8.9 | 9.4 ± 0.4 |
| hump | MeanFM | 2 | 10/10 | 0/10 | 0/10 | 35.6 ± 8.1 | 18.2 ± 0.4 |
| hump | MeanFM | 3 | *fetch* | | | | |
| hump | CI-MeanFM | 1 | 10/10 | 0/10 | 0/10 | 36.6 ± 8.4 | 9.3 ± 0.4 |
| hump | CI-MeanFM | 2 | *fetch* | | | | |
| hump | CI-MeanFM | 3 | 10/10 | 0/10 | 0/10 | 35.6 ± 7.6 | 27.1 ± 0.4 |
| hump | FM | 1 | 10/10 | 0/10 | 0/10 | 34.7 ± 6.8 | 9.1 ± 0.4 |
| hump | FM | 2 | 10/10 | 0/10 | 0/10 | 32.6 ± 7.8 | 17.5 ± 0.3 |
| hump | FM | 3 | 10/10 | 0/10 | 0/10 | 32.6 ± 8.0 | 26.3 ± 0.4 |
| hump | FM | 5 | 10/10 | 0/10 | 0/10 | 32.0 ± 7.9 | 43.6 ± 0.4 |
| hump | FM | 20 | 10/10 | 0/10 | 0/10 | 31.3 ± 7.6 | 172.3 ± 2.6 |
| hump | Diffusion ★ | 20 | 10/10 | 0/10 | 0/10 | 28.6 ± 7.7 | 178.2 ± 2.7 |

**Reading (for the hole "Result sentence, before projection").** Every model at every budget crosses the finish line on
every flight, and no flight is violation-free on either constraint: the tilt is crossed on 74–99 control steps per flight,
the hump on 29–37. The unprojected paths cross the tilt from x ≈ −1.2 at the earliest to the window end and the hump
over its whole apex region (x ∈ [−0.5, 0.5]); the deepest step per cell is 0.37–0.48 m. So the scene is a repair case on both constraints, as
the chapter anticipated. Diffusion has the most violating steps on the tilt (99.4) and the fewest on the hump (28.6); the flow
models sit between 74.5 and 94.7 (tilt) and 31.3 and 36.6 (hump) — differences inside one standard deviation. Because
every model reaches the goal unprojected, no before-projection frontier is drawn (the chapter's own rule).

### 3.2 `tab:uav-corridor` — after projection, best rule per projector

Best rule = most S&C; ties → fewer mean violating steps → lower ms/step (ties are the rule here: S&C is 0 almost everywhere,
so the rule printed is the one with the fewest violating steps).

| block | model | nfe | per-step S&C | rule | ms/step | endpoint S&C | rule | ms/step |
| :-- | :-- | --: | --: | :-- | --: | --: | :-- | --: |
| tilt | MeanFM | 1 | 0/10 | c | 27.4 | — | — | — |
| tilt | MeanFM | 2 | 0/10 | c | 36.4 | — | — | — |
| tilt | MeanFM | 3 | *fetch* | | | *fetch* | | |
| tilt | CI-MeanFM | 1 | 0/10 | c (r, t *fetch*) | *fetch* | — | — | — |
| tilt | CI-MeanFM | 2 | 0/10 | c (r, t *fetch*) | *fetch* | — | — | — |
| tilt | CI-MeanFM | 3 | *fetch* | | | *fetch* | | |
| tilt | FM | 1 | 0/10 | c | 27.6 | — | — | — |
| tilt | FM | 2 | 0/10 | c | 34.7 | — | — | — |
| tilt | FM | 3 | 0/10 | c | 79.2 | 0/10 | c | 71.6 |
| tilt | FM | 5 | 0/10 | c | 116.9 | 0/10 | c | 115.2 |
| tilt | FM | 20 | 0/10 | c | 360.4 | 0/10 | c | 416.9 |
| tilt | Diffusion ★ | 20 | 0/10 | r (c *fetch*) | 679.7 | — | — | — |
| hump | MeanFM | 1 | 0/10 | c | 26.4 | — | — | — |
| hump | MeanFM | 2 | 0/10 | r | 35.0 | — | — | — |
| hump | MeanFM | 3 | *fetch* | | | *fetch* | | |
| hump | CI-MeanFM | 1 | 0/10 | t | 26.6 | — | — | — |
| hump | CI-MeanFM | 2 | 0/10 | c (r, t *fetch*) | 35.6 | — | — | — |
| hump | CI-MeanFM | 3 | *fetch* | | | *fetch* | | |
| hump | FM | 1 | 0/10 | c | 26.2 | — | — | — |
| hump | FM | 2 | 0/10 | r | 34.5 | — | — | — |
| hump | FM | 3 | 0/10 | c | 77.1 | 0/10 | c | 71.9 |
| hump | FM | 5 | 0/10 | c | 114.1 | 0/10 | c | 112.4 |
| hump | FM | 20 | 0/10 | c | 355.2 | **8/10** | single | 291.0 |
| hump | Diffusion ★ | 20 | *run* (t *fetch*) | | | — | — | — |

**Reading (for the hole "Result sentence, after projection").** On the tilt no configuration succeeds violation-free, at
any budget, under either projector: every flow flight crosses the line and every one leaves a residue at the window end
(§1.1). On the hump the only violation-free successes are FM at K20 under endpoint projection (single 8/10, 291 ms/step);
per-step projection at K1/K2 stops every flight on the roof (0/10 success; 10/10 violation-free except MeanFM K2 t, 7/10), and at K3–K20 it crosses
every flight with an apex residue. The diffusion baseline's projected tilt flights mostly run out of steps before the
line (2/10 and 3/10 cross; §3.5).

The budget paragraph's facts (hole "Budget paragraph"): the residue shrinks monotonically with the budget on both
constraints — tilt, per-step, mean violating steps 9.8–11.1 (K1/K2) → 5.8–6.7 (K3) → 4.4–5.4 (K5) → 2.1–3.2 (FM K20; the baseline 3.2–6.1);
endpoint 6.9–8.2 (K3) → 4.9–6.3 (K5) → 1.4–2.5 (K20); hump per-step 5.6–5.9 (K3) → 3.9–4.3 (K5) → 1.8–2.2 (K20), endpoint
8.3–10.1 → 3.9–5.6 → 0.2–0.7. No budget reaches zero on the tilt; the hump reaches it only with endpoint projection at K20
(4–8 of 10 flights). The price rises with the budget: per-step 26–29 ms (K1) → 35–42 (K2) → 77–79 (K3) → 112–118 (K5) →
354–360 (K20); endpoint 44–72 (K3) → 73–115 (K5) → 289–425 (K20); the baseline's per-step 651–680 ms.

### 3.3 `tab:uav-corridor-projection` — S&C under every rule (K ≥ 3; K1/K2 per-step added for completeness)

| block | model | nfe | per-step r | c | t | endpoint single | r | c | t |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| tilt | MeanFM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | MeanFM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | MeanFM | 3 | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* |
| tilt | CI-MeanFM | 1 | *fetch* | 0/10 | *fetch* | — | — | — | — |
| tilt | CI-MeanFM | 2 | *fetch* | 0/10 | *fetch* | — | — | — | — |
| tilt | CI-MeanFM | 3 | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* |
| tilt | FM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | FM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| tilt | FM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | FM | 5 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | FM | 20 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| tilt | Diffusion ★ | 20 | 0/10 | *fetch* | 0/10 | — | — | — | — |
| hump | MeanFM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | MeanFM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | MeanFM | 3 | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* |
| hump | CI-MeanFM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | CI-MeanFM | 2 | *fetch* | 0/10 | *fetch* | — | — | — | — |
| hump | CI-MeanFM | 3 | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* | *fetch* |
| hump | FM | 1 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | FM | 2 | 0/10 | 0/10 | 0/10 | — | — | — | — |
| hump | FM | 3 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hump | FM | 5 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hump | FM | 20 | 0/10 | 0/10 | 0/10 | **8/10** | 6/10 | 6/10 | 4/10 |
| hump | Diffusion ★ | 20 | *run* | *run* | *fetch* | — | — | — | — |

**Reading (hole "Corridor result sentence, per-step against endpoint").** Where the two can be compared (FM K3, K5, K20),
endpoint projection matches per-step projection on S&C at K3 and K5 (0/10 both) at a lower or equal price (single 44 vs
77–79 ms at K3; 73 vs 112–118 ms at K5) with somewhat more residue at K3 (6.9–10.1 vs 5.6–6.7 steps); at K20 it is the only
projector with violation-free flights on the hump (4–8/10) and it leaves fewer violating steps on the tilt (1.4–2.5 vs
2.1–3.2), the single-candidate variant at 289–291 ms against per-step's 354–360. Rule differences within one projector are
small (≤ 2 violating steps; S&C differs only on the hump at K20, single > r = c > t).

### 3.4 `fig:uav-corridor-tradeoff` — the frontier

| block | eligible points (S&C > 0) | frontier |
| :-- | --: | :-- |
| tilt | 0 of 13 readable | **none** — no configuration succeeds violation-free |
| hump | 1 of 12 readable | FM K20, endpoint, single — 8/10 at 291.0 ms/step |

With the chapter's own eligibility rule the tilt panel is empty and the hump panel has one point. A figure over these would
be a ranking of zeros; see §4.3 for the options (the author's call — nothing drawn yet).

### 3.5 `fig:uav-corridor-paths` — facts for the paths paragraph

- **Unprojected**, both constraints: every flight follows its route and crosses the finish line; on the tilt the body
  enters the leaned plane from x ≈ −1.2 at the earliest to the window end, on the hump over the apex (x ∈ [−0.5, 0.5]).
- **Per-step, tilt**: every flow flight moves sideways and down under the plane and holds ≈ 5 cm clearance until x ≈ 1.75,
  then clips the plane in the last 0.25 m of the window while its setpoint is already past the window end (diagnostic
  figure (a), [`DA_20260923_corridor_v3_preliminary/fig_diag_tilt_residue_hump_stall.png`](DA_20260923_corridor_v3_preliminary/fig_diag_tilt_residue_hump_stall.png)).
  Crosses the line 10/10 at every budget.
- **Per-step, hump**: at K1/K2 the flight climbs onto the roof's limit and stops 0.02–0.17 m before the apex (final altitude
  1.39–1.50 m, step cap 10/10 — figure (b)); from K3 it rides over the apex (max altitude 1.46 m) and returns toward the
  route altitude, 10/10 across the line.
- **Endpoint, K ≥ 3**: the same shapes as per-step at the same budget, fewer violating steps at K20.
- **Diffusion, projected, tilt**: descends and moves sideways like the flow models but slower (377–378 steps): 7–8 of 10
  flights run out of steps 0.2 m short of the finish line (final x ≈ 2.55–2.60) — the v2 corridor's "projected diffusion
  stalls short of the line" again. Violation-free 3/10.
- The budget at which a model stops violating the constraint is **none** on the tilt; on the hump it is K20 with endpoint
  projection only. The budget at which it stops reaching the end is K1/K2 under per-step projection on the hump (all
  three flow models) and never on the tilt.

### 3.6 `fig:uav-corridor-altitude` — the altitude numbers

Extreme altitude in the constraint's window, per flight: tilt = minimum z over x ∈ [0.5, 2.0]; hump = maximum z over
x ∈ [−0.5, 0.5]. Δz = projected − unprojected, **same trial index** (same route and seed).

| block | model | nfe | unprojected band (m) | per-step Δz min / median / max (m) | endpoint Δz min / median / max (m) |
| :-- | :-- | --: | :-- | :-- | :-- |
| tilt | MeanFM | 1 | 0.84–1.17 | r −0.25 / −0.13 / +0.09 · c −0.28 / −0.19 / +0.04 · t −0.23 / −0.14 / +0.06 | — |
| tilt | MeanFM | 2 | 0.87–1.17 | r −0.24 / −0.14 / +0.05 · c −0.28 / −0.19 / +0.01 · t −0.23 / −0.16 / +0.05 | — |
| tilt | CI-MeanFM | 1 | 0.83–1.15 | c −0.26 / −0.16 / +0.04 (r, t *fetch*) | — |
| tilt | FM | 1 | 0.92–1.17 | −0.31 … −0.27 / −0.19 … −0.21 / −0.02 … +0.00 | — |
| tilt | FM | 2 | 0.92–1.22 | −0.35 … −0.32 / −0.22 … −0.25 / −0.02 … −0.01 | — |
| tilt | FM | 3 | 0.93–1.22 | −0.41 / −0.32 / −0.05 (all three rules) | −0.38 … −0.37 / −0.28 … −0.30 / −0.04 … −0.03 |
| tilt | FM | 5 | 0.94–1.22 | −0.41 / −0.33 / −0.07 | −0.41 … −0.40 / −0.32 / −0.07 … −0.06 |
| tilt | FM | 20 | 0.96–1.23 | −0.42 / −0.35 / −0.12 | −0.45 / −0.37 / −0.17 … −0.16 |
| tilt | Diffusion ★ | 20 | 0.99–1.27 | r −0.55 / −0.25 / −0.04 · t −0.55 / −0.30 / −0.08 | — |
| hump | MeanFM | 1 | 0.89–1.21 | +0.24 … +0.28 / +0.33 … +0.38 / +0.55 … +0.60 | — |
| hump | MeanFM | 2 | 0.92–1.20 | +0.25 … +0.28 / +0.33 … +0.39 / +0.53 … +0.58 | — |
| hump | CI-MeanFM | 1 | 0.90–1.20 | +0.25 … +0.29 / +0.33 … +0.39 / +0.54 … +0.59 | — |
| hump | FM | 1 | 0.92–1.18 | +0.22 … +0.24 / +0.32 … +0.35 / +0.47 … +0.50 | — |
| hump | FM | 2 | 0.93–1.22 | +0.18 … +0.20 / +0.27 … +0.30 / +0.46 … +0.49 | — |
| hump | FM | 3 | 0.94–1.22 | +0.24 / +0.33 … +0.34 / +0.52 … +0.53 | +0.25 … +0.27 / +0.35 … +0.36 / +0.55 … +0.56 |
| hump | FM | 5 | 0.94–1.23 | +0.24 / +0.33 / +0.52 … +0.53 | +0.25 / +0.33 … +0.34 / +0.52 … +0.53 |
| hump | FM | 20 | 0.96–1.23 | +0.24 / +0.33 / +0.52 | +0.25 / +0.33 … +0.34 / +0.53 |

**Reading (hole "Altitude numbers per constraint").** Unprojected, every model flies level in a 0.83–1.27 m band on both
scenes. Projected, every flight of every readable cell moves altitude in the direction its constraint asks: under the tilt
it descends by a median 0.13–0.37 m (more with the budget; the baseline 0.25–0.30 m), over the hump it climbs by a median
0.27–0.39 m (up to 0.60 m), at every budget — the projector moves z on demand, where on the level corridor of the earlier
evaluation it moved z only as a by-product of the lateral correction. The pilot's values (−0.15 to −0.29, +0.30 to +0.40) sit
inside these ranges.

### 3.7 The remaining holes and the §6.4 summary rows

- **Conclusion line `sec:res:uav:conclusion`, UAV-corridor** (facts to phrase): tilt — every flow model at every budget
  flies the descent and crosses the line, none violation-free (a hand-over residue of 1.4–11 steps at the window end, the
  commanded setpoint clean throughout); the baseline crosses 2–3/10. Hump — only FM K20 with endpoint projection succeeds
  violation-free (8/10); per-step projection stalls on the roof at K1/K2. The expected shape written in the hole
  ("average-velocity models ahead of instantaneous-velocity matching on the constraint") is **not** visible on the
  readable cells: under per-step projection at K1/K2 the three flow models leave 9.8–11.1 violating steps per flight on the
  tilt and all three stall on the hump
  (CI-MeanFM K3 and MeanFM K3 are *fetch*).
- **§6.4 rows that still quote the v2 corridor** (`tab:summary-models` "MeanFM, CI-MeanFM (12/12)", `tab:summary-projection`
  "per-step at K3 … endpoint at most 7 of 12", `tab:summary-combinations` "MeanFM K3 per-step t, 12/12") — **none of them
  holds on v3**: no tilt configuration succeeds violation-free, and the only hump configurations that do are FM K20 endpoint.
  Those three rows need rewriting from §3.2–§3.4 once the final DA lands (the chapter's 🔒 applies — author's call).
- Chapter 5 statements the data now confirm: the demonstrations never satisfy either constraint (unprojected violation-free
  0/10 everywhere); sideways-only at the launch altitude is infeasible on the tilt (every projected flight descends).

## 4 · The two structural results in detail

### 4.1 Tilt: the window-end hand-over residue

Per projected flow cell: violating steps only at x ∈ [1.75, 2.00] (5–95 % range of all violating steps), max depth
0.089–0.119 m (K1/K2), 0.048–0.079 (K3), 0.034–0.048 (K5), 0.010–0.022 (K20); setpoint feasible at 100 %. At the first
violating step: setpoint x 2.28–2.48, lead over the vehicle 0.52–0.57 m (MeanFM K1 t, FM K3 t, FM K20 endpoint single);
0.23 m for the baseline, which flies slower. The gate `x_active: [-2.0, 2.0]` is evaluated on the setpoint's x under
`-pdes` (`eval_mix_uav.py`, U16 fix); once the setpoint leaves the window the plan is no longer bound by the plane and
returns toward the route altitude, and the vehicle — still 0.5 m inside the window — follows it up into the plane.

### 4.2 Hump: the K1/K2 stall

- Where: 16/16 readable K1/K2 per-step cells, 10/10 flights each, stop at x −0.02 to −0.17 m, altitude 1.39–1.50 m, on the
  step cap (396). The tightened planning limit at those x is 1.39–1.50 m — the flights stop **on** it.
- Plan box: stored plans sit on the altitude clip 1.2992 m for 55–61 % of all plan points at K1/K2 (19–30 % at K3/K5,
  46–54 % at K20). The apex needs 1.485 m (1.515 tightened), so no plan can represent the apex; the setpoint gets there only
  through the action increments.
- Correction metric: the projector minimises ‖Δ‖² in min–max normalised coordinates (`Q = I`, `mix_uav/sampling/projection.py`);
  with x spanning 5.6 m and z 0.40 m of the plan box, one metre of climb costs ≈ (5.6/0.4)² ≈ 200× one metre of retreat, so the
  correction of a plan that runs into the rising roof is almost purely backward along x.
- Gate: the rising row is active while the setpoint's x < 0 and is applied to all eight horizon points, including those
  past the apex, where it asks for more altitude than the falling side allows.
- At K ≥ 3 the flow's later integration steps act after the intermediate projection and the flights cross; the last
  per-step correction still leaves the apex residue (≤ 2.8 cm).
- The one exception, MeanFM K2 `t` (7/10 violation-free): 3 flights (routes R, R, L) get over the apex at 1.52–1.68 m and
  then drift sideways into the corridor **wall** (y −0.79 to −0.97 against the ±0.64 drone-centre limit; 2 of them also
  into the wall-end caps), with the commanded setpoint itself violating (52–65 of their violating steps are the walls or
  caps, 0–3 the roof), and end 1.15–1.79 m from the goal on the step cap — sideways drift at an altitude no demonstration
  flew, not a roof violation.

### 4.3 Options for the author (not applied; each needs a go)

| option | what | cost | effect |
| :-- | :-- | :-- | :-- |
| A · report as is | the tables above; Chapter 6 reads the corridor on violating steps, depth and descent/climb, with S&C reported as 0 and the hand-over residue named as the reason | none | honest, but the tilt ranks nothing on S&C |
| B · score the tilt within the window the plan is bound to | a DA-only secondary metric: violation-free **while the setpoint is inside the window** (the residue after the hand-over excluded) — computable from the stored npz, no run | minutes | separates the configurations; must be named as a secondary metric in the chapter |
| C · re-fly the tilt with the gate on the vehicle's x (or the window extended to x = 2.5) | one config/code change, the tilt C2–C5 waves again | ~1 GPU-day | removes the hand-over residue if the vehicle-gated plane is tracked; changes the stack against v2 |
| D · hump: per-horizon-point row selection (or drop the hump K1/K2 per-step rows as "outside the plan box") | code change in `setup_dpcc_projector` for `plane: xz` windows | code + hump re-run | would test whether the stall is the gate or the plan box |

## 5 · Reproduce

```bash
python3.14 Data_Analysis/DA_in_Paper/analysis/corridor_v3_grid.py                                   # this document's tables
CORRIDOR_V3_CORPUS=<complete plans root> python3.14 Data_Analysis/DA_in_Paper/analysis/corridor_v3_grid.py --json cells.json
python3.14 Data_Analysis/DA_in_Paper/analysis/DA_20260923_corridor_v3_preliminary/make_diag_fig.py   # the diagnostic figure
```

---
Claude (Opus 5.5, Claude Code, U19 corridor chat) · 2026-09-23 · computed locally with python3.14 (numpy, yaml); nothing run on
the cluster; the v3 draft was not edited.
