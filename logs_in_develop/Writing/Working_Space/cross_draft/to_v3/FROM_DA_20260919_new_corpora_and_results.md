# FROM DA → v3 · 2026-09-19 — new corpora, four results, one scene withdrawn

**No draft text was edited.** This note is the hand-off: the DA of record has moved to new batches and
Chapter 6 now has numbers where it had markers. Full analysis:
[`DA_20260919_wave_1718_corridor_endpoint_and_scurve.md`](../../../../../Data_Analysis/DA_in_Paper/analysis/DA_20260919_wave_1718_corridor_endpoint_and_scurve.md)
· index: [`analysis/INDEX.md`](../../../../../Data_Analysis/DA_in_Paper/analysis/INDEX.md)

## Corpora of record — update §"Corpora" and `tab:corpora`

| task | old | new |
| :-- | :-- | :-- |
| D3IL-avoiding | `15-09/batch_avoiding_combined_20260915_100757` | **`19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703`** |
| quadrotor | `15-09/batch_uav_20260915_100816` | **`19-09-UAV-Pillars-Exclude/batch_uav_20260919_111701`** |
| D3IL-aligning | `15-09/batch_va2_20260915_100754` | unchanged — nothing ran |

Every previously published number is reproduced exactly from the new batches (regression check in the DA
§preamble), so **no existing Chapter 6 number changes**. The new batches only add rows.

## 1 · `tab:avoiding-dpcc-protocol` is complete — the `\hole` can go (R8, R19)

The four *not yet evaluated* rows now have data: instantaneous-velocity matching and
consistency-interpolated average-velocity matching (U-Net, floor 0.2) at $\nfe=1,2$, five seeds × two
episodes, all three selection rules. Numbers in DA §1.

The headline this gives Chapter 6, at DPCC's own protocol and architecture-matched (4.0 M U-Net against
4.0 M U-Net), against the pinned target (diffusion $\nfe=20$, `dpcc-c`, S&C 1.000 · 70.1 steps · 553.4 ms):

* flow matching $\nfe=1$ `dpcc-c` — 1.000 · 67.0 steps · 17.3 ms
* CI-MeanFM $\nfe=1$ `dpcc-t` — 1.000 · 59.2 steps · 18.1 ms

Equal constraint satisfaction, fewer control steps, ≈ 31× less compute per step: Pareto-dominant on all
three axes, so this can be stated as a beat rather than a trade-off. Flow matching is the only model holding
1.000 under **all three** rules at both budgets; CI-MeanFM is rule-sensitive (0.900 under random selection
at $\nfe=2$, and an 89.7-step stall under `dpcc-c` — the same effect already described for MeanFM's 98.0).

## 2 · `sec:res:uav:projection` — the endpoint comparison on corridor is now complete (R20)

`hardflow_sls-r` and `hardflow_sls-c` exist at $\nfe=3,5$ for all three flow models, so the section no
longer compares on two of six configurations. **The conclusion does not change: endpoint projection does not
overtake per-step projection on corridor under any selection rule** — best endpoint cell 0.58 (MeanFM,
minimum-cost, $\nfe=3$) against 1.00 for every per-step rule at the same budget, at roughly half the
projection cost. It also degrades as the budget grows (0.58 → 0.33 at $\nfe=5$).

The $\nfe=1$ endpoint cells will never exist: at one ODE step the only step is the terminal step, so there
is no guided step to place. That is a model limit, not a pending run — the `\guard` can say so plainly.

**New, unplanned, and worth a sentence:** the untightened companion `dpcc-c-bounds_free-pdes` scores 0.00 at
every budget for every model, against 1.00 for its tightened twin. On corridor the tightening margin is not
a refinement of the projector, it is the whole of its constraint satisfaction.

## 3 · `sec:res:uav:scurve` — R10's re-run is done and yields nothing

The post-fix endpoint rows (tag `u18sc`) are **all zero**, S&C and collision-free alike, for FM $\nfe=20$,
MeanFM $\nfe=10$ and CI-MeanFM $\nfe=5$. The reason is physical: the divergence guard ends 57 %, 92 % and
87 % of flights respectively because the drone inverts — **including on the unprojected plan** (4/10, 10/10,
8/10). So this is the scene and the tracking controller, not the projector.

Suggested handling: keep s-curve as the failure case it is, and print the abort counts beside the scores.
The `\guard` withholding the pre-fix rows can be replaced by this, but the section should not present the
zeros as a comparison between methods.

## 4 · 🔴 UAV-pillars is withdrawn from the results

`pillars_hg` enforces the constraint its own demonstration generator was built to satisfy, so its projected
configurations measure how little of an already-feasible plan each method disturbs
([`PENDING_20260918_pillars_geometry_redesign.md`](../../data_status/PENDING_20260918_pillars_geometry_redesign.md)).
The DA now gates the scene off entirely (`PILLARS_EXCLUDED` in `uav_results.py`), including the $\nfe=1$ and
$\nfe=2$ cells the 09-18 wave produced.

**This affects `tab:uav-pillars`, `tab:uav-pillars-best`, `tab:uav-pillars-endpoint`, `sec:res:uav:pillars`
and any Chapter 6 or 7 claim resting on them** — including the criterion-swap note in `analysis/INDEX.md`
(FM leads at the goal-passed criterion, MeanFM at the strict one), which is a pillars statement. Nothing is
being asked of v3 yet beyond not building new text on the scene: the replacement runs on the enlarged
geometry (`pillars_xl`, `pillars_xxl`) are in
[`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](../../data_status/SLURM_RUNBOOK_20260919_pillars_enlarged.md).

## Ledger effect

✅ R8, R19 closed · ✅ R6 closed at DPCC's protocol (extended protocol still seed 6) · ✅ R20 closed for
$\nfe=3,5$, ❌ not runnable at $\nfe=1$ · ✅ R10 run, all-zero · ❌ R22 not applicable (the diffusion engine
exposes no velocity field) · ⏸ R12, R13, R21 superseded by the pillars redesign · ⏳ R18 still open, and its
phase B is now unblocked since the 09-17 queue is clear.
