# TO v3 — R44a is ready: Table 6.14 `tab:uav-scurve` is complete, and its selection is FM at nfe = 1

**2026-09-24 · from the DA side (R44 run chat).** Nothing in `v3/` was touched.
- **Analysis of record:** `Data_Analysis/DA_in_Paper/analysis/DA_20260924_scurve_R44a_raw_grid.md`, script
  `scurve_r44.py`; a row was added to `analysis/INDEX.md`.
- **Jobs:** 26167–26176, all verified.
- **Nothing printed moves.** The six cells Table 6.14 already prints reproduce **flight for flight** under the new tag:
  the same checkpoints, and 0 of 600 outcome values differ. Only ms/step moves, by ≤ 1.2 ms. The table can now come from
  one tag, as the author wanted.

## 1 · Table 6.14 — the whole body from the new tag (the four bold rows were *pending (R44a)*)

```latex
    MeanFM        & 1  & 3/10 & 0/10 & 2.023 & 7/10  & 9.2 \\
    MeanFM        & 2  & 0/10 & 0/10 & 2.515 & 9/10  & 17.9 \\
    MeanFM        & 20 & 0/10 & 0/10 & 2.772 & 10/10 & 176.4 \\
    \addlinespace[3pt]
    CI-MeanFM, $\alpha_{\mathrm{end}}=0.2$ & 1  & 6/10 & 0/10 & 1.306 & 4/10 & 9.3 \\
    CI-MeanFM, $\alpha_{\mathrm{end}}=0.2$ & 2  & 1/10 & 0/10 & 2.082 & 7/10 & 18.0 \\
    CI-MeanFM, $\alpha_{\mathrm{end}}=0.2$ & 20 & 1/10 & 0/10 & 2.511 & 9/10 & 180.4 \\
    \addlinespace[3pt]
    FM            & 1  & 9/10 & 0/10 & 0.571 & 1/10 & 8.9 \\
    FM            & 2  & 7/10 & 0/10 & 1.110 & 3/10 & 17.2 \\
    FM            & 20 & 6/10 & 0/10 & 1.193 & 4/10 & 168.3 \\
    \midrule
    \mainconfig{\baselinemark\ Diffusion} & \mainconfig{20} & \mainconfig{0/10} & \mainconfig{0/10} & \mainconfig{2.432} & \mainconfig{8/10} & \mainconfig{174.3} \\
```
Pending rows: MeanFM 1 and 20, CI-MeanFM 20, FM 1. Previously printed ms/step, now from the new tag: 18.1 → 17.9,
9.2 → 9.3, 18.5 → 18.0, 17.4 → 17.2, 169.5 → 168.3, 173.9 → 174.3.

**Dataref, replacement:**
- `batch_uav_20260924_081422`, `per_rollout_detail.csv`; geometry `s_curve_hg`, variant `diffuser`, controller
  `pid_stopgo`, seed 6.
- Tag `p23scgrid` (CI-MeanFM `EPlatest_p23scgrid`: α_end 0.2, U-Net, latest epoch); jobs 26167–26176.
- The earlier tags (`u7hg`, `EPlatest_u6unet_ae02`) are reproduced flight for flight, with the same checkpoint files,
  which closes the "whether it is the `u7hg` checkpoint" clause.
- The column definitions are unchanged.

## 2 · The `\hole` after Table 6.14

The configuration with the most crossings is **FM at nfe = 1: 9/10** (1 aborted, 0.571 m). Next are FM at nfe = 2 (7/10),
then FM at nfe = 20 and CI-MeanFM at nfe = 1 (6/10 each).

At nfe = 1 endpoint projection has no guiding step, so the evaluation does not run it. Tables 6.15/6.16 are **not flown
yet**: the author picks between per-step only at nfe = 1 and FM at nfe = 20 with both methods (spec §2). Until then, keep
6.15/6.16 pending. The section's sentence "none at one or two" already explains an empty endpoint block.

## 3 · Prose to revisit

1. **The paragraph after Table 6.14.** Four new cells:
   - FM crosses on 9 of 10 flights at one evaluation (1 inverted).
   - MeanFM crosses on 3 at one evaluation and 0 at twenty (all ten inverted).
   - CI-MeanFM crosses on 1 at twenty (9 inverted).

   **For every flow-based model, crossings fall and aborts rise from one to two to twenty evaluations** (FM 9/7/6,
   CI-MeanFM 6/1/1, MeanFM 3/0/0). Flights invert 11.6–15.9 s into the flight (control steps 386–530); successful flights
   reach the line at step 590–635.
2. **"No flight succeeds within the constraints"** now holds for all ten cells: no flight of any cell is violation-free.
   The flow models' flights track their plans with a mean error of 0.30–0.33 m, while the demonstrated route clears the
   constraint boundaries by 0.121 m (the evaluation's own feasibility check). The baseline tracks at 0.558 m.
3. **Success definition** (Ch 5 `sec:setup:metrics:uav`, and the Table 6.14 caption). The evaluation counts a crossing
   only in safe flight: not aborted (already stated), within the contact limit, and **airborne, lowest altitude above
   0.2 m** (`eval_mix_uav.py:1994–2027`). Three flights in this grid crossed after reaching the floor and are not
   successes (MeanFM nfe 2; CI-MeanFM nfe 2 twice). One clause.
4. **FLAW §B.** FM leads on this scene and is the selected configuration. FM alone is sampled from half-scale initial noise
   (`mix_uav/models/engine_registry.py:274`; `data_status/FLAW_20260920_known_flaws.md` §B). A sentence ranking FM above
   MeanFM or CI-MeanFM here carries §B2's caveat; adding it is the author's call, as §B2 says. FLAW §B2's quadrotor row
   ("the only flow model that never reaches S&C 1.00") does not describe this scene, where no model reaches S&C above 0.
5. **`sec:res:uav:controller`** (for when R44c lands). The pilot's failure, MeanFM nfe 10 with all ten inverted, is the
   majority failure of MeanFM, CI-MeanFM and the baseline in this grid. The selected FM nfe 1, however, crosses 9/10 under
   the cascaded geometric controller. Its limit is violation-free flight (tracking error 0.303 m against a 0.121 m
   clearance), not crossing. Table 6.16 on FM nfe 1 will show whether MuJoCo MPC turns crossings into violation-free
   crossings; "the scene is limited by the controller" should be phrased against that result.
6. `sec:setup:protocol:uav` says the selected configuration is flown "under both projection methods". Whether that holds
   depends on the author's pick (§2).

## 4 · Not affected

Figure 6.9 and Table 6.17 (the MeanFM nfe 10 pilot) and `app:uav-scurve-budgets` (nfe 5/10 for the record).
