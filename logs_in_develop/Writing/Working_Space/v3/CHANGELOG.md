# CHANGELOG — `Working_Space/v3`

Every change to the v3 draft, its figure pipeline or its inheritance machinery gets an entry here,
newest first. Format follows [`../v2/CHANGELOG.md`](../v2/CHANGELOG.md): what changed · why · what it
is sourced from · what it left open.

**Rules for this file**

- One entry per working pass, not per edit.
- **Every number injected into the draft names its evidence of record** — a batch directory and the
  DA that published it, or a repo file with line numbers. "From a summary table" is not a source:
  the readiness ledger is a *pointer*, and what gets cited is what it points at.
- Mechanical checks (`python3 tools/check.py`) are re-run after every pass and their result is
  recorded. **There is no TeX toolchain in this container, so "checked" never means "compiled".**
- A pass that absorbs a v2 change records which v2 revision it absorbed, from
  `inherited/SYNC_STATE.json`.

---

## v3.30 — 2026-09-17 · Results chapter traversed against the pending-data ledger → [`changelogs/v3.30_20260917_results_ledger_traversal.md`](changelogs/v3.30_20260917_results_ledger_traversal.md)

- Matched every `\hole`, `\provisional`, `\todofigure` and `\guard` in Chapter 6 to a ledger row; added **R12–R17** for six gaps that had none, two of which carry no marker in the draft at all (the baseline's missing pillars configurations, and the pillars budget ladder stopping at K=2).
- Closed **D3** (the MuJoCo MPC overhead claim was removed in v3.28, not verified); marked **D10** as needing the cluster; rewrote stale section numbers as labels.
- Added a coverage table to the ledger mapping all 26 draft locations to their rows. No thesis prose changed.

## v3.29 — 2026-09-17 · v2.23 absorbed, three RQs, captions cleaned, smoothness section, trajectory figures flagged → [`changelogs/v3.29_20260917_v2_sync_captions_smoothness_and_trajectory_holes.md`](changelogs/v3.29_20260917_v2_sync_captions_smoothness_and_trajectory_holes.md)

- Absorbed **v2.23** (`sync_v2.py merge`) and closed the open v2 item: three RQs instead of four, with the budget question folded into RQ1; Ch 8 answers renumbered.
- Table 6.1's caption no longer explains what was excluded; the remaining gap is recorded in the new `data_status/PENDING_20260917_dpcc_protocol_rows_table61.md`, including the `Folder_Name` backbone trap.
- Table 6.4 reordered to MeanFM, CI-MeanFM, FM, Diffusion.
- §6.1.1.4 is now "Smoothness of the Plans" — the same evidence, stated as what the section shows rather than what the metrics miss.
- Trajectory figures flagged: `fig:uav-scurve-paths` placeholder after Table 6.12 plus holes for pillars, corridor and alignment, with `REQUEST_20260917_trajectory_figures.md` naming the cluster runs to download (ledger D10).
- `check.py` passes; bundles rebuilt, 23/23 figures. Not compiled.

## v3.28 — 2026-09-17 · Chapter 6: missing protocol rows recovered, caveat reasoning, goal-passed criterion → [`changelogs/v3.28_20260917_ch6_protocol_rows_metrics_and_goal_passed.md`](changelogs/v3.28_20260917_ch6_protocol_rows_metrics_and_goal_passed.md)

- Table 6.1 filled with MeanFM at K=1/K=2 over five seeds and two episodes: the rows were in the batch all along, hidden because `Folder_Name` does not carry the backbone.
- §6.1 intro trimmed to a roadmap and names the candidate-selection rules as v2 defines them; §6.1.1 separates the diffusion and ODE meanings of K (sent to v4).
- The Table 6.3 guard is now a result; the caveat names the inverse kinematics and reads DPCC's Table 2 as the size of the downstream margin.
- §6.2.1 leads with the finding; Table 6.6 marks its selected row; every model table is ordered MeanFM, CI-MeanFM, FM, Diffusion.
- UAV success is now "the goal passed" (crossing the finish line, as on D3IL-avoiding). **On UAV-pillars this swaps the lead: FM 0.718 against MeanFM 0.627, with MeanFM shorter by 120 control steps.** Chapter 8 and the recap tables follow.
- `check.py` passes; bundles rebuilt, 23/23 figures. Not compiled.

## v3.27 — 2026-09-17 · Chapter 5: platform facts, constraint figure, incentive, metric definition → [`changelogs/v3.27_20260917_ch5_platforms_constraints_metrics.md`](changelogs/v3.27_20260917_ch5_platforms_constraints_metrics.md)

- `tab:platforms`: IK wording checked against the controller code and abbreviated, robot named *Franka Emika Panda* with its cylindrical rod end effector (not a gripper), table widened, self-attacking real-time clause removed.
- §5.2.2 gains the incentive for the alignment environment; §5.4.2 carries D3IL's own success rates (0.359 from images against 0.950 on avoiding from the state); §5.4.3 loses a stray baseline sentence.
- `tab:uav-scenes` is now a constraint table without the workspace box, and new Fig 5.7 draws the three aerial constraint sets like the avoiding one.
- §5.5.1 defines the time metric as DPCC's "time to compute one action" with no caveat; the machine caveat moved to the appendix and the timing caveat to v4.
- Fig 5.5 re-framed from the rollout where the in-hand box is complete and centred.
- `check.py` passes (23 figures); bundles rebuilt. Not compiled.

## v3.26 — 2026-09-17 · MuJoCo wording, Ch 5 holes, no statistics, two caveats, UAV restructure → [`changelogs/v3.26_20260917_mujoco_holes_caveats_uav_restructure.md`](changelogs/v3.26_20260917_mujoco_holes_caveats_uav_restructure.md)

- MuJoCo named as the simulator throughout Ch 5 (Fig 5.5 labels, captions); D3IL = task files, data, cameras.
- Filled the holes on the UAV workspace box and tightening, on demonstration counts, gains and splits, and on the UAV controller; rewrote §5.4.3 (original MuJoCo MPC task).
- Removed every p-value and test name (Ch 5, Ch 6, Ch 8 draft); new translation rules 7–8.
- New caveat subsections: DPCC metrics and smoothness (§6.1), tracking controller (§6.3). §6.2.2 shows endpoint projection as the D3IL-aligning method (`tab:va-projection`).
- §6.3 restructured around all four models per scene with UAV-pillars as the main test; §6.4 is a three-table recap.
- `check.py` passes; bundles `thesis_v3_20260917_161355_new` and `_new_clean`, 22/22 figures. Not compiled.

## v3.25 — 2026-09-17 · thesis-style visuals, Chapter 5 audit, and clean review bundle → [`changelogs/v3.25_20260917_thesis_visuals_ch5_and_clean_bundle.md`](changelogs/v3.25_20260917_thesis_visuals_ch5_and_clean_bundle.md)

- Reworked the Chapter 5 opening and task terminology, filled the alignment workspace bounds, audited
  the UAV demonstration method against v2, and gave Table 5.7 a relaxed dedicated float page.
- Simplified Figures 6.1, 6.2, 6.4 and 6.5 to thesis-style plots, changed the model/constraint palette
  to dark slate, and made selected table configurations unmistakable with a bold black triangle.
- Added a dated v3.25 handoff for v2's Section 1.2 and added an optional clean bundle mode that hides
  source notes while keeping holes visible.
- `check.py` passes; both `thesis_v3_20260917_144454_new.tex/.zip` and
  `thesis_v3_20260917_144454_new_clean.tex/.zip` contain 22/22 rendered figures and verify
  byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.24 — 2026-09-17 · consistent model labels in every figure and table → [`changelogs/v3.24_20260917_figure_table_abbreviations.md`](changelogs/v3.24_20260917_figure_table_abbreviations.md)

- Standardised every v3 figure and table body on **Diffusion**, **FM**, **MeanFM** and **CI-MeanFM**,
  including the Figure 6.3 matrix, appendix mappings, and all four generated Chapter 6 charts.
- Section 5.3 and the affected figure captions explicitly map each short label to its full scientific
  name; running prose continues to use the full mechanism names.
- Updated the canonical translation table and official plotting vocabulary, rebuilt and visually
  checked the four affected charts, and re-exported the figure store to v3.
- `check.py` passes; `thesis_v3_20260917_110958_new.tex/.zip` contains 22/22 rendered figures and
  verifies byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.23 — 2026-09-17 · monochrome table emphasis → [`changelogs/v3.23_20260917_monochrome_table_emphasis.md`](changelogs/v3.23_20260917_monochrome_table_emphasis.md)

- Replaced the red main-configuration styling in Chapter 6 with bold monochrome emphasis for grayscale
  and accessibility robustness. Table 6.2 names its selected operating point directly because bold
  already denotes the best rule per model there.
- Audited the remaining colour commands: they are drafting diagnostics that must be removed before
  submission, not final-text styling; figure colours remain data encodings.
- `check.py` passes; `thesis_v3_20260917_110526_new.tex/.zip` contains 22/22 rendered figures and
  verifies byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.22 — 2026-09-17 · CI-MeanFM table-label hotfix → [`changelogs/v3.22_20260917_ci_meanfm_hotfix.md`](changelogs/v3.22_20260917_ci_meanfm_hotfix.md)

- Replaced the rejected `$\alpha$-MeanFM` table label with the author-approved **CI-MeanFM**:
  consistency-interpolated MeanFM. Full prose remains **consistency-interpolated average-velocity
  matching**.
- Updated Section 5.3, all affected result tables, the canonical translation table and the prior layout
  entry so the rejected label is not left as current guidance.
- `check.py` passes; `thesis_v3_20260917_105825_new.tex/.zip` has 22/22 rendered figures and verifies
  byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.21 — 2026-09-17 · compact UAV floats and result-table model labels → [`changelogs/v3.21_20260917_compact_floats_and_table_labels.md`](changelogs/v3.21_20260917_compact_floats_and_table_labels.md)

- Reduced the footprint of Table 5.3, Figure 5.6 and Table 5.4; the UAV scene figure now occupies
  78% of the text width, and both tables use compact spacing and permissive float placement before the
  existing section barrier.
- Defined the table-only labels **Diffusion**, **FM**, **MeanFM** and **CI-MeanFM** alongside the
  full mechanism names in Section 5.3 and recorded them in the canonical translation table.
- Replaced long model cells throughout the v3 results tables, split the non-wrapping headers in Table
  6.7, and tightened Tables 6.7, 6.8 and 6.10 to prevent overlap.
- `check.py` passes; the new-only bundle `thesis_v3_20260917_104554_new.tex/.zip` contains 22/22
  rendered figures and verifies byte-faithful. Not compiled: this container has no TeX toolchain.

## v3.20 — 2026-09-17 · mechanism names, Chapter 5 layout, and obstacle-avoidance emphasis → [`changelogs/v3.20_20260917_naming_ch5_layout_ch6_highlights.md`](changelogs/v3.20_20260917_naming_ch5_layout_ch6_highlights.md)

- Applied the author-approved names **instantaneous-velocity matching**, **analytic average-velocity
  matching**, and **consistency-interpolated average-velocity matching** throughout v3-owned prose,
  tables, captions, appendix mappings and the affected official figure; updated the canonical
  translation table and sent the Chapter 4 rename to v2.
- Clarified MuJoCo as the common simulator, D3IL as the Panda integration, and the X2 lineage through
  MuJoCo Menagerie and the MuJoCo MPC task patch. D3IL-avoiding is now explicitly the exact DPCC
  benchmark used for controlled validation.
- Kept each §5.2 task's figures inside its subsection, reduced their footprint, and transposed the
  oversized Table 5.5 into a compact protocol-by-metric layout.
- Marked the main §6.1 configurations in red and revised its conclusion: the
  consistency-interpolated objective does show the remembered single-seed edge over the analytic
  objective, but it is not statistically resolved; the analytic objective carries the five-seed result.
- Rebuilt and visually checked `fig_avoiding_raw_models`, re-exported v3, passed `check.py`, and built
  and byte-verified both the full and v3-new-sections-only 22-figure bundles.

## v3.19 — 2026-09-17 · D3IL-avoiding: DPCC's protocol first, every selection rule shown

- 🔴 **Protocol corrected.** DPCC's released evaluation runs `n_trials: 2` for each of five training seeds (`aux_repo/dpcc/config/projection_eval.yaml`, `scripts/eval.py`) — the paper's "five training seeds and ten test seeds" is **10 episodes per geometry**, not 50 as §5.5.1 said. §5.5.1 rewritten: 10 episodes = the protocol of DPCC (primary), 100 = the extended evaluation (stronger evidence).
- **§6.1 restructured** (author): opening states the two protocols; *Generative Models* now has (a) **At the Protocol of DPCC** — new `tab:avoiding-dpcc-protocol` with all three rules $r/c/t$; (b) **Extended Evaluation** — `tab:state-headline` rebuilt with all rules for diffusion K20, flow matching K1/K2, MeanFlow K1/K2, best rule in bold; then α-Flow via `tab:state-models`.
- **Results.** At DPCC's protocol the baseline reproduces the paper: DPCC-C tightened **S&C 1.000 in 70.1 steps** (published 0.98 in 69.0); flow matching at the same K20 reaches 1.000 in 63.2 steps at 477 vs 553 ms. Extended, at each model's best rule: MeanFlow K1 $t$ 0.993 / **61.0** steps / 18.1 ms; flow matching K1 $c$ 0.997 / 67.5 / 18.9; diffusion $c$ 0.983 / 69.0 / 563.5 → **MeanFlow ahead of both; flow matching level with diffusion on the path at 1/30 of its time.** The rule matters most for MeanFlow ($c$ picks stalled 98-step plans at K2).
- `tab:target` gains a *protocol of DPCC* column (1.000 / 70.1 / 0 violations / 0.553 s).
- 🔴 **Data gap, written as placeholder:** flow matching K1/K2, MeanFlow **U-Net** and α-Flow **U-Net** were never evaluated at 5 × 2 (the 5-seed 2-episode MeanFlow/α-Flow runs are DiT/SiT; U-Net α-Flow exists for seed 6 only). *Pending* rows + `\hole`; cluster commands in `DA_in_Paper/plotting/REQUEST_20260917_avoiding_dpcc_protocol.md`; ledger R8 raised to 🔴.
- Removed a truncated sentence ("the diffusion baseline at") and a duplicated flow-matching paragraph from the old §6.1.1.
- New `DA_in_Paper/analysis/avoiding_rules_by_protocol.py` + `INDEX.md` row. v2 notified (protocol wording; a new 14.9 cm table in its Ch 4).

---

## v3.18 — 2026-09-17 · v2.17 sync; projection-cost fix; UAV rates in simulated time; α-Flow naming

**Sync.** v2.16 → **v2.17** (7 inherited files, all fast-forward, merged by name). Acting on v2.17's notes:
- 🔴 removed v3's duplicate `tabularx` / `\newcolumntype{L}` / `secnumdepth` — the inherited preamble now defines them, and a second `\newcolumntype{L}` is a LaTeX error;
- 🔴 **projection-cost contradiction** (v2's `to_v3` note): the `\guard` below `tab:hf-ladder` quoted "1.86–3.57× the cost at equal candidates" — that is **UAV-corridor** data (DA_20260824 §5), misattributed. The D3IL-avoiding K3/K5 comparison gave both methods **4 candidates** (`…_A1_B4_…hfmink…`), so "about half the time" stands. Guard corrected; resolution + a consistent cost sentence sent to v2; inbox item ✅.

**"33 Hz plan / 100 Hz controller" (author: "99 % fake").** Checked in code: *not invented, but misleading*. Physics dt 0.01 s; planner every `decim = 3` steps (0.03 s), the demonstrations' recording interval (`dataset_writer.py:73`); tracker every physics step (`eval_mix_uav.py:1609–1735`). But the evaluation is **lock-step** — the simulation waits for the planner, which takes 9–4878 ms wall clock (21 % of cells under 30 ms). `tab:platforms` now states intervals of simulated time and says so; `tab:uav-demos` "(33 Hz)" → "(0.03 s)". v2's Ch 4 rate table/equation flagged in `cross_draft`.

**Naming.** Read from the papers: MeanFlow is the paper's own name (average velocity) — kept; "mean flow matching" is not a term. What we call "consistency training" is **α-Flow** (a family between trajectory flow matching α = 1 and MeanFlow α → 0; α = consistency step ratio), and "Consistency Training" is a *different* published method (Song et al.). Renamed across Ch 5, 6, 8, appendix: **α-Flow**, final value **α_end = 0.2 / 0.05** (was "floor"); headings via `\texorpdfstring`; one definition of all three flow models at §5.3; figure labels (`fig:avoiding-raw-models`) rebuilt. Translation table §2 and rule 1 revised; v2 notified (10 occurrences, §4 title); memory updated.

**Tooling.** 🔴 The rename's first pass wrote BEL control characters (Python expanded `\a` in `\alpha`); caught by reviewing the diff, repaired (46). `check.py` now fails on control characters — verified with an injected one.

check.py pass; bundle `thesis_v3_20260917_*_new.zip`, 22/22.

---

## v3.17a — 2026-09-16 · hotfix: remaining environment prefixes; D3IL credit on the camera figure

- **Prefixes, second pass** (paragraph-level scan, catches names split across lines): Ch 5 intro now introduces D3IL-avoiding, D3IL-aligning, UAV-corridor/-pillars/-s-curve; the UAV scene description list; §5.3.3 "as on D3IL-aligning"; `tab:train` caption; Fig 5.4 caption; Ch 6 intro; D3IL-avoiding conclusion; D3IL-aligning budget text; RQ2. Object descriptions ("walls and pillars", "pillar pairs") left as they are.
- **Fig 5.5 (`fig:aligning-cameras`)**: caption credits D3IL — "Adapted from D3IL (appendix, Fig. 8): the camera set-up and the two views are D3IL's; the frames are rendered in the alignment environment of this thesis". Panel labels and §5.1.2 text use D3IL's names, *front view* and *in-hand view*, replacing "fixed overhead / wrist-mounted camera".
- check.py pass; bundle 22/22.

---

## v3.17 — 2026-09-16 · seed policy: five seeds shown once, single seed stated once

Follows the author's multi-seed assessments (`Gen14/Campaign_20260916_va_5seed_assessment/`, `Gen15/Campaign_20260916_uav_5seed_training/`): no further seeds for D3IL-aligning (cost) or UAV (cluster disk).

- **§5.5 intro:** one sentence — D3IL-avoiding over five seeds; D3IL-aligning and UAV with seed 6 "within the storage and compute available to this work". No further justification.
- **§5.5.1:** new `tab:seed-spread` + paragraph, over seeds 6–10 on the two geometries every seed covers: MeanFlow K1 S&C 0.995 ± 0.011, 62.1 ± 1.8 steps, 18.0 ± 0.2 ms; flow matching K2 68.6 ± 2.2 steps; diffusion 73.8 ± 11.3 steps; diffusion/MeanFlow time 28–33× in every seed. Script `DA_in_Paper/analysis/avoiding_seed_spread.py`, `INDEX.md` row.
- **Removed** single-seed caveats in Ch 6 (six `\guard`/`\provisional` sentences and one conclusion sentence) and Ch 8; non-seed content of those caveats kept. Neutral protocol statements ("seed 6" in captions and tables) kept.
- Ledger: R1 and R4 closed ❌ with reasons; seed policy recorded. `cross_draft/to_v4` findings updated so the Discussion does not re-open it.
- check.py pass; bundle 22/22.

---

## v3.16b — 2026-09-16 · environment prefixes where environments appear together

- Defined once at the start of §5.1: **D3IL-avoiding, D3IL-aligning, UAV-corridor, UAV-pillars, UAV-s-curve** (UAV spelled out).
- Applied wherever several environments are shown or compared together: §6.4 prose and all three summary tables, the UAV conclusion and cross-scene consistency-training text, Ch 8 (summary and RQs), `tab:train` / `tab:eval` headers, §5.5.3, `tab:uav-scenes`, `tab:uav-demos`, Fig 5.4 panel labels, appendix `tab:corpora`. Single-environment sections keep plain names; object descriptions ("corridor walls") unchanged; `\autoref`/`\label`/`\texttt`/`\dataref` untouched.
- Recorded in the translation table; v2 notified in `cross_draft/INBOX.md`. check.py pass; bundle 22/22.

---

## v3.16a — 2026-09-16 · review of v3.16 (Codex) + isolated platform renders

- **Review: v3.16 holds.** Platform specs re-derived from `quadrotor_modified.xml` (1.325 kg, rotor centres 0.228 m, 0.54×0.62 m, 0–13 N) and `panda.xml` (wrist camera); camera frames are from our own expert-replay GIF; the Ch 6 conclusions match `va_results.py` / `uav_results.py`; check.py and bundle pass.
- 🔴 **Fixed a silent defect of the restructure:** the new `\subsubsection` parts were unnumbered under `scrbook` defaults, so 13 `\autoref`s would print the parent subsection's number. `secnumdepth` raised in `parts/00_preamble_v3.tex`; logged for v2 in `cross_draft/INBOX.md`.
- **Fig 5.1** now shows the Panda and the X2 on their own — MuJoCo renders of the model files with the background removed by the segmentation pass (`PLATFORM_RENDERS`, `prep/render_mujoco_scenes.py`). This also removes one use of the D3IL README GIF.
- Fixed `render_mujoco_scenes.py --check`, which always reported the pillars render stale (nested tuples). 5/5 current.
- Bundle `thesis_v3_20260916_142003_new.zip`: 22/22 figures render.
- ⚠️ Still open: Figs 5.2 and 5.3 use crops of D3IL's own README GIF, captioned "authentic simulator frame", without credit.

---

## v3.16 — 2026-09-16 · platforms, alignment camera inputs, and two-part result structure → [`changelogs/v3.16_20260916_platforms_cameras_results_structure.md`](changelogs/v3.16_20260916_platforms_cameras_results_structure.md)

- Chapter 5 now introduces the Panda and Skydio X2 with source-derived specifications and shows the two
  actual $96\times96$ alignment observations from one expert-demonstration instant.
- Every Chapter 6 environment now follows **Generative Models → Projection Methods → Conclusion**, with a
  final table of supported model--projection operating points.
- Obstacle avoidance explicitly keeps per-step projection at its one/two-evaluation operating point;
  alignment and UAV state their supported combinations and evidence limits.
- The s-curve section now isolates the tracking-controller effect and distinguishes the flyable
  unprojected plan from the unresolved projected stack.

---

## v3.15 — 2026-09-16 · §6.2 alignment and §6.3 quadrotor rewritten from the 15-09 data → [`changelogs/v3.15_20260916_va_and_uav_results.md`](changelogs/v3.15_20260916_va_and_uav_results.md)

- Every §6.2–6.3 number recomputed by two scripts in `DA_in_Paper/analysis/`; analyses of record reproduced exactly.
- Alignment: MeanFlow ahead of both; flow matching ≈ diffusion; consistency training level at K2, behind at K20. Evaluated on **training** contexts — disclosed.
- Quadrotor: corridor and pillars written in full (MeanFlow > flow matching > consistency training on pillars); s-curve as a listing with the controller discussion.
- 🔴 Five false statements in the previous draft corrected (diffusion violation-freedom on alignment; pillars "left the constraint set"; MF>FM>diffusion on alignment; ≥2 guiding steps; pooled ablations).

---

## v3.14a — 2026-09-16 · plan matrix and Codex handover

- **`fig:raw-plans` is now a 2×4 matrix**: MeanFlow, flow matching, diffusion, consistency training × $\nfe\in\{1,2\}$. Three panels exist (MeanFlow K1/K2, diffusion K1); the other five are visible `\todofigure` placeholders, registered in `PLANNED`, with the cluster runs in `DA_in_Paper/plotting/REQUEST_20260916_cluster_fm_plan_panels.md`. Laid out 2×4 rather than 4×2, which would not fit on a page.
- `tools/check.py`: panel grids of minipages are now measured by their declared widths (the character estimate had flagged this figure as 35 cm wide); verified to warn at an oversized width.
- **Plan** for §6.2 alignment and §6.3 quadrotor results (first written as a Codex handover, reassigned to v3 by the author): `plans/PLAN_20260916_VA_and_UAV_results.md` — sources, expected story, known traps (held-out split, an unexplained `d3il_baseline` engine in the alignment data, consistency training below flow matching on pillars, the partially invalid MPC-vs-PID report), rules and deliverables.

---

## v3.14 — 2026-09-16 · MuJoCo renders, §5.3 baselines, §5.5 parameters → [`changelogs/v3.14_20260916_mujoco_renders_baselines_protocol.md`](changelogs/v3.14_20260916_mujoco_renders_baselines_protocol.md)

- Fig. 5.4 top row: real MuJoCo renders with the generator's reference path; 4 SVG-only figures given PNGs (18/18 render).
- Fig. 6.3 flow matching: no plan data off the cluster; exact run request written, panels registered as planned.
- §5.3.2 D3IL reference and `tab:va-vs-d3il`; §5.3.3 MuJoCo MPC quadrotor task as precedent.
- §5.5 `tab:train` / `tab:eval` from the training logs. 🔴 Training is **not** uniform across models (batch, lr, action weight, EMA) — disclosed in a `\guard`.

---

## v3.13 — 2026-09-15 · authentic environment views, figure explanations, raw-plan source boundary → [`changelogs/v3.13_20260915_environment_renders_and_figure_explanations.md`](changelogs/v3.13_20260915_environment_renders_and_figure_explanations.md)

- **Chapter 5 now shows all three environments.** Authentic D3IL frames cover obstacle avoidance and
  alignment; authentic rollout frames cover corridor v2, pillars and s-curve. Alignment and UAV views
  are paired with readable geometry reconstructions whose exact context/MJCF sources are audited.
- **Figures 6.1 and 6.2 are explained in the prose.** Figure 6.2 no longer mislabels its smaller
  five-seed by two-episode diffusion sample as the published DPCC protocol.
- **Figure 6.3 now includes MeanFlow at $K=2$.** Naïve flow matching at $K=1$ and $K=2$ cannot be
  rendered from this checkout: neither diagnostic panels nor saved plan states exist locally. Both
  cluster render sources are registered as `PLANNED`; no panel was fabricated from aggregate rows.
- Full figure store rebuilt and 18 referenced figures exported; mechanical checks pass with no
  `\todofigure`. No TeX toolchain is installed, so this pass was not compiled.

## v3.12 — 2026-09-15 · consistency training into the results, the protocol made exact, two figures rebuilt → [`changelogs/v3.12_20260915_af_results_protocol_and_figures.md`](changelogs/v3.12_20260915_af_results_protocol_and_figures.md)

- **Consistency training was missing from the results.** Built in from `Report_20260903_AF_UNet`: the
  new four-model Table 6.2 (`tab:state-models`) and a new figure `fig:avoiding-raw-models` — goal
  reached with the projection off, 20/20 against MeanFlow's and flow matching's 17/20 and the baseline's
  12/20. 🔴 Caught while building it: averaging each model over the seeds it happens to have reads
  0.97/0.97/0.92 instead of 0.85/0.85/0.60, because only the consistency-training folders are seed-6-only.
- 🔴 **§5.5.1: ten episodes is not "the published number".** The paper is five training seeds x ten test
  seeds = **50 rollouts per geometry**; our two settings are 5x2 = 10 (smaller) and 5x20 = 100 (twice).
  Set out explicitly, and the same wrong label removed from `sources.py`.
- **Figure 6.3**: the flow-matching and consistency-training plan panels have never been rendered
  (`Report_20260903` section 8, all eight still placeholders). Caption now says which models it shows;
  both missing panels registered in `PLANNED` with their plan directories.
- **Figure 6.4 rebuilt**: banned vocabulary on the canvas ("arm", "genuine", "degenerate", "not
  citable"), "lower is better" in an axis label, a truncated subtitle, and 🔴 a semantic error — both
  series were drawn hollow at K=2, though degeneracy applies only to endpoint projection.

---

## v3.11 — 2026-09-15 · v2 checked (unmoved), §6.1.1 renamed, smoothness argued with the baseline's own margin, endpoint projection at low budget → [`changelogs/v3.11_20260915_smoothness_margin_and_lowK_endpoint.md`](changelogs/v3.11_20260915_smoothness_margin_and_lowK_endpoint.md)

- **v2 has not moved** — still v2.16, both hashes identical to `SYNC_STATE.json`; nothing to merge. Read
  it anyway: it leaves a `\hole` at `:332` for a result sentence from Chapter 6, now supplied in
  `handover/HANDOVER_20260915_for_v2_result_sentences.md`. ✅ **v2 is not missing the low-budget
  endpoint-projection point** — Ch. 1 `:331`, Ch. 3 `:590` and §4.5.4 `:1594–1641` all carry it; what it
  lacks is the result, which it explicitly delegates to v3.
- **§6.1.1 "Generative Models against the Baseline" → "Flow Matching against Diffusion"**: the baseline
  is a diffusion model, so the old title implied a distinction that does not exist.
- **§6.1.3 → "Smoothness of the Plans before Projection"**, now with a number behind the claim that the
  control stack hides plan quality: DPCC's own Table 2 — a dynamics model wrong by a factor of four
  still satisfies the constraints in 0.77 of episodes. Cited as `\textcite{romer2025diffusion}` and read
  off the PDF this pass.
- **The $\nfe=10$ endpoint-projection evaluation existed and was missing from the draft.** Added as
  `tab:hf-ladder` with the guiding-step count per budget, the measured solve counts confirming zero at
  $\nfe\le2$, the bit-identical $\nfe=2$ parity run at 2.45× cost, and a `\guard` for the candidate-count
  mismatch.
- **Headline stated**: this task's operating point is one network evaluation, MeanFlow is at $0.993$
  there, and endpoint projection has no step to act on at that budget — in §6.1's opening, §6.1.4's
  close, §6.4.2 and RQ3.

---

## v3.10 — 2026-09-15 · the plans figure cut to its panel, "genuine steps" renamed, checkpoint data recorded → [`changelogs/v3.10_20260915_raw_plans_cut_and_guiding_steps.md`](changelogs/v3.10_20260915_raw_plans_cut_and_guiding_steps.md)

- **Where the lines in the obstacle-avoidance figures come from**, settled by diffing against
  `/workspaces/aux_repo/dpcc/`: the 96 curves are recorded D3IL demonstrations, the constraint geometry
  is byte-identical to DPCC's config, the goal line and obstacles come from a helper identical to
  DPCC's. No model output, nothing invented — the evidence for Chapter 5's "1:1" claim.
- **Figure 6.3 rebuilt.** It was shipping a 3000x1000 six-panel diagnostic dashboard; the panel the
  section argues from was ~12 mm wide. Now cut to that panel, with the cut **declared** in
  `VENDORED_CROP` and applied by a new `plotting/prep/crop_vendored.py`; `make_figs.py` refuses to fall
  back to the uncut image. 🔴 Its caption claimed "every plan" — the renderer draws every fourth control
  step, up to four candidates each; corrected.
- **"Genuine steps" renamed to guiding steps** ($n\sidx{guide}$), §6.1.5 retitled *Steps at Which
  Projection Guides Sampling*; Chapter 8 and the appendix follow; recorded in the translation table.
  Chapter 4 is v2's and is written up in `handover/HANDOVER_20260915_for_v2_ch4_naming.md`, not edited.
- **`analysis_results_checkpoint/15-09` recorded** as the data source of record (committed, all three
  environments); no figure re-pointed, and its long-format schema noted as the obstacle to doing so.
- Fixed the bundler's figure accounting: it counted files, not figures, and warned about placeholders
  that were not placeholders. This build: 7 figures, 7 render, 0 placeholder.

---

## v3.9 — 2026-09-15 · quadrotor dataset values, tables that fit, all three geometries, avoiding figures → [`changelogs/v3.9_20260915_tables_geometries_avoiding_figures.md`](changelogs/v3.9_20260915_tables_geometries_avoiding_figures.md)

- Synced v2.15 → **v2.16**; §5.2.3 now holds the quadrotor demonstration values (`tab:uav-demos`), read
  from the collector code. *Cubic fillets* corrected to circular arcs of radius 0.45 m.
- **Table 6.1's overflow fixed**: non-wrapping `\multicolumn` notes about 20 cm wide. All 13 tables are
  now `tabularx` at `\linewidth`; notes moved into captions; `check.py` gained a width warning.
- **All three geometries** in every obstacle-avoidance table, and a new `tab:avoiding-geometries` in
  Chapter 5. 🔴 The claim that MeanFlow and consistency training are also *faster per episode* than flow
  matching holds only on top-left-hard; corrected in the text and in `tab:summary-models`.
- **Obstacle-avoidance figures built**: the scene with its 96 demonstrations, the three constraint
  geometries, and the control-steps-against-time grid. New extraction step and an SVG preview renderer in
  `DA_in_Paper/plotting`.

- **For v2:** `tools/check.py`'s new table-width guard reports one row in Chapter 4's notation table
  (`04_method.tex:106`): the `\multicolumn{3}{l}{\emph{deployment ... so the collisions are visible}}`
  subheading is about 18.6 cm of unwrapping text against a text width of about 14.7 cm, so it likely runs
  into the margin. A `p{}`/`tabularx` column or a shorter subheading fixes it.
- **For v2:** v3's tables use `tabularx`; `\usepackage{tabularx}` and the `L` column type live in
  `parts/00_preamble_v3.tex` and must be folded into `settings.tex` when the drafts merge.

---

## v3.8 — 2026-09-14 · figures and plotting moved to `Data_Analysis/DA_in_Paper` → [`changelogs/v3.8_20260914_figures_move_to_DA_in_Paper.md`](changelogs/v3.8_20260914_figures_move_to_DA_in_Paper.md)

- `DA_in_Paper` is now the official store: `analysis/INDEX.md` (17 thesis results, links checked),
  `plotting/` (moved from `v3/plots/` + `v3/tools/svg2pdf.sh`; output byte-identical), and
  `figures/{da,demo,env,schematic}/`.
- New `plotting/export_to_draft.py`: `v3/figures/` now holds only the four figures v3 includes, as
  copies with `EXPORTED.md`.
- Concise pointer note in `data_status/`.

---

## v3.7 — 2026-09-14 · Chapters 5–8 restructured by environment; Discussion handed to v4 → [`changelogs/v3.7_20260914_ch5-8_restructure.md`](changelogs/v3.7_20260914_ch5-8_restructure.md)

Full KP-by-KP record in the file above.
- **Synced first:** v2.9 → **v2.15**, inherited files merged one by one.
- **Ch 5:** rebuilt as five sections × the three environments, in study order (DPCC's benchmark
  unchanged → two environments built for this thesis). DPCC's published baseline row sits next to v3's
  reproduction. The caveats, the regime taxonomy and the global definition of "better" are deleted.
- **Ch 6:** one section per environment, plus 6.4 answering both comparisons per environment.
  Quadrotor follows corridor (new U16 results) → pillars → s-curve.
- **Ch 7:** headings only, now v4's; the old prose is in `handover/`.
- **Ch 8:** concise.
- **Shared notes:** 12 translation rows; `DRAFT_OWNERSHIP.md` revised.

- **For v2:** no label v2 references was removed. Where they now sit:
  - `sec:res:fewstep` → *Subsection* 6.1.2 *Step Budget*;
  - `sec:res:constraints:degenerate` → *Subsection* 6.1.5 *Genuine Steps*;
  - `sec:res:constraints` → *Subsection* 6.4.2 *Projection Methods*.

  `\autoref` will print "Subsection" for these.
- **For v2:** `sec:disc:interpretation` and `sec:disc:negative` are now headings with no text (Ch 7 is
  v4's). Ch 4 line ~506 cites `sec:disc:negative` for the consistency-training result; that result is
  in `sec:res:summary:models` (6.4.1).
- **For v2:** two of Ch 4's `\hole`s point at `sec:setup:protocol` and are now answered in its
  subsections: the sampling-prior question in `sec:setup:protocol:avoiding`, and which controller
  produced the UAV results in `sec:setup:protocol:uav`.
- **For v2:** v2.13 asked for one result sentence per contribution. Not yet — the alignment and
  quadrotor results rest on one training seed.

---

## v3.6 — 2026-09-14 · the default bundle builds only the new sections

**Asked for:** since v3 does not touch the v2 sections in normal editing, the default bundle should
build only the new sections and mark the v2 ones in the `.tex` as v2 — if that premise is right.

**The premise is right, with one qualification that the tool now enforces.** All seven inherited
files are byte-identical to `inherited/v2_base/` today. But `02_background` and `04_method` carry the
`merge` policy — v3 is *allowed* to edit them (the planned `sec:bg:fewstep` change) — so "untouched"
is true now, not guaranteed. The bundler therefore checks it on every build instead of assuming it.

### What changed in `bundle/make_bundle.py`

- **Default is now new-sections-only; `--full` builds everything.** Files are named `_new` / `_full`.
- **A v2 chapter collapses to a skeleton:** its numbered `\chapter`/`\section` headings with their
  labels, plus a grey `\fmpccvtwo` box — *"v2 SECTION — not built in this bundle"* — naming the file,
  its length and the v2 revision it was inherited at (read from `SYNC_STATE.json`). The macro is
  injected into the preamble only when something was collapsed.
- **Guard 1 — only v2 content chapters are candidates.** Ownership comes from `tools/sync_v2.py`'s
  `POLICY`, imported rather than restated, so the bundler cannot disagree with the sync tool.
  `parts/` is always inlined: the document does not compile without the preamble, the front matter
  (`\mainmatter`, the ToC) or the back matter (the acronym list).
- **Guard 2 — collapse only if byte-identical to the v2 baseline.** An edited chapter is inlined in
  full and the build prints `inlined IN FULL because v3 has edited them: …`.
- **`--verify` understands collapsed chapters:** no text to diff, so it compares the source's SHA-256
  against the one recorded in the banner.
- `--list`, `--prune` and the `--verify` default order bundles by modification time, since a `_full`
  and a `_new` built in the same second would otherwise sort by name.

### Why the skeleton keeps headings rather than dropping the chapters

Dropping Chapters 1–4 outright would renumber everything: *Experimental Setup* would become Chapter 1,
and all nine `\autoref`s from v3 chapters into v2 ones would print `??`. Every one of those nine is a
`sec:` reference, so keeping the numbered headings with their labels resolves all of them. Starred
headings move no counter and are dropped.

### Verified, not assumed

| check | result |
| :-- | :-- |
| new-sections-only build | 10 files inlined, 4 v2 chapters collapsed (1 821 lines not built); 2 347 lines against 4 108 |
| cross-references in the new-only build | **all resolve** |
| numbered headings, new vs full | **58 and 58, identical sequence** — so every chapter and section number matches |
| `--verify`, new-only bundle | 9 inlined byte-faithful + 4 collapsed unchanged |
| `--verify`, full bundle | 13 inlined byte-faithful |
| **guard test** — appended one line to `02_background.tex` | `--verify` on the old bundle: `DIFF … collapsed v2 section; the source has changed since the build`. Rebuild: `inlined IN FULL because v3 has edited them: chapters/02_background.tex`, the other three still collapsed. File restored, identical to baseline again |

**Use `--full` for anything that has to be the whole thesis** — the Overleaf upload of record, a
supervisor copy, and submission. **Not compiled.**

---

## v3.5 — 2026-09-12 · raw plan quality: the half of the headline that was missing

**Asked for:** the smoothness of the raw network output matters — a chaotic, low-quality plan can be
masked downstream by the IK/controller, while MeanFlow holds the same or better quality — with the
figures from `Report_20260819_MF_UNet` (and the consistency target, if applicable). Write it into the
tex if absent, and into a note so it is not lost. Skip either if already done.

**Checked first, as asked: neither was done.** The tex mentioned smoothness exactly once, in
`sec:disc:limitations`, and only to say no metric exists — the *argument* was absent. v2's tex: zero.
The notes: one clause inside the readiness ledger's list of limits. So both jobs were needed.

### Why this was a real gap, not a nice-to-have

Every number in \autoref{ch:results} is measured after a projector **and** a tracking controller,
and both are low-pass. Without this section a reader can reasonably ask whether the $31\times$ is
bought by degrading the plan. The evidence says the opposite, and it is the missing half of the
headline: **the MeanFlow engine is not merely cheaper at one evaluation — its plans are usable at one
evaluation, where the baseline's are not.**

### 1 · `sec:res:state` — a new subsection, *The plans themselves, before any projection*

With `fig:raw-plans`, two panels side by side at one network evaluation, projector off: MeanFlow's
replans form a tight ribbon around the executed path; the diffusion baseline's are a high-frequency
scribble across the workspace. Matched backbone size, matched budget, and on this cell matched
compute — 0.0097 against 0.0094\,s per step, three per cent apart.

Quantitatively: **6.0 unprojected violations against 28.0, a factor of 4.7**, at 58.0 against 56.5
steps. A second evaluation brings MeanFlow to 12.0 and tightens the ribbon, so the one-evaluation
panel is the low end of a trend, not a lucky sample.
\dataref{Report\_20260819\_MF\_UNet \S7}

**The load-bearing sentence:** the baseline's *commanded* traces are smooth, but they are smooth
because the controller integrates the scribble away, not because the plan was good.

The consistency target's raw arm is reported too — 20/20 goals at one evaluation against MeanFlow's
and flow matching's 17/20 and the baseline's 12/20 at its full budget — with a `\guard` stating that
this is three episodes on one seed, that **the margin does not survive projection**, and that the raw
arm therefore supports *the few-step objectives draw better plans than diffusion* and **not** an
ordering among them. That keeps it consistent with the standing negative result.
⚠️ The successes-only step basis is named, because the toolchain has two `n_steps` definitions and
mixing them is a real error.

### 2 · `sec:disc:threats` — the general form

New subsection, *Downstream metrics can hide the generator*: an evaluation that measures only what
survives the control stack measures **the stack's tolerance** as much as the planner's quality.
Reporting the unprojected arm costs one configuration per cell. This also bounds the aerial entry —
with a tracking error of 0.30--0.49\,m, that stack absorbs more than any scene leaves as clearance.

### 3 · Two smaller consequences

`sec:setup:metrics` now says *why* stage 1 of the funnel is evaluated on the unprojected arm, and
that the reason applies beyond the all-fail regime. `sec:disc:limitations` was upgraded from "no
smoothness metric is reported" to naming the specific missing measurement — **jerk, path length or
curvature over the saved plan files, which needs no new runs**, only a pass over plans already on
disk. It is now the cheapest outstanding measurement in the project.

### 4 · Vendored figures, with provenance enforced

These four panels come from the evaluation's own diagnostics and cannot be rebuilt from a CSV, so
they are **copied, not redrawn**. `plots/sources.py` gains a `VENDORED` registry — destination name,
source path, and a provenance string — and `make_figs.py` copies them and prints them in their own
table in `figures/MANIFEST.md`.

🚨 **The registry exists because of the v3.3 near-miss.** All four were checked against
`/workspaces/aux_repo/` and are ours; the rule is now explicit in the registry's own header, since
`figures/avoiding*.png` at the repo root are byte-identical to the baseline authors' copies.

### 5 · The note

`Auxiliary/NOTES_plan_quality_and_smoothness.md`, registered in `Auxiliary/README.md`: the claim, the
two tables, the ladder warning, the `n_steps`-basis trap, the owed measurement, the figure-provenance
rule, and the standing caveats (unprojected arms are unconstrained, so raw violation counts are never
a safety result; the raw gap and the projected gap are different sizes).

### 🔴 A bug I introduced, caught by the tooling rather than by reading

The new figure was written with `\fmpccgraphic` in the **source**. That macro is injected by
`bundle/make_bundle.py` and does not exist in a direct build of `thesis_v3.tex`, so the bundle would
have compiled while the split draft failed — the worst way for this to break, because the artefact
that gets uploaded still works.

`make_bundle.py --verify` flagged it: it un-rewrites `\fmpccgraphic` to `\includegraphics` before
diffing, so a source that already said `\fmpccgraphic` came back as a mismatch. `check.py` also
under-counted the figures, since it looks for `\includegraphics`. Both symptoms, one cause.

Fixed to `\includegraphics`, and **`check.py` now fails on any bundler-only macro appearing in a
source file** — tested by reintroducing it, which the guard catches by name and file.

### Checks

`tools/check.py`: 4 019 lines, 146 labels, **4 figure references**, 15 `\hole`, 20 `\guard`,
20 `\dataref`; all references and citations resolve; braces and environments balanced.
6 generated + 4 vendored figures. Both bundles rebuilt; `--verify` byte-faithful. **Not compiled.**

---

## v3.4 — 2026-09-12 · absorbed v2.7–v2.9, and applied their critique to v3's own chapters

**Asked for:** two jobs. (1) Sync v2's updates into v3 and move the version marker. (2) See whether
v2's new critiques — the *"bootstrapped target is jargon, not the scientific name"* one in
particular — also apply to the sections v3 wrote.

### Job 1 · Sync

Six files moved in v2 (`01_frontmatter`, `99_backmatter`, `01_introduction`, `02_background`,
`04_method`, `bibliography.bib`), all **fast-forwards** — v3 had edited none of them. Merged clean.
Absorbed v2.7, v2.8 and v2.9; baseline stamped at **v2.9**. 3 939 lines; 28 bibliography entries,
all cited.

🔴 **The stamp was wrong on the first attempt, and the tool was at fault.** `v2_version()` took the
*first* `## v2` heading, and v2's changelog is not strictly newest-first — a stale `v2.6` entry sits
above `v2.9`. v3 would have recorded itself as inheriting from v2.6 while in fact holding v2.9. The
parser now takes the **highest** version, and `status` prints a note when the changelog is out of
descending order. A version marker that can be silently wrong is worse than no marker.

One thing the merge could have broken and did not: v2.7 **deleted the `\acro{PID}` declaration**
(the controller is not a PID). No v3 chapter used `\ac{PID}`, so nothing dangled.

### Job 2 · v2's critique, applied to Chapters 5–8

**A · The coinage. `bootstrap` 21 → 0.** v2.8 established that *"bootstrapped target"* was this
draft's own coinage and hid the method: it is ordinary **consistency training** — the identity closed
by a **finite difference at an intermediate transport time, taken from a stop-gradient copy of the
network**, where MeanFlow closes it with an analytic derivative. v3 was still using the retired term
21 times across five chapters, in prose, in a results table, and in `tab:names`.

Renamed to **the consistency target** throughout, and the two places that state the *mechanism* were
rewritten to v2's account rather than merely relabelled — the old text explained the failure as a
*"field bias of order $(1-\alpha)$"*, which was my own formulation of it. It now reads: a
finite-difference target closed with the network's own stop-gradient output **inherits a fraction of
that network's error**, at a rate set by $\alpha$; the inherited term does not shrink as the interval
shrinks, so it is budget-independent, and the objective admits self-consistent fields that are wrong.
`plots/sources.py` renamed with it, so figure legends and prose cannot diverge.

**B · The controller's name.** v2.7 established that `CascadedPID` is a misnomer on three counts (no
integral term anywhere; the inner loop is a coordinate-free $SO(3)$ error, not a scalar loop; and
"cascaded PID" implies nested scalar loops). v3 said *"cascaded geometric controller"*; now
**"the cascaded geometric tracking controller"**, matching v2 exactly.

**C · The tone.** v2.9's finding — the prose had been arguing with an imagined examiner — applied to
v3's chapters at least as much. Swept to zero:

| pattern | before | after |
|---|---|---|
| `"worth stating / worth naming"` · `"It is worth"` | 4 | **0** |
| `"is not a defect"` · `"is not a post-hoc excuse"` | 3 | **0** |
| `"must not be"` · `"does not claim"` · `"makes no claim"` | 3 | **0** |
| `"is what makes"` (justifying form) | 8 | **0** |
| `"rather than"` | 58 | **48** (survivors are factual contrasts: *CPU-limited rather than accelerator-limited*) |

Representative rewrites, keeping every fact and dropping the defending:

- *"The taxonomy is not a post-hoc excuse for weak cells. It is diagnostic, it is measured rather than
  asserted … and it is used in three chapters."* → *"The regime is measured, not assigned: an
  all-pass scene is one where the violation counter reads zero on the unprojected arm."*
- *"Retiring it as a ranking scene is the correct action, and keeping it as the demonstration … is a
  stronger use of it than a failed ranking would have been"* → *"It is kept as the demonstration of
  where the control stack runs out."*
- *"That is not a defect of the experiment — both arms are reported — but it is a reason …"* → *"Both
  arms are reported, and the comparison is a pair rather than an ordering."*
- *"Two claims explicitly not made"* → *"Two results that carry no claim"*, with the defending
  sentences deleted and the numbers left to speak.
- One real redundancy surfaced by the sweep: the sentence explaining that the halfspaces are absent
  from the demonstrations appeared **twice**, 130 lines apart. One survives.

### Not done

v2 §2.4 still carries *"No mathematics is given for it in this draft"* for the consistency target,
so the deferral v3.0 logged as *"first item of v3.1"* is still open **in v2**, which owns that
chapter. v2.8 did add the mechanism to Engine 3, so the inconsistency is smaller than it was. The
one-way rule holds: that edit belongs in v2 and syncs down.

### Checks

`tools/check.py`: 14 files, 145 labels, all references and citations resolve, braces and environments
balanced, `bootstrap` 0. `sync_v2.py status`: clean at v2.9. Both bundles rebuilt;
`make_bundle.py --verify`: 13 inlined sources, byte-faithful. **Not compiled.**

---

## v3.3 — 2026-09-11 · the environments and their constraint sets, shown rather than only described

**Asked for:** the setup section should carry an image and a description of the environment *and* the
constraint set, for `avoiding`, `aligning` and each of the three aerial scenes, with placeholders
marked TODO where no image exists — plus a demonstration of the different constraints.

**Found:** the chapter described all three environments and showed nothing. It also under-described
the constraint sets, which are the more important half — the projector is inherited and held fixed,
so **the geometry is the part that had to be designed**, and it was getting one paragraph.

### 🔴 The finding that changes what may be shipped

`figures/avoiding.png`, `figures/avoiding_constraints.png` and `figures/avoiding_data.png` exist in
this repository and are exactly the panels the chapter needs. **All three are byte-identical to the
baseline authors' own copies in `aux_repo/dpcc/figures/`** (`cmp` verified); they arrived with the
"Add DPCC Code" commit. They are the baseline's artefacts, not ours, and shipping them as this
thesis's environment figures would be the visual form of the error the naming rules exist to prevent.

**The remedy is cheap and is recorded in the figure specs:** the script that produces them,
`scripts/visualize_data_constraints.py`, is in this repository and regenerates both panels from our
own configuration and dataset. It needs numpy/matplotlib and the vendored environment, so it is a
**cluster job**, not something runnable here. Re-running it makes the figure ours.

### Added

- **`\todofigure[height]{spec}`** in `parts/00_preamble_v3.tex`: a red-bordered box for a figure that
  is *planned but does not exist*. Distinct from `\fmpccgraphic`'s fallback, which means "this file
  should be here and is not". It carries the **specification** — what the panel must show and where
  the asset comes from — in the draft itself, so the gap is visible on every read and whoever renders
  it does not have to guess. Counted by `tools/check.py`.
- **Five specified figures** in `sec:setup:tasks`: the state-based benchmark, the vision-conditioned
  task (specified to show *the policy's own two camera streams*, not a third-person render, because
  what the network sees is what distinguishes the entry), the three aerial scenes at a common scale
  with the vehicle drawn to scale, the three halfspace geometries with the tightened margin drawn as
  a ring, and the scene-clearance-against-tracking-error panel.
- **A proper `Constraint sets` subsection**, replacing the three-sentence one: `tab:constraint-families`
  giving the five families and marking which three are ours; the ablation rule; tightening written up
  as an **experimental factor** with the measured claim that it is a larger lever than the choice of
  constraint arm; and the honest-geometry defect — 0.000/0.060/0.120\,m of clearance against a
  0.30--0.49\,m tracking error — with the gate that could not tell "safe" from "exactly on the edge".

### Two holes this pass makes visible rather than fixes

- 🔴 **The vision-conditioned feasible set is undefined in prose** — not here and not in the
  methodology sources — although every table on that entry is measured against it. Now an explicit
  `\hole` that names it as **the largest single gap in the methodology of this thesis**.
- The numeric tightening margin, in metres, for both embodiments, and what the projector-threshold
  sweep varies. Configuration values, not derived quantities.

### Checks

`tools/check.py`: 3 871 lines, 143 labels, all references resolve, all citations resolve, braces and
environments balanced. Drafting macros now 14 `\hole`, 5 `\provisional`, 18 `\guard`, 35
`\srcnote`, 18 `\dataref`, **5 `\todofigure`**.

🔴 **A second tooling bug, caught by the tooling:** `make_bundle.py --verify` undid the
`\\includegraphics` → `\\fmpccgraphic` rewrite on whole lines, while the rewrite itself only touches
the part of a line TeX executes. The moment this pass added a *comment* mentioning `\\fmpccgraphic`
by name, verify corrupted that comment and reported a mismatch against an untouched file. The two
now share one `split_comment` helper, so the inverse is a real inverse.

🔴 **A counter bug fixed in passing:** `check.py` counted drafting macros with a bare `\\macro\{`
pattern, which does not match `\todofigure[0.3\textwidth]{...}`. It reported **zero** planned
figures while five existed — a pre-submission check that silently hides exactly what it exists to
find. Now matches the optional-argument form.

---

## v3.2 — 2026-09-11 · template-conformance audit

**Asked for:** sanity check that the TUM template is still preserved in v3.

**Answer: yes.** `Template_DONT_CHANGE/` is untouched (clean `git status`, last touched by
`c721f7d4`), and the `\ifstandalone`/`\else` merge bridge survives verbatim in both
`parts/00_preamble.tex` and `parts/01_frontmatter.tex` — v3 inherits them byte-for-byte and
`sync_v2.py status` confirms no drift. v3 adds **no packages**.

### One real divergence found, and fixed

`settings.tex:44-52` capitalises the `\autoref` names through babel's language hook. The standalone
branch never loads `settings.tex`, so a standalone build fell back to hyperref's own lowercase
defaults: **"section 5.2" where the merged template build says "Section 5.2"**, across 133 `\autoref`
calls. Mirrored into `parts/00_preamble_v3.tex` using the *same* `\addto\extrasamerican` hook — a
plain `\providecommand` in the preamble would be overwritten when babel selects the language at
`\begin{document}` — and guarded by `\ifstandalone` so the merged build still takes `settings.tex`'s
copy and it never fires twice.

### 🔴 Known, inherited from v2, NOT fixed: the standalone build is not PDF/A

`settings.tex` loads `\usepackage[a-2u]{pdfx}` and the template ships `main.xmpdata`; the standalone
branch loads neither. So **a bundle built here is not PDF/A-2u**, and if the submission requires it
the final PDF must come from the template path, not from the Overleaf bundle. This is a v2 divergence
that v2's own README does not list among its declared ones. Left as-is deliberately: adding `pdfx`
would change the standalone build substantially and risks breaking the Overleaf compile that v3.1
exists to enable. Flagged rather than silently carried.

Also absent from the standalone branch, and harmless because the draft uses none of them: `listings`,
`lstautogobble`, `scrhack`, `tikz`, `pgfplots`, `pgfplotstable`, `caption`, `ifthen`, `pagecolor`,
the TUM corporate colours and the `\BeforeTOCHead` PDF bookmark. All arrive with `settings.tex` on
merge.

### The merge path, restated for the split layout

v2's bridge assumed one file. v3 is fourteen — but `bundle/make_bundle.py` reconstructs exactly that
single-file shape, so the bridge is intact and in fact simpler to execute than before: **flatten, set
`\standalonefalse`, drop the bundle at the root of a template copy.** Its `\else` branch then pulls
`settings.tex`, `pages/cover`, `pages/title` and the rest from the template, which is why those
`\input`s are deliberately left unresolved in the bundle. `parts/00_preamble_v3.tex` folds into
`settings.tex` at that point, together with v2's own `amsmath`/`amssymb`/`amsthm` addition.

### Checks

`tools/check.py` passes; `make_bundle.py --verify`: 13 inlined sources, byte-faithful. Both bundles
rebuilt.

---

## v3.1 — 2026-09-10 · a flattened build, because the split draft would not compile

**Asked for:** the split draft cannot be compiled on a remote Overleaf. Write a tool — not an
LLM-driven copy-paste — that aggregates the parts into one full `.tex`, timestamped, in a subfolder
under `v3/`, with the tool living in that subfolder. Write it and run it.

**Added:** `bundle/make_bundle.py`, plus `bundle/README.md` and `bundle/BUNDLE_LOG.md`.

### What it does

Recursively inlines every `\input` whose target exists, wrapping each in `BEGIN`/`END` banners
carrying the source path and its SHA-256 prefix, and writes
`bundle/thesis_v3_<YYYYMMDD_HHMMSS>.tex` plus a `.zip` holding that file, both bibliography
resources and `figures/` — an Overleaf upload in one artefact.

**An `\input` whose target does not exist is left verbatim, and that is correct rather than a
fallback.** The inherited preamble and front matter carry `\input{settings}` and
`\input{pages/cover}` inside the `\ifstandalone … \else` branch, for the day the draft is merged
into the TUM template; `\standalonetrue` is set, so LaTeX never reads them. Six such lines survive
in the bundle and all six are in dead branches — checked, not assumed.

### The figure problem, and how it is handled

The figures are SVG and `\includegraphics` cannot read SVG; no converter exists in this container.
The tool rewrites every `\includegraphics` to `\fmpccgraphic` and injects that macro, resolving
`.pdf` → `.png` → *(mode)* → placeholder. Default mode draws a framed box naming the missing file, so
**the document always builds**; `--svg-package` adds `\usepackage{svg}` for hosts with Inkscape,
Overleaf among them. Opt-in rather than default because it is a property of the build host, not of
the document. A real `.pdf`/`.png` wins in every mode, so running `tools/svg2pdf.sh` and rebuilding
upgrades the figures with no source change.

🔴 **Ordering constraint, recorded because it is easy to get backwards:** the rewrite must run
*before* the shim is injected. The shim's own body calls `\includegraphics`; injecting first would
rewrite those calls too and make `\fmpccgraphic` infinitely recursive.

### Verification, which is the part that matters

`--verify` extracts every inlined source back out of a bundle, undoes the one transformation the tool
applies, and diffs against the tree. **All 13 inlined sources round-trip byte-for-byte.** Tested in
both directions: appending one line to a chapter made it report `DIFF … first difference at source
line 127`, and reverting restored `byte-faithful throughout`. Run against an *older* bundle it
answers a different and equally useful question — *was this built from what is on disk now?*

The build path additionally refuses to write at all unless: no resolvable `\input` remains, no
content was lost, braces balance, environments balance, and `\documentclass`, `\begin{document}`
and `\end{document}` each appear exactly once.

### Two bugs found by running it rather than by reading it

- 🔴 **The placeholder would have failed on exactly the filenames it exists to print.** It set the
  missing name in `\texttt{figures/#1}`, and every generated figure name contains underscores
  (`fig_avoiding_k_ladder`); a bare `_` in text mode is a subscript ⇒ *"Missing $ inserted"*. Now
  `\texttt{\detokenize{...}}`, with a comment saying why it is not decorative.
- **Two runs in the same second collided** and the second silently overwrote the first, defeating the
  point of stamping them. A bundle is now never overwritten: the tool suffixes instead.

### Result

14 source files, 3 648 source lines → **3 719 output lines, 228 KB**, 90 KB zipped. Two bundles
built, one per figure mode. **Still not compiled** — there is no TeX toolchain here, the tool checks
structure rather than typesetting, and it says so on every run. The first real build is the author's.

---

## v3.0 — 2026-09-10 · the workspace, the figure pipeline, and the experiments

**Asked for:** branch a v3 off v2 that can be worked on **in parallel** with it, so that a later v2
change can be identified and pulled down; give it a changelog like v2's; give it figure code kept as
a template that takes current data and emits current plots, so that when data updates the plots
update; and write the experiments — the state-based `avoiding` entry in full, the bones filled, and
what can be written for the other two entries.

**Branched from** v2 at `v2.6 — 2026-09-09 · the compute environment, and the first prose written into
Chapter 5`, recorded in `inherited/SYNC_STATE.json` together with the file digests.

### 1 · The parallel-work machinery

v2 is one 2 151-line file, which makes parallel work impossible: any v2 edit conflicts with any v3
edit. v3 is therefore v2 **split by chapter**, with a merge base and a tool.

- **`tools/split_v2.py`** splits `thesis_v2.tex` on *content markers* — `\begin{document}`,
  `\mainmatter{}`, each unstarred `\chapter{}`, `\appendix{}`, `\addchap{Abbreviations}` — never on
  line numbers, so it keeps working as v2 grows. It asserts that every input line lands in exactly
  one output file, and it **refuses to run** if v2 adds a chapter whose label the manifest does not
  know, rather than inventing a filename.
- **`inherited/v2_base/`** is the merge base: v2 as it stood at the last sync. **`tools/sync_v2.py`**
  re-splits the current v2 on demand and reports, per file, whether v2 moved and whether v3 moved;
  `merge` runs `git merge-file` with the base and advances the baseline **only on a clean merge**, so
  a conflicted file can simply be redone.
- **Per-file policy** in `inherited/MANIFEST.md`: `inherit` (v3 never edits — fast-forwards),
  `merge` (both edit), `own` (v3 wrote it; the baseline exists so drift is *visible*, not silent).
- **The dependency is one-way.** `tools/` reads `../v2/` and never writes to it. A fix belonging in
  both drafts is made in v2 and synced down.

**Tested end to end, not merely written.** In a scratch copy: v2 edited an `inherit` chapter and a
`merge` chapter; v3 independently edited the same `merge` chapter elsewhere. `status` classified both
correctly, `merge` fast-forwarded the first and three-way-merged the second with **both** edits
present and **zero** conflict markers, and re-stamped the baseline.

**No temporary aggregation sections were needed.** The split is clean at chapter granularity, so
every chapter has exactly one owner. `sec:bg:fewstep` is the single place where v3 would write inside
an inherited chapter, which is why `02_background.tex` carries the `merge` policy — and v3.0 did not
in fact write there (see *Not done*).

### 2 · The bibliography, split for the same reason 🆕

Raised mid-pass by the author: the `.bib` needs the same treatment, and v3 should record *which*
version of v2's bibliography it holds.

- `bibliography.bib` stays **inherited byte-for-byte**; v3 never appends to it. Its policy is
  therefore `inherit`, not `merge`, so it is a permanent fast-forward.
- `bibliography_v3.bib` is new and holds v3's own entries. `parts/00_preamble_v3.tex` registers it
  with a second `\addbibresource`, guarded by `\ifstandalone` for the same reason v2 guards its own:
  the template's `settings.tex` already registers a resource in the merged build.
- **Why not one file:** two live drafts appending at the end of the same file is the worst conflict
  shape there is, and it would recur on every pass.
- **The version stamp:** `sync_v2.py` reads the newest `## v2.x` heading from v2's changelog and
  writes it, with SHA-256 prefixes of both v2 files, into `inherited/SYNC_STATE.json`. `status`
  prints it, and `stamp` **refuses** to record a version while any baseline file differs from v2 —
  so the stamp cannot become a lie.
- `tools/check.py` reports a key defined in *both* `.bib` files. Biber would too; catching it here is
  cheaper.

### 3 · The figure pipeline

`plots/`, stdlib-only, plus `figures/` and a generated manifest.

- **`sources.py` is the only file containing a path.** Corpus entries mirror `DATASTATUS §10` row for
  row and carry the *protocol* and the readiness *grade*, not just the directory. **When data lands,
  editing this one file rebuilds every figure with its subtitle and provenance updated.**
- **`fmpcc_svg.py`** is carried over from
  `Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/make_figs.py`, so the thesis figures
  are drawn by the same code as the reports of record. No matplotlib: this container has no
  scientific Python stack, and a figure script that could not run where the writing happens would
  guarantee that figures drift from text.
- **`figures.py`** — one builder per figure, each returning `None` when its corpus is absent so a
  partial checkout builds what it can. **`make_figs.py`** writes `figures/MANIFEST.md` recording
  which corpus each figure came from, so a stale figure is a visible fact.
- **Six figures built** from data on disk: four cost/quality frontiers (aggregate + three
  geometries), the step-budget ladder, and the constraint-arm cost crossover.
- SVG is the committed artefact; `tools/svg2pdf.sh` converts where the document is built. The
  draft uses extension-less `\includegraphics`, so it needs no change either way.

#### 🔴 The aggregation rule, and a contradiction it would have caused

Entry-1 cells must be aggregated **per geometry first, then across geometries** — never as a flat
mean over `(seed × geometry)` cells. The two differ whenever a geometry carries a different seed
count, which is exactly the case for the pinned baseline, whose `both-hard` cell is seed-6 only.

Geometry-mean reproduces the DAs of record **exactly**: baseline 0.983 / 69.0 steps / 0.5635 s
against DA_20260827 §10.1's 0.983 / 69.0 / 564 ms; MeanFlow K1 0.993 / 61.0 / 0.0181 against its
0.993 / 61.0 / 18.1 ms; and the baseline at the published protocol 1.000 / 70.1 / 0.5534 against
DA_20260906 §3's 1.000 / 70.13 / 0.5534. **A flat cell-mean gives 0.977 / 72.5 / 544 ms** and would
have put the figures in silent contradiction with the text. `geometry_mean` takes the geometry column
index as a *required* argument, because inferring it from the key length picked the wrong column for
one of the two loaders and produced an empty figure rather than a wrong one only by luck.

### 4 · Chapter 5 — Experimental Setup

The compute-environment subsection v2 wrote is preserved **verbatim**. Around it:

- **Three entries, organised by what each adds** — the state-based benchmark, then a
  vision-conditioned task that adds harder control *and* harder perception, then an aerial embodiment.
- **Why the aerial entry is a different problem, not a harder one**: fully actuated and
  quasi-statically stable against underactuated and open-loop unstable; kinematic against dynamic
  feasibility; one loop at planner rate against a cascaded multi-rate one.
- **The regime taxonomy** — all-pass / discriminating / all-fail — defined once, measured (an
  all-pass scene is one where the unprojected arm's violation counter reads zero), and used by three
  chapters.
- **The pinned baseline** as `tab:target`, with the coverage asymmetry stated: the transport rows are
  300 episodes and the baseline 220, because its hardest geometry has one seed. 🔴 **This corrects a
  summary line in DA_20260827 §10.1**, which says 300 a side; the batch has 11 baseline cells, not 15.
- **`\autoref{def:improvement}`** — Pareto dominance with quality as a gate rather than a third
  objective — plus the funnel for all-fail regimes, and the explicit statement that the aerial
  per-step budget is never a pass/fail criterion.
- **The two protocol tiers**, with the measured cost of the difference as `tab:tiers`: a 1.00 at ten
  episodes means "≥ 0.90 at ±0.10", untightened mid-range cells move by up to 0.35 in both
  directions, tightened ones are stable to ~0.08. The consequence is a methodological point, not a
  formality: **the published protocol's resolution is coarser than most differences the field reports
  on this benchmark.**

### 5 · Chapter 6 — Results

**Entry 1 in full.** `tab:state-headline`: MeanFlow at a single network evaluation against the pinned
baseline at twenty — 0.993 / 61.0 steps / 18.1 ms against 0.983 / 69.0 / 563.5, architecture-matched
at 4.0 M. The compute margin and the quality margin are read differently *in the text*: 31× is far
outside noise, 0.993 against 0.983 is *inside* the across-seed standard error and is therefore the
**gate** of `def:improvement`, not its margin.

- **The flow-matching result is written as a cost claim**, `tab:state-fm`, with the matched-budget
  control that decomposes the 21× into ≈1.4× of engine and ≈15× of budget. The quality version is
  refuted on Entry 2 and is withdrawn in `sec:disc:negative`.
- **The low-budget pair** is a *pair*: three reasons in the text forbid the ladder — p = 0.231, a
  wash over all six rules, and **two floors trained with the better quoted**, a selection effect a
  single seed cannot absorb.
- **A defect reported as one:** the cumulative-cost rule stalls on the flagship (98.0 steps at K2,
  72.0 at K1, against ~61 elsewhere) and on the bootstrapped target, but not on the baseline —
  so the sign tracks the projection-cost landscape, not the architecture.
- **`sec:res:fewstep`** carries `fig:k-ladder`, with the baseline's published-protocol low-budget
  cells drawn **dashed and hollow** as a separate series. Mixing 10-episode and 100-episode cells
  into one solid line would have been the most misleading thing that figure could do.
- **`sec:res:constraints`** states the citable claim **per entry and never pools**, and
  `sec:res:constraints:degenerate` gives the genuine-step condition, confirmed by the shipped code at
  run time (solve counts 1 : 1.51 : 2.52 against a predicted 1 : 1.5 : 2.5).
- **Entries 2 and 3 written to grade.** Entry 2's engine result is decisive and is written as such;
  its projector claim selects **one** operating point (K = 10, 0/9, p = 0.0039) and explicitly
  refuses the K = 20 safety pairing, which is one discordant rollout at p = 1.0000. One table is
  **held**. Entry 3's chapter is written and its **ranking is not banked** — every ordering sentence
  carries seed 6, n = 10 — while the embodiment argument, the regime taxonomy and the
  zero-unsafe-rollouts-in-210 result stand independently of it.

### 6 · Chapters 7, 8 and the appendix

Discussion written in full: where Goal A and Goal B pull against each other and why the budget is
really a constraint-enforcement decision; the bootstrapped target as a **negative result with a
derived mechanism** and with **both α floors named**; the withdrawn quality claim; task saturation as
a general threat to validity, argued from this work's own corridor scene, where the projector spends
155–202 ms/step removing violations that never existed.

Conclusion written with the RQ answers settled at the strength the evidence supports; headline
numbers and the abstract left for last, on purpose. The appendix gains `tab:names` — the **reverse**
map from thesis names to the tokens in released artefacts, which `NAMING §8` records as owed and
which the results chapters make urgent, since they use mechanism names exclusively — and
`tab:corpora`.

### Not done, on purpose

- 🔴 **`sec:bg:fewstep` still says the bootstrapped target's mathematics is "deliberately deferred".**
  `DATASTATUS §7` asks for that to be reversed now that the fallback has fired and the mechanism *is*
  the contribution. v3.0 leaves Chapter 2 byte-identical and writes the mechanism where the result is
  (`sec:disc:negative`). **First item of v3.1.** Adding a pointer instead of the derivation would have
  been worse than leaving it.
- **v2's `\hole` in `sec:method:engine` about the sampling prior is answered from the protocol side
  only**, as a `\provisional`: the flow-matching numbers were produced by the shipped sampler and
  were *not* re-run at matched scale, the bias direction is unestablished, no result is attributed to
  it, and the flagship MeanFlow rows are unaffected. `04_method.tex` is left byte-identical so it
  stays a clean fast-forward.
- **Entry-2 and Entry-3 figures**, the abstract, the headline numbers, and the cross-entry synthesis
  table. Reasons in `README.md`.
- **Dataset counts and splits** are a `\hole` in `sec:setup:data`. Book-keeping, not blocked — read
  them off the dataset files rather than a dev log.

### Verification done in this pass

- Every Entry-1 number was **recomputed from the batch CSVs** and matched against the DA of record
  before being written: the target, the flagship at K1/K2, the per-geometry flow-matching cells
  (1.00/1.00 vs 1.00/0.95; 65.5/71.6 vs 70.0/77.6 steps; 1.8/1.9 vs 39.1/40.2 s/ep), the
  published-protocol budget ladder (0.667 → 1.000 → 1.000), and the cumulative-cost stall (98.0).
- Entry-2 and Entry-3 headline numbers were **spot-checked against their closure DAs directly**, not
  taken from the readiness ledger's transcription: the −0.3744 m / 0/10 / p = 0.0020 pairing, the
  −324.96 ms / 0/9 / p = 0.0039 latency sweep, the 0.635 / 0.359 / 0.235 aerial means, and the
  0-of-210 against 43-of-270 safety tally. All four reproduce.

### Mechanical checks

`python3 tools/check.py`: 14 files, 3 634 lines, 137 labels, 26 distinct citations of 26 bib
entries, 2 figure references — **dangling references: none · duplicate labels: none · missing bib
keys: none · keys in both `.bib` files: none · unbalanced environments: none · brace delta: 0 in
every file · `\includegraphics` targets: all present**. Drafting macros: 12 `\hole`, 5
`\provisional`, 17 `\guard`, 35 `\srcnote`, 18 `\dataref`. `python3 tools/sync_v2.py status`: clean —
no inherited file has moved. **Still not compiled**, and `check.py` says so on every run.

Two bugs were found and fixed in the tooling by these checks rather than by inspection: a
`[^%]*` character class in the input scanner matched across newlines under `re.MULTILINE` and
silently loaded only 5 of 13 files; and the brace counter subtracted escaped braces once instead of
excluding them from both counts, reporting a −2 delta per `\{…\}` pair on two untouched inherited
chapters.
