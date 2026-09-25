# Claude's answer to `THESIS_FIRST_READING_REVIEW.md`

**Written:** 2026-09-25 13:10 · **By:** the Orchestra chat (Claude Fable 5.1, Claude Code), job O002 · **Answers:** `feedback/THESIS_FIRST_READING_REVIEW.md` (412 lines, written by a reviewing agent that read only the compiled PDF and this release's LaTeX)
**Release reviewed:** `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE` · PDF compiled on Overleaf 2026-09-25 14:09, 185 pages
**Draft revisions:** v2 **v2.28** (2026-09-24) · v3 **v3.100b** (2026-09-25) · v4 **v4.2** (2026-09-25) — the revisions the release was built on, and still the live ones at the time of writing (checked with `Orchestra/tools/orchestra.py status`)
**Status:** an answer, nothing applied. The author ticks items in §9 (four groups: format bugs · tone · content · runs); the Orchestra then applies the ticked items in `Working_Space/v5` (the aggregate, since job O003) and hands the DA its items as a note.
**Revised:** 2026-09-25 14:48 (job O005) — §9 re-cut into the four groups on the author's instruction ("too much process; list the pure writing/format bugs first, then tone/style/redundant text, then content, then runs/data; the current version is highly curated — caution on factual changes"); §4 W6 corrected (`07:63` already has the exact form). §0–§8 are unchanged evidence.
**Applied:** 2026-09-25 15:01 (job O006) — §9 group A applied in `Working_Space/v5` → **v5.2** (`v5/changelogs/v5.2_20260925_review_tier_a_format_bugs.md`); a *status* column marks each A row. No release built.
**Revised (2):** 2026-09-25 15:35 (job O009) — §9 B rebuilt by risk, size and target (B-i minor · B-ii giant · B-iii no target, with word counts over the released text); §9 C tagged 🔴 dangerous / 🟠 careful / 🟢 minor, 6 · 9 · 9.
**Applied:** 2026-09-25 15:20 (job O007) — A6 and A9 fixed at the DA source and re-exported → **v5.3** (`v5/changelogs/v5.3_20260925_review_tier_a_figures_at_source.md`).
**Released:** 2026-09-25 15:23 (job O008) — `RELEASE/output/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A/`, the PURE BUG FIX build of group A (v5.2 + v5.3), marked so in `RELEASE/CHANGELOG.md`; A11 (metadata) still prints as TODO in it.

## How every finding was checked

The reviewer knew nothing of the code, the data or the earlier audits, so each of its points was verified before it earned a verdict:

1. **The release LaTeX** — `latex/chapters/*.tex`, `latex/pages/abstract.tex`, `latex/main.tex`. Line numbers below are of the release copy (the live drafts differ by a few lines because the release strips comments and drafting macros).
2. **The compiled pages** — the reviewer's own page images in `feedback/_review_work/page_NNN.png` (PDF page = printed page + 8) and the PDF text, for every layout claim.
3. **The live drafts and their changelogs** (`Working_Space/v2`, `v3`, `v4`), for what was already decided and why.
4. **The code that produced the numbers** — `FM_v3_uav_test/eval_fm_uav.py`, `fm_visual_aligning_test/eval_fm_visual_aligning.py`, `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py`, `flow_matcher_v3_meanflow/sampling/{projection,hardflow_projection}.py`, `config/uav_projection.yaml`, `uav_env_test/flight_controller.py`, the vendored D3IL dataset code.
5. **The analyses of record** — `Data_Analysis/DA_in_Paper/analysis/`, and `logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md` for the one number that turned out to come from nowhere else.

Nothing was re-run and nothing was compiled; the `??` diagnosis is from the LaTeX rules, not from a rebuild.

**Verdicts:** ✅ **confirmed** — the reviewer is right, a concrete fix follows · 🟡 **partly** — right in substance, but the thesis already carries part of it or the fix is a scoping choice the author makes · ❌ **not supported / declined** — the critique is wrong on the facts, or it runs against a decision the author has taken (named) · ⚪ **needs data or a run** — cannot be settled from the release; a DA re-aggregation or a cluster job would.

---

## 0. The short version

**Tally.** Content C1–C12: 2 ✅ · 10 🟡 (two of them with a ❌ half: C9's remedy, C11's "fixed before evaluation"). Mathematics M1–M9: 6 ✅ · 3 🟡 (M4 with a ❌ half). Consistency rows (14): 12 ✅ · 1 🟡 · 1 ❌. Writing W1–W6: the author's call, with one factual item (W6 ✅). Layout (§6–7): 9 confirmed on the pages, the rest are judgement calls the author weighs.

**What is simply wrong in the PDF and must be fixed before any further reading round (P1):**

1. **Three `(??)` equation references** (printed pp. 29, 30, 31). Cause found: `04_method.tex:648` carries **two `\label`s on one `equation`** (`eq:bg:fm:loss` and `eq:method:engine:fm`); amsmath keeps only the last label of a display and drops the first with the warning *Multiple \label's: label will be lost*. Every `\eqref{eq:bg:fm:loss}` (lines 671, 687, 769) prints `??`; `\eqref{eq:method:engine:fm}` (line 827) resolves. The release tool's label check cannot see this (both labels exist in the source). **v2:** one label on that equation; point the three references at it. **RELEASE tool:** warn on two `\label`s inside one display.
2. **An internal audit identifier is printed in the thesis** — `06_results.tex:772` ends with `(CLOSURE\_20260907 R.4)`, and the counts it supports ("succeeds in seven … in two") come from `logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md` §R.4 ("on contexts-ever-solved at K = 2, mf solves 7 of 10 and af 2 of 10"), a **2026-09-07 analysis on the 14 102 rollouts of that time**, written into v3 at v3.15 (2026-09-16) and never re-derived after the corpora of record moved at v3.41 (2026-09-19). No table of the thesis carries these counts. **v3:** delete the sentence and the identifier, or have the DA recompute the per-cell "ends in position" counts (the metric is declared at `05_setup.tex:637–639` but has no table column) and print them.
3. **Table 6.12 overflows the page** (printed p. 122): its last row sits on the page number. **v3.**
4. **Figure 6.7's caption runs over the page number** (p. 131); the figure block is taller than the text height. **v3** (and DA for the empty sixth panel).
5. **Figures 5.8 and 5.11: the captions cross the footer rule** (pp. 61, 67). **Figure 5.10's subtitle and Figure 5.11's third legend line are cut off inside the image** (pp. 64, 67). **v3** for the float heights, **DA** for the two exports.
6. **Figure 6.2's caption says "triangles temporal consistency"; the figure and its legend use squares** (p. 93). **v3.**
7. **Nine sentences contradict the tables they stand beside** (the reviewer's consistency rows 2–10, every one confirmed below: "every model … the same checkpoint", "every comparison … five seeds", "no unprojected row exceeds 0.100", "15.5 to 19.0", "ten" solves, "four of the ten", "closest at K = 20", "do not move the box", "the one place in the thesis"). **All v3.**
8. **Table 6.6's caption promises "mean ± sample standard deviation" for ms/step; the rows carry means only.** **v3.**

*Status 2026-09-25 15:01 (v5.2, job O006): items 1, 3, 4, 6, 7, 8 fixed in v5; item 2's identifier removed (the sentence itself is the author's call, §9 Ci-5); item 5's float heights fixed in v5.2 and its two figures re-exported from the fixed DA source in v5.3 (15:20, job O007). The "v2 / v3" owners named in §0–§8 are the legacy sources; every fix went into `Working_Space/v5`.*

**The two technical points that hold (P2):**

- **M1 — the 0.31 m offset is not the vehicle's extent in every direction.** With rotors at (±0.14, ±0.18) m and radius 0.13 m the reach is 0.31 m across, 0.27 m along, **0.356 m on the diagonal and 0.358 m enclosing**. Checked against `config/uav_projection.yaml`: the corridor and s-curve walls are axis-parallel (0.31 exact), the tilt plane and the hump are approached through a normal with a large z-component (0.31 is conservative there, the vehicle being thin), but **every keep-out disk** (`sphere_outside` on `['x','y']`: the four wall-end caps of the corridor, the corner balls of the s-curve) can be approached from any horizontal direction at the zero yaw the controller holds — there the offset under-covers by **up to 4.8 cm**. The violation scorer uses the same 0.31 m, so "violation-free" is a statement about the 0.31 m surfaces; physical contact is scored separately (the wall-contact fraction). UAV-pillars already uses the radial 0.36 m (`05_setup.tex:577`), so the author knows the number. **Qualify in v2 §4.6.3, v3 §5.6.3, v4 Fig. A.1 caption; optional DA re-score of the stored flights against the disks at 0.36 m.**
- **M2 — the mismatch is the change of the tracking error, not the tracking error.** All three evaluation loops accumulate the setpoint: `FM_v3_uav_test/eval_fm_uav.py:1155` (`p_des = p_des + action`), `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:716` (`next_pos_des = action + obs[:2]`, the commanded position), `fm_visual_aligning_test/eval_fm_visual_aligning.py:600` ("des_c_pos accumulates"); the demonstrations define the action the same way (`d3il/environments/dataset/avoiding_dataset.py:55–60`). Under that update, the measured-channel mismatch of `eq:method:dpcc:euler` is `w_t = e_t − e_{t+1}`. The sentence at `04_method.tex:351–354` ("the residual is precisely the w_t") is wrong by exactly the reviewer's derivation. **v2: one sentence.**

**Where the reviewer is wrong, or the author has already decided otherwise (❌):** C9's uncertainty intervals and paired tests (author's rule: counts only, no tests); §5's "move Discussion before Conclusion" (author, v4.1: Conclusion first); row 13's Appendix B heading flag (author, v4.2: deliberately visible in the release); C2's "restore the original completion metric" as stated (the thesis explains at `05_setup.tex:793–803` why D3IL's success rate does not transfer, and it does declare a position-only count); M4's "give explicit index ranges" (they are given, `04_method.tex:1006–1010`); C11's implication that the finish line was chosen after the fact without saying so (the thesis says at `05_setup.tex:658–664` why the goal line was replaced and prints the goal-point counts where they differ).

---

## 1. Content, evidence and interpretation (C1–C12)

### C1 · Alignment success has incompatible meanings — ✅ confirmed (the worst item of the review)

- **Checked.** `06_results.tex:770–772`: "Counting the contexts in which any configuration reaches the target, analytic average-velocity matching succeeds in seven and consistency-interpolated average-velocity matching in two (CLOSURE\_20260907 R.4)." `06_results.tex:1116–1119`: "No context ends in position, within the 1.8 cm of the task's own test" — for the selected MeanFM K = 20 configurations. The two are not the same population: the seven/two are "contexts ever solved by *any* K = 2 variant" in the Gen14 closure of 2026-09-07 (its §R.4 adds that nine of the consistency model's eleven successes are one context re-solved across variants). That closure predates the corpora of record (v3.41, 2026-09-19); the sentence entered v3 at v3.15 and was never re-derived. No thesis table shows these counts; the audit identifier is dev residue that the release tool cannot recognise (it is plain text, not a drafting macro).
- **Verdict.** ✅. The reviewer's reading — "no reader-accessible evidence establishes the distinction" — is exactly right, and the number's provenance fails the thesis's own rule that every number names its evidence of record.
- **Fix.** v3: delete both sentences (the "level result at K = 2 is in distance only" point can stand on Table 6.5's *unmoved* column without them), or ⚪ ask the DA for the per-cell count of contexts ending within 0.018 m (the eval records `success` = D3IL's strict test and `success_relaxed` = position-only, per rollout: `fm_visual_aligning_test/eval_fm_visual_aligning.py:1418–1463`) and add that column to Tables 6.5–6.8. Also replace "the model that solves the task does so at twenty" (`06:945`) with a distance statement — the reviewer is right that nothing "solves" the task by the task's test.

### C2 · Restore the original completion metric — 🟡 partly

- **Checked.** `05_setup.tex:793–803` explains why D3IL's success rate (0.018 m *and* 8.6° at once) is not used: a different network, target and output. `05_setup.tex:637–639` *does* declare "whether the box ends in position: within 0.018 m … (a count of contexts); the orientation half of that test is not used" as a metric — but no table has that column; the count appears once, in prose (`06:1116–1119`), for two configurations.
- **Verdict.** 🟡. The reviewer's demand to *report* the original success is not answered by the explanation for not *selecting on* it. The data exist (strict and position-only flags per rollout, see C1). Whether the strict count is zero for every reported cell is not established in the thesis — it is implied for the selected configurations only.
- **Fix.** ⚪ DA: per-cell counts (strict, position-only) for every aligning cell of record, from the stored rollouts, no runs. v3: one column in Tables 6.5 and 6.8 (or a sentence per table if the counts are all zero), and "solves / delivered" reserved for that test. The author decides whether orientation error is worth printing (it is recorded, since the strict flag needs it).

### C3 · Evaluation on seen contexts — 🟡 partly

- **Checked.** `05_setup.tex:456–458`: 60 training and 60 held-out contexts exist; `06:627`: "The ten contexts are drawn from the training contexts"; `06:1119`: "The conclusion is limited to training-set contexts." The abstract, §1.4 and Ch 7 do not say so. Checkpoint selection: `05:482–483` "a random tenth of the training windows"; the windows are built with stride one (`range(T − window_size + 1)`, `d3il/environments/dataset/avoiding_dataset.py:92`; `fm_visual_aligning/datasets/sequence.py:149, 294`) and split at window level (`torch.utils.data.random_split`, `flow_matcher_v3_meanflow/utils/training.py:99`), so the held-out tenth shares demonstrations with the training set. It selects the checkpoint only; no reported number is computed on it.
- **Verdict.** 🟡. Ch 6 already scopes the conclusion; the abstract and the summaries do not. The window-overlap point is correct and costs one clause.
- **Fix.** v2: "on ten training-set contexts" where the abstract and §1.4 state the alignment result. v4: same in Ch 7 (RQ1) and Ch 8 Limitations. v3: one clause at `05:482–483` (windows overlap; the tenth selects the checkpoint only). A held-out evaluation is ⚪ (a cluster job; the author's call, see §8).

### C4 · A rejected context counted as a clean execution — 🟡 partly

- **Checked.** `06:1020–1026`: context 7 is never attempted on the tightened set (the tightening overlaps its box), "scores clean and unmoved in every tightened row", every x/10 is x−1 of nine attempted; "Comparisons between rows are unaffected." `09_appendix.tex:102` repeats it. The disclosure comes after Tables 6.6–6.8 have been read; the medians of the tightened tables include the unmoved 0.45 m of that context.
- **Verdict.** 🟡. The disclosure is honest and the ordering argument is right; the reviewer is right that "10/10 violation-free" reads as ten executions, and that the medians carry the frozen value.
- **Fix.** v3: name the guard in Ch 5 (§5.5.2 protocol, before the tables) and in the captions of Tables 6.6–6.8 ("nine contexts attempted; context 7 is held by the box–obstacle guard and counts as unmoved"); consider printing x/9 for the tightened rows. Recomputing the medians over nine contexts is ⚪ (a DA re-aggregation; it changes printed numbers in three tables and Fig 6.4) — the author's call; the cheap alternative is the caption clause.

### C5 · The abstract needs its limitations — 🟡 partly (the abstract is the author's)

- **Checked.** `latex/pages/abstract.tex:16–25`: "On vision-conditioned alignment, analytic average-velocity matching dominates the diffusion model again, and at the larger budget that task needs, endpoint projection is the better projector on the constraints." No "training contexts", no "not within the tolerance", no "simulation". "Dominates" is defined in `06:350` (non-dominated on the plotted axes) and used in that sense.
- **Verdict.** 🟡. The reviewer's replacement text is a fair reading; the author's release rule is that the abstract is copied as it stands, so this is a v2 change on the author's word only.
- **Fix.** v2, if the author agrees: add the scope ("on ten training-set contexts, in simulation; neither the unprojected policy nor the selected projected one reaches the position tolerance") and name the endpoint of "better projector" (more violation-free contexts). Keep the avoiding sentence.

### C6 · The CI-MeanFM plateau — ✅ confirmed (two halves)

- **Checked.** `06:753–758`: "it does not improve when it is given a larger budget … Raising the budget spends time and buys nothing"; `06:1103–1104`: "does not improve with the budget and reaches 60 % only at a hundred evaluations". Table 6.5 (`06:664–680`), CI-MeanFM α = 0.2: **0.3738 (K = 2) → 0.4247 (10) → 0.3897 (20) → 0.1791 (100)**. So "does not improve" holds from 2 to 20 and fails at 100 — the text at 1103 half-admits it in the same sentence. v4 already scoped its version ("non-monotonic to K = 20, better only at K = 100", v4.2). The mechanism sentence (`06:766–769`, "The mechanism is in the training target …") is an untested account: the author's own style rule (`Writing_Hints/HINT_20260920`, 🟡 "a *because* the writer invented to explain a measurement") forbids exactly this form.
- **Verdict.** ✅ both: scope the range; state the account as a hypothesis or move it to Ch 8.
- **Fix.** v3 §6.2.1 and §6.4: "no improvement from 2 to 20, improvement at 100 (0.1791 m, 60 %)"; the stop-gradient account becomes "consistent with … ; not tested here". v4 Ch 7/8: check the same account (`08_discussion.tex` future-work item on the sampler) reads as a test to run, which v4.2 already made it.

### C7 · Human versus generated demonstrations — 🟡 partly (the author's storyline)

- **Checked.** `07:36–37, 44–46` and `06 §6.4`: "on straight-line demonstrations the generative model does not matter, the budget does"; "with human demonstrations the average-velocity models lead". This is the author's storyline (`GUIDE_20260924_results_storyline_author.md`) and the v4 rules of 2026-09-24/25, applied as instructed. The reviewer's confound list (task, vehicle, controller, dimension, sampling) is factually right; the thesis nowhere claims to have isolated the demonstration source.
- **Verdict.** 🟡. Keep the storyline; the wording "does not matter" and "cannot be told apart" overreach for ten flights and one seed, and the reviewer's "may contribute; its effect was not isolated" is the safer form for the *explanation*, while the *observation* (order changes across tasks) stays.
- **Fix.** v4 Ch 7: keep the ordering per task; write the demonstration-source account as the reading, not the cause ("the one thing that separates these tasks in the data is the demonstrations; the other differences were not varied"). v3 §6.4: same sentence. The author decides how far.

### C8 · Architecture matching does not isolate the objective — 🟡 partly (already disclosed)

- **Checked.** Table 5.8 (`05:955–960`): avoiding batch 8×2 vs 32×2, learning rate 1e-4 vs 5e-4, EMA for the two average-velocity models; `05:1001–1016` says so in words and ends "the results are those of the trained configurations as Table 5.8 states them, not of the objectives in isolation"; `05:1123–1128`: FM sampled from σ = 0.5 against σ = 1 training. The zero-interval query of the endpoint sampler is disclosed at `06:437–440` ("a difference between the rows belongs to the sampler and the point of projection together").
- **Verdict.** 🟡. The thesis states everything the reviewer asks for; what remains is that Ch 7 sometimes attributes the outcome to the objective ("the average-velocity models lead"). The matched-noise ablation and the fixed-query comparison are real experiments, not edits.
- **Fix.** v4 Ch 7 RQ1: one clause "as trained (Table 5.8)". ⚪ The two ablations go on the author's list of runs, if any.

### C9 · The uncertainty argument and the Pareto claims — 🟡 partly / ❌ on the remedy

- **Checked.** `05:708–709`: "It is also the larger of the two spreads available and therefore the conservative one." The reviewer's objection is right: a standard deviation is dispersion, not the uncertainty of a mean, and "larger, therefore conservative" is not an argument. Table 6.3 (`06:440–452`): 4 seeds at K = 3, 1 seed at K = 10, CI-MeanFM K = 10 on twenty episodes, no ± — as the reviewer says.
- **Verdict.** 🟡 on the sentence and the labels; ❌ on "paired differences and uncertainty intervals": the author's rule for the thesis is counts, no tests, no p-values.
- **Fix.** v3: delete the "larger … conservative" sentence; "fronts of observed means" (or "of the printed means") where a frontier is named; the seed/episode exceptions of Table 6.3 stated beside the K = 3 / K = 10 blocks in the text (the caption has them).

### C10 · Selection and confirmation intertwined — 🟡 partly

- **Checked.** `06:13–15`: "The models are then compared at their best rule"; Table 6.12: "at its best selection rule". Both sides of every comparison, including the baseline, are taken at their best rule on the same grid — that is DPCC's own protocol and the DA rule the author set ("the paper baseline at its best projection variant").
- **Verdict.** 🟡. Symmetric selection is fair; the reviewer's point that a selected UAV cell of ten flights is an exploratory optimum is right, and "selected configuration" is already the thesis's term.
- **Fix.** v3: one sentence in §5.5 (evaluation protocol) saying the best rule is selected on the evaluated grid for every model alike, the baseline included; no "validated operating point" language anywhere (none found).

### C11 · The UAV success definition — 🟡 partly; ❌ on "was it fixed before evaluation"

- **Checked.** `05:651–665`: success = crossing the finish line at the end of the walls (x = 2.0 / 3.0 m), in controlled flight (≤ 2 % / 8 % wall-contact steps); the goal line 0.8 m further is not used, and why (the episode limit was set from level-corridor demonstrations); "The counts at the goal point are printed beside the others where they differ." `06:1405–1408` and Table 6.12's caption print them: only the baseline differs, 0.000 (tilt) and 0.300 (hump) at the goal against 1.000 at the exit. The finish line was moved on 2026-09-24 (v3.80–v3.92, the DA notes R44/R45); the thesis states the reason, which is what the reviewer asks for.
- **Verdict.** 🟡. The information is complete but split between prose and caption; a reader of the table alone sees only the exit counts. "Success is not collision-free" is stated in the definition.
- **Fix.** v3: a "goal" column (or two rows) in Table 6.12 for the cells where the two counts differ, or a table note; keep the definition. A common longer-timeout run is ⚪ (author's call).

### C12 · Controller timing is not controller-only — 🟡 partly

- **Checked.** Qualified: `06:1881–1882, 1891, 1902–1908` ("the controller and the simulator step together"), `07:40–41`. Unqualified: `06:1916, 1928, 1987, 2092` ("a twenty-fourth of MuJoCo MPC's controller cost per control step"). Two execution environments: stated at `08_discussion.tex:67` and in Appendix C.
- **Verdict.** 🟡. Ch 7 already carries the qualifier; four Ch 6 sentences do not.
- **Fix.** v3: "controller-and-simulator cost" in the four places. Instrumenting the controller alone is ⚪.

## 2. Mathematics and specification (M1–M9)

### M1 · The 0.31 m footprint inflation — ✅ confirmed as a bound, 🟡 on where it bites

- **Checked.** `04_method.tex:1477–1482`: the reach is "the distance from the centre to the rim of a rotor disk along a body axis, with the vehicle at zero yaw and every approach in these scenes lateral"; `05:1081–1085`: 0.18 + 0.13 = 0.31 across, 0.14 + 0.13 = 0.27 along, the larger taken; `05:577`: UAV-pillars maps the rod radius to "the vehicle's radial reach of 0.36 m". Geometry: the support of four horizontal disks at (±0.14, ±0.18) with radius 0.13 in a horizontal direction at angle φ is 0.14|cos φ| + 0.18|sin φ| + 0.13 — 0.31 at φ = 90°, **0.356 at 45°**, enclosing radius √(0.14² + 0.18²) + 0.13 = **0.358 m** (the figure's red 0.228 m line is that centre distance). Scenes (`config/uav_projection.yaml`, `corridor_v3_tilt`, `corridor_v3_ablation_hump`, `s_curve`): walls at y = ±0.95 / ±0.35 …, axis-parallel → 0.31 exact; the tilt plane is the v2 slide (trace (−2, 0.95)→(2, −0.05)) leaned −60° (`z_lean`), so its unit normal has |n_xy| = 0.5 and the vehicle's support along it is about 0.17 m plus the body's half-height — 0.31 is conservative; the hump's halfspaces lie in x–z → 0.27·|n_x| + body height, conservative; **the keep-out disks** (`sphere_outside`, `dimensions: ['x','y']`, radius 0.05 at the wall ends, the s-curve corner balls) are cylinders approachable from any horizontal direction → **0.31 under-covers by up to 0.048 m**. Yaw is held at zero: `uav_env_test/flight_controller.py:80` (`yaw_des=0.0`), called without a yaw argument at `FM_v3_uav_test/eval_fm_uav.py:1175`. The scorer uses the same 0.31 m offset (`05:1079–1081`; `eval_fm_uav.py:410–458`), so the reported violation counts are consistent with the planner; physical contact is scored by the wall-contact fraction.
- **Verdict.** ✅ the bound is not general and Fig. A.1 shows it; 🟡 in these scenes it is exact on the walls, conservative on the tilt and the hump, and short by ≤ 4.8 cm on the disks only. The reviewer's closing caveat — this does not establish the cause of any recorded collision — stands.
- **Fix.** v2 §4.6.3: state the support-function fact and the disk exception (one sentence with the 0.356 / 0.358 numbers). v3 §5.6.3 (the "0.31 is geometry" paragraph): same, plus "the scorer and the planner share the offset, so a violation-free flight can carry a rotor tip up to 4.8 cm past a nominal disk". v4 Fig. A.1 caption: "the dashed 0.31 m circle is the per-axis reach; the enclosing radius is 0.358 m". ⚪ DA (optional): re-score the stored corridor and s-curve flights against the disks at 0.36 m with the existing E9 scorer, to say whether any flight changes class.

### M2 · Tracking error versus dynamics mismatch — ✅ confirmed

- **Checked.** `04:351–354`: "the arm does not exactly reach p_des within one step — the residual is precisely the w_t that (4.x) pushes into the mismatch bound". The setpoint accumulates in every loop (`eval_fm_uav.py:1155`; `eval_flow_matching_v3_meanflow.py:716`, where `obs[:2]` is the commanded position; `eval_fm_visual_aligning.py:600`), and the demonstrations' action is the difference of consecutive commanded positions (`avoiding_dataset.py:55–60`). With p_des,t+1 = p_des,t + Δp_t and e_t = p_des,t − p_t, the measured-channel mismatch of `eq:method:dpcc:euler` is w_t = p_{t+1} − p_t − Δp_t = e_t − e_{t+1}: the *change* of the tracking error, as the reviewer derives.
- **Fix.** v2: rewrite the sentence ("the change of the residual tracking error over one step is the w_t …"); check that Ch 8's lead numbers (0.49–0.61 m, ~0.3 m) are described as tracking error, not as w_t (they are — `08:` D4 wording of v4.2 — no change).

### M3 · The mismatch-bound calibration — 🟡 partly

- **Checked.** `04:205` and `04:1167`: "γ is estimated from unconstrained rollouts" — inherited DPCC method text. `05:1071–1074`: margins 0.025 m (DPCC's value, taken over for the quadrotor scenes) and 0.03 m (aligning), no calibration reported. `08_discussion.tex:11` says feasibility is a statement about the model; v4.2's Ch 8 gives the measured leads (0.49–0.61 m on the corridor at the first violating step; about 0.3 m on the s-curve) — far above 0.025 m, and the thesis draws the consequence there.
- **Verdict.** 🟡. The reviewer is right that the margins are inherited or chosen, not calibrated per plant, and that the text should say so instead of implying an estimate.
- **Fix.** v2 §4.5.2: "here γ is not re-estimated; the configured margins are stated in Ch 5". v3 §5.6.3: "0.025 m is DPCC's margin, taken over; 0.03 m was chosen for the alignment task; neither was calibrated on rollouts of this thesis" + the one measured fact per plant that exists (the UAV lead is in Ch 8). ⚪ exceedance rates would need a DA over stored paths.

### M4 · Full-state feasibility versus the constrained channels — 🟡 partly, ❌ on the index ranges

- **Checked.** `04:1006–1010` (`eq:bg:mpc:dyn`): the dynamics row runs over t′ ∈ {t, …, t+H−2} — the range is explicit. `04:1414–1418` (`eq:method:dep:varows`) and `04:1445–1447`: the six dynamics rows anchor the commanded and the measured position; the UAV's measured-velocity channels carry no row.
- **Verdict.** 🟡: the general statement and the instantiation are both there, but no sentence says the velocity channels are prediction/conditioning channels outside the constraint, nor on which channels the mismatch ball acts.
- **Fix.** v2: two sentences in §4.6.3 (the spatial plan): the constrained channels, the free ones, the space γ lives in. No index change.

### M5 · Disk, cylinder or sphere — ✅ confirmed (specification)

- **Checked.** `04:1373–1375`: ‖p_i − o‖² ≥ ρ² with p_i ∈ ℝ³ on the quadrotor. Implementation: every keep-out entry is `sphere_outside` with `dimensions: ['x','y']` (`config/uav_projection.yaml`, corridor caps, pillars, s-curve corners; `flow_matcher_v3_uav/sampling/projection.py:93, 547`) — the constraint is on the horizontal distance, i.e. a vertical cylinder, and the prose ("keep-out disks") is right.
- **Fix.** v2: write the equation with the planar projection, ‖P_xy p_i − o_xy‖² ≥ ρ², and say "a vertical cylinder" once.

### M6 · Sampling steps, network evaluations, projection solves — ✅ confirmed (a known open item)

- **Checked.** `main.tex:54`: `\newcommand{\nfe}{K}` — the macro prints K; Table 4.1 defines K (sampling-step budget) and NFE (network evaluations per candidate) separately; `05:1163–1176`: n_guide = ⌈ηK⌉ − 1, so endpoint projection at K = 20, η = 0.2 makes 23 field evaluations. The prose says "at twenty evaluations" for endpoint rows (`06:1103, 1112` and elsewhere). `05:1066–1067` states the diffusion gate's eleven solves; `06:98–99` says ten. v4.2 already sent v3 an FYI on the K/NFE wording (INBOX → v3, open).
- **Fix.** v3: a wording pass — "K = 20" or "twenty sampling steps" wherever a projected row is meant; "eleven" at 06:98; a two-line NFE/solve accounting note in §5.6.4 (the reviewer's table). v4 Ch 7: v4.2 already uses K; re-check two sentences.

### M7 · The planning horizon and switched walls — ✅ confirmed for the switched walls

- **Checked.** `04:1455–1463` (`eq:method:env:switched`): W(x_t) is chosen from the current x and applied to every horizon step; "a wall the horizon will only reach later enters at the replan that reaches it". `08_discussion.tex:109–113`: "A longer horizon would hand the projector its constraints earlier". For the switched walls that does not follow; for the disks, the box, the tilt and the hump (always active) it does.
- **Fix.** v4: qualify the future-work item — earlier for the constraints that are always active; the switched walls need a rule on the predicted position, with the cost the reviewer names (an irrelevant wall over the whole horizon).

### M8 · Target equivalence versus objective equivalence — ✅ confirmed (one word)

- **Checked.** `04:825–827`: at h = 0 "(4.38) reduces to u_tgt = v and the objective becomes (4.30)". The loss (4.39) keeps two heads and the adaptive reweighting, so the *target* reduces to the instantaneous one; the objective does not become (4.30).
- **Fix.** v2: "the target reduces to the instantaneous-velocity target of (4.30)". The α-expansion paragraph (`04:868–880`) is already qualified as the reviewer asks; no change.

### M9 · Controller and solver specification — 🟡 partly

- **Checked.** SLSQP: `04:1047, 1055` ("return its last iterate on non-convergence, so feasibility of that output is not guaranteed"); code `flow_matcher_v3_meanflow/sampling/projection.py:143–147` — `maxiter = 1000`, `ftol` at SciPy's default 1e-6; non-convergence is counted for the endpoint sampler and written to the avoiding results (`nlp_failures_total`, `eval_flow_matching_v3_meanflow.py:832`), not for the per-step projector. Controller reference: the evaluation calls `tracker.compute(p, q, v, om, p_des, v_des)` (`eval_fm_uav.py:1175`) — no acceleration feed-forward and zero yaw; the thesis says "without velocity feed-forward" (`05:1145`) but not what the acceleration term receives.
- **Fix.** v2 §4.5.x: the two solver settings and "last iterate on non-convergence" (present). v3 §5.5: "a_des = 0 and yaw_des = 0 in evaluation". ⚪ DA: the endpoint non-convergence count per avoiding cell exists in the results and can be reported; the per-step count does not exist.

## 3. Direct consistency corrections (the reviewer's table)

| # | Reviewer's row | Checked | Verdict | Fix (owner) |
| :-- | :-- | :-- | :-- | :-- |
| 1 | three `(??)` refs, pp. 29–31 | cause: two `\label`s on one `equation` (`04:648`); `eq:bg:fm:loss` lost, three `\eqref` to it print `??` | ✅ | one label, three refs re-pointed (v2); a double-label warning in the release tool |
| 2 | "Every model … the same checkpoint at every budget" `05:906–907` | contradicts `06:30–32` (every diffusion budget is its own checkpoint) | ✅ | "every flow-based model" (v3) |
| 3 | "Every comparison … five training seeds and two episodes" `06:11–12` | Table 6.3: 4 seeds at K = 3, 1 at K = 10, one cell of twenty episodes | ✅ | "unless a table states otherwise" + the exceptions named (v3) |
| 4 | "no unprojected row exceeds 0.100 S&C" `06:39` | Table 6.1 FM K = 20: 0.133 — and `06:81–82` itself says 0.133 | ✅ | 0.133 (v3) |
| 5 | "Between 15.5 and 19.0 control steps" `06:84–85` | Table 6.1 diffusion K = 2: 20.4 | ✅ | 15.5 to 20.4 (v3) |
| 6 | baseline projection "ten" times `06:98` | `05:1066–1067` says the gate is inclusive: eleven | ✅ | eleven (v3) |
| 7 | "Four of the ten configurations" on the frontier `06:729` | Fig 6.3 plots every cell of Table 6.5 except the two α = 0.05 rows: 13 | ✅ | "four of the thirteen" (v3) |
| 8 | MeanFM "comes closest at K = 20" `06:777` | Table 6.5: 0.0673 m at K = 100 < 0.0741 m at K = 20; `06:734–735` says so | ✅ | "closest at the selected budget; K = 100 is 0.0068 m closer at 4.7× the cost" (v3) |
| 9 | CI-MeanFM and FM "do not move the box" `06:1027` | Table 6.5: 2/10 and 4/10 unmoved, medians 14 % and 10 % | ✅ | "barely move the box (medians …)" (v3) |
| 10 | "the one place in the thesis where a flow-based model is ahead … on every axis" `06:255–257` | `06:1096–1099`: alignment also ahead on every axis (9 vs 4 contexts, 0.179 vs 0.462 m, 266 vs 809 ms) | ✅ | scope to D3IL-avoiding or drop "in the thesis" (v3) |
| 11 | Fig 6.2 caption "triangles" | figure and legend: squares; hollow points defined in the text (`06:350–359`), not in the caption | ✅ | "squares"; hollow = outside the 0.05 band, in the caption (v3) |
| 12 | front-matter TODOs | `main.tex:12–17`; known HOLES of every release | ✅ (author) | the author's metadata in v2's `\get…` block |
| 13 | Appendix B heading flag | author's instruction, v4.2 (2026-09-25): visible in the release on purpose | ❌ for now | remove at the submission build only |
| 14 | diffusion "fixed during training" vs implicit samplers | `abstract.tex:5`, `02:25`; `03:38` discusses implicit samplers; Table 4.1 scopes K to "the diffusion baseline, fixed with its schedule at training" | 🟡 | v2: "the inherited implementation" in the abstract's and Ch 1's sentence (author's word for the abstract) |

## 4. Academic writing (W1–W6)

The author's rules already fix the register (storytelling from the person who ran the experiment; no self-justification; no invented mechanism; mechanism names, not brands). The reviewer's W1–W5 are taste plus a real risk: the repeated aphorism-plus-explanation cadence is where the invented-mechanism sentences (C6, C7) live. I do not recommend rewriting for tone; I recommend that every sentence the reviewer lists be tested against the two author's alerts (🔴 request-in-the-thesis, 🟡 invented *because*) and kept if it passes.

- **W1** (six slogans) — author's call. "The price of a projection is set by what the projector is handed" and "A task saturated at one evaluation hides …" are the two that carry an explanation; the others are fine.
- **W2** ("Three things follow", "Read down the success column first") — author's call; the counting openers are v2's Ch 4 habit (`04:199, 265, 351, 470, 669, 764, 817`), the reading instructions v3's §6.1.1. The reviewer's replacement paragraph for Table 6.1 is accurate.
- **W3** (literal verbs; "the better projector on the constraint") — 🟡: the metric behind "better on the constraint" is violation-free contexts (9 → 10 of ten at K = 20, Table 6.6); name it where the phrase stands alone (abstract, Ch 7).
- **W4** (contribution 6 and the RQ answers are long) — author's call; v4.2 already trimmed RQ1 once.
- **W5** (repetition of full model names; "proves", "the mechanism is", "does not matter") — 🟡: "proves" is not in Ch 7/8; "the mechanism is in the training target" (`06:766`) is C6; "does not matter" (`07:37`) is C7.
- **W6** (the percentage is not per-context progress) — ✅: `05:633–635` defines 0 % correctly ("not a box left unmoved"), but `06:732` still says "the line at which the box was not moved at all" (v3.100's F03 removed the other instances; this one stayed), and Ch 6 uses the shorthand "closes 84 % of the distance" at `06:636, 705, 710–711, 1100–1101, 1972–1973`; Ch 7 (`07:21, 63`) already has the exact form, "reduces the median final distance by 84 % of the mean initial distance". **Fix:** delete the "not moved" clause (§9 A10); whether Ch 6 adopts Ch 7's form is the author's call (§9 B6). *(Corrected in job O005: the first version of this paragraph named `07:63` as a shorthand instance; it is not.)*

## 5. Material to shorten, merge or move

Author's decisions, all of them; the page count (185 compiled; Ch 5–6 alone about 95 pages against the institute's 60–80 orientation) is known. Notes:

- "Move Discussion before Conclusion" — ❌: the author fixed Ch 7 = Conclusion, Ch 8 = Discussion on 2026-09-24 (v4.1).
- §6.4 "adds no new result" (`06:1952`) and the repeated 59.2 vs 70.1 steps / 18.1 vs 553.4 ms tour — the reviewer's count is right; whether §6.4 shrinks to a synthesis table is the author's storyline decision (the author asked for §6.4's per-environment answers in v3.96).
- Tables 6.6–6.8 repeat the MeanFM K = 20 block — true (`06:893`, `1006`); a merge is a v3 restructuring, not a fix.
- Background vs Related Work overlap, the MeanFM derivation displays, §4.6 vs §5.2 scene descriptions — v2/v3 structure; the author's call.

## 6. Tables — confirmed on the pages

- **Table 6.12 (p. 122):** ✅ the last row ("FM 20 single …") sits on the page number. v3: shorten the caption (the goal-point sentence is in the text at `06:1405`), or split tilt and hump into two floats. Do not shrink the font further.
- **Table 6.6 (p. 105):** ✅ caption promises mean ± SD for ms/step (`06:820–822`), rows carry means (`06:838–842`). v3: fix the caption.
- **Table 6.3 (p. 95):** ✅ mixed replication, no spread — see C9.
- **Tables 5.8, 5.9, 6.2, 6.7–6.8, 6.11, 6.14–6.16, C.1:** presentation judgements (long captions, prose in cells, `1.000 ± 0.000` for ten-flight counts, repeated conditions). The `1.000 ± 0.000` point is worth taking (the author's rule "counts are printed as counts" at `05:718–720` argues for 10/10 in the UAV tables). The rest is the author's call; C.1 is deliberately facts-only (author, v4.0).

## 7. Figures — confirmed on the pages

| figure | reviewer | confirmed? | owner of the fix |
| :-- | :-- | :-- | :-- |
| 5.8 (p. 61) | caption crosses the footer rule | ✅ (last caption line on the rule, page number below it) | v3: shorter caption or `[p]` float; the figure is DA's |
| 5.10 (p. 64) | subtitle cut off at the right | ✅ ("… none the l") | DA: re-export with the subtitle inside the canvas (`fig_expert_aligning`) |
| 5.11 (p. 67) | third legend line truncated; caption crosses the footer | ✅ both ("… is in the surfa"; page number inside the caption) | DA: legend; v3: float height |
| 6.1 (p. 87) | small ticks, empty K = 20 row | ✅ visible; layout choice | DA/author |
| 6.2 (p. 93) | triangles vs squares; hollow undefined; dense labels | ✅ squares; hollow only in the text | v3 caption (P1); DA labels (optional) |
| 6.3 (p. 102) | labels overlap near the 0 % line; log-of-remaining-distance axis | ✅ overlap ("10 20 20" cluster); the axis is the author's choice | DA (optional) |
| 6.4 (p. 111) | large "better" box; grey reference from the untightened set; CI-MeanFM colour differs | ✅ all three (purple here, teal in 6.2/6.3/6.6) | DA: one colour per model across figures (P2); the reference ring is declared in the caption |
| 6.6 (p. 125) | dense labels | ✅ ("3 3 3" clusters) | DA (optional) |
| 6.7 (p. 131) | severe overflow; empty sixth slot | ✅ (caption on the page number) | v3: `height=` cap on the includegraphics or `[p]`; DA: 2×3 or 3×2 layout (P1) |
| 6.8 (p. 133) | small subtitles | judgement | DA (optional) |
| A.1 (p. 152) | dashed circle fails to enclose the rotors | ✅ | v4 caption (M1); DA: add the 0.358 m enclosing circle (optional) |
| B.1–B.3 | small panels, long captions | judgement | v4/DA (optional) |

Vector figures: every figure ships as PNG (the release notes say so); the institute asks for vector. Known HOLE, unchanged.

## 8. Missing evidence — what exists, what would need a run

1. **Per-context alignment table** (initial/final error, position success, strict success, constraint outcome, attempted/rejected): ⚪ **DA, no run** — every field is in the stored rollouts (`success`, `success_relaxed`, final distance, violations; context 7's status is known). Recommended: it settles C1, C2 and C4 at once and can live in Appendix B.2 (v4).
2. **Held-out contexts and more seeds:** ⚪ **cluster runs** (60 held-out contexts exist in the dataset). Without them, the scope words of C3/C5 are the answer.
3. **Solver / mismatch diagnostics:** ⚪ partly — endpoint non-convergence counts exist for D3IL-avoiding (`nlp_failures_total`); per-step counts and margin exceedance do not; a tracking-error table over stored UAV paths is a DA job.
4. **Reproducibility identifiers** (seeds, context order, checkpoints, code revision, exact MuJoCo/MJX versions): the artefact map exists as writing material (`v4/notes/`), kept out of the thesis on the author's instruction (App C = facts only). ❌ as a thesis table unless the author reverses that; the code revision and versions are a one-row addition to Table C.1 if wanted.
5. **Internal-validity paragraph:** ❌ as a section — the author archived Threats to Validity as "suicide" (v4.1). The individual disclosures exist (Table 5.8 paragraph, `06:437–440`, `06:1020–1026`); Ch 8 Limitations can gather them in one paragraph if the author wants (v4).
6. **Observation → account → test** for the CI-MeanFM plateau and the demonstration-source reading: see C6, C7 — the tests are named in v4's Future Work (v4.2); the *account* wording is the change.

---

## 9. The list for the author — four groups (revised 2026-09-25 14:48, job O005)

*Supersedes the per-owner change list of 13:10. Since 14:32 the thesis is one workspace (`Working_Space/v5`, job O003), so the
items are no longer sorted by v2 / v3 / v4; each names its file (the same file name in v5) with the line of the release copy.
The verdicts and the evidence stay in §0–§8; this section is only what to do. The current text is curated: **group A can go into
v5 in one pass on your go; every item of C is your call, one by one, and the smallest edit that answers the finding is named**
— nothing in C changes a printed result unless its "touches" cell says so.*

Legend: ✅ apply · ✍️ your decision · ❌ declined (the reviewer is wrong on the facts, or it runs against a decision you took) ·
*touches* = what changes in the curated text.

### A · Pure writing / format bugs — no judgement involved (✅)

**Applied in v5.2** (job O006, 2026-09-25 15:01; record `v5/changelogs/v5.2_20260925_review_tier_a_format_bugs.md`): every v5-side item, twenty edits in Ch 4–6, nothing of record changed, `check.py` and the dry run clean, not compiled. **v5.3** (job O007, 15:20; `v5/changelogs/v5.3_20260925_review_tier_a_figures_at_source.md`): A6 and A9 fixed at the DA source (builders, store, re-export), the pure bugs among the leftovers. Still open: your metadata (A11) and the two tooling items; Figure 6.7's empty sixth slot is cosmetic and stays. The *status* column says which.

| # | what is wrong → the fix | where (release copy) | touches | status |
| :-- | :-- | :-- | :-- | :-- |
| A1 | three `(??)` equation references (pp. 29–31): two `\label`s on one `equation`, amsmath keeps only the last → one label, the three `\eqref` re-pointed | `04_method.tex:648` (refs at 671, 687, 769) | LaTeX labels only | ✅ v5.2 |
| A2 | an internal identifier is printed: "(CLOSURE\_20260907 R.4)" → delete the parenthesis (the sentence it supports is Ci-5) | `06_results.tex:772` | three words | ✅ v5.2 (identifier; the sentence is Ci-5) |
| A3 | Table 6.12 runs onto the page number (p. 122) → shorter caption (the goal-point sentence is in the text at `06:1405`) or two floats; do not shrink the font further | `06:1412–1420` | layout | ✅ v5.2 (caption cut; confirm at the next compile) |
| A4 | Figure 6.7's caption runs over the page number (p. 131) → cap the `\includegraphics` height or `[p]`; the empty sixth panel → DA relayout (2×3 or 3×2) | `06`, Fig 6.7; DA | layout, one figure file | ✅ v5.2 height cap · relayout: cosmetic, not a bug — left |
| A5 | Figures 5.8 and 5.11: the caption sits on the footer rule (pp. 61, 67) → float height or shorter caption | `05`, Figs 5.8, 5.11 | layout | ✅ v5.2 (height caps; confirm at the next compile) |
| A6 | Figure 5.10's subtitle is cut at the right ("… none the l"); Figure 5.11's third legend line is cut ("… is in the surfa") → DA re-export (`fig_expert_aligning`, `fig_expert_uav`) | DA | two figure files | ✅ v5.3 (fixed at the DA source: two-line subtitle, wrapped legend key; re-exported) |
| A7 | Figure 6.2's caption says "triangles temporal consistency"; the figure and its legend use squares; hollow points are defined only in the text (`06:350–359`) → "squares", hollow = outside the 0.05 band, in the caption | `06:371` | one caption | ✅ v5.2 |
| A8 | Table 6.6's caption promises "mean ± sample standard deviation" for ms/step; the rows carry means → fix the caption | `06:820–822` | one caption | ✅ v5.2 |
| A9 | Figure 6.4 colours CI-MeanFM purple; Figures 6.2 / 6.3 / 6.6 teal → one colour per model across the figures | DA | figure files | ✅ v5.3 (palette = `ENGINE_COLOUR_DISTINCT`, as Figs 6.2 / 6.3 / 6.6) |
| A10 | "the line at which the box was not moved at all" — the 0 % line is defined at `05:633–635` as a median final distance equal to the mean initial one, explicitly *not* an unmoved box (v3.100's F03 removed the other instances; this one stayed) → "the 0 % line, a median final distance equal to the mean initial one" | `06:732` | one clause | ✅ v5.2 |
| A11 | front matter `TODO: Author / Supervisor / Advisor / Submission date` → your metadata in the `\get…` block (the known holes of every release) | `main.tex:12–17` | metadata | ⏳ your metadata |

**Prose that contradicts the table beside it** — the table is the record; each fix is a number or a word in prose. Verify the
cell once, then apply (✅):

| # | the sentence says | the table says | the fix | where | status |
| :-- | :-- | :-- | :-- | :-- | :-- |
| A12 | "Every model is trained once per training seed, and the same checkpoint is then evaluated under every step budget" | `06:30–32`: every diffusion budget is its own checkpoint | "Every flow-based model …" | `05_setup.tex:906–907` | ✅ v5.2 |
| A13 | "Every comparison in this section … five training seeds and two episodes per seed" | Table 6.3: four seeds at K = 3, one seed at K = 10, one cell of twenty episodes | "… unless a table states otherwise", the exceptions named beside the K = 3 / K = 10 blocks | `06:11–12` (Table 6.3 `06:440–470`) | ✅ v5.2 |
| A14 | "no unprojected row exceeds 0.100 success with constraint satisfaction" | Table 6.1: FM K = 20 is 0.133 (`06:81–82` says 0.133 itself) | 0.133 | `06:39` | ✅ v5.2 |
| A15 | "Between 15.5 and 19.0 control steps … violate a constraint" | Table 6.1: diffusion K = 2 is 20.4 | 15.5 to 20.4 | `06:84–85` | ✅ v5.2 |
| A16 | the baseline's gate projects on "ten" steps at K = 20 | `05:1066–1067`: the gate is inclusive — eleven | eleven | `06:98` | ✅ v5.2 |
| A17 | "Four of the ten configurations are on it" (the frontier of Fig 6.3) | the figure plots thirteen cells (all of Table 6.5 but the two α = 0.05 rows) | "four of the thirteen" | `06:729` | ✅ v5.2 |
| A18 | analytic MeanFM "comes closest to the target at K = 20" | Table 6.5: 0.0673 m at K = 100 against 0.0741 m at K = 20 (`06:734–735` says so itself) | "closest at the selected budget; K = 100 ends 0.0068 m closer at 4.7× the cost" | `06:777` | ✅ v5.2 |
| A19 | CI-MeanFM and FM "do not move the box" | Table 6.5: 2/10 and 4/10 unmoved, medians 14 % and 10 % | "barely move the box (medians 14 % and 10 %)" | `06:1027` | ✅ v5.2 (both instances) |
| A20 | "the one place in the thesis where a flow-based model is ahead of the baseline on every reported axis at once" | `06:1096–1099`: alignment is also ahead on every axis (9 vs 4 contexts, 0.179 vs 0.462 m, 266 vs 809 ms) | "the one place on D3IL-avoiding …", or drop "in the thesis" | `06:255–257` | ✅ v5.2 |

Known and intentional, not bugs: the Appendix B heading flag (your instruction, v4.2: visible until the submission build) · every
figure ships as PNG (the institute asks for vector; the standing hole) · the one-word `\getDoctype` patch the release tool applies.

Tooling (the RELEASE tool, not the thesis; ⏳ code, on your go): warn when one display carries two `\label`s (A1 could not be caught) · flag `CLOSURE_`,
`DA_2026`, `R\d+` in prose as residue (A2).

### B · Tone, writing style, redundant text — sorted by risk, size and whether the reviewer named a target (✍️ all yours; none is a factual defect)

*Revised 2026-09-25 15:35 (job O009). **size** S = a word or one sentence · M = a paragraph, one caption or one float · L = a
section, several floats or a rebuild across chapters. **risk** = what an unwanted edit costs: 🟢 wording only, nothing of record ·
🟠 a definition, a storytelling device or a figure design you chose · 🔴 the storyline or the structure. Counts of a word are over
the released text (`…_ORCH_v5.3_BUGFIX_A/latex`).*

**B-i · Low risk, minor — the reviewer names the sentence; the edit is local**

| # | item (W = the reviewer's section) | where (release copy · page) | the edit | size · risk |
| :-- | :-- | :-- | :-- | :-- |
| B-i-1 | W1 "buys expressiveness with computation" | `01_introduction.tex:5` · p. 1 | keep, or the literal form ("requires repeated network evaluations during planning") | S · 🟢 |
| B-i-2 | W1 "structural, and the structure is what makes it addressable" | `01_introduction.tex:13` · p. 1 (the reviewer's quote is not verbatim) | say which structure: the inherited denoising schedule fixes the budget at training | S · 🟢 |
| B-i-3 | W1 "The price of a projection is set by what the projector is handed." | `06_results.tex:2057` · §6.4, p. 139 | a slogan for an observed fact (Table 6.3: the endpoint solve is the cheaper one); keep, or the literal sentence | S · 🟢 |
| B-i-4 | W1 "A task saturated at one evaluation hides what a task that rewards more evaluations exposes." | `06:2061` · §6.4 | the §6.4 echo of the CI-MeanFM account — stands or falls with Ci-7 | S · 🟠 (tied to Ci-7) |
| B-i-5 | W1 "a plant that holds nothing by itself" | `07_conclusion.tex:31` · p. 142 | "an underactuated quadrotor that needs feedback to hover" | S · 🟢 |
| B-i-6 | W1 "The results stop at three things" | `08_discussion.tex:3` · p. 145 | delete the sentence; the headings carry it | S · 🟢 |
| B-i-7 | W2 counting openers ("Three things follow", "Two consequences matter", …) | `04_method.tex:199, 265, 351, 470, 669, 764, 817` | delete the opener where the items are not genuinely distinct — seven independent one-clause edits | S × 7 · 🟢 |
| B-i-8 | W2 reading instructions for Table 6.1 | `06:79, 83, 95` ("Read down the success column first", "Read the next column", "The last column prices generation alone") | delete the three clauses (S · 🟢); the reviewer's synthesis paragraph for §6.1.1 is accurate but replaces your read-through device (M · 🟠) | S · 🟢 / M · 🟠 |
| B-i-9 | W3 "the better projector on the constraints" | `abstract.tex:22`; `06` × 6; `07` × 2 | name the metric where the phrase stands alone: more violation-free contexts (9 → 10 of ten at K = 20, Table 6.6); the abstract on your word | S × 3 · 🟢 (abstract 🟠) |
| B-i-10 | W6 the shorthand "closes N % of the distance" | `06:636, 705, 710–711, 1100–1101, 1972–1973` | Ch 7's exact form ("reduces the median final distance by N % of the mean initial distance"); the definition at `05:630–635` stands | S × 5 · 🟢 |
| B-i-11 | §5 the arm / quadrotor contrast said twice | `02_background.tex:130–140` | merge the two paragraphs into one | M · 🟢 |
| B-i-12 | §5 the Discussion's candidate-selection paragraph | `08` · p. 147 | one implication and one representative example instead of the counts and timings | M · 🟠 (prunes numbers you put there) |
| B-i-13 | §6 `1.000 ± 0.000` for ten-flight outcomes | Tables 6.11, 6.12, B.1–B.4 | `10/10` — your own rule (`05:718–720`, counts are printed as counts); mechanical, many cells | M (many cells) · 🟢 |
| B-i-14 | §6 one meaning per dash | Tables 6.6 ("— no guiding step", "not run"), 6.12 ("—: no flight reaches the end") | each caption already defines its dash; a table note where a caption was cut (A3) | S · 🟢 |
| B-i-15 | §7 red / green as the only outcome code on path figures | Figs 5.9, 5.11, 6.5, 6.8, B.1 (DA) | a line style or marker beside the colour | M (DA) · 🟢 |
| B-i-16 | §7 label thinning, the boxed "better" arrows, tick and subtitle sizes | Figs 6.2 / 6.3 / 6.6 (labels), 6.2 / 6.3 / 6.4 / 6.6 (arrow), 6.1 (ticks), 6.8 (subtitles), B.1 (panel size) — DA | each a small builder change; the arrow is your device (v3.63) | M (DA) · 🟢 (arrow 🟠) |
| B-i-17 | §6 / §7 Appendix B.1–B.4 captions and B.1's panel size | `09_appendix.tex`, `app_long/` | shorter captions (condition, encoding, exceptions); a larger B.1 | M · 🟢 |

**B-ii · Giant — a rebuild of a section, several floats or a chapter; each moves pages, cross-references or the storyline. A decision first, then its own job**

| # | item | where | what it entails | size · risk |
| :-- | :-- | :-- | :-- | :-- |
| B-ii-1 | W4 / §5 six contributions → three or four; the RQ answers once, short, with the limitation beside the claim | §1.4 (pp. 2–4); Ch 7 (pp. 141–143) | a rewrite of §1.4 and Ch 7; the contributions ↔ RQ mapping is your storyline (v4.1 / v4.2 set it; RQ1 was trimmed once) | L · 🔴 |
| B-ii-2 | §5 Background vs Related Work overlap | Ch 2–3 (pp. 6–15) | remove the repeated diffusion / flow / projection descriptions: Ch 2 defines, Ch 3 places the gap — two chapters of v2's text | L · 🔴 |
| B-ii-3 | §5 the MeanFM development's repeated displays | Ch 4 (pp. 30–34) | remove repeated target / sampler displays, move inherited derivation — equation numbers and `\eqref`s move, and C-ii items point into this section | L · 🔴 |
| B-ii-4 | §5 §4.6 vs §5.2 scene descriptions | Ch 4 / Ch 5 (pp. 43–61) | cross-reference instead of repeating (a v2 / v3 seam): Method keeps variables and constraints, Setup the numbers and the protocol | L · 🟠 |
| B-ii-5 | §5 / §6 Tables 6.6–6.8 | pp. 105–109 | one full cross-model comparison in the body, the threshold / rule ladder to the appendix, the MeanFM K = 20 block printed once — every `\autoref` and sentence reading those tables follows | L · 🔴 |
| B-ii-6 | §5 Table 6.11: the two rescorings as one row per model / budget | p. 120 | the DA confirms which timing fields are shared before columns merge | L (DA + table) · 🟠 |
| B-ii-7 | §5 §6.4 → one synthesis table | pp. 136–140 | §6.4 says of itself that it "adds no new result"; you asked for its per-environment answers at v3.96 — a decision, then a rebuild | L · 🔴 |
| B-ii-8 | W3 the vocabulary sweep ("reads / prices / hands / buys / carries / of record") | Ch 6 above all: "read(s)" 36, "carries" 18, "prices" 17, "of record" 10, "hands" 8, "buys" 5; Ch 4–5 and 7–8 a handful each | a chapter-wide re-wording; "the baseline of record" is a defined term (▶, Ch 5) | L · 🟠 |
| B-ii-9 | §6 caption shortening on the Table 6.2 model; "Planner time [ms/action]" for "ms/step" (× 18 in Ch 6); decimal alignment | every results table | a pass over about sixteen tables — low risk each, large in count | L · 🟢 |
| B-ii-10 | §6 Table 6.2 compact + full grid to the appendix; Tables 5.8 / 5.9 as a settings matrix with the prose out of the cells; Tables 6.14–6.16 as one controller comparison | pp. 77, 79, 91, 132–135 | table surgery, appendix additions, and the text that reads them | L · 🟠 |
| B-ii-11 | §7 figure redesigns: 4.1 (vendored; attribution), 5.3 (typography), 5.6 vs 5.10 (one to the appendix), 6.1 (2 × 4 + inset), 6.3 (axis), 6.5 (enlarge or appendix), 6.6 (split the frontier from the within-group comparison), 6.7 (selected panels + appendix) | DA builders + captions | each a redesign; 6.3's axis and 6.4's reference ring are your decisions (v3.62 / v3.63); 6.7's empty slot is cosmetic (its page fit is v5.2's) | L (DA) · 🟠 |
| B-ii-12 | §5 Discussion before Conclusion | Ch 7 / 8 | ❌ your order (v4.1) stands | — |
| B-ii-13 | §7 vector export | all 41 figures | the standing hole: the builders write SVG, but this machine has no rasteriser for `svg2pdf.sh`; a run on yours | L (tooling) · 🟢 |

**B-iii · No specific target — a taste or a pattern claim with no sentence to act on; you would first pick the places**

| # | the reviewer says | what the text shows | so |
| :-- | :-- | :-- | :-- |
| B-iii-1 | W1 "the persistent pattern: memorable contrast, confident explanation, restatement", beyond the six quoted | six sentences are named (B-i-1 … 6); the rest is a reading impression | act on the six; more needs a list |
| B-iii-2 | W3 "Results throughout; Discussion pp. 146–147; Conclusion pp. 141–143" | page ranges, no sentences; the counts are in B-ii-8 | the sweep is L, or nothing |
| B-iii-3 | W5 "proves"; "unqualified dominates" | "proves": 0 hits in Ch 1–8; "dominat…": 31 hits in Ch 6, 3 in Ch 7, 2 in the abstract — a term the thesis defines (`06:350`, non-dominated on the plotted axes) and uses in that sense | no target for "proves"; "dominates" is defined — keep, or restate the definition where Ch 7 / the abstract first use it |
| B-iii-4 | W5 full model names repeated "several times in one paragraph" | Ch 6: "instantaneous-velocity matching" 44 ×, "analytic average-velocity matching" 25 ×, "consistency-interpolated …" 5 × — your rule (mechanism names in prose; brands only in Related Work); the short labels live in tables and figures | runs against your naming rule; no paragraph named — decline unless you want the short labels in prose |
| B-iii-5 | W4 "put the important limitation in the same paragraph as the claim it qualifies" | general; Ch 7 (v4.2) already carries its qualifiers inside the RQ answers | no target |
| B-iii-6 | W6 "prefer the absolute median distance in the headline" | a storyline choice, not a defect: you chose the share (v3.62) | a decision, not an edit |
| B-iii-7 | §6 "decimal-aligned columns, consistent precision for the same metric" | no cell named; precision is consistent within each table as far as checked | no target |
| B-iii-8 | §7 "typography at the final printed dimensions; 8–9 pt for tick and legend text" | a DA standard, not a finding; the SVG builders scale fonts per figure | a guideline for B-ii-11, if it runs |
| B-iii-9 | §5 "the thesis's length is not by itself the objection" (185 pages; Ch 5–6 ≈ 95) | the page count is known (60–200 limit, 60–80 orientation) | the L rows of B-ii are where pages would come from |

### C · Content — factual issues and suggested changes (the curated text: one by one, the smallest edit named)

*Revised 2026-09-25 15:35 (job O009): a **risk** column. 🔴 **dangerous** — a result statement, a printed number or a table's structure changes, the abstract or Ch 7's storyline is touched, or data are needed; a wrong edit corrupts a claim. 🟠 **careful** — a technical statement or an explanation changes (verified against code or data), or one wording moves in many places; nothing of record moves, but read the new sentence once. 🟢 **minor** — a clause or a word adds scope or precision in one place; no number, no claim. Tally: C-i 🔴 2 · 🟠 3 · 🟢 2; C-ii 🔴 4 · 🟠 6 · 🟢 7 — in all **🔴 6 · 🟠 9 · 🟢 9**.*

**C-i · wrong as printed** — the sentence does not match the code, the data or the thesis's own definitions. Recommended ✅,
still yours to confirm before it is applied.

| # | finding | evidence | smallest fix | touches | risk |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Ci-1 | **M2** "the residual is precisely the w_t" — under an accumulating setpoint the mismatch of `eq:method:dpcc:euler` is the *change* of the tracking error, e_t − e_{t+1} | all three evaluation loops accumulate p_des; the demonstrations define the action the same way (§2 M2) | "the change of the residual tracking error over one step is the w_t …" | `04_method.tex:351–354`, one sentence; Ch 8's lead numbers stay (they are tracking error and described as such) | 🟠 a mathematical statement in Ch 4 changes; verified against the three evaluation loops; no number |
| Ci-2 | **M8** at h = 0 "the objective becomes (4.30)" — the *target* reduces; the loss keeps its two heads and the reweighting | §2 M8 | "the target reduces to the instantaneous-velocity target of (4.30)" | `04:825–827`, one clause | 🟢 one clause |
| Ci-3 | **M5** the keep-out row is written on p_i ∈ ℝ³ (a sphere); every keep-out entry is `sphere_outside` on (x, y) — a vertical cylinder; the prose "keep-out disks" is right | `config/uav_projection.yaml`; `projection.py:93, 547` | ‖P_xy p_i − o_xy‖² ≥ ρ², "a vertical cylinder" once | `04:1373–1375`, one equation | 🟠 a displayed equation changes (`eq:method:env:disk`); verified against the config and the projector |
| Ci-4 | **C6, the scope** "it does not improve when it is given a larger budget" — true from 2 to 20, false at 100 (Table 6.5: 0.1791 m, 60 %; `06:1103` half-admits it) | Table 6.5 `06:664–680` | "no improvement from 2 to 20; at 100 it does (0.1791 m, 60 %)" — Ch 7 (v4.2) already says so | `06:753–758, 1103–1104`, two sentences; no number changes | 🔴 a result sentence in Ch 6 changes — backed by Table 6.5 and already said in Ch 7 |
| Ci-5 | **C1** "succeeds in seven … in two" — counts from a 2026-09-07 analysis of 14 102 rollouts, entered at v3.15 and never re-derived after the corpora of record moved (v3.41); no table carries them | §1 C1 | delete the sentence (keep "The level result at K = 2 is in distance only"; Table 6.5's *unmoved* column carries the point) — or recompute them (D1) | `06:770–772`, one sentence; A2 goes with it | 🔴 a result sentence (two counts) is deleted or recomputed (D1) |
| Ci-6 | **C1** "the model that solves the task does so at twenty" — nothing solves the task by the task's own test (`06:1116–1119` says so) | §1 C1 | a distance statement: "comes closest at twenty" | `06:945`, one clause | 🟢 one word, "solves" → a distance statement |
| Ci-7 | **C6, the account** "The mechanism is in the training target …" — an untested *because*; your own 🟡 alert forbids the form | §1 C6 | "consistent with …; not tested here", or move it to Ch 8, where the test is already named (v4.2) | `06:766–769`, `06:1104`, one sentence each | 🟠 an explanation becomes a hypothesis; storyline-adjacent |

**C-ii · a claim wider than its evidence, or a specification left implicit** — a qualifier or a sentence is added; nothing
printed changes. In the order I would take them; every one ✍️ yours.

| # | finding | smallest fix | where | touches | risk |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Cii-1 | **C3 / C5** the alignment result is on ten *training-set* contexts (`06:627, 1119` say so; the abstract, §1.4, Ch 7 and Ch 8 do not); no context within the position tolerance; simulation | "on ten training-set contexts" in §1.4, Ch 7 RQ1, Ch 8 Limitations; the abstract only on your word (C5's replacement text is fair) | abstract l. 16–25; §1.4; `07`; `08` | a clause in three or four places | 🔴 the abstract's headline scope (your text); the §1.4 / Ch 7 / Ch 8 clauses alone are 🟢 |
| Cii-2 | **C3** the checkpoint-selection tenth is drawn at window level from stride-one windows, so it shares demonstrations with training; it selects the checkpoint only | one clause | `05:482–483` | one clause | 🟢 one clause in Ch 5 |
| Cii-3 | **C4** context 7 is never attempted on the tightened set; disclosed *after* Tables 6.6–6.8 (`06:1020–1026`); every x/10 is x−1 of nine; the medians carry its frozen 0.45 m | name the guard in §5.5.2 before the tables and in the three captions ("nine contexts attempted; context 7 held by the guard, counted unmoved"); optionally x/9 in the tightened rows; recomputing the medians is D2 | `05` protocol; `06:817–1010` captions | captions + one sentence; numbers unchanged unless D2 | 🟠 three captions + one protocol sentence; x/9 in the rows would be 🔴 (D2) |
| Cii-4 | **C12** "a twenty-fourth of the controller's cost" — controller *and simulator* step together (`06:1881–1908` and Ch 7 say so; four sentences do not) | "controller-and-simulator cost" | `06:1916, 1928, 1987, 2092` | four words | 🟢 four words |
| Cii-5 | **M1** 0.31 m is the per-axis reach; the enclosing radius is 0.358 m (0.356 m on the diagonal); exact on the axis-parallel walls, conservative on the tilt and the hump, short by ≤ 4.8 cm on the keep-out disks only; planner and scorer share the offset | one sentence each: §4.6.3 (the support-function fact, the disk exception), §5.6.3 (+ "a violation-free flight can carry a rotor tip up to 4.8 cm past a nominal disk"), Fig. A.1's caption ("dashed = per-axis reach; enclosing radius 0.358 m") | `04:1477–1482`; `05:1076–1086`; `09:51–53` | numbers added, none changed | 🔴 new numbers (0.356 / 0.358 m, 4.8 cm) enter three places and qualify a safety statement; derived from the config, not in any table |
| Cii-6 | **M3** "γ is estimated from unconstrained rollouts" is inherited method text; the margins 0.025 m (DPCC's) and 0.03 m (chosen) were not calibrated here | "γ is not re-estimated here; the configured margins are stated in Ch 5" · "0.025 m is DPCC's margin, taken over; 0.03 m was chosen for alignment; neither calibrated on rollouts of this thesis" (the measured UAV lead is already in Ch 8) | `04:1167`; `05:1071–1074` | two sentences | 🟠 removes an implied claim (γ estimated) in two places; no number changes |
| Cii-7 | **M4** no sentence says which channels the dynamics rows and the mismatch ball act on (the UAV's velocity channels are prediction channels, unconstrained) | two sentences in §4.6.3 | `04:1445–1447` | two sentences · ❌ the index ranges (explicit at `04:1006–1010`) | 🟢 two explanatory sentences |
| Cii-8 | **M7** "a longer horizon would hand the projector its constraints earlier" — not for the switched walls (W(x_t) is chosen from the current state for the whole horizon) | earlier for the always-active constraints; the switched walls need a rule on the predicted position, at the cost the reviewer names | `08:109–113` | one sentence | 🟢 one future-work sentence |
| Cii-9 | **C9** "the larger of the two spreads … therefore the conservative one" — dispersion is not the uncertainty of a mean, and "larger, therefore conservative" is not an argument; Table 6.3's mixed replication is in the caption only | delete the sentence; "fronts of observed means" where a frontier is named; the seed / episode exceptions beside the K = 3 / K = 10 blocks (= A13) | `05:708–709`; `06:440–470` | one sentence deleted, one phrase · ❌ paired tests / intervals (your rule: counts, no tests) | 🟠 deletes a justification sentence; "fronts of observed means" recurs where a frontier is named |
| Cii-10 | **C10** the best selection rule is picked on the evaluated grid for every model alike, the baseline included — right, and unstated | one sentence in §5.5 | `05` protocol | one sentence | 🟢 one sentence |
| Cii-11 | **C11** the goal-point counts (baseline 0.000 tilt / 0.300 hump at the goal against 1.000 at the exit) are in the caption and the prose (`06:1405–1408`), not in Table 6.12's cells | a "goal" column or a table note for the cells that differ; the definition stays | `06:1412–1420` | a table column · ❌ the "fixed after the fact" implication (`05:658–664` states the reason) | 🔴 Table 6.12's structure changes (it overflowed once already); the numbers exist |
| Cii-12 | **C7** "on straight-line demonstrations the generative model does not matter, the budget does" — ten flights, one seed; the demonstration source was not isolated (task, vehicle, controller, dimension and sampling all differ) | keep the per-task order (your storyline); write the demonstration-source account as the reading, not the cause ("the one thing that separates these tasks in the data …") | `07:36–37, 44–46`; §6.4 | the wording of two sentences | 🔴 storyline sentences in Ch 7 and §6.4 |
| Cii-13 | **C8** Ch 7 attributes the outcome to the objective; Table 5.8 and `05:1001–1016` already say the trained configurations differ | "as trained (Table 5.8)" in RQ1 | `07` | one clause | 🟢 one clause |
| Cii-14 | **M6** "at twenty evaluations" on an endpoint row means K = 20 sampling steps (23 field evaluations) | the convention is declared in Ch 5 (`sec:setup:protocol:eval`) and restated at `06:83–87`, so v5.1 closed it as no-change; the stricter wording ("K = 20" / "twenty sampling steps" on projected rows, a two-line NFE / solve note in §5.6.4) is optional | Ch 5–6 | an optional wording pass; A16 is the only hard error | 🟠 an optional wording sweep over Ch 5–6; the convention is declared |
| Cii-15 | **M9** solver: `maxiter = 1000`, `ftol` 1e-6, last iterate on non-convergence (present in part); controller in evaluation: a_des = 0 and yaw_des = 0 (not stated) | the two settings in §4.5; one sentence in §5.5 | `04:1047–1055`; `05:1145` | two sentences | 🟢 two specification sentences from the code |
| Cii-16 | **row 14** "fixed during training" for the diffusion baseline, against implicit samplers | "the inherited implementation" as the scope | abstract l. 5 (your word); `02:25` | two words | 🟠 two words, but in the abstract |
| Cii-17 | **§8-4 / §8-5** reproducibility identifiers; an internal-validity paragraph | the code revision and the exact MuJoCo / MJX versions as one row of Table C.1, if wanted; one Limitations *paragraph* gathering the disclosures that exist (the Table 5.8 paragraph, `06:437–440`, `06:1020–1026`) | `09`; `08` | ❌ as a table or a section (App C facts-only, v4.0; Threats to Validity archived, v4.1) | 🟠 a new Ch 8 paragraph and a Table C.1 row; Threats to Validity was archived (v4.1) |

**C-iii · declined ❌** — the reviewer is wrong on the facts, or it runs against a decision you took: C9's paired differences and
uncertainty intervals (counts, no tests) · C2 "restore the original completion metric" as stated (`05:793–803` explains why it does
not transfer; *reporting* it is D1) · C11's implication that the finish line was moved without saying so (`05:658–664`) · M4's
"give explicit index ranges" (given) · §5 "Discussion before Conclusion" (v4.1) · row 13's Appendix B flag (v4.2) · a
Threats-to-Validity section (v4.1) · a reproducibility table (v4.0).

### D · Suggested runs and data

**From stored data — the DA only, no cluster:**

| # | what | settles | changes printed numbers? |
| :-- | :-- | :-- | :-- |
| D1 | a per-context alignment table: strict and position-only success (`success`, `success_relaxed` per rollout), final distance, violations, attempted / rejected — for every aligning cell of record; App B.2 | C1, C2, C4 at once | adds a table; nothing existing changes |
| D2 | the tightened alignment medians over the nine attempted contexts | C4 | **yes** — Tables 6.6–6.8 and Fig 6.4; the cheap alternative is Cii-3's caption clause |
| D3 | re-score the stored corridor and s-curve flights against the keep-out disks at 0.36 m (the E9 scorer) | M1 | only if a flight changes class |
| D4 | endpoint non-convergence counts per avoiding cell (`nlp_failures_total` is in the results) | M9 | adds a column or a sentence |
| D5 | a tracking-error / margin-exceedance table over the stored UAV paths | M3 | adds a table |
| D6 | figure work already listed in A: the re-exports (A6), one colour per model (A9), Fig 6.7's relayout (A4); optional: label thinning (B9), the 0.358 m circle on A.1 | — | figures only |

**Cluster runs — your call, each its own job:**

| # | what | settles |
| :-- | :-- | :-- |
| D7 | alignment on the 60 held-out contexts (they exist in the dataset); more seeds | C3, C5 — without it, Cii-1's scope words are the answer |
| D8 | FM trained at the sampled noise scale (the matched-noise ablation); the endpoint sampler with the same query interval as the plain one | C8 |
| D9 | the corridor under a common longer timeout (the goal line 0.8 m further) | C11 |
| D10 | controller-only timing (instrumentation) | C12 |
| D11 | a per-step projector non-convergence count (does not exist; instrumentation) | M9 |

**How this proceeds.** A is done (v5.2 / v5.3; released as `…_ORCH_v5.3_BUGFIX_A`). **B:** tick rows of B-i (each local, S or M); a B-ii row is a decision before it is a job (its own Orchestra job and v5.N, pages and cross-references re-checked); B-iii needs your list of places first. **C:** the 🟢 rows can go in one pass on a single yes (→ v5.4); 🟠 one by one, the new sentence written into the changelog as it goes in; 🔴 only on your explicit decision each — Ci-5 and Cii-11 may want their DA item (D1, D2) first. **D:** D1–D6 become a note to the DA, D7–D11 a run list.
re-run); of B and C only the ticked items, each with its *touches* cell as the limit of the edit; D1–D6 become a note to the DA,
D7–D11 a run list. Nothing is applied before your ticks.

---

Orchestra (Claude Fable 5.1, Claude Code), O002 · 2026-09-25 13:10, §9 revised and §4 W6 corrected in job O005 · 2026-09-25 14:48 · group A applied in v5 (v5.2) in job O006 · 2026-09-25 15:01 · A6 / A9 fixed at the DA source (v5.3) in job O007 · 2026-09-25 15:20 · B / C re-sorted by risk in job O009 · 2026-09-25 15:35 · nothing compiled, nothing committed; no file of v2, v3, v4 or the release's `latex/` touched; DA: two builders and three store figures changed (O007).
