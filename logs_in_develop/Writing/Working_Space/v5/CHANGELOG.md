# CHANGELOG — `Working_Space/v5` (the thesis; the Advance Orchestra)

Every working pass on v5 gets an entry here, newest first — **one revision per pass, `## v5.N`, the next number** — and a
detailed, signed MD under `changelogs/`. The author's scheme (2026-09-25): v5.0 = the init with the v2 / v3 / v4 revisions,
v5.1 = the first pass, and so on. Format follows the drafts' changelogs: what changed · why (the author's words) · what it is
sourced from · what was checked · what it left open.

**Rules for this file**

- One entry per working pass, not per edit; the heading names the Orchestra job and links the detailed MD.
- The heading also names the pass's **source** (which audit / review / author instruction) and its **kind**: `PURE BUG FIX` (form,
  format, consistency — no claim, no number of record, no figure content changes) or `CONTENT` (a claim, number, figure or
  structure changes; the author's tick per item). Author, 2026-09-25: "mark in the version/changelog this time is from which
  audit and it is pure bug fixing."
- The drafts' sourcing rules travel with their chapters (`README.md`, rule 2): an equation names its source, a number names its
  evidence of record, Ch 7–8 restate Ch 6 with a `\dataref`.
- Mechanical checks (`tools/check.py`, `tools/make_release_v5.py --dry-run`) are re-run after every pass and their result is
  recorded. **There is no TeX toolchain in this container, so "checked" never means "compiled".**
- A pass that absorbs a legacy change (`tools/absorb.py merge`) records which v2 / v3 / v4 revision it absorbed
  (`inherited/ABSORB_STATE.json`) and marks the owner's INBOX row `🔀 v5.N`.
- A release built from this revision is named in the entry (`RELEASE/output/<stamp>_thesis_release_ORCH_v5.N…`); the
  date-time is the build's identity.
- Every entry and every MD under `changelogs/` is signed: who, model, date, not compiled.

---

## v5.8 — 2026-09-25 · ROUND2 FIX · source: the second reading of v5.7 (ROUND2_FEEDBACK…), answer list R2-A · twenty factual and consistency corrections in Ch 5–8, six of them undoing regressions of v5.5–v5.7; the controller reference, the guard, the Steps definitions, the dominance definition; no number of record changed (Orchestra O023) → [`changelogs/v5.8_20260925_round2_fix.md`](changelogs/v5.8_20260925_round2_fix.md)

- **Source and kind: ROUND2 FIX (factual and consistency corrections).** The second reading of v5.7
  (`RELEASE/output/20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX/feedback/ROUND2_FEEDBACK_TO_RESPONSE_AND_V5.7_20260925.md`) as answered in
  `feedback/CLAUDE_ANSWER_to_ROUND2_20260925_2157_v5.7.md` — its list R2-A, the twenty ✅ DO items; the author: "ship. and v5 changelog and
  release". Six of the twenty undo regressions of v5.5–v5.7; no number of record, table cell or figure changed.
- **Changed (28 edit operations, four files):** **6A** Ch 5 no longer says the corridor / s-curve controller receives the planned velocity — it
  receives the accumulated position setpoint with zero velocity and acceleration feed-forward and a zero yaw reference (the reported flights
  used the `pid_stopgo` tracker; Ch 4 was right). **4.2a** "the vehicle's per-axis reach, not only its centre point, clears it" with a pointer
  to the directional limit. **4.2b** Ch 8: the manipulator's guarantee is conditional on the one-step change of the tracking error staying
  within the uncalibrated margin. **4.3** "Including the same guarded context preserves the ordering of violation-free counts; the reported
  distances and timings also include that context" in §6.2.2; "one held by the guard" at Ch 7's two "all ten" places. **4.4a** "the selected
  model is run at its $\nfe=20$ operating point". **4.4b** the horizon item names the span-switched constraints (the walls and the hump's two
  halves) and lists no "always-active" ones. **4.4c** six endpoint sentences say $\nfe=20$ / sampling steps, not "twenty evaluations".
  **4.5** the two certainty clauses ("as this one does", "as D3IL-avoiding is") dropped from the CI-MeanFM account. **6B** "half of it
  projected" deleted. **6C** the generic episode definition no longer ends a run at the finish line; the quadrotor's latch and continuation
  are stated. **6D-1** "every flow-based row at its operating budget is cheaper than every diffusion row at equal S\&C". **6D-2** "the
  diffusion model trained and run at one denoising step loses it". **6D-3** "only" removed in §6.2.1 and Ch 7. **6D-4** the dangling "with
  the fewest violating steps" deleted. **6E** Steps defined: episode length over every episode on D3IL-avoiding and UAV-pillars; steps to
  the end over the flights that reach it on UAV-corridor (§5.4.1, §5.4.3, Table 6.11's caption). **6F-1** non-dominance as the figure
  computes it (at least as fast and fewer control steps), "dominance refers to the reported means on the evaluated configurations", and
  Ch 7's gloss "equal success with constraint satisfaction, fewer control steps and less time per action". **6F-2** the guiding-step
  shorthand for $\eta>0$, the general form referenced. **6F-3** "None of them arrives legally as a rule". **6F-4** the merged caption: "among
  the rules with the fewest aborts random selection has the fewest violating steps". **D4** "Endpoint non-convergence was uncommon".
- **Not in this pass (the answer's R2-B / R2-C):** the abstract; the corridor wording "did not separate the objectives"; the two optional cuts;
  the `10/10` cells — the author's word each.
- **Checked:** `tools/check.py` 17 files, 7802 lines, 277 labels, 53/53 citations, 41 figures, all pass; `tools/make_release_v5.py
  --dry-run` 19 + 48 files, 9 holes, 1 finding, 4 residue hits. Record: `changelogs/v5.8_20260925_round2_fix.md`. **Not compiled.**
- **Release (ROUND2 FIX):** `RELEASE/output/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX/` (job O024); page estimate ~179 (152-215); `diff -rq` against the v5.7 build: 4 file(s) differ — `chapters/05_setup.tex`, `chapters/06_results.tex`, `chapters/07_conclusion.tex`, `chapters/08_discussion.tex`; no file only on one side.
- **Left open:** R2-B / R2-C above; the ⛔ rows and 🟡 remainders of the round-one answer's C; the held B-i rows. Nothing committed.
  Signed: Orchestra (Claude Fable 5.1, Claude Code), O023 · 2026-09-25.


## v5.7 — 2026-09-25 · DATA FIX · source: the first-reading audit (M9 / §8-3), answer group D under the conservative verdicts — D4 only: the endpoint solves' non-convergence rate (0.1–1.1 %, a number of record in the DA) stated in one sentence of §6.1.2; D7–D11 rejected by the author, D1 / 2 / 3 / 5 not made (Orchestra O020) → [`changelogs/v5.7_20260925_review_D_data_fix.md`](changelogs/v5.7_20260925_review_D_data_fix.md)

- **Source and kind: DATA FIX.** Group D of the review answer (`…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` §9 D; the first-reading audit's
  §2 M9 and §8 item 3) under the verdicts of job O019: D4 was the one item offered, "one sentence with a number already of record"; the author:
  "ship the D with v5 changelog and build release as Data fix". D7–D11 are rejected by the author's ruling; D1, D2, D3, D5 are NOT; D6 was
  done where it was a bug (v5.3).
- **Changed (one sentence and its data note, `chapters/06_results.tex` §6.1.2):** after the sentence on why the endpoint solve is cheap:
  "The endpoint solves also converge: across the six endpoint cells of Table 6.3, $0.1$ to $1.1\,\%$ of them returned their last iterate without
  converging (§4.5.1)." The number is of record — `Data_Analysis/DA_in_Paper/analysis/DA_20260924_R16_R36_must_need.md` §5.3 counts the
  non-converged SLSQP solves per endpoint cell: $189/17\,568$ ($1.08\,\%$), $36/17\,796$ ($0.20\,\%$), $80/7\,900$ ($1.01\,\%$) for the R36 cells and
  $0.64$, $0.32$, $0.09\,\%$ for the three printed earlier — the source's own data note under Table 6.3 already cited the R36 figures. No table
  cell, no other number changed; nothing was run.
- **Checked:** `tools/check.py` 17 files, 7798 lines, 277 labels, 53/53 citations, 41 figures, all pass; `tools/make_release_v5.py
  --dry-run` 19 + 48 files, 9 holes, 1 finding, 4 residue hits. **Not compiled.** Record: `changelogs/v5.7_20260925_review_D_data_fix.md`.
- **Release (DATA FIX):** `RELEASE/output/20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX/` (job O021); page estimate ~179 (152-215); `diff -rq` against the v5.6 build: 1 file(s) differ — `chapters/06_results.tex`; no file only on one side.
- **Left open:** nothing of group D (D1, 2, 3, 5 NOT; D7–D11 rejected by the author); the ⛔ rows and 🟡 remainders of C; the held B-i rows.
  Nothing committed. Signed: Orchestra (Claude Fable 5.1, Claude Code), O020 · 2026-09-25.


## v5.6 — 2026-09-25 · FACTUAL CHANGE · source: the first-reading audit, answer group C under the conservative verdicts (14 DO + 7 minima) · two self-contradictions scoped or removed, three technical statements corrected against the code, the CI-MeanFM mechanism marked as an untested account, scope and specification qualifiers added; no number of record changed (Orchestra O017) → [`changelogs/v5.6_20260925_review_C_factual_change.md`](changelogs/v5.6_20260925_review_C_factual_change.md)

- **Source and kind: FACTUAL CHANGE.** The content rows of the review answer (`…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` §9 C; the
  first-reading audit of the golden release, §1 C1–C12 and §2 M1–M9) under the conservative verdicts of job O016: the 14 ✅ DO rows and the
  minima of the 7 🟡 rows; the 3 ⛔ rows untouched. The author: "Go new changelog for V5 and release, mark as Factual Change."
- **What changes in what the thesis says (34 edit operations, eight files):** two statements that contradicted the thesis's own table or
  sourcing rule are scoped or removed — the CI-MeanFM plateau now reads "from two to twenty evaluations … only at a hundred does it"
  (Table 6.5) in §6.2.1 and §6.2.3 (Ci-4), and the seven / two counts with no evidence of record are gone (Ci-5); three technical
  statements are corrected against the code — the mismatch $w_t$ is the *change* of the tracking error (Ci-1), the MeanFM anchor case
  reduces the *target*, not the objective (Ci-2), the keep-out row acts on the horizontal components, a vertical cylinder (Ci-3); the
  CI-MeanFM mechanism is stated as an account "consistent with … not tested here" in §6.2.1, §6.2.3 and §6.4 (Ci-7, with B-i-4); and
  qualifiers are added without new numbers — "on ten training-set contexts" in Ch 7 (twice) and Ch 8 (Cii-1), the stride-one windows
  (Cii-2), the context-7 guard in §5.5.2 and three captions (Cii-3), "controller-and-simulator cost" in four places (Cii-4), the per-axis
  reach in §4.6.3 and Fig. A.1's caption (Cii-5; the only figure it names is ≈ 0.36 m, the radial reach UAV-pillars already prints),
  the margins not calibrated here (Cii-6, three places), the unconstrained velocity channels (Cii-7), the horizon item for switched walls
  (Cii-8), the "larger, therefore conservative" sentence deleted (Cii-9), the selection-on-the-grid sentence (Cii-10), "cannot be told
  apart" for "does not matter" in Ch 7 (Cii-12), "as trained" in RQ1 (Cii-13), the solver and controller settings (Cii-15), "in the
  inherited implementation" in Ch 2 (Cii-16). **No number of record changed; no table cell, no figure.**
- **Not done (⛔ or the 🟡 remainders):** the abstract (Cii-1, Cii-16), x/9 in the rows and the recount (Cii-3, D2), the "4.8 cm past a disk"
  sentence (Cii-5), the "fronts of observed means" sweep (Cii-9), the causal rewrite of the demonstration-source account (Cii-12), the
  goal column of Table 6.12 (Cii-11), the K / NFE sweep (Cii-14), the Ch 8 paragraph and Table C.1 row (Cii-17), moving the account to
  Ch 8 (Ci-7), the recount of the seven / two (D1).
- **Checked:** `tools/check.py` 17 files, 7793 lines, 277 labels, 53/53 citations, 41 figures, all pass; `tools/make_release_v5.py
  --dry-run` 19 + 48 files, 9 holes, 1 finding, 4 residue hits. Every edit before → after: `changelogs/v5.6_20260925_review_C_factual_change.md`.
  **Not compiled.**
- **Release (FACTUAL CHANGE):** `RELEASE/output/20260925_212026_thesis_release_ORCH_v5.6_FACTUAL_CHANGE/` (job O018); page estimate ~179 (152-214); `diff -rq` against the v5.5 build: 7 file(s) differ — `chapters/02_background.tex`, `chapters/04_method.tex`, `chapters/05_setup.tex`, `chapters/06_results.tex`, `chapters/07_conclusion.tex`, `chapters/08_discussion.tex`, `chapters/09_appendix.tex`; no file only on one side.
- **Left open:** the ⛔ rows and 🟡 remainders above (your word each); the held B-i rows; group D. Nothing committed.
  Signed: Orchestra (Claude Fable 5.1, Claude Code), O017 · 2026-09-25.


## v5.5 — 2026-09-25 · B WRITING (length) · source: the first-reading audit, answer groups B-ii / B-iii relaxed for length · 47 edits: duplicates cut in Ch 1, 3, 4, 5, 6, 7, three tables merged, two tables and three figures to the appendix; no claim or number changed (Orchestra O014) → [`changelogs/v5.5_20260925_review_B_writing_length.md`](changelogs/v5.5_20260925_review_B_writing_length.md)

- **Source and kind: B WRITING (length).** The relaxed rows of the review answer's groups B-ii / B-iii (`…_GOLDEN_TEMPLATE/feedback/
  CLAUDE_ANSWER_…md` §9; the first-reading audit of the golden release, §4–§7), on the author's word: "Go. do B-ii … mark what is
  been done and which is not … then release mark as B writing." Wording, duplication and the placement of floats; **no claim, no
  number of record changed** — numbers were deleted only where the same sentence restated them, and the merged table carries the
  cells of the three it replaces.
- **Changed (47 edits, eight files):** B-ii-1 the six contributions keep their number and order but lose their result narratives
  (a section pointer each); RQ1's avoiding numbers → a table pointer. B-ii-2 three Related-Work passages that re-described DPCC's loop,
  the deployment-time budget and the geometric controller → a clause and a cross-reference each. B-ii-3 the one MeanFM display no
  `\eqref` cites (`eq:method:engine:mfid`) removed. B-ii-4 §5.2 keeps numbers, pictures and protocol and points to §4.6 for the
  definitions: the avoiding-geometry paragraph, the "only be seen" paragraph, the pillars item, the rotor-reach paragraph, the figure
  walkthrough and one datasets sentence shortened or cut. B-ii-7 §6.4 compressed inside itself: the "one picture" paragraph (a third
  telling), the duplicate avoiding numbers in the combination list, one "one episode in thirty"; the synthesis table **not** made
  (your v3.95: "some parts of 6.4 are not suitable for a table"). B-ii-9 one interpretive caption sentence (Table 6.8). B-ii-10
  Tables 6.14–6.16 → one table (`tab:uav-controller`; the planner and loop times as two columns; references re-pointed); the prose of
  Tables 5.8 / 5.9 into the text. B-ii-5 Tables 6.6 / 6.7 → App B (the ladders). B-ii-11 Figure 6.1 → its $\nfe=1$ row in the body,
  the full grid in App B; Figures 6.5 and 6.7 → App B whole. B-iii-3 one clause at Chapter 7's first "Pareto-dominates".
- **Not done (and why):** B-ii-6 (Table 6.11 merge: DA + no pages), B-ii-8 (the vocabulary sweep: no pages), B-ii-12 (declined),
  B-ii-13 (vector export: your machine), Table 6.2's split (the text reads it row by row), §6.4 as a table (v3.95), the DA panel cuts
  for Figures 6.5 / 6.7 (moved whole instead); B-iii-1, 2, 4–7, 9 (no target or no pages).
- **Honest size:** the source went from 7 835 to 7764 lines (71 fewer); the release tool's page estimate is in the
  build below. Text removed ≈ 3 pages of the PDF; floats moved to the appendix ≈ 5 pages of the body — less than the ≈ 11–14 the
  reviewer's spans suggested, because §4.6 / §5.2 and §6.4 duplicate less than they seemed to.
- **Checked:** `tools/check.py` 17 files, 7764 lines, 277 labels, 53/53 citations, 41 figures, all pass (the moved labels
  resolve; `fig:raw-plans-full`, `app:avoiding-plans`, `app:aligning-ladders`, `app:uav-pillars-paths`, `app:uav-scurve-plans` new);
  `tools/make_release_v5.py --dry-run` 19 + 48 files, 9 holes, 1 finding, 4 residue hits. Every edit before → after:
  `changelogs/v5.5_20260925_review_B_writing_length.md`. **Not compiled.**
- **Release (B WRITING):** `RELEASE/output/20260925_210607_thesis_release_ORCH_v5.5_B_WRITING/` (job O015, the author: "then release mark as B writing"); page estimate ~177 (151-213)
  against ~180 for the v5.4 build; `diff -rq` against the v5.4 build: 7 file(s) differ — `chapters/01_introduction.tex`, `chapters/03_related_work.tex`, `chapters/04_method.tex`, `chapters/05_setup.tex`, `chapters/06_results.tex`, `chapters/07_conclusion.tex`, `chapters/09_appendix.tex`; no file only on one side.
- **Left open:** the held B-i rows; group C (🔴 6 · 🟠 9 · 🟢 9) and D on the author's ticks. Nothing committed.
  Signed: Orchestra (Claude Fable 5.1, Claude Code), O014 · 2026-09-25.


## v5.4 — 2026-09-25 · B WRITING FIX · source: the first-reading audit (answer group B-i, the low-risk wording items) · 27 wording edits in Ch 1, 2, 4, 6, 7, 8; no claim, no number, no table cell changed (Orchestra O010) → [`changelogs/v5.4_20260925_review_B_writing_fix.md`](changelogs/v5.4_20260925_review_B_writing_fix.md)

- **Source and kind: B WRITING FIX.** The low-risk, minor wording items (group **B-i**) of the first-reading audit of the golden release
  (`RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md` §4 W1–W3, W6, §5, §6; sorted in
  `feedback/CLAUDE_ANSWER_…md` §9 B-i). Wording only: no claim, no number of record, no table cell, no figure changed.
- **Changed (27 edits, six files):** the two Introduction aphorisms made literal (B-i-1, 2); the seven counting openers of Ch 4 without their
  counts (B-i-7); §6.1.1's three reading instructions (B-i-8); "the better projector on the constraint" names its metric where it stood alone,
  three places in Ch 6 (B-i-9); the shorthand "closes N % of the distance" → the defined reduction of the median, six places in Ch 6 (B-i-10);
  the §6.4 slogan on projection cost made literal (B-i-3); "a plant that holds nothing by itself" (Ch 7, B-i-5); Ch 8's opener (B-i-6); the
  arm / quadrotor contrast of §2.4 said once (B-i-11); Table 6.14's caption defines its dash (B-i-14).
- **Held (decisions, not minor):** B-i-13 `10/10` — your v3.83 instruction (rates ±) against your Ch 5 rule (counts get no ±); B-i-17 the appendix
  captions carry protocol by your v3.49 rule; B-i-4 tied to Ci-7; B-i-9's abstract instance; B-i-12 prunes numbers; B-i-15 / 16 are DA work.
- **Why:** the author (2026-09-25): "start on B … we may no risk fix B-i first" · "Go. do B-i." · "in B update, we build a new changelog and
  mark as B writing fix". Detail per edit (before → after, line): `changelogs/v5.4_20260925_review_B_writing_fix.md`.
- **Checked:** `tools/check.py` 17 files, 7 835 lines, 275 labels, 53/53 citations, 41 figures, all pass; `tools/make_release_v5.py --dry-run`
  19 + 48 files, 9 holes, 1 finding, 4 residue hits — as v5.3. **Not compiled.**
- **Release (temporary review marker):** `RELEASE/output/20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi/` (job O011; the author:
  "release it (as a temp marker, I will use for review now for the B-i changes in PDF)") — not a milestone; 19 + 48 files, ~180 pages
  estimated, 9 holes, 1 finding, 4 residue hits; against the v5.3 build `…_BUGFIX_A` only the six chapter files v5.4 edited differ.
- **Left open:** the held rows; B-ii / B-iii / C / D on the author's ticks. Nothing committed.
  Signed: Orchestra (Claude Fable 5.1, Claude Code), O010 · 2026-09-25.


## v5.3 — 2026-09-25 · **PURE BUG FIX** · source: the same first-reading audit (answer group A, items A6 and A9) · Figures 5.10 / 5.11 (clipped text) and 6.4 (palette) rebuilt in the DA store and re-exported; no text change (Orchestra O007) → [`changelogs/v5.3_20260925_review_tier_a_figures_at_source.md`](changelogs/v5.3_20260925_review_tier_a_figures_at_source.md)

- **Source and kind: PURE BUG FIX.** The two group-A items of the first-reading audit of the golden release — `RELEASE/output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md` (a reviewing agent, 2026-09-25, PDF only), answered and sorted in `feedback/CLAUDE_ANSWER_to_THESIS_FIRST_READING_REVIEW_20260925_1310_…md` §9 that v5.2 could not make inside the draft (A6 clipped
  text in two exported figures, A9 one frontier figure coloured unlike the other three), fixed at the figure source. No data,
  marker, axis, number or caption changed.
- **Changed (figures only; no `.tex` of v5 touched):** three figures rebuilt at their source in `Data_Analysis/DA_in_Paper` and
  re-exported (`plotting/export_to_draft.py <v5>`: 6 files copied, 55 current; `figures/EXPORTED.md` regenerated). **A6** Figure 5.10
  (`fig_expert_aligning`): the subtitle ran past the 620 px canvas ("… none the l") → two lines, the top margin widened to hold them
  (`builders/expert.py`, PNG 1860×2817 → 1860×2874). **A6** Figure 5.11 (`fig_expert_uav`): the third legend key ran past the
  canvas ("… is in the surfa") → `_legend` wraps a key at `\n`, the key split in two (PNG 2844×3402 → 2844×3498; the v5.2 height
  cap absorbs the extra 96 px). **A9** Figure 6.4 (`fig_aligning_projected_tradeoff`): its private palette (`#1F4E79 / #8B3F71 /
  #C45B24`) replaced by `sources.ENGINE_COLOUR_DISTINCT`, so CI-MeanFM is teal, MeanFM blue and FM orange as in Figures 6.2 / 6.3 /
  6.6 (`builders/frontier.py`; the SVG diff is colour strings only; PNG 2280×1500 unchanged).
- **Why:** the author (2026-09-25): "is the still open A pure bugs and zero chance of other issues? if pure bug, just fix and back
  to source." A6 and A9 are pure bugs (clipped text; a palette no decision asked for — v3's changelog names none) and were fixed at
  the source, not in the draft, as the DA rule requires. **Not done, not bugs:** Figure 6.7's empty sixth panel (cosmetic; the page
  fit is v5.2's height cap), A11 (the author's front-matter metadata), the two RELEASE-tool items (code hardening).
- **Verified:** the store's PNGs are exactly `plotting/svg/preview_png.py --scale 3` (a re-render of two unchanged figures was
  pixel-identical to the store), so the re-rendered PNGs are the pipeline's, not a preview; the three new PNGs were read by eye
  (subtitle and key inside the canvas; the palette as in Figure 6.2); the data behind the expert figures (`data/expert_paths.json`)
  and the frontier's corpus are unchanged, so nothing but text layout and colour moved. `figures/MANIFEST.md` restored after the
  partial builds (a matched `make_figs.py` run rewrites it with only the matched rows).
- **Checked:** `tools/check.py` 17 files, 7 833 lines, 275 labels, 53/53 citations, 41 figures, all pass; `tools/make_release_v5.py
  --dry-run` 19 + 48 files, 9 holes, 1 finding (`\getDoctype`), 4 residue hits — as v5.2. **Not compiled.**
- **Release:** `RELEASE/output/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A/` (job O008, on the author's word: "release it. (mark
  from v5.2/bug fix A)") — the PURE BUG FIX build of the audit's tier A, v5.2 + v5.3; 19 + 48 files, ~179 pages estimated, 9 holes,
  1 finding, 4 residue hits; against the golden release's `latex/` only the six files v5.2 / v5.3 changed differ. Not compiled here.
- **Left open:** groups B, C, D of the review answer (the author's ticks); A11; the tooling items. Nothing committed (the DA store's
  six figure files and two builders are modified in the working tree for the author to commit).
  Signed: Orchestra (Claude Fable 5.1, Claude Code), O007 · 2026-09-25.


## v5.2 — 2026-09-25 · **PURE BUG FIX** · source: the first-reading audit of the golden release `20260925_115125_…_GOLDEN_TEMPLATE` (`feedback/THESIS_FIRST_READING_REVIEW.md`, answer group A) · the format and consistency bugs A1–A20 in Ch 4–6; no content change (Orchestra O006) → [`changelogs/v5.2_20260925_review_tier_a_format_bugs.md`](changelogs/v5.2_20260925_review_tier_a_format_bugs.md)

- **Source and kind: PURE BUG FIX.** Every item comes from the first-reading audit of the golden release — `RELEASE/output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md` (a reviewing agent, 2026-09-25, PDF only), answered and sorted in `feedback/CLAUDE_ANSWER_to_THESIS_FIRST_READING_REVIEW_20260925_1310_…md` §9, group A ("pure writing / format bugs"): a lost
  equation label, a printed audit identifier, four floats over the page, two captions misdescribing their figure / table, one
  retired wording, nine sentences contradicting the table beside them. No claim, no number of record, no figure content changed;
  the audit's content items (answer groups B–D) wait for the author's ticks.
- **Changed (20 edits, three files; the drafting macros and every number of record untouched):** `chapters/04_method.tex` — A1 the FM-loss
  display carried two `\label`s (amsmath keeps only the last, so the three `\eqref{eq:bg:fm:loss}` printed `??`): one label, the one
  `\eqref{eq:method:engine:fm}` re-pointed. `chapters/05_setup.tex` — A12 "Every model is trained once … the same checkpoint under every
  budget" → "Every flow-based model …; the diffusion baseline's budget is fixed with its noise schedule at training, so each of its budgets
  is a separately trained checkpoint"; A5 Figures 5.8 and 5.11 (`fig_constraints_uav`, `fig_expert_uav`) capped at `height=0.70\textheight`
  (their captions sat on the footer rule). `chapters/06_results.tex` — A2 the printed `(CLOSURE\_20260907 R.4)` removed (provenance stays in
  the `\dataref`); A3 Table 6.12's caption shortened (tie-break order and the goal-point sentence, both in the text / App B.4); A4 Figure 6.7
  capped at `0.70\textheight`; A7 Figure 6.2's caption: "triangles" → "squares", hollow points defined; A8 Table 6.6's caption: ms/step is a
  mean, not mean ± SD; A10 "the line at which the box was not moved at all" → "the $0\,\%$ line"; A13 "Every comparison … five seeds" gains
  "except where a table states otherwise" with Table 6.3's four / one seeds and its twenty-episode cell; A14 $0.100$ → $0.133$; A15 $19.0$ →
  $20.4$; A16 "ten" → "eleven" gated steps at the baseline's budget; A17 "four of the ten" → "of the thirteen"; A18 "comes closest at
  $\nfe=20$" → "$\nfe=20$ is its operating budget — the $\nfe=100$ point ends only $0.0068$\,m closer at $4.7$ times the cost"; A19 "do not
  move the box" → "barely move the box (medians $14$ and $10\,\%$, unmoved in two and four of ten)" in both places; A20 "the one place in the
  thesis where …" → "Here a flow-based model is ahead …".
- **Why:** the author (2026-09-25): "Init a Changelog and update in v5 for the tier A. and no need to release. after fix also mark in the
  Answer md saying been fixed." Tier A = group A of the review answer
  (`RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` §9): the pure writing / format bugs and the nine sentences
  that contradict the table beside them. Every edit was re-verified against the v5 table or definition it cites before it was made.
- **Not in this pass (not v5's):** A6 (Figures 5.10 / 5.11 re-exported by the DA), A9 (one colour per model, DA), Figure 6.7's relayout
  (DA), A11 (the author's front-matter metadata), the two RELEASE-tool items (code).
- **Checked:** `tools/check.py` 17 files, 7 833 lines, **275 labels** (276 minus the dropped duplicate), 53/53 citations, 41 figures, all
  pass; drafting macros unchanged (hole 1, provisional 1, guard 19, srcnote 78, dataref 60, flawed 2, longdata 1). `tools/make_release_v5.py
  --dry-run` 19 + 48 files, 9 holes, 1 finding (`\getDoctype`), 4 residue hits — as before. The three float fixes and the caption cut are
  sized from the reviewer's page images and the PNG aspect ratios (1.131, 1.196, 1.408 against a text block of about 1.465); only a compile
  confirms them. **Not compiled.**
- **Left open:** groups B, C, D of the answer (the author's ticks); the DA items above. No release built; nothing committed; no file of
  v2 / v3 / v4 touched. Signed: Orchestra (Claude Fable 5.1, Claude Code), O006 · 2026-09-25.


## v5.1 — 2026-09-25 · The seven INBOX rows open for v2 / v3 / v4 resolved in v5: absorbed by construction, answered, or not applicable; no text change (Orchestra O004) → [`changelogs/v5.1_20260925_inbox_rows_resolved.md`](changelogs/v5.1_20260925_inbox_rows_resolved.md)

- **Resolved in v5 (no thesis text changed):** the seven INBOX rows open on 2026-09-25 — → v2: Orchestra O001 (FYI; superseded by
  O003) · → v3: Orchestra O001 (superseded), v4.2 (two Ch 6 FYI items: the budget convention is stated in Ch 5 `sec:setup:protocol:eval`
  l. 1262–1269 and Ch 6 l. 83–87, so "at twenty evaluations" names $\nfe = 20$ — no change; the 18.1 ms sentence at `06_results.tex:437–439`
  already names analytic average-velocity matching — no change), v2.28 (absorbed by construction: v5 holds v2.28's Ch 1–4, abstract and
  acronym list since v5.0; MJPC gone, PD declared and used) · → v4: Orchestra O001 (superseded), v3.100b (not applicable: v5 has no
  inherited copies and no sync tool; the cross-chapter references resolve in `check.py`; `v4/tools/sync_v3.py` untouched), v2.28
  (absorbed by construction). Each row is marked `🔀 v5.1` with its resolution; the three O003 FYI rows stay open for the owner chats.
- **Why:** the author (2026-09-25): "First job is to resolve any the INBOX for v2,v3,4. to v5.1."
- **Checked:** `tools/check.py` 17 files, 7 824 lines, 276 labels, 53/53 citations, 41 figures, all pass; `tools/make_release_v5.py --dry-run`
  9 holes, 1 finding (`\getDoctype`), 4 residue hits; `tools/absorb.py status` nothing to absorb — all unchanged from v5.0. **Not compiled.**
- **Left open:** the Ch 6 K / NFE wording sweep and the rest of the parked review answer (job O002); the author's decisions listed under v5.0.
- No file of v2 / v3 / v4 touched; no release built; nothing committed. Signed: Orchestra (Claude Fable 5.1, Claude Code), O004 · 2026-09-25.


## v5.0 — 2026-09-25 13:43 · v5 initialised: the aggregate of v2.28 · v3.100b · v4.2 (Advance Orchestra, job O003) → [`changelogs/v5.0_20260925_init_from_v2.28_v3.100b_v4.2.md`](changelogs/v5.0_20260925_init_from_v2.28_v3.100b_v4.2.md)

- **Built from:** v2 **v2.28** (2026-09-24 · the seven open cross notes applied) → `parts/00_preamble`, `01_frontmatter` (abstract),
  `99_backmatter` (acronyms), `chapters/01–04`, `bibliography.bib` (the monolith split at its markers); v3 **v3.100b** (2026-09-25 ·
  v3's copies of Ch 7–9 archived) → `chapters/05–06`, `parts/00_preamble_v3`, `bibliography_v3.bib`; v4 **v4.2** (2026-09-25 · the
  audit of v4.1a applied) → `chapters/07–09` (+ `app_long/`, `app_ntrial20_feasible`), `parts/00_preamble_v4`, `bibliography_v4.bib`.
  Byte-identical to the sources of the golden release `20260925_115125_…_GOLDEN_TEMPLATE`. Record: `inherited/INIT_STATE.json`,
  `inherited/MANIFEST.md`; the copies in `inherited/materials/` are the merge base for later legacy changes.
- **Why:** the author (2026-09-25): the major parts of the thesis are set; the split v2 / v3 / v4 flow is kept but used less; the
  Orchestra is the major workspace "from the thesis → release", all together, with the RELEASE kept and the build marked as the
  Orchestra's, the date-time as a build's identity, and a changelog from v5.0 on.
- **Tools:** `tools/check.py` (v4's, re-pointed), `tools/make_release_v5.py` (builds `RELEASE/output/<stamp>_thesis_release_ORCH_v5.N/`
  by importing the legacy release tool), `tools/absorb.py` (a legacy change three-way-merged into v5). Runbook: `README.md`.
- **Checked:** `check.py` 17 files, 7 824 lines, 276 labels, 53/53 citations, 41 figures, all pass; `make_release_v5.py --dry-run`
  9 holes, 1 finding (the standing `\getDoctype` patch), 4 residue hits; **a v5.0 test build reproduces the golden release's `latex/`
  tree byte for byte** (67 files, `diff -rq` clean; page estimate ~179); `absorb.py status` nothing to absorb. **Not compiled.**
- **Pending at init:** seven open INBOX rows (→ v2 one, → v3 three, → v4 three) — resolved in v5.1; the author's decisions in
  `v4/notes/OPEN_20260924_v4_open_items.md`; the parked review answer (job O002); the `\getDoctype` finding.
- No file of v2 / v3 / v4 changed; nothing under `RELEASE/output/` touched; nothing committed.
  Signed: Orchestra (Claude Fable 5.1, Claude Code), O003 · 2026-09-25.
