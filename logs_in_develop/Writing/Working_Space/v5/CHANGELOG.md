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
