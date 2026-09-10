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
