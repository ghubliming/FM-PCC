# v3 — the experiments

**Created:** 2026-09-10 · **Master file:** [`thesis_v3.tex`](thesis_v3.tex) · **Change history:** [`CHANGELOG.md`](CHANGELOG.md)
**Branched from:** [`../v2/thesis_v2.tex`](../v2/thesis_v2.tex) at **v2.6** — see [`inherited/SYNC_STATE.json`](inherited/SYNC_STATE.json)
**Governed by:** [`../TARGET_20260905_thesis_claim_ladder.md`](../TARGET_20260905_thesis_claim_ladder.md) (goals) ·
[`../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md`](../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md) (**fired** — the active engine plan)
**What may be written:** [`../data_status/DATASTATUS_20260910_v3_entry_readiness.md`](../data_status/DATASTATUS_20260910_v3_entry_readiness.md)
**Vocabulary:** [`../../Auxiliary/Naming/NAMING_20260910_master_table.md`](../../Auxiliary/Naming/NAMING_20260910_master_table.md) — canonical

---

## What v3 is

v1 fixed the argument. v2 wrote the mathematics it stands on. **v3 writes the experiments**:
Chapters 5–8 stop being bone.

The foundation entry — the state-based manipulation benchmark — is written **in full**, because it is
the only multi-seed entry and it carries the principal claim. The vision-conditioned and aerial
entries are written **to the strength their data actually supports**, and every guarded claim says so
in its own sentence rather than in a footnote.

| chapter | v2 | v3 |
|---|---|---|
| 1 Introduction · 2 Background · 3 Related Work · 4 Method | filled + mathematics + citations | **inherited unchanged** |
| 5 Experimental Setup | bone, except the compute environment | **written** — tasks, the regime taxonomy, the pinned baseline, the definition of "better", the two protocol tiers |
| 6 Results | bone | **written** — foundation entry in full; the other two to grade |
| 7 Discussion | bone | **written** — interpretation, negative results with mechanisms, threats, limitations |
| 8 Conclusion | bone | **written** — RQ answers settled; headline numbers left for last |
| Appendix | `app:repro` opened | **extended** — the reverse name map, the corpora of record |

3634 lines against v2's 2151. 6 generated figures, 5 new tables, 12 `\hole`s, 5 `\provisional`s.
No dangling reference, no missing bib key, no unbalanced environment. **Never compiled — there is no
TeX toolchain in this container.**

---

## v2 and v3 run in parallel. Here is how that works.

**The problem.** v2 is one 2 151-line file. If v3 were a copy of it, then every later v2 pass on
Chapters 1–4 would have to be re-applied to v3 by hand, and the two drafts would silently diverge on
the mathematics.

**The fix.** v3 is v2 **split by chapter**, plus a merge base, plus a tool. v2 remains the upstream
for Chapters 1–4; v3 owns 5–8. When v2 moves, `sync_v2.py` says so and merges it down.

```bash
python3 bundle/make_bundle.py      # flatten to ONE .tex + an Overleaf-ready .zip
python3 tools/sync_v2.py status    # has v2 moved? which files? which need a real merge?
python3 tools/sync_v2.py diff      # what exactly did v2 change
python3 tools/sync_v2.py merge     # three-way-merge it in, advance the baseline, re-stamp
python3 tools/check.py             # labels, citations, environments, braces, figures
python3 plots/make_figs.py         # rebuild every figure from the batch CSVs
```

`status` prints which v2 revision the inherited half is at, so the answer to *"which v2 is in this
v3?"* is one command, not an archaeology exercise.

### Who owns what

Full policy table with rationale: [`inherited/MANIFEST.md`](inherited/MANIFEST.md). In short —
`inherit` = v3 never edits it, a v2 change fast-forwards; `merge` = both edit it, a v2 change needs a
three-way merge; `own` = v3 wrote it, and the baseline exists only so that drift becomes *visible*.

The split is clean at chapter granularity, so **every chapter has exactly one owner and no temporary
"aggregation section" is needed**. The single place where v3 writes inside an inherited chapter is
`sec:bg:fewstep`, which is why `02_background.tex` carries the `merge` policy.

### The bibliography is split for the same reason

- `bibliography.bib` — **inherited from v2 byte-for-byte.** v3 never appends to it.
- `bibliography_v3.bib` — **v3's own entries.** v2 never sees it.

Both drafts appending to one `.bib` is the worst possible conflict shape: two sides adding at the end
of the same file, recurring on every pass. Keeping them separate makes the inherited half a
permanent fast-forward. biblatex merges the two resources at build time and biber reports a key
defined in both — `tools/check.py` catches that case earlier and more cheaply.

Which v2 revision the inherited half is at lives in `inherited/SYNC_STATE.json`, stamped by the tool
rather than typed by hand.

### The one-way rule

`tools/` reads `../v2/` and never writes to it. **A fix that belongs in both drafts is made in v2 and
synced down.** Making it in v3 first strands it — there is deliberately no tool pointing the other
way, because two-way sync between two live drafts is how both get corrupted.

---

## Layout

```
thesis_v3.tex          master: preamble, \input order, nothing else
parts/
  00_preamble.tex      INHERITED byte-for-byte — do not edit, or every sync conflicts
  00_preamble_v3.tex   v3-only packages and macros. This is where v3 preamble edits go.
  01_frontmatter.tex   INHERITED
  99_backmatter.tex    MERGE — one acronym environment, and v3 declares more
chapters/01..09        01–04 inherited · 05–09 v3
bibliography.bib       INHERITED · bibliography_v3.bib  v3's own
inherited/
  v2_base/             the merge base: v2 as it stood at the last sync. Never edit by hand.
  SYNC_STATE.json      which v2 revision that is, and when
  MANIFEST.md          per-file policy, with the reasoning
tools/
  split_v2.py          content-based splitter (markers, never line numbers)
  sync_v2.py           status / diff / merge / stamp
  check.py             mechanical checks — NOT a compiler
  svg2pdf.sh           figures for the LaTeX build
plots/                 the figure pipeline — see plots/README.md
figures/               generated SVG + MANIFEST.md saying what came from where
bundle/                flattened builds — see bundle/README.md
  make_bundle.py       inlines every \input, verifies, zips for Overleaf
  thesis_v3_<stamp>.tex/.zip   build output. NEVER edit these; rebuild instead.
```

## Compiling it

The split layout is for editing. **To build or upload, flatten it first:**

```bash
python3 bundle/make_bundle.py                 # -> bundle/thesis_v3_<stamp>.tex + .zip
python3 bundle/make_bundle.py --svg-package   # same, with the SVG figures rendered
python3 bundle/make_bundle.py --verify        # prove a bundle matches the tree, byte for byte
```

Upload the `.zip` to Overleaf — it carries the flat `.tex`, both `.bib` files and `figures/`.
Compiler **pdfLaTeX**, bibliography **Biber**. Full detail, including the two figure modes and why
`--svg-package` is opt-in, in [`bundle/README.md`](bundle/README.md).

The bundle is **build output**: never edit it, and nothing reads it back. Its header lists every
source file with a SHA-256 prefix, so any bundle can be traced to exactly what produced it.

## Drafting macros — all four must be gone before submission

`tools/check.py` counts them on every run.

| macro | means | v3.0 |
|---|---|---|
| `\hole{...}` | inherited from v2: something not yet written | 12 |
| `\provisional{...}` | **new in v3.** True of today's data, below the strength the thesis wants, and it will move when a named run lands | 5 |
| `\guard{...}` | **new in v3.** The caveat that must travel with a claim *wherever it is quoted* — seed depth, protocol tier, a selection effect. In the running text, not a footnote, because a footnote gets dropped when the sentence is quoted | 17 |
| `\dataref{...}` | **new in v3.** The *evidence* of record — batch and DA. v2's `\srcnote` points at code and answers "how is this implemented"; `\dataref` points at data and answers "where does this number come from" | 18 |

---

## What v3.0 deliberately did not do

1. **`sec:bg:fewstep` still says the bootstrapped target's mathematics is "deliberately deferred".**
   Under the fired fallback that mathematics *is* the contribution, and `DATASTATUS §7` asks for the
   deferral to be reversed. v3.0 leaves Chapter 2 byte-identical instead, and writes the mechanism
   where the result is (`sec:disc:negative`). **This is the first item of v3.1** — doing it half-way,
   as a note rather than as the derivation, would have been worse than not doing it.
2. **Entry-2 and Entry-3 figures.** Their sections quote paired tests and per-scene tables, and both
   entries have owed runs that would change what the right figure is. Their corpora are already
   registered in `plots/sources.py`.
3. **The abstract and the headline numbers.** Rewritten last, in whichever version ships.
4. **The cross-entry synthesis table** in `sec:res:summary`. Writing it now would put three results
   of very different strengths into one visual register, which is the thing `sec:setup:metrics`
   exists to prevent.

## What would change the text when it lands

Ordered as `DATASTATUS §8` orders it — cheapest and most unblocking first.

| owed run | what in v3 changes |
|---|---|
| Entry 2 at 50 contexts | every Entry-2 test moves from 10 paired samples to 50 |
| seeds on the aerial pillars scene | `sec:res:uav`'s `\provisional` ranking becomes a claim |
| the published-protocol tier on Entry 1 | `tab:tiers` gains its transport rows; the budget-ladder mechanism sentence becomes a powered comparison |
| the 3-D **state** aligning leg | `sec:disc:limitations` loses its largest item, and RQ4 gains the state-to-vision transfer |
| a budget sweep of the endpoint arm at 5 seeds | `sec:res:constraints`'s `\provisional` becomes a Pareto claim |
