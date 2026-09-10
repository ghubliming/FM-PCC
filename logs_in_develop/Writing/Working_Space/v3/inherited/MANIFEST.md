# Inheritance manifest — who owns which file

**`v2_base/`** is a pristine snapshot of `../v2/thesis_v2.tex` **as it stood when v3 last synced**.
It is the *merge base*, nothing else. Never edit it by hand; `tools/sync_v2.py merge` advances it,
and only on a clean merge.

```
base    inherited/v2_base/<file>   v2 at the last sync
ours    <file>                     the v3 working copy
theirs  (v2 re-split on demand)    v2 right now
```

## Policy per file

| file | policy | meaning |
| :-- | :-- | :-- |
| `parts/00_preamble.tex` | **inherit** | byte-identical to v2. v3 additions live in `parts/00_preamble_v3.tex`, which v2 does not have. A v2 preamble change fast-forwards. |
| `parts/01_frontmatter.tex` | **inherit** | title page, abstract, ToC. The abstract is rewritten last, in whichever version ships. |
| `chapters/01_introduction.tex` | **inherit** | v2's. |
| `chapters/02_background.tex` | **merge** | v3 edits it — `sec:bg:fewstep` reverses v2's α-Flow deferral (`DATASTATUS §7`, `FALLBACK §6.1`). |
| `chapters/03_related_work.tex` | **inherit** | v2's. |
| `chapters/04_method.tex` | **merge** | v3 discharges v2's `\hole`s in `sec:method:engine` and `sec:method:deployment` from the protocol section. |
| `parts/99_backmatter.tex` | **merge** | one `acronym` environment, and v3 declares more of them. Cannot be split. |
| `bibliography.bib` | **merge** | v3 adds entries; v2 may too. |
| `chapters/05_setup.tex` | **own** | bone in v2, written in v3. |
| `chapters/06_results.tex` | **own** | bone in v2, written in v3. |
| `chapters/07_discussion.tex` | **own** | bone in v2, written in v3. |
| `chapters/08_conclusion.tex` | **own** | bone in v2, written in v3. |
| `chapters/09_appendix.tex` | **own** | v2 opened `app:repro`; v3 extends all three appendices. |

**`own` still keeps a baseline.** Not to merge into — to make drift *visible*. If v2 ever writes
into its own Chapter 5–8 bone, `sync_v2.py status` says so and a human decides. That is the whole
point: silence is never the same as agreement.

## The one-way rule

`tools/` reads `../v2/` and never writes to it. If a fix belongs in both drafts, make it in v2 and
sync it down. Making it in v3 first strands it — v2 has no tool pointing the other way, on purpose:
two-way sync between two live drafts is how both get corrupted.

## Aggregation at the end

The final thesis is **one** of these versions, not a splice. If the two ever have to be recombined:

1. `python3 tools/sync_v2.py status` — see what moved where.
2. `merge` the inherited files until `status` is clean.
3. Chapters 5–8 come from v3 unconditionally; 1–4 are then already identical.
4. `parts/00_preamble_v3.tex` folds into `settings.tex` at template-merge time, together with v2's
   own divergences (`amsmath`/`amssymb`/`amsthm`, listed in `../v2/README.md`).

No chapter needs a temporary "aggregation section": the split is clean at chapter granularity, so
every chapter has exactly one owner. `sec:bg:fewstep` is the single place where v3 writes inside an
inherited chapter, and it is a `merge`-policy file for that reason.
