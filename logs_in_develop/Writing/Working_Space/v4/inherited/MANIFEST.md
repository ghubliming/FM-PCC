# Inheritance manifest — who owns which file (v4)

**`v3_base/`** is a pristine snapshot of `../v3/` (parts, chapters, both `.bib` files) **as it stood
when v4 last synced** — at branch time, **v3.98 (2026-09-24 20:19)**, which carries **v2.27
(2026-09-23)** in its inherited half. It is the *merge base*, nothing else. Never edit it by hand;
`tools/sync_v3.py merge` advances it, and only on a clean merge. `SYNC_STATE.json` records the
versions and is written by the tool, not by hand.

```
base    inherited/v3_base/<file>   v3 at the last sync
ours    <file>                     the v4 working copy
theirs  ../v3/<file>               v3 right now (already split; no re-split needed)
```

The flow is one way, **v2 → v3 → v4**: a v2 change is absorbed by v3 (`v3/tools/sync_v2.py`) and
arrives here as a v3 change. `tools/` reads `../v3/` and never writes into `../v3/` or `../v2/`.

## Policy per file (also the table `tools/sync_v3.py` and `bundle/make_bundle.py` read)

| file | policy | owner | meaning |
| :-- | :-- | :-- | :-- |
| `parts/00_preamble.tex` | **inherit** | v2 | byte-identical to v3's copy of v2's. v4 additions live in `parts/00_preamble_v4.tex`. |
| `parts/00_preamble_v3.tex` | **inherit** | v3 | v3's preamble additions (`\dataref`, `\guard`, `\provisional`, …), taken as is. |
| `parts/01_frontmatter.tex` | **inherit** | v2 | title page, abstract, ToC. |
| `parts/99_backmatter.tex` | **merge** | v2 | one `acronym` environment; v4 may declare more (a note goes to v2 when it does). |
| `chapters/01_introduction.tex` … `04_method.tex` | **inherit** | v2 | v2's, through v3. **Collapsed by default in the bundle.** |
| `chapters/05_setup.tex`, `06_results.tex` | **inherit** | v3 | v3's. Built in full in the bundle. |
| `chapters/07_discussion.tex`, `08_conclusion.tex` | **watch** | v3 | v3's frozen copies (headings-only Ch 7; the stale 20-09 Ch 8). v4 carries no file under these names since v4.1; `sync_v3.py status` reports if v3 edits them, nothing is merged. |
| `chapters/07_conclusion.tex` | **own** | v4 | the Conclusion (summary, RQ answers), Chapter 7 since v4.1; no upstream counterpart. |
| `chapters/08_discussion.tex` | **own** | v4 | the Discussion (limitations, towards deployment, future work), Chapter 8 since v4.1; no upstream counterpart. |
| `chapters/09_appendix.tex` | **own** | v4 | restructured in v4 (the long-data switch, Reproducibility as facts); `chapters/app_long/*.tex` hold the four data records it inputs. |
| `chapters/app_ntrial20_feasible.tex` | **own** | v4 | curated by another agent, used as is (author, v3.92). |
| `bibliography.bib` | **inherit** | v2 | v2's entries. |
| `bibliography_v3.bib` | **inherit** | v3 | v3's entries. `bibliography_v4.bib` is v4's and has no baseline. |

**`own` and `watch` keep v3's copy at branch time as a baseline** so that if v3 ever edits its own
Ch 7–9 again, `sync_v3.py status` says so and a human decides; `merge` skips both on purpose.
