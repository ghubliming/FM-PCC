# DRAFT OWNERSHIP — which version writes which chapter

**Read this first in any v2, v3 or v4 session.** Three drafts are worked on by separate chats.
Each owns a fixed set of chapters; **no chat edits another draft's chapters or files.**
**Created:** 2026-09-14 · **Revised:** 2026-09-14 (v3.7 — Discussion moved to v4, Conclusion shared) · **Authority:** the author. Change this file only when the author re-divides the work.

---

## The map

| draft | status | **owns — the only things it writes** | where |
| :-- | :-- | :-- | :-- |
| **v2** | 🟢 live | **Ch 1 Introduction · Ch 2 Background · Ch 3 Related Work · Ch 4 Method** · abstract & front matter · preamble · acronym list · `bibliography.bib` | `v2/thesis_v2.tex` |
| **v3** | 🟢 live | **Ch 5 Experimental Setup · Ch 6 Results** · a concise draft of **Ch 8 Conclusion** (v4 refines) · the **appendix** (for now — see below) · `bibliography_v3.bib` · `parts/00_preamble_v3.tex` · bundle and tools | `v3/chapters/05, 06, 08, 09`, `v3/parts/00_preamble_v3.tex`, `v3/bundle/`, `v3/tools/`; `v3/figures/` holds exported copies only |
| **v4** | ⚪ **planned, not started** | **Ch 7 Discussion** (moved from v3, 2026-09-14) · refinement of **Ch 8 Conclusion** and its **future work** · per the author: *next steps, real-world practice, appendix material*. **Starts only after v2 and v3 are finished.** | `v4/` (currently notes only). Until v4 starts, `v3/chapters/07_discussion.tex` holds the headings only |

**Flow is one-way: v2 → v3 → v4.** v3 inherits Ch 1–4 from v2 through `v3/tools/sync_v2.py`. v4 will
branch off the finished v3 the same way.

## Rules for each chat

### v2
- Write only Ch 1–4, the abstract, the preamble, acronyms and `bibliography.bib` in `v2/thesis_v2.tex`.
- **Do not write into v2's own Ch 5–8.** They are placeholder headings; the real Ch 5–8 live in v3.
- **Do not edit anything under `v3/` or `v4/`.**
- If a v2 decision affects Ch 5–8 — a renamed term, a renamed or removed `\label` — **record it in
  `v2/CHANGELOG.md` under a line starting `For v3:`** and stop there. v3 picks it up on its next sync.
  Renaming a `\label` breaks v3's cross-references; say so explicitly when it happens.

### v3
- Write only `v3/chapters/05`, `06`, `08` (concise draft) and `09`, `parts/00_preamble_v3.tex`,
  `bibliography_v3.bib`, and v3's `bundle/`, `tools/`, `handover/`. **Figures are produced only in
  `Data_Analysis/DA_in_Paper/`** (since 2026-09-14) and copied into `v3/figures/` with
  `plotting/export_to_draft.py v3` — never drawn or edited in the draft.
- **Ch 7 Discussion is v4's.** v3 keeps its headings and labels (Ch 4 references them) and writes no
  prose there. What v3 had written is in `v3/handover/HANDOVER_20260914_discussion_prose_for_v4.tex`.
- **Never edit v3's copies of Ch 1–4, `00_preamble`, `01_frontmatter`, `99_backmatter` or
  `bibliography.bib`** — not even the ones whose policy is `merge`. The change belongs in v2; sync it
  down with `python3 tools/sync_v2.py merge`. Editing the copy creates merge conflicts, and the
  default bundle then stops collapsing that chapter.
- **Never edit anything under `v2/` or `v4/`.**
- Sync from v2 only when the author asks. Before merging, read v2's `For v3:` lines.

### v4
- **Do not start drafting until the author declares v2 and v3 finished.** Until then, write only
  notes inside `v4/`.
- When it starts: branch from the finished v3 the way v3 branched from v2 (copy, keep a baseline, sync
  one-way), and add v4's own row to this file.
- **Never edit anything under `v2/` or `v3/`.**

## The overlaps, settled

| topic | who holds it now | handover |
| :-- | :-- | :-- |
| **Appendix** (`app:derivations`, `app:results`, `app:repro`) | **v3** — it has extended `app:repro` (name map, corpora of record) | passes to **v4** when v4 starts. v3 keeps it limited to what the results chapters need |
| **Future work / next steps** (`sec:conc:future`) | **v3** — a `\hole` pointing at `future_work/` | the prose is **v4's**. v3 does not expand it |
| **Discussion** (Ch 7) | **v4** — decided by the author 2026-09-14 | v3 keeps headings and labels only; its earlier prose is in `v3/handover/` for v4 to reuse |
| **Conclusion** (Ch 8) | **v3 and v4** | v3 keeps a concise version that follows Ch 6; v4 refines the prose |

## Shared material — any chat may read, update with care

`Auxiliary/` (the naming table in `Auxiliary/Naming/` is canonical for all drafts), `data_status/`,
`fallback_target/`, `future_work/`, `TARGET_20260905_thesis_claim_ladder.md`.
When one draft's session changes a shared note, it says so in that note and in its own CHANGELOG.
`../Template_DONT_CHANGE/` is read-only for everyone.

## If a request would cross the line

**Stop and say so.** Name the owning draft and the file, and put the needed change as a `For v3:` /
`For v2:` line in your own CHANGELOG instead of making it.

## Open items as of 2026-09-14 (after v3.7)

- v3 synced to **v2.15** on 2026-09-14 (inherited files merged one by one). v2's rename of its
  placeholder Ch 6 heading (*Constraint Arms* → *Projection Methods*) is adopted in v3 as
  `\subsection{Projection Methods}\label{sec:res:constraints}`, and the baseline was advanced.
- `v3/tools/sync_v2.py merge` without a file argument still also merges files v3 owns (Ch 5–9). Merge
  inherited files by name until it is fixed.
