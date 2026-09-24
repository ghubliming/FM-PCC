# DRAFT OWNERSHIP — which version writes which chapter

**Read this first in any v2, v3 or v4 session.** Three drafts are worked on by separate chats.
Each owns a fixed set of chapters; **no chat edits another draft's chapters or files.**
**Created:** 2026-09-14 · **Revised:** 2026-09-14 (v3.7 — Discussion moved to v4, Conclusion shared); 2026-09-24 (v4.0 — v4 live: the endings and the appendix are v4's; v4.1 — Ch 7 = Conclusion, Ch 8 = Discussion) · **Authority:** the author. Change this file only when the author re-divides the work.

---

## The map

| draft | status | **owns — the only things it writes** | where |
| :-- | :-- | :-- | :-- |
| **v2** | 🟢 live | **Ch 1 Introduction · Ch 2 Background · Ch 3 Related Work · Ch 4 Method** · abstract & front matter · preamble · acronym list · `bibliography.bib` | `v2/thesis_v2.tex` |
| **v3** | 🟢 live | **Ch 5 Experimental Setup · Ch 6 Results** · `bibliography_v3.bib` · `parts/00_preamble_v3.tex` · bundle and tools. Its copies of Ch 7–9 are **frozen at v3.98** (handed to v4, 2026-09-24); the live endings are v4's | `v3/chapters/05, 06`, `v3/parts/00_preamble_v3.tex`, `v3/bundle/`, `v3/tools/`; `v3/figures/` holds exported copies only |
| **v4** | 🟢 live (since 2026-09-24, v4.0; order v4.1) | **Ch 7 Conclusion** (summary, RQ answers) · **Ch 8 Discussion** (limitations, *Towards Deployment*, future work) · **the appendix** (handed over by v3 at v3.98, restructured in v4) · `parts/00_preamble_v4.tex` · `bibliography_v4.bib` · bundle and tools | `v4/chapters/07_conclusion.tex, 08_discussion.tex, 09_appendix.tex` + `chapters/app_long/`, `v4/parts/00_preamble_v4.tex`, `v4/bundle/`, `v4/tools/`; `v4/figures/` holds exported copies only. Branched from **v3.98**, which carries **v2.27** (`v4/inherited/SYNC_STATE.json`) |

**Flow is one-way: v2 → v3 → v4.** v3 inherits Ch 1–4 from v2 through `v3/tools/sync_v2.py`; v4
inherits Ch 1–6 from v3 through `v4/tools/sync_v3.py` (branched 2026-09-24 at v3.98 / v2.27). **v3's copies
of Ch 7–9 are frozen at v3.98**: the live Ch 7–9 are v4's.

## Cross-draft findings → `cross_draft/`

When a draft finds something another draft must change, it writes a note in `cross_draft/to_<target>/`
and a line in `cross_draft/INBOX.md` — **never an edit in the other draft**. Every session starts by
reading its section of `INBOX.md`. Rules: `cross_draft/README.md`.

## Binding on every chat: the prompt is not thesis text

What the author asks for enters the thesis as a **fact about the figure, table or result** — never as a
restatement of the request, a *"which is why ..."* justification for having done it, or a claim that it
was done (🔴). Nor does a measurement get a *because* the evaluation did not produce, or a paragraph
answering an objection the reader has not made (🟡). The full rule, with worked examples and the caption checklist, is
[`../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md`](../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md)
(author, 2026-09-20). It holds for v2, v3 and v4 alike, and it is the one rule an AI agent breaks most
easily, because the request is the freshest thing in its context.

## Rules for each chat

### v2
- Write only Ch 1–4, the abstract, the preamble, acronyms and `bibliography.bib` in `v2/thesis_v2.tex`.
- **Do not write into v2's own Ch 5–8.** They are placeholder headings; the real Ch 5–8 live in v3.
- **Do not edit anything under `v3/` or `v4/`.**
- If a v2 decision affects Ch 5–8 — a renamed term, a renamed or removed `\label` — **record it in
  `v2/CHANGELOG.md` under a line starting `For v3:`** and stop there. v3 picks it up on its next sync.
  Renaming a `\label` breaks v3's cross-references; say so explicitly when it happens.

### v3
- Write only `v3/chapters/05` and `06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib`, and v3's
  `bundle/`, `tools/`, `handover/`. Its `07`, `08`, `09` and `app_ntrial20_feasible.tex` are frozen at v3.98
  (v4 owns the live ones); a new Ch 6 fact goes to `cross_draft/to_v4/`. **Figures are produced only in
  `Data_Analysis/DA_in_Paper/`** (since 2026-09-14) and copied into `v3/figures/` with
  `plotting/export_to_draft.py v3` — never drawn or edited in the draft.
- **Ch 7 Discussion is v4's.** v3 keeps its headings and labels (Ch 4 references them) and writes no
  prose there. What v3 had written is in `cross_draft/to_v4/FROM_v3_20260914_discussion_prose.tex`.
- **Never edit v3's copies of Ch 1–4, `00_preamble`, `01_frontmatter`, `99_backmatter` or
  `bibliography.bib`** — not even the ones whose policy is `merge`. The change belongs in v2; sync it
  down with `python3 tools/sync_v2.py merge`. Editing the copy creates merge conflicts, and the
  default bundle then stops collapsing that chapter.
- **Never edit anything under `v2/` or `v4/`.**
- Sync from v2 only when the author asks. Before merging, read v2's `For v3:` lines.

### v4 (live since 2026-09-24)
- Write only `v4/chapters/07_conclusion`, `08_discussion`, `09_appendix` (+ `chapters/app_long/`), `parts/00_preamble_v4.tex`,
  `bibliography_v4.bib`, and v4's `bundle/`, `tools/`, `notes/`, `withheld/`. **Figures are produced only in
  `Data_Analysis/DA_in_Paper/`** and copied in with `plotting/export_to_draft.py v4`.
- **Never edit v4's copies of Ch 1–6, the inherited parts or the inherited `.bib` files**, not even
  `99_backmatter.tex` (policy `merge`) without a `For v2:` note. The change belongs upstream; sync it down
  with `python3 tools/sync_v3.py merge` (a v2 change reaches v4 after v3 has absorbed it).
- **Never edit anything under `v2/` or `v3/`.** Sync from v3 only when the author asks; before merging,
  read v3's `For v4:` lines.
- Every number in Ch 7–8 restates a Ch 6 number and names its table or section (`\dataref`). If Ch 6
  moves, Ch 7–8 move with it; v3 hands the new facts through `cross_draft/to_v4/`.
- The bundle is built **without v2 but with v3 + v4** (author, 2026-09-24): `bundle/make_bundle.py`
  collapses the v2-owned chapters by default and names the v2/v3 revisions in every header.

## The overlaps, settled

| topic | who holds it now | handover |
| :-- | :-- | :-- |
| **Appendix** (`app:derivations`, `app:results`, `app:repro`) | **v4** since 2026-09-24 (v3.98 handover; v4.0: long-data switch, dead block dropped, Reproducibility as facts; v4.1: A Supplementary Material, B the five kept sections, C Compute Environment) | v3's copy is frozen at v3.98; the twenty-episode section is used as is; index changes are notified to v3, never edited there |
| **Future work / next steps** (`sec:disc:future`, in the Discussion since v4.1) | **v4** — written at v4.0 from `future_work/` and the author's notes | v3's `\hole` is superseded |
| **Discussion** (Ch 8 since v4.1) | **v4** — limitations, towards deployment, future work; the Interpretation / Negative Results / Threats sections of the bone are dropped (author, 2026-09-24) | v3's headings-only copy is frozen |
| **Conclusion** (Ch 7 since v4.1) | **v4** since 2026-09-24 — summary and RQ answers only | v3's concise 20-09 draft was stale against Ch 6 at v3.98 and is not reused; v4.0 rewrote it |

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

## Open items as of 2026-09-24 (after v4.0)

- v4 branched from v3.98 / v2.27 and is in sync (`v4/tools/sync_v3.py status`). Three v3-owned sentences
  point at appendix content v4 changed (INBOX → v3, v4.0 row); the alias label
  `app:avoiding-twenty-episode` in v4 keeps them resolving meanwhile.
- The abbreviation recheck found the list complete; two consistency findings are on the → v2 table.
- Whether the appendix's long-data tables are hidden (`\appendixfullfalse`) and the web link that
  replaces them are the author's (`v4/notes/OPEN_20260924_v4_open_items.md`).
