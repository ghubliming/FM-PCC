---
name: thesis-draft-ownership
description: Thesis drafts v2/v3/v4 are edited by separate chats and each owns fixed chapters (v2=Ch1–4, v3=Ch5–6, v4=Ch7 Conclusion + Ch8 Discussion + appendix, live since 2026-09-24 at v3.98/v2.27) — never cross-edit; read Working_Space/DRAFT_OWNERSHIP.md first
metadata:
  node_type: memory
  type: project
  originSessionId: 1c56946d-b40c-4293-9648-70342972edfd
  modified: 2026-09-24T21:00:00.000Z
---

The thesis drafts in `logs_in_develop/Writing/Working_Space/` are worked on by **separate chats**, and each owns fixed chapters (set by the author 2026-09-14; v4 live 2026-09-24):

- **v2** — Ch 1–4 (Intro, Background, Related Work, Method), abstract, preamble, acronyms, `bibliography.bib`, in `v2/thesis_v2.tex`. Must not write its own Ch 5–8 placeholders.
- **v3** — Ch 5 Setup, Ch 6 Results, `parts/00_preamble_v3.tex`, `bibliography_v3.bib`, its bundle/tools. Figures only in `Data_Analysis/DA_in_Paper` ([[da-in-paper-official-store]]). Never edits its copies of Ch 1–4. **Its copies of Ch 7–9 are frozen at v3.98** (handed to v4 in `v4/FROM_v3_v3.98_20260924_201920/`).
- **v4** — **live since 2026-09-24 (v4.0)**: **Ch 7 Conclusion** (summary + RQ answers only) and **Ch 8 Discussion** (limitations, *Towards Deployment*, future work) — this order since v4.1, files `chapters/07_conclusion.tex`, `08_discussion.tex` — the appendix (long-data switch `\ifappendixfull`, dead block dropped, Reproducibility as facts), `parts/00_preamble_v4.tex`, `bibliography_v4.bib`, `v4/tools/` (`sync_v3.py` status/diff/merge/stamp against `../v3`, `check.py`), `v4/bundle/make_bundle.py` (**v2 chapters collapsed, v3+v4 built**, v2/v3 versions in every header). Branched from **v3.98**, which carries **v2.27** (`v4/inherited/SYNC_STATE.json`). Never edits its copies of Ch 1–6.

**Why:** parallel chats editing the same chapters produce merge conflicts and silently overwrite each other's work; the flow is one-way v2 → v3 → v4 through the sync tools.

**How to apply:** at the start of any Writing/v2, v3 or v4 task, read `Working_Space/DRAFT_OWNERSHIP.md` and the draft's README. If a request would touch another draft's file, stop and say so, and log the needed change as a `For v3:` / `For v2:` / `For v4:` line in your own draft's CHANGELOG instead. Related: [[master-thesis-writing-tum]], [[thesis-prose-style]], [[thesis-v4-conclusion-rules]].

**The endings are v4's (author, 2026-09-24):** v3's job is Ch 5 and Ch 6. When Ch 6 results change, v3 hands the new facts to v4 through its inbox; Ch 7–8 restate Ch 6 numbers only (a `\dataref` names the source table). Open v4 items: `v4/notes/OPEN_20260924_v4_open_items.md`.

**Cross-draft findings (since 2026-09-16):** never edit another draft — write `Working_Space/cross_draft/to_<v2|v3|v4>/FROM_<src>_<date>_<topic>.md` and a line in `cross_draft/INBOX.md`; read your INBOX section at session start. The target draft closes items (✅ + its version).
