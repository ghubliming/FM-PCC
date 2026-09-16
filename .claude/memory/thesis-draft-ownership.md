---
name: thesis-draft-ownership
description: Thesis drafts v2/v3/v4 are edited by separate chats and each owns fixed chapters (v2=Ch1–4, v3=Ch5–6+appendix+Ch8 draft, v4=Ch7+Ch8 refine) — never cross-edit; read Working_Space/DRAFT_OWNERSHIP.md first
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c56946d-b40c-4293-9648-70342972edfd
  modified: 2026-09-14T15:19:19.442Z
---

The thesis drafts in `logs_in_develop/Writing/Working_Space/` are worked on by **separate chats**, and each owns fixed chapters (set by the author 2026-09-14):

- **v2** — Ch 1–4 (Intro, Background, Related Work, Method), abstract, preamble, acronyms, `bibliography.bib`, in `v2/thesis_v2.tex`. Must not write its own Ch 5–8 placeholders.
- **v3** — Ch 5 Setup, Ch 6 Results, a concise Ch 8 Conclusion draft, and the appendix for now, plus bundle/tools. Figures are produced only in `Data_Analysis/DA_in_Paper` and exported into drafts ([[da-in-paper-official-store]]). Keeps Ch 7's headings only. Never edits its copies of Ch 1–4 (those change in v2 and sync down one-way).
- **v4** — planned, not started: **Ch 7 Discussion** (moved from v3 by the author, 2026-09-14), refining Ch 8 and its future work, next steps, real-world practice, appendix material. Starts only after v2 and v3 are declared finished; will branch from v3. v3's earlier discussion prose is in `v3/handover/`.

**Why:** parallel chats editing the same chapters produce merge conflicts and silently overwrite each other's work; v3 already inherits Ch 1–4 from v2 through a one-way sync tool.

**How to apply:** at the start of any Writing/v2, v3 or v4 task, read `Working_Space/DRAFT_OWNERSHIP.md`. If a request would touch another draft's file, stop and say so, and log the needed change as a `For v3:` / `For v2:` line in your own draft's CHANGELOG instead. Related: [[master-thesis-writing-tum]], [[thesis-prose-style]].

**Cross-draft findings (since 2026-09-16):** never edit another draft — write `Working_Space/cross_draft/to_<v2|v3|v4>/FROM_<src>_<date>_<topic>.md` and a line in `cross_draft/INBOX.md`; read your INBOX section at session start. The target draft closes items (✅ + its version).
