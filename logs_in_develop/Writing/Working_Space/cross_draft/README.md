# cross_draft — what one draft found that another draft must act on

v2, v3 and v4 are written in separate chats and **never edit each other's files** (`../DRAFT_OWNERSHIP.md`).
When one draft finds something that belongs to another, it writes a note here instead of editing.

```
cross_draft/
├── INBOX.md      one line per open item — read this first, every session
├── to_v2/        found by v3 or v4, to be acted on in v2 (Ch 1–4, abstract, preamble, bib)
├── to_v3/        found by v2 or v4, to be acted on in v3 (Ch 5–6, appendix, Ch 8 draft)
├── to_v4/        found by v2 or v3, for v4 (Ch 7 Discussion, Ch 8 refinement, future work)
└── to_v5/        since 2026-09-25: for the Orchestra, which edits the thesis in ../v5 — findings about the thesis
                  text, and announcements of a change made in a legacy draft's own files (to be absorbed into v5)
```

## Rules
- **Writing a note:** `to_<target>/FROM_<source>_<YYYYMMDD>_<topic>.md` — what, where (file:line or
  `\label`), why, and the evidence. Then add one line to `INBOX.md`. Do not edit the target's files.
- **Reading:** at the start of a session, check `INBOX.md` for your draft.
- **Closing:** the *target* draft acts, records it in its own CHANGELOG, and marks the INBOX line ✅ with
  its version (e.g. `✅ v2.17`). Notes stay in place as the record; nobody deletes them.
- **Declining:** mark ❌ with a one-line reason. The author decides disputes.
- **A fourth source since 2026-09-25 (Orchestra O001):** notes named `FROM_Orchestra_<date>_O###_<topic>.md` and rows
  `Orchestra O### · <date>` come from the Orchestra chat (`../Orchestra/README.md`), which makes minor and cross-linked
  edits directly in the drafts on the author's request and distributes TODO lists. The target closes them like any
  other row; the Orchestra never marks its own rows ✅.
- v2 → v3 changes to inherited chapters still flow through `v3/tools/sync_v2.py`; this folder is for
  findings, not for copying text.
- **Since 2026-09-25 (Orchestra O003) the thesis lives in `../v5`** (the aggregate; runbook `../v5/README.md`). A finding
  that concerns the thesis text goes under `## → v5` (folder `to_v5/`), where the Orchestra acts on it in v5. A change
  made in a legacy draft's own files is announced the same way (a `## → v5` row naming the files and the revision), so
  that the Orchestra absorbs it with `v5/tools/absorb.py`. A row the Orchestra resolves in v5 is marked **`🔀 v5.N`**;
  that never touches the legacy draft's own files, so the owner has nothing to redo.
