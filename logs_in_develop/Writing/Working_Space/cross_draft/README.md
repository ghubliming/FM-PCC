# cross_draft — what one draft found that another draft must act on

v2, v3 and v4 are written in separate chats and **never edit each other's files** (`../DRAFT_OWNERSHIP.md`).
When one draft finds something that belongs to another, it writes a note here instead of editing.

```
cross_draft/
├── INBOX.md      one line per open item — read this first, every session
├── to_v2/        found by v3 or v4, to be acted on in v2 (Ch 1–4, abstract, preamble, bib)
├── to_v3/        found by v2 or v4, to be acted on in v3 (Ch 5–6, appendix, Ch 8 draft)
└── to_v4/        found by v2 or v3, for v4 (Ch 7 Discussion, Ch 8 refinement, future work)
```

## Rules
- **Writing a note:** `to_<target>/FROM_<source>_<YYYYMMDD>_<topic>.md` — what, where (file:line or
  `\label`), why, and the evidence. Then add one line to `INBOX.md`. Do not edit the target's files.
- **Reading:** at the start of a session, check `INBOX.md` for your draft.
- **Closing:** the *target* draft acts, records it in its own CHANGELOG, and marks the INBOX line ✅ with
  its version (e.g. `✅ v2.17`). Notes stay in place as the record; nobody deletes them.
- **Declining:** mark ❌ with a one-line reason. The author decides disputes.
- v2 → v3 changes to inherited chapters still flow through `v3/tools/sync_v2.py`; this folder is for
  findings, not for copying text.
