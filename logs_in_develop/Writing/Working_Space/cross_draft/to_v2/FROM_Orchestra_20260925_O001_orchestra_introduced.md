# FROM Orchestra → v2 · 2026-09-25 · O001 · orchestra_introduced

**Who writes this:** the Orchestra chat (`Working_Space/Orchestra/`), which handles minor and cross-linked changes across v2 / v3 / v4 on the author's request. Job record: [`../../Orchestra/jobs/O001_20260925_1219_orchestra_init_folder_runbook_tool_templates.md`](../../Orchestra/jobs/O001_20260925_1219_orchestra_init_folder_runbook_tool_templates.md). Draft revisions at the time of writing: v2.28 · v3.100b · v4.2.

**Nothing is asked of v2 now; nothing of yours was changed.** This note introduces a new chat and its protocol, so that a change you did not make is never a surprise.

## What the Orchestra does (author, 2026-09-25)

The author runs a separate **Orchestra** chat (`Working_Space/Orchestra/`, runbook in its `README.md`) for **minor changes, and changes that are cross-linked between chapters of different drafts**. It edits directly inside `v2/` — **only the files you own** per `DRAFT_OWNERSHIP.md`, never an inherited copy, never your README / state files / notes — and records every edit in **your `CHANGELOG.md`** as a revision entry, newest first, in your format. Big changes stay with you; the author brings them to your chat directly. When a heavy job is split up, you receive your part as a **TODO list** through this inbox.

## How an Orchestra edit shows up in your files

1. **A revision entry in your `CHANGELOG.md`** — **v2: the next number** (v2.28 → v2.29), because `v3/tools/sync_v2.py` reads no letter suffix; the heading is tagged `(Orchestra O###)` and links the job file. Your next pass then continues at the number after it. The body names the file and lines, the author's request, the checks run (**never compiled**), and what is left to you.
2. **A cross note + INBOX row** (`cross_draft/to_v2/FROM_Orchestra_<date>_O###_<topic>.md`, row `Orchestra O### · <date>`) for every job that touched your files, and an FYI note when a change elsewhere affects something you reference (a label, a term, a restated number).
3. **Bundle and checks** are rerun by the Orchestra after its edit (v3 / v4: `tools/check.py`, `bundle/make_bundle.py`; v2: the release tool's `--dry-run`), and the result is in the entry.

## What you do when you find one

Read the entry and the note at your session start, refresh your `README.md` / state files if the revision line there is stale, and mark the INBOX row ✅ with your revision (❌ with a reason if you disagree; the author decides). Your ownership, your sync policy and your changelog rules are unchanged.

**Releases** are still built from your live files by `RELEASE/tools/make_release.py`; when the Orchestra runs one on the author's word, the build is tagged `ORCH_O###` and its note names the job.

**What was left to you:** your `README.md`, `CROSS_STATE.json` / `inherited/SYNC_STATE.json` and any "current revision" line were **not** touched — refresh them at your next pass. Mark this INBOX row ✅ with your revision once read and absorbed (❌ with a reason if you disagree; the author decides).

Orchestra (Claude Fable 5.1, Claude Code), O001 · 2026-09-25 · no file of yours touched beyond what is listed above.
