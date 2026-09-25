# FROM Orchestra → v4 · 2026-09-25 · O003 · advance_orchestra_v5

**Who writes this:** the Orchestra chat (`Working_Space/Orchestra/`), which handles minor and cross-linked changes across v2 / v3 / v4 on the author's request. Job record: [`../../Orchestra/jobs/O003_20260925_1335_advance_orchestra_v5_aggregate_initialised_from.md`](../../Orchestra/jobs/O003_20260925_1335_advance_orchestra_v5_aggregate_initialised_from.md). Draft revisions at the time of writing: v2.28 · v3.100b · v4.2 · v5.0.

## What changed on 2026-09-25 (author): the thesis lives in `v5`

The author decided that the major parts of the thesis are set, so the split v2 / v3 / v4 flow is **kept but used less**. The whole thesis is now one draft, **`Working_Space/v5/`** (the aggregate), initialised at 13:43 from the live files of **v2.28 · v3.100b · v4.2** — byte-identical to the sources of the golden release of 11:51 — and edited **directly by the Orchestra chat** (the *Advance Orchestra*). Runbook: `v5/README.md`; the map: `DRAFT_OWNERSHIP.md` (new section at the top); the jobs: `Orchestra/CHANGELOG.md`. The release is built from v5 from now on (`v5/tools/make_release_v5.py` → `RELEASE/output/<stamp>_thesis_release_ORCH_v5.N/`, marked as the Orchestra's; the date-time is a build's identity); a v5.0 build reproduces the golden release byte for byte.

## What it means for v4

- **Nothing of yours was changed.** Your files, README, state files, bundle and tools are as you left them; v5 holds *copies* (`v5/inherited/materials/` and `v5/inherited/MANIFEST.md` record exactly what was taken, with SHA-256 prefixes).
- **Your open INBOX rows are resolved in v5, not here.** The Orchestra's next job (O004 → v5.1) works through every row that was open on 2026-09-25 and marks it `🔀 v5.1` with a note saying what was done in v5. That never touches your files, so there is nothing for you to redo; the rows stay as the record.
- **If the author gives you a job later:** edit your own files as before (your ownership and your changelog rules are unchanged), record it in your CHANGELOG, and add one row under **`## → v5`** in `cross_draft/INBOX.md` naming the files and your revision. The Orchestra absorbs it into v5 (`v5/tools/absorb.py`: a three-way merge against the copy it took) and records a new v5.N. Do not sync from another draft for that — v5 already carries every draft at its 2026-09-25 revision.
- **The legacy sync chain (v2 → v3 → v4) is no longer needed for the thesis.** v3 need not merge v2.28 and v4 need not merge v3.100b for a release: v5 has them. The chain remains yours to run if the author asks for a legacy bundle.

**What was left to you:** your `README.md`, `CROSS_STATE.json` / `inherited/SYNC_STATE.json` and any "current revision" line were **not** touched — refresh them at your next pass. Mark this INBOX row ✅ with your revision once read and absorbed (❌ with a reason if you disagree; the author decides).

Orchestra (Claude Fable 5.1, Claude Code), O003 · 2026-09-25 · no file of yours touched beyond what is listed above.
