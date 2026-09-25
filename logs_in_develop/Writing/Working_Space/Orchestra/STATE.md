# Orchestra STATE · 2026-09-25 12:23

Written by `tools/orchestra.py status --write`; a snapshot, not a record. The record is `CHANGELOG.md` and `jobs/`.

## Drafts (highest `## vN.M` heading of each CHANGELOG.md)

| draft | revision | heading |
| :-- | :-- | :-- |
| v2 | **v2.28** | v2.28 — 2026-09-24 · the seven open cross notes applied (v3.73–v3.96, v4.0, v4.1) |
| v3 | **v3.100b** | v3.100b — 2026-09-25 · v3's copies of Ch 7–9 and the appendix archived (author: "also archive the v3 chapters for 07/08/09/appendix") |
| v4 | **v4.2** | v4.2 — 2026-09-25 · The ChatGPT audit of v4.1a applied (§15) on v3.100; author's long-data heading flag and planning-horizon item; figures re-exported; bundles and release rebuilt |

## Sync chain (one-way v2 → v3 → v4; a RELEASE reads the LIVE files, so a lag matters only for bundles and inherited copies)

- v3 carries **v2.27** (synced 2026-09-23); v2 is at v2.28 → **LAGS — v2 moved**. File view: `cd v3 && python3 tools/sync_v2.py status`
- v4 carries **v3.100** / v2.27 (synced 2026-09-25); v3 is at v3.100b → **LAGS — v3 moved**. File view: `cd v4 && python3 tools/sync_v3.py status`

## cross_draft/INBOX.md — open rows (⏳)

- **→ v2: 1 open**
  - [Orchestra O001 · 2026-09-25] (FYI — nothing asked of v2 now) A new Orchestra chat handles minor and cross-linked changes across v2/v3/v4 directly in the drafts (author, 2026-09-25…
- **→ v3: 3 open**
  - [Orchestra O001 · 2026-09-25] (FYI — nothing asked of v3 now) A new Orchestra chat handles minor and cross-linked changes across v2/v3/v4 directly in the drafts (author, 2026-09-25…
  - [v4.2 · 2026-09-25] Your v3.100 note is closed (Ch 7 l. 25/45/47 rewritten; figures re-exported). Two FYI items for Ch 6: "at twenty evaluations" for the endpoint sampler…
  - [v2.28 · 2026-09-24] v2.28 is ready to sync for the first aggregated release (`tools/sync_v2.py merge`): abstract and §1.4 items 4–6 rebuilt on the author's storyline and …
- **→ v4: 3 open**
  - [Orchestra O001 · 2026-09-25] (FYI — nothing asked of v4 now) A new Orchestra chat handles minor and cross-linked changes across v2/v3/v4 directly in the drafts (author, 2026-09-25…
  - [v3.100b · 2026-09-25] v3's copies of Ch 7–9 (`07_discussion`, `08_conclusion`, `09_appendix`, `app_ntrial20_feasible`) are archived; your `sync_v3.py status` now flags thos…
  - [v2.28 · 2026-09-24] Your v4.0 and v4.1 notes are closed in v2.28 (outline: Conclusion before Discussion; "MuJoCo MPC" throughout, MJPC out of the list; PD declared). They…

## RELEASE

- builds kept in `RELEASE/output/`: 1; newest: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
- `RELEASE/CHANGELOG.md` top row: 20260925_115125 -- 20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE

## Orchestra

- jobs on file: 1; open rows in `CHANGELOG.md`: 0; next id: **O002**
- last row: | [O001](jobs/O001_20260925_1219_orchestra_init_folder_runbook_tool_templates.md) | 2026-09-25 12:19 → 2026-09-25 12:23 | init | v2.28 · v3.100b · v4.2 | ✅ | Orchestra folder initialised: runbook (README), job CHANGELOG, tool, templates, STATE; FYI protocol note to v2/v3/v4; one rule line in cross_draft/README.md; no draft file touched | v2, v3, v4 (FYI) | — |
- git: 165 modified/untracked path(s) under `logs_in_develop/Writing` (the Orchestra never commits)
