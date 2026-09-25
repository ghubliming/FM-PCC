# Orchestra STATE · 2026-09-25 15:36

Written by `tools/orchestra.py status --write`; a snapshot, not a record. The record is `CHANGELOG.md` and `jobs/`.

## Drafts (highest `## vN.M` heading of each CHANGELOG.md)

| draft | revision | heading |
| :-- | :-- | :-- |
| v2 | **v2.28** | v2.28 — 2026-09-24 · the seven open cross notes applied (v3.73–v3.96, v4.0, v4.1) |
| v3 | **v3.100b** | v3.100b — 2026-09-25 · v3's copies of Ch 7–9 and the appendix archived (author: "also archive the v3 chapters for 07/08/09/appendix") |
| v4 | **v4.2** | v4.2 — 2026-09-25 · The ChatGPT audit of v4.1a applied (§15) on v3.100; author's long-data heading flag and planning-horizon item; figures re-exported; bundles and release rebuilt |
| v5 | **v5.3** | v5.3 — 2026-09-25 · **PURE BUG FIX** · source: the same first-reading audit (answer group A, items A6 and A9) · Figures 5.10 / 5.11 (clipped text) and 6.4 (palette) rebuilt in the DA store and re-exported; no text change (Orchestra O007) — **the thesis** (the aggregate; Advance Orchestra since 2026-09-25) |

## v5 — the aggregate (Advance Orchestra, since 2026-09-25; runbook `v5/README.md`)

- v5 carries (last absorbed 2026-09-25 13:43:19 (init, v5.0)): v2 **v2.28** · v3 **v3.100b** · v4 **v4.2**; legacy drafts moved since: **none**. File view: `cd v5 && python3 tools/absorb.py status`

## Sync chain (one-way v2 → v3 → v4; a RELEASE reads the LIVE files, so a lag matters only for bundles and inherited copies)

- v3 carries **v2.27** (synced 2026-09-23); v2 is at v2.28 → **LAGS — v2 moved**. File view: `cd v3 && python3 tools/sync_v2.py status`
- v4 carries **v3.100** / v2.27 (synced 2026-09-25); v3 is at v3.100b → **LAGS — v3 moved**. File view: `cd v4 && python3 tools/sync_v3.py status`

## cross_draft/INBOX.md — open rows (⏳)

- **→ v2: 1 open**
  - [Orchestra O003 · 2026-09-25] (FYI — nothing asked of v2 now) Since 2026-09-25 the thesis lives in `v5` — the aggregate of v2.28 · v3.100b · v4.2, edited directly by the Orchestra …
- **→ v3: 1 open**
  - [Orchestra O003 · 2026-09-25] (FYI — nothing asked of v3 now) Since 2026-09-25 the thesis lives in `v5` — the aggregate of v2.28 · v3.100b · v4.2, edited directly by the Orchestra …
- **→ v4: 1 open**
  - [Orchestra O003 · 2026-09-25] (FYI — nothing asked of v4 now) Since 2026-09-25 the thesis lives in `v5` — the aggregate of v2.28 · v3.100b · v4.2, edited directly by the Orchestra …
- **→ v5: 0 open**

## RELEASE

- builds kept in `RELEASE/output/`: 2; newest: `20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A`
- of these built by the Orchestra from v5 (`_ORCH_`): 1; newest: `20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A` (`cd v5 && python3 tools/make_release_v5.py`)
- `RELEASE/CHANGELOG.md` top row: 20260925_152307 -- 20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A

## Orchestra

- jobs on file: 9; open rows in `CHANGELOG.md`: 1; next id: **O010**
- last row: | [O009](jobs/O009_20260925_1533_answer_md_group_b_rebuilt_by_risk_size_and.md) | 2026-09-25 15:33 → 2026-09-25 15:36 | check | v2.28 · v3.100b · v4.2 · v5.3 | ✅ | Answer MD §9 re-sorted for the author: B rebuilt as B-i low-risk minor (17 rows, file:line, size, risk), B-ii giant rebuilds (13 rows, incl. the W3 sweep quantified), B-iii no target (9 rows, word counts vs the claims); C tagged 🔴 dangerous 6 · 🟠 careful 9 · 🟢 minor 9 with a reason each; 'How this proceeds' by tag. Nothing applied to the thesis. | — | — |
- git: 24 modified/untracked path(s) under `logs_in_develop/Writing` (the Orchestra never commits)
