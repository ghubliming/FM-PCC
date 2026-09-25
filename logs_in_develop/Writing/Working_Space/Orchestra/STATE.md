# Orchestra STATE · 2026-09-25 22:08

Written by `tools/orchestra.py status --write`; a snapshot, not a record. The record is `CHANGELOG.md` and `jobs/`.

## Drafts (highest `## vN.M` heading of each CHANGELOG.md)

| draft | revision | heading |
| :-- | :-- | :-- |
| v2 | **v2.28** | v2.28 — 2026-09-24 · the seven open cross notes applied (v3.73–v3.96, v4.0, v4.1) |
| v3 | **v3.100b** | v3.100b — 2026-09-25 · v3's copies of Ch 7–9 and the appendix archived (author: "also archive the v3 chapters for 07/08/09/appendix") |
| v4 | **v4.2** | v4.2 — 2026-09-25 · The ChatGPT audit of v4.1a applied (§15) on v3.100; author's long-data heading flag and planning-horizon item; figures re-exported; bundles and release rebuilt |
| v5 | **v5.8** | v5.8 — 2026-09-25 · ROUND2 FIX · source: the second reading of v5.7 (ROUND2_FEEDBACK…), answer list R2-A · twenty factual and consistency corrections in Ch 5–8, six of them undoing regressions of v5.5–v5.7; the controller reference, the guard, the Steps definitions, the dominance definition; no number of record changed (Orchestra O023) — **the thesis** (the aggregate; Advance Orchestra since 2026-09-25) |

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

- builds kept in `RELEASE/output/`: 7; newest: `20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX`
- of these built by the Orchestra from v5 (`_ORCH_`): 6; newest: `20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX` (`cd v5 && python3 tools/make_release_v5.py`)
- `RELEASE/CHANGELOG.md` top row: 20260925_220823 -- 20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX

## Orchestra

- jobs on file: 24; open rows in `CHANGELOG.md`: 1; next id: **O025**
- last row: | [O024](jobs/O024_20260925_2208_release_from_v5_8_marked_round2_fix.md) | 2026-09-25 22:08 → 2026-09-25 22:08 | release | v2.28 · v3.100b · v4.2 · v5.8 | ✅ | Release from v5.8 on the author's word: output/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX — the ROUND2 FIX build; page estimate ~179 (152-215); differs from the v5.7 build in the four chapter files v5.8 edited. | — | `20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX` |
- git: 39 modified/untracked path(s) under `logs_in_develop/Writing` (the Orchestra never commits)
