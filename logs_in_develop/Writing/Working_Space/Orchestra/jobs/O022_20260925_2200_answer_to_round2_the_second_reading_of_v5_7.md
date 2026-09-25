# O022 — Answer to ROUND2 (the second reading of v5.7): every point checked against the v5.7 source and the code; the list for the author

**Opened:** 2026-09-25 22:00 · **Kind:** check (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.7** (the thesis) · last release: `20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "answer the ROUND2, write new MD near ROUND2. …/20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX/feedback/ROUND2_FEEDBACK_TO_RESPONSE_AND_V5.7_20260925.md" (author, 2026-09-25)

## Scope check (before touching anything)

- [x] Kind `check`: an answer MD written beside the round-2 review in the v5.7 build's `feedback/` folder; no thesis file touched; the code read for two points (the UAV controller's velocity reference; the Steps population).
- [ ] **Advance (kind `advance` / `absorb`):** the edit is made in `../../v5/` — the Orchestra owns it (no ownership check, no cross note); one `## v5.N` entry in `v5/CHANGELOG.md` + `v5/changelogs/`, `tools/check.py`, `tools/make_release_v5.py --dry-run`; INBOX rows resolved there are marked `🔀 v5.N`. The items below are for the legacy flow.
- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [ ] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [ ] No README / CROSS_STATE / SYNC_STATE / MANIFEST of a draft edited; no `RELEASE/output/`, no `Template_DONT_CHANGE/`.
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| RELEASE feedback | `…_v5.7_DATA_FIX/feedback/CLAUDE_ANSWER_to_ROUND2_20260925_2157_v5.7.md` | the answer: 27 points checked against the v5.7 release copy and, for 6A / 6E / 6F-1, the code; six regressions of v5.5–v5.7 named (the Ch 5 controller sentence, the per-axis clause, "comes closest at twenty", the always-active list, the Ch 7 dominance gloss, the merged caption); verdicts ✅ DO 20 · 🟡 yours 5 · ⛔ 1 · production 1; the list for the author (R2-A one pass → v5.8 "ROUND2 FIX"; R2-B the abstract, the corridor wording, two optional cuts; R2-C the 10/10 cells; R2-D compile) | the MD itself |
| thesis | **nothing** | untouched | — |

## Checks

- No thesis file touched. Code checked: `Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh:246`, `fetch_20260924_p23cv3_corridor_paths.sh:51–58` (run folders `…_pid_stopgo_…`), `FM_v3_uav_test/eval_fm_uav.py:1046–1052, 1156–1167`, `mix_uav_test/eval_mix_uav.py:762–766, 1895`, `uav_env_test/flight_controller.py:79–91` (6A); `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:745–748`, `DA_in_Paper/analysis/avoiding_rules_by_protocol.py:90–95`, `plotting/extract/corridor_v3_frontier.py:15, 44–57`, `analysis/scurve_r44.py:65–85` (6E); `plotting/builders/frontier.py:77–83` (6F-1). Every quoted passage read at its v5.7 line.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | the author ticks §3 of the answer |

## Release

none.

## Closed

2026-09-25 22:00 · versions after: v2.28 · v3.100b · v4.2 · v5.7 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
