# MANIFEST — what v5 was initialised from (v5.0)

**Initialised:** 2026-09-25 13:43:19 · **Orchestra job:** O003 · **Machine-readable:** [`INIT_STATE.json`](INIT_STATE.json) · **Last release at init:** `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE` (its sources were byte-identical to these files).

## The revisions v5 carries (the highest `## vN.M` heading of each legacy CHANGELOG.md at init)

| draft | revision | heading | CHANGELOG sha256 (16) |
| :-- | :-- | :-- | :-- |
| v2 | **v2.28** | v2.28 — 2026-09-24 · the seven open cross notes applied (v3.73–v3.96, v4.0, v4.1) | `23a09aa329a5476b` |
| v3 | **v3.100b** | v3.100b — 2026-09-25 · v3's copies of Ch 7–9 and the appendix archived (author: "also archive the v3 chapters for 07/08/09/appendix") | `dc068728c207275d` |
| v4 | **v4.2** | v4.2 — 2026-09-25 · The ChatGPT audit of v4.1a applied (§15) on v3.100; author's long-data heading flag and planning-horizon item; figures re-exported; bundles and release rebuilt | `9530f7535cd7dc9b` |

## File by file

`materials/<draft>/…` holds each source as v5 last absorbed it (at init: these exact bytes); `tools/absorb.py` merges a later legacy change against it. `materials/v2_split/` is the split of the v2 monolith (the seven parts v5 takes and the five placeholders it does not).

| v5 file | from | how | lines | sha256 (16) | source sha256 (16) |
| :-- | :-- | :-- | --: | :-- | :-- |
| `parts/00_preamble.tex` | v2 `v2/thesis_v2.tex` | split: lines before \begin{document} | 161 | `b386c7b93aa033c2` | `6863f3a6866c426c` |
| `parts/01_frontmatter.tex` | v2 `v2/thesis_v2.tex` | split: \begin{document}+1 .. first \chapter (title page, abstract, contents, \mainmatter) | 79 | `a08634de135d83f5` | `6863f3a6866c426c` |
| `chapters/01_introduction.tex` | v2 `v2/thesis_v2.tex` | split: \chapter ch:introduction | 172 | `2323d403aa0a3be4` | `6863f3a6866c426c` |
| `chapters/02_background.tex` | v2 `v2/thesis_v2.tex` | split: \chapter ch:background | 153 | `419b89fba4699ed3` | `6863f3a6866c426c` |
| `chapters/03_related_work.tex` | v2 `v2/thesis_v2.tex` | split: \chapter ch:related | 166 | `9801b196cd7158bb` | `6863f3a6866c426c` |
| `chapters/04_method.tex` | v2 `v2/thesis_v2.tex` | split: \chapter ch:method | 1707 | `1796c37a81d4a07e` | `6863f3a6866c426c` |
| `parts/99_backmatter.tex` | v2 `v2/thesis_v2.tex` | split: \microtypesetup .. before \end{document} (acronyms, lists, bibliography) | 30 | `fe62846cdeafb4d0` | `6863f3a6866c426c` |
| `bibliography.bib` | v2 `v2/bibliography.bib` | copy | 587 | `5c046b2902675a09` | `5c046b2902675a09` (identical) |
| `parts/00_preamble_v3.tex` | v3 `v3/parts/00_preamble_v3.tex` | copy | 198 | `7dde03651a170309` | `7dde03651a170309` (identical) |
| `chapters/05_setup.tex` | v3 `v3/chapters/05_setup.tex` | copy | 1500 | `4321c7942aeec67e` | `4321c7942aeec67e` (identical) |
| `chapters/06_results.tex` | v3 `v3/chapters/06_results.tex` | copy | 2582 | `5e2715bb809f5e9a` | `5e2715bb809f5e9a` (identical) |
| `bibliography_v3.bib` | v3 `v3/bibliography_v3.bib` | copy | 54 | `a84dbb745f2e377b` | `a84dbb745f2e377b` (identical) |
| `parts/00_preamble_v4.tex` | v4 `v4/parts/00_preamble_v4.tex` | copy | 42 | `3e2afbe3eba2c4f9` | `3e2afbe3eba2c4f9` (identical) |
| `chapters/07_conclusion.tex` | v4 `v4/chapters/07_conclusion.tex` | copy | 124 | `0fff00f679f66b12` | `0fff00f679f66b12` (identical) |
| `chapters/08_discussion.tex` | v4 `v4/chapters/08_discussion.tex` | copy | 169 | `37041a03fb37c40a` | `37041a03fb37c40a` (identical) |
| `chapters/09_appendix.tex` | v4 `v4/chapters/09_appendix.tex` | copy | 314 | `c1d5eaf909afb096` | `c1d5eaf909afb096` (identical) |
| `chapters/app_long/uav_corridor_rules.tex` | v4 `v4/chapters/app_long/uav_corridor_rules.tex` | copy | 217 | `e439689d30ebd8de` | `e439689d30ebd8de` (identical) |
| `chapters/app_ntrial20_feasible.tex` | v4 `v4/chapters/app_ntrial20_feasible.tex` | copy | 149 | `ddde881204c5781a` | `ddde881204c5781a` (identical) |
| `bibliography_v4.bib` | v4 `v4/bibliography_v4.bib` | copy | 10 | `8866985bfc96766e` | `8866985bfc96766e` (identical) |

**Not used by v5** (v2's placeholders, kept in `materials/v2_split/chapters/`): `chapters/05_setup.tex`, `chapters/06_results.tex`, `chapters/07_discussion.tex`, `chapters/08_conclusion.tex`, `chapters/09_appendix.tex`.

**Figures:** 78 files in `figures/` — v4/figures (v4.2 export of 2026-09-25) + v3/figures + v2/figures where v4 had no copy; `v4/figures/EXPORTED.md (copied as is)`.

## Open at init (the INBOX rows marked ⏳ on 2026-09-25, resolved in v5.1)

| target | INBOX line | from | item |
| :-- | --: | :-- | :-- |
| → v2 | 8 | Orchestra O001 · 2026-09-25 | (FYI — nothing asked of v2 now) A new Orchestra chat handles minor and cross-linked changes across v2/v3/v4 directly in the drafts (author, 2026-09-25): an edit of yours shows up as a tagged revision … |
| → v3 | 44 | Orchestra O001 · 2026-09-25 | (FYI — nothing asked of v3 now) A new Orchestra chat handles minor and cross-linked changes across v2/v3/v4 directly in the drafts (author, 2026-09-25): an edit of yours shows up as a tagged revision … |
| → v3 | 45 | v4.2 · 2026-09-25 | Your v3.100 note is closed (Ch 7 l. 25/45/47 rewritten; figures re-exported). Two FYI items for Ch 6: "at twenty evaluations" for the endpoint sampler's budget is an unresolved K/NFE terminology follo… |
| → v3 | 46 | v2.28 · 2026-09-24 | v2.28 is ready to sync for the first aggregated release (`tools/sync_v2.py merge`): abstract and §1.4 items 4–6 rebuilt on the author's storyline and v3.99's conclusions; outline in v4.1's order; "MuJ… |
| → v4 | 77 | Orchestra O001 · 2026-09-25 | (FYI — nothing asked of v4 now) A new Orchestra chat handles minor and cross-linked changes across v2/v3/v4 directly in the drafts (author, 2026-09-25): an edit of yours shows up as a tagged revision … |
| → v4 | 78 | v3.100b · 2026-09-25 | v3's copies of Ch 7–9 (`07_discussion`, `08_conclusion`, `09_appendix`, `app_ntrial20_feasible`) are archived; your `sync_v3.py status` now flags those four rows as "v3 moved" — retire them. v3's `che… |
| → v4 | 80 | v2.28 · 2026-09-24 | Your v4.0 and v4.1 notes are closed in v2.28 (outline: Conclusion before Discussion; "MuJoCo MPC" throughout, MJPC out of the list; PD declared). They reach v4 after v3's `sync_v2.py merge` and your `… |

Also carried over: the author's open decisions in `../../v4/notes/OPEN_20260924_v4_open_items.md`; the parked answer to the first-reading review (Orchestra job O002); the standing `\getDoctype` finding of the release tool.

*Written by the Orchestra (Claude Fable 5.1, Claude Code), job O003, 2026-09-25. Generated from INIT_STATE.json; not compiled.*
