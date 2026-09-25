# O007 — Tier A leftovers fixed at the source: the two clipped expert figures (A6) and the projected-frontier palette (A9), rebuilt in the DA store and exported to v5 (v5.3)

**Opened:** 2026-09-25 15:17 · **Kind:** advance (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.2** (the thesis) · last release: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "is hte still open A pure bugs and zero chance of other issues? if pure ubg, just fix and back to source" (author, 2026-09-25)

## Scope check (before touching anything)

- [x] **Advance (kind `advance` / `absorb`):** figures only — fixed in the DA store (the figure source, per the DA rule) and exported into v5; no `.tex` touched.
- [ ] **Advance (kind `advance` / `absorb`):** the edit is made in `../../v5/` — the Orchestra owns it (no ownership check, no cross note); one `## v5.N` entry in `v5/CHANGELOG.md` + `v5/changelogs/`, `tools/check.py`, `tools/make_release_v5.py --dry-run`; INBOX rows resolved there are marked `🔀 v5.N`. The items below are for the legacy flow.
- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [x] No figure drawn or edited *inside a draft*: the three figures were rebuilt in `Data_Analysis/DA_in_Paper/` and reached v5 through `export_to_draft.py` (6 files copied, `EXPORTED.md` regenerated).
- [ ] No README / CROSS_STATE / SYNC_STATE / MANIFEST of a draft edited; no `RELEASE/output/`, no `Template_DONT_CHANGE/`.
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| DA source | `plotting/builders/expert.py` (`_legend` + `LEGEND_LINE`, l. 76–79ff; key l. 237; margin l. 263; subtitle l. 275–277), `plotting/builders/frontier.py` (l. 298) | A6: wrapped legend key, two-line subtitle; A9: palette = `ENGINE_COLOUR_DISTINCT` | `v5/changelogs/v5.3_20260925_review_tier_a_figures_at_source.md` |
| DA store | `figures/env/fig_expert_aligning.{svg,png}`, `figures/env/fig_expert_uav.{svg,png}`, `figures/da/fig_aligning_projected_tradeoff.{svg,png}` | rebuilt (`make_figs.py <match>`), PNGs re-rendered (`preview_png.py --scale 3`, the store's own method); `MANIFEST.md` restored after the partial build | same |
| v5 | `figures/` — the six files + `EXPORTED.md` | exported (`export_to_draft.py <v5>`) | `v5/CHANGELOG.md` **v5.3** |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` | A6 / A9 status → `✅ v5.3`; A4 relayout → "cosmetic, left"; header + §0 status + signature | the MD itself |
| v2 / v3 / v4 | **nothing** | untouched | — |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7 833 lines, 275 labels, 53/53 citations, 41 figures, all pass · `python3 tools/make_release_v5.py --dry-run` → 19 + 48 files, 9 holes, 1 finding, 4 residue hits (as v5.2) · `absorb.py status` not re-run (no legacy file changed)
- Figures: the three PNGs read by eye; store ↔ v5 SHA-256 equal for all six files; the rasteriser proven to be the store's (pixel-identical re-render of unchanged figures).
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | the store is the source for every draft; v3 / v4 would pick the fixed figures up at their next export — nothing asked of them |

## Release

none — not asked.

## Closed

2026-09-25 15:19 · versions after: v2.28 · v3.100b · v4.2 · v5.3 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
