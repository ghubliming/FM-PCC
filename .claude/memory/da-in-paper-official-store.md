---
name: da-in-paper-official-store
description: Data_Analysis/DA_in_Paper is the official, live store for thesis numbers (analysis/INDEX.md) and ALL thesis figures + plotting code; drafts only receive copies via export_to_draft.py
metadata:
  type: project
---

Since 2026-09-14 (author's decision) `Data_Analysis/DA_in_Paper/` is the **official source of everything the thesis shows**:

- `analysis/INDEX.md` — every thesis result → the analysis it comes from, with protocol. Numbers in a draft are quoted from an analysis listed there.
- `plotting/` — **all** figure code: `sources.py` (the only file with data paths; CORPORA, VENDORED, PLANNED, EXCLUDED), `builders/` (per-environment, stdlib SVG), `svg/` (canvas + svg2pdf.sh), `mpl/` (matplotlib, cluster/laptop only), `make_figs.py`, `export_to_draft.py`.
- `figures/{da,demo,env,schematic}/` — **every** thesis figure (DA charts, raw plan/rollout demos, environment renders, diagrams) + generated MANIFEST.md.

**Why:** one place for figures and their provenance; drafts (v2/v3/v4) are edited by separate chats and must not each grow their own plotting code or figure versions.

**How to apply:** never draw or edit a figure inside `Working_Space/<draft>/figures/`. Produce it in DA_in_Paper (`make_figs.py`, or a script in `mpl/`), then `python3 plotting/export_to_draft.py <draft>` copies only the figures that draft's .tex includes and writes `EXPORTED.md`. Copied-in panels are checked against `aux_repo/` first (repo-root `figures/avoiding*.png` are DPCC's and EXCLUDED). The old `v3/plots/` was moved here. Related: [[thesis-draft-ownership]], [[no-unrequested-urls-or-artifacts]].
