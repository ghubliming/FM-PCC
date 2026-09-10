# `plots/` — the figure pipeline

**Figures are code, not files.** Every figure in `../figures/` is generated from a batch CSV by a
script here, so when a run lands the figures are rebuilt rather than redrawn.

```bash
python3 plots/make_figs.py            # rebuild everything into ../figures/
python3 plots/make_figs.py k_ladder   # just the ones whose name matches
python3 plots/make_figs.py --list     # what exists, and which corpora are on disk
tools/svg2pdf.sh                      # SVG -> PDF, on a machine that has a converter
```

## When new data lands, edit one file

**`sources.py` is the only file that contains a path.** Point its `Corpus` entry at the new batch
directory, update the `protocol` string, re-run `make_figs.py`. Every figure that uses that corpus is
rebuilt, its subtitle picks up the new protocol, and `../figures/MANIFEST.md` records what came from
where. No figure script needs touching.

The registry mirrors §10 of
[`../../data_status/DATASTATUS_20260910_v3_entry_readiness.md`](../../data_status/DATASTATUS_20260910_v3_entry_readiness.md)
row for row. **If the two disagree, that file is right and this one is stale.**

## The four files

| file | role |
| :-- | :-- |
| `fmpcc_svg.py` | The SVG canvas. Carried over from `Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/make_figs.py`, so the thesis figures are drawn by the same code as the reports of record. |
| `sources.py` | Corpus registry, loaders, the aggregation rule, the Pareto definition. The vocabulary too — engine colours and thesis-facing labels. |
| `figures.py` | One function per thesis figure. Each returns `(path, provenance)` or `None`. |
| `make_figs.py` | CLI, and the manifest writer. |

## Two decisions worth knowing before editing

**Stdlib only, no matplotlib.** The container the thesis is written in has no scientific Python
stack. A figure script that imports numpy could not be run where the writing happens, which would
make "regenerate the figures" a cluster round-trip and guarantee that figures drift from text. The
canvas is about 200 lines and has paid for itself.

**SVG, converted to PDF at build time.** `\includegraphics` cannot read SVG, and no converter is
installed here either. So SVG is the committed artefact — diffable, viewable in a browser and in a
Markdown preview — and `tools/svg2pdf.sh` produces PDFs wherever the document is actually built.
`thesis_v3.tex` uses extension-less `\includegraphics`, so it needs no change either way.

## Adding a figure

1. Write `def fig_<name>(outdir) -> (path, provenance) | None` in `figures.py`.
2. Add it to `ALL` at the bottom of that file.
3. `python3 make_figs.py <name>`.

Four things a builder must do, all of them enforced by convention rather than by code:

- **Return `None` when its corpus is absent**, so a partial checkout builds what it can. `temp/` is a
  local drop directory and is not version-controlled — a fresh machine has none of it.
- **Put the protocol in the subtitle.** TARGET §6 rule 7: never a bare *n*.
- **Use thesis names, never code tokens** (`Auxiliary/Naming/NAMING_20260910_master_table.md`). The
  tokens belong in `sources.py`'s folder patterns and nowhere else.
- **Draw a different protocol differently.** The K-ladder plots the baseline's published 10-episode
  cells dashed and hollow alongside 100-episode cells; mixing them into one solid series would be
  the most misleading thing that figure could do.

## The aggregation rule, because it is the easy thing to get wrong

Entry-1 cells aggregate **per geometry first, then across geometries** — never as a flat mean over
`(seed × geometry)` cells. The two differ whenever a geometry carries a different seed count, which
is exactly the case for the pinned baseline (`both-hard` is seed-6 only).

Geometry-mean reproduces the DAs of record exactly:

| | S&C | steps | s/step | matches |
| :-- | --: | --: | --: | :-- |
| baseline K20, cumulative-cost tightened | 0.983 | 69.0 | 0.5635 | DA_20260827 §10.1 (0.983 / 69.0 / 564 ms) |
| MeanFlow K1, temporal-consistency tightened | 0.993 | 61.0 | 0.0181 | same, 0.993 / 61.0 / 18.1 ms |
| baseline K20 at the published protocol | 1.000 | 70.1 | 0.5534 | DA_20260906 §3 (1.000 / 70.13 / 0.5534) |

A flat cell-mean gives 0.977 / 72.5 / 544 ms and would silently contradict the text. `geometry_mean`
takes `geom_index` as a **required** argument for the same reason: inferring it from the key length
picked the wrong column for one of the two loaders, and produced an empty result rather than a wrong
one only by luck.

## What is not built yet

`figures.py` covers Entry 1 only. Entry 2 and Entry 3 figures are not written, because their
sections quote paired tests and per-scene tables rather than frontiers, and because both entries have
owed runs that will change what the right figure is. The corpora are already registered in
`sources.py`, so those builders start from a loader that works.
