# `plotting/` — the thesis figure pipeline

Moved here from `Working_Space/v3/plots/` on 2026-09-14. Output is byte-identical to the v3 pipeline it
replaces (checked on all ten figures at the move).

```bash
python3 make_figs.py                 # build everything into ../figures/<group>/
python3 make_figs.py k_ladder        # only figures whose name contains this
python3 make_figs.py --list          # corpora, builders, vendored, planned
python3.14 prep/extract_env_frames.py # prepare registered stills from source GIFs
python3 export_to_draft.py v3        # copy the figures v3 uses into the draft (--dry-run, --prune)
svg/svg2pdf.sh                       # SVG -> PDF for the whole store
```

## Layout

| path | role |
| :-- | :-- |
| `sources.py` | **The only file with data paths.** `CORPORA` (batch directories + protocol + readiness), `VENDORED` (figures copied in, with group and provenance), `PLANNED` (figures a draft asks for that do not exist yet), `EXCLUDED` (images never to be used), the aggregation rule and the vocabulary. |
| `builders/` | One module per environment; `builders/__init__.py` lists every builder with its group. `avoiding.py` holds the six obstacle-avoidance charts. |
| `svg/` | `fmpcc_svg.py`, the stdlib SVG canvas (runs in the AI container); `svg2pdf.sh`. |
| `mpl/` | matplotlib scripts, for figures that need it — run where matplotlib exists (cluster, laptop). See `mpl/README.md`. |
| `extract/` | Steps that need numpy or PyYAML (so `python3.14`, not `python3`): they read configs, datasets and logs once and write plain JSON into `../data/`, which the builders then read. `avoiding_scene.py` writes the obstacle-avoidance geometry and its 96 demonstrations. |
| `prep/` | Deterministic preparation of vendored inputs. `extract_env_frames.py` selects registered frames and crops from source GIFs, including the two $96\times96$ alignment camera observations from an expert demonstration; `crop_vendored.py` removes dashboard chrome. Both provide `--check`. |
| `svg/preview_png.py` | Renders a figure SVG to PNG with PIL (`python3.14`), for checking it by eye — the container has no SVG rasteriser. A preview, not the print path. |
| `make_figs.py` | Builds generated figures, copies vendored ones, writes `../figures/MANIFEST.md`. |
| `export_to_draft.py` | Copies the figures a draft references into `<draft>/figures/`, writes `EXPORTED.md`, reports planned and missing figures (missing exits 1). |

## Figure groups

| group | holds |
| :-- | :-- |
| `da` | charts computed from evaluation data |
| `demo` | raw demonstrations of behaviour: plans, rollouts, frames |
| `env` | environments and constraint sets: renders, scene and constraint panels |
| `schematic` | diagrams of the method |

A figure name is unique across groups; both tools refuse a name found in two groups.

## Checking a figure

There is no SVG viewer here, so render a preview and look at it:

```bash
python3.14 svg/preview_png.py ../figures/env/fig_constraints_avoiding.svg /tmp/check.png
```

It supports exactly the SVG that `fmpcc_svg.py` writes.

## When new data lands

Point the `Corpus` in `sources.py` at the new batch directory, update its protocol string, then
`make_figs.py` and `export_to_draft.py <draft>`. No builder contains a path.

## Adding a figure

- **From data, stdlib:** a builder `fn(outdir) -> (path, provenance) | None` in `builders/<env>.py`, added to
  that module's `ALL` and to `builders/__init__.py` with its group.
- **With matplotlib:** a script in `mpl/` that saves into `../figures/<group>/<name>.{pdf,png}`.
- **Copied from elsewhere:** check it against `/workspaces/aux_repo/`, then add it to `VENDORED`.
- **Asked for but not made:** add it to `PLANNED` with the draft label and the spec; remove it once the file
  exists.

Each builder returns `None` when its corpus is absent, puts the protocol in its subtitle, and uses the
thesis names (translation table), never code tokens.

## The aggregation rule

Obstacle-avoidance results are averaged **per geometry first, then across geometries** — never as a flat
mean over (seed × geometry) cells. The baseline's both-hard geometry has one seed, so the two differ;
only the per-geometry mean reproduces the analyses of record (0.983 / 69.0 steps / 0.5635 s for the
baseline, 0.993 / 61.0 / 0.0181 s for MeanFlow at K = 1). `geometry_mean` takes the geometry column index
as a required argument.

## Also usable

The analysis code in `Data_Analysis/` (`DA_Code_v3`, `DA_VA_v2`, `DA_UAV_v1`, the HTML visualizers and
their LaTeX table export) and the scripts in `Data_Analysis/DA_Result_Curated_MD/Report_*/make_figs.py`
can be reused here — see `logs_in_develop/Writing/Auxiliary/NOTES_plotting_sources.md`.
