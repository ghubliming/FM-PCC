# NOTES — where thesis figures and tables can come from

**Created:** 2026-09-13 · **Type:** hint · **Applies to:** every draft's figure pipeline
(`Data_Analysis/DA_in_Paper/plotting/`, official since 2026-09-14)

> **Hint.** Thesis figures do not have to come only from the curated report plots in
> `Data_Analysis/DA_Result_Curated_MD/`. The analysis **code** in `Data_Analysis/` — the DA pipelines
> and the visualizers — can be referred to, and its principles and code reused.

---

## The three sources, from cheapest to most flexible

| source | what it gives | reuse how |
| :-- | :-- | :-- |
| **`DA_Result_Curated_MD/`** — reports of record | finished panels, each with its own `make_figs.py` (e.g. `Report_20260903_AF_UNet/`) and the numbers the text cites | copy a panel, or copy its script. **Check provenance first** (see below) |
| **DA pipelines** — `DA_Code_v3/` (avoiding), `DA_VA_v2/` + `DA_Visual_Aligning/` (visual aligning), `DA_UAV_v1/` (UAV) | discover → load → aggregate → plot → report. `DA_Code_v3/batch_visualizer.py` already has Pareto frontier, success and time comparison, robustness boxplot, constraint heatmap and matrix plots | reuse the **aggregation semantics** and plot designs; run the pipeline for new outputs |
| **HTML visualizers** — `Visualizer/`, `Visualizer_VA_v2/`, `Visualizer_UAV_v1/`, `Visualizer_Visual_Aligning/` | the de-facto definitions of the Pareto band and front, the degenerate / `geo_free` tagging, and per-candidate highlighting | read them as the spec. **The page's `EXPORT ZIP` already emits PNG + LaTeX result matrices** — a ready starting point for thesis tables |

## Things worth knowing before reusing

- **The DA pipelines need matplotlib / numpy / pandas.** This container has none, so running them is
  a cluster job (`Slurm_Codes/sbatch/DA/`, `--no-plots` skips the slow PNGs and keeps the CSVs).
  `DA_in_Paper/plotting/svg/` is stdlib-only for exactly this reason, and `DA_in_Paper/plotting/mpl/` is
  where matplotlib scripts go. Porting a DA plot therefore means
  one of two things: re-implement its semantics in the stdlib pipeline, or run the DA on the cluster
  and bring the output in as a **vendored** figure with its provenance recorded.
- **Principles already shared, keep them shared.** v3's Pareto band and front follow
  `Visualizer_VA_v2/index.html`; its per-geometry-then-across aggregation reproduces the DAs of record
  exactly. Reusing DA code is the easiest way to keep thesis figures and DA numbers from drifting apart.
- **UAV corpora are richer than avoiding ones.** `batch_uav_*` carries `per_rollout_detail.csv`,
  `uav_k_sweep.csv` and `uav_units_long.csv` — per-episode data, so distributions and paired tests
  are possible there. The avoiding corpus stores per-cell aggregates only.
- **`Visualizer_VA_v2` is generated from the DAv3 page** by `build_from_dav3.py`, so a fix to
  `Visualizer/index.html` propagates by re-running that script, not by editing the VA page.
- **Serve the visualizers over HTTP**, not `file://` — they `fetch()` CSVs relative to their path
  (see `Data_Analysis/README.md`).

## 🚨 Provenance, whichever source is used

A copied panel is checked against `/workspaces/aux_repo/` before it enters a draft. The repo-root
`figures/avoiding*.png` are byte-identical to the DPCC authors' own copies and were nearly used as this
thesis's environment figures (`Working_Space/v3/CHANGELOG.md` v3.3). Every copied panel is recorded in
`Data_Analysis/DA_in_Paper/plotting/sources.py` → `VENDORED`, with its group, source path and provenance.
