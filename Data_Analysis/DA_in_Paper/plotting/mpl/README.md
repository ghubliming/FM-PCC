# `mpl/` — matplotlib figure scripts

For figures the stdlib SVG canvas cannot draw well (renders, dense scatter, heatmaps, image grids).

- **Runs where matplotlib exists** — the cluster environment or a laptop. The AI container has none.
- **Saves straight into the store:** `../../figures/<group>/<name>.pdf` (plus `.png` if useful). The
  name must match what the draft includes and must not exist in another group.
- **Reads data paths from `../sources.py`**, never hard-coded.
- Once the file exists, remove the figure's entry from `PLANNED` in `sources.py`, run `make_figs.py` to
  refresh the manifest, then `export_to_draft.py <draft>`.

Starting point for the obstacle-avoidance constraint panel (`fig_constraints_avoiding`, PLANNED):
`scripts/visualize_data_constraints.py` at the repo root — it needs the dataset and the D3IL package.
Its saved outputs in the repo-root `figures/` are identical to the DPCC authors' copies and are not
used; regenerate instead.
