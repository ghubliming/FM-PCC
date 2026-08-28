# Analysis scripts for DA_20260827_mpc1_full_seeds_state_avoiding.md

Pure-Python stdlib only (this container has no numpy/scipy/matplotlib).

| file | role |
|---|---|
| `stats.py` | exact Wilcoxon signed-rank (full enumeration), exact sign test, paired bootstrap CI, Wilson CI |
| `svglib.py` | minimal SVG writer used by the figure scripts |
| `figs.py` | Figures A (cost decomposition) and B (safety–cost plane) |
| `figs2.py` | Figures C (paired blocks) and D (selection collapse) |
| `tables.py` | `results_mpc_fan_20260827.csv` and `table_mpc_fan.tex` |
| `png.py` | dependency-free PNG encoder + 5×7 bitmap font, used for `simple_summary.png` |

Input is `perseed.json`, extracted from `candidates_multidimensional_raw.csv` of batch DA
`batch_avoiding_combined_20260827_224347` (candidates 12, 17, 45, 47, 66, 67, 138, 149, 151),
keyed `<leg>|<variant>|<metric>` -> `<seed>|<scenario>` -> value.

## Figures

| file | use |
|---|---|
| `simple_summary.png` | one-glance overview — safety bars + speed-up bars |
| `figA_cost_decomposition.svg` | generator vs projection, fan 4 and fan 1 |
| `figB_safety_cost_pareto.svg` | safety–cost plane with fan 4 → fan 1 arrows |
| `figC_paired_blocks.svg` | per-block paired differences, bootstrap CIs, exact p |
| `figD_selection_collapse.svg` | -r / -c / -t identity at fan 1 |

The SVGs take their colours from `var(--fig-*, fallback)`, so they render standalone and also inherit a host page's theme when inlined.
