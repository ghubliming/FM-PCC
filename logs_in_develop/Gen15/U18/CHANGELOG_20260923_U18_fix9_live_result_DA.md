# CHANGELOG — Gen15 U18 · fix9 (2026-09-23) · R39 live results read, DA of record, figures

**Inspection of jobs 26151–26154 (tag `p23uavpv2live`):** complete and clean — 240 flights, 4 × 30 cells, no error, no
divergence, no step-cap end, sidecars one file per env close (fix8 works: 15 files per CI-MeanFM job, 3 per MeanFM
seed call). Not our fault: every failed flight is a pillar contact at exactly the rotor reach, and in 50 of 51 the
planner's commanded path itself passes within the reach (analysis §4 has the checks). The batch folder the author ran
(`pillars-batch_uav_20260923_172422`) is the UAV reporter over `UAV_MIX`/`UAV_FM` and does not contain these cells
(they live in the avoiding tree) — not used.

| file | what |
| :-- | :-- |
| `Data_Analysis/analysis_results_checkpoint/23-09-UAV-Pillars-v2/live_p23uavpv2live/` | corpus of record (13 MB): the 120 npz, eval logs, provenance, 60 sidecars, 4 job logs |
| `Data_Analysis/DA_in_Paper/analysis/pillars_v2_live.py` | the analysis script: table vs air, per geometry, failure anatomy, plant stats |
| `Data_Analysis/DA_in_Paper/analysis/DA_20260923_pillars_v2_live.md` | the DA of record; `analysis/INDEX.md` row (pillars_hg row struck) |
| `Data_Analysis/DA_in_Paper/plotting/extract/pillars_v2_paths.py` | writes `data/uav_paths.json` figures `pillars` (MeanFM nfe 1) and `pillars_cimf` (CI-MeanFM nfe 1): 6 panels each, flown paths mapped into the arena |
| `Data_Analysis/DA_in_Paper/plotting/builders/paths.py` | per-panel avoiding geometry (`UAV_PILLARS_GEOMETRIES`), `fig_uav_pillars_paths` re-pointed, `fig_uav_pillars_paths_cimf` added; legend text |
| `figures/da/fig_uav_pillars_paths{,_cimf}.svg/.png` | built by `make_figs.py`; PNG previews by `svg/preview_png.py` |
| `cross_draft/to_v3/FROM_DA_20260923_pillars_v2_live_result.md` + INBOX row | the number and the finding for the chapter |
| `PENDING_20260923…` §2, `SLURM_RUNBOOK_20260923…` §5 | DONE markers |
| `Slurm_Codes/temp_bash/export_20260923_p23_pillars_live.sh` | the tar.gz export used to fetch the results |

Not done / open: the thesis text itself (v3's); the figure export to the draft (`export_to_draft.py v3`, once referenced).
