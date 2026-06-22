# U4 fix_2 — visual-avoiding eval seed/yaml routing — CHANGELOG

**Date:** 2026-06-22
**Why:** visual-avoiding eval only ran seed 6 despite all of 6–10 trained. Root cause (see
[REPORT.md](REPORT.md)): the active eval read the **shared** `config/projection_eval.yaml`, while the
similarly-named `config/visual_avoiding_eval.yaml` (a wrong clone of the visual-*aligning* eval) was an
orphan — and the whole eval config (16 keys, not just seeds) came from one yaml. Fix: give visual avoiding
its own correctly-schema'd config and reactivate it.

## Changes
- **`config/visual_avoiding_eval.yaml`** — **rewritten** from aligning-style junk
  (`geo_constraint_variants`/`n_contexts`/scalar `dt`) to a faithful **avoiding** config mirroring
  `projection_eval.yaml` (half-spaces, obstacles, bounds, dims, projection variants, `ax_limits`), with its
  own `seeds: [6,7,8,9,10]`. Decoupled from the 41 scripts that share `projection_eval.yaml`.
- **Reactivated** — repointed the 4 active visual-avoiding scripts from `projection_eval.yaml` →
  `visual_avoiding_eval.yaml`:
  - `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py`
  - `fm_visual_avoiding_test/eval_fm_visual_avoiding.py`
  - `diffuser_visual_avoiding_test/load_results_visual_avoiding_dpcc.py`
  - `fm_visual_avoiding_test/load_results_fm_visual_avoiding.py`
- **Added** to both evals: `[ eval ] Seed list for this run: … (from config/visual_avoiding_eval.yaml)` —
  ends the "which seeds / which yaml" ambiguity in every log.

## Verified
- `visual_avoiding_eval.yaml` parses; all 16 keys the active eval reads present; `avoiding-d3il` constraint
  key present; `seeds=[6,7,8,9,10]`.
- All 4 scripts `py_compile` clean; both `open()` calls now target `visual_avoiding_eval.yaml`; no active
  visual-avoiding script still reads `projection_eval.yaml`.

## Not done / notes
- `projection_eval.yaml` untouched (shared by 41 scripts).
- Legacy `(legacy_based_on_visual_aligning)` scripts still expect the old schema — deprecated, out of scope.
- Per-seed fault isolation (try/except + continue) in the evals — not added yet (optional hardening).
- Local only; no commit/push, no cluster run. Next remote eval log should show all of `[6,7,8,9,10]`.
