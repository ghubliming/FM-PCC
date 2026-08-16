# CHANGELOG — Gen12 fix_5: FMv3ODE-style output path layout

**Date:** 2026-07-25 · **Type:** fix (output paths) · **Status:** code complete, verified static
**Follows:** U4/U4.2 coding + resolver fix. **Nothing committed.** Nothing run.

---

## 0. TL;DR

Gen12's eval buried the run knobs (K, activation threshold, MPC candidate count) as a `run_tag`
subdir **under** `results/halfspace_<hv>/`. fix_5 moves them up into a proper **eval-name folder**,
so the layout strictly follows the FMv3ODE / visual convention:

```
plans/flow_matching_v3_hardflow/<TRAIN-NAME>/<EVAL-NAME>/<seed>/results/halfspace_<hv>/<variant>.{npz,png}
```

- **TRAIN-NAME** = the loaded checkpoint's identity, e.g.
  `H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10`
- **EVAL-NAME** = the run's eval config, e.g. `K20_thres0.5_mpc1_n2`

This mirrors `flow_matching_v3_ode_selectable` (whose visual sibling writes
`…T0.5_D…_mpc4/6/results/halfspace_both-hard/diffuser.png` — threshold and `mpc` in the
eval-name folder, seed next, then `results/`).

## 1. Before → after

| | before (U4/U4.2) | after (fix_5) |
|---|---|---|
| eval knobs live in | `results/halfspace_<hv>/K…_thres…_mpc…/` (buried) | **`<train>/<eval-name>/`** (a folder level, FMv3ODE-style) |
| full npz path | `…/H8_K10_D…/6/results/halfspace_both-hard/K20_n2_thres0.5_mpc1/hardflow_new.npz` | `…/H8_D…_aw10/K20_thres0.5_mpc1_n2/6/results/halfspace_both-hard/hardflow_new.npz` |
| train vs eval identity | mashed into one folder (`H8_K10_D…`) | **split**: train-name (checkpoint) / eval-name (run) |

## 2. Files changed

| file | change |
|---|---|
| `FM_v3_hardflow_test/hf_paths.py` | **new** — single source of the path layout: `train_name()`, `eval_name()`, `eval_root()`. Imported by both eval and load_results so they can't drift. |
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | after model load, rebuild `args.savepath = <train>/<eval-name>/<seed>` via `hf_paths`; `results/halfspace_<hv>/` no longer carries a run_tag; `all_seeds/` sits at `<train>/<eval>/all_seeds/`. |
| `FM_v3_hardflow_test/load_results_FM_v3_hardflow.py` | reads the same `hf_paths` layout per seed; summary-plot name is the eval-name; deduped imports; docstring updated. |

`hf_paths` (the layout contract):

```python
train_name = basename(checkpoint_dir or diffusion_loadpath)     # checkpoint identity
eval_name  = f'K{K}_thres{thr:g}_mpc{batch}_n{n}'               # run identity
eval_root  = <logbase>/<dataset>/plans/flow_matching_v3_hardflow/<train>/<eval>/<seed>
```

## 3. Why the split matters

- **Train vs eval identity are now distinct folders**, exactly like FMv3ODE: the checkpoint you load
  is one level, the eval configuration (K, threshold, MPC fan, n) is the next. Sweeping any eval knob
  makes a sibling eval-name folder under the same train-name — no collisions, and every run is
  self-describing from its path.
- **The knobs are visible at a glance** in the eval-name (`K20_thres0.5_mpc1_n2`) rather than buried
  four levels down under `results/`.
- Arms A/B are threshold/mpc-invariant, so their numbers repeat across eval-name folders (expected);
  for a pure arm-C sweep, set `projection_variants` to arm C only.

## 4. Verification (static — nothing executed)

- All three touched files compile.
- Rendered layout confirmed end-to-end:
  ```
  plans/flow_matching_v3_hardflow/H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/
      K20_thres0.5_mpc1_n2/6/results/halfspace_both-hard/hardflow_new.npz
  ```
- eval writes and load_results reads the **same** path (shared `hf_paths`).

## 5. Migration note

Results from the previous runs (jobs 23829–23831) sit in the OLD layout
(`…/H8_K10_D…/6/results/halfspace_*/K20_n2_thres*_mpc*/…`) and will **not** be found by the new
`load_results`. They are not deleted — re-run eval to populate the new layout, or read the old npz
directly. No production data is lost; only the directory convention changed.

## 6. Unchanged
- The sampler / NLP / U4 threshold / U4.2 selection logic — untouched. fix_5 is purely where files
  land.
- The K=20-vs-10 cluster-config issue is orthogonal (still: check the cluster plan block / pass
  `--flow-steps`).
