# C5 — Consolidate visual-aligning per-rollout pkl into the NPZ (IMPLEMENTED)

**Date:** 2026-07-09. **Scope:** `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7)
**and** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4) — the two
visual-**aligning** eval scripts. **Status:** code written for both; **must be validated on
the cluster** (no Python in the Docker container). See §Siblings for the scope decision.

Implements the plan in `TODO_npz_pkl_consolidation.md`. Before writing code, every claim in
that TODO was re-verified against the current source (misnamed `obs_all`, the 3 pkl-only
arrays, the trivial `results_seed` pkl, and that no repo consumer reads the pkls). All held.
Two decisions in the TODO were resolved toward the lower-risk / strictly-better option.

---

## What changed

### 1. NPZ is now the single source of raw truth (UAV-style schema)
The per-rollout raw + metric arrays are now built by a shared helper
`_collect_per_rollout_arrays(agent)` used by **both** the variant-end write and the new
crash-safety partial. Schema changes to `<variant>.npz`:

- **`obs_all` widened 3-D → 6-D:** `[des_c_pos(0:3) | c_pos(3:6)]`, matching
  `FM_v3_uav_test/eval_artifacts.py` and the model's actual 6-D input.
  - Chosen over the TODO's "separate `c_pos_all` key" option because widening matches the
    UAV schema exactly **and** is column-additive: cols `0:3` are byte-identical to the old
    3-D `obs_all` (still the commanded/desired path), so any consumer slicing `obs_all[:, :3]`
    is unaffected. **This resolves the old misnaming** — the actual executed path (was the
    pkl-only `c_pos_history`) now lives in cols `3:6`. **No key was renamed**, so no loader
    update is required.
- **`sampled_trajectories_all` now holds the FULL MPC candidate fan** (per replan step a
  `(B,H,3)` array) instead of the selected plan only. The fan was previously pkl-only
  (`all_candidates`). This also gains the `B` dimension that `npz_analysis/analyze_npz.py`'s
  plan-fan analysis already expects.
- **`selected_idx_all` added** (was pkl-only): which candidate index executed per replan.
- **`complete` flag added** (`True` on the authoritative variant-end write).
- All other keys are unchanged (same names, same construction): `success_*`, `n_steps`,
  `avg_time`, `mean_dist_per_rollout`, `physical_tracking_errors`, `context_*`, `contact_*`,
  `constraint_exec_*` / `constraint_plan_*`, `mode_encoding`, `mean_distance`, `args`.

⇒ The new npz is a **strict superset** of the pre-C5 keys (nothing dropped), plus
`selected_idx_all` / `complete`, with `obs_all` widened and `sampled_trajectories_all`
upgraded to the fan.

### 2. Deleted redundant pkls
- **`diagnostics/rollout_<r>_data.pkl`** write removed (it dumped the entire
  `master_rollout_history[...]` dict). Every raw array it held now lives in the npz.
- **`results_seed_<s>.pkl`** write removed (3 scalars `success_rate`/`entropy`/`elapsed`,
  all already implied by the npz).
- **Kept:** `diagnostics/rollout_<r>_stats.json` (metrics-only, human-readable, and the DA
  JSON-fallback path depends on it). `pickle` import stays — still used for expert data and
  normalizer loading.

### 3. Crash-safety: incremental `<variant>.partial.npz` sidecar
The pre-C5 scheme wrote pkls **per rollout** (survive a SLURM kill) but the npz **once at
variant end** — so naively dropping the pkls would lose all raw arrays for a killed variant.
Chosen mitigation (TODO §3 option b, "incremental"):

- New `_save_partial_npz(...)` writes **all raw arrays gathered so far** every
  `partial_npz_every` rollouts (default **5**, `geo_config`-settable via
  `partial_npz_every`). Write is **atomic** (`tmp.npz` → `os.replace`).
- **Deliberately a sidecar, not `<variant>.npz`.** The DA pipeline only reads
  `<variant>.npz`, which is still written exactly once, atomically, at variant END. This
  preserves today's clean *"missing npz ⇒ variant didn't finish"* semantics and avoids a
  silent-correctness hazard where DA would otherwise aggregate over a partial run thinking
  it was complete. The partial (`complete=False`, best-effort aggregates) is pure raw-data
  insurance and is **deleted on successful variant completion**.
- Trigger lives in `_export_rollout_realtime`, **outside** the plotting try/except, so a plot
  failure never skips the snapshot (and a snapshot failure is reported distinctly).

⇒ **Net: strictly ≥ the old pkl scheme in every case.** Completed variants get a single
clean file that now *also* carries the actual path + full fan + selected_idx the npz
previously lacked; killed variants keep the last N-block of rollouts' raw arrays in the
sidecar (vs. losing them under a naive npz-only approach).

---

## Downstream impact (repo-side, verified)
- `Data_Analysis/DA_Visual_Aligning/data_loader.py::_load_npz` reads generic `data.files`
  into a dict — additive keys are safe; **no rename**, so nothing to update. It never opened
  the deleted pkls. Its `load_all_candidates` refers to *experiment folders*, not the MPC
  fan.
- `npz_analysis/analyze_npz.py` reads `obs_all`/`sampled_trajectories_all` generically and
  has no `aligning` branch (uses CLI-fallback cols) — unaffected by the widening/fan change.
- No consumer of `_data.pkl` / `results_seed_*.pkl` exists under `Data_Analysis/`,
  `npz_analysis/`, `misc_tools_in_develop/`, `Results_and_Data_Analysis_Colab_T4/`,
  `ipynbs_Colab/`. (Ad-hoc **cluster-side** scripts, if any, would need updating — flagged.)

## Siblings — scope decided
- ✅ **`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4)** — mirrored
  with the **identical** change (this is the parent the FM script was copy-modified from; the
  agent state, master-rollout dict keys, npz save block, PNG grid, and `_export_rollout_realtime`
  are byte-for-byte equivalent, so `_collect_per_rollout_arrays` / `_save_partial_npz` and all
  edits transfer verbatim). `py_compile` OK.
- ⛔ **`imf_visual_aligning_test/`** — out of scope: not ready yet.
- ⛔ **`fm_visual_avoiding_test/`, `diffuser_visual_avoiding_test/`,
  `ddpm_encdec_vision_test/`, `fm_encdec_vision_test/`** — out of scope: these are **not**
  the visual-aligning task, so the C5 consolidation is not applied here.

⇒ Both in-scope aligning eval scripts (FM + diffuser) now share the C5 schema. Validate on
cluster before considering any further propagation.

## Verify (RUN ON CLUSTER — no Python in Docker)
0. `python -m py_compile` passed **locally** for both scripts (stdlib-only, no torch/numpy
   import) — syntax is valid; functional checks below still need the cluster.
1. `python -m py_compile fm_visual_aligning_test/eval_fm_visual_aligning.py
   diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`.
2. One small eval **per engine** (FM + diffuser); confirm `<variant>.npz` now has `obs_all`
   shape `(T,6)`, a full-fan
   `sampled_trajectories_all`, `selected_idx_all`, and `complete=True`; that **no**
   `_data.pkl` / `results_seed_*.pkl` are written; and that a mid-run `<variant>.partial.npz`
   appears and is gone after the variant completes.
3. `DA_Visual_Aligning/main_da_batch.py` still loads clean (npz + json sources).
