# C5 (TODO / not yet implemented) — consolidate visual-aligning per-rollout pkl into the npz

**Status:** PLAN ONLY. No code written yet. Author to implement later.
**Motivation:** The visual-aligning eval artifact layout is chaotic — raw trajectory data
is split across a per-variant `.npz` AND per-rollout `_data.pkl` files, with overlapping,
misleadingly-named, and partially-orphaned content. Consolidate onto the **cleaner UAV
pattern from `logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_12`** (one `<variant>.npz`
holding all raw arrays; JSONs are metrics-only; no per-rollout pkl).

---

## The reference pattern (UAV, Gen11 Fix_10/Fix_12)

`FM_v3_uav_test/eval_artifacts.py` writes ONE `<variant>.npz` per variant with object
arrays, one entry per rollout:

- `obs_all` — (T, 9) `[p_des(0:3) | p(3:6) | v(6:9)]` → **both commanded AND actual
  position in one array**
- `act_all` — (T, 3) executed action per step
- `sampled_trajectories_all` — list of (B, H, obs_dim) → **the FULL MPC candidate fan of
  every replan** (not just the selected plan)
- flat metric vectors (`success_*`, `phys_*`, `constraint_*`, `goal_*`)

Per-rollout `diagnostics/rollout_<i>_stats.json` = **metrics only** (heavy arrays stripped
via `HEAVY_KEYS`). No `_data.pkl`. This is the target shape.

---

## Current visual-aligning state (what's wrong)

File: `fm_visual_aligning_test/eval_fm_visual_aligning.py`

### The npz (`np.savez` at ~line 2238) stores:
- `obs_all` ← `d['real_robot_pos']` — ⚠ **misnamed**: `history_real_pos` actually appends
  `des_robot_pos_np` (lines 1622/1672), i.e. these are DESIRED positions, not real.
- `act_all` ← `d['desired_actions']`
- `sampled_trajectories_all` ← `d['full_plans']` — ⚠ the **SELECTED candidate's plan only**,
  not the full fan.
- many metric vectors (success, context, constraint_metrics, contact steps, …).

### The per-rollout `diagnostics/rollout_<r>_data.pkl` (written at ~line 1167) stores the
ENTIRE `master_rollout_history['rollout_<r>']` dict (built ~lines 995–1035):
- `real_robot_pos` (desired), `c_pos_history` (**ACTUAL executed path**),
  `desired_actions`, `full_plans` (selected plan),
- `all_candidates` (**FULL candidate fan**, list of (B,H,3)), `selected_idx`,
- scalars/dicts: `success`, `success_relaxed`, `mean_distance`, `mode`, `steps`,
  `avg_time`, `max_physical_tracking_error`, `act_magnitudes`, `dist_to_target`,
  `clamp_events`, `context_info`, contact first/last step+pos, `constraint_metrics`.

### `results_seed_<s>.pkl` (next to npz, ~line 2286): trivial — just
`{success_rate, entropy, elapsed}` (3 scalars, already implied by the npz).

### Three raw arrays live ONLY in the pkl today (would be lost if pkls deleted naively):
1. `c_pos_history` — the actual executed trajectory
2. `all_candidates` — the full MPC candidate fan
3. `selected_idx` — which candidate was executed each replan

---

## Downstream-consumer audit (already done — safe to consolidate)

Verified nothing in the repo's analysis pipeline reads the pkls:
- `Data_Analysis/DA_Visual_Aligning/data_loader.py` reads `<variant>.npz`
  (`np.load(..., allow_pickle=True)`, generic `data.files`) and can reconstruct from
  `diagnostics/rollout_*_stats.json`. It NEVER opens `_data.pkl` or `results_seed_*.pkl`.
- `npz_analysis/analyze_npz.py`, `Data_Analysis/DA_Code_v3/*` — npz-only.
- No consumer of `_data.pkl` / `results_seed_*.pkl` anywhere under `Data_Analysis/`,
  `npz_analysis/`, `misc_tools_in_develop/`, `Results_and_Data_Analysis_Colab_T4/`,
  `ipynbs_Colab/`.

⇒ Repo-side consolidation is safe. Only ad-hoc cluster-side scripts (if any exist) would
need updating.

---

## Plan (implement later)

### 1. Extend the npz to be the single source of raw truth
In the `np.savez` call, add/repair:
- **Fix `obs_all`** to carry both desired AND actual position, mirroring UAV's
  `[p_des | p | ...]`. Either widen `obs_all` to a concatenated `[des | c_pos]` per step,
  or add a separate `c_pos_all` (actual executed path from `c_pos_history`). Widening is
  closer to the UAV schema; a separate key is a smaller diff. **Decide one; document it.**
- **Add `sampled_candidates_all`** ← `all_candidates` (the FULL fan, list of (B,H,3)) so
  the fan is no longer pkl-only. Keep `sampled_trajectories_all` (= selected `full_plans`)
  for backward compat, OR rename cleanly and bump a schema note.
- **Add `selected_idx_all`** ← per-rollout `selected_idx`.
- Fold any remaining pkl-only scalar/list fields into flat vectors (most already are:
  context, constraint_metrics, contacts — confirm none are dropped).
- ⚠ **Rename note:** `obs_all`-is-desired is a latent trap; if renaming, update
  `DA_Visual_Aligning/data_loader.py` + `DA_Code_v3` in the same change (they read keys
  generically, so ADDING keys is safe; RENAMING existing keys needs a loader update).

### 2. Drop the redundant pkls
- Remove the `rollout_<r>_data.pkl` write (~line 1167) once (1) lands.
- Remove the `results_seed_<s>.pkl` write (~line 2286) — its 3 scalars are already in npz.
- Keep `diagnostics/rollout_<r>_stats.json` (small, human-readable, and the DA JSON
  fallback path depends on it).

### 3. Guard the crash-safety regression (IMPORTANT)
Pkls are currently written **per rollout as each finishes**; the npz is written **once at
variant end**. On 24 h SLURM-limit jobs, a killed variant currently keeps completed
rollouts' pkls but npz-only would lose all raw data for that variant. Mitigations (pick):
- (a) keep `rollout_<r>_stats.json` per rollout (metrics survive a kill) — minimum, and
  already true; OR
- (b) re-write the npz incrementally every N rollouts; OR
- (c) accept the risk and document it.
UAV eval accepts this (npz once at end) + relies on per-rollout `.log`/stats — matching (a)
is the cheapest consistent choice.

### 4. Mirror across sibling generations (repo convention)
The pkl-per-rollout pattern is Gen6V4 heritage shared by siblings. Decide + document
whether to mirror or intentionally defer for each:
- `diffuser_visual_aligning_test/`
- `fm_visual_avoiding_test/`, `diffuser_visual_avoiding_test/`
- `imf_visual_aligning_test/`
- `ddpm_encdec_vision_test/`, `fm_encdec_vision_test/`
(Check each with the copy-modify sibling convention before editing only one.)

### 5. Changelog
Write a Fix_12-style changelog in this folder
(`logs_in_develop/Gen7_FMPCC_Viusal_Aligning/npz_pkl_Org_C5/`) covering schema changes,
the `obs_all` rename decision, deleted files, crash-safety choice, and which siblings were
mirrored vs deferred.

### 6. Verify (on cluster — no Python in Docker)
- `py_compile` the eval script locally.
- Run one small eval; confirm `<variant>.npz` now contains `c_pos`/fan/`selected_idx`,
  no `_data.pkl` written, and `DA_Visual_Aligning/main_da_batch.py` still loads clean.

---

## Note
Cross-reference: root diagnosis of the artifact chaos and the UAV reference schema are in
`logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_12/REPORT_fix12_constraint_strictness_homotopy_rawdata.md`
(Q3) and `.../Fix_12/CHANGELOG_fix12_constraint_feasibility_and_homotopy.md`.
