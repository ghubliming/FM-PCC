# Gen11 E6 — U2: Per-Scene FM models + scene-keyed run structure (PLAN)

**Date:** 2026-06-22
**Status:** PLAN only — no code yet.
**Supersedes:** the EPOCH6_PLAN "milestone = pooled `--scene all`" idea.

---

## 0. Why (the decision, settled)

A single **universal state-only** FM across the 4 scenes is **underdetermined**, not just imbalanced: the FM
sees only `obs=[p_des,p,v]` (no obstacles, no scene id), so the *same* state maps to *contradictory* expert
outputs across scenes (straight in `empty` vs weave in `pillars` around a pillar it can't see) → it blends
the modes and samples wrong-scene behavior. **Per-scene works** because within one scene the geometry is
constant, so state→action is well-defined (the fixed obstacles are implicitly baked into the weights) —
exactly how every legacy FMPCC/DPCC model works (one model per fixed env). A universal model only becomes
meaningful with **vision (Gen7)** or explicit scene/obstacle conditioning — not this Epoch.

⇒ **This Epoch: one FM per scene (4 models). `--scene all` is demoted to experimental.**

---

## 1. Goal & guiding constraint

Train + eval **one FM per scene**, organized under a **scene-keyed run tree**, via a thin **outer loop**
over the existing per-scene `train_fm_uav.py` / `eval_fm_uav.py` (which already take `--scene <one>`).
**Minimal change** — add orchestration + output nesting; do **not** touch the FM model/dataset/training code.

Desired nesting (your spec): **scene → {train line, eval line} → seed → projection-variant → …**

---

## 2. Proposed run / output structure

```
logs/fm_uav/                                   ← top-level UAV-FM root
├── empty/                                      ← SCENE (top level "which scene")
│   ├── train/
│   │   └── seed_<s>/  { weights/, losses.pkl, *_config.pkl }     ← FM TRAIN line (per seed)
│   ├── eval/
│   │   └── seed_<s>/
│   │       └── <projection>/ { results.json, plots/, timing/ }   ← FM EVAL line (per seed × projection)
│   └── SCENE_SUMMARY.json                       ← this scene, across seeds (success/contact/timing)
├── corridor/   … (same)
├── s_curve/    … (same)
├── pillars/    … (same)
└── ALL_SCENES_SUMMARY.json                      ← roll-up across the 4 scenes
```

- **`<projection>` axis** (your "projection like both hard"): this Epoch is state-only FM, so the only value
  is **`fm_only`** (no DPCC yet). The level is created now so the DPCC variants (`dpcc-c`,
  `dpcc-c-tightened`, …) slot in **without restructuring** when Phase-3 projection lands. (It is the UAV
  analogue of the avoiding eval's `projection_variants`; UAV **scenes** are the analogue of the avoiding
  `halfspace_variants` — they live at the *top*, not inside eval.)
- One model **per (scene, seed)**. Seeds give the generative-policy spread, same as legacy FM evals.

---

## 3. What changes vs what stays (minimal-change ledger)

**Stays exactly as-is (no edits):**
- `flow_matcher_v3_uav/` (model, dataset `uav-<scene>` branch, sampling, training loop).
- `config/uav.py` (block `flow_matching_v3_uav`).
- The core `train_fm_uav.py` / `eval_fm_uav.py` logic — they already run **one scene** per invocation.

**Small edits (output nesting only):**
- `train_fm_uav.py`: write under `logs/fm_uav/<scene>/train/seed_<s>/` instead of the Parser default
  `logs/uav-<scene>/<exp_name>/<seed>/`. *(Option A: leave the Parser path as-is and just symlink/alias —
  zero code change; Option B: small savepath override. Pick A to honor "don't change too much".)*
- `eval_fm_uav.py`: add a `--projection {fm_only}` arg (default `fm_only`) and write results under
  `eval/seed_<s>/<projection>/`; keep the existing metrics. Roll up `SCENE_SUMMARY.json`.

**New (the outer loop — the actual deliverable):**
- `Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh` — loops `scene × seed`, submits `train_fm_uav.sh <scene>
  <seed>` for each (4 scenes × N seeds = fan-out jobs).
- `Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh` — loops `scene × seed`, submits `eval_fm_uav.sh`.
- `Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh` — one-shot: for each scene×seed, train→eval chained
  (`--dependency=afterok`); fully batched (no login-session steps).
- A tiny `aggregate_scene_summaries.py` (stdlib) → `ALL_SCENES_SUMMARY.json`.

---

## 4. How you'd run it (target UX)

```bash
# all scenes, all seeds, train→eval chained, one submit:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh "empty corridor s_curve pillars" "5 6 7"

# or just one scene's line:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh pillars "5 6 7"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh  pillars "5 6 7"
```
Read: `logs/fm_uav/<scene>/SCENE_SUMMARY.json` per scene, `logs/fm_uav/ALL_SCENES_SUMMARY.json` overall.

---

## 5. Decisions to confirm before coding

| # | Decision | Lean |
|---|---|---|
| D1 | Output path: alias/symlink to existing Parser path (A) vs savepath override (B) | **A** (zero FM-code change) |
| D2 | `--scene all` pooled model | **keep as experimental flag, not in the loop / not the milestone** |
| D3 | Seeds per scene | match legacy (`5 6 7` or `5..9`) — your call |
| D4 | `<projection>` level now | create with single value `fm_only`; DPCC variants deferred to Phase 3 |
| D5 | One model per (scene,seed) vs per scene only | **per (scene,seed)** — needed for the generative spread |

---

## 6. Risks / notes

- **Job count:** 4 scenes × N seeds × {train,eval} — e.g. 4×3×2 = 24 jobs. The fan-out pipeline handles it,
  but watch the queue. Per-scene `train_all_scenes.sh <scene>` lets you do them in waves.
- **`max_path_length`** (separate bug, still open): `600` truncates long s_curve episodes (~726 steps) →
  bump to ~750 in `config/uav.py` before training s_curve. Fold into this work.
- **Imbalance** is now moot per-scene (each scene uniform within itself).
- Universal/visual model = **Gen7, next Epoch** — this structure leaves room (`<projection>` + a future
  `visual/` sibling) without rework.

---

## 7. Deliverables (when approved)

| File | Purpose |
|---|---|
| `Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh` | outer loop: train per scene×seed |
| `Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh` | outer loop: eval per scene×seed |
| `Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh` | one-shot train→eval fan-out |
| `FM_v3_uav_test/eval_fm_uav.py` (small edit) | `--projection` arg + `eval/seed_/<proj>/` nesting + `SCENE_SUMMARY.json` |
| `FM_v3_uav_test/aggregate_scene_summaries.py` | `ALL_SCENES_SUMMARY.json` roll-up |
| `config/uav.py` (1-line) | `max_path_length: 600 → 750` |
| CHANGELOG.md (this folder) | record on implementation |

*No code changed — plan only. Confirm §5 (esp. D1/D2) and I'll implement.*
