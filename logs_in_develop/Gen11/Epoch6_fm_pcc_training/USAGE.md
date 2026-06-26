# Gen11 Epoch 6 — UAV FM: USAGE (how to run)

All real runs are on the **cluster** (the Docker dev box has no torch/GPU/MuJoCo). Submit through the
repo standard wrapper `./Slurm_Codes/submit.sh <script> <args>` (gives dated unified logs).

---

## 0. Prerequisite — produce the dataset, then curate it (uses the E4 pipeline)

> **Nothing here deletes or moves your raw data.** Curate **copies** (`shutil.copy2`) raw pkls into
> `data/uav_fm/v1/`; `logs/uav_expert_data/` is left untouched. The only manual archive step is the old
> pillars, and we **`mv` (archive), never `rm`** — so no data is lost.
>
> **Case A — you already have good empty/corridor/s_curve, only pillars to redo (likely):**
> ```bash
> mv logs/uav_expert_data/pillars logs/uav_expert_data/_archive_pillars_274   # archive, don't delete
> ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars 500   # fresh pillars only
> ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/prepare_uav_fm_data.sh # curate all 4 + verify
> ```
> Do **NOT** run the full pipeline here — it re-collects all 4 scenes and would mix into your good dirs.
>
> **Case B — fresh start (no data yet), fully batched, no login-node steps:**
> ```bash
> ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/collect_to_ready_pipeline.sh 500
> ```
> When the `prepare` job logs `DATASET READY ✓`, go to §1. (Details: Aux-1 CHANGELOG.) The manual steps
> below remain valid if you prefer to run them yourself.

The trainer reads **only** the curated, **flat** tree `data/uav_fm/v1/<scene>/*.pkl`. The raw E4 collection
is nested (`logs/uav_expert_data/<scene>/<homotopy>/*.pkl` + `run_summary.json`), so the curate step
(which **flattens** the homotopy subdirs and drops `run_summary`) is **required**, not optional.

### 0.1 Collect with the E4 SLURM pipeline (one job per scene)
Uses `Slurm_Codes/sbatch/uav_expert_data/collect.sh` — headless MuJoCo PID rollouts, no GPU; it runs
`collect.py` then `stats_validator.py`. Output → `logs/uav_expert_data/<scene>/`.
```bash
# args: <scene> <n_trials> [gain] [seed_offset] [homotopy]
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty    500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  500   # BLEND_RADIUS=0.45 fix is already in trajectories.py
```
- **Pillars:** the U9 fix (`BLEND_RADIUS=0.45`) is already applied, so a fresh collect is clean. **Archive
  the old 274-episode pillars dir first (move, never delete)** so the recollect doesn't mix with it:
  `mv logs/uav_expert_data/pillars logs/uav_expert_data/_archive_pillars_274`, per U9 "do not mix".
- empty/corridor/s_curve from the U9 run are already clean — re-collect only if you don't have them.
- Smoke first if unsure: `… collect.sh empty 10`.

### 0.2 Curate raw → flat training set (`data/uav_fm/v1/`)
`curate_dataset.py` is pure-stdlib (no GPU/torch) — run it on the **cluster login node** (or anywhere the
raw pkls are). It recurses the homotopy subdirs, copies only accepted `*.pkl` (skips `run_summary`/stress),
**flattens** them per scene, and writes a `manifest.json`:
```bash
python uav_expert_data_collect/curate_dataset.py \
    --scenes empty corridor s_curve pillars \
    --out data/uav_fm/v1
# → data/uav_fm/v1/<scene>/*.pkl (flat) + manifest.json
```
(If you recollected pillars into a separate dir instead of clearing, add
`--pillars-src logs/uav_expert_data/<your_pillars_dir>`.)

### 0.3 Manual alternative (if you'd rather move files yourself)
The loader just needs **flat per-scene pkls**:
```
data/uav_fm/v1/<scene>/<episode>.pkl      # NO homotopy subdirs, NO run_summary.json
```
So you can skip 0.2 and move them by hand, e.g.
`mkdir -p data/uav_fm/v1/pillars && find logs/uav_expert_data/pillars -name '*.pkl' -exec cp {} data/uav_fm/v1/pillars/ \;`
(flatten the homotopy subdirs). Or point the loader elsewhere: `export UAV_FM_DATA_ROOT=/abs/flat/root`.

> ⚠ Don't point `UAV_FM_DATA_ROOT` at the raw `logs/uav_expert_data` — the loader lists a scene dir
> non-recursively, so it would see the homotopy *subdirs* (not `.pkl`s) and load **zero** episodes.

### 0.4 Optional cheap gate (data-side, not E6 code)
```bash
python uav_expert_data_collect/mini_fm_sanity.py --data-dir data/uav_fm/v1/empty
```

---

## 1. Train

**Pooled 4-scene model (the milestone):**
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh all 5
```
**A single scene (ablation / debugging):**
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh empty 5
# scenes: empty | corridor | s_curve | pillars
```
Local (cluster shell, no SLURM): `python FM_v3_uav_test/train_fm_uav.py --scene all --seed 5`
Multi-seed: `--seeds 5 6 7`. W&B: add `--use-wandb`.

Output → `logs/uav-<scene>/flow_matching_v3_uav.../<seed>/` (`weights/`, `losses.pkl`, `*_config.pkl`).

---

## 2. Eval (closed-loop MuJoCo)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh all 5 20      # scene seed n_trials
```
Local: `python FM_v3_uav_test/eval_fm_uav.py --scene all --seed 5 --n-trials 20`

Writes `logs/uav-<scene>/.../<seed>/eval/results.json` per scene and, for `--scene all`,
`logs/uav-all/SUMMARY.json`.

---

## 3. One-shot pipeline (train → eval, chained)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh all 5 20
```
Eval runs only if train succeeds (`--dependency=afterok`).

---

## 4. Read results

Per scene: `logs/uav-<scene>/.../<seed>/eval/results.json` → `summary.success_rate`,
`contact_frac_mean`, `track_err_mean`, `fm_ms_mean`/`fm_ms_p95`, `goal_dist_mean`.
Cross-scene: `logs/uav-all/SUMMARY.json`.

```bash
python - <<'EOF'
import json; d=json.load(open('logs/uav-all/SUMMARY.json'))
for s,v in d.items():
    print(f"{s:9s} success={v['success_rate']:.3f}  contact={v['contact_frac_mean']:.3f}  "
          f"fm_ms p95={v['fm_ms_p95']:.1f}")
EOF
```

---

## Notes / gotchas
- **Timing gate:** `fm_ms_p95` is the deployability number (target ≤ ~30 ms for 33 Hz). It's measured
  live during eval; do not read it from a reloaded model.
- **Success metric:** defaults to *contact-free + airborne* (the expert's own gate). The state-only FM is
  not goal-conditioned, so `goal_dist` is secondary — see CHANGELOG "Open design decision".
- **Schema:** H=8, obs=9 `[p_des|p|v]`, action=3 `Δp_des`, transition=12. Dims auto-derive from data.
- **Scene ↔ output:** `--scene X` ⇒ dataset string `uav-X` ⇒ both the data branch and the output path.
