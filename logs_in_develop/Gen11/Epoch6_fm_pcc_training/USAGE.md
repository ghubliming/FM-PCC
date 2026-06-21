# Gen11 Epoch 6 — UAV FM: USAGE (how to run)

All real runs are on the **cluster** (the Docker dev box has no torch/GPU/MuJoCo). Submit through the
repo standard wrapper `./Slurm_Codes/submit.sh <script> <args>` (gives dated unified logs).

---

## 0. Prerequisite — curated dataset must exist

The trainer reads **only** `data/uav_fm/v1/<scene>/` (never the raw E4 tree). Build it once (Phase-0 prep,
after the pillars recollect):
```bash
python uav_expert_data_collect/curate_dataset.py \
    --scenes empty corridor s_curve pillars \
    --pillars-src logs/uav_expert_data/pillars_v2 \
    --out data/uav_fm/v1
# → data/uav_fm/v1/<scene>/*.pkl + manifest.json
```
If your curated set lives elsewhere, set `export UAV_FM_DATA_ROOT=/abs/path` (loader honours it).

Optional cheap gate (data-side, not E6 code): `python uav_expert_data_collect/mini_fm_sanity.py
--data-dir data/uav_fm/v1/empty`.

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
