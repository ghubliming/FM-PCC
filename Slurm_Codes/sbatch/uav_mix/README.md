# Gen15 — UAV Mix-ML SLURM Scripts

All submitted via `bash Slurm_Codes/submit.sh <script> [args]` from repo root.
Seeds always loop **inside** one job — adding seeds adds time, not jobs.

**`engine` is always the FIRST argument** (`fm` | `mf` | `af`). This is the only CLI difference
from Gen11's `uav_fm/` scripts; everything after it keeps Gen11's argument order.

| engine | objective | source |
|---|---|---|
| `fm` | Flow Matching + Euler ODE | Gen11 (the incumbent / parity arm) |
| `mf` | MeanFlow, arXiv 2505.13447 | Gen3v6 |
| `af` | α-Flow, arXiv 2510.20771 | Gen3v7 |

Gen15 writes to **`logs/UAV_MIX/`**. Gen11's `logs/UAV_FM/` is never touched.

---

## Scripts

| Script | What it does | Args |
|---|---|---|
| `gates_mix_uav.sh` | **Run this first.** Wiring gates, ~minutes | `$1=device` `$2=gates` |
| `train_mix_uav.sh` | Train one arm, one scene, N seeds sequentially | `$1=engine` `$2=scene` `$3=seeds` |
| `eval_mix_uav.sh` | Eval one arm, one scene, N seeds sequentially | `$1=engine` `$2=scene` `$3=seeds` `$4=n_trials` `$5=projection` `$6=record` `$7=K` |
| `eval_k_sweep.sh` | One eval job per K, in parallel — **the Gen15 experiment** | `$1=engine` `$2=scene` `$3=seeds` `$4=K list` `$5=n_trials` `$6=projection` |
| `aggregate_summaries.sh` | Roll up one arm's `results.json` → `SCENE_SUMMARY.json` + `ALL_SCENES_SUMMARY.json` | `$1=engine` `$2=scenes` `$3=projection` |
| `train_all_scenes.sh` | One `train_mix_uav` job **per scene**, in parallel | `$1=engine` `$2=scenes` `$3=seeds` |
| `eval_all_scenes.sh` | One `eval_mix_uav` job per scene in parallel, then auto-aggregate | `$1=engine` `$2=scenes` `$3=seeds` `$4=n_trials` `$5=projection` `$6=K` |
| `uav_mix_pipeline.sh` | Single scene: train → eval chained (`afterok`) | `$1=engine` `$2=scene` `$3=seed` `$4=n_trials` `$5=projection` `$6=record` `$7=K` |
| `uav_mix_all_pipeline.sh` | One arm, all scenes: train→eval per scene in parallel, then aggregate | `$1=engine` `$2=scenes` `$3=seeds` `$4=n_trials` `$5=projection` `$6=K` |

---

## Order of operations

**0. Gates — before anything else.** A wiring mistake here costs 30 seconds; the same mistake
found during a sweep costs a day.

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/gates_mix_uav.sh
```

**1. Smoke train, one arm, one scene, one seed** (confirms the arm learns at all):

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh mf corridor "6"
```

**2. Parity check on the `fm` arm** — Gen15-`fm` must reproduce Gen11. See the Gen15 changelog
for the G1 procedure (structural half is `gates_mix_uav.py --gates G1`, behavioural half is a
rollout comparison on the `diffuser` + `dpcc-c` variants).

**3. Full train, one arm at a time:**

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_all_scenes.sh fm "empty corridor s_curve pillars" "6 7 8 9 10"
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_all_scenes.sh mf "empty corridor s_curve pillars" "6 7 8 9 10"
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_all_scenes.sh af "empty corridor s_curve pillars" "6 7 8 9 10"
```

**4. The K sweep — the actual experiment.** Same K list for every arm:

```bash
for e in fm mf af; do
  bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh \
      $e corridor "6 7 8 9 10" "1 2 4 10 20"
done
```

**5. Aggregate per arm** (never pooled — three different objectives):

```bash
for e in fm mf af; do
  bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/aggregate_summaries.sh \
      $e "empty corridor s_curve pillars" dpcc-c
done
```

---

## Or, end to end for one arm

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_all_pipeline.sh \
    mf "empty corridor s_curve pillars" "6 7 8 9 10" 20 dpcc-c
```

---

## Things that will bite you

- **Matched budget or nothing.** When comparing arms, every arm gets the same `K`. A win at
  K=2 against another arm's K=20 is not a result.
- **`K` is real in Gen15 and was not in Gen11.** Gen11's `flow_steps_v3` reached neither the
  sampler nor the folder name — its evals sampled at the pickled training value (10) inside
  folders labelled `K20`. Gen15 pins K onto the loaded model *and* writes it into the path.
  Do not carry Gen11 K-labelled numbers into a Gen15 comparison without re-running them.
- **Arms are aggregated separately.** `aggregate_summaries.sh` takes an engine for a reason.
- **`logs/UAV_FM/` is Gen11's.** Nothing here writes there. If you see a Gen15 job touching it,
  something is wrong with `logbase` in `config/uav_mix.py`.
- **`--time` = 2× expected, 24 h cap.** The submitters already scale by seed count.
- Constraints come from the **shared, read-only** `config/uav_projection.yaml`. Editing it
  changes Gen11 and Gen15 alike — that is intended (comparability), but it means an edit
  mid-sweep silently splits your results. Each run snapshots the yaml it actually used.
