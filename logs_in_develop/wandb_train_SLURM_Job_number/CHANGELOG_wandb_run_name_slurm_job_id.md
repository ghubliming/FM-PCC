# Changelog — W&B run names carry the Slurm job ID

**Date:** 2026-07-21
**Branch:** `update_into_FM`
**Scope:** repo-wide (all active training entrypoints + all non-legacy sbatch scripts)
**Status:** code written locally, **not yet run** — needs a cluster run to confirm (see *Verification* below)

---

## 1. Goal

Make every W&B run traceable back to the Slurm job that produced it:

```
iMF-seed-6            ->  iMF-seed-6-slurm-4183726
HF-iMF-hf_imf-seed0   ->  HF-iMF-hf_imf-seed0-slurm-4183726
aligning-fm_v3-S6     ->  aligning-fm_v3-S6-slurm-4183726
```

## 2. Approach (two independent layers)

| Layer | What it does | Where |
|---|---|---|
| **A. Run name suffix** | appends `-slurm-<SLURM_JOB_ID>` to the W&B run *name* | Python, ~19 `wandb.init()` call sites |
| **B. Run tag** | sets `WANDB_TAGS="slurm-<id>"` so the ID is a filterable W&B tag | Bash, 19 sbatch scripts |

Layer A was needed because **every** training script passes `name=` explicitly to
`wandb.init()`, which makes W&B ignore the `WANDB_NAME` env var — so the sbatch side
alone cannot change the name. Layer B is free (no `tags=` kwarg exists anywhere in the
repo, so the env var is always honoured) and gives a clean filter dimension that does not
pollute the name string.

**Both layers are no-ops off-cluster** (`SLURM_JOB_ID` unset → suffix is `''`, tag never
exported), so local/Colab runs keep their old names exactly.

**Group names are deliberately left untouched** — appending the job ID to `group=` would
split every seed of an experiment into its own group and break the seed-aggregation plots.

## 3. Code pattern

Two shapes, depending on how each file builds its name.

**A1 — files with an f-string name inline:**
```python
# Tag the run with the Slurm job id (empty off-cluster)
slurm_suffix = f"-slurm-{os.environ['SLURM_JOB_ID']}" if os.environ.get('SLURM_JOB_ID') else ''
run = wandb.init(..., name=f'iMF-seed-{seed}{slurm_suffix}', ...)
```

**A2 — files with a precomputed `wandb_name` variable:**
```python
# Tag the run name with the Slurm job id (no-op off-cluster); group stays clean
if os.environ.get('SLURM_JOB_ID'):
    wandb_name = f"{wandb_name}-slurm-{os.environ['SLURM_JOB_ID']}"
```

**B — sbatch, inserted inside the existing `if [ -f .wandb_api_key ]` block:**
```bash
# Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
```
Written as a full `if/fi` rather than `[ ... ] && export ...` on purpose: several of these
scripts run under `set -e`, where a trailing false `&&` chain would abort the job.

## 4. Files touched — Python (19 files, all `wandb.init(...)` name= sites)

| File | Function / block | Pattern | Name before → after |
|---|---|---|---|
| `scripts/train.py` | `__main__` seed loop, wandb block | A2 | `<savepath>-S<seed>` → `…-slurm-<id>` |
| `FM_test/train_FM.py` | seed loop wandb block | A1 | `{dataset}-seed-{seed}` → `…-slurm-<id>` |
| `FM_v2_test/train_FM_v2.py` | idem | A1 | idem |
| `FM_v3_test/train_FM_v3.py` | idem | A1 | idem |
| `FM_Unet_v2_test/train_FM_Unet_v2.py` | idem | A1 | idem |
| `FM_hp_tune_test/train_FM_hp_tune.py` | idem | A1 | idem |
| `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` | `__main__`, `# Setup W&B` block | A1 | `iMF-seed-{seed}` → `iMF-seed-{seed}-slurm-<id>` |
| `FM_v3_imeanflow_test/train_flow_matching_v3_ode_selectable.py` | seed loop wandb block | A2 | savepath-derived |
| `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py` | idem | A2 | savepath-derived |
| `FM_v3_drifting_test/train_flow_matching_v3_drifting.py` | idem | A2 | savepath-derived |
| `FM_v3_uav_test/train_fm_uav.py` | idem | A2 | savepath-derived |
| `fm_visual_aligning_test/train_fm_visual_aligning.py` | seed loop wandb block | A2 | `{exp}-{exp_name}-S{seed}` → `…-slurm-<id>` |
| `fm_visual_avoiding_test/train_fm_visual_avoiding.py` | idem | A2 | idem |
| `imf_visual_aligning_test/train_imf_visual_aligning.py` | idem | A2 | idem |
| `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` | idem | A2 | idem |
| `diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py` | idem | A2 | idem |
| `d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py` | hydra `main()` wandb block | A1 | `{agent}_seed{seed}` → `…-slurm-<id>` |
| `HardFlow/run/train_fm.py` | `init_wandb()` | A1 | `HF-FM-{exp}-seed{seed}` → `…-slurm-<id>` |
| `HardFlow/run/train_imf.py` | `init_wandb()` | A1 | `HF-iMF-{exp}-seed{seed}` → `…-slurm-<id>` |

`os` was already imported in all 19 files — no new imports.

## 5. Files touched — sbatch (19 files, `WANDB_TAGS` export)

`Slurm_Codes/sbatch/`:
`train_dpcc_job.sh`, `eval_dpcc_job.sh`, `train_fmv3_ode_job.sh`, `eval_fmv3_ode_job.sh`,
`iMF/train_imf.sh`, `iMF/eval_imf.sh`, `iMF/load_results_imf.sh`,
`hardflow/_hardflow_common.sh` (sourced by **all** hardflow train/eval/pipeline scripts),
`Drifting/train_drifting.sh`, `Drifting/eval_drifting.sh`, `Drifting/load_results_drifting.sh`,
`uav_fm/train_fm_uav.sh`, `uav_fm/eval_fm_uav.sh`,
`fm_visual_aligning/train_fm_visual_aligning.sh`, `fm_visual_avoiding/train_fm_visual_avoiding.sh`,
`imf_visual_aligning/train_imf_visual_aligning.sh`,
`diffuser_visual_aligning/train_visual_aligning_dpcc.sh`,
`diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh`,
`d3il_visual_aligning_baseline/train_d3il_baseline.sh`.

Insertion point is the existing W&B login block (`if [ -f "$HOME/FMPCC/.wandb_api_key" ]`).
The d3il baseline script has no `WANDB_MODE` line, so the export was attached to its
`WANDB_API_KEY` line instead.

## 6. Deliberately NOT touched

- `Slurm_Codes/sbatch/(legacy)Visual_Aligning/*` and
  `fm_visual_avoiding_test (legacy_based_on_visual_aligning)/` — legacy/dead code.
  (The legacy avoiding trainer *was* edited by accident and reverted with `git checkout`;
  the **active** `fm_visual_avoiding_test/` copy is patched.)
- `Archived_Codes/**`, `*/datasets/minari-dataset-generation/scripts/antmaze/train_ant.py`
  (vendored upstream), `d3il/.../d3il_sim/core/logger.py` (vendored).
- All `group=` arguments — see §2.
- `MASTER_TEST_HISTORY.md` — not updated (per convention, only on request).

## 7. Risk assessment

| Risk | Verdict |
|---|---|
| Breaks W&B run resumption | **No** — no `resume=` / `WANDB_RUN_ID` in any active training script. |
| Breaks array-job naming | **No** — no `--array` jobs anywhere in `Slurm_Codes/sbatch`. |
| Breaks downstream analysis | **No** — nothing in `Data_Analysis/`, `Slurm_Codes/`, `ipynbs_Colab/`, `Results_and_Data_Analysis_Colab_T4/` parses W&B run names (no `wandb.Api()` usage). |
| Overwrites existing `tags=` | **No** — no `wandb.init(..., tags=...)` call exists in active code. |
| Aborts a `set -e` script | **No** — export written as a full `if/fi`; see §3. |
| Multi-seed loop in one job | All seeds share one job ID, but names still differ by seed → still unique. |
| Pipeline jobs (train+eval in one job) | Train and eval runs share the job ID — intended. |

## 8. Verification done here

- `python3 -m py_compile` on all 19 Python files → **pass** (syntax only; no deps in this container).
- `bash -n` on all 19 sbatch scripts → **pass**.
- **Not run on cluster.** Next real Slurm job should show `…-slurm-<jobid>` as the W&B run
  name and a `slurm-<jobid>` tag on the run page. If the name looks right but the tag is
  missing, check that the job actually found `$HOME/FMPCC/.wandb_api_key`.
