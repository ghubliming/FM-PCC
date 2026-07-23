# CHANGELOG — Gen12 fix_1: eval config decouple + direct-checkpoint loader

**Date:** 2026-07-23 · **Type:** fix · **Status:** code complete, **NOTHING RUN** (no deps in this container)
**Follows:** [`../init/CHANGELOG_Gen12_coding1.md`](../init/CHANGELOG_Gen12_coding1.md)
**Nothing committed.**

This fix has three parts: **§1–§5** decouple the eval from `projection_eval.yaml`;
**§6** replaces the templated loadpath with a direct `checkpoint_dir`, after we found the
init loadpath pointed at a checkpoint that does not exist / is the wrong model class;
**§7** renames the misleading `*_pipeline.sh` (Gen12 has no training) and pins down the
real-deploy entrypoint.

---

## 0. TL;DR

Gen12's evaluation must be driven **only** by its own `config/hardflow_projection_eval.yaml`,
never by the shared `config/projection_eval.yaml`. Coding pass 1 already had `eval` and
`load_results` reading the Gen12 file, but **`gates_hardflow.py` still opened
`projection_eval.yaml`** — a leak that meant the pre-flight gates could assert one geometry while
the eval enforced another. Fixed. The Gen12 eval path is now single-sourced.

The Gen12 yaml stays **self-contained and meaningful** (the design the user chose to keep): it
already holds every constraint key the gates need, so no yaml redesign was required — only the
consumer was repointed.

## 1. The one real bug

`FM_v3_hardflow_test/gates_hardflow.py::build_constraints` hard-coded
`open('config/projection_eval.yaml')`. If someone edited an obstacle radius or halfspace in the
Gen12 yaml (its own copy — see init §6.2), the gates would still build constraints from the shared
file. A gate could then pass on geometry the eval never runs, or fail on geometry the eval doesn't
use. Since the gates exist precisely to prove the seam (PLAN §3.2/§3.3), reading a different config
than the eval defeats their purpose.

## 2. Files changed

| file | change |
|---|---|
| `FM_v3_hardflow_test/gates_hardflow.py` | `build_constraints`, `gate_g2`, `gate_g3` now take a `config_path` (default `CONFIG_PATH = 'config/hardflow_projection_eval.yaml'`); `main()` gains `--config`; the run banner prints the config in use |

No other file needed touching:

- `eval_FM_v3_hardflow.py` — already `--config` default `hardflow_projection_eval.yaml` ✅
- `load_results_FM_v3_hardflow.py` — already `--config` default `hardflow_projection_eval.yaml` ✅
- `Slurm_Codes/sbatch/hardflow_fmv3/gates_hardflow_fmv3.sh` — unchanged; the new `--config`
  default already points at the Gen12 yaml, so the existing invocation is correct.

## 3. Verification (static — nothing executed)

- **No `projection_eval.yaml` read remains in the Gen12 code path.** The only surviving mentions
  are the `# never config/projection_eval.yaml` comment in `gates_hardflow.py` and the `--config`
  defaults, which all name the Gen12 file. (`grep` over `FM_v3_hardflow_test/*.py` +
  `flow_matcher_v3_hardflow/`.)
- **All three Gen12 scripts default to the same file** — gates, eval, load_results.
- **The Gen12 yaml has every key the gates read** — `halfspace_constraints`,
  `obstacle_constraints`, `bounds`, `observation_indices`, `action_indices` all present, so
  `build_constraints` runs against it with no missing-key error.
- `gates_hardflow.py` compiles.

## 4. Known, deliberate NON-change

`config/avoiding-d3il.py` still reads `projection_eval.yaml` at **import time** for
`diffusion_timestep_threshold`. This is **shared config infrastructure** loaded by every
generation's `Parser`, not the eval config, and the Gen12 plan block (`plan_fm_v3_hardflow`) does
**not** consume the threshold. Repointing it would change behaviour for every generation, which is
out of scope for a Gen12 fix. Left as-is intentionally; flagged here so it is not mistaken for a
missed leak.

## 5. Net effect

Gen12 eval is now single-sourced on `config/hardflow_projection_eval.yaml`: change a constraint
there and the gates, the eval and the aggregator all move together. The shared
`projection_eval.yaml` (5-seed eval default, restored by the user) drives Gen3/Gen3v2/Gen3v6/Gen3v7
as before and no longer has any influence on Gen12.

Still nothing run — the gates remain the first cluster step (PLAN §4).

---

## 6. Direct-checkpoint loader — the init loadpath was pointing at nothing

### 6.1 The bug this exposes

The init `plan_fm_v3_hardflow` block resolved its `diffusion_loadpath` to:

```
logs/avoiding-d3il/flow_matching_v3/H8_K10_Dmodels.diffusion.GaussianDiffusion/<seed>/
```

**That checkpoint does not exist on the cluster, and there is no sbatch that trains it.**
The actual trained baseline the user has is a *different model class in a different folder*:

```
logs/avoiding-d3il/flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/<seed>/
```

Two independent init mistakes, both from copy-modifying the wrong base (`flow_matcher_v3` /
`GaussianDiffusion`) when the real checkpoint is the Gen3v2 ode_selectable / `FlowMatchingODE`
model:

1. **Wrong folder + wrong class + wrong tag** (`flow_matching_v3/…GaussianDiffusion/K10` vs
   `flow_matching_v3_ode_selectable/…FlowMatchingODE/a1.5_b1.0_aw10`).
2. The eval routed model type by **class name** (`== 'GaussianDiffusion'`), so even if a
   `FlowMatchingODE` loaded, it fell to the states-only branch → `action_dim = 0` → the whole
   dof layout the sampler depends on collapses.

Symptom the user hit: the pipeline submitted fine (gates → eval → aggregate), but the gates need
no checkpoint (stub field), so a green gates job says nothing; the **eval** job is where it would
have died with `FileNotFoundError` on `dataset_config.pkl`.

### 6.2 The redesign — one direct path, any FMv3-family model

Replaced the per-model templated loadpath with a single **`checkpoint_dir`** knob in the eval YAML
(the user's ask: *"a EVAL BLOCK direct need a path … we can load what we want fmv3ode or imf or
whatever"*).

| file | change |
|---|---|
| `config/hardflow_projection_eval.yaml` | new `checkpoint_dir:` key (**default `null`**), documented with the ode_selectable example path |
| `eval_FM_v3_hardflow.py` | `checkpoint_dir` set → load `<checkpoint_dir>/<seed>/` with the pickle's **own** class (`target_class=None`, native load); `null` → fall back to the templated `plan_fm_v3_hardflow.diffusion_loadpath` |
| `eval_FM_v3_hardflow.py` | model-type routing now keys on `action_dim > 0`, not class name — so `FlowMatchingODE` / `MeanFlowODE` / … all route to `states_actions` correctly |
| `eval_FM_v3_hardflow.py` | `args.horizon` re-synced from the loaded model (a direct-path model may not be H8) |

**Why native load needs no new class in the Gen12 package:** `HardFlowPolicy` uses the model as a
pure black box — verified `FlowMatchingODE` exposes the *identical* interface the sampler calls
(`_predict_velocity(x, cond, t, returns=None)`, `__call__`, `p_sample_loop`, `action_dim`,
`transition_dim`, `goal_dim`). So the pickled `FlowMatchingODE` loads natively from its own
`flow_matcher_v3_ode_selectable` package (which is on `PYTHONPATH`), and the three arms run
unchanged. No `FlowMatchingODE` copy into `flow_matcher_v3_hardflow` was needed.

### 6.3 Bad-path behaviour (user request)

Default `checkpoint_dir: null`. Before loading each seed, the eval checks the resolved dir exists
**and** contains a `state_*.pt`. If not, it prints a boxed **`[ WARNING ]`** to stderr naming the
exact path it looked for (and whether `checkpoint_dir` was set or null), then **skips that seed**
rather than crashing the whole job — so a partially-trained sweep (e.g. seeds 6–9 present, 10
missing) still evaluates what exists.

### 6.4 What this does NOT fix (stated)

- **`checkpoint_dir` is still `null` by default** — the user must set it to their real path (the
  ode_selectable folder) before the eval loads anything. Left null intentionally; a wrong hard-coded
  default is exactly what caused §6.1.
- **`hardflow_new` assumes a standard single-time velocity `v = f(x, t)`.** `FlowMatchingODE` and
  `GaussianDiffusion` qualify. A two-time iMF/MeanFlow field (`u(z, τ, h)`) will *load* through this
  path, and arms A/B (which call the model's own `p_sample_loop`) will run, but **arm C's sampler
  math is only correct for single-time FM fields.** Loading an iMF checkpoint here is not
  automatically a valid arm-C experiment — flagged, not blocked.
- **`load_results` unchanged** — results are written under the plan block's `savepath`
  (`plans/flow_matching_v3_hardflow/…`), independent of where the checkpoint was loaded from, so
  aggregation already reads the right place.

### 6.5 Verification (static)

- `eval_FM_v3_hardflow.py` compiles.
- `checkpoint_dir: null` parses to Python `None`; the override branch is correctly skipped when
  `target_class=None` (native class load).
- `FlowMatchingODE._predict_velocity` signature is byte-identical to the `GaussianDiffusion` one the
  sampler was written against.

Still nothing run. Next cluster step is unchanged (gates first, PLAN §4), but the eval now needs
`checkpoint_dir` pointed at the real ode_selectable folder to load anything.

---

## 7. "pipeline" renamed — Gen12 has no training

### 7.1 The confusion

In this repo a `*_pipeline.sh` chains **TRAIN → EVAL** (see `AlphaFlow/alphaflow_pipeline.sh`,
`MeanFlow/meanflow_pipeline.sh`). Gen12 shipped a `hardflow_fmv3_pipeline.sh` — but **Gen12 trains
nothing** (PLAN §1: it reuses a pre-trained FMv3 checkpoint). The name implied a training stage
that does not, and should not, exist.

### 7.2 Confirmed: there is zero training surface in Gen12

- `FM_v3_hardflow_test/` has **no train script** — `train_FM_v3.py` was deleted at init
  (init changelog §2.1). Contents: `eval_FM_v3_hardflow.py`, `gates_hardflow.py`,
  `fit_dynamics_fmv3.py`, `load_results_FM_v3_hardflow.py`.
- `Slurm_Codes/sbatch/hardflow_fmv3/` has **no train sbatch**.
- The chain script itself contains **no train job** (grep `train` → only the "there is NO training
  job" comment).

So nothing had to be deactivated — the training surface was already absent. The only fix needed
was the misleading name.

### 7.3 Change

| before | after |
|---|---|
| `Slurm_Codes/sbatch/hardflow_fmv3/hardflow_fmv3_pipeline.sh` | `…/hardflow_fmv3_debug_chain.sh` (`git mv`) |
| `#SBATCH --job-name=hffm_pipeline` | `hffm_debug_chain` |
| header "Pipeline Master Script" | "DEBUG / BRING-UP CHAIN (NOT a train pipeline)" + why |
| log banners `PIPELINE START/END` | `DEBUG-CHAIN START/END` |

The chain's behaviour is unchanged: gates → eval → aggregate, `afterok` so a gate failure cancels
the rest. Only the name and the messaging changed. `bash -n` passes.

### 7.4 The real-deploy entrypoint (the user's question, pinned down)

- **Real deploy = the eval job ALONE.** Once the gates have passed once, submit:
  ```
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
  ```
  This is now stated in the chain's header and echoed at submit time.
- **`hardflow_fmv3_debug_chain.sh`** is the convenience wrapper for validation / bring-up (it just
  front-loads the gates and back-loads the aggregation around that same eval job).
- **`gates_hardflow_fmv3.sh`** — run once after any change to the sampler/layout; needs no checkpoint.
- **`load_results_hardflow_fmv3.sh`** — aggregation; run after eval.
- **`fit_dynamics_hardflow_fmv3.sh`** — only if the YAML is switched to `dynamics_mode: linear_fit`.

### 7.5 Stale references left as historical record

`init/CHANGELOG_Gen12_coding1.md` (§6, §14) still names `hardflow_fmv3_pipeline.sh`. Init
changelogs are historical records of what was built at init time and are **not** rewritten; this
fix_1 doc is the current-state authority. If you copy a submit command, take it from here (§7.4),
not from the init changelog.

Still nothing run.
