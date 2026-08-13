# CHANGELOG — `custom_msg` results-path token (eval-only)

**Date:** 2026-08-13
**Status:** ✅ IMPLEMENTED (3 config modules) — **not committed** (user commits manually)
**Plan doc:** [`CHECKLIST_custom_msg_path_token.md`](CHECKLIST_custom_msg_path_token.md)
**Motivation:** run the final "pick the best horse" comparison at `n_trials: 20` (up from 2) over
5 seeds **without overwriting the existing 2-trial results**.

---

## 1. What changed

An **eval-only, opt-in message token** is appended to the results-folder name of every `plan_*`
(evaluation) block. Empty by default ⇒ **every path is byte-identical to before**.

```
msg OFF (default)
  logs/avoiding-d3il/plans/flow_matching_v3_meanflow/H8_D..._dp0.5/H8_K2_Meuler_T1_A1_B1_D.../6/

msg ON  (FMPCC_RUN_MSG=20trials)
  logs/avoiding-d3il/plans/flow_matching_v3_meanflow/H8_D..._dp0.5/H8_K2_Meuler_T1_A1_B1_D..._msg20trials/6/
                                                                                            ^^^^^^^^^^^^^
```

### Three ways to set it

| how | scope | notes |
|---|---|---|
| `FMPCC_RUN_MSG=20trials ./Slurm_Codes/submit.sh <sbatch>` | the whole job | **primary.** No git edit — `submit.sh:42` already passes `--export=ALL` |
| edit `custom_msg = ...` in the config module | every arm of that dataset | git-tracked |
| `'custom_msg': 'ode_only_rerun',` inside one plan block | one arm | for tagging a single arm differently |

### Mechanism (identical in all three modules)

```python
def _sanitize_msg(text): ...          # keep [A-Za-z0-9._-], collapse rest to '-', cap 40 chars
custom_msg = _sanitize_msg(os.environ.get('FMPCC_RUN_MSG', ''))
def _msg_suffix(args):                # '_msg<token>' or ''
    msg = _sanitize_msg(getattr(args, 'custom_msg', ''))
    return f'_msg{msg}' if msg else ''
def watch_plan(lst):                  # watch() + the suffix — PLAN BLOCKS ONLY
    _fn = watch(lst)
    return lambda args: _fn(args) + _msg_suffix(args)
```

Then per plan block: `watch(...)` → `watch_plan(...)`, plus one new `'custom_msg': custom_msg,` line.

---

## 2. Files changed

| file | plan blocks tagged | train blocks touched |
|---|---|---|
| `config/avoiding-d3il.py` | **12** | 0 |
| `config/aligning-d3il-visual.py` | **10** (4 explicit + 2 derived + 4 mix arms) | 0 |
| `config/avoiding-d3il-visual.py` | **2** | 0 |

`config/avoiding-d3il.py` blocks: `plan` ⭐, `plan_fm`, `plan_fm_unet_v2`, `plan_fm_v2`,
`plan_fm_v3`, `plan_fm_v3_hardflow`, `plan_fm_v3_ode_selectable` ⭐, `plan_fm_v3_drifting`,
`plan_fm_v3_imeanflow`, `plan_fm_v3_meanflow` ⭐, `plan_fm_v3_alphaflow` ⭐, `plan_fm_hp_tune`.
⭐ = the four arms of this campaign (DPCC K20 / FMv3ODE / mf_unet / af_sit).

`config/aligning-d3il-visual.py` note: only **5** call sites were edited — the 4 explicit plan
blocks plus `_mix_plan_block`'s `blk['exp_name']`. The derived blocks
(`plan_ddpm_encdec_vision_nonvisual`, `plan_imf_visual_aligning`) and the 4 mix arms inherit
`custom_msg` through `_mix_plan_common` / their copy-from-parent construction — verified below.

`config/avoiding-d3il-visual.py` also gained `import os` (it had none).

**Not changed:** any `config/*.yaml`, `diffuser/utils/setup.py`, any `eval_*.py`, any
`load_results_*.py`, `Data_Analysis/**`, `Slurm_Codes/**`.

---

## 3. Why an eval ARG and not a projection-YAML key

Rev 1 of the plan put `custom_msg` in the projection-eval YAMLs. Changed on user direction — the
message describes *this eval run*, not the constraint projection, so it belongs next to
`diffusion_epoch` / `suffix` in the plan block. Three concrete wins:

1. **Kills a real footgun.** `scripts/eval.py` and `eval_flow_matching_v3_ode_selectable.py` never
   publish `FMPCC_PROJ_CFG`, so the config module falls back to `config/projection_eval.yaml`.
   That is the file those two happen to load — correct *by coincidence*, not construction. A
   YAML-sourced message could have been read from a file the eval never opened, which is exactly
   the class of bug `FIX_9_CFG_PROVENANCE` was written to close.
2. **`config/*.yaml` stays a pure constraint-projection surface** (repo convention).
3. **Free provenance.** As a config key it lands in `self._dict` → `args.json` and the config
   snapshot, so the folder tag is provable from the run's own artifacts.

---

## 4. Why a wrapper instead of editing the `args_to_watch_*` lists 🔴

Adding `('custom_msg', 'msg')` to a watch list looks simpler but would have moved **checkpoint**
folders and broken every `diffusion_loadpath`, because several lists are shared between train and
plan blocks:

- `args_to_watch_v3` — train `flow_matching_v3` **and** plan `plan_fm_v3` / `plan_fm_v3_hardflow`
- `args_to_watch` — train `flow_matching` / `flow_matching_unet_v2` / `flow_matching_v2` **and**
  their `plan_*` counterparts

`watch_plan` is a separate callable, so the lists are untouched. Second line of defence:
`watch_plan` reads `args.custom_msg`, and **no train block defines that key** — a misapplied
wrapper still renders the empty token. Both properties are asserted by the test in §5.

Also note `watch()` skips absent keys but would render a present-but-empty value as a bare label
(`..._msg`), which is why the suffix is applied conditionally outside `watch()` rather than as a
list entry.

---

## 5. Verification — done here, offline ✅

`Data_Analysis`-free harness at
`scratchpad/check_msg_all.py` (session-scratch, not committed): stubs `yaml` and
`diffuser.utils` (injecting the real dependency-free `watch()`), imports **both**
`git show HEAD:<file>` and the patched file, renders every block's `exp_name`, and asserts:

| # | assertion | result |
|---|---|---|
| 1 | msg **unset** → every block byte-identical to HEAD | ✅ 47/47 blocks |
| 2 | msg **set** → TRAIN blocks still byte-identical to HEAD | ✅ 23/23 train blocks |
| 3 | msg **set** → PLAN blocks gain exactly `_msg20trials` as a suffix | ✅ 24/24 plan blocks |
| 4 | sanitizer: `''`/`'  '`/`None`→`''`, `'20 trials'`→`'20-trials'`, `'a/b'`→`'a-b'`, `'final run #2'`→`'final-run-2'`, `'--x--'`→`'x'`, 60 chars→40 | ✅ |

```
PASS  config/avoiding-d3il.py           (12 plan / 11 train blocks)
PASS  config/avoiding-d3il-visual.py    ( 2 plan /  2 train blocks)
PASS  config/aligning-d3il-visual.py    (10 plan / 10 train blocks)
ALL PASS — msg unset is a byte-identical no-op; msg set tags PLAN blocks only.
```

Assertion 1 is the no-regression gate: **no existing results folder is renamed, moved or
overwritten by this patch.**

### Still to confirm on the cluster (nothing here can run the real stack)
- [ ] One real eval with msg **unset** → `[ utils/setup ] Made savepath:` matches an existing dir.
- [ ] One real eval with `FMPCC_RUN_MSG=20trials` → `_msg20trials` leaf, **and** the model-load
      line still points at the unchanged checkpoint dir.
- [ ] One **train** job launched with `FMPCC_RUN_MSG` exported → checkpoint folder name unchanged.
- [ ] `custom_msg` present in the eval's `args.json` / config snapshot.

---

## 6. Deliberately NOT patched: `config/uav.py`, `config/uav_mix.py` ⚠️

Planned in the checklist (§2.4), then **dropped after reading the UAV evals** — the plan block's
`exp_name` is **dead for pathing** in the whole UAV family. Both
`FM_v3_uav_test/eval_fm_uav.py:1276` and `mix_uav_test/eval_mix_uav.py:1341` build the output
directory by hand:

```python
_model_dir = os.path.relpath(os.path.dirname(parsed.savepath), scene_root)   # from the TRAIN block
seed_dir   = os.path.join(scene_root, 'plans', _model_dir, eval_params_dir, _seed_str)
```

`parsed` comes from `build_experiment()`, which parses the **training** experiment; the
eval-parameter level is `_uav_eval_tag(config, controller)`, built from
`config/uav_projection.yaml` — **not** from any `plan_*` `exp_name`. Patching those two plan
blocks would have produced a knob that silently does nothing, which is worse than no knob.

**Correct UAV hook, if wanted later:** add `custom_msg` to `config/uav_projection.yaml` and append
it inside `_uav_eval_tag()` in both eval scripts (2 files, ~3 lines each). That is the YAML route
— appropriate there precisely because UAV's eval-tag is already yaml-driven. Not done: it is a
different generation from this campaign and was not requested.

---

## 7. Known gap that still blocks the stated campaign 🔴

**FMv3ODE has no K knob.** `plan_fm_v3_ode_selectable` hard-codes `'flow_steps_v3': 10`,
`eval_flow_matching_v3_ode_selectable.py` has no `--flow-steps`, and `Parser.add_extras` (the
generic `--key value` override) is **commented out** at `diffuser/utils/setup.py:76`. So
"FMv3ODE K ∈ {1,2,5,10,20}" is currently only reachable by hand-editing the config between
submits. MeanFlow/AlphaFlow already have `MF_FLOW_STEPS` / `AF_FLOW_STEPS`.

**Not implemented — not requested.** The one-line fix mirrors this patch's own idiom:
```python
'flow_steps_v3': int(os.environ.get('FMV3_FLOW_STEPS', 10)),   # in plan_fm_v3_ode_selectable
```
plus a `for K in $FMV3_FLOW_STEPS` loop in `Slurm_Codes/sbatch/eval_fmv3_ode_job.sh`. K already
renders into the path as `_K{flow_steps_v3}_`, so each K lands in its own folder.

Other campaign-time risks unchanged from the checklist: **runtime ~10×** (both eval sbatch scripts
are already at the 24 h cap — trim `projection_variants` first) and **npz size ~10×** (per-trial
`obs_all` / `act_all` / `sampled_trajectories_all`; commit `567af3d7` already had to harden DA
auto-scan against OOM).

---

## 8. Campaign run-book

```bash
export FMPCC_RUN_MSG=20trials     # SAME value for every arm, or DA cannot group them
```

- [ ] `n_trials: 20` in `config/projection_eval.yaml`, `meanflow_projection_eval.yaml`,
      `alphaflow_projection_eval.yaml` (+ `hardflow_` if used).
- [ ] Trim `projection_variants` to the kept set (drops `dpcc-c0p5` etc.) — buys back the runtime.
- [ ] DPCC baseline K20/aw10 — `Slurm_Codes/sbatch/eval_dpcc_job.sh`
- [ ] FMv3ODE K sweep — `Slurm_Codes/sbatch/eval_fmv3_ode_job.sh` ⚠️ blocked by §7
- [ ] mf_unet K sweep — `Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh` (`MF_FLOW_STEPS="1 2 5 10 20"`)
- [ ] af_sit K sweep — `Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh` (`AF_FLOW_STEPS=...`)
- [ ] ⚠️ **`load_results_*` jobs need the same `FMPCC_RUN_MSG`** — they rebuild the path through the
      same Parser (`scripts/load_results.py:38`), so without it they silently read the OLD
      2-trial folder.
- [ ] ⚠️ Do **not** put the export in `~/.bashrc` — it would tag every unrelated eval too.
- [ ] **DA needs no change.** `run_da_batch_v3.sh --parent-path logs/avoiding-d3il/plans` +
      `max_depth=10` (`main_da_batch.py:182`) finds the new folders; the `_msg20trials` suffix
      makes them distinguishable from the 2-trial siblings in every table; the aggregator is
      trial-count agnostic (`data_loader.py:163`, `float(np.mean(value))`).

---

## 9. Rollback

`git checkout -- config/avoiding-d3il.py config/aligning-d3il-visual.py config/avoiding-d3il-visual.py`.
Nothing else references the new symbols. Leaving the patch in place with `FMPCC_RUN_MSG` unset is
already a no-op (§5, assertion 1).
