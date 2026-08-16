# Checklist — `custom_msg` path token for the "20-trials, pick-the-best-horse" DA campaign

**Date:** 2026-08-13
**Status:** ✅ IMPLEMENTED 2026-08-13 (not committed) — see
[`CHANGELOG_custom_msg_path_token.md`](CHANGELOG_custom_msg_path_token.md) for what actually
shipped. Deviations from this plan: `config/uav.py` / `config/uav_mix.py` were **dropped**
(their plan-block `exp_name` is dead for pathing — CHANGELOG §6), and §5.1 (FMv3ODE K knob)
was **not** implemented (not requested — CHANGELOG §7).
**Goal:** run the final head-to-head at `n_trials: 20` (instead of 2) across 5 seeds **without
overwriting any existing 2-trial results**, by adding an optional user-settable message token to
the eval results path.

**Rev 2 (per user):** `custom_msg` is an **eval concern, not a projection concern** → it lives as a
**new arg line inside the `plan_*` blocks of `config/*.py`**, not in the projection-eval YAMLs.
**No YAML file is touched by this patch.**

---

## 0. Verdict: is it possible?

**Yes — and it is a small, low-risk patch.** The results directory is fully derived from
`config/<dataset>.py`:

```
diffuser/utils/setup.py:176
    args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name, str(args.seed))
```

`exp_name` for every *plan* (= eval) block is a callable built by `watch(...)`
(`diffuser/utils/setup.py:21`), and `watch` reads **`args`** — so a new key in the plan block is
immediately available to it. Appending one token is enough:

```
BEFORE  logs/avoiding-d3il/plans/flow_matching_v3_meanflow/H8_D..._dp0.5/H8_K2_Meuler_T1_A1_B1_D.../6/
AFTER   logs/avoiding-d3il/plans/flow_matching_v3_meanflow/H8_D..._dp0.5/H8_K2_Meuler_T1_A1_B1_D..._msg20trials/6/
```

Everything downstream follows for free:
- `results/halfspace_*/<variant>.npz` (`FM_v3_test/eval_FM_v3.py:251`) is under `savepath`.
- the `all_seeds` aggregate (`.../eval_*.py:262`, `os.path.dirname(args.savepath)`) is under `exp_name`.
- `config_snapshot_*` audit stamps (`setup.py:194`) land in the new folder → DA's "Last Run" column works.
- `load_results_*.py` rebuild the path through the **same Parser** → they find the new folder automatically
  (as long as the same msg is in effect for that job — see §4 ⚠️).
- **bonus:** because it is a real config key it is captured in `self._dict` → written to `args.json`
  and into the config snapshot, so the folder tag is *provable* from the run's own artifacts.

---

## 1. Design decisions (and why)

### 1.1 Token suffix on the leaf folder — NOT a new directory level
Considered inserting `.../plans/<model>/<msg>/H8_.../H8_K2_.../seed`. Rejected:

| | leaf-suffix `_msg20trials` | extra directory level |
|---|---|---|
| tree depth | **unchanged** | +1 |
| DA candidate `name` (= leaf dir, `multi_candidate_discovery.py:265`) | **carries the msg** → old vs new visibly distinct in every table/plot | identical for old & new → two rows called `H8_K2_Meuler_T1_A1_B1_D...`, indistinguishable |
| `discover_candidates_recursive` | fine | fine at `max_depth=10` (`main_da_batch.py:182`), **but breaks** the module default `max_depth=3` and any caller using it |

→ **leaf suffix wins.**

### 1.2 It is an eval arg, defaulted from the environment
One module-level line supplies the default, and each plan block carries **one new arg line**:

```python
custom_msg = _sanitize_msg(os.environ.get('FMPCC_RUN_MSG', ''))   # module level, next to `logbase`
...
'custom_msg': custom_msg,                                          # inside each plan_* block
```

Three ways to drive it, in order of convenience:
- **env at submit time** — `FMPCC_RUN_MSG=20trials ./Slurm_Codes/submit.sh <script>`.
  `Slurm_Codes/submit.sh:42` already uses `--export=ALL`, so it propagates into the job.
  **No git edit** — which is what the user asked for ("I will set in remote").
- **edit the module-level default** — one line, applies to every arm at once, git-tracked.
- **per-block literal** — `'custom_msg': 'ode_only_rerun',` to tag a single arm differently.

### 1.3 Why not the projection YAML (rev-1 design, dropped)
`custom_msg` describes *this eval run*, not the constraint projection — it belongs with
`diffusion_epoch` / `max_episode_length` / `suffix`, in the plan block. Dropping the YAML route
also **kills a real footgun**: `scripts/eval.py` and `eval_flow_matching_v3_ode_selectable.py` never
publish `FMPCC_PROJ_CFG`, so the config module falls back to its default
`config/projection_eval.yaml`. That happens to be the file those two load — correct *by
coincidence*, not by construction. With the arg-line design that whole class of mismatch is gone,
and `config/*.yaml` stays a pure constraint-projection surface (repo convention).

### 1.4 Default is empty → byte-identical old paths
Unset ⇒ token is `''` ⇒ `exp_name` is character-for-character what it is today.
**No existing folder is renamed, moved, or overwritten.** Old 2-trial data stays exactly where it is.

### 1.5 PLAN blocks only — never TRAIN blocks 🔴
If the token reached a *training* `exp_name`, checkpoint folders would move and every
`diffusion_loadpath` would break. Two concrete traps:
- `args_to_watch_v3` is shared by the **train** block `flow_matching_v3` (`config/avoiding-d3il.py:416`)
  **and** the plan blocks `plan_fm_v3` (:1047) / `plan_fm_v3_hardflow` (:1105).
- `args_to_watch` is shared by `flow_matching` (:254) / `flow_matching_unet_v2` (:303) /
  `flow_matching_v2` (:359) train blocks and their `plan_*` counterparts.

→ **Do not edit the `args_to_watch_*` lists.** Add a *wrapper* `watch_plan()` and use it only in
plan blocks. The lists stay untouched. Double safety: `watch_plan` reads `args.custom_msg`, and no
train block defines that key — a train block would render the empty token even if misused.

---

## 2. The patch (exact edits)

### 2.1 `config/avoiding-d3il.py` — sanitizer + default + wrapper (next to `logbase = 'logs'`, ~line 167)

```python
# ── CUSTOM RUN MESSAGE (2026-08-13) — opt-in results-path tag, EVAL ONLY ──────────────
# Lets a re-run of the SAME config at a different budget (n_trials 2 -> 20) write to its own
# folder instead of overwriting the old numbers. Empty  =>  paths are byte-identical to before,
# so nothing existing is touched.
#   env FMPCC_RUN_MSG=20trials ./Slurm_Codes/submit.sh ...   (submit.sh uses --export=ALL)
#   or edit the default below, or set 'custom_msg' inside one plan block to tag a single arm.
# 🔴 PLAN BLOCKS ONLY (see watch_plan). It must never reach a training exp_name: that would move
#    every checkpoint folder and break every diffusion_loadpath.
def _sanitize_msg(text):
    """Filesystem-safe, stable token: keep [A-Za-z0-9._-], collapse the rest to '-'."""
    raw = str(text if text is not None else '').strip()
    if not raw:
        return ''
    out = ''.join(ch if (ch.isalnum() or ch in '._-') else '-' for ch in raw)
    while '--' in out:
        out = out.replace('--', '-')
    return out.strip('-._')[:40]


custom_msg = _sanitize_msg(os.environ.get('FMPCC_RUN_MSG', ''))
if custom_msg:
    print(f'[ config/avoiding-d3il ] custom_msg="{custom_msg}" -> results dirs end in "_msg{custom_msg}"')


def watch_plan(args_to_watch_list):
    """watch(), plus the plan block's custom_msg suffix. Use in PLAN blocks only."""
    _fn = watch(args_to_watch_list)

    def _suffix(args):
        msg = _sanitize_msg(getattr(args, 'custom_msg', ''))
        return f'_msg{msg}' if msg else ''

    return lambda args: _fn(args) + _suffix(args)
```

`_sanitize_msg` is applied again inside `_suffix` so a hand-typed per-block literal
(`'custom_msg': 'final run #2'`) is cleaned too, not just the env value.

### 2.2 `config/avoiding-d3il.py` — per plan block: **1 swap + 1 new arg line**

Swap `watch(` → `watch_plan(` on the `exp_name` line, and add `'custom_msg': custom_msg,` next to
the block's other serialization keys (`prefix` / `exp_name` / `suffix`):

| line | block | `exp_name` today |
|---|---|---|
| 906 | `plan` (DPCC baseline) ⭐ | `lambda args: f"plans/diffusion/..." + watch([...])(args)` → also append the same `_msg` suffix |
| 946 | `plan_fm` | `watch(args_to_watch)` |
| 979 | `plan_fm_unet_v2` | `watch(args_to_watch)` |
| 1012 | `plan_fm_v2` | `watch(args_to_watch)` |
| 1047 | `plan_fm_v3` | `watch(args_to_watch_v3)` |
| 1105 | `plan_fm_v3_hardflow` | `watch(args_to_watch_v3)` |
| 1141 | `plan_fm_v3_ode_selectable` ⭐ | `watch(args_to_watch_fmv3_ode_plan)` |
| 1191 | `plan_fm_v3_drifting` | `watch(args_to_watch_fmv3_ode_plan)` |
| 1237 | `plan_fm_v3_imeanflow` | `watch(args_to_watch_fmv3_ode_plan)` |
| 1318 | `plan_fm_v3_meanflow` ⭐ | `watch(args_to_watch_fmv3_hf_plan)` |
| 1415 | `plan_fm_v3_alphaflow` ⭐ | `watch(args_to_watch_fmv3_hf_plan)` |
| 1566 | `plan_fm_hp_tune` | `watch(args_to_watch)` |

⭐ = the four arms named for this campaign (DPCC K20, FMv3ODE, mf_unet, af_sit).
The other eight are patched for consistency so no live generation is left half-supported.

Example (`plan_fm_v3_meanflow`, :1315-1318):
```python
        'prefix': 'f:plans/flow_matching_v3_meanflow/' +
                  'H{horizon}_D{diffusion}_aw{action_weight}_obj{mf_objective}_bb{imf_backbone}_ts{t_schedule}_dp{meanflow_data_proportion}/',
        'exp_name': watch_plan(args_to_watch_fmv3_hf_plan),   # was: watch(...)
        'custom_msg': custom_msg,   # NEW — '' => path unchanged; else '..._msg<value>'
```

**Train blocks at lines 206, 254, 303, 359, 416, 472, 526, 657, 762, 890, 1539 — DO NOT TOUCH.**
They get neither the swap nor the arg line.

### 2.3 YAML files — **no change**
`config/projection_eval.yaml`, `meanflow_projection_eval.yaml`, `alphaflow_projection_eval.yaml`,
`hardflow_projection_eval.yaml` are untouched by this patch. (`n_trials: 2 → 20` is a separate,
pre-existing knob you change at campaign time — see §4.)

### 2.4 (same treatment for the other datasets — do only if wanted)
`config/aligning-d3il-visual.py`, `config/avoiding-d3il-visual.py`, `config/uav.py`,
`config/uav_mix.py` all follow the identical `watch(...)`-in-a-plan-block pattern, so the same
two-step patch applies verbatim.
⚠️ Visual-aligning has **no `n_trials`** — it uses `n_contexts` (`config/visual_aligning_eval.yaml:19`),
so "20 trials" there means bumping `n_contexts`. Out of scope unless requested.

---

## 3. Verification (cluster — no Python in this container)

- [ ] **Dry-run path print, msg OFF.** Run any eval with `--aggregate-only` (or just read the
      `[ utils/setup ] Made savepath:` line) and confirm the path is **identical** to a folder that
      already exists on disk. This is the no-regression gate.
- [ ] **Dry-run path print, msg ON** (`FMPCC_RUN_MSG=20trials`): confirm `_msg20trials` on the leaf,
      and that `diffusion_loadpath` / the loaded checkpoint dir are **unchanged**
      (`[ utils/setup ]` / model-load log lines).
- [ ] **Train job smoke check:** launch (or dry-run) one train job **with `FMPCC_RUN_MSG` exported**
      and confirm the checkpoint folder name is unchanged (this is the trap in §1.5).
- [ ] Confirm `custom_msg` shows up in the eval's `args.json` / config snapshot.
- [ ] `ls` the old 2-trial folders after the first new run — untouched, no new files inside.

---

## 4. Campaign run-book (after the patch lands)

Set the SAME msg for every arm, or DA cannot group them.

```bash
# one export, reused by every submit of this campaign
export FMPCC_RUN_MSG=20trials
```

- [ ] Set `n_trials: 20` in `config/projection_eval.yaml`, `meanflow_projection_eval.yaml`,
      `alphaflow_projection_eval.yaml` (+ `hardflow_` if used).
- [ ] Trim `projection_variants` to the ones being kept (user plans to drop `dpcc-c0p5` etc.) —
      this is what buys back the runtime lost to 10× trials.
- [ ] **DPCC baseline K20 / aw10 / GaussianDiffusion** — `Slurm_Codes/sbatch/eval_dpcc_job.sh`
      (this is the pinned paper Target; K other than 20 is only a conservative extra check).
- [ ] **FMv3ODE K ∈ {1,2,5,10,20}** — `Slurm_Codes/sbatch/eval_fmv3_ode_job.sh`
      ⚠️ **see §5.1 — there is currently no K knob for this arm.**
- [ ] **mf_unet K sweep** — `Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh`
      (`MF_FLOW_STEPS="1 2 5 10 20"`, or `HFFM_FLOW_STEPS=<K>` for a single K)
- [ ] **af_sit K sweep** — `Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh`
      (`AF_FLOW_STEPS`, same convention)
- [ ] ⚠️ **`load_results_*` jobs need the same `FMPCC_RUN_MSG`** — they rebuild the path through the
      same Parser (`scripts/load_results.py:38`, `FM_v3_ode_selectable_test/load_results_...py:38`).
      Without it they read the OLD folder and silently report the 2-trial numbers.
- [ ] **DA:** `Slurm_Codes/sbatch/DA/run_da_batch_v3.sh` needs **no change** —
      `--parent-path logs/avoiding-d3il/plans` + `max_depth=10` picks the new folders up, and the
      `_msg20trials` suffix makes them distinguishable in every table.
      The aggregator is trial-count agnostic (`data_loader.py:163`, `float(np.mean(value))`).

---

## 5. Risks / gaps found while checking

### 5.1 🔴 FMv3ODE has no K override — blocks "K1,2,5,10,20" as stated
`plan_fm_v3_ode_selectable` hard-codes `'flow_steps_v3': 10` (`config/avoiding-d3il.py:1150`),
`eval_flow_matching_v3_ode_selectable.py` has no `--flow-steps`, and `Parser.add_extras` (the
generic `--key value` override) is **commented out** at `diffuser/utils/setup.py:76`. So today the
FMv3ODE K sweep is only doable by hand-editing the config between submits.
**Fix (small, mirrors MeanFlow/AlphaFlow, and is the same "arg line reads env" idiom as this patch):**
`'flow_steps_v3': int(os.environ.get('FMV3_FLOW_STEPS', 10)),` plus a `for K in ...` loop in
`eval_fmv3_ode_job.sh`. K already appears in the path as `_K{flow_steps_v3}_`, so each K gets its
own folder. **Say the word and I fold this into the same patch.**

### 5.2 ⏱️ Runtime is ~10× — sbatch `--time` must go up
`eval_dpcc_job.sh:8` and `eval_fmv3_ode_job.sh:8` are already at the 24 h cap. 20 trials × 3
halfspace variants × 5 seeds × N projection variants will not fit unless `projection_variants` is
cut hard. Plan: trim the variant list first, then re-estimate from a single-seed run.

### 5.3 💾 npz size / DA memory is ~10×
`eval_*.py:254` stores `obs_all`, `act_all`, `sampled_trajectories_all` for **every trial** in the
npz. 20 trials ⇒ ~10× file size. Commit `567af3d7` already had to harden the DA auto-scan against
OOM; re-check `main_da_batch.py` memory on the first 20-trial folder before scanning all arms.

### 5.4 Token length / charset
Sanitizer caps at 40 chars and strips anything outside `[A-Za-z0-9._-]`. Keep msgs short and
boring (`20trials`, `final_v1`) — the parent paths are already long and some filesystems cap a
single component at 255 bytes.

### 5.5 One env var, one campaign
`FMPCC_RUN_MSG` is global to the job. Exporting it in your cluster shell profile would tag *every*
subsequent eval, including unrelated ones. Export it per-submit (or per-campaign shell), not in
`~/.bashrc`.

---

## 6. Files this patch touches

```
config/avoiding-d3il.py     sanitizer + `custom_msg` default + watch_plan
                            + 12 plan blocks × (exp_name swap, new 'custom_msg' arg line)
[optional §2.4]  config/aligning-d3il-visual.py, avoiding-d3il-visual.py, uav.py, uav_mix.py
[optional §5.1]  config/avoiding-d3il.py (flow_steps_v3 env) + Slurm_Codes/sbatch/eval_fmv3_ode_job.sh
```

No change needed in: **any `config/*.yaml`**, `diffuser/utils/setup.py`, any `eval_*.py`, any
`load_results_*.py`, `Data_Analysis/**`, `Slurm_Codes/submit.sh`.

---

## 7. Open questions for the user

1. **Token spelling** — `_msg20trials` (proposed) or something shorter like `_20t`? It becomes part
   of every folder name for this campaign.
2. **§5.1 FMv3ODE K override** — fold into this patch, or handle separately?
3. **§2.4 other datasets** (visual-aligning / visual-avoiding / UAV) — patch now, or avoiding-d3il only?
