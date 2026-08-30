# Gen14 U11 — the projection threshold as an overridable, sweepable, *recorded* knob

**Date** 2026-08-29 · **Scope** eval only, `aligning-d3il-visual` · **Arm** any (used here for `mf`)
**Status** code + sbatch only. **Nothing has been run** (no torch in the AI container). Wall-clock
figures below are extrapolations from measured `avg_time_ms`, and are labelled as such.

---

## Why

`diffusion_timestep_threshold` (`T`) sets the fraction of the LATE ODE over which the DPCC projector
runs: the sampler projects from `int((1 - T) * K)` to the end, so solves/replan ≈ `T*K`.

| T | K | solves / replan |
|---|---|---|
| 0.5 (shipped) | 100 | **50** |
| **0.1** | 100 | **10** |
| **0.05** | 100 | **5** |
| 0.5 | 2 | 1 |

At the shipped `T = 0.5`, **every K ≥ 50 projected cell in this tree has hit the 24 h Slurm wall**:
`mf` K100 `dpcc-r` measured 14 988 ms/step, needs **50 h** for 30 rollouts, and died at 11/30.
Lowering `T` is the cheapest way to make a K = 100 projected cell finishable, and it is a real
question in its own right — *does late-only projection buy the same constraint satisfaction?*

Before U11 the only way to change it was editing `config/visual_aligning_eval.yaml`.

## 🔴 The problem with the YAML edit

```python
# config/aligning-d3il-visual.py:7-13
with open('config/visual_aligning_eval.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)
_yaml_threshold = _proj_config['diffusion_timestep_threshold']
```

Read **once**, at config import, and fed to **every** block in the file — Gen6v4, Gen7 and all four
Gen14 arms (lines 570/621/678/763). Editing it to run one job silently re-points every eval that
imports the config afterwards, and it is exactly the kind of change that gets left behind in the
working tree.

## ✅ Why there is no collision risk either way

`('diffusion_timestep_threshold', 'T')` is **already** in `args_to_watch_mix_visual_plan`, so the
results directory carries it:

```
T = 0.5   ->  plans/mix_visual_aligning_mf/.../H8_K100_Meuler_T0.5_..._Emf/6/     (the truncated run)
T = 0.1   ->  plans/mix_visual_aligning_mf/.../H8_K100_Meuler_T0.1_..._Emf/6/     (new, empty)
T = 0.05  ->  plans/mix_visual_aligning_mf/.../H8_K100_Meuler_T0.05_..._Emf/6/    (new, empty)
```

**A new threshold is a new folder, always.** No existing result can be overwritten, and the
**CHECKPOINT** path is untouched (`T` is eval-only, and is in `_SAMPLING_OVERRIDE_KEYS`
`eval_mix_visual_aligning.py:2693-2699`) — so this reuses the existing `mf` weights and implies no
retraining.

## What changed

### 1. `config/aligning-d3il-visual.py` — `MIX_PROJ_T`

New block immediately after `_yaml_threshold` is read. Absent → the YAML value is used and nothing
changes. Present → parsed as a float, validated to `[0, 1]`, announced on **stderr**, and substituted
before any block is built. Rejects non-floats and out-of-range values at config-import, before a GPU
is allocated.

### 2. `Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh`

Two new blocks before the python call.

**`MIX_PROJ_T`** — validates with `awk` (numeric, `[0, 1]`), echoes the resolved value, prints the
resulting results-dir fragment, and computes the projector calls per replan from `T` and `$4`
(`FLOW_STEPS`) so the budget is visible before the job starts:

```
[ eval ] projection threshold T = 0.1  (config default 0.5)
[ eval ]   -> results dir becomes H8_K<K>_Meuler_T0.1_...  (new folder, no overwrite)
[ eval ]   -> 10 projector call(s) per replan (K=100 x T=0.1)
[ eval ]   -> vs 50 at the shipped T=0.5
```

and, when `T` is left at the default with `K >= 50`, a **warning naming the wall failure**:

```
[ eval ]   ⚠  WARNING: K=100 at T=0.5 means ~50 SLSQP solves per replan.
[ eval ]      Every K>=50 projected cell in this tree has hit the 24 h wall and
[ eval ]      truncated (mf K100 died at 11/30, needing 50 h). ...
```

**Arm C status** — echoes whether `HFFM_VARIANTS` enabled HardFlow, which NLP backend resolved, that
`slsqp` writes `hardflow_sls-*` (so the IPOPT corpus cannot be overwritten,
`hardflow_projection.artifact_variant_label`), and whether `HFFM_ACT_THRESHOLD` has deliberately
unmatched arms B and C.

## Arms B and C stay matched for free

`config/visual_aligning_eval.yaml:458 activation_threshold: null` means arm C **inherits**
`diffusion_timestep_threshold` (`eval_mix_visual_aligning.py:3127-3130`). So `MIX_PROJ_T=0.1` moves
**both** the DPCC projector and HardFlow to 10 solves per replan, and the comparison stays
like-for-like without a second knob. `HFFM_ACT_THRESHOLD` breaks the match on purpose if wanted.

## Not changed

- No math. `T` was always a supported knob; U11 only makes it settable per job and audible in the log.
- The YAML default stays `0.5`, so every existing path and every job that does not set `MIX_PROJ_T`
  is byte-identical to before.
- `mf`/`fm`/`diffusion`/`af` blocks, training paths, checkpoints — untouched.

## Verified here (container-side only)

- `ast.parse` on the config: OK. `bash -n` on the sbatch: OK.
- The sbatch block dry-run on four inputs: `T=0.1` → 10 calls; `T=0.05` → 5 calls; `T` unset with
  `K=100` → the wall warning; `T=1.5` → error, exit 1.

## 🔴 Still to verify on the cluster

1. The config actually imports under `MIX_PROJ_T` (needs `diffuser`; cheapest check is
   `gates_mix_visual.sh`).
2. The eval log's results path really contains `T0.1`, **before** letting the job run to the wall.
3. That the projector call count in the eval's own `--flow-steps` banner
   (`eval_mix_visual_aligning.py:2842-2845`) agrees with the sbatch's estimate.

## Wall-time extrapolation (NOT measured — read the caveat)

Backed out of this batch's measured `avg_time_ms`: sampler-only `mf` K100 = 893 ms/step; DPCC solve
≈ 282 ms (from `(14988 − 893)/50`); HardFlow-SLSQP solve ≈ 34 ms (IPOPT 147 ms ÷ the 4.33× SolverSwap
ratio measured in job 25121).

| config | est. ms/step | est. h for 30 rollouts | fits 24 h? |
|---|---|---|---|
| K100 `T=0.5` `dpcc-r` *(the run that died)* | 14 988 ✅measured | **50.0** | ❌ |
| **K100 `T=0.1` `dpcc-r`** | ~3 712 | **~12.4** | ✅ |
| **K100 `T=0.05` `dpcc-r`** | ~2 302 | **~7.7** | ✅ |
| **K100 `T=0.1` `hardflow_sls-*`** | ~1 232 | **~4.1** | ✅ |
| **K100 `T=0.05` `hardflow_sls-*`** | ~1 063 | **~3.5** | ✅ |

⚠️ The DPCC per-solve cost is uncertain by ~10×: backing it out of the K = 2 cell instead gives
≈28 ms, which would put `T=0.1` at ~3.9 h. **282 ms is the conservative end** and is the number the
table uses. Either way both thresholds clear the wall with room; run one variant first and read the
real `avg_time_ms` before launching the full sweep.

---

## Added after the first pass — three gaps closed

### A. `--proj-threshold` CLI flag, and 🔴 the two-place application bug

The env-only version had a real defect: **the threshold is read in two independent places.**

| consumer | source | drives |
|---|---|---|
| `config/aligning-d3il-visual.py` `_yaml_threshold` | config module | `args` → `exp_name` → **the results folder name** |
| `eval_mix_visual_aligning.py` `config = yaml.safe_load(...)` | its OWN load of the same YAML | `_gc = dict(config)` → `setup_dpcc_projector` → **the actual projector**, and HardFlow's inherited threshold |

`MIX_PROJ_T` alone moved only the first. A run would have written a folder named `T0.1`
while the projector ran at `0.5` — the exact mislabelling this file already warns about for
`--flow-steps`. Both are now set from **one** resolved value (CLI > env > YAML), applied
before any `Parser().parse_args()`, and a guard raises if they ever disagree:

```
[ eval ] ERROR: projection threshold disagrees between the config module (0.1, drives the
results folder name) and the eval yaml (0.5, drives the actual projector). A run in this
state would be mislabelled.
```

### B. The snapshot now reflects the overrides

`Parser.snapshot_configs()` (`diffuser/utils/setup.py:188-220`) copies **files on disk**, so
an override leaves no trace — and it hardcodes `config/projection_eval.yaml`, which **this
generation does not read** (it reads `config/visual_aligning_eval.yaml`; both files exist, so
the snapshot has always looked plausible while describing a config no visual-aligning run
ever used).

Rather than add a parallel manifest, this reuses the existing `diffuser/utils/provenance`
(already imported and called by this eval, U10.1):

- **`TRACKED_ENV` extended** with `MIX_PROJ_T`, `MIX_BONE*`, `MIX_VIS_*`, `MIX_TRAIN_STEPS`,
  `MIX_AF_ALPHA_*`, `FMPCC_HF_NLP_BACKEND`. Purely additive — other generations list them
  under `env_absent` and are otherwise unaffected.
- **`yaml_path` corrected** to `config/visual_aligning_eval.yaml`.
- **`resolved` gained** `diffusion_timestep_threshold`, `t_override_source`, `flow_steps_v3`,
  `n_diffusion_steps`, `projector_calls_per_replan`, `hf_nlp_backend`, `engine`.

`t_override_source` is the point: it distinguishes *"T=0.1 because the submitter asked"* from
*"T=0.5 because the yaml said so"* — invisible in the results path, since both write a `T`
token.

### C. `MIX_PROJ_T` is now a LIST — one command, both thresholds

`MIX_PROJ_T="0.1 0.05"` runs one full eval pass per value, sequentially, in the same
allocation. Each pass gets its own results dir (`T` is a plan path key), so passes cannot
collide with each other or with the existing `T0.5` run.

`set -e` is relaxed around the loop so a failure in pass 1 does not discard pass 2; each pass
reports its own status and the job exits non-zero at the end if any failed. Inside the loop
the env form is unset (`MIX_PROJ_T= run_eval "$T"`) — a lingering list would fail the eval's
`float()` and kill the pass.

Dry-run:

```
[ eval ] projection threshold sweep: T = 0.1 0.05   (config default 0.5)
[ eval ]   T=0.1    -> 10 projector call(s)/replan  -> dir H8_K100_Meuler_T0.1_...
[ eval ]   T=0.05   ->  5 projector call(s)/replan  -> dir H8_K100_Meuler_T0.05_...
```

Invalid entries are rejected before the first pass starts (`MIX_PROJ_T="0.1 7"` → exit 1).

---

## Fix — job 25215: `MIX_PROJ_T= run_eval` exported a BLANK, not an unset

**Both sweep passes died at config-import**, before a single rollout:

```
File "config/aligning-d3il-visual.py", line 41, in <module>
    _env_T = float(_env_T)
ValueError: could not convert string to float: ''
ValueError: CRITICAL: MIX_PROJ_T='' is not a float.
[ eval ] ❌ PASS T=0.1 FAILED (exit 1)
[ eval ] ❌ PASS T=0.05 FAILED (exit 1)
```

**Cause — mine.** Inside the sweep loop I wrote `MIX_PROJ_T= run_eval "$T"` to clear the list
form before handing the pass its value via `--proj-threshold`. **`VAR= cmd` does not unset VAR —
it exports it as the empty string.** So `os.environ.get('MIX_PROJ_T')` returned `''`, the
`is not None` guard was True, and `float('')` raised. The sweep was the only caller that used
that idiom, so it broke exactly the feature it was added for, and broke it on both passes.

Everything upstream of the crash was correct — the banner resolved and printed the whole plan
(`T=0.1 -> 10 calls`, `T=0.05 -> 5 calls`, arm C on, `slsqp -> hardflow_sls-*`, arms B/C matched),
so the failure is isolated to this one shell idiom.

**Fixed in two independent places, deliberately — keep both.**

1. **The cause, in the sbatch.** Save / real `unset` / restore around the call, so the child sees
   no variable at all:
   ```bash
   MIX_PROJ_T_SAVED="$MIX_PROJ_T"
   unset MIX_PROJ_T                 # a real unset, so the child sees no variable at all
   run_eval "$T"
   rc=$?
   MIX_PROJ_T="$MIX_PROJ_T_SAVED"   # restore for the next iteration's bookkeeping
   ```
   Simulated: child now sees `MIX_PROJ_T=None` on both passes; the old form gave `''`.

2. **The brittleness, in the readers.** A knob must not be one shell quirk away from a crash, so
   **blank now means unset** everywhere. New `_env_or_none()` in the config (returns `None` for
   unset *or* whitespace-only, and strips), routed through by `MIX_PROJ_T` **and** the U10
   `MIX_AF_ALPHA_*` readers, which had the identical latent defect. The eval's own reader switched
   from `is not None` to `.strip()` truthiness, and its error message now names the likely cause
   (a sweep list reaching the child).

| `MIX_PROJ_T` | before | after |
|---|---|---|
| unset | inert ✅ | inert ✅ |
| `''` / `'   '` | 💥 **ValueError at import** | inert ✅ |
| `'0.1'` / `' 0.05 '` | override ✅ | override ✅ (stripped) |
| `'abc'` | rejected ✅ | rejected ✅ |
| `'1.5'` | rejected ✅ | rejected ✅ |
| `'0.1 0.05'` reaching the child | 💥 bare `ValueError` | `SystemExit` naming the cause ✅ |

⚠️ **Sibling not touched:** `_mix_u9_keys()` reads `MIX_VIS_PRETRAINED` / `MIX_VIS_LR_SCALE` /
`MIX_VIS_COND` with the same `if raw is not None` shape and would raise the same way on a blank.
No current sbatch uses the `VAR=` idiom on those, so it is latent, not live — flagged here rather
than changed, since it is outside U11.

**Re-run:** the same command. Nothing else about the job changes.
