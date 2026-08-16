# CHANGELOG — Gen15 U2: the HardFlow arm returns, + pillars & s_curve with GIF recording

**Date:** 2026-08-15 · **Type:** feature (new guidance arm) + campaign setup
**Status:** code complete, **NOTHING RUN** — syntax-checked only; all execution is a cluster job
**Files:** `mix_uav/sampling/hardflow_projection.py` (new), `mix_uav/models/engine_registry.py`,
`mix_uav_test/eval_mix_uav.py`, `mix_uav_test/gates_mix_uav.py`, `config/uav_mix.py`,
`Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh`
**Retraining required:** none for the arm itself (eval-time sampler). New scenes need their own
training runs.

---

## 1. 🔴 The init plan's HardFlow blocker was wrong — correcting it

Init plan §1.9 put HardFlow out of scope because *"the UAV frame has no linear-dynamics `.npz`
in UAV normalizer units (the Gen12 refit warning)"*.

**That applies only to `dynamics_mode='linear_fit'`, which is optional and not the default.**
The default is `'deriv'`, which writes the dynamics rows straight into the NLP:

```python
unnorm[t + 1][x_idx] == unnorm[t][x_idx] + self.dt * unnorm[t][dx_idx]
```

No fitted `A, B, c` anywhere. Gen3v6's own eval runs `deriv`. Verified compatibility with UAV
before writing a line:

| requirement | UAV reality | ok |
|---|---|---|
| NLP supports the emitted constraint kinds | HardFlow handles `ineq`, `lb`/`ub`, `sphere_outside/inside`, `deriv`; UAV emits exactly those four | ✅ |
| `deriv` assert `x_idx >= action_dim` | UAV deriv rows are `(3,0)…(8,2)`, action_dim = 3 | ✅ |
| `dt` | HardFlow default 1.0; UAV uses `dt: 1.0` (the action IS Δp_des) | ✅ |
| `goal_dim == 0` assert | UAV forces `goal_dim = 0` already | ✅ |
| normalizer shape | `HardFlowPolicy` derives mins/maxs from `normalizer.normalizers['actions'/'observations']` | ✅ |

So the arm is portable with **no refit**. `linear_fit` remains unavailable and unused.

## 2. Two engine-specific values that would have been silent bugs

Gen3v6's port hosted **one** engine and hard-defaulted both. Gen15 hosts three and they disagree:

| engine | `init_noise_scale` | source | `two_time` | source |
|---|---|---|---|---|
| `fm` | **0.5** | `diffusion.py:184` `0.5 * torch.randn` | **False** | `_predict_velocity(x, cond, t, returns)` — **no `h`** |
| `mf` | 1.0 | `mf_diffusion.py:205` `torch.randn` | True | `_predict_velocity(..., h=None, ...)` |
| `af` | 1.0 | `af_diffusion.py:261` `torch.randn` | True | same |

- **`init_noise_scale`** — Gen3v6's `HardFlowPolicy` defaulted it to `1.0`. On Gen15's `fm` arm
  that starts the ODE at **twice** its trained noise scale: an out-of-distribution τ=0 state that
  looks like nothing but a slightly worse model. This is Gen3v6's own `fix_4` bug, re-armed by
  multi-engine hosting.
- **`two_time`** — decides which **field** the NLP is handed. The two-time engines emit the
  interval average `u(x,r,t)` and must be queried at `h=0`, where `u(x,t,0) = v(x,t)` exactly.
  `fm` emits `v` directly and **has no `h` parameter** — passing one is a `TypeError`; omitting
  it on a two-time arm silently swaps the field.

**Both are now REQUIRED arguments with no defaults** (`assert ... is not None`), published per
engine by `engine_registry`, and asserted by gate **G8** against the engines' own source
(`inspect.getsource(p_sample_loop)` for the sigma, `inspect.signature(_predict_velocity)` for the
field). The gate cannot drift from the thing it guards.

## 3. Where the variants live — and why not in the yaml

`hardflow_new`, `hardflow_new-c`, `hardflow_new-t` are declared in **`config/uav_mix.py`**
(`hardflow_variants`), not in `config/uav_projection.yaml`.

The yaml is **shared read-only with Gen11**. Adding an arm there would make Gen11's next eval
try to run a variant Gen11 has no code for. The split is deliberate and matches the reason the
yaml is shared in the first place: **constraints stay shared** (that is the half that must match
for a DPCC-vs-HardFlow comparison to be valid), **arms do not**.

The NLP is built from the **same `constraint_list` the DPCC `Projector` consumes** —
`setup_dpcc_projector(..., return_constraint_list=True)` returns the list before it builds the
Projector. If the two arms enforced different sets, the comparison would be void.

`UAV_MIX_HF_OFF=1` disables the arm for a DPCC-only run.

Threshold matching: `activation_threshold = 0.5` equals the DPCC `diffusion_timestep_threshold`,
and both use DPCC polarity (higher = more projection), so the two arms are threshold-matched by
construction and the `T0.5` token in the output path stays honest.

## 4. ⚠️ FAIRNESS: HardFlow costs 2K network evals, DPCC costs K

`hardflow_new` evaluates the network **twice per ODE step** — the reference step and the
terminal predict. So **arm C at K=10 spends 20 NFE where a DPCC arm at K=10 spends 10.**

"Same K" is therefore **not** the same generation budget. The summary now records the truth:

```
'hardflow': {'is_hardflow', 'nfe_total', 'nfe_per_plan', 'nlp_solves_total',
             'nlp_failures_total', 'activation_threshold', 'init_noise_scale', 'two_time'}
```

**`nfe_per_plan` is the number to quote in the DA**, not K. A DPCC-vs-HardFlow table at matched
K silently gives HardFlow double the compute.

## 5. Selection-rule fix

`_selection_for` matched on the literal substrings `'dpcc-t'` / `'dpcc-c'`, so `hardflow_new-c`
would have fallen through to `'random'` — the `-c` arm would not have been a min-cost arm at all.
Now it also matches `-c` / `-t` suffixes. Verified against all 11 variant spellings; the existing
DPCC mappings are unchanged.

## 6. Scenes: pillars + s_curve, `--record all`

| scene | why | max_episode_length | note |
|---|---|---|---|
| **pillars** | 4 sphere obstacles → exercises HardFlow's `sphere_outside` NLP rows, which corridor never touched (corridor is halfspace-driven). Fixed start/goal, so goal-reaching is well defined. | 634 | the scene Fix_12 repaired most — both channels were closed before it |
| **s_curve** | non-convex, ~24 cm bands, switched per segment. `U_13` found every geometry-keeping DPCC variant eventually crashes here and `geo_free` does best — so it is the strongest available test of whether HardFlow's NLP beats the linear projector. | 871 | highest risk of a null result **and** of a 24 h timeout |

### 6.1 🔴 Runtime warning — read before submitting s_curve

Gen11's own `Fix_11` note says s_curve at **18 variants × 10 trials can brush or exceed the 24 h
wall clock with no rendering at all**. U2 adds to that load in three ways at once:

- **+3 variants** (the HardFlow arm) → 23 total;
- **HardFlow solves an IPOPT NLP per activated ODE step**, on CPU;
- **`--record all` renders a GIF per rollout** — 23 × 10 = 230 GIFs per scene — and the render
  time lands **inside the measured per-step wall clock**, contaminating the `total_ms` column
  that Gen15 exists to measure.

**Recommendation: pillars first at full 10 trials; s_curve at 3–5 trials for the recorded run.**
If the timing numbers matter more than the GIFs for a given scene, run that scene twice —
`--record none` for metrics, a short `--record all` job for the artifacts. The corridor K=10
numbers already in hand were recorded with `--record none`, so they stay the clean timing
reference.

---

## 7. How to run it

**Step 0 — gates (now 8, ~1 min).** G8 is new and cheap; it is the whole defence for §2.

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/gates_mix_uav.sh
```
Expect `G8 PASS` with three lines confirming `init_noise_scale` / `two_time` against the engine
source, plus a casadi line (`⊘` off-cluster, `✓` on it).

**Step 1 — train the two new scenes** (the corridor checkpoint does not transfer; the FM is
state-only and cannot tell scenes apart, so one model per scene):

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh mf pillars "6"
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh mf s_curve "6"
```

**Step 2 — eval with GIFs.** `uav_mix_pipeline.sh` chains train→eval, but the trains above are
already submitted, so eval directly (args: `engine scene seeds n_trials projection record K`):

```bash
# pillars — full trials
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh mf pillars "6" 10 fm_only all 10
# s_curve — fewer trials (see §6.1)
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh mf s_curve "6" 3 fm_only all 10
```

**Step 3 — the K sweep**, once the arm is proven on one scene. `eval_k_sweep.sh` now takes
`$7=record`; leave it `none` so the timing column stays clean:

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf pillars "6" "1 2 5 10 20" 10 fm_only none
```

### What to check in the first eval log

```
[ eval ] HardFlow arm: +3 variants ['hardflow_new', 'hardflow_new-c', 'hardflow_new-t'] (from config/uav_mix.py, NOT the shared yaml)
[hardflow] engine=mf K=10 noise_sigma=1.0 two_time=True dyn=deriv A=0.5 sel=random
```
`noise_sigma` must be **1.0** on `mf`/`af` and **0.5** on `fm`. If it reads 1.0 on an `fm` run,
stop — §2's bug is back.

---

## 8. Not done / open

1. **`linear_fit` dynamics mode** stays unavailable — it needs an `A,B,c` refit in UAV normalizer
   units. `deriv` is the faithful UAV formulation anyway (the action *is* the position delta), so
   this is a deferred curiosity, not a gap.
2. **Gen3v6/Gen3v7 are not patched** with the `two_time` / required-`init_noise_scale` changes.
   They host one engine each, so their defaults are correct there. Isolation preserved.
3. **HardFlow on `fm` has never been run anywhere** — Gen12 ported it to FMv3ODE but Gen15's
   `fm` arm is the first UAV host. Expect the first `fm` + `hardflow_new` job to surface
   something.
4. **The `-tightened` siblings** are not offered for the HardFlow arm. Tightening applies an
   `enlarge_constraints` margin to spatial rows; whether that is meaningful inside a prox-NLP
   rather than a projection is an open question, and corridor showed tightening inverts on this
   task anyway (`dpcc-c` 1.00 → 0.60).

---

## 9. ⚠️ Late addition — HardFlowPolicy had the Fix_1 hole too

Caught on review, before any run: `HardFlowPolicy` came from Gen3v6, the same generation whose
`policies.py` had no real-time logging. It therefore set **none** of the attributes the UAV eval
reads after every plan — `last_proj_ms`, `last_proj_cost`, `last_which_trajectory`, `last_infos`,
`projector`.

Every read on the eval side is a `getattr(..., default)`, so **nothing would have crashed.** It
would have reported `proj_ms = 0.0`, and since the eval derives `fm_ms = total_ms − proj_ms`, the
**entire IPOPT solve time would have been booked as pure network inference** — the identical
corruption Fix_1 repaired on `mf`/`af`, on a third class, in the same week.

Fixed at both ends:

1. `HardFlowNLP` now accumulates `solve_ms` around `opti.solve_limited()` (a *failed* solve is
   timed too — it still costs), and the sampler emits the per-call delta as
   **`infos['projection_ms']`** — the same key, units and meaning the DPCC engines use. That is
   what keeps `proj_ms` / `fm_ms` comparable between arm B (project-after) and arm C (NLP-inside).
2. `HardFlowPolicy.__call__` publishes `last_proj_ms`, `last_proj_cost` (prox cost of the
   *executed* candidate), `last_which_trajectory` and `last_infos`; `self.projector = None` so the
   eval's circuit-breaker probes correctly read "never tripped" (arm C has no DPCC projector —
   its convergence health is `nlp.n_failures`, reported in the summary instead).

Gate **G7** already asserts this contract and now covers arm C by construction.

**Lesson worth keeping:** `getattr(x, k, default)` across a component boundary converts a crash
into a plausible-looking number. Three instances of the same bug have now been found in Gen15
(mf, af, HardFlow). When adopting a class from another generation, diff its published attribute
surface against what the host actually reads — do not trust "drop-in replacement" in a docstring,
because it was drop-in for *that* generation's Policy, not this one's.
