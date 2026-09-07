# CHANGELOG — Gen15 Fix_16: the degenerate (constant) action channel

**Date:** 2026-09-01
**Scope:** `mix_uav/` + `mix_uav_test/` + `Slurm_Codes/sbatch/uav_mix/` (Gen15 only — see §5)
**Evidence:** [`../Study/STUDY_20260901_mf_unguided_failure_uav_pillars.md`](../Study/STUDY_20260901_mf_unguided_failure_uav_pillars.md)
**Retrain required:** ❌ **No.** The fix is checkpoint-compatible — proven in §3.
**Status:** patched locally, **not committed**, **not yet run on the cluster.**

---

## 1. The problem

### 1.1 What was observed

On `pillars`, unguided (`variant=diffuser`), MeanFlow scored **0/30 goal-line crossings and 30/30
divergence aborts** at every K, while FlowMatching scored 0.90 and α-Flow 0.93 on the same scene,
seeds, controller and code rev. The drone climbed vertically out of the flight envelope from the start
line, aborting at FM step 47–144 having covered ~0.24 m of a 6.4 m course.

### 1.2 Root cause — three links, only one of them engine-specific

| # | link | shared or `mf`-only |
|---|---|---|
| 1 | **The expert data contains no vertical control signal at all.** On `pillars` the vertical action is *exactly* constant (`Δz ≡ 0`) — every eval banner has always printed `Constant data in dimension 2 \| max = min = 0.0`. **No training example anywhere shows an altitude error being corrected**, so the learned vertical feedback gain is unconstrained by data. | **shared** |
| 2 | **`SafeLimitsNormalizer(eps=1)` turned that channel into the loudest output in the action space.** For a constant dim it widens `[c,c]` → `[c−1, c+1]`, which makes `unnormalize` the **identity**: the model's raw *normalized* output is emitted verbatim in metres, and `LimitsNormalizer`'s clip to `[−1,1]` becomes a **±1 m ceiling** — applied **silently**, because the warning print was commented out. | **shared** |
| 3 | `mf` learned a **positive** vertical feedback gain, `b₁ = +0.11 m` commanded per metre of altitude error; `fm`/`af` landed at **−0.03…+0.01**. Closed loop → `e_z ← e_z·(1+b₁)` → 3.3 m by step ~60. | 🔴 **`mf`-only** |

Measured consequences of link 2, from the raw `act_all` artifacts:

- `max |Δz| = 1.00000` **exactly** for `mf` and `af`; `mf` K10 sat on that ceiling for **55.5 %** of every executed step.
- `max |Δx| ≈ 0.044` for **every** arm — the real channels are compressed by their data range. The two ceilings are **23× apart**, an asymmetry created entirely by the normalizer.
- The **projector could not stop it either**: `action_bounds:'auto'` derives its cap from `act_normalizer.mins/.maxs` (`eval_mix_uav.py:1004-1006`), so the vertical action bound *was* **±1 m**. That is why `bounds_free` made no measurable difference in the constraint-group ablation.
- The *observation* normalizer has the same defect on `p_des_z` (also constant), so the plan's own altitude channel was identity-scaled too.

### 1.3 Is this a bug?

**No — and the changelog should not claim it is.** `SafeLimitsNormalizer` does exactly what it says: it
avoids `0/0 = NaN` and **round-trips the data exactly** (`normalize(c) = 0`, `unnormalize(0) = c`). A
correctness audit passes it, correctly. It is a **scale-calibration defect**: `eps` is a magic constant
expressed in *data units*, benign at D3IL/maze's O(1) actions and three orders of magnitude out of
scale at the UAV's O(0.02 m) actions. It is **inherited from upstream** —
`aux_repo/dpcc/diffuser/datasets/normalization.py:182` carries the same `eps=1`.

It is an **amplifier, not the cause**: a model that learned the channel emits ~0 and is untouched,
which is exactly what `fm` does. Fixing it does **not** by itself explain link 3.

---

## 2. What changed

### 2.1 `mix_uav/datasets/normalization.py`

| change | why |
|---|---|
| **`SafeLimitsNormalizer.__init__(eps=1)` → `eps=None` + `_resolve_eps()`** | The widening is now derived from the data: `eps = FMPCC_SAFE_EPS_FRAC × median(half-width of the non-constant dims)`, floored at `1e-8`. The degenerate channel becomes **quieter** than every real one instead of louder. |
| **`FMPCC_SAFE_EPS_MODE=legacy` escape hatch** | Restores `eps=1.0` byte-for-byte so pre-fix runs can be reproduced and A/B'd without checking out an old rev. |
| **All-dimensions-constant fallback** | If there is no non-constant dim to reference (e.g. the `rewards` field), fall back to `eps=1.0` **and say so** rather than invent a scale. |
| **`LimitsNormalizer.unnormalize`: clip warning un-commented** | The clip was silent. A batch could sit on the ceiling for >50 % of its steps with no log line anywhere. Now warns **once per normalizer** (batch logs must stay quiet) and keeps `_clip_events` / `_clip_max` counters. |
| **`Normalizer.__init__` takes `key=None`; `DatasetNormalizer` forwards it** | Diagnostics can now say `actions[2]` instead of a bare `2`. The forward is `try/except TypeError` with a fallback to the original call, so every other normalizer class still constructs unchanged. The pre-existing bare `except:` was narrowed to `except Exception:`. |
| **`degenerate_dims` / `degenerate_eps` attributes** | So the eval can report them instead of the reader reverse-engineering them from artifacts. |

### 2.2 `mix_uav_test/eval_mix_uav.py`

| change | why |
|---|---|
| **`_report_degenerate_dims()`**, called before the projector is built | Prints, per degenerate channel: which field and index, the eps chosen, what a saturated output now commands, and the resulting `action_bounds:'auto'` vector. This is the diagnostic whose absence made the defect invisible. |
| **`_uav_eval_tag()` accepts `FMPCC_UAV_EVAL_TAG`**, appended last and sanitised to `[A-Za-z0-9._-]` | 🔴 **Required for A/B.** Two evals differing only in an env knob previously produced an *identical* folder name and the second **silently overwrote** the first. |
| `import re` added | Used by the tag sanitiser. |

### 2.3 `Slurm_Codes/sbatch/uav_mix/`

`eval_mix_uav.sh` and `eval_k_sweep.sh` export `FMPCC_SAFE_EPS_MODE` (default `scaled`),
`FMPCC_SAFE_EPS_FRAC` (default `1e-3`) and `FMPCC_UAV_EVAL_TAG`, and echo all three into the job
banner so a run is reproducible from its log alone. No change to GPU/EGL isolation, walltime, or
buffering.

---

## 3. Why no retrain is needed — and the verification

For a constant dimension, `normalize(x) = 2·(x−(c−eps))/((c+eps)−(c−eps)) − 1 = 0` **for any `eps`**.
The normalized training data is therefore bit-identical before and after; only the `unnormalize` of
*model outputs away from 0* changes. The normalizer is rebuilt from the dataset at eval time
(`dataset.normalizer`), so the fix applies to existing checkpoints immediately.

Verified locally on a synthetic UAV-shaped dataset (`dx,dy` real; `dz ≡ 0`):

| | `eps` | saturated `+1.0` output → | projector auto bound (z) | `normalize(train)[:,2] == 0` | round-trip |
|---|---|---|---|---|---|
| **legacy** | 1.0 | **1.0 m/step** | ±1.000 | ✅ | ✅ exact |
| **scaled (new default)** | 3.998e-05 | **3.998e-05 m/step** | ±4.0e-05 | ✅ | ✅ exact |

**A 25 000× reduction** on the saturating channel; worst-case accumulated drift over a full 634-step
episode falls from 634 m to **0.025 m**. The real `observations`/`actions` channels are untouched, and
`GaussianNormalizer` / `LimitsNormalizer` / `DebugNormalizer` still construct through the fallback path.

⚠️ **Not yet validated on the cluster.** Everything above is static analysis plus an offline unit test;
nothing has been rolled out in MuJoCo. §4 is the run that closes that.

---

## 4. How to run it — SLURM, with non-overwriting folders

`FMPCC_UAV_EVAL_TAG` is what keeps the runs apart. **Without it the A and B arms write to the same
folder and the second overwrites the first.**

**A — the fix (primary run):**

```bash
FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=fix16scaled \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf pillars "6" "1 2 5"
```

**B — the pre-fix control (same code, same checkpoint, legacy eps):**

```bash
FMPCC_SAFE_EPS_MODE=legacy FMPCC_UAV_EVAL_TAG=fix16legacy \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf pillars "6" "1 2 5"
```

**C — the other two arms, to confirm the fix does not regress them** (`fm` should be ~unchanged; it
never approached the ceiling):

```bash
FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=fix16scaled \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh fm pillars "6" "1 2 5"
FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=fix16scaled \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af pillars "6" "1 2 5"
```

Output folders become e.g.
`logs/UAV_MIX/uav-pillars/plans/mix_uav_mf/H8_D…/Emf_K1_mpc4_pid_stopgo_T0.5_fix16scaled/6/…`
alongside the untagged pre-fix `Emf_K1_mpc4_pid_stopgo_T0.5/6/…` already on disk — **nothing is
overwritten.**

### 4.1 What to check in the log

```
[ eval ] Fix_16 DEGENERATE actions[2]: constant in the expert data — no training signal for this
         channel. Widened by eps=3.998e-05; a saturated model output on it commands +/-3.998e-05
         data-units/step.
[ eval ] Fix_16 projector action_bounds=auto → lb=[...] ub=[...]  (degenerate dims now bounded, not +/-1)
```

### 4.2 Falsifiable prediction

🧪 **`mf` unguided (`diffuser`) should now fly `pillars`.** Its forward channel is healthy — per
step-bin it equals or exceeds `fm` and `af` — and the only thing killing it was the vertical loop,
whose gain is now multiplied by `4e-05` instead of `1.0`.

🔴 **If `mf` still aborts 30/30, the mechanism in the study is wrong** and links 1–2 are incidental.
That is the point of arm B: it isolates this one variable on an otherwise identical run.

### 4.3 ✅ RESULT — the prediction passed (2026-09-03)

Run on the cluster, jobs **25316–25321**, all on git rev `def8fdf`. Full analysis:
`logs_in_develop/Gen15/DA/DA_20260903_fix16_AB_mf_pillars.md`.

| `mf` `pillars` `diffuser` | K1 | K2 | K5 |
|---|---|---|---|
| abort %, arm B (`legacy`) | 100.0 | 100.0 | 100.0 |
| abort %, arm A (`scaled`) | **0.0** | **0.0** | **0.0** |
| goal_dist legacy → scaled (m) | 6.50 → **0.62** | 6.46 → **0.66** | 6.49 → **0.36** |
| success % legacy → scaled | 0 → 10 | 0 → 30 | 0 → **90** |
| phys_safe legacy → scaled | 0.00 → **1.00** | 0.00 → **1.00** | 0.00 → **1.00** |

Supporting evidence, in order of weight:

- **Arm B reproduced the pre-fix run bit-for-bit** — every metric of the `fix16legacy`
  candidates equals the untagged pre-fix candidates to the printed decimal. The A/B therefore
  isolates exactly one variable, and `SAFE_EPS_MODE=legacy` is an exact rollback switch.
- **The `*-geo_free` control arms, already healthy pre-fix, did not move** (K1 abort 0 % → 0 %,
  track_err 0.329 → 0.325). The fix moves only what was broken.
- **Aborts fall on 9 of 10 non-`geo_free` variants** at K1/K2; `overspeed` aborts vanish
  entirely and `off_route` falls 16 → 1 at K1.
- Realised `eps` on the cluster was **3.086e-05** (offline prediction: 3.998e-05); the projector
  z action-bound went ±1.000 → **±3.1e-05**, a 32,000× reduction.

Two things the result does **not** say:

- 🔴 **`pillars` is still unsolved.** Success+constraints is **0 / 2876** rollouts — every
  engine, every K, both arms. Fix_16 removes a failure mode; it does not make the scene solvable.
- 🔴 **`fm` and `af` were not re-run**, so no cross-engine comparison is available yet.

Two follow-ups the run surfaced: both **K5 jobs died at the wall clock** (the fixed drone now
flies ~600 steps instead of dying at ~77, so a rollout costs ~10× more wall time — `--time`
must go up), and **per-step projection cost rose 1.2–1.6×** because the z decision variable is
now a near-equality constraint for SLSQP.

---

## 5. Scope, and what was deliberately NOT changed

- **Gen15 only.** `SafeLimitsNormalizer(eps=1)` exists in **20 sibling copies** of
  `datasets/normalization.py` across the repo (and in upstream `aux_repo/dpcc`). Per the
  copy-modify-isolation convention this patch touches **only `mix_uav/`**. The other generations are
  **unfixed and still carry the defect** — it is latent for any of them whose data has a constant
  dimension. Syncing them is a separate, explicit decision.
- **No model, sampler, objective or training code was touched.** Fix_16 changes normalization,
  diagnostics and run-naming only.
- **Link 3 is not addressed.** Why MeanFlow extrapolates to a positive gain in a channel with no
  training signal remains open; the study's §4.2 2×2 (`mf` on the SiT bone) is the experiment for it.
  Fix_16 removes the amplifier, not the cause.
- **`config/uav_mix.py` unchanged** — the fix needs no config change, and `normalizer:
  'SafeLimitsNormalizer'` stays the right choice.

---

## 6. Files touched

| file | lines | nature |
|---|---|---|
| `mix_uav/datasets/normalization.py` | +123 / −15 | the fix + diagnostics |
| `mix_uav_test/eval_mix_uav.py` | +40 | reporting + unique eval tag |
| `Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh` | +11 | env knobs + banner |
| `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` | +7 | env knobs + banner |

**Nothing committed** — awaiting review.
