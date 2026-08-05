# DA — Gen0 Fix2 cluster validation (θ = 1.0 A/B against the pre-fix run)

**Date:** 2026-08-05
**Analyst input:** `temp/0508/` (post-fix) vs `temp/0408/` (pre-fix)
**Scope:** Gen0 `scripts/eval.py` only — defects **A** (threshold not forwarded) and **B** (`post_processing` not defined).
**Verdict: THE FIX WORKS.** Both defects are closed, proven at the byte level with a clean negative control.

---

## 1. Verdict up front

| Claim | Status | Strongest evidence |
|---|---|---|
| The YAML threshold now reaches `Projector` | ✅ **PROVEN** | `diffuser` bit-identical while all 12 projector variants changed; dpcc-r 0.41 → 1.70 s/step |
| `post_processing` is no longer a duplicate of `dpcc-r` | ✅ **PROVEN** | was byte-identical in 3/3 halfspaces, now differs in 3/3 |
| `post_processing` is one final projection (paper definition) | ✅ **PROVEN** | costs +0.014 s/step over `diffuser` = exactly one solve, and output ≠ `diffuser` |
| Defect **C** (bare-float gate in the FM model packages) | ⬜ **NOT TESTED** | this run exercises `scripts/eval.py` only — no FM path touched |

Neither run is a performance measurement. See §7.

---

## 2. Run identification

| | pre-fix | post-fix |
|---|---|---|
| Folder | `temp/0408/` | `temp/0508/` |
| Log | `18_03_01_eval_dpcc_job_24254.log` | `00_36_44_eval_dpcc_job_24279.log` |
| Slurm job | 24254 | 24279 |
| Node | i6-gpu-1 | i6-gpu-1 |
| `GIT REV` | pre-`205c494` | **`205c494`** ← the Fix2 commit |
| YAML `diffusion_timestep_threshold` | 1 | 1 |
| Threshold **actually used** | **0.5** (constructor default) | **1.0** |
| Seeds / trials / halfspaces | `[6]` / 2 / 3 | `[6]` / 2 / 3 |
| Model | `H8_K20_Dmodels.GaussianDiffusion_aw10`, step 91000 | identical |

Both runs requested `T = 1` in `projection_eval.yaml`. Only the code differs. That is what makes this a clean A/B: **the config is a constant, the fix is the only variable.**

`K = n_timesteps = 20`, gate form C (`t <= T·K`, `t` counting down from `K−1`) → `n_active = min(floor(T·K)+1, K)`:

| θ | n_active | which run |
|---|---|---|
| 0.5 | `min(10+1, 20)` = **11** | pre-fix (every run ever, regardless of YAML) |
| 1.0 | `min(20+1, 20)` = **20** | post-fix `dpcc-*`, `gradient`, `model_free` |
| 0.0 | `min(0+1, 20)` = **1** | post-fix `post_processing` ← one solve, at `t = 0`, after the last denoising step |

---

## 3. Evidence 1 — the echo fires

`00_36_44_eval_dpcc_job_24279.log:20`:

```
[ eval ] diffusion_timestep_threshold (from YAML) = 1
```

Absent from the pre-fix log. Confirms the new `scripts/eval.py:59` ran. On its own this proves only that the value was *read* — §4–§6 prove it was *used*.

---

## 4. Evidence 2 — the negative control is bit-identical

`diffuser` is the one variant that gets `projector = None` (`scripts/eval.py:235`). It must be unaffected by any threshold change. Hashing each decompressed `.npz` member:

| halfspace | `diffuser` members that differ pre→post |
|---|---|
| top-right-hard | `avg_time.npy` only |
| top-left-hard | `avg_time.npy` only |
| both-hard | `avg_time.npy` only |

`obs_all`, `act_all`, `sampled_trajectories_all`, `n_success`, `n_steps`, `n_violations`, `total_violations`, `collision_free_completed`, `args` — **all byte-identical across all three halfspaces.** `avg_time` is wall-clock and is expected to jitter.

This is the load-bearing control. It establishes that between the two runs the model weights, the seeds, the RNG stream, the environment and the constraint set were **identical**. Therefore every difference reported in §5 is attributable to the fix and to nothing else.

> ⚠️ Methodological note: hashing the `.npz` **file** is useless here — zip metadata carries timestamps, so all 39 files show as "different" including `diffuser`. Members must be decompressed and hashed individually. Script: `scratchpad/npzcmp.py`.

---

## 5. Evidence 3 — everything that touches the projector moved

Content signature = sha256 over all members except `avg_time.npy`.

| variant | uses projector? | top-right | top-left | both |
|---|---|---|---|---|
| `dpcc-r` / `-tightened` | yes | CHANGED | CHANGED | CHANGED |
| `dpcc-c` / `-tightened` | yes | CHANGED | CHANGED | CHANGED |
| `dpcc-t` / `-tightened` | yes | CHANGED | CHANGED | CHANGED |
| `gradient` / `-tightened` | yes (`gradient=True` mode) | CHANGED | CHANGED | CHANGED |
| `post_processing` / `-tightened` | yes | CHANGED | CHANGED | CHANGED |
| `model_free` / `-tightened` | yes | CHANGED | CHANGED | CHANGED |
| **`diffuser`** | **no (`projector=None`)** | **SAME** | **SAME** | **SAME** |

12 changed / 1 unchanged, and the 1 unchanged is exactly the one with no projector. Perfect separation.

`gradient` and `model_free` are *not* exempt — `scripts/eval.py:232` constructs a `Projector` for them too (`gradient=True` only switches the projector's internal mode), and `:235` nulls it for `diffuser` alone. Their moving is correct, not collateral damage.

---

## 6. Evidence 4 — the `post_processing` duplicate is gone

This is the direct test of defect **B**.

| run | halfspace | `post_processing` ≡ `dpcc-r` ? | `post_processing` ≡ `diffuser` ? | `pp-tightened` ≡ `dpcc-r-tightened` ? |
|---|---|---|---|---|
| pre-fix | top-right-hard | **IDENTICAL** | different | **IDENTICAL** |
| pre-fix | top-left-hard | **IDENTICAL** | different | **IDENTICAL** |
| pre-fix | both-hard | **IDENTICAL** | different | **IDENTICAL** |
| post-fix | top-right-hard | different | different | different |
| post-fix | top-left-hard | different | different | different |
| post-fix | both-hard | different | different | different |

Pre-fix, `post_processing` was a **bit-for-bit clone of `dpcc-r`** in every halfspace — the variant was reporting `dpcc-r`'s numbers under a different name in Table 1. Post-fix it is distinct from `dpcc-r` **and** from `diffuser`, i.e. it is neither "projection throughout denoising" nor "no projection". That is the paper's post-processing: one optimization applied after the last denoising step.

### 6.1 Timing corroborates the count

Mean s/step over the three halfspaces:

| variant | n_active | pre-fix | post-fix | marginal cost over `diffuser` |
|---|---|---|---|---|
| `diffuser` | 0 | 0.181 | 0.179 | — |
| `post_processing` | 1 (was 11) | 0.410 | **0.192** | **+0.014 s** = one solve |
| `dpcc-r` | 20 (was 11) | 0.412 | **1.704** | +1.525 s |

Pre-fix `post_processing` (0.410) sat on top of `dpcc-r` (0.412) — the timing already whispered what the hashes then confirmed. Post-fix it drops to 0.192, a hair above the projector-free `diffuser` at 0.179. One projection, as specified.

### 6.2 Side finding — projection cost is strongly superlinear in `n_active`

From the marginal costs above:

| projections | total projector cost | mean per projection |
|---|---|---|
| 1 (final step, `t=0`) | 0.014 s | 0.014 s |
| 11 (`t = 10…0`) | 0.231 s | 0.021 s |
| 20 (`t = 19…0`) | 1.525 s | 0.076 s |
| **the extra 9 (`t = 19…11`)** | **1.294 s** | **0.144 s** |

The nine earliest projections cost **~10× more each** than the final one. Expected: at `t = 19` the iterate is near-pure noise and grossly violates every constraint, so SLSQP needs far more iterations to converge than it does on a nearly-clean sample at `t = 0`.

**Consequence for the cost model:** the additive form `t = a·N_netcalls + b·NPE` with a *constant* `b` (used in the Gen12 HardFlow DA) is only valid at fixed `n_active`. It underpredicts badly when `n_active` grows toward `K`. Do not extrapolate `b` fitted at θ=0.5 to θ→1.

---

## 7. What this run does **not** establish

- **Not a performance result.** `seeds: [6]`, `n_trials: 2`. Every success-rate cell is one of {0.0, 0.5, 1.0} from two rollouts on a single seed. The SR / CS / violation columns in the raw logs have essentially no statistical power and must not be quoted as findings. What is proven here is **mechanism**, and mechanism is proven at the byte level, which needs no sample size.
- **Defect C is untested.** The bare-float→guarded-gate change lives in the five FM model packages (`flow_matcher_v3/`, `flow_matcher_v3_hardflow/`, `fm_visual_aligning/`, `fm_visual_avoiding/`, `mix_visual_aligning/`). This run never loads any of them.
- **Gen9 defect A is untested.** `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` got the same wiring fix and has not been re-run.
- **The θ=0.5 regression guard has not been run.** At θ=0.5, K=20 the fixed code must reproduce the old numbers exactly (that was the value silently in force). Until that runs, "no past result moves" rests on arithmetic, not on measurement.

---

## 8. Side finding — θ=1.0 is not a usable setting for `gradient`

Now that the knob is live, the extreme setting exposes real behaviour:

| halfspace | `gradient` SR (pre → post) | avg violations (pre → post) | total violation (pre → post) |
|---|---|---|---|
| top-right-hard | 0.5 → **1.0** | 15.5 → **0.00** | 2.787 → 0.073 |
| top-left-hard | 1.0 → **0.0** | 14.5 → 0.00 | 0.489 → 0.125 |
| both-hard | 1.0 → **0.0** | 22.5 → **120.50** | 0.439 → **121.640** |

On `both-hard` the rollout blows up. Applying gradient guidance across all 20 denoising steps — including the pure-noise ones — destabilizes generation. This is the knob behaving, not the fix misbehaving: pre-fix these rows were frozen at θ=0.5 and could never have shown it.

`Avg number of steps: 0.00` in those rows is not a crash — `scripts/eval.py:403` averages steps over successful trials only and prints `0` when `n_success` is all-zero.

The `dpcc-*` variants stay well-behaved at θ=1.0 (violations near zero, `-tightened` arms at SR+C 1.0 on 5 of 6 halfspace/variant cells), which is the expected asymmetry: full-schedule *projection* is stable, full-schedule *gradient guidance* is not.

---

## 9. Reproduction

Scripts live next to this file. Both are pure stdlib (no numpy) so they run in the AI-coding container.

```bash
cd /workspaces/FM-PCC
# member-level npz comparison (whole-file hashes are contaminated by zip timestamps)
python3 logs_in_develop/Gen0_FMPCC_DPCC_Code_Updates/Fix2/npzcmp.py    # §4, §5, §6 tables
python3 logs_in_develop/Gen0_FMPCC_DPCC_Code_Updates/Fix2/metrics.py   # §8 metric deltas
grep -n "diffusion_timestep_threshold" temp/0508/00_36_44_eval_dpcc_job_24279.log   # §3
```

Both scripts have the `temp/0408` and `temp/0508` paths hard-coded at the top; repoint them to compare any other pair of runs.

Ten-second version of §6, runnable on any future eval output:

```bash
python3 -c "
import zipfile,hashlib
d='temp/0508/H8_K20_T1_Dmodels.GaussianDiffusion/6/results/halfspace_both-hard'
h=lambda v: hashlib.sha256(zipfile.ZipFile(f'{d}/{v}.npz').read('obs_all.npy')).hexdigest()[:12]
print('dpcc-r', h('dpcc-r')); print('post_processing', h('post_processing')); print('diffuser', h('diffuser'))
"
# post_processing must match NEITHER of the other two.
```

---

## 10. Remaining validation queue

| # | Check | Why it matters |
|---|---|---|
| 1 | Gen0 at **θ=0.5, K=20** | Regression guard — must reproduce pre-fix numbers exactly, confirming no archived result moved |
| 2 | Gen0 at **θ=0.05** | `dpcc-t-tightened` should land ~0.19–0.24 s/step (n_active=2), not 0.53 |
| 3 | **Gen9** `eval_visual_avoiding_dpcc.py` at θ≠0.5 | The second defect-A site, found by audit, still unvalidated |
| 4 | **Gen7** FM visual aligning, `post_processing` | Defect C path — must differ from `diffuser` (bare-float gate previously gave n_active=0 at θ=0) |
| 5 | **Gen14** `fm` arm vs `diffusion`/`mf`/`af` arms | Gate arithmetic now unified across three of four engines |
| 6 | **Gen12** HardFlow Part I/II | Expected unchanged — every threshold used gives integer `(1−T)·K` |

---

## 11. Bottom line

The pre-fix run requested `T = 1` and executed `T = 0.5`, with `post_processing` emitting a byte-exact copy of `dpcc-r`. The post-fix run, same config, same node, same seed, executed `T = 1` with `post_processing` at one final projection — while the projector-free `diffuser` arm stayed bit-identical, ruling out every alternative explanation.

Defects A and B are closed on the Gen0 path. Defects on the FM paths (C) and the Gen9 DPCC path remain to be exercised.
