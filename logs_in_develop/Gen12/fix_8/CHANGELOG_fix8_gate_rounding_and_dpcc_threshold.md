# Gen12 fix_8 — gate rounding parity (ceil→floor) + restore the orphaned DPCC threshold

**Dev flag:** every edit is tagged **`[Gen12fix8]`** in-code. Retrieve with:

```bash
grep -rn "Gen12fix8" --include=*.py .
```

**Numbering note (index cleanup).** `fix_6` = threshold *polarity*; `fix_7` = DPCC-parity *batched
compute*; **`fix_8` = this** (gate *rounding* + DPCC threshold wiring). Both earlier folders are
untouched. The diagnosis that led here is
[`ANALYSIS_fix8_gate_floor_vs_ceil_blast_radius.md`](./ANALYSIS_fix8_gate_floor_vs_ceil_blast_radius.md).

**Status:** code changed, `py_compile` clean, logic verified locally by pure-Python simulation of both
gates. **No cluster run yet** — no GPU/torch/casadi in this container.

---

## Two independent fixes

### 8a — HardFlow activation gate: `ceil` → `floor` (3 files)

fix_6 matched DPCC's *polarity* but not its *rounding*. DPCC truncates the boundary with `int()`
(floor); HardFlow compared against the raw float (ceil). Result: **HardFlow did one FEWER projection
step than DPCC — and than upstream HardFlow — whenever `(1−θ)·K` is not an integer.**

```diff
- active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)
+ active = (k >= int((1.0 - self.activation_threshold) * K)) or (k == K - 1)
```

Reference implementations, both **floor**:
- DPCC: `int((1.0 - threshold) * flow_steps_v3)` — `flow_matcher_v3_ode_selectable/models/diffusion.py:207`
- upstream HardFlow: `if k < self.oc_N_steps // 2` — `aux_repo/HardFlow/hardflow/models_flow/flow_policy.py:868`

### 8b — DPCC `diffusion_timestep_threshold` was orphaned config (1 file)

`config/hardflow_projection_eval.yaml` carries `diffusion_timestep_threshold: 0.5` (copied verbatim
from `config/projection_eval.yaml`), but **`eval_FM_v3_hardflow.py` never read it** and never passed it
to `Projector`, so arms A/B silently used `Projector`'s hardcoded default of `0.5`. The YAML knob did
nothing. The FMv3ODE sibling wires it correctly
(`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py:54` → `Projector(..., :242)`);
Gen12's port dropped both lines.

Harmless to date (YAML value `0.5` == the default), but it **blocked any θ≠0.5 DPCC sweep** — i.e. the
entire Test_NFE θ=0.1 plan.

---

## Files touched (4)

| # | file | change |
|---|---|---|
| 1 | `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | 8a: gate `int()`; comment block |
| 2 | `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` | 8a: same, synced (Gen3v6) |
| 3 | `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` | 8a: same, synced (Gen3v7) |
| 4 | `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | 8b: read `dpcc_threshold`, pass to `Projector`, record in npz |

**Sync note:** 1–3 are copy-modify siblings sharing a byte-identical gate line; all three were fixed
together per the repo's cross-generation sync convention. **Gen3v4 (iMeanFlow) and Gen14
(`mix_visual_aligning`) have no HardFlow sampler and were correctly left alone.**

### File 4 detail — three edits

```python
# a) resolve the threshold (next to the other arm knobs)
dpcc_threshold = float(os.environ.get('DPCC_THRESHOLD',
                                      config.get('diffusion_timestep_threshold', 0.5)))

# b) pass it through
projector = Projector(..., solver='scipy', diffusion_timestep_threshold=dpcc_threshold)

# c) record it, so a run can never be silently mislabeled
np.savez(npz_path, ..., activation_threshold=hf_act_threshold, dpcc_threshold=dpcc_threshold, ...)
```

Edit (c) exists because `hf_paths.eval_name()` encodes **only** the HF activation threshold in the
results dir name. Now that DPCC's threshold is independently settable, a folder called
`K20_thres0.1_...` could otherwise contain arms A/B that actually ran at 0.5. **Keep the two equal
unless you deliberately want them to differ.**

New env override **`DPCC_THRESHOLD`**, mirroring `HFFM_ACT_THRESHOLD` on the arm-C side. Precedence:
`DPCC_THRESHOLD` env → YAML `diffusion_timestep_threshold` → `0.5`.

---

## Verification (local, pure-Python — no torch/casadi in this container)

```
py_compile:  all 4 files OK

Parity, K=1..20 x θ=.05..1.00 (400 combos):
  HF vs DPCC mismatches BEFORE : 283 / 400
  HF vs DPCC mismatches AFTER  :   0 / 400      <- exact three-way parity

Regression — every (K, θ) with results on disk:
  K=20 θ=0.5 : 10 -> 10   UNCHANGED
  K=20 θ=0.0 :  1 ->  1   UNCHANGED
  K=10 θ=0.0 :  1 ->  1   UNCHANGED
  K= 5 θ=0.0 :  1 ->  1   UNCHANGED
  K= 2 θ=0.0 :  1 ->  1   UNCHANGED
```

**278/400 grid points change behaviour**, but **0 of the 5 configurations ever actually run** do — every
existing Gen12/v6/v7 result sits on an integer boundary and **reproduces bit-identically**. Nothing to
re-run; no fix_7 timing conclusion is affected.

8b is a no-op at the current YAML value (`0.5` == the old hardcoded default), so it too reproduces
existing runs exactly.

---

## What this unblocks

The Test_NFE equal-cost plan
([`../Test_NFE/PLAN_hardflow_vs_dpcc_equal_cost_test.md`](../Test_NFE/PLAN_hardflow_vs_dpcc_equal_cost_test.md)):

```bash
# θ=0.1 on BOTH arms (edit YAML to diffusion_timestep_threshold: 0.1, or use the env var)
HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.1 DPCC_THRESHOLD=0.1 HFFM_FLOW_STEPS="20" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
```

Writes a **new** dir `K20_thres0.1_mpc4_n2` — existing `K20_thres0.5_mpc4_n2` data is untouched, and
no `FORCE_OVERWRITE` is needed.

**Cluster validation still required** (per the repo's no-Python-locally rule): confirm the θ=0.5
re-run reproduces the fix_7 seed-6 numbers before trusting the θ=0.1 arm.

---

## Residual / not done

- **`hf_paths.eval_name()` still encodes only the HF threshold.** Deliberately not changed — altering
  it would rename every existing results directory. Mitigated by recording `dpcc_threshold` in the npz
  (edit c). Revisit only if a run intentionally uses two different thresholds.
- **No cluster run performed.**
- Gen3v6/v7 (files 2–3) received the gate fix but were **not** otherwise touched or re-validated;
  their HardFlow arms inherit the same integer-boundary guidance.
