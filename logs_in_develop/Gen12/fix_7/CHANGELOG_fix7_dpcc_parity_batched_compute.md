# Gen12 fix_7 — DPCC-parity batched compute for the HardFlow sampler

**Note on numbering:** the user asked for this "as Fix6", but `fix_6` already exists
(`CHANGELOG_fix6_dpcc_threshold_polarity.md`, the DPCC-polarity activation gate). That file is
untouched; this new, distinct fix is filed as **fix_7** to avoid clobbering it.

## Problem (from the timing analysis)

`WHY_hardflow_slower_than_dpcc_timing.md` established that HardFlow's ~4× per-step cost vs DPCC is
**not** the NLP (both engines do the same 40 constrained solves per plan at threshold 0.5) but the
**generation loop**: the old `sample()` integrated the candidate fan with a Python
`for b in range(batch_size)` loop, calling the U-Net at **batch-1** and doing a **CPU↔GPU round-trip
on every velocity eval** (`v_dof.cpu().numpy()`). Result per plan (K=20, B=4, thr=0.5):

| | old HardFlow | DPCC |
|---|---|---|
| network GPU calls | **120 (batch-1)** | 20 (batch-4) |
| CPU↔GPU transfers | **120** (per eval) | ~20 (per active step, batched) |
| constrained solves | 40 (ipopt, serial) | 40 (scipy, serial) |

The solve count and structure already matched DPCC; the generation did not.

## The question the user posed

> "if same math then same theoretical compute time … ensure same operation / parallel same to dpcc
> code. If there is nothing to fix, it cannot — abort and tell me why."

**It is fixable, and this commit fixes it.** Verified prerequisites before changing anything:

1. **Candidates are independent** — each has its own initial noise `x_init[b]`, its own pinned `s0`,
   and its velocity/NLP depend only on that candidate's own state. No cross-candidate coupling → a
   batched network pass produces per-row-identical outputs (eval-mode GroupNorm, no BatchNorm).
2. **DPCC does NOT parallelize its solver either** — `Projector.project` loops
   `for i in range(batch_size)` doing one scipy SLSQP per candidate (`projection.py:131`). So
   "match DPCC parallelism" means **batched network + serial per-candidate solves**, which HardFlow
   can reproduce exactly. (A truly parallel batched NLP would go *beyond* DPCC, not match it.)
3. **`_velocity` had no other callers** — safe to change its contract.

So there was something to fix, and the target is well-defined: replicate DPCC's operation pattern,
not invent a new one.

## The fix (`flow_matcher_v3_hardflow/sampling/hardflow_projection.py`)

Rewrote `HardFlowSampler._velocity` → `_velocity_batch` and restructured `sample()`:

- **Batched network, on GPU.** `_velocity_batch(X, τ, s0_all, …)` takes `(B, dof)` and runs the
  U-Net once on `(B, H, T)` — **K+n_active calls per plan instead of B·(K+n_active)** — and returns a
  torch tensor (no `.cpu().numpy()`). Mirrors DPCC's `p_sample` on `(B, H, T)`.
- **Flow loop outside, candidate loop only around the solve.** The ODE now integrates `X` as one
  GPU tensor across `k = 0..K-1`. At `active` steps the per-candidate Python loop survives **only**
  around `self.nlp.solve(...)` — the exact analog of DPCC's `project()` inner loop. `set_s0` is
  called per candidate there (same s0 semantics as before).
- **CPU↔GPU transfer only at the NLP boundary.** One transfer out (`X1_ref → numpy`) and one back
  (`X1_proj → torch`) per active step; inactive steps stay entirely on the GPU. This is DPCC's
  transfer pattern.
- **Metrics preserved.** `nfe += batch_size` per batched call (still counts *per-sample* evals, so
  NFE is unchanged); `candidate_costs` (prox) and `candidate_costs_control` accumulate per candidate
  as before; `dof_chains` is stacked once at the end `(B, K+1, dof)` — identical shape/content.

### Post-fix operation count (per plan, K=20, B=4, thr=0.5)

| | HardFlow (fix_7) | DPCC |
|---|---|---|
| network GPU calls | **30 (batch-4)** | 20 (batch-4) |
| CPU↔GPU transfers | **~20** (2 per active step) | ~20 |
| constrained solves | 40 (ipopt, serial) | 40 (scipy, serial) |

HardFlow now matches DPCC's parallelism **operation-for-operation**. The only residual differences
are **inherent to the algorithm, not the implementation**, and therefore cannot be removed without
changing the math:

- **+10 network calls (30 vs 20):** the extra `v_next` eval per active step builds the endpoint
  prediction `x1_ref = x_ref + (1−τ)·v_next` that the NLP projects. This is the flow-matching
  "predict-x1-then-project" step; DPCC's diffusion parameterization projects `x` directly and needs
  no endpoint eval. Keeping it *is* keeping the math.
- **ipopt vs scipy per solve:** in-loop interior-point vs post-hoc SLSQP — the honest price of
  enforcing constraints *during* generation (the capability behind HardFlow's zero-margin safety win).

## Math preservation — what is and isn't identical

- **Identical operations & results (to float tolerance):** each candidate sees the same sequence of
  velocity evaluations and the same per-candidate NLP solve with the same `s0`/`x1_ref`. Batching a
  GroupNorm U-Net in eval mode does not change per-row outputs.
- **Not guaranteed bit-identical:** the ODE arithmetic now runs in GPU float32 (torch) rather than
  the old CPU float32/float64-mixed numpy path, and batched cuBLAS may pick different kernels than
  batch-1. Differences are at the ~1e-6 level and must not change success/violation outcomes; a
  borderline trial could in principle flip. **Re-run on the cluster and diff against the existing
  `K20_thres0.5_mpc4_n2` seed data to confirm parity** (esp. `n_success_and_constraints`,
  `total_violations`, and `candidate_costs` ranking so `-c` selection is unchanged).

## Expected effect

- **avg_time:** should drop substantially — generation was ~75% of the old per-step budget and its
  GPU-call count falls 4× (120→30). Estimated landing ~1.5–2× DPCC (down from ~4×), the remainder
  being the extra `v_next` evals and ipopt vs scipy. **This is the number to check post-run.**
- **All other metrics (`n_success`, `succ+con`, violations, nlp_solves, nfe):** expected unchanged
  within float32 tolerance.

## Validation status & how to run

- **Local:** `python3 -m py_compile flow_matcher_v3_hardflow/sampling/hardflow_projection.py` passes.
  No torch/casadi in this container, so no numerical run is possible here — **must be validated on
  the cluster.**
- **No config / sbatch / API change.** `HardFlowPolicy` → `sample()` signature is unchanged, so the
  eval driver, YAML, and `Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh` are untouched.
  Re-run the same command; it writes the same `K20_thres0.5_mpc4_n2` folder (use `FORCE_OVERWRITE=1`
  to overwrite the existing seed data for a clean before/after timing diff):

  ```bash
  FORCE_OVERWRITE=1 HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 HFFM_FLOW_STEPS="20" \
    ./submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
  ```

  Compare `avg_time` (should fall) and `n_success_and_constraints` / `total_violations` (should hold)
  against the pre-fix numbers in `ANALYSIS_U5_mpc4_full_run_2707.md`.
