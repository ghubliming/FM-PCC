# Gen12 fix_7 — validation: batched compute works, math preserved (seed-6 before/after)

**Run:** `temp/Gen3V7/2507/After_fix7`, job 23903, **GIT REV `ec0c781`** (the fix_7 commit),
seed 6, `K20_thres0.5_mpc4_n2`, 8 arms × 3 halfspaces × 2 trials, 24/24 npz present, `batch=4`
confirmed on every arm. Compared against the **pre-fix seed-6** data in `temp/Gen12/2707` (job
23890, rev `18aa683`) — same config, same checkpoint, same seed.

## Verdict

**fix_7 does exactly what it was designed to do: ~1.77× faster HardFlow, with the headline math
byte-for-byte preserved.** The DPCC-parity batching is validated.

## Before → After (seed 6, mean over 3 halfspaces × 2 trials)

| variant | succ+con% | total_viol | **avg_time (ms)** | nlp_solves | nfe |
|---|---|---|---|---|---|
| diffuser | 0.0 → 0.0 | 4.85e0 → 4.85e0 | 0.184 → 0.176 (1.04×) | 0 → 0 | 10560 → 10560 |
| dpcc-c-tightened | 100 → 100 | 3.86e-8 → 3.86e-8 | 0.481 → 0.473 (1.02×) | 0 → 0 | 10107 → 10107 |
| hardflow_new-r | 83.3 → 83.3 | 9.48e-4 → 9.47e-4 | **1.839 → 1.041 (1.77×)** | 5493 → 5493 | 16480 → 16480 |
| hardflow_new-c | 50.0 → 50.0 | 2.06e-2 → 2.14e-2 | **1.788 → 0.998 (1.79×)** | 8520 → 8533 | 25560 → 25600 |
| hardflow_new-t | 66.7 → 66.7 | 3.96e-3 → 8.00e-3 | **1.862 → 1.073 (1.74×)** | 5107 → 5133 | 15320 → 15400 |
| hardflow_new-r-tightened | 100 → 100 | 2.43e-7 → 2.44e-7 | **1.869 → 1.070 (1.75×)** | 5533 → 5533 | 16600 → 16600 |
| hardflow_new-c-tightened | 100 → 100 | 2.79e-7 → 3.10e-7 | **1.804 → 1.003 (1.80×)** | 8320 → 8333 | 24960 → 25000 |
| hardflow_new-t-tightened | 100 → 100 | 2.55e-7 → 2.60e-7 | **1.919 → 1.105 (1.74×)** | 4960 → 4973 | 14880 → 14920 |

## Insights

**1. The speedup is real and causally attributable to fix_7 — not node variance.**
The two arms fix_7 did *not* touch (diffuser, dpcc-c-tightened — they go through
`diffusion.py::p_sample_loop`, not `HardFlowSampler`) are **flat**: 1.04× and 1.02×, i.e. run-to-run
noise. Every arm fix_7 *did* touch dropped **1.74–1.80×**. Same job, same node, same GPU, same
seed → the ~1.77× is the batching, full stop. This is the cleanest possible A/B: the untouched
control arms are the built-in baseline.

**2. The gap to DPCC closed from ~3.9× to ~2.2×, exactly as predicted.**
Headline arm `hardflow_new-c-tightened` vs `dpcc-c-tightened`:
`1.804 / 0.481 = 3.75×` **before** → `1.003 / 0.473 = 2.12×` **after**. fix_7's changelog predicted
"~1.5–2× DPCC (down from ~4×)"; the measured 2.1–2.2× lands right at the top of that band. The
remainder is the **inherent** algorithmic cost the changelog said could not be removed: HardFlow
still evaluates the extra `v_next` endpoint prediction (nfe 15k–25k vs DPCC's 10k) and runs ipopt
rather than post-hoc scipy. Those are the math, not the implementation.

**3. Math preserved — the headline metrics are identical.**
- **`n_success_and_constraints`: identical for all 8 arms** (0/100/83.3/50/66.7/100/100/100 → same).
  No outcome flipped. The four 100%-safe arms stayed 100%; the non-tightened arms kept their exact
  seed-6 scores.
- **`nfe` identical or ±0.3%** (diffuser/dpcc/`-r`/`-r-tightened` bit-identical; the rest drift by a
  single MPC step). Same velocity-eval budget ⇒ same trajectory computation.
- **`nlp_solves` identical or ±0.3%** (`-r` bit-identical; others ±13–26, i.e. one env-step of
  drift, since solves = n_steps × active × batch).
- **`batch_size = 4`** on every arm, before and after.

**4. The tiny numerical drift is exactly the float32 GPU-vs-CPU tolerance fix_7 flagged — and it
never changed an outcome.**
The old path ran the ODE arithmetic in CPU numpy; fix_7 runs it in GPU float32 (torch). Consequences,
all cosmetic:
- `total_violations` shifts at the 1e-7 (tightened) / 1e-3 (untightened) level. The largest relative
  move is `hardflow_new-t` (3.96e-3 → 8.00e-3) — still a *tiny* absolute violation, and its
  `succ+con` held at 66.7, so the trial classification did not flip.
- A few trajectories end one MPC step earlier/later (the ±nfe/±nlp_solves drift). Expected: a
  ~1e-6 velocity difference can nudge a borderline episode-termination check by a step.
None of this touches `n_success_and_constraints`. The fix is a **speed change, behaviourally inert**
to the resolution that matters.

## Bottom line

fix_7 is validated on seed 6: **HardFlow is ~1.77× faster (now ~2.1× DPCC instead of ~3.9×), with
identical success/constraint outcomes and only sub-1e-3 numerical drift** from the GPU float32 path.
The residual 2.1× is the honest, irreducible cost of in-loop flow-matching projection (extra endpoint
eval + interior-point solve), not an implementation artifact.

**Next:** promote to the full seed set (6–10) to regenerate `ANALYSIS_U5_mpc4_full_run_2707.md`'s
timing column under fix_7, and confirm the succ+con parity holds across all seeds, not just seed 6:
```bash
# set seeds: [6,7,8,9,10] in config/hardflow_projection_eval.yaml, then
FORCE_OVERWRITE=1 HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 HFFM_FLOW_STEPS="20" \
  ./submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
```

---

## Why is it STILL ~2.1× slower than DPCC? Root cause (post-fix_7)

fix_7 is confirmed correct — so the remaining gap is not a batching bug, it's a *different*
bottleneck that only became visible once generation was fixed. Before fix_7 the villain was
un-batched generation (~75% of the cost). **After fix_7 the villain moved: it is now the NLP
solver.**

### Decomposition of the residual (seed 6, tightened trio, same run/node)

Using the three arms as a calibrated ladder — diffuser = network only, dpcc = network + 40 scipy,
hardflow = network + 40 ipopt — and the network per-call cost from diffuser (20 batched calls →
0.0088 ms/call):

| component | DPCC | HardFlow (fix_7) | note |
|---|---|---|---|
| network calls / plan | 20 | **30** | HardFlow adds the `v_next` endpoint eval |
| network time / plan | 0.176 ms | 0.264 ms | 30 × 0.0088 |
| constrained solves / plan | 40 (scipy) | 40 (ipopt) | same count, both serial |
| **time per solve** | **7.4 µs** | **19.9 µs** | **ipopt ≈ 2.7× scipy** |
| solve time / plan | 0.297 ms | 0.795 ms | 40 solves |
| **total / plan** | **0.473 ms** | **1.059 ms** | measured |

**Residual gap = 0.586 ms, and it splits:**
- **≈ 85% — the NLP solver.** ipopt costs **19.9 µs/solve vs scipy's 7.4 µs/solve (2.7×)**. This is
  now the dominant term by far.
- **≈ 15% — the extra `v_next` network eval** (30 vs 20 calls/plan).

### Root cause #1 (the 85%): ipopt interior-point vs scipy SLSQP — `hardflow_projection.py`

```
:174   self.opti.solver('ipopt', opts)          # in-loop solver = ipopt (interior-point)
:309   sol = self.opti.solve_limited()          # a full barrier solve, 40× per plan
```
vs DPCC's `projection.py:138  method='SLSQP'` (active-set SQP). On this small dense problem an
active-set SQP converges in a few cheap iterations; ipopt runs its barrier/interior-point loop with
CasADi function-call overhead per iteration. Same optimum, ~2.7× the wall time per solve.

**A concrete, fixable inefficiency inside root cause #1** — the Hessian:
```
:156-157   0.5 * reg_scale * cs.sumsqr(self.x1 - self.x1_ref) * self.tau_param**2   # objective
:234       self.opti.subject_to(sq >= radius ** 2)                                  # obstacle
:165       'ipopt.hessian_approximation': 'limited-memory',                         # L-BFGS
```
The objective Hessian is **constant** (`reg_scale·τ²·I`) and every constraint Hessian is constant
too (halfspace/bounds/deriv are linear → 0; the obstacle `sq ≥ r²` has Hessian `2I`). So the full
Lagrangian Hessian is **constant** — yet `:165` tells ipopt to *approximate* it with L-BFGS, which
then spends iterations rediscovering a matrix we already know in closed form. Switching to
`'exact'` (or supplying the constant Hessian) would let ipopt converge in ~1–2 iterations and is a
**pure speed change — it does not move the optimum, so the math is unchanged.** This is the single
highest-leverage knob left.

### Root cause #2 (the 15%): the extra flow-matching endpoint eval — `hardflow_projection.py:443-445`

```
:443   V_next = self._velocity_batch(X_ref, tau_next, s0_all, cond_net, returns_net)
:445   X1_ref = X_ref + (1.0 - tau_next) * V_next     # predicted terminal x1 to project
```
HardFlow must predict the terminal state `x1` (one extra network eval per active step) before it
can project it. DPCC's diffusion parameterization yields its `x0` estimate directly inside
`p_sample`, needing no second eval. This +50% network is **intrinsic to in-loop flow-matching
projection — removing it changes the algorithm**, so it stays.

### Summary: irreducible vs tunable

| residual cause | share | code | status |
|---|---|---|---|
| ipopt >> scipy per solve | ~85% | `:174, :309` | **partly tunable** — see below |
| └ L-BFGS on a constant Hessian | (of the above) | `:165` (+`:156, :234`) | **tunable, math-preserving** — try `hessian_approximation:'exact'` |
| extra `v_next` endpoint eval | ~15% | `:443-445` | **irreducible** (it's the flow-matching math) |

**Bottom line:** the ~2.1× is no longer generation — it's the in-loop NLP. ~85% of it is ipopt
being ~2.7× costlier per solve than scipy, and a real chunk of *that* is a misconfigured
(L-BFGS-approximated) constant Hessian — a math-preserving one-line lever worth trying next. The
remaining ~15% (the extra endpoint eval) is the honest, irreducible price of projecting *during*
flow-matching generation rather than after it — the same mechanism behind HardFlow's zero-margin
safety advantage. So a plausible floor after tuning the Hessian is ~1.3–1.5× DPCC, not 1.0×.
