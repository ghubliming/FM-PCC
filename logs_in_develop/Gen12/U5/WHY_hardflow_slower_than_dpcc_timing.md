# Gen12 U5 — why is HardFlow ~4× slower than DPCC per step, when both use threshold 0.5?

**Observed** (`K20_thres0.5_mpc4_n2`, ms/step): diffuser 0.184 · dpcc-c-tightened **0.477** ·
hardflow arms **1.80–1.90**. The user's instinct — "threshold is 0.5 for both, so this feels
weird" — is exactly right to be suspicious: **the threshold makes the two engines do the *same
number of solves per plan*. It does nothing about the generation loop, and that is where the 4× lives.**

TL;DR: **the NLP is NOT the main cost.** The cost is that HardFlow integrates the ODE **one
candidate at a time, batch-1, with a CPU↔GPU round-trip on every network eval**, whereas DPCC
integrates **all candidates in one batched GPU pass** and only touches the CPU to project.

---

## 1. What "threshold 0.5" actually controls (identical for both)

Both gate on the *same* rule — project/solve on the late half of the K=20 flow steps:

- DPCC: `near_end = loop_idx >= (1 - 0.5)*K` → steps 10..19 (`diffusion.py:178`).
- HardFlow: `active = (k >= (1 - 0.5)*K) or k==K-1` → steps 10..19 (`hardflow_projection.py:442`).

→ **10 active steps** each. With batch=4, both perform **10 × 4 = 40 constrained solves per plan**.
Confirmed in the log: hardflow `NLP solves = 5760` over the run = 40 × 144 plans. DPCC runs the
*same* 40 scipy projections/plan — they just aren't reported by the hardflow-only `NLP solves`
counter (its log line reads `NLP solves=0`). **Solve count is a tie.** So the threshold is not the
lever; it equalizes the one thing that *is* equal.

## 2. Per-plan operation count (K=20, batch=4, thr=0.5)

| per plan (one env step) | **DPCC** (post-hoc) | **HardFlow** (in-loop) | code |
|---|---|---|---|
| constrained solves | 40 (scipy SLSQP) | 40 (ipopt) | `projection.py:131` / `hardflow_projection.py:453` |
| **network forward passes** | **20, each batch-4** | **120, each batch-1** | `diffusion.py:173-184` / `hardflow_projection.py:429-447` |
| sample-evals counted (NFE) | 80 | 120 | — |
| **CPU↔GPU transfers** | **10** (one per active step, batched) | **120** (one per velocity eval) | `diffusion.py:190` / `hardflow_projection.py:388` |
| solver graph builds | n/a | 1 (reused, warm-started) | `hardflow_projection.py:147,306` |

Two structural asymmetries jump out — **neither is affected by the threshold**:

**(a) Un-batched, serialized generation.** DPCC's `p_sample_loop` runs the ODE on a tensor of
shape `(batch=4, H, T)`: **20 GPU calls total**, each doing all 4 candidates at once
(`diffusion.py:173`). HardFlow's `sample()` has an outer `for b in range(batch_size)` and inside
calls `_velocity` on a **batch-1** reshape (`hardflow_projection.py:415, 383`): **120 GPU calls**,
one candidate at a time. Kernel-launch + Python-dispatch overhead scales with the **number of
calls**, not the number of samples — so HardFlow pays **~6× the launch overhead** (120 vs 20)
even though it only evaluates 1.5× more samples.

**(b) A CPU↔GPU round-trip on every single velocity eval.** Because the NLP is a **CPU**
casadi/ipopt solver, `_velocity` ends with `v_dof.cpu().numpy()` (`hardflow_projection.py:388`) —
a GPU→CPU sync **120 times per plan**. DPCC keeps the whole ODE on the GPU as a torch tensor and
only drops to numpy inside `project()` — **10 times per plan**, batched. Per-step syncs serialize
the GPU and are pure overhead.

**(c) HardFlow also does ~50% more network evals.** At each active step it needs an *extra*
velocity eval `v_next` to build the endpoint prediction `x1_ref = x_ref + (1-τ)·v_next` that the
NLP projects (`hardflow_projection.py:445-447`). So each candidate is 20 + 10 = **30 evals**,
×4 = 120. DPCC projects `x` directly and needs no endpoint eval → 20 batched evals.

## 3. The math: attributing the 1.85 ms

Use the three arms as a natural decomposition (all batch-4, same checkpoint, same 40 active-solve
budget where applicable):

```
diffuser         0.184  = batched generation only          (20 batched evals, no solve)
dpcc-c-tightened 0.477  = batched generation + 40 scipy     => 40 scipy projections ≈ 0.29 ms
hardflow_new-*   ~1.85  = UN-batched generation + 40 ipopt
```

- The **40 constrained solves** cost DPCC ≈ **0.29 ms** (0.477 − 0.184). Even if ipopt is ~1.5–2×
  heavier than SLSQP on this small dense problem, the 40 solves account for at most ~0.3–0.6 ms of
  HardFlow's budget.
- That leaves **~1.25–1.55 ms** — the majority of HardFlow's per-step time — for **generation
  alone**: the 120 batch-1 GPU calls + 120 CPU syncs + the extra `v_next` evals. HardFlow's
  generation is therefore **~7–8× more expensive than DPCC's batched generation** (≈1.4 ms vs
  0.18 ms), and it, not the NLP, is the dominant term.

So the breakdown of a HardFlow step is roughly: **~75% serialized/CPU-synced generation, ~15–30%
NLP, the rest bookkeeping.** The solver everyone assumes is the villain is the *smaller* half.

## 4. Why the threshold intuition misleads

Raising/lowering the threshold changes **how many of the 10 late steps are active** → it scales
the **solve count** and the **extra `v_next` evals**, which per §3 is only ~15–30% of HardFlow's
time. It has **zero** effect on the batch-1 serialization and the per-eval CPU sync (those happen
on *every* step, active or not). That is why two engines at the identical threshold 0.5 still
differ 4×: the threshold governs the part that's already equal (40 solves), and leaves untouched
the part that isn't (un-batched generation).

Corollary: dropping HardFlow's threshold to 0.0 (terminal-only NLP) would **not** make it
DPCC-fast — it would still run 120 batch-1 GPU calls with 120 CPU syncs. It would only shave the
~0.3–0.5 ms of solves, landing around ~1.4 ms, still ~3× DPCC.

## 5. Is it fundamental? No — it's an implementation artifact

The 4× is **not** intrinsic to in-loop constrained sampling; it's how this port is wired:

1. **Batch the candidate fan.** Run the ODE on `(batch, H, T)` like DPCC instead of a Python
   `for b` loop → collapses 120 GPU calls back toward 20. (Needs a batched NLP or a per-candidate
   CPU thread pool for the solve.)
2. **Keep the ODE on GPU.** Only transfer to numpy at the active steps that actually solve, not on
   every velocity eval → 120 syncs → ~40 (or fewer if the solve is batched).
3. **Avoid the redundant `v_next`.** Cache/reuse the velocity across the ref/endpoint construction
   where the schedule allows.
4. **Warm-start harder / cap ipopt iters.** The graph is already reused and warm-started
   (`set_initial`, `:306`); a tighter `solve_limited` iteration cap trades a little feasibility
   margin for time.

Items 1–2 target the ~75% that dominates and would likely bring HardFlow within ~1.5–2× of DPCC.
The remaining gap (a real interior-point solve per active step vs a single post-hoc projection) is
the honest, irreducible price of enforcing constraints *during* generation rather than *after* —
which is exactly the capability that buys HardFlow its zero-margin safety advantage (see the
non-tightened §3 comparison in `ANALYSIS_U5_mpc4_full_run_2707.md`).

---

### One-line answer
Both engines solve 40 constraints per plan (threshold 0.5 guarantees that), so the NLP isn't the
difference — HardFlow is 4× slower because it integrates the ODE **one candidate at a time on the
GPU with a CPU round-trip every step**, while DPCC integrates **all four candidates in one batched
GPU pass**. Generation, not the solver, is ~75% of HardFlow's per-step cost, and it's fixable by
batching.
