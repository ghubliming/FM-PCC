# Gen12 U4 — PLAN: threshold-gated (late-only) NLP projection

**Date:** 2026-07-25 · **Status:** plan only, **no code written**
**Motivation:** the K-sweep (fix_3 RESULTS) showed arm C (`hardflow_new`) solving an NLP at **every**
ODE step is (a) the slowest arm and (b) *fails at low K*. The user's proposal — solve the NLP only in
the **later** steps, like DPCC's `diffusion_timestep_threshold=0.5` — is examined here against the
HardFlow paper (arXiv 2511.08425v3) and the aux-repo code.
**Verdict: ✅ mathematically sound and explicitly endorsed by the paper.** This plan says why, and how
to build it.

---

## 1. The question

Currently arm C runs `activation: all` → one prox-NLP per ODE step (K solves per plan). Per fix_3:
- **cost:** ~2–3× slower than DPCC (arm B) at every K;
- **low-K failure:** at K=2/5 arm C fails (never reaches goal on `top-right`, or violates on `both`),
  while arm B is 100%/100% from K=2 up.

The user's idea: **don't solve the NLP every step — activate it only in the last fraction of the ODE**
(e.g. τ ≥ 0.5), exactly as DPCC applies its projection only near the end (`diffusion_timestep_threshold`).

## 2. Verdict from the paper — this is sound, and recommended

### 2.1 The safety guarantee is a TERMINAL property (Prop. "safety_guarantee")

HardFlow's feasibility guarantee (`h(x_N) ≤ 0`) is proved **from the final step alone**. At `t_N = 1`
the affine scheduler has `α₁ = 1, β₁ = 0`, so the state update collapses to

```
x_N = α₁ · x̂_N*  +  β₁ · (…)  =  x̂_N*        (paper App. proof of Prop. safety_guarantee)
```

and since the final NLP returns `h(x̂_N*) ≤ 0`, we get `h(x_N) ≤ 0`. **The intermediate steps
contribute nothing to the guarantee.** Formally, `M_{t_N}^θ(x_N) = x_N` — the posterior estimator is
*exact* at the terminal time regardless of model quality (paper §"Feasibility, Stability, Efficiency").

**Consequence:** skipping the NLP on early steps does **not** weaken the terminal safety guarantee —
*provided the final step's NLP is still solved.* This is the one hard invariant of the upgrade (§5.2).

### 2.2 The paper explicitly recommends skipping early steps

Verbatim (paper, App. "Feasibility, Stability, and Efficiency"):

> *"Empirically, we find that it is not necessary to solve the constrained optimization problem at
> every sampling step. In the early stages of sampling, both the posterior estimator and the
> fixed-point approximation are less accurate, so optimization at those stages is often unnecessary.
> Instead, we can skip the early steps and activate constrained optimization only in the later stages,
> which can achieve a good balance between efficiency and performance."*

So the user's proposal is not a hack — it is the paper's own recommended efficiency mode.

### 2.3 It also aligns with the paper's core "minimal-intervention" principle

The paper's central design principle (App. "Minimal-Intervention") is to perturb the pretrained field
**as little as possible**. Solving the NLP every step over-intervenes on transient states that are
"algorithmic artifacts" (paper Intro). The regularizer already carries an `α_{t_{i+1}}²` weight
(= τ² for a linear scheduler — the same τ² factor Gen12 fix_2 identified), which *down-weights* early
intervention. Late-only activation is the discrete, harder version of the same idea: intervene only
where it matters.

## 3. Why this should ALSO fix the low-K failure (not just speed)

fix_3's mechanistic finding: at low K, arm C produces feasible-but-wrong plans because
`x̂₁ = x_ref + (1−τ)·v` — the terminal prediction — is built from a 2–5-step Euler integration of the
field and is **inaccurate at small τ**. Projecting onto a bad terminal estimate early, then pulling the
whole trajectory toward it, corrupts the plan.

The paper names this exact failure mode: *"In the early stages … the posterior estimator and the
fixed-point approximation are less accurate."* So **early projections are not just wasted compute —
they are actively harmful when the field is coarse.** Removing them should:

- **improve** low-K success (stop corrupting the plan with bad early projections), and
- **reduce** cost (fewer NLP solves),

simultaneously. At the extreme (activate only the final step), arm C degenerates to *post-hoc
projection of the terminal trajectory* — essentially arm B, which fix_3 showed is robust at every K.
So a threshold sweep **interpolates arm C (all steps) ↔ arm B (terminal only)** and should recover
B-like robustness at low K while keeping C's in-loop steering at high K.

## 4. What already exists vs. what to build

Gen12 **already** has a binary switch (`flow_matcher_v3_hardflow/sampling/hardflow_projection.py`):

```python
active = self.activation == 'all' or k >= K // 2        # 'late' = last half
```

Gaps:
1. It is **binary** (`all` / `late`), hardcoded at `K//2`, not a **continuous threshold** like DPCC's
   `diffusion_timestep_threshold` (0.5). No way to sweep τ_act ∈ {0.0, 0.3, 0.5, 0.7, 0.9}.
2. It has **no explicit final-step guard** — it works today only because `k=K−1 ≥ K//2` for K≥2, but a
   general threshold (e.g. 0.9 at K=2) could otherwise skip the terminal solve and silently void the
   safety guarantee (§2.1). The guard must be explicit.
3. We only ever ran `activation: all`. The `late` path is untested.

## 5. The upgrade

### 5.1 A continuous activation threshold (DPCC-analogous)

Replace the binary flag with `hardflow_activation_threshold ∈ [0, 1)`:

```
solve NLP at step k  ⇔  τ_{k+1} ≥ hardflow_activation_threshold   OR   k == K-1
```

Semantics, matching DPCC's `diffusion_timestep_threshold`:
- `0.0` → every step (today's `all`; the fix_3 arm C).
- `0.5` → last half of the ODE (τ ≥ 0.5) — the user's DPCC-parity setting.
- `→1.0` → only the terminal step — pure post-hoc terminal projection (≈ arm B, in-loop-free).

Keep `activation: 'all'|'late'` as back-compat aliases mapping to thresholds 0.0 / 0.5.

### 5.2 🔴 The one invariant — the final step is ALWAYS active

The `OR k == K-1` clause is **not optional**. Prop. safety_guarantee holds only if the terminal NLP is
solved. A gate (§6) must assert: for any threshold and any K, step `k=K−1` is active and the returned
`x_N` equals the projected terminal `x̂_N*` (so `h(x_N) ≤ 0`). Without this, a high threshold at low K
would produce an unguided sample with no feasibility guarantee.

### 5.3 Config wiring

- Add `hardflow.activation_threshold` to `config/hardflow_projection_eval.yaml` (arm-C block), default
  `0.5` (the user's proposal / DPCC parity). `activation: all|late` stays as alias.
- Thread it through `HardFlowPolicy` → `HardFlowSampler` (replace the `k >= K//2` line).
- Record the threshold in the results provenance (dir name or npz), so a threshold sweep does not
  collide — e.g. `K{K}_n{n}_act{thr}`.

### 5.4 No change to the NLP or the loading

This is purely a *when-to-solve* gate. The NLP, the FMv3ODE loading, the constraint geometry, and the
pull-back math are untouched. NFE drops because skipped steps still take the reference Euler step but
skip the **second** (terminal-prediction) forward pass and the solve.

## 6. Gates (pre-flight, cluster; extend `gates_hardflow.py`)

- **G4 final-step invariant:** for threshold ∈ {0.0, 0.5, 0.9, 0.99} and K ∈ {2, 5, 10}, assert step
  `k=K−1` is active and the returned terminal is the NLP solution (feasible). This is the §5.2
  safety invariant — the most important new gate.
- **G5 monotone activation count:** number of NLP solves is non-increasing in the threshold, and equals
  the count of steps with τ ≥ thr (plus the forced final one). Confirms the gate does what it says.
- **G6 threshold→1 ≈ post-hoc:** at threshold → 1.0, the swept trajectory (early steps unguided) matches
  a plain unguided ODE up to the final projection — i.e. arm C collapses to terminal-only projection.

## 7. Experiment design

Matched-K, as always (PLAN §5 / Gen13 fix_7). Add the threshold axis:

| axis | values |
|---|---|
| K | {2, 5, 10} |
| activation_threshold | {0.0 (all), 0.5 (DPCC-parity), 0.9, →1.0 (terminal-only)} |
| arms | A `diffuser`, B `dpcc-c-tightened`, C `hardflow_new` (× threshold) |

Report per cell: success (goal), success (goal+constraints), violations, **s/plan**, NFE, **NLP
solves**. Seed 6, raise `n_trials` toward n ≥ 100 (fix_3 §7).

**The questions this answers:**
1. Does `threshold=0.5` **fix the low-K failure** (K=2/5 success → 1.0)? (Expected yes, §3.)
2. Does it **cut cost** toward arm B while staying safe? (Expected yes — fewer solves.)
3. Is there a threshold where arm C **beats** arm B on success-per-second? (The real prize — in-loop
   steering on the *late* steps, at DPCC-like cost.) If not, arm C at best ties B → Gen12 stays a
   clean negative, but now a *cheap* one.

## 8. Traps

1. **Final-step guard (§5.2)** — the whole safety guarantee rides on it. Gate G4 exists for this.
2. **Threshold semantics vs DPCC.** DPCC: project when `loop_idx ≥ (1 − thr)·K`. Gen12 here: solve when
   `τ ≥ thr`. For a uniform grid these coincide at thr=0.5 (last half). State the convention in the
   config comment so the two generations remain comparable.
3. **Provenance.** A threshold sweep writes many result dirs; encode `act{thr}` in the path or the
   npz, or a sweep silently overwrites (PLAN §3.6 lesson, already bitten twice).
4. **Low-K + high threshold = few/one solve.** At K=2, thr=0.5 → only the final step solves. That is
   *intended* (≈ post-hoc projection) and is the robust regime — do not "fix" it back to all-steps.
5. **Don't confuse this with the τ² regularizer weight.** §2.3: the α_t²/τ² factor is a *soft* early
   down-weighting inside the cost; the activation threshold is a *hard* on/off gate. They are
   complementary, not the same knob.

## 9. Success criteria

- **Minimum (correctness):** with `threshold=0.5`, the final-step invariant holds (G4), the terminal
  sample is feasible, and low-K success is **no worse** than `activation: all`.
- **Target (the point of U4):** `threshold=0.5` **fixes** the low-K failure (K=2/5 → 100%/100%) **and**
  cuts arm C's s/plan substantially toward arm B — i.e. same safety, near-DPCC cost.
- **Stretch:** some threshold gives arm C a **success-per-second win** over arm B at some K — the first
  positive result for the contribution. A clean negative (ties B, at now-comparable cost) is still a
  publishable refinement of fix_3.

## 10. Out of scope

- Retraining anything (Gen12 is eval-only).
- The `linear_fit` dynamics mode (still `deriv`).
- iMF/MeanFlow backbones (Gen12 is FMv3ODE-only, fix_1 §8).
- The batch-4 arm-C confound (fix_3 §5) — orthogonal; run it separately.

---

### Appendix — exact references in the paper (arXiv 2511.08425v3)

- **Prop. safety_guarantee** (terminal feasibility) + its proof (App. "Proof of Proposition
  safety_guarantee"): guarantee derives from the final step, `α₁=1, β₁=0 ⇒ x_N = x̂_N*`.
- **App. "Feasibility, Stability, and Efficiency":** explicit recommendation to skip early steps and
  activate only in later stages.
- **App. "Minimal-Intervention Principle":** perturb as little as possible; `α_t²` weight down-weights
  early control.
- **Algorithm (main):** per-step NLP `argmin_{x̂_N} C(x̂_N) + (λ/2Δt)·α²·‖x̂_N − x̄_N‖²  s.t.  h(x̂_N)≤0`.
- aux-repo `HardFlow/hardflow/models_flow/flow_policy.py` implements the same `hardflow_activation
  all|late` binary that Gen12 inherited — U4 generalises it to a continuous threshold with an explicit
  final-step guard.
