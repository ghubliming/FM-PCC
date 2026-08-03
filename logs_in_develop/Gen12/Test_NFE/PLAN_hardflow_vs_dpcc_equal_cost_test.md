# Gen12 Test_NFE — HardFlow vs DPCC at **equal cost**: test plan + cost math (θ=0.1 edge probe)

> **Status: PLAN ONLY — no code changes.** Nothing in `flow_matcher_v3_hardflow/` or
> `FM_v3_hardflow_test/` was modified for this document. Every count below is derived by reading the
> existing code (line refs at the end); the run matrix in §6 uses only existing env-var / YAML knobs.

**Goal.** Compare HardFlow (in-loop NLP) vs DPCC (post-hoc SLSQP) on the **FMv3ODE** checkpoint under a
*controlled compute budget* rather than at a fixed K — and use **θ = 0.1** to probe extreme-edge
behaviour (projection only at the very last steps). Two questions to settle:

1. What is "same NFE"? And is "same NFE" even the right parity — or should it be **same NLP solves**?
2. Is the **naive additive** cost model (`T ≈ net-evals + NLP-solves`, everything else ignored) legit?

---

## 1. The two activation gates — both guard the final step; they differ only by floor-vs-ceil

⚠️ **Which DPCC class actually runs.** Gen12 loads the checkpoint's **own** class
(`target_class=None`, `eval:148`) = **`flow_matcher_v3_ode_selectable/models/diffusion.py::FlowMatchingODE`**
— *not* the local `flow_matcher_v3_hardflow/models/diffusion.py::GaussianDiffusion`. The local file is
dead code for this eval and has a *different, weaker* gate. The gate that runs is:

```python
# DPCC (the one that runs)  flow_matcher_v3_ode_selectable/models/diffusion.py:207-208
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)

# HardFlow                  sampling/hardflow_projection.py:454
active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)
```

**Both have the SAME final-step guard `or (… == K−1)`.** So **neither arm can ever do zero
projections** — the terminal solve always fires, at every K and every θ. Let `θ` = threshold,
`K` = flow steps, `x = (1−θ)·K`:

```
n_active(DPCC) = max( K − floor(x), 1 )      ← int() truncates  ⇒ floor
n_active(HF)   = max( K − ceil(x),  1 )      ← float `>=`        ⇒ ceil
```

### The real difference: **floor vs ceil**, worth exactly one step

`int()` truncates, so **DPCC starts one step EARLIER than HF whenever `x` is not an integer** —
i.e. DPCC does **one extra projection step**. They are identical **iff `(1−θ)·K` is an integer**
(or both are clamped to 1 by the guard). Across a K=1…20 × θ=0.05…1.0 grid, **283 of 400
combinations differ** — always by exactly 1, always DPCC ≥ HF.

| K | θ | `(1−θ)K` | DPCC | HF | |
|---|---|---|---|---|---|
| 20 | 0.5 | 10.0 ✔ int | 10 | 10 | ✅ match |
| 20 | **0.1** | **18.0 ✔ int** | **2** | **2** | ✅ **match — the probe is safe** |
| 20 | 0.35 | `12.999999999999998` | 8 | 7 | ❌ differ — **float trap**: `(1−0.35)·20` is *not* exactly 13 |
| 5 | 0.5 | 2.5 | 3 | 2 | ❌ differ (**50% more solves for DPCC**) |
| 2 | 0.5 | 1.0 ✔ int | 1 | 1 | ✅ match |
| 1 | any | — | 1 | 1 | ✅ match (guard) |

> **Design rule for every arm in this test: choose K and θ so that `(1−θ)·K` is an INTEGER —
> and verify it in float, not on paper.** `(1−0.35)·20` evaluates to `12.999999999999998`, not 13,
> which alone flips DPCC to an extra solve step. Safe values are those where `1−θ` is exactly
> representable (θ ∈ {0.5, 0.25, 0.75, 1.0}) or where the product lands cleanly — **θ=0.1, K=20 gives
> exactly 18.0 ✔** (verified). All arms in §6 check out: 18.0, 10.0, 1.0, 0.0.
>
> Quick check before any new (K, θ):
> ```python
> x = (1.0 - theta) * K
> assert x == int(x), f"(1-θ)K = {x!r} is not integral — gates will differ by one step"
> ```

### Where HF's guard comes from — it is a **Gen12 addition**, not upstream

Upstream HardFlow (`aux_repo/HardFlow/hardflow/models_flow/flow_policy.py:864-873`) has **no
continuous threshold and no explicit final-step guard**. It has a binary switch:

```python
projection_flag = True
if self.cfg.projection_option == "all":      # <- DEFAULT (config/flow_matching.py:78)
    pass                                     #    project at EVERY step
elif self.cfg.projection_option == "late":
    if k < self.oc_N_steps // 2:             #    integer floor division
        projection_flag = False
```

Two things follow:

1. **Upstream's final step is always projected anyway** — `k = N−1 ≥ N//2` holds for all `N ≥ 1`, so
   the guard exists *in effect* without being written. Gen12 made it **explicit** (`:450-454`: *"PLUS
   the forced final step (the terminal solve carries the safety guarantee)"*, tagged U4 + fix_6) and
   generalised `late` (a hardcoded half) into a **continuous θ**. So: **yes, Gen12 HF has the guard,
   DPCC has the guard, and upstream has it implicitly.** No arm can skip the terminal solve.

2. ⚠️ **But upstream uses `N//2` — floor — exactly like DPCC's `int()`. Gen12 HF uses a float `>=`,
   which is ceil.** So at odd K, **Gen12's HF is the odd one out, doing one FEWER projection than both
   upstream HardFlow and DPCC**:

| K | upstream `late` | DPCC θ=0.5 | **Gen12 HF θ=0.5** |
|---|---|---|---|
| 3 | 2 | 2 | **1** ❌ |
| 5 | 3 | 3 | **2** ❌ |
| 7 | 4 | 4 | **3** ❌ |
| 2, 4, 6, 10, 20 | = | = | = ✅ |

This is a **port-fidelity deviation in Gen12's HF gate**, not a DPCC quirk: `int((1−θ)·K)` (floor)
would match both references. It is invisible at even K and at every arm in §6 — **but it means the
`or (k == K−1)` guard is currently doing real work at odd K, masking what would otherwise be a
zero-projection step.** Worth fixing to `int(...)` for exact three-way parity if odd K is ever used;
**not needed for this test** (all arms use K ∈ {1, 2, 10, 20}).

**Correction note:** an earlier draft of this document claimed DPCC does *zero* projections below
θ=1/K and therefore "becomes the diffuser arm". **That was wrong** — it came from reading the unused
local `GaussianDiffusion`, which lacks the guard. The `FlowMatchingODE` that actually runs always
projects at least once, exactly like HardFlow.

---

## 2. Operation counts — the exact accounting

> **Reporting rule for this whole test: NEVER aggregate the two costs into one number.**
> Always report **NN calls** and **NLP solves** as *separate columns*, with their *separate* time
> contributions. They scale differently (NN is batched ≈ flat in `B`; NLP is serial ≈ linear in `B`),
> they respond differently to θ (NN barely moves; NLP moves 5×), and they have different per-unit
> costs per arm (ipopt ≈ 2.7× SLSQP). A single summed "cost" hides every effect this test is designed
> to isolate. The `NFE-eq` scalar in §5 is a *derived convenience only* — never the primary readout.

Per **plan** (one MPC replan), with `K` flow steps, `B` candidates, threshold `θ`:

| | **DPCC** | **HardFlow** |
|---|---|---|
| batched net calls | `K` | `K + θK = K(1+θ)` |
| **per-sample NFE** (the logged metric) | `K·B` | `K(1+θ)·B` |
| constrained solves (serial in B) | `θK·B` | `θK·B` |

**The single structural difference:** HF's extra `v_next` endpoint eval at every active step
(`hardflow_projection.py:456`), which DPCC does not need. Nothing else differs — fix_7 already made
the network batching and the serial-solve loop identical.

### Concrete: K=20, B=4 — costs kept SEPARATE

#### Where the time constants come from (provenance — read this before trusting any ms below)

Source: the **fix_7 seed-6 run**, `temp/Gen3V7/2507/After_fix7/.../K20_thres0.5_mpc4_n2`, reading
`avg_time` out of the 8 arms × 3 halfspaces × 2 trials npz files. **`avg_time` is SECONDS per
`policy()` call** — `avg_time[i] += time.time() - start` around the policy call (`:394-396`) then
`avg_time[i] /= _` over the episode's steps (`:424`), printed as *"Average computation time per
step"* (`:473`). One policy call = one full plan, so **per-step == per-plan**.

Measured means (seconds/plan): `diffuser` **0.1760**, `dpcc-c-tightened` **0.4733**, HF tightened-trio
**1.0595**.

| constant | value | how obtained | status |
|---|---|---|---|
| `a` — per *batched* NN call | **8.8 ms** | `0.1760 s / 20 calls` — diffuser runs K=20 calls and **zero** NLP | ✅ **directly measured** |
| `b_scipy` — per solve | **7.43 ms** | `(0.4733 − 0.1760) / 40` | ⚠️ **back-solved** (assumes the additive model) |
| `b_ipopt` — per solve | **19.89 ms** | `(1.0595 − 30·a) / 40` | ⚠️ **back-solved** (assumes the additive model) |

**Only `a` is an independent measurement.** `b_scipy`/`b_ipopt` are *derived by assuming* the additive
model, so any table built from them **cannot** be used to confirm that model — that is exactly the
circularity §4 sets out to break with the θ=0.1 run. Treat all predicted ms below as **hypotheses to
be tested**, not results. (8.8 ms for a batched U-Net forward is physically sensible, which is a
useful sanity check on `a`.)

| K=20, B=4 | arm | **NN calls** | **NN time** | **NLP solves** | **NLP time** | total | NLP share |
|---|---|---|---|---|---|---|---|
| **θ=0.5** | DPCC | 20 | 176 ms | 40 | 297 ms | **473 ms** *(measured)* | 63% |
| | HF | **30** (+50%) | 264 ms | 40 | **796 ms** | **1060 ms** *(measured)* | **75%** |
| **θ=0.1** | DPCC | 20 | 176 ms | **8** | 59 ms | **235 ms** *(predicted)* | 25% |
| | HF | **22** (+10%) | 194 ms | **8** | **159 ms** | **353 ms** *(predicted)* | 45% |

Per-sample NFE (the logged metric) = NN calls × B: DPCC 80 both thresholds; HF 120 (θ=0.5) → 88 (θ=0.1).

**Read the two columns independently:**
- **NN column:** DPCC is *flat* (20 calls at either θ — the network runs every flow step regardless).
  Only HF's column moves, and only by the `v_next` count (+10 → +2 calls).
- **NLP column:** both arms move together, 40 → 8, a **5× cut**. This is the term θ actually controls.
- The **NLP share** row is the headline: HF at θ=0.5 is a *solver-bound* workload (75%); at θ=0.1 it
  becomes nearly *network-bound* (45%). **θ changes the character of the workload, not just its size.**

**Key insight for the test:** dropping θ 0.5→0.1 cuts solves 5× (40→8) while leaving DPCC's net calls
**completely unchanged** (20 either way) and HF's nearly unchanged (30→22). **θ is a lever that moves
one cost term while pinning the other.** That is exactly what makes the additive model falsifiable
(§4), and it is why this run is worth doing beyond the edge-behaviour question.

---

## 2b. Low-K regime: **K=1 and K=2** — the guard floor

Relevant because Gen13's headline was **FM @ K=2**. Both arms carry the final-step guard (§1), so at
low K the guard — not θ — sets the projection count.

### The guard floor: you cannot buy fewer than one projection step

`n_active ≥ 1` always, for **both** arms. So the **effective** threshold is

```
θ_eff = n_active / K   ≥   1/K
```

| K | min achievable `θ_eff` | consequence |
|---|---|---|
| 20 | **0.05** | θ=0.1 is genuinely reachable ✅ |
| 5 | 0.20 | θ<0.2 all collapse to the same run |
| **2** | **0.50** | **θ = 0.5, 0.3, 0.1 are the SAME run** (all → n_active=1) |
| **1** | **1.00** | **θ is completely inert — always 1 projection** |

**This is the real low-K limitation** (not "DPCC goes unguided", which was the earlier error): at
K=1 the threshold knob does nothing at all, and at K=2 everything at or below 0.5 is one identical
configuration. **To probe a genuinely sparse projection schedule you need large K** — which is
precisely why the θ=0.1 edge probe must run at **K=20**.

### K=2, B=4 — costs kept SEPARATE (all ms **predicted** from `a`/`b`, not measured)

| θ | n_act | arm | **NN calls** | **NN time** | **NLP solves** | **NLP time** | total | HF/DPCC |
|---|---|---|---|---|---|---|---|---|
| **1.0** | 2 / 2 | DPCC | 2 | 17.6 ms | 8 | 59.4 ms | **77.0 ms** | — |
| | | HF | 4 | 35.2 ms | 8 | 159.1 ms | **194.3 ms** | 2.52× |
| **0.5** | 1 / 1 | DPCC | 2 | 17.6 ms | 4 | 29.7 ms | **47.3 ms** | — |
| | | HF | 3 | 26.4 ms | 4 | 79.6 ms | **106.0 ms** | 2.24× |
| **≤0.5** | 1 / 1 | — | — | — | — | — | *identical to the θ=0.5 row* | 2.24× |
| **0.55–0.95** ❌ | **2 / 1** | — | — | — | — | — | *gates DIFFER — do not use* | — |

### K=1, B=4 — θ is inert (n_active = 1 for every θ)

| θ | arm | **NN calls** | **NN time** | **NLP solves** | **NLP time** | total | HF/DPCC |
|---|---|---|---|---|---|---|---|
| **any** | DPCC | 1 | 8.8 ms | 4 | 29.7 ms | **38.5 ms** | — |
| | HF | 2 | 17.6 ms | 4 | 79.6 ms | **97.2 ms** | **2.52×** |

### The key structural result: **the time ratio is independent of K**

Write the split model in terms of the *effective* threshold `θ_eff = n_active/K` (equal for both arms
whenever `(1−θ)K` is an integer, §1):

```
T_DPCC = a·K       + b_s·θ_eff·K·B               HF     a(1+θ_eff) + b_i·θ_eff·B
T_HF   = a·K(1+θ_eff) + b_i·θ_eff·K·B           ──── = ────────────────────────   ← no K !
                                                 DPCC     a + b_s·θ_eff·B
```

**K cancels.** So HF/DPCC = **2.24× at θ_eff=0.5 and 1.50× at θ_eff=0.1 — at K=1, K=2 or K=20 alike**
(the tables confirm: 2.24× at K=2 θ=0.5, matching the 2.24× *measured* at K=20 θ=0.5). Three
consequences:

1. **Shrinking K does *not* close HF's relative gap.** It scales both arms down together. Only
   **θ_eff** or **B** move the ratio. This kills "just run HF at low K to make it competitive".
2. **Low K cannot reach low θ_eff** (guard floor above), so low-K runs are *cheaper*, not *sparser*.
   At K=1, θ_eff is pinned at 1.0 — the **most** projection-heavy regime, worst case for HF (2.52×).
3. **The ratio is set by θ_eff alone**, so the cheap K=1/K=2 arms are a *legitimate* proxy for the
   expensive K=20 arms at the same θ_eff — which is what makes the smoke test below worth running.

> ⚠️ **Consequences 1 and 3 above are REFUTED by real K=2 data — see §2c.** They are kept here
> because the *algebra* is right; what fails is the premise that `b_ipopt` is a constant. Read §2c
> before using anything in this subsection.

---

## 2c. EMPIRICAL TEST of the K-invariance claim — **REFUTED**

**Source:** `temp/0308/plot_n_success_and_constraints_both-hard_20v_2c_20260803_1812_tables.tex`
(Visualizer LaTeX export, batch `batch_avoiding_combined_20260802_092307`, env `both-hard`,
mean ± SEM over all available seeds). Two **K=2, T=0.5** candidates:

- **CAND_32** — Gen3v7 α-Flow (`H8_K2_Meuler_T0.5_…AlphaFlowODE`)
- **CAND_102** — Gen3v6 MeanFlow (`H8_K2_Meuler_T0.5_…MeanFlowODE`)

Both are at `n_active = 1` and `(1−θ)·K = 1.0` (integral) → **fix_8-invariant**, and `θ_eff = 1/2 = 0.5`,
exactly matching the Gen12 K=20 reference's `θ_eff = 10/20 = 0.5`. So this is a like-for-like test of
the K-invariance prediction, on the formula's own terms.

### Measured (avg_time, mean over the -r/-r-tightened/-t/-t-tightened arms)

| run | `a` (per NN call) | DPCC | HF | **HF/DPCC** |
|---|---|---|---|---|
| CAND_32 (α-Flow, K=2) | 6.15 ms | 20.1 ms | 67.5 ms | **3.36×** |
| CAND_102 (MeanFlow, K=2) | 8.30 ms | 24.9 ms | 77.8 ms | **3.13×** |
| **Gen12 FMv3ODE, K=20** | 8.80 ms | 473 ms | 1060 ms | **2.24×** |

**§2b predicted 2.24× at every K. The measured value at K=2 is 3.1–3.4×.** The prediction fails, and
it fails in the *unfavourable* direction: **shrinking K WIDENS HF's relative gap.**

### Where it breaks — the solver term, not the network

Backing out each run's own constants (`a` from its diffuser arm, so the different DiT-vs-U-Net
backbone is *not* a confound), with `u = b_scipy·B`, `v = b_ipopt·B`:

| run | `b_scipy·B` | `b_ipopt·B` | **`b_ipopt / b_scipy`** |
|---|---|---|---|
| CAND_32, K=2 | 7.78 ms | 49.0 ms | **6.30×** |
| CAND_102, K=2 | 8.27 ms | 52.9 ms | **6.39×** |
| Gen12, K=20 | 7.42 ms/solve | 19.90 ms/solve | **2.68×** |

**`B` cancels in the ratio**, so the 6.3× vs 2.68× comparison is free of the unknown-batch confound.

The decisive check — take Gen12's constants and predict CAND_32 using *its own* `a`:

```
DPCC:  2·6.15 + 7.43  = 19.7 ms      measured 19.9 ms    ✅ transfers (1% error)
HF  :  3·6.15 + 19.89 = 38.3 ms      measured 66.7 ms    ❌ off by 1.74×
```

**`b_scipy` is portable across K, model family and batch (7.4 → 7.8 → 8.3 ms). `b_ipopt` is not
(19.9 → 49.0 → 52.9 ms, ~2.5×).** The additive model holds for DPCC and breaks for HardFlow.

### Why: ipopt's cost tracks how far `x1_ref` starts from feasible

`HardFlowNLP.solve` seeds the solve with the *reference*, not the previous solution:

```python
self.opti.set_value(self.x1_ref, x1_ref)
self.opti.set_initial(self.x1, x1_ref)     # cold-ish: initial guess = the unprojected reference
```

So per-solve cost is set by the **quality of `x1_ref`**, which depends on K:

- **K=20, 10 active steps:** by the time each solve runs, the ODE has taken many fine Euler steps
  *and* the previous 9 solves already pushed the trajectory into the feasible set. Each solve starts
  nearly-feasible → few barrier iterations → **~20 ms**.
- **K=2, 1 active step:** a single solve on a raw reference produced by **two** coarse Euler steps,
  with no prior projection to lean on. Far from feasible → many barrier iterations → **~50 ms**.

scipy SLSQP (active-set) is far less sensitive to this, which is why `b_scipy` barely moves. In short:
**consecutive in-loop projections amortise; an isolated one does not.** ipopt's per-solve cost is a
function of the projection *schedule*, not a constant of the solver.

### What this changes

1. ❌ **"Shrinking K does not close HF's gap" (§2b #1) — WRONG in a worse way.** Low K doesn't just
   fail to help; it actively **hurts** HF (2.24× → 3.1–3.4×). "Run HF at low K to look competitive"
   is not merely ineffective, it is **counterproductive**.
2. ❌ **"Cheap K=1/K=2 arms are a legitimate proxy for K=20" (§2b #3) — WRONG.** Arm 0 is still worth
   running, but as its **own** operating point, **not** as a stand-in for the K=20 result.
3. ✅ **The disaggregation rule (§2) is vindicated.** A single summed cost would have hidden this
   entirely: the NN term behaved exactly as predicted, and *only* the NLP term broke. Reporting
   `b_scipy` and `b_ipopt` separately is what made the failure visible and localisable.
4. ⚠️ **The θ=0.1 predictions in §2/§4 (DPCC 235 ms, HF 353 ms) are now suspect on the HF side only.**
   θ=0.1 at K=20 drops HF from 10 active steps to **2** — a large move toward the un-amortised regime.
   Expect HF **above** 353 ms; the DPCC 235 ms prediction should still hold. **This makes the θ=0.1
   run more interesting, not less: it now tests amortisation, not just the additive model.**

### Revised prediction for the θ=0.1 run

| | DPCC | HF |
|---|---|---|
| §4 additive model | **235 ms** (trust) | 353 ms (**lower bound only**) |
| with amortisation (`b_ipopt` → ~40–50 ms at 2 active steps) | 235 ms | **~500–600 ms** |
| implied HF/DPCC | — | **2.1–2.6×** (vs 1.50× naive) |

**Falsifiable either way:** if the measured θ=0.1 HF time lands near 353 ms, `b_ipopt` is constant
after all and the K=2 gap must be explained by something else (model family, node, batch). If it
lands near 500–600 ms, amortisation is confirmed and `b_ipopt` must be modelled as a function of
`n_active`.

### Caveats

- The K=2 rows come from **α-Flow / MeanFlow** checkpoints (Gen3v6/v7, DiT backbone), not FMv3ODE.
  Using each run's own `a` removes the *network*-cost confound, and `b_ipopt/b_scipy` is
  `B`-independent — but a *different checkpoint produces a different `x1_ref` quality*, which is
  precisely the mechanism at issue. **A same-checkpoint K sweep (FMv3ODE at K=2 vs K=20) would settle
  it cleanly** and is the single most valuable follow-up run.
- `B` (mpc candidates) is not recorded in the LaTeX export. It cancels in every ratio quoted above,
  but the absolute `b·B` figures are not directly comparable to Gen12's per-solve numbers.

---

### The minimal exact-parity pair (useful, and Gen13-relevant)

Applying `K_H = K_D(1−θ_D)`, `θ_H = θ_D/(1−θ_D)` (§3) at the smallest usable size:

> **HF @ K=1, θ=1.0  ≡  DPCC @ K=2, θ=0.5** — both **2 NN calls** and **4 NLP solves**. Exact
> double-parity, at the cheapest possible scale (~100 ms/plan vs ~1 s for the K=20 arms). Predicted times **97.2 vs 47.3 ms**.

This is a fast smoke-test of the whole cost model: it costs ~10× less than the K=20 arms, and if the
measured ratio there is **not** ≈2.25× the K-invariance claim above is falsified immediately. **Run
this pair first — it's minutes, not hours.**

---

## 3. What does "same NFE" mean here? (and why NFE-parity ≠ solve-parity)

Your question — *"K20 DPCC projects 10 times at θ=0.5; what's the same-NFE HF?"*

**Equal per-sample NFE** requires `K_H(1+θ_H) = K_D`. At θ=0.5, K_D=20:

```
K_H = 20 / 1.5 = 13.33  →  K_H = 13
```

But then HF's solves = `θ·K_H·B` = 0.5·13·4 = **26**, vs DPCC's **40**. So **matching NFE
under-provisions HF's projection** — you'd be comparing 26 solves against 40 and HF would look
artificially safe-but-cheap. **NFE-parity alone is a misleading budget.**

**Equal solves** requires `θ_H·K_H = θ_D·K_D` — at θ=0.5, K=20 that's just K_H=20, θ_H=0.5, i.e. the
current config, where HF spends 50% more NFE.

### The double-parity point (both matched exactly)

Solve both constraints simultaneously:

```
K_H(1+θ_H) = K_D          (equal net calls / NFE)
θ_H·K_H    = θ_D·K_D      (equal solves)
   ⇒   K_H = K_D·(1 − θ_D)        θ_H = θ_D / (1 − θ_D)
```

| DPCC reference | **HF double-parity twin** | net calls | solves |
|---|---|---|---|
| K=20, θ=0.5 | **K=10, θ=1.0** | 20 = 20 | 40 = 40 |
| K=20, θ=0.1 | **K=18, θ=0.111** | 20 = 20 | 8 = 8 |

**This is the fairest possible comparison** and I'd make it the headline arm: HF at **K=10, θ=1.0**
against DPCC at **K=20, θ=0.5** is *exactly* equal on both cost axes — same 20 batched net calls, same
40 solves. Any remaining time gap is then **pure per-operation cost** (ipopt vs SLSQP), with zero
budget confound. (Valid only for `θ_D ≤ 0.5`, since `θ_H ≤ 1`.)

Note K=10/θ=1.0 means HF projects at **every** flow step — a genuinely different, interesting regime
(constraint-guided from noise onward) that happens to cost the same.

---

## 4. Is the naive additive cost model legit?

**Your instinct is right — but it needs to be *proven*, and it currently isn't.** The model:

```
T_plan  ≈  a · N_netcalls  +  b · N_solves
```

Fitted against the fix_7 seed-6 measurements (`a` = 8.8 ms/batched call from the `diffuser` arm;
`b_scipy` = 7.43 ms; `b_ipopt` = 19.89 ms — provenance and units in §2):

| arm | predicted | measured | err |
|---|---|---|---|
| DPCC θ=0.5 | 20·8.8 + 40·7.43 = **473 ms** | 473 ms | 0.2% |
| HF θ=0.5 | 30·8.8 + 40·19.89 = **1060 ms** | 1060 ms | 0.1% |

Looks perfect — **but this is partly circular**: `b_scipy` and `b_ipopt` were *back-solved* from those
same totals, so a <1% fit is guaranteed, not evidence. **One threshold gives one equation; you cannot
validate a two-term model from a single operating point.**

### θ=0.1 is the independent falsification test

Because DPCC's net-call count is **θ-independent** (20 at both thresholds), DPCC becomes a **clean
instrument**:

```
b_scipy  =  [ T(θ=0.5) − T(θ=0.1) ] / (40 − 8)          ← no network confound at all
```

If that value comes out ≈ 7.43 ms, the additive model is **confirmed on independent data**. If the
measured T(θ=0.1) is materially above the prediction, there's a fixed per-plan overhead the model
misses. **Predictions to check (K=20, B=4):**

| arm | predicted T at θ=0.1 | ratio HF/DPCC |
|---|---|---|
| DPCC | 176 + 8·7.43 = **235 ms** | — |
| HF | 22·8.8 + 8·19.89 = **353 ms** | **1.50×** (vs 2.24× at θ=0.5) |

**Falsifiable claim:** the HF/DPCC time ratio should fall from ~2.2× to ~**1.5×** at θ=0.1, purely
because the expensive-solve term shrinks. If it doesn't move, the additive model is wrong.

### What the naive sum legitimately ignores — and one thing it hides

Ignored terms are genuinely small and *not* worth modelling: the Euler update, the `x₁` extrapolation,
`apply_conditioning`, cost accumulation — all O(B·dof) vectorised arithmetic, nanoseconds against an
8.8 ms net call.

**But one caveat.** The CPU↔GPU transfers and `set_s0`/casadi parameter-setting happen **once per
active step** (`hardflow_projection.py:460-469`), i.e. they are **collinear with the solve count**.
The fitted `b_ipopt = 19.89 ms` therefore is *not* pure solver time — it is **"ipopt + transfer +
param-set, per solve."** The additive model works precisely *because* these overheads scale with the
same counter; it just means **`b` is an effective constant, and you must not quote it as ipopt's
intrinsic cost.**

### A second lever worth remembering: batching asymmetry

- **Net calls are batched** → cost ≈ flat in `B` (until GPU saturation).
- **NLP solves are serial** → cost **linear in `B`** (`for b in range(batch_size)`, and DPCC's
  `projection.py:131` identically).

So raising `B` inflates the solve term only. If you ever want to *stress* the solve term without
touching θ, raise B — a third axis on the same cost model.

---

## 5. Proposed terminology (the "invent a word" ask)

`NFE` alone is the wrong yardstick here — it counts only network evals and is blind to the NLP, which
is the **dominant** term (§4: 63% of DPCC's plan time, 75% of HF's). Proposal — two counters plus one
scalarisation:

| symbol | name | definition |
|---|---|---|
| **NFE** | net function evals | per-sample network evals = `K(1+θ)·B` (HF) / `K·B` (DPCC) |
| **NPE** | **N**umber of **P**rojection **E**valuations | constrained solves = `θK·B` (both arms) |
| **NFE‑eq** | **NFE-equivalent** heavy-op cost | `N_netcalls + (b/a)·NPE`, in *batched-net-call* units |

with the measured exchange rates `b/a`: **1 ipopt solve ≈ 2.26 net calls**, **1 SLSQP solve ≈ 0.84 net
calls**. Then:

| config | NFE-eq | ratio |
|---|---|---|
| DPCC K20 θ0.5 | 20 + 40(0.84) = **53.6** | 1.00 |
| HF K20 θ0.5 | 30 + 40(2.26) = **120.4** | 2.25 ✓ (matches measured 2.24) |
| DPCC K20 θ0.1 | 20 + 8(0.84) = **26.7** | — |
| HF K20 θ0.1 | 22 + 8(2.26) = **40.1** | 1.50 |

**NFE‑eq is the single number to report** — it is the "only NN queries and NLP solves cost anything"
intuition made quantitative, and it predicts wall time to <1%. (Caveat: `b/a` is hardware- and
problem-size-specific; re-fit it per node.)

---

## 6. Recommended run matrix

All on the **FMv3ODE** checkpoint in Gen12, `avoiding-d3il`, mpc=4, B=4, seeds 6–10.

| # | purpose | DPCC | HardFlow | cost |
|---|---|---|---|---|
| **0** | cheap probe at its **own** operating point (§2c: **NOT** a K=20 proxy) | K2 θ0.5 | **K1 θ1.0** | ~10× cheaper |
| **A** | baseline (have it) | K20 θ0.5 | K20 θ0.5 | — |
| **B** | **edge probe** — your θ=0.1 | K20 θ0.1 | K20 θ0.1 | ~½ of A |
| **C** | **double-parity headline** | K20 θ0.5 | **K10 θ1.0** | ≈ A |
| D | *(optional)* NFE-parity-only, to show it's misleading | K20 θ0.5 | K13 θ0.5 | — |

**§2c already killed K-invariance** using existing K=2 data, so arm 0 is no longer a validation of it.
Its value now is measuring `b_ipopt` at `n_active=1` on the *FMv3ODE* checkpoint — the same-checkpoint
control the §2c caveat calls for. **Arm B (θ=0.1) is the priority run**; it tests amortisation.

⚠️ **Every arm above sits on an INTEGER `(1−θ)·K`** (18.0, 10.0, 1.0, 0.0) so the two gates give the
same `n_active` (§1). Do **not** improvise other (K, θ) pairs without checking that — e.g. K=5 θ=0.5
gives DPCC 3 solve-steps vs HF 2, a silent 50% budget advantage to DPCC.

**A+B** validate the cost model and give the edge behaviour. **C** is the scientifically fair
head-to-head. **D** is optional and mainly rhetorical — it demonstrates why NFE-parity alone
under-provisions HF.

### What to read out

**Report every arm as this row — never a single aggregated cost (§2):**

```
arm | K | θ | NN calls | NLP solves | NFE(per-sample) | avg_time | succ+con | total_viol
```

`nfe` and `nlp_solves` are already logged per run (`eval_FM_v3_hardflow.py:483`), so both columns come
straight out of the npz — divide by plan count to get per-plan figures and check them against §2/§2b.

1. **succ+con** at θ=0.1 — the real question: with only 2 projection steps, does HF's *in-loop*
   projection still hold safety where DPCC's *post-hoc* single-shot correction fails? This is where
   HF should win: DPCC at θ=0.1 projects a nearly-final trajectory twice with no chance for the flow
   to re-absorb the correction, while HF's terminal solve is structurally guaranteed.
2. **avg_time vs the predictions** — DPCC vs 235 ms (additive model, trusted); HF vs **both** 353 ms
   (naive) and ~500–600 ms (amortisation-corrected, §2c). Which one it lands on decides whether
   `b_ipopt` is a constant or a function of `n_active`.
3. **`b_scipy` back-out** from DPCC's two thresholds — the clean, unconfounded per-solve measurement.
4. **Arm C** — equal NFE *and* equal NPE; any time gap left is pure ipopt-vs-SLSQP.

### Config knobs
- θ: `config/hardflow_projection_eval.yaml` → `HFFM_ACT_THRESHOLD` (HF) and the DPCC projector's
  `diffusion_timestep_threshold` — **both must be set**, they are separate fields.
- K: `HFFM_FLOW_STEPS`.
- Submit via `./submit.sh` (not raw sbatch), `FORCE_OVERWRITE=1` for a clean re-run.

---

### Code references
- DPCC gate: `flow_matcher_v3_hardflow/models/diffusion.py:178`; projection `:188-194`; net eval per step `:182/:184`.
- HF gate: `flow_matcher_v3_hardflow/sampling/hardflow_projection.py:454`; extra endpoint eval `:456-457`; serial solve loop `:462-466`; transfer boundary `:460, :469`.
- DPCC serial solve loop: `flow_matcher_v3_hardflow/sampling/projection.py:131`; `method='SLSQP'` `:138`.
- Cost constants (seed 6, fix_7): `logs_in_develop/Gen12/fix_7/ANALYSIS_fix7_validation_before_after_seed6.md` §"Decomposition of the residual".
