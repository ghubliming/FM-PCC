# Gen3v6 U3 — HardFlow arm: in-loop constrained sampling on the mean-flow checkpoint (matched-K sweep)

**Runs:** eval job **23981** (K=2, `temp/2907/2907/…H8_K2_…`) plus the matched-K sweep jobs
**24021 (K=1) / 24022 (K=5) / 24023 (K=20)** (`temp/2026-07-30/`, git `bed63b3`).
**Setup:** the U2 `mf_dit` aw10 checkpoint (seed 6, EMA weights), **matched K** for all arms,
**mpc=4** (`HFFM_BATCH=4`), HardFlow activation threshold **0.5** (== DPCC's), 3 halfspace
scenarios, **2 trials/cell**. `hardflow_new-*` queries the u-head at `h=0`, so `u(x,t,0)=v`.

*(The rest of `temp/2907/2907` — `H16_imf_*`, `H16_ml_*` — is Gen13 HardFlow-native, a different
codebase; out of scope here.)*

> 🛑 **INVALIDATED 2026-07-30 (fix_4) — every `hardflow_new-*` number on this page is unsound.**
> The audit before the Gen3v7 port found that arm C opened its ODE at **σ=0.5** while the MeanFlow
> checkpoint is trained and sampled at **σ=1.0** — the U3 port had inherited Gen12's noise law
> verbatim. Arm C therefore ran off-distribution, and arms B/C did **not** share a start
> distribution, so no DPCC-vs-HardFlow comparison below is valid. In particular the "identical
> inputs" replan-0 evidence was *not* identical: arm C's `x_init` is exactly 0.5× arm B's under the
> same seed, so the claim that **the in-loop NLP manufactures the collapsed trajectories is
> observed but not established.** Full severity breakdown, including what survives, in
> [`../fix_4/CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md`](../fix_4/CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md) §2.
> **Arms A/B are unaffected** (different code path) — the DPCC results and the whole `dpcc-c`
> investigation stand. Re-run the sweep before citing anything here.
>
> **Update 2026-07-30.** This file originally covered K=2 only and concluded "HardFlow reaches
> DPCC-parity." **The K-sweep does not support that as a general claim** — parity holds at K=1–2 and
> breaks down at K≥5, where DPCC improves to a clean sweep and HardFlow does not. §K-sweep below is
> authoritative where it disagrees with the K=2-only reading. The port's *structure* is still
> correct (h=0 identity, dof layout, constraints — all re-audited in fix_4 §6); its *noise law* was
> not.

---

## Headline (revised after the sweep)

**The port is correct, but the two engines have opposite K-dependence.** At K=1–2 in-loop HardFlow
does match DPCC's safety, which is what the original K=2 run showed. Extend the budget and they
diverge: **DPCC gets monotonically better** (all three tightened arms reach 1.0 goal-and-constraints
on all three scenarios at K=5 *and* K=20), while **HardFlow gets worse** — only `-r-tightened` keeps
up, `-t-tightened` decays to 0.5, and `-c-tightened` collapses to **0.0 everywhere**.

The cause is a HardFlow-side defect the K=2 run could not have revealed: **repeated in-loop NLP
intervention manufactures degenerate motionless trajectories.** At K=5/K=20, ~29–31% of HardFlow's
candidates collapse to zero net displacement, while the *same checkpoint, same seed, same noise*
under DPCC produces **0%** (§Two distinct `-c` collapses). This is the port's real limitation and it
is orthogonal to the K=2 generative-collapse bug documented in
[`INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](INVESTIGATION_dpcc-c_stuck_at_point_K2.md).

The `h=0` identity itself remains validated: HardFlow's NLP consumes a genuine instantaneous
velocity, produces sensible trajectories, and reaches **0 violations** wherever it reaches the goal,
at every K tested.

## Port correctness — confirmed

- **Ran clean:** matched K=2 applied (`train=10→eval=2`), EMA weights, NFE/NLP metrics logged, no
  crashes. Savepath correctly encoded `_K2_` (the collision fix holds).
- **Parity safeguard:** `dpcc-c-tightened` and `hardflow_new-c-tightened` **both hit 0 violations in
  all 3 scenarios** — the shared tightened feasible set is enforced identically by the scipy
  (post-hoc) and casadi (in-loop) solvers. The port is clean.
- **NLP health:** 0 failures on both-hard and top-right; failures appear only on top-left-hard and
  only for **untightened** HF arms (8 for `-r`, 36 for `-t`) — the tightened arms solve cleanly
  (0 failures, 0 violations). Tightening stabilises the interior-point solve.

## Results (K=2, mpc=4, 2 trials — directional). g&c = goal-AND-constraints

| variant | top-right | top-left | both-hard | note |
|---|---|---|---|---|
| **diffuser** (raw) | 0.0 (12.5 viol) | 0.0 (24.5 viol) | 0.5 (11.5 viol) | unsafe floor |
| dpcc-r | **1.0** | 0.5 | **1.0** | |
| dpcc-r-tightened | 0.5 | **1.0** | **1.0** | DPCC best |
| dpcc-t-tightened | 0.5 | **1.0** | **1.0** | DPCC best |
| dpcc-c / -c-tightened | 0.0 | 0.0 | 0.0 | **collapsed (see below)** |
| **hardflow_new-r-tightened** | 0.5 | **1.0** | **1.0** | HF best |
| **hardflow_new-t-tightened** | 0.0 | **1.0** | **1.0** | HF best |
| hardflow_new-c-tightened | 0.5 | 0.5 | 0.5 | 0 viol throughout |
| hardflow_new-r / -t (untightened) | 0.5 | 0.0 (5–7 viol) | 0.5–1.0 | margin leaks w/o tightening |

**Reading:**
- **HardFlow ≈ DPCC.** On top-left and both-hard, the HF tightened `-r`/`-t` arms match DPCC's best
  (g&c=1.0, 0 violations). On top-right-hard both engines mostly land at 0.5 — it's a genuinely hard
  geometry, not an engine difference. Raw `diffuser` is unsafe everywhere (11–24 violations).
- **Tightening matters for the in-loop arm too:** untightened HF `-r`/`-t` ride the zero-margin
  boundary and leak (top-left `-t`: 7.5 viol / 0.541 total, 36 NLP failures); the +0.025 margin
  fixes it (0 viol, 0 failures) — same behaviour DPCC shows, now confirmed for casadi in-loop.
- **The `-c` (minimum-projection-cost) selection is degenerate — for BOTH engines.** `dpcc-c*`
  collapses to succ=0 everywhere; `hardflow_new-c*` is weak (0.5/0.5/0.0). At mpc=4, "least
  intervention" selects the candidate that barely moves — safe but goal-missing.
  ⚠️ **Superseded by the sweep:** the K=2-only reading, that this is one shared selection-rule
  artifact, is wrong. The two engines fail here for **two different reasons at two different K**
  — see §Two distinct `-c` collapses.

---

# K-sweep (jobs 24021 / 24022 / 24023) — matched K ∈ {1, 2, 5, 20}

## Goal-AND-constraints matrix, tightened arms (2 trials/cell: 0.0 / 0.5 / 1.0)

| K | h/step | engine | `-r-tightened` (TR/TL/BH) | `-t-tightened` | `-c-tightened` | Σ/9 |
|---|---|---|---|---|---|---|
| 1  | 1.00 | DPCC     | 1.0 / 0.5 / 1.0 | 1.0 / 1.0 / 1.0 | 0.0 / 0.0 / 1.0 | 6.5 |
| 1  | 1.00 | HardFlow | 1.0 / 1.0 / 1.0 | 0.5 / 1.0 / 1.0 | 0.5 / 1.0 / 1.0 | **8.0** |
| 2  | 0.50 | DPCC     | 0.5 / 1.0 / 1.0 | 0.5 / 1.0 / 1.0 | 0.0 / 0.0 / 0.0 | 5.0 |
| 2  | 0.50 | HardFlow | 0.5 / 1.0 / 1.0 | 0.0 / 1.0 / 1.0 | 0.5 / 0.5 / 0.5 | 6.0 |
| 5  | 0.20 | DPCC     | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | **9.0** |
| 5  | 0.20 | HardFlow | 1.0 / 1.0 / 1.0 | 0.5 / 0.5 / 0.5 | 0.0 / 0.0 / 0.0 | 4.5 |
| 20 | 0.05 | DPCC     | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | **9.0** |
| 20 | 0.05 | HardFlow | 0.5 / 1.0 / 1.0 | 0.5 / 0.5 / 0.5 | 0.0 / 0.0 / 0.0 | 4.0 |

**The crossover is the story.** HardFlow leads at K=1 (8.0 vs 6.5) and K=2 (6.0 vs 5.0) — the
few-step regime the port was built for — then loses decisively at K=5/K=20 (4.5/4.0 vs a perfect
9.0). Raw `diffuser` never exceeds 1/9 at any K and rides 10–27 violations, so both engines are
doing real safety work throughout; the divergence is in *goal-reaching*, not safety.

**`hardflow_new-r-tightened` is the one arm that holds up everywhere** (3.0 / 2.5 / 3.0 / 2.5 across
K=1/2/5/20). If a single HF configuration goes in the paper, it is that one.

## Two distinct `-c` collapses — same symptom, different causes, different K

Both engines' `-c` arm freezes the robot at its start pose. **They are not the same bug.** Frozen
executed actions (`ACT (±0.000,±0.000)`) on `both-hard`, trial 0:

| K | `dpcc-c` frozen | `hardflow_new-c` frozen |
|---|---|---|
| 1  | 2/83 | 2/71 |
| 2  | **198/200** | 7/71 |
| 5  | 0/53 | **200/200** |
| 20 | 1/59 | **200/200** |

Exact mirror images. Measuring the candidate fan (net planned horizon displacement per candidate,
`both-hard`, both trials) shows why:

| K | `dpcc-c` collapsed (`<1e-3`) | `hardflow_new-c` collapsed |
|---|---|---|
| 1  | 0.0% (0/788) | 0.0% (0/132) |
| 2  | **28.1%** | 16.5% |
| 5  | 0.0% (0/432) | **28.7%** |
| 20 | 0.0% (0/608) | **30.9%** |

- **DPCC's `-c` failure at K=2 is a *generation* bug** — the checkpoint's own two-time field has a
  degenerate "stay put" mode at the `(r,t)` coordinates K=2 visits. Fully documented in
  [`INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](INVESTIGATION_dpcc-c_stuck_at_point_K2.md).
- **HardFlow's `-c` failure at K≥5 is a *port* bug — the in-loop NLP creates the collapse itself.**
  At K=5/K=20 the generator is provably clean (DPCC sees 0/432 and 0/608 collapsed from the same
  checkpoint, seed and noise), yet HardFlow's post-NLP fan is ~29–31% collapsed.

### The decisive single-input comparison

Replan 0, trial 0, `both-hard` — identical start pose, identical `torch.manual_seed(0)` noise, so
both engines receive the **same four raw candidates**. Only the constraint machinery differs:

```
K=5    dpcc-c          candidate spans = [0.09642, 0.02572, 0.09692, 0.08729]
       hardflow_new-c  candidate spans = [0.04724, 0.02594, 0.00005, 0.06538]
                                                              ^^^^^^^ crushed to 5e-5
K=20   dpcc-c          candidate spans = [0.06940, 0.02572, 0.09773, 0.08490]
       hardflow_new-c  candidate spans = [0.05391, 0.02599, 0.00001, 0.06586]
```

Candidate 2 goes from a healthy 0.097 to 5e-5; candidates 0 and 3 are attenuated ~30–45%; candidate 1
(which needed little intervention) is untouched at 0.0257→0.0259. **The NLP is not filtering these
trajectories, it is destroying them** — and `NLP failures = 0` in these cells, so this is not a solver
failure. Ipopt converges, to a degenerate answer.

**Mechanism (consistent with the numbers, not yet proven at solver level):** the NLP solve count per
candidate per replan is `K − int(0.5·K)` = **1, 1, 3, 10** for K = 1, 2, 5, 20 (confirmed against the
logged `NLP solves`: 552/560/1056/4800). Collapse appears only once there is **more than one** in-loop
solve. A single terminal solve (K=1, K=2) cannot feed back; from K=5 on, each intervention hands the
*next* ODE step an `x` that has been pushed off the model's data manifold, and the error compounds
across the remaining steps into a degenerate fixed point. This predicts collapse should scale with
active-step count, not with `h` — testable by sweeping `HFFM_ACT_THRESHOLD` at fixed K (§Lacking data).

`-c` does not cause either collapse; it is simply the only selection rule that actively *seeks*
minimum intervention, so it finds whichever degenerate candidate exists. `-r` and `-t` don't, which is
why they survive both failure modes.

## Compute cost — the "4× in-loop price" is K-dependent and mostly disappears

Mean per-step compute, averaged over all arms of each engine and all 3 scenarios:

| K | diffuser | DPCC | HardFlow | HF / DPCC |
|---|---|---|---|---|
| 1  | 10 ms | 27 ms | 118 ms | **4.4×** |
| 2  | 17 ms | 24 ms | 100 ms | **4.2×** |
| 5  | 43 ms | 212 ms | 224 ms | **1.06×** |
| 20 | 160 ms | 883 ms | 1058 ms | **1.20×** |

**Revision of the K=2-only claim.** The original "~4× DPCC, the honest in-loop price" is a *low-K*
statement. DPCC's projection count also scales with K, so by K=5 the two engines cost essentially the
same per step, and at K=20 HardFlow is only 1.2× DPCC. The in-loop premium is real only in the
few-step regime — which is, awkwardly, the regime where HardFlow is *also* the better engine (Σ 8.0
vs 6.5 at K=1). The honest framing: **HardFlow buys its advantage at K=1–2 for a 4× per-step premium
over a much cheaper DPCC; at K≥5 it costs the same as DPCC and performs worse.**

## Answering the original open questions

1. ✅ **The other K points** — done, this section.
2. ⬜ **Seeds / trials** — still 1 seed × 2 trials per cell. Unchanged.
3. ✅ **The top-right-hard 0.5 ceiling was a few-step field-quality limit, not constraint geometry.**
   It lifts cleanly for DPCC at K=5 and K=20 (all tightened arms 1.0, 0 violations), and raw
   `diffuser` even reaches 1.0 there at K=20. HardFlow does **not** clear it (`-t`/`-c-tightened`
   stay at 0.5/0.0) — further evidence the residual limitation is in the port, not the scenario.

## What this establishes (and what it doesn't)

**Establishes:** the U3 port is correct and usable — the MeanFlow checkpoint can be driven by
HardFlow's in-loop constrained sampler via `u(x,t,0)=v`, with 0 violations wherever it reaches the
goal, at every K. In the **few-step regime it was designed for (K=1–2) it beats DPCC** on the
goal-and-constraints matrix. And the sweep produced a concrete, localized, actionable defect: in-loop
NLP intervention compounding into trajectory collapse at ≥2 active solves per candidate.

**Does NOT establish:** a statistically-backed ranking — still **1 seed, 2 trials/cell**, so every
number is 0.0/0.5/1.0 and the Σ column is directional. It also does not establish that the collapse
mechanism is the compounding-feedback story above; that is the hypothesis the numbers fit, and it has
a cheap falsification test (below).

---

## ⚠️ Lacking data — what's needed next

1. **Falsify-or-confirm the compounding hypothesis: sweep `HFFM_ACT_THRESHOLD` at fixed K.** The
   claim is that collapse tracks *number of in-loop solves*, not `h` or `K`. At K=20, threshold 0.95
   gives 1 solve/candidate, 0.5 gives 10. If collapse vanishes at threshold 0.95 and returns at 0.5,
   the mechanism is confirmed and the fix direction (fewer, later interventions) follows immediately:
   ```
   HFFM_FLOW_STEPS=20 HFFM_ACT_THRESHOLD=0.95 HFFM_BATCH=4 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
   HFFM_FLOW_STEPS=20 HFFM_ACT_THRESHOLD=0.75 … (same)
   ```
   **This is the highest-value next run** — one knob, existing script, directly tests the mechanism.
2. **Seeds / trials.** 1 seed × 2 trials makes every cell 0.0/0.5/1.0 — directional only. The paper
   table needs seeds 6–10 (⇒ train 7–10 first) and/or more trials for error bars.
3. **Inspect the NLP's degenerate solutions directly.** `NLP failures = 0` while producing 5e-5-span
   trajectories means ipopt is converging to a feasible-but-useless point. Logging the NLP objective
   value and the pre/post-solve `x` for a collapsed candidate would show whether the objective is
   under-specified (nothing penalises standing still) or the warm-start/bounds are pulling it there.
4. **Drop `-c` from the HF arm for paper purposes**, or replace its cost with an
   intervention-per-unit-progress ratio. As specified, minimum-intervention selection is degenerate
   for both engines; it is only a question of which K exposes it.

## Net

U3's port is **validated and its operating envelope is now mapped**: HardFlow-into-Gen3v6 is the
better engine at **K=1–2** (Σ 8.0/6.0 vs DPCC's 6.5/5.0) at a ~4× per-step premium, and the worse one
at **K≥5** (Σ 4.5/4.0 vs a perfect 9.0) at roughly equal cost — because repeated in-loop NLP
intervention collapses ~30% of its candidates. Use `hardflow_new-r-tightened`; avoid `-c` entirely.
The next run is the `HFFM_ACT_THRESHOLD` sweep, then seeds.
