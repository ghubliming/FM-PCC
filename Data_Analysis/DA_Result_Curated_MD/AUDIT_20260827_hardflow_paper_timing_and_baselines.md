# Audit — is HardFlow's D3IL table honest? A line-by-line check of the code behind it

**2026-08-27** · source audit of `aux_repo/HardFlow` (branch `d3il`) + `arXiv-2511.08425v3/main.tex`. Nothing run; every claim below cites a file and line.
**Companion:** [`RESPONSE_20260826_did_HardFlow_ever_beat_DPCC.md`](./RESPONSE_20260826_did_HardFlow_ever_beat_DPCC.md) Q6 — the measurement this audit was spun out of.

## 0 · The direct answer — why their HardFlow is faster and ours is not

**Their HardFlow beats *their* baseline. Ours loses to *ours*. The two baselines are different programs.** HardFlow's solve is genuinely cheaper on both machines; what changes is what it is measured against.

| solve, per call | their machine | our cluster |
|---|---:|---:|
| **HardFlow** — IPOPT on the near-feasible predicted endpoint | 7.0 ms | ~30 ms |
| **their baseline** — IPOPT on the noisy iterate | 22.6 ms | ~160 ms (est.) |
| **our baseline** — **scipy SLSQP** on the noisy iterate | *never run* | **2.1–21 ms** |

- **HF vs *their* baseline: wins ~3.2× on their machine, would win ~5× on ours.** Hardware is not the cause.
- **HF vs *our* baseline: loses 1.4–14×**, because SLSQP costs **2.1 ms** at K=1–2 where our IPOPT costs ~30 ms.

**HardFlow's advantage is a 3.2× easier NLP. Our baseline's advantage is a ~12× cheaper solve. 12 beats 3.2.** Their table never tests that, because every row in it is IPOPT.

### 0.1 · "It is only a solver — should it really be 12×?" No, and that is the point

**It is not solver quality. It is a solver-class mismatch on a tiny problem.** Our NLP has **44 variables** (`dof = H·(state+action) − state = 8·6 − 4`). IPOPT is an interior-point code for large *sparse* NLPs; its per-call setup — CasADi `Opti` parameter substitution, KKT assembly, linear-solver initialisation — is **size-independent** and dominates at this scale. scipy SLSQP is a dense active-set Fortran routine with near-zero setup, which is the right tool for 44 dense variables.

**The scaling proves it.** Two measured IPOPT points on our cluster:

| | variables | ms / solve |
|---|---:|---:|
| our port, H8 | 44 | 30.0 |
| their code on our cluster, H16 (job 23565) | 92 | 49.7 |

**2.09× the variables buys only 1.66× the time.** Genuine optimisation work on a dense NLP scales *super*linearly — ~4.4× if `O(n²)`, ~9× if `O(n³)`. Sublinear scaling means a size-independent term dominates. Fitting `t = f + a·n²` gives **f ≈ 24 ms fixed and ≈ 6 ms of actual work at H8 — 81 % overhead.**

So at K = 1–2 **we are not measuring IPOPT against SLSQP; we are measuring CasADi/IPOPT per-call startup against a solve that barely exists.** Secondary contributor: our IPOPT runs `hessian_approximation: limited-memory` (L-BFGS) while DPCC's SLSQP receives analytic Jacobians for both cost and constraints — more iterations, each less informed.

⚠️ Two-point fit, and the two points come from different code paths (our port at H8, their code at H16). The fitted split is indicative; the **sublinear scaling itself is not** — it holds regardless of the fit.

**✅ CONFIRMED 2026-08-27 (job 25121) — this is no longer a fit.** The Gen12 solver bench ran both projectors side by side on the *identical* NLP, `horizon=8`, 3 seeds × 50 reps. On the near-feasible reference HardFlow actually solves, **IPOPT 47.6 ms vs SLSQP 11.0 ms = 4.33×**. And the overhead claim is now measured on a second axis — same problem size, harder problem: **IPOPT 47.6 → 54.2 ms (1.14×), SLSQP 11.0 → 34.0 ms (3.09×)**. SLSQP's time is work; IPOPT's is floor. Consequence: HardFlow's endpoint trick is worth **3.09× to SLSQP and only 1.14× to IPOPT** — *HardFlow ships the one solver that cannot cash in its own central optimisation.* Full numbers: `logs_in_develop/Gen12/Solver_Bench/RESULTS_20260827_solver_bench_ipopt_vs_slsqp.md`.

**This is measured, not reconstructed.** The fan-matched parity run (K=2, both arms at fan 1) timed it directly: DPCC **2.4 ms/step**, HardFlow **30 ms/step**, on top of an 18.5 ms generator.

### 0.2 · So which run? Give HardFlow our solver — not our baseline their solver

**Decision: swap arm C's IPOPT for DPCC's SLSQP. Do not give arm B IPOPT.**

Giving arm B IPOPT only reproduces a published number on our machine, and §0.1 shows that number is ~81 % per-call plumbing. It would make our baseline artificially slow to match a measurement we have just shown is not measuring solvers. No finding in it. (Keep it in reserve only if a reviewer disputes the reconciliation in §0.)

Giving arm C SLSQP is the **fair** comparison: same solver on both arms, so the only remaining difference is the algorithm — *what gets projected*, the predicted endpoint versus the noisy iterate.

**It is nearly free, because the two NLPs are already the same problem.** Our HF cost is `0.5·reg_scale·τ²·‖x − x_ref‖²`. That scalar weight multiplies the entire objective, so it does not move the argmin: HF's NLP is exactly `Π_S(x_ref)` — which is what DPCC's `Projector.project` already computes (`Q = I`, `r = −x_ref`, `projection.py:74`). The edit is one call site in the HF sampler:

```
self.nlp.solve(X1_ref_np[b], tau_next)    →    Projector.project(X1_ref)
```

on a projector object that already lives in the same package. No new solver code, no new constraint plumbing. Residual formulation gaps (`s_0` scope, the `Bounds(−5,5)` box) are the ones catalogued in companion 4c and are small.

**Decision rule once it lands:**

| outcome | conclusion |
|---|---|
| cost collapses to parity (HF still pays 1 extra NFE per active step) | cost stops being the story; **S&C and steps decide** |
| HF still loses S&C and steps | **HardFlow is finished for us** — and cost never has to be argued again |
| HF's quality improves | the gap was partly **IPOPT failures**, not the algorithm — see below |

**The one way this run surprises us.** Our IPOPT fails **1.4–1.8 %** of solves at high K and **12.5–13.5 %** on visual-avoiding TL untightened (companion 2b), each failure returning a possibly-infeasible last iterate. If part of HardFlow's quality deficit is those silent failures rather than the method, SLSQP could close it. Nothing else in the corpus would reveal that.

Either way the question closes, which is why this is the run.

### 0.3 · What follows

**HardFlow's cost disadvantage in our harness is mostly an engineering tax, not the algorithm.** Give the HF arm a solver sized for a 44-variable dense problem — SLSQP, or IPOPT with the per-call setup hoisted — and its solve should fall toward ~6 ms plus setup. At that point HardFlow's genuine 3.2× easier NLP would actually surface, and arm B vs arm C would become an algorithm comparison rather than a solver-plumbing one.

**✅ Measured 2026-08-27 (job 25121).** The predicted fall came in at **11.0 ms**, not ~6 ms — the fitted "work" term was low — but the direction and size hold: **4.33× off HardFlow's solve.** Extrapolating that ratio onto the fan-matched parity run moves arm C from 48.5 → 25.4 ms/step against DPCC's 20.9, i.e. **2.32× → 1.22×**. That is §0.2's first decision row: *cost stops being the story, S&C and steps decide* — and chapters 1–3 already decide those against HardFlow. Two caveats: it is an extrapolation until an arm-C-with-SLSQP eval actually runs, and it changes only the solve term, not the solve count or the generator. Separately, the bench reproduced the IPOPT-failure defect offline (**26 % non-convergence** on noisy references, 3× larger residual than SLSQP) with no GPU and no checkpoint — see `RESULTS_20260827_*` §5 for why most of that is the synthetic reference regime rather than the eval.

**Correction this forces on the companion doc:** chapter 4 calls the solver swap "the smallest term". That is true of the **trajectory** — the `fm` rollouts came out bit-identical, so the solver does not change *what* is produced. It is false of **cost**, where the solver is the largest term by far. Both statements are now carried in 4f.

---

## Verdict on the paper

**They are not lying, and I could not construct a case that they are.** I went looking for a thumb on the scale and checked every candidate: the activation schedule, the solver, the solver options, the NLP formulation, the constraint margin, the candidate fan, the ODE step count, and the network-evaluation path. **Seven of eight are matched across all methods.** The eighth — an implementation asymmetry that *would* have been a scale-thumb — cancels arithmetically, and their headline result replicates on our cluster.

**What I did find is one real measurement error in the headline table, plus four reporting defects.** The measurement error inflates HardFlow's apparent advantage over the *unguided* baseline by ~1.3× and makes one row incomparable. None of it changes the paper's ranking of methods.

If you want the one-sentence version: **the speed claim is earned by the algorithm, the safety claim replicates, and the Computation Time column is not measuring what its caption says it measures.**

---

## 1 · What is matched — the checks that came back clean

| # | candidate for a thumb on the scale | finding | evidence |
|---:|---|---|---|
| 1 | **Constraint margin** | `obstacle_margin = 0.02` in **every** run script — HardFlow, Projection-All/Late/Relaxed, OC-Flow, Gradient Guidance | `run_scripts/eval_*.sh:21-31` |
| 2 | **ODE step count** | `ode_t_steps = 10` for every method | `run_scripts/eval_*.sh` |
| 3 | **Candidate fan** | `warmstart_batch = 1` for every method; `batch_size == 1` hard-asserted in every forward | `run_scripts/*`, `flow_policy.py:798, 1293` |
| 4 | **Horizon / cadence** | `horizon=16`, `replan_steps=8` for every row | `run_scripts/*` |
| 5 | **Solver + options** | `ipopt` with `hessian_approximation: limited-memory` in both formulations | `flow_policy.py:544, 748` |
| 6 | **NLP structure** | identical `oc_dof`, identical `_apply_obstacle_constraints` + `_apply_dynamics_constraints` with the same `X_index_selector="projection"`. Only the cost's scalar weight differs (`reg_scale · t²`) | `flow_policy.py:498-544` vs `:683-748` |
| 7 | **Number of solves** | HardFlow `hardflow_activation="all"` → **10 solves**; Projection-All → **10 solves**. Not fewer. | `eval_hardflow_new.sh:33`, `flow_policy.py:1328-1336, 866-880` |
| 8 | **Their headline replicates** | our cluster, their repo, their released checkpoint, 50 episodes: 4 % → **100 % safe, 0 violations, 50.7 steps** (paper: 52.5) | job 23565 |

**On #1 specifically** — this corrects a claim in the companion doc. I had written that their projection baselines run "untightened". **That is wrong.** All methods share `obstacle_margin = 0.02`, comparable to our own 0.025 tightening. Their baselines are margin-matched to HardFlow; the reason Projection-All only reaches 0.46 safety is not a missing margin.

---

## 2 · The asymmetry that would have been cheating — and why it isn't

`run/eval.py:577-580` routes the network through the **L4CasADi bridge** for `projection`, `projection_relaxed` and `hardflow`, but **not** for `hardflow_new` — the variant that is the paper's row, which calls PyTorch directly (`flow_policy.py:1313`).

That is a genuine implementation asymmetry: the baselines pay a slower per-evaluation path than the proposed method. Had the numbers depended on it, it would be a scale-thumb.

**It cancels.** HardFlow evaluates the network **twice per step** (ODE step + endpoint lookahead, `:1338-1339`) against projection's once. Priced from their own rows:

- L4CasADi eval ≈ **12.3 ms** → projection pays 10 × 12.3 = **123 ms**
- torch eval ≈ **6.0 ms** → HardFlow pays 20 × 6.0 = **120 ms**

**A 3 ms difference on a 159 ms gap.** The eval path explains none of it.

**Where the gap actually comes from — and it is algorithmic.** All four unit costs are derivable from their published rows with no free parameters:

| quantity | derivation | value |
|---|---|---:|
| torch flow eval | `Original` = 10 evals, 0 solves = 0.060 | 6.0 ms |
| **projection IPOPT solve** | `All − Late` = 5 solves = 0.349 − 0.236 | **22.6 ms** |
| L4CasADi flow eval | (`All` − 10 solves) ÷ 10 | 12.3 ms |
| **HardFlow IPOPT solve** | (`HF` − 20 torch evals) ÷ 10 | **7.0 ms** |

Cross-check: `Projection-Relaxed` = 0.116 ≈ 10 bridged evals with near-free augmented-Lagrangian steps. Four rows, one consistent model.

**HardFlow's solve is 3.2× cheaper at matched solver, matched solve count and matched formulation**, because of *what it projects*: the **predicted clean endpoint** `x_next_ref + (1−t−dt)·v_next` (`:1339`), which lies near the data manifold and is usually already near-feasible, versus projection's **noisy intermediate iterate** `x_k + v_k·dt` (`:861`), which at early `t` is far from the constraint set. The `t²` cost weight reinforces it. **This is a real property of the method and their speed claim rests on it legitimately.**

---

## 3 · 🔴 The real defect: the Computation Time column mixes two different quantities

**This is the finding that matters.** Whether the base generation pass is inside the timer **differs by method**:

| method | calls `warmstart()` before `t_start`? | does the reported time include generation? | evidence |
|---|---|---|---|
| `Original` | no — times the full ODE solve | ✅ **includes** | `flow_policy.py:171-179` |
| **Gradient Guidance** | no — times the full ODE solve | ✅ **includes** | `:1449-1451` |
| OC-Flow | **yes** | ❌ excludes | `:1536-1538` |
| Projection-All / Late / Relaxed | **yes** | ❌ excludes | `:809-810`, `:970-971` |
| **HardFlow (`hardflow_new`)** | **yes** | ❌ **excludes** | `:1302-1305` |

`warmstart()` is a full N=10 ODE sample plus value selection (`:753-793`). So **the headline 0.190 s is the guidance loop only** — the cost of producing the sample it guides is not in it, while `Original`'s 0.060 s and Gradient Guidance's 0.992 s both contain it.

**Consequences, in order of severity:**

1. **The caption's comparison against `Original` is wrong.** *"incurring only mild computational overhead"* compares 0.190 (no generation) against 0.060 (with generation). Add generation back and HardFlow is ~0.250 s — **4.2× Original, not 3.2×**. The overhead is understated by about a third.
2. **The Gradient Guidance row is not comparable to the projection rows at all.** GG (0.992, with generation) against Projection-All (0.349, without) is apples-to-oranges. Normalised, GG ≈ 0.932 of guidance vs Projection-All's 0.349 — the qualitative conclusion (guidance is far more expensive) survives, but the printed ratio does not.
3. **The ranking among the hard-constrained methods is unaffected**, because Projection-*, OC-Flow and HardFlow all exclude generation identically. HardFlow still leads that group.

**Is this deliberate?** Nothing suggests it. `Original` and Gradient Guidance simply do not have a warm-start stage to exclude — the inconsistency falls out of the code structure rather than being introduced. But it is an error in a headline table and it happens to flatter the proposed method.

---

## 4 · Four reporting defects, ranked

| # | defect | why it matters | evidence |
|---:|---|---|---|
| **D1** | **Provenance misreporting.** *"This task follows the setup of [romer2025diffusion]"* while silently changing `H` 8→16, `T` 1→8, `max_episode_length` 200→100, and adding novel test-time obstacles | invites a cross-paper comparison the numbers cannot support | `main.tex:728` vs `dpcc/config/avoiding-d3il.py:22, 68`, `hardflow/config/flow_matching.py:53` |
| **D2** | **DPCC is miscited.** Filed under **Projection-All** ("projecting after every sampling step"). DPCC ships `diffusion_timestep_threshold = 0.5` and projects only the later half — by their own taxonomy it is **Projection-Late** | the row a reader would take as "DPCC" (0.46 safety) is not DPCC; the closer row is Projection-Late (0.76) | `main.tex:720` vs `dpcc/diffuser/sampling/projection.py:8`, `diffusion.py:186` |
| **D3** | **Per-episode compute never reported.** Computation Time is defined *per replanning step* (`main.tex:737`) and `T = 8`, so per-episode planning cost is ~8× lower than a replan-every-step method. The practicality framing rests on a cadence the reader is not shown | `T` is invisible in the only cost metric printed | `main.tex:737`, `run/eval.py:391` |
| **D4** | **`H` and `T` are never ablated.** Stated once in an appendix sentence, no sensitivity study, and the repo's own dataclass defaults are still DPCC's `horizon=8, replan_steps=1`, overridden by every run script | "fewest steps 52.5" is plausibly in part a horizon effect; nothing in the paper separates them | `main.tex:1242`; `hardflow/config/flow_matching.py:12, 44, 47` vs `run_scripts/*.sh` |

**On D4 — the counterfactual is cheap and nobody has run it.** Their config already defaults to H8/T1. One retrain plus one eval, no code change, settles whether "fewest steps" survives the horizon.

---

## 5 · 🔴 The strongest criticism: no row in their table is DPCC

**The valid way to claim you beat DPCC is to run DPCC's released code on DPCC's benchmark and report both.** HardFlow does not do that. It re-implements the *idea* of projection on its own flow model, under its own settings, and the paper's *"this task follows the setup of [romer2025diffusion]"* invites the reader to treat the result as a DPCC comparison.

Every defining setting of DPCC is different in the reconstruction:

| | DPCC as published | HardFlow's "Projection-*" rows | source |
|---|---|---|---|
| generative model | **Gaussian diffusion**, `n_diffusion_steps = 20` | flow matching, `ode_t_steps = 10` | `dpcc/config/avoiding-d3il.py:23` vs `eval_projection.sh:19` |
| **candidate fan** | **`batch_size = 4` + a trained value model for selection** | **`warmstart_batch = 1`, no selection** | `dpcc/config/avoiding-d3il.py:69, 91` vs `eval_projection.sh:25` |
| planning horizon | `H = 8` | `H = 16` | `:22` vs `run_scripts/*` |
| replan cadence | every step (`T = 1`) | `T = 8` | `dpcc/scripts/eval.py:231` vs `run_scripts/*` |
| episode cap | `200` | `100` | `:68` vs `hardflow/config/flow_matching.py:53` |
| solver | scipy **SLSQP** + analytic Jacobians, `Bounds(−5, 5)` | **IPOPT**, no box | `dpcc/…/projection.py` vs `flow_policy.py:544` |
| `s_0` | decision variable, only `deriv` dims pinned | NLP parameter, fully pinned | `projection.py:149-157` vs `flow_policy.py:517` |
| obstacles | original pillars | pillars **+ novel test-time regions** | `main.tex:728` |

**The candidate fan is the one that hurts most.** DPCC is *Diffusion Predictive Control* — sampling several candidate plans and selecting among them with a value model **is** the control part of the method. Dropping to a single sample is not a neutral simplification; it removes a component whose contribution we have measured directly. In our own fan-parity run, moving DPCC from fan 4 to fan 1 changed `dpcc-c` from S&C 0.667 to 0.333 (companion doc, Q3). **Their "DPCC" is a weakened DPCC by construction.**

**In fairness, two things.** First, some substitutions are forced: HardFlow's math needs a flow ODE, so a diffusion baseline cannot be dropped in unchanged, and re-implementing projection on their own model is the only way to hold the generator fixed across rows — which is genuinely the right call *for their internal comparison*. Second, **they never write "we beat DPCC" and DPCC is not a row in the table.** The table is internally honest (§1).

**But that is exactly the problem.** A paper that cites DPCC, says it follows DPCC's setup, and prints a row labelled with DPCC's citation at safety 0.46, will be read as having beaten DPCC — and no experiment in the paper supports that reading. The comparison that would support it (their method against DPCC's released code, DPCC's settings, DPCC's benchmark) is absent, and nothing in the paper flags the gap.

**What this means for us, concretely:** when we cite HardFlow, we must not report their Projection-All row as a DPCC number, and we should say plainly that **the HardFlow-vs-DPCC comparison does not exist in the literature — ours is the first.** That is a contribution, not a footnote: our arm B is DPCC's actual projector, at DPCC's fan, with DPCC's solver, on DPCC's benchmark.

---

## 6 · What is fair to say about this paper

**Fair:**
- The safety claim (1.00, zero violations) is real and **replicates on independent hardware**.
- The speed advantage over projection baselines is **earned by the algorithm** — projecting a near-feasible predicted endpoint instead of a noisy iterate makes the NLP 3.2× cheaper at matched everything else.
- Their internal comparison is **margin-, fan-, step- and solver-matched**. It is a cleaner benchmark than most.

**Fair criticism:**
- The Computation Time column **is not measuring what the caption says** (§3), and the error flatters the method against `Original` by ~1.3×.
- The baselines run through a **slower network-evaluation path** than the proposed method (§2). It cancels here, but they neither report it nor could have known it cancels without checking.
- The DPCC comparison a reader will draw is **not supported** — different H, T, episode cap, obstacle set, and DPCC is filed under the wrong baseline family (D1, D2).

**Not fair, and unsupported by anything in this audit:**
- That the baselines were handicapped **relative to each other**. They were not — §1. (That is a separate question from whether any of them is DPCC — §5.)
- That the speed result is an implementation artifact. It is not — §2.
- That the numbers were fabricated. Their headline reproduced on our cluster to within the step count.

---

## 7 · What this changes for us

1. **Correct the companion doc.** Q6 Part 4 claims their projection baselines are untightened. **Wrong** — `obstacle_margin = 0.02` throughout (§1). The reason our `dpcc-*-tightened` outperforms their Projection-All is task difficulty (novel obstacles), candidate fan (4 vs 1), and *what is projected*, not a missing margin.
2. **Never quote their absolute times against ours.** Our cluster runs their own code 4.5× slower end-to-end and **7.1× slower on IPOPT specifically** (Q6 Part 6). Only within-machine ratios transfer.
3. **The experiment that closes the loop** is still unrun: give our DPCC arm **IPOPT on the noisy iterate** — their Projection-All/Late — inside our harness. Expected: our HardFlow beats it ~3×, reproducing their result, while still losing to SLSQP. That would show both papers measure the same phenomenon and that neither is wrong.
