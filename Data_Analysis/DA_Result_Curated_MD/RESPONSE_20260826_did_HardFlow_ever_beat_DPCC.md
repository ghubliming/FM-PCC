# HF Q&A — HardFlow vs DPCC

**2026-08-26** · re-analysis of existing runs, nothing new executed, no code touched.
**Cost convention:** `ep-time s` in every table = one episode's total wall-clock, `avg_time × n_steps` — the "time × steps" metric. "episodes" always means rollout count (sample size), never time. Each section names its own data source and states whether the candidate fan was matched natively or re-priced.

> 🔴 **Degeneracy rule — applies to every row below.** A row is HardFlow only if `n_genuine ≥ 1`, where `n_genuine = max(K − floor((1−A)·K), 1) − 1`. **Every run here is A = 0.5, so K=1 and K=2 have `n_genuine = 0`** — those rows are `Π_S(Euler sample)`, sample-then-project, DPCC's algorithm with IPOPT instead of SLSQP. **No HardFlow-specific math runs in them.** K=3 is alive but marginal (`n_genuine = 1`); K≥5 gives ≥2 genuine steps, matching the paper's own N=10 / A=0.5. Rows are tagged ❌ degenerate / ⚠️ marginal / ✅ genuine. **A quality or cost claim about *HardFlow* may only be built on ✅ rows.** (Gen12 ships A=1.0 and is degenerate at K=1 only — do not blanket-apply this.)

**Structure:** `#` = one chapter per major question asked · `##` = each sub-question or follow-up within it · `###` = parts of a single answer. Later major questions append as `# 6`, `# 7`, …

---

# 1 · `avoiding-d3il` — did HardFlow ever beat DPCC?

**Data:** `temp/2508/batch_avoiding_combined_20260825_143212/candidates_multidimensional_raw.csv`, `bbunet` parent only — the DiT checkpoint writes identically-named folders, so split by `Full_Path` or the two average together. **Fan:** raw logs ran DPCC at 4 candidates and HF at 1, so both arms are re-priced at fan 1 throughout; the reconstruction checks out against the one measured fan-1 cell (DPCC 0.0208 predicted vs 0.021 measured, HF 0.0503 vs 0.049).

## Q1 · Did HF beat DPCC at H8+8 (H=16, execute 8)?

In one cell, K=5, and it is the only ✅ cell tested. n=6 per cell (1 seed × 2 trials × 3 scenes).

| config | K | | DPCC S&C · steps · ep-time s | HF S&C · steps · ep-time s | outcome |
|---|---:|---|---|---|---|
| H16/r1 | 1 | ❌ | 1.000 · 58.8 · 1.64 | 1.000 · 60.5 · 3.81 | not HF — IPOPT 2.3× costlier than SLSQP |
| H16/r1 | 2 | ❌ | 1.000 · 62.7 · 2.36 | 1.000 · 63.8 · 4.68 | not HF — IPOPT 2.0× costlier |
| H16/r1 | 5 | ✅ | 1.000 · 59.0 · 20.85 | 1.000 · 61.2 · 12.23 | trade-off: cheaper, +2.2 steps |
| H16/r8 | 1 | ❌ | 1.000 · 62.3 · 0.26 | 1.000 · 63.2 · 0.54 | not HF — IPOPT 2.1× costlier |
| H16/r8 | 2 | ❌ | 1.000 · 56.0 · 0.31 | 0.833 · 54.0 · 0.53 | not HF |
| **H16/r8** | **5** | ✅ | 1.000 · 65.7 · 3.11 | **1.000 · 58.0 · 1.57** | **HF dominates all 3 axes** |

So the H8+8 question has exactly **two** HardFlow data points, both at K=5: a trade-off at r1, a clean win at r8. Everything at K≤2 answers a different question (which *solver* is cheaper for one terminal projection — answer: SLSQP). The K=5 win comes from horizon scaling, not from the guidance: doubling H costs SLSQP ~8× per solve and IPOPT ~1.86×.

## Q2 · Is the HF paper cheating?

No. Under-reported, and their headline replicates.

- Every row of their D3IL table is re-implemented in their own harness at the same H16/T8. DPCC's published numbers are not a row, so no baseline is handicapped.
- Fair to criticise: *"follows the setup of [DPCC]"* while changing H 8→16, T 1→8, fan 4→1, episode cap 200→100, plus novel obstacles — none of it flagged.
- Their config still ships DPCC's `horizon=8, replan_steps=1` as unused defaults; H=16 arrives with the flow-guidance fork. Repo has initial-commit only, no H8 checkpoints, no sweeps. Reads as inherited, not chosen.
- `T` appears in zero equations. Safety is per-sample and holds at T=1; T=8 buys compute (~35 NLP solves/episode instead of ~250).
- Our replication (job 23565, their repo + released checkpoint, 50 episodes, **N=10 / A=0.5 → ✅ genuine**): `original` 4% safe → `hardflow_new` **100% safe, 0 violations, 50.7 steps** (paper 52.5).

At H8/T1 their safety claim should survive; "fewest steps 52.5" likely would not (the cost is `‖s_{H−1}−target‖²`, so H=16 buys 16 steps of lookahead), and "0.190 s mild overhead" would not (metric is per *replan*, so T=8 is invisible in it).

## Q3 · Final result — did HF beat DPCC in our avoiding runs?

No. DPCC wins every axis at every K, ✅ rows included. H8/replan-1, UNet MeanFlow 4.0M, 5 seeds × 20 trials = 300 episodes/cell, best tightened arm each side.

| K | | DPCC S&C · steps · ep-time s | HF S&C · steps · ep-time s | ΔS&C | Δsteps | HF cost |
|---:|---|---|---|---:|---:|---:|
| 1 | ❌ | 0.993 · 61.0 · 0.72 | 0.950 · 63.4 · 2.66 | −0.043 | +2.41 | 3.71× |
| 2 | ❌ | 0.993 · 60.4 · 1.26 | 0.900 · 67.1 · 3.37 | −0.093 | +6.65 | 2.68× |
| 5 | ✅ | 0.980 · 60.8 · 5.53 | 0.897 · 67.3 · 9.48 | −0.083 | +6.47 | 1.71× |
| 10 | ✅ | 0.973 · 61.0 · 10.40 | 0.846 · 67.4 · 17.20 | −0.127 | +6.43 | 1.65× |
| 20 | ✅ | 0.936 · 65.3 · 25.46 | 0.914 · 69.7 · 32.92 | −0.022 | +4.48 | 1.29× |

Restricted to the ✅ rows, which is the only version of this claim that is about HardFlow: **HF loses S&C by 0.02–0.13, loses 4.5–6.5 steps, and costs 1.3–1.7× more.** The K≤2 rows say only that SLSQP is a cheaper solver than IPOPT for one projection.
Measured fan-1 cell at K=2 (❌, no reconstruction): on `fm` both arms produced bit-identical rollouts — step delta exactly 0.00 — at 2.45× the cost. That is the degeneracy made visible: same algorithm, different solver, 2.45× the price.
Against the pinned DA target (`H8_K20 GaussianDiffusion aw10`, `dpcc-c-tightened` 1.000 · 70.1 · 38.81 s): DPCC on our MeanFlow checkpoint beats it at 0.993 · 61.0 · 0.72 s, and beats every HF row too. No HF row on `avoiding` improves on it.

## Q4 · Did HF really fail at H8/1, scored on `time × steps`?

Yes, on the ✅ rows, and the step half fails first.

- Steps: HF is worse at every ✅ K (+4.5 to +6.5). The gap is conservative — HF also scores lower S&C, and failed episodes terminate early and log fewer steps.
- Time: the K≥5 advantage visible in raw logs is the 4:1 fan artifact. Matched at fan 1, HF costs 1.3–1.7× more per episode at ✅ K.
- Trend worth keeping: the HF/DPCC cost ratio falls monotonically with K (1.71 → 1.29 across ✅ rows), because SLSQP per-solve cost grows with K (14.9 → 21.0 ms) while IPOPT's does not. Break-even extrapolates to K ≈ 30–40 even at H=8 — but S&C and steps would still be losing there.

## Q5 (asked) · Is H16/r8's final average step count better than our H8/1?

Yes on both arms, and the HF gap is larger once degenerate rows are excluded. Best arm at S&C ≥ 0.99 per configuration; ❌ rows are not eligible for the HardFlow line.

| arm | H8/r1 (n=300) | H16/r8 (n=6) | Δsteps |
|---|---|---|---:|
| DPCC | 0.993 · **60.41** (K2, `-t-tight`) | 1.000 · **56.00** (K2, `-t-tight`) | **−4.41** |
| HardFlow ✅ only | 0.897 · **67.31** (K5) — best-S&C ✅ is K20 at 69.73 | 1.000 · **58.00** (K5) | **−9.31** (−11.73 vs K20) |

Matched-arm across the K grid, `dpcc-c-tightened`: 72.0 / 98.0 / 64.0 at H8/r1 → 64.7 / 64.7 / 65.7 at H16/r8. Consistent with `DA_20260818` §2.2 (4 of 6 matched rows better by 7.3–33.3 steps, 1 tie, 1 worse by +1.7).
Two conditions: the H16 side is n=6, so the step delta is not statistically resolved; and r8 costs 1.6–6.0× worse worst-case per-tick latency and only works if the controller can buffer an 8-step segment.

**Where HF does earn something** — (1) H16/r8/K=5, the one dominant cell, and it is ✅. (2) Horizon scaling: per solve H 8→16 costs SLSQP ~8× and IPOPT ~1.86×, so under a latency budget with growing H, HF is the arm that survives; per-solve parity extrapolates to H≈24–26 at K=1–2 and H≈10 at K=5. (3) UAV corridor K=5 (✅): 5 wins, 0 losses (1 seed, n=10), still at 1.9–3.6× DPCC's cost — and on the same sweep K=1–2 (❌) lost 2–4, which is what degeneracy predicts.
**Retracted from the previous version of this note:** "untightened, in-loop beats post-hoc" citing the fan-matched K=2 cell (`hardflow_new-c` 0.500 vs `dpcc-c` 0.333). K=2 at A=0.5 is ❌ — no in-loop step ran. That gap is IPOPT vs SLSQP on a nonconvex terminal projection, not the paper's mechanism. **We have no ✅ evidence for the in-loop-beats-post-hoc claim on `avoiding`.**

**New this pass** — the "K=5 NLP-failure spike" from `DA_20260824` does not reproduce. On the 300-episode ladder failures are monotone in K, opposite signs by tightening: tightened 0.56% (K1) → 1.36% (K5) → 1.76% (K20); untightened 1.04% → 0.64% → 0.21%. A failed IPOPT returns a possibly-infeasible last iterate, so at K=20 tightened the hard guarantee is ~98.2% hard.

**Caveats on everything above** — all H16 rows are n=6 (cost columns are near-deterministic and safe; S&C ties and step deltas are not). The fan correction is a reconstruction validated to 2%, measured directly only at K=2. W2 (NFE re-baseline, 2026-08-24) post-dates every run here, so these NFE/time figures do not compare to post-2026-08-24 runs. A=0.5 throughout, so the ❌/✅ split above is fixed; it moves if `HFFM_ACT_THRESHOLD` changes.

**Open HF items** — (1) n=6 → n=300 at H16/r8/K=5; the whole HardFlow claim in this project rests on 6 episodes at the single ✅ H16 cell, gate registered 2026-08-18 and never run. (2) `HFFM_BATCH=4` at H16/K=5 to measure the fan correction directly — must use `-r`/`-t`, since at B=4 the `-c` key selects a 161–164-step timeout candidate. (3) Clean sweep: `avoiding`, K∈{3,5,10} (drop K≤2 or keep it only as the solver control), `fm` engine, fan 1 both arms — `mf`/`af` carry the arm-C `h=0` confound at low K. (4) H8/T1 on their repo and checkpoint, which settles Q2 with their own defaults and no code change. (5) `temp/2608/batch_va2_20260826_142750` carries HF rows on VisualMeanFlow/VisualAlphaFlow at **K=2 (❌)**, and every arm floors at S&C 0.000–0.067 including DPCC and `diffuser` — doubly unusable, no HF signal there.

## Q6 (asked) · Is their "Projection-All" our DPCC, and why is HardFlow the *cheap* arm in their table but the *expensive* one in ours?

### Part 1 — "Projection-All" is not DPCC; their citation is misfiled

They define three projection baselines (`main.tex:720`): **Projection-All** — "projecting after every sampling step"; **Projection-Late** — "projecting only in the later sampling steps"; **Projection-Relaxed** — augmented-Lagrangian iterations per step. They cite `romer2025diffusion` (= DPCC) under **Projection-All**.

That is wrong for DPCC as shipped. DPCC's default is `diffusion_timestep_threshold = 0.5` (`aux_repo/dpcc/diffuser/sampling/projection.py:8`) and it projects only when `t <= threshold · n_timesteps` (`diffusion.py:186`) — **the later half**. By HardFlow's own taxonomy, **DPCC is Projection-Late, not Projection-All.** Our arm B runs the same 0.5, so our DPCC arm corresponds to their *Projection-Late* row. Neither row is DPCC exactly: theirs are re-implementations on their flow model, at fan 1, **untightened**, against novel test-time obstacles.

### Part 2 — what their code actually runs (`aux_repo/HardFlow`, branch `d3il`)

Read from source, not inferred. All paper runs use `ode_t_steps = 10`, `warmstart_batch = 1`.

| | solves per plan | flow evals per plan | net eval path |
|---|---:|---:|---|
| `projection`, option `all` | **10** (`flow_policy.py:866-880`) | 10 | **L4CasADi bridge** |
| `projection`, option `late` | **5** (skips `k < N//2`) | 10 | L4CasADi bridge |
| `hardflow_new` (the paper's row) | **10** — `hardflow_activation="all"` (`eval_hardflow_new.sh:33`) | **20** (two per step) | **direct PyTorch** |

**The two NLPs are structurally identical.** `projection_formulate` (`:498`) and `hardflow_formulate` (`:683`) build the same `oc_dof` variable, the same `_apply_obstacle_constraints` + `_apply_dynamics_constraints` with the same `X_index_selector="projection"`, and the same solver (`ipopt`, `hessian_approximation: limited-memory`). The **only** formulation difference is a scalar weight on the cost: `0.5·‖x−x_ref‖²` versus `0.5·reg_scale·‖x−x_ref‖²·t²`.

**So HardFlow does the same number of solves of the same NLP, and twice as many network evaluations — and is still reported 1.84× faster than Projection-All.** The solve-count story is wrong; something else is going on.

**Also note what the timer excludes.** `t_start` is set *after* `self.warmstart(conditions)` in every guided method (`:810`, `:1305`). `warmstart` runs a full N=10 ODE sample plus value selection (`:753-793`). **Their Computation Time is the guidance loop only — the base generation pass is not in it.**

### Part 3 — why HardFlow is fast: its NLP is easier, not rarer

Their own three projection rows pin every unit cost, with no assumptions:

| quantity | derivation | value |
|---|---|---:|
| torch flow eval | `Original` = 10 evals, no solves = 0.060 | **6.0 ms** |
| **projection IPOPT solve** | `All − Late` = 5 solves = 0.349 − 0.236 | **22.6 ms** |
| L4CasADi flow eval | `All` − 10 solves = 0.349 − 0.226, ÷10 | **12.3 ms** |
| **HardFlow IPOPT solve** | `HF` − 20 torch evals = 0.190 − 0.120, ÷10 | **7.0 ms** |

Cross-check: `Projection-Relaxed` = 0.116 ≈ 10 L4CasADi evals with near-free augmented-Lagrangian steps (no NLP solves) — consistent with 11.6–12.3 ms/eval. The model closes on all four rows.

**The L4CasADi asymmetry is real but is *not* the explanation.** `run/eval.py:577-580` gates the bridge to `projection` / `projection_relaxed` / `hardflow`; `hardflow_new` bypasses it. But HardFlow does **twice** the evals, so eval cost is ~120 ms (20 × 6.0) against projection's ~123 ms (10 × 12.3) — **a wash**. Checked and excluded.

**The entire 0.159 s gap is per-solve IPOPT cost: 226 ms vs 70 ms — a 3.2× cheaper solve, at matched solver, matched solve count, matched NLP structure.** The cause is *what gets projected*:

| | projected point | why it costs what it costs |
|---|---|---|
| Projection-All/Late | `x_next_ref = x_k + v_k·dt` — the **noisy intermediate ODE iterate** (`:860-862`) | at early `t` this sits far from the constraint set; IPOPT needs many iterations to restore feasibility |
| **HardFlow** | `x_terminal_predicted_ref = x_next_ref + (1−t−dt)·v_next` — the **predicted clean endpoint** (`:1339`) | lies near the data manifold, so it is usually *already near-feasible*; IPOPT converges in few iterations |

The `t²` weight reinforces it: at early steps the proximal objective is nearly flat, so there is little to trade against feasibility. **This is a genuine algorithmic advantage and their speed claim on it is legitimate.** Projecting a noisy iterate is both more expensive and less useful — which is also why Projection-All scores only 0.46 safety: constraints enforced on a noisy iterate get destroyed by the remaining ODE steps.

**Our port is faithful on exactly this point** — verified in `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:712-726`: `X1_ref = X_ref + (1−tau_next)·V_next`, `nlp.solve(X1_ref, tau_next)`, `X_next = X_ref + tau_next·(X1_proj − X1_ref)`. Same predicted-endpoint projection, same reconstruction.

### Part 4 — why the sign flips in our harness: our baseline is not IPOPT

| | HardFlow's table | our arm B vs arm C |
|---|---|---|
| HF arm | IPOPT on the predicted endpoint → **7.0 ms/solve** | IPOPT on the predicted endpoint → **~30 ms/solve** |
| baseline arm | **IPOPT** on the noisy iterate → **22.6 ms/solve** | **scipy SLSQP** on the noisy iterate → **2.1–21 ms/solve** |
| result | HF **3.2× cheaper** | HF **1.4–14× more expensive** |

Both are internally consistent, and neither is wrong. HardFlow's easier NLP wins by 3.2× **against IPOPT**. It does not recover the gap against **SLSQP with analytic Jacobians**, which on this problem class is 2–10× faster than IPOPT before the easier-problem discount is applied. Their table never tests that, because every row in it is IPOPT.

Baseline strength cuts the same way: their best projection row reaches safety **0.76** untightened; our `dpcc-t-tightened` reaches S&C **0.98–1.00**. **Constraint tightening**, which their baselines do not carry, is load-bearing on every task in this note.

### Part 5 — how to cite this without overclaiming

- ❌ "HardFlow is cheaper than DPCC" — their table does not show it. Projection-All ≠ DPCC, Projection-Late ≈ DPCC-untightened, and every row is IPOPT.
- ❌ "Their speed result is an implementation artifact" — we checked the L4CasADi asymmetry explicitly and it cancels. Their advantage is real.
- ❌ "Our result contradicts theirs" — different baseline solvers, both measured correctly.
- ✅ **"HardFlow projects a predicted clean endpoint rather than a noisy intermediate iterate, which makes its NLP ~3.2× cheaper per solve at matched solver, solve count and formulation — a genuine advantage, and the reason it leads their table. On our harness the DPCC projector uses scipy SLSQP rather than IPOPT, which is fast enough on this problem that HardFlow's easier NLP does not close the gap."** Every clause is measured.

### Part 6 — "more NFE but an easier solve" — yes, and it holds in our code too

**The trade, stated plainly.** HardFlow spends **2 network evals per step** (one for the ODE step, one for the endpoint lookahead) against projection's 1, and buys a solve that is **3.2× cheaper** (7.0 ms vs 22.6 ms). On their machine that trade is strongly profitable: it pays 60 ms extra in evals to save 156 ms in solves. **More NFE, easier NLP — correct.**

**Our port makes the same trade.** Verified at `hardflow_projection.py:712-726`: `X1_ref = X_ref + (1−tau_next)·V_next` then `nlp.solve(X1_ref, tau_next)` — the predicted endpoint, not the noisy iterate. Our NFE accounting says the same thing: `K + n_active − 1` evals per plan, i.e. one extra eval per active step. Nothing about the trade is lost in the port.

**So why does it not pay off for us? Two reasons, and neither is the algorithm.**

**(1) Our baseline is not IPOPT.** Their comparison is IPOPT-on-endpoint vs IPOPT-on-iterate, so the easier-NLP discount is the *only* thing moving. Ours is IPOPT-on-endpoint vs **SLSQP**-on-iterate, and SLSQP with analytic Jacobians is the faster solver on this problem class by more than the 3.2× the easier NLP wins back.

**(2) Our cluster is disproportionately slow at IPOPT.** This is measurable, because we ran *their* code on *our* cluster (job 23565, chapter 1 Q2):

| | their machine (paper) | our cluster (their code, H16) | ratio |
|---|---:|---:|---:|
| `Original` per step | 0.060 s | 0.175 s | 2.9× slower |
| `hardflow_new` per step | 0.190 s | 0.847 s | 4.5× slower |
| ⇒ torch flow eval | 6.0 ms | 17.5 ms | **2.9×** |
| ⇒ IPOPT solve | 7.0 ms | **49.7 ms** | **7.1×** |

**Our cluster is 2.9× slower at the network but 7.1× slower at IPOPT** — the penalty lands squarely on the solve, which is the half HardFlow is made of. Our own port's ~30 ms/solve at **H8** is consistent with 49.7 ms at **H16** on the same hardware, so the port is performing as expected; the machine is the difference.

**Consequence for reading any cross-paper number:** absolute times are not comparable between their table and ours — only within-machine ratios are. Their 0.190 s and our 0.847 s are the same code doing the same work.

**The missing experiment, and it is the cheapest one in this note.** We have never run *their* baseline — IPOPT projecting the noisy iterate (their Projection-All / Projection-Late) — inside our harness. If we did, we should see **our HardFlow beat it by ~3×, reproducing their result, while still losing to SLSQP.** That single arm would show both papers are measuring the same phenomenon and would close this question.

Note this is chapter 4's rung 1 run **in the opposite direction**: not "give HardFlow SLSQP", but **"give the DPCC arm IPOPT"**. It is the more informative direction, because it reproduces their published comparison rather than manufacturing a new one.

---

# 2 · Visual avoiding — how does HardFlow perform on the hard env?

**Short answer — it has never been run there.** Exactly one visual-avoiding candidate carries arm C, and it is **K=2 at A=0.5 → ❌ degenerate**: no HardFlow-specific math executes in it.

## 2a · The one candidate with arm C

Gen16 `mf` (VisualMeanFlow, U-Net FiLM-v1), K=2, seed 6, 30 trials × 3 geometries = **90 episodes**. Unlike everything in chapter 1, **both arms ran at fan 4**, so the cost column is directly measured — no reconstruction. Data: `temp/2508/batch_avoiding_combined_20260825_143212`, folder `H8_K2_…A0.5_…VisualMeanFlow_VTrue_mpc4_filmv1_Emf`.

| arm | TR (the failing geometry) | TL | BH | pooled S&C | ep-time s |
|---|---|---|---|---:|---:|
| `dpcc-c-tightened` | 0.767 | 1.000 | 1.000 | 0.922 | **1.73–2.28** |
| `dpcc-r-tightened` | 0.833 | 1.000 | 1.000 | 0.944 | **1.89–2.08** |
| `dpcc-t-tightened` | 0.833 | 1.000 | 1.000 | 0.944 | **1.72–2.09** |
| `hardflow_new-c-tightened` | **0.867** | 1.000 | 1.000 | **0.956** | 7.01–7.58 |
| `hardflow_new-t-tightened` | 0.800 | 1.000 | 1.000 | 0.933 | 7.13–7.45 |
| `hardflow_new-r-tightened` | 0.700 | 1.000 | 1.000 | 0.900 | 5.96–7.71 |

Read it as **+0.012 pooled S&C for the best HF rule over the best DPCC rule — one episode in 90 — at 3.3–3.6× the cost.** It does not survive the obvious check: HF's own rule-to-rule spread on TR (0.700 / 0.800 / 0.867) is **wider than its margin over DPCC**, so "HF's best rule beats DPCC" is rule selection, not a method effect. HF's worst rule (0.900 pooled) sits below DPCC's worst (0.922).
Where the arms coincide, the degeneracy is visible directly: BH untightened, both give 1.000 · 51.0 steps, HF at 3.4× the price; TL untightened, both give 0.000 · ~81.7 steps · ~0.9 violations, HF at 4.7×.

## 2b · The IPOPT-failure defect

On TL untightened, HF's IPOPT **fails 12.5–13.5 % of solves** (0.00–0.5 % everywhere else on this task). A failed solve returns a possibly-infeasible last iterate, so on that cell the hard guarantee is off for roughly one plan in eight. Untightened `dpcc-c` also scores 0.00 there at `succ = 1.00` — the policy reaches the goal every time and clips the obstacle every time — so **tightening is load-bearing on this env for both arms**, not an optimisation.

## 2c · Net on visual avoiding

What exists is a fan-matched, well-sampled (90-episode) measurement that **IPOPT costs 3.3–3.6× more than SLSQP for one terminal projection, buys about one episode in ninety, and fails 13 % of its solves on the hardest untightened cell.** `SNAPSHOT_20260826_visual_avoiding_env_status.md` §5.1 reports the 0.87-vs-0.83 TR row as `mf`'s best row without flagging the K=2 degeneracy — that row needs the ❌ tag. A real answer costs one eval: **rerun that candidate at K ≥ 5** (or A = 1.0, which makes K=2 genuine), same seed and trials; arm C is already wired in `config/visual_avoiding_mix_eval.yaml`. Until then the `hf_act_threshold` sweep that snapshot asks for in §8 item 7 would sweep a knob that is not doing anything.

*Follow-up asked inside this question: check `mf` / `af` / `fm` — which models have the HF projector enabled?*

## 2d · Cross-engine coverage — who has arm C at all?

Sweeping every folder in the state-`avoiding` batch that carries `hardflow_new-*` rows:

| engine | backbone | K with HF rows | genuine (A=0.5, K≥5)? |
|---|---|---|---|
| `mf` (MeanFlow) | **UNet** — architecture-matched | 1, 2, 5, 10, 20 (+ A sweep at 5, 10) | ✅ K = 5, 10, 20 |
| `mf` | `mf_dit` | 1, 2, 5, 10, 20 | ✅ K = 5, 10, 20 |
| `af` (AlphaFlow) | SiT | 1, 2, 5, 10, 20 | ✅ K = 5, 10, 20 |
| `af` | UNet | 1, 2, 5, 10 (pre-Fix_9, no `A` token — W4 risk) | ⚠️ provenance |
| **`fm` (naive FM)** | UNet | **K=2 only** (the fan-parity cell, different batch) | ❌ **none** |
| `imf`, `diffusion` | — | **none** | ❌ never run |
| `mf` visual avoiding | UNet-FiLM | K=2 only | ❌ |

**`fm` is the gap that matters.** `DA_20260824` §6 established that arm C queries the instantaneous field (`h = 0`) while arms A/B use the trained interval field (`h = dt`) — a confound that hits `mf`/`af`/`imf` and fades as K rises. **`fm` is the only engine free of it, and it is the one engine with no genuine-K HF run on this task.**

## 2e · Does the engine change the verdict?

All rows ✅ genuine, both arms re-priced at fan 1, `-t-tightened` each side, n=15 cells.

| engine | bone | K | DPCC S&C · steps · ep-t | HF S&C · steps · ep-t | ΔS&C | Δsteps | HF cost |
|---|---|---:|---|---|---:|---:|---:|
| `af` | SiT | 5 | 1.000 · 67.8 · 4.87 | 1.000 · 69.4 · 7.26 | +0.000 | +1.57 | 1.49× |
| `af` | SiT | 10 | 0.967 · 64.4 · 8.41 | **1.000** · 68.5 · 12.85 | **+0.033** | +4.10 | 1.53× |
| `af` | SiT | 20 | 1.000 · 68.7 · 22.21 | 0.967 · 74.7 · 27.22 | −0.033 | +6.03 | 1.23× |
| `mf` | mf_dit | 5 | 0.967 · 70.2 · 5.80 | 0.967 · 72.2 · 8.72 | +0.000 | +2.03 | 1.50× |
| `mf` | mf_dit | 10 | 0.967 · 71.0 · 10.88 | 0.933 · 71.9 · 15.49 | −0.033 | +0.90 | 1.42× |
| `mf` | mf_dit | 20 | 0.967 · 71.4 · 25.87 | 0.933 · **68.4** · 29.23 | −0.033 | **−3.00** | 1.13× |
| **`mf`** | **UNet** | 5 | 0.980 · 60.8 · 5.53 | 0.897 · 67.3 · 9.48 | −0.083 | +6.48 | 1.71× |
| **`mf`** | **UNet** | 10 | 0.973 · 61.0 · 10.40 | 0.846 · 67.4 · 17.21 | −0.127 | +6.42 | 1.65× |
| **`mf`** | **UNet** | 20 | 0.936 · 65.2 · 25.46 | 0.914 · 69.7 · 32.92 | −0.021 | +4.48 | 1.29× |

**No engine rescues HardFlow, and the pattern is a backbone pattern, not an engine one.** HF looks least bad on `af`/SiT (one +0.033 row), middling on `mf_dit` (one −3.0-step row, the only cross-engine cell where HF shortens the trajectory), and worst on **`mf`/UNet — which is the architecture-matched comparison, i.e. HF looks worst exactly where the comparison is fair.** HF costs more in all nine cells (1.13–1.71×) and takes more steps in eight of nine.

## 2f · The `A` sweep — what does HardFlow's guidance itself buy?

`mf`/UNet ships a sweep of the activation threshold at fixed K, seed 6 (n = 3 cells = 6 episodes). This is the cleanest experiment in the corpus: same checkpoint, same K, same seed, **only the number of genuine guidance steps changes.**

| K | A | n_genuine | | HF S&C · steps · ep-t | NLP solves | DPCC control (A-independent) |
|---:|---|---:|---|---|---:|---|
| 5 | 0.0 | 0 | ❌ | 1.000 · 68.5 · **5.19** | 139 | 1.000 · 59.3 · 13.07 |
| 5 | 0.1 | 0 | ❌ | 1.000 · 68.5 · 5.23 | 139 | 1.000 · 59.3 · 13.16 |
| 5 | 0.25 | 1 | ⚠️ | 1.000 · 68.2 · 7.08 | 277 | 1.000 · 59.3 · 13.08 |
| 5 | 0.5 | 2 | ✅ | 1.000 · 67.7 · **8.82** | 412 | 1.000 · 59.3 · 13.09 |
| 10 | 0.0 | 0 | ❌ | 1.000 · 69.0 · **8.55** | 140 | 1.000 · 64.2 · 23.29 |
| 10 | 0.1 | 0 | ❌ | 1.000 · 69.0 · 8.56 | 140 | 1.000 · 64.2 · 23.34 |
| 10 | 0.25 | 2 | ✅ | 1.000 · 68.0 · 11.78 | 414 | 1.000 · 64.2 · 23.11 |
| 10 | 0.5 | 4 | ✅ | 1.000 · 69.3 · **16.03** | 703 | 1.000 · 64.2 · 23.27 |

Two things fall out. **First, the degeneracy formula is confirmed on a second axis:** solve counts go 1 : 1 : 2 : 3 at K=5 and 1 : 1 : 3 : 5 at K=10, exactly `n_active`. The DPCC control is flat throughout, confirming `A` touches only arm C.
**Second — turning HardFlow's guidance on costs 1.70× (K=5) and 1.87× (K=10) and buys nothing measurable here:** S&C is 1.000 in every cell, and steps move −0.8 (K=5) and +0.3 (K=10). ⚠️ **This is ceiling-limited** — both arms are already at 1.000 on 6 episodes, so the sweep *cannot* detect an improvement even if one exists. What it does establish is the price of the guidance in isolation, and that the degenerate rows are ~1.7–1.9× cheaper than the genuine ones on the same checkpoint. **Rerunning this sweep on a harder cell with headroom is the single highest-value HF experiment available — it isolates the guidance with everything else nailed down.**

---

# 3 · Visual aligning — how does HardFlow perform there?

**Short answer — no better, and the one significant test runs against it.** Same degeneracy caveat: both candidates carrying arm C are **K=2 at A=0.5 → ❌**. Answered independently in `SNAPSHOT_20260823_visual_aligning_env_status.md` §5; the numbers below are re-derived from `temp/2608/batch_va2_20260826_142750/va2_aggregated_long.csv` and agree with it to 4 decimals.

## 3a · S&C is the wrong metric here

Every arm — HF, DPCC, and the unguided `diffuser` alike — scores `succ` 0.000–0.077 with 380–400 steps against a 400 cap. The task is essentially unsolved by all methods, so success carries no signal. The informative metrics are **distance to target**, **violation magnitude**, and cost. (This is what my first pass got wrong: I read S&C, saw a floor, and reported "no signal" when the DA had already answered the question on distance.)

## 3b · The paired comparison (n = 30 contexts)

Arm C exists on two candidates: C14 `mf` v1 K2 and C6 `af` v1 K2.

| model | pool | best DPCC (dist · viol · ms) | best HF (dist · viol · ms) | Δ dist | HF wins | cost |
|---|---|---|---|---:|---:|---:|
| `mf` ❌ | untightened | `dpcc-t` 0.2867 · 66.2 · 52.8 | `hardflow_new-c` 0.3249 · 60.3 · 194.0 | **+0.0382** | 10/30 | 3.7× |
| `mf` ❌ | tightened | `dpcc-r` 0.3145 · 5.2 · 48.8 | `hardflow_new-t` 0.3112 · 0.38 · 167.9 | −0.0033 | **9/30** | 3.4× |
| `af` ❌ | untightened | `dpcc-t` 0.3378 · 52.6 · 53.4 | `hardflow_new-r` 0.3316 · 31.4 · 177.5 | −0.0063 | 12/30 | 3.3× |
| `af` ❌ | tightened | `dpcc-t` 0.3486 · 1.23 · 49.1 | `hardflow_new-c` 0.3128 · 5.19 · 182.9 | −0.0358 | 14/30 | 3.7× |

HF lands within **±0.04 m** of the best DPCC arm in all four blocks, splits the clean `<15 cm` tail 2–1–1, and costs a flat **3.3–3.7×**. **The only sign test that clears 0.05 runs against HardFlow** — C14 tightened, 9 of 30.

## 3c · The constraint row — DPCC owns it

`dpcc-t` + tightening on `mf` reaches **`viol` = 0.00 at 48.9 ms**. HF never matches that: its best is `hardflow_new-t` at 0.38 violations and 167.9 ms — **worse constraint satisfaction at 3.4× the price**. On `af` tightened it is not close: `dpcc-t` 1.23 violations vs HF's 3.62 / 5.19 / 5.69 across the three rules, all at ~180 ms.

## 3d · Where HF helps — and why it is not a reason to adopt it

Matched-rule and untightened, HF rescues DPCC's *weak* selection rules: `-c` on `mf` goes 0.4094 → 0.3249 (**−0.084 m**, violations 69.5 → 60.3) and `-r` on `af` goes 0.4220 → 0.3316 (**−0.090 m**, violations 34.7 → 31.4). Where arm B already selects well (`-t`), HF is slightly worse. **Its benefit scales inversely with how good DPCC's rule already is — it rescues a bad rule rather than beating a good one.** Picking the good rule is free; HF costs 3.3–3.7×.

## 3e · Net on visual aligning

No, HardFlow does not beat the DPCC projectors here — and because both candidates are K=2 at A=0.5, this is again IPOPT-vs-SLSQP rather than a HardFlow test. `SNAPSHOT_20260823` §5 reaches the same verdict without flagging the degeneracy; its rows need the ❌ tag too. Both that snapshot and this note agree on what is missing: **HardFlow at a *lower* projection threshold than DPCC has never been run**, which is the comparison the benchmark hierarchy actually asks for — and it should be run at K ≥ 5, not K = 2.

---

# 4 · Swap HF's NLP solver for DPCC's — would the degenerate rows then agree, and should the diff be ~0?

**Short answer: worth doing, but no — the diff would shrink, not vanish, because the solver is only one of five differences. And we already have one cell where the two arms agree *bit-for-bit*, which points the finger somewhere else.** This was raised in `DEGENERACY_HardFlow_at_low_K.md` §0.3 as "the gate that would settle it" — offline, cheap, **still not built**.

## 4a · What arm C actually is — not a projector

`hardflow_new-*` does not take arms A/B's trajectory and project it. It **replaces the sampler**: `HardFlowSampler.sample()` (`flow_matcher_v3_meanflow/sampling/hardflow_projection.py:592`) runs its **own Euler ODE loop** — `for k in range(K): V = self._velocity_batch(...); X_ref = X + V*dt` — and fires the NLP *inside* that loop. Arms A/B call the model's own `p_sample_loop` instead. So arm C differs from arms A/B in **two independent places: the trajectory it generates, and the solve applied to it.**

Whether the first difference is real depends on the engine:

| engine | arm A/B field query | arm C field query | same base trajectory? |
|---|---|---|---|
| **`fm`** (naive FM) | `v(x,t)` — instantaneous | `v(x,t)` — `h` is meaningless for FM | ✅ **yes — identical Euler loop** |
| **`mf` / `af`** (MeanFlow family) | trained **interval** field, `h = dt` (`mf_diffusion.py:278`) | **`h = 0`**, instantaneous (`hardflow_projection.py:583`) | ❌ **no** |

On `mf`/`af` the gap is not subtle at low K. MeanFlow's entire point is the few-step **jump**: at K=1 arms A/B take one *trained* jump from `t=0` to `t=1`; arm C takes a first-order Euler extrapolation across `Δt = 1` using the instantaneous velocity. The `h=0` query is mathematically defensible — the MeanFlow identity gives `u(x,t,0) = v(x,t)` exactly, and the code comment says so — but it **discards the jump that makes MeanFlow few-step**, so at low K arm C is effectively sampling at naive-FM quality. The two converge as K rises.

**So your model is exactly right on `fm`, and only on `fm`.** There, arm C and arms A/B integrate the same ODE and the *only* difference is the solve — which is why the `fm` K=2 parity cell came out bit-identical (4b). On `mf`/`af`, "same trajectory, different projector" is false.

## 4b · Why `h=dt` ≠ `h=0`, and why FM is exempt

The objection "MeanFlow's average velocity is just a training target, at sampling we use the instantaneous `v` anyway" is the natural reading, but the sampler does not do that. `mf_diffusion.py:211` sets `h_batch = dt` once and uses it at **every** step including the last (`:278`); `h=0` never appears in arms A/B.

The two updates are different objects:

| | update | error |
|---|---|---|
| arms A/B (MeanFlow) | `x ← x + dt · u(x, t, dt)` | **exact over `[t, t+dt]`** if `u` is the true interval average — this is the few-step *jump* MeanFlow is trained for |
| arm C | `x ← x + dt · v(x, t)` | plain first-order Euler, local error `O(dt²)` |

`u(x,t,0) = v(x,t)` is a **training identity** — it anchors the target at `r = t` — not what the sampler evaluates. At K=1 (`dt = 1`) arms A/B take one trained jump `t: 0 → 1`; arm C linearly extrapolates across the entire path. They converge as `dt → 0`, which is exactly the "arms agree at high K" pattern in the data.
**For `fm` there is no such split:** `_predict_velocity` returns `v(x,t)` regardless of `h`, so arm C's Euler loop and arms A/B's are the same computation. Hence the bit-identical `fm` cell.

## 4c · The solve-side differences — corrected

My first pass listed five and overstated two. Re-checked against the source:

| # | claim | verdict |
|---:|---|---|
| 1 | **solver**: SLSQP + analytic Jacobians (`projection.py:143-151`) vs IPOPT + limited-memory BFGS via `solve_limited()` (`hardflow_projection.py:209, 342`) | ✅ **real** |
| 2 | **`s_0` scope**: HF's `dof = H·T − state_dim` — `s_0` is *not a decision variable at all* (`Layout.state_index(0)` raises). DPCC keeps all `H·T` variables and pins only the `deriv` `x_idx` dims of step 0 via `b[counter·H] = s_0[x_idx]`. | ✅ **real, but narrower than I wrote** — it concerns only the *non-`x_idx`* state dims of step 0 (velocities), free in DPCC, absent in HF |
| 3 | **box bounds**: DPCC's `Bounds(−5, +5)` | ⚠️ **overstated.** It is a loose *numerical* guard in **normalized** coordinates; normalized data sits near ±1, so it is essentially never binding. Both arms carry the real task bounds (`ub`/`lb`) from the shared `constraint_list`. Demote to a footnote. |
| 4 | **sphere encoding**: normalized-pre-substituted vs unnormalized | ❌ **not a known difference** — it is an *unverified assumption*. Both apply the obstacle over `t = 1…H−1`; whether the two encodings describe the same set has never been checked by any gate. Belongs on the to-verify list, not the difference list. |
| 5 | **failure fallback**: HF keeps IPOPT's last iterate, *not guaranteed feasible* (`:346-360`); DPCC's breaker returns the trajectory **unprojected** | ✅ **real, and both are silent** |

So the honest count is **two solid differences (1, 5), one structural-but-narrow (2), one overstated (3), one that was never a difference (4)** — plus the sampler split in 4a, which is the large one.

## 4d · 🔴 Fidelity gap — we have never run HardFlow's dynamics

HardFlow's formulation constrains the plan with a least-squares-**fitted linear model** `s' = A s + B a + c`. Our port implements it — `dynamics_mode='linear_fit'`, described in-code as "HardFlow-faithful mode" (`hardflow_projection.py:293-305`) — but **every shipped config uses `dynamics_mode: deriv` with `linear_dynamics_path: null`** (`config/meanflow_projection_eval.yaml:132-133`, `config/visual_avoiding_mix_eval.yaml:124-125`). `deriv` is **DPCC's** integrator constraint `x[t+1] = x[t] + dt·dx[t]`.

So arm C as run is: **HardFlow's sampler + HardFlow's `τ²`-weighted proximal NLP schedule, wearing DPCC's dynamics constraint.**

That cuts two ways and both should be said:
- **It is deliberate and defensible.** The whole point of arm B vs arm C is to vary one thing; the UAV wiring says so explicitly — *"if the two arms enforced different constraint sets the comparison would be void."* Sharing `constraint_list` is what makes the comparison legitimate.
- **But it means HardFlow-as-published has never run in our harness.** Every number in chapters 1–3 is HardFlow's *guidance schedule* under DPCC's constraint model — never the paper's algorithm end to end. The published-repo replication (chapter 1, Q2) is the only place the real thing ran, and that was their code, their checkpoint, no comparison to our DPCC.

**Flipping it is one config line plus a fitted `.npz`** (`logs/<env>/dynamics_gen12/linear_model_H8_mpl150.npz`). It should be run as a *third* arm, not as a replacement for the current one — the matched-constraint arm is what keeps arm B vs arm C interpretable.

## 4e · The evidence on hand points at the sampler, not the solver

In the fan-parity run (`DA_20260824_mpc1_parity_MF_vs_FM.md`, K=2, fan 1 both arms) — see chapter 1, Q3 — **on the `fm` engine the two arms produced bit-identical rollouts**: step delta exactly 0.00 (71.0 / 61.5 / 62.5 in both), zero violations on both sides, at 2.45× the cost. `fm` is the one engine where arm C and arms A/B integrate the *same* ODE (4a). On `mf` in the same run they diverge (HF loses TR 0.50 vs 1.00).
Read together: **when the base trajectory is shared, differences 1, 2 and 5 produced no observable difference at all; when the sampler differs, the arms diverge.** That makes the *sampler* the primary suspect and the solver the secondary one — the opposite of the intuitive ordering. (Caveat: both arms logged zero violations on that cell, so the constraint may simply have been slack, in which case both projectors returned the reference untouched and the cell tests nothing. The offline gate in 4f distinguishes those two readings, and nothing else does.)

## 4f · The ladder, cheapest first

Each rung removes one difference; watch the residual. ⚠️ **Scope note:** "smallest term" below refers to the **trajectory** — the solver does not change *what* is produced (`fm` rollouts were bit-identical). On **cost** the solver is the *largest* term: ~81 % of an H8 IPOPT solve is fixed per-call overhead, so IPOPT costs ~30 ms where SLSQP costs 2.1 ms ([audit §0.1](./AUDIT_20260827_hardflow_paper_timing_and_baselines.md)). ⚠️ **Superseded ordering — see 4i, and see Q6 Part 6 for the *direction*: giving the DPCC arm IPOPT reproduces their published comparison, which is more informative than giving HardFlow SLSQP.** Revised priority: rung 0 stays first, but the `h = dt` fix in 4i outranks rungs 1–2, which chase the smallest term.

| rung | what to run | cost | what it settles |
|---:|---|---|---|
| **0** | **The offline projector diff — build the §0.3 gate.** One fixed `X_ref` batch, run both projectors, report `‖Π_HF − Π_DPCC‖`, the max constraint residual of **each** output, and `nlp_failures` / `last_proj_skipped`. | **no rollout, minutes** | Is either output *infeasible*? If yes → bug, stop here; §8.3's `2/6 vs 6/6` stops being mysterious. If both feasible → the gap is local-minimum choice, and rung 1 becomes worth it. |
| 1 | **The solver swap** — route the degenerate terminal step through DPCC's `Projector`, or add a solver knob. | small code change ×6 copies | Isolates solver from formulation. Expect the gap to shrink, not close. |
| 2 | **Formulation alignment** — pin `s_0` the same way, add/remove the box, unify the sphere coords. | larger | Should reach ~0 up to local-minimum multiplicity. If it does, "degenerate ≡ DPCC" is proven operationally. |

**Do rung 0 first.** It costs no cluster time, needs no code swap, and it can *end* the question: if one projector is returning infeasible output, that is a bug worth more than the whole comparison, and rungs 1–2 are unnecessary to find it.

## 4g · Implementation note

The solver is hardcoded — `self.opti.solver('ipopt', opts)` (e.g. `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:209`); there is a `solver_opts` passthrough but no solver-*name* knob. Per the repo's sibling convention this file exists in **six copies** (`flow_matcher_v3_{meanflow,alphaflow,hardflow}`, `mix_uav`, `mix_visual_{aligning,avoiding}`), so rung 1 is a six-way mirrored edit — which is another reason to spend rung 0 first.

## 4h · What it buys

If the degenerate rows reproduce the DPCC rows exactly once the code paths are unified, then every ❌ row in this note is *provably* a relabelling rather than a separate measurement, and — more useful — **the `A` sweep (2f) becomes a clean price tag on HardFlow's guidance alone**, with the projector held identical. That is the experiment that would finally answer "what does HardFlow's in-loop guidance actually buy", which nothing in this corpus currently answers.

## 4i · Is `h=0` a bug on `mf`/`af`? No — but there is a one-line fix

**Not a bug.** HardFlow is defined on an ODE step `x_{i+1} = x_i + v(x_i, t_i)·Δt + u_i·Δt`; it *requires* an instantaneous velocity field. The MeanFlow identity `u(x,t,0) = v(x,t)` holds exactly, so `h=0` hands HF precisely the field its math expects. **As an implementation of HardFlow, arm C is faithful on this point.**

**And "mf/af is just a velocity provider to HF" is exactly what the code does.** The field is a black box `v = f(x,t)` evaluated *outside* the solver (module docstring `:12`); it reaches the NLP only through `X_ref = X + V·dt`, and `solve()` takes just `(x1_ref, tau)`. Nothing in the NLP knows or cares which engine produced `V`.

**The cost.** Querying at `h=0` converts a MeanFlow model into a naive-FM model at sampling time — the few-step jump you trained for is discarded and replaced by many-step Euler. Severe at K=1–2, negligible by K=20. So on `mf`/`af`, **arm B vs arm C is not "same generator, different constraint handling"**; it is *MeanFlow-as-designed* vs *MeanFlow-downgraded-to-FM*, **plus** different constraint handling. Two things move at once. On `fm` only one moves — which is precisely why `fm` came out bit-identical (4e) and `mf` did not.

**The fix — one line, and it supersedes the 4f ladder as the cheapest high-value experiment.** In `_velocity_batch`, query `h = dt` instead of `h = torch.zeros_like(t)`. Because the field is black-box and enters only via `X_ref`, no NLP code changes.

| arm | field query | what arm B vs arm C then isolates |
|---|---|---|
| **C** (as run, all chapters above) | `h = 0` | constraint method **+ generator downgrade** — confounded on `mf`/`af` |
| **C-jump** (proposed) | `h = dt` | **constraint method only** — base trajectory identical to arms A/B |

C-jump is a deliberate deviation from the paper, which assumes an instantaneous-velocity model, so it must be labelled **"MeanFlow-native HardFlow", not HardFlow**. But it is the arm that would actually answer *"does HardFlow's in-loop guidance beat DPCC's post-hoc projection on a MeanFlow backbone"* with nothing else moving.

**Revised priority for chapter 4.** The three candidate experiments, cheapest-first and now re-ranked:

| rank | experiment | cost | why this order |
|---:|---|---|---|
| **1** | **4f rung 0** — the offline projector diff (`‖Π_HF − Π_DPCC‖`, per-output constraint residual, `nlp_failures`) | no cluster time | Can *end* the question: if either projector returns infeasible output that is a bug worth more than the whole comparison. |
| **2** | **C-jump (`h = dt`)** | 1 line ×6 sibling copies + one eval | Removes the *largest* difference (the sampler, 4a/4b), not the smallest. Makes every `mf`/`af` row in chapters 1–3 interpretable for the first time. |
| **3** | 4f rungs 1–2 — solver swap, then formulation alignment | 6-way edit, larger | Chases differences 1/2/5, which produced **zero** observable effect on `fm` (4e). Smallest term; do last. |

**Not on this list but still open:** `dynamics_mode: linear_fit` (4d) — that one answers a *different* question ("is HardFlow-as-published better?") and belongs as a third arm, not as a fix to the current one.

---

# 5 · Does the V_A frame/GIF recording inflate `avg_time` and distort the results?

## 5a · Verdict — no run wasted, no number retracted, no fix required

| question | answer |
|---|---|
| Does recording enter `avg_time`? | **No**, in either visual eval. |
| Are chapters 2–3 wasted? | **No.** Every HF-vs-DPCC ratio stands as reported. |
| Is a fix needed? | **No.** One optional code tidy-up, no re-run. |

## 5b · Visual avoiding (chapter 2)

The timer wraps *only* the `policy(...)` call (`eval_mix_visual_avoiding.py:815-820`). The observation render (`:810-812`), the recorder (`:843`), and all `savefig` calls (`:1004+`) are outside it. **There is no GIF code in this eval at all** — 0 matches for `gif`/`imageio`/`save_video`/`mp4`. Recording on/off cannot move `avg_time`.

## 5c · Visual aligning (chapter 3)

A recorder exists and is on by default (`--record`, default `all`, `:2716`), but per-step frame capture runs at `:2267` — *before* the replan timer opens at `:2397` — and `imageio.mimsave` runs at rollout end (`:1687`). Also outside.

## 5d · The one real imperfection — and it cuts in our favour

The aligning timer spans ~170 lines and does include bookkeeping the avoiding eval excludes: candidate-array copies, a GPU→CPU sync (`action_traj[0].detach().cpu().numpy()`), and a diag block that writes a file every 50th replan. That overhead is **common-mode** — both arms pay it — and adding a constant to both sides *compresses* a ratio. So chapter 3's "HF costs 3.3–3.7×" is **conservative**; the true multiple is at least that. Nothing to retract.

## 5e · Two things to know, neither blocking
1. **Do not compare `avg_time` between chapters 2 and 3** — the two evals time different spans, so cross-env cost numbers are not the same quantity. Within a chapter, arm-vs-arm is valid.
2. **`bp_cam.get_image()` is excluded from `avg_time`** — that is the visual *observation* render, a mandatory deployment cost, not recording. So chapter 2's visual `avg_time` understates true per-step cost, equally for all arms. Fine for arm-vs-arm; wrong for any "visual planning costs only X ms" claim.

## 5f · Optional tidy-up (no GPU)

narrow the aligning timer to the sampler + candidate-selection call so both evals report the same quantity.
