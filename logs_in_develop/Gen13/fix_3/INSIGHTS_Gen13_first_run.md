# Gen13 — FIRST FULL RUN: results, verdict, and insights

**Date:** 2026-07-19
**Jobs:** pipeline `23578` → train `23579` → eval `23580` (chained via `afterok`, node i6-gpu-1, env `hardflow_clone`, git `ab3c624`)
**Logs:** `temp/HF_iMF_first_run/` (3 files)
**Headline:** the pipeline worked end-to-end and produced the full E1–E4 matrix. **iMF does NOT yet beat FM** — it fails the non-negotiable safety criterion (94% vs 100%) — but it achieves a **4.6× NFE reduction**, and the K1→K2 trend points at a concrete, cheap path to closing the gap.

---

## 1. What ran

| Stage | Outcome |
|---|---|
| Orchestrator (fix_3) | ✅ submitted + chained both jobs in **<1 s**, exited clean — the `PartitionTimeLimit` bug is gone |
| Gates G0/G1 | ✅ **ALL GATES PASSED** on cluster (sign gate: mean\|x\| 2.089, K1~K2 W1 0.067) |
| Training | ✅ 100k steps in **4h 12m** (~3.95 it/s), all 10 checkpoints + `metrics.csv` written |
| Eval E1–E4 | ✅ all four 50-episode runs completed (1m03s / 1m14s / 3m24s / 3m44s; whole job 9m26s) |

## 2. Results (50 episodes each)

| Run | Method | K | Success | Safety | Violations | Steps | NFE/plan |
|---|---|---|---|---|---|---|---|
| E1 | iMF original | 1 | 0% | 0% | 50 | 18.9 | 3 |
| E2 | iMF original | 2 | 2% | 2% | 49 | 21.7 | 5 |
| E4a | iMF hardflow_new | 1 | 80% | 80% | 10 | 56.6 | 5 |
| **E3** | **iMF hardflow_new** | **2** | **94%** | **94%** | **3** | **51.7** | **9** |
| B1 *(frozen FM)* | FM original | — | 4% | 4% | 48 | 19.9 | ~11 |
| **B2 *(frozen FM)*** | **FM hardflow_new** | — | **100%** | **100%** | **0** | **50.7** | **~41** |

NFE accounting independently verified against the code (warmstart + 2×loop + x1_estimate): FM = 10+20+11 = 41; iMF K2 = 2+4+3 = 9 — both match the logged values exactly, so the instrumentation is trustworthy.

## 3. Verdict against the plan §5 criteria

| Criterion | Target | E3 actual | Result |
|---|---|---|---|
| 1. Safety parity (**non-negotiable**) | 100% safe, 0 violations | 94%, **3 violations** | ❌ **FAIL** |
| 2. Efficiency win | NFE/compute < B2 | **9 vs 41 NFE (4.6×)**; 2 vs 10 NLP solves (5×) | ✅ PASS |
| 3. Quality not degraded | steps within ~±20% of 50.7 | 51.7 (+2%) | ✅ PASS |

**→ "iMF superior to FM in HardFlow" is NOT declared.** Criterion 1 is explicitly non-negotiable (constraint satisfaction is table stakes), and 3 violations in 50 episodes fails it. The efficiency result is nevertheless large and real.

## 4. THE key insight — K is a *constraint-projection* knob on the guided path, not just an NFE knob

This is the most important finding, and it invalidates a planning assumption (D7):

- Plan D7 set K∈{1,2} as primary because **Gen3v4 showed iMF is K-invariant** — more steps don't improve *generation* quality.
- That reasoning came from **unguided generation** and **does not transfer to the guided path.** In `hardflow_new_imf`, `K == ode_t_steps` also sets **the number of prox-NLP constraint projections** (one per step).
- Evidence: K1→K2 moved safety **80% → 94%** (violations 10 → 3). That is *not* the generative field improving (it's K-invariant); it is **twice as many constraint projections**.
- FM's B2 gets 100% because it runs **10 projections** (`ode_t_steps=10`), not because its field is better.

**Implication:** the safety gap is almost certainly a *projection-count* deficit, not a fundamental iMF weakness — and it is cheap to test. iMF at K=4/K=5 would still use ~17/21 NFE (well under FM's 41) while doubling/2.5×-ing the projections. **This is the single highest-value next experiment.**

## 5. Secondary insights

**a) Unguided iMF is worse than unguided FM (0–2% vs 4%) — expected, and pre-declared acceptable.** The plan (§5, §7) predicted the 96-demo data ceiling would leave the raw `u`-field coarse. Confirmed quantitatively: `raw_mse_u` plateaus ≈10–15 summed over 96 dims (H16×6) → **≈0.37 per-dim residual**, vs Gen3v4's ≈0.25/dim on the easier H8 task. The field *is* coarse. The claim under test was always *guided efficiency*, not raw-field beauty — so this does not count against the method.

**b) Training met its gate, but only just, and it is spiky.** `raw_mse_u` 61.5 → ~13 = **4.7× drop** (G2 wanted ≥3×) ✅. But: `a0_mse` settled ≈0.2–0.35 vs the plan's <0.15 reference ⚠️, and large spikes persist to the end (132.6 at step 98,400; 46.7 at 91,200). Spikes are the known JVP-predicted-v-tangent variance (Gen3v4 §1) — not divergence — but combined with the marginal `a0`, this suggests **training headroom remains**: the LR cosine-annealed to 0 while still spiky, exactly the Gen3v4 §5 "froze on a noisy plateau" pattern. A longer run or constant-LR tail is a *secondary* lever (behind K).

**c) The adaptive loss behaved exactly as documented** — pinned at 1.996–1.999 for the entire run. Anyone reading that curve alone would conclude "not learning". The `raw_mse_*` logging added in coding_1 is what made this run interpretable; keep it.

**d) Guided iMF episodes run ~2.7× longer than unguided (18.9 → 51.7 steps)** — same signature as FM (19.9 → 50.7): trajectories stop dying at the first obstacle and actually complete the task.

## 6. ⚠️ Engineering issues found (both are MY bugs, not cluster problems)

**(i) Training log is 98% tqdm garbage — the fix_2 progress-bar fix did not cover training.**
`23_57_33_hf_imf_train_23579.log` is **4.6 MB in 586 lines**; three single lines are **791k / 865k / 742k characters**. Cause: `run/train_imf.py` still wraps the 100k-step loop in `tqdm.tqdm(...)`, which under `submit.sh` (redirected to a file, not a tty) dumps every one of 100,000 updates as raw text. fix_2 only fixed `run/eval_imf.py`. **This directly violates the SLURM memory rule written in the previous turn.** Fix: gate the bar behind `sys.stdout.isatty()` (the per-200-step `[ train_imf ]` lines already carry all the real information — the bar adds nothing in batch mode).

**(ii) Eval log is clean per-line but 70k lines, ~44% IPOPT solver output.**
The fix_2 quiet-episode fix worked (max line length **213 chars**, one tidy line per episode ✅). The remaining bulk is IPOPT's per-solve report from `solver_print_level=5`. Lowering it to `0` for the iMF eval scripts would cut the log ~2× and lose nothing (the "Overall NLP error" values were all ~1e-9/1e-10, i.e. the solver is healthy and uninformative). Deliberately *not* changed yet: the FM baselines used level 5, so this is a comparability-cosmetics call for the user.

## 7. Recommended next steps (ranked)

1. **Run E5: `hardflow_new_imf` at K=4 and K=5.** Directly tests §4's projection-count hypothesis. One eval job, ~10 min, no retraining:
   ```bash
   IMF_METHODS="hardflow_new" IMF_KS="4 5" \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_imf_hardflow.sh
   ```
   If K=4/5 reaches 100% safe at ≤21 NFE, **iMF wins outright** (still ~2× cheaper than FM's 41) and Gen13's claim flips to PASS.
2. **Fix the training tqdm bloat** (§6i) before the next training run.
3. Only if (1) plateaus below 100%: revisit training (constant-LR tail / longer budget per §5b), since that attacks field quality rather than projection count.
4. Optional: `solver_print_level=0` for iMF eval logs (§6ii).

## 8. Bottom line (first run)

The Gen13 machinery is **fully validated end-to-end** — orchestrator, gates, training, seam-swapped constrained sampler, NFE instrumentation all work, and the numbers are internally consistent. The scientific verdict is **"not yet superior, but promising"**: iMF matches FM's task quality at **4.6× fewer network evaluations**, and misses the safety bar by 3 episodes for a reason we can now name precisely (projection count, not field quality) and test in a single 10-minute job.

---

# UPDATE — K-sweep results (K=4, K=5) and revised verdict

**Date:** 2026-07-19 · **Run:** `IMF_METHODS="hardflow_new" IMF_KS="4 5"`, same checkpoint, no retraining. CSVs carry the fix_4 `nlp_*` columns.

## 9. The projection-count hypothesis is CONFIRMED — and it saturates

| Run | Success | Safety | Viol | Steps | s/plan | NFE/plan | NLP/episode | NLP failures |
|---|---|---|---|---|---|---|---|---|
| iMF hardflow_new K1 | 80% | 80% | 10 | 56.6 | — | 5 | — | — |
| iMF hardflow_new K2 | 94% | 94% | 3 | 51.7 | — | 9 | — | — |
| **iMF hardflow_new K4** | **96%** | **96%** | **2** | 51.9 | **0.403** | **17** | 28 | **0** |
| **iMF hardflow_new K5** | **98%** | **98%** | **1** | 51.7 | **0.487** | **21** | 35 | **0** |
| **FM hardflow_new (B2)** | **100%** | **100%** | **0** | 50.7 | 0.847 | ~41 | 10 | — |

Safety is **monotone in K** — 80 → 94 → 96 → 98% — exactly as §4 predicted. But the returns collapse: **K1→K2 = +14 pts, K2→K4 = +2, K4→K5 = +2.** Reaching a *deterministic* 0/50 by extrapolation would need roughly K≈8–10, i.e. ~33–41 NFE — **precisely FM's budget**, which would erase the efficiency advantage entirely. Predicted NFE (17 / 21) matched the logged values exactly, confirming the accounting.

**Zero NLP failures across all 100 episodes** (fix_4's new instrumentation earning its keep). This **cleanly eliminates** the alternative hypothesis that residual violations come from failed projections — every NLP converged. The violations are the *approximation*, not the solver.

## 10. ⚠️ The statistics say we cannot actually call this a loss

Fisher exact test, each iMF run vs FM's 0/50:

| Comparison | p-value | Conclusion |
|---|---|---|
| K1 (10/50) vs FM (0/50) | **0.0012** | significantly worse ✅ real effect |
| K2 (3/50) vs FM | 0.242 | **not** distinguishable |
| K4 (2/50) vs FM | 0.495 | **not** distinguishable |
| K5 (1/50) vs FM | **1.000** | **completely** indistinguishable |

95% Clopper–Pearson upper bounds on the true violation rate: FM 0/50 → **≤7.1%**; K5 1/50 → **≤10.6%**. These intervals overlap almost entirely.

**So at n=50 the experiment is underpowered to resolve the very difference the verdict hinges on.** "98% vs 100%" reads like a gap but is one episode, and one episode is noise at this sample size. Symmetrically — and this is the honest part — **failing to detect a difference is not evidence of equivalence**: with a true 2% violation rate, n=50 has only a **64%** chance of showing even one violation. Both "iMF is as safe as FM" and "iMF is slightly less safe" remain live.

Power (P(observe ≥1 violation | true rate = 2%)): n=50 → 64%, **n=100 → 87%, n=200 → 98%**.

## 11. Revised verdict

| Criterion | Target | K5 actual | Result |
|---|---|---|---|
| 1. Safety parity (**non-negotiable**) | 100%, 0 violations | 98%, **1 violation** | ❌ **FAIL as written** — but statistically indistinguishable from FM (p=1.0) |
| 2. Efficiency win | < B2 | **21 vs 41 NFE (1.95×)**; **0.487 vs 0.847 s/plan (1.74×)** | ✅ PASS |
| 3. Quality not degraded | steps ≈ 50.7 ±20% | 51.7 (+2%) | ✅ PASS |

**Gen13 still cannot declare "iMF superior"** — the criterion is a deterministic 0 violations and K5 produced 1. But the honest characterization has shifted substantially from the first run: iMF at K=5 is **statistically indistinguishable from FM on safety while using half the compute**, rather than "6 points worse".

**The criterion itself is now the bottleneck.** A deterministic "0/50" bar is not measurable at n=50 — any method with a true violation rate under ~2% passes or fails it by luck. The verdict should be restated as a *rate with a confidence bound* at adequate n, not a coin flip on one episode.

## 12. Next steps (revised priority)

1. **Re-run K5 (and FM B2) at n=200.** This is now the decisive experiment, not more K values. It moves power from 64%→98% and converts "1 vs 0" into a real rate comparison with tight CIs. B2 must be re-run at the same n for a fair paired comparison — its 0/50 has a 7.1% upper bound, so *FM's own safety is not established at 100% either*.
2. **Do NOT chase higher K.** §9's saturation shows K≈8–10 would be needed for a deterministic 0, at FM-equivalent cost — self-defeating. K=5 is the efficiency sweet spot (1.74× faster, 1.95× fewer NFE).
3. If n=200 still shows a small nonzero rate, the remaining levers are **training quality** (§5b: `a0_mse` marginal, LR froze while spiky) and the **`consistency` objective** (`Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md` §4) — not more projections.

## 13. Bottom line (revised)

The projection-count hypothesis was **right**: safety rose monotonically with K, and zero solver failures proved the residual is approximation error, not solver failure. iMF at K=5 delivers **~2× the efficiency of FM at a safety level statistically indistinguishable from it** — a genuinely promising result. What blocks the "superior" declaration is no longer a 6-point gap but a **measurement-resolution problem**: at n=50 neither method's true violation rate is pinned down. The next run should be **more episodes, not more K**.

---

# 14. INITIAL CONCLUSION — has iMF beaten the baseline, scientifically and strictly?

**Short answer: NO — not strictly, not yet. But the efficiency half of the claim is already PROVEN beyond doubt, and the safety half is genuinely undetermined rather than lost.** Breaking the question into its two independent halves is what makes this answerable:

## 14.1 Efficiency — ✅ **PROVEN. Strictly, decisively, no caveats.**

| Measure | FM B2 | iMF K5 | Gap |
|---|---|---|---|
| NFE per plan | 41 | **21** | **1.95×** (a deterministic count, no uncertainty at all) |
| Compute per plan | 0.8467 ± 0.0345 s | **0.4865 ± 0.0344 s** | **1.74×**, permutation test **p ≈ 0** (<1/20 000) |

And the strongest form of evidence available: **the two distributions do not overlap at all.** FM's *fastest* episode (0.776 s) is slower than iMF's *slowest* (0.581 s). Every one of 50 iMF episodes beat every one of 50 FM episodes. This is not a statistical inference that could reverse with more data — it is complete separation.

**On efficiency, iMF has strictly beaten the baseline.**

## 14.2 Safety — ⚠️ **UNDETERMINED. Not proven equal, not proven worse.**

- Frequentist: K5's 1/50 vs FM's 0/50 → Fisher exact **p = 1.000**. Literally the least significant result obtainable.
- But absence of evidence ≠ evidence of absence. Bayesian posteriors (uniform priors) on the *true* violation rates:

| Quantity | Value |
|---|---|
| P(iMF's true rate is worse than FM's) | **75%** |
| P(the gap is < 2 percentage points) | **55%** |
| Median gap | **+1.67 pts** (iMF worse) |
| 95% credible interval for the gap | **[−4.3, +9.0] pts** |

That interval spans zero *and* is enormously wide — the data simply cannot resolve this. The 75% figure is the honest headline: **the data leans toward iMF being slightly less safe, but nowhere near conclusively** (75% is weak evidence; it is not the 95% one would demand).

## 14.3 The strict verdict

**iMF has NOT beaten the baseline strictly**, for two independent reasons:

1. **The pre-registered criterion is unmet by its own terms.** Plan §5 criterion 1 demanded *"100% safe, 0 violations — non-negotiable"*. K5 produced 1 violation. Failing a criterion you wrote in advance is a fail; retroactively softening it to "statistically indistinguishable" would be moving the goalposts, and this document declines to do that.
2. **Equivalence was never demonstrated, only non-distinguishability.** Claiming parity requires a non-inferiority test with a pre-declared margin and adequate power. At n=50 with 1 event, that test cannot even be run.

## 14.4 Is there a *great chance* it wins with more data? — **A qualified yes, and here is the honest split**

**Arguments that it likely wins (Pareto sense — equal-ish safety at half cost):**
- The efficiency win is permanent and cannot be eroded by more data (§14.1).
- The safety trend has a **mechanistic explanation, not a curve fit**: safety tracks projection count (80→94→96→98%), and the fix_4 instrumentation proved **0 NLP failures in 100 episodes**, eliminating the main alternative cause. We understand *why* the number moves.
- The gap under test is one episode. If iMF's true rate were, say, 1%, an n=200 run would very likely show it as within noise of FM.

**Arguments for caution (these are real, not pro-forma):**
- **The returns are saturating**, and that pattern is not favorable: +14, +2, +2. Extrapolation suggests true parity may need K≈8–10 — precisely FM's budget, which would erase the entire advantage. There is no guarantee the curve reaches 100% at any K that is still cheap.
- **There is a mechanistic reason to expect a residual gap.** iMF's field is measurably coarser (≈0.37/dim vs Gen3v4's ≈0.25/dim). The NLP guarantees the *predicted* endpoint is feasible (constraint violation ~1e-16); executed-trajectory violations arise from *prediction error*. A coarser field ⇒ more prediction error ⇒ a plausibly irreducible floor that more projections cannot fully remove.
- The posterior already leans 75% toward iMF being worse. That is not damning, but it is not encouraging either.

**Also worth stating plainly:** FM's "100%" is **not** an established guarantee either — 0/50 has a 95% upper bound of **7.1%**. Neither method has a closed-loop hard guarantee: the NLP enforces constraints on the *plan*, while violations occur in *execution* (model–plant mismatch + receding-horizon replanning). This is an empirical rate comparison between two imperfect methods, not a proof-versus-proof contest. That framing is more favorable to iMF than "98 vs 100" makes it sound.

## 14.5 Verdict, stated precisely

> **Efficiency: iMF has strictly and decisively beaten the FM baseline (1.95× fewer NFE, 1.74× faster, zero distribution overlap).**
> **Safety: undetermined. iMF is not measurably worse (p = 1.0), but parity is unproven and the posterior leans 75% toward a small real gap.**
> **Overall: no strict win yet. A Pareto win (equal safety, half cost) is a live and reasonable possibility — perhaps roughly even odds — but it is not yet supported, and the saturating-returns pattern plus the coarse-field argument are genuine reasons it might not materialize.**

**The single experiment that resolves this:** a paired **n = 200** run of iMF K5 *and* FM B2 (power 64%→98%). That converts "1 vs 0" into two rates with tight intervals and settles the question either way. Until then, the defensible public claim is the efficiency result, with safety reported as *"no significant difference detected at n=50 (p=1.0); parity not established."*

---

# 15. NEXT STEP — the decisive n=200 paired run

## 15.1 The experiment

**One job, two arms, same seed, same everything except the method:**

| Arm | What | n | Purpose |
|---|---|---|---|
| **A** | iMF `hardflow_new_imf`, **K=5**, cp 4 | **200** | the candidate (best efficiency/safety point) |
| **B** | FM `hardflow_new`, `ode_t_steps=10` | **200** | the baseline — **must be re-run**, its 0/50 is not established either (95% CI ≤7.1%) |

**Why both arms.** Re-running only iMF and comparing against the frozen 0/50 would be an unfair, underpowered comparison — FM's own rate is unknown to ±7 pts. A paired run at identical n is the only way to get two intervals that can actually be compared.

**Why K=5 and not a K-sweep.** §9 showed saturation; K=5 is the efficiency sweet spot (1.95× fewer NFE, 1.74× faster, zero distribution overlap). More K values would spend budget on a question already answered.

## 15.2 What it will settle

| Outcome at n=200 | Interpretation | Action |
|---|---|---|
| iMF ≈ FM (e.g. 2–4 vs 1–3 violations; CIs overlap, non-inferiority margin met) | **Pareto win** — equal safety at half the cost | Declare the Gen13 result; write it up |
| iMF clearly worse (e.g. ≥8 vs ≤2, p<0.05) | Real, mechanistic gap — consistent with the coarse-field floor (§14.4) | Go to §15.4 levers, **not** more K |
| **Both** show a nonzero rate (e.g. 4 vs 3) | Neither method is a closed-loop "hard" guarantee — reframes the criterion itself | Restate the plan §5 criterion as a *rate with CI*, not "0 violations" |
| iMF better than FM | Unexpected; suspect a confound | Re-check seeds/config before believing it |

Power: with a true 2% rate, n=200 has a **98%** chance of revealing ≥1 violation (vs 64% at n=50).

## 15.3 How to run it — two practical blockers to handle first

**Blocker 1 — `random_repeat=50` is hardcoded in both eval scripts** (`eval_hardflow_new_imf.sh:18`, `eval_hardflow_new.sh:18`).
- iMF side: trivial — make it an env knob (`random_repeat="${RANDOM_REPEAT:-50}"`), a Gen13-owned file.
- FM side: ⚠️ `eval_hardflow_new.sh` is **pre-existing HardFlow code under the no-edit rule.** Do **not** edit it. Instead add a new Gen13-owned sibling (e.g. `run_scripts/eval_hardflow_new_n200.sh`) that calls `run/eval.py` with the identical paper parameters plus `--random_repeat 200` and a distinct `exp_name` — additive, and it leaves the frozen baseline artifacts untouched.

**Blocker 2 — exp_name collision.** Both arms must write to *new* directories (e.g. `H16_imf_hardflow_new_K5_n200`, `H16_1e6steps_hardflow_new_n200`) so the frozen n=50 baselines are preserved for reference (§1's rule).

**Free regression check (worth exploiting):** `env.set_seed(cfg.seed)` is called once before the episode loop and episodes run sequentially, so **the first 50 episodes of a 200-episode run should reproduce the original 50 exactly.** If they don't, something non-deterministic changed between runs — check this before trusting the new numbers.

**Runtime estimate:** iMF K5 was 3m44s/50 eps → ~15 min for 200. FM is 1.74× slower per plan → ~26 min for 200. Total ≈ 45 min. Per the SLURM rule (2× expected, 24h cap): **request `--time=02:00:00`**, single job, no retraining, no GPU-hour concern.

## 15.4 If the gap turns out to be real — the levers, in order

Do **not** reach for more projections (§9 saturation makes that self-defeating). In priority order:

1. **Training quality** (§5b). The clearest untapped headroom: `a0_mse` settled at 0.2–0.35 vs the <0.15 reference, and the cosine LR annealed to 0 while `raw_mse_u` was still spiking (132.6 at step 98.4k) — the Gen3v4 "froze on a noisy plateau" pattern. A constant-low-LR tail or a longer budget attacks the **coarse-field floor**, which §14.4 identifies as the mechanistic cause of any residual violations. This is the highest-value lever because it targets the actual hypothesised cause.
2. **The Newton/MF pull-back (THEORY Level 2).** Gen13 built only the "Level 1" seam (plan D8): the pull-back still uses HardFlow's `τ` gain. `THEORY_DeepMix_HF_iMF.md` shows that gain delivers only ~11% of the requested correction at τ=0.1, and that the correct Jacobian `∇F = I + (1−τ)∇u` is available by JVP — precisely because we now have `u`. **This is the one advantage iMF has that Gen13 has not yet cashed in**, and it improves correction accuracy without adding projections.
3. **`value_objective="consistency"`** (`Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md` §4). Built, wired, and currently switched off; plausibly matters more for a coarser field. Cheap to test — one flag, no retraining.

## 15.5 Cheap diagnostics worth adding alongside (optional)

- **Foresight-fan plotting** — `x_chain`/`x1_estimation` are already returned and discarded (`_, _`). Capturing them would let us *see* whether iMF's exact endpoint prediction differs qualitatively from FM's Euler shot — currently our only un-inspected diagnostic.
- **Measure S1 directly** — mean squared second difference of the planned trajectory, before vs after projection. Turns "the NLP manufactures smoothness" from a claim into a number, and would show how much work the projection is doing for each backbone.

## 15.6 Summary of the next action

> **Run one job: iMF K5 and FM B2, both at n=200, same seed, new exp_names, ~2h walltime.** Handle the two hardcoded-`random_repeat` blockers first (env knob for iMF; a new additive sibling script for FM — never edit the frozen baseline script). Verify the first 50 episodes reproduce the frozen results, then compare rates with confidence intervals. That single run converts Gen13's central open question from "undetermined" into a defensible yes or no.
