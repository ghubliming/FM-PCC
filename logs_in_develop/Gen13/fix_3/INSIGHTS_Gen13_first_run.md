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

## 8. Bottom line

The Gen13 machinery is **fully validated end-to-end** — orchestrator, gates, training, seam-swapped constrained sampler, NFE instrumentation all work, and the numbers are internally consistent. The scientific verdict is **"not yet superior, but promising"**: iMF matches FM's task quality at **4.6× fewer network evaluations**, and misses the safety bar by 3 episodes for a reason we can now name precisely (projection count, not field quality) and test in a single 10-minute job.
