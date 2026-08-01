# DISCUSSION — What should "SMOOTH" actually mean, and how should we measure it?

**Date:** 2026-08-01
**Status:** open exploration. **No code was written or changed.** This is an idea document.
**Trigger:** *"design a metric to quantify SMOOTH … from start to end is optimal path straight line, or like zigzag? but this is buggy, maybe avoiding obstacles; or compare to the original learned data? … from eye it is easy to inspect what is smooth, it is a SMOOTH feeling … smooth like surface smooth, not brutal zigzag between points (actions in H8). CHECK WHAT IS THE CURRENT METRIC in Gen13 SLURM console output."*

**Companions:** `../../HF_iMF/Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md` (S1/S2/S3 split — the conceptual prerequisite for everything below) · `../../Gen13/fix_7/RESULTS_Gen13_fix7_smoothness_2x2.md` (the numbers) · `../../Gen13/U13_smoothness_fan_diag_ml/CHANGELOG_Gen13_U13_smoothness_fan_diag_ml.md` (the MF/AF extension).

---

## 0. TL;DR

1. **The current Gen13 metric is one line of math**: mean squared second difference of the planned x-y path. It is *correct but thin* — scale-dependent, horizon-dependent, xy-only, and unable to distinguish "an aggressive but clean turn" from "jitter". The instinct in the prompt that a straight-line/zigzag framing is "buggy" is **right, and for a precise reason** (§3).
2. **There are three different `roughness` definitions live in this repo right now**, two of them sharing the column name `plan_roughness`. That is a real hazard for cross-generation analysis (§1.4).
3. The eye's "smooth feeling" is **not** low curvature — it is **low curvature *variation* at the sampling rate**, i.e. high-frequency energy. Any metric that penalises curvature per se will rank the *safest* obstacle-avoiding path as the *worst* one (§3.2).
4. The single best idea in the prompt is **"compare to the original learned data"**. Absolute smoothness thresholds do not exist and never will; the D3IL demonstrations define what smooth means *for this task*, for free, with no new rollouts (§4.7). That should be the calibration layer under whatever raw metric we pick.
5. Under HardFlow the *projected* plan is smooth **by construction** (hard dynamics equality), so measuring smoothness there is nearly degenerate — the information lives in the **raw** plan and the **executed** path (§5.3).

---

## 1. AUDIT — what is measured today

### 1.1 Gen13 / HardFlow — the number in the SLURM console

`HardFlow/run/eval_imf.py:132` (added in Gen13 fix_7, tagged `fix_7` in-place):

```python
def _traj_smoothness(plan, action_dim):
    px, py = action_dim + 2, action_dim + 3
    p = np.stack([plan[:, px], plan[:, py]], axis=-1)   # (H, 2)
    d2 = p[2:] - 2.0 * p[1:-1] + p[:-2]
    return float((d2 ** 2).sum(axis=-1).mean())
```

$$\text{rough} \;=\; \frac{1}{H-2}\sum_{t} \big\lVert\, p_{t+1} - 2p_t + p_{t-1} \,\big\rVert^2 \qquad [\text{m}^2]$$

**Where it is computed:** once per replan, on the **final planned horizon** `x_chain[0,-1,:,:]` (`eval_imf.py:220`), and again on the **raw pre-NLP warmstart** via `policy.raw_plan()` (`eval_imf.py:226`, fix_7.2). Averaged over replans → one value per episode.

**Where it surfaces:**
- console, per episode: `eval_imf.py:289` → `  rough=2.194e-06 raw=2.106e-04`
- CSV columns `plan_roughness`, `plan_roughness_raw` (`eval_imf.py:506`)
- SLURM summary table, `Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_ml_hardflow.sh:49-70`:
  `projected=… raw=… ratio=…x safe=…%` — the **ratio** (raw/projected) is the headline quantity, i.e. *how much smoothing the NLP manufactured*.

**Properties.** Zero for straight *or* constant-velocity paths. Grows with jitter. Sanity-ordered in fix_7 §2 on three synthetic cases: straight `3.7e-33` < mild jitter `1.6e-03` < zigzag `4.0e-02`.

### 1.2 npz_analysis — a *different* `roughness`

`npz_analysis/analyze_npz.py:140-178` computes a whole quality vector on the executed path *and* on plan snapshots (`plan_path_len`, `plan_straightness`, **`plan_roughness`**, `plan_max_jerk`, `plan_max_step`, `plan_max_abs`, `plan_cand_spread`, `plan_exec_div`). But there:

| name | definition |
|---|---|
| `straightness` | `net_disp / path_len` — 1 = straight, →0 = wandering |
| **`roughness`** | **`max_step / median_step`** — a *spike index*, not a second difference |
| `mean_jerk` / `max_jerk` | mean / max of `‖2nd difference‖` (**L2 norm, not squared**) |
| `max_abs` | explosion detector over all dims |

### 1.3 Drifting generation

`flow_matcher_v3_drifting/utils/drift_metrics.py:62` — `compute_trajectory_smoothness` = mean `‖acceleration‖₂` over the whole batch. Third convention.

### 1.4 ⚠️ The naming collision

**`plan_roughness` in a Gen13 `trajectories.csv` and `plan_roughness` in an `analyze_npz` output are different quantities with different units and different scaling behaviour.** One is $\text{m}^2$ and quadratic; the other is dimensionless and ratio-like. They are not comparable, not convertible, and will silently produce nonsense if a future analysis joins them. Whatever comes out of this discussion, a rename (`plan_d2_energy` vs `plan_step_spike`) is cheap and should ride along.

---

## 2. Is the current metric good enough? Honest assessment

**What it does well**
- Dirt cheap, no extra rollouts, already wired into both the iMF/FM 2×2 and the MF/AF (U13) matrix.
- It answered its actual question decisively: raw-vs-projected **ratio** 96× (iMF) vs 7.4× (FM) → "the NLP manufactures smoothness" is now a measurement, not a claim.
- It produced a genuinely useful negative result (fix_7 §5): guided iMF plans are *smoother* than guided FM plans yet violate more, which **eliminated roughness as the cause of the safety gap**.

**What it cannot do**
1. **Not scale-free.** Units m², so it scales with the *size* of the motion. A fast episode traversing more distance scores rougher than a slow one of identical shape. Cross-task comparison (avoiding vs aligning vs UAV) is meaningless.
2. **Not sampling-rate-free.** Second differences scale like $\Delta t^2$. **Comparing H8 against H16, or two dt's, without normalisation is invalid.** With H16 now standard in Gen13 and H8 elsewhere, this is not hypothetical.
3. **Squared ⇒ single-kink dominated.** One bad waypoint out of 16 can set the whole value. Combined with only the *mean over replans and episodes* being reported (no median/IQR anywhere), a heavy-tailed quantity is being summarised by its least robust statistic.
4. **xy-only, and hard-coded.** `px, py = action_dim+2, action_dim+3` is an `avoiding-v0` layout assumption. Joint-space jitter that never shows in the end-effector xy plane is invisible.
5. **Blind to shape.** This is the real one — see §3.

---

## 3. What "smooth" means to the eye (and why the straight-line framing is buggy)

### 3.1 Separate the axes that keep getting bundled

The prompt already senses these are different things. They are:

| axis | question | metric family | is it "smooth"? |
|---|---|---|---|
| **A. Directness** | does it go straight to the goal? | `straightness = net/path_len` | ❌ **no** — this is efficiency |
| **B. Curvature magnitude** | how hard does it turn? | mean ‖2nd diff‖, turning angle | ⚠️ partly — but see 3.2 |
| **C. Curvature *variation*** | does the turn direction flip back and forth? | sign reversals, HF spectral energy | ✅ **this is the zigzag** |
| **D. Spikes** | one insane waypoint? | max/median step, max jerk | ✅ separate failure |
| **E. Replan kinks (S3)** | does plan@t agree with plan@t+K? | plan-overlap L2 | ✅ separate failure |

The current metric is **B**, squared. The eye's complaint is almost always **C** (and sometimes D). That mismatch is the whole problem.

### 3.2 Why "straight = optimal" is wrong here

In `avoiding-v0` the agent must weave between obstacles. **The optimal path is curved by construction.** A metric that rewards straightness or penalises curvature magnitude ranks the safest trajectory worst — it is anti-correlated with the actual objective. Concretely: an aggressive-but-clean S-curve around two pillars and a per-step zigzag of the same amplitude can produce *the same* mean-squared second difference, and the eye calls one beautiful and the other broken.

**Note what this means for fix_7's validation.** Its sanity check was straight / mild-jitter / zigzag — three cases that a curvature metric passes trivially. **The discriminating case, "clean aggressive turn", was never tested.** That is exactly where a false positive would live. Any new metric proposal should be validated against that case first, not the easy three.

### 3.3 The right mental model: it's a frequency question

"Surface smooth, not brutal zigzag between points" is, mathematically, a statement about **where the path's energy sits in frequency**. Low-frequency content = "a curve" (fine, even if sharp). Energy near the Nyquist rate of the waypoint grid = "jitter" (bad). The eye is a band-pass detector; the current metric is a broadband one. That single sentence is the design brief.

---

## 4. Candidate metric families

Ordered roughly by cost. None require retraining; most are post-processing on data we already dump (`*_fan.npz`, `trajectories.csv`, `sampled_trajectories_all` in the FMPCC npz).

### 4.1 Normalised second difference (dimensionless) — *the free fix*

$$\tilde R \;=\; \frac{\text{mean}_t \lVert d^2 p_t\rVert^2}{\big(\text{mean}_t \lVert d^1 p_t \rVert^2\big)} \quad\text{, i.e. curvature energy \emph{per unit travel}}$$

Kills the scale- and dt-dependence in §2.1–2.2 at zero cost. **Everything else in this section is optional; this one is close to free and should probably happen regardless.**

### 4.2 Turn-direction reversal rate — *the pure zigzag detector*

Per step vector $v_t = p_{t+1}-p_t$, take the signed turn $\theta_t = \angle(v_t, v_{t+1})$. Report the **fraction of consecutive pairs where $\theta$ flips sign** (optionally weighted by $|\theta|$, so meaningless flips near 0 don't count).

- Scale-free, dt-robust, bounded in [0,1], directly interpretable ("38% of waypoints reverse the turn").
- **Clean turn → ~0. Zigzag → ~1.** Exactly axis C, nothing else.
- Weakness: ignores amplitude — a tiny high-frequency wobble scores like a violent one. Pair with 4.1.

### 4.3 Turning-angle decomposition

Report $\text{mean}|\theta_t|$ (**B**) and $\sum_t(\theta_{t+1}-\theta_t)^2$ (**C**) as *two* numbers. Cheapest honest way to stop conflating "turns a lot" with "turns erratically".

### 4.4 High-frequency energy fraction (spectral)

DFT the H-length path per axis; report $\sum_{|f|>f_{\text{nyq}}/2}|\hat p|^2 \big/ \sum|\hat p|^2$.

- The most faithful formalisation of "surface smooth", and dimensionless by construction.
- **Caveat:** H = 8 or 16 is a very short window. Only ~4–8 usable bins; leakage and windowing choices will matter more than the signal. Probably too fragile at H8; plausible at H16 and clearly fine on the **executed** path (50+ steps).

### 4.5 Smoothing-fit residual — *closest to "surface smooth"*

Fit a low-order curve (cubic spline with few knots, or a degree-3 polynomial) to the H waypoints; report **RMS residual normalised by path length**.

- Interpretation is exactly the intuition: *what fraction of the path is not explainable by a smooth curve.*
- Curvature-agnostic — a hard clean turn fits a spline fine and scores ~0. **Passes the §3.2 discriminating case.** That makes it the most promising *single* number here.
- One knob (knots / df) that must be fixed and documented, otherwise it is tunable-until-it-agrees.

### 4.6 Robust / heavy-tail-aware statistics

Independent of which raw quantity is chosen: report **median + IQR across plans**, and a **P95** alongside the mean. fix_7 argued n=5 suffices because roughness is a per-plan mean rather than a rare-event rate — true, but only for the *mean* of a quantity whose distribution nobody has looked at. A histogram of per-replan roughness for one cell would cost nothing and might change how all existing numbers are read.

### 4.7 ⭐ Demo-calibrated smoothness — *the best idea in the prompt*

"Compare to the original learned data." **Yes.** There is no absolute answer to "is 2.19e-06 smooth?" — but the D3IL expert demonstrations are, by definition, the reference for what a good trajectory looks like in this task.

Proposal: compute whichever raw metric on the **demonstration set**, once, to get a reference distribution; then report every plan as a **percentile / z-score against it**.

> `plan_roughness = 2.19e-06 → 71st percentile of demos` — *within the human/expert band*
> `raw_plan = 2.11e-04 → 99.9th percentile` — *far outside anything a demo ever did*

- Turns an uninterpretable float into a statement anyone can act on.
- Automatically scale-, unit-, and task-correct: the reference is measured in the same units on the same task, so §2.1–2.2 mostly dissolve.
- **Free** — demos are already on disk, no rollouts, no GPU.
- Stronger variant: compare *distributions*, not just a scalar — e.g. Wasserstein distance between the plan's turning-angle histogram and the demos'. Catches "same average curvature, wrong texture".
- Caveat to state up front: this measures **similarity to demonstrations**, not smoothness in the absolute. A method that is genuinely smoother than the demos will show as "anomalous". That is arguably fine — but it must be labelled honestly, and it means this cannot be the *only* number.

### 4.8 Cross-replan consistency (S3) — the axis nobody measures

Every metric above, including the current one, scores a *single planned horizon*. The executed path can kink at every replan seam even when every individual plan is perfectly smooth. Two cheap ideas:
- **Plan-overlap disagreement:** L2 between plan@t and plan@(t+K) over their shared timesteps. Pure S3, no new machinery.
- **Measure smoothness on the executed closed-loop path**, not just the plan. `analyze_npz` already does this for FMPCC; the HardFlow side does not.

And the standing observation from the companion doc: HardFlow *ships* `value_objective="consistency"` (`run/eval.py:127`) and **every** avoiding script uses `"distance"` instead. The S3 mechanism exists and has never been switched on.

---

## 5. Where the metric should be applied (this matters as much as the formula)

### 5.1 Three objects, not one

| object | what it tells you | already captured? |
|---|---|---|
| **raw plan** (pre-NLP / pre-projection) | **generative model quality (S1)** — the actual research question | ✅ `plan_roughness_raw` |
| **projected plan** (post-NLP) | what the optimiser certified | ✅ `plan_roughness` |
| **executed path** | what the robot did, incl. replan seams (S3) + tracking error | ❌ not on the HardFlow side |

### 5.2 The ratio is the real signal

fix_7's most informative column was neither raw nor projected but **raw/projected** — the *work done by the projection*. It is dimensionless, which is precisely why it travelled well (96× vs 7.4× is a claim you can state without knowing the units). Whatever new metric is adopted, **keep the ratio form**.

### 5.3 ⚠️ Under HardFlow, post-projection smoothness is near-degenerate

HardFlow imposes $A s_t + B a_t + c = s_{t+1}$ as a **hard NLP equality** for every consecutive pair. A dynamically-infeasible zigzag cannot survive it. So the projected plan is smooth *by construction* — which is why fix_7 found iMF-guided and FM-guided differing by a mere 0.82×, i.e. essentially noise, despite a 9.5× gap upstream.

**Implication:** for Gen13-family work, effort spent refining the post-projection smoothness metric has low information yield. The discriminating measurements are **raw plan** and **executed path**. For DPCC/FMPCC (softer projection, "generative brain + physical brakes") the projected plan *is* still informative — so the same metric carries different weight in the two paradigms, and any write-up needs to say which.

---

## 6. If I had to pick — a minimal proposal

Not a single score. **A vector of four**, each mapping to one failure mode, each reported for raw / projected / executed, each also given as a demo percentile:

| # | metric | axis | why |
|---|---|---|---|
| 1 | **normalised d² energy** (§4.1) | B | the current number, made comparable — preserves continuity with all existing results |
| 2 | **turn-reversal rate** (§4.2) | C | the literal zigzag detector; scale-free |
| 3 | **spline residual / path length** (§4.5) | C | survives the §3.2 clean-turn test; the closest thing to "surface smooth" |
| 4 | **max_step / median_step** (§1.2, already exists) | D | spike / explosion detector, already implemented in `analyze_npz` |

Plus, unconditionally: **median + IQR, not just mean**, and **demo percentiles** for 1–3.

**Deliberately not proposed:** a single weighted "smoothness score". Collapsing these hides *which* thing broke, and the weights would be unfalsifiable. Four labelled numbers beat one arbitrary one.

**Validation gate before trusting any of it** — the ranking must come out right on synthetic cases, *including the hard one*:

```
straight  <  smooth arc  <  aggressive clean S-curve  ≪  jitter  <  zigzag
                            └─ the case fix_7 never tested; metric 1 will
                               likely FAIL here and metrics 2–3 should PASS
```

If metric 1 ranks the clean S-curve near the jitter case, that is the empirical demonstration of §3.2 — and it is a result worth writing down on its own, since it retroactively qualifies how every existing `plan_roughness` number should be read.

---

## 7. Open questions

1. **Units across generations.** HardFlow's plan is unnormalised (metres); FMPCC npz plans may be normalised. Any cross-generation smoothness table needs the unit stated per row, or it is fiction.
2. **Is the demo set actually smooth?** §4.7 assumes it. Nobody has measured the demonstrations' own roughness distribution. That single histogram is the cheapest next step in this whole document and it gates the best idea in it.
3. **Does smoothness predict anything?** fix_7 §5 already showed it does *not* predict violations in HardFlow. Before investing in a better metric, worth asking what decision it would change. Honest possible answer: it is a **model-quality readout** (S1) for choosing between MF / AF / iMF fields, *not* a task metric — which is a perfectly good reason to have it, but a different one from why it was originally added.
4. **H8 vs H16.** Any comparison across horizons using the *current* metric is invalid (§2.2). Worth checking whether any existing table in `logs_in_develop/` already does this.
5. **Naming.** `plan_roughness` means two different things in two tools (§1.4). Fix before someone joins them.

---

## 8. Status

Idea document only. Nothing here has been implemented, and **nothing should be implemented from it without a decision on §6 and a run of the §6 validation gate.** The current `_traj_smoothness` is frozen and every published Gen13 number stands unchanged.
