# U6 — DEBUG: DiT eval trajectories explode while train loss & metrics stay sane

**Date:** 2026-06-18
**Symptom (reported):** the iMF **DiT** trains cleanly — best train/val loss of any run — but on `eval`,
the **`diffuser`** plots show **exploded / chaotic trajectory lines** (they trend toward the goal but are
not smooth, far worse than the DPCC UNet baseline). **Yet the numeric metrics** (avg steps, violations,
success) **look reasonable, not ridiculous.**
**Backbone:** `imf_backbone='dit'` (U6). UNet does not show this.
**This doc:** isolates *why* the plot and the numbers disagree, ranks the root causes against the actual
code, and gives ordered diagnostics + fixes. **No code changed yet — this is the investigation.**

---

## 1. The paradox, stated precisely

Two facts that seem contradictory:
- **Plot exploded** ⇒ the trajectory the model produces is bad.
- **Metrics sane** ⇒ the trajectory the robot follows is fine.

They are only contradictory if "the plot" and "the metrics" are the **same trajectory**. **They are
not.** Resolving that is the whole bug.

---

## 2. Root reconciliation — open-loop *plan* (plotted) vs closed-loop *execution* (metrics)

What each thing actually is, in code:

**The blue exploded lines = the model's full open-loop H=8 plan.**
`eval:323` stashes `samples.observations[:, :, :]` (the entire horizon), and `eval:353–357` draws **all 8
planned waypoints** connected. This is the raw multi-step output of `p_sample_loop`.

**The metrics + black path = closed-loop execution using only the FIRST action.**
- `Policy.__call__` returns `action = actions[which_trajectory, 0]` — **only waypoint 0's action**
  (`policies.py:86`).
- `eval:305–309` executes that one action, gets a **fresh** observation, and **re-plans** next step
  (receding-horizon MPC).
- The black path (`eval:346`) and every metric (`eval:285–303` violations, `eval:367–372`) are built from
  this **executed** `obs_buffer`, never from waypoints 2…7.
- Tracking uses only waypoint **1** (`desired_next_pos = samples.observations[0, 1, …]`, `eval:322`).

**Consequence:** the closed loop only ever consumes **waypoint 0's action + waypoint 1's position**. But
**that does NOT make the executed path good** — see the correction below. Both the open-loop plan *and*
the executed motion are bad; the metrics survive for a *different* reason (§8), not because the first
step is clean.

> ### ⚠️ CORRECTION (supersedes an earlier draft of this doc)
> An earlier version claimed "the first step is good, only the tail explodes, so the executed path is
> smooth." **That is wrong.** The **executed trajectory itself is bad as hell** — the action you run is
> the model's own (corrupted) output, and `apply_conditioning` pins **only the observation dims** at
> position 0, **not the action dims** that get executed (`policies.py:82–86`; the executed
> `action = actions[which,0]` is pure model output). So H1/H2/H3 corrupt the **executed action**, and the
> real motion is jerky/exploded.
> **The right question is therefore: why does the metric log stay "not so bad" when the real trajectory
> is terrible?** Answer in **§8** — the metrics are coarse *task-success* numbers that are **blind to
> motion quality.**

---

## 3. What the "best train loss" rules OUT

A clean, best-in-class train/val loss eliminates the structural DiT bugs:
- **Not a token/position scramble** — if the DiT mapped timestep `p` to the wrong output position, the
  per-position loss vs per-position targets would be *high*, not best. Good loss ⇒ patch-embed↔unpatchify
  ordering is correct.
- **Not a gross magnitude / RoPE-pairing error** — those would inflate the regression loss too.
- **Not a NaN / weight-load failure** — loss converged.

So the explosion is **not** a broken forward pass. It is a **train-objective ↔ deployment mismatch**: the
loss the DiT minimizes is *not* "smooth integrated 8-step trajectory." See §4.

---

## 4. Ranked root causes (with code evidence)

> ### Primer — what `ω` and CFG actually do (read before H1)
> **CFG = classifier-free guidance.** It *strengthens* the effect of the conditioning by predicting twice
> and extrapolating between the two:
> ```
> u_guided = u_uncond + ω · (u_cond − u_uncond)
>            └─ no conditioning   └─ "the conditioning direction"
> ```
> - `u_cond` = prediction **with** the conditioning; `u_uncond` = prediction with the conditioning
>   **dropped** (`force_dropout=True`).
> - **`ω` is the guidance scale (the dial):** `ω=0` → ignore conditioning; `ω=1` → plain conditioned
>   prediction; **`ω>1` → extrapolate *past* the conditioned prediction**, pushing further along the
>   `(u_cond − u_uncond)` direction. Bigger ω = sharper/stronger conditioning **but** also bigger
>   overshoot — if that direction is wrong or noisy, ω multiplies the error.
> - **"Interval" CFG** = apply this **only** while the flow time `τ` is inside `[t_min, t_max]` (here
>   `[0.4, 0.6]`); outside the window, `ω` is effectively 0 and the plain prediction is used
>   (`imf_diffusion.py:230`).
>
> **The catch CFG depends on:** the extrapolation is only meaningful if `(u_cond − u_uncond)` is a *real*
> conditioning signal. If dropping the "conditioning" barely changes the output (or changes it randomly),
> then `(u_cond − u_uncond)` is **noise**, and multiplying noise by `ω=4` and injecting it into the
> trajectory is pure damage. **That is exactly the DiT's situation — see H1.**

### H1 — Interval-CFG ω=4 amplifies the DiT's near-meaningless class-token guidance, on the *decisive* flow step  ⟵ strongest, test first
The plan defaults (config) are `meanflow_cfg_omega=4.0`, interval `[0.4, 0.6]`. In the sampler
(`imf_diffusion.py:230`):
```
step_cfg = ω  if (cfg_on and t_min ≤ τ ≤ t_max) else 0
```
At **2 NFE**, τ ∈ {0.0, 0.5}. **τ=0.5 ∈ [0.4, 0.6] ⇒ CFG fires on the final, data-side step** — the step
that most shapes the output trajectory. There, `u = u_uncond + 4·(u_cond − u_uncond)` (`:163–166`).

**Why this hurts the DiT specifically:** for the DiT, `u_cond` vs `u_uncond` differ *only* by the
class-token index (label 0 vs the null index) — the CFG-dropout switch I added (`imf_dit_trajectory.py`,
`force_dropout` → null class). But the **real** conditioning (the start observation) is pinned into `x`
**externally** and is **identical** in both calls. So `(u_cond − u_uncond)` is **not** a meaningful
"conditioning direction" — it's a small, largely arbitrary learned offset between two near-identical
predictions. Multiplying it by **4** and injecting it into the final step perturbs the whole horizon in a
roughly arbitrary direction → **exploded plan**. The UNet survives the same CFG because (a) its
conditioning enters as a summed real bias and (b) its conv smoothness damps per-position noise.
'`diffuser`' does **not** disable this — it only nulls the *projector* (`eval:241`); interval-CFG is
internal to `p_sample_loop` and stays on.

**This is the prime amplifier and the cheapest to confirm (diag D1).**

### H2 — The DiT has no horizon-smoothness prior, and the loss is velocity-only + heavily step-0-weighted
- The training target is on **u (a velocity field)**, not on the integrated trajectory. Low u-loss does
  **not** imply a smooth few-step integral — integration error compounds, and nothing in the loss
  penalizes jaggedness across waypoints.
- The loss is **skewed to waypoint-0 action**: `get_loss_weights` sets
  `loss_weights[0, :action_dim] = action_weight` (=10) while every other entry is 1 (`imf_diffusion.py:132`,
  `discount=1.0`), and the JVP loss multiplies by exactly these weights (`:451`); `a0_loss` (`:469`) also
  spotlights step-0 action. So the optimizer is *most* rewarded for nailing **waypoint-0 action** — the
  one quantity that drives the metrics — and only weakly constrained on waypoints 2…7.
- A **conv UNet** masks this: convolution structurally correlates neighbouring timesteps, so even loosely
  supervised tail steps come out smooth. The **DiT (attention)** has no such bias; it fits the
  well-weighted start tightly and lets the under-weighted tail wander → **best loss, exploded tail.**

This explains **why the DiT explodes where the UNet doesn't despite a better loss.**

### H3 — Constant-h, t-blind few-step stepping (`dit_condition_on_t=False`)
At inference, every step uses `h = dt` constant and the DiT **ignores `t`** (official recipe;
`imf_dit_trajectory.py` only adds the `t` token if `condition_on_t=True`). So for a 2-step rollout, both
steps query `u(x, h=0.5)` with **no signal for "which half-interval am I in"** except the value of `x`
itself. The UNet's sampler explicitly conditions on **both** `t` and `h` and warns *never to freeze t*
(the U3-B1 guardrail in `p_sample_loop:212–220`) precisely because t-blindness is OOD for *it*. The DiT
trains t-blind too, so it's *self-consistent*, but it is **more fragile**: the tail waypoints have the
least `x`-information to localize the interval, compounding H2.

### H4 — Low-NFE compounding (cross-cutting)
At `flow_steps_v3=2` the integral is coarse; any per-step velocity error in the tail is integrated with
`dt=0.5` and not corrected. This is a multiplier on H1–H3, not an independent cause. (Bumping NFE is a
*diagnostic*, not the fix — iMF's point is low NFE.)

---

## 5. Diagnostics (run in this order — cheap/decisive first)

**D1 — Kill CFG, re-eval (confirms H1).** In `plan_fm_v3_imeanflow` set `meanflow_cfg_omega: 0.0`
(leave everything else). Re-run `diffuser`.
- *Blue plans de-explode* ⇒ **H1 confirmed** as the dominant amplifier. (Expected biggest single win.)
- *Still exploded* ⇒ H1 is secondary; go to D2.

**D2 — Plot the open-loop plan vs executed path side by side (quantifies §2).** You already have both:
black = `ax[i,4]` (executed), blue = `ax[i,5]` (plan). Confirm the **black is smooth and the blue is
jagged** — that *proves* the closed-loop-masking reconciliation and tells you the metrics are trustworthy.

**D3 — Per-waypoint error profile (confirms H2).** For a few plans, compute the deviation of waypoints
`0,1,…,7` from a smooth reference (or just the discrete curvature ‖x_{k+1} − 2x_k + x_{k-1}‖). Expect
**small at k=0,1, growing toward k=7**. A rising tail ⇒ H2 (tail under-constrained).

**D4 — NFE sweep (sizes H4).** Eval at `flow_steps_v3 = 1, 2, 4, 8`. If the plan smooths out markedly by
4–8 ⇒ compounding (H4) is large and the DiT velocity field is fine but the integration is too coarse.

**D5 — A/B the backbone under identical settings.** Same config, `imf_backbone: 'unet'` vs `'dit'`,
CFG off. If the UNet plan is smooth and the DiT is not ⇒ confirms the **missing smoothness prior** (H2/H3)
rather than anything in the shared objective/sampler.

---

## 6. Fixes (mapped to cause)

| Cause | Fix | Where |
|---|---|---|
| **H1** | Set `meanflow_cfg_omega=0` for the DiT (its class-token CFG is not a real conditioning lever); **or** redesign DiT CFG to drop a *meaningful* signal (e.g. drop the **pinned-observation token** / a returns token), not the null class. | plan + train blocks; `imf_dit_trajectory.py` CFG path |
| **H2** | Add an explicit **trajectory-smoothness / acceleration penalty** (‖x_{k+1}−2x_k+x_{k-1}‖²) to the loss, **or** raise supervision on non-zero waypoints (flatten the step-0 action skew), **or** give the DiT a mild locality bias. | `_p_losses_meanflow_jvp`, `get_loss_weights` |
| **H3** | Try `dit_condition_on_t=True` so the DiT can localize the interval at inference (costs official-recipe fidelity; A/B it). | config `dit_condition_on_t` |
| **H4** | Report plan quality at 4–8 NFE alongside 1–2; keep low NFE only if quality holds. | plan `flow_steps_v3` |

**Recommended first action:** **D1 (CFG off)** — one-line config change, highest expected payoff, directly
targets the strongest hypothesis.

---

## 7. What is NOT broken (so you don't chase ghosts)

- **The DiT forward / load / training is sound** (best loss; §3 rules out scramble/magnitude/NaN).
- **The eval plotting is sound** — it faithfully shows you a genuinely bad trajectory.
- **The metrics are *computed* correctly** — but they are **not a quality measure** (§8). Do **not** read
  "success≈ok, violations≈low" as "the model is good." That is the trap this whole doc is about.

> **One-line takeaway:** the DiT produces a **genuinely bad, jerky trajectory** (corrupted executed
> action, not just a bad tail); interval-CFG ω=4 via the DiT's meaningless null-class guidance (firing on
> the decisive final flow step) is the prime amplifier, and the DiT's lack of a smoothness prior under a
> velocity-only, step-0-weighted loss is why it explodes where the UNet stays smooth. The **metric log
> looks fine only because the avoiding metrics score binary goal-reaching + discrete obstacle-disk checks
> — they are blind to motion quality (§8).** Start by turning CFG off (D1).

---

## 8. Why the metric log stays "not so bad" while the real trajectory is bad as hell

This is the crux the user pointed at. The eval metrics are **task-completion** numbers, **not** motion-
quality numbers. Every one of them can be satisfied by a jerky, exploded, demonstration-unfaithful path,
because of *how* they are computed (`eval_flow_matching_v3_imeanflow.py`):

1. **Success is binary, goal-region, and sticky.** `success = info[1]` (`:310`); `if success: n_success[i]=1`
   (`:327`) and it never resets. The avoiding task calls "success" = *reached a goal zone*. The action is
   a **position delta** (`next_pos_des = action + obs[:2]`, `:308`), so a **net-upward but chaotic** stream
   of deltas still climbs into the goal region. Quality is never inspected — only arrival.

2. **Violations are discrete obstacle-disk checks, sampled once per env step.** `:296`:
   `‖obs(x,y) − center‖ < radius`. This fires **only** when the *landed* observation is **inside the
   obstacle circle**. So:
   - chaotic motion **in free space** (the vast majority of the arena) ⇒ **zero** violations;
   - high-frequency jitter **between** the discrete env steps is **never sub-sampled** — only the per-step
     landed point is tested;
   - the obstacle is a **small disk**; a path can weave wildly and still never enter it.
   `total_violations` (`:298`) is **penetration depth**, again zero unless you're *inside* the disk.

3. **`collision_free_completed` measures "finished without entering an obstacle," not smoothness.** It
   starts at 1 and only drops on an actual disk/halfspace entry (`:293,299`) or a timeout-without-success
   (`:328`). A jerky-but-successful, obstacle-missing run keeps it **= 1**.

4. **`avg_time` / `n_steps` are latency and length** (`:306,330`) — orthogonal to quality.

5. **There is NO smoothness / jerk / path-length / imitation-fidelity metric anywhere.** The single
   quality-ish number is `pos_tracking_errors` (`:321`) — and it is (a) printed **only for `diffuser`**
   (`:374`), (b) a **max over steps**, and (c) measured against the model's **own** next waypoint
   (`desired_next_pos = samples.observations[0,1]`, `:322`), **not** against a smooth/demo reference. So a
   model that *consistently* plans badly can still show a modest "tracking error" because it's tracking its
   own bad plan. (Notably, for the DiT this is the **one** logged number that may actually look elevated —
   worth checking explicitly; see D6.)

6. **Everything is averaged over trials and reported as a mean** (`:367–373`). A spread of
   "reached-the-goal-eventually" runs averages to numbers that "feel reasonable."

**Net:** the avoiding metrics answer *"did the controller get to the goal without sitting inside the
obstacle disk?"* — a **coarse, binary, discretely-sampled** question. The DiT can answer **yes** while
emitting motion that is **visually and dynamically terrible** (jerky, exploded, nothing like the smooth
demonstrations). That is the entire gap between "log looks ok" and "trajectory is bad as hell." **The
plots are the honest signal; the success/violation numbers are the misleading one.**

### D6 — make the badness measurable (so the log stops lying)
Add quality metrics to the eval and the gap disappears:
- **Jerk / curvature:** mean ‖x_{k+1} − 2x_k + x_{k-1}‖ over the executed path (and over each plan).
- **Path length ratio:** executed length ÷ straight-line (or demo) length — chaos inflates it.
- **Demonstration fidelity:** DTW / Fréchet distance of the executed path vs a held-out demo for the same
  start. **Print the diffuser `Tracking error` prominently** and compare DiT vs UNet — expect it
  materially worse for the DiT even when success/violations match.

This turns the qualitative "bad as hell" into a number that will rank DiT below the UNet/DPCC baseline and
make the regression visible in the log, not just the picture.

---

## 9. Caveats
- Analysis is from code logic (`imf_diffusion.py`, `imf_dit_trajectory.py`, `policies.py`,
  `eval_flow_matching_v3_imeanflow.py`) on branch `update_into_FM`; no local runtime — confirm with the
  cluster diagnostics in §5.
- H1 vs H2 weighting is hypothesis-ranked, not yet measured; **D1 then D5** settle it quickly.
- The §8 metric-blindness is **independent of the root cause** — even after H1–H4 are fixed, add the §8/D6
  quality metrics so motion regressions can't hide behind binary success again.
