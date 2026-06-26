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

**Why the official repo gets away with the very same CFG (and we don't):** the official imfDiT runs CFG at
even *higher* `ω≈8–10.5` and it *helps* — because the signal it **drops** is a **real ImageNet class label
`y`** (`imfDiT.py` `y_embedder` / `class_tokens`). Dropping `y` genuinely changes the prediction, so
`(u_cond − u_uncond)` points in a *meaningful* "more like class y" direction; amplifying it sharpens the
sample. **Our port has no real label** — the trajectory's actual conditioning (the start observation) is
pinned into `x` *outside* the network, and the class token is a **dummy** (one class + a null). So our
`(u_cond − u_uncond)` is **noise, not signal**, and the same machinery that *sharpens* images *explodes*
our trajectories. The difference is not the DiT or CFG itself — it is that **we kept the CFG knob but
removed the thing it guides on.**

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

> #### Primer — what "DiT conditioning" is, and why `t` matters here
> **"Conditioning" = the side-information you *tell* the network about the current generation step**, on
> top of the trajectory itself. In the **DiT** each conditioning signal is turned into its **own token**
> (a learned vector) and **prepended** to the trajectory tokens; attention then lets every trajectory
> step "read" them (`imf_dit_trajectory.py`, the prefix tokens). Our conditioning signals are:
> - **`h = t − r`** — the *interval size* the average-velocity `u` is being asked to cover (the step size);
> - **`ω, t_min, t_max`** — the CFG knobs (the §4 primer);
> - a **class token** — used only as the CFG on/off switch;
> - **`t`** — *where we currently are* along the noise→data flow (τ=0 pure noise … τ=1 data). **Optional**:
>   added only if `dit_condition_on_t=True`.
> (Contrast the **UNet**, which doesn't tokenize — it *sums* all of these into one bias vector.)
>
> **Why `t` matters:** sampling walks `τ` from 0→1 in `N` steps. The correct `u` at an **early** step
> (mostly noise) is *different* from the correct `u` at a **late** step (near data). If the DiT is **not
> told `t`**, it must infer "how far along am I?" purely from the statistics of `x` — a weak, indirect cue.

> #### ⚠ Rebuttal from the official repo — H3 is WEAK, do not lead with it
> **The official imfDiT is `t`-blind on purpose** ("We don't explicitly condition on time t, only on
> h = t − r", `/workspaces/imeanflow/models/imfDiT.py:370`) and still achieves **SOTA 1–2-NFE generation**.
> So **`t`-blindness by itself is not broken** — the MeanFlow method is *designed* to need only `h`, and a
> well-trained t-blind DiT integrates fine. Our DiT is `t`-blind in the **same** way. Therefore H3 is
> **demoted to "probably not the cause"** — we keep it only as a cheap A/B (D-toggle below), not as a
> suspect. (The UNet's "never freeze t" guardrail is about the **UNet**, which was *trained* with `t`;
> it does not imply a t-blind DiT is wrong.)

At inference every step uses `h = dt` constant and the DiT ignores `t` (official recipe;
`imf_dit_trajectory.py` only adds the `t` token if `condition_on_t=True`) — **identical to the official
model that works.** The only residual worry is that our few-step *trajectory* setting gives the tail
waypoints little `x`-information to localize the interval; but since the official is t-blind and fine,
treat this as a **minor** contributor at most. If you want to rule it in/out, flip `dit_condition_on_t=True`
and A/B — but **spend your first effort on H1/H2, not here.**

### H4 — Low-NFE compounding (cross-cutting)
At `flow_steps_v3=2` the integral is coarse; any per-step velocity error in the tail is integrated with
`dt=0.5` and not corrected. This is a multiplier on H1–H3, not an independent cause. (Bumping NFE is a
*diagnostic*, not the fix — iMF's point is low NFE.)

---

## 4B. Is the convergence "fake"? — training-objective & training-setup analysis

> **"Do we have such a saying — fake convergence?"** Not as a standard term, but **the phenomenon is real
> and well-named.** A loss can converge low while the network is a bad generator, via: **proxy mismatch**
> (the loss isn't the thing you care about), a **degenerate fixed point of a bootstrapped objective**, or
> **loss-masking by reweighting**. So "converged val loss + bad model" is **not** a contradiction — it's a
> known failure mode. For iMF, three mechanisms make our converged loss **necessary but not sufficient**.

### Mechanism 1 — the MeanFlow loss is a SELF-CONSISTENCY target → low loss ≠ correct field
The target is `u_target = (v_inst + h·du/dr).detach()` (`imf_diffusion.py:444–447`) — it contains the
network's **own** derivative. Minimizing `‖u_pred − u_target‖` means *"be consistent with my own
identity,"* **not** *"match a ground-truth average velocity."* Self-referential (bootstrapped) objectives
have **trivial / degenerate fixed points**: the field can drift toward a self-consistent-but-biased
solution (or partially collapse toward the instantaneous `v`, ignoring `h`) and **still show low loss**.
This is the same failure class as **consistency-model collapse**. **Low loss certifies self-consistency,
not generation quality** — the textbook "fake-good" setup.

### Mechanism 2 — adaptive loss weighting HIDES the hard samples in the reported number
The loss multiplies each sample by `w = 1/(‖Δ‖² + c)^p`, `p=0.5` (`imf_diffusion.py:454`). That **down-
weights exactly the high-error samples.** So the printed train/val loss is dominated by the **easy**
samples; the regions the model is worst at — typically the **long / near-data intervals** the few-step
sampler most relies on — contribute least to the number. A "best val loss" can literally mean **"best at
the easy stuff,"** while the hard regime that drives the explosion is unmeasured.

### Mechanism 3 — the loss is velocity-MSE, never integrated-trajectory quality
Nothing in the objective integrates `u` into a trajectory or penalizes jerk; compounding error at low NFE
is **invisible** to a per-element velocity loss. (This is H2, seen from the training side: you can drive
velocity-MSE to its floor and still integrate to a chaotic path.)

### Training-setup suboptimalities vs the official method (the "fake-good" amplifiers)
The official PyTorch repo is **inference-only** (training is JAX-only — `imf.py:29` asserts inference;
README points to the JAX MeanFlow for training), so exact hyperparameters aren't copyable. But the
*method* + our config (`config/avoiding-d3il.py`) expose concrete gaps for a **high-variance bootstrapped
DiT** objective:

| Knob | Ours | Why it's likely suboptimal for iMF-JVP |
|---|---|---|
| `batch_size` | **32** (eff. 64 with `gradient_accumulate_every=2`) | **The single biggest suspect.** The JVP/MeanFlow target is **high-variance**; the method leans on **large batches** to average it. Eff-64 ⇒ noisy gradients ⇒ the net fits a *smoothed/biased* mean velocity (low average loss) but a poorly-estimated field where variance is high. |
| `learning_rate` | **5e-4** | High for a transformer/DiT (DiTs typically ~1e-4 **with warmup**). High LR on a self-referential target can **lock in a bad fixed point**. |
| `ema_decay` | **0.995** | Loose — averaging window ≈ 200 steps. Image DiTs use ~**0.9999**. Eval uses the EMA weights, so they may be **noisier** than the raw best. |
| `warmup_epochs` | **0** | No curriculum. A bootstrapped target benefits from **first learning `v`** (instantaneous) before trusting the JVP; a cold start can settle a **degenerate field**. |
| `n_train_steps` | **100k** @ eff-batch 64 | ~6.4M samples; the **long-`h` / near-data** regime (rare under the schedule) is plausibly **undertrained** — exactly where Mechanism 2 hides it. |

**Net (the training-side root cause beneath H1/H2):** the DiT can minimize a **reweighted self-consistency
proxy** (Mechanisms 1+2), under a **small-batch, high-LR, loose-EMA, no-warmup** regime, and report a
"best ever" converged loss **while being a chaotic generator.** The convergence is real; its *meaning* is
weak.

### Training diagnostics (do these alongside §5)
- **T1 — un-mask the loss.** Re-log the **un-weighted** loss (`w≡1`) and the loss **split by `h`-bucket**
  (small/med/large). If small-`h` is great but large-`h` is bad ⇒ **Mechanism 2 confirmed**; the converged
  number was hiding the hard regime.
- **T2 — track a quality metric *during* training**, not just loss: 1-NFE reconstruction error, or the
  **jerk** of generated plans, logged every N steps. If loss falls but this doesn't ⇒ **fake convergence
  confirmed**.
- **T3 — fix the regime and re-eval:** raise effective batch (accum 8–16), drop LR to ~1e-4 **+ warmup**,
  tighten EMA to ~0.9999. Re-check plan smoothness.
- **T4 — isolate the bootstrap:** does an `fm_equivalent` **DiT** (plain velocity-MSE, no JVP) also explode?
  If **yes** ⇒ it's the DiT/training regime, not the JVP. If **only `meanflow_jvp`** explodes ⇒
  **Mechanism 1** (bootstrap fixed point) is implicated. This cleanly separates objective from architecture.

### "But it really converged — if so, do batch size / LR / EMA even matter?"

**Yes it converged, and yes they still matter — because "converged" and "correct" are different claims.**
"Converged" means *the loss stopped dropping* = the optimizer reached **a** stable fixed point of **this**
objective under **this** regime. It does **not** mean that fixed point is the *right* field. Three reasons
the regime is not neutralized by convergence:

1. **Converged ≠ unique.** Different batch/LR/EMA settings land in **different** stable points. "Loss
   flat" tells you *where you stopped*, not that it's the best reachable solution.

2. **The objective is bootstrapped, so batch noise is baked INTO the fixed point — not averaged away.**
   This is the crux, and it's where MeanFlow differs from plain FM:
   - **Fixed-target regression (`fm_equivalent` / plain FM):** the label `x₁−ε` is **constant**. Small-batch
     noise mostly affects *speed*; once converged, the solution is ≈ the same regardless of batch. Here
     *"converged ⇒ done"* is roughly true — **this is the intuition you're applying.**
   - **Self-referential target (`meanflow_jvp`):** the target **contains the network's own derivative**, so
     it **moves as the net moves.** Gradient noise perturbs the *target itself*, and the network converges
     to be self-consistent **with that noisy target** — the bias becomes **part of** the fixed point. A
     larger batch gives a **less-biased target ⇒ a different, better converged solution.** So here batch
     size changes the **destination**, not just the speed. **Your fixed-target intuition does not transfer
     to a bootstrapped objective.**

3. **The converged *number* is a reweighted proxy, and EMA picks the deployed weights.** Adaptive weighting
   (Mech. 2) can drive the displayed loss low independent of the hard regime; and EMA decides **which**
   weights you actually evaluate, regardless of whether the *raw* loss converged. A flat, low curve
   certifies neither field-correctness nor a good deployed snapshot.

> **One line:** convergence proves you found a stable fixed point of a *reweighted self-consistency*
> objective; **batch / LR / EMA decide *which* fixed point and whether it's a good generator.** For a
> bootstrapped target they are **not "speed-only" knobs — they move the answer.**

### Final — recommended new training parameters for the iMF DiT (A/B these)

| Param | Current | Recommended | Why |
|---|---|---|---|
| **effective batch** | 64 (`bs32 × accum2`) | **256+** (`bs32 × accum8`, or `bs64 × accum4`) | average the **high-variance JVP target** → less-biased fixed point (Mech. 1 / §setup). **Do this first.** |
| **learning_rate** | 5e-4 | **1e-4** | DiT-stable; high LR on a moving target locks bad fixed points |
| **lr warmup** | none (`warmup_epochs=0`) | **2–5k steps linear** | let `v`/representation form before the JVP target is trusted |
| **ema_decay** | 0.995 | **0.9999** | deploy a stabler averaged snapshot (eval uses EMA weights) |
| **meanflow_aux_weight** | 0.05 | **0.1–0.25** | stronger shared `v`-anchor pulls the bootstrap toward the true instantaneous velocity |
| **meanflow_r_equals_t_frac** | 0.25 | **0.25–0.50** | more FM-anchor mass (`h=0`) grounds the field; the paper uses up to ~50% |
| **n_train_steps** | 100k | **200–300k** (keep sample budget after batch↑) | cover the rare **long-`h` / near-data** regime |
| **meanflow_cfg_omega** (eval) | 4.0 | **0.0** | orthogonal eval fix (H1/D1) — stop amplifying a meaningless signal |
| `dit_condition_on_t` | False | **False** (leave) | H3 rebutted — official is t-blind; not the lever |

**Order to try:** (1) **batch↑ + LR↓ + warmup** (attacks Mechanism 1 / the biased fixed point), then
(2) **EMA↑**, then (3) **aux/anchor↑**; (4) **CFG=0** is the independent eval-side fix from D1. Re-judge
with the **quality metrics (T1–T2, D6)** — *not* the converged loss, which is the very number that misled
us here.

---

## 4C. Config-semantics gotchas — what each param *actually* does (train vs eval vs path)

The config looks chaotic because a key can live in the **train block**, the **plan block**, or both, and
its **effective domain** is not obvious from where it sits. Two that genuinely mislead:

### `meanflow_cfg_omega` — split-role, and subtly broken in training
- **Not eval-only, not fake.** In **training** (`_p_losses_meanflow_jvp:425–433`) `ω>0` feeds the network
  an **ω-conditioning token** (`omega_c = full_like(t, 4.0)`, held constant through the JVP) — but it does
  **no guidance** (no cond/uncond mixing). In **eval** (`p_sample_loop:230` → `_predict_velocity:163`) the
  same value **both** conditions the net **and** is the **guidance scale** in `u_uncond + ω·(u_cond−u_uncond)`.
- **⇒ train and eval `ω` MUST match** — a different `ω` at eval feeds an OOD token the net never trained on.
- **The defect:** training feeds a **constant `ω=4` to every sample**, so the net never sees `ω` *vary* and
  **cannot learn ω-dependence** — the ω token is just a fixed bias. The official **samples `ω` per batch**
  so the net becomes *guidance-aware*. Ours is not ⇒ at eval, CFG extrapolates along an axis the network
  treats as constant. **This is a second, independent reason the DiT's CFG is meaningless (on top of the
  dummy class token in H1).** Fix: either `ω=0` (D1), or **sample ω during training** if you want real CFG.

### `action_weight` — training-only compute, **path-only at eval**
- Feeds `get_loss_weights:132` (`loss_weights[0,:action_dim]=10`), used **only** in the training loss
  (`:451`). `p_sample_loop` computes **no loss**, so `action_weight` **never touches the generated
  trajectory at eval.** The eval script doesn't read `args.action_weight` at all.
- Its **only** eval role: the `aw{action_weight}` token in `diffusion_loadpath`/`prefix` (`:825,862`) — it
  must match the trained value so the **checkpoint folder resolves**. (So: yes it "plays a role" in eval —
  but only **path resolution**, not the math.)

### The effective-domain map (read this before touching the config)

| Param | Train: compute? | Eval: compute? | In loadpath? | Notes |
|---|---|---|---|---|
| `imf_objective` | ✅ selects loss | — (model from pickle) | ✅ | path must match trained |
| `time_beta_alpha_v3` / `_beta_v3` | ✅ samples `t` (`loss()`) | ❌ (sampler uses uniform `τ=i/N`) | ✅ | **train-only math**; at eval it's *path-only* |
| `flow_steps_v3` | ❌ | ✅ NFE in `p_sample_loop` | ❌ | **eval-only** (set NFE here) |
| `meanflow_cfg_omega/t_min/t_max` | ✅ conditioning (constant) | ✅ conditioning **+ guidance** | ❌ | **must match**; constant-ω defect above |
| `dual_head`, `interval_cfg`, `dit_*`, `imf_backbone` | ✅ builds arch | ✅ rebuilds arch | `imf_backbone` ✅ | **must match or `state_dict` fails** |
| `meanflow_aux_weight`, `_r_equals_t_frac`, `_adaptive_p/c` | ✅ loss | ❌ | ❌ | **train-only** |
| `action_weight` | ✅ loss weight | ❌ | ✅ `aw{}` | **train-only math; eval = path-only** |
| `n_train_steps`, `batch_size`, `lr`, `ema_decay`, `warmup` | ✅ | ❌ | ❌ | **train-only** (the §4B knobs) |

**Rule of thumb:** at **eval**, the model architecture + weights come from the **pickled checkpoint**; the
plan block only (a) sets **sampler** knobs (`flow_steps_v3`, the CFG trio), (b) supplies **arch flags** that
must equal training (or load fails), and (c) fills the **path** so the right checkpoint is found. Anything
that's purely a **loss** knob (`action_weight`, `aux_weight`, schedule `time_beta_*`) is **inert at eval
except where it appears in the path.**

### Practical: ω=0 — eval-only (diagnose) vs both (fix); and `interval_cfg` / `action_weight`

**Setting `meanflow_cfg_omega=0` is the move — the only question is *where*.**

- **Hard rule:** train ω and eval ω must be **equal**. A mismatch feeds the net an ω-token it never
  trained on → OOD → garbage. The **one sanctioned exception** is the eval-only test below, because you
  are probing an *already-trained* model whose train-time ω you cannot retroactively change.

- **Eval-only ω=0 — quick DIAGNOSTIC, no retrain.** On the existing ω=4 checkpoint, set
  `meanflow_cfg_omega: 0.0` in **`plan_fm_v3_imeanflow` only**. Kills the guidance extrapolation
  instantly; introduces a *tiny* OOD shift (the ω-token moves 4→0) that is acceptable for a test. If the
  plan de-explodes ⇒ **H1 confirmed**. This is "enough" only in the sense of **confirming the cause**, not
  as the final state.

- **Both ω=0 — the real FIX, needs a retrain.** Set `meanflow_cfg_omega: 0.0` in **both** the train and
  plan blocks. Clean, consistent, no OOD. This is the going-forward default for trajectories (there is no
  real signal to guide on, so CFG only risks H1).

**Do you also need `interval_cfg=False`? No — ω=0 is enough.**
- `interval_cfg` is an **architecture** flag (it *builds* the ω/t_min/t_max layers, baked into the
  `state_dict`); **ω=0 makes those layers inert.** `interval_cfg=False` merely *deletes* the now-dead
  params (cosmetic) and **changes the `state_dict`**.
- On the **existing checkpoint you cannot flip it** (load would fail) — so ω=0 is your only lever anyway.
- On a **retrain** it's optional tidy-up; if you do it, set it in **both** blocks. Otherwise skip it.

**`action_weight`: leave it.** It still exists and is used — but **training-only** (weights the first
action 10× in the loss; at eval it only fills the `aw{}` folder path). It is a deliberate controller
choice (MPC executes the first action), **not a bug**. Don't touch it until after CFG-off + the §4B
training-regime fixes; only then consider flattening it (→1) if the tail is still loose — and that needs a
retrain with both blocks kept matched.

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
