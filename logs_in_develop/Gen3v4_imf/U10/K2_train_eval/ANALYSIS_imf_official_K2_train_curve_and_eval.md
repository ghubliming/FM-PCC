# iMF-official (U10) — K2 train + eval: why it's *stable* but *not extraordinary*

**Run:** `imf_train` job 23392 (seed 6, `obj=imf_official`, `bb=dit`, 100 k steps, git `b9fa1c6`)
**Eval:** `eval_imf` job 23420 (K2, EMA, CFG-off) · logs in `temp/00_25_08_imf_train_23392.log`, `temp/12_29_46_eval_imf_23420.log`
**Checkpoint:** `H8_..._objimf_official_bbdit_tslogit_normal/6`

**One-line verdict:** the *headline* loss curve looking "non-converging" is a **measurement artifact** of the adaptive loss — the model *does* fit (the real MSE curves drop 3–5×). But it plateaus **coarse** (summed-MSE ≈ 2–4 with recurrent spikes), and the reasons K2/K10 aren't sharp are **fundamental to average-velocity FM on this task**, not a tuning miss. Details below.

---

## 0. First: read the RIGHT curve. The "flat loss" is not a convergence failure.

The `imf_official` objective (`imf_diffusion.py:744–749`) is the **official self-normalizing adaptive loss**:

```python
adp = lambda L: L / (L + 0.01).detach().pow(1.0)   # p=1, eps=0.01
loss = (adp(loss_u) + adp(loss_v)).mean()
```

`adp(L) = L/(L+0.01)` is **bounded in [0,1)** for every sample: any residual `L ≫ 0.01` maps to ≈ 1.0. So the reported `loss`/`diffusion_loss` **cannot** move much — two saturating terms summed ⇒ it sits near ~2.0 (and `loss` = that /2 because `gradient_accumulate_every=2`, `training.py:145,199`, which is exactly why the log shows `loss≈0.95` next to `diffusion_loss≈1.9`).

> **Watching `loss`/`diffusion_loss`/`test/loss` for convergence is meaningless here — they are a *ratio*, flat by construction.** This is the whole point of the adaptive loss (equalize per-sample gradient magnitude), and it is why wandb "doesn't look converged."

**The real signals are the un-normalized MSEs** (`raw_mse`=`loss_u.mean()`, `aux_loss`=`loss_v.mean()`, `a0_loss`), and they *did* converge:

| metric | what it is | ep0 | end | verdict |
|---|---|---|---|---|
| `raw_mse` | u-head (avg-velocity) summed MSE, 48 dims | 11.3 | ~2–4 (spikes 45,71,15,14) | dropped ~3–5×, **plateaus coarse** |
| `aux_loss` | v-head (instant-velocity) summed MSE | 9.7 | ~1.4–3 | dropped ~4× |
| `a0_loss` | action MSE at horizon step 0 | 1.49 | 0.03–0.14 | **converged well** |
| `test/loss` | adaptive (bounded) | 1.00 | 0.97 | monotone but meaningless magnitude |

So: **training worked.** The model is not diverging, not exploding (K2 fix held). But it settles on a **coarse** field, and that coarseness is the "not smooth / not extraordinary" you see.

---

## 1. What the coarse plateau means, numerically

`raw_mse` is a **sum over 48 dims** (H8 × state 6). A plateau at ≈ 3 ⇒ per-dim residual ≈ √(3/48) ≈ **0.25 in normalized units** on the average-velocity field. That is not "converged to the data manifold"; it's "learned the gross flow, missed the fine structure." When a 2-step (K2) sampler integrates a field that's ~0.25/dim off, the endpoint drifts — hence non-smooth, slightly-off trajectories, and the raw `diffuser` variant in eval violating constraints (tracking error **0.132**, 12–17 violations before the projection brakes clean it to 1.0).

**The recurrent spikes** (`raw_mse` 45 @ep27, 71 @ep43, 15 @ep31, 14 @ep89; `test/a0_loss` 3.48 @ep28, 1.23, 1.12, 0.66) are the **JVP-tangent variance**: `imf_official` feeds the *predicted* `v_c` as the JVP direction (`imf_diffusion.py:734`). When the v-head is momentarily wrong, `u_target = v_g + h·du_dr` (line 741) gets a bad tangent → a huge target → a spike. This is inherent to *improved*-MeanFlow (predicted-v tangent) and is the price of its defining feature. It keeps the field from settling smoothly.

---

## 2. Why K2 **and** K10 are both mediocre → it's the **field**, not the step count

If K10 were clearly better than K2, the bottleneck would be *discretization* (too few ODE steps). You report K10 ≈ K2. That is the diagnostic tell: **more steps of a biased field don't fix bias.** The ceiling is the field quality (§1), so adding steps buys almost nothing. Three fundamental drivers of that bias:

### (a) The training interval distribution starves exactly the regime K-step sampling uses
`_p_losses_imf_official` draws `t,r` from **two independent logit-normals concentrated near data** (`τ→1`, `imf_diffusion.py:667–670`) and then makes **50 % of samples FM anchors** with `h=t−r=0` (`meanflow_data_proportion=0.5`, line 671–673). Result: the interval width `h` seen in training is **heavily skewed toward 0**. But a K2 sampler takes **h≈0.5 per step**; even K10 takes h≈0.1. The model is trained mostly on `h≈0` (instantaneous) and asked at eval to trust **large-h average-velocity** predictions it rarely practiced. Classic MeanFlow few-step degradation — and here it's baked into the sampling proportions.

### (b) Average-velocity is a *2-time-argument* object — far more data-hungry than plain FM
UNet-FM fits `v(z,t)` (one time). iMF fits `u(z,r,t)` (an interval field) **plus** a v-head **plus** their JVP consistency. The dataset is **96 demonstrations** (`buffer: observations (96,150,4)`). For the harder object, 96 trajectories is severely data-starved, so iMF's field is structurally coarser than FM's on *this* task **regardless of tuning**. This is the honest reason "iMF < UNet-FM/DPCC here" — not a bug.

### (c) Eval runs with the guidance machinery **switched off**
Trained with CFG `ω∈[1,7]` (`smax=7`) on a narrow interval `[0.4,0.6]` (`cfg_omega=4, t_min=0.4, t_max=0.6`). Eval overrides to **`ω=1.0` (CFG off), `[0,1]`** (see the 5 `[ config->pkl ] INFO` lines in the eval log — this is the two-tier override working as designed, and CFG-off is what *stops the earlier explosion*). Net effect: the capacity the model spent learning the guided/null-token split is **unused at eval**, and with `condition_guidance_w=0` the returns-conditioning gives no steering either. The model therefore emits the **marginal/averaged** behavior — smoother-but-blander, i.e. "not extraordinary." You cannot currently turn CFG *on* to sharpen it, because that reintroduces the explosion (U9/U10 kill-chain). That tension is the real limitation of this checkpoint.

---

## 3. Things that are **inert** on this path (don't waste tuning on them)

- `u_loss_weight=1.0 / v_loss_weight=0.1 / meanflow_aux_weight` — **ignored** by `imf_official`. That path hardcodes **equal** `adp(loss_u)+adp(loss_v)` (`imf_diffusion.py:749`; `info['u_weight']=v_weight=1.0`). They only affect the legacy `meanflow_jvp`/single-t arms (lines 481, 628). So the v-head is *not* under-weighted on this run — good — but also you can't rebalance u/v via config here.
- `meanflow_adaptive_p / meanflow_adaptive_c` from config — also ignored; the official `adp` hardcodes `p=1, eps=0.01` (line 748). If you want to experiment with the adaptive strength you must edit that line.

---

## 4. If you want to push it (ranked by expected leverage, cheapest first)

These are hypotheses to A/B, not guarantees — the §2(b) data ceiling caps all of them.

1. **Rebalance the interval distribution toward large h.** Lower `meanflow_data_proportion` (0.5→0.25) and/or widen the (t,r) sampler so more mass lands at `h≈0.3–0.7`. Directly attacks §2(a) — the single most likely lever for K2/K10 smoothness. *(Edit the sampling in `_p_losses_imf_official`; `data_proportion` is already a config knob.)*
2. **Make CFG usable at eval instead of off.** The reason ω>1 explodes is the null-token branch; a milder guided-eval (small ω, e.g. 1.5, on the trained interval) *with* the U9/U10 clamp may recover sharpness without blow-up. Worth a single K10 sweep of ω∈{1.0,1.3,1.6}.
3. **Longer / re-warmed schedule for the u-field.** LR cosine hit 0 at ep99 while `raw_mse` was still spiky (not settled). The field never got a low-LR fine-anneal at *stable* variance. A short constant-low-LR tail (or EMA-heavier) may shave the plateau.
4. **Accept iMF as the generative brain + lean on the brakes.** Eval already shows **1.0 success** on every *projected* variant (`dpcc-c-tightened`, `post_processing-tightened`, `dpcc-t-tightened`). If the research claim is "FM/iMF + DPCC projection," the coarse raw field is *tolerable* — the projection is doing its job. The raw-`diffuser` number is the honest "generative quality" metric and it's mediocre; that's the story to report, not hide.

---

## 5. Do we still have training-side tuning headroom? — **Yes, but bounded.**

**Short answer:** yes, there is *real* headroom, and it is **not** just wishful tuning — the evidence says the plateau is **not** a clean data-noise floor:
- `raw_mse` was **still spiking** (45, 71, 14) when the cosine LR hit **0** at ep99 — the field never got a stable low-LR fine-anneal. That's *optimization* headroom left on the table.
- Both heads plateau **well above** the adaptive floor (`raw_mse≈3`, `aux≈2`, not ≈0.01), and the interval distribution is **mis-matched to the sampler** (§2a) — that's *configuration* headroom, independent of data.

**But the §2b ceiling is not crossable by any hyperparameter:** the avg-velocity field `u(z,r,t)` is a 2-time object learned from **96 demos**. Tuning can lower the plateau and sharpen K2/K10; it will **not** manufacture information that isn't in 96 trajectories, so beating UNet-FM/DPCC on *this* task remains unlikely. Set expectations accordingly: the goal of tuning here is "close the coarse-plateau gap," not "win outright."

### Live knobs on the `imf_official` path (what tuning can actually touch)

| knob (current) | line | leverage | direction & mechanism |
|---|---|---|---|
| `meanflow_data_proportion` (0.5) | `:671` | **HIGH** | ↓ **0.25**. It's the fraction of `h=0` FM anchors; 0.5 means half the batch teaches *instantaneous* velocity while K2 needs *large-h* average velocity (§2a). Lowering shifts supervision to the interval regime the sampler queries. Single biggest lever. |
| `p_mean` (−0.4), `p_std` (1.0) | `:667-668` | **HIGH** | ↑ `p_std` (→ ~1.4) or add a uniform-`t,r` component. Two logit-normals near `τ→1` produce **small** `h`; widening spreads mass so `h∈[0.3,0.7]` (the K2 step size) is actually trained. Pairs with the row above. |
| `meanflow_cfg_smax` (7.0) | `:687` | MED | ↓ (→ 2–3). A ceiling of 7 trains a very aggressive guided branch that the CFG-off eval never uses and that drives the null-token instability (§2c). A gentler ceiling may make a *small* eval-ω usable. |
| `meanflow_class_dropout_prob` (0.1) | `:716` | MED | the null-token training rate. If you want CFG usable at eval, this + `smax` must be co-tuned; too low → null token undertrained (the original explosion root). |
| `meanflow_cfg_omega / t_min / t_max` (4.0 / 0.4 / 0.6) | `:687-688` | MED | training guidance interval. Widening `[t_min,t_max]` spreads guidance supervision; only worth it in tandem with the CFG-usable-at-eval goal. |
| trainer: `n_train_steps`, LR tail, `ema_decay` (0.995) | trainer_cfg | MED | give a **constant low-LR tail** (don't cosine-to-0 while spiky) and/or heavier EMA — directly attacks the "froze on a noisy plateau" evidence above. |

### Inert on this path — do **not** tune these (they do nothing for `imf_official`)
`action_weight` (10) and all `loss_weights` — the official loss is an **unweighted per-sample sum** (`:744-749`, "no DPCC weights"); `u_loss_weight`/`v_loss_weight`; `meanflow_adaptive_p`/`_c` (hardcoded `p=1,eps=0.01` at `:748`); `meanflow_r_equals_t_frac` (that's the *other* arm at `:564`, not `:671`). Changing any of these will look like it "did nothing" — because it does nothing here.

### Recommended minimal sweep (cheap, high-information)
1. **Retrain A/B:** `meanflow_data_proportion 0.5→0.25` **+** `p_std 1.0→1.4`. One run. Directly tests the §2a hypothesis — watch `val/raw_mse` plateau height and K2 smoothness, *not* `loss`.
2. **No-retrain:** at K10, sweep eval `ω∈{1.0, 1.3, 1.6}` with the U9/U10 clamp on. Tests whether §2c sharpness is recoverable for free.
3. Only if (1) helps: add a **constant-LR tail** (last ~10 k steps) to settle the spiky plateau.

If (1)+(2) don't move `val/raw_mse` below ~2 or sharpen K2, you've hit the §2b data ceiling — at which point the honest call is "use iMF as the generative brain behind the DPCC brakes (eval already 1.0 on projected variants) and report the raw `diffuser` quality as-is."

### Yes — the step budget can (and probably should) be **cut**
`n_train_steps=100 k` looks like overkill for this run:
- `raw_mse` does most of its work in the **first ~10 epochs** (11.3 → ~4 by ep10), then just oscillates on the 2–4 plateau for the remaining **~90 epochs**.
- `a0_loss` is essentially converged by **ep10–20** (→ ~0.1).
- `test/loss` (adaptive) makes most of its 1.00→0.97 move in the **first half**, flat after.

So **~40–50 k steps captures nearly all the useful learning** and roughly **halves the ~12 h wall-clock** — which matters most because it makes the §5 sweep iterate 2× faster. Two caveats:
- **Reduce via `n_train_steps`, not by killing early** — the cosine LR anneals over the *full* budget, so setting `n_train_steps=50000` gives a proper anneal-to-0 at 50 k (killing a 100 k run at step 50 k leaves LR high and the field unsettled).
- **Don't judge the cut by `loss`** (flat by construction, §0) — confirm the shortened run's **`val/raw_mse`** plateau matches the 100 k run's. `ema_decay=0.995` (≈200-step window) is fine at 40–50 k.

Caveat on the ceiling: fewer steps saves compute, it does **not** lower the §2b data ceiling — a 50 k run will plateau at about the same `raw_mse`, just sooner.

---

## 6. Proposed config for the **next TRAIN + EVAL (K10)** — exact settings

All edits are in `config/avoiding-d3il.py`. "Baseline" = the settings that produced job 23392/23420. Only the **bold** rows change; everything else stays.

### 6A · TRAIN block — iMF (`config/avoiding-d3il.py`, imf train block ≈ L482–550)

| Parameter | Line | Baseline | **Next run** | Leverage / why |
|---|---|---|---|---|
| **`meanflow_data_proportion`** | 489 | 0.5 | **0.25** | **HIGH** — fewer `h=0` FM anchors → more large-`h` (interval) supervision that K-step sampling actually queries (§2a). |
| **`p_std`** | 550 | 1.0 | **1.4** | **HIGH** — widens the `(t,r)` logit-normals so `h∈[0.3,0.7]` (the K2/K10 step size) gets real training mass (§2a). |
| **`n_train_steps`** | 532 | 100000 | **50000** | compute — ~all useful learning is done by ~ep40; halves the ~12 h wall-clock, 2× faster sweeps (§5). Cosine LR re-anneals to 0 at 50 k automatically. |
| `meanflow_cfg_smax` | 488 | 7.0 | **3.0** *(optional)* | MED — gentler guided branch; only worth it if you also intend a CFG-on eval (row-set 6B note). Leave at 7.0 if not. |
| `p_mean` | 549 | −0.4 | −0.4 (keep) | keeps median anchor ≈ 0.40 on the τ axis. |
| `meanflow_class_dropout_prob` | 490 | 0.1 | 0.1 (keep) | null-token rate; change only in tandem with CFG-on eval. |
| `imf_objective` / `imf_backbone` | 482 / 510 | `imf_official` / `dit` | keep | identity — DiT is required (unet no-ops cond_drop). |
| `meanflow_cfg_omega/t_min/t_max` | 501–503 | 4.0 / 0.4 / 0.6 | keep | train-time guidance interval. |

### 6B · EVAL / plan block — running **K10** (`config/avoiding-d3il.py`, imf plan block ≈ L845–896)

| Parameter | Line | **Next eval (K10)** | Note |
|---|---|---|---|
| **`flow_steps_v3`** | 854 | **10** | this is the K in K10 (baseline plan block was `2`=K2). Sampling knob → overrides pkl, prints `[ config->pkl ] INFO`. **This is the only knob that sets K.** |
| ~~`ode_inference_steps_v3`~~ | — | **do NOT set** | **DEAD** — `imf_diffusion.py:104` overwrites it to `=flow_steps_v3` and sampling never reads it (`p_sample_loop` uses `self.flow_steps_v3`, `:223`). The config itself labels it *"DEAD code (compatibility alias)"*. Ignore. |
| `meanflow_cfg_omega` | 877 | 1.0 (CFG **off**) | baseline eval operating point. *Optional experiment:* sweep `{1.0, 1.3, 1.6}` with the U9/U10 clamp on to test if §2c sharpness is recoverable — no retrain needed. |
| `meanflow_cfg_t_min` / `t_max` | 878–879 | 0.0 / 1.0 | inert while ω=1; set to the train interval only if you turn CFG on. |
| `condition_guidance_w` | 880 | 0.0 | the real returns-CFG neutralizer; keep 0. |
| `returns_condition` | 885 | True | match pkl (inert; neutralized by `condition_guidance_w=0`). |

### 6C · ⚠ Mirror the identity knobs, or eval will WARN
`meanflow_data_proportion`, `p_std`, `p_mean`, `meanflow_cfg_smax`, `meanflow_class_dropout_prob` are **identity keys** (not in the sampling-override allowlist), so at eval the **pkl value wins** and a `[ config->pkl ] WARNING` fires on any mismatch. They don't affect sampling behavior, but to keep the console clean **copy the new TRAIN values into the plan block too**:

| plan-block line | set to (match new pkl) |
|---|---|
| `meanflow_data_proportion` (888) | **0.25** |
| `p_std` (858) | **1.4** |
| `meanflow_cfg_smax` (887) | **3.0** *(only if you changed it in 6A)* |
| `p_mean` (857), `meanflow_class_dropout_prob` (889) | unchanged (−0.4 / 0.1) |

**Read-the-result reminder:** judge the new run by **`val/raw_mse`** plateau height and K10 trajectory smoothness — **not** by `loss`/`test/loss` (flat by construction, §0). Success = `val/raw_mse` settles below ~2 and K10 trajectories look smoother than this run's.

---

## 7. "More K → *more* chaotic" (raw `diffuser`) — is it TRAIN or EVAL?

**Verdict: TRAIN (the field). The eval sampler is faithful — I checked it.** And critically, **expecting iMF to smooth with rising K like DPCC is a category error** — that behavior is *not* what MeanFlow does, so the K-sweep is the wrong lens. Reasoning:

### The sampler is a correct MeanFlow composition (not the bug)
Eval uses `legacy_euler`/`euler` (`config` L861–862). The loop (`imf_diffusion.py:284–345`) does, per step `i` of `K`:
```
τ = i/K ;  h = dt = 1/K
x += u(x, r=τ, h=dt) · dt
```
This is the **textbook MeanFlow multi-step**: anchor the query at the interval *start* `r=i/K`, width `h=dt`, displace by `u·h`. It matches training exactly (in `_p_losses_imf_official` the net is queried at the noise-side anchor `r` with `h=t−r`). Query stays in-domain (`τ+h=(i+1)/K ≤ 1`), final step lands at `τ=1` (data). **So the composition is right — the chaos is not a sampler bug.**

### First, what the iMF paper (arXiv:2512.02012v1) *actually* claims about NFE — I checked
Its validated results are **only NFE 1 and 2**, on **ImageNet 256²** (FID metric, huge sample):

| NFE | iMF-XL/2 FID |
|---|---|
| 1 | 1.72 |
| 2 | **1.54** (−0.18) |

So the paper does say **2-NFE > 1-NFE** — a *small* refinement — with **one-step as the primary design**, and gives **no results at NFE ≥ 4** (silent beyond 2). The proper reading of iMF is **K-invariance**: a well-fit average-velocity field lands on data in *one* step, a 2nd step refines marginally, and there's no reason to integrate further. It is **not** an ODE integrator that keeps improving with K.

### Correcting my earlier framing
Two honest corrections to what I wrote above:
1. **iMF does not predict "monotonically worse with K."** I overstated that. Its prediction is **K-invariant + a tiny 1→2 gain**. Neither DPCC's "smoother with K" nor "worse with K" is the iMF law — a *well-fit* iMF should look **flat** across small NFE. Your K10/K50 chaos is **off-paper** (the paper never tests there), so it neither confirms nor refutes the paper.
2. **Different metric.** The paper measures **FID over a sample distribution**; you're eyeballing **one raw trajectory's smoothness**. A field can be distributionally acceptable yet emit individually jittery paths. Not the same axis — don't read our single-trajectory jitter as a FID-style refutation.

### Why *our* raw trajectory still degrades as K rises (an **under-fit-field** effect, not an iMF law)
Given the field is under-fit (§1–§2), the off-paper high-K regime behaves badly for two reasons:
1. **As K↑, h→0, and `u(x,r,h) → v(x,r)`** — the *instantaneous* velocity, which here is the coarse, JVP-trained, **spiky** object (`aux_loss`/`raw_mse` plateau ≈ 2–3, §1). High-K integrates the model's **roughest** field, many times.
2. **A non-smooth field surfaces as jitter under many small steps.** K2 samples its kinks twice → coarse-but-smooth; K50 samples them 50× → every kink becomes a visible wiggle. It's the field's **roughness being resolved**, not error magnitude growing.

A **well-fit** iMF would *not* show this (the field would be smooth, so high-K ≈ low-K). So the K-degradation is a **symptom of the under-fit field**, fully consistent with the paper — the paper's clean 1/2-NFE result assumes a field ours hasn't reached on 96 demos (§2b).

### The comparison that actually matches the paper: **K1 vs K2** (you haven't run it)
Your K2-vs-K10 sweep is **not** the paper's test. The paper's headline is **1-NFE vs 2-NFE**. So run **K1** and compare to **K2**:
- **K2 sharper than K1** ⇒ you *reproduced the paper's direction* (2>1) — the field is fitting; push it with the §6 retrain and stay at NFE≤2.
- **K1 ≈ K2, both coarse** ⇒ the field is too under-fit to even show the 1→2 refinement ⇒ §2b data ceiling is binding.
- **K1 already coarse** ⇒ proves it's **100 % the field (TRAIN)**: one query, no accumulation, no roughness-resolution — any coarseness is the map itself.

Either way, **evaluate iMF at NFE 1–2 (the paper's regime), not at K10/K50.** K1 is a single `u`-query — no accumulation, no roughness-resolution — so it isolates the field cleanly and costs nothing to run alongside your K50.

### What the §6 retrain will / won't do
- **Will:** target **low-K (K1–2) sharpness** — the *correct* iMF operating regime. More large-`h` mass (`data_proportion↓`, `p_std↑`) directly improves the big-jump average-velocity the one-shot map uses.
- **Won't:** make **high-K "DPCC-smooth."** Stop chasing high-K for iMF — it's not its regime, and a smoother field will show up as *better K1/K2*, not as a DPCC-style K-convergence curve.
- **If even K1/K2 stays coarse after retrain** ⇒ §2b data ceiling is binding ⇒ honest call: use iMF as the generative brain behind the DPCC brakes (projected variants already 1.0) and report raw `diffuser` as the method's limitation at this data scale.

> **Judge iMF at K1–K2, not by a K-sweep.** The K-sweep is a diagnostic (it proved the field is rough), not the metric.

---

## 8. Bottom line for the writeup

- **Not a training bug.** K2 stability fix held; a0/raw_mse/aux all dropped; adaptive `loss` is flat *by design*.
- **The model is genuinely coarse** (raw_mse plateau ≈ 3, per-dim ≈ 0.25) because: interval-sampling starves large-h (§2a), average-velocity is data-starved at 96 demos (§2b), and eval runs CFG-off so trained guidance is dead weight (§2c).
- **On the K-sweep (§7):** the paper (arXiv:2512.02012v1) validates **only NFE 1→2** (FID 1.72→1.54), one-step-primary, silent past 2 — iMF is **K-invariant**, not an ODE integrator. Our K10/K50 "chaos" is **off-paper** and is a symptom of the **under-fit field** (high-K resolves its roughness), *not* a refutation of the paper and *not* an iMF "worse-with-K" law. **The paper's real test is K1 vs K2 — still unrun.** Judge iMF there.
- **iMF here is faithful but structurally out-matched by UNet-FM/DPCC on this small task** — the honest, fundamental reason, consistent with `../debug_notes/INVESTIGATION_imf_fidelity_vanilla_vs_improved_meanflow.md`.

---

## 9. Addendum — full K-sweep K1…K100 (empirical): **commitment ↔ smoothness trade-off**

*Added after running the actual sweep (K∈{1,2,10,50,100}, **one seed**). Metrics: raw-trajectory smoothness = the `diffuser` (unprojected) variant; success & constraint = the projected **`dpcc-r-tightened`** variant (user's "dpcc-rtc"). After projection every variant is smooth — smoothness below always means the **raw** `diffuser` path.*

### What the sweep actually showed
| K (NFE) | raw `diffuser` smoothness | task (success & constraint), `dpcc-r-tightened` |
|---|---|---|
| 1 | coarse / angular (few segments) | **best** on top-left & top-right-hard |
| 2 | coarse / angular | **best** on top-left & top-right-hard |
| 10 | **slightly** smoother | — |
| 50 | smoother | **worse** than K1/2 (top-L/R) |
| 100 | smoother — but **still not FM/DPCC-smooth** | **worse** than K1/2 (top-L/R) |
| any | (both-hard) | all K reach full success & constraint; avg steps ≈ equal |

Two things move in **opposite** directions as K rises: raw smoothness **improves a little**, task success **degrades**. That opposition is the whole story.

### Correction to §7 (I had the smoothness direction wrong)
My §7 "high-K resolves the field's roughness → *more* jitter" is **not** what the data shows. Higher K is **mildly smoother**, not more chaotic. **Supersede the §7 jitter mechanism with this:** the trade-off below. (§7's *verdict* — TRAIN not EVAL, sampler is faithful — still stands; only the "worse-with-K jitter" mechanism was wrong.)

### The mechanism: **NFE trades mode-commitment for line-smoothness on a multimodal task**
`avoiding-d3il` is **multimodal by construction** — at the obstacle you go *left* or *right*, two valid demo modes. That is the key.

- **Low K (1–2): a big average-velocity jump *commits to one mode*.** `u(x, 0, 1)` integrates the whole interval in one/two shots; the jump is dominated by whichever demo mode the initial noise is closest to, so the endpoint lands **on a real, single-mode trajectory**. The path is *angular/coarse* (only 1–2 segments) but **decisive and geometry-correct** → after projection it satisfies goal + constraints → **best task metrics**.
- **High K (50–100): fine integration of the instantaneous field *averages across modes*.** As K↑, `u→v(x,τ)`, and the under-fit instantaneous field near the decision point is a **blend of the left and right modes** (it was fit to both). Integrating that blend in many small steps drifts the path toward the **between-mode middle** — a smoother *line* that heads into the ambiguous region (often toward the obstacle it should skirt). Projection then smooths it further but **cannot restore the lost mode decision** → **worse success/constraint**.

So: **more steps buy a smoother line at the cost of averaging away the mode commitment** — and on a decision task, commitment is what the metric rewards. This is exactly *why* iMF is designed one-shot, and the task numbers agree with the design: **K1/K2 win.**

### Why raw smoothness rises with K yet never reaches FM/DPCC
- *Rises with K:* 1–2 giant jumps are piecewise-linear/angular; 50–100 small steps trace a finer polyline → visually smoother.
- *Never reaches FM/DPCC:* FM/DPCC integrate an **accurate instantaneous field**, so fine integration is genuinely smooth **and** on-mode. Our instantaneous field is **under-fit** (§1, `raw_mse` plateau ≈ 3), so even at K100 the polyline is smoother-but-still-coarse **and** mode-averaged. iMF's field simply isn't accurate enough (data ceiling, §2b) to be *both* smooth and committed the way FM's is. **Raw-trajectory smoothness is therefore a misleading proxy for policy quality here** — the smoother K100 path is the *worse* policy.

### Reconciliation with the paper (arXiv:2512.02012v1)
Consistent, not contradictory: the paper reports **1-NFE ≈ 2-NFE** (FID 1.72→1.54) and stops at 2 — precisely the regime where **commitment is intact**. Our sweep independently lands on the same operating point: **NFE 1–2 is where iMF should run.** The paper never enters the high-K mode-averaging regime; neither should we.

### Honest caveats (do not over-read this)
- **n = 1 seed**, and D3IL success is coarse (~2 trials/condition → 0/0.5/1.0 granularity, `±` of ~1–2 violations). "K1/2 > K50/100" is **suggestive, not established** — it needs a **multi-seed** rerun before it goes in a paper.
- both-hard being all-success at every K means projection can rescue that layout regardless; the discriminating evidence is **top-left/right-hard**, where the single-side mode decision actually matters.

### Implications
1. **Operate iMF at NFE 1–2.** Both the paper and your task metrics point there; ignore high-K for policy quality.
2. **Report task success, not raw smoothness, as the quality metric** — and note explicitly that raw smoothness *anti-correlates* with success here (a genuinely interesting result worth stating).
3. **The §6 retrain still applies and is now better motivated:** its goal is to make the **low-K committed jump** *also* smooth — i.e. sharpen the one-shot map (more large-`h` mass), not to chase high-K. If it works you'll see it as **better K1/K2** on *both* smoothness and success.
4. **Confirm with ≥3 seeds** at K1 and K2 before treating "low-K wins" as a finding.
