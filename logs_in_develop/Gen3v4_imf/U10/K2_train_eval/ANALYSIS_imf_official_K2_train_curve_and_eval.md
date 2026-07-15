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

## 7. Bottom line for the writeup

- **Not a training bug.** K2 stability fix held; a0/raw_mse/aux all dropped; adaptive `loss` is flat *by design*.
- **The model is genuinely coarse** (raw_mse plateau ≈ 3, per-dim ≈ 0.25) because: interval-sampling starves large-h (§2a), average-velocity is data-starved at 96 demos (§2b), and eval runs CFG-off so trained guidance is dead weight (§2c). K10≈K2 confirms it's field-bias, not discretization.
- **iMF here is faithful but structurally out-matched by UNet-FM/DPCC on this small task** — the honest, fundamental reason, consistent with `../debug_notes/INVESTIGATION_imf_fidelity_vanilla_vs_improved_meanflow.md`.
