# REPORT — `avoiding-d3il`: MeanFlow works on the U-Net, α-Flow does not. Why?

**Date:** 2026-08-30, rewritten 2026-08-31 · **Type:** code audit + reconstruction of the evaluation record
**Task:** `avoiding-d3il` (state-space, H=8, D=6) · **Code rev:** `938641c`
**Scope:** **Gen3v6** (`flow_matcher_v3_meanflow`) vs **Gen3v7** (`flow_matcher_v3_alphaflow`), backbone `unet` vs `sit` / `mf_dit`
**Nothing was run for this report** — no cluster job. It audits the code and re-reads the existing eval record and train logs. Items marked 🧪 are untested predictions.

**Supersedes** [`STUDY_why_af_sit_works_unet_not_and_mf_unet_works.md`](./STUDY_why_af_sit_works_unet_not_and_mf_unet_works.md) (2026-08-19) on its §4.3 / §6.2 / §7.3 / §10.

> **Rewrite note (2026-08-31).** This file previously grew by four append-only updates and had become
> self-contradictory (its verdict said "AF-UNet is fine" while its own raw-quality section said the opposite). It is now reorganised around
> the seven questions actually asked, and every earlier finding is folded into the section that answers
> one of them. **Gen14 was only ever used as the *trigger* for the "is α-Flow enabled?" check and as one
> capacity-control reference — it is confined to §4.4 and §8.3 and is not the subject of this report.**

---

## The seven questions this report answers

| # | question | answer | where |
|---|---|---|---|
| Q1 | AF works on SiT but is bad on the U-Net, while MF-UNet works well — is there a **bug**? | **No bug.** Two of the three "AF-UNet fails" data points are provenance artifacts; the third is real but is a *raw-field* deficit, not a failure | §2, §3, §5 |
| Q2 | Is α-Flow **actually enabled** in the current setup, and was it in the past DA runs? | **Yes**, verified five ways including the train logs of the runs themselves — but it silently **breaks checkpoint selection** | §4.1–§4.3 |
| Q3 | Does `avoiding` have the **α → 0 snap** that Gen14 U10 found? | **The snap: yes, bit-for-bit. The damage: no** — no `avoiding` eval ever loaded a snapped checkpoint | §4.4 |
| Q4 | AF-UNet's **raw trajectory quality** is bad and is rescued by IK/projection — it does *not* beat MF-UNet | **Correct.** AF-UNet is behind MF-UNet in every unprojected / lightly-projected arm | §3 |
| Q5 | **Verdict:** is it my code / parameter setup, or something else? | **~85–90 % setup** — and the dominant piece is a *parameterisation* choice, not a defect | §0, §6 |
| Q7 | **How to test?** What is the mental model, what do I tune, and how do I find when K1/K2 `diffuser` quality matches MeanFlow? | Tune **`dt = α·h` against the backbone's time resolution**; endpoint is the **paired K2→K1 slope**, not the K1 level; two runs ≈ 8.5 GPU-h | §9 |
| Q6 | Can α-Flow be **theoretically better** than MeanFlow on the U-Net? Is there a real math reason it fails? Worth further testing? | **Same fixed point, so no better optimum** — but a real, non-obvious math reason it fails on this backbone, and a bounded case for testing | §7 |

---

## 0. VERDICT (Q5)

> **It is your setup — but the dominant cause is a *parameterisation* choice in the two-time conditioning, not a bug and not a limit of the U-Net.**
>
> The argument that closes the "or something else?" branch: **α-Flow contains MeanFlow.** At α = 0 the α-Flow target is byte-identical to Gen3v6's JVP target (gate G2, `af_diffusion.py:552`), and the shipped schedule sits at α = 0 for the last 29 k steps. **So the very weights that make MF-UNet work are a feasible point of the AF-UNet problem.** Capacity and expressivity of the U-Net are therefore exonerated — the solution lies inside its hypothesis class. Only the **training signal** can be failing to steer there, and the one component of that signal which is both α-Flow-only and backbone-dependent is the finite-difference bootstrap probe and how each backbone's time conditioning represents it (§6.1).

### 0.1 The 2×2 kills every single-factor explanation

| | **U-Net** (4.0 M) | **SiT / DiT** (10.0 M) |
|---|---|---|
| **MeanFlow** (Gen3v6) | ✅ works | ✅ works (`mf_dit`) |
| **α-Flow** (Gen3v7) | ❌ weak raw field, rough | ✅ works (`sit`) |

| candidate cause | killed by |
|---|---|
| "the U-Net is broken / too small" | MF-UNet works, and has **fewer steps in 13/13 arms** than `mf_dit` |
| "α-Flow is broken" | AF-SiT has the **best raw field in the matrix** (`diffuser` S&C 0.267) |
| "capacity — the SiT is 2.5× bigger" | the identical 2.5× gap exists for `mf_dit` vs MF-UNet, and there the **U-Net wins** (§8) |
| "the task / data / pipeline" | state-space `avoiding-d3il`, no vision; MF runs the same loader, normaliser and horizon |
| "expressivity — the U-Net can't represent the AF solution" | AF ⊇ MF at α = 0; the working MF-UNet solution is feasible for AF |

One failing cell out of four is an **interaction**, and the interaction lives in the wrapper code.

### 0.2 Apportionment — what share of "AF-UNet looks bad" sits where

| # | cause | share 🧪 | yours? | cost to remove |
|---|---|---|---|---|
| 1 | **§6.1 coordinate mismatch** — the U-Net is conditioned on `E_τ(r) + E_h(h)`, so α-Flow's probe `(r+dt, h−dt)` moves both terms in *opposite* directions and cancels to first order. The SiT is conditioned on `E_t(r+h) + E_r(r)`, where the endpoint term does not move at all | **~45 %** | ✅ a wrapper choice | ~20 lines, **same param count** (§10.4) |
| 2 | **§6.2 time-code resolution** — `freq_dim` is one knob for *both* channel width and time-embed width, so Fix_8's necessary 256 → 32 also cut the time code to 32-D / 16 frequencies (SiT: 256-D / 128), injected as a shift with no scale or gate | ~20 % | ✅ a config coupling defect | +0.06 M params (§10.3) |
| 3 | **§4.3 checkpoint selection** — AF's `state_best.pt` is picked by a loss whose dominant term is `0.75 + 0.25·α`, so every AF number on record is a mid-homotopy model; MeanFlow is unaffected | ~15 % | ✅ a selection bug | **zero training** (§10.2) |
| 4 | **Measurement debt** — stale seed 6 (§2.2), no `n_trials=20` row, K=2 only, 2.5× unmatched capacity | ~10 % | ✅ | one eval + one train (§10.5) |
| 5 | **Genuinely harder optimisation** — a bootstrapped self-distillation target has higher gradient variance than an analytic JVP (AF's train `raw_mse_u` tail hits **638.8** vs MF's spikes > 20), and a local-receptive-field conv trunk may be a worse host for it than global attention, independently of 1–2 | ~15 % | ❌ | only measurable once 1+2 are removed |
| 6 | not yet found | ~5 % | — | — |

This is **not** a second Fix_8. Nothing here is *wrong* in the sense of failing a gate. It is a faithful port of α-Flow's objective onto a backbone written in the wrong coordinates for it — a subtler and more publishable result.

---

## 1. The record, with provenance

Every AF/MF × backbone cell measured on `avoiding-d3il`, in date order. **Nothing above the Fix_8 line is interpretable.**

| date | cell | width | result | seeds | verdict |
|---|---|---|---|---|---|
| 07-25 | AF + DiT | — | 0.50 goal+constr | 6 (n=2) | ✅ |
| 07-25 | **AF + UNet** | **256 → 253 M** | **0.07**, per-dim RMS 0.96 | 6 (n=2) | ❌ **width defect** |
| 07-25 | MF + DiT | — | 0.49 goal+constr | 6 (n=2) | ✅ |
| 07-25 | **MF + UNet** | **256 → 253 M** | **0.14**, per-dim RMS 0.98 | 6 (n=2) | ❌ **width defect** |
| — | — | — | ↑ *`2d85f03` Fix_8: `freq_dim` 256 → 32 (253 M → 4.0 M)* | — | — |
| 08-07 | MF + UNet@32 | 4.0 M | `raw_mse_u` 19.3 → 1.90, per-dim RMS **0.199** | 6 | ✅ |
| 08-08 | **AF + UNet@32** | 4.0 M | per-dim RMS 0.34–0.44; `dpcc-t-tight` **1.00 / 58.4 / 0.030** | **7–10 only** ⚠️ | ⚠️ 4 seeds |
| 08-11 | AF + UNet@32, 5-seed roll-up | 4.0 M | `dpcc-t-tight` **0.833** / 61.8 / 0.0732 | 6–10 | 🔴 **contaminated — §2.2** |
| 08-11 | AF + SiT | 10.0 M | `dpcc-r-tight` **1.000** / 67.6 / 0.0202 | 6–10 | ✅ |
| 08-11 | MF + UNet@32 | 4.0 M | `dpcc-t-tight` 0.967 / 59.4 | 6–10 | ✅ |
| 08-11 | MF + `mf_dit` | 10.0 M | `dpcc-t-tight` 0.967 / 68.4 | 6–10 | ✅ |
| 08-15 | MF + UNet@32 | 4.0 M | `dpcc-t-tight` **0.993** `[0.980, 1.000]` | 6–10, **n=20 (300 ep)** | ✅ **verified** |
| 08-17 | AF + SiT | 10.0 M | `dpcc-r-tight` K1 **1.000**, 300/300 | 6–10, **n=20** | ✅ **verified** |
| — | **AF + UNet@32** | — | — | — | 🔴 **never run at n=20** |

**AF-UNet is the one cell in the matrix with no verified measurement.** That is why the impression "AF-UNet is bad" has survived unexamined for so long — and why §3, which *does* find a real deficit, is the load-bearing evidence rather than any of the S&C rows.

---

## 2. Q1 — the two "AF-UNet fails" data points are provenance artifacts

### 2.1 Cause #1 — the `freq_dim` width defect (fixed in `2d85f03`)

`freq_dim` was read as a "frequency embedding size" but is the U-Net's **channel width**:

```python
# unet1d_temporal_cond.py:105-110
dims   = [transition_dim, *map(lambda m: dim * m, dim_mults)]  # dim = freq_dim, dim_mults=(1,2,4,8)
self.time_dim = dim                                            # ...and the time-embed width
```

`freq_dim=256` ⇒ channels (256, 512, 1024, 2048) ⇒ **253.0 M** params against DPCC/FMv3ODE's **3.97 M**, on 96 demonstrations.

**This cause is objective-independent.** Both α-Flow and MeanFlow land at per-dim RMS ≈ 1.0 — per-dimension error equal to the full normalised data scale, i.e. a model that has fit nothing. `fix_1`'s conclusion *"the MeanFlow JVP objective requires the DiT"* was falsified by the same fix. **Nothing about the α-Flow objective is implicated by this row.**

### 2.2 Cause #2 — the stale seed-6 checkpoint behind the `0.833` 🔴

`DA_20260811_MF_UNet32_full5seeds_avoiding.md` §4.4 is where the current belief comes from:

> | family | DiT/SiT | U-Net | winner |
> | **MeanFlow** | `mf_dit` 0.967 / 68.43 | **`bbunet` 0.967 / 59.43** | **U-Net** |
> | **AlphaFlow** | **`bbsit` 1.000 / 67.60** | `bbunet` 0.833 / 61.77 | **SiT** |

That DA pools **seeds 6–10**. Its own companion — `Unet_study/RESULTS_20260809_…` §1, written two days earlier by the same pipeline — shows AF-UNet's seed 6 is not the fixed model:

```
133:  [ AFTrajectoryModel ] backbone=unet  unet_width(freq_dim)=32  params=4.0M   ← the freshly BUILT model
134:  [ train ] Seed 6 already reached 100000 steps — skipping                    ← the ON-DISK weights are pre-Fix_8
```

`--auto-resume` found a 100 k-step checkpoint already in the `_bbunet_` tree and skipped seed 6. Gen3v6's Fix_8 run cleared the tree first; Gen3v7 did not.

**The arithmetic closes exactly.** `Unet_study` §3.1 gives `dpcc-t-tightened` per seed: seed 6 = **0.17**, seeds 7–10 = **1.00** each.

| quantity | 4 valid seeds | stale seed 6 | 5-seed mean | `DA_20260811` reports |
|---|---|---|---|---|
| S&C | 1.00 | 0.17 | **0.834** | **0.833** ✅ |
| steps | 58.4 | ⇒ 75.3 | **61.8** | **61.77** ✅ |
| s/step | 0.030 | ⇒ 0.246 | **0.0732** | **0.0732** ✅ |

Three axes, three significant figures. The implied seed-6 cost of **0.246 s/step** is itself the signature: `Unet_study` §1.1 independently measures seed 6's `dpcc-r` arm at **0.242 s/step vs 0.028 on seeds 7–10** — the NLP iterating against an off-manifold field, not a slower network. The `.npz` evidence agrees: seed 6's MPC foresight contains single horizon steps of **0.968** against an arena diagonal of 0.92, at every K, on seed 6 and nowhere else.

> [!IMPORTANT]
> **`DA_20260811` §4.4's "AlphaFlow + U-Net is markedly worse" is one broken checkpoint averaged into four good ones.** A 4 M U-Net cannot cost 3.6× more per step than a 10 M SiT at generation time — that 0.0732 is the projector fighting the 253 M model's output.

### 2.3 What the four valid seeds say

`Unet_study` §4.3, **matched seeds 7–10, same K=2, same arms**:

| arm | AF **UNet@32** K2 | AF **SiT** K2 |
|---|---|---|
| mean S&C, 3 `-tightened` | **0.958** | 0.722 |
| `dpcc-t-tightened` | **1.00** / 58.4 / 0.030 | 0.92 / 67.9 / 0.024 |
| `dpcc-c-tightened` | **0.96** / 91 steps | 0.25 / 177 steps *(timeout mode)* |
| `dpcc-r`, `dpcc-t` *(untightened)* | 0.42 / 0.46 | **0.79 / 0.83** |
| s/step, `diffuser` arm | 0.030 | **0.019** |

⚠️ Four seeds, K=2 only, `n_trials=2` — per `DA_20260815` §1.4, **any n=2 row reading exactly 1.000 is unverified, not perfect.**

**But read the split, not the mean:** the U-Net owns the **tightened** arms; the SiT owns the **untightened** ones. §3 shows that split is the whole story.

---

## 3. Q4 — the raw field: AF-UNet does **not** beat MF-UNet. The tightened projector does.

This is the part the earlier version of this report got wrong by leading with the tightened arms. Sorted by how much projection each arm applies:

| arm (K=2) | projection | **AF UNet@32** (s7–10, 24 ep) | **MF UNet@32** (s6–10, 30 ep) | **AF SiT** (s6–10, 30 ep) |
|---|---|---|---|---|
| `diffuser` | **none** | 0.04 | 0.033 | **0.267** |
| `dpcc-r` | light | 0.42 | **0.467** | **0.79** |
| `dpcc-t` | light | 0.46 | **0.467** | **0.83** |
| `dpcc-c` | light | 0.75 | **0.800** | — |
| `dpcc-t-tightened` | **heavy** | **1.00** | 0.967 | 0.933 |
| `dpcc-r-tightened` | **heavy** | 0.92 | 0.967 | **1.000** |

Sources: `Unet_study` §3 (AF-UNet); `DA_20260811` §4.1.1 and §4.4.2 (MF-UNet, AF-SiT).

**The pattern is monotone in how much the projector is doing.** AF-UNet trails MF-UNet on all three lightly-projected arms and only moves ahead once the *tightened* constraint set is applied. Each individual gap is ~1 episode of 24–30 and is not significant alone — but the **sign is identical in every unprojected and lightly-projected cell**, and it reverses exactly where the projector's authority increases.

The training-side regression metric says the same, more loudly:

| signal (UNet@32, end of training) | **MF** (job 24317) | **AF** (job 24389, seeds 7–10) | ratio |
|---|---|---|---|
| `per_dim_rms_u` | **0.199** (min 0.179) | 0.336 / 0.402 / 0.435 / 0.358 → mean **0.383** | **≈1.9× worse** |
| `raw_mse_u` (val) | **1.90** (min 1.54) | 6.86 / 18.50 / 15.87 / 8.55 → mean **12.4** | **≈6.5× worse** |
| `raw_mse_u` (train, tail) | spikes > 20 on 14/100 epochs | ends at **638.8** (s7) / 70.9 (s10) | AF far noisier |

⚠️ Two provenance caveats, both closable for free (§10.1):
1. **Target-comparability.** `raw_mse_u` is the error against *each family's own* `u_target`, so it is not cross-family comparable in general — the same reason `Unet_study` §2 forbids comparing `test_loss` 0.983 to 0.912. It **becomes** comparable at α = 0, where AF's target is byte-identical to MF's (gate G2) — i.e. over the last 29 k steps. The figures above are end-of-training, so they fall inside that window and the comparison is legitimate *there*.
2. **Train vs val, single seed.** MF's 1.90 / 0.199 come from the Fix_8 **seed-6** signal table; AF's are four-seed `val/` summaries. The direction is robust — AF's train-side numbers are *worse* than its val-side, so matching the split can only widen the gap — but the exact ratio needs a matched re-read.

### 3.1 "Low quality, not smooth" is the predicted signature

§6.1 predicts the field degrades **at large `h`** — where the probe matters most and 1–2-NFE sampling lives — while staying **bounded**: a rough, under-pinned prior rather than a divergence. Four independent signals match:

- raw regression 1.9×–6.5× worse than MF-UNet (table above);
- the *only* arms where AF-UNet loses to MF-UNet are the raw and lightly-projected ones; the tightened arms — where the projector has authority — are where it wins;
- the projector works measurably harder on AF-UNet output: **0.030 s/step** vs AF-SiT's 0.019 on the same NLP;
- but it does **not** diverge — AF-UNet has no `dpcc-c` "crushed to a point" timeout mode, which is exactly the SiT's failure (0.96 vs 0.25, §2.3).

Roughness with bounded magnitude, rescued by projection, worst at large `h`. **Underfitting looks different** — it produces a smooth, over-averaged, low-variance field, which is not what AF-UNet does.

### 3.2 What this changes

- ❌ It does **not** revive "AF-UNet is broken" — §2.1/§2.2 stand: the 0.07 and the 0.833 remain the 253 M model and the stale seed 6.
- ✅ It **does** retire "AF-UNet ties MF-UNet". That is true only post-projection and must never be stated unqualified ([[benchmark-hierarchy-who-beats-whom]], [[pareto-definition-of-good]]).
- ✅ It **strengthens §6.1 considerably**, and AF-SiT is the positive control the mechanism predicts: same objective, conditioning written in the coordinates the bootstrap is invariant in, and the best raw field in the entire matrix (`diffuser` 0.267 — **8×** MF-UNet's 0.033).

---

## 4. Q2 + Q3 — is α-Flow actually enabled, and does the α → 0 snap hurt us?

The generation's own PLAN calls a dead α schedule *"the #1 silent failure of this generation"* — α pinned at 1.0 is plain flow matching wearing an α-Flow folder name. So it was checked directly. **The trigger for this check was Gen14's U10 finding; the check itself is entirely about Gen3v7.**

### 4.1 The schedule is live — five independent confirmations

| # | evidence | result |
|---|---|---|
| 1 | **Guard armed.** `AlphaFlowODE.__init__` raises if `af_alpha_end_step != n_train_steps`, and `af_n_train_steps=args.n_train_steps` **is** passed (`train_flow_matching_v3_alphaflow.py:473`). Both landed in the AF init commit `4342ed4f` (2026-07-23) — **before any AF run executed.** | ✅ |
| 2 | **`set_train_step` wired.** `utils/training.py:179-180` pushes `self.step` once per *optimizer* step (outside the `gradient_accumulate_every` loop, so the schedule is not halved); `load()` restores it on resume. | ✅ |
| 3 | **Pre-flight banner in every AF train log on disk** — including `2026-08-07/00_04_45_af_train_24389.log`, the AF-**UNet** run the DA rows come from:<br>`alpha schedule: sigmoid 1.0 -> 0.0 over [0, 100000] gamma=25.0 clamp=0.005`<br>`alpha: 1.000 1.000 1.000 0.993 0.924 0.500 0.076 0.007 0.000 0.000 0.000`<br>matching the independently recomputed curve to 3 decimals. | ✅ |
| 4 | **Per-step telemetry moves.** `alpha=` and `discrete_frac=` on every epoch line; `discrete_frac` ≈ 0.5 while α > 0, then 0.000 once α snaps. | ✅ |
| 5 | **Folder provenance.** Every AF checkpoint tree — `bbdit`, `bbunet`, `bbsit` — carries `ai1.0_ae0.0_ag25.0_rf0.5`. A constant-α misconfiguration would read `ae1.0`. | ✅ |

**And the port is faithful.** Against `aux_repo/alphaflow`: `get_ratio` (`src/training/loss.py:390-427`) normalises sigmoid progress by `(change_end − change_init)`, so **γ is scale-free** — copying γ=25 verbatim while rescaling `end_step` 400 k → 100 k is exactly right, not a mistake. `clamp_value=0.005`, `ratio_fm=0.5`, `clamp_utgt=4.0` match `alphaflow-sigmoid-latentspace-B-2` token-for-token, and the branch weighting `weight_c = 1`, `weight_d = α` (`loss.py:590-593`) matches `w_br` in `_p_losses_alphaflow`.

> **Q2 answer: yes — α-Flow was genuinely enabled on every run in the DA record, including the U-Net ones.**

### 4.2 …but only ~12 % of the training signal is α-Flow-specific

Weight mass on the u-loss over the 100 k budget, from the shipped config (`ratio_fm=0.5`, γ=25):

| branch | `u_tgt` | weight | share of u-loss mass |
|---|---|---|---|
| FM anchor, h = 0 | `v` | 1 | **55.9 %** |
| α = 1, h > 0 (short-circuit) | `v` | 1 | **16.1 %** |
| **bootstrap, 0 < α < 1** | **`α·v + (1−α)·u_next`** | **α** | **11.8 %** ⭐ |
| α = 0, h > 0 | `v + h·du/dr` (MeanFlow JVP) | 1 | 16.1 % |

**72 % of α-Flow's u-loss mass regresses `u → v`, i.e. is plain flow matching**; only **11.8 %** carries the bootstrapped target that is α-Flow's actual contribution. The squeeze is threefold: the bootstrap window is 42 % of the budget, only 50 % of each batch has h > 0, and those rows are additionally down-weighted by α. Because γ is scale-free this split is **identical upstream** — a property of the method at γ=25, not a porting error. It belongs in any Gen3v6-vs-Gen3v7 write-up: the two objectives share ~72 % of their u-loss, which tempers how large an AF-vs-MF difference should be expected at all.

### 4.3 🔴 The real defect: `state_best.pt` is selected by α, not by quality

Selection is `if test_loss < self.best_test_loss: self.save_best()` (`utils/training.py:235-237`). For α-Flow:

```
test_loss ≈ [ mean(w_br)·A(err_u) + A(err_v) ] / 2 ,   A(e) = e/(e+1e-3) ≈ 1  (err is a SUM over 48 dims)
mean(w_br) = ratio_fm·1 + (1−ratio_fm)·α = 0.5 + 0.5·α   for 0 < α < 1;   = 1.0 at α = 1 and at α = 0
⇒  test_loss ≈ 0.75 + 0.25·α ,  with a discontinuous jump back to ≈1.00 when α snaps to 0
```

**This is what the logs show, not a derivation awaiting confirmation.** Parsed from all four seeds of job 24389:

| step | α | `loss_test` | predicted `0.75 + 0.25α` |
|---|---|---|---|
| 999 | 1.000 | 1.000 | 1.00 |
| 40 999 | 0.905 | 0.977 | 0.98 |
| 50 999 | 0.438 | 0.872 | 0.86 |
| 60 999 | 0.060 | 0.758 | 0.77 |
| **68 999** | **0.009** | **0.739 ← global min** | 0.75 |
| 70 999 | 0.005 | 0.746 | 0.75 |
| **72 000** | **0.000** | **0.988 ← jumps 0.25** | 1.00 |
| 98 999 | 0.000 | 0.983 | 1.00 |

Bucketed over all 400 logged rows, mean `loss_test` rises monotonically with α: 0.773 (α≈0.1) → 0.886 (0.5) → 0.997 (1.0). **The metric measures α.**

Consequences, applying to *every* α-Flow number in *every* DA:

1. **`state_best.pt` is the ~step-69 k checkpoint at α ≈ 0.009**, on every seed. The eval loads it — `'diffusion_epoch': 'best'` has been in the AF plan block since `4342ed4f` and `eval_flow_matching_v3_alphaflow.py:309` passes it explicitly.
2. **The pure-MeanFlow tail is discarded** for a mechanical reason. α-Flow "ends as MeanFlow" in *training*; what is **deployed is mid-homotopy**.
3. **The artifact is 50× the signal.** The α term spans 0.25; genuine quality variation inside the α = 0 plateau spans ~0.005 (0.988 → 0.983).
4. **MeanFlow is immune** — verified, not assumed: job 24317 (`bbunet@32`) shows `loss_test` falling **monotonically** 1.000 → 0.912 across the full 100 k, so its `best` *is* the end of training. **Gen3v6 vs Gen3v7 is therefore not checkpoint-matched: MF at 100 k, AF at ~69 k.**
5. **It may be *flattering* α-Flow.** `INSIGHT_Gen3v7_first_train_curve` §1 recorded the field is best mid-anneal and *degrades* in the α = 0 tail (`h_mse_b3` spiking to 269). So this is **a confound to measure, not a bug to fix blind** — a naive switch to `latest` could make every AF number worse.
6. **It confounds this report's backbone question**: whichever backbone peaks near step 69 k is flattered, and neither AF-UNet nor AF-SiT was measured at any other checkpoint.

### 4.4 Q3 — the α → 0 snap: present here, but it never reached an eval

Gen14's `DA_20260831…U10` found that `af_alpha_clamp = 0.005` snaps α to exactly 0 partway through training, and that the snapped model's `raw_mse_u` degrades sharply. Does `avoiding-d3il` have it?

**Yes, bit-for-bit.** `config/avoiding-d3il.py:901-915` and `config/aligning-d3il-visual.py:1425-1429` carry the same five keys — `sigmoid`, `1.0 → 0.0`, γ = 25.0, `clamp = 0.005`, `end_step = n_train_steps = 100000` — so the α curves are identical. Measured, not inferred: **α = 0.0 from step 71 000 in every `avoiding` α-Flow run on disk** (23759, 23810, 23929, and all four seeds of 24389). That is 29 000 steps — **29 % of the budget** — training the pure MeanFlow target.

**But it never reached an eval.** Because `test_loss ≈ 0.75 + 0.25·α` (§4.3), `state_best.pt` always lands *before* the snap:

| run | best `self.step` | α there | α = 0 from | final `loss_test` |
|---|---|---|---|---|
| 24389 seed 7 | 68 000 | 0.0086 | 71 000 | 0.982 |
| 24389 seed 8 | 64 000 | 0.0230 | 71 000 | 0.984 |
| 24389 seed 9 | 68 000 | 0.0086 | 71 000 | 0.984 |
| 24389 seed 10 | 68 000 | 0.0086 | 71 000 | 0.985 |
| 23810 (SiT) | 65 000 | 0.018 | 71 000 | — |
| 23929 (DiT) | 68 000 | 0.0086 | 71 000 | — |
| 23759 (SiT) | 65 000 | 0.018 | 71 000 | — |

> **Q3 answer: every α-Flow number in every `avoiding-d3il` DA came from a pre-snap checkpoint at α ≈ 0.009–0.023. The snap burned 29 % of the training compute but never reached an evaluation. No `avoiding` result is invalidated by U10 — the cost was wasted compute, not wrong numbers.**

**Side finding, for the Gen14 owner (not this report's subject).** The same trace applied to Gen14's own comparator says its shipped α→0 run (job 24457) has `best` at `self.step` **71 000** — the *first* snapped step, `raw_mse_u` 5.87 — not the 100 k endpoint (8.504); and the α-const run 25241 has `best` at **80 000**, `raw_mse_u` 2.95, not the 2.626 endpoint. So that DA's "8.504 → 2.626, 3.2×" compares two final-step values, **neither of which was rolled out**; deployed-to-deployed is ≈2×. Its §1.5/§1.6 conclusions are unaffected. The step-stamp convention was validated against eval 25242 printing `Restored loss history from checkpoint at step 80000`. *(Flagged here only — not edited into that DA.)*

⚠️ **Parsing trap, for anyone re-deriving these numbers:** the tqdm postfix sorts keys alphabetically, so `a0_loss_test=` immediately precedes `loss_test=` and a loose regex reads the wrong series. Parse the postfix into a dict and read the exact key. Also, the postfix `step=N` is the progress-bar index; the trainer tests at `self.step % log_freq == 0` and stamps checkpoints with `self.step = N − 999`.

---

## 5. Code audit — what was checked and found clean

Audited at `938641c`. Every item was a candidate for an AF-specific U-Net defect; **none is one.**

| # | checked | result |
|---|---|---|
| 1 | `unet1d_temporal_cond.py` AF vs MF | **byte-identical** (`diff` empty, 493 lines). No backbone code differs between the generations |
| 2 | `dual_head` plumbing on the U-Net arm | ✅ `dual_head: True` → `AlphaFlowEngine` (`af_engine.py:86`) → `AFTrajectoryModel` → `Flow_matcher_U_Net_v2(dual_head=True)`. The v-head shares the trunk, same as the SiT's `final_layer_v`. The legacy orphan `aux_head` path is **not** taken |
| 3 | non-determinism between the target forward and the prediction forward | ✅ none. AF computes `u_next` and `u_pred` in **separate** forwards (MF reuses the JVP primal) — safe only if the backbone is deterministic, and it is: `returns_condition=False` is hardcoded for the U-Net so `mask_dist` is never sampled, and `Conv1dBlock` is `Conv1d→GroupNorm→Mish` with no `nn.Dropout` |
| 4 | `GroupNorm(n_groups=8)` at `freq_dim=32` | ✅ channels 32/64/128/256, all divisible by 8 |
| 5 | α = 0 branch vs Gen3v6 | ✅ `compute_u_target`'s continuous branch is Gen3v6's `_p_losses_meanflow` body with the same tangents `(v_inst, +1, −1)`; gate G2 asserts it |
| 6 | `af_alpha_end_step == n_train_steps` | ✅ both 100 000, hard-raise otherwise (`af_diffusion.py:161-166`) |
| 7 | `af_alpha_clamp` degenerate-dt guard | ✅ 0.005 present; without it every sample takes the discrete branch at dt ≈ 0 |
| 8 | `u_next` under `torch.no_grad()` | ✅ present (gate G5) — the target is not self-referential through the graph |
| 9 | adaptive-loss eps 1e-3 (AF) vs 1e-2 (MF) | ✅ near-inert either way: `err` is a per-sample **SUM** over H·D = 48, so `err ≫ eps` and the weight sits at ≈1. Not a backbone-specific lever |
| 10 | plan-block backbone token | ✅ `plan_fm_v3_alphaflow` mirrors `imf_backbone` into `diffusion_loadpath`; a mismatch fails loudly at `state_dict` load |

**One live hazard, unrelated to the backbone:** `--auto-resume` still silently skips a seed whose old checkpoint sits in the target tree, and the `params=4.0M` banner prints the *freshly built* model, not what is on disk. That produced §2.2 and is not guarded today.

### 5.1 What did *not* change from MF to AF

Since the question is "AF is based on MF — what broke?":

| # | MF → AF change | harder for a U-Net? |
|---|---|---|
| 1 | JVP target → bootstrap target | **architecturally easier** (no forward-AD through Conv1d/GroupNorm/Mish); **representationally riskier** — §6.1 ⭐ |
| 2 | +1 no-grad forward when 0 < α < 1 | no — backbone-agnostic |
| 3 | α homotopy 1 → 0 | **easier at the start** — 29 k steps of pure FM warm-up MF never gets |
| 4 | α = 0 tail | identical code to MF |
| 5 | `ratio_fm` 0.5 | identical to MF's `meanflow_data_proportion` |
| 6 | adaptive eps 0.01 → 0.001 | inert under SUM reduction (§5 item 9) |
| 7 | branch weight α on discrete samples | continuous fade, no discontinuity |
| 8 | target clamp ±4.0 | **stabilising** |
| 9 | default backbone `mf_dit` → `sit` | config default, not a code change |

**Item 1 is the only one with a plausible backbone interaction**, and it is narrow and specific — not "the U-Net can't learn this".

---

## 6. The mechanism — why the U-Net is the wrong host *for this objective* 🧪

Both subsections are hypotheses with code-verified premises. §10 gives the tests.

### 6.1 The bootstrap step is invariant in the SiT's coordinates and not in the U-Net's ⭐

α-Flow's target is an **interval-composition identity at a fixed endpoint** (`af_diffusion.py:529-635`):

```
z_shift = z_r + dt·v                       dt = α·h
u_next  = u(z_shift,  r+dt,  h−dt)         ← no_grad; note r+h = t is UNCHANGED
u_tgt   = (dt·v + (h−dt)·u_next) / h
```

| | conditioning vector | along the probe step `(r → r+dt, h → h−dt)` |
|---|---|---|
| **SiT** (`af_sit_trajectory.py`) | `c = E_t(t) + E_r(r)`, with `t_abs = r + h` computed **inside** the backbone; 256-D each, adaLN-zero 6·d per block × 8 blocks + 2·d final | `E_t(t)` is **bitwise identical** at both points. Only `E_r` moves, by `+dt`. The identity is a recursion in **one** coordinate, anchored by a term that does not move |
| **U-Net** (`unet1d_temporal_cond.py:136-147, 246-252`) | `c = time_mlp(r) + h_mlp(h)`, **32-D**, injected as a pure **additive shift** (`blocks[0](x) + time_mlp(t)` — no scale, no gate) | **both** terms move, in **opposite** directions, and only their **sum** reaches the network: `Δc ≈ dt·[E'_τ(r) − E'_h(h)]`. Nothing in the architecture pins "same `t`" |

α-Flow's own backbone is written in the coordinates its own identity is invariant in. The U-Net is written in `(r, h)`, so the composition step becomes a **difference of two independently-learned embedders**. If those learn similar derivative directions, `Δc → 0`, the network cannot distinguish the query from the probe, and `u_tgt → α·v + (1−α)·u_pred` — a shrinkage toward `v` that constrains the large-`h` field only through the h = 0 anchors (50 % of each batch). The field does not blow up; it stops being *pinned* at large `h`, which is exactly where 1–2-NFE sampling lives — and exactly what §3.1 measures.

**Why MeanFlow is immune:** MF's target is the **analytic** derivative in that same direction — `torch.func.jvp` with tangents `(v_inst, +1, −1)`, i.e. literally `∂/∂r − ∂/∂h`. It is the exact derivative of whatever conditioning the backbone has, however coarse, and it is *added to* an analytic anchor: `u_tgt = v + h·du/dr`. When `du/dr → 0` the MF target degrades gracefully to `v` — still a valid flow-matching target. **α-Flow's finite-difference version of the same derivative has no such floor.**

### 6.2 Fix_8 cut the U-Net's time-embedding resolution 8×, as a side effect

`freq_dim` is *both* the channel width and the time-embed width — the code says so and warns "never raise `freq_dim` to improve the embedding". So Fix_8's necessary 256 → 32 also took `SinusoidalPosEmb(dim)` from 256 to 32:

| | embedding width | frequencies | frequencies with argument > 0.1 rad anywhere on τ, h ∈ [0,1] |
|---|---|---|---|
| **UNet@32** `time_mlp` / `h_mlp` | **32** | 16 | **4** |
| **SiT** `TimestepEmbedder(frequency_embedding_size=256)` | **256** | 128 | **32** |

(Both use `max_period=10000`, tuned for integer diffusion timesteps 0…1000 rather than continuous times on [0,1] — so most channels are near-constant in both, but the U-Net has **8× fewer** that resolve anything.)

This compounds §6.1: the U-Net must express the probe as a small difference between two summed embeddings using a quarter as many active frequencies. MeanFlow again does not care — autodiff returns the exact derivative of a coarse embedding; α-Flow's finite difference needs the embedding to actually *move*.

**The two effects are confounded in the record**, because the single knob `freq_dim` changes capacity and time resolution together. That is a design defect worth fixing regardless of the outcome (§10.3).

### 6.3 Where in training this should show — the α timeline

Computed from `_get_ratio` with the shipped config:

| steps | α | branch | AF ≡ |
|---|---|---|---|
| 0 → **28 800** | 1.0 | short-circuit `u_tgt = v` (gate G1, **bitwise**) | **plain flow matching** |
| **28 800 → 71 000** | 1.0 → 0.0 | **bootstrap** (`u_next` forward) | α-Flow proper |
| **71 000** → 100 000 | 0.0 | JVP | **MeanFlow, byte-identical** |

🧪 **Prediction.** If §6.1/§6.2 hold, AF-UNet's `raw_mse_u` / `per_dim_rms_u` should **track AF-SiT's and MF-UNet's for the first ~29 k steps** (all three are doing plain FM, which the U-Net demonstrably does well), separate during the bootstrap window, and partially recover after 71 k. Separation should concentrate in the **`h_mse_b3` bucket (h ≥ 0.6)** — already logged by `_build_info`. **Checkable from existing W&B histories at zero compute** (§10.1).

---

## 7. Q6 — CAN α-Flow beat MeanFlow at all? The math, and whether AF-UNet is worth pursuing

**Question (2026-08-31):** *"Can α-Flow theoretically be better than MeanFlow on the U-Net? The U-Net works but AF doesn't — is there a real math reason? If it is theoretically better, maybe it is still worth further testing."*

### 7.1 Both objectives have the **same fixed point** — so α-Flow's ceiling *is* MeanFlow's ceiling

Both learn the **average velocity** of the same probability path:

```
U(z_r, r, h)  :=  (1/h) ∫_r^{r+h} v(z_s, s) ds        (h = t − r;  h → 0 ⇒ U → v)
```

They differ only in which identity they regress against:

| | identity | target | class |
|---|---|---|---|
| **MeanFlow** | differentiate `h·U = ∫ v` in `r` | `u_tgt = v + h·du/dr`, `du/dr = ∂u/∂z·v + ∂u/∂r − ∂u/∂h` | **differential** (local) |
| **α-Flow** | split the interval at `r+dt`, `dt = α·h` | `u_tgt = [dt·v + (h−dt)·u(z_r+dt·v, r+dt, h−dt)] / h` | **integral** (finite-difference) |

α-Flow's target makes two approximations to the *exact* composition identity: it replaces `U(z_r, r, dt)` by `v(z_r, r)` (error `O(dt)`, carried with weight `dt/h`) and the true flow point by one Euler step (error `O(dt²)`). Net **target bias `O(α²h)`** — so the true `U` is an exact fixed point of MeanFlow's objective and only an `O(α²h)`-approximate one of α-Flow's, for α > 0. At α = 0 the bias vanishes and the two targets are **byte-identical** (gate G2, `af_diffusion.py:552`).

> **Answer, part 1: no — α-Flow cannot have a *better optimum* than MeanFlow. It has the same one.** Its asymptotic objective *is* MeanFlow's. Any α-Flow advantage is necessarily an **optimisation / curriculum** advantage: a better *path* to the same solution, never a better solution. Any claim of the form "α-Flow's field is fundamentally better than MeanFlow's" is unsupportable on this math.

The bias itself is not the problem: at α ≤ 0.1 it is ≤ 1 % of `h`. **α-Flow's difficulty is signal, not bias** (§7.3).

### 7.2 Where α-Flow *can* legitimately win — three real reasons, one of them ours

1. **Finite-interval semigroup constraint ↔ few-NFE sampling.** MeanFlow's identity is infinitesimal: it implies the composition property over a finite step only if it is satisfied *exactly, everywhere*. α-Flow constrains `u` at `h` directly against `u` at `h−dt` — a **finite-range** constraint, which is what 1–2-step sampling actually consumes. This is the same argument that motivates consistency/shortcut-model bootstrapping over purely local self-consistency. **It is also this project's whole point: we deploy at K = 1–2.** ⇒ if a genuine α-Flow > MeanFlow win exists, **it lives at K = 1, not K = 20.**
2. **No forward-mode AD.** MeanFlow needs `torch.func.jvp` through the backbone every step; α-Flow needs one extra plain forward under `no_grad`. Cheaper, and it removes a stiff second-order path through `Conv1d → GroupNorm → Mish`.
3. **Curriculum.** α = 1 is *bitwise* plain flow matching (gate G1), so α-Flow gets ~29 k steps of externally-anchored FM before any self-consistency is asked for. MeanFlow never gets that warm-up and must estimate `du/dr` from an untrained network from step 1.

### 7.3 …and where it pays — the real math reason it fails on *this* backbone ⭐

Expand α-Flow's probe. With conditioning code `c(r,h)` and network `f(z, c)`:

```
u_next − u_pred  ≈  ∂f/∂z·(dt·v)  +  ∂f/∂c·Δc ,     Δc = c(r+dt, h−dt) − c(r, h)
⇒ (u_next − u_pred)/dt  →  ∂f/∂z·v + ∂f/∂c·(∂c/∂r − ∂c/∂h)  =  du/dr
```

**The two objectives ask for exactly the same quantity.** α-Flow estimates `du/dr` by a finite difference; MeanFlow computes it by autodiff. That is the entire difference — and it is the whole story, because the two estimators fail *differently* when the conditioning cannot resolve the probe:

| | target when the `(r,h)` code stops resolving the probe (`Δc → 0`) | consequence |
|---|---|---|
| **MeanFlow** | `u_tgt = v + h·(collapsing term)` → **`v`** | degrades **gracefully** to the plain-FM target, at **weight 1**. The `h > 0` rows keep a full-strength anchor. Worst case: `u(·,h) ≈ v` — a crude but coherent field |
| **α-Flow** | `u_tgt = α·v + (1−α)·u_pred` ⇒ restoring signal `α·(v − u)`, and those rows are *additionally* weighted `w_br = α` | same fixed point (`u = v`), but the effective gradient is **`O(α²)`**. As α anneals down, the `h > 0` rows **stop training**. The large-`h` field is not driven anywhere — it **freezes mid-anneal**, constrained only by the `h = 0` anchors, which say nothing about `h`-dependence |

**That is the real math reason, and it is not "the U-Net can't learn this".** It is: *α-Flow's objective is singular in `Δc → 0` (its signal vanishes as `α²`), MeanFlow's is not (its signal floors at `v`).* An architecture whose `(r,h)` code has low resolution therefore costs α-Flow far more than it costs MeanFlow — which is exactly the observed 2×2, and it predicts the observed symptom: a **stale, under-determined, rough** large-`h` field rather than a divergent one (§3.1).

**And `dt = α·h` shrinks monotonically through training.** So there is a **crossover step** at which the probe drops below a given backbone's `(r,h)` resolution floor. 🧪 That step is *earlier* for a 32-D additive shift (§6.2) than for 256-D adaLN across 8 blocks — which is a sharp, falsifiable prediction and the thing §10.1 should look for.

**`af_alpha_clamp` is precisely the knob that encodes this floor — and it is a fixed constant where the correct value is backbone-dependent.** The config comment already names the failure mode exactly:

> *"snap-to-exact-0/1 guard. Without it α becomes a tiny-but-nonzero number and every sample takes the discrete branch with dt≈0 ⇒ a degenerate near-identity target."*

Upstream set `0.005` for a SiT in latent space. **For a U-Net conditioned by a 32-D additive shift the degenerate regime begins far earlier than α = 0.005** — and everything between the true floor and 0.005 is training on a near-identity target at weight α. That window is not a bug in anyone's code; it is one global constant standing in for an architecture-dependent quantity.

### 7.4 One more prediction that follows, and contradicts the current setup

At α = 0 the JVP branch resumes at **full weight 1** — so the last 29 k steps are exactly the phase that should **repair** a frozen `h > 0` field. **And `'diffusion_epoch': 'best'` throws that phase away** (§4.3, §4.4): every AF number on record is the ~step-69 k checkpoint, i.e. the model *at its most frozen*, immediately before the repair phase.

🧪 **Prediction: re-evaluating at `'latest'` should help AF-UNet *more* than AF-SiT** — the U-Net has more to repair. ⚠️ The evidence is genuinely mixed and this could go the other way: `INSIGHT_Gen3v7_first_train_curve` §1 recorded the field *degrading* in the α = 0 tail on the DiT (`h_mse_b3` → 269), and AF-UNet's train `raw_mse_u` still ends at 638.8. Which is why §10.2 is a **measurement, not a fix**.

### 7.5 🟢 The cheapest AF-UNet experiment falls straight out of §7.3 — one config key

```python
# config/avoiding-d3il.py — AF training block, U-Net arm only
'af_alpha_clamp': 0.05,   # was 0.005 — snap α to 0 once the probe drops below the
                          # U-Net's (r,h) resolution floor, instead of training ~15 k
                          # steps on a near-identity target at weight α.
```

No code change, no new architecture, no new gate. It shortens the degenerate window and hands the `h > 0` field to the full-weight JVP branch earlier. **Its floor is MF-UNet-quality**, because after the snap AF *is* MeanFlow — plus a 29 k-step pure-FM warm-up MeanFlow never gets (§7.2 item 3). Screen at 1 seed, sweep `{0.005 (today), 0.05, 0.15}`; pair with `'diffusion_epoch': 'latest'` so §4.3 does not mask the result.

### 7.6 Target-matched evidence that it is the *path*, not the objective

In the last 29 k steps α = 0, so AF and MF optimise a **byte-identical target** on the **same backbone**, data, horizon and normaliser. At that point AF-UNet's train `raw_mse_u` ends at **638.8** while MF-UNet spikes above 20 on 14 of 100 epochs. Same target, same architecture, very different error ⇒ **the anneal left AF-UNet's weights somewhere materially worse**, which is an optimisation-path statement, not an objective statement. This is the cleanest target-matched comparison available in the record, and it is what §7.3 predicts.

### 7.7 So — is AF-UNet worth further testing? **Yes, but with a bounded expectation.**

| | |
|---|---|
| **Realistic upside** | AF-UNet **catches up to** MF-UNet and trains cheaper (no JVP) — *not* "AF-UNet beats MF-UNet by a lot". §7.1 caps it: same fixed point |
| **Where a genuine AF > MF win could exist** | **K = 1** few-NFE, where the finite-interval constraint is the thing being consumed (§7.2 item 1). Not at K = 20 |
| **Why it matters scientifically** | the paper needs an **architecture-matched** α-Flow row. Today the only working AF arm is a **10.0 M** SiT against a **4.0 M** baseline — a confounded claim by [[architecture-matched-beat-is-the-strong-claim]]. A working 4.0 M AF-UNet converts it into a matched one |
| **What would make it not worth it** | if §10.1 shows AF-UNet already behind MF-UNet **before step 28 800** — where AF is bitwise plain flow matching — then none of §7.3 applies, the problem is upstream of α-Flow, and this whole line closes |
| **Cost ladder** | §10.1 (free, W&B) → §7.5 (one config key, 1-seed screen) → §10.2 (re-eval, no training) → §10.5 (the `E_t + E_r` fix, same param count) |

---

## 8. Why "the SiT is just bigger" is not the answer

AF-SiT is **10.0 M**, AF-UNet@32 is **4.0 M** — 2.5×, so any SiT > UNet reading is confounded ([[architecture-matched-beat-is-the-strong-claim]]). Three things about that:

1. **It cannot explain the interaction.** The same 10.0 M vs 4.0 M gap exists between `mf_dit` and MF-UNet, and there the **U-Net wins** (0.967 / 59.4 vs 0.967 / 68.4 steps, and fewer steps in 13/13 arms). Capacity alone predicts the wrong sign for MeanFlow.
2. **It cannot explain expressivity.** AF ⊇ MF at α = 0, so the working 4.0 M MF-UNet solution is a feasible point of the AF-UNet problem (§0).
3. **But it is a genuine, unfixed confound in the config.** The avoiding AF block still pairs `'imf_backbone': 'sit'` with `'dit_hidden_size': 256` against `'freq_dim': 32`. The Gen14 tree, which runs the same engine, pins its transformer bone to `dit_hidden_size=160` (≈3.9 M vs 4.0 M, 0.97×) and its config states plainly that an unmatched bone A/B is *"exactly the Fix_8 defect that already forced one public retraction"*. **The avoiding config should adopt the same 160 before any bone claim is published** (§10.5).

---

## 9. How to test AF-UNet — the mental model, the endpoint, and what is worth paying for

**Question (2026-08-31):** *"How to test? What is the correct mental model? What to tune for AF-UNet? I need to find when K1/K2 on the `diffuser` arm returns the same or better quality than MeanFlow. This is training-phase tuning and probably expensive — pick the most hopeful things."*

### 9.1 The mental model

> **AF-UNet is not a different model from MF-UNet. It is the same model reached by a different path**, and the path has a middle phase whose usefulness depends on exactly one number:
> ```
> dt = α · h          ← the size of the probe the network must resolve
> ```
> Above the backbone's time-resolution floor, that phase teaches the **finite-interval** property that few-NFE sampling consumes (§7.2). Below it, the phase teaches nothing *and costs*: the `h > 0` rows carry `O(α²)` signal and freeze (§7.3).
>
> **So you are not tuning "how much α-Flow". You are tuning the probe size relative to what the conditioning can resolve** — and, equivalently, *when to hand the `h > 0` field over to the full-weight JVP branch.*

Two numbers make this concrete. `dt = α·h`, and with `t_schedule='logit_normal'` (p_mean −0.4, p_std 1.0, two independent draws, min/max) `E[h] ≈ 0.25–0.3`. So at α = 0.05, `dt ≈ 0.013`; at the clamp α = 0.005, `dt ≈ 0.0013`. Against a time code with **~4 resolving frequencies on [0,1]** (§6.2) that probe is essentially invisible; against the SiT's **~32** it is ~8× more visible. **The U-Net needs a probe roughly 8× larger than the SiT to see the same thing** — which is where `clamp 0.005 → ≈0.04–0.05` comes from. It is an order-of-magnitude argument, not a theorem, but it is the argument that sets the knob.

### 9.2 The endpoint — and why `diffuser` S&C alone will not resolve it

The `diffuser` arm is the right arm: unprojected, so it measures the field and nothing else, and U-Net-vs-U-Net is the only *natively capacity-matched* comparison in the project. But its S&C is a near-floor binary (0.033–0.267 = 1–8 episodes of 30), so on its own it cannot separate AF-UNet 0.067 from MF-UNet 0.033. Design around that:

| tier | endpoint | why | power |
|---|---|---|---|
| **primary** | zero-violation fraction on `diffuser`, **paired** by (seed × context × trial), **exact McNemar** | this is the statistic that actually separated the families before (`DA_20260811` §4.4.2: the difference is in the *tail*, not the mean violation count — AF-SiT 0.267 vs MF-UNet 0.033 while their mean violations are 15.27 vs 15.50) | usable at `n_trials=20`; hopeless unpaired at n=2 |
| **primary** | **the K2 → K1 degradation slope**, within model | ⭐ the theory predicts α-Flow's advantage is the *finite-interval* property, so the discriminating quantity is **how much you lose going 2 → 1 NFE**, not the level at either. A within-model contrast, far more sensitive than a cross-model level | high |
| secondary | roughness ratio: `mean‖Δ realised per control step‖` from any `diffuser.npz` — healthy ≈ 1.0, the pre-Fix_8 seed 6 scored 6.6–8.4 (`Unet_study` §7) | this is the direct measurement of *"not smooth"*, and it is a two-line check on files you already have | high, free |
| secondary | `h_mse_b3` (h ≥ 0.6) on the training side | ⭐ **at K = 1 the sampler evaluates `u(z₀, 0, 1)` — h = 1.** So `h_mse_b3` *is* the training-time proxy for K = 1 quality, and it is already logged | high, free |
| **screen only** | `raw_mse_u` / `per_dim_rms_u` at steps ≥ 71 000 | the only target-legitimate cross-family regression comparison (α = 0 ⇒ identical targets). Use it to **reject** bad configs cheaply | ⚠️ **never as the verdict** — `Gen14 U10 §1.6` established `raw_mse_u` is not a proxy for rollout quality |

**Matched-checkpoint rule.** MF's `best` is the end of training, AF's is ~step 69 k (§4.3). Any AF-vs-MF comparison must be run at **`'diffusion_epoch': 'latest'` on both**, or as a 2×2 with `best`. Otherwise the result is the selection artifact, not the field.

### 9.3 The cost-aware ladder — do the free things first, then buy two runs

AF-UNet trains at **≈4 h 14 m/seed** on one A5000 (`Unet_study` §2). So a 1-seed screen ≈ 4.2 h and the whole recommendation below is **≈8.5 GPU-hours**.

**Free, and one of them can kill the entire line (do first):**

- **F1 — the W&B overlay (§10.1).** Steps 0 → 28 800 hold α = 1.0, where gate G1 makes α-Flow *bitwise* plain flow matching. Overlay `h_mse_b3` and `per_dim_rms_u` for AF-UNet (24389, s7–10) vs MF-UNet (24317) vs AF-SiT. **If AF-UNet is already behind at 28 800, stop — the problem is upstream of α-Flow and none of §7 applies.** If they coincide there and separate after, the bootstrap window is confirmed as the cause and the tuning below is justified.
- **F2 — the `latest` re-eval (§10.2).** No training; the checkpoints are on disk. Tests §7.4's prediction that the α = 0 tail *repairs* the U-Net's `h > 0` field, and it removes the matched-checkpoint confound from every later comparison.
- **F3 — the roughness ratio** on the existing `diffuser.npz` files. Two lines, and it quantifies the "not smooth" complaint on data already collected.

**Then buy exactly two training runs, 1 seed each:**

| # | change | why it is the most hopeful | cost | floor |
|---|---|---|---|---|
| **T1** | `af_alpha_clamp: 0.005 → 0.05` — **one config key, no code** | derived directly from §9.1's 8× resolution ratio. Ends the degenerate window early and hands `h > 0` to the full-weight JVP branch sooner | ~4.2 h | **MF-UNet quality** — after the snap α-Flow *is* MeanFlow, plus a pure-FM warm-up MeanFlow never gets |
| **T2** | `E_t(t) + E_r(r)` conditioning on the U-Net (§10.5) — ~20 lines, **identical parameter count** | attacks the root cause instead of routing around it, and it is the only change here that would be a **publishable α-Flow × parameterisation result** rather than a tuning note | ~4.2 h + a small port | unchanged behaviour if `t = r + h` is embedded and the injection is kept additive |

**A legitimate 40 % cost saving for the screen:** the crossover is predicted mid-training and `h_mse_b3` separates well before 100 k. Screening at **60 k steps** is fine *provided* `af_alpha_end_step` is set to 60 000 too (the constructor asserts they match) — but be explicit that this **compresses the α curve**, so it is a different schedule and every arm in the screen must use the same one. Never compare a 60 k-schedule arm against a 100 k-schedule arm.

### 9.4 What NOT to spend training on yet

- **`af_ratio_fm: 0.5 → 0.25`** — it raises the α-Flow-specific loss mass from 11.8 % to ~17.7 % (§4.2), which sounds right, but under §7.3 it *adds more rows whose probe is invisible while removing FM anchors*. **Theory says it makes things worse until the probe is fixed.** Revisit only after T1 or T2 succeeds.
- **`p_std` ↑ (wider `(r,t)` gap ⇒ larger `E[h]` ⇒ larger `dt`)** — sound in principle and it attacks the same `dt` product, but `t_schedule` is shared with Gen3v6 and baked into `diffusion_loadpath`, so changing it **breaks the controlled MF-vs-AF A/B** unless mirrored into Gen3v6 and retrained on both sides. Twice the cost, second-tier.
- **α-const (`af_alpha_end: 0.05`, never snapping)** — keeps the finite-interval constraint alive but **forfeits the full-weight JVP repair phase** that §7.4 says the U-Net needs most. T1 is the same idea pointed the other way and is cheaper to reason about.
- **More seeds before a config wins.** Screen at 1 seed on the free/continuous endpoints; only spend seeds 6–10 × `n_trials=20` on a config that has already moved `h_mse_b3`.

---

## 10. What to run

Ordered by information per GPU-hour.

**10.1 🟢 FREE — read the existing W&B curves. Do this first.**
Overlay `raw_mse_u`, `per_dim_rms_u`, `h_mse_b0..b3`, `alpha`, `discrete_frac`, `clamp_frac` for **AF-UNet (24389, seeds 7–10)**, **AF-SiT** and **MF-UNet (24317)** on the step axis, with the **28 800** and **71 000** boundaries marked. Two decisive readings:
- curves **together** before 28.8 k (where gate G1 makes AF bitwise plain flow matching) and **diverging after** ⇒ the bootstrap window is the cause, §0 confirmed near-decisively;
- AF-UNet **already** behind at 28.8 k ⇒ the problem is upstream of α-Flow entirely (data, normalisation, LR, EMA) and §6 is wrong.
Same pull also closes §3's caveats: compare `per_dim_rms_u` on the **same split** at **steps ≥ 71 000**, where the two families regress against a byte-identical target and the comparison is target-legitimate.

**10.2 🔴 Re-evaluate the existing AF checkpoints at `'diffusion_epoch': 'latest'`** (§4.3). No retraining — the numbered checkpoints are on disk. This is the α ≈ 0.009 vs α = 0 A/B; it decides whether every AF number in the study is flattered or penalised, and it must be settled before Gen3v6-vs-Gen3v7 is written up as matched. **Do not change the selection metric before measuring both.**

**10.3 🟢 Raise `af_alpha_clamp` on the U-Net arm — one config key, no code (§7.5).** `0.005 → 0.05`, screened at 1 seed, sweeping `{0.005, 0.05, 0.15}`, paired with `'diffusion_epoch': 'latest'`. It shortens the window in which α-Flow trains on a near-identity target at weight α and hands the `h > 0` field to the full-weight JVP branch earlier. **Its floor is MF-UNet quality**, since after the snap α-Flow *is* MeanFlow — plus a 29 k-step pure-FM warm-up MeanFlow never gets. Highest value per GPU-hour of anything that requires training.

**10.4 🟡 Decouple the U-Net's time-embed width from its channel width.** Add `time_embed_freq_dim` (default 32 ⇒ byte-identical to today) used only by the two `SinusoidalPosEmb`s, and run AF-UNet at 256 with `dim` held at 32. Cost ≈ **+0.06 M** params (4.0 M → ~4.06 M) — capacity stays matched to the DPCC baseline. Direct test of §6.2; mirror into Gen3v6 per the sibling convention.

**10.5 🟡 The decisive test of §6.1.** Give the U-Net an `E_t(t) + E_r(r)` conditioning option — compute `t = r + h` inside the backbone, embed both endpoints, keep the additive injection. **Identical parameter count.** If AF-UNet improves and MF-UNet is unchanged, §6.1 is confirmed and it is a genuine α-Flow × parameterisation result, not a U-Net result.

**10.6 🔴 The missing measurement: AF-UNet@32 at `n_trials=20`, seeds 6–10, K ∈ {1,2}.** Delete `logs/avoiding-d3il/flow_matching_v3_alphaflow/H8_D…_bbunet_…/6/` first and retrain seed 6 — otherwise `--auto-resume` reproduces §2.2 exactly. Run the SiT arm at `dit_hidden_size=160` alongside so the bone A/B is finally capacity-matched (§8.3). Until this exists, every AF-UNet claim in this repo — including §2.3 — rests on 4 seeds × 2 trials.

**10.7 ⚪ Guard the auto-resume hazard** that caused §2.2: print the *loaded* checkpoint's parameter count and config fingerprint, not the freshly built model's, and refuse to skip a seed whose on-disk fingerprint differs from the current one.

---

## 11. Confidence ledger

| claim | confidence | basis |
|---|---|---|
| Failure #1 is the `freq_dim` width bug, not the objective | **high** | MF and AF die identically at 256, both recover at 32 |
| Failure #2 (`0.833`) is the stale seed-6 checkpoint | **high** | three-axis arithmetic reconstruction (§2.2) + independent `.npz` foresight evidence + the train-log line |
| No bug in today's AF U-Net code path | **high** | 10-item audit, §5 |
| AF-UNet's **raw** field is behind MF-UNet's; its 1.00 is projector-supplied | **medium-high** | same sign in all four unprojected / lightly-projected arms, plus a ~1.9× `per_dim_rms_u` gap; each individual arm gap is ~1 episode and the seed sets are not matched (24 vs 30 ep) |
| AF-SiT has the best raw field in the matrix | **high** | `DA_20260811` §4.4.2 — `diffuser` 0.267 vs 0.067 / 0.067 / 0.033, 5 seeds; untightened arms agree (0.79 / 0.83) |
| AF-UNet@32 ≥ AF-SiT on **tightened** arms | **low** | 4 seeds × 2 trials, K=2 only, no seed 6 |
| α schedule genuinely enabled on all runs incl. U-Net | **high** | assert + `set_train_step` since the AF init commit; pre-flight banner and per-step telemetry in the logs themselves; folder tokens; upstream port verified token-for-token |
| Only ~11.8 % of the u-loss mass is α-Flow-specific | **high** | computed from the shipped config; upstream weighting confirmed at `loss.py:590-593`; γ scale-free so identical upstream |
| `state_best.pt` ≈ step 69 k, α ≈ 0.009, on every AF run | **high** | analytic `0.75 + 0.25α` matches the parsed `loss_test` curve of all four seeds to ~0.01; global min at bar-step 68 999; MF control (24317) is monotone as predicted |
| Whether that selection helps or hurts AF | **unknown** | needs the `latest` re-eval (§10.2) |
| The α → 0 snap is present in `avoiding` too | **high** | identical schedule keys; α = 0.0 measured at step 71 000 in 23759, 23810, 23929 and all four seeds of 24389 |
| …but no `avoiding` eval ever loaded a snapped checkpoint | **high** | per-run running-argmin of `loss_test` = step 64 k–68 k; `'diffusion_epoch': 'best'` in the plan block and passed explicitly at eval |
| AF-UNet's failure is not an expressivity or capacity limit | **high** | gate G2: at α = 0 the AF target is byte-identical to MF's, so the MF-UNet solution is feasible for AF |
| ~85–90 % of the effect is in setup-controlled causes | **medium** | §0.2 is a judgement over five code-verified mechanisms, not a measurement; §10.1 moves it either way |
| α-Flow and MeanFlow share the same fixed point; AF has no better optimum | **high** | AF's target is an `O(α²h)`-biased finite difference of the identity MF differentiates exactly, and is byte-identical to MF's at α = 0 (gate G2) — §7.1 |
| α-Flow's signal vanishes as `O(α²)` when the `(r,h)` code cannot resolve the probe, MeanFlow's floors at `v` | **medium-high** | algebra of the two targets (§7.3); the premise (`Δc` small on a 32-D additive shift) is code-verified, the effect size is not measured |
| `af_alpha_clamp` is a global constant standing in for a backbone-dependent floor | **medium** | the config comment names the exact failure mode it guards; that the correct threshold differs by backbone follows from §6.2 but is untested (§7.5) |
| §6.1 conditioning-parameterisation mechanism | **hypothesis** | code-verified premises, untested consequence |
| §6.2 time-embedding resolution | **hypothesis** | numbers computed from the two implementations; effect size unknown |
| AF-SiT is the *safe* AF arm for the paper today | **high** | the only AF cell with a 300-episode verified row (`DA_20260817`) |

---

## 12. One-line verdict

**Nothing is broken, and the U-Net is not too small — α-Flow contains MeanFlow at α = 0, so the working MF-UNet weights are a feasible point of the AF-UNet problem; what fails is the training signal, because α-Flow's target is a finite difference of the derivative MeanFlow computes analytically, and the U-Net's `E_τ(r) + E_h(h)` conditioning is the one parameterisation in which that difference cancels — which is why AF-UNet's raw field is the weakest in the matrix and its only perfect arm is the one the projector rescues, why AF-SiT's raw field is the strongest, and why ~85–90 % of the gap sits in things you control and can test for free from logs already on disk.**

---

## 13. Claims for the paper, as they stand today

- ✅ **AF-SiT is the α-Flow arm** — the only AF cell with a verified 300-episode row (`DA_20260817`) and the best raw field measured.
- ✅ **MF-UNet is the architecture-matched arm** ([[architecture-matched-beat-is-the-strong-claim]]) — 4.0 M against the baseline's 4.0 M, `n=20` verified.
- ❌ **Do not write "α-Flow needs a transformer"** — unsupported, and false if §6.1 holds.
- ❌ **Do not write "AF-UNet ties MF-UNet"** — true only post-projection (§3).
- ❌ **Do not claim α-Flow's field is fundamentally better than MeanFlow's** — they share a fixed point (§7.1). α-Flow's case is a *few-NFE* and *training-cost* case, and it must be argued at **K = 1**.
- ⏳ **AF-UNet is unresolved**, for reasons that are ~85–90 % ours and mostly cheap to remove. Its realistic upside is parity with MF-UNet at lower training cost — which is still worth having, because it is the **architecture-matched** α-Flow row the paper currently lacks.
