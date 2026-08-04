# DA — Gen14 first mf / af visual-aligning train + eval (seed 6)

**Date:** 2026-08-04
**Source:** `temp/0408/`
**Arms:** `mix_visual_aligning_mf` (MeanFlow, Gen3v6 lineage) and `mix_visual_aligning_af` (α-Flow, Gen3v7 lineage), both at `film_mode=v1`, U-Net backbone, `if_vision=True`, seed 6, 100 k steps.
**Question:** the first end-to-end mf/af visual run finished. Did it work, and if not, what is the mechanism?

> **Headline.** Both arms are task failures (1/30 success). But the training logs contain a clean, unambiguous causal finding that is worth more than the eval numbers: **α-Flow's test error jumps 2.9× the moment `af_alpha_clamp` snaps α to exactly 0** — i.e. the moment the objective switches from the bootstrapped target to the JVP MeanFlow target — and lands exactly on MeanFlow's own error plateau. Two independent runs, one estimator, one error floor. On top of that sit three configuration defects (100 % gradient clipping, K=100 on a few-step model, `epoch='latest'` checkpoint selection) that each independently suffice to explain the eval result.

---

## 1. Provenance

| item | mf | af |
|---|---|---|
| Train exp | `mix_visual_aligning_mf/H8_D…VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal/6` | `mix_visual_aligning_af/H8_D…VisualAlphaFlow_…_filmv1_Eaf_tslogit_normal_afschsigmoid/6` |
| Plan exp | `H8_K100_Meuler_T0.5_…_VTrue_mpc4_filmv1_Emf/6` | `…_mpc4_filmv1_Eaf/6` |
| Config snapshot | `2026-08-02 08:20:13` | `2026-08-02 20:59:04` |
| Loss history | `mf_losses.pkl` (100 log points, 0 → 99 000) | `af)losses.pkl` (same) |
| Geo variant | `combined_5` (dynamics + geo_bounds + halfspace + obstacles + action bounds) | same |
| Projection variants run | `diffuser` (30/30), `dpcc-r` (**11/30, truncated**) | `diffuser` (30/30), `dpcc-r` (**11/30, truncated**) |

Training hyperparameters, from the snapshot: `lr=2e-4` (1 k warmup, cosine → 5.02e-8), `batch_size=64`, `ema_decay=0.995`, `n_train_steps=1e5`, `gradient_clip=1.0`, `dual_head=True`, `interval_cfg=False`, `t_schedule=logit_normal`, `p_mean=-0.4`, `p_std=1.0`. af adds `af_alpha_scheduler=sigmoid`, `α: 1→0` over `[0, 1e5]`, `af_alpha_gamma=25.0`, **`af_alpha_clamp=0.005`**.

**Housekeeping.** `temp/0408/` contains the af tree **four times** at different nesting depths (`./6/…`, `./H8_K100_…/6/…`, `./H8_D…/H8_K100_…/6/…`, `./mix_visual_aligning_af/…`) — byte-identical (md5 `1b70a03136…`, 70 200 B on every copy of `eval_diffuser_train_set.log`). A drag-and-drop / partial-path download artifact, not four runs. The mf tree appears once. `01_01_20_eval_fmv3_ode_job_24198.log` is unrelated — it is a **state-only avoiding-d3il** FMv3-ODE eval (job 24198, `flow_matcher_v3_ode_selectable`), not part of this generation; ignored here.

---

## 2. Headline eval numbers

`diffuser` = generation only, no projection. Full 30 contexts, both arms.

| metric | **mf** | **af** |
|---|---:|---:|
| Success rate | **0.0333** (1/30) | **0.0333** (1/30) |
| Final mean distance | 0.3113 ± 0.2231 m | 0.4165 ± 0.2592 m |
| Best single rollout | 0.0276 m | 0.0108 m |
| Steps (all trials) | 394.5 ± 29.4 | 390.7 ± 49.9 |
| Physical tracking error (mean / max) | 0.1028 / **2.2999** m | 0.1775 / **2.7194** m |
| Inference time / replan | 0.893 s | 0.902 s |
| Exec constraint sat rate | 0.804 ± 0.238 | 0.718 ± 0.298 |
| Violated steps / rollout | 78.4 (bounds 58.2, hs 16.7, obs 5.4) | 111.8 (bounds 86.4, hs 16.4, obs 14.6) |
| Plan post-projection viol rate | 0.104 | 0.224 |
| Zero-violation rollouts | 6/30 | 6/30 |

The box does move — mean box-to-target distance goes 0.4547 → 0.2200 m (mf) and 0.4547 → 0.3598 m (af) — so the policy is not inert. It just does not finish: 3/30 (mf) and 2/30 (af) rollouts never moved the box at all, and 4/30 (mf) / 7/30 (af) pushed it **further away** than it started.

`dpcc-r` was **killed by the wall clock at rollout 11/30 in both arms** (see §6), so those cells are 11-context partials and are only ever compared against the first 11 `diffuser` contexts below.

---

## 3. The finding: the α cliff

![training curves](figs/fig1_training.png)

`train/loss` and `test/loss` are pinned near 1.0 by adaptive weighting (mf 0.926, af 0.989 at the end) and carry no signal — as expected. The real quantity is `raw_mse_u`.

α-Flow's `compute_u_target` has **two structurally different branches** (`af_diffusion.py:552` / `:581`):

- **α > 0** — bootstrapped / discrete: step toward data by `dt = α·h` at the analytic velocity, query the model there, form the target from the two-point difference.
- **α ≤ 0** — the continuous branch: Gen3v6's forward-mode JVP, `u_tgt = v + h · d/dr[u]` via `_jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))`.

`af_alpha_clamp=0.005` snaps α to exactly 0 once the sigmoid drops below 0.005. In this run that happens between logged steps 71 000 and 72 000:

| step | α | `discrete_frac` | **test raw MSE (u)** |
|---:|---:|---:|---:|
| 69 000 | 0.008577 | 0.406 | 2.779 |
| 70 000 | 0.006693 | 0.547 | **2.657**  ← af's best |
| 71 000 | 0.005220 | 0.484 | 2.911 |
| **72 000** | **0.000000** | **0.000** | **8.504** |
| 73 000 | 0.000000 | 0.000 | 6.256 |
| 74 000 | 0.000000 | 0.000 | 7.914 |
| 75 000 | 0.000000 | 0.000 | 7.117 |

**2.911 → 8.504 in one logging interval — a 2.9× step change**, against a curve that had been descending smoothly and monotonically for 70 k steps (12.58 → 2.66). It never recovers: steps 72 k–99 k sit in the band 6.3–8.6.

That band is **MeanFlow's band**. mf, which runs the α=0 JVP branch from step 0, plateaus at test MSE 7–10 from step ~10 k onward and finishes at 7.29 (best 6.65 @ 91 k). So:

> Two independently-initialised, independently-trained runs land on the same test-error floor as soon as — and only as soon as — they use the same estimator. α-Flow at α ≈ 0.005–0.02 reaches **2.66**, i.e. **2.5× better than either run ever achieves under the JVP target.**

The interval-bucket panel (bottom-right of fig. 1) makes the same point per-`h`: af's `h_mse_b1` and `h_mse_b3` (the long-interval buckets, which is where few-NFE sampling actually lives) drop to ~1e-3 in the window 55 k–71 k and jump back to 10⁰–10¹ at the cliff. mf never gets below ~0.5 in b2/b3 until very late and is unstable there.

**Two candidate readings, and they are separable:**

1. **The JVP target is genuinely worse on this problem** (visual conditioning, 128-d latent, U-Net). The `d/dr[u]` tangent is a single-sample estimate of a derivative through the whole conditioned network; the bootstrapped target at small α is a two-point finite difference of the *same* quantity, and finite differences are famously better-conditioned than exact derivatives when the function is noisy.
2. **The clamp is the bug, not the branch.** α does not decay to 0 — it is *snapped* there from 0.0052. The model spent 70 k steps adapting to a target that always had a discrete component (`discrete_frac` ≈ 0.5 throughout, since it also depends on `mf_mask`), and then that component vanishes entirely in one step. This is a distribution shift in the *target*, not in the data.

Reading 2 predicts the jump would shrink or vanish if α were annealed to, say, 1e-4 without clamping, or if the clamp were removed. Reading 1 predicts it would not. **That is a one-run experiment** and it is the single highest-value follow-up in this report — see §7.

Either way, the operational conclusion is the same and immediate: **`af_alpha_end: 0.0` is throwing away the best model this generation has produced.** af at α ≈ 0.007 is the best checkpoint in either arm by a factor of 2.5.

---

## 4. Every gradient step was clipped

| | mf | af |
|---|---:|---:|
| pre-clip ‖g‖ at step 0 | 2.46 | 2.45 |
| median over logged steps | **67.2** | **72.6** |
| p90 | 88.4 | 97.4 |
| max | 121.9 | 114.9 |
| fraction of logged points with ‖g‖ > `gradient_clip`=1.0 | **1.000** | **1.000** |

`gradient_clip: 1.0` is inherited from Gen3v6/v7, where it is applied to a **state-only** model. Here the gradient norm is two orders of magnitude larger — plausibly because the ResNet-18 vision encoders train end-to-end (`mf_freeze_vision_encoder: False`) and contribute the bulk of the norm.

Consequence: after step ~2 k, **every** update is renormalised to unit norm. The optimiser is running normalised/sign-like descent at ~1/70 the nominal step size, and the cosine LR schedule underneath it is largely decorative — the *direction* is preserved but the *relative* scaling between the encoder and the trunk is destroyed on every step. This is the strongest single explanation for why mf's test curve is flat from step 10 k to step 99 k (10.4 → 7.3 over 89 k steps).

This confirms, on two new arms, the standing lead from the earlier Gen14 curve analysis. It is not a new hypothesis; it is now a measurement on `mf` and `af` as well.

---

## 5. The policy collapses to "do nothing"

Parsing the `[ DIAG replan=… ]` lines (sampled every 50 replans, 236 samples for mf-diffuser, 234 for af-diffuser):

| run | fraction of sampled replans with ‖a₀‖ < 1 mm | direction z < −0.9 | **both** | median ‖a₀‖ |
|---|---:|---:|---:|---:|
| mf `diffuser` | 0.513 | 0.538 | **0.513** | 0.426 mm |
| af `diffuser` | 0.325 | 0.350 | **0.325** | 8.23 mm |
| mf `dpcc-r` | 0.211 | 0.232 | 0.211 | 8.16 mm |
| af `dpcc-r` | 0.191 | 0.213 | 0.191 | 8.17 mm |

The stall and the "pressing straight down" direction are **the same event** (the `both` column equals the `stalled` column exactly). And the stalled output is not arbitrary. The act normalizer is `mins=[-0.0083,-0.0083,-0.0083]`, `maxs=[0.0083,0.0083,0.0134]`. A physical zero action normalises to

```
z_norm = (0 − 0.00255) / 0.01085 = −0.235
```

and the observed stalled outputs are `norm|a0| ≈ 0.23`–`0.25` with `dir ≈ [~0, ~0, −1]`. **The model is emitting, to three digits, the normalised encoding of the zero action** — i.e. the marginal mean of the action distribution. This is textbook regression-to-the-mean / mode collapse, exactly what a heavily-clipped, under-converged L2 objective produces.

mf stalls on **51 %** of sampled replans vs af's **33 %** — consistent with mf's 2.4× worse test `raw_mse_u`. The per-bucket breakdown shows mf's stall concentrated in replans 100–300 (73 % / 62 %), i.e. the policy gets the approach right, reaches the box, and then stops.

---

## 6. Projection works. It just cannot be afforded.

![eval scatter](figs/fig2_eval.png)

Paired on the 11 contexts `dpcc-r` reached:

| | mf `diffuser` → `dpcc-r` | af `diffuser` → `dpcc-r` |
|---|---|---|
| Constraint sat rate | 0.839 → **0.872** | 0.805 → **0.966** |
| Violated steps / rollout | 64.0 → 51.4 | 78.2 → **13.5** |
| bounds / hs / obs | 37.0/21.5/6.9 → 46.3/0.0/5.1 | 48.3/0.2/29.7 → 6.7/0.8/6.0 |
| Plan post-proj viol rate | 0.128 → **0.018** | 0.155 → **0.009** |
| Max tracking error (mean) | 0.376 → **0.186** m | 0.384 → **0.156** m |
| Rollouts with tracking error > 1 m | 6/30 → **0/11** | 9/30 → **0/11** |
| Final mean distance | 0.247 → 0.368 m | 0.370 → **0.294** m |

The tracking-error panel of fig. 2 is bimodal: a controlled cluster at 0.03–0.09 m and a blown-up cluster at 0.3–2.7 m. **DPCC-R eliminates the blown-up cluster entirely** — zero rollouts above 1 m in either arm. Constraint satisfaction and post-projection violation rate both improve substantially, af dramatically (13.5 violated steps vs 78.2). Task distance is unchanged within noise. The projector is doing its job; the generator is the problem.

**But the cost is prohibitive:**

| run | s / replan | s / rollout | measured | projected 30 rollouts |
|---|---:|---:|---|---:|
| mf `diffuser` | 0.893 | 377 | 11 326 s (30/30) ✅ | 3.15 h |
| af `diffuser` | 0.902 | 377 | 11 302 s (30/30) ✅ | 3.14 h |
| mf `dpcc-r` | 14.99 (min 9.3, max 19.7) | ~6 020 | 66 239 s at 11/30, `~114 413 s to go` | **~50.2 h** |
| af `dpcc-r` | 16.42 (min 4.0, max 29.9) | ~6 375 | 70 130 s at 11/30, `~121 134 s to go` | **~53.1 h** |

Against a 24 h Slurm cap, `dpcc-r` cannot finish 30 contexts. Both jobs died mid-sweep. And the eval YAML lists **12 projection variants**, of which 11 involve projection — a full sweep at this cost is ≳ 500 h per seed. That is not a tuning problem, it is a design problem.

The `⚠ PROJECTION CIRCUIT-BREAKER TRIPPED` warning fired in 1 mf rollout and 2 af rollouts ("sustained SLSQP slowness", 4 unprojected steps in af's rollout 10) — those rollouts' constraint metrics are explicitly flagged invalid by the eval script itself.

---

## 7. K=100 on a model whose entire premise is K≈1

The plan folder is `H8_K100_…`. `flow_steps_v3 = 100` was inherited verbatim from `plan_fm_visual_aligning`, because `_mix_plan_common` copies every key from the FM plan template that is not `prefix`/`exp_name`/`diffusion`/`diffusion_loadpath`. Nothing in the mf/af plan blocks overrides it.

`mf_diffusion.py:202` reads `flow_steps = num_steps if num_steps is not None else self.flow_steps_v3`, so the sampler really does run 100 Euler steps of the two-time u-head, × `mpc_batch_size=4` candidates = **400 backbone evaluations per replan**. At 0.89 s/replan that is ~8.9 ms per (batched) call, which matches the 8.4–9.4 ms/NFE measured in the state-only DA.

This is not merely wasteful — it **inverts the claim the generation exists to test**. The Gen3v6/v7 state-only DA (`logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md`) evaluated MF and AF at **K=2**, and found AF at 0.0122 s and MF at 0.0168 s generation time, 11–15× faster than FM at K=20, with 1.000 success under `dpcc-r-tightened`. Gen14 is running the same two methods at **50× that step count** and reporting 0.9 s.

Cross-generation contrast, for the record:

| | Gen3v6/v7 (state-only) | Gen14 (visual) |
|---|---|---|
| K | 2 | **100** |
| Backbone | MF-DiT / SiT | **U-Net + FiLM v1** |
| Task | avoiding-d3il, halfspace | aligning-d3il-visual, combined_5 |
| Gen time | 0.012–0.017 s | 0.89–0.90 s |
| Success (best projection) | 1.000 (AF), 0.967 (MF) | 0.033 |

Three axes differ at once (K, backbone, task), so this is not an attribution — but K is the one that is free to fix and is unambiguously wrong for a few-step method.

At K=1 the `diffuser` sweep would drop from 3.15 h to ~2 min. It would **not** rescue `dpcc-r`: generation is only 0.89 s of the 15–16 s there, the remaining ~14 s is SLSQP. Fixing the K and fixing the projection budget are independent problems.

---

## 8. The evaluated checkpoint is not the best checkpoint

`eval_mix_visual_aligning.py:2284` — `if epoch == 'latest': epoch = utils.get_latest_epoch(loadpath)`. Both evals therefore loaded the **step-99 000** checkpoint.

| arm | best test `raw_mse_u` | @ step | final (evaluated) | penalty |
|---|---:|---:|---:|---:|
| mf | 6.648 | 91 000 | 7.293 | 1.10× |
| af | **2.657** | 70 000 | 6.959 | **2.62×** |

For mf this is minor. For af it is the whole ballgame: the evaluated checkpoint is on the wrong side of the α cliff, 2.6× worse than the model that existed 29 k steps earlier. **The α-Flow arm has never been evaluated at its own optimum.** If intermediate checkpoints were retained, evaluating af @ 70 000 is a zero-training-cost experiment.

(Train `raw_mse_u` shows the same shape: mf best 2.705 @ 94 k → 8.844 final, 3.3×; af best 1.685 @ 58 k → 5.570 final, 3.3×.)

---

## 9. What this run does and does not establish

**Established:**

1. The mf and af arms train end-to-end and evaluate end-to-end under vision. The plumbing works — no crashes, correct normalizers, correct paths, correct engine dispatch, 100 clean logging intervals on both.
2. The JVP MeanFlow target (α=0) sits at a distinctly worse error floor than the bootstrapped target (α ≈ 0.005–0.02) on this task, demonstrated twice: as a step change within the af run, and as a plateau across the whole mf run. Effect size ≈ 2.5×.
3. `gradient_clip=1.0` clips 100 % of steps at a median pre-clip norm of 67–73.
4. DPCC-R projection removes all tracking-error blowups and cuts post-projection violations by 7–17×, at 17–18× the inference cost.
5. The `dpcc-r` sweep as configured cannot complete inside the Slurm cap.

**Not established:**

- **Nothing about FiLM v1 vs v2 on these arms.** This run is v1 only. The U5 v2 port has never been trained or evaluated, and G7 has still never run.
- **Nothing about mf vs af as methods.** They tie at 1/30 success, and the af arm was evaluated 29 k steps past its optimum. The comparison is not yet meaningful.
- **No seed variance.** Seed 6 only.
- **Nothing attributable to the backbone.** Both are U-Net; the state-only comparison points are DiT/SiT (see U5 changelog §8).

---

## 10. Recommended next moves, ordered by information-per-GPU-hour

1. **Re-evaluate af @ step 70 000** (pre-cliff), same settings. Zero training cost if the checkpoint survives. Directly tests whether the α-cliff regression is what the eval is measuring.
2. **Set `flow_steps_v3` to 1–4 for the mf/af plan blocks.** They are few-step methods; K=100 makes the `diffuser` sweep 50–100× more expensive than it needs to be and misrepresents the method. Note `flow_steps_v3` is a *plan* path key — changing it produces a new results folder, it does not collide with existing output.
3. **Disambiguate the α cliff** — one af run with `af_alpha_clamp` lowered (e.g. 1e-4) or `af_alpha_end` set to a small positive value instead of 0. If the jump disappears, it is a clamp artifact; if it persists, the JVP target is genuinely inferior here and that is a *result*, not a bug. This is the highest-value single job in this list.
4. **Raise `gradient_clip`** to ~10 (or disable it and rely on the LR schedule) for the visual arms. The Gen3v6/v7 value was tuned on a state-only model with ~1/70 the gradient norm.
5. **Make the projection sweep affordable** before running it again — cut `n_contexts`, cut `projection_variants` to the 2–3 that matter, or profile SLSQP. As configured it is ≳ 500 h/seed.
6. Only after 1–4: run the FiLM v2 arms (U5's port) and the `diffusion`/`fm` reference arms, so there is something to compare against.

---

## 11. Reproduction

Figures are regenerated by `logs_in_develop/Gen14/U5/da_20260804_analysis.py` (numpy + matplotlib only,
no torch, no GPU — it runs in the AI-coding container):

```bash
python logs_in_develop/Gen14/U5/da_20260804_analysis.py     # → figs/fig1_training.png, figs/fig2_eval.png
```

Inputs, all under `temp/0408/`:

```
mf_losses.pkl
af)losses.pkl                                        # note the ')' in the filename as delivered
mix_visual_aligning_mf/H8_D…Emf_tslogit_normal/H8_K100_…_Emf/6/results_train_set/combined_5/
    {diffuser,dpcc-r}_train_set/eval_*.log
6/results_train_set/combined_5/                      # = the af tree (top-level duplicate, see §1)
    {diffuser,dpcc-r}_train_set/eval_*.log
```

The tabulated statistics (§2, §5, §6) come from the same two sources: `[ Seen Training Context N Finished ]`
blocks and `[ constraints ] …` blocks in the eval logs, plus `constraint_metrics.json` for the two completed
`diffuser` runs. Where a summary already exists in the eval log's own trailer, this report quotes it rather
than recomputing — the `first_violation_step` means, in particular, are the log's (violating rollouts only),
not an average over the `-1` sentinels.

All analysis in this document is read-only over `temp/0408/`. No code was changed and no cluster job was run.
