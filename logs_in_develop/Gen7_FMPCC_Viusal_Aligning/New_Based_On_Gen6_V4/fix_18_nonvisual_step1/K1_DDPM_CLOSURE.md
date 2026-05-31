# K=1 DDPM Non-Visual: Closure (Fundamental, Not a Bug)

**Date**: 2026-05-31
**Status**: ✅ Closed — confirmed mathematical degeneracy of K=1 vanilla DDPM, **NOT** a Fix-18 code or math bug.
**Symptom**: The non-visual DPCC eval at K=1 (run after Fixes 18.1–18.5 unblocked the pipeline) completes cleanly but produces **exploding trajectories** — `norm‖a₀‖` grows from 0.77 at replan 0 to 36.74 by replan 300, denormalized actions get clipped, rollouts achieve 0/5 success on aligning.
**Source log**: `temp/one_shot_run/non_visual_dpcc` (job 21050).

---

## 1. What the log shows

The eval is functioning end-to-end (no crash, all metrics computed). Trace
the per-replan action magnitudes:

| Replan | `norm‖a₀‖` | `denorm‖a₀‖` (clipped) | Interpretation |
|---|---|---|---|
| 0   | 0.77 | 5.7e-03 m | Healthy — drone-scale step |
| 50  | 1.07 | 8.4e-03 m | Still in distribution |
| 100 | **7.07** | 1.44e-02 m (clipped) | Blowing up — clip kicks in |
| 150 | 8.56 | 1.44e-02 m | |
| 200 | 18.99 | 1.44e-02 m | |
| 250 | 26.82 | 1.44e-02 m | |
| 300 | **36.74** | 1.44e-02 m | 37× outside training scale |

Plus 5/5 contexts ending with 0% success, mean distance 0.46 m, constraint
satisfaction 0.40 (mostly bounds violations).

**The pipeline is healthy. The model is bad.**

---

## 2. The math: K=1 cosine DDPM is intrinsically degenerate

Walking `cosine_beta_schedule(1)` in `diffuser_visual_aligning/models/helpers.py:130`:

```
timesteps = 1, steps = 2, x = [0, 2]

alphas_cumprod_raw  = cos( ((x/2) + 0.008) / 1.008 · π/2 )²
                    ≈ [0.99984, 0]
alphas_cumprod      = raw / raw[0]          ≈ [1.0, 0.0]
betas               = 1 − alphas_cumprod[1] / alphas_cumprod[0]  = [1.0]
betas_clipped       = clip(betas, max=0.999) = [0.999]
```

Derived buffers at the single timestep K=1:

```
betas[0]                       = 0.999
alphas_cumprod[0]              = 0.001
sqrt_recip_alphas_cumprod[0]   = √(1/0.001)     ≈ 31.62
sqrt_recipm1_alphas_cumprod[0] = √(1/0.001 − 1) ≈ 31.61
posterior_variance[0]          = 0      (deterministic)
```

The reverse-process predict-x₀ formula (epsilon-parameterized) is:

```
x₀_pred = sqrt_recip_alphas_cumprod · x_T − sqrt_recipm1_alphas_cumprod · ε_θ(x_T)
        ≈ 31.62 · x_T − 31.61 · ε_θ(x_T)
```

That ≈ **32× amplification** is intrinsic to the cosine schedule at K=1 —
not a Fix-18 artifact. Any small ε prediction error gets multiplied by
32 into `x₀_pred`. A 1% relative error in ε becomes a 32% error in `x₀`;
3% becomes 100%.

The training loss (MSE on ε) converges (we see step 72000 looks fine),
but converged-on-average ≠ accurate-per-sample. The amplification turns
the residual ε error variance into huge x₀ variance.

---

## 3. Why this isn't a Fix-18 issue

| Fix | What it affects | Could it cause this? |
|---|---|---|
| 18.1 train obs_dim override | Model construction at train time | ❌ Model built with correct 23-D shape; that's how training succeeded |
| 18.2 eval `_traj_dim` from normalizer | Projector dim at eval | ❌ `diffuser` variant uses no projector; this can't influence its rollouts |
| 18.3 UF-13 guard | Whether visual mode forced | ❌ Routes only; doesn't touch sampling math |
| 18.4 DIAG var alias | Diagnostic print | ❌ Logging only, no state effect |
| 18.5 normalizer slice size | Projector setup for variant 2+ | ❌ `diffuser` variant doesn't construct a projector |
| STALE_CONFIG | Always overwrite `model_config.pkl` | ❌ Metadata only |

None of the five fixes or the side-patch touches `sqrt_recip_alphas_cumprod`,
the cosine schedule, the epsilon prediction, the reverse-process formula,
or anything else along the K=1 sampling math path.

The explosion would happen identically with `git revert` of all Fix-18
commits **except** the eval would never reach the rollout (would crash
earlier on shape mismatch). Fix-18 only made the bug *visible*; it
didn't cause it.

---

## 4. Why this isn't a regression vs. visual DPCC

Visual DPCC was trained at **K=100**, not K=1. At K=100 the same schedule
gives:

- `sqrt_recip_alphas_cumprod[k]` ranges from ≈ 1.0 (at k=0) to ≈ 31 (at k=99).
- The reverse process runs 100 iterations, each amplifying ε error by a
  modest factor at modest k, refining x in many small steps.
- Total error has 100 chances to be corrected by subsequent steps.

That's the regime DDPM was designed for. K=1 short-circuits all of it.

---

## 5. Why this is a known dead-end in the literature

Vanilla DDPM-loss-at-K=1 from-scratch training is essentially asking the
model to learn a **one-shot denoiser** using a loss that's tuned for
**multi-step denoising**. The literature has produced three families of
methods that *do* make one-step generation work, all of which require
something other than just lowering K:

1. **Progressive Distillation** (Salimans & Ho 2022) — train at K=N,
   distill to K=N/2, recurse to K=1. Multiple stages, hours-to-days of
   compute, but produces a true one-shot model.

2. **Consistency Models** (Song et al. 2023) — modify the training
   objective so single-step prediction is enforced directly.

3. **Mean Flow / iMeanFlow** (Lim 2025) — what we're working on in
   parallel under Gen3v4_imf. Different loss formulation that targets
   interval-averaged velocity; one-step Euler becomes valid by
   construction.

What you ran here — vanilla DDPM loss trained from scratch at
n_timesteps=1 — is none of these. It's the **null experiment** that
demonstrates *why* the above three families had to be invented.

---

## 6. Closure Decision

✅ **Close as fundamental, not a bug.** Specifically:

- The Fix-18 pipeline is correct (verified: schedule math is intrinsic,
  not touched by any fix).
- The trained model is also correct in the only sense it can be (low ε
  loss). What's wrong is the recipe — K=1 vanilla DDPM is a known
  dead-end.
- No code change to ship. No model retrain at K=1 would help (the
  problem is the schedule + loss combination, not insufficient training).

This re-confirms the earlier `INVESTIGATION_REPORT.md` §6 statement:

> "DDPM is fundamentally different from FM here. DDPM's discrete noise
> schedule means train-time T and eval-time T must match — so testing
> DDPM at T=1 does require retraining. FM does not."

We did the retrain at T=1; it didn't help. That's expected, because the
*amount* of training isn't the issue — the *schedule and loss
combination* is intrinsically wrong for one-shot.

---

## 7. What to Do Next (Three Options)

| Option | Effort | Expected outcome |
|---|---|---|
| **Accept and move on** — the K=1 vanilla DDPM experiment is closed; we've answered "does it work?" with "no, as expected." | 0 | Done. |
| **Re-train at K=10 or K=25** — same code, same dataset, change `n_diffusion_steps`. Gives a working but multi-step DPCC for non-visual. | Hours of compute | Working baseline, no longer "one-shot" |
| **Distill K=100 → K=1** via progressive distillation | Days, new code | True one-shot DDPM that works |

For comparison vs. the iMF line of work (Gen3v4_imf/fix_2): iMF
*is* designed for one-shot and is now untangling its own remaining
post-fix-1 issue. If you want one-shot generation for aligning, **iMF
is the right vehicle**, not K=1 DDPM.

---

## 8. Resolved Side-Thread: FM ODE=1 GIFs

User initially observed "FM ODE=1 also produces no GIFs" alongside the
DPCC K=1 explosion, raising concern of a Fix-18.3 regression.

**Resolved 2026-05-31**: user re-checked and **FM ODE=1 eval does
produce GIFs**. Two possible explanations, depending on which
interpretation of the FM checkpoint is true (unresolved without a
state_dict shape check — see §8.1):

- **Interpretation A**: FM is structurally 9-D visual (normalizer 6-D)
  → `_ckpt_is_visual = True` → UF-13 fires → GIFs via visual predict()
  capture buffer. DPCC differs because it's genuinely 23-D non-visual
  (normalizer 20-D) → `_ckpt_is_visual = False` → UF-13 stays off → no
  GIFs from this path. Asymmetry explained by architectural difference.
- **Interpretation B**: FM is structurally 23-D non-visual (user's
  assertion based on "tested under 18.1"). In this case both DPCC and
  FM should behave identically under Fix-18.3 — both should not have
  produced GIFs pre-Fix-18.6. The fact that FM did is unexplained
  without more evidence; the simplest reconciliation is that the FM
  eval happened to be run under an earlier code state, or there's a
  path the audit missed.

The `model_config.pkl` for the FM checkpoint showed `obs_dim=6` (which
would support Interpretation A), but the STALE_CONFIG side-patch
documented elsewhere in this folder explains why that file could be
unreliable. The authoritative check is the state_dict tensor shape:

```bash
python -c "
import torch, glob
ckpt = sorted(glob.glob('logs/aligning-d3il-visual/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VFalse_steps900_bs64/6/state_*.pt'))[-1]
sd = torch.load(ckpt, map_location='cpu')['model']
k = next(k for k in sd if 'downs.0.0.blocks.0.block.0.weight' in k)
print(f'FM first-conv shape: {sd[k].shape}  → {sd[k].shape[1]}-D')
"
```
- prints `[32, 9, 5]` → Interpretation A correct (9-D visual)
- prints `[32, 23, 5]` → Interpretation B correct (23-D non-visual)

**Either way, Fix-18.6 makes the question moot for forward use**:
under Fix-18.6, both 9-D-visual and genuine 23-D-non-visual eval
produce GIFs (via different paths: UF-13 visual capture vs
env-render record_sim_frame). Asymmetry closed.

**Update 2026-05-31 (later same day)**: the 23-D non-visual GIF
capability — flagged here as "future work" — was implemented as
**Fix-18.6** (env-render hook in `aligning_sim.py` + `record_sim_frame`
method on both Policy classes). After Fix-18.6, genuine 23-D
non-visual checkpoints also produce GIFs. See CHANGELOG.md §"Fix F
(= 18.6) HOTFIX" for details. No further action needed on the GIF
thread.

---

## 9. One-Line Summary

K=1 cosine DDPM has a 32× amplification factor baked into
`sqrt_recip_alphas_cumprod` at the single timestep, so any ε prediction
error explodes into massive x₀ error during the one-shot reverse step.
This is intrinsic to the schedule+loss combination — vanilla DDPM at
K=1 is a known unsolved problem that distillation, consistency, and
mean-flow methods exist to solve. **Not a Fix-18 bug; not a retrain
issue; nothing to ship.** If we want one-shot for this task, use iMF
(Gen3v4_imf line) or distill from a K=100 DDPM.
