# Gen9E2U2F4 — Problem & Solution

**Date:** 2026-06-10 · **Updated:** 2026-06-11 (review integration + clip_denoised elevation, see §Review) · **Scope:** Avoiding Visual-DPCC (DDPM) "exploded chaotic lines" at eval
**Working assumption (user-confirmed):** this is a **CODE problem**, not a training problem.
**Critical caveat:** the codebase has since been **reset on the remote** — the local workspace audited in this session is **NOT the version that produced the failing runs**. Every "byte-identical / exonerated" claim from earlier audits is therefore **invalid for the failing runs** and must be re-established against the actual remote code state.

---

## Problem

DDPM avoiding eval produces exploded lines with **zero visible structure** — while the same training run's loss converged to ~0.0015 (the model fit its training data essentially perfectly). A model that memorized training data but shows *nothing* at eval is the classic signature of a **train↔eval pipeline mismatch in code**: the eval feeds the model something different from what it was trained on, or mangles the model's output on the way to the robot. The FM engine on the same task works, which narrows the bug to the DDPM-specific code path *as it existed on the remote*.

## Step 0 — Pin down the actual code version (do this before anything else)

Nothing below is decidable without this:

1. On the cluster: identify the commit/state the failing train and eval jobs ran under (`git log -1` at the repo, plus `git diff`/`git stash list` for uncommitted edits; the Slurm job logs record the submit time — match against reflog).
2. Diff that state against current local `diffuser_visual_avoiding/`, `diffuser_visual_avoiding_test/`, and `config/avoiding-d3il-visual.py`.
3. Any divergence in the files listed in the checklist below is a prime suspect — the local "clean" audit only proves the *current* code is clean, not the code that ran.

## Primary suspects — CODE (ranked, checked against the REMOTE version)

These are the places where a bug produces exactly "perfect train fit, zero eval structure". For each: what to check in the remote-version code.

### S1 — Visual conditioning never (correctly) reaches the model at eval
- `eval_visual_avoiding_dpcc.py` `VisualAgent.predict`: image must be BGR→RGB, CHW, `/255.`, float32 — byte-for-byte the same transform as `ParityAvoidingDataset` at training. A raw 0–255 image into an encoder trained on 0–1 saturates the FiLM latent → model output decorrelates from the scene → garbage trajectories despite perfect training.
- `VisualGaussianDiffusion.forward` cond unpacking (`{0: (bp_imgs, obs_seq)}` → `{'visual': ..., 0: snap_obs}`): a remote variant with the 2-cam aligning unpack, or a wrong tuple order, silently feeds obs as image or drops the image.
- `apply_conditioning` string-key guard: if the remote version lacked the `isinstance(t, str)` skip, the `'visual'` key would be applied as a timestep index and corrupt the trajectory tensor every step.

### S2 — Normalizer / action-decode mismatch at eval
- Eval must load the **same** `obs_normalizer.pkl` / `act_normalizer.pkl` saved by the training run, and apply them in the right direction (normalize obs in, unnormalize actions out). Wrong normalizer file, skipped unnormalize, or normalize/unnormalize swapped → every action becomes a near-max delta → robot flies off in straight-ish exploded lines. This visually matches the symptom exactly.
- Check `next_pos_des = action + obs[:2]` delta-decode against the dataset's `actions = des_xy[1:] − des_xy[:-1]` encode in the remote version.

### S3 — Loader / class-swap silently building the wrong model
- `load_diffusion_with_override`: the Fix_1-era class-swap prune (when pkl class ≠ target class) drops `_dict` keys. On the remote version, verify it was a no-op for this run and that `load_state_dict` ran **strict** — a `strict=False` load that skipped encoder weights leaves a randomly-initialized vision encoder: training looks fine, eval is structureless.
- Cross-package import quirk (the DPCC eval imports `fm_visual_avoiding.utils`; `import_class` prepends the package name — Fix_1 worked around it with importlib): if the remote version predates that workaround, the wrong class can be instantiated.

### S4 — Sampling-loop parameter mismatch baked in at eval
- `n_timesteps` used by `p_sample_loop` comes from the frozen `diffusion_config.pkl`; the `.py` config is **never read at eval** (no override mechanism exists, despite the intended design). If the remote eval injected config values anywhere (e.g. rebuilding the diffusion from config instead of pkl), train/eval chain-length or schedule mismatch follows.
- ~~Verify `clip_denoised` ended up False at runtime on the remote (the eval has a setter line — confirm it executed and what value it pulled).~~ ← corrected 2026-06-11: **no setter line exists** in `eval_visual_avoiding_dpcc.py` (verified by grep — zero `clip_denoised` references in the eval script). The pkl value wins unconditionally. See S5 below — this item is elevated to its own co-primary suspect.

### S5 — `clip_denoised=False` frozen in the checkpoint pkl ⚠️ ELEVATED to co-primary (2026-06-11)

Originally a one-line verify inside S4; elevated after review integration exposed a chain of facts:

1. **Fix_3 already diagnosed this exact mechanism** (`Fix_3/Diffu_FM_Comparison.md`: "Verdict: Code bug (config error) — clip_denoised=False in DDPM eval config"). With `clip_denoised=False` the reverse chain runs with **zero clamping**; `x_recon` errors compound across the K denoising steps → exploded trajectories. The base `GaussianDiffusion.p_mean_variance` **refuses to run** unclamped (`raise RuntimeError`) — the original authors considered it unsafe; `VisualGaussianDiffusion`'s override silently skips the clamp instead (verified: `visual_gaussian_diffusion.py:65-66`, no else branch).
2. **The Fix_3 fix never took effect at eval.** Fix_3 changed the `.py` plan config (False→True, commit ca7b232) — but the eval reads `clip_denoised` exclusively from the frozen `diffusion_config.pkl`; `args.clip_denoised` is never injected into the constructor (review Fact 1, verified). The `.py`-level fix was structurally a **no-op**. The hypothesis "clip_denoised=True fixes the explosion" has therefore **never actually been tested**.
3. **The config was then reverted to False** (commit 9797209, 2026-06-10, "misc trivial NOT important reverts") — so any future retrain freezes False into the new pkl again. Live trap.
4. **The symptom signature fits perfectly:** training loss is healthy because `p_losses` never runs the reverse chain (single-step epsilon MSE only); FM on the same task works because its deterministic ODE has no stochastic chain to diverge; only the DDPM explodes.

**Cheapest test in the whole list (do before/alongside S1):** read the frozen value from the failing checkpoint's `diffusion_config.pkl`; if False, patch it with the existing `diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py` tool and re-run eval only. Zero retrain, minutes of effort, and it either confirms or permanently eliminates the strongest historical suspect.

## Secondary factors — POSSIBLE, explicitly NOT the main reason

Kept for completeness only; do not prioritize:

- *(possible)* Checkpoint trained at `n_diffusion_steps=20` while the corrected config says 100 (checkpoint dir `H8_K20_...`); the working aligning recipe is K=100.
- *(possible)* `losses.pkl` shows test-loss minimum at step 11k vs eval at step 99k (5.19× worse) — overfit checkpoint selection. Note this **cannot** explain zero structure on its own; at most it amplifies whatever the code bug produces.

## Solution (for the implementing agent)

1. **Cheap test first (S5):** read `clip_denoised` from the failing checkpoint's frozen `diffusion_config.pkl`. If False → patch with `diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py`, re-run eval only. If the explosion disappears, done (then also re-apply the `.py` config True + add the comment back, reverting commit 9797209's revert, so retrains don't re-freeze the trap).
2. **Execute Step 0** — recover the exact remote code state of the failing runs. All other conclusions flow from this.
3. **Audit S1→S4 in that version**, in order. S1/S2 are the highest-yield: instrument the eval with a one-batch probe — dump (a) the image tensor stats entering the encoder, (b) the normalized obs cond, (c) the raw model trajectory output, (d) the unnormalized action — and compare each against the same quantities computed from one training batch. The first place the two diverge **is** the bug. *(Probe construction note, from review: fetch the training batch via DataLoader, not raw `dataset[idx]` — `to_device` raises on numpy inputs that the DataLoader would otherwise auto-convert.)*
4. **Fix the divergence found**, re-run eval only (no retrain — training is healthy by its own loss).
5. Add the missing **config-vs-pkl mismatch warning** at eval load (`n_diffusion_steps`, `clip_denoised`, `dim`, `horizon`) so silent precedence can never eat a fix again.
6. Only if S5 + S1–S4 all verify clean on the true remote version: revisit the secondary factors above (cheap test: re-eval a ~step-11k checkpoint if one exists).

**Definition of done:** the one-batch probe (step 3) shows train and eval pipelines numerically identical at every interface, and the eval rollout shows structured (even if imperfect) behaviour.

---

## Review integration (2026-06-11, from `Gen9E2U2F4 — Technical Review.md`)

The review's code verifications all check out (re-verified independently: eval image transform, `apply_conditioning` string guard, strict `load_state_dict`, pkl-only loading, `to_device` behaviour). Its Fact 1 (the `.py` config is never injected at eval — pkl wins) and Fact 2 (the subclass override never crashes regardless of the flag) are both correct and adopted above.

**One review conclusion is rejected** — Section 2's "clip_denoised is not a concern for this investigation. Keep the status quo — don't waste time on it." This is marked wrong in the review file. The review analyzed only crash-vs-no-crash and never considered the numerical divergence mechanism documented in Fix_3's own comparison doc; combined with its own Fact 1 it actually *strengthens* the clip_denoised hypothesis (the Fix_3 fix was a no-op at eval → never tested). Hence S5 above.
