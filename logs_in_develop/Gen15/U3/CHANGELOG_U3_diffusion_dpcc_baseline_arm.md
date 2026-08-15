# CHANGELOG — Gen15 U3: the DPCC baseline arm (`diffusion` / DDPM)

**Date:** 2026-08-15 · **Type:** new engine arm · **Status:** code complete, **NOTHING RUN**
**Files:** `mix_uav/models/ddpm_diffusion.py` (new), `mix_uav/models/unet1d_ddpm_cond.py` (new),
`mix_uav/models/{__init__,engine_registry}.py`, `mix_uav_test/{eval_mix_uav,gates_mix_uav}.py`,
`config/uav_mix.py`, `Slurm_Codes/sbatch/uav_mix/*.sh`
**Retraining:** required — this arm has no checkpoint anywhere. One training run per scene.

Gen15 now hosts **four** engines: `fm | mf | af | diffusion`.

---

## 1. Why

On `avoiding-d3il`, DPCC's `GaussianDiffusion` **is the Target** every DA is measured against
(`H8_K20_Dmodels.GaussianDiffusion_aw10`, 1.000 S&C / 70.13 steps). On UAV that row never
existed — Gen11 went straight to flow matching. That is why every Gen15 claim so far has been
capped at *"vs naive FM + DPCC"* and could never be *"beats DPCC"* (init plan §1.5, run report
§6). This arm closes the gap.

It also supplies the row that makes Gen15's real-time thesis legible: a denoiser at K=20 cannot
be cut to 1–2 steps — that inability is what flow matching exists to fix. From gate G6's
measurements (~8.7 ms per network eval at this size) it should land near **175 ms/plan against a
30.3 ms budget**, before the projector. It is the baseline that shows *why* few-step matters.

## 2. What was copied, and what changed

| file | from | delta |
|---|---|---|
| `mix_uav/models/ddpm_diffusion.py` | `diffuser/models/diffusion.py` | header + the Fix_1 contract (§3) |
| `mix_uav/models/unet1d_ddpm_cond.py` | `diffuser/models/unet1d_temporal_cond.py` | verbatim |

**Copied, not imported.** `diffuser/` is shared by every generation in this repo — Gen11,
Gen12, Gen14 and the DPCC baselines themselves. Patching it there would reach all of them.
`git status diffuser/` is clean.

Dependencies check out: `cosine_beta_schedule`, `extract`, `apply_conditioning`, `Losses` all
exist in `mix_uav/models/helpers.py` (whose only diff from `diffuser`'s is a docstring on
`cosine_beta_schedule` saying it is "kept for compatibility… Flow Matching does not rely on a
beta schedule" — now it does again).

**Backbone is architecture-matched to the `fm` arm.** `UNet1DTemporalCondModel` and
`Flow_matcher_U_Net_v2` are the same temporal U-Net at `dim=32`, `dim_mults=(1,2,4,8)`; the only
structural difference upstream is an optional `use_cond_projection`/`cond_mlp` branch, left OFF.
Gate G3 asserts the param counts still agree across all four arms.

## 3. Fix_1, fourth instance — and a fifth gap it exposed

`diffuser`'s `GaussianDiffusion` does **not** emit `infos['projection_ms']` (grepped: 0 hits).
Since the eval derives `fm_ms = total_ms − proj_ms`, this arm would have arrived with the exact
bug already fixed on `mf`/`af` (Fix_1) and on HardFlow (U2 §9). Patched in the Gen15 copy.

**A gap that patch exposed:** the *gradient* projector runs **inside** `p_sample`, not at a call
site, so it cannot be timed the way `project()` can. The Gen15 copy charges the whole guided
step, which makes `proj_ms` on `gradient*` variants a slight **over**-estimate of pure projector
cost. Documented in code rather than silently mixed with the exact DPCC numbers.

⚠️ **The same gap exists on the `fm` arm and is inherited from Gen11** (`mix_uav/models/
diffusion.py`, legacy-euler branch: `x = self.p_sample(..., projector=projector, ...)` with no
timer). It affects 2 of 20 variants (`gradient`, `gradient-tightened`) and only their `proj_ms`
/ `fm_ms` split — `total_ms` is unaffected. Left as-is for now; fixing it changes the `fm` arm's
reported split versus Gen11's.

## 4. 🔴 Two ways this arm is genuinely not like the others

**(a) K is a TRAINING-time property.** `p_sample_loop` iterates
`reversed(range(0, self.n_timesteps))` and the beta schedule is built from `n_timesteps` in
`__init__`. Re-pointing it at eval — the way `flow_steps_v3` re-points the flow arms — would
desynchronise the schedule from the loop bound and silently corrupt the sampler.

Consequences, all now enforced:
- `engine_registry.apply_nfe(..., engine=...)` is a **no-op** on this arm and says so in the log.
- `n_diffusion_steps` is an **exp_name token** (`_K{n}`), so two budgets cannot share a
  checkpoint folder. This is the only arm whose *training* path carries K — and it matches how
  the avoiding-d3il baselines are organised (`diffusion/H8_K20_…`, `H8_K10_…`, `H8_K1_…` are
  three separately trained models).
- A K sweep on this arm = **separate training runs**, not `eval_k_sweep.sh`.
- Gate G6 reports the single budget the checkpoint has rather than sweeping a sampler that
  cannot exist.

**(b) No HardFlow arm.** HardFlow's NLP needs an instantaneous velocity field `v = f(x, t)`; a
DDPM predicts noise/x0 and has no `_predict_velocity`. The registry marks it
`supports_hardflow=False`, the eval drops the `hardflow_*` variants with a printed reason
instead of crashing inside the sampler, and gate G8 asserts the exclusion is *correct* (it fails
if the class turns out to have `_predict_velocity` after all).

So this arm runs **20 DPCC variants**, where `fm`/`mf`/`af` run 23.

## 5. ⚠️ `action_weight = 1`, not DPCC's 10

DPCC's baseline block uses `action_weight: 10`; every Gen15 arm and all of Gen11 use `1`.
**This arm uses 1.**

Rationale: Gen15's question is *"which objective wins on this task"*, not *"reproduce DPCC's
hyperparameters"*. `aw1` keeps this arm comparable to `fm`/`mf`/`af` — same task config, same
backbone, same budget, same normalizer — which is what makes the four-way meaningful.

**Consequence to state in any write-up:** this is **not** a like-for-like reproduction of the
avoiding-d3il Target row. It is "DDPM under Gen15's UAV config". If the paper-faithful comparison
is ever wanted, train a second `aw10` variant — and add an `aw` token to `exp_name_tokens` first,
or the two will collide in one folder.

## 6. 🔴 A silent bug this pass found in the `af` arm

While verifying the new arm's exp_name tokens I dumped the token table for all four engines and
found the `af` row declared **`af_alpha_start`** — a key that does not exist. The real config key
is **`af_alpha_init`**.

`_uav_mix_exp_name` renders a token only `if hasattr(args, key)`, so the misspelling was
**completely silent**: the `as` token never appeared on any α-Flow path. No α-Flow run exists
yet, so nothing is invalidated — but two `af` runs differing only in `af_alpha_init` would have
shared a checkpoint directory.

Fixed, and gated: **G2 gained part (d)** — every declared `exp_name_token` key must actually
exist in that engine's train block. This is the guard the original G2 lacked: it tested that 16
knob combinations produced 16 distinct names, which passed *by accident* because the remaining
tokens still differed.

## 7. How to run it

```bash
# 0. gates — G0/G3/G7 now cover four arms; G2(d) is new; G8 asserts the hardflow exclusion
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/gates_mix_uav.sh

# 1. train the baseline (one run per scene; K=20 is baked in)
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh diffusion corridor "6"

# 2. eval — 20 DPCC variants, no hardflow arm. The trailing K is IGNORED on this arm (§4a).
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh diffusion corridor "6" 10 fm_only none 20
```

Expected log lines:
```
[ train ] Gen15 UAV Mix-ML — engine: diffusion  (DDPM / DPCC baseline (GaussianDiffusion))
[ train ] savepath: logs/UAV_MIX/uav-corridor/mix_uav_diffusion/H8_D…GaussianDiffusion_9D_K20/6
[ registry ] engine 'diffusion': K is train-time (n_timesteps=20); ignoring the eval-side flow_steps=20
[ eval ] engine 'diffusion' does not support HardFlow (no instantaneous velocity field) → dropping [...]
```
The `_K20` in the **training** path and the `[ registry ]` line are the two things to confirm.

**Comparison note:** corridor `mf` @ K=10 is already in hand, so `diffusion` @ K=20 on corridor
gives the first four-way-capable row. Quote NFE honestly — `diffusion` at K=20 is 20 network
evals, `mf` at K=10 is 10, and a HardFlow arm at K is 2K (U2 §4).

## 8. Open

1. **`aw10` sibling** for a paper-faithful DPCC row (§5) — needs an `aw` exp_name token first.
2. **The `fm` arm's gradient-branch timing gap** (§3) — 2 variants, split only, inherited from
   Gen11.
3. **K sweep on this arm** means N training runs. If that is wanted, `{1, 5, 10, 20}` matches the
   avoiding-d3il ladder and the `_K{n}` token keeps them separate automatically.
