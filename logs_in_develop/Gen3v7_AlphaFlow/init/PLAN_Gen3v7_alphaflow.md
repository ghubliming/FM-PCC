# PLAN — Gen3v7: **α-Flow** (bootstrapped MeanFlow target + α-anneal) as a Gen3v4 sibling

**Date:** 2026-07-22 · **Type:** implementation plan · **NO CODE WRITTEN YET**
**Status:** ready for hand-off · **implement SECOND**, after Gen3v6 is trained and evaluated
**Sibling plan:** [`../../Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md`](../../Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md)
**Upstream:** `/workspaces/aux_repo/alphaflow` @ `b0fef77` — snap-research, **official PyTorch**, arXiv 2510.20771
**Background:** [`../../imeanflow_train/AUDIT_...md`](../../imeanflow_train/AUDIT_port_vs_upstream_and_the_train_in_imeanflow_proposal.md) §9.3

---

## 0. Why this generation exists

Gen13's matched-NFE battery refuted iMF on avoiding ([`../../Gen13/fix_7/RESULTS_Gen13_fix7.3_VERDICT_imf_refuted.md`](../../Gen13/fix_7/RESULTS_Gen13_fix7.3_VERDICT_imf_refuted.md)), and the diagnosis in [`../../HF_iMF/Research/COMPARE_gen13_hardflow_vs_gen3v4_imf_training.md`](../../HF_iMF/Research/COMPARE_gen13_hardflow_vs_gen3v4_imf_training.md) §8.2 is structural:

> The MeanFlow residual sees only `δ_u − h·δ_D`. **Any error with `δ_u = h·δ_D` is invisible to the loss** — but the sampler uses `u` alone. Conditioning degrades as `h → 1`, which is exactly where 1–2 NFE lives.

**α-Flow removes that mechanism.** It replaces the JVP target with a **self-bootstrapped target computed under `no_grad`** — a *fixed tensor* at optimisation time. A fixed target has no blind direction: the loss measures `u` directly. And `α` anneals **1 → 0**, so training starts as plain flow matching (a regime this project has already validated at 100 % safety) and introduces the differential constraint only as the field becomes accurate.

**This is the one remaining idea with a real chance of making few-NFE work on this task.** Unlike the distillation alternative it needs no teacher model.

⚠️ **Honest prior:** α-Flow's own ImageNet margin over MeanFlow is real but modest (FID 43.1→40.2 at B/2 no-cfg; 3.47→2.95 at XL/2-cfg), at 400 k–1.2 M steps. And §7's transfer-gap argument still stands — a better-posed objective does not create data, and 96 episodes is still 96 episodes. Pre-register the kill criteria in §8.

---

## 1. Decisions

- **Generation number: Gen3v7.** `Gen3v5` is reserved (BNS Solver, `MASTER_TEST_HISTORY.md:23`); `Gen3v6` is MeanFlow.
- **Base: `flow_matcher_v3_imeanflow` (Gen3v4)** — same reasoning as the Gen3v6 plan §0.2: α-Flow needs a two-time network `u(z, τ, h)`, dual heads, a JVP path (for the α=0 branch) and a `u`-interval sampler. FMv3ODE has none of these.
- ⭐ **Better: base on Gen3v6 if it is already merged.** α-Flow's α→0 limit **is** MeanFlow (proved in §3.4). If `flow_matcher_v3_meanflow/` exists and passed its gates, copy *that* — you inherit the faithful `(t,r)` sampling, the official adaptive loss, the h-stratified metrics and the working gradient clip, and the α=0 branch is then literally Gen3v6's loss. **This is the reason Gen3v6 goes first.** Fall back to Gen3v4 only if Gen3v6 slipped.
- **Scope: state-only avoiding-d3il**, DiT backbone, matching Gen3v4/Gen3v6 so the three-way A/B is controlled.
- **CFG: OFF** for the first run (`omega=1.0`, `kappa=0.0`) — mirrors α-Flow's own `alphaflow-latentspace-B-2` non-cfg config and keeps the comparison to Gen3v6 clean. The `velocity_cfg` variable below is then just `v`.

---

## 2. 🔴 Convention mapping — get this right or nothing else matters

α-Flow and Gen3v4 use **opposite time axes**. Do the mapping once, here, and never again in the code.

**α-Flow** (`src/training/loss.py`, `RecFlowLoss.apply_noise`, verified):
```
x_t = (1 − t)·x + t·e        t = 1 noise  →  t = 0 data
v   = e − x                  (compute_targets)
step:  x_{t_next} = x_t − (t − t_next)·u        t_next < t
```
**Gen3v4 / Gen3v6 (DATA-AT-1)**:
```
z_τ = τ·x₁ + (1 − τ)·x₀      τ = 0 noise  →  τ = 1 data
v   = x₁ − x₀
step:  z_{τ+h} = z_τ + h·u
```
Mapping: **`τ = 1 − t`**, `u_ours = −u_αFlow`, `v_ours = −v_αFlow`.

| α-Flow symbol | our symbol | meaning |
|---|---|---|
| `t` | `r` | the **query point** (noise side) |
| `t_next` | `t` | the **destination** (data side) |
| `t − t_next` | `h` | interval width |
| `dt = α·(t − t_next)` | `dt = α·h` | the bootstrap sub-step, **toward data** |
| `x_t − dt·v_cfg` | `z_r + dt·v` | shifted point (sign flips twice ⇒ `+`) |
| `net(·, sigma_next=t_next, sigma=t−dt)` | `model(z_shift, τ=r+dt, h=h−dt)` | remaining-interval average velocity |

⚠️ Note the **`r`/`t` role swap**: α-Flow's `t` is our `r`. Naming a variable `t` on our side while meaning α-Flow's `t` is the single most likely bug in this generation. **Use only our names in the code.**

---

## 3. The objective

### 3.1 Batch routing

Each batch splits three ways (α-Flow `forward` + `sample_traj_params`):

| branch | when | target |
|---|---|---|
| **FM anchors** | fraction `ratio_fm` of the batch; forced `r = t` ⇒ `h = 0` | `u_tgt = v` |
| **discrete (bootstrapped)** | `h > 0` and `α > 0` | §3.3 |
| **continuous (JVP)** | `h > 0` and `α == 0` | §3.2 — **identical to Gen3v6's loss** |

`ratio_fm` and `α` are both **step-scheduled**. α-Flow ships `ratio_fm ∈ {0.25, 0.5, 0.75}` constant; Gen3v4 uses 0.5. Start at **0.5**.

### 3.2 Continuous branch (α = 0) — reuse Gen3v6 verbatim

```python
u_pred, du_dr = jvp(_u_of, (z_r, r, h), (v, ones, -ones))
u_tgt = (v + h·du_dr).detach()
```
This is Gen3v6's `_p_losses_meanflow` body unchanged. **Do not re-derive it.** (α-Flow's own continuous branch uses the *predicted* `velocity_cfg` as tangent, which under CFG-off equals `v` — so with CFG off the two agree exactly.)

### 3.3 ⭐ Discrete branch (α > 0) — the new thing

α-Flow's formula, transported into our convention:

```python
dt      = alpha * h                                   # (B,)
z_shift = z_r + pad(dt) * v                           # step toward data by dt at velocity v
with torch.no_grad():
    u_next, _ = self._predict_uv(z_shift, cond, r + dt, h=h - dt, returns=returns)
u_tgt = (pad(dt) * v + pad(h - dt) * u_next) / pad(h)
u_tgt = u_tgt.clamp(-self.af_clamp_utgt, self.af_clamp_utgt)   # 4.0
u_tgt = apply_conditioning(u_tgt, cond, action_dim, goal_dim=goal_dim, noise=True).detach()
```

⭐ **Because `dt = α·h`, this collapses to a convex combination:**
```
u_tgt = α·v + (1 − α)·u_next
```
Two consequences the implementing agent should exploit:
- **α = 1 ⇒ `u_tgt = v` exactly** — skip the `u_next` forward entirely (α-Flow does this too, via its `isclose(1 − dt/(t−t_next), 0)` guard). At α=1 the objective *is* flow matching.
- It is a **one-step interval-composition identity**: traverse `[r, r+dt]` at the instantaneous velocity `v`, then `[r+dt, r+h]` at the model's own average velocity, and average by arc length. Obviously correct, and it needs **no derivative of the network**.

⚠️ `u_next` **must** be under `torch.no_grad()`. If gradient flows into it you have re-created a self-referential target and thrown away the entire point of the generation.

### 3.4 Sanity: the α → 0 limit *is* the MeanFlow identity (verify this on paper before coding)

Expand `u_next = u(z_{r+dt}, r+dt, h−dt)` to first order in `dt`:
```
u_next ≈ u + dt · D_tot ,        D_tot = ∂_z u·v + ∂_r u − ∂_h u
u_tgt  = α·v + (1−α)(u + α·h·D_tot)
```
Setting `u_tgt = u` (the fixed point the regression drives toward) and dividing by α:
```
v − u + (1−α)·h·D_tot = 0   --α→0-->   u = v + h·D_tot
```
— exactly Gen3v4/Gen3v6/Gen13's MeanFlow identity. ✅ **The α-anneal is a genuine homotopy from FM (α=1) to MeanFlow (α=0).** This also gives you gate **G2** in §7 for free.

### 3.5 Loss and weighting

```python
err   = (u_pred - u_tgt).pow(2).sum(dim=(1, 2))          # per-sample SUM
w_br  = torch.where(discrete_mask, alpha_t, ones)        # α-Flow: weight_d = alpha
loss  = (w_br * err / (err + self.af_adp_eps).detach()).mean()
```
`af_adp_eps = 1e-3` (α-Flow's `adaptive_loss_weight_eps`). ⚠️ **Note this differs from MeanFlow/iMF's `0.01`** — keep α-Flow's value for the α-Flow arm, and say so in a comment so nobody "harmonises" it.

Keep the v-head loss from Gen3v6 (`loss_v`, same adaptive form) — the dual head is inherited and `dual_head=True`.

### 3.6 The α schedule

α-Flow's headline recipe (`infra/experiments/experiments-alphaflow.yaml:155`):
```yaml
alpha: {scheduler: sigmoid, initial_value: 1.0, end_value: 0.0,
        change_init_steps: 0, change_end_steps: 400000, gamma: 25.0, clamp_value: 0.005}
```
Port `get_ratio` (`loss.py:390-427`) — sigmoid branch plus the `clamp_value` snap:
```python
progress = (step - init) / (end - init) - 0.5        # centred
val = init_v + (end_v - init_v) / (1 + math.exp(-progress * gamma))
if val < clamp_value: val = 0.0                       # snap to the exact JVP branch
if val > 1 - clamp_value: val = 1.0
```
🔴 **Rescale `change_end_steps` to OUR budget.** α-Flow anneals over 400 k steps; Gen3v4 trains **100 k**. Set `af_alpha_end_step = n_train_steps` (100 000). Copying 400 000 verbatim means α never leaves ~1.0 and you have trained plain flow matching for 100 k steps and called it α-Flow. **This is the single most likely silent failure of this generation.**

⚠️ The `clamp_value: 0.005` snap matters: without it α becomes a tiny-but-nonzero number and every sample takes the discrete branch with `dt ≈ 0`, i.e. `u_next` evaluated ~at the query point — a degenerate near-identity target. The snap routes those samples to the **JVP** branch instead. Do not drop it.

### 3.7 🔴 Plumbing: the loss needs the global step

`flow_matcher_v3_*/utils/training.py:144` calls `self.model.loss(*batch)` — **no step is passed**. Required change, in the new sibling only:

```python
# utils/training.py, in train_epoch, immediately before the loss call:
if hasattr(self.model, 'set_train_step'):
    self.model.set_train_step(self.step)
loss, infos = self.model.loss(*batch)
```
and on the diffusion class:
```python
def set_train_step(self, step: int):
    self._train_step = int(step)
```
Use `self.step` (the **optimizer**-step counter, incremented once per accumulation group) — not the inner `for step in range(...)`. With `gradient_accumulate_every=2` the inner loop runs twice per optimizer step; using it would halve the effective schedule length.

⚠️ **Resume must restore it.** If a run resumes from a checkpoint, `self.step` must be restored *before* the first loss call or α restarts at 1.0 and the model unlearns. Verify against the existing resume path.

Also **log `alpha` every `log_freq`** — a run whose α never moved is otherwise indistinguishable from a working one.

---

## 4. Deliverables

| # | artifact |
|---|---|
| D1 | `flow_matcher_v3_alphaflow/` — copy of `flow_matcher_v3_meanflow/` (or `_imeanflow/`), then modify |
| D2 | `FM_v3_alphaflow_test/` — copy of the matching test folder |
| D3 | `config/avoiding-d3il.py`: `flow_matching_v3_alphaflow` + `plan_fm_v3_alphaflow` + `args_to_watch_fmv3_af_train` |
| D4 | `Slurm_Codes/sbatch/AlphaFlow/` — copy-modify of `sbatch/iMF/` |
| D5 | `logs_in_develop/Gen3v7_AlphaFlow/<epoch>/CHANGELOG_*.md` |
| D6 | MASTER row (§9) — **prepared, not applied** |

Mechanics of the copy, the module-path `sed`, the script renames and the experiment keys are **identical to the Gen3v6 plan §2** — follow it, substituting `meanflow` → `alphaflow`. Same hard rule: touch no existing generation.

---

## 5. Config block

**5.1** `args_to_watch_fmv3_af_train`:
```python
args_to_watch_fmv3_af_train = [
    ('prefix', ''), ('horizon', 'H'), ('diffusion', 'D'), ('action_weight', 'aw'),
    ('imf_backbone', 'bb'), ('t_schedule', 'ts'),
    ('af_alpha_init', 'ai'),        # 1.0
    ('af_alpha_end', 'ae'),         # 0.0
    ('af_alpha_gamma', 'ag'),       # 25.0
    ('af_ratio_fm', 'rf'),          # 0.5
]
```
⭐ Every α knob is in the folder name **on purpose** — `POST_U10_II` §1.1 documents a live overwrite hazard where four un-watched knobs let two different runs write the same directory. α-Flow has more sweepable knobs than any previous generation, so this matters more here than anywhere.

**5.2** `'flow_matching_v3_alphaflow'` — start from Gen3v6's block, then:
```python
'model':     'flow_matcher_v3_alphaflow.models.AlphaFlowEngine',
'diffusion': 'flow_matcher_v3_alphaflow.models.AlphaFlowODE',
'prefix':    'flow_matching_v3_alphaflow/',
'exp_name':  watch(args_to_watch_fmv3_af_train),

## α schedule (α-Flow experiments-alphaflow.yaml:155, rescaled to OUR budget)
'af_alpha_scheduler':  'sigmoid',
'af_alpha_init':       1.0,
'af_alpha_end':        0.0,
'af_alpha_init_step':  0,
'af_alpha_end_step':   100000,   # 🔴 == n_train_steps. Upstream's 400000 would be a no-op here.
'af_alpha_gamma':      25.0,
'af_alpha_clamp':      0.005,

'af_ratio_fm':    0.5,           # FM anchors (h=0)
'af_clamp_utgt':  4.0,           # α-Flow target clamp — no prior generation has this
'af_adp_eps':     1e-3,          # α-Flow value; NOT MeanFlow's 0.01

'dual_head': True, 'interval_cfg': False,
'imf_backbone': 'dit', 'dit_condition_on_t': False,
'p_mean': -0.4, 'p_std': 1.0, 't_schedule': 'logit_normal',
'n_train_steps': 100000, 'batch_size': 32, 'gradient_accumulate_every': 2,
'learning_rate': 5e-4, 'gradient_clip': 1.0, 'ema_decay': 0.995,
```
🔴 As in Gen3v6: **`gradient_clip` is a dead key in this lineage** (`POST_U10_III` §4.1 — never read by `utils/training.py`). Implement it for real. It matters *more* here: the discrete branch has no JVP and should be calmer, so a surviving spike is diagnostic rather than background noise.

**5.3** `'plan_fm_v3_alphaflow'` — copy `plan_fm_v3_imeanflow`, swap `prefix`/`diffusion`/`exp_name`, and make `diffusion_loadpath` reproduce `args_to_watch_fmv3_af_train` **token for token**. Architecture keys must equal the training block. `eval_use_ema: True`. Sweep `flow_steps_v3 ∈ {1, 2, 5, 10}`.
✅ **α is training-only** — it does not appear at eval, and the sampler is unchanged (`x += dt·u`). Nothing about inference differs from Gen3v4/Gen3v6, which is exactly what makes the three-way comparison clean.

---

## 6. SLURM

```bash
cp -r Slurm_Codes/sbatch/iMF Slurm_Codes/sbatch/AlphaFlow
cd Slurm_Codes/sbatch/AlphaFlow
mv train_imf.sh train_alphaflow.sh; mv eval_imf.sh eval_alphaflow.sh
mv load_results_imf.sh load_results_alphaflow.sh; mv imf_pipeline.sh alphaflow_pipeline.sh
```
Edit **only**: `--job-name` (`af_train`/`af_eval`/`af_pipeline`), the python paths (`FM_v3_alphaflow_test/...`), `--wandb-project FMPCC-AlphaFlow`, and `SBATCH_DIR="Slurm_Codes/sbatch/AlphaFlow"` + the two script names in the pipeline.

🔴 Leave untouched: conda activation, `PYTHONPATH`, `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl` / `MPLBACKEND=agg`, the **`MUJOCO_EGL_DEVICE_ID` GPU-leak abort guard**, the W&B key-file block, `trap on_exit`, the `latest.log` symlink. Keep `--time=24:00:00`. No live progress bars in batch logs.

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/alphaflow_pipeline.sh
```

**Do NOT vendor α-Flow's own infra.** It is hydra + `torchrun --nproc-per-node=8` + 5-D `[b,t,c,h,w]` video tensors + its own `src/train.py`. We take the **objective only** (`AlphaFlowLoss._compute_mean_velocity_d`, `get_ratio`, `sample_traj_params`) and run it inside our trainer on our existing sbatch. Importing their launcher would drag in a second training stack.

---

## 7. Gates

| gate | what | pass criterion |
|---|---|---|
| **G0** | build + one forward | shapes OK; no stale module refs |
| **G1** | **α = 1 ⇒ pure FM** — force `af_alpha_init = af_alpha_end = 1.0`, `ratio_fm = 0` | `u_tgt == v` **exactly** (bitwise); `raw_mse` matches an FM run's curve. This is the safety floor: **α-Flow at α=1 is a method we already know reaches 100 % safety.** |
| **G2** | **α = 0 ⇒ MeanFlow** — force `α = 0` | loss values match **Gen3v6**'s `_p_losses_meanflow` to `<1e-5` on identical inputs. Uses §3.4. **This is why Gen3v6 ships first.** |
| **G3** | intermediate α consistency | at α = 0.05 and small `h`, `‖u_tgt(α) − u_tgt(α→0 JVP)‖ / ‖u_tgt‖ < 0.1` — first-order agreement per §3.4 |
| **G4** | schedule is alive | 200-step smoke run; `alpha` logged and **moving**; `n_discrete`/`n_continuous` counts shift over training |
| **G5** | no gradient leaks into `u_next` | assert `u_tgt.requires_grad is False` |
| **G6** | full train 1 seed + eval at `flow_steps_v3 ∈ {1,2,5,10}` | §8 |

**G1 and G2 together prove the homotopy is wired correctly at both endpoints.** If either fails, stop — every intermediate α is then meaningless.

---

## 8. Success / kill criteria — pre-register BEFORE looking

The bar is the same as everywhere in this project: **matched-NFE against FM**, whose best config is `FM @ K=2 → 100 % safe, 0.1894 s/plan` (fix_7.3 §2, HardFlow lineage — state which FM comparator you use).

| outcome | reading |
|---|---|
| ⭐ α-Flow @ K=1–2 ≥ 95 % safe at ≤ 0.12 s/plan | **the generation succeeded** — few-NFE works once the target is fixed rather than self-referential. This is the result worth a paper. |
| α-Flow ≫ MeanFlow(Gen3v6) and ≫ iMF(Gen3v4), still < FM@K=2 | the blind direction was *a* cause but not the binding one ⇒ the remaining gap is data scale (`POST_U10_III` §7.2) |
| α-Flow ≈ MeanFlow ≈ iMF | 🔴 **the blind direction was never the operative cause.** COMPARE §8.2 is refuted as *the* explanation, and the honest conclusion is the fix_7.3 one: few-NFE MeanFlow-family methods do not transfer to 96-episode constrained control. **Stop the line.** |
| α-Flow worse than MeanFlow | suspect the schedule (§3.6 rescale) or a `u_next` gradient leak **before** believing the science — re-run G1/G2/G5 |

**Diagnostics that must be in the results MD** (not optional — they are what made fix_7.3 trustworthy):
1. `h`-stratified residual, inherited from Gen3v6 — does large-`h` accuracy finally improve?
2. **Endpoint error `‖x̂₁ − x₁‖` at the sampler's own grid** `{(τ=0,h=1), (0,0.5), (0.5,0.5)}`. iMF's was **flat in K at ~0.155** (fix_7.3 §4) — the signature of field error, not discretisation. **If α-Flow's is flat too, the objective change did not help.** This is the single most decisive number in the generation.
3. Matched-K safety and s/plan against MeanFlow (Gen3v6), iMF (Gen3v4) and FM.
4. Spike census: `raw_mse` max/median. iMF hit 65–500×; a calm α-Flow curve is itself a result.

⚠️ **Matched budget or nothing.** fix_7.3 §9: the entire Gen13 claim died because one hard-coded `k_steps=10` made the decisive control unrunnable, and the confound survived four rounds of analysis. Build the K-grid into the eval config from day one.

---

## 9. MASTER_TEST_HISTORY row — **prepared, not applied**

```markdown
| **Gen3v7 (α-Flow)** | [flow_matcher_v3_alphaflow/](../flow_matcher_v3_alphaflow) | [FM_v3_alphaflow_test/](../FM_v3_alphaflow_test) | July 2026 | **α-Flow (arXiv 2510.20771, snap-research)** — replaces the MeanFlow JVP target with a self-bootstrapped no-grad target `u_tgt = α·v + (1−α)·u_next`, with α annealed 1→0 (sigmoid) so training is a homotopy from flow matching to MeanFlow. Targets COMPARE §8.2's blind direction, the diagnosed cause of the Gen13 iMF refutation. Adds target clamping (4.0) and a step-scheduled objective. | working on |
```
Insert after the Gen3v6 row. Do **not** self-edit the master file.

---

## 10. Hand-off checklist

- [ ] Based on **Gen3v6** if available (else Gen3v4); folders copied; `__pycache__` purged; zero stale module refs
- [ ] Three-way batch routing implemented (FM anchor / discrete / continuous) with counts logged
- [ ] §3.3 discrete target implemented **in our convention** (§2), `u_next` under `no_grad`, clamped to `af_clamp_utgt`
- [ ] α=1 short-circuits the `u_next` forward
- [ ] Continuous branch is Gen3v6's JVP loss, unmodified
- [ ] `get_ratio` sigmoid + `clamp_value` snap ported; 🔴 `af_alpha_end_step == n_train_steps`
- [ ] `set_train_step` plumbed from `utils/training.py`; uses the **optimizer** step; survives resume; `alpha` logged
- [ ] `af_adp_eps = 1e-3` (α-Flow), commented as deliberately ≠ MeanFlow's 0.01
- [ ] 🔴 `gradient_clip` actually applied in `utils/training.py`
- [ ] 2 config blocks + `args_to_watch_fmv3_af_train` with **every α knob watched**; `diffusion_loadpath` token-for-token
- [ ] `Slurm_Codes/sbatch/AlphaFlow/` copied from `iMF/`; only listed lines changed; EGL guard untouched
- [ ] G0–G5 pass before the full run
- [ ] Endpoint-error-at-sampler-grid diagnostic implemented (§8.2) — **this is the decisive metric**
- [ ] changelog MD written; MASTER row handed over, not applied; nothing committed without permission

---

## 11. Traps specific to this generation

1. 🔴 **Copying `change_end_steps: 400000` verbatim.** α then never leaves ~1.0 → you trained flow matching for 100 k steps. **The #1 silent failure.** Assert `af_alpha_end_step == n_train_steps` at construction.
2. 🔴 **Gradient leaking into `u_next`.** Re-creates a self-referential target and voids the generation's entire premise. G5 exists for this.
3. **The `r`/`t` role swap (§2).** α-Flow's `t` is our `r`. Never name a variable after their convention.
4. **Dropping the `clamp_value` snap** ⇒ degenerate near-identity targets at tiny α.
5. **Using the inner accumulation-loop counter** for the schedule ⇒ effective schedule length halved.
6. **Resume restarting α at 1.0** ⇒ the model unlearns on every requeue. Verify against the resume path.
7. **Harmonising `af_adp_eps` to 0.01** because "MeanFlow uses that". Different method, different constant.
8. **Vendoring α-Flow's hydra/torchrun infra.** Take the objective, not the launcher.
9. **Folder-name collision** from un-watched α knobs (`POST_U10_II` §1.1).
10. **The inherited window-level `train_test_split` leak** (`POST_U10_III` §4.2) — at H=8 adjacent windows share 7 of 8 frames. Either fix it episode-level (hold out ~19 of 96 episodes) or label every val number as leaking. Do not present it as generalisation.
