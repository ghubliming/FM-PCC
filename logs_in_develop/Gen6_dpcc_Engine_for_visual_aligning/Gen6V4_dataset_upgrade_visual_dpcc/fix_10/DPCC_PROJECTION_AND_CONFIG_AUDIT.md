# DPCC Projection and Configuration Audit — `aligning-d3il-visual.py`

**Date:** 2026-05-20
**Investigator:** Antigravity (Claude Opus / Gemini 3.5 Flash)
**Triggered by:** Developer flagging dead params in `plan_visual_aligning_dpcc` config block

---

## The Question

The developer found these "dead" parameters in `config/aligning-d3il-visual.py`:

| Param | Claimed status |
|---|---|
| `policy: 'sampling.Policy'` | Eval uses VisualAgentWrapper — never instantiated |
| `test_ret: 0` | Only read when `returns_condition: True` |
| `value_loadpath` | No value function in DPCC pipeline |
| `dynamic_loss: False` | Training-only flag, meaningless at eval time |

**User's concern:** If the goal is to replicate DPCC behaviour for visual aligning, why are key modules like `sampling.Policy` and value functions not used? Are these signs of a severe architectural bug?

---

## Finding 1 — `sampling.Policy` is **intentionally bypassed, not broken**

### How the original DPCC pipeline works (`scripts/eval.py`)

The non-visual DPCC pipeline (obstacle avoidance) uses a clean architecture:

```
config → Parser → args.policy = 'sampling.Policy'
                              ↓
scripts/eval.py:214  →  policy = Policy(model=diffusion, normalizer=..., projector=..., test_ret=...)
                              ↓
scripts/eval.py:291  →  action, samples = policy(conditions={0: obs}, ...)
                              ↓
                     Policy.__call__() handles:
                       - Normalizing conditions
                       - Calling diffusion model
                       - Trajectory selection (temporal_consistency / minimum_projection_cost)
                       - Un-normalizing actions
                       - Returning single action + full trajectory
```

`Policy` is a **middleware class** that wraps the diffusion model, handles normalization, trajectory selection, and action extraction in one clean `__call__`.

### How the visual pipeline works (`eval_visual_aligning_dpcc.py`)

Both visual eval scripts (`eval_visual_aligning_dpcc.py` and `eval_ddpm_encdec_vision.py`) replace `sampling.Policy` with a custom `VisualAgentWrapper`:

```
eval_visual_aligning_dpcc.py:772  →  agent = VisualAgentWrapper(diffusion_model=..., projector=..., ...)
                                            ↓
                                  Aligning_Sim.test_agent(agent)
                                            ↓
                                  agent.predict(state, if_vision=True)
                                            ↓
                                  VisualAgentWrapper.predict() handles:
                                    - Image context window management
                                    - 6D obs construction (des_c_pos + c_pos)
                                    - Normalization (obs_normalizer / act_normalizer)
                                    - Calling diffusion model directly
                                    - Trajectory selection (same 3 strategies)
                                    - Un-normalizing actions
                                    - Mental robot position tracking
                                    - Video frame capture
                                    - Diagnostics logging
```

### Why `Policy` was replaced

`sampling.Policy` was designed for **state-only** tasks (avoiding, pointmaze, antmaze). It assumes:
- Observations are 1D numpy vectors
- Normalization uses `dataset.normalizer` (a `GaussianNormalizer` or `LimitsNormalizer` with `.normalize()/.unnormalize()` keyed by `'observations'`/`'actions'`)
- Conditioning is `{0: obs_vector}`

The visual pipeline needs:
- **Image** conditioning (bp_cam + inhand_cam as 4D tensors)
- **Multi-modal** condition dict: `{0: (bp_batch, inhand_batch, obs_batch)}`
- Separate `obs_normalizer` and `act_normalizer` (LimitsNormalizer instances, not keyed dict)
- D3IL agent interface (`reset()`, `predict()`, `update_rollout_info()`)
- Mental position tracking for open-loop MPC
- Per-step video capture

**`sampling.Policy` cannot handle any of this.** It would crash at `_format_conditions()` because it tries to normalize tuples through `utils.apply_dict()`.

### Verdict: ✅ NOT A BUG

`VisualAgentWrapper` is a purpose-built replacement that **reimplements all of Policy's responsibilities** (normalization, trajectory selection, action extraction) plus visual-specific features. The config's `policy: 'sampling.Policy'` is dead because the eval script correctly ignores it.

---

## Finding 2 — `test_ret: 0` is **correctly dead**

`test_ret` feeds into `returns`-conditioned generation:

```python
# sampling/policies.py:48
returns = to_device(test_ret * torch.ones(batch_size, 1), 'cuda')
# → passed to: self.model(conditions, returns=returns, ...)
```

Inside the U-Net (`unet1d_temporal_cond.py:230`):
```python
if self.returns_condition:
    assert returns is not None
    returns_embed = self.returns_mlp(returns)
    t = torch.cat([t, returns_embed], dim=-1)
```

When `returns_condition: False` (as set in all visual configs), the `returns` argument is **never read** by the model — the `if` branch is skipped entirely. `VisualAgentWrapper.predict()` never passes `returns` at all.

### Verdict: ✅ NOT A BUG

Returns conditioning is a reward-guided generation feature from Diffuser (Janner et al.). D3IL's aligning task doesn't use reward-guided generation. `test_ret` is dead by correct architectural choice.

---

## Finding 3 — `value_loadpath` is **correctly dead**

`value_loadpath` would point to a trained value function used for guided sampling in Diffuser-style reward-guided planning. The visual DPCC pipeline:
- Does not train a value function
- Does not use value-guided sampling
- Uses DPCC projection (SLSQP constraint enforcement) instead of gradient-based value guidance

Neither eval script references `args.value_loadpath`. No value function `.pt` file exists in the checkpoint directory.

### Verdict: ✅ NOT A BUG

Value-guided planning and DPCC constraint projection are **alternative** approaches to trajectory quality improvement. Visual-DPCC chose projection. The value path is dead by design.

---

## Finding 4 — `dynamic_loss: False` is **correctly dead at eval**

`dynamic_loss` is a training-time flag that would modify the loss function. At eval time, only the forward sampling path (`p_sample_loop`) is used — no loss computation occurs. Grepping `diffuser_visual_aligning/` for `dynamic_loss` returns zero results; the visual pipeline never reads it.

### Verdict: ✅ NOT A BUG

---

## The Real Question: Is the Visual Pipeline a Faithful DPCC Replication?

The user's underlying concern is whether the visual pipeline properly replicates DPCC. Let me compare the **functional capabilities**:

| DPCC Feature | `scripts/eval.py` (original) | `eval_visual_aligning_dpcc.py` (visual) | Match? |
|---|---|---|---|
| Diffusion model inference | `Policy.__call__()` → `self.model(cond, ...)` | `VisualAgentWrapper.predict()` → `self.model(cond, ...)` | ✅ |
| SLSQP constraint projection | `Projector` passed to `Policy` → `self.model(..., projector=...)` | `Projector` passed to wrapper → `self.model(cond, projector=...)` | ✅ |
| Trajectory selection (3 modes) | `Policy.__call__()` lines 65-76 | `VisualAgentWrapper.predict()` lines 547-572 | ✅ |
| Batch candidate generation | `batch_size=6` from config | `batch_size=6` hardcoded for DPCC variants | ✅ |
| Action normalization/denormalization | `Policy`: `dataset.normalizer.unnormalize(actions, 'actions')` | Wrapper: `self.act_normalizer.unnormalize(act_np)` | ✅ |
| Obs normalization | `Policy._format_conditions()` | Wrapper: `self.obs_normalizer.normalize(obs_6d_np)` | ✅ |
| `max_episode_length` from config | `scripts/eval.py:263`: `for _ in range(args.max_episode_length)` | **❌ DEAD (Fix 10 bug)** — env hardcodes 400 | ⚠️ **Fixed** |
| `diffusion_timestep_threshold` | Passed to Projector | Passed to Projector | ✅ |
| Action chunking (action_seq_size) | `Policy` returns single action per call | Wrapper supports `action_seq_size` > 1 | ✅ (superset) |
| Image conditioning | N/A (state-only) | Full image pipeline | ✅ (extended) |

### The one structural difference that matters

In `scripts/eval.py`, the eval loop is:
```python
for _ in range(args.max_episode_length):     # ← controlled by config
    action, samples = policy(conditions={0: obs}, ...)
    obs, rew, terminated, truncated, info = env.step(action)
    if success or terminated or _ == args.max_episode_length - 1:
        break
```

In `eval_visual_aligning_dpcc.py`, episode length is delegated to D3IL's `Aligning_Sim.eval_agent()`:
```python
while not done:                               # ← controlled by GymEnvWrapper.is_finished()
    pred_action = agent.predict(...)          #    which reads max_steps_per_episode (hardcoded 400)
    obs, reward, done, info = env.step(pred_action)
```

The original DPCC eval **directly uses `args.max_episode_length`** in its step loop. The visual pipeline delegates to D3IL's env, which **ignores the config** (Fix 10 bug). This is the only real architectural gap, and it's already documented and fixed.

---

## Summary Verdict

| Dead Param | Severe Bug? | Reason |
|---|---|---|
| `policy: 'sampling.Policy'` | ❌ No | Replaced by `VisualAgentWrapper` — a visual-capable superset |
| `test_ret: 0` | ❌ No | Returns conditioning correctly disabled (`returns_condition: False`) |
| `value_loadpath` | ❌ No | DPCC uses projection, not value guidance — alternative by design |
| `dynamic_loss: False` | ❌ No | Training-only flag, irrelevant at eval |
| `max_episode_length: 1000` | ⚠️ **YES** | Was dead due to missing wiring (Fix 10) — **already fixed** |

**None of the four flagged parameters indicate a missing module or broken DPCC replication.** They are config noise inherited from the original Diffuser/D3IL template. The visual pipeline correctly reimplements all DPCC functionality through `VisualAgentWrapper` + `Projector`.

The only real bug was `max_episode_length` (Fix 10), which was a wiring failure, not a missing module.

---

### Recommendation

These dead params should be annotated in the config to prevent future confusion:

```python
'plan_visual_aligning_dpcc': {
    # ...active params...

    # ── LEGACY / UNUSED (safe to remove, kept for parser compatibility) ──
    'policy': 'sampling.Policy',    # DEAD: visual eval uses VisualAgentWrapper, not Policy
    'test_ret': 0,                  # DEAD: returns_condition=False, value never read
    'value_loadpath': '...',        # DEAD: no value function in DPCC (uses projection instead)
    'dynamic_loss': False,          # DEAD: training-only flag, eval never reads it
    'predict_epsilon': True,        # DEAD: hardcoded in model __init__, config value ignored
}
```

No code changes needed. No modules missing. No severe bugs from these params.

---

## Appendix — How DPCC Projection Actually Works (Original `/workspaces/dpcc` vs Visual Pipeline)

**User's core question:** If `sampling.Policy` is bypassed, how do DPCC concepts like projection still work in `diffuser_visual_aligning`?

**Answer:** Projection does **NOT** live inside `Policy`. It lives inside the **diffusion model itself** (`GaussianDiffusion.p_sample_loop`). `Policy` is just a convenience wrapper — projection would work with or without it.

---

### The DPCC Projection Architecture (from `/workspaces/dpcc`)

The DPCC projection engine (`Projector`) is injected into the **denoising loop**, not into the policy wrapper. Here's the full call chain in the original DPCC codebase:

```
┌─────────────────────────────────────────────────────────────────────┐
│ scripts/eval.py                                                     │
│   policy = Policy(model=diffusion, projector=projector, ...)        │
│   action, samples = policy(conditions={0: obs}, ...)                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ sampling/policies.py :: Policy.__call__()                           │
│   L51: projector = self.projector if not disable_projection else None│
│   L52: samples, infos = self.model(cond, projector=projector, ...)  │
│                                    │                                │
│   (Policy just PASSES projector through to the model)               │
│   (Policy does NOT do any projection itself)                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ models/diffusion.py :: GaussianDiffusion.forward()                  │
│   → conditional_sample()                                            │
│     → p_sample_loop(shape, cond, projector=projector, ...)          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GaussianDiffusion.p_sample_loop()  ← THIS IS WHERE DPCC HAPPENS    │
│                                                                     │
│   for i in reversed(range(n_timesteps)):                            │
│                                                                     │
│     ┌─ Gradient-based projection (during denoising) ──────────┐     │
│     │ L179: if projector.gradient and t <= threshold:         │     │
│     │         x = p_sample(x, cond, t, projector=projector)   │     │
│     │           → p_mean_variance():                          │     │
│     │               grad = projector.compute_gradient(x_recon)│     │
│     │               model_mean = model_mean + grad            │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                     │
│     ┌─ Post-processing SLSQP projection (after denoising) ───┐     │
│     │ L186: if not projector.gradient and t <= threshold:     │     │
│     │         x, costs = projector.project(x)                 │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                     │
│   return x, infos={'projection_costs': costs}                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** `Policy` is a pass-through for the projector. It receives `projector` in its constructor, stores it as `self.projector`, and blindly forwards it to `self.model(cond, projector=projector)` at L52. **All actual projection logic executes inside `GaussianDiffusion.p_sample_loop()`.**

---

### The Visual Pipeline — Same Projection, Different Entry Point

The visual pipeline bypasses `Policy` but calls the **exact same model method** with the **exact same `projector=` kwarg**:

```
┌─────────────────────────────────────────────────────────────────────┐
│ eval_ddpm_encdec_vision.py                                          │
│   agent = VisualAgentWrapper(diffusion_model=..., projector=...)    │
│   sim.test_agent(agent)                                             │
│     → agent.predict(state, if_vision=True)                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ VisualAgentWrapper.predict()                                        │
│   L486: if self.projector is not None:                              │
│             trajectory, infos = self.model(cond, projector=self.projector)│
│         else:                                                       │
│             trajectory, infos = self.model(cond)                    │
│                                    │                                │
│   (Wrapper PASSES projector directly to model — same as Policy)     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ VisualGaussianDiffusion.forward()                                   │
│   → reformats visual cond dict                                      │
│   → super().forward(new_cond, projector=projector, ...)             │
│     → conditional_sample()                                          │
│       → p_sample_loop(shape, cond, projector=projector, ...)        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GaussianDiffusion.p_sample_loop()  ← EXACT SAME CODE AS DPCC       │
│                                                                     │
│   (identical gradient-based and SLSQP projection hooks)             │
│   (diffuser_visual_aligning/models/diffusion.py is a FORK of       │
│    dpcc/diffuser/models/diffusion.py — same p_sample_loop logic)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Code-Level Proof: The Diffusion Models Are Forks

I diffed the two `p_sample_loop` implementations:

| Component | `/workspaces/dpcc/diffuser/models/diffusion.py` | `FM-PCC/diffuser_visual_aligning/models/diffusion.py` | Identical? |
|---|---|---|---|
| `p_sample_loop` signature | `(shape, cond, returns=None, ..., projector=None, constraints=None, repeat_last=0)` | Same | ✅ |
| Gradient-based projection (L179) | `if projector is not None and projector.gradient and t <= threshold` | Same | ✅ |
| SLSQP post-processing (L186) | `if projector is not None and not projector.gradient and t <= threshold` | Same | ✅ |
| `projector.project(x)` call | L191 | L191 | ✅ |
| `projector.compute_gradient(x_recon)` | L149 | L149 | ✅ |
| `infos['projection_costs']` return | L200 | L200 | ✅ |

The `VisualGaussianDiffusion` subclass **inherits** `p_sample_loop` from `GaussianDiffusion` — it only overrides `p_mean_variance` (for selective action clamping) and `forward` (for visual cond reformatting). The projection hooks in the denoising loop are **byte-for-byte identical** to `/workspaces/dpcc`.

---

### The `Projector` Class Itself

The visual pipeline's `Projector` (at `FM-PCC/diffuser_visual_aligning/sampling/projection.py`) is a **direct fork** of `/workspaces/dpcc/diffuser/sampling/projection.py` with only two modifications:

1. **Fix 9.1:** No-op guard when no constraints are active (prevents SLSQP corruption on unconstrained diffuser baseline)
2. **Fix 9.2:** Zero-gradient guard for gradient-based mode
3. **Fix 9.3:** Per-sample initial state anchor (moved from batch[0] global to per-sample)

The core SLSQP solver, constraint formulation (`A`, `b`, `C`, `d` matrices), dynamics constraints (`deriv` type), and bounds constraints are **structurally identical** to the original DPCC.

---

### Why Bypassing Policy Does NOT Bypass DPCC

```
DPCC = Projector injected into GaussianDiffusion.p_sample_loop()
Policy = convenience wrapper that formats obs + forwards projector to model

Bypassing Policy ≠ Bypassing DPCC
```

The visual pipeline replaces `Policy` with `VisualAgentWrapper` which:
1. Handles image+state conditioning (Policy can't do this)
2. Calls `self.model(cond, projector=self.projector)` — **same call as Policy L52**
3. The model's `p_sample_loop` runs the **same DPCC projection hooks** at every denoising step

**`Policy` was never the DPCC engine. It was just the mailman. The visual pipeline hired a different mailman (`VisualAgentWrapper`) who delivers the same package (`projector`) to the same address (`model.forward()`).**

---

*Investigator: Antigravity (Claude Opus 4.6 Thinking)*
*2026-05-20*

