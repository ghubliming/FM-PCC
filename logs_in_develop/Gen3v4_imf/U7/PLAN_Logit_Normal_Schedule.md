# U7 — Logit-Normal Time Scheduling for iMF

**Date:** 2026-06-28  
**Lineage:** Gen3v4 iMF → U3 (mean-flow) → U4 (MeanFlow-JVP + CFG) → U5 (interval-CFG per-sample) → U6 (DiT backbone) → **U7 (logit-normal schedule)**

---

## Why this change exists

### The investigation that triggered U7

Two conflicting claims about what iMF uses for training time `t` sampling:
1. **"Uniform"** — `t ~ Uniform(0, 1)`
2. **"Logit-Normal"** — `t = sigmoid(randn * 1.0 - 0.4)`

**Research result: both claims are partially wrong/right.**

| Source | Schedule | Implementation |
|---|---|---|
| Reference JAX iMF (`/workspaces/imeanflow/imf.py` L120-124) | **Logit-Normal** | `t = nn.sigmoid(rnd_normal * P_std + P_mean)`, `P_mean=-0.4, P_std=1.0` |
| Our FM-PCC (`flow_matcher_v3_imeanflow/models/imf_diffusion.py` L313-316) | **Beta distribution** | `t = 1 - Beta(alpha=1.5, beta=1.0)` |
| Beta(1,1) special case (one config in `avoiding-d3il.py:538`) | **Uniform** | `Beta(1,1) = Uniform(0,1)` — comment "uniform for real iMF" was wrong |

**Bottom line:** We are using Beta distribution, not logit-normal. The reference JAX iMF is logit-normal.

---

## Distribution shape comparison

```
Logit-Normal (reference iMF):         Our Beta(1.5, 1.0):
                                        
  PDF                                   PDF
  ↑   bell-shaped                       ↑   monotone-decreasing
  |       ╭─╮                           |  ╲
  |      ╱   ╲                          |   ╲
  |    ╱       ╲                        |    ╲
  |  ╱           ╲                      |     ╲___
  +──────────────→ t                    +──────────────→ t
  0    0.4      1.0                     0             1.0
  
  median  = sigmoid(-0.4) ≈ 0.40        mean = 0.40
  shape   = bell (concentrates at 0.4)  shape = decaying (no peak)
  from paper:  yes (canonical)          custom approximation
```

Both have mean ≈ 0.40, but **shape is fundamentally different**:
- Logit-normal: bell-shaped → **concentrates training at the difficult intermediate t values** (principled: matching EDM/iMF loss weighting)
- `1 - Beta(1.5, 1.0)`: monotone decreasing → more training near t=0 (noise), less near t=1 (data), no concentration at the "hard" intermediate range

---

## Why logit-normal is better (motivation)

1. **Matches the paper / reference implementation exactly.** The iMF paper and its official JAX codebase use `sigmoid(N(P_mean, P_std))`. Our Beta was a custom approximation with the same mean but wrong shape.

2. **Bell-shape concentrates training budget at the right place.** Near t=0 (noise), the model output is almost pure noise anyway — loss is high but gradients are uninformative. Near t=1 (data), the problem is near-trivial. The intermediate range (t≈0.3–0.6) is where the mean-flow field is hardest to learn. Logit-normal naturally focuses more samples there.

3. **`P_mean` and `P_std` are tunable hyperparameters.** The bell can be shifted (`P_mean`) or widened/sharpened (`P_std`) — makes schedule an explicit design choice rather than an accident of Beta parameterization.

4. **Eliminates confusion.** The `time_beta_alpha_v3` / `time_beta_beta_v3` config keys have accumulated incorrect comments ("uniform for real iMF"). Switching to `p_mean` / `p_std` makes the intent explicit and matches the paper nomenclature.

---

## Status

- [x] Step 1: modify `flow_matcher_v3_imeanflow/models/imf_diffusion.py`
- [x] Step 2: update `config/avoiding-d3il.py` + `config/aligning-d3il-visual.py`
- [x] Step 3: mirror changes to `imf_visual_aligning/models/imf_diffusion.py`
- [ ] Step 4: cluster re-train + A/B eval (USER)

**Implementation note (vs original plan):** kept `time_beta_alpha_v3`/`time_beta_beta_v3` params in place — no path restructuring. Simply added logit-normal as a 3rd branch alongside the existing Beta path. `t_schedule` key added to checkpoint paths so old Beta checkpoints never collide with new logit-normal ones. See `CHANGELOG_U7.md` for exact touched lines.

---

## Scope of change

### Affected files (minimal surface — only scheduling logic)

| File | Change |
|---|---|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | Add `t_schedule`, `p_mean`, `p_std` params; add logit-normal branch in `loss()` |
| `imf_visual_aligning/models/imf_diffusion.py` | Same (parallel copy) |
| `config/avoiding-d3il.py` | Add `t_schedule`, `p_mean`, `p_std` to iMF block; keep Beta params for backward compat |
| `config/aligning-d3il-visual.py` | Same for iMF block |

### NOT changing

- `_p_losses_meanflow_jvp()` — the JVP path still receives `t` as argument; schedule is orthogonal
- `sample_tr()` / `p_losses()` internals — only the `t` sampling line changes
- Eval code, projector, backbone — entirely orthogonal
- Config checkpoint path format — `p_mean` / `p_std` added to path key so old checkpoints remain accessible under old keys

---

## Implementation plan

### Step 1 — `imf_diffusion.py`: add logit-normal schedule

In `__init__`, add:
```python
t_schedule: str = "logit_normal",   # 'logit_normal' | 'beta' | 'uniform'
p_mean: float = -0.4,               # logit-normal: mean in logit space (iMF default)
p_std: float = 1.0,                 # logit-normal: std in logit space (iMF default)
```
Store as `self.t_schedule`, `self.p_mean`, `self.p_std`.

In `loss()`, replace the current Beta block:
```python
# BEFORE (U6):
alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
beta_dist = torch.distributions.Beta(alpha, beta)
t = 1.0 - beta_dist.sample((batch_size,))

# AFTER (U7):
if self.t_schedule == 'logit_normal':
    # Canonical iMF schedule (reference: /workspaces/imeanflow/imf.py L120-124)
    rnd = torch.randn(batch_size, device=x.device) * self.p_std + self.p_mean
    t = torch.sigmoid(rnd)
elif self.t_schedule == 'uniform':
    t = torch.rand(batch_size, device=x.device)
else:  # 'beta' — backward-compatible default
    alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
    beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
    beta_dist = torch.distributions.Beta(alpha, beta)
    t = 1.0 - beta_dist.sample((batch_size,))
```

**Default for new runs:** `t_schedule='logit_normal'`.  
**Default for backward compat:** keep `t_schedule='beta'` if old checkpoints exist (controlled by config).

### Step 2 — Config: add schedule keys

In `config/avoiding-d3il.py` (iMF block) and `config/aligning-d3il-visual.py`:
```python
# U7: logit-normal schedule (canonical iMF, matches reference imf.py)
't_schedule': 'logit_normal',   # 'logit_normal' | 'beta' | 'uniform'
'p_mean':     -0.4,             # logit-normal P_mean (iMF default)
'p_std':      1.0,              # logit-normal P_std (iMF default)
```

Update `args_to_watch` and `prefix`/`diffusion_loadpath` to include `t_schedule`:
```python
('t_schedule', 'ts'),
# path: ...H{horizon}_D{diffusion}_ts{t_schedule}_aw{action_weight}...
```

Old checkpoints trained with Beta stay accessible via `t_schedule=beta` in the path — no collision.

### Step 3 — Duplicate into `imf_visual_aligning/models/imf_diffusion.py`

The visual-aligning iMF is a parallel copy. Apply the identical change so both are in sync.

### Step 4 — Verify on cluster (USER task)

1. Retrain one scene with `t_schedule=logit_normal` vs current `t_schedule=beta`.
2. Compare: eval success rate, loss curve shape (expect faster initial convergence), trajectory smoothness.
3. The logit-normal bell-shaped distribution should produce more stable intermediate-t gradients.

---

## Config path key strategy

Current key pattern (U6 and prior):
```
H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}_obj{imf_objective}_bb{imf_backbone}
```

U7 new pattern:
```
H{horizon}_D{diffusion}_ts{t_schedule}_aw{action_weight}_obj{imf_objective}_bb{imf_backbone}
```

The `a`/`b` keys are implicit in `ts=beta` (config still has the numeric values). For logit-normal, `p_mean` and `p_std` are implicit in defaults (no checkpoint bloat for the standard -0.4/1.0 case). If you experiment with non-default `p_mean`/`p_std`, extend the path with `_pm{p_mean}_ps{p_std}`.

---

## Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Logit-normal may perform similarly to Beta (same mean) | Low | Easy A/B test; if no gain, it still documents correct alignment with paper |
| Old Beta checkpoints incompatible with new default | None | `t_schedule` in path key; set `t_schedule=beta` to resume old runs |
| Change to `imf_visual_aligning` breaks incomplete source | None | iMF visual-aligning is incomplete per user (2026-06-28) — changes are additive |
| JVP path (`meanflow_jvp`) not updated | None | JVP path receives `t` as argument, schedule happens before it — orthogonal |

---

## Status

- [ ] Step 1: modify `flow_matcher_v3_imeanflow/models/imf_diffusion.py`
- [ ] Step 2: update `config/avoiding-d3il.py` + `config/aligning-d3il-visual.py`
- [ ] Step 3: mirror changes to `imf_visual_aligning/models/imf_diffusion.py`
- [ ] Step 4: cluster re-train + A/B eval (USER)
