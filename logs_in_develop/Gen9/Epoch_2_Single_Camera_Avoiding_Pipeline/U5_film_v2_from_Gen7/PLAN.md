# U5 — Port FiLM v2 from Gen7 (Visual Aligning) to Visual Avoiding

**Source:** `fm_visual_aligning/` (Gen7, dual-cam, 128D latent)
**Target:** `fm_visual_avoiding/` (Gen9 E2, single-cam, 64D latent)
**Scope:** FM visual avoiding only (3 files); no retrain of existing v1 checkpoints

---

## What FiLM v2 Is

`v1` (current): visual latent is CONCATENATED with time embedding → additive bias only
`v2` (true FiLM): visual latent produces per-channel `γ` (scale) + `β` (shift) injected
inside every ResidualTemporalBlock. Zero-initialised → identity at step 0, stable training.

```
v1 per block:  out = Conv(x) + time_mlp([t ‖ visual])
v2 per block:  out = (1 + γ(visual)) · (Conv(x) + time_mlp(t)) + β(visual)
```

---

## 3 Files to Change

### 1. NEW `fm_visual_avoiding/models/unet1d_temporal_film.py`

Copy from `fm_visual_aligning/models/unet1d_temporal_film.py`.
One import fix — change the helpers import to the avoiding package:

```python
# before (aligning):
from .helpers import (SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock)

# after (avoiding) — same line, same helpers.py exists in fm_visual_avoiding:
from .helpers import (SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock)
```
Actually identical — `from .helpers import ...` is relative, so no change needed.
Just copy the file verbatim.

### 2. EDIT `fm_visual_avoiding/models/visual_unet.py`

Add `film_mode` branching (same pattern as aligning `visual_unet.py` lines 61-94).
Key difference: `cond_dim=64` (single-cam, avoiding) vs `128` (dual-cam, aligning).

In `__init__`, after the existing `backbone_kwargs` block:

```python
self.film_mode = getattr(config, 'film_mode', 'v1')

if self.film_mode == 'v2':
    from fm_visual_avoiding.models.unet1d_temporal_film import UNet1DTemporalFiLMModel
    self.backbone = UNet1DTemporalFiLMModel(**backbone_kwargs).to(self.device)
    print('[ VisualUNet ] film_mode=v2 — TRUE FiLM backbone ACTIVE')
else:
    self.backbone = UNet1DTemporalCondModel(**backbone_kwargs).to(self.device)
```

`backbone_kwargs` already contains `cond_dim=latent_dim` (=64 when `if_vision=True`),
so `UNet1DTemporalFiLMModel` receives `cond_dim=64` automatically — no extra change.

### 3. EDIT `config/avoiding-d3il-visual.py`

**a) Add `film_mode` to `args_to_watch_fm_visual_train`** (bakes into checkpoint folder name):
```python
args_to_watch_fm_visual_train = [
    ...
    ('batch_size', 'bs'),
    ('film_mode', 'film'),   # ← add this line
]
```

**b) Add `film_mode` to `args_to_watch_fm_visual_plan`**:
```python
args_to_watch_fm_visual_plan = [
    ...
    ('mpc_batch_size', 'mpc'),
    ('film_mode', 'film'),   # ← add this line
]
```

**c) Add `'film_mode': 'v1'` to the existing FM visual train block** (backward compat —
existing checkpoints were trained without this key, so they live under `..._bs8/` with
no `_film` suffix; adding `v1` as the new explicit default would rename the folder.
**Option**: leave existing block unchanged and add a NEW train block for v2):

```python
# existing block (leave untouched — no film suffix in path):
'fm_visual_avoiding': { ... }

# new block for v2 (separate folder, separate retrain):
'fm_visual_avoiding_filmv2': {
    ...same as fm_visual_avoiding...
    'film_mode': 'v2',
}
```

This avoids any naming collision with existing v1 checkpoints.

---

## Checkpoint Path Result

v1 (existing, unchanged): `logs/avoiding-d3il-visual/plans/fm_visual_avoiding/H8_K20_...`
v2 (new):                  `logs/avoiding-d3il-visual/plans/fm_visual_avoiding/H8_K20_..._filmv2/`

---

## What Does NOT Change

- `fm_visual_avoiding/models/visual_gaussian_diffusion.py` — untouched
- `fm_visual_avoiding/models/unet1d_temporal_cond.py` — untouched (v1 still works)
- All existing v1 checkpoints — loadable without any code change
- `diffuser_visual_avoiding/` (DPCC/ddpm blocks) — out of scope for this U5
