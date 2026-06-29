# U5 Changelog — FiLM v2 Port from Gen7 to Visual Avoiding

**Scope:** additive opt-in; all existing v1 code paths and checkpoints unchanged
**Requires retrain:** yes — new `fm_visual_avoiding_filmv2` config block

---

## Files Changed

### NEW `fm_visual_avoiding/models/unet1d_temporal_film.py`

Copied verbatim from `fm_visual_aligning/models/unet1d_temporal_film.py`.
Relative imports (`from .helpers import ...`) work unchanged since both packages
share the same `helpers.py` structure.

True FiLM per-block: `out = (1 + γ(v)) · (Conv(x) + time_mlp(t)) + β(v)`
γ/β zero-initialised → identity at step 0, stable early training.

### EDIT `fm_visual_avoiding/models/visual_unet.py`

Added `film_mode` branching in `__init__`. When `film_mode='v2'`, constructs
`UNet1DTemporalFiLMModel` instead of `UNet1DTemporalCondModel` using the same
`backbone_kwargs` dict — `cond_dim=64` (single-cam, 64D latent) flows in automatically.
Default `film_mode='v1'` → existing behaviour byte-identical.

### EDIT `config/avoiding-d3il-visual.py`

**`args_to_watch_fm_visual_train`** — added `('film_mode', 'film')` as last entry.
`watch()` skips missing keys → existing v1 blocks (no `film_mode` key) produce
unchanged folder names.

**`args_to_watch_fm_visual_plan`** — same addition.

**NEW block `fm_visual_avoiding_filmv2`** — identical to `fm_visual_avoiding` + `'film_mode': 'v2'`.
Checkpoint path: `fm_visual_avoiding/H8_D..._bs64_filmv2/`

**NEW block `plan_fm_visual_avoiding_filmv2`** — identical to `plan_fm_visual_avoiding` + `'film_mode': 'v2'`.
`diffusion_loadpath` points to the v2 train folder (`..._film{film_mode}`).

---

## Checkpoint Paths

| config block | path suffix |
|---|---|
| `fm_visual_avoiding` (existing) | `..._bs64/` — unchanged |
| `fm_visual_avoiding_filmv2` (new) | `..._bs64_filmv2/` |

---

## Usage

```bash
# train v2
python train_visual_avoiding.py --config fm_visual_avoiding_filmv2 --seed 0

# eval v2
python eval_visual_avoiding.py --config plan_fm_visual_avoiding_filmv2 --seed 0
```
