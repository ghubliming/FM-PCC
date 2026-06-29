# U5 Changelog — FiLM v2 Port from Gen7 to Visual Avoiding

**Scope:** additive opt-in; same single block, just change `film_mode` value
**Requires retrain:** yes for v2 (separate checkpoint folder via watch())

---

## Files Changed

### NEW `fm_visual_avoiding/models/unet1d_temporal_film.py`

Copied verbatim from `fm_visual_aligning/models/unet1d_temporal_film.py`.

### EDIT `fm_visual_avoiding/models/visual_unet.py`

Added `film_mode` branching — same pattern as Gen7 `fm_visual_aligning/models/visual_unet.py`:
- `film_mode='v1'` (default) → `UNet1DTemporalCondModel` — unchanged behaviour
- `film_mode='v2'` → `UNet1DTemporalFiLMModel` — true FiLM per-block γ/β

### EDIT `config/avoiding-d3il-visual.py`

- `args_to_watch_fm_visual_train`: added `('film_mode', 'film')` → path auto-discriminates
- `args_to_watch_fm_visual_plan`: same
- `fm_visual_avoiding` train block: added `'film_mode': 'v1'`
- `plan_fm_visual_avoiding` plan block: added `'film_mode': 'v1'`, updated `diffusion_loadpath` to include `_film{film_mode}`

---

## Usage — Exactly Like Gen7

**To train v1 (current):** nothing changes, `film_mode: 'v1'` is already set.

**To train v2:** change ONE line in `fm_visual_avoiding` block:
```python
'film_mode': 'v2',
```
And ONE line in `plan_fm_visual_avoiding` block:
```python
'film_mode': 'v2',
```
That's it. Checkpoint path auto-becomes `..._filmv2/`.
