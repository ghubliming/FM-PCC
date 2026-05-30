# Fix-18 — Severity, Scope, and Retrain Impact

**Date**: 2026-05-30
**Status**: ⚠️ This MD has been **rewritten twice** as new evidence
emerged. The current version is the corrected understanding after the
user shared the actual `model_config.pkl` from the FM `_VFalse_`
checkpoint.

---

## TL;DR (corrected)

| Question | Answer |
|---|---|
| Is this a severe fix? | **Yes for FM eval (Fix B), low for FM/DPCC train (Fix A).** Fix A is a latent bug nobody had actually hit yet because no one had successfully trained a *genuine* 23-D non-visual model. Fix B is hit immediately by any eval of a `_VFalse_` checkpoint with UF-13 active. |
| Was the old non-visual *code* wrong? | The model-construction path was wrong for genuine 23-D non-visual mode. But **no genuine 23-D non-visual checkpoint actually exists** on disk — see §2. |
| Is this specific to `K=1` / `ODE=1`? | **No.** Step count is irrelevant. The bug would fire for any non-visual training attempt that goes through the 23-D dataset path. |
| Do I need to retrain anything? | **No retrain required for any existing checkpoint.** All existing `_VFalse_` checkpoints are actually 9-D visual models with a misleading flag name — see §2. Fix A is for *future* genuine non-visual training, which has never been run before. |

The earlier MD versions over-claimed severity and gave wrong retrain
guidance. The actual picture is: **the existing FM `_VFalse_` checkpoint
is a 9-D visual model wearing a non-visual nameplate. Keep using it as
a visual checkpoint. There is no real 23-D non-visual model to lose.**

---

## 1. What the user's model_config tells us

From the FM checkpoint loaded at eval time
(`logs/.../fm_visual_aligning/H8_..._VFalse_steps900_bs64/6/model_config.pkl`):

```python
action_dim=3, obs_dim=6, if_vision=False
```

Trace `VisualUNet.__init__` (`fm_visual_aligning/models/visual_unet.py:70-74`):

```python
if self.if_vision:            # False → take else branch
    transition_dim = self.TRANSITION_DIM   # 9 (skipped)
else:
    obs_dim = getattr(config, 'obs_dim', 20)   # reads 6
    transition_dim = config.action_dim + obs_dim   # 3 + 6 = 9
```

So the **first conv weights in this checkpoint expect 9-channel input**, not
23. The model is structurally a 9-D model. The `if_vision=False` flag is
cosmetic; it influenced things downstream (UF-13 record-mode auto-enables
visual rendering anyway) but it did NOT cause the model to be built for 23-D.

Training did not crash because the dataset code at the time of that training
run was **also** producing 9-D trajectories. Either:

- UF-17's `StateOnlyAligningDataset` branch wasn't reached (UF-17 commit was
  `19952b7`; the checkpoint may predate it), or
- The training script's `if_vision` branch on the dataset side happened to
  load 9-D data for some other reason.

Either way: **9-D model + 9-D data = successful but non-visual-in-name-only
training.**

---

## 2. Severity by code path (corrected)

### 🟢 Visual path (DPCC + FM, `if_vision=True`)
- Not affected. Never was.

### 🟢 Existing FM `_VFalse_` checkpoint
- **Effectively a 9-D visual model.** The `_VFalse_` in the path comes from
  the args templating `_V{if_vision}_`, not from any structural difference
  in the trained weights.
- **No retrain needed.** Treat as a visual checkpoint that happens to log
  `if_vision=False`. Train-time `obs_dim=6` confirms 9-D structure.
- Fix B is what makes this checkpoint **evaluable** with current eval
  scripts (without it, the projector dim derivation crashes because args
  say `False` but the actual model is 9-D).

### 🟡 DPCC non-visual via the `ddpm_encdec_vision_nonvisual` variant
- Variant block exists in config (`obs_dim=20, if_vision=False`).
- **BUT** no train script invokes this variant — every train script
  hardcodes `experiment='visual_aligning_dpcc'` or `'fm_visual_aligning'`
  (see `train_visual_aligning_dpcc.py:143`, `train_fm_visual_aligning.py`).
- So the variant is **effectively dead code**. It was never exercised.
- Fix A is what enables it (or any other genuine 23-D path) to actually
  produce a valid model when invoked.

### 🔴 *Genuine* 23-D non-visual training (whether for DPCC or FM)
- This path has **never produced a working checkpoint**. Before Fix A,
  attempting it produces the iteration-0 first-conv crash.
- The user's recent K=1 run was the first attempt at a genuine 23-D path
  (because `if_vision=False` in the visual variant + the 23-D dataset
  branch finally lined up post-UF-17). That attempt hit the bug.
- **No legacy checkpoints to migrate** — there's nothing to retrain
  because there's nothing that ever finished.

### 🟡 FM/DPCC eval with UF-13 record mode active
- Fix B is required. Pre-fix, the projector dim derivation reads
  `args.if_vision` which UF-13 can flip silently, producing the `(23,) (9,)`
  broadcast crash. Post-fix, the derivation reads the saved normalizers
  (immutable ground truth).
- Affects evaluation only, not the checkpoint itself.

---

## 3. Do I need to retrain?

| Scenario | Retrain? | Why |
|---|---|---|
| Any visual checkpoint (`_VTrue_`, any K, any ODE step count) | ❌ No | Path was never broken |
| FM checkpoint with `_VFalse_` in path | ❌ **No** | It's a 9-D visual model with a misleading flag (see §1). Keep using it as visual. Just need Fix B for eval to work. |
| DPCC checkpoint with `_VFalse_` in path | ❌ **No (probably)** | Same reasoning — but check `model_config.pkl` to confirm `obs_dim=6` (i.e. 9-D). If `obs_dim=20`, it's a genuine 23-D model and was somehow trained successfully (would require investigation). |
| Genuine 23-D non-visual checkpoint | N/A — none exists | Train fresh with Fix A patched if you want one |

### Quick verification (run on cluster)

```bash
python - <<'PY'
import pickle
# Edit the path to point at any suspected non-visual checkpoint dir
ck = 'logs/aligning-d3il-visual/fm_visual_aligning/H8_..._VFalse_steps900_bs64/6'
cfg = pickle.load(open(f'{ck}/model_config.pkl','rb'))
ns  = cfg['config']
print(f'obs_dim   = {ns.obs_dim}     # 6 → 9-D visual model (no retrain). 20 → genuine 23-D non-visual.')
print(f'action_dim= {ns.action_dim}')
print(f'if_vision = {ns.if_vision}   # cosmetic only; not load-bearing for model structure')
PY
```

**Rule of thumb:** structural identity of a checkpoint is `(obs_dim,
action_dim)`, NOT `if_vision`. If `obs_dim=6 and action_dim=3`, it's a 9-D
visual-style model regardless of flag.

---

## 4. What Fix-18 actually buys you

1. **Fix B unblocks evaluation of existing `_VFalse_` checkpoints.** Before
   Fix B, eval would crash mid-run on the projector dim mismatch. After
   Fix B, the projector dim follows the saved normalizers (which are 9-D
   for any existing checkpoint), so projection works.
2. **Fix A makes future genuine 23-D non-visual training possible.** Before
   Fix A, attempting non-visual training (via any invocation path) would
   crash at iteration 0 because the model was built 9-D while data was
   23-D. After Fix A, the model is built 23-D when the dataset is 23-D, so
   training can complete.
3. **No legacy capability was lost.** Visual path remains identical.

---

## 5. Why earlier MD versions had wrong severity claims

- **v1**: claimed this was "documentation-only" and "no code changes
  needed." Wrong — code changes are needed for both eval (Fix B) and
  future non-visual training (Fix A).
- **v2**: claimed "non-visual DPCC training always crashed." Partially
  wrong — implied that genuine non-visual training had been happening and
  was now broken. Actually nobody had ever successfully trained a genuine
  23-D model.
- **v3 (this one)**: corrected after user shared the actual `obs_dim=6`
  from the `_VFalse_` model_config. The `_VFalse_` checkpoint is not a
  non-visual model; it's a visual model with a misleading flag. The
  severity is therefore: **Fix B unblocks current eval, Fix A unblocks
  future capability**, no legacy retrain is needed.

---

## 6. Bottom line

- **Use the existing FM `_VFalse_` checkpoint as a visual checkpoint.** No
  retraining required.
- **Don't infer model dimensionality from the `_VFalse_` in the path** —
  always check `obs_dim` in `model_config.pkl`.
- **Fix B is needed now** to eval the existing checkpoint with current
  scripts.
- **Fix A is needed if/when you want to train a genuine 23-D non-visual
  model** (DPCC or FM). That experiment has never been run before in this
  repo; Fix A is what makes it possible, not what fixes a regression.
- **One-shot K=1 / ODE=1 is unrelated** to either fix. Step count and
  trajectory dimensionality are orthogonal concerns; your one-shot attempt
  just happened to surface a latent bug in a path you hadn't tried before.
