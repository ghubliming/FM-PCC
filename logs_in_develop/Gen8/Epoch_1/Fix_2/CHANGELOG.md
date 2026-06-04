# Gen8 Epoch 1 — Fix_2: Missing `model_config.pkl` (Engine Not Wrapped in `utils.Config`)

**Date**: 2026-06-04
**Status**: ✅ Fixed (uncommitted)
**Triggered by**: `temp/debug_gen8/eval_outputs` — Slurm eval job 21175
**Parent**: [`../Fix_1/CHANGELOG.md`](../Fix_1/CHANGELOG.md)

---

## 1. Symptom

Eval job 21175 crashed immediately after printing the load path:

```
[ eval loading ] Loading from logs/aligning-d3il-visual/imf_visual_aligning/
    H8_Dimf_visual_aligning.models.visual_imf_diffusion.VisualIMF_a1.5_b1.0_aw1_VTrue_steps1000_bs64/6

Traceback (most recent call last):
  File "eval_imf_visual_aligning.py", line 1707, in <module>
    exp = load_diffusion_with_override(...)
  File "eval_imf_visual_aligning.py", line 1660, in load_diffusion_with_override
    model_config = utils.load_config(*loadpath, 'model_config.pkl')
FileNotFoundError: [...]/6/model_config.pkl
```

Training had completed successfully. The checkpoint directory existed but contained only:
`dataset_config.pkl`, `diffusion_config.pkl`, `trainer_config.pkl`, `obs_normalizer.pkl`, `act_normalizer.pkl`, and `state_*.pt`.

`model_config.pkl` was absent.

---

## 2. Root cause — Gen8 engine not wrapped in `utils.Config`

**Gen7 pattern** (`train_fm_visual_aligning.py`):
```python
model_config = utils.Config(
    VisualUNet,
    savepath=(args.savepath, 'model_config.pkl'),  # ← writes model_config.pkl
    config=args,
)
model = model_config()
diffusion_config = utils.Config(VisualFlowMatching, savepath=(args.savepath, 'diffusion_config.pkl'), ...)
diffusion = diffusion_config(model)
```

**Gen8 original pattern** (`train_imf_visual_aligning.py`):
```python
engine = iMeanFlowEngine(  # ← direct instantiation — no utils.Config, no .pkl written
    state_dim=_transition_dim, seq_len=args.horizon, ...,
    if_vision=_if_vision, vis_config=args,
)
diffusion = diffusion_config(engine)
```

Gen8 was written with Option A (engine wired inside `iMFTrajectoryModel`), and the standalone
`VisualUNet` construction was removed. But the `utils.Config` wrapper that saved `model_config.pkl`
was also removed. The eval script's `load_diffusion_with_override` always loads all four pkl files
in sequence — `dataset_config`, `model_config`, `diffusion_config`, `trainer_config` — and crashes
on the first missing one.

---

## 3. Fix

**File**: `imf_visual_aligning_test/train_imf_visual_aligning.py` — §2 Engine block

Wrapped `iMeanFlowEngine` instantiation in `utils.Config` with `savepath=(args.savepath, 'model_config.pkl')`:

```python
# Before:
engine = iMeanFlowEngine(
    state_dim=_transition_dim, seq_len=args.horizon, freq_dim=...,
    dropout_rate=..., device=args.device, if_vision=_if_vision, vis_config=args,
)

# After:
model_config = utils.Config(
    iMeanFlowEngine,
    savepath=(args.savepath, 'model_config.pkl'),
    state_dim=_transition_dim,
    seq_len=args.horizon,
    freq_dim=getattr(args, 'dim', 128),
    dropout_rate=getattr(args, 'condition_dropout', 0.1),
    device=args.device,
    if_vision=_if_vision,
    vis_config=args,
)
engine = model_config()
```

The `utils.Config` call writes `model_config.pkl` at construction time (before `model_config()` is called). The resulting `engine` object is identical to the previous direct instantiation — no behavioral change during training.

At eval time, `load_diffusion_with_override` now finds `model_config.pkl` and reconstructs:
```python
engine    = model_config()            # iMeanFlowEngine(...)
diffusion = diffusion_config(engine)  # VisualIMF(engine, ...params...)
trainer   = trainer_config(diffusion_model=diffusion, dataset=dataset)
trainer.load(epoch)                   # loads state_<step>.pt
```

---

## 3b. Eval fallback for existing pre-Fix_2 checkpoints

The training fix (§3) prevents the problem going forward but does not help the **already-saved
checkpoint** (job from before Fix_2 was applied). That checkpoint directory exists but has no
`model_config.pkl`.

**Additional fix — `eval_imf_visual_aligning.py`**: added `_rebuild_engine_config_from_args(lp, device)` helper and a conditional in `load_diffusion_with_override`:

```python
_mc_path = os.path.join(lp, 'model_config.pkl')
if os.path.exists(_mc_path):
    model_config = utils.load_config(*loadpath, 'model_config.pkl')
else:
    print('[ eval ] model_config.pkl missing (pre-Fix_2 checkpoint) — rebuilding from args.json')
    model_config = _rebuild_engine_config_from_args(lp, device=device)
```

`_rebuild_engine_config_from_args` loads `args.json` (written by the parser at training time),
reconstructs `utils.Config(iMeanFlowEngine, ...)` with the same kwargs the train script would
have used, **writes `model_config.pkl`** to the checkpoint dir for future runs, and returns the Config.

This means: no retrain needed. The first eval run auto-heals the checkpoint; subsequent runs use the pkl directly.

---

## 4. Why training itself was unaffected

Training does not use `model_config.pkl` at all — it only reads the returned Python object.
The missing file only matters at eval load time. Training completed and wrote a valid checkpoint
(`state_*.pt` files), but eval could not reload the architecture to wrap around those weights.

---

## 5. Verification

| Check | Result |
|---|---|
| AST parse: `train_imf_visual_aligning.py` | ✅ |
| `model_config.pkl` savepath present in Config call | ✅ |
| Save order: `model_config` before `diffusion_config` | ✅ (lines 227 vs 246) |
| `engine = model_config()` produces same object as direct call | ✅ (same kwargs) |
| AST parse: `eval_imf_visual_aligning.py` | ✅ |
| `_rebuild_engine_config_from_args` fallback added | ✅ |
| Fallback writes pkl so second run uses fast path | ✅ |
| Eval `load_diffusion_with_override` load order matches | ✅ dataset → model → diffusion → trainer |

**Cluster-side expectation**: eval re-run on existing checkpoint will trigger the `args.json` fallback, write `model_config.pkl`, and proceed to model construction. No retrain needed.

---

## 6. Why Phase-0 and Fix_1 checks missed this

Fix_1's `__init__.py` import audit verified that all symbol names resolved correctly — but it only checked *names*, not whether the training script actually *called* `utils.Config` for every loadable component. A simple grep for `utils.Config.*model_config` on the training script would have caught this.

**Lesson for future ports**: when copying a train script across architectures, audit every `utils.load_config(...)` call in the eval script against every `utils.Config(..., savepath=..., 'xxx.pkl')` call in the train script. They must be 1-to-1.

---

## 7. Files touched

```
M  imf_visual_aligning_test/train_imf_visual_aligning.py    (wrap engine in utils.Config)
M  imf_visual_aligning_test/eval_imf_visual_aligning.py     (fallback: rebuild model_config from args.json)
```

---

## 8. Cross-references

| Document | Content |
|---|---|
| [`../Fix_1/CHANGELOG.md`](../Fix_1/CHANGELOG.md) | Previous fix (UNet name mismatch + FlowMatchingODE alias) |
| `imf_visual_aligning_test/eval_imf_visual_aligning.py:1656` | `load_diffusion_with_override` — loads all 4 pkl files |
| `fm_visual_aligning_test/train_fm_visual_aligning.py:214` | Gen7 reference pattern: `utils.Config(VisualUNet, savepath='model_config.pkl')` |
