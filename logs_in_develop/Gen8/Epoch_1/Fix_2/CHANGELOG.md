# Gen8 Epoch 1 — Fix_2 Series: Missing `model_config.pkl`

**Date**: 2026-06-04
**Status**: ✅ Fixed (uncommitted)
**Parent**: [`../Fix_1/CHANGELOG.md`](../Fix_1/CHANGELOG.md)

---

## Fix_2 — Train script: engine not wrapped in `utils.Config`

**Triggered by**: `temp/debug_gen8/eval_outputs` — Slurm eval job 21175

### Symptom

```
FileNotFoundError: [...]/6/model_config.pkl
```

Training completed. Checkpoint directory existed but contained only `dataset_config.pkl`,
`diffusion_config.pkl`, `trainer_config.pkl`, `obs_normalizer.pkl`, `act_normalizer.pkl`,
`state_*.pt` — no `model_config.pkl`.

### Root cause

Gen7 wraps `VisualUNet` in `utils.Config(savepath='model_config.pkl')` before passing it to
the diffusion wrapper. Gen8 removed the standalone `VisualUNet` construction (Option A wires
it inside `iMFTrajectoryModel`) but also dropped the `utils.Config` wrapper entirely —
`iMeanFlowEngine` was instantiated directly, so no pkl was ever written.

The eval's `load_diffusion_with_override` loads four pkls in sequence and crashes on the first
missing one.

### Fix — `train_imf_visual_aligning.py`

```python
# Before (direct instantiation — no pkl):
engine = iMeanFlowEngine(state_dim=..., seq_len=..., ..., if_vision=_if_vision, vis_config=args)

# After (wrapped — writes model_config.pkl at Config construction time):
model_config = utils.Config(
    iMeanFlowEngine,
    savepath=(args.savepath, 'model_config.pkl'),
    state_dim=_transition_dim, seq_len=args.horizon,
    freq_dim=getattr(args, 'dim', 128),
    dropout_rate=getattr(args, 'condition_dropout', 0.1),
    device=args.device, if_vision=_if_vision, vis_config=args,
)
engine = model_config()
```

No behavioral change during training. Prevents the problem on all future training runs.

---

## Fix_2.2 — Eval fallback attempt via `args.json` (failed)

**Triggered by**: `temp/debug_gen8/outputs_2` — Slurm eval job 21191

### Symptom

Same `FileNotFoundError` despite Fix_2 being applied — because the **existing checkpoint**
(saved before Fix_2) still has no `model_config.pkl`. Fix_2 only helps future runs.

### Attempted fix

Added `_rebuild_engine_config_from_args(lp, device)` fallback in `load_diffusion_with_override`:
load `args.json` from the checkpoint dir, reconstruct the engine Config from those values.

### Why it failed

`args.json` is also absent. The parser saves it only when `experiment == 'train'` (literal
string) — but the training experiment name is `'imf_visual_aligning'`, never `'train'`.
So neither `model_config.pkl` nor `args.json` exist in the checkpoint.

---

## Fix_2.3 — Eval fallback via checkpoint path parsing (partial — wrong dim)

**Triggered by**: `temp/debug_gen8/outputs_2` — Slurm eval job 21192

### What was done

Replaced the `args.json` fallback with `_rebuild_engine_config_from_path(lp, device)`.
Parsed `H(\d+)` → `horizon=8` and `V(True|False)` → `if_vision=True` from the exp dir name.
Fallback ran successfully and wrote `model_config.pkl`.

### Why it still failed (Fix_2.4 symptom)

`dim=128` was hardcoded in the fallback. The training config has `'dim': 32` for all aligning
variants. `VisualUNet` backbone was reconstructed with 4× too many channels → `RuntimeError`
on `load_state_dict`:

```
size mismatch for model.model.velocity_net.backbone.time_mlp.1.weight:
  checkpoint shape torch.Size([128, 32])  ←  dim=32
  current model   torch.Size([512, 128])  ←  dim=128
```

`dim` is not encoded in the path, so path-parsing alone is insufficient.

---

## Fix_2.4 — Infer dim from checkpoint weights (partial — stale pkl not evicted)

**Triggered by**: `temp/debug_gen8/outputs_2` — Slurm eval job 21193

### What was done

Updated `_rebuild_engine_config_from_path` to infer `dim` from `time_mlp.1.weight` shape in
`state_*.pt` (`[4*dim, dim]` → `dim = shape[1]`). Falls back to `dim=32` if no file found.

### Why it still failed (Fix_2.5 symptom)

Fix_2.3 had already written `model_config.pkl` (dim=128) to the cluster checkpoint dir.
The `os.path.exists(_mc_path)` check found it → loaded the stale pkl → skipped the fallback
entirely. The dim-inference code in `_rebuild_engine_config_from_path` never ran.

---

## Fix_2.5 — Validate loaded pkl dim against checkpoint (final fix)

**Triggered by**: `temp/debug_gen8/output_2.4` — Slurm eval job 21195

### Root cause

The existence check is insufficient: a `model_config.pkl` can exist but be wrong.
Must validate the stored `vis_config.dim` against the checkpoint weight shape after loading.

### Fix — `eval_imf_visual_aligning.py` `load_diffusion_with_override`

After loading `model_config.pkl`, validate immediately:

```python
_ckpt_dim   = int(state['model.model.velocity_net.backbone.time_mlp.1.weight'].shape[1])
_stored_dim = int(getattr(model_config._dict.get('vis_config'), 'dim', -1))
if _stored_dim != _ckpt_dim:
    os.remove(_mc_path)   # evict stale pkl
    model_config = _rebuild_engine_config_from_path(lp, device=device)  # writes correct one
```

On mismatch: removes the stale pkl, calls `_rebuild_engine_config_from_path` which infers
`dim=32` from the checkpoint and writes the correct `model_config.pkl`. Next run finds a valid
pkl and the validation passes — no overhead.

**No retrain needed.**

---

## Files touched

```
M  imf_visual_aligning_test/train_imf_visual_aligning.py    (Fix_2:   wrap engine in utils.Config)
M  imf_visual_aligning_test/eval_imf_visual_aligning.py     (Fix_2.3: fallback rebuild from path)
```

---

## Verification

| Check | Result |
|---|---|
| AST parse: `train_imf_visual_aligning.py` | ✅ |
| `model_config.pkl` savepath in Config call (train) | ✅ |
| Save order: `model_config` before `diffusion_config` | ✅ |
| AST parse: `eval_imf_visual_aligning.py` | ✅ |
| Regex `H(\d+)` + `V(True\|False)` on actual path | ✅ verified |
| `dim` inferred from `time_mlp.1.weight` shape in state_*.pt | ✅ |
| Falls back to `dim=32` if no checkpoint present | ✅ |
| Loaded pkl dim validated against checkpoint — stale pkl evicted on mismatch | ✅ |

**Cluster expectation**: validation detects dim=128 vs dim=32 mismatch → removes stale pkl → rebuilds with dim=32 → loads cleanly.

---

## Lesson

When porting a train script across architectures, audit every `utils.load_config(...)` call
in the eval script against every `utils.Config(..., savepath='xxx.pkl')` call in the train
script. They must be 1-to-1.

---

## Cross-references

| Document | Content |
|---|---|
| [`../Fix_1/CHANGELOG.md`](../Fix_1/CHANGELOG.md) | Previous fix (UNet name mismatch + FlowMatchingODE alias) |
| `eval_imf_visual_aligning.py:1656` | `_rebuild_engine_config_from_path` + `load_diffusion_with_override` |
| `fm_visual_aligning_test/train_fm_visual_aligning.py:214` | Gen7 reference: `utils.Config(VisualUNet, savepath='model_config.pkl')` |
