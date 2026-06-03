# Gen9 Epoch 2 — Fix-6: Camera Resolution Mismatch (1024×1024 vs 96×96)

**Date**: 2026-06-03
**Status**: ✅ Fixed (uncommitted)
**Triggered by**: `temp/debug_Gen9E2/outputs_5` — Slurm job 21161
**Parent**: [`../Fix_5/CHANGELOG.md`](../Fix_5/CHANGELOG.md)

---

## 1. Symptom

Job 21161 reached `Context 0 Rollout 0` and crashed on the first `agent.predict()` call:

```
File "fm_visual_avoiding/models/visual_unet.py", line 113, in forward
    visual_cond = self.encode_visual(bp_imgs)
File "fm_visual_avoiding/models/visual_unet.py", line 103, in encode_visual
    features = self.obs_encoder(obs_dict)
File "d3il/agents/models/vision/multi_image_obs_encoder.py", line 162, in forward
    assert img.shape[1:] == self.key_shape_map[key]
AssertionError
```

The `MultiImageObsEncoder` asserts that incoming images match the shape it was initialized with.

---

## 2. Root cause

`BPCageCam` (the bp_cam attached to `ObstacleAvoidanceEnv`) is initialized with default resolution:
```python
class BPCageCam(MjCamera):
    def __init__(self, width: int = 1024, height: int = 1024, ...):
```

So `env.bp_cam.get_image(depth=False)` returns a `(1024, 1024, 3)` array.

After `bp_img_raw[:, :, ::-1].transpose((2, 0, 1)) / 255.` this becomes `(3, 1024, 1024)`.

But `MultiImageObsEncoder` was initialized with:
```python
shape_meta = {'obs': {'agentview_image': {'shape': [3, 96, 96], 'type': 'rgb'}}}
```

→ expects `(3, 96, 96)`. The assertion fires at shape mismatch.

**Training resolution verified**: `ParityAvoidingDataset._load_images()` loads images as:
```python
img = cv2.imread(p)                                          # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float32)  # → RGB
frames.append(torch.from_numpy(img.transpose(2, 0, 1)))     # (3, H, W)
# stored shape: (3, 96, 96) — from zeros(0, 3, 96, 96) fallback
```

Images on disk are 96×96. The camera produces 1024×1024. No resize was applied at eval time.

**Channel order is correct**: `Avoiding_Sim.eval_agent` does `bp_img_raw[:, :, ::-1]` (BGR→RGB) matching `ParityAvoidingDataset`'s `cvtColor(BGR2RGB)`. Only resolution needed fixing.

---

## 3. Fix

`d3il/simulation/avoiding_sim.py`:

Added `import cv2` and constant `_IMG_W = _IMG_H = 96`.

Applied `cv2.resize(..., (_IMG_W, _IMG_H), cv2.INTER_AREA)` immediately after both
`env.bp_cam.get_image(depth=False)` calls (visual path + non-visual GIF capture path):

```python
# Before:
bp_img_raw = env.bp_cam.get_image(depth=False)             # (1024, 1024, 3)
bp_image = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)).copy() / 255.

# After:
bp_img_raw = env.bp_cam.get_image(depth=False)             # (1024, 1024, 3)
bp_img_raw = cv2.resize(bp_img_raw, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA)
bp_image = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)).copy() / 255.   # (3, 96, 96) RGB
```

`INTER_AREA` is the correct downsampling interpolation (averages pixel blocks; avoids
aliasing artifacts that `INTER_LINEAR`/`INTER_NEAREST` introduce at 10× downscale).

---

## 4. File touched

```
M  d3il/simulation/avoiding_sim.py   (import cv2 + _IMG_W/H constant + 2× resize lines)
```

---

## 5. Verification

| Check | Result |
|---|---|
| AST parse | ✅ |
| Both `get_image` calls followed by `cv2.resize` | ✅ |
| Resize target `(96, 96)` matches `shape_meta` `[3, 96, 96]` | ✅ |
| Channel order: `[::-1]` (BGR→RGB) matches `ParityAvoidingDataset.cvtColor(BGR2RGB)` | ✅ |

**Cluster-side expectation**: `encode_visual` receives `(B, T, 3, 96, 96)` tensors; the
`MultiImageObsEncoder` assertion passes; the model produces a trajectory; rollout completes.

---

## 6. Cross-references

| Document | Content |
|---|---|
| [`../Fix_5/CHANGELOG.md`](../Fix_5/CHANGELOG.md) | Previous fix (record_context_info + expert gen) |
| `fm_visual_avoiding/datasets/sequence.py:_load_images` | Ground truth: 96×96 RGB training images |
| `fm_visual_avoiding/models/visual_unet.py:encode_visual` | Crash site: shape assertion |
| `d3il/.../avoiding_objects.py:BPCageCam` | Default 1024×1024 resolution |
