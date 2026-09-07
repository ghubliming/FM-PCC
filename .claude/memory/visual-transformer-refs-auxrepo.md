---
name: visual-transformer-refs-auxrepo
description: "aux_repo/visual_transformer_refs_(Claude_pulled)/ — diffusion_policy + act, pulled 2026-08-20 for Gen14 U8; diffusion_policy is the true upstream of D3IL's vision encoder"
metadata:
  node_type: memory
  type: reference
---

`/workspaces/aux_repo/visual_transformer_refs_(Claude_pulled)/` — pulled by Claude Agent on 2026-08-20 for **Gen14 U8** (visual DiT/SiT ML-bone; plan at `logs_in_develop/Gen14/U8/PLAN_Gen14_U8_visual_dit_bone.md`). Shallow clones, read-only, never imported or run. Has its own `README.md` with per-file pointers.

- **`diffusion_policy/`** (real-stanford, Chi et al.) — 🔴 **the true upstream of FM-PCC's vision encoder.** `d3il/agents/models/vision/multi_image_obs_encoder.py` IS this repo's `model/vision/multi_image_obs_encoder.py`; verified by diff, only import paths and whitespace differ. So the encoder every Gen6–Gen14 visual model uses was written by these authors. Key files: `model/diffusion/transformer_for_diffusion.py` (obs enters as **tokens** + cross-attention decoder — *not* adaLN) and `model/diffusion/conditional_unet1d.py` (their U-Net conditions by **FiLM**, i.e. our `film_mode=v2`).
- **`act/`** (tonyzhaozh) — for Phase B / spatial visual tokens only. D3IL vendors just a cut-down `act_vae.py`; the full DETR-style `detr/models/{backbone,transformer,position_encoding}.py` is the reference for keeping ResNet spatial maps as tokens.

Not pulled, deliberately: `facebookresearch/DiT` (redundant — `aux_repo/MeanFlow/models/dit.py` is already the adaLN-zero source our ports came from) and `octo` (JAX, heavy, design-reading only).

**Why:** every other repo in `aux_repo/` is state-only; none answers "how do you condition a *trajectory* transformer on *image* features?", which is the question U8 turns on.

**How to apply:** consult these before designing any visual-conditioning path for a transformer backbone; cite `transformer_for_diffusion.py` rather than arguing from first principles. See [[fmpcc-dev-logs-navigation]], [[meanflow-family-upstreams]].
