# U6 — How to Use & Replicate the iMF **DiT** Backbone on Avoiding

**Goal:** train and evaluate the real-iMF method with the official-style **DiT** backbone
(`imf_backbone='dit'`) on `avoiding-d3il`, and A/B it against the UNet and the FM baseline.
**Companion:** [CHANGELOG_DiT_Backbone.md](./CHANGELOG_DiT_Backbone.md) · [PLAN_Switchable_DiT_Backbone.md](./PLAN_Switchable_DiT_Backbone.md).
**Key fact:** the backbone is chosen in **`config/avoiding-d3il.py`**, not by a CLI flag. The train and
plan blocks **must agree** (the checkpoint folder name and `state_dict` both depend on it).

---

## 0. TL;DR (3 steps)

1. **Edit the config** — set `'imf_backbone': 'dit'` in **both** the `flow_matching_v3_imeanflow`
   (train) block **and** the `plan_fm_v3_imeanflow` (plan) block. Keep every `dit_*` value identical
   across the two blocks.
2. **Sync to cluster & run the pipeline** (train → eval, chained):
   ```bash
   sbatch Slurm_Codes/sbatch/iMF/imf_pipeline.sh
   ```
3. **Read results** — outputs land under a folder tagged `…_bb dit` (see §5), kept fully separate from
   the UNet run. A/B per §7.

> Reverting is the same edit back to `'imf_backbone': 'unet'` (the default).

---

## 1. What selects the DiT

One key, in **two** config blocks (`config/avoiding-d3il.py`):

```python
# ── flow_matching_v3_imeanflow (TRAIN block) ──
'imf_backbone': 'dit',        # 'unet' (default) | 'dit'
'dit_depth': 8,               # total transformer blocks
'dit_hidden_size': 256,       # token width — keep small for H=8
'dit_num_heads': 4,
'dit_aux_head_depth': 2,      # private blocks per u/v head
'dit_patch_size': 1,          # MUST divide horizon (8): 1 or 2
'dit_condition_on_t': False,  # official recipe: condition on h=t−r only

# ── plan_fm_v3_imeanflow (PLAN/EVAL block) — MUST be identical ──
'imf_backbone': 'dit',
'dit_depth': 8, 'dit_hidden_size': 256, 'dit_num_heads': 4,
'dit_aux_head_depth': 2, 'dit_patch_size': 1, 'dit_condition_on_t': False,
```

**Why both:** training bakes the architecture into `model_config.pkl` and into the folder name
(`…_bb{imf_backbone}`). Eval rebuilds/loads against the plan block; if the `dit_*` or `imf_backbone`
differ, you get either a **wrong folder** (`diffusion_loadpath` not found) or a **`state_dict`
mismatch**. Treat the two blocks as one unit.

Everything else stays at the U5 all-power defaults (`imf_objective='meanflow_jvp'`, uniform schedule,
`dual_head`, `interval_cfg`, `ω=4.0`, `flow_steps_v3=2`). The DiT carries its dual heads + interval
conditioning natively, so those flags remain consistent.

---

## 2. Pre-flight on the cluster (do this BEFORE a full run)

There is **no torch in the Docker dev box** — the forward pass and the JVP have only been
`py_compile`-checked, never executed. Run this **micro smoke test on the cluster** (seconds, 1 GPU or
even CPU) to de-risk the 24 h training job. It validates the two things most likely to break: the
**forward shapes** and the **MeanFlow JVP** through the DiT.

```python
# scratch_dit_smoke.py  — run on cluster: python scratch_dit_smoke.py
import torch
from torch.func import jvp
from flow_matcher_v3_imeanflow.models import IMFDiTTrajectory

B, H, D = 4, 8, 6                      # transition_dim D = obs+action for avoiding
net = IMFDiTTrajectory(horizon=H, transition_dim=D,
                       hidden_size=256, depth=8, num_heads=4,
                       aux_head_depth=2, patch_size=1).cuda()

x = torch.randn(B, H, D, device='cuda')
t = torch.rand(B, device='cuda'); h = torch.rand(B, device='cuda')
omega = torch.full((B,), 4.0, device='cuda')
tmin = torch.full((B,), 0.4, device='cuda'); tmax = torch.full((B,), 0.6, device='cuda')

# (1) forward shape parity: u, v must be [B, H, D]
u, v = net(x, None, t, h=h, omega=omega, t_min=tmin, t_max=tmax, return_v=True)
assert u.shape == (B, H, D) and v.shape == (B, H, D), (u.shape, v.shape)

# (2) the hard gate — MeanFlow JVP through the DiT (mirrors imf_diffusion._u_of)
def u_of(z, t_in, h_in):
    return net(z, None, t_in, h=h_in, omega=omega, t_min=tmin, t_max=tmax, return_v=True)[0]
ones = torch.ones_like(t)
u_pred, du_dr = jvp(u_of, (x, t, h), (torch.randn_like(x), ones, -ones))
assert u_pred.shape == du_dr.shape == (B, H, D)
print("DiT smoke OK — forward + JVP clean", u_pred.shape)
```

- **If (1) fails:** a shape/patch bug — check `dit_patch_size` divides `H=8`.
- **If (2) fails:** the JVP gate. The real-valued RoPE + RMSNorm were chosen to be forward-AD-safe; an
  error here points at a stray in-place op or dtype cast — capture the traceback before the big run.

Also run a config-parse check (no GPU needed) to catch typos in the two blocks:
```bash
python -c "import diffuser.utils as u; u.Parser().parse_args(experiment='flow_matching_v3_imeanflow')"
```

---

## 3. Replication recipe (exact steps)

1. **Branch / sync.** Commit the config edit locally, push, and `git pull` on the cluster (the project's
   git-sync workflow). Do **not** hand-edit on the cluster only — keep the dev box authoritative.
2. **Edit `config/avoiding-d3il.py`** per §1 (both blocks → `'dit'`).
3. **Smoke test** per §2 on the cluster.
4. **Launch** the chained pipeline (§4).
5. **Verify** per §6 as logs arrive.

---

## 4. Running

**Recommended — chained pipeline (train → eval on success):**
```bash
sbatch Slurm_Codes/sbatch/iMF/imf_pipeline.sh
# submits train_imf.sh, then eval_imf.sh with --dependency=afterok
```

**Or run the stages individually:**
```bash
# Train (seeds 6–10, W&B on — as wired in the sbatch script)
sbatch Slurm_Codes/sbatch/iMF/train_imf.sh
#   → python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10 --use-wandb

# Evaluate (reads plan_fm_v3_imeanflow; resolves the _bb{dit} checkpoint)
sbatch Slurm_Codes/sbatch/iMF/eval_imf.sh
#   → python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py

# Aggregate plots only (after per-seed eval exists)
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py --aggregate-only
```

> Monitor: `tail -f Slurm_Codes/logs/latest.log` (the sbatch scripts symlink the active job's stdout
> there).

**Advanced (CLI override instead of editing the file):** the trainer forwards unknown args to the
`utils.Parser`, so `... train_flow_matching_v3_imeanflow.py --seeds 6 --imf_backbone dit` works for a
quick training probe — **but** the eval script does not take that override, so for a real train→eval
cycle **edit the config** so both stages agree. The CLI path is for one-off training smoke runs only.

---

## 5. Where the outputs land (and why they don't collide)

The U6 watch tag `('imf_backbone','bb')` puts the backbone in the folder name. So the DiT and UNet
checkpoints live in **separate directories**:

```
# UNet (default)
…/flow_matching_v3_imeanflow/H8_D…_a1.0_b1.0_aw10_objmeanflow_jvp_bbunet/
# DiT
…/flow_matching_v3_imeanflow/H8_D…_a1.0_b1.0_aw10_objmeanflow_jvp_bbdit/
```

Eval's `diffusion_loadpath` carries the same `_bb{imf_backbone}` suffix, so it auto-resolves the right
one. You can keep a UNet run and a DiT run side by side without overwriting.

---

## 6. Verification checklist (what "working" means)

| # | Check | Pass criterion |
|---|---|---|
| 1 | **JVP gate** (§2 smoke or training step) | training starts, no `torch.func.jvp` error |
| 2 | **Forward parity** | DiT `u`/`v` are `[B, 8, D]` like the UNet |
| 3 | **Train loss descends** | `train/loss` falls; no NaN (zero-init gates should keep it stable early) |
| 4 | **1-NFE reconstruction** sanity | matches the U4/U5 reconstruction check |
| 5 | **Interval-CFG** | ω sweep (0 → 4 → 8) trades diversity/quality monotonically |
| 6 | **Eval resolves checkpoint** | no "loadpath not found" / `state_dict` mismatch (means §1 blocks agree) |

---

## 7. The A/B you actually want (DiT vs UNet vs FM)

Same data, seeds (6–10), schedule, NFE, projector — only the backbone/objective changes:

| Arm | Config | NFE |
|---|---|---|
| **FM baseline** | `imf_objective='fm_equivalent'`, `imf_backbone='unet'` | 10 |
| **UNet real-iMF** | `imf_objective='meanflow_jvp'`, `imf_backbone='unet'` | 1 / 2 / 4 |
| **DiT real-iMF** | `imf_objective='meanflow_jvp'`, `imf_backbone='dit'` | 1 / 2 / 4 |

Report **success rate / trajectory quality** *and* **`fm_ms`** (latency) per arm. The U6 question:
**does the DiT's tokenized conditioning + deep dual heads beat the UNet at 1–2 NFE?**
(See [UNet_vs_DiT_for_iMF_Principle](../U5/UNet_vs_DiT_for_iMF_Principle.md) for the hypothesis — the
DiT's edge here is conditioning capacity, not receptive field, since `H=8` is already globally coupled.)

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.func.jvp` error at train start | a non-AD-safe op slipped into the DiT path | run §2 smoke, capture traceback; verify real-RoPE is used (not a complex cast) |
| `horizon 8 not divisible by patch` | `dit_patch_size` ∤ 8 | use `dit_patch_size` ∈ {1, 2, 4} |
| `diffusion_loadpath … not found` at eval | plan block backbone/`dit_*` ≠ train | make the two blocks identical (§1) |
| `Error(s) in loading state_dict` | architecture mismatch (depth/hidden/heads/patch) | match **all** `dit_*` across train+plan |
| Eval loads the UNet by mistake | forgot `_bb` in a custom loadpath | keep the `_bb{imf_backbone}` suffix in `prefix`/`diffusion_loadpath` |
| OOM / slow | DiT too big for `H=8` | shrink `dit_hidden_size`/`dit_depth` (256/8 is already small; the receptive-field win is moot at H=8) |

---

## 9. Reverting to the UNet

Set `'imf_backbone': 'unet'` in **both** blocks (the default). Nothing else changes; the UNet run is
byte-for-byte the U5 behaviour and its checkpoints are in the `_bbunet` folder, untouched by any DiT run.
