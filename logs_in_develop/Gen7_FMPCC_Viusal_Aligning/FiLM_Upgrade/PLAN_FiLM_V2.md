# Implementation Plan — FiLM_V2 (True FiLM) as an OPT-IN Additional Module

**Date**: 2026-06-27
**Status**: PLAN ONLY (no code written yet)
**Companion doc**: [Ideas.md](./Ideas.md) — feasibility study (proposes *in-place* surgery; this plan rejects that in favor of an *additive* design)

---

## 0. The One-Sentence Difference From `Ideas.md`

> [!IMPORTANT]
> `Ideas.md` proposes **modifying** `ResidualTemporalBlock` + `UNet1DTemporalCondModel` in place — which **breaks every existing visual checkpoint** and changes the default code path.
>
> **This plan instead ADDS a parallel module (`FiLM_V2`)** selected by a new optional config key `film_mode`. When `film_mode` is absent or `'v1'`, **every existing pipeline, checkpoint, and config runs byte-identically**. True FiLM is only constructed when a user explicitly sets `film_mode: 'v2'`.

**Design contract:**
1. Current "Fake FiLM" (additive bias via time-embedding concat) = **`v1`** = the default = untouched.
2. True FiLM (per-block γ scale + β shift) = **`v2`** = new files = opt-in.
3. No existing `.pth` checkpoint becomes unloadable. No existing config needs to change to keep working.
4. The two never share a checkpoint directory (path isolation via `film_mode` in `exp_name`).

---

## 1. Goal & Non-Goals

### Goals
- Add a **real FiLM** denoiser backbone: `out = (1 + γ(v)) · Conv(x) + β(v)` with per-ResBlock, per-channel γ and β derived from the 128-D visual latent, routed **separately** from the time embedding.
- Make it selectable per training/eval run via a single config flag.
- Keep `v1` as the untouched default so all current runs and checkpoints are unaffected.
- Isolate `v1` and `v2` checkpoints so they cannot collide on disk.

### Non-Goals (explicitly out of scope for this plan)
- No cross-attention conditioning (Diffusion Policy style) — separate future effort.
- No change to the vision encoder (`MultiImageObsEncoder` stays; still emits `(B, 128)`).
- No change to the diffusion/flow engines, datasets, samplers, MPC, or projector.
- No change to non-visual pipelines (UAV, state-only avoiding) — they never use `cond_mlp` and are not touched.

---

## 2. Pipelines In Scope

The "Fake FiLM" mechanism (`use_cond_projection=True` → `cond_mlp` → concat with `t`) lives in `UNet1DTemporalCondModel`, used by the **visual aligning** pipelines:

| Pipeline | Backbone file | Wrapper file | Confirmed uses `cond_mlp`? |
|---|---|---|---|
| `fm_visual_aligning` | `fm_visual_aligning/models/unet1d_temporal_cond.py` | `fm_visual_aligning/models/visual_unet.py` | ✅ yes (`use_cond_projection=self.if_vision`) |
| `diffuser_visual_aligning` | `diffuser_visual_aligning/models/unet1d_temporal_cond.py` | `diffuser_visual_aligning/models/visual_unet.py` | ✅ yes |
| `imf_visual_aligning` | `imf_visual_aligning/models/unet1d_temporal_cond.py` | `imf_visual_aligning/models/visual_unet.py` | ⚠️ **VERIFY FIRST** — iMF may route visual cond through `iMFTrajectoryModel`/`Flow_matcher_U_Net_v2` (no `cond_mlp`) instead. Do not implement here until the actual visual-cond path is confirmed. |

> [!NOTE]
> **Primary targets: `fm_visual_aligning` and `diffuser_visual_aligning`.** These are the two pipelines proven to inject the 128-D visual latent through `UNet1DTemporalCondModel.cond_mlp`. Treat `imf_visual_aligning` as a Phase 3 follow-up gated on a verification step (§9, Task V0).

---

## 3. The Config Switch — `film_mode`

A single new optional key, read with a safe default so **absence = current behavior**.

```python
# read inside VisualUNet.__init__
film_mode = getattr(config, 'film_mode', 'v1')   # 'v1' (default, fake FiLM) | 'v2' (true FiLM)
```

| `film_mode` | Backbone constructed | Behavior |
|---|---|---|
| absent / `'v1'` | `UNet1DTemporalCondModel` (existing) | Identical to today. Loads all current checkpoints. |
| `'v2'` | `UNet1DTemporalFiLMModel` (**new**) | True FiLM. Fresh checkpoints only. |

**No config file edit is required for existing runs to keep working** — `getattr(..., 'v1')` makes the key optional. Editing the config is only needed when a user *wants* `v2`.

---

## 4. Files to ADD (new code — nothing overwritten)

### 4.1 New backbone file (one per in-scope pipeline)

`<pipeline>/models/unet1d_temporal_film.py`

Contains two new classes. **It does not import from or modify** `unet1d_temporal_cond.py`; it reuses only the shared low-level blocks from `helpers.py` (`SinusoidalPosEmb`, `Conv1dBlock`, `Downsample1d`, `Upsample1d`), exactly as `unet1d_temporal_cond.py` does.

#### Class A — `FiLMResidualTemporalBlock`

Same skeleton as the existing `ResidualTemporalBlock`, **plus** a separate FiLM projection. Sketch:

```python
class FiLMResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon,
                 kernel_size=5, cond_dim=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        # time path — UNCHANGED, embed_dim is time-only (NOT widened by cond)
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        # ── True FiLM head: visual latent → (gamma, beta) ──
        self.use_film = cond_dim > 0
        if self.use_film:
            self.film_proj = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels * 2),   # γ ‖ β
            )
            nn.init.zeros_(self.film_proj[-1].weight)    # IDENTITY INIT
            nn.init.zeros_(self.film_proj[-1].bias)      # (1+0)·x + 0 = x
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, cond=None):
        out = self.blocks[0](x) + self.time_mlp(t)
        if self.use_film and cond is not None:
            gamma, beta = self.film_proj(cond).chunk(2, dim=-1)   # (B, out_ch) each
            out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
```

> [!TIP]
> **Zero-init γ/β is mandatory.** At step 0 the block computes `(1+0)·x + 0 = x + time_mlp(t)` — identical to a no-FiLM block. Training starts stable and the network *grows into* the gates. This is the same identity-gate trick used by DiT/AdaLN-Zero.

#### Class B — `UNet1DTemporalFiLMModel`

A copy of `UNet1DTemporalCondModel` with **four** differences. **Critically, it keeps the exact same `forward(...)` signature** so the `VisualUNet` wrapper's forward path needs no branching.

| # | Change vs `UNet1DTemporalCondModel` | Why |
|---|---|---|
| 1 | `embed_dim = dim` (NOT `dim + cond_embed_dim`) | Visual no longer widens the time embedding; cond is routed separately. (returns still adds `dim` if `returns_condition`.) |
| 2 | Build `FiLMResidualTemporalBlock(..., cond_dim=film_cond_dim)` where `film_cond_dim = dim` when `use_cond_projection and cond_dim>0` else `0` | Hands each block its own γ/β head. |
| 3 | In `forward`, compute `cond_emb = cond_mlp(cond_pooled)` but **do NOT** `torch.cat([t, cond_emb])`; pass `cond_emb` as the `cond=` arg to every block call | This is the actual FiLM routing. |
| 4 | Every `resnet(x, t)` / `mid_block(x, t)` call becomes `resnet(x, t, cond=cond_emb)` — in **both** `forward` and `get_pred` | All 4 down + 2 mid + 4 up block calls, plus the `get_pred` mirror. |

Keep `cond_mlp` (the `Linear(128→dim)→Mish→Linear(dim→dim)` projector) — it still pools and projects the visual latent; only the *delivery* changes (separate arg instead of concat). `returns_condition` handling stays byte-identical (still concatenated into `t`); all visual configs set `returns_condition=False` so this path is dormant anyway.

> [!CAUTION]
> **Do not forget `get_pred`.** `UNet1DTemporalCondModel` has a second forward loop (`get_pred`) that also iterates the down/mid/up blocks. If any engine calls it, every block call there must also thread `cond=cond_emb`. Thread `cond` through `get_pred`'s signature too (default `None`).

### 4.2 Why a NEW file instead of editing the old one

- Old `unet1d_temporal_cond.py` stays **bit-for-bit unchanged** → every `v1` checkpoint still loads, every `v1` run is reproducible.
- The two backbones live side by side; `VisualUNet` picks one at construction time.
- Diffs are isolated and reviewable; rollback = stop setting `film_mode: 'v2'`.

---

## 5. Files to MODIFY (small, backward-compatible)

### 5.1 `<pipeline>/models/visual_unet.py` — add backbone selection

Only `__init__` changes. The `forward` method is **untouched** because both backbones share the same call signature `backbone(x, visual_cond, t, returns=..., use_dropout=..., force_dropout=...)`.

```diff
+        film_mode = getattr(config, 'film_mode', 'v1')   # 'v1' (default) | 'v2'
+
-        from fm_visual_aligning.models.unet1d_temporal_cond import UNet1DTemporalCondModel
         ...
-        self.backbone = UNet1DTemporalCondModel(
-            horizon=self.padded_horizon,
-            transition_dim=transition_dim,
-            cond_dim=latent_dim,
-            dim=getattr(config, 'dim', 128),
-            dim_mults=getattr(config, 'dim_mults', (1, 2, 4, 8)),
-            returns_condition=getattr(config, 'returns_condition', False),
-            condition_dropout=getattr(config, 'condition_dropout', 0.1),
-            use_cond_projection=self.if_vision,
-        ).to(self.device)
+        common_kwargs = dict(
+            horizon=self.padded_horizon,
+            transition_dim=transition_dim,
+            cond_dim=latent_dim,
+            dim=getattr(config, 'dim', 128),
+            dim_mults=getattr(config, 'dim_mults', (1, 2, 4, 8)),
+            returns_condition=getattr(config, 'returns_condition', False),
+            condition_dropout=getattr(config, 'condition_dropout', 0.1),
+            use_cond_projection=self.if_vision,
+        )
+        if film_mode == 'v2':
+            from fm_visual_aligning.models.unet1d_temporal_film import UNet1DTemporalFiLMModel
+            self.backbone = UNet1DTemporalFiLMModel(**common_kwargs).to(self.device)
+            print('[ VisualUNet ] FiLM_V2 (true scale+shift FiLM) backbone ACTIVE')
+        else:
+            from fm_visual_aligning.models.unet1d_temporal_cond import UNet1DTemporalCondModel
+            self.backbone = UNet1DTemporalCondModel(**common_kwargs).to(self.device)
```

Repeat per pipeline with that pipeline's import path.

### 5.2 `config/aligning-d3il-visual.py` — register the optional key + path isolation

Two minimal additions:

**(a)** Add `'film_mode': 'v1'` to the relevant train/plan blocks (`fm_visual_aligning`, `plan_fm_visual_aligning`, `visual_aligning_dpcc`, `plan_visual_aligning_dpcc`). Setting it to `'v1'` keeps current behavior; a user flips to `'v2'` to opt in. *(Strictly optional — `getattr` defaults to `'v1'` — but making it explicit documents the knob.)*

**(b)** **Checkpoint path isolation.** Append `film_mode` to the `args_to_watch_*` lists so `v1` and `v2` runs serialize to different directories and can never overwrite each other:

```diff
 args_to_watch_fm_visual_train = [
     ('prefix', ''),
     ('horizon', 'H'),
     ...
     ('batch_size', 'bs'),
+    ('film_mode', 'film'),     # film_v1 / film_v2 — isolates true-FiLM checkpoints
 ]
```

…and mirror the same `_film{film_mode}` fragment into the matching `plan_*` block's `prefix` and `diffusion_loadpath` strings so eval resolves the correct checkpoint.

> [!IMPORTANT]
> **Path isolation is the safety net.** Without it, a `v2` training run could write into a `v1` checkpoint folder name and a later `v1` eval would try to load architecturally-incompatible weights → load error. The `film` fragment in `exp_name`/`loadpath` guarantees `H8_..._filmv1` and `H8_..._filmv2` are distinct directories.

---

## 6. Files That Do NOT Change

| Component | Why untouched |
|---|---|
| `unet1d_temporal_cond.py` (all pipelines) | `v1` backbone — left byte-identical so every existing checkpoint loads. |
| `visual_unet.py` `forward()` | Both backbones share the same call signature. |
| `MultiImageObsEncoder` / vision encoder | Still emits `(B, 128)`; FiLM consumes the same latent. |
| Diffusion / Flow / iMF engines, samplers, MPC, DPCC projector | They call `model(x, cond, t)`; the wrapper API is unchanged. |
| `helpers.py` | Shared low-level blocks; reused, not modified. |
| Datasets (`ParityAligningDataset`, etc.) | Trajectory dims unchanged (still 9-D). |
| All non-visual pipelines (UAV, state-only avoiding, `flow_matcher_v3_*`) | No `cond_mlp`; separate code path; zero contact. |

---

## 7. Backward-Compatibility Guarantees (the checklist that must stay TRUE)

1. ✅ Run any current training/eval command with **no config edits** → constructs `UNet1DTemporalCondModel` (v1) → identical to today.
2. ✅ Every existing `.pth` visual checkpoint loads without error (the v1 class is unchanged).
3. ✅ `v2` weights never land in a `v1` directory (path isolation via `film_mode` fragment).
4. ✅ Non-visual pipelines: zero diff, zero risk.
5. ✅ `film_mode: 'v2'` produces a model with new `film_proj.*` tensors and narrower in-block `time_mlp` (`Linear(dim→out)` not `Linear(2·dim→out)`) — a **fresh** architecture requiring training from scratch (expected, isolated by §5.2b).

---

## 8. Architecture Delta (v1 vs v2), Concrete

| | **v1 — Fake FiLM (default, unchanged)** | **v2 — True FiLM (new, opt-in)** |
|---|---|---|
| Formula per block | `Conv(x) + time_mlp([t ‖ cond])` | `(1 + γ(v))·(Conv(x) + time_mlp(t)) + β(v)` |
| In-block `time_mlp` input width | `embed_dim = dim + dim = 2·dim` | `embed_dim = dim` (time only) |
| Visual delivery | concatenated into `t`, fed to `time_mlp` | separate `film_proj` per block (γ, β) |
| γ (scale/gate) | ❌ none (implicit 1) | ✅ learned, zero-init |
| New params | — | ~1.2 M across 16 `film_proj` heads |
| Checkpoint compatibility | loads all current | fresh only (isolated dir) |
| Identity at init | n/a | ✅ behaves as `x + time_mlp(t)` |

---

## 9. Task Breakdown (execution order)

**Phase 1 — `fm_visual_aligning` (reference implementation)**
- **T1.** Add `fm_visual_aligning/models/unet1d_temporal_film.py` (`FiLMResidualTemporalBlock`, `UNet1DTemporalFiLMModel`). Mirror `forward` **and** `get_pred`.
- **T2.** Edit `fm_visual_aligning/models/visual_unet.py` `__init__` for `film_mode` selection (§5.1).
- **T3.** Edit `config/aligning-d3il-visual.py`: add `film_mode` key + `('film_mode','film')` to `args_to_watch_fm_visual_train`; mirror `_film{film_mode}` into `plan_fm_visual_aligning` `prefix`/`diffusion_loadpath`.
- **T4.** Shape test (§10).

**Phase 2 — `diffuser_visual_aligning` (DDPM variant)**
- **T5–T7.** Repeat T1–T3 with `diffuser_visual_aligning` import paths and the `visual_aligning_dpcc` / `plan_visual_aligning_dpcc` config blocks + `args_to_watch_dpcc_train`.

**Phase 3 — `imf_visual_aligning` (gated)**
- **V0. (verify first)** Trace how `imf_visual_aligning` injects the 128-D visual latent: confirm whether it flows through `UNet1DTemporalCondModel.cond_mlp` (then this plan applies verbatim) or through `iMFTrajectoryModel`/`Flow_matcher_U_Net_v2`/`IMFDiTTrajectory` (then a *different* FiLM insertion point is needed — out of scope until traced).
- **T8+.** Implement only if V0 confirms the `cond_mlp` path.

---

## 10. Verification Plan

> [!NOTE]
> Per project memory, the Docker dev box has **no Python runtime** — all execution happens on the remote Slurm cluster via git sync. The checks below are written to run there, not locally.

1. **Construction smoke test** — instantiate `VisualUNet` with `film_mode='v2'`; assert backbone is `UNet1DTemporalFiLMModel`; print param count delta (~+1.2 M).
2. **Forward shape parity** — feed `x=(B,8,9)`, `cond={'visual': (bp,inhand,obs)}`, `t=(B,)`; assert output `(B,8,9)`, no NaN.
3. **Identity-at-init check** — with γ/β zero-init, a single forward should equal the same backbone with FiLM disabled (within fp tolerance) → confirms the zero-init contract.
4. **v1 regression** — run the existing fm_visual eval with no config change; confirm it still loads the current checkpoint and reproduces prior metrics (proves zero breakage).
5. **Path isolation** — dry-run `exp_name`/`diffusion_loadpath` rendering for `film_mode` ∈ {v1, v2}; confirm distinct directory strings.
6. **Short train smoke** — ~200 steps with `film_mode: 'v2'`; confirm loss decreases and no shape errors in the `get_pred`/sampling path.

---

## 11. Risk Table

| Risk | Severity | Mitigation |
|---|---|---|
| `get_pred` loop missed → cond not threaded at sampling | MED | §9 T1 explicitly mirrors `get_pred`; §10 step 6 exercises the sampling path. |
| v2 weights overwrite a v1 dir | HIGH→LOW | Path isolation via `film_mode` fragment (§5.2b); §10 step 5 dry-run. |
| Forward signature drift between v1/v2 → wrapper branch needed | LOW | v2 deliberately keeps the identical `forward(...)` signature (§4.1 Class B). |
| iMF wrong insertion point | MED | Phase 3 gated behind verify task V0 (§9). |
| Training instability from FiLM | LOW | Zero-init γ/β identity start (§4.1); §10 step 3 asserts it. |
| Reviewer confusion "is this FiLM?" | LOW | v2 **is** real FiLM (γ scale + β shift, Perez et al. 2018); name it FiLM only for `film_mode='v2'`. Keep calling v1 "embedding-concat conditioning," never FiLM. |

---

## 12. Definition of Done

- [ ] `film_mode` absent → current behavior, current checkpoints load (regression green).
- [ ] `film_mode: 'v2'` builds `UNet1DTemporalFiLMModel`, forward returns `(B,H,D)`, no NaN.
- [ ] γ/β zero-init verified (identity at step 0).
- [ ] v1 and v2 serialize to distinct checkpoint directories.
- [ ] `fm_visual_aligning` + `diffuser_visual_aligning` covered; `imf_visual_aligning` gated on V0.
- [ ] No diff in `unet1d_temporal_cond.py`, engines, datasets, or non-visual pipelines.

---

## 13. Summary

This plan delivers **true FiLM as a bolt-on, opt-in module** (`film_mode: 'v2'`), leaving the existing "fake FiLM" (`v1`) as the untouched default. New code lives in a new file per pipeline (`unet1d_temporal_film.py`); the only edits to existing files are a backbone-selection branch in `visual_unet.__init__` and an optional, path-isolating config key. Nothing currently runnable stops running; no current checkpoint becomes unloadable; the upgrade is reversible by simply not setting the flag.
