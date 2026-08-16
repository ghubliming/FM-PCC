# CHANGELOG — Fix_8: UNet channel width (`dim=freq_dim`), applied across 5 generations

**Date:** 2026-08-05 · **Report:** [`REPORT_Fix_8_unet_width_freq_dim_defect.md`](./REPORT_Fix_8_unet_width_freq_dim_defect.md)
**Retrieval flags:** `FIX_8_UNET_WIDTH` (the width defect) and `FIX_8_BACKBONE_DEFAULT`
(the fallback-backbone defect, §6) — **46 sites, 16 files.** Find them all with:

```bash
grep -rn FIX_8_ --include='*.py' . | grep -v Archived_Codes
```

The flag exists because this fix crosses five generations that are otherwise deliberately
isolated from each other. When one of them is next forked, the flag is how the next copy learns
that the line it is about to inherit was wrong once.

---

## 1. One-line summary

`freq_dim` (256) was being passed into `Flow_matcher_U_Net_v2`'s `dim` argument, which sets the
UNet's **channel width** — building a **253.0 M** backbone where the DPCC/FMv3ODE baseline is
**3.97 M** at `dim=32`. A **63.8×** capacity error, silent, on a 96-demonstration dataset.
`freq_dim` is now `32` everywhere.

## 2. Files touched — 16 in total

**13 files for the width defect (§2.1-2.3, §3), plus 3 more train scripts for the
fallback-backbone defect (§6).** Nothing was deleted but the replaced lines themselves.

### 2.1 Config — 1 file, 3 sites

| file | line (pre-edit) | change |
|---|---|---|
| `config/avoiding-d3il.py` | `:497` Gen3v4 iMeanFlow | `'freq_dim': 256,` → `32` + 8-line flagged comment |
| | `:611` Gen3v6 MeanFlow | same |
| | `:711` Gen3v7 α-Flow | same |

These three are **the actual fix**. Everything below is a backstop or a signpost.

### 2.2 Engines — 6 files, 1 site each

Signature default `freq_dim: int = 256` → `32`.

| file | generation |
|---|---|
| `flow_matcher_v3_meanflow/models/mf_engine.py:28` | Gen3v6 |
| `flow_matcher_v3_alphaflow/models/af_engine.py:28` | Gen3v7 |
| `flow_matcher_v3_imeanflow/models/imf_engine.py:27` | Gen3v4 |
| `mix_visual_aligning/models/mf_engine.py:28` | Gen14 |
| `mix_visual_aligning/models/af_engine.py:28` | Gen14 |
| `imf_visual_aligning/models/imf_engine.py:27` | Gen8 |

### 2.3 Trajectory models — 6 files, 3 sites each

Each got: (a) the same signature default, (b) an annotation on the `dim=freq_dim` line,
(c) a build-time parameter-count print.

| file | generation | branch status |
|---|---|---|
| `flow_matcher_v3_meanflow/models/mf_trajectory_model.py` | Gen3v6 | live (`imf_backbone='unet'`) |
| `flow_matcher_v3_alphaflow/models/af_trajectory_model.py` | Gen3v7 | live |
| `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` | Gen3v4 | live |
| `mix_visual_aligning/models/mf_trajectory_model.py` | Gen14 | **dormant** — unreachable at `if_vision=True` |
| `mix_visual_aligning/models/af_trajectory_model.py` | Gen14 | **dormant** |
| `imf_visual_aligning/models/imf_trajectory_model.py` | Gen8 | **dormant** |

## 3. The four changes, verbatim

### (a) Config value — the fix

```python
## architecture sizing (UNet arm; DiT sizing is the dit_* block below)
# 🔴 FIX_8_UNET_WIDTH (2026-08-05) — THIS KEY IS THE UNET CHANNEL WIDTH.
# Its only consumer anywhere is `dim=freq_dim` in models/*_trajectory_model.py,
# and Flow_matcher_U_Net_v2 uses that one argument for BOTH the channel width
# (:106) and the time-embed width (:110). At 256 the backbone was 253.0 M params
# (channels 256/512/1024/2048) against the DPCC/FMv3ODE baseline's 3.97 M at 32
# — a 63.8x capacity error on 96 demonstrations, which silently confounded every
# imf_backbone='unet' run. DiT / SiT / mf_dit ignore this key (they size from
# dit_hidden_size). Full audit: logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/.
'freq_dim': 32,
```

### (b) Signature defaults — the backstop

```python
freq_dim: int = 32,          # 🔴 FIX_8_UNET_WIDTH — UNet CHANNEL width (was 256 => 253 M params);
                             # ignored by the DiT/SiT backbones. See logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/.
```

**This is the only part that changes behaviour outside the three state-only generations.** Gen14
and Gen8 never pass `freq_dim`, so their dormant UNet branch used to fall through to 256. It now
falls through to 32. No current Gen14/Gen8 config reaches that branch (both run `if_vision=True`,
which takes `VisualUNetTwoTime` / `VisualUNet` instead), so nothing running today changes.

### (c) Call-site annotation — the signpost

```python
cond_dim=state_dim,
# 🔴 FIX_8_UNET_WIDTH — `dim` is BOTH the channel width and the time-embed
# width (unet1d_temporal_cond.py:106,110). `freq_dim` is this repo's
# only source for it, so its value IS the backbone size: 32 => 3.97 M,
# 256 => 253.0 M. Never raise freq_dim to "improve the embedding".
dim=freq_dim,
```

### (d) Build-time guard — so it cannot recur silently

```python
_n_params = sum(p.numel() for p in self.velocity_net.parameters())
print(f'[ MFTrajectoryModel ] backbone={imf_backbone}  unet_width(freq_dim)={freq_dim}  '
      f'params={_n_params / 1e6:.1f}M')
```

Placed after the whole backbone-selection chain, so it reports **every** backbone, not just the
UNet. The visual variants print `vision={self.if_vision}` as well. Nothing in the previous train
logs stated a parameter count anywhere — that is why a 63× error survived ~3 months.

Expect on the next Gen3v6 train job:

```
[ MFTrajectoryModel ] backbone=mf_dit  unet_width(freq_dim)=32  params=10.xM
```

and, if the UNet arm is ever run again:

```
[ MFTrajectoryModel ] backbone=unet  unet_width(freq_dim)=32  params=4.0M     ← correct
[ MFTrajectoryModel ] backbone=unet  unet_width(freq_dim)=256  params=253.0M  ← the bug
```

## 4. Deliberately NOT changed

| | why |
|---|---|
| `args_to_watch_fmv3_{imf,mf,af}_train` | Adding `('freq_dim','fd')` changes `exp_name` for **every** run in all three generations, not just UNet ones. Every existing `bbmf_dit`/`bbsit`/`bbdit` checkpoint — all of Gen3v6's headline results — would become unreachable at its recorded path. Rejected; see report §5.2. |
| the three `diffusion_loadpath` templates | follows from the above |
| plan/eval config blocks | the backbone is rebuilt from the pickled `model_config.pkl`, not from the plan block (`eval_flow_matching_v3_meanflow.py:118,186`); the CONFIG-OVERRIDES-PKL loop at `:147` touches `diffusion_config` only. Old checkpoints keep evaluating correctly. |
| renaming `freq_dim` → `unet_dim` | the key still lies about what it is, but a rename plus a re-run in one commit makes a failed run ambiguous. Deferred. |
| deleting the inert `mlp_dim` / `time_dim` / `depth` / `num_heads` from the MF/AF blocks | same reason — they do nothing, but removing them is cosmetic and belongs with the rename |
| `Flow_matcher_U_Net_v2` itself | `dim` doing double duty is upstream DPCC's design. Splitting it would fork the class away from 10 sibling copies. |
| `logs_in_develop/MASTER_TEST_HISTORY.md` | never self-edited (standing convention). Gen3v4's row attributes the pre-U6 failure to the objective; that wants a note. |

## 5. Verification

- **`ast.parse` — all 16 touched files pass.**
- **Line endings preserved.** 11 of 16 files are CRLF, 5 are LF. All patches ran through a
  `newline=''` reader/writer that re-emits the file's own terminator; `file` reports the same
  ending for every file before and after. (The `git diff` warning on the four
  `mix_visual_aligning/` files is pre-existing repo `autocrlf` behaviour, not introduced here.)
- **Flag coverage:** `grep -rn FIX_8_` returns **46 sites across 16 files** — 27 `FIX_8_UNET_WIDTH`
  (3 config + 6 engine + 18 trajectory-model) and 19 `FIX_8_BACKBONE_DEFAULT` (6 signature
  defaults + 3 train fallbacks + 4 Gen14 keep-as-is notes, some spanning two flag mentions),
  matching the intended counts.
- **Parameter arithmetic** re-derived independently from the layer definitions before the fix:
  `dim=32` → 3,968,268 (`dual_head=True`) / 3,962,854 (`False`); `dim=256` → 253,036,556.
  Ratio 63.8×.
- **NOT run** — no Python environment in this container. Every runtime claim needs a cluster job.

### 5.1 What to check on the first cluster run

1. The new `[ ...TrajectoryModel ]` line appears and reports the expected size.
2. A Gen3v6 `mf_dit` train job is **unchanged** — `freq_dim` is not read on that branch, so the
   loss curve must match the existing runs step for step. This is the regression test.
3. Eval of an **existing** checkpoint still loads (the pkl carries the old width; nothing about
   the load path changed).

### 5.2 🔴 Before re-running the UNet arm

Delete or rename the existing `_bbunet_` checkpoint trees. `utils/config.py:36-38` writes
`model_config.pkl` **only if absent**, so re-training into the old directory would leave a pkl
claiming 256 beside a 32-wide checkpoint, and eval would die in `load_state_dict`. Loud, but
avoidable.

## 6. Second defect, same family: the fallback backbone was `'unet'`

**Raised by the user, 2026-08-05, after the width fix landed:** *"config default should be own NN
not UNET, for each 3 gen3v4/6/7"*. Correct, and it is the more dangerous of the two.

Every `imf_backbone` default in the three state-only generations read `'unet'` — a Gen3v4-era
leftover from before U6 introduced the selector, when the UNet was the only backbone there was.
So a missing or misspelled config key did not fail; it **silently selected the one arm whose every
run is confounded by the width defect**. Two failure modes stacked on the same fallback path.

### 6.1 Changed — 9 sites, 9 files

Signature defaults, `'unet'` → each generation's own backbone:

| file | new default |
|---|---|
| `flow_matcher_v3_meanflow/models/mf_engine.py:46` | `'mf_dit'` |
| `flow_matcher_v3_meanflow/models/mf_trajectory_model.py:51` | `'mf_dit'` |
| `flow_matcher_v3_alphaflow/models/af_engine.py:46` | `'sit'` |
| `flow_matcher_v3_alphaflow/models/af_trajectory_model.py:52` | `'sit'` |
| `flow_matcher_v3_imeanflow/models/imf_engine.py:45` | `'dit'` |
| `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py:51` | `'dit'` |

And the same fallback in the three train scripts, which had it a second time:

| file | change |
|---|---|
| `FM_v3_meanflow_test/train_flow_matching_v3_meanflow.py:367` | `getattr(args, 'imf_backbone', 'unet')` → `'mf_dit'` |
| `FM_v3_alphaflow_test/train_flow_matching_v3_alphaflow.py:425` | → `'sit'` |
| `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py:206` | → `'dit'` |

Each matches what `config/avoiding-d3il.py` already sets (`:555` `dit`, `:660` `mf_dit`, `:785`
`sit`), so **no behaviour changes for any current run** — the config always supplies the key. The
default only matters when it doesn't, which is precisely when a wrong default does damage.

### 6.2 🔴 NOT changed — Gen14's `'unet'` default is load-bearing

`mix_visual_aligning/models/{mf,af}_{engine,trajectory_model}.py` keep `imf_backbone: str = 'unet'`.
This is **not** an oversight and must not be "harmonised":

- `config/aligning-d3il-visual.py` contains **no `imf_backbone` key at all** — verified by grep.
  Gen14 relies entirely on the signature default.
- The visual graft guards on it: `mf_trajectory_model.py:77`
  `if imf_backbone not in ('unet',): raise ValueError(...)`.

Changing Gen14's default to a DiT/SiT would make **every Gen14 mf/af visual training job raise on
construction**. All four files now carry a `FIX_8_BACKBONE_DEFAULT` comment saying so, because
the next person doing a consistency sweep will otherwise "fix" it.

## 7. Follow-ups this leaves open

- **Gen3v6 `bbunet` re-run at width 32** — the missing architecture control. Expectation is
  low (see report §3.1), but it is the only way `fix_1`'s verdict becomes a statement about
  architecture rather than about capacity.
- **Correction notes** in `Gen3v6_MeanFlow/fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md`
  (§0 and §1's "one variable" claim) and in the Gen3v4 U1–U5 documents. Not written yet.
- **`MASTER_TEST_HISTORY.md`** Gen3v4 row — flagged, not edited.
- The rename / dead-key cleanup from §4.
