# Fix-18 (Gen7) — Non-Visual One-Shot Run: Code Fixes Applied

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Related**:
- [`INVESTIGATION_REPORT.md`](INVESTIGATION_REPORT.md) — full evidence + line refs
- [`SEVERITY_AND_RETRAIN_IMPACT.md`](SEVERITY_AND_RETRAIN_IMPACT.md) — what's actually broken vs. what isn't, and what to do with existing checkpoints
- [`STALE_CONFIG_PATCH.md`](STALE_CONFIG_PATCH.md) — side-patch (`utils.Config` always-overwrite) + one-off regen script for pre-Fix-A `model_config.pkl` left over on disk
**Scope**: **Five code-level fixes** (18.1 Fix A / 18.2 Fix B / 18.3 Fix C / 18.4 Fix D / 18.5 Fix E) + one side-patch (`utils.Config`), so the non-visual `K=1` DPCC train + non-visual DPCC eval (all projection variants) + ODE=1 FM eval all run end-to-end. Visual path remains unchanged.
**Source logs**: `temp/one_shot_run/visual_dpcc`, `temp/one_shot_run/visual_fm`, plus the 2026-05-31 console log captured in [`fix_console_logs`](fix_console_logs) (regen-script execution that produced the fresh `model_config.pkl`) and SLURM job `21046` stderr (the UF-13 broadcast crash that motivated Fix C).

---

## Summary

User attempted a "one-shot DGM" experiment: train Visual-DPCC at
`n_diffusion_steps=1` and eval Visual-FM with `flow_steps_v3=1`. Both runs
were launched against the visual variants with a CLI override
`if_vision=False`. Both crashed.

- **DPCC train**: crashed on iteration 0 — `RuntimeError: weight of size
  [32, 9, 5], expected input[64, 23, 8] to have 9 channels`. The visual
  `visual_aligning_dpcc` variant hardcodes `obs_dim=6`; the CLI override
  flipped `if_vision=False` only, so dataset switched to 23-D
  (`StateOnlyAligningDataset`) but the model still built 9 input channels
  (`3+6`).
- **FM eval**: ran one variant (0/5 success — *separate* issue, see below),
  then crashed on the next projector setup —
  `ValueError: operands could not be broadcast together with shapes (23,) (9,)`
  at `Projector.build_matrices`. Root cause:
  `eval_fm_visual_aligning.py:1849` derives `_traj_dim = 23 if not if_vision
  else 9` from `args.if_vision` alone, ignoring that the loaded checkpoint
  and normalizer were visual (9-D).

**Verification by user (post-investigation)**: re-ran the same one-shot
experiment on the visual variants with `if_vision=True` (default). **Both
DPCC train and FM eval completed without crashing.** This confirms the two
bugs are **non-visual-path-only**, triggered exclusively by mis-mixing
visual variants with a CLI `if_vision=False` override.

---

## Code Fixes Applied

Two fixes were applied to source. Both are minimal and additive — they only
affect the non-visual code path; the visual path is bit-for-bit unchanged.

### Fix A — train scripts: override `args.obs_dim` for non-visual

**Files**:
- `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` (around line 196)
- `fm_visual_aligning_test/train_fm_visual_aligning.py`         (around line 198)

**Symptom prevented**: `RuntimeError: weight of size [32, 9, 5], expected
input[64, 23, 8] to have 9 channels` at first conv on iteration 0.

**Root cause**: `VisualUNet.__init__` (`models/visual_unet.py:73-74`) computes
`transition_dim = action_dim + obs_dim` in the non-visual branch by reading
`args.obs_dim`. The visual variants in `config/aligning-d3il-visual.py`
hardcode `obs_dim=6` (the visual obs anchor). When a user CLI-overrides
`if_vision=False`, the dataset switches to 23-D but `args.obs_dim` stays at
6, so the model builds with 9 input channels.

**Patch**: after the dataset is constructed, if `if_vision=False`, override
`args.obs_dim` to `dataset.obs_normalizer.mins.shape[0]` (= 20 for
`StateOnlyAligningDataset`) **before** building `VisualUNet`. Logs an
explicit `[ train ] FIX-18: overriding args.obs_dim 6 → 20` line when the
override fires.

```python
if not _if_vision:
    _dataset_obs_dim = dataset.obs_normalizer.mins.shape[0]
    if getattr(args, 'obs_dim', None) != _dataset_obs_dim:
        print(f'[ train ] FIX-18: overriding args.obs_dim '
              f'{getattr(args, "obs_dim", None)} → {_dataset_obs_dim} '
              f'(non-visual; from dataset normalizer)')
        args.obs_dim = _dataset_obs_dim
```

### Fix B — eval scripts: derive `_traj_dim` from saved normalizers

**Files**:
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`         (around line 1848)
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around line 1849)

**Symptom prevented**: `ValueError: operands could not be broadcast together
with shapes (23,) (9,)` at `Projector.build_matrices`.

**Root cause**: `_traj_dim = 9 if args.if_vision else 23` reads the CLI flag
which can be flipped by UF-13's "record-mode auto-enable visual" path. The
checkpoint's saved normalizers are the immutable ground truth for what the
trained model actually produces.

**Patch**: derive `_traj_dim` from `act_normalizer.mins.shape[0] +
obs_normalizer.mins.shape[0]`. Adds a sanity warning if the sum is
unexpected (anything other than 9 or 23).

```python
_act_dim_norm = act_normalizer.mins.shape[0]
_obs_dim_norm = obs_normalizer.mins.shape[0]
_traj_dim = _act_dim_norm + _obs_dim_norm
```

### Fix C (= 18.3) — eval scripts: guard UF-13 record-mode flip on actual checkpoint type

**Added**: 2026-05-31 (after Fix A + Fix B unblocked training but eval still crashed downstream).

**Files**:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around lines 1902-1920)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`           (around lines 1903-1921)

**Symptom prevented**: `ValueError: operands could not be broadcast together
with shapes (1,6) (20,)` at `normalize()` inside `predict()`, when
running eval on a **non-visual** checkpoint with `--record all`.

**Root cause**: UF-13 used to indiscriminately set `if_vision = True`
whenever recording was on, regardless of what the checkpoint was actually
trained for. With a non-visual checkpoint (20-D obs_normalizer), this
forced `Aligning_Sim` into the visual code path, which then called
`agent.predict((bp_image, inhand_image, des_robot_pos, robot_pos),
if_vision=True)`. Inside, the visual branch built a 6-D obs vector and
tried to normalize against the 20-D normalizer → broadcast crash.

The pre-Fix-C UF-13 line:
```python
if not if_vision and args_cli.record != 'none':
    if_vision = True   # ← flips even when there's no image encoder
```

**Patch**: guard the flip on the saved normalizer dim. Only flip when
`obs_normalizer.mins.shape[0] == 6` (i.e. the checkpoint *is* visual).
For non-visual checkpoints, print a NOTE explaining that GIFs/videos
cannot be captured (the model has no image encoder) and proceed with
non-visual rollouts.

```python
_ckpt_is_visual = (obs_normalizer is not None
                   and obs_normalizer.mins.shape[0] == 6)
if not if_vision and args_cli.record != 'none':
    if _ckpt_is_visual:
        if_vision = True
        print('[ eval ] WARNING: ... auto-enabling visual mode ... (UF-13).')
    else:
        print('[ eval ] NOTE: record_mode is active but checkpoint is non-visual '
              f'(obs_normalizer dim = {obs_normalizer.mins.shape[0]}). '
              'Cannot auto-enable visual mode (this model has no image encoder); '
              'proceeding with non-visual rollouts. No GIFs/videos will be captured.')
```

**Consequence (initially)**: eval on a non-visual checkpoint with
`--record all` succeeded and produced metrics + logs, but no GIFs
(there's no image encoder in the model to render through). This was
considered acceptable since the alternative was the crash above.

**Superseded by Fix F (=18.6)**: the "no GIFs for non-visual" caveat
was eliminated by adding `record_sim_frame(env)` — an env-render hook
independent of the policy's image-handling capability. After Fix-18.6,
genuine 23-D non-visual eval ALSO produces GIFs. See §"Fix F (= 18.6)
HOTFIX" below for details. The NOTE message printed by Fix-18.3 was
updated to reflect this.

**Out-of-band patch shipped alongside Fix C** (see
[`STALE_CONFIG_PATCH.md`](STALE_CONFIG_PATCH.md)): `utils.Config.save()`
in both DPCC and FM `utils/config.py` previously skipped overwriting
`model_config.pkl` if a stale copy existed on disk. That mismatch caused
eval to instantiate a 9-D model from the stale config and fail to load a
fresh 23-D state dict (the bug surfaced *between* Fix-18 train success
and Fix C; the patch makes future training runs always overwrite, and
the one-off `regen_stale_model_config.py` script repairs existing
broken checkpoints without re-training). Not strictly part of Fix-18's
non-visual fixes but the same investigation thread; documented in its
own MD to keep this changelog focused.

### Fix D (= 18.4) — eval scripts: first-replan DIAG block referenced visual-only var names

**Added**: 2026-05-31 (after Fix C let eval reach the non-visual `predict()` path).

**Files**:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around lines 1571-1582)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`           (around lines 1557-1568)

**Symptom prevented**: `UnboundLocalError: local variable 'obs_6d_np' referenced before assignment`
at the first-replan diagnostic print, only on the non-visual code path.

**Root cause**: the visual branch of `predict()` defines `obs_6d_np` and
`obs_6d_norm` (the 6-D obs anchor + its normalized form). The non-visual
branch defines `obs_20d_np` and `obs_norm` instead — different names.
The one-shot first-replan diagnostic block hardcoded the visual names,
so reaching it from the non-visual path raised UnboundLocalError.

**Patch**: branch the diagnostic on `if_vision` and bind a pair of
local aliases (`_diag_obs_raw`, `_diag_obs_norm`) to whichever pair
exists. Also generalised the print to include the actual obs dim:

```python
if if_vision:
    _diag_obs_raw  = obs_6d_np      # (6,)
    _diag_obs_norm = obs_6d_norm    # (6,)
else:
    _diag_obs_raw  = obs_20d_np     # (20,)
    _diag_obs_norm = obs_norm       # (20,)
diag_lines += [
    f'[ DIAG obs ] des_c_pos={np.round(_diag_obs_raw[:3], 4)}  '
    f'c_pos={np.round(_diag_obs_raw[3:6], 4)}',
    f'[ DIAG obs ] obs_norm (dim={_diag_obs_norm.shape[0]})='
    f'{np.round(_diag_obs_norm, 4)}',
]
```

For non-visual, only the first 6 entries of the 20-D obs are
des_c_pos + c_pos (positions 0-2 and 3-5); the remaining 14 entries
(box pose, target pose) print fully via `obs_norm`. The "image health"
sub-block remains visual-only as before — that's correctly guarded.

**Consequence**: non-visual eval now passes the first-replan diagnostic
and continues into the rollout loop. No effect on visual path.

### Fix F (= 18.6) HOTFIX — non-visual GIF capture via env-render hook

**Added**: 2026-05-31 (after user observed: "DPCC K=1 non-visual eval produces no GIFs, but FM non-visual DOES produce GIFs — inconsistent").

**Files**:
- `d3il/simulation/aligning_sim.py` (around line 137-142, non-visual rollout branch)
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (new `record_sim_frame` method on Policy; updated Fix-18.3 NOTE message)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` (same)

**Symptom prevented**: Asymmetric GIF behavior across non-visual eval runs.
The legacy FM `_VFalse_` checkpoint (structurally 9-D visual) still
produced GIFs because UF-13 routed it through the visual `predict()`
path, which has its own frame capture buffer. The new DPCC K=1 `_VFalse_`
checkpoint (genuinely 23-D non-visual, post-Fix-18.1) produced no GIFs
because Fix-18.3 correctly kept UF-13 off (avoiding the (1,6) vs (20,)
crash), but the non-visual `Aligning_Sim` branch had no frame-capture
mechanism at all.

**Root cause**: GIF capture in the codebase was historically baked
INSIDE the visual `predict()` path (next to where bp/inhand images are
already being constructed for the model). The non-visual rollout never
had a parallel mechanism — it didn't need one when "non-visual" was
always actually-visual-with-flag-confusion. Once Fix-18.1 enabled
genuine 23-D non-visual training, the gap became visible.

**Patch**: add a render-from-sim hook decoupled from the policy:

1. **`d3il/simulation/aligning_sim.py`** — call
   `agent.record_sim_frame(env)` after every `env.step()` in the
   non-visual branch (visual branch unchanged). Hook is optional via
   `hasattr` check so Aligning_Sim stays compatible with agents that
   don't implement it.

2. **Both eval scripts** — add `Policy.record_sim_frame(env)`. It
   pulls `env.bp_cam.get_image(...)` + `env.inhand_cam.get_image(...)`
   directly from MuJoCo (both cams exist on Robot_Push_Env regardless
   of the env's `if_vision` flag), formats a side-by-side BGR→RGB
   frame with a step-counter overlay (matching the visual branch's
   format), and appends to `self.video_frames`.

3. **The existing save path** in `update_rollout_info` (`if
   self.record_mode != 'none' and self.video_frames: imageio.mimsave(...)`)
   automatically picks up the new frames — no save-side change needed.

4. **Updated Fix-18.3 NOTE message** in both eval scripts: removes
   "No GIFs/videos will be captured" claim (now stale; was always
   conditional on the missing hook).

**Consequence**: non-visual eval (both DPCC and FM, at any K/ODE step
count) now produces GIFs of the actual rollout, captured from the sim's
own cameras independent of whether the policy consumed them. Behavior
table now uniform:

| Checkpoint type | UF-13 fires? | GIF source | Result |
|---|---|---|---|
| Visual `_VTrue_` | N/A (already visual) | visual `predict()` capture buffer | ✅ GIF |
| Legacy 9-D `_VFalse_` (cosmetic non-visual) | Yes (Fix-18.3) | visual `predict()` capture buffer | ✅ GIF |
| Genuine 23-D `_VFalse_` (Fix-18.1 path) | No (Fix-18.3) | **env-render hook (Fix-18.6)** | ✅ GIF |

**Safety**:
- `record_sim_frame` is a no-op if `record_mode == 'none'`.
- All env calls wrapped in `try/except` so a misbehaving camera
  cannot crash a rollout.
- Does NOT alter policy state, predictions, or metrics — pure
  side-channel rendering.
- Visual rollout path is untouched (its existing capture inside
  visual `predict()` is unchanged; the new hook also doesn't fire
  there because the visual branch in `aligning_sim.py` was not
  modified).

### Fix F.1 (= 18.6.1) — `record_sim_frame` produced inverted-color GIFs (R↔B swap)

**Added**: 2026-06-01 (immediately after Fix-18.6 ship, surfaced by
the user's next non-visual DPCC K=1 eval run).

**Symptom**: GIFs produced by Fix-18.6's `record_sim_frame` hook had
inverted colors — blue floor appeared red, red box appeared blue.
Other metrics (positions, success rates) were correct; only the
visual output was wrong.

**Root cause**: Fix-18.6's `record_sim_frame` mistakenly copy-pasted
the `cv2.cvtColor(..., COLOR_BGR2RGB)` call from the visual
`predict()` capture path WITHOUT realizing the two paths receive
images in different color orders.

| Path | Source of image | Color order arriving at the capture code |
|---|---|---|
| Visual `predict()` (pre-existing) | `env.step` returns `bp_image` AFTER `aligning.py:212` did `RGB → BGR` | **BGR** — so `BGR2RGB` in capture is correct (un-does the env's conversion) |
| Fix-18.6 `record_sim_frame` | Directly calls `env.bp_cam.get_image(depth=False)` — **bypasses** env.step's RGB→BGR conversion | **RGB** (per MuJoCo MjCamera spec) — so `BGR2RGB` here applied to RGB data **swaps R↔B** → "inverted" GIFs |

**Patch**: in both eval scripts' `record_sim_frame`, remove the
`cv2.cvtColor(..., COLOR_BGR2RGB)` calls. The camera output is
already RGB; `imageio.mimsave` writes RGB; no conversion needed:

```python
# BEFORE (Fix-18.6, wrong):
bp_vis = cv2.cvtColor(bp.astype(np.uint8), cv2.COLOR_BGR2RGB)
ih_vis = cv2.cvtColor(ih.astype(np.uint8), cv2.COLOR_BGR2RGB)

# AFTER (Fix-18.6.1, correct):
bp_vis = bp.astype(np.uint8)
ih_vis = ih.astype(np.uint8)
```

**Files**:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py:971-972`
- `fm_visual_aligning_test/eval_fm_visual_aligning.py:957-958`

**Scope**: localized to Fix-18.6's `record_sim_frame` only. None of
Fix-18.1–18.5 are affected. Visual `predict()` capture path is
untouched (still correctly does `BGR2RGB` because it receives BGR).

**No model retraining required.** The bug was purely cosmetic in the
GIF channel — no policy state, no metrics, no normalizers affected.

### Fix F.2 (= 18.6.2) — Replace `record_sim_frame` with `capture_frame`; reuse the visual capture pipeline verbatim

**Added**: 2026-06-02 (after Fix-18.6.1's "no-conversion" patch was
verified by the user to STILL produce inverted GIFs on the cluster —
falsifying the 18.6.1 hypothesis that the camera output was RGB in our
runtime).

**Symptom**: Fix-18.6.1's `bp_vis = bp.astype(np.uint8)` (no cvtColor)
also produced inverted-color GIFs. So *both* Fix-18.6 (with `BGR2RGB`)
and Fix-18.6.1 (without) produced wrong colors. Either the d3il
`MjCamera` docstring ("returns RGB") is wrong on this build, or there
is some other subtle source-order issue in `bp_cam.get_image()` direct
calls. Empirically the *only* color pipeline known to work on this
cluster is the visual `predict()` capture path that runs through
`aligning.py:212`'s `RGB → BGR` first.

**Root cause (architectural)**: Fix-18.6's `record_sim_frame` invented
a NEW image-acquisition path (call `env.bp_cam.get_image()` directly)
instead of REUSING the proven pipeline the visual rollout uses. That
introduced an unverified color-order assumption (camera output is
RGB), and 18.6.1's "fix" doubled down on the same assumption by
flipping the cvtColor — still based on a guess. The actual fix is to
**stop guessing** and route through the pipeline whose output we have
ground-truth evidence for (visual GIFs look correct).

**Patch**: replace `record_sim_frame(env)` (which took the env and
called the camera itself) with `capture_frame(bp_np, inhand_np)`
(which receives images already-processed through the visual
pipeline). Move the image-acquisition + RGB→BGR step into
`Aligning_Sim`'s non-visual branch, where it mirrors the visual
branch line-for-line.

| Stage | Visual rollout (proven) | Non-visual rollout (NEW, Fix-18.6.2) |
|---|---|---|
| 1. Camera | `bp_cam.get_image()` → RGB | `bp_cam.get_image()` → RGB |
| 2. RGB→BGR | `aligning.py:212` `cv2.cvtColor(RGB2BGR)` | `aligning_sim.py` `[:, :, ::-1]` (numpy form of the same channel swap; no `cv2` import needed) |
| 3. Transpose + /255 | `aligning_sim.py:120-121` | `aligning_sim.py` (new block, **same code**) |
| 4. Handoff to agent | `agent.predict(..., if_vision=True)` | `agent.capture_frame(bp_np, ih_np)` |
| 5. Capture cvtColor | `predict()` visual block: `cv2.cvtColor(... COLOR_BGR2RGB)` | `capture_frame()`: **byte-identical** `cv2.cvtColor(... COLOR_BGR2RGB)` |
| 6. Frame assembly | `np.concatenate([bp_vis, ih_vis], axis=1)` + `cv2.putText` | **byte-identical** assembly + `putText` |
| 7. Append | `self.video_frames.append(frame)` | `self.video_frames.append(frame)` |
| 8. Save GIF | `imageio.mimsave(... fps=10)` (unchanged) | same unchanged save path |

Stages 4 onward are a literal copy of the visual capture lines. Stages
1–3 are mirrored from the visual branch of `aligning_sim.py`. If the
visual GIF looks correct (which it does — user-confirmed) then the
non-visual GIF is structurally guaranteed to look correct because it
goes through the same channel-swap sequence end-to-end.

**Files**:
- `d3il/simulation/aligning_sim.py` — non-visual branch (~line 138), added image render + `agent.capture_frame()` call. Uses `[:, :, ::-1]` instead of `cv2.cvtColor` to avoid a new `cv2` import in this file.
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — removed `record_sim_frame` (Fix-18.6/18.6.1 deleted); added new `capture_frame(bp_np, inhand_np)` method near `record_step_info` (line ~953). Internal cvtColor lines are byte-identical to predict()'s visual block.
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` — same change as DPCC, mirrored.
- Misleading "No GIFs/videos will be captured" warning updated in both eval scripts to: "GIFs/videos WILL be captured via Aligning_Sim non-visual hook → agent.capture_frame()."

**What was removed**:
- `record_sim_frame(env)` method from both eval scripts (Fix-18.6 and 18.6.1's combined surface).
- The `if hasattr(agent, 'record_sim_frame'): agent.record_sim_frame(env)` hook in `aligning_sim.py` non-visual branch.

**What was kept**:
- Fix-18.1 through Fix-18.5 — all load-bearing for the non-visual eval pipeline crashes, completely orthogonal to the GIF path.
- Visual `predict()` capture path — byte-identical to UF-18.1 (verified by `diff` on the `cvtColor`/`putText`/`append` lines).
- UF-13 auto-enable for 6-D-normalizer (visual) checkpoints — still fires via UF-18.3's `_ckpt_is_visual` guard, which is why FM with a 9-D visual checkpoint and `config if_vision=False` continues to produce GIFs through the visual path the way it did at UF-18.1.

**Scope**: cosmetic GIF channel only. No policy state, metrics,
normalizers, or model code is touched. No retraining required.

**Color-pipeline guarantee**: structural, not empirical. The
non-visual GIF and the visual GIF traverse the same RGB→BGR→BGR2RGB
sequence; only Stage 4's hand-off differs. If one is correct the
other must be.

**Reverts**: localized to ~40 lines across the 3 files. To roll back
Fix-18.6.2 specifically (returning to "no non-visual GIFs"):
```
git checkout a361854 -- d3il/simulation/aligning_sim.py \
    diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py \
    fm_visual_aligning_test/eval_fm_visual_aligning.py
```
(That's the UF-18.5 commit — strips Fix-18.6, 18.6.1, *and* 18.6.2 in
one go. Visual path remains unaffected because it's untouched at
UF-18.5.)

### Fix E (= 18.5) — eval scripts: `setup_dpcc_projector` slices normalizer to wrong width for 23-D trajectory

**Added**: 2026-05-31 (after Fix D let the first ("diffuser") variant complete a full 5-context rollout; crash moved to the second variant's projector setup).

**Files**:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around lines 90-100, inside `setup_dpcc_projector`)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`           (around lines 90-100, same function name)

**Symptom prevented**: `ValueError: operands could not be broadcast
together with shapes (23,) (9,)` at `Projector.build_matrices` line 401
(`a = bound[0] * (x_max - x_min) / 2`), when the second eval variant
needs a projector and the trajectory is non-visual (23-D). The first
variant (`diffuser`) does NOT instantiate a projector, so it ran clean
through all 5 contexts before the crash on variant 2.

**Root cause**: `setup_dpcc_projector` always sliced `obs_normalizer`
down to its first 6 dims:
```python
proj_obs_normalizer = obs_normalizer
if hasattr(obs_normalizer, 'mins') and len(obs_normalizer.mins) > 6:
    ... = obs_normalizer.mins[:6] ...   # ← hardcoded 6
```
Visual: obs_normalizer is 6-D → no-op. Fine.
Non-visual: obs_normalizer is 20-D → trimmed to 6-D, leaving the
projector's `self.normalizer` with `3 act + 6 obs = 9-D` ranges. But the
halfspace bound vector is built at the full `trajectory_dim = 23` width
by `formulate_halfspace_constraints`. `(23,) * (9,)` → crash.

**Patch**: derive the slice target from `trajectory_dim - action_dim`
(=20 for non-visual, =6 for visual), so the slice only fires when the
normalizer is truly oversized for the trajectory at hand:

```python
_target_obs_dim = trajectory_dim - 3   # action_dim hardcoded 3 throughout
proj_obs_normalizer = obs_normalizer
if hasattr(obs_normalizer, 'mins') and len(obs_normalizer.mins) > _target_obs_dim:
    ...slice to _target_obs_dim...
```

Why this is safe for the trailing 14 trajectory dims (positions 9-22 in
non-visual): they carry **zero bound coefficients** from
`formulate_halfspace_constraints` (which only emits non-zero entries at
the explicit `_DIM` indices, all in 0-8). So `bound[0] * range = 0 *
anything = 0` for those positions, contributing nothing to the
constraint matrix. The slice change keeps PCC's robot-kinematic
constraints (dims 0-8) bit-identical to before; it only enlarges the
normalizer so the shape arithmetic works.

**Consequence**: non-visual eval can now build the projector for any
variant after `diffuser` (`dpcc-r`, `dpcc-c`, `dpcc-t`, post-processing,
gradient, …). Visual eval is unchanged (the slice condition `len > 6`
was already false for visual; new condition `len > 6` is still false).

### What was NOT touched

- `config/aligning-d3il-visual.py` — variants are correct.
- `*/models/visual_unet.py` — the non-visual branch logic is fine; only the
  obs_dim value it reads was wrong.
- `*/datasets/sequence.py` — `StateOnlyAligningDataset` (UF-17) already
  produces 23-D correctly.
- `Aligning_Sim` (`d3il/simulation/aligning_sim.py`) — non-visual branch
  already worked correctly; Fixes C and D just let the eval driver reach
  it and survive its first-replan diagnostic.
- `*/sampling/projection.py` — `Projector` and `build_matrices` are
  correct; the bug was in how the call site sized the normalizer it
  passed in (Fix E).
- `formulate_halfspace_constraints` — emits zero-padded bound vectors
  correctly; the consumer's normalizer just needed to be sized to match.
- The visual path of any of the above scripts.

---

## Files Changed in This Fix

### Code (sources)

| Action | File | Fix |
|---|---|---|
| Modified | `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` | A (= 18.1) |
| Modified | `fm_visual_aligning_test/train_fm_visual_aligning.py`         | A (= 18.1) |
| Modified | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`  | B (= 18.2), C (= 18.3), D (= 18.4), E (= 18.5), F-hotfix (= 18.6) |
| Modified | `fm_visual_aligning_test/eval_fm_visual_aligning.py`          | B (= 18.2), C (= 18.3), D (= 18.4), E (= 18.5), F-hotfix (= 18.6) |
| Modified | `d3il/simulation/aligning_sim.py` | F-hotfix (= 18.6) — one-line `record_sim_frame` hook in non-visual branch |
| Modified | `diffuser_visual_aligning/utils/config.py`                    | side-patch (STALE_CONFIG, always-overwrite `model_config.pkl`) |
| Modified | `fm_visual_aligning/utils/config.py`                          | side-patch (STALE_CONFIG, always-overwrite `model_config.pkl`) |

### Docs / one-off scripts

| Action | File |
|---|---|
| Created | `fix_18_nonvisual_step1/INVESTIGATION_REPORT.md` |
| Created | `fix_18_nonvisual_step1/CHANGELOG.md` (this file) |
| Created | `fix_18_nonvisual_step1/SEVERITY_AND_RETRAIN_IMPACT.md` |
| Created | `fix_18_nonvisual_step1/STALE_CONFIG_PATCH.md` (documents the `utils.Config` side-patch + regen script) |
| Created | `fix_18_nonvisual_step1/regen_stale_model_config.py` (one-off cleanup for pre-Fix-A `model_config.pkl` left over on disk) |
| Edited (report) | Status banner added confirming visual-path verification (2026-05-30) |
| Edited (report) | Recommendation #2 and §6 rewritten — vanilla FM train/eval ARE decoupled, so the 1-step FM under-integration is fixed by eval re-run, **not** retraining. Only mean-flow / iMeanFlow requires retraining. |

### Fix numbering recap

- **18.1 (Fix A)** — train scripts override `args.obs_dim` so the model is built 23-D for non-visual.
- **18.2 (Fix B)** — eval scripts derive `_traj_dim` from the saved normalizer (immune to UF-13).
- **18.3 (Fix C)** — eval scripts guard the UF-13 record-mode `if_vision` flip on the saved normalizer dim, so a non-visual checkpoint isn't forced into the visual `predict()` path.
- **18.4 (Fix D)** — eval scripts' first-replan DIAG block aliases `obs_6d_np`/`obs_6d_norm` (visual) vs `obs_20d_np`/`obs_norm` (non-visual) so neither path hits `UnboundLocalError`.
- **18.5 (Fix E)** — `setup_dpcc_projector` now slices the obs normalizer to `trajectory_dim - action_dim` instead of a hardcoded 6, so the 23-D non-visual trajectory gets matching 20-D obs ranges and the projector's bound × range arithmetic stops broadcast-erroring at variant 2+.
- **18.6 (Fix F)** — `Policy.record_sim_frame(env)` env-render hook added; non-visual rollouts now produce GIFs via direct bp_cam/inhand_cam access.
- **18.6.1 (Fix F.1)** — `record_sim_frame` had inverted color (R↔B swap) because it copy-pasted `cv2.cvtColor(BGR2RGB)` from the visual capture path, but the camera output is already RGB (it bypasses env.step's RGB→BGR conversion). One-line removal of the cvtColor calls. Bug introduced and fixed within the same epoch; no model effect.
- **18.6.2 (Fix F.2)** — 18.6.1's "no-conversion" patch ALSO produced inverted GIFs on the cluster, falsifying the "camera output is RGB" assumption empirically. Replaced the env-render hook (`record_sim_frame`) entirely with a `capture_frame(bp_np, ih_np)` agent method whose image-acquisition is moved into `Aligning_Sim`'s non-visual branch and mirrors the visual branch line-for-line (RGB→BGR via `[:, :, ::-1]`, transpose+/255, hand to agent). The agent-side `capture_frame` then uses byte-identical `cv2.cvtColor(... COLOR_BGR2RGB)` lines copied from `predict()`'s visual block. Structural guarantee: visual GIF correct ⇒ non-visual GIF correct.
- **Side-patch (STALE_CONFIG)** — `utils.Config.save()` always overwrites `model_config.pkl`; sibling regen script repairs pre-existing broken checkpoints without re-training.

---

## Key Findings

1. **Visual path is internally consistent.** `visual_aligning_dpcc` and
   `fm_visual_aligning` both define `obs_dim=6`, `if_vision=True`, model
   spec, dataset spec, and normalizer dims in a single self-consistent
   configuration. There is no plumbing to mismatch.

2. **Non-visual path requires using the matching variant.** The
   `ddpm_encdec_vision_nonvisual` variant (UF-17) is the only correct way to
   run non-visual aligning. CLI-overriding `if_vision=False` on a visual
   variant produces inconsistent state because three components read
   different sources of truth:
   - Dataset reads `config.if_vision` (CLI-mutable).
   - Model reads `config.obs_dim` (frozen at variant definition).
   - Eval projector reads `args.if_vision` (CLI-mutable).

3. **FM 0% success at 1-step Euler is expected**, not a bug. The model
   learned a curved velocity field; integrating it with `Δt=1` lands far
   outside the data manifold. Cure: more eval steps, OR switch to mean-flow
   if you want one-shot validity by construction.

4. **DDPM is fundamentally different from FM here.** DDPM's discrete noise
   schedule means train-time T and eval-time T must match — so testing
   "DDPM at T=1" *does* require retraining. FM does not.

---

## Recommended Next Steps (Per Report §5–§6)

Not applied as code in this fix; tracked here for follow-up:

| Priority | Action |
|---|---|
| High | Re-run DPCC training with `prefix=ddpm_encdec_vision_nonvisual/`, `n_diffusion_steps=1`. Drop the `if_vision=False` CLI override (the variant sets it already). |
| High | Sweep FM eval `flow_steps_v3 ∈ {1, 2, 5, 10, 20}` on the existing checkpoint to characterize curvature of the learned flow. |
| Medium | Add guardrails (§5 of the report) — `_traj_dim` derived from normalizer dims, train-time `obs_dim` assertion, variant/flag coherence check at CLI parse. |
| Low | Only if step-count sweep shows fundamental 1-step failure: train an iMeanFlow variant. |

---

## Sync Note

Documentation-only fix. **Sync to Gen6V4 is parallel**, not a code copy. See
[`Gen6V4_dataset_upgrade_visual_dpcc/Gen7_fix18_applied/CHANGELOG.md`](../../../Gen6_dpcc_Engine_for_visual_aligning/Gen6V4_dataset_upgrade_visual_dpcc/Gen7_fix18_applied/CHANGELOG.md).

---

## Final Post-Fix-18 Audit (2026-05-31)

Every fix 18.1 through 18.6 (plus the STALE_CONFIG side-patch) was
re-audited at the time of the Fix-18.6 commit. Conclusion: **none are
hallucinated, none can be safely reverted.**

| Fix | Without it | Verdict |
|---|---|---|
| 18.1 train obs_dim override | Non-visual training crashes at first conv (model 9-D, data 23-D) | Load-bearing |
| 18.2 eval `_traj_dim` from normalizer | Defensive only — if `args.if_vision` matches the checkpoint, old `9 if if_vision else 23` works. Could in principle be reverted, but cleanup gain is 3 lines and it adds a real robustness margin | Defensive — kept |
| 18.3 UF-13 normalizer-dim guard | Genuine 23-D non-visual eval crashes with `(1,6) vs (20,)` broadcast (UF-13 forces visual predict path on a model that can't consume 6-D obs against 20-D normalizer) | Load-bearing |
| 18.4 DIAG var alias | Non-visual predict() crashes at first-replan diagnostic with UnboundLocalError | Load-bearing |
| 18.5 projector slice `_target_obs_dim` | Projector setup for variant 2+ crashes with `(23,) vs (9,)` (obs normalizer trimmed to 6-D vs 23-D bound vectors) | Load-bearing |
| 18.6 record_sim_frame | Genuine 23-D non-visual eval produces no GIFs (visual capture path blocked by 18.3 guard, no fallback) | **Superseded by 18.6.2** |
| 18.6.1 no-conversion patch | Tried to fix 18.6's R↔B by removing cvtColor — *still inverted* on cluster (assumption "camera = RGB" empirically false) | **Superseded by 18.6.2** |
| 18.6.2 capture_frame (reuse visual pipeline) | Without it, 23-D non-visual eval again produces no GIFs (the 18.6/18.6.1 hook was removed; visual capture path is still blocked by 18.3 guard) | Load-bearing for GIFs |
| STALE_CONFIG | `model_config.pkl` becomes stale after retraining → misleads eval (shape mismatch) AND human audit (the exact dim confusion that consumed hours of this session) | Load-bearing for sanity |

Visual path is bit-for-bit unchanged at every fix. Verified by every
fix's guard condition resolving to either "no-op on visual" (e.g.,
Fix-18.1's `if not _if_vision:`) or "same result on visual" (e.g.,
Fix-18.5's `_target_obs_dim = 6` when trajectory_dim is 9).

### Dim inventory at commit time

| Model | Trajectory dim | Notes |
|---|---|---|
| Visual DPCC / FM (any K, any ODE) | **9-D** | Canonical since Gen6V4; not touched by any Fix-18 |
| DPCC K=1 non-visual `_VFalse_` | **23-D** | Trained this week under Fix-18.1; verified by the [32,23,5] shape that appeared in the original shape-mismatch crash |
| FM ODE=1 `_VFalse_` | **Unverified — could be 9 or 23** | model_config showed `obs_dim=6` (suggests 9-D), but STALE_CONFIG bug makes that file unreliable; user asserts the checkpoint was trained under Fix-18.1 (suggests 23-D). Authoritative check is state_dict tensor shape (one-line script in `K1_DDPM_CLOSURE.md` §8). Either way is functionally equivalent post-Fix-18.6 — GIFs work for both. |

### "Why DPCC didn't record GIF but FM did" — final answer

Depends on which interpretation of the FM dim is correct:
- If FM is 9-D visual: asymmetry explained by architectural difference. FM uses UF-13 visual-capture path; DPCC needed Fix-18.6's env-render path.
- If FM is 23-D non-visual: asymmetry unexplained without more data. Both should have behaved identically.

Fix-18.6 closes the asymmetry regardless of which interpretation is true.
