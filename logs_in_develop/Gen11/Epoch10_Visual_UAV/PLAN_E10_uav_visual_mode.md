# PLAN — Gen11 Epoch 10: **UAV Visual Mode** (state-only UAV FM-PCC → dual-camera visual FM-PCC)

**Date:** 2026-08-11 · **Type:** future-facing plan / investigation · **NO CODE WRITTEN**
**Status:** draft for review — nothing may be implemented until §7 decisions are confirmed.
**Scope:** extend the Gen11 UAV closed-loop pipeline (`flow_matcher_v3_uav/` ↔ `FM_v3_uav_test/`)
from a **state-only** policy to an **image-conditioned** policy, by grafting the Gen7 visual
stack (`fm_visual_aligning/models/visual_unet.py` + `visual_gaussian_diffusion.py`) onto the UAV
frame, fed by **two cameras** (overhead + nose FPV) rendered from MuJoCo.

> **Governing principle (inherited from Gen14/Gen15):** *write the least code — reassemble, don't
> rewrite.* The **UAV frame is the host**; the vision stack is the **graft**. Never the other way
> round: the visual folders have no MuJoCo scenes, no PID/MJPC tracker, no real-time logging, no
> UAV constraint geometry.

---

## 0. The question asked: *was a Gen11 visual plan ever made, and is it current?*

**Answer: it was declared, never written, and what exists is stale.**

| Where | What it says | Length | State |
|---|---|---|---|
| [`Epoch6_fm_pcc_training/IDEAS.md`](../Epoch6_fm_pcc_training/IDEAS.md) | 5-phase spine `P0 data → P1 mini-FM → P2 state-FM → P3 DPCC → **P4 visual**` | 1 line | forward pointer only |
| [`Epoch6_fm_pcc_training/EPOCH6_PLAN.md` §"Phase 3"](../Epoch6_fm_pcc_training/EPOCH6_PLAN.md) (L336–341) | *"Visual FM-PCC — DEFERRED to next Epoch (based on Gen7 `fm_encdec_vision`): WS-A camera collection, dual-camera FiLM-ResNet encoder conditioning the FM, retrain with visual conditioning."* | 5 lines | **the entire existing "plan"** |
| [`Epoch7…/PLAN.md` §3](../Epoch7_fm_pcc_FULL_PCC_MPC/PLAN.md) (L75–76) | *"**No vision.** Drop all image/ResNet conditioning; UAV is state-only."* | — | E7 explicitly closed the door for E7 |
| [`Epoch5…/init_0/CLOSURE.md`](../Epoch5_visual_and_validation/init_0/CLOSURE.md) (L45) | *"WS-A camera image collection (`collect_camera_images.py`) — **not run**; not required before training"* | — | collector shipped, **never executed** |

Epochs 7, 8 and 9 then went to PCC projection / MJPC thrust control / constraint geometry. **Vision
was never picked up again.** Nothing under `Gen11/Epoch{7,8,9}` contains a visual-observation design.

### 0.1 Why the E6 stub is not usable as-is — five concrete stalenesses

1. **Wrong target folder.** It names `fm_encdec_vision` — that name survives only inside
   `config/aligning-d3il-visual.py` (Gen6 diffusion bridge). The live Gen7 FM visual arm is
   **`fm_visual_aligning/`**, and as of **Gen14** the maintained multi-engine visual frame is
   **`mix_visual_aligning/`**.
2. **Wrong observation schema.** The stub assumes the E6 layout `obs=[p_des|p|v]` (9-D obs, 12-D
   transition). Since E8 the default is `cond_mode='pos_only'` → **obs `[p_des|p]` 6-D, transition
   9-D** (`config/uav.py:133`, `:190`). This change is what makes the graft *cheap* (see §2) — the
   stub predates it and misses the point entirely.
3. **Predates the whole control stack.** No DPCC projector, no `pid_stopgo`/MJPC tracker, no
   real-time deadline accounting (`proj_ms`, `total_ms_p95`, `over_budget`), no circuit breaker
   (E9 Fix 15.3), no `config_override_pkl` two-tier reconciliation. A visual plan that ignores the
   30.3 ms replan budget is not a plan.
4. **Predates the Gen7 encoder's own evolution** — `film_mode='v2'` (true per-block FiLM,
   `Gen7_FMPCC_Viusal_Aligning/FiLM_Upgrade_C1/`) and the fact that `VisualUNet` **hardcodes
   `TRANSITION_DIM = 9`**.
5. **Predates Gen14/Gen15**, i.e. the established graft methodology and the engine-registry axis
   this generation must compose with, not collide with (§3.3).

**Verdict: not up to date → this document replaces both stubs.** The old stubs are left untouched
as history.

---

## 1. Asset audit — what already exists today (nothing here needs inventing)

| # | Asset | Path | State |
|---|---|---|---|
| A1 | **Dual-camera expert-image collector** (replay E4 pickles, inject `qpos/qvel/q`, render 2 cams @96×96 BGR PNG) | `uav_expert_data_collect/collect_camera_images.py` (294 L) | written, **never run** |
| A2 | SLURM wrapper for A1 (EGL + GPU, 8 h) | `Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh` | written, never run |
| A3 | **Vision encoder + FiLM U-Net** (`MultiImageObsEncoder`, dual ResNet18@96² → 64+64=128-D latent → FiLM) | `fm_visual_aligning/models/visual_unet.py` | live (Gen7/Gen14) |
| A4 | **Visual FM engine** (`VisualFlowMatching(FlowMatchingODE)`, `cond = {0: (bp, wrist, obs_seq)}`) | `fm_visual_aligning/models/visual_gaussian_diffusion.py` | live |
| A5 | **Image dataset pattern** (`_load_images`, BGR→RGB /255, sliding windows) | `fm_visual_aligning/datasets/sequence.py:154` | live |
| A6 | **Visual eval agent** (per-step camera grab, `deque(maxlen=window)` image context, batch repeat) | `fm_visual_aligning_test/eval_fm_visual_aligning.py:1979–1999` | live |
| A7 | **UAV closed loop** with per-scene renderer + EGL discipline (`_make_overhead_renderer`, `_free_renderer`, `_render_overhead`) | `FM_v3_uav_test/eval_fm_uav.py:780–840` | live |
| A8 | **UAV episode schema** with attitude `q(T,4)` — required for a correct FPV view | `uav_expert_data_collect/dataset_writer.py:40–95` | live |
| A9 | UAV data loader (`uav-<scene>` → curated pkl) | `flow_matcher_v3_uav/datasets/d4rl.py` | live |
| A10 | Vision encoder source (hydra-instantiated, no D3IL env needed) | `d3il/agents/models/vision/{multi_image_obs_encoder,model_getter}.py` | vendored |

**Consequence: no new *model* and no new *offline collector* is needed** — but the **online** side
(policy cameras inside the closed loop) does not exist yet, and neither does the plumbing that keeps
the offline and online renders identical. See §1.2 for the exists-vs-new split.

### 1.1 Defects found in the never-run collector (A1) — fix before running

| ID | Finding | Evidence |
|---|---|---|
| **D-1** | Output dir is `images/track-cam/…` but the E5 spec and every downstream doc say `fpv-cam`; the camera actually rendered is named `fpv` (`_TRACK_CAM_NAME='fpv'`). Docstring still says *"from …xml `track`"*. | `collect_camera_images.py:17,57,250` vs `Epoch5…/EPOCH5_PLAN.md:28,81` |
| **D-2** | The "bp-cam" is **not** a fixed world camera despite the docstring — `render_frame_overhead` builds a free camera with `lookat = data.qpos[:3]`, i.e. it **follows the drone** (ego-centric top-down, 5 m out). This is a *modelling decision*, not a bug, but it is undeclared. See §7-D2. | `collect_camera_images.py:124–137` |
| **D-3** | Attitude fallback is silent: `q_stored = episode.get('q', None)` → level attitude if absent. Any pre-E4-U3 episode renders an **attitude-blind FPV stream** that silently disagrees with eval. Must hard-fail instead. | `collect_camera_images.py:152,172` |
| **D-4** | No image↔obs count assertion is written to disk; the E5 verification step is a manual shell snippet. | `Epoch5…/EPOCH5_PLAN.md:92–101` |

### 1.2 Tooling: what is reusable vs **what must be newly written**

§1 says "~90 % wiring", which is true of the *model* but **not of the tools**. Split honestly:

| | Tool | Status | Note |
|---|---|---|---|
| **REUSE** | `collect_camera_images.py` — offline 2-cam render from E4 pickles | exists, never run | needs the WP0 fixes (D-1…D-4) — **not** a rewrite |
| **REUSE** | `collect_camera_images.sh` (EGL/GPU sbatch) | exists | |
| **REUSE** | `generate_trajectory_gifs._render_overhead` (proven overhead cam) | exists, already imported by eval | |
| **REUSE** | `MultiImageObsEncoder` + `get_resnet` | vendored in `d3il/` | pure torch — no D3IL env needed |
| **NEW — T1** | **`uav_camera_rig.py`** — one module owning both camera specs (`fpv` cam id, overhead free-cam pose/distance, 96², BGR↔RGB, `mj_forward` vs live-physics call sites) | **must be written** | ⭐ *the key new tool* — see below |
| **NEW — T2** | **Live dual-camera capture inside the closed loop** | **must be written** | the UAV eval has **no** policy camera at all today — only a 140 px debug-GIF renderer (`eval_fm_uav.py:780`). The arm gets images from the D3IL env API (`env.bp_cam.get_image()`); **the UAV has no such env API**, so this cannot be copied from Gen7 — it must be written against raw `mujoco.Renderer`. |
| **NEW — T3** | `UAVVisualSequenceDataset` (lazy `uint8`, ep_id→image-dir map) | **must be written** | pattern from Gen7, but the eager-float32 loading *cannot* be copied (§WP2) |
| **NEW — T4** | Corpus packer + verifier (PNG→`uint8` `.npy`, count parity, contact sheets, `meta.json` fingerprint) | **must be written** | small, but it is the only thing standing between us and a silently-corrupt 2 M-frame corpus |
| **NEW — T5** | Eval-startup **camera-spec fingerprint assertion** (corpus spec == eval spec) | **must be written** | the structural fix for Risk 1 |
| **NEW — T6** | Visual train/eval sbatch set (`Slurm_Codes/sbatch/uav_fm_visual/`) | **must be written** | copies of `uav_fm/` |
| **NEW — T7** | *(optional)* policy-cam frame dump into the eval npz for post-hoc DA | **must be written** | extends `eval_artifacts.py` |

> ⭐ **T1 is the load-bearing new tool, and it is worth stating why.** Today the camera geometry
> would exist in **three** disconnected places: the offline collector, the debug-GIF renderer, and
> (new) the eval policy capture. Training images and eval images agreeing would then be a matter of
> *convention* — and the single largest risk in this epoch (Risk 1) is exactly that they silently
> stop agreeing. `uav_camera_rig.py` makes the agreement **structural**: the collector and the eval
> loop both call `rig.render_policy_views(model, data)` and both stamp `rig.fingerprint()` into
> their outputs, so a drift becomes a loud assertion instead of a quietly-wrong model.
> It is ~100 lines and it subsumes the existing render helpers rather than duplicating them.

**So: yes, new tools are needed** — but the *expensive* one (offline 2-camera collection) already
exists; what is missing is the **online** half of the same capability, plus the plumbing that keeps
the two halves honest.

---

## 2. The load-bearing coincidence: **the UAV and the arm already share a 9-D transition**

`VisualUNet` hardcodes `TRANSITION_DIM = 9` and ignores `config.obs_dim` (a deliberate "fix_5
lesson" guard). Since E8, UAV `cond_mode='pos_only'` is *also* 9-D, with the same
`[action | anchor | actual]` structure:

| | action (3) | cols 3:6 | cols 6:9 | transition |
|---|---|---|---|---|
| **Gen7 visual aligning** | `Δdes_c_pos` | `des_c_pos` | `c_pos` | 9-D |
| **Gen11 UAV `pos_only`** | `Δp_des` | `p_des` | `p` | 9-D |

**Implications (all verified against the code, not assumed):**

- `VisualUNet` and `VisualFlowMatching` can be copied **byte-for-byte** — no dimension surgery.
- The DPCC dynamics constraint differs only by *which* index pair is bound, and that was already
  resolved in E7: bind `p_des` (cols 3:6), not `p` — `('deriv',[3,0]),([4,1]),([5,2])`
  (`Epoch7…/PLAN.md:80–86`). The visual arm binds `[6,0..2]` because its arm tracks perfectly.
  **Nothing in the projector changes for the visual UAV.**
- The normalizer stays `SafeLimitsNormalizer` on the 6-D obs / 3-D action (`config/uav.py:138`).
- Only the **conditioning path** changes: `cond = {0: obs_6d}` → `cond = {0: (bp, fpv, obs_6d)}`.

This is why Epoch 10 is a graft and not a generation-scale rewrite.

### 2.1 The network: **reuse `VisualUNet` — no new architecture**

**Recommendation: copy Gen7's `fm_visual_aligning/models/visual_unet.py` verbatim** (2 cameras,
two independent ResNet18 trunks @96² → 64-D each → concat 128-D → FiLM into the 1-D temporal U-Net).
Three code-grounded reasons:

1. **It already fits the UAV tensor shapes** — `TRANSITION_DIM = 9` matches `pos_only` exactly (§2).
2. **The encoder is generic over cameras.** `MultiImageObsEncoder` loops over every `type: rgb` key
   in `shape_meta['obs']` (`multi_image_obs_encoder.py:43–52,116`) and concatenates per-key
   features. Camera **count is a config change, not a code change** — the only manual edits are the
   two hardcoded constants `LATENT_DIM` (= 64 × n_cams) and `TRANSITION_DIM`.
   `fm_visual_avoiding/models/visual_unet.py` is the **1-camera precedent in this repo**
   (`LATENT_DIM=64`, `TRANSITION_DIM=6`) — proof the pattern flexes cleanly.
3. **It keeps the comparison architecture-controlled.** Gen7 (arm, visual), Gen14 (arm, visual,
   multi-engine) and Gen11-E10 (UAV, visual) would then share one backbone, so a UAV-vs-arm
   difference is attributable to the *task*, not the vision stack.

> Note: `mix_visual_aligning/models/visual_unet.py` (Gen14) is **byte-identical** to Gen7's except
> for two import paths (verified by `diff`). Copy from Gen7 to keep the `fm`-engine lineage; a later
> merge into Gen15's `mix_uav/` stays mechanical either way.

**Alternatives considered and ruled out (for run 1):**

| Option | Verdict | Why |
|---|---|---|
| **A. Gen7 `VisualUNet`, 2 cams, separate trunks** | ✅ **adopt** | §2.1 above |
| B. `share_rgb_model=True` (one trunk for both cams) | ❌ reject | halves encoder cost we don't need to save (§6: projector dominates), and the two views are semantically *unrelated* (world-layout top-down vs body-frame FPV) — a shared trunk is the wrong inductive bias. Arm precedent is `False`. |
| C. Smaller/custom CNN for latency | ❌ reject | solves a non-problem: encode is ~2–5 ms against a 30.3 ms budget already blown by 770–1240 ms of projection. Would also break parity with Gen7/Gen14. |
| D. **ImageNet-pretrained** ResNet18 | ⏸ hold as **first ablation** | `get_resnet` hardcodes `pretrained=False` (`model_getter.py:14–19`), so E10's encoder learns from scratch on synthetic MuJoCo renders — that is Risk 3. Pretrained weights are the obvious remedy **if the visual arm underfits**, but flipping it in run 1 confounds the observation-space ablation with an initialisation change. Change one thing at a time. |
| E. New architecture (ViT / CLIP / DINO / R3M) | ❌ reject | no such encoder exists in any `aux_repo/` upstream for this task — **UAV-Flow's Python side has no vision model at all** (it only logs waypoints/velocity commands; physics and perception live in Unreal), so there is nothing to "port". Introducing one destroys the architecture-controlled comparison and adds latency and a second unvalidated dependency. |
| F. 3rd camera (e.g. chase view) | ❌ reject for run 1 | free to add later (encoder is key-generic), but every extra camera re-renders the whole corpus (§WP1) and adds a render to every control step. Two views already span *global layout* + *local obstacle proximity*. |
| G. Feed images to the **projector** as well | ❌ out of scope | the DPCC projector operates on the 9-D transition and analytic geometry; vision changes the *generative* path only. Explicit non-goal (§9). |

**Config keys this implies** (mirroring Gen7 so checkpoints stay legible):
`if_vision=True`, `film_mode='v1'` (v2 = ablation), `LATENT_DIM=128`, `n_cams=2`,
`shape_meta` keys kept at the arm's names (`agentview_image` ← bp-cam, `in_hand_image` ← fpv-cam)
with the mapping documented — renaming buys nothing and widens the diff.

---

## 3. Why do it — the scientific case (and what would falsify it)

### 3.1 The hypothesis
Gen11's decisive E6 finding is **homotopy ambiguity**
([`…/U3/FINDING_homotopy_ambiguity_4scene_AB.md`](../Epoch6_fm_pcc_training/U3/FINDING_homotopy_ambiguity_4scene_AB.md)):
single-homotopy scenes fly (empty 1.00, s_curve 0.95 stability-success), multi-homotopy scenes
(corridor 3 modes, pillars 4 modes) collapse to 0.00 and *explode* — a single unselected sample
oscillates between modes and a 2nd-order drone cannot absorb it.

E7–E9 attacked this **downstream**: candidate fan + selection + DPCC projection pick one mode
*after* the model has been ambiguous. **Vision attacks it upstream:** the obstacle layout is *in
the image*, so the conditional distribution can collapse toward the mode the scene actually
affords, before any projector runs.

> **Falsifiable claim for E10:** on `corridor` and `pillars`, the visual arm reduces
> **plan-fan mode-spread** (`plan_cand_spread`, `npz_analysis_tool`) and raises unguided
> (`variant='diffuser'`, no projection) stability-success versus the state-only arm **trained on
> the same episodes with the same engine and horizon**. If the fan spread does not shrink, vision
> bought nothing and the arm should be reported as a negative result, not tuned into a win.

### 3.2 Why the *comparison* is clean
Same engine (`FlowMatchingODE`), same data, same 9-D transition, same projector, same trackers,
same scenes. E10 is a **pure observation-space ablation** — the only such ablation in this repo
where the downstream controller is a real-time closed loop.

### 3.3 How this composes with Gen15 (do not collide)
Gen15 (`mix_uav/` ↔ `mix_uav_test/`) adds the **engine axis** (`fm | mf | af`) to the UAV frame.
E10 adds the **observation axis** (state | visual). They graft onto the *same host* and touch
almost-disjoint files (Gen15: `models/mf_*.py`, engine registry; E10: dataset + `visual_unet` +
`cond` plumbing).

**Rule: E10 ships `fm`-only.** The visual × engine cross-product is a *later* generation, and only
if both axes independently pay. Keep the config key names compatible with Gen15's registry
(`engine='fm'`, `if_vision=True`) so the merge is later mechanical.

---

## 4. Target architecture

### 4.1 Training data flow
```
data/uav_fm/v1/<scene>/<ep_id>.pkl          logs/uav_expert_data/images/
  obs (T,6) [p_des|p]                          bp-cam /<scene>/<homotopy>/<ep_id>/{t}.png
  actions (T-1,3) Δp_des                       fpv-cam/<scene>/<homotopy>/<ep_id>/{t}.png
        │                                                 │
        └──────────────► UAVVisualSequenceDataset ◄───────┘   (index by ep_id; 1 frame ⇔ 1 obs row)
                                   │
                  Batch(trajectories (H,9), conditions={0: (bp(W,3,96,96), fpv(W,3,96,96), obs(6,))})
                                   │
                       VisualFlowMatching.loss  →  VisualUNet
                                                     ├─ MultiImageObsEncoder (2× ResNet18) → 128-D
                                                     └─ UNet1DTemporalCond/FiLM (9-D, H=8)
```

### 4.2 Eval (closed loop, per replan @ `control_hz=33`)
```
MuJoCo data ──► renderer.update_scene(cam=free-overhead)  ─► bp   96×96 ─┐
            └─► renderer.update_scene(cam='fpv')          ─► fpv  96×96 ─┤
                                                                          ├─ encode ONCE → 128-D
 obs = [p_des | p] (6,) ──────────────────────────────────────────────────┘        │
                                                                                   ▼
                       policy({0:(bp,fpv,obs)}, batch_size=B) ─► B candidate plans (H,9)
                                     │ (latent broadcast to B — do NOT repeat images)
                       DPCC Projector (unchanged) ─► selection ─► first Δp_des ─► PID/MJPC ─► sim
```

---

## 5. Work packages

> Every runnable step is **cluster-only** (`i6-gpu-1`, EGL, FMPCC env). Nothing below can be
> validated in the dev container.

### WP0 — Collector hardening *(no new features)*
- Fix **D-1** (`track-cam` → `fpv-cam`, docstring), **D-3** (hard-fail when `q` missing, with the
  offending `ep_id` printed), **D-4** (write a per-episode `meta.json` with `T`, `n_frames`,
  `scene`, `homotopy`, `git_rev`, camera spec, resolution).
- Decide **D-2** (§7-D2) and *record the camera spec in `meta.json`* so a later re-render is
  detectable.
- **Acceptance:** smoke run `--max-episodes 1` per scene; `n_frames == len(obs)` for both cams;
  neither stream near-uniform (reuse the Gen9 preflight std-check idea from
  `collect_visual_avoiding_data.py:134`).

### WP0b — **`uav_camera_rig.py`** (T1) — the new shared tool
- One module, ~100 lines, owning: `fpv` camera lookup, overhead free-cam pose (`lookat`, `distance`,
  `azimuth`, `elevation`), policy resolution (96²), colour convention, and a
  `fingerprint()` → dict/hash of all of the above + scene XML mtime/`git_rev`.
- Two entry points: `render_policy_views(model, data)` (used **identically** by the offline
  collector and the online eval) and `fingerprint()` (stamped into `meta.json` at collection, and
  asserted at eval startup — T5).
- Refactor `collect_camera_images.py` to call it; leave the 140 px **debug-GIF** renderer alone —
  it is deliberately a different camera and must not be merged into the rig.
- **Acceptance:** collector output before/after the refactor is pixel-identical on one episode;
  `fingerprint()` changes when (and only when) a camera parameter or the scene XML changes.

### WP1 — Run the collection (cluster) + storage budget
- 1,952 accepted E4 episodes; per-scene path lengths 360–750 steps (`config/uav.py:48–54`).
  Order-of-magnitude: **~1.5–2 M frames total across 2 cameras.**
- **Storage:** ~10–20 KB/PNG → **15–40 GB and ~2 M inodes.** ⚠️ Cluster quotas are usually
  inode-limited, not byte-limited. → §7-D8: strongly consider **one `uint8` `.npy` per
  (episode, camera)** (`(T,96,96,3)`, ~20 MB/episode, **2 files per episode instead of ~2×T**).
- **Acceptance:** count-parity report over *all* episodes, and a contact sheet of 20 random
  (bp, fpv) pairs per scene reviewed by eye before any training is queued.

### WP2 — `UAVVisualSequenceDataset`
- New file in the fork; pattern from `fm_visual_aligning/datasets/sequence.py:53–174`, with
  **two deviations forced by scale**:
  1. **Lazy `uint8`, not eager `float32`.** The arm dataset loads every frame as
     `float32/255` into RAM. At UAV scale that is `2M × 96×96×3 × 4 B ≈ 220 GB` — impossible.
     Store/emit `uint8` and do `→float/255 + imagenet norm` **on GPU per batch**
     (`uint8` still ≈ 55 GB → memmap `.npy` per episode, page-cached, never fully resident).
  2. **Index map** `curated pkl → image dir`: the pkl carries `scene` + `homotopy` + `episode_id`
     (`dataset_writer.py:75–95`), the images live at
     `images/<cam>/<scene>/<homotopy_safe>/<ep_id>/` — reconstruct, and **assert** the dir exists
     for every curated episode at construction time (fail loudly, never silently drop).
- Windows: `(ep, start, end)` valid only where **both** obs rows and frames exist.
- **Acceptance:** one batch printed with shapes `(B,H,9)`, `(B,W,3,96,96)×2`, `(B,6)`; RAM high-water
  mark logged; a `__getitem__` frame visually matched against its `p` from `obs`.

### WP3 — Model graft
- Copy `visual_unet.py` + `visual_gaussian_diffusion.py` into the new model folder **unchanged
  except**: `shape_meta` key names (`agentview_image`→`bp_image`, `in_hand_image`→`fpv_image`) —
  *or keep the arm's key names verbatim to minimise diff and just document the mapping*
  (recommended: **keep verbatim**, per the Gen15 fidelity rule).
- `film_mode`: default `'v1'`; `'v2'` (true FiLM) as an explicit ablation only.
- **Acceptance:** `diff -u` against the Gen7 originals fits on one screen; a forward pass on a
  synthetic batch returns `(B,H,9)`.

### WP4 — Train entry + config + sbatch
- `train_*` copied from `FM_v3_uav_test/train_fm_uav.py`, dataset class swapped, `if_vision=True`.
- `config/uav.py`: new experiment block; checkpoint path must get a **`_VIS` tag** next to the
  existing `_9D` dim tag (`config/uav.py:69–91`) so visual and state-only checkpoints can never
  land in the same folder.
- `Slurm_Codes/sbatch/uav_fm_visual/` mirroring `uav_fm/` (train, eval, all-scenes, pipeline).
  ⚠️ per repo rule: **no tqdm/live bars in batch logs**; `--time = 2× expected` (24 h cap).
- **Acceptance:** 2 k-step smoke train on `empty` completes, loss decreases, no NaN
  (watch the E6-F3 `SafeLimitsNormalizer` constant-column lesson).

### WP5 — Eval loop: live dual-camera capture (T2 — genuinely new code)
- ⚠️ **Nothing here can be copied from Gen7.** The arm's eval pulls frames from the D3IL env API
  (`env.bp_cam.get_image(...)`); the UAV has **no env wrapper** — the eval drives raw
  `mujoco.MjModel/MjData`. The capture must be written against `mujoco.Renderer` via the WP0b rig.
- In `rollout_one`, call `rig.render_policy_views(...)` each replan (2 × 96²), and keep the
  **GIF renderer separate** (140 px, `frame_stride=3`) — *never* conflate the policy camera with the
  debug GIF camera (that conflation is exactly what Fix_7/Fix_9 warn about, `eval_fm_uav.py:780–795`).
- Assert `rig.fingerprint()` equals the corpus fingerprint at startup (T5) — fail fast, never warn.
- **One renderer per scene, reused**; `_free_renderer` on exit (EGL teardown discipline, E6-U3).
- **Encode once, broadcast.** The arm's eval repeats the image tensor across `batch_size`
  candidates (`eval_fm_visual_aligning.py:1996`). For the UAV that would multiply the ResNet cost
  by `B` inside a 30 ms budget for **zero** information gain — the latent is identical for all
  candidates. Encode once, `expand` the 128-D latent. *(This is a justified UAV-specific deviation
  from the arm; document it in the changelog.)*
- Image context window `W`: `deque(maxlen=W)`, warm-started by repeating frame 0 (arm behaviour).
- **Acceptance:** per-step `render_ms` / `encode_ms` logged next to `proj_ms`; a `[DIAG img]`
  near-black guard on both streams (arm has this at `eval…:2161–2169`); one rollout GIF that shows
  the fpv stream alongside the overhead.

### WP6 — Parity + A/B protocol
- **Parity gate first:** run the visual arm with `if_vision=False` on the *same* data and confirm
  it reproduces the current state-only Gen11 numbers within noise. If it doesn't, the graft is
  wrong and no visual result means anything.
- Then the A/B: state-only vs visual, **same seeds, same scenes, same projection variants**,
  unguided (`diffuser`) *and* guided.
- **Report both**: unguided (does vision fix the model?) and guided (does vision survive the
  projector?).

### WP7 — Docs + DA
- Changelog per repo convention under `logs_in_develop/Gen11/Epoch10_Visual_UAV/<update>/`.
- DA per the standing rule: fix the **best baseline row as Target**; here the baseline is
  **Gen11 state-only FM + DPCC** (there is *no* diffusion-DPCC UAV checkpoint in this repo — the
  same limitation Gen15 §1.5 documents). ⚠️ Any claim must therefore read *"vs Gen11 state-only
  FM + DPCC"*, never *"beats DPCC"*.
- **Do not edit `MASTER_TEST_HISTORY.md`** — propose a row, let the user add it.

---

## 6. Real-time budget — where vision actually lands

| Item | Cost | Source |
|---|---|---|
| Replan budget @ 33 Hz | **30.3 ms** | `Epoch9…/notes/PROJECTION_VARIANTS_ANALYSIS_s_curve.md:127` |
| Current full-stack projection | **770–1240 ms** (`total_ms_p95` 5–8 s) | ibid. :177 |
| Cheapest useful geometry variants | ~75–90 ms (near budget) | ibid. :224 |
| **+ 2 × 96² EGL renders** | to be measured; expect **single-digit ms** | WP5 |
| **+ dual ResNet18 @96², B=1, encode-once** | to be measured; expect **~2–5 ms GPU** | WP5 |

**Honest reading:** vision does **not** meaningfully worsen the deadline picture — the projector
already dominates it by 1–2 orders of magnitude. But two things must be enforced anyway:
(a) **encode-once-broadcast** (otherwise cost scales with the candidate fan), and
(b) `render_ms`/`encode_ms` must be logged separately from `proj_ms`, so that the E9 over-budget
accounting stays interpretable and vision never becomes a hidden term in it.

---

## 7. Decisions to confirm before any code

| # | Decision | Recommendation |
|---|---|---|
| **D1** | Folder pair | **`fm_visual_uav/` ↔ `fm_visual_uav_test/`** (new sibling, copy of `flow_matcher_v3_uav/` @ HEAD). *Not* in-place — it would break Gen11 rollback and invalidate every `logs/UAV_FM/uav-<scene>/flow_matching_v3_uav/…` checkpoint path. |
| **D2** | Overhead camera: **drone-following** (current code) vs **world-fixed per scene** (D3IL's actual convention) | **Keep drone-following for run 1**, and *declare it*: the view becomes translation-invariant (better generalisation across starts) while absolute position is still supplied by `p`/`p_des` in the obs anchor. Fixed-world is the natural A/B if run 1 underperforms. This is the single most consequential undeclared choice in the collector. |
| **D3** | `cond_mode` | **`pos_only` locked** (9-D transition ⇒ zero-surgery graft, §2). |
| **D4** | Image window `W` | **`W=1`** for run 1 (lowest latency, simplest); `W=2` mean-pooled as ablation. The arm mean-pools a window — a drone at 33 Hz gets little from 30 ms of history. |
| **D5** | Engine | **`fm` only.** MF/AF is Gen15's axis (§3.3). |
| **D6** | Scenes / order | **`pillars` + `corridor` first** — the multi-homotopy scenes where the hypothesis lives. `empty`/`s_curve` are near-saturated and can only show regressions. |
| **D7** | Attitude fidelity | **Require `q`** in every episode; hard-fail otherwise (D-3). An attitude-blind FPV stream is a silent train/eval mismatch. |
| **D8** | On-disk format | **`uint8` `.npy` per (episode, camera)** over per-frame PNGs (inode + load-time; §WP1). If PNG parity with D3IL is wanted for inspection, emit PNGs for a *sampled* subset only. |
| **D9** | **Which NN** | **Reuse Gen7 `VisualUNet` verbatim** (2× ResNet18 → 128-D → FiLM). No new architecture, no pretrained weights, no shared trunk in run 1 — see the §2.1 options table. Pretrained-ResNet is the designated *first* ablation if the arm underfits. |
| **D10** | **Camera count / roles** | **2: `bp-cam` (global layout) + `fpv-cam` (local proximity)**. The encoder is key-generic, so 1 or 3 is a config change — but each extra camera re-renders the whole corpus and costs a render every control step. |
| **D11** | Copy the visual stack from Gen7 or Gen14? | **Gen7 `fm_visual_aligning/`** — `fm`-engine lineage. The Gen14 file is byte-identical modulo import paths, so a later merge into `mix_uav/` is mechanical regardless. |

---

## 8. Risks

1. **Train/eval render mismatch (highest).** Training frames come from **state injection +
   `mj_forward`** (no physics); eval frames come from **live physics**. Lighting/scene identical,
   but any XML, camera-spec, or resolution drift silently invalidates the entire image corpus.
   → `meta.json` camera fingerprint (WP0) + a startup assertion in eval that the eval camera spec
   equals the corpus spec.
2. **Any scene-geometry change ⇒ full re-render** (~cluster-hours, not minutes). Freeze scene XMLs
   before WP1; the E9 constraint work has been editing *YAML* geometry, which is fine, but XML
   edits are not.
3. **Encoder is randomly initialised** (`get_resnet(..., pretrained=False)`,
   `d3il/agents/models/vision/model_getter.py:14–19`). It must learn from this dataset alone — if
   the visual arm underfits, that, not "vision doesn't help", is the first suspect.
4. **Dataloader throughput** becomes the training bottleneck at ~2 M frames; budget CPU workers and
   verify GPU utilisation before blaming the model.
5. **Ego-centric overhead view may be *too* invariant** — if the drone-centred crop excludes the
   goal, the image cannot disambiguate homotopy at long range and the hypothesis fails for a
   trivial framing reason. Check the 5 m camera distance covers the corridor/pillars layout in the
   WP0 contact sheet **before** WP1.
6. **Negative result is a real outcome.** If fan spread doesn't shrink, report it (the Gen13 iMF
   precedent) rather than tuning until the number moves.

---

## 9. Non-goals for Epoch 10

- No changes to the DPCC projector, constraint YAMLs, trackers, or the physics loop.
- No MF/AF/iMF engines (Gen15's axis).
- No new UAV expert-data collection — E10 re-renders **existing** accepted episodes only.
- No real-time optimisation of the projector (that is E9's open front).
- No edits to `MASTER_TEST_HISTORY.md` (propose only).

---

## 10. References (file:line)

- Stale stubs: `Gen11/Epoch6_fm_pcc_training/EPOCH6_PLAN.md:336–341`, `…/IDEAS.md` (P4)
- E7 "no vision" + `deriv` mapping: `Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/PLAN.md:75–86`
- Collector: `uav_expert_data_collect/collect_camera_images.py:17,57,124–137,152,248–250`
- Collector sbatch: `Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh`
- Episode schema (`obs`,`q`,`actions`,`homotopy`): `uav_expert_data_collect/dataset_writer.py:40–95`
- Visual encoder/backbone: `fm_visual_aligning/models/visual_unet.py:11–100`
- Encoder is generic over N cameras: `d3il/agents/models/vision/multi_image_obs_encoder.py:43–52,116`
- ResNet18 trunk, `pretrained=False`: `d3il/agents/models/vision/model_getter.py:7–19`
- 1-camera precedent (LATENT_DIM=64, TRANSITION_DIM=6): `fm_visual_avoiding/models/visual_unet.py:19–28`
- Gen14 VisualUNet is byte-identical to Gen7's modulo imports: `diff fm_visual_aligning/models/visual_unet.py mix_visual_aligning/models/visual_unet.py`
- Visual FM engine: `fm_visual_aligning/models/visual_gaussian_diffusion.py:6–100`
- Image loading pattern: `fm_visual_aligning/datasets/sequence.py:53–174`
- Visual eval agent (window, batch repeat, diagnostics): `fm_visual_aligning_test/eval_fm_visual_aligning.py:1937–1999, 2161–2169`
- UAV eval renderer + EGL discipline: `FM_v3_uav_test/eval_fm_uav.py:780–840`
- UAV obs layout switch: `FM_v3_uav_test/eval_fm_uav.py:937–943`; `config/uav.py:126–133,190`
- Path lengths / scene budgets: `config/uav.py:48–54`
- Homotopy ambiguity finding: `Gen11/Epoch6_fm_pcc_training/U3/FINDING_homotopy_ambiguity_4scene_AB.md`
- Timing budget & projection cost: `Gen11/Epoch9_PCC_Constraints/notes/PROJECTION_VARIANTS_ANALYSIS_s_curve.md:127,177,224`
- Graft methodology: `Gen15/init/PLAN_Gen15_uav_mix_ml.md`, `Gen14/init/PLAN_Gen14_visual_mix_ml.md`
- Gen9 dual-camera collection precedent: `collect_visual_avoiding_data/collect_visual_avoiding_data.py`
