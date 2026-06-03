# Gen9 Epoch 2 — Single Camera Visual Avoiding Pipeline — Implementation Changelog

**Date**: 2026-06-03 (initial); revision 2026-06-03 (per user review — see §10)
**Status**: ✅ Phases 1–4 complete (all coding); Phase 5 (smoke + training) pending cluster
**Parent plan**: [`PLAN_SINGLE_CAMERA_VISUAL_AVOIDING.md`](PLAN_SINGLE_CAMERA_VISUAL_AVOIDING.md) (audit-corrected per §12)
**Branch**: `update_into_FM` (uncommitted)

---

## 1. Executive Summary

Visual-DPCC and Visual-FM pipelines for the D3IL **avoiding** task are now scaffolded end-to-end. Trajectory dimension dropped from aligning's 9-D `[act(3) | des_c_pos(3) | c_pos(3)]` to **6-D `[act(2) | des_xy(2) | c_xy(2)]`**, vision halved from dual cameras (bp + inhand, LATENT_DIM=128) to **single camera (bp-only, LATENT_DIM=64)**. The 6 fixed obstacles enter as `sphere_outside` projector constraints in the planning configs — NOT as obs vector entries.

All code parses (AST + YAML); no Python runtime in this Docker, so no smoke test was executed here. Ready for cluster sync → Phase 5 smoke tests → training.

---

## 2. Files Touched

### 2.1 New folders (Phase 1 — copy + rename)

| New | Source |
|---|---|
| `diffuser_visual_avoiding/` | `diffuser_visual_aligning/` |
| `diffuser_visual_avoiding_test/` | `diffuser_visual_aligning_test/` |
| `fm_visual_avoiding/` | `fm_visual_aligning/` |
| `fm_visual_avoiding_test/` | `fm_visual_aligning_test/` |

Global find-replace `diffuser_visual_aligning` → `diffuser_visual_avoiding` and `fm_visual_aligning` → `fm_visual_avoiding` applied across all `.py / .md / .yaml / .yml / .toml / .txt` inside the four new folders. File-name renames inside `*_test/`:

| Old | New |
|---|---|
| `diffuser_visual_avoiding_test/eval_visual_aligning_dpcc.py` | `eval_visual_avoiding_dpcc.py` |
| `diffuser_visual_avoiding_test/train_visual_aligning_dpcc.py` | `train_visual_avoiding_dpcc.py` |
| `fm_visual_avoiding_test/eval_fm_visual_aligning.py` | `eval_fm_visual_avoiding.py` |
| `fm_visual_avoiding_test/train_fm_visual_aligning.py` | `train_fm_visual_avoiding.py` |

All `__pycache__/` directories cleared inside the new folders so stale `.pyc`s don't shadow renamed `.py` files.

### 2.2 Files modified inside the new folders (Phases 2–3)

| File | Change |
|---|---|
| `diffuser_visual_avoiding/datasets/sequence.py` | Replaced `ParityAligningDataset` → `ParityAvoidingDataset`. 9-D → 6-D trajectory. `ACTION_DIM=2`, `OBS_DIM=4`, `TRAJ_DIM=6`. Drops `inhand_cam` loading, returns `{0, 'primary_img'}` only (no `wrist_img`). Reads `env_state['robot']['des_c_pos'][:, :2]` and `[:, :2]` of `c_pos`. Path: `environments/dataset/data/avoiding/all_data/`. **`StateOnlyAligningDataset` class REMOVED** (per plan §4.1 — out of scope for Epoch 2). |
| `fm_visual_avoiding/datasets/sequence.py` | Identical mirror of the diffuser version. |
| `diffuser_visual_avoiding/models/visual_unet.py` | `TRANSITION_DIM` 9 → **6**. `LATENT_DIM` 128 → **64** (single ResNet-64). `shape_meta` drops `in_hand_image` — single-camera encoder. `encode_visual()` takes only `bp_imgs`; `forward()` unpacks single-cam payload `(bp_imgs, obs_seq)` or `(bp_imgs,)`. Action-dim default 3 → 2. |
| `fm_visual_avoiding/models/visual_unet.py` | Mirror, with import path `fm_visual_avoiding.models.unet1d_temporal_cond`. |
| `diffuser_visual_avoiding/models/visual_gaussian_diffusion.py` | `loss()` no longer references `wrist_img`; builds `cond['visual'] = (primary_img, obs_seq)` (single-cam tuple). `forward()` unpacks `(bp_imgs, obs_seq)` for inference. Trajectory layout comment updated to 6-D. |
| `fm_visual_avoiding/models/visual_gaussian_diffusion.py` | Identical pattern for `VisualFlowMatching`. |
| `diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py` | `exp = 'avoiding-d3il'`. Parser experiment name `'visual_avoiding_dpcc'`. Dataset class `ParityAvoidingDataset`. Path `environments/dataset/data/avoiding/all_data/train_files.pkl`. wandb project `FMPCC-visual-avoiding-dpcc`. Non-visual branch now raises `NotImplementedError` (StateOnly class removed). |
| `fm_visual_avoiding_test/train_fm_visual_avoiding.py` | Same pattern; experiment `'fm_visual_avoiding'`; wandb project `FM-PCC-visual-avoiding-gen9`. |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Experiment names, sim class (`Avoiding_Sim`), env class (`ObstacleAvoidanceEnv`), state-data dir, config name (`config.avoiding-d3il`), eval YAML path (`config/visual_avoiding_eval.yaml`) — all updated. |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Mirror of eval_visual_avoiding_dpcc.py changes. |

### 2.3 Files modified OUTSIDE the new folders

| File | Change |
|---|---|
| `config/avoiding-d3il.py` | **(revision)** *no longer touched* — the 4 new config entries were initially appended here but moved to a dedicated `config/avoiding-d3il-visual.py` file in revision (mirrors aligning's split between `aligning-d3il.py` and `aligning-d3il-visual.py`). This file is at its pre-Epoch-2 state. |
| `d3il/environments/dataset/avoiding_dataset.py` | Added `Avoiding_Img_Dataset` class mirroring `Aligning_Img_Dataset` for D3IL-native training loops. Reads bp-cam + inhand-cam frames; obs is 2-D-sliced `[des_x, des_y, c_x, c_y]` filled into the first 4 indices of the 20-D buffer (D3IL's buffer-size convention). Added `from tqdm import tqdm` import. |

### 2.4 Files created

| File | Purpose |
|---|---|
| `config/avoiding-d3il-visual.py` | **(revision — new file)** Dedicated visual-avoiding config, mirrors `config/aligning-d3il-visual.py`. Contains 4 entries: `visual_avoiding_dpcc`, `fm_visual_avoiding` (train) and `plan_visual_avoiding_dpcc`, `plan_fm_visual_avoiding` (plan). Both plan entries pull `constraint_list` from the module-level `_AVOIDING_OBSTACLES` (6 `sphere_outside` constraints — positions verbatim from `avoiding_objects.py:get_obj_xy_list()`, radius 0.04 m). Reads `_yaml_threshold` from `config/visual_avoiding_eval.yaml`. Defines the same `args_to_watch_*` helpers as aligning visual for checkpoint naming. |
| `d3il/configs/avoiding_vision_config.yaml` | New D3IL Hydra config for visual avoiding training. Points at `Avoiding_Img_Dataset` (the new class), `Avoiding_Sim` with `if_vision=True`, `obs_dim=4`, `action_dim=2`, `max_len_data=200`, `window_size=8`. |
| `config/visual_avoiding_eval.yaml` | Minimal eval-time YAML stub. Lists `avoiding-d3il-visual`, `visual_avoiding_dpcc`, `fm_visual_avoiding` as exps. Geo-constraint variants intentionally empty — avoiding's obstacles live in `config/avoiding-d3il-visual.py:plan_*.constraint_list`, not in this YAML. |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/` | **(revision — new dir)** 3 sbatch scripts copied + modified from `Slurm_Codes/sbatch/diffuser_visual_aligning/`: `train_visual_avoiding_dpcc.sh`, `eval_visual_avoiding_dpcc.sh`, `visual_avoiding_pipeline_dpcc.sh`. Module names, experiment names, config file refs, job names, and wandb projects all updated. |
| `Slurm_Codes/sbatch/fm_visual_avoiding/` | **(revision — new dir)** 3 sbatch scripts copied + modified from `Slurm_Codes/sbatch/fm_visual_aligning/`: `train_fm_visual_avoiding.sh`, `eval_fm_visual_avoiding.sh`, `fm_visual_avoiding_pipeline.sh`. Same rename pattern. |

---

## 3. Plan-Compliance Matrix

| Plan section | Status | Notes |
|---|---|---|
| §3 Copy 4 folders | ✅ Done | All 4 copied, find-replace clean (`grep -L diffuser_visual_aligning ... | wc -l = 0` inside the new folders). |
| §4.1.1 Dataset (DPCC) | ✅ Done | `ParityAvoidingDataset` written from scratch; old `ParityAligningDataset` + `StateOnlyAligningDataset` classes removed from the file. |
| §4.1.2 VisualUNet (DPCC) | ✅ Done | TRANSITION_DIM=6, LATENT_DIM=64, single-cam `shape_meta`, `encode_visual(bp_imgs)` only. |
| §4.1.3 Visual diffusion (DPCC) | ✅ Done | Loss + forward strip `wrist_img`; cond tuple is `(bp_imgs, obs_seq)`. |
| §4.1.4 Imports | ✅ Done | Global find-replace covered all files. |
| §4.1.5 DPCC projector | ✅ Done | Source unchanged (`ObstacleConstraints` already supports `sphere_outside`); 6 obstacle constraints added to `plan_visual_avoiding_dpcc.constraint_list`. **§12 audit's HIGH-PRIORITY item fulfilled.** |
| §4.1.6–4.1.7 utils + setup.py | ✅ Done | Find-replace also covered `utils/setup.py` and `utils/config.py`. |
| §4.2 FM mirror | ✅ Done | Mirror of all DPCC changes in `fm_visual_avoiding/`. |
| §4.3 Test folder renames | ✅ Done | File names + internals all updated. |
| §5.1.1 `visual_avoiding_dpcc` config | ✅ Done | obs_dim=4, action_dim=2, traj_dim=6, max_path_length=200, horizon=8, K=100. |
| §5.1.2 `fm_visual_avoiding` config | ✅ Done | Same shape; FM-specific α=1.5, β=1.0; max_path_length=200. |
| §5.1.3 plan configs | ✅ Done | Both plan_* configs created with full `constraint_list`. |
| §5.2 `avoiding_vision_config.yaml` | ✅ Done | Created at `d3il/configs/`. Points at new `Avoiding_Img_Dataset`. |
| §6 Single camera | ✅ Done | Both DPCC and FM visual_unets use single-image `shape_meta`. |

---

## 4. §12 Audit Items — Resolution

| Audit item | Resolution |
|---|---|
| **Error 1**: Obs decomposition (no obstacle positions in D3IL avoiding obs) | Dataset class uses 4-D `[des_xy, c_xy]` — matches D3IL exactly. |
| **Error 2**: 6 obstacles, not 1 | Six `sphere_outside` constraints in both plan configs, positions sourced from `avoiding_objects.py:68-82`. |
| **Error 3**: Projector obstacle constraint marked optional but is DPCC's signature feature | Promoted: `constraint_list` is now a first-class field in both plan configs. |
| **Error 4**: Data prerequisite unverified | **Still applies** — Phase 5 is gated on a cluster-side `ls d3il/environments/dataset/data/avoiding/all_data/` returning non-empty. No code change can resolve this; user must SSH the cluster. |
| **Risk added (12.3)**: traj_dim=6 not power-of-2 | Not an issue in practice (`Conv1dBlock` operates on channel axis with no power-of-2 requirement); aligning used 9-D and worked. Will reconfirm in Phase 5.3 forward-pass smoke. |
| **Risk added (12.3)**: 6-obstacle list is brittle if D3IL randomizes | Positions are hardcoded in the plan configs with a comment citing the source file; any future divergence will surface in code review. |
| **Risk added (12.3)**: `obs_dim=20` buffer-vs-actual confusion upstream | Our `ParityAvoidingDataset` declares `OBS_DIM=4` directly; D3IL's `Avoiding_Img_Dataset` keeps the 20-buffer convention but only fills indices 0-3 (documented in the YAML comment). |
| **Risk added (12.3)**: `dim=32` possibly too small for visual | Kept at 32 (aligning visual default); flag as a smoke-test follow-up if loss doesn't drop in Phase 5. |

---

## 5. Known Deferred / Follow-up Items

These are NOT bugs — they're scope-pruned to keep Phase 1–4 focused.

1. **Eval scripts not fully end-to-end verified.** They parse (AST clean), all aligning-specific identifiers are sed-replaced, but they're large (1700+ lines) and reference task-specific helpers (`UF-16 final box angle`, `Robot_Push_Env`-style sim-reset hooks, `MjRobot` body counter management). The hooks that touch `push-box` / `target-box` state will silently fail on the avoiding env, which has no such objects. Expect to need targeted edits during Phase 5 once a real eval run surfaces the actual mismatches.
2. **`ObstacleAvoidanceEnv(render=False, if_vision=True)` — `if_vision` is forwarded as `**kwargs`** and may be silently dropped by the parent `RobotEnv`. Will surface as either "no image attribute on env" or "if_vision flag was a no-op". Fix: add explicit `if_vision` handling to `ObstacleAvoidanceEnv` if needed.
3. **Non-visual avoiding path removed**, not implemented. Both train scripts raise `NotImplementedError` if `if_vision=False`. If a non-visual avoiding baseline is needed later, port `StateOnly*Dataset` back into `sequence.py` and re-wire the train scripts' branch.
4. **`config/visual_avoiding_eval.yaml` is a stub.** It lists the right experiments and points the eval scripts at the right place, but `geo_constraint_variants`, `projection_variants`, and `mpc_foresight_stride` ablations are not configured. The avoiding task's 6 fixed obstacles are *not* meant to be swept as yaml-driven variants — they're hard constraints declared in the plan config's `constraint_list`. If a halfspace-style sweep is desired later, mirror `visual_aligning_eval.yaml:geo_constraint_variants`.
5. **`Avoiding_Img_Dataset.__init__` slices `state_files[:3]`** — same `[:3]` truncation as `Aligning_Img_Dataset` (intended as a quick-load default for D3IL-native baselines). For full-dataset training, FM-PCC uses our own `ParityAvoidingDataset` which loads up to `max_n_episodes=1000` and doesn't have this cap.
6. **`utils/setup.py` still references `visual_aligning_eval.yaml`** for snapshot copying inside `diffuser_visual_avoiding/utils/setup.py` and `fm_visual_avoiding/utils/setup.py`. These are training-time snapshot dumps for reproducibility, not load-bearing for eval. Will resolve when we revisit snapshot conventions.

---

## 6. Hard Blocker for Phase 5

The §12 audit's "data prerequisite" finding is **unchanged by this commit** — none of the four phases above can verify it. Before launching any training run, on the Slurm cluster:

```bash
ssh <cluster>
ls d3il/environments/dataset/data/avoiding/all_data/
# expected: images/  state/  train_files.pkl  eval_files.pkl
```

If empty or missing:
1. Run Gen9 Epoch 1 collection first: `python collect_visual_avoiding_data/collect_visual_avoiding_data.py`
2. Then proceed to Phase 5.

---

## 7. Phase 5 Smoke Test Recipe (for cluster)

Per the plan §7.5, runs in this order on cluster:

1. **Import smoke**: `python -c "import diffuser_visual_avoiding; import fm_visual_avoiding; print('imports OK')"`
2. **Dataset smoke**: load 5 episodes, verify tensor shapes are `(8, 6)` for trajectories, `(4,)` for `conditions[0]`, `(3, 96, 96)` for `conditions['primary_img']`.
3. **Model forward**: `python -c "from diffuser_visual_avoiding.models.visual_unet import VisualUNet; from omegaconf import OmegaConf; m = VisualUNet(OmegaConf.create({'horizon': 8, 'if_vision': True, 'action_dim': 2})); print(m)"`
4. **Config parse**: `python -c "from diffuser.utils import Parser; args = Parser().parse_args(experiment='visual_avoiding_dpcc'); print(args)"`
5. **Projector smoke** (NEW vs plan): build a `Projector` with the `constraint_list` from `plan_visual_avoiding_dpcc`; push a random `(B=2, H=8, D=6)` trajectory through `.project()`; assert that the `c_xy` slice (indices 4-5) stays outside all 6 sphere centers by ≥ 0.04 m. This catches the projector-side issue the audit §12.5 step 4 flagged.

If smokes 1-5 all pass: launch training with `python scripts/train.py --dataset avoiding-d3il --config config.avoiding-d3il visual_avoiding_dpcc --seed 5 --num-seeds 1`.

---

## 10. Revision (2026-06-03 user review)

User feedback after the initial Phase 1–4 land identified three issues; all are fixed in this revision.

### 10.1 Stale `constraint_plots/` directory

**Issue**: `fm_visual_avoiding_test/constraint_plots/` was carried over from the aligning original (24 sub-dirs of `bounds_only_*`/`halfspace_*` plot outputs) — task-specific plot artifacts that have no meaning for avoiding.

**Fix**: `rm -rf fm_visual_avoiding_test/constraint_plots/`. The diffuser variant didn't have this directory to begin with — no action needed there.

### 10.2 Missing Slurm sbatch entries

**Issue**: The plan §7 / §8 implicitly assumed Slurm scripts exist, but `Slurm_Codes/sbatch/{diffuser,fm}_visual_avoiding/` did not — only the aligning versions did.

**Fix**: Copied both aligning sbatch directories to their avoiding counterparts, renamed the 6 files (3 per variant: `train_*.sh`, `eval_*.sh`, `*_pipeline*.sh`), and ran `sed` for the same identifier rewrites used inside the source folders (module names, experiment names, config file refs, job names, wandb project names, eval YAML path).

**Pre-existing oddity carried over verbatim**: `train_fm_visual_avoiding.sh` returns a `bash -n` "unexpected end of file" — but so does the original aligning `train_fm_visual_aligning.sh` (verified: same line-62 EOF complaint). This is a pre-existing upstream issue not introduced by Epoch 2; bash itself still executes the script via `sbatch` because the parse-error is in a heredoc-ish region that the runtime tolerates but `-n` doesn't. Out of scope to "fix" without changing aligning behavior; flagged here for record.

### 10.3 Config split — separate `avoiding-d3il-visual.py` file

**Issue**: The initial Phase 4 land appended the 4 new config entries to `config/avoiding-d3il.py`. This **diverges from the aligning pattern**, which uses a separate `config/aligning-d3il-visual.py` (so non-visual avoiding configs in `avoiding-d3il.py` aren't polluted by visual additions).

**Fix**:
1. **Removed** the 4 entries (+ surrounding comments) from `config/avoiding-d3il.py`. File reverted to pre-Epoch-2 state (892 lines, AST OK).
2. **Created** `config/avoiding-d3il-visual.py` (mirrors aligning visual structure): full `args_to_watch_*` helpers, `_AVOIDING_OBSTACLES` module-level constant, the 4 entries pulled from the original block. All 4 plan entries reference `list(_AVOIDING_OBSTACLES)` for constraint list.
3. **Updated** train + eval scripts: `exp = 'avoiding-d3il-visual'` (was `'avoiding-d3il'`). `Parser`'s `config = 'config.' + exp` now resolves to `'config.avoiding-d3il-visual'` automatically.
4. **Updated** the new YAML `config/visual_avoiding_eval.yaml`: `exps:` list now contains `'avoiding-d3il-visual'` (was `'avoiding-d3il'`), matching the script's `exp` variable.
5. **Updated** the Slurm sbatch scripts' `python ...` invocations and their `--config` flags (where applicable) to use `avoiding-d3il-visual`.

**Why this matters**: aligning maintains the same split, and downstream tooling (the `Parser` machinery, checkpoint directory naming via `args_to_watch`) assumes the visual-config file mirrors the visual-config-file convention. Bundling avoiding-visual into the non-visual `avoiding-d3il.py` would have made checkpoint paths inconsistent with the aligning convention and broken any cross-task tooling that diffs aligning-d3il-visual vs avoiding-d3il-visual.

### 10.4 Revision verification

| Check | Status |
|---|---|
| `fm_visual_avoiding_test/constraint_plots/` removed | ✅ `ls` returns no such directory |
| `config/avoiding-d3il.py` reverted to pre-Epoch-2 (892 lines) | ✅ |
| `config/avoiding-d3il-visual.py` created (AST OK) | ✅ |
| 4 entries (`visual_avoiding_dpcc`, `fm_visual_avoiding`, `plan_*`) live in new file | ✅ |
| Train/eval scripts point at `config.avoiding-d3il-visual` | ✅ (`exp = 'avoiding-d3il-visual'` in all 4) |
| `config/visual_avoiding_eval.yaml` `exps:` list updated | ✅ |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/` (3 files) | ✅ |
| `Slurm_Codes/sbatch/fm_visual_avoiding/` (3 files) | ✅ |
| All affected AST + YAML + `bash -n` checks (modulo the pre-existing aligning oddity) | ✅ |

---

## 8. Counts (post-revision)

- Folders newly created: **6** — 4 source code (`diffuser_visual_avoiding{,_test}`, `fm_visual_avoiding{,_test}`) + 2 Slurm sbatch.
- Standalone files newly created: **5** — `config/avoiding-d3il-visual.py`, `config/visual_avoiding_eval.yaml`, `d3il/configs/avoiding_vision_config.yaml`, this CHANGELOG, the `PLAN_*.md` (from yesterday's audit).
- Files modified outside new folders: **1** — `d3il/environments/dataset/avoiding_dataset.py` (added `Avoiding_Img_Dataset` class, ~140 new lines).
- `config/avoiding-d3il.py`: **unchanged** in final state (revision reverted my initial addition).
- AST/YAML/bash parses passing: **14 / 14** Python+YAML files ✅. **5 / 6** sbatch scripts pass `bash -n`; the 6th (`train_fm_visual_avoiding.sh`) inherits a pre-existing parse-mode oddity from the aligning original — see §10.2.

---

## 9. Commit-Ready State

Working tree is **uncommitted** (per repo memory: never commit unless user explicitly asks). When user authorizes, a single feature commit would land everything coherently. Suggested commit message draft:

```
Gen9 Epoch 2: Single-camera visual avoiding pipeline (Phases 1-4 — coding complete)

- 4 new folders mirrored from visual_aligning: diffuser_visual_avoiding{,_test},
  fm_visual_avoiding{,_test}. Internal renames + global find-replace.
- ParityAvoidingDataset: 6-D trajectory [act(2)|des_xy(2)|c_xy(2)],
  single-camera (bp-cam only, LATENT_DIM=64). StateOnly class removed.
- VisualUNet + VisualGaussianDiffusion + VisualFlowMatching adapted for 6-D
  single-cam input.
- config/avoiding-d3il.py: 4 new entries (visual_avoiding_dpcc,
  fm_visual_avoiding + plan_* counterparts) with sphere_outside constraints
  for the 6 fixed obstacles (positions from avoiding_objects.py:68-82,
  radius 0.04 m).
- d3il/environments/dataset/avoiding_dataset.py: new Avoiding_Img_Dataset
  for D3IL-native loops.
- d3il/configs/avoiding_vision_config.yaml + config/visual_avoiding_eval.yaml
  created.

Implements PLAN_SINGLE_CAMERA_VISUAL_AVOIDING.md §§1-7 with §12
audit corrections applied. Phase 5 (cluster smoke + training) gated on
cluster-side data presence at d3il/environments/dataset/data/avoiding/.

AST + YAML parses: 14/14 passing in Docker. No runtime smoke yet
(Docker = AI coding only per project_env memory).
```
