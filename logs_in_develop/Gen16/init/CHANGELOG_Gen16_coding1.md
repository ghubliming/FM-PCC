# CHANGELOG — Gen16 (Visual-Avoiding Mix-ML) · coding pass 1

**Date:** 2026-08-21 · **Status:** code complete, **NOTHING HAS RUN ON HARDWARE.**
This container has no Python packages; every check below is stdlib-level. All execution is a
cluster job on i6-gpu-1.
**Plan:** [`PLAN_Gen16_visual_avoiding_mix_ml.md`](./PLAN_Gen16_visual_avoiding_mix_ml.md)

Gen9's visual-avoiding task now takes the ML objective as a config switch:
`--engine diffusion | fm | mf | af` → visual DDPM (Gen6V4) · visual Flow Matching (Gen7) ·
MeanFlow (Gen3v6) · α-Flow (Gen3v7) — on a locked `VisualUNet` backbone, with the DPCC
projector (arm B) and the HardFlow in-loop sampler (arm C) both available.

**This answers the question the generation was opened for: how do MeanFlow and α-Flow behave
in visual *avoiding*, on the architecture-matched U-Net bone.**

---

## 1. What was decided, and why not the two obvious alternatives

| Option | Verdict |
|---|---|
| **Upgrade Gen9 in place** (`fm_visual_avoiding/`) | ❌ Gen9's eval is 643 lines; Gen14's is 3168. Upgrading means hand-back-porting ~2500 lines of eval machinery (DPCC variant harness, HardFlow arm C, HF fan parity, provenance, receding-horizon cadence) into the folder that has to produce paper numbers. |
| **Merge Gen9 into Gen14** | ❌ Gen14 is the most active arm in the repo (commits 2026-08-20/21). Folding a second environment into it breaks the sibling convention and puts the aligning results at regression risk for zero gain. |
| **New sibling: Gen14 @ HEAD + the avoiding delta** | ✅ **Chosen.** The same move Gen15 made for UAV. Measured first (§2), then executed. |

**On "there is a rebuild anyway" (GEN_X):** GEN_X is a one-file DRAFT concept, unstarted.
Results are not blocked on it. And this port is not wasted work — `visual_avoiding` is already
axis-3 of the rebuild's experiment matrix, so a registry-driven avoiding sibling with
dims/cameras in one module *is* the pilot for GEN_X's environment axis.

---

## 2. The measurement that made this cheap

Before writing anything, the aligning→avoiding delta was measured on the Gen7 pair
(`fm_visual_aligning/` vs `fm_visual_avoiding/`): **8 files differ, ~640 diff lines**, all
localised.

| file | Δ lines | what it is |
|---|---|---|
| `datasets/sequence.py` | 282 | avoiding dataset, 6-D |
| `sampling/projection.py` | 110 | ⚠️ **not an avoiding delta** — Gen9 is BEHIND (§4.3) |
| `models/visual_unet.py` | 83 | `TRANSITION_DIM 9→6`, `LATENT_DIM 128→64`, single cam |
| `models/visual_gaussian_diffusion.py` | 73 | cond tuple drops `wrist_img` |
| `utils/training.py` | 47 | episode split + EMA test scoring |
| `utils/setup.py` | 28 | snapshot yaml name |
| `utils/config.py`, `models/helpers.py`, `models/diffusion.py`, `datasets/__init__.py` | 14 | package rename only |

The train-script delta was 54 lines.

---

## 3. Impact on existing code: **NONE**

```
$ git status --porcelain
?? config/avoiding-d3il-visual-mix.py
?? config/visual_avoiding_mix_eval.yaml
?? mix_visual_avoiding/
?? mix_visual_avoiding_test/
?? Slurm_Codes/sbatch/mix_visual_avoiding/
```

**Zero modified files.** Not even append-only: Gen14's coding pass had to append 267 lines to
`config/aligning-d3il-visual.py`; Gen16 ships its own config module instead, so
`config/avoiding-d3il-visual.py` (Gen9), `config/aligning-d3il-visual.py` (Gen14) and
`config/visual_avoiding_eval.yaml` are **byte-untouched**. Every existing checkpoint and
results path in Gen7 / Gen9 / Gen14 resolves exactly as before.

Gen16 writes to `logs/avoiding-d3il-visual-mix/…` and
`mix_visual_avoiding_<engine>/…` exclusively.

---

## 4. Files created

### 4.1 `mix_visual_avoiding/` — 47 `.py`

Copy of `mix_visual_aligning/` @ HEAD (`c19bbde2`), package prefix repointed, then **18
declared edits**. Gate A0 enumerates them and asserts every *other* file is byte-identical
to Gen14's after reversing the rename. **A0 currently reports 30 files clean / 18 edits.**

| kind | file | what changed |
|---|---|---|
| **NEW** | `models/visual_spec.py` | the single source of truth (§5.1) |
| **NEW** | `sampling/policies.py` | `VisualPolicy` / `VisualHardFlowPolicy` (§5.2) |
| spec-driven | `models/visual_unet.py`, `visual_unet_twotime.py`, `visual_dit_twotime.py` | encoder + dims from the spec; `encode_visual(*cam_imgs)` |
| spec-driven | `models/visual_{gaussian,fm,mf,af}_diffusion.py` | cond packing via the spec; `_encode_once(*cam_imgs)` |
| replaced | `datasets/sequence.py` | `ParityAvoidingDataset` (from Gen9) + `episode_split`, spec-driven |
| edited | `utils/training.py`, `utils/training_twotime.py` | §5.3 — the same three edits in both |
| edited | `utils/serialization.py` | `DiffusionExperiment` gains `ema` |
| edited | `utils/setup.py` | snapshot follows `FMPCC_PROJ_CFG` (§4.3) |
| edited | `sampling/hardflow_projection.py` | spec-driven cond + the `fm` fix (§5.4) |
| edited | `models/{__init__,engine_registry}.py` | exports / provenance strings only |

### 4.2 `mix_visual_avoiding_test/` — 3 `.py`

| file | lines | provenance |
|---|---|---|
| `train_mix_visual_avoiding.py` | 610 | Gen14's train script + 4 blocks marked `Gen16` |
| `eval_mix_visual_avoiding.py` | 780 | **NOT Gen14's** — see §5.5 |
| `gates_mix_visual_avoiding.py` | 640 | new, the A0–A9 battery |

### 4.3 Config + Slurm

- `config/avoiding-d3il-visual-mix.py` (**new, 520 ln**) — Gen14's mix helpers copy-modified
  for avoiding. *Copied, not imported*, following Gen15's `config/uav_mix.py` precedent.
- `config/visual_avoiding_mix_eval.yaml` (**new**) — Gen9's avoiding geometry **byte-for-byte**
  + the 13-variant three-arm list + the `hardflow` block.
- `Slurm_Codes/sbatch/mix_visual_avoiding/` — the Gen14 quartet + a K-sweep submitter.

**Three latent bugs were NOT carried across**, because taking Gen14 @ HEAD as the base means
Gen16 gets the newer file wherever Gen9 was behind:

1. `sampling/projection.py` — Gen9 lacks Fix_15.2's sustained-slowness circuit breaker
   entirely. The 110-line "delta" in §2 is Gen14 being *ahead*, not an avoiding difference.
   Gen16 takes Gen14's verbatim; the file has **zero** task-specific content.
2. `utils/setup.py` — Gen9 hardcoded the snapshot filename and had to ship a cleanup for the
   stale wrong-task file it used to write. Gen16 reads `FMPCC_PROJ_CFG`, so the snapshot
   follows `--config` and threshold sweeps for free. The stale-file cleanup is kept.
3. `checkpoint_epoch_best` handling arrives with Gen14's loader.

---

## 5. The five substantive pieces of engineering

### 5.1 `models/visual_spec.py` — the hoist

Gen14 spreads the task's observation spec across **nine files**: `TRANSITION_DIM` /
`LATENT_DIM` in three backbones, `'in_hand_image'` in the same three, and
`bp, inhand, obs_seq = cond[...]` in those three plus the four engine wrappers.

That was fine with one visual task. Gen16 is the second, and it differs in **exactly those
nine places**. Replaying the edit nine times per new ML bone is how a bone silently keeps the
other task's camera count.

So the spec is declared once:

```python
CAMERA_KEYS   = ('agentview_image',)   # bp-cam only
COND_IMG_KEYS = ('primary_img',)
LATENT_DIM    = N_CAMERAS * RGB_OUTPUT_SIZE   # 64 — DERIVED, not chosen
ACTION_DIM, OBS_DIM, TRANSITION_DIM = 2, 4, 6
```

plus `build_obs_encoder()`, `build_obs_dict()`, `pack_visual()`, `split_visual()`,
`images_from_conditions()`.

**The rule for this generation:** no module below `models/` may name a camera, a latent width
or a trajectory dimension. **Gate A2 enforces it** — it AST-strips comments and docstrings
(so the files can still *explain* what they no longer hardcode) and scans the remaining
executable text. It currently reports **47 modules spec-driven**.

Three consequences worth stating:

- `LATENT_DIM` is **derived**, not written down. Setting it by hand is how a camera count and
  a FiLM `cond_dim` drift apart into a shape error 40 minutes into a GPU allocation.
- `split_visual()` **raises** on a short payload rather than silently dropping a camera.
- Gen14's two-camera `(bp, inhand, obs_seq)` payload is exactly `pack_visual` at
  `N_CAMERAS == 2`. The convention is unchanged; only the arity moved, and only one module
  knows it. This is what makes the DiT/SiT bones (Gen14 U8) port for free.

### 5.2 `sampling/policies.py` — the bridge

Gen14 drives its engines through `VisualAgentWrapper`, ~700 lines shaped by D3IL's
`Aligning_Sim` callback protocol. **The avoiding task has no such harness** — it is a plain
gym loop, and its whole lineage (Gen3v2 → Gen3v6 → Gen3v7 → Gen12) drives it through
diffuser's `Policy`:

```python
action, Trajectories(actions, observations) = policy(
    conditions={0: obs}, batch_size=B, horizon=H, disable_projection=bool)
```

So Gen16 does not port Gen14's agent. It keeps the avoiding loop and supplies the **same
`Policy` surface** over the visual engines:

- `VisualPolicy` — arms A/B. Normalises the obs anchor, repeats the condition across the
  candidate fan, packs `{0: (*cam_seqs, obs_seq)}`, calls the engine, unnormalises, selects.
- `VisualHardFlowPolicy` — arm C. Wraps the HardFlow sampler. Has **no `projector` attribute
  at all**, so the double-projection mistake is not expressible.
- `VisualNormalizer` — the adapter. The visual train script pickles obs/act normalizers
  separately; the `Projector`, `build_hardflow_sampler` and this file all want one object
  with a `.normalizers` dict AND a `normalize(x, key)` method. Gen9's `ProjectorNormalizer`
  covered only the first. Covering all three is what lets the rollout loop stay unmodified.

The candidate-selection block is a faithful copy of Gen3v6's, **including the fix_5
`which_trajectory` / `executed_idx` split** — the two indices address different arrays
(`actions` is never reordered, `observations` is, under `-t`) and collapsing them re-creates
the bug where the executed plan is not the recorded plan.

### 5.3 The trainers — one split, one selection criterion

Both trainers got the **same three edits**, and the "same" is the point.

**(a) Episode-level split.** `ParityAvoidingDataset` yields overlapping sliding windows, so a
random *window* split puts near-duplicate windows from the same episode on both sides. The
test loss then reads far too low and `state_best` is selected against a leaked set. Gen9 U4
diagnosed this; `episode_split()` cuts on episode boundaries.

> 🔵 **This removes a standing Gen14 confound.** Gen14 PLAN §4 had to document that mf/af use
> a *seeded* random split while diffusion/fm use an *unseeded* one — "accept, document, and
> compare arms on unguided TASK SUCCESS, never on test_loss". `episode_split()` is
> deterministic (episode index, no RNG), so in Gen16 **all four arms train and validate on
> exactly the same windows** and `test_loss` is comparable across arms. `split_seed` stays
> wired for datasets without the method, and is inert here.

**(b) EMA-consistent `test()`.** Upstream DPCC scored `self.model` *and* deployed
`self.model` — self-consistent. Scoring raw while deploying EMA is not: `state_best` becomes
the step that was best for a network nobody runs. Both Gen16 trainers now score
`self.ema_model`, matching `eval_use_ema=True`. Revert path is documented in both files:
change **both**, never one.

**(c) Final save.** `self.step` is incremented *after* the periodic-save check, so the loop
only ever saved at `0, save_freq, …, n_train_steps − save_freq`. The last `save_freq` steps of
every run were discarded (Gen9 B8).

⚠️ **(b) and (c) are a deliberate divergence from the Gen3v6/v7 "verbatim" two-time trainer.**
The reasoning: checkpoint *selection* is not split-independent — it decides which weights get
deployed. Four arms selecting `state_best` under two different criteria is a confound in the
deployed weights themselves. Making the criterion identical everywhere is worth the small
loss of byte-identity with Gen3v7, and A0 records it as a declared edit rather than hiding it.

### 5.4 A real bug found in the Gen14 HardFlow port

`encode_visual_cond()` called `model._encode_once(...)` unconditionally. Only the **two-time**
wrappers define `_encode_once`; `VisualFlowMatching` does not. **Arm C on the `fm` host would
have raised `AttributeError` on its first step.**

Gen16 routes by capability: two-time hosts get a pre-encoded `visual_latent` (a JVP constant),
and `fm` gets `visual` — raw images, which `VisualUNet.forward` encodes itself. That is what
the non-HardFlow `fm` path already does, and `_VISUAL_COND_KEYS` already allow-lists both keys
so the blind-field guard accepts either. **Gate A8 pins both branches.**

*This is reported as found, not fixed upstream — Gen14 is untouched (§3). Worth mirroring
there if the aligning `fm` arm ever runs arm C.*

### 5.5 The eval is built on the AVOIDING harness, not Gen14's

This is the decision most likely to be second-guessed later, so it is recorded explicitly.

Gen14's 3168-line eval is aligning-specific in depth: `Aligning_Sim`, context management, box
geometry (`_box_obstacle_overlap`, `_scan_box_obstacle_conflicts`), box drawing, gif export.
Almost none of it transfers.

`eval_mix_visual_avoiding.py` is therefore structured on
**`FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` @ HEAD** — the mature avoiding
harness — with three things layered on:

1. `--engine` dispatch through `engine_registry` (from Gen14)
2. the visual policies of §5.2 instead of diffuser's `Policy`
3. the bp-cam frame captured per env step (from Gen9 Epoch 2)

Everything else is the avoiding lineage's, **deliberately unchanged**, so Gen16 numbers land
in the same frame as Gen3v6/Gen3v7's state-only numbers: the K sweep, the U10 receding-horizon
cadence, U10.1 run provenance, B4_PARITY arm-C fan resolution, the RTRecorder, the npz schema.

Carried forward verbatim, and worth knowing about:

- **B4_PARITY.** Arm C's fan defaults to 4 (the DPCC arms' `batch_size`), not 1. Both arms
  loop serially over candidates around a CPU solve; a mismatched fan made arm C look ~25%
  cheaper than DPCC when its per-solve cost is ~1.8–2.2× DPCC's. Gate A9 checks the yaml's
  `hardflow.batch_size` against the config's `mpc_batch_size` and runs **offline**, so a fan
  mismatch is caught before submitting.
- **`-c` at B>1 is a known-bad arm.** 49% timeouts across 750 B=4 avoiding cells. The warning
  banner is preserved verbatim.

Two Gen16 additions:

- `--flow-steps` is **refused on the `diffusion` arm** with an explanation.
  `n_diffusion_steps` is the DDPM chain length and a checkpoint-path key: changing it needs a
  retrain, not an eval flag.
- Arm-C variants are **skipped with a printed notice** on the `diffusion` arm rather than
  crashing, so ONE yaml serves all four engines. A DDPM reverse chain has no velocity field,
  so `hardflow_new` is undefined on it, not merely unsupported.

---

## 6. Gates

`mix_visual_avoiding_test/gates_mix_visual_avoiding.py` — ten gates, tiered by what they need.

| gate | what it proves | needs |
|---|---|---|
| **A0** | **copy fidelity** — every non-declared file is byte-identical to Gen14's | stdlib |
| A1 | spec coherence — the derived constants actually derive | stdlib |
| A2 | no stray camera/dim literals outside `visual_spec.py` | stdlib |
| A9 | arm-C fan == arm-B fan; the two activation gates agree | stdlib |
| A3 | registry wiring; no diffusion/fm entry reaches a two-time module | torch |
| A4 | `diffusion_loadpath` reproduces the train block's `exp_name`, per arm | torch |
| A6 | dataset constants and camera plumbing match the spec | torch |
| A5 | all three bones report the same dims/latent | GPU |
| **A7** | **four arms, one visual training step, single camera, finite loss** | GPU |
| A8 | HardFlow host policy + the §5.4 fm path | GPU-free torch |

**A0 is load-bearing**: Gen16's claim is "Gen14's frame, one task swapped". If A0 fails the
claim is false and every cross-generation comparison is suspect.

**A7 answers the research question** — mf/af differentiate the network with a forward-mode
JVP and keep the vision encoder out of it by pre-encoding the latent. If that repack is wrong
for a one-camera payload, A7 surfaces it in seconds instead of nine hours into a training job.

### Ran here (stdlib only)

```
$ python3 -m mix_visual_avoiding_test.gates_mix_visual_avoiding --gate offline
  A0: PASS    30 files byte-identical to Gen14, 18 declared edits
  A1: PASS    6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)] · 1 camera (agentview_image)
  A2: PASS    47 modules are spec-driven
  A9: PASS    arm-C fan 4 == arm-B fan 4; both gates at 0.5
```

**A4 was additionally verified off-cluster** by re-running the config module against a stubbed
`diffuser.utils.watch` (no torch). All four arms round-trip; the resolved paths are:

```
mix_visual_avoiding_diffusion/H8_K20_D…VisualGaussianDiffusion_aw10_VTrue_steps200_bs64_filmv1_Ediffusion
mix_visual_avoiding_fm/H8_D…VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps200_bs64_filmv1_Efm
mix_visual_avoiding_mf/H8_D…VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps200_bs64_filmv1_Emf_tslogit_normal
mix_visual_avoiding_af/H8_D…VisualAlphaFlow_a1.5_b1.0_aw1_…_Eaf_tslogit_normal_afschsigmoid
```

That check caught one defect — **in the gate, not the config**: A4 was comparing the loadpath
against `f'mix_visual_avoiding_{eng}/{train_name}'`, double-prefixing, because `train_name`
already carries the prefix fragment (`prefix` is the first watch entry, with an empty label,
and `watch()` collapses the `'/_'` join back to `'/'`). Fixed, with the reason written into
the gate so nobody re-adds it.

Syntax: every `.py` compiles under `py_compile`; every `.sh` passes `bash -n`.

---

## 7. What has NOT been verified — read before trusting a number

1. **Nothing has executed a tensor op.** A3–A8 have never run. Run
   `gates_mix_visual_avoiding.sh` before the first training job.
2. **The dataset has not been opened.** `collect_visual_avoiding_data` has not been touched
   since 2026-05-29. **Confirm `environments/dataset/data/avoiding/all_data/images/bp-cam/`
   still exists on i6-gpu-1 before submitting.** `_load_images` raises with a named cause if
   the on-disk resolution disagrees with `visual_spec.IMG_SHAPE` — deliberately, instead of
   silently resizing.
3. **No `ddpm`-arm parity check against Gen9 has been run.** `diffuser_visual_avoiding`
   (Gen9's DDPM baseline) trained and evaluated in June 2026. **The Gen16 `diffusion` arm
   should reproduce it, and that parity is the gate on the whole generation** — if it holds,
   the mf/af arms are architecture-matched by construction (same `VisualUNet`, same data,
   same split). Note the arms will NOT be bit-identical: §5.3(a)/(b) changed the split and the
   `state_best` criterion for every Gen16 arm. Compare on rollout metrics, not on loss curves.
4. **`n_trials: 30` × 13 variants × 3 halfspaces × 5 seeds is a large first run.** Consider
   `--seed 6` and a trimmed variant list for the first submission.
5. **No linear-dynamics `.npz` exists for this generation.** `dynamics_mode: deriv` is the
   default and needs none. Do NOT point `linear_dynamics_path` at another generation's file —
   it is fit on a different normalizer and would silently enforce the wrong physics.

---

## 8. How to run it

```bash
# 0. gates first — the whole battery
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_avoiding/gates_mix_visual_avoiding.sh

# 1. one arm, one seed, train -> eval chained
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_avoiding/mix_visual_avoiding_pipeline.sh mf "6"

# 2. the K sweep (the headline experiment) — SAME K list for every arm
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_avoiding/eval_k_sweep.sh mf "6 7" "1 2 5 10 20"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_avoiding/eval_k_sweep.sh af "6 7" "1 2 5 10 20"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_avoiding/eval_k_sweep.sh fm "6 7" "1 2 5 10 20"
```

Results: `logs/avoiding-d3il-visual-mix/plans/mix_visual_avoiding_<engine>/<ckpt-id>/<eval-id>/results/halfspace_<variant>/`

The npz carries `engine`, `ml_bone`, `flow_steps_K` and `replan_steps` alongside the metrics,
so a Data_Analysis sweep cannot pool two arms by mistake.

---

## 9. Reporting rules for this generation

- **Lead with the `unet` bone.** The DPCC baseline is a U-Net, so only the U-Net row is an
  unconfounded comparison. `MIX_BONE=*` DiT/SiT results are a secondary, confounded finding
  and must be labelled as such, with backbone + parameter count in every table.
- **The baseline is diffusion-DPCC**, and mf/af must also beat naive FM. Arm C (HardFlow) has
  to beat the DPCC projector at a *lower* projection threshold to mean anything.
- **"Good" means Pareto-dominant**: at equal success + constraints, fewer steps AND lower
  `avg_time`. Otherwise say "trade-off" / "non-dominated", never "best".
- **⚠️ Gen16 = DPCC math (arms A/B) + the HardFlow sampler as arm C. Gen13 (HF_Mix_ML) is
  built ON HardFlow and is a different mechanism. Never pool their results.**

---

## 10. Open items

- `MASTER_TEST_HISTORY.md` has **not** been touched. Gen16 needs a Master Trace Map row —
  say the word and I will add it.
- A `load_results_mix_visual_avoiding.py` was not written; the eval's npz schema matches the
  avoiding lineage's, so the existing `Data_Analysis` avoiding path should read it. Unverified.
- Gen14's §5.4 bug is unfixed in Gen14 itself, by design (isolation). Mirror it there if the
  aligning `fm` arm ever runs arm C.
