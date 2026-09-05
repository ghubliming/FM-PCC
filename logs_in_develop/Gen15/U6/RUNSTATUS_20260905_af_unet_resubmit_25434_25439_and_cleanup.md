# Run status 2026-09-05 — the UAV AF-UNet resubmit (25434 / 25439), what it can answer, and what may be deleted

*Gen15 · U6 · source: `temp/0509` export (job logs to 2026-09-05 15:21 UTC, tree captured 17:12
local, DA batch `batch_uav_20260905_163742`). Successor to
[`RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md`](RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md).*

**Two answers, up front.**

1. **Does the UAV AF-UNet work? Mechanically, yes — proven.** α is genuinely on at 0.2 on a
   3.97 M U-Net, and the checkpoint/eval plumbing resolves. §2.
2. **Can it carry a DA against `mf_unet` or `af_sit` today? No — not a clean one.** The only
   finished AF-UNet arm is on `pillars`, and on that scene (a) every arm scores S&C = 0, and
   (b) **every available comparator was evaluated at a different checkpoint selector** — AF-UNet
   is `latest`, MF-UNet and AF-SiT are both `best`. §3. The fix is cheap and needs **no retrain**. §4.

---

## 1. What is queued right now

| pipeline | knobs given | train | evals | resolves to |
|---|---|---|---|---|
| **25434** 15:20:42 UTC | `UAV_MIX_BONE_AF=unet` only | **25435** | 25436 K1 / 25437 K2 / 25438 K5 | `…_ae0_bbunet/6`, eval `Eaf_K*_mpc4_pid_stopgo_T0.5` |
| **25439** 15:21:29 UTC | `BONE_AF=unet`, `AF_ALPHA_END=0.2`, `EPOCH=latest` | **25440** | 25441 K1 / 25442 K2 / 25443 K5 | `…_ae0.2_bbunet/6`, eval `Eaf_K*_…_EPlatest` |

All six children `(Dependency)`; both trains `(AssocGrpCpuLimit)` PENDING.

**Against what the predecessor's §5 actually prescribed** — two commands, both
`BONE_AF=unet AF_ALPHA_END=0.2 EPOCH=latest FMPCC_UAV_EVAL_TAG=u6unet_ae02`, on `s_curve "1 2 5"`
and on `pillars "1 2"`:

- **25439 = the `s_curve` command**, minus the eval tag. ✅
- **25434 was not prescribed here.** It is the α-off (`ae0`) U-Net control, the same shape as the
  `u6unet_ae0` submission of 09-03. Defensible as a control — see the mismatch in §5.
- The `pillars` command is **already satisfied** by 25393/25394/25395 (the git-pull timing accident
  of the predecessor's §7). Do not resubmit it.

Both pipeline logs print the U6 echo block before the START banner, so both chains see the knobs.
25434's shipped `⚠ af_alpha_end = 0.0 … ends on the MEANFLOW target` warning is expected for a
deliberate α-off control, not a fault.

## 2. The α gate is GREEN — first time on UAV

Evidence from `pillars` (job 25393/25394), the only AF-UNet arm that has finished end to end:

| gate | evidence | file |
|---|---|---|
| backbone is the U-Net | `[ AFTrajectoryModel ] backbone=unet … params=3,969,222 (3.97 M)` | `2026-09-04/00_11_47_uav_mix_train_25393.log:146-147` |
| α floored, not annealed to 0 | `alpha=0.2` at epochs 96–99; `wandb train/alpha 0.2` | `…25393.log:246-249, 311` |
| bootstrap branch actually taken | `discrete_frac = 0.5 / 0.375 / 0.375 / 0.25`; final `test/discrete_frac 0.51` | `…25393.log:246-249, 298` |
| eval loads the floored checkpoint | `checkpoint selector = 'latest' (source: env UAV_MIX_EPOCH)`; `alpha(step 100000) = 0.2000 … alpha-Flow objective ACTIVE` | `…00_11_47_uav_mix_eval_25394.log:14, 36-37` |
| path keys propagate | savepath `…AlphaFlowODE_9D_as1_ae0.2_bbunet/6`; results `Eaf_K1_…_EPlatest_u6unet_ae02` | `…25394.log:20, 48` |

U6 does what it was built to do. This is the UAV counterpart of the Gen14 U12 green gates.

## 3. Why a `pillars` DA cannot be run cleanly today

### 3.1 Two eval tags on this scene are lying about the checkpoint

`_EP<sel>` only enters the results path from commit `ca0eb314` (U6) onward, and pre-U6
`UAV_MIX_EPOCH` was **read by nothing**. So a folder whose *tag text* says "latest" but whose
*path* has no `_EPlatest` token was evaluated at `best`:

| arm | folder | `GIT REV` of the eval | selector line in log | actually loaded |
|---|---|---|---|---|
| AF-UNet α→0.2 | `…_ae0.2_bbunet/Eaf_K{1,2}_…_EPlatest_u6unet_ae02` | `ca0eb31` | `checkpoint selector = 'latest' (source: env UAV_MIX_EPOCH)` | **latest** ✅ |
| AF-SiT | `…_ae0_bbsit/Eaf_K{1,2}_…_u6sitlatest` | **`43d684c`** | *line absent* | **best** ❌ tag lies |
| MF-UNet | `…_bbunet/Emf_K*_…_fix16scaled` | pre-U6 | *no `_EP` token* | **best** ❌ |

(`temp/0509/2026-09-04/00_04_48_uav_mix_eval_2538{0,1}.log:5-6,14,19` — `EVAL_TAG=u6sitlatest`,
`GIT REV 43d684c`, and no selector line anywhere in either file.)

So **every comparator differs from the AF-UNet arm in two ways at once** — objective/backbone *and*
checkpoint selection. Checkpoint selection is exactly the variable U12 was built to control, so
this confound is not ignorable.

### 3.2 And `pillars` cannot rank on the primary axis anyway

All 24 `pillars` candidates in the batch — Gen11 included — score **S&C = 0.00 %** and
**collision-free = 0.00 %**, and the scene carries **no `diffusion` target arm**. Per the
benchmark hierarchy there is nothing to beat there. Secondary axes, seed 6, n = 10 rollouts:

| arm | ckpt | K | S&C | success | goal dist (m) | steps→goal | ms/replan |
|---|---|---|---|---|---|---|---|
| **AF-UNet α→0.2** (c42) | latest | 1 | 0.00 | 11.0 % | **0.509** | 507.0 | **69.2** |
| AF-SiT α=0, 10.0 M (c45) | best | 1 | 0.00 | 13.0 % | 0.711 | 486.0 | 57.7 |
| MF-UNet dp0.5 (c57) | best | 1 | 0.00 | 11.0 % | 0.651 | 478.3 | 94.6 |
| **AF-UNet α→0.2** (c43) | latest | 2 | 0.00 | 14.0 % | 0.547 | 479.8 | 79.3 |
| AF-SiT α=0, 10.0 M (c47) | best | 2 | 0.00 | 14.0 % | **0.496** | 495.3 | 53.1 |
| MF-UNet dp0.5 (c60) | best | 2 | 0.00 | **20.0 %** | 0.530 | 469.2 | 83.3 |

Read as a *sanity check* it is reassuring — the 3.97 M AF-UNet is in the same band as the 10.0 M
AF-SiT and the MF-UNet. Read as a *result* it is nothing: no S&C separation, n = 10, one seed,
and the selector confound of §3.1 sitting on top.

`s_curve` is the discriminating scene — it has the `diffusion` K20 target arm (candidate 78) — and
no AF-UNet run exists there yet. That is what 25440 is for.

## 4. To make `pillars` usable — eval only, no retrain

`diffusion_epoch: _UAV_EPOCH` lives in the shared `_UAV_PLAN` block (`config/uav_mix.py:305,317`),
which every engine's plan block spreads (`:483, :526, :554, :624`), and the comment there states it
is eval-only and lands beside the `best` pass rather than overwriting it. So the confound of §3.1
is removed by two eval-only K-sweeps against checkpoints that already exist:

```bash
# MF-UNet pillars at 'latest' — matched selector for the AF-UNet arm
UAV_MIX_EPOCH=latest FMPCC_UAV_EVAL_TAG=u12latest \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf pillars "6" "1 2"

# AF-SiT pillars at 'latest', on post-U6 code this time (the 09-04 pass silently used 'best')
UAV_MIX_BONE_AF=sit UAV_MIX_EPOCH=latest FMPCC_UAV_EVAL_TAG=u6sitlatest_real \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af pillars "6" "1 2"
```

Use a **new** tag on the AF-SiT pass — `u6sitlatest` is taken by the mislabelled `best` results and
reusing it would pool two different checkpoints under one folder.

Even after this, `pillars` still cannot produce a win claim (§3.2). It can support an
*architecture/parameter* observation — 3.97 M AF-UNet vs 10.0 M AF-SiT at matched K and matched
selector — and nothing stronger.

## 5. One mismatch in the queued pair

25434 has no `UAV_MIX_EPOCH`, so its evals load `best`; 25439 loads `latest`. If 25434 is meant as
the α-off control for 25439, the pair reproduces exactly the confound of §3.1 on `s_curve`. Either
resubmit 25434 with `UAV_MIX_EPOCH=latest`, or do not read the two against each other. Neither
carries `FMPCC_UAV_EVAL_TAG`; not fatal (the parent model dirs differ, so DA_UAV_v1 will not pool
them) but the `run_tag` axis comes back blank.

At α = 0 `af_diffusion.py` routes to the Gen3v6 MeanFlow body, so 25435 trains a MeanFlow objective
through the AF code path — a useful same-code-path control, at the cost of one full train (~2.9 h)
plus three evals.

## 6. Deletion — RE-VERIFIED 2026-09-05, and largely WITHDRAWN

Paths relative to `/u/home/llim/FMPCC/FM-PCC/`. **Two of the three claims in the first draft of
this section were wrong. Corrected below from the job logs.**

### 6.1 WITHDRAWN — the empty `s_curve` `ae0.2_bbunet` dir

Listing it was wrong: it is empty, so deleting frees nothing, and it is **25440's own output
directory**. It has since been deleted by hand; that is harmless and needs no repair — `mkdir()`
recreates it and prints `Made savepath` (`mix_uav/utils/serialization.py:16-18`, `setup.py:190`),
and with no `args.json` present `Parser.save` writes no `_resume_N` file (`setup.py:55-60`).

### 6.2 WITHDRAWN — the `s_curve` AF-SiT checkpoint is **not** corrupt

`logs/UAV_MIX/uav-s_curve/mix_uav_af/H8_…_ae0_bbsit` — 21 files, 917.3 MiB. **Keep it.**

The `*_resume_1.json` files are a *config-filename* artifact, not a training resume: `Parser.save`
renames when `args.json` already exists (`setup.py:55-60`). Job 25388 in fact **restarted from
scratch** — its first progress line is `Epoch 0 … alpha=1, lr=0.0001, step=999`
(`…00_05_59_uav_mix_train_25388.log:139`) and it ran a full 100 epochs to `step=1e+5, alpha=0,
discrete_frac=0`. Every periodic `state_*.pt` name it passed overwrote 25383's file of the same
name, and `state_best.pt` (19:09 UTC) and `losses.json` (20:22 UTC) both fall inside 25388's window.

So the directory holds **one clean 100-epoch run** — an AF-SiT arm at α = 0, i.e. MeanFlow-on-SiT,
10.0 M params. 25383's GPU-hours were wasted, but nothing it left behind survives. This is the only
`s_curve` AF-SiT checkpoint and is a legitimate backbone-ablation comparator for 25440's 3.97 M
AF-UNet. Delete only under disk pressure.

### 6.3 Narrowed — only the K=1 results are irreproducible

`logs/UAV_MIX/uav-s_curve/plans/mix_uav_af/H8_…_ae0_bbsit` — 1342 files, 471.8 MiB, DA candidates
75 / 76 / 77. Eval start times against 25388's training window (18:02:30 → 20:24:47 UTC):

| eval | K | started (UTC) | checkpoint it loaded | status |
|---|---|---|---|---|
| 25384 | 1 | **18:03:32** — 62 s into 25388 | 25383's `state_best`, **overwritten at 19:09 UTC** | ❌ cand 75 cannot be reproduced |
| 25385 | 2 | 20:24:48 — after 25388 ended | 25388's `state_best` | ✅ cand 76 valid |
| 25386 | 5 | 21:27:42 — after 25388 ended | 25388's `state_best` | ✅ cand 77 valid |

**The unattributed 17:09-local write is explained:** it is 25386's own completion —
`JOB END: Sat Sep 5 15:09:23 UTC` after a 17.7 h K=5 run, `Job completed successfully`
(`…00_05_16_uav_mix_eval_25386.log`). No aborted job ever touched this tree. The earlier
"check the provenance first" caveat is discharged.

Only this one folder is defensible to delete, and only because its numbers cannot be regenerated
from any surviving checkpoint:

```bash
rm -rf logs/UAV_MIX/uav-s_curve/plans/mix_uav_af/H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0_bbsit/Eaf_K1_mpc4_pid_stopgo_T0.5_u6unet_ae0   # 148.5 MiB
```

### 6.4 What genuinely remains wrong

The eval tag `u6unet_ae0` sits on a **SiT** model — a trap for anyone reading folder names. It is
cosmetic: DA_UAV_v1 records `backbone = sit` for all three candidates, so no number is misstated.

**Do not delete** `logs/UAV_MIX/uav-pillars/mix_uav_af/H8_…_ae0.2_bbunet/` or its plans subtree —
the only genuine α-on UAV checkpoint in existence (§2).

## 7. Code bug — needs a go-ahead, not fixed

`mix_uav/utils/setup.py:183-191` guards `plan_*` experiments against creating a ghost *plans* dir,
but the eval path still resolves the **training** savepath through the `else: self.mkdir(...)`
branch. A missing-checkpoint eval therefore creates an empty model directory and only then dies at
`mix_uav/utils/serialization.py:36` with `FileNotFoundError: …/dataset_config.pkl` — which is
exactly how the dir of §6.1 appeared. A pre-flight existence check on `dataset_config.pkl` would
fail fast with a readable message and leave no litter.

## 8. Queue ownership map — 2026-09-05 (post-resubmit `squeue`)

| job(s) | what | origin | action |
|---|---|---|---|
| 25416 RUNNING, 25417 PEND | `eval_mix_visual_aligning` | Gen14 VA `af` evals, `MIX_AF_ALPHA_END=0.2 / 0.05`, `MIX_EPOCH=latest`, `MIX_PROJ_T=0.2` | **keep** — feeds the U12 DA, unrelated to UAV |
| 25419–25424 PEND `uav_mix_eval` | `fm` / `mf` K=2 sweep, tag `u7hg`, scenes pillars / corridor / s_curve | children of `eval_k_sweep` 25410–25415 | **keep** — user's own U7 sweep, not from this file |
| **25435 + 25436/37/38** | `af s_curve` train + K1/K2/K5 evals, **`ae0`, no `UAV_MIX_EPOCH` ⇒ `best`** | pipeline 25434 | **CANCEL** — §5 mismatch; see below |
| **25440 + 25441/42/43** | `af s_curve` train + K1/K2/K5 evals, **`ae0.2`, `EPOCH=latest`** | pipeline 25439 | **keep** — this is the prescribed run |

Everything else in that `squeue` belongs to other users (`caoh`, `erene`, `linze`, `saju`).
Pipelines 25434 and 25439 have already exited; only their children are queued.

**Why cancel rather than resubmit 25435's chain.** Its purpose was the α-off control. But `s_curve`
already carries an MF-UNet arm at **K1 / K2 / K5 / K10 / K20** (DA candidates 71 / 73 / 74 / 70 / 72),
so the MeanFlow comparator exists and costs nothing. 25435 would add only a *same-code-path* α-off
variant — a nice-to-have — for one full train (~2.9 h) plus three evals on a queue already blocked
at `AssocGrpCpuLimit`. Recommend: cancel now, resubmit later with `UAV_MIX_EPOCH=latest` only if the
same-code-path control turns out to be needed.

**`s_curve` is a scene that can actually discriminate**, unlike `pillars`: `mf` K=2 reaches
S&C = 1.0 % with 71 % collision-free, `mf` K=20 reaches 82 % collision-free, and the `diffusion`
K20 target arm exists (candidates 64 and 78). That is why 25440 is the run worth waiting for.

**Residual confound on `s_curve`, same shape as §3.1:** the existing `mf` evals carry no `_EP`
token, so they ran at `best`, while 25441–25443 will run at `latest`. One eval-only sweep removes
it — no retrain, the checkpoint already exists:

```bash
UAV_MIX_EPOCH=latest FMPCC_UAV_EVAL_TAG=u12latest \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf s_curve "6" "1 2 5"
```

**Status 2026-09-05, after the cancel:** `scancel 25435 25436 25437 25438` was executed — they are
gone from `squeue` and 25435 never started, so no directory was created and nothing needs cleaning.
The only UAV `af` work outstanding is the **25440 chain** (train + evals 25441 / 25442 / 25443), all
still `PENDING` behind `AssocGrpCpuLimit` while 25416 holds the allocation.
