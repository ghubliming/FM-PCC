# Run status 2026-09-04 — the three UAV `af` pipelines were submitted from a **pre-U6** checkout

**Status: none of the AF-UNet runs exist. All three submitted pipelines will train the old
MeanFlow-on-SiT arm, and the two `s_curve` jobs will fight over the same directory.
Recommended: cancel, pull, resubmit.**

---

## 1. What was submitted

| pipeline job | scene | K list | children | log |
|---|---|---|---|---|
| 25382 (22:05:16 UTC) | `s_curve` | 1 2 5 | train **25383** → eval 25384 / 25385 / 25386 | `2026-09-04/00_05_16_uav_mix_ksweep_pipeline_25382.log` |
| 25387 (22:05:59 UTC) | `s_curve` | 1 2 5 | train **25388** → eval 25389 / 25390 / 25391 | `2026-09-04/00_05_59_uav_mix_ksweep_pipeline_25387.log` |
| 25392 (22:11:48 UTC) | `pillars` | 1 2 | train **25393** → eval 25394 / 25395 | `2026-09-04/00_11_47_uav_mix_ksweep_pipeline_25392.log` |

No train or eval log for 25383–25395 exists in the 2026-09-04 export (taken after 14:46 UTC, when
job 25380 finished), so none of them had started at export time. They are queued or early.

Separately, jobs **25379 → 25380 / 25381** were an *eval-only* K-sweep. That one is real and
finished (25380 K=1 complete; 25381 K=2 still running at export) — see §4.

## 2. The evidence that U6 is not on the cluster

`uav_mix_ksweep_pipeline.sh` prints an **unconditional** block for `ENGINE = af`, *before* the
START banner (`Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh:59-70`):

```
[ pipeline ] af bone = <bone>  (U6 default; 'sit' via UAV_MIX_BONE_AF=sit)
[ pipeline ]   ⚠  af_alpha_end = 0.0 (shipped): alpha snaps to EXACTLY 0 …
```

**All three pipeline logs go straight from the `libtinfo` warning to
`UAV-MIX K-SWEEP PIPELINE START`.** The block is absent ⇒ the script that ran is the pre-U6 one.

Corroborating: commit `ca0eb314` (Gen15 U6) is timestamped **2026-09-03 22:04:12 UTC**, ~1 min
before the first submission — and the jobs that *did* run in that window report
`GIT REV: 43d684c` (job 25376). `Slurm_Codes/submit.sh` performs **no** git sync; it submits
whatever is in the cluster checkout. The commit was made locally but never pulled on `i6-gpu-1`.

## 3. What that means for the three jobs

Pre-U6 `config/uav_mix.py` hardcodes the `af` arm:

```
git show 43d684cb:config/uav_mix.py
  432:  'imf_backbone': 'sit',      # mix_uav_af
  441:  'af_alpha_end': 0.0,
  465:  'imf_backbone': 'sit',      # plan_mix_uav_af
```

so `UAV_MIX_BONE_AF` / `UAV_MIX_AF_ALPHA_END` / `UAV_MIX_EPOCH` are **read by nothing**, even if
they were exported. Consequences:

1. **25383, 25388, 25393 are not AF-UNet runs.** They train `af` on the ~9.4 M SiT with
   `af_alpha_end = 0.0`, i.e. α snaps to exactly 0 from ~71.2 % of the budget ⇒ the **MeanFlow**
   objective, evaluated at `best`. That is precisely the arm U6 exists to replace.
2. **25383 and 25388 collide.** Identical engine / scene / seed and no path-differentiating knob ⇒
   both resolve the same savepath
   `logs/UAV_MIX/uav-s_curve/mix_uav_af/H8_D…AlphaFlowODE_9D_as1_ae0_bbsit/6`. Two trainers writing
   one checkpoint directory; whichever finishes last wins, and the six dependent evals will read a
   checkpoint neither of them fully owns.
3. Nine of the eleven queued jobs are wasted GPU time.

## 4. The eval-only job (25379–25381) is fine but mislabelled

It ran against the existing SiT checkpoint and completed K=1:

```
[ fix_16 ] SAFE_EPS_MODE='scaled' SAFE_EPS_FRAC='1e-3' EVAL_TAG='u6sitlatest'
… /mix_uav_af/H8_D…AlphaFlowODE_9D_as1_ae0_bbsit/Eaf_K1_mpc4_pid_stopgo_T0.5_u6sitlatest/6/
```

`_as1_ae0_bbsit` confirms SiT + α_end=0 (⇒ MeanFlow objective). The results folder carries **no
`EP` fragment**, which under U6 would be added by `UAV_MIX_EPOCH=latest` — so despite the
`u6sitlatest` run tag, **this eval loaded `best`, not `latest`.** The folder name says something
the run did not do. Treat it as a `best`-checkpoint MeanFlow-on-SiT row, or re-tag it.

Result for the record (pillars, K=1, seed 6, post-Fix_16): success 0.000 / success_relaxed 1.000 /
safe 1.000 / goal_reached 0.000 across all 10 projection variants; `track_err ≈ 0.34`;
`fm_ms = 6.4`, `proj_ms = 11.4`. At K=2 (25381, partial) `dpcc-r` reaches `goal_reached = 0.600`
but `safe = 0.000`. Consistent with the standing picture: **pillars is not solvable by any arm.**

## 5. Corrective sequence

```bash
# 1. cancel the pre-U6 chain (train + every dependent eval)
scancel 25383 25384 25385 25386 25388 25389 25390 25391 25393 25394 25395

# 2. bring the cluster checkout to the U6 commit
cd /data/home/llim/FMPCC/FM-PCC && git pull        # expect ca0eb314 or later

# 3. sanity-gate before spending a GPU
sbatch Slurm_Codes/sbatch/uav_mix/gates_mix_uav.sh   # G9 covers the U6 knobs

# 4. resubmit — and CHECK the pipeline log shows the U6 echo before the START banner:
#      [ pipeline ] UAV_MIX_BONE_AF=unet
#      [ pipeline ] af bone = unet  (U6 default; 'sit' via UAV_MIX_BONE_AF=sit)
UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest \
  FMPCC_UAV_EVAL_TAG=u6unet_ae02 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh \
  af s_curve 6 "" fm_only none "1 2 5"

UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest \
  FMPCC_UAV_EVAL_TAG=u6unet_ae02 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh \
  af pillars 6 "" fm_only none "1 2"
```

Checkpoint tree to expect afterwards: `…AlphaFlowODE_9D_as1_ae0.2_bbunet/6` (`_fmt` uses `%g`,
so the default `0.0` still renders `ae0` and every existing path stays byte-identical), results
tree `Eaf_K<k>_…_EPlatest_u6unet_ae02`. Neither exists today, so nothing is overwritten.

**Before reading any result**, confirm the α gate: `train/discrete_frac > 0` at the final epochs and
`alpha(step 100000) = 0.2000 … ACTIVE` in the eval log. If `discrete_frac == 0.0`, the run is
MeanFlow again and the folder name is lying.

## 6. Cross-reference

Gen14's parallel port **did** land — see
`logs_in_develop/Gen14/DA_20260904_Gen14_U12_alpha_floor_and_latest_checkpoint.md`. All four U12
gates are green there (α alive at `state_100000.pt`, `discrete_frac = 0.50173`, path safety, and a
correct fail-fast). Note its verdict before spending UAV GPU time: on `aligning-d3il-visual` the α
floor traded task progress for constraint satisfaction and did **not** reproduce the `avoiding`
win. Whether the UAV scenes behave like `avoiding` or like Visual Aligning is exactly what these
runs are for — and `pillars` still cannot rank anything (S&C = 0 for every arm, no `diffusion`
target arm for the scene), so `s_curve` is where the answer will come from.

---

## 7. RESOLUTION — 2026-09-05, from the `temp/0509` export

**§5 was followed only in part: step 2 (`git pull`) was done; step 1 (`scancel`) and step 4
(resubmit) were not.** Every job below is from the user's original **Sep 3 22:05–22:11** batch —
no `af` pipeline was ever resubmitted (the next `af`-family job ids, 25410–25415 on 09-05, are the
`fm`/`mf` `u7hg` sweep). The cluster was pulled to `ca0eb314` between 20:24 and 22:56 UTC on 09-04,
**while the queued children of §1 were still draining**. A Slurm job checks out nothing; it runs
whatever the working tree holds **at its start time**, so the pull cut the three pipelines in half:

| job | start (UTC) | `GIT REV` | savepath resolved | outcome |
|---|---|---|---|---|
| 25383 train `s_curve` | 09-04 15:40 | `43d684c` (pre-U6) | `…_as1_ae0_bbsit/6` | ran; α dead |
| 25384/85/86 eval `s_curve` | 18:03 / 20:24 / 21:27 | pre-U6 | same | ran; rows tagged `u6unet_ae0` — **tag lies**, this is SiT + α=0 |
| 25388 train `s_curve` | 09-04 18:02 | `43d684c` (pre-U6) | `…_as1_ae0_bbsit/6` | ran; **collided** with 25383 as predicted in §3.2 |
| 25389/90/91 eval `s_curve` | 09-04 22:56 | **`ca0eb31`** (U6) | `…_as1_ae0.2_bbunet/6` | ❌ **crashed in 5 s** — `FileNotFoundError: …/dataset_config.pkl` |
| 25393 train `pillars` | 09-04 22:56 | **`ca0eb31`** (U6) | `…_as1_ae0.2_bbunet/6` | ✅ trained the real arm |
| 25394 / 25395 eval `pillars` | 09-05 01:50 / 02:59 | **`ca0eb31`** | `Eaf_K{1,2}_…_EPlatest_u6unet_ae02` | ✅ complete, 10 variants × 10 rollouts |

**This corrects §1's guess** that all three pipelines would produce pre-U6 arms. The *pipelines*
were pre-U6 (§2 stands — the U6 echo block is absent from all three stubs, and all three started
Sep 3 22:05–22:11), but job 25392's children happened to start after the pull and therefore ran
U6 code. The `pillars` α-Flow arm exists because of that timing accident, not because it was
submitted correctly.

### 7.1 What is real

`logs/UAV_MIX/uav-pillars/plans/mix_uav_af/H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet/Eaf_K{1,2}_mpc4_pid_stopgo_T0.5_EPlatest_u6unet_ae02`

`ae0.2` in the **checkpoint** dir (α floored at train time) and `EPlatest` in the **results** dir
(step-100 000 save, not `best`) — both U6 knobs took. First genuine α-Flow arm on UAV.

### 7.2 What is still missing, and why

| gap | classification | cause |
|---|---|---|
| `s_curve` α-Flow (K 1/2/5) | **lacking — will never appear** | train ran pre-U6, evals ran post-U6; the eval crashed on the absent checkpoint. Needs a fresh train+eval. |
| `corridor` α-Flow | **lacking — never submitted** | no `corridor` pipeline in the batch at all |
| `pillars` α-Flow **K=5** | **lacking — never submitted** | 25392's K list was `1 2`; the `s_curve` pipelines carried `1 2 5` |
| `fm`/`mf` K=2 × 3 scenes, tag `u7hg` (25419–25424) | **unknown — needs `sacct`** | parents 25410–25415 started 09-05 04:13 and scheduled the six children; no child log and no `u7hg` row anywhere in the 16:38 batch |

### 7.3 Process note — and a gap in §5

Both failure modes here are the same root cause: **`git pull` on the cluster while jobs are
queued**. `submit.sh` performs no git sync and Slurm holds no snapshot, so a chained pipeline can
straddle two revisions — the exact hazard that produced the train/eval savepath mismatch.

**§5 under-specified this.** It listed `scancel` as step 1 and `git pull` as step 2, but it never
said *why the order matters*, so running step 2 alone looks harmless — it is not. Jobs 25389–25391
are literally in §5's `scancel` list; had step 1 run, they would have been cancelled instead of
crashing, and 25393–25395 would have been resubmitted cleanly rather than succeeding by accident.
Any future corrective sequence in this repo must state the hazard inline, not just the order:

> ⚠ **Never `git pull` on the cluster with jobs queued.** `squeue -u $USER` first. A queued job
> runs the tree as of its *start* time, so a pull mid-drain can hand a pipeline's train and eval
> stages two different revisions — and two different savepaths.

### 7.4 Attribution — these were prescribed commands, and the prescription was defective

The Sep-3 batch was not improvised. Four of the five jobs are verbatim from the command block in
`CHANGELOG_Gen15_U6_af_unet_default_and_alpha_epoch_knobs.md`:

| job | changelog line | role |
|---|---|---|
| 25379 | `:233-234` | Stage 1, the free re-eval of the SiT tree |
| 25382 | `:265-267` | s_curve **arm A** — architecture fix alone, α off |
| 25387 | `:270-273` | s_curve **arm B** — α actually on. → the 25389/90/91 crashes |
| 25392 | `:284-287` | pillars **arm B** |

(pillars arm A, `:280-281`, was never submitted.)

So 25382 and 25387 are **not a duplicate submission**. They are arms A and B of a deliberate
α-ablation, written as consecutive commands, which is why they landed 43 s apart. With the knobs
inert on the pre-U6 tree they collapsed onto one savepath: trains 25383 and 25388 **both wrote
`…_as1_ae0_bbsit/6/`**.

**Correction (2026-09-05, from `uav_mix_tree.txt`).** An earlier revision of this section called
those three s_curve rows *contaminated*. That was stronger than the evidence. The collision was
benign on the config side:

* 25388 wrote `dataset_config_resume_1.json`, `model_config_resume_1.json`,
  `diffusion_config_resume_1.json`, `trainer_config_resume_1.json` — **`.json` only, no `.pkl`.**
  It took the trainer's *resume* branch, not a clobber.
* `load_diffusion` reads the **`.pkl`** configs. Those are still 25383's originals, untouched.
* The two trainers overlapped by **61 s** (25383 END 18:03:31, 25388 START 18:02:30), not for hours.

So evals 25384/85/86 are **mistagged, not corrupt** — the same category as `u6sitlatest` in §4. The
`u6unet_ae0` tag names an arm that was never trained (this is SiT + α_end=0). Read them as a
MeanFlow-on-SiT s_curve row, or retag. Do not delete them.

⚠️ **One unresolved anomaly.** `state_best.pt` is stamped 09-04 **21:09** and the periodic set
`state_<N>.pt x5` **21:32** — both *after* 25388 exited at 20:24 — and the span is `[0..80000]`
while both trainers' configs request `n_train_steps: 100000`. Nothing in the 09-04 log set accounts
for a writer in that window. Re-verify these weights (step count in `losses.json`, and an
`--epoch` echo in a fresh eval) before publishing any number derived from them.

**The defect is in the handover.** The changelog's command block opens directly with
`UAV_MIX_BONE_AF=… ./Slurm_Codes/submit.sh …`. It has **no step 0**. `submit.sh` performs no git
sync, commit `ca0eb314` was authored 22:04:12 UTC, and the block was executed from 22:05:16 — the
race was all but guaranteed by the ordering as written. A command block that depends on a commit
must say so in the block itself; a `⚠` further down the page does not travel with the copy-paste.

**Rule adopted (applied in the U7 block, `CHANGELOG_20260904_honest_geometry_and_slack_gate.md`):**
every cluster command block in this repo begins with the sync-and-verify pair, inside the fenced
block, above the first `submit.sh`:

```bash
cd /path/to/FM-PCC && git pull && git log --oneline -1   # must show the commit this block needs
squeue -u "$USER"                                        # must be EMPTY, or drain/scancel first
```

The second line is the half §7.3 identifies as the real hazard: pulling is safe only when nothing
is queued.

**Follow-up:** [`RUNSTATUS_20260905_af_unet_resubmit_25434_25439_and_cleanup.md`](RUNSTATUS_20260905_af_unet_resubmit_25434_25439_and_cleanup.md)
