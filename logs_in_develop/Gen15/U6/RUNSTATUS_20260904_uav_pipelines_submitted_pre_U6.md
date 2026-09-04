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
