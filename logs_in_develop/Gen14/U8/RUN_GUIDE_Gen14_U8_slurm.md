# RUN GUIDE — Gen14 U8 visual DiT/SiT bone: cluster submission

> All commands run from the repo root on **i6-gpu-1**. Nothing here has been executed —
> this container has no Python. See [`CHANGELOG_Gen14_U8_visual_dit_bone.md`](./CHANGELOG_Gen14_U8_visual_dit_bone.md) §6.

---

## 0. 🔴 Step 1 — gates, BEFORE any training

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh bone
```

~1 min on one GPU. Runs G-B1/B2/B3/B4-5/B6/B7 and nothing else. **This is the first
execution of any U8 code.** Do not skip it:

| Gate | Catches |
|:--|:--|
| G-B1 | the visual token leaking into state-only builds (would break Gen3v4/v6/v7) |
| G-B2 | an unmatched bone (~9.9 M vs ~4.0 M) — a confounded A/B |
| G-B3 | an image-blind model that still reports `if_vision=True` |
| G-B4/5 | the JVP / α-Flow bootstrap dying on the extra token |
| **G-B6** | **a half-applied prefix bump — trains fine, reads the WRONG positions** |
| G-B7 | checkpoint-path collision, and `_film..` on a DiT path |

Exit code is non-zero if any gate fails, so it is safe to chain. Run the full suite too if you
want the pre-U8 gates re-confirmed:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh all
```

---

## 1. The bone knob

| Env var | Effect |
|:--|:--|
| `MIX_BONE_MF=<bone>` | mf arm only — `unet` \| `mf_dit` \| `dit` |
| `MIX_BONE_AF=<bone>` | af arm only — `unet` \| `sit` \| `dit` |
| `MIX_BONE=<bone>` | both two-time arms (use when the mf-vs-af comparison must stay controlled) |
| *(unset)* | `unet` — the pre-U8 baseline, paths byte-identical to existing runs |

A bone that belongs to the other arm (`sit` on mf, `mf_dit` on af) is **rejected**, in the
sbatch script and again in the config.

---

## 2. Train + eval, chained, one job per seed (the normal path)

`mix_visual_aligning_pipeline.sh` submits `gates ──> train(seed) ──> eval(seed)` with
`--dependency=afterok`, fanned out per seed so each gets its own 24 h wall.

```bash
# mf arm on the official MeanFlow DiT, all five seeds
MIX_BONE_MF=mf_dit \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh \
  mf "6 7 8 9 10"

# af arm on alpha-Flow's own SiT
MIX_BONE_AF=sit \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh \
  af "6 7 8 9 10"

# both arms on the SAME bone (the controlled mf-vs-af comparison) — two submissions
MIX_BONE=dit ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf "6 7 8 9 10"
MIX_BONE=dit ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af "6 7 8 9 10"
```

Args: `$1` engine, `$2` seeds, `$3` NFE override (blank ⇒ config default; mf/af = 2).

### 2.1 A first, cheap smoke run

One seed end-to-end before committing five:

```bash
MIX_BONE_MF=mf_dit \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh \
  mf "6"
```

---

## 3. Train and eval separately

```bash
# train one seed
MIX_BONE_MF=mf_dit \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/train_mix_visual_aligning.sh mf 6

# eval that checkpoint  ($3 = record_mode, default all)
MIX_BONE_MF=mf_dit \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6 all
```

🔴 **The eval needs the same `MIX_BONE_*` as the train.** The bone is baked into
`diffusion_loadpath` (`..._B{bone}_E{arm}`), and a DiT path carries no `_film..` fragment —
so an eval without it resolves to the U-Net directory or to nothing. The backbone itself is
always rebuilt from the train-time `model_config.pkl`, so the architecture can never diverge
from the weights: the failure mode is a wrong/missing **path**, not wrong math.

---

## 4. The baseline arm to compare against

The U-Net rows are the headline (PLAN §11). If you do not already have current U-Net runs for
these seeds, submit them the same way with the bone unset:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf "6 7 8 9 10"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af "6 7 8 9 10"
```

These land in the **existing** pre-U8 checkpoint trees — U8 did not move them.

---

## 5. What to check in the first log

```
[ train ] ml_bone = mf_dit — VisualDiTTwoTime, visual latent enters as ONE PREPENDED TOKEN
          (hidden=160, depth=8); parameter-matched to the ~4.0M U-Net
[ VisualDiTTwoTime ] MultiImageObsEncoder initialized — LATENT_DIM=128 ...
[ VisualDiTTwoTime ] bone=mf_dit (MFDiTOfficialTrajectory) hidden=160 depth=8 heads=4
                     patch=1 cond_dim=128 (visual token ON)
[ MFTrajectoryModel ] backbone=mf_dit vision=True ... params=X.XM
```

| Symptom | Meaning |
|:--|:--|
| `cond_dim=0 (visual token OFF)` | 🔴 image-blind — abort, G-B3 should have caught it |
| `🔴 WARNING: dit_hidden_size=256` | the sizing plumbing broke; the A/B would be confounded |
| `params=` far from ~4.0 M | not parameter-matched (PLAN §8) |
| any `_film` in the run directory | the `_DROP` deletion regressed |
| `ml_bone = unet` when you set the env | narrowing bug — the arm-specific var did not survive `--export` |

Logs: `Slurm_Codes/logs/<date>/<time>_<jobname>_<jobid>.log`, and
`Slurm_Codes/logs/latest.log` symlinks the running job.

---

## 6. Results land in

```
logs/aligning-d3il-visual/plans/mix_visual_aligning_<arm>/H8_..._B<bone>_E<arm>_ts<...>/results/<seed>/
```

The `_B<bone>` fragment is what keeps these separate from the U-Net results — which have no
`_B` fragment at all, exactly as before U8.
