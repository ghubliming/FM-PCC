# DA 2026-09-04 — Gen14 U12: the α floor is alive, and it does not help

*Gen14 · `aligning-d3il-visual` · DA batch `batch_va2_20260904_221037` (26 candidates).
Two new arms landed: `MIX_AF_ALPHA_END=0.2` (job 25372→25373) and `MIX_AF_ALPHA_END=0.05`
(job 25376→25377), both deployed at `MIX_EPOCH=latest`. This is the first evaluation of the
U12 recipe — the port of the Gen3v7 `avoiding` AF-UNet result into Visual Aligning.*

**Verdict in one line: every U12 mechanism works exactly as designed, and the resulting policy
is not better. The α floor buys constraint satisfaction by ceasing to make task progress.**

**And the ladder does not come out the way we wanted.** At matched K=2, matched 26.4 M visual
U-Net, matched seed and matched contexts, the ordering on task progress is **`mf` > `af` > `fm`**,
not `af > mf > fm`. `af > mf > fm` holds on the **constraint axis only**, and only for the
α_end=0.05 setup — the one that has all but stopped moving the box. Every arm, the pinned
DPCC target included, scores **S&C = 0.000**. See §3.0–3.1 for the four direct answers.

---

## Conventions

Same as `DA_20260831_…U10_alpha_const…md`, kept deliberately identical so the two reports compose:

- **Distance** is `context_final_xy_dist` — raw box→target XY in metres. **Progress** is
  `context_init_xy_dist − context_final_xy_dist`; positive means the box moved *toward* the target.
  `mean_dist_per_rollout` is **not used** (it is `0.5·(pos_dist_3D + rot_err/π)`, not a distance).
- **0-viol** is `collision_free_completed`, identical to `constraint_exec_zero_violation` in every
  rollout row of this batch.
- `avg_time_ms` is **per replan step**, not per rollout.
- Tests are exact two-sided sign test (distance) and exact McNemar (0-viol), pure stdlib.
- **Pairing.** Fingerprinting each rollout by `(box_init_xy, target_xy, box_angle)` shows the
  10-context runs are an exact **prefix subset** of the 30-context runs — verified for candidates
  5, 8, 9 against 7, 22 and 12 (`intersection = 10`, `⊆ = True` in all cases). Every comparison
  below is paired per context. The n=10 vs n=30 asymmetry is **not** a confound.
- All rows are `split = train` (seen training set) and `geo = combined_5`. Gen14 evals on this
  scene have no held-out split. Seed **6**, n=1 seed everywhere.

### The arms

| cand | arm | α schedule | ckpt | bone | n ctx |
|---|---|---|---|---|---|
| 7 | `af` shipped | sigmoid 1.0 → **0.0** — α snaps to exactly 0 at ≈71.2 % ⇒ **ends as MeanFlow** | `best` | visual U-Net, FiLM v1 | 30 |
| 5 | `af` U10 | **constant 0.05** from step 0 | `best` | same | 10 |
| **8** | **`af` U12 NEW** | **sigmoid 1.0 → 0.05, floored** | **`latest`** | same | 10 |
| **9** | **`af` U12 NEW** | **sigmoid 1.0 → 0.2, floored** | **`latest`** | same | 10 |
| 22 | `mf` | n/a — MeanFlow objective throughout | `best` | same | 30 |
| 12 | `diffusion` K20 aw10 | n/a | `best` | same | 30 |

Backbone is identical across 5/7/8/9/22: `[ AFTrajectoryModel ] backbone=unet vision=True
unet_width(freq_dim)=32 params=26.4M`. This is an **architecture-matched** comparison; the
26.4 M figure includes the `MultiImageObsEncoder`, so it is not the 4.0 M state-space U-Net.

---

## Part 1 — The mechanism: four gates, all green

The U12 changelog named four things that had to be true. Each is confirmed from the job logs.

### Gate 1 — the final checkpoint actually lands

`save_freq = n_train_steps // 5` with `self.step` reaching only `n_train_steps − 1` meant the
newest numeric checkpoint was **step 80 000 of 100 000** — `latest` could never see the end of the
curriculum. The purely-additive final save fired in both trainers:

```
[ utils/training_twotime ] final checkpoint: step 100000 (the periodic save only reaches 80000);
                           "latest" now resolves to the end of the schedule
```
*(jobs 25372 and 25376, identical line)*

### Gate 2 — `latest` resolves to it, and α is alive there

```
job 25373:  [ eval loading ] checkpoint = state_100000.pt  (trained to step 100000)
            [ eval loading ] alpha(step 100000) = 0.2000  [schedule sigmoid 1.0 -> 0.2 over 100000 steps, clamp 0.005]
            [ eval loading ]   alpha-Flow objective ACTIVE at this checkpoint.

job 25377:  [ eval loading ] checkpoint = state_100000.pt  (trained to step 100000)
            [ eval loading ] alpha(step 100000) = 0.0500  [schedule sigmoid 1.0 -> 0.05 over 100000 steps, clamp 0.005]
            [ eval loading ]   alpha-Flow objective ACTIVE at this checkpoint.
```

Training side agrees. Final wandb summary:

| job | `val/alpha` | `val/discrete_frac` |
|---|---|---|
| 25372 (`α_end=0.2`) | **0.2** | **0.50173** |
| 25376 (`α_end=0.05`) | **0.05** | **0.50173** |

`discrete_frac = 0.50173` at the last epoch is the machine-readable proof: the bootstrapped branch
carried ~half of every batch to the end of training. The defect signature is
`train/discrete_frac == 0.0` at the final epochs. **The α-never-on defect is fixed on Gen14.**

### Gate 3 — path safety

| tree | job 25373 | job 25377 |
|---|---|---|
| checkpoint | `…_afschsigmoid_AFAFend0p2/6` | `…_afschsigmoid_AFAFend0p05/6` |
| results | `…_filmv1_Eaf_EPlatest_msgafon02_s6/6` | `…_filmv1_Eaf_EPlatest_msgafon005_s6/6` |

Both fragments are new directories. The pre-existing `…_afschsigmoid/…_filmv1_Eaf` tree
(candidate 7) is byte-identical and was neither read nor written. The DA discovered both as fresh
candidates (8 and 9) with no pooling — `run_tag` handling from commit `43d684cb` did its job.

### Gate 4 — the fail-fast fires

Job **25369** requested `--epoch latest` against the *old* `afschsigmoid` tree, which predates the
final-save fix and holds only `state_best.pt`:

```
[ eval loading ] ERROR: --epoch latest found NO numeric state_<step>.pt in …
  present: ['state_best.pt']
  The periodic saves are missing, not the checkpoint: state_best.pt alone cannot answer 'latest'.
```

That is the designed behaviour, not a failure. It cost one job slot and correctly refused to
silently deploy `best` under a `_EPlatest` folder name.

---

## Part 2 — The numbers

### 2.1 Paired tests (n = 320 rollouts = 10 contexts × 32 variant/geo cells)

`A < B / A > B` counts contexts where A's final distance is smaller / larger. `b/c` are the
McNemar discordant counts on 0-viol.

| A vs B | dist A / B | A<B / A>B | p (sign) | 0-viol A / B | b/c | p (McNemar) | sat A / B | abort A / B |
|---|---|---|---|---|---|---|---|---|
| **α_end=0.05 @latest** vs α=0.05 const @best | 0.4073 / 0.3642 | 121/148 | 0.11 | **0.637 / 0.284** | 128/15 | **1.6e-23** | 0.902 / 0.817 | **0.025 / 0.331** |
| **α_end=0.2 @latest** vs α=0.05 const @best | 0.3793 / 0.3642 | 98/160 | **1.4e-4** | 0.316 / 0.284 | 49/39 | 0.34 | 0.761 / 0.817 | **0.081 / 0.331** |
| **α_end=0.2** vs **α_end=0.05** *(one knob)* | 0.3793 / 0.4073 | 154/119 | **0.039** | 0.316 / 0.637 | 28/131 | **4.0e-17** | 0.761 / 0.902 | 0.081 / 0.025 |
| α_end=0.05 @latest vs **α_end=0 @best (shipped)** | 0.4073 / 0.3202 | 97/171 | **7.3e-6** | 0.637 / 0.444 | 96/34 | **4.9e-8** | 0.902 / 0.877 | 0.025 / — |
| α_end=0.2 @latest vs **α_end=0 @best (shipped)** | 0.3793 / 0.3202 | 119/152 | 0.052 | 0.316 / 0.444 | 37/78 | **1.7e-4** | 0.761 / 0.877 | 0.081 / — |
| α_end=0.05 @latest vs **mf** | 0.4073 / 0.3241 | 108/166 | **5.5e-4** | 0.637 / 0.494 | 79/33 | **1.6e-5** | 0.902 / 0.887 | 0.025 / — |
| α_end=0.2 @latest vs **mf** | 0.3793 / 0.3241 | 103/172 | **3.8e-5** | 0.316 / 0.494 | 25/82 | **2.9e-8** | 0.761 / 0.887 | 0.081 / — |
| α_end=0.05 @latest vs **diffusion K20 (target)** | 0.4287 / 0.4171 | 27/28 | 1.00 | **0.610 / 0.186** | 29/4 | **1.1e-5** | 0.898 / 0.621 | 0.034 / — |
| α_end=0.2 @latest vs **diffusion K20 (target)** | 0.3836 / 0.4171 | 30/23 | 0.41 | 0.186 / 0.186 | 7/7 | 1.00 | 0.679 / 0.621 | 0.102 / — |

*(the target arm exists on only 6 projection variants, hence n=59 on its two rows;
`divergence_aborted` is a newer metric and is absent from candidates 7, 22 and 12)*

### 2.2 Where it actually shows: the anneal is a large stability win over constant α

Against the **correct like-for-like control** — same α value (0.05), same everything else, changed
only the schedule and the checkpoint selector:

- zero-violation rollouts **0.284 → 0.637** (`p = 1.6e-23`)
- divergence aborts **0.331 → 0.025** — **13× fewer**
- constraint satisfaction **0.817 → 0.902**
- distance unchanged (`p = 0.11`)

This is unambiguous and it is the U12 recipe doing its job. ⚠ **It confounds two changes** — the
sigmoid anneal *and* `best`→`latest`. Nothing here separates them; that would need
`α_end=0.05 @best` as a third cell.

### 2.3 …and why it is not a win

Restricting to the **common 10 contexts** and the **6 variants where the pinned DPCC target
exists**, with task progress made explicit:

| variant | arm | S&C | sat | 0-viol | final d | **progress** | abort | ms |
|---|---|---|---|---|---|---|---|---|
| `diffuser` | af α_end=0 @best | 0.000 | 0.832 | 0.200 | 0.2394 | **+0.2136** | — | 27.1 |
| | af α_end=0.05 @latest | 0.000 | 0.899 | **0.700** | 0.3358 | +0.1172 | 0.100 | 27.6 |
| | af α_end=0.2 @latest | 0.000 | 0.565 | 0.100 | 0.3574 | +0.0955 | 0.200 | 28.7 |
| | mf | 0.000 | 0.854 | 0.600 | 0.3676 | +0.0854 | — | 28.4 |
| | diffusion K20 **(TARGET)** | 0.000 | 0.702 | 0.300 | 0.4677 | −0.0147 | — | 298.3 |
| `dpcc-c` | af α_end=0 @best | 0.000 | 0.775 | 0.200 | 0.2891 | **+0.1639** | — | 57.3 |
| | af α_end=0.05 @latest | 0.000 | 0.829 | 0.400 | 0.5084 | **−0.0555** | 0.000 | 70.4 |
| | af α_end=0.2 @latest | 0.000 | 0.686 | 0.100 | 0.4271 | +0.0259 | 0.100 | 67.6 |
| | mf | 0.000 | 0.847 | 0.200 | 0.3542 | +0.0988 | — | 59.3 |
| | diffusion K20 **(TARGET)** | 0.000 | 0.753 | 0.300 | 0.3259 | +0.1271 | — | 1804.5 |
| `dpcc-t` | af α_end=0 @best | 0.000 | 0.868 | 0.400 | 0.3137 | +0.1393 | — | 53.1 |
| | af α_end=0.05 @latest | 0.000 | **0.907** | 0.500 | 0.4745 | **−0.0215** | 0.100 | 54.9 |
| | af α_end=0.2 @latest | 0.000 | 0.754 | 0.300 | 0.4034 | +0.0496 | 0.000 | 52.9 |
| | mf | 0.000 | 0.828 | 0.200 | 0.2721 | **+0.1809** | — | 50.0 |
| | diffusion K20 **(TARGET)** | 0.000 | 0.611 | 0.000 | 0.4580 | −0.0050 | — | 1980.3 |
| `dpcc-r` | af α_end=0 @best | 0.000 | 0.887 | 0.500 | 0.3657 | +0.0873 | — | 54.8 |
| | af α_end=0.05 @latest | 0.000 | **0.917** | 0.600 | 0.4534 | **−0.0004** | 0.000 | 61.7 |
| | af α_end=0.2 @latest | 0.000 | 0.746 | 0.200 | 0.3891 | +0.0639 | 0.100 | 62.5 |
| | mf | 0.000 | 0.868 | 0.300 | 0.3028 | **+0.1502** | — | 49.8 |
| | diffusion K20 **(TARGET)** | 0.000 | 0.743 | 0.200 | 0.3380 | +0.1150 | — | 1877.0 |
| `gradient` | af α_end=0 @best | 0.000 | 0.777 | 0.300 | 0.2657 | **+0.1872** | — | 28.7 |
| | af α_end=0.05 @latest | 0.000 | 0.895 | **0.700** | 0.3414 | +0.1116 | 0.000 | 28.7 |
| | af α_end=0.2 @latest | 0.000 | 0.569 | 0.100 | 0.3429 | +0.1101 | 0.100 | 28.7 |
| | mf | 0.000 | 0.781 | 0.300 | 0.3206 | +0.1324 | — | 29.5 |
| | diffusion K20 **(TARGET)** | 0.000 | 0.554 | 0.200 | 0.4273 | +0.0256 | — | 316.7 |
| `post_processing` | af α_end=0 @best | 0.000 | 0.887 | 0.500 | 0.3659 | +0.0871 | — | 55.4 |
| | af α_end=0.05 @latest | 0.000 | **0.922** | **0.700** | 0.4422 | +0.0108 | 0.000 | 56.1 |
| | af α_end=0.2 @latest | 0.000 | 0.748 | 0.300 | 0.3843 | +0.0687 | 0.100 | 59.4 |
| | mf | 0.000 | 0.876 | 0.300 | 0.3016 | **+0.1514** | — | 49.8 |
| | diffusion K20 **(TARGET)** | 0.000 | 0.334 | 0.111 | 0.4934 | −0.0357 | — | 379.0 |

Two facts kill the win:

1. **`S&C = 0.000` for every arm in every cell, including the pinned DPCC target.** Success and
   constraint satisfaction never co-occur for anyone. Over the full 13-variant set the ceiling is
   `mf` at 0.067 (2/30 on `dpcc-r`), and the target is flat 0.000 at n=30. **`aligning-d3il-visual`
   is an unsolved scene**, so nothing here can be ranked.
2. **`α_end=0.05` earns its constraint numbers by not doing the task.** Its progress is **negative**
   on `dpcc-c`, `dpcc-t` and `dpcc-r` — the box ends *further* from the target than it started —
   while the shipped α-dead arm advances it +0.09…+0.21 m and `mf` +0.09…+0.18 m. Averaged over
   all rollouts: progress **0.046 m** (α_end=0.05) vs **0.145 m** (shipped) vs **0.164 m** (`mf`)
   out of a 0.45 m initial gap. It is not degenerately frozen — 42.8 % of its rollouts still move
   >1 cm, comparable to the other `af` arms — but the magnitude collapses ~3×.

`α_end=0.2` is the more balanced of the two: progress +0.026…+0.110, aborts 4× below the constant-α
arm, but 0-viol *worse* than the shipped arm on 5 of 6 variants.

### 2.4 The one defensible statement against the target

On the 59 paired rollouts where the target arm exists, `α_end=0.05 @latest` is statistically tied
on distance (`p = 1.00`) and strictly better on constraints (**0-viol 0.610 vs 0.186**,
`p = 1.1e-5`; sat 0.898 vs 0.621) at **~50 ms vs ~1290 ms** per replan (2 NFE vs K=20).

Report this as a **speed-and-constraints observation, not a win**: both arms have S&C = 0 and both
are near-static in task terms (progress +0.046 vs +0.053 m), so "better constraints at equal
success" is being measured where neither policy solves the task.

---

## Part 3 — Verdict

### 3.0 The engine ladder, on one set of contexts

Everything below reads off this table. **Common 10 contexts, `geo = combined_5`, `split = train`,
seed 6.** `prog` = `init_dist − final_dist` in metres (positive = box moved toward target, out of a
0.45 m initial gap); `prog_ok` recomputes it excluding diverged rollouts, to prove the ranking is
not an abort artefact. Sorted by `prog`.

| arm | K | n | **prog** | prog_ok | **0-viol** | sat | abort | steps | ms/replan |
|---|---|---|---|---|---|---|---|---|---|
| `af` α_end=0 @best *(α dead ⇒ **is** MeanFlow)* | 2 | 190 | **0.1399** | 0.1399 | 0.311 | 0.847 | 0.000 | 397.1 | 66.8 |
| **`mf`** | 2 | 190 | **0.1357** | 0.1357 | 0.332 | 0.857 | 0.000 | 395.7 | 67.4 |
| `af` α=0.05 **const** @best *(U10)* | 2 | 160 | 0.0956 | 0.0940 | 0.144 | 0.761 | **0.356** | 332.0 | 53.3 |
| **`af` α_end=0.2 @latest** *(U12 🆕)* | 2 | 160 | 0.0723 | 0.0784 | 0.156 | 0.677 | 0.094 | 375.1 | 50.9 |
| `fm` T0.5 | 20 | 100 | 0.0520 | 0.0520 | 0.000 | 0.471 | 0.000 | 400.0 | 648.7 |
| `fm` T0.2 | 20 | 190 | 0.0433 | 0.0382 | 0.058 | 0.618 | 0.268 | 356.7 | 439.2 |
| **`af` α_end=0.05 @latest** *(U12 🆕)* | 2 | 160 | **0.0418** | 0.0435 | **0.537** | **0.873** | 0.044 | 391.4 | 50.7 |
| `diffusion` K20 aw10 **(TARGET)** | 20 | 59 | 0.0366 | 0.0366 | 0.186 | 0.621 | 0.000 | 400.0 | 1121.7 |

`S&C = 0.000` in **every** row. Excluding aborts moves nothing by more than 0.006 m, so the ordering
is real and not a survivorship effect.

### 3.1 The four questions, answered

**Q1 — Is α-enabled better now than the old setups?**
**Against the previous α-enable attempt (U10 constant α=0.05): yes, but only on the safety axes.**
α_end=0.05 takes 0-viol `0.144 → 0.537`, sat `0.761 → 0.873`, aborts `0.356 → 0.044` (8× fewer).
That is the single largest effect in this report (`p = 1.6e-23`). But task progress goes
`0.096 → 0.042` — it got *worse at the task*. α_end=0.2 is milder in both directions: aborts
`0.356 → 0.094`, progress `0.096 → 0.072`, 0-viol flat, sat worse.
**Against the shipped α_end=0 @best: no.** That arm leads on progress (0.140) and is beaten on
0-viol only by α_end=0.05 (0.537 vs 0.311). ⚠ Remember what the shipped arm *is*: α snaps to 0 at
≈71.2 %, so it trains the **MeanFlow** target. "α-Flow loses to the shipped af arm" is really
"α-Flow loses to MeanFlow wearing an `af` folder name."

**Q2 — Does any af setup beat `mf` now?**
**No.** This is the cleanest comparison in the batch — same K=2, same 26.4 M visual U-Net, same
seed, same contexts, same variant set — and af loses it:

| | af α_end=0.2 | af α_end=0.05 | `mf` | who wins |
|---|---|---|---|---|
| progress (m) | 0.0723 | 0.0418 | **0.1357** | **mf**, by 1.9× and 3.2× |
| 0-viol | 0.156 | **0.537** | 0.332 | af α_end=0.05 |
| sat | 0.677 | **0.873** | 0.857 | af α_end=0.05, marginally |
| aborts | 0.094 | 0.044 | **0.000** | **mf** |
| S&C | 0.000 | 0.000 | 0.000 | tie at zero |

Paired significance: distance worse than `mf` at `p = 3.8e-5` (α=0.2) and `p = 5.5e-4` (α=0.05);
0-viol better at `p = 1.6e-5` (α=0.05) but **worse** at `p = 2.9e-8` (α=0.2).
So one af setup leads on constraints — and it is the setup that has all but stopped doing the task.
**No af setup is Pareto-dominant over `mf`, and none is even close on the task axis.**

**Q3 — Does `af > mf > fm` hold now?**
**No — not as a general ordering. It holds on the constraint axis only, and inverts on the task axis.**

| axis | ordering observed |
|---|---|
| **task progress** | `mf` (0.136) **>** `af` (0.072, best af setup) **>** `fm` (0.052) → **mf > af > fm** |
| **0-viol** | `af` α_end=0.05 (0.537) **>** `mf` (0.332) **>** `fm` (0.058) → **af > mf > fm ✅** |
| **constraint sat** | `af` α_end=0.05 (0.873) ≈ `mf` (0.857) **>** `fm` (0.618) → af ≈ mf > fm |
| **cost** | `af` ≈ `mf` (51–67 ms, K=2) **≪** `fm` (439–649 ms, K=20) |
| **S&C** | all zero — **no ordering exists** |

Two things are solid. **`fm` is last on every axis** — and it is last while spending 9–13× the
compute (K=20 vs K=2), so that ranking is safe even though `fm` is not NFE-matched. And
**`af` vs `mf` is properly matched** (same K, bone, seed, contexts, variants) — that comparison is
the one with no excuses, and af loses it on the axis the task is scored on.
The headline `af > mf > fm` **cannot be claimed from this scene.** The defensible sentence is:
*"at matched K=2 and matched backbone, MeanFlow leads on task progress; α-Flow with a 0.05 floor
leads on constraint satisfaction; naive FM is last on both at 9× the cost — and none of the three
achieves a single success-and-constraints rollout."*

**Q4 — Is it still worth refining AF?**
**Not on this scene. Yes elsewhere — and three cheap items here are still worth buying.**

Against, on `aligning-d3il-visual`:
- `S&C = 0.000` for every arm **including the pinned DPCC target** at n=30. There is no ranking
  signal to refine *toward*; any α sweep here moves numbers that do not compose into a claim.
- The negative is consistent, not noisy: the same direction across 13 projection variants, both
  α floors, and two independent axes, at `p ≤ 3.8e-5`.
- The mechanism is now proven working (Part 1), so a further null here is a statement about the
  **method on this task**, not about the plumbing. That question is answered.

For, elsewhere:
- On `avoiding` (state-space, 4.0 M U-Net) the same recipe turned **0/2 → 2/2 goals at K=1**. AF is
  not dead — the Gen14 result localises the failure to *this scene / this 26.4 M visual bone*, which
  is itself a finding worth one paragraph in the write-up.
- `s_curve` (Gen15) has a live `diffusion` target arm, so a result there can actually be ranked.
  That is where the next AF GPU-hour belongs. Carry **α_end=0.2**, not 0.05 — 0.05 suppresses the
  policy, and a policy that stops moving scores zero against a real target too.

Three items here are still worth buying, in this order — see Part 4.

### 3.2 What this run did and did not establish

| | |
|---|---|
| ✅ **Established** | The U12 machinery works end to end — final checkpoint saved, `latest` resolves to it, α alive at deploy (`discrete_frac = 0.50173`), path safety intact, fail-fast correct. |
| ✅ **Established** | The floored **anneal** is far more stable than **constant** α at the same α value: 8× fewer divergence aborts, `p = 1.6e-23` on 0-viol. |
| ✅ **Established** | At matched K, bone, seed and contexts, **`mf` leads `af` on task progress** and **`fm` is last on every axis**. |
| ❌ **Not established** | That α-Flow beats MeanFlow anywhere on this scene. It does not. |
| ❌ **Not established** | Any `af > mf > fm` ordering. Only the constraint axis supports it. |
| ❌ **Not established** | Anything about generalisation (train split only) or seed variance (n=1). |
| ⚠️ **Confounded** | The big U12 win pairs the anneal with `best → latest`. One train job separates them. |

**Neither new arm is Pareto-dominant** over the shipped arm or `mf`. Both are non-dominated in the
strict sense — better on constraints, worse on distance — but the trade goes the wrong way: the
axis that got worse is the one the task is scored on.

### Caveats that must travel with these numbers

- **n = 1 seed** (seed 6) for every arm. No seed variance estimate exists.
- Arms 5/8/9 are **n = 10 contexts**, arms 7/22/12 are **n = 30**. Handled by exact pairing on the
  common 10, but the 30-context means quoted in §2.1 for candidates 7/22/12 are over their own
  full context sets.
- `split = train` only — seen training set, no generalisation claim available.
- **`fm` is not NFE-matched** to `af`/`mf` (K=20 vs K=2). Its last place is safe *because* it is
  last at 9–13× the compute, but no equal-budget `fm` row exists in this batch.
- **The new arms have no `hardflow_new-*` rows.** Candidates 7 and 22 do (3 variants each). The
  HardFlow axis is uncovered for U12 and cannot be compared.
- `α_end=0.05 @latest` vs `α=0.05 const @best` confounds schedule with checkpoint selector.
- Per-arm mean `avg_time_ms` is **not** comparable across arms in the batch summary because the
  variant sets differ (the missing HardFlow rows, ~180 ms each, bias the new arms low). Only the
  per-variant figures in §2.3 and §3.0 are comparable.

---

## Part 4 — What would actually settle it

Ranked by information per GPU-hour. None of these are submitted.

1. **`MIX_AF_ALPHA_END=0.05 MIX_EPOCH=best`** — one train job. Separates the anneal from the
   checkpoint selector, which is the single largest confound in this report and is cheap to remove.
   This is the only experiment that could change how §3.1 Q1 reads.
2. **A second seed on `α_end=0.2`** — the least-bad af setup. Everything here is n=1, so the
   *magnitude* of the af-vs-mf gap is unpinned even though its direction is consistent.
3. **HardFlow coverage on arms 8/9** — eval-only, no retrain. Closes the variant gap against
   candidates 7 and 22.
4. **Then stop, and move AF to `s_curve` (Gen15).** With the pinned DPCC target at `S&C = 0.000`
   and the strongest arm ever run on this scene at 0.067, `aligning-d3il-visual` cannot support a
   ranked claim from any engine. Further α sweeps here buy resolution on a metric that is zero for
   everyone. `s_curve` has a live `diffusion` arm; carry **α_end=0.2 @latest** there.

---

## Part 5 — Where the new runs live on disk

Everything is under `logs/aligning-d3il-visual/` on the cluster (`logs/` is gitignored — cluster
only). Two trees per run: **checkpoints** under the engine folder, **rollouts** under `plans/`.

To keep the table readable, `<CLS>` stands for the model class string that both trees repeat:

| shorthand | expands to |
|---|---|
| `<AF>` | `Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow` |
| `<MF>` | `Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow` |
| `<GD>` | `Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion` |

### 5.1 The two new runs — full literal paths

**`α_end=0.2` (job 25372 → 25373, DA candidate 9)**

```
# checkpoints  ← state_100000.pt lives here
logs/aligning-d3il-visual/mix_visual_aligning_af/
  H8_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Eaf_tslogit_normal_afschsigmoid_AFAFend0p2/
    6/

# rollouts
logs/aligning-d3il-visual/plans/mix_visual_aligning_af/
  H8_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Eaf_tslogit_normal_afschsigmoid_AFAFend0p2/
    H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow_VTrue_mpc4_filmv1_Eaf_EPlatest_msgafon02_s6/
      6/
```

**`α_end=0.05` (job 25376 → 25377, DA candidate 8)** — identical, with
`_AFAFend0p05` in place of `_AFAFend0p2` and `_msgafon005_s6` in place of `_msgafon02_s6`.

### 5.2 The whole comparison set, side by side

Read the **last two fragments** of each row — that is all that differs between the α-Flow arms.
🆕 marks a directory that did not exist before 2026-09-03.

| # | arm | checkpoint folder (under `mix_visual_aligning_<engine>/`) | plan folder (under `plans/.../<ckpt folder>/`) |
|---|---|---|---|
| **9** | 🆕 **af α_end=0.2 @latest** | `H8_<AF>_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Eaf_tslogit_normal_afschsigmoid`**`_AFAFend0p2`** | `H8_K2_Meuler_T0.5_<AF>_VTrue_mpc4_filmv1_Eaf`**`_EPlatest_msgafon02_s6`** |
| **8** | 🆕 **af α_end=0.05 @latest** | `…_afschsigmoid`**`_AFAFend0p05`** | `…_filmv1_Eaf`**`_EPlatest_msgafon005_s6`** |
| 7 | af α_end=0 @best *(shipped)* | `…_afschsigmoid` *(no AF tag)* | `…_filmv1_Eaf` *(no EP, no msg)* |
| 5 | af α=0.05 const @best | `…_Eaf_tslogit_normal`**`_afschconstant_AFAFconst0p05`** | `…_filmv1_Eaf` |
| 10 | af filmv2 α_end=0 @best | `H8_<AF>_…_bs64`**`_filmv2`**`_Eaf_tslogit_normal_afschsigmoid` | `H8_K2_Meuler_T0.5_<AF>_VTrue_mpc4`**`_filmv2`**`_Eaf` |
| 22 | mf | `H8_<MF>_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal` | `H8_K2_Meuler_T0.5_<MF>_VTrue_mpc4_filmv1_Emf` |
| 12 | diffusion K20 aw10 **(target)** | `H8_K20_<GD>_aw10_VTrue_steps1000_bs64_filmv1_Ediffusion` | `H8_K20_T0.5_<GD>_VTrue_mpc4_filmv1_Ediffusion` |

Candidates 7 and 5 share a plan-folder *name* but sit under different checkpoint parents, so they
never collide — that is the whole point of putting the α tag on the **checkpoint** segment.

### 5.3 What every fragment means

**Checkpoint folder** (built by `watch(args_to_watch_mix_visual)`):

| fragment | meaning |
|---|---|
| `H8` | horizon = 8 |
| `<AF>` / `<MF>` / `<GD>` | the diffusion/flow class — the engine's identity |
| `a1.5_b1.0` | loss discount `α=1.5`, `β=1.0` |
| `aw1` / `aw10` | `action_weight` — **10 on the diffusion target**, 1 elsewhere |
| `VTrue` | `if_vision = True` (visual observations) |
| `steps1000` | `n_diffusion_steps = 1000` at train time |
| `bs64` | batch size 64 |
| `filmv1` / `filmv2` | FiLM conditioning mode |
| `Eaf` / `Emf` / `Ediffusion` | engine |
| `tslogit_normal` | `t_schedule = logit_normal` |
| `afschsigmoid` / `afschconstant` | `af_alpha_scheduler` |
| 🆕 **`AFAFend0p2`** | **α path key (U10 token, U12 value).** Reads `AF` + `end0p2` ⇒ `af_alpha_end = 0.2`. The doubled `AF` is the label `('af_alpha','AF')` meeting a value that already starts with `AF` — cosmetic, deliberate, don't "fix" it. `0.05 → 0p05`; `.` becomes `p`. |
| `AFAFconst0p05` | same token, constant schedule: `af_alpha_init = af_alpha_end = 0.05` |
| `/6` | **seed** |

**Plan folder** (`watch(...)` + the plan-only suffixes):

| fragment | meaning |
|---|---|
| `H8` | horizon 8 |
| `K2` / `K20` | **sampler steps (NFE per replan).** Eval-time knob: the checkpoint stores `flow_steps_v3=100`, the eval overrides down — look for `[ config->pkl ] INFO flow_steps_v3: train=100 -> eval=2` |
| `Meuler` | ODE solver = Euler (absent on the diffusion arm) |
| `T0.5` | projection threshold `diffusion_timestep_threshold = 0.5` |
| `mpc4` | `mpc_batch_size = 4` |
| 🆕 **`EPlatest`** | **checkpoint selector (U12).** `latest` ⇒ `state_100000.pt`. Absent ⇒ `best`. A run **without** this fragment loaded `state_best.pt` no matter what its run tag claims. |
| 🆕 `msgafon02_s6` | `FMPCC_RUN_MSG` free-text run tag (`afon02_s6`). Cosmetic only — never a functional knob. |
| `/6` | seed |

### 5.4 Inside a plan folder

```
<plan folder>/6/
├── run_provenance.json                      ← env knobs, git rev, resolved checkpoint
└── results_train_set/
    ├── run_provenance.json
    ├── expert_references/expert_rollout_<N>.gif
    ├── combined_5/                          ← nominal constraint set
    │   ├── constraint_overview.png (+ .svg)
    │   └── <variant>_train_set/             ← 19 projection variants
    │       ├── constraint_metrics.json      ← the numbers this DA aggregates
    │       ├── diag_first_replan.txt
    │       └── diagnostics/rollout_<N>.mp4  ← ⚠ MP4 backend missing on the cluster; PNG grids do land
    └── combined_5-tightened/                ← same, tightened constraints
```

`<variant>` ∈ `diffuser`, `dpcc-c`, `dpcc-t`, `dpcc-r`, `dpcc-c-dt{0p25,0p5,2p0,4p0}`,
`geo_free`, `geo_free-bounds_free`, `geo_free-model_free`, `bounds_free`, `model_free`,
`model_free-bounds_free`, `post_processing`, `gradient`. **`hardflow_new-{c,r,t}` are absent on
candidates 8 and 9** — the coverage gap noted in the caveats.

### 5.5 Two-second sanity check before trusting a folder

```bash
CKPT=logs/aligning-d3il-visual/mix_visual_aligning_af/H8_..._afschsigmoid_AFAFend0p2/6
ls $CKPT                       # expect state_best.pt AND state_100000.pt
                               # only state_best.pt  ⇒ pre-U12 tree, `latest` will fail-fast
grep -m2 'alpha(step' <eval log>   # expect: alpha(step 100000) = 0.2000 … ACTIVE
```

If the plan folder has **no `_EP` fragment**, the run deployed `best` — a mid-curriculum
checkpoint — regardless of what the `_msg` tag says.

---

## Part 6 — Can AF be refined until it beats MF?

Short answer: **there are real, untried levers, and the most promising one is free — but none of
them can produce a defensible "af beats mf" *on this scene*, because `S&C = 0.000` for every arm
including the pinned target. Run the free diagnostics here; spend GPU on the paid ones where a
target arm exists.**

### 6.0 The lead: we may have deployed the wrong point on the curriculum

Recomputing the shipped α schedule (`sigmoid`, γ = 25, over 100 000 steps) at the steps that are
actually on disk:

| step | α (`end=0`) | α (`end=0.05`) | α (`end=0.2`) | what the model is |
|---|---|---|---|---|
| 0 – 20 000 | 1.000 | 1.000 | 1.000 | **pure Flow Matching** |
| 40 000 | 0.924 | 0.928 | 0.939 | ~FM, barely mixed |
| 50 000 | 0.500 | 0.525 | 0.600 | **the actual crossover** — no checkpoint here |
| 60 000 | 0.076 | **0.122** | **0.261** | **genuinely mid-curriculum** |
| 80 000 | 0.000 | 0.051 | 0.200 | at the floor |
| 100 000 | 0.000 | 0.050 | 0.200 | at the floor ← **what U12 deployed** |

Two things fall out.

**(a) γ = 25 is not a curriculum, it is a step function.** The entire FM → MeanFlow transition
happens between step ~40 000 and ~60 000 — **20 % of the budget**. The other 80 % is spent either at
α = 1 (pure FM) or pinned at the floor. Whatever "α-Flow as a curriculum" is supposed to buy, this
schedule barely does it.

**(b) The strongest `af` arm on task may already be a mid-curriculum model — by accident.** The
shipped `α_end=0` arm leads every af arm on progress (0.140), and it is evaluated at **`best`**.
`state_best.pt` is selected on `test_loss ≈ 0.75 + 0.25·α`, which **structurally prefers a
mid-curriculum checkpoint** (U12 changelog). So the best-performing "α-Flow" model we have is
plausibly one where α was still partly on — and U12's contribution was to move *away* from it,
to the endpoint. If that is right, the recipe is **deploy mid-curriculum, not at the endpoint** —
the opposite of what this run tested.

`save_freq = n_train_steps // 5 = 20 000`, so the two new trees hold
`state_{20000,40000,60000,80000,100000}.pt` **plus** the U12 final save. **Step 60 000 is a genuine
mid-curriculum α-Flow checkpoint and it is already on disk.** Testing (b) costs zero training.

### 6.1 The levers, ranked by information per GPU-hour

| # | lever | cost | why it could move af past mf |
|---|---|---|---|
| **1** | **Checkpoint sweep on the two existing trees** — `MIX_EPOCH=40000 / 60000 / 80000` | **eval only, no retrain** | Directly tests §6.0(b). Step 60 000 gives α = 0.122 / 0.261 — the only real mid-curriculum points we can reach. Also converts §3.0 from three α values into an α→performance *curve*. |
| **2** | **`MIX_AF_ALPHA_GAMMA=5`** (or 8) | 1 train job | Spreads the transition across most of training instead of 20 % of it. This is the knob that makes α-Flow an actual curriculum. Path key `_AFg5`, already wired (`config/aligning-d3il-visual.py:1545`). |
| **3** | **`MIX_SAVE_EVERY=5000`** on the next af train | free rider on #2 | Turns the α curve from 5 samples into 20. Should be standard on every future af run; there is no reason to sample a curriculum 5 times. |
| **4** | **Match `af_adp_eps` to `mf_adp_eps`** (1e-3 → 0.01) | 1 train job + **a config edit** | `config/aligning-d3il-visual.py:1675-1678` calls the 10× difference *deliberate*. Deliberate or not, it is **uncontrolled**: until it is matched, "af vs mf" is a two-knob comparison and the α conclusion is not clean. No env knob exists — this one needs a code change and a go-ahead. |
| 5 | More α floors (0.1, 0.3, 0.5) | 1 train job each | **Lowest priority.** The three floors we have are non-monotone in progress (0.140 → 0.042 → 0.072 for α = 0 / 0.05 / 0.2). At n = 1 seed that is as likely to be seed noise as structure. Adding floors without adding seeds buys resolution on a curve we cannot yet distinguish from noise. |
| 6 | More seeds | 1 train job each | Not a lever *toward* a win, but nothing above is interpretable at n = 1. At minimum, whichever configuration looks best gets a second seed before it is written up. |

### 6.2 The free sweep, ready to submit

```bash
# α_end=0.2 tree — α at these steps: 0.939 / 0.261 / 0.200
for EP in 40000 60000 80000; do
  MIX_AF_ALPHA_END=0.2 MIX_EPOCH=$EP FMPCC_RUN_MSG=afon02_ep${EP}_s6 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh af 6
done

# α_end=0.05 tree — α at these steps: 0.928 / 0.122 / 0.051
for EP in 40000 60000 80000; do
  MIX_AF_ALPHA_END=0.05 MIX_EPOCH=$EP FMPCC_RUN_MSG=afon005_ep${EP}_s6 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh af 6
done
```

Six eval jobs, no training. Each writes to its own `_EP<step>_msg…` results folder, so nothing
existing is touched. Read `prog` and `0-viol` off them and extend the §3.0 table. **If step 60 000
beats step 100 000 on progress, §6.0(b) is confirmed and the whole recipe changes direction.**

### 6.3 What "beating mf" would have to look like to count

On this scene it cannot count, and it is worth being blunt about why. `S&C = 0.000` for `af`, `mf`,
`fm` **and the pinned diffusion-DPCC target**. With the primary metric at zero for everyone, any
"win" is on a secondary axis inside failed rollouts, with no target anchor to normalise against.
That will not survive review, and it should not.

A claim that would count needs all of:

1. **A scene where the pinned target is non-zero** — `avoiding` (where AF-UNet already went
   **0/2 → 2/2 goals at K=1**) or `s_curve` (live `diffusion` arm). Not here.
2. **Matched backbone and matched K** — af and mf are already matched at K=2 on the 26.4 M visual
   U-Net, and that is exactly the comparison af currently loses. Keep it matched; do not let a win
   arrive via a backbone or NFE change.
3. **Pareto-dominance, or an explicit trade-off statement.** At equal success and constraints,
   fewer steps *and* lower `avg_time` — otherwise the word is "non-dominated", not "beats".
4. **≥ 2 seeds** on the winning configuration.
5. **The sweep reported as a sweep.** If levers 1–4 produce 12 cells and one of them beats mf, the
   honest artefact is all 12 cells with the α curve, not the single flattering cell. Picking the
   winner after the fact from a family this size is how a null becomes a false positive — and the
   non-monotone floor result in §6.1 row 5 is exactly the shape of noise that would do it.

### 6.4 Recommendation

- **Here, now:** run §6.2 (free, six evals). It is a genuine hypothesis test, not a fishing trip,
  and it is the only thing that could reverse the direction of the U12 recipe.
- **Next af train job, wherever it runs:** `MIX_AF_ALPHA_GAMMA=5 MIX_SAVE_EVERY=5000`. Cheap,
  and it fixes the two structural problems — a transition compressed into 20 % of the budget, and a
  curriculum sampled 5 times.
- **Where to spend real GPU on AF:** `s_curve` (Gen15), with `α_end=0.2 @latest`, against the live
  `diffusion` target. That is where an af-beats-mf result can be *measured*, and it is already the
  resubmit block in `logs_in_develop/Gen15/U6/RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md`.
- **Do not** keep sweeping α floors on `aligning-d3il-visual`. The scene cannot rank them.

---

## Provenance

| item | value |
|---|---|
| DA batch | `batch_va2_20260904_221037` — 26 candidates, 13 083 rollout rows |
| pipelines | 25370 (`α_end=0.2`), 25374 (`α_end=0.05`) |
| gates | 25371, 25375 — both pass |
| train | 25372 (`GIT REV ba05cb7`), 25376 (`GIT REV 43d684c`) — both `Job completed successfully` |
| eval | 25373, 25377 — both `Job completed successfully` |
| failed | 25369 — `--epoch latest` against a pre-U12 tree; **designed fail-fast**, not a defect |
| wandb | `bhmfs1c6` (α_end=0.2), `2twhzwc0` (α_end=0.05), project `FM-PCC-visual-aligning-gen14` |
| code | Gen14 U12 — `CHANGELOG_Gen14_U12_checkpoint_selector_MIX_EPOCH.md`, commit `ba05cb7c` |
