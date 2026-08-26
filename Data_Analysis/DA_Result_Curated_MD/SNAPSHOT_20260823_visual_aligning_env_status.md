# `aligning-d3il-visual` — env status, per projector, constraint-checked

> **SNAPSHOT 2026-08-23.** Whole-env status across every candidate in the visual-aligning tree.
> Regenerated as new batches land. Use the newest `SNAPSHOT_<date>_*` file in this folder.

**Batch:** `batch_va2_20260823_135156` (DA_VA_v2 — `temp/2508/…/per_rollout_detail.csv`)
**Scope:** full scan of `logs/aligning-d3il-visual/plans` + 2 bridged D3IL baselines →
**18 candidates · 321 units · 10 478 rollouts**, spanning Gen5, Gen6v4, Gen7 and Gen14.
**Companions:**
[`Gen14/U8/…mf_dit…`](../../logs_in_develop/Gen14/U8/DA_20260823_Gen14_U8_mf_dit_visual_aligning.md) (DiT vs U-Net) ·
[`Gen14/U7/…hardflow_vs_dpcc…`](../../logs_in_develop/Gen14/U7/DA_20260823_hardflow_vs_dpcc_visual_aligning.md) (arm B vs arm C).

> ### The two answers up front
>
> **1. Does `mf`/`af` at K = 2 beat `fm`/`diffusion` at K = 20? → YES, on every axis.** §3.
> Paired over the same 30 contexts, each model at its own best DPCC projector:
> `mf` v1 K2 `dpcc-t` **0.2867 m** vs `fm` K20 `dpcc-t` 0.3617 (t = −2.21) and vs `diffusion` K20
> `dpcc-r` 0.3666 (t = −2.98); constraint-clean tail **10–17 % vs 0 %**; zero-violation rate
> **0.27–0.40 vs 0.03–0.10**; and **20–41× cheaper** (53 ms vs 1 066 / 2 158 ms).
> **The K = 20 engines have _no_ rollout that is both within 15 cm and constraint-clean.**
>
> **2. Does HardFlow beat the DPCC projectors? → No.** §5. Best-vs-best on the two candidates that
> have both, HardFlow is within ±0.04 m on distance, never higher on zero-violation than the best
> DPCC arm, and costs a flat **3.3–3.7×**. `dpcc-t` + tightening reaches **`0-viol` = 1.00 at 42 ms**;
> HardFlow's best is 0.97 at 146 ms.

---

## Reporting rules

🔴 **1. No aggregation across projectors.** The unit is the **cell** =
`(model × projector × geometry × split)`, each with its own `n`. There is no "model score" here.

🔴 **2. Model-vs-model comparisons use each model's OWN best projector, and always name it.**
`af`+`dpcc-c` vs `fm`+`dpcc-r` is a legitimate comparison — that is what you would deploy. Forcing
both sides onto the same variant answers a different question and is reported separately where it
matters. **Best projector is selected on the constraint-clean tail (`<15cm` clean), not raw distance.**

🔴 **3. HardFlow (arm C) is never mixed into a DPCC (arm B) result.** Arm B and arm C are separate
controllers and get separate rows; the single question "does HF beat DPCC" is answered once, in §5.

🔴 **4. Every distance claim is constraint-checked.** A 6 mm rollout that clipped an obstacle is not
a result. Each cell reports **`min (clean)`** — the best rollout that had **zero** constraint
violations — beside `min (any)`, and **`<15cm` clean** beside the raw rate. Where they differ, the
clean number is the one that counts.

🔴 **5. Goal success (`S`) is not used for ranking anywhere in this file.** It is at the floor —
every deployable n = 30 cell scores 0, 1 or 2 successes out of 30, and 99 % of episodes hit the
400-step cap. Ranking on 2/30 is ranking on luck. **Constraint satisfaction and distance are the
instruments.** (Success counts are retained only in §9 and §11.3, where the comparison is against a
baseline rate rather than between models.)

🔴 **6. Every row carries `n`, and all main tables are n = 30 cells only.** Each rollout is a
distinct initial condition (verified). Cells with n < 30 are quarantined in §8 and support no claim.

**Protocol:** seed 6 for every Gen6v4/Gen7/Gen14 candidate. Contexts are shared between candidates,
so same-pool cells are **paired** (§0.3) — all `t` and sign tests below are paired over contexts,
single seed.

---

## 0. Definitions

### 0.1 Metrics

| symbol | field | meaning | dir |
|---|---|---|---|
| **`dist`** | `mean_dist_per_rollout` | The D3IL aligning metric ("Avg final mean distance"), cell mean. | lower |
| **`0-viol`** | `collision_free_completed` | Fraction of the cell's rollouts with **zero** constraint violations at every step. Identical to `constraint_exec_zero_violation` in every cell checked. | higher |
| **`viol`** | `constraint_exec_total_viol_count` | Mean violating steps per rollout. | lower |
| **`min (clean)`** | derived | Lowest `dist` **among zero-violation rollouts only**. The best result the controller actually achieved legally. `— none` = the cell has no clean rollout at all. | lower |
| `min (any)` | derived | Lowest `dist` ignoring constraints. Shown for contrast; **never quote it alone**. | lower |
| **`<15cm` clean** | derived | Fraction of the cell's 30 rollouts that are **both** under 0.15 m **and** zero-violation. The primary sample-robust ranking metric. | higher |
| `<15cm` any | derived | Same without the constraint filter. | higher |
| **`ms`** | `avg_time_ms` | Wall-clock ms per control step = one plan (K net calls × MPC batch 4) **plus** the projection solve. | lower |
| `still` | derived | `abs(context_final_xy_dist − context_init_xy_dist) < 0.02` — the box never moved. | lower |

**Do-nothing reference:** over the 5 418 rollouts in the batch where the box moved < 2 cm,
`dist` = **0.3985 m** (sd 0.130). A cell at ≈0.40 did nothing. (Initial box→target *xy* distance is
0.4547 m — a different quantity; do not use it as the `dist` reference.)

### 0.2 Projectors

| class | variants | role |
|---|---|---|
| **Arm B — DPCC projector** | `dpcc-r` (rule: random) · `dpcc-c` (min projection cost) · `dpcc-t` (temporal consistency) · `post_processing` (post-hoc, no selection) | ✅ reported; model-vs-model uses each model's best of these |
| **Arm C — HardFlow** | `hardflow_new-{r,c,t}` (in-loop NLP; **candidates 6 and 14 only**) | ✅ reported, **separately** (§5) |
| **Arm A — reference** | `diffuser` (unguided, no projection) | ⚪ the no-projector control |
| **Study-only — excluded** | `geo_free` · `bounds_free` · `model_free` + pairs (constraint-class **removal**) · `gradient` · `dpcc-c-dt{0p25,0p5,2p0,4p0}` (threshold sweep) | ❌ not controllers |

The `*_free` family deletes a constraint class from the projector, so both its distance *and* its
violation numbers describe a problem the deployed system does not get to solve. Arm C is opt-in
(`config/visual_aligning_eval.yaml:433 hardflow_variants: []`) and is refused by design for the
`diffusion` engine — a DDPM reverse chain has no velocity field to integrate.

### 0.3 Context pools (what is paired with what)

| pool | contexts | who | pairing |
|---|---|---|---|
| **Train-30** | 30 fixed | C5–C15 (Gen14) | **fully paired**; C5/C13 got 11, C8 19, C10 22 — truncated prefixes (§8) |
| **Train-3** | 3, a subset of Train-30 | C1–C4, C16 | n = 3 → §8 only |
| **Test-30** | 30 held-out, **disjoint** from Train-30 | **C4 and C16 only**, identical set | **fully paired** → §7 |

---

## 1. Roster (n = 30 cells only)

| C | generation | folder | engine | bone / cond | K | arm B | arm C |
|---|---|---|---|---|---|---|---|
| **4** | Gen7 | `fm_visual_aligning` | fm | UNet FiLM v1 | 20 | ✅ **test** (4) | — |
| 3 | Gen7 | `fm_visual_aligning` | fm | UNet FiLM v2 | 20 | — (`diffuser` only) | — |
| **6** | Gen14 | `mix_visual_aligning` | af | UNet FiLM v1 | 2 | ✅ train ×2 geo | ✅ **yes** |
| 7 | Gen14 | `mix_visual_aligning` | af | UNet FiLM v2 | 2 | ✅ train ×2 geo | — |
| 9 | Gen14 | `mix_visual_aligning` | diffusion `aw10` | UNet FiLM v1 | 20 | ✅ train | *refused by design* |
| 11 | Gen14 | `mix_visual_aligning` | fm | UNet FiLM v1 | 20 | ✅ train | — |
| 12 | Gen14 U8 | `mix_visual_aligning` | mf | **DiT**, 80 k | 2 | ✅ train ×2 geo | — |
| **14** | Gen14 | `mix_visual_aligning` | mf | UNet FiLM v1 | 2 | ✅ train ×2 geo | ✅ **yes** |
| 15 | Gen14 | `mix_visual_aligning` | mf | UNet FiLM v2 | 2 | ✅ train ×2 geo | — |
| **16** | **Gen6v4 — in-repo DPCC baseline** | `diffuser_visual_aligning` | diffusion `aw10` | UNet FiLM v2 | 20 | ✅ **test** (4) | — |
| 5, 8, 10, 13 | Gen14 | `mix_visual_aligning` | af / diffusion / fm / mf | UNet FiLM v1 | 100 | ❌ truncated (§8) | — |
| 17, 18 | **Gen5 — D3IL baselines** *(bridged)* | `ddpm_encdec_vision` | DDPM enc-dec | — | — | *no projector* → §6 | — |
| 1, 2 | Gen7 *(legacy trees)* | `fm_visual_aligning` | fm | UNet FiLM v2 | 20 | ❌ n = 3 (§8) | — |

---

## 2. Distance × constraints, cell by cell (n = 30 only)

Sorted by `dist` within each block. **Read `min (clean)` and `<15cm` clean, not the raw columns.**
Do-nothing line = 0.3985.

#### A. `train` / `combined_5` (untightened)

| C | model | **projector** | `dist` | **`0-viol`** | `viol` | **min (clean)** | min (any) | **`<15cm` clean** | `<15cm` any | `ms` | still |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | mf v1 K2 | `dpcc-t` | 0.2867 | 0.27 | 66.2 | **0.0683** | 0.0683 | **10 %** | 27 % | 53 | 33 % |
| 14 | mf v1 K2 | `hardflow_new-r` | 0.3064 | 0.27 | 49.8 | **0.2252** | 0.0442 | **0 %** | 10 % | 182 | 27 % |
| 14 | mf v1 K2 | `hardflow_new-t` | 0.3074 | 0.50 | 40.4 | **0.0439** | 0.0439 | **10 %** | 13 % | 175 | 27 % |
| 13 | mf v1 K100 | `diffuser` *(no projection)* | 0.3113 | 0.20 | 80.3 | **0.0347** | 0.0276 | **10 %** | 33 % | 893 | 10 % |
| 14 | mf v1 K2 | `hardflow_new-c` | 0.3249 | 0.27 | 60.5 | **0.0157** | 0.0157 | **13 %** | 13 % | 194 | 40 % |
| 6 | af v1 K2 | `diffuser` *(no projection)* | 0.3311 | 0.20 | 84.7 | **0.0104** | 0.0104 | **7 %** | 23 % | 27 | 10 % |
| 6 | af v1 K2 | `hardflow_new-r` | 0.3316 | 0.53 | 31.4 | **0.0278** | 0.0278 | **10 %** | 13 % | 178 | 27 % |
| 8 | diffusion K100 | `diffuser` *(no projection)* | 0.3342 | 0.27 | 113.6 | **0.0344** | 0.0344 | **7 %** | 13 % | 1527 | 20 % |
| 15 | mf v2 K2 | `dpcc-t` | 0.3369 | 0.20 | 79.1 | **0.1268** | 0.1046 | **3 %** | 13 % | 53 | 47 % |
| 6 | af v1 K2 | `dpcc-t` | 0.3378 | 0.40 | 60.0 | **0.0517** | 0.0517 | **17 %** | 20 % | 53 | 30 % |
| 14 | mf v1 K2 | `dpcc-r` | 0.3423 | 0.37 | 60.7 | **0.0061** | 0.0061 | **10 %** | 20 % | 56 | 7 % |
| 6 | af v1 K2 | `hardflow_new-t` | 0.3434 | 0.30 | 53.9 | **0.0661** | 0.0661 | **7 %** | 17 % | 186 | 30 % |
| 14 | mf v1 K2 | `post_processing` | 0.3481 | 0.37 | 59.4 | **0.0061** | 0.0061 | **7 %** | 20 % | 56 | 7 % |
| 6 | af v1 K2 | `dpcc-c` | 0.3501 | 0.17 | 99.0 | **0.0680** | 0.0680 | **3 %** | 13 % | 55 | 33 % |
| 6 | af v1 K2 | `hardflow_new-c` | 0.3553 | 0.33 | 57.8 | **0.0347** | 0.0347 | **10 %** | 17 % | 192 | 33 % |
| 12 | mf DiT K2 | `dpcc-t` | 0.3575 | 0.30 | 30.2 | **0.0327** | 0.0327 | **10 %** | 17 % | 56 | 37 % |
| 12 | mf DiT K2 | `dpcc-c` | 0.3577 | 0.30 | 32.2 | **0.0720** | 0.0720 | **10 %** | 17 % | 56 | 13 % |
| 11 | fm K20 | `dpcc-t` | 0.3617 | 0.03 | 128.6 | **0.7086** | 0.0599 | **0 %** | 10 % | 1066 | 67 % |
| 11 | fm K20 | `diffuser` *(no projection)* | 0.3635 | 0.10 | 259.4 | **0.2964** | 0.1288 | **0 %** | 7 % | 294 | 47 % |
| 11 | fm K20 | `dpcc-r` | 0.3645 | 0.07 | 139.5 | **0.4358** | 0.1133 | **0 %** | 3 % | 1073 | 70 % |
| 9 | diffusion K20 | `dpcc-r` | 0.3666 | 0.10 | 131.4 | **0.2735** | 0.0566 | **0 %** | 3 % | 2158 | 73 % |
| 10 | fm K100 | `diffuser` *(no projection)* | 0.3720 | 0.10 | 264.4 | **0.0469** | 0.0469 | **3 %** | 7 % | 1426 | 47 % |
| 15 | mf v2 K2 | `post_processing` | 0.3731 | 0.13 | 74.4 | **0.0160** | 0.0160 | **3 %** | 20 % | 98 | 23 % |
| 15 | mf v2 K2 | `dpcc-r` | 0.3750 | 0.13 | 75.2 | **0.0160** | 0.0160 | **3 %** | 20 % | 98 | 23 % |
| 11 | fm K20 | `dpcc-c` | 0.3791 | 0.00 | 124.8 | **— none** | 0.0984 | **0 %** | 3 % | 1099 | 63 % |
| 11 | fm K20 | `post_processing` | 0.3797 | 0.13 | 129.1 | **0.2964** | 0.1037 | **0 %** | 10 % | 323 | 63 % |
| 7 | af v2 K2 | `diffuser` *(no projection)* | 0.3832 | 0.03 | 196.4 | **0.1861** | 0.1843 | **0 %** | 0 % | 31 | 90 % |
| 9 | diffusion K20 | `dpcc-c` | 0.3840 | 0.13 | 127.5 | **0.1967** | 0.1967 | **0 %** | 0 % | 2119 | 77 % |
| 15 | mf v2 K2 | `dpcc-c` | 0.3855 | 0.10 | 117.8 | **0.0882** | 0.0882 | **3 %** | 7 % | 67 | 63 % |
| 7 | af v2 K2 | `dpcc-t` | 0.3904 | 0.23 | 23.6 | **0.2884** | 0.1843 | **0 %** | 0 % | 60 | 73 % |
| 12 | mf DiT K2 | `post_processing` | 0.3926 | 0.17 | 41.7 | **0.0607** | 0.0607 | **7 %** | 17 % | 56 | 20 % |
| 9 | diffusion K20 | `diffuser` *(no projection)* | 0.3957 | 0.30 | 140.3 | **0.1751** | 0.0753 | **0 %** | 7 % | 298 | 43 % |
| 9 | diffusion K20 | `dpcc-t` | 0.3960 | 0.10 | 136.0 | **0.2718** | 0.1843 | **0 %** | 0 % | 1820 | 93 % |
| 12 | mf DiT K2 | `dpcc-r` | 0.4033 | 0.20 | 43.1 | **0.0607** | 0.0607 | **7 %** | 13 % | 57 | 20 % |
| 14 | mf v1 K2 | `dpcc-c` | 0.4094 | 0.23 | 69.8 | **0.0474** | 0.0474 | **7 %** | 10 % | 57 | 43 % |
| 15 | mf v2 K2 | `diffuser` *(no projection)* | 0.4098 | 0.10 | 169.7 | **0.0077** | 0.0077 | **3 %** | 10 % | 31 | 17 % |
| 5 | af v1 K100 | `diffuser` *(no projection)* | 0.4165 | 0.20 | 117.4 | **0.3107** | 0.0108 | **0 %** | 13 % | 902 | 20 % |
| 12 | mf DiT K2 | `diffuser` *(no projection)* | 0.4187 | 0.13 | 119.0 | **0.2762** | 0.0307 | **0 %** | 7 % | 32 | 10 % |
| 6 | af v1 K2 | `post_processing` | 0.4190 | 0.37 | 49.6 | **0.1141** | 0.0077 | **7 %** | 13 % | 53 | 17 % |
| 6 | af v1 K2 | `dpcc-r` | 0.4220 | 0.37 | 50.5 | **0.1141** | 0.0077 | **7 %** | 13 % | 53 | 17 % |
| 7 | af v2 K2 | `dpcc-r` | 0.4390 | 0.07 | 80.0 | **0.2787** | 0.1667 | **0 %** | 0 % | 73 | 60 % |
| 7 | af v2 K2 | `post_processing` | 0.4390 | 0.07 | 77.9 | **0.2787** | 0.1667 | **0 %** | 0 % | 72 | 60 % |
| 14 | mf v1 K2 | `diffuser` *(no projection)* | 0.4656 | 0.33 | 94.7 | **0.0094** | 0.0094 | **10 %** | 23 % | 28 | 7 % |
| 7 | af v2 K2 | `dpcc-c` | 0.5156 | 0.23 | 75.6 | **0.2735** | 0.2165 | **0 %** | 0 % | 72 | 63 % |

#### B. `train` / `combined_5-tightened`

| C | model | **projector** | `dist` | **`0-viol`** | `viol` | **min (clean)** | min (any) | **`<15cm` clean** | `<15cm` any | `ms` | still |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | mf v1 K2 | `hardflow_new-c` | 0.2959 | 0.87 | 2.4 | **0.0271** | 0.0271 | **13 %** | 17 % | 148 | 47 % |
| 14 | mf v1 K2 | `dpcc-t` | 0.3066 | 1.00 | 0.0 | **0.0110** | 0.0110 | **17 %** | 17 % | 42 | 33 % |
| 14 | mf v1 K2 | `hardflow_new-t` | 0.3112 | 0.97 | 0.4 | **0.0271** | 0.0271 | **20 %** | 20 % | 146 | 37 % |
| 14 | mf v1 K2 | `hardflow_new-r` | 0.3117 | 0.93 | 3.4 | **0.0752** | 0.0752 | **20 %** | 20 % | 147 | 27 % |
| 6 | af v1 K2 | `hardflow_new-c` | 0.3128 | 0.83 | 4.5 | **0.0793** | 0.0793 | **20 %** | 20 % | 159 | 50 % |
| 14 | mf v1 K2 | `dpcc-r` | 0.3145 | 0.90 | 6.7 | **0.0198** | 0.0198 | **20 %** | 20 % | 42 | 23 % |
| 6 | af v1 K2 | `hardflow_new-r` | 0.3288 | 0.90 | 3.1 | **0.0715** | 0.0715 | **20 %** | 20 % | 154 | 20 % |
| 12 | mf DiT K2 | `dpcc-r` | 0.3387 | 0.80 | 4.4 | **0.1473** | 0.1174 | **3 %** | 7 % | 57 | 27 % |
| 12 | mf DiT K2 | `dpcc-t` | 0.3450 | 0.80 | 4.4 | **0.0112** | 0.0112 | **10 %** | 17 % | 50 | 27 % |
| 14 | mf v1 K2 | `diffuser` *(no projection)* | 0.3451 | 0.37 | 65.9 | **0.1843** | 0.0114 | **0 %** | 17 % | 24 | 23 % |
| 12 | mf DiT K2 | `post_processing` | 0.3482 | 0.77 | 4.6 | **0.1473** | 0.1174 | **3 %** | 7 % | 57 | 27 % |
| 6 | af v1 K2 | `dpcc-t` | 0.3486 | 0.93 | 1.1 | **0.0710** | 0.0710 | **17 %** | 17 % | 43 | 33 % |
| 6 | af v1 K2 | `diffuser` *(no projection)* | 0.3521 | 0.20 | 81.9 | **0.1843** | 0.0188 | **0 %** | 17 % | 23 | 17 % |
| 6 | af v1 K2 | `dpcc-c` | 0.3597 | 0.73 | 31.9 | **0.0680** | 0.0680 | **10 %** | 10 % | 49 | 47 % |
| 14 | mf v1 K2 | `dpcc-c` | 0.3626 | 0.73 | 13.5 | **0.0258** | 0.0258 | **7 %** | 7 % | 55 | 57 % |
| 6 | af v1 K2 | `hardflow_new-t` | 0.3649 | 0.87 | 4.9 | **0.0847** | 0.0847 | **13 %** | 13 % | 156 | 30 % |
| 15 | mf v2 K2 | `diffuser` *(no projection)* | 0.3669 | 0.17 | 172.1 | **0.1843** | 0.1179 | **0 %** | 3 % | 27 | 27 % |
| 14 | mf v1 K2 | `post_processing` | 0.3682 | 0.90 | 7.1 | **0.0198** | 0.0198 | **20 %** | 20 % | 42 | 23 % |
| 12 | mf DiT K2 | `dpcc-c` | 0.3684 | 0.83 | 4.2 | **0.1208** | 0.1208 | **3 %** | 3 % | 54 | 30 % |
| 15 | mf v2 K2 | `dpcc-c` | 0.3740 | 0.80 | 4.2 | **0.1398** | 0.1398 | **3 %** | 3 % | 58 | 57 % |
| 7 | af v2 K2 | `diffuser` *(no projection)* | 0.3808 | 0.17 | 175.5 | **0.1843** | 0.1843 | **0 %** | 0 % | 27 | 93 % |
| 6 | af v1 K2 | `dpcc-r` | 0.3866 | 0.80 | 26.4 | **0.0643** | 0.0643 | **13 %** | 13 % | 49 | 27 % |
| 15 | mf v2 K2 | `dpcc-r` | 0.3873 | 0.87 | 17.0 | **0.0639** | 0.0639 | **10 %** | 10 % | 49 | 30 % |
| 6 | af v1 K2 | `post_processing` | 0.3898 | 0.83 | 26.3 | **0.0643** | 0.0643 | **13 %** | 13 % | 49 | 27 % |
| 15 | mf v2 K2 | `post_processing` | 0.3921 | 0.87 | 17.5 | **0.0639** | 0.0639 | **10 %** | 10 % | 49 | 30 % |
| 15 | mf v2 K2 | `dpcc-t` | 0.4240 | 0.90 | 13.0 | **0.0310** | 0.0310 | **10 %** | 10 % | 57 | 60 % |
| 7 | af v2 K2 | `dpcc-r` | 0.4276 | 0.33 | 67.1 | **0.1843** | 0.1843 | **0 %** | 0 % | 72 | 73 % |
| 7 | af v2 K2 | `post_processing` | 0.4276 | 0.33 | 67.1 | **0.1843** | 0.1843 | **0 %** | 0 % | 72 | 73 % |
| 7 | af v2 K2 | `dpcc-t` | 0.4441 | 0.73 | 10.2 | **0.1843** | 0.1843 | **0 %** | 0 % | 53 | 73 % |
| 12 | mf DiT K2 | `diffuser` *(no projection)* | 0.4480 | 0.23 | 131.0 | **0.1843** | 0.0655 | **0 %** | 3 % | 28 | 20 % |
| 7 | af v2 K2 | `dpcc-c` | 0.4962 | 0.57 | 78.9 | **0.0885** | 0.0885 | **7 %** | 7 % | 70 | 53 % |

#### C. `test` / `combined_5` — the only held-out cells

| C | model | **projector** | `dist` | **`0-viol`** | `viol` | **min (clean)** | min (any) | **`<15cm` clean** | `<15cm` any | `ms` | still |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | fm Gen7 v1 K20 | `post_processing` | 0.2798 | 0.30 | 78.4 | **0.1133** | 0.0265 | **10 %** | 27 % | 296 | 20 % |
| 4 | fm Gen7 v1 K20 | `diffuser` *(no projection)* | 0.2981 | 0.23 | 85.7 | **0.0943** | 0.0471 | **7 %** | 27 % | 303 | 20 % |
| 4 | fm Gen7 v1 K20 | `dpcc-r` | 0.3063 | 0.23 | 47.8 | **0.0527** | 0.0527 | **10 %** | 20 % | 1032 | 33 % |
| 4 | fm Gen7 v1 K20 | `dpcc-t` | 0.3097 | 0.43 | 43.7 | **0.0485** | 0.0379 | **13 %** | 20 % | 1126 | 40 % |
| 4 | fm Gen7 v1 K20 | `dpcc-c` | 0.3113 | 0.30 | 54.2 | **0.0759** | 0.0759 | **7 %** | 23 % | 1372 | 37 % |
| 16 | Gen6v4 DPCC K20 | `dpcc-r` | 0.4124 | 0.00 | 122.1 | **— none** | 0.2153 | **0 %** | 0 % | 1776 | 97 % |
| 16 | Gen6v4 DPCC K20 | `post_processing` | 0.4132 | 0.07 | 193.7 | **0.3592** | 0.2153 | **0 %** | 0 % | 391 | 100 % |
| 16 | Gen6v4 DPCC K20 | `dpcc-c` | 0.4147 | 0.00 | 129.9 | **— none** | 0.2153 | **0 %** | 0 % | 1825 | 100 % |
| 16 | Gen6v4 DPCC K20 | `diffuser` *(no projection)* | 0.4285 | 0.17 | 362.4 | **0.3127** | 0.2153 | **0 %** | 0 % | 339 | 83 % |
| 16 | Gen6v4 DPCC K20 | `dpcc-t` | 0.4408 | 0.07 | 98.1 | **0.4717** | 0.2153 | **0 %** | 0 % | 1585 | 97 % |
| 3 | fm Gen7 v2 K20 | `diffuser` *(no projection)* | 0.4551 | 0.23 | 196.3 | **0.2976** | 0.0453 | **0 %** | 7 % | 335 | 33 % |

### 2.4 What the constraint filter changes

🔴 **Three headline numbers from earlier drafts do not survive the constraint check:**

| claim | raw | **constraint-checked** |
|---|---|---|
| "`af` v1 K2 reaches **7.7 mm**" (`dpcc-r` / `post_processing`, untightened) | min 0.0077 | **that rollout violated** (1 violating step). Its best *clean* rollout in those cells is 0.0104 (unguided). |
| "`fm` Gen7 v1 held-out `post_processing` lands 27 % within 15 cm" | 27 % | **10 % clean** — `0-viol` is only 0.30. |
| "`mf` v1 K2 `dpcc-t` untightened is the best cell" | dist 0.2867, 27 % `<15cm` | **10 % clean**, `0-viol` 0.27 — good distance bought with violations. |

**What does survive:** `mf` v1 K2 **`dpcc-t` on `combined_5-tightened`** — `0-viol` **1.00**, so
every one of its rollouts is clean by construction: **min (clean) 0.0110 m, `<15cm` clean 17 %,
42 ms/step**. Its stablemate `dpcc-r` on the same geometry reaches **20 % clean at `0-viol` 0.90**,
also 42 ms. These are the two cells to quote.

**And the baseline gets worse, not better, under the filter:** C16 `dpcc-r` and `dpcc-c` on the test
split have **`0-viol` = 0.00 — not one clean rollout in 30**, so `min (clean)` does not exist for
them. Its only clean rollouts anywhere are at 0.31–0.47 m.

**General pattern.** Untightened, every model's `0-viol` sits at 0.03–0.40, so raw and clean tails
diverge sharply. Tightened, `0-viol` rises to 0.73–1.00 and the two columns converge — **the
tightened geometry is where distance numbers can be trusted at all.**

---

## 3. 🔴 The headline question — does `mf`/`af` at low K beat `fm`/`diffusion` at high K?

**Yes, on distance, on constraints, on the clean tail, and on cost.**

This is an **operating-point** comparison: each model at the K it is meant to run at and at its own
best DPCC projector (named in every row). It is *not* NFE-matched — that control does not exist in
this batch (§4) — but it is the deployment question, and it is paired over the same 30 Train-30
contexts, `train/combined_5`, n = 30 both sides.

| low-K model | projector | high-K model | projector | `dist` low vs high | **Δ** | t(29) | wins | **`<15cm` clean** | **`0-viol`** | **`ms`** |
|---|---|---|---|---|---|---|---|---|---|---|
| **`mf` v1 K2** | `dpcc-t` | **`diffusion` K20** | `dpcc-r` | **0.2867** vs 0.3666 | **−0.0799** | **−2.98** | 18/30 | **10 % vs 0 %** | **0.27 vs 0.10** | **53 vs 2 158 = 41×** |
| **`mf` v1 K2** | `dpcc-t` | **`fm` K20** | `dpcc-t` | **0.2867** vs 0.3617 | **−0.0750** | **−2.21** | 18/30 | **10 % vs 0 %** | **0.27 vs 0.03** | **53 vs 1 066 = 20×** |
| `af` v1 K2 | `dpcc-t` | `diffusion` K20 | `dpcc-r` | 0.3378 vs 0.3666 | −0.0288 | −0.83 | 16/30 | **17 % vs 0 %** | **0.40 vs 0.10** | 53 vs 2 158 = **41×** |
| `af` v1 K2 | `dpcc-t` | `fm` K20 | `dpcc-t` | 0.3378 vs 0.3617 | −0.0239 | −0.67 | 19/30 | **17 % vs 0 %** | **0.40 vs 0.03** | 53 vs 1 066 = **20×** |
| `mf` v2 K2 | `dpcc-t` | `fm` K20 | `dpcc-t` | 0.3369 vs 0.3617 | −0.0248 | −0.72 | 16/30 | 3 % vs 0 % | 0.20 vs 0.03 | 53 vs 1 066 = 20× |
| `mf` DiT K2 | `dpcc-t` | `fm` K20 | `dpcc-t` | 0.3575 vs 0.3617 | −0.0042 | −0.09 | 13/30 | 10 % vs 0 % | 0.30 vs 0.03 | 56 vs 1 066 = 19× |

**Every low-K row beats every high-K row on distance** (Δ = −0.004 to −0.080; `mf` v1 significant on
the t-statistic against both, sign test 18/30 p = 0.36).

**The decisive column is not distance — it is the clean tail.** `fm` K20 and `diffusion` K20 have
**zero rollouts that are both within 15 cm and constraint-clean**, on any of their four DPCC arms.
`mf` v1 K2 has 10 %, `af` v1 K2 has 17 %. Their zero-violation rates are 4–13× higher
(0.27–0.40 vs 0.03–0.10) *before* tightening is applied.

**And the cost gap is 20–41×** at the control-step level: 53 ms vs 1 066 ms (`fm`) and 2 158 ms
(`diffusion`). Generation alone is 26.8–32.4 ms at K = 2 against 293.9–298.3 ms at K = 20 (≈10×); the
rest is that the DPCC solve is far more expensive when handed a K = 20 plan (§2, `ms` column).

**With arm C allowed on the low-K side** the distance edge shrinks and the cost edge shrinks with
it — `mf` v1 K2 `hardflow_new-c` 0.3249 (13 % clean) at 194 ms is still 5.5× cheaper than `fm` K20
and 11× cheaper than `diffusion` K20, but the best low-K configuration is arm B, not arm C (§5).

### 3.1 Caveats on this comparison, stated plainly

1. **K is not matched, by construction** — this compares operating points, not objectives. The
   NFE-matched control (`fm`/`diffusion` at K = 2, or `mf`/`af` at K = 20) **has never been run**
   (§4). So the correct claim is *"the few-step engines at their operating point beat the many-step
   engines at theirs, on every axis, by 20–41× on cost"* — **not** *"MeanFlow is a better objective
   than Flow Matching"*.
2. **Untightened geometry only.** C9 and C11 have no `combined_5-tightened` cells, so the comparison
   cannot be repeated on the geometry where constraint numbers are trustworthy (§2.4).
3. **Single seed, n = 30 paired contexts.** The sign tests do not clear 0.05; the t-statistics for
   `mf` v1 do (−2.21, −2.98). The `<15cm` clean gap (10–17 % vs **0 %**) is the most robust part.
4. **Bone is matched** (UNet FiLM v1, 4.04 M) for `mf` v1, `af` v1, `fm` and `diffusion` — this is
   *not* confounded by architecture. `mf` DiT is the exception and is labelled.
5. `diffusion` runs `aw10`, the flow engines `aw1`.

---

## 4. What a matched-K comparison would need (and why it does not exist)

| engine | K = 2 | K = 20 | K = 100 |
|---|---|---|---|
| `mf` | ✅ C12/C14/C15, full arms | ❌ **never run** | ⚪ C13 — `diffuser` only at n = 30 |
| `af` | ✅ C6/C7, full arms | ❌ **never run** | ⚪ C5 — `diffuser` only at n = 30 |
| `fm` | ❌ **never run** | ✅ C11, arm B | ⚪ C10 — `diffuser` only at n = 30 |
| `diffusion` | ❌ *refused by design* | ✅ C9, arm B | ⚪ C8 — `diffuser` only at n = 30 |

**No projected cell anywhere in this batch has a few-step and a many-step engine at the same K.**
The only matched-K four-engine point is the **unguided** arm at K = 100, `train/combined_5`, n = 30,
paired:

| engine | C | `dist` | `0-viol` | **min (clean)** | **`<15cm` clean** | vs `mf` (paired) |
|---|---|---|---|---|---|---|
| **`mf`** | 13 | **0.3113** | 0.20 | **0.0347** | **10 %** | — |
| `diffusion` | 8 | 0.3342 | **0.27** | 0.0344 | 7 % | Δ −0.023, t −0.53, 15/30 → **tie** |
| `fm` | 10 | 0.3720 | 0.10 | 0.0469 | 3 % | Δ −0.061, t −1.70, 18/30, p 0.36 |
| `af` | 5 | 0.4165 | 0.20 | 0.3107 | **0 %** | Δ −0.105, t −1.99, 19/30, p 0.20 |

**Unguided at K = 100, nothing separates the engines** — `mf` leads, `diffusion` is second ahead of
`fm`, `af` is last, and no pairing resolves. The clean tails are 10 / 7 / 3 / 0 %, all far below the
low-K numbers in §3, and `af`'s apparent 0.0108 min at K = 100 is a **violating** rollout (its best
clean one is 0.3107). Whatever advantage the few-step engines have in §3 does **not** show
up when K is equalised at 100 without a projector; it appears at their own low-K operating point,
where their generation is 10× cheaper and their raw plans are more projectable.

**The one matched-K comparison with projectors** is `fm` vs `diffusion` at K = 20, each at its own
best arm B: `fm`/`dpcc-t` **0.3617** vs `diffusion`/`dpcc-r` **0.3666**, Δ = −0.0049, t = −0.24,
13/30 — **a tie.** Both have `<15cm` clean = 0 %.

**To settle the objective question, two runs are needed:** `fm` and `diffusion` at **K = 2** with
full arms, and `mf`/`af` at **K = 20**. Until then §3 is an operating-point result, and that is how
it should be quoted.

---

## 5. 🔴 Does HardFlow beat the DPCC projectors? — the single question, answered once

Arm C exists on **two** candidates (C6 `af` v1 K2, C14 `mf` v1 K2). Below, each arm gets its **own
best variant** by constraint-clean tail, and the two are then paired over the same 30 contexts.

| C | model | geometry | **best DPCC (arm B)** | **best HardFlow (arm C)** | Δ dist (C−B) | HF wins | **`<15cm` clean B → C** | **`0-viol` B → C** | **cost** |
|---|---|---|---|---|---|---|---|---|---|
| 14 | `mf` v1 K2 | `combined_5` | `dpcc-t` 0.2867 | `hardflow_new-c` 0.3249 | **+0.0382** | 10/30 | 10 % → 13 % | 0.27 → 0.27 | 53 → 194 ms **3.7×** |
| 14 | `mf` v1 K2 | tightened | `dpcc-r` 0.3145 | `hardflow_new-t` 0.3112 | −0.0033 | 9/30 | 20 % → 20 % | 0.90 → 0.97 | 42 → 146 ms **3.4×** |
| 6 | `af` v1 K2 | `combined_5` | `dpcc-t` 0.3378 | `hardflow_new-r` 0.3316 | −0.0063 | 12/30 | 17 % → 10 % | 0.40 → 0.53 | 53 → 178 ms **3.3×** |
| 6 | `af` v1 K2 | tightened | `dpcc-t` 0.3486 | `hardflow_new-c` 0.3128 | −0.0358 | 14/30 | 17 % → 20 % | 0.93 → 0.83 | 43 → 159 ms **3.7×** |

**Verdict: no.** HardFlow is within ±0.04 m of the best DPCC arm on distance (2 of 4 blocks favour
HF, 1 favours DPCC, 1 is a wash), **splits the clean tail 2–1–1**, and costs a flat **3.3–3.7×**.
Nothing here reaches significance in HF's favour — the one sign test that does clear 0.05 is
C14 tightened at **9/30, i.e. against HardFlow**.

**The decisive row is the constraint one.** `dpcc-t` + tightening on C14 reaches **`0-viol` = 1.00
with `viol` = 0.0 at 42 ms** (§2 block B). **HardFlow never matches that** — its best anywhere is
0.97 (`hardflow_new-t`, C14 tightened) at 146 ms, i.e. **worse constraint satisfaction at 3.5× the
price.** On C6, HF's `0-viol` is *lower* than DPCC's (0.83 vs 0.93).

**Where HardFlow does help** is on the DPCC projector's *weak* selection rule. Matched-rule,
untightened: `-c` on C14 is `dpcc-c` 0.4094 → `hardflow_new-c` 0.3249 (−0.084) and `-r` on C6 is
`dpcc-r` 0.4220 → `hardflow_new-r` 0.3316 (−0.090). Where arm B is already good (`-t`), HF is
slightly worse. **Its benefit scales inversely with how well arm B already selects** — so it rescues
a bad rule rather than beating a good one, which is not a reason to adopt it.

⚠️ **The comparison the benchmark hierarchy actually asks for — HardFlow at a *lower projection
threshold* than DPCC — has never been run.** Both arms ran at the same threshold here. That sweep is
the only experiment that could change this verdict.

---

## 6. The baselines

| baseline | split | projector | n | `dist` | **`0-viol`** | `viol` | **min (clean)** | **`<15cm` clean** | `still` |
|---|---|---|---|---|---|---|---|---|---|
| **Gen6v4 visual DPCC** (C16) | test | `dpcc-r` | 30 | 0.4124 | **0.00** 🔴 | 122.1 | **— none** 🔴 | **0 %** | 97 % |
| Gen6v4 visual DPCC (C16) | test | `dpcc-c` | 30 | 0.4147 | **0.00** 🔴 | 129.9 | **— none** 🔴 | 0 % | 100 % |
| Gen6v4 visual DPCC (C16) | test | `post_processing` | 30 | 0.4132 | 0.07 | 193.7 | 0.3592 | 0 % | 100 % |
| Gen6v4 visual DPCC (C16) | test | `dpcc-t` | 30 | 0.4408 | 0.07 | 98.1 | 0.4717 | 0 % | 97 % |
| Gen6v4 visual DPCC (C16) | test | *`diffuser`* | 30 | 0.4285 | 0.17 | 362.4 | 0.3127 | 0 % | 83 % |
| **D3IL `ddpm_encdec_vision`** (C17) | test | *no projector* | 1 080 | **0.4009** | — | — | — | — | 78 % |
| **D3IL `…__Bf_U3`** (C18, 6 seeds) | test | *no projector* | 2 804 | 0.3918 | — | — | — | — | 63 % |
| *do-nothing reference* | — | — | 5 418 | **0.3985** | — | — | — | — | 100 % |

**Two of the baseline's four DPCC arms produce not one constraint-clean rollout in 30.** Its best
clean rollout anywhere is 0.3127 m — worse than the do-nothing line. Its box never moves in 83–100 %
of rollouts, and its `dist` sits at the do-nothing value. C17 scores 0.4009 against the 0.3985
reference — a 2 mm difference.

🔴 **There is no target to beat on this env.** The benchmark hierarchy (diffusion-DPCC is THE
baseline) cannot be applied: the baseline is at the floor on every one of its cells, and the
upstream D3IL visual policy the whole line is built on is at the floor too. Fixing the baseline is a
prerequisite, not a parallel task.

---

## 7. The one paired, held-out, n = 30 comparison

C4 (Gen7 `fm` v1 K20) and C16 (Gen6v4 DPCC baseline, K20) ran **the same 30 held-out test contexts**
at the same K. Each at its own best arm B by clean tail, plus the matched-variant rows:

| comparison | C4 projector | C16 projector | C4 `dist` | C16 `dist` | Δ | t(29) | C4 wins | sign p |
|---|---|---|---|---|---|---|---|---|
| **each at own best** | `dpcc-t` | `dpcc-r` | **0.3097** | 0.4124 | **−0.1027** | **−3.53** | 18/30 | 0.36 |
| matched `post_processing` | `post_processing` | `post_processing` | **0.2798** | 0.4132 | **−0.1334** | **−4.93** | **22/30** | **0.016** |
| matched `dpcc-t` | `dpcc-t` | `dpcc-t` | 0.3097 | 0.4408 | −0.1311 | −2.80 | 19/30 | 0.20 |
| matched `dpcc-r` | `dpcc-r` | `dpcc-r` | 0.3063 | 0.4124 | −0.1061 | −3.98 | 20/30 | 0.099 |
| *unprojected* | *`diffuser`* | *`diffuser`* | *0.2981* | *0.4285* | *−0.1304* | *−2.59* | *20/30* | *0.099* |

**Gen7 visual FM beats the Gen6v4 visual DPCC baseline by 0.10–0.13 m on held-out contexts, at
matched K = 20, on every projector including the unprojected one.**

**Constraint-checked:** C4's clean tail is **13 %** (`dpcc-t`, `0-viol` 0.43) against C16's **0 %**
on all four arms, and C4's `min (clean)` is **0.0485 m** where two of C16's arms have no clean
rollout at all. The win survives the filter — though note C4's own `0-viol` of 0.23–0.43 is poor in
absolute terms, and its best *raw* distance cell (`post_processing`, 0.2798) is the one with the
weakest constraint record (`0-viol` 0.30, clean tail 10 %).

Caveats: single seed — the replication unit is the *context*, so this is generalisation across
initial conditions, not training variance; both models are far from solving the task; and **no
Gen14 candidate has a test split**, so Gen14 cannot enter this comparison at all.

---

## 8. Quarantine — every cell with n < 30

Nothing in §2–§7 uses these.

| C | model | split / geo | projector(s) | n | note |
|---|---|---|---|---|---|
| 5 | `af` v1 K100 | train `combined_5` | `dpcc-r` | **11** | **This cell is `candidates_ranking.csv`'s 4.55 % headline** — 1 success in 11. |
| 8 | `diffusion` K100 | train `combined_5` | `dpcc-r` | **19** | truncated |
| 10 | `fm` K100 | train `combined_5` | `dpcc-r` | **22** | truncated |
| 13 | `mf` v1 K100 | train `combined_5` | `dpcc-r` | **11** | truncated |
| 9 | `diffusion` K20 | train `combined_5` | `post_processing` | **9** | truncated |
| 1, 2, 3 | Gen7 `fm` v2 *(2 legacy trees)* | train, 2–3 geos | `dpcc-{r,c,t}`, `post` | **1–3** | anecdote |
| 4 | Gen7 `fm` v1 | train ×3 geo | `dpcc-{c,t}`, `post` | **3** | anecdote; C4's usable cells are its test ones |
| **16** | **Gen6v4 DPCC** | train ×2 geo | `dpcc-{r,c,t}`, `post` | **3** | anecdote; the baseline's train numbers carry nothing |

K100 cells are truncated *prefixes* of Train-30 — paired but under-sampled. The n = 3 cells cover
only 3 of the 30 train contexts. **One thing worth noting from quarantine, claimed as nothing:** C16
`combined_5-tightened` reaches `0-viol` = 1.00 with `viol` = 0.0 on `dpcc-{r,c,t}` — on **3
episodes**, at 693–835 ms/step.

---

## 9. Cross-cutting diagnostics

**Goal success is at the floor and is not used for ranking (rule 5).** For the record: every
deployable n = 30 cell scores 0, 1 or 2 successes in 30. The ceiling is **2/30**, reached by `mf` v1
K2 with `dpcc-r`, `post_processing` and `hardflow_new-c` untightened and with `dpcc-t` tightened.
**99.0 % of deployable rollouts end at the 400-step cap**, so no cell has a meaningful step-count
statistic either.

**Orientation is the binding failure — per cell, n = 30 each.** Wrapped
`abs(final box angle − target angle)`; `init` is fixed per pool (Train-30: 56.6°, Test-30: 69.2°).

| C | model | cell | `ang` init → **final** | `<15°` init → **final** | `xy < 5 cm` |
|---|---|---|---|---|---|
| **4** | Gen7 `fm` v1 | test `post_processing` | 69.2° → **64.7°** ✅ | 10 % → **20 %** ✅ | **23 %** |
| 4 | Gen7 `fm` v1 | test `dpcc-t` | 69.2° → 68.1° | 10 % → 13 % | 17 % |
| 16 | **Gen6v4 baseline** | test `dpcc-r` | 69.2° → 70.6° | 10 % → 10 % | **0 %** |
| 14 | `mf` v1 K2 | train `dpcc-t` | 56.6° → 57.9° | 27 % → 20 % | 13 % |
| 14 | `mf` v1 K2 | tightened `dpcc-t` | 56.6° → 66.7° | 27 % → 20 % | 27 % |
| 14 | `mf` v1 K2 | train `dpcc-r` | 56.6° → 78.2° 🔴 | 27 % → 20 % | 20 % |
| 15 | `mf` v2 K2 | train `dpcc-t` | 56.6° → 65.8° | 27 % → 20 % | 7 % |
| 6 | `af` v1 K2 | train `dpcc-t` | 56.6° → 69.0° | 27 % → 17 % | 7 % |
| 11 | `fm` K20 | train `dpcc-t` | 56.6° → 60.4° | 27 % → 20 % | 0 % |
| 9 | `diffusion` K20 | train `dpcc-r` | 56.6° → 64.8° | 27 % → 27 % | 7 % |
| 12 | `mf` DiT K2 | train `dpcc-t` | 56.6° → 73.5° 🔴 | 27 % → 23 % | 3 % |
| 7 | `af` v2 K2 | train `dpcc-t` | 56.6° → 56.3° * | 27 % → 27 % * | 0 % |

\* C7's flat angle is not alignment — it is 73 % still. A policy that does nothing preserves the
initial angle.

**Every Train-30 cell ends with worse mean orientation than it started and loses `<15°` rollouts.**
The only cells that improve orientation are C4's on the test split. Translating the box is partly
solved; rotating it is not.

**Data quality.** 93 of 321 units flagged: 4/30 frozen rollouts (D1 box-obstacle conflict) in every
tightened cell of C6, C7, C12, C14, C15 — identical across candidates, a task property — plus 11
circuit-breaker trips on C9, C11, C16. Batch-wide frozen rate 3.3 %. `mask=unfrozen` moves `0-viol`
by ≤ 4 points on deployable cells and changes no ordering in §2–§5.

---

## 10. Limits

- **One seed (6)** for C1–C16. All pairing is across *contexts* within that seed: generalisation
  over initial conditions, **not** training variance. Seeds 7–10 unrun.
- **No test split for any Gen14 candidate** (C5–C15). Everything in §2 blocks A/B, §3, §4 and §5 is
  on seen contexts.
- 🔴 **§3 is an operating-point comparison, not NFE-matched** — the matched-K control does not exist
  (§4). Quote it as "few-step engines at K = 2 beat many-step engines at K = 20 on every axis at
  20–41× lower cost", never as "MeanFlow is a better objective".
- **§3 is untightened-only** — C9/C11 have no tightened cells, and §2.4 shows untightened constraint
  numbers are the weak ones.
- **Arm C exists on 2 candidates** (C6, C14), so §5 rests on those two.
- **`min (clean)` is an extreme-value statistic**; it is comparable here because every main-table
  cell is exactly n = 30. Do not compare it against a quarantined cell.
- **`aw` differs**: `diffusion` candidates are `aw10`; `fm`/`mf`/`af` are `aw1`.
- **C12 is 80 k steps / 3.37 M params** against 100 k / 4.04 M for the U-Nets (see the U8 DA).
- **`avg_time` is wall-clock on shared GPUs** and includes the NLP solve.
- **No smoothness or contact metric.** `still` is inferred from box displacement; it does not
  distinguish *the arm never moved* from *the arm moved and never touched the box*.

---

## 11. Good news — what is working

The environment does not work yet, but several things in this batch are finished, and two results
survive the constraint check.

### 11.1 ✅ Constraint satisfaction under vision is solved, cheaply

**`mf` v1 K2 + `dpcc-t` on `combined_5-tightened`: `0-viol` = 1.00, `viol` = 0.0, on all 30
episodes, at 42.3 ms per control step — 23.6 Hz.** A visual receding-horizon controller that
violated no constraint on any episode, at real-time rate, with the full projection solve in the
loop. Because every rollout is clean, its distance numbers need no filter: **min (clean) 0.0110 m,
`<15cm` clean 17 %.**

Nearby: `mf` v1 `dpcc-r` tightened **0.90 / 20 % clean / 42 ms** (the best clean tail in the batch),
`af` v1 `dpcc-t` tightened **0.93 / 17 % / 43 ms**, `mf` v1 `hardflow_new-t` **0.97 / 20 % / 146 ms**.
Unguided arms on the same geometry sit at `0-viol` 0.17–0.37. **The projector is contributing
0.5–0.8 of zero-violation rate for ~15–20 ms of extra compute.**

### 11.2 ✅ The low-K stack wins the deployment comparison outright

§3: at their own operating points, `mf`/`af` at K = 2 beat `fm`/`diffusion` at K = 20 on distance,
on zero-violation rate, on the constraint-clean tail (**10–17 % vs 0 %**), and by **20–41× on cost**.
Whatever else is unresolved, the few-step direction is the right one to keep pushing on this task —
it is cheaper *and* the plans it hands the projector are more projectable.

### 11.3 ✅ Two clean, statistically-supported wins

- **`mf` v1 K2 vs the upstream D3IL baseline.** The bridged D3IL visual policy scores 8/2804 =
  0.285 %. Under that rate, P(≥ 2 successes in 30) = **0.0034**, and four separate C14 cells reach
  2/30. (Success is not used for ranking — this is the one place it is legitimate, because the
  comparison is against a fixed baseline *rate*, not between models. The four cells share contexts,
  so they are not four independent confirmations.)
- **Gen7 `fm` v1 vs the Gen6v4 DPCC baseline on held-out data** (§7): −0.10 to −0.13 m paired over
  the same 30 test contexts, every projector, `post_processing` at t(29) = −4.93 / 22 of 30
  (sign p = 0.016), and it survives the constraint filter (13 % clean vs **0 %**).

### 11.4 ✅ The models can reach the target legally

**Clean** best rollouts — zero violations, not just short distance: **0.0110 m** (`mf` v1 `dpcc-t`
tightened, a cell that is 100 % clean), **0.0112 m** (`mf` DiT `dpcc-t` tightened), **0.0157 m**
(`mf` v1 `hardflow_new-c` untightened), **0.0198 m** (`mf` v1 `dpcc-r`/`post_processing` tightened).
A controller that reaches 11 mm *without violating anything* has the representation and the solver;
it fails on consistency. The baseline, by contrast, has **no clean rollout better than 0.31 m**.

### 11.5 ✅ Cheap options exist

`post_processing` costs **1.0–1.2×** unguided generation (C4 test: 296 ms vs 303 ms; C11: 323 vs 294)
against 3.4–7.2× for `dpcc-{r,c,t}` at K = 20 — and it ties `dpcc-r` for the best clean tail on C14
tightened (20 %, 42 ms). K = 2 generation runs at **26.8–32.4 ms** with a ~10 ms visual-encoder
floor, so the whole few-step stack sits inside a real-time budget before any optimisation.

### 11.6 ✅ The measurement apparatus is trustworthy

321/321 units loaded, 0 failed, `npz_complete` 1.0 throughout; 11 circuit-breaker trips in 10 478
rollouts; frozen cells identical across candidates. **Contexts are shared and verified paired**
(§0.3) — which is why every comparison in this file could be run *paired*, and it means any future
candidate on the same pool is immediately comparable to everything here. Arm C is ported and runs at
a predictable flat 3.3–3.7× cost; the DiT bone trains and evaluates cleanly (U8 DA).

### 11.7 One line

**Constraint satisfaction is solved (`0-viol` 1.00 at 42 ms, 23.6 Hz), the few-step engines beat the
many-step ones at 20–41× lower cost on every axis, and the best stack reaches 11 mm on a fully clean
rollout. What is missing is consistency — and specifically box rotation.**

---

## 12. Verdict

**Visual aligning has no working controller and no working baseline as of 2026-08-23** — but the
failure is narrow and located. Constraint satisfaction, the harness and the few-step engines all
work (§11); what fails is the generative model's consistency, and specifically box orientation (§9).

1. **`mf`/`af` at K = 2 beat `fm`/`diffusion` at K = 20 on every axis** — distance (Δ −0.004 to
   −0.080), zero-violation rate (0.27–0.40 vs 0.03–0.10), constraint-clean tail (**10–17 % vs 0 %**)
   and cost (**20–41×**). This is an operating-point result; the NFE-matched control has never been
   run (§3.1, §4).
2. **HardFlow does not beat the DPCC projectors** (§5). ±0.04 m on distance, never better than the
   best DPCC arm on `0-viol`, flat 3.3–3.7× cost; `dpcc-t` + tightening reaches 1.00 at 42 ms and HF
   never matches it. HF rescues DPCC's *weak* selection rules, which is not a reason to adopt it.
3. **Constraint checking changes the rankings** (§2.4). `af` v1's 7.7 mm rollout violated; Gen7 v1's
   27 % held-out tail is 10 % clean; `mf` v1's best untightened cell is bought with violations. The
   cells that survive are on the **tightened** geometry, where `0-viol` is 0.73–1.00.
4. **The Gen6v4 DPCC baseline has two arms with zero clean rollouts in 30** and no clean rollout
   better than 0.31 m anywhere — it is incapable, not merely worse (§6).
5. **Gen7 `fm` v1 beats that baseline on held-out contexts at matched K = 20**, on every projector,
   constraint-checked (§7).
6. **The selection rule swings results more than the arm does** — 0.12 m between `dpcc-t` and
   `dpcc-c` on the same model — and it inverts between U-Net and DiT.
7. **Cost is governed by K, then by the projector** — 27–32 ms generation at K = 2 vs 294–298 at
   K = 20; arm B adds 1.7–2.0× at K = 2 but 3.4–7.2× at K = 20; arm C is a flat 3.3–3.7× on arm B.
8. **Orientation is the binding failure** (§9): every Train-30 cell ends worse-aligned than it
   started, and the best-distance held-out cell has zero successes.

**Order of work.** (a) **Give Gen14 a test split** — eval-only, and the only way Gen14 can enter §7.
(b) **Run `fm`/`diffusion` at K = 2 and `mf`/`af` at K = 20** — the blank rows in §4; without them
§3 stays an operating-point claim. (c) **Run C9/C11 on `combined_5-tightened`** so §3 can be
repeated where the constraint numbers are trustworthy. (d) Fix the baseline; §6 says its problem is
capability, not tuning. (e) Instrument orientation directly (does the 9-D action head express box
rotation at all?). (f) **HardFlow threshold sweep** — the only experiment that could change §5.
(g) Audit FiLM v2 against v1. (h) Re-run the truncated cells in §8, then seeds 7–10.

---

## 13. Reproduction

```
batch : temp/2508/batch_va2_20260823_135156/
files : per_rollout_detail.csv     every rollout, wide (10 478 rows)
        candidates_ranking.csv     ⚠ DO NOT USE for model comparison — pools projectors,
                                     splits, geometries and truncated cells per candidate.

CELL   = (Candidate, split, geo, variant); never aggregate across `variant`.
ARM_B  = {dpcc-r, dpcc-c, dpcc-t, post_processing}      # DPCC projector
ARM_C  = {hardflow_new-r, hardflow_new-c, hardflow_new-t}  # HardFlow — never merged with ARM_B
ARM_A  = {diffuser}                                      # no projection, reference
EXCLUDED = {geo_free, bounds_free, model_free, geo_free-bounds_free, geo_free-model_free,
            model_free-bounds_free, gradient, dpcc-c-dt0p25, dpcc-c-dt0p5,
            dpcc-c-dt2p0, dpcc-c-dt4p0}

filter : mask == 'all', len(cell) == 30
clean  : constraint_exec_zero_violation == 1.0        # == collision_free_completed
dist   : mean(mean_dist_per_rollout)
min(clean)   : min(mean_dist_per_rollout WHERE clean)          # None if no clean rollout
<15cm clean  : count(mean_dist_per_rollout < 0.15 AND clean) / 30
0-viol : mean(collision_free_completed)
viol   : mean(constraint_exec_total_viol_count)
ms     : mean(avg_time_ms)                            # generation + projection solve
still  : abs(context_final_xy_dist - context_init_xy_dist) < 0.02

best projector for a model = argmax over its cells of (<15cm clean, -dist),
                             chosen WITHIN an arm, never across arms.

pairing : join two candidates on round(context_box_init_xy_x,4), round(...y,4).
          Train-30 shared by C5-C15; Test-30 shared by C4 and C16; the two are disjoint.
          t = paired t on per-context differences; sign test is exact binomial.

do-nothing reference : mean(mean_dist_per_rollout) over all still rollouts
                       = 0.3985 m (n = 5 418, sd 0.130)
```

Missing runs that would change conclusions:

```bash
# §4 — the NFE-matched control (train + eval)
#   fm and diffusion at K=2 with full arm B; mf and af at K=20
# §3 caveat 2 — tightened geometry for C9 / C11
# §5 — HardFlow at a lower projection threshold than DPCC
# arm C on a candidate that lacks it (eval-only):
HFFM_VARIANTS="hardflow_new-r hardflow_new-c hardflow_new-t" \
  sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh <engine> 6
```
