# Gen15 — the five-mission UAV campaign (submitted 2026-09-07 04:21 UTC)

> **This is not `logs_in_develop/MASTER_TEST_HISTORY.md`.** That file is the repo-wide chronological
> index. *This* file is the local master for one campaign: the five UAV tests submitted on
> 2026-09-07, what each is supposed to answer, and how far each actually got.

*Gen15 · campaign folder · opened 2026-09-08 · owner thread: the UAV/`af_unet` chat (lost in the
2026-09-07 Claude Code history wipe — see `.claude/CRISIS_RECOVERY_2026-09-07_lost_history.md`).*

---

## 0. What we are doing

The thesis claim ladder (`Writing/Working_Space/TARGET_20260905_thesis_claim_ladder.md`) wants
**af_unet > mf > fm > diffusion (DPCC baseline)**. Gen15 is the UAV leg of that ladder. By Sep 6 the
α-Flow U-Net (`af_unet`, 3.97 M params) was *mechanically* proven on UAV — α genuinely on at 0.2,
checkpoint plumbing resolving — but it had never been ranked against `mf`/`fm` on a scene with a
usable dynamic range, and the `s_curve` scene was failing outright.

The five missions are the campaign that was supposed to close that. They split into three questions:

| | question | missions |
|---|---|---|
| **A** | Does the af_unet arm hold up at matched K on a *second* and *third* scene? | **1** (pillars), **2** (corridor) |
| **B** | Does HardFlow-SLSQP beat the DPCC projector once K is high enough to be genuine? | **3** (pillars K=5, fm/mf/af) |
| **C** | Why does `s_curve` fail — is it the **budget**, or is it the **controller**? | **4** (high K), **5** (mjpc vs pid) |

C is the interesting one. Mission 4 asks *"give it more NFE and does it recover?"*; mission 5 asks
*"is the policy fine and the PID tracker simply not strong enough to fly what it plans?"* — the
question the user posed on Sep 7 as *"let's see if the controller is not powerful."*

### 🔴 What this campaign structurally cannot deliver

**No mission runs `engine=diffusion`.** The wave is `af`, `fm` and `mf` only. Per
[[benchmark-hierarchy-who-beats-whom]] the diffusion-DPCC arm is *the* baseline every claim must
clear, so this campaign **cannot on its own produce the headline result** — it can only order
af/mf/fm among themselves. A `diffusion` arm at matched K on the same scenes is still owed.

Two naming traps that bite readers of these logs:

* the variant called **`diffuser`** is **not** the diffusion engine. It is the *no-projection* row
  (`proj=off`) of whatever engine is loaded. Under `engine=af` it is the raw α-Flow U-Net plan.
* **`-tightened`** and **`-geo_free`** are projector configurations, not engines.

---

## 1. The five missions and every Slurm index

| # | test | wrapper → children | engine · scene · K | state |
|---|---|---|---|---|
| **1** | af_unet K-sweep, **pillars** | `25486` → **25497** (K1), **25499** (K2) | af · pillars · 1,2 | ✅ **complete** |
| | | `25490` → **25501** (K5) | af · pillars · 5 | ❌ cancelled 5/17 → resubmit **25542** ⚠️ malformed |
| **2** | af_unet pipeline, **corridor** | `25487` → **25494** train, **25495** (K1), **25498** (K2) | af · corridor · 1,2 | ✅ **complete** → [**DA**](DA_20260908_T2_af_unet_corridor_K1_K2.md) |
| **3** | **HF-SLSQP** vs DPCC projector, pillars K=5 | `25488` → **25496** | fm · pillars · 5 | ✅ 17/17 |
| | | `25489` → **25503** | mf · pillars · 5 | 🟡 running |
| | | `25490` → **25501** *(shared with #1)* | af · pillars · 5 | ❌ no HF row ran → **25542** ⚠️ |
| **4** | **s_curve** explore at high K | `25491` → **25500** | fm · s_curve · 20 | ❌ 6/8 → resubmit **25543** |
| | | `25492` → **25502** | mf · s_curve · 10 | 🟡 running |
| **5** | **mjpc vs pid** controller, worst s_curve case | **25514** | s_curve, fewer trials | 🔴 **never started** |

Shared knobs across the wave: `FMPCC_SAFE_EPS_MODE=scaled`, `FMPCC_UAV_EVAL_TAG=u7hg`,
`proj=fm_only`, `record=none`, seed **6**, **n_trials=10** (from `config/uav_projection.yaml`),
`controller=pid_stopgo`. The af arm additionally carries `UAV_MIX_BONE_AF=unet`,
`UAV_MIX_AF_ALPHA_END=0.2`, `UAV_MIX_EPOCH=latest`.

Missions 1 and 3 **share job `25490 → 25501`** — the af pillars K=5 leg is one job serving two
questions, which is why a single cancellation opened two holes.

---

## 2. Live status

Full ledger, with cancellation forensics and ETAs:
[`../U10/RUNSTATUS_20260908_wave_25486_25514_status.md`](../U10/RUNSTATUS_20260908_wave_25486_25514_status.md)

| | |
|---|---|
| 🟡 running | `25502` (mf s_curve K10, wall 09-09 10:30) · `25503` (mf pillars K5, wall 09-09 10:50) |
| 🔴 pending, CPU-quota blocked | `25514` (mission 5) · `25542` (mission 1+3 resume, **malformed**) · `25543` (mission 4 resume) |
| ⚠️ needs action | `25542` carries a newline inside `UAV_MIX_VARIANTS` → child exits 2. `scancel` + resubmit on one line. |

---

## 3. Findings

### Mission 2 — analysed ✅ [`DA_20260908_T2_af_unet_corridor_K1_K2.md`](DA_20260908_T2_af_unet_corridor_K1_K2.md)

Source: `temp/0809/batch_uav_20260908_153947` (`DA_UAV_v1`, 1263 units, 0 failed).

1. **`corridor` is saturated** — S&C = 1.000 in all 20 af cells and all 10 mf cells.
2. 🔴 **af_unet ≡ mf_unet at K=2.** af leads 5/10 variants, mf 5/10, mean |Δ| = 0.83 steps against
   σ = 5–14. **The ladder step `af > mf` is not supported on this scene.**
3. **fm is the only engine corridor breaks** — S&C 0.800 on all three `temporal_consistency` rows
   (goal-reach failures, `phys_safe` stays 1.000) and **3–6× the step dispersion** of af/mf.
   `af, mf > fm` holds, weakly.
4. **K=1→K=2 is flat for af**: +92 % network time for −1.9 % track_err and +1.9 % steps.
5. The projector trades tracking fidelity for path length **monotonically and identically in all
   three engines**; `-geo_free` gets ~99 % of the quality at 1/14 the projection cost of `-tightened`.

**Consequence:** `corridor` joins `s_curve` (floored at 0.00) and pre-U7 `pillars` (void) as a scene
that cannot rank this arm. The af_unet UAV arm still has **no scene with usable dynamic range** —
which is exactly what mission 5 exists to attack.

### The other four — not yet analysed

A DA may only be written from the `DA_UAV_v1` batch CSVs, never from the sbatch/eval job logs —
see [[da-requires-csv-never-from-logs]]. Missions 1, 3, 4 and 5 have no complete result set in
`batch_uav_20260908_153947`, so they stay un-analysed until their runs land and a fresh batch is cut.

Run status — which job finished, which died, what is queued — is *not* a DA and lives in
[`../U10/RUNSTATUS_20260908_wave_25486_25514_status.md`](../U10/RUNSTATUS_20260908_wave_25486_25514_status.md).

### DA readiness per mission

| # | results on disk? | DA can start when |
|---|---|---|
| **1** pillars K1/K2 | ✅ complete (25497, 25499) | a `batch_uav_*` drop covering them arrives |
| **1/3** pillars K5 af | ❌ partial (25501, 5/17) + resubmit 25542 pending | 25542 is fixed and lands |
| **2** corridor K1/K2 | ✅ complete (25495, 25498) | ✅ **DONE** — `batch_uav_20260908_153947` |
| **3** pillars K5 fm/mf | fm ✅ (25496) · mf 🟡 (25503 running) | 25503 lands, then a batch covering all three engines at matched K=5 |
| **4** s_curve | fm ❌ partial (25500, 6/8) + 25543 pending · mf 🟡 (25502 running) | both land |
| **5** mjpc vs pid | ❌ nothing — 25514 never started | 25514 runs at all |

### Carry-over context (from earlier, separately analysed work — not this wave)

Prior DA on the af_unet UAV arm, for framing only:
`../DA/DA_20260907_af_unet_uav_s_curve_pillars_K_sweep.md` reports S&C = 0.00 on 27 of 30 legal
`s_curve` cells and raw-plan success *falling* with K, and marks pre-U7 `pillars` candidates 46/48
**void for ranking**. That is the state this campaign was launched to move past; it is not a result
*of* this campaign.

---

## 4. Files in this folder

| file | contents |
|---|---|
| `MASTER_20260908_five_missions_campaign.md` | this file — campaign definition, index, status, DA readiness |
| `DA_20260908_T2_af_unet_corridor_K1_K2.md` | **mission 2** DA — af_unet on `corridor`, K=1/K=2, vs mf and fm at matched K=2 |

Planned, as the runs land: a DA per remaining mission, then a cross-mission closure.

---

## 5. Provenance

Job **logs** (status only, never DA input): `temp/0809/2026-09-07/` (supersedes `temp/0609/II/2026-09-07/`, which was missing
25501/25502/25503). No aggregated CSVs cover this wave yet — the last DA batch,
`batch_uav_20260907_141115`, ran at 14:11 on Sep 7, **before** 25500–25503 wrote anything. A fresh
`DA_UAV_v1` batch is required once the running jobs land.
