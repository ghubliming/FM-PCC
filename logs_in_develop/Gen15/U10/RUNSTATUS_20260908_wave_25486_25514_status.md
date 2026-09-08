# Run status 2026-09-08 — the 25486–25514 wave: what finished, what died, what never started

*Gen15 · U10 · source: `temp/0809/2026-09-07` (sbatch logs, incl. the three that were missing from
the `temp/0609/II` drop: 25501/25502/25503) + a `squeue` snapshot at **≈2026-09-08 12:54 UTC**.
Successor to [`../U6/RUNSTATUS_20260905_af_unet_resubmit_25434_25439_and_cleanup.md`](../U6/RUNSTATUS_20260905_af_unet_resubmit_25434_25439_and_cleanup.md).
Reconstructed after the Claude Code history loss — see `.claude/CRISIS_RECOVERY_2026-09-07_lost_history.md` §5.*

## 0. TL;DR

1. **Six of the seven Sep-7 jobs that were *supposed* to finish did not.** Four evals completed
   clean (25494/25495/25497/25498/25499 + 25496); **two were `scancel`led part-way** on Sep 8
   morning (25500, 25501) and **two are still running** (25502, 25503). §2
2. 🔴 **Mission 5 — `25514`, the mjpc-vs-pid controller test — has never started.** `TIME 0:00`,
   reason **`QOSMaxCpuPerUserLimit`**. It is not waiting on a GPU (it requests none); it is blocked
   by **the user's own two running jobs**. It cannot start before ~Sep 9 unless one is cancelled. §3
3. The dossier's §5 "✅ delivered" for missions 1/3/4 is **too generous** — those arms are partial.
   Corrected mission table in §4.

---

## 1. The wave, as submitted (Sep 7 04:21 UTC)

All seven wrappers ran and exited fine; they only submit children.

| wrapper | kind | arm | children |
|---|---|---|---|
| **25486** | `eval_k_sweep` | af · pillars · K=1,2 | 25497, 25499 |
| **25487** | `ksweep_pipeline` | af · corridor | train **25494** → 25495 (K1), 25498 (K2) |
| **25488** | `eval_k_sweep` | fm · pillars · K=5 | 25496 |
| **25489** | `eval_k_sweep` | mf · pillars · K=5 | **25503** |
| **25490** | `eval_k_sweep` | af · pillars · K=5 | **25501** |
| **25491** | `eval_k_sweep` | fm · s_curve · K=20 | **25500** |
| **25492** | `eval_k_sweep` | mf · s_curve · K=10 | **25502** |

Shared knobs: `SAFE_EPS_MODE=scaled`, `EVAL_TAG=u7hg`, `proj=fm_only`, `record=none`, seed 6,
10 trials/variant, `--time 24:00:00` per eval (8:00:00 for the 25486 pair).
25491/25492 additionally carry the **U9 variant subset** (8 of 17); the pillars jobs run the full 17.

## 2. Worker jobs — actual outcome

| job | arm | window (UTC) | wall | state | variants finished |
|---|---|---|---|---|---|
| 25494 | **train** af corridor | 09-07 04:22 → 07:15 | 2.9 h | ✅ complete | 3.97 M U-Net, α=0.2 |
| 25495 | eval af corridor K=1 | 09-07 07:15 → 07:53 | 0.6 h | ✅ complete | 10/10 |
| 25497 | eval af pillars K=1 | 09-07 07:53 → 09:10 | 1.3 h | ✅ complete | 10/10 |
| 25498 | eval af corridor K=2 | 09-07 09:10 → 09:51 | 0.7 h | ✅ complete | 10/10 |
| 25499 | eval af pillars K=2 | 09-07 09:51 → 10:59 | 1.1 h | ✅ complete | 10/10 |
| 25496 | eval **fm** pillars K=5 | 09-07 06:18 → 18:26 | 12.1 h | ✅ complete | **17/17** |
| **25500** | eval **fm** s_curve K=20 | 09-07 10:59 → **09-08 10:30** | 23.5 h | ❌ **CANCELLED** | **6/8** |
| **25501** | eval **af** pillars K=5 | 09-07 18:26 → **09-08 10:50** | 16.4 h | ❌ **CANCELLED** | **5/17** |
| **25502** | eval **mf** s_curve K=10 | 09-08 10:30 → *running* | 2.4 h⁺ | 🟡 **RUNNING** | 1/8 + dpcc-r ~90 % |
| **25503** | eval **mf** pillars K=5 | 09-08 10:50 → *running* | 2.1 h⁺ | 🟡 **RUNNING** | 1/17 + dpcc-r ~60 % |

### 2.1 The two cancellations were manual, not timeouts

Slurm printed `*** JOB … CANCELLED AT … ***` with **no** `DUE TO TIME LIMIT` suffix, and neither
job was near its 24 h wall in a way that explains it (25501 died at 16.4 h of 24 h). The timing is
the tell:

```
25500 cancelled 10:30:28  →  25502 started 10:30:30   (+2 s)
25501 cancelled 10:50:32  →  25503 started 10:50:34   (+2 s)
```

Each cancellation freed the QOS slot that the next queued job took two seconds later. This looks
deliberate — trading the two long-running arms for the two that had not started at all.

**Partial results are not lost.** Each variant writes its own `results.json` on completion, so
25500 keeps 6 usable variants and 25501 keeps 5. What is missing:

- **25500** (fm s_curve K=20): missing `hardflow_new-c` (died at trial 4/10) and `hardflow_new-t`.
  → the fm s_curve HardFlow arm is **2 variants short of complete**.
- **25501** (af pillars K=5): got only `diffuser`, `dpcc-r`, `dpcc-r-tightened`, `dpcc-c`,
  `dpcc-c-tightened`; died in `dpcc-t` at trial 5/10. → **no HardFlow variant ran at all**, so the
  af arm of the HF-SLSQP test (mission 3) produced **nothing**.

### 2.2 Why the af arm is so much slower

Per-variant cost for the *same* scene/K (pillars, K=5, `dpcc-r`):

| engine | job | `dpcc-r` wall | ratio vs fm |
|---|---|---|---|
| fm | 25496 | 7 143 s | 1.00 |
| **mf** | 25503 | ~10 020 s (extrapolated from trial 6/10) | **1.40** |
| **af** | 25501 | 14 372 s | **2.01** |

The full 17-variant pillars sweep costs 12.1 h for fm. At 2.0× that is ~24 h, i.e. **the af pillars
K=5 arm never fits its own 24 h wall** — cancelling 25501 only pre-empted a timeout. Any resubmit
of that arm needs the U9 variant subset or a longer `UAV_EVAL_HOURS`.

### 2.3 ETA for the two live jobs

| job | hard wall | estimated finish | margin | risk |
|---|---|---|---|---|
| **25502** mf s_curve K=10 | **09-09 10:30:30** | ~09-09 06:00–07:00 | ~4 h | ⚠️ medium — `dpcc-t` alone cost 7.0 h in 25500 |
| **25503** mf pillars K=5 | **09-09 10:50:34** | ~09-09 04:00 | ~7 h | ⚠️ medium — 17 variants; the `-tightened` pair is the slow half |

Estimates scale 25500 / 25496 by the measured `dpcc-r` ratio in §2.2. Both should land, neither
comfortably.

## 3. 🔴 25514 — the one job that matters and has not run

```
25514  llim  0:00  gpu-1-student  (QOSMaxCpuPerUserLimit)  N/A  PENDING  eval_k_sweep
```

- `TIME 0:00` → it has **never been allocated**, ~1.5 days after submission on Sep 7.
- `TRES_PER_NODE = N/A` → it asks for **no GPU**. It is not competing with `schan`/`saju` for the
  cards; the queue is not the problem.
- `QOSMaxCpuPerUserLimit` → it is blocked by **llim's own CPU quota**, consumed by 25502 + 25503.

So the earliest it starts is when one of those two ends — **~Sep 9 morning UTC** — and even then
25514 is only a *wrapper*: it submits child eval jobs, which must then queue for a GPU behind
`schan` (4 GPUs, 17 h in) and `saju`. Realistically the mjpc-vs-pid answer is **2+ days out** on
the current queue.

**The choice is the user's:** cancel one running mf arm to unblock 25514 now, or let both finish
and accept the delay. 25503 (mf pillars K=5) is the weaker sacrifice — pillars is the scene the
Sep-7 DA already flagged as **void for ranking** under pre-U7 geometry, whereas 25502 is s_curve,
the scene mission 5 is actually about.

**When 25514's log finally appears, check these first** (U10 §5 lists them as never-exercised):

1. `[ env-select ] … -> conda env 'FMPCC_mjx'` — does that env exist on the cluster at all?
2. `[ U10 ] controller: 'pid_stopgo' -> 'mjpc' (env override)` — did `UAV_MIX_CONTROLLER=mjpc`
   survive into the job's captured environment?
3. a results path containing `_mpc4_mjpc_T0.5`
4. `MJPCTracker` has **never** run on a Gen15 UAV scene — an early `import mujoco.mjx` failure is
   the expected first-run mode.

## 4. Corrected mission table (supersedes `.claude/CRISIS_RECOVERY…` §5)

| # | mission | indices | true state |
|---|---|---|---|
| 1 | af_unet K-sweep on **pillars** | 25486→25497/25499; 25490→25501 | **K=1, K=2 ✅ complete**; **K=5 ❌ 5/17, cancelled** |
| 2 | af_unet pipeline on **corridor** | 25487→25494/25495/25498 | ✅ **fully complete** (train + K1 + K2) |
| 3 | **HF-SLSQP** test, pillars K=5 | fm 25488→25496 · mf 25489→**25503** · af 25490→25501 | fm ✅ 17/17 · mf 🟡 **running** · af ❌ **no HardFlow variant ran** |
| 4 | **S_curve explore** | fm K20 25491→25500 · mf K10 25492→**25502** | fm ❌ **6/8** (no `hardflow_new-c/-t`) · mf 🟡 **running** |
| 5 | **mjpc vs pid** on worst s_curve | **25514** | 🔴 **PENDING, never started** — blocked by 1 & 4's own jobs |

## 5. What the new drop does *not* contain

`temp/0809` ships **logs only**. Unlike `temp/0609/II`, there are no `batch_uav_*` /
`batch_va2_*` CSV folders. The Sep-7 DA batch referenced inside `da_uav_manual.log`
(`batch_uav_20260907_141115`, 13 report files, completed in 47.5 s) was produced **before**
25500–25503 wrote anything, so **no aggregated CSV in the repo covers the mf arms or the partial
25500/25501 results**. A fresh DA run is required once 25502/25503 land.

---

## 6. Addendum — the 2026-09-08 resubmits (25542, 25543)

Submitted ≈13:30 UTC to fill the two cancellation gaps of §2.1, using the U9 `UAV_MIX_VARIANTS`
subset knob as a **resume**: per-variant `results.json` folders sit under a stable eval-tag path and
`run_provenance.json` is written once then "kept if unchanged", so a subset job with identical knobs
lands *alongside* the variants already on disk instead of replacing them.

| job | fills | variants | est. wall | state |
|---|---|---|---|---|
| **25542** | 25501 (af pillars K=5) → missions 1 & 3 | 12 remaining | ~11 h (≈14 h at the worst observed af ratio) | 🔴 **malformed — see §6.1** |
| **25543** | 25500 (fm s_curve K=20) → mission 4 | `dpcc-t-geo_free`, `hardflow_new-c`, `hardflow_new-t` | ~9 h | 🟡 PENDING, clean |

**`n_trials` was deliberately left at the yaml default of 10.** The finished variants in both arms
ran at n=10; resuming the remainder at a lower count would leave a 10-trial `dpcc-r` next to a
5-trial `hardflow_new` inside the *same* arm, scene and K. The `UAV_MIX_VARIANTS` guard checks
variant **composition** only (`eval_mix_uav.py:658` refuses a HardFlow-only subset) — it does **not**
check trial counts, so that mistake would pass silently. Reduced trials are correct only for
mission 5 / 25514, which has no matched partner.

25543 needed `dpcc-t-geo_free` purely to satisfy that guard: its own remainder is HardFlow-only,
which exits 2. A `geo_free` row is the cheap key, and it was never in the U9 subset, so it is new
data rather than a redundant re-run.

### 6.1 🔴 25542 carries a newline inside `UAV_MIX_VARIANTS`

The submitting shell wrapped the quoted list without a line-continuation backslash, so bash's PS2
continuation swallowed a real newline into the value:

```
...,hardflow_new-t,hardflow_new-r-geo_fr
  ee,hardflow_new-c-geo_free,...
```

Element 10 resolves to `hardflow_new-r-geo_fr\n  ee`, not `hardflow_new-r-geo_free`.
`eval_mix_uav.py:646-654` splits on commas and validates every name against the assembled variant
list; one unknown name raises `SystemExit(2)` and **kills the whole child job**, not just that row.
It fails before any rollout, so the cost is a queue slot, not wall-clock.

Confirm via the wrapper's own echo (`eval_k_sweep.sh:113`), which will print across two lines:
`grep -A1 "U9 . variant subset" Slurm_Codes/logs/2026-09-08/*25542*.log`
Then `scancel 25542` and resubmit the identical command **on a single line**.

Until that is done, **mission 1's K=5 leg and mission 3's af leg remain open.**

### 6.2 Queue state after the resubmits (squeue ≈13:30 UTC)

```
RUNNING  25502  mf s_curve K=10   3:01:18   hard wall 09-09 10:30:30
RUNNING  25503  mf pillars K=5    2:41:13   hard wall 09-09 10:50:34
PENDING  25514  eval_k_sweep  (QOSMaxCpuPerUserLimit)  ← mission 5, waiting since 09-07
PENDING  25542  eval_k_sweep  (QOSMaxCpuPerUserLimit)  ← malformed, see §6.1
PENDING  25543  eval_k_sweep  (QOSMaxCpuPerUserLimit)
```

All three pending jobs are **wrappers** (`TRES = N/A`, no GPU) blocked on llim's own CPU quota, held
by 25502 + 25503. They release in job-ID order, so **25514 starts first** — the desired order, since
it is the longest-waiting and the only test with no substitute. Wrappers exit within seconds of
starting; their *children* then queue for GPUs behind `schan` (4 cards, 17:53 in) and `saju`.
