# DA — Gen15 UAV Mix-ML: K sweep on corridor (`fm` K∈{1,2,5,10,20}; `mf` FAILED TO LAUNCH)

**Date:** 2026-08-20
**Batch:** `temp/2008/batch_uav_20260820_092522` (DA_UAV_v1, auto-scan over `logs/UAV_MIX` + `logs/UAV_FM`)
**Job logs:** `temp/2008/2026-08-19/` (submitters 24707/24713, evals 24708–24718)
**Scope:** corridor, seed 6, 10 trials/cell, 23 projection variants, `--projection fm_only`, `--record none`
**Supersedes/extends:** [`DA_20260819_fm_vs_mf_3scenes_K10.md`](DA_20260819_fm_vs_mf_3scenes_K10.md) §7 item 0

---

## 0. TL;DR

1. **Half the experiment did not run.** All five `mf` jobs (24708–24712) died **~5 seconds after launch** with `FileNotFoundError: state_-1.pt`. The `mf` arm contributed **zero new data**. Everything labelled `mf` below is the pre-existing K=10 run from 2026-08-15.
2. **The `fm` collapse curve is now measured, and it is sharp.** `dpcc-c` S&C goes **0.00 → 0.00 → 0.90 → 1.00 → 1.00** across K = 1, 2, 5, 10, 20. The cliff sits **between K=2 and K=5**.
3. **The question the sweep was built to answer is still open.** The interesting region is K≤2 — exactly where we have no MeanFlow data. This DA cannot say whether MeanFlow holds there.
4. **New, unplanned result: `hardflow_new` is more K-robust than the DPCC projector.** At K=5 all three HardFlow selectors hit **1.00** S&C while all three DPCC selectors sit at **0.90**, and the tightened DPCC variants only reach 0.30. HardFlow buys back ~1 K-halving of budget.
5. **Nothing is real-time.** Even the cheapest full-success config (`post_processing` @ K=10, 111.9 ms) is **3.7× over** the 30.3 ms budget. The network alone costs **~8.6 ms per NFE**, so only K≤3 fits the budget *before any projection at all*.
6. **`mf`'s `fm_ms` is not comparable to `fm`'s `fm_ms`** — the 2026-08-15 `mf` data predates Gen15 Fix_1 (`0da86dc6`, 2026-08-15) and reports `proj_ms=0.0` with the projector time folded into `fm_ms`. Totals are fine; the split is not.

---

## 1. What actually ran

| Job | Arm | K | Outcome |
|---|---|---:|---|
| 24708 | `mf` | 1 | 🔴 **crash @ 4 s** — `state_-1.pt` not found |
| 24709 | `mf` | 2 | 🔴 **crash @ 9 s** — same |
| 24710 | `mf` | 5 | 🔴 **crash @ 14 s** — same |
| 24711 | `mf` | 10 | 🔴 **crash @ 18 s** — same |
| 24712 | `mf` | 20 | 🔴 **crash @ 23 s** — same |
| 24714 | `fm` | 1 | ✅ complete, 23/23 variants |
| 24715 | `fm` | 2 | ✅ complete, 23/23 variants |
| 24716 | `fm` | 5 | ✅ complete, 23/23 variants |
| 24717 | `fm` | 10 | ✅ complete, 23/23 variants |
| 24718 | `fm` | 20 | 🟡 **cancelled at 24 h TIME LIMIT** — 22/23 done, died in `hardflow_new-t` trial 2/10 |

Wall times: K=1 → 4 h 50 m, K=2 → 6 h 42 m, K=5 → 8 h 37 m, K=10 → 11 h 50 m, K=20 → hit the cap.

### 1.1 Why every `mf` job died

The traceback is identical in all five logs:

```
FileNotFoundError: [Errno 2] No such file or directory:
'logs/UAV_MIX/uav-corridor/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/6/state_-1.pt'
```

The chain, traced through the code:

- `mix_uav_test/eval_mix_uav.py:173` — `--epoch` defaults to `'latest'`, and `Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh` never overrides it.
- `mix_uav/utils/serialization.py:99` — `'latest'` routes to `get_latest_epoch(loadpath)`.
- `mix_uav/utils/serialization.py:22-31` — that function globs `state_*`, and `int(state.replace('state_','').replace('.pt',''))` **raises `ValueError` on `state_best.pt`, which is caught and scored as epoch `-1`**. With no numbered checkpoint present, `latest_epoch` stays at its `-1` initialiser.
- `mix_uav/utils/training_twotime.py:491` — loads `state_{-1}.pt`. Boom.

So: **the `mf` corridor training directory currently has no numbered `state_<N>.pt`, only `state_best.pt`.** The training itself is fine — the checkpoint the earlier K=10 run used is presumably still there under the `best` name.

Two things are wrong here, and they are separable:

- **(a) Environment.** The numbered checkpoints for `mix_uav_mf` are gone (disk cleanup is the likely cause — `training_twotime.py:544` explicitly anticipates "if the periodic `state_*.pt` files had been deleted to free disk"). Needs a `ls` on the cluster to confirm.
- **(b) Code.** `eval_mix_uav.py:193` is `ep = epoch if epoch == 'latest' else int(epoch)` — so **the UAV eval has no way to ask for `state_best.pt` at all**. `--epoch best` would die on `int('best')`. The D3IL side does exactly this via `'diffusion_epoch': 'best'`; the UAV frame never got the equivalent. `get_latest_epoch` also silently swallows the parse failure instead of reporting "found only `state_best.pt`".

Fix (b) is small and I have **not** made it — flagging for your go-ahead. It is a prerequisite for re-running the `mf` half of this sweep.

---

## 2. The `fm` collapse curve — the actual result

### 2.1 S&C across every variant × K

corridor, seed 6, 10 trials per cell.

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|---:|
| `diffuser` (no projection) | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 |
| `dpcc-c` | 0.00 | 0.00 | **0.90** | 1.00 | 1.00 |
| `dpcc-r` | 0.00 | 0.00 | **0.90** | 1.00 | 1.00 |
| `dpcc-t` | 0.00 | 0.00 | **0.90** | 1.00 | 1.00 |
| `dpcc-c-tightened` | 0.00 | 0.00 | 0.30 | 0.60 | 1.00 |
| `dpcc-r-tightened` | 0.00 | 0.00 | 0.30 | 0.70 | 1.00 |
| `dpcc-t-tightened` | 0.00 | 0.00 | 0.30 | 0.70 | 1.00 |
| `hardflow_new` | 0.00 | 0.10 | **1.00** | 1.00 | 1.00 |
| `hardflow_new-c` | 0.00 | 0.00 | **1.00** | 1.00 | 1.00 |
| `hardflow_new-t` | 0.00 | 0.00 | **1.00** | 1.00 | (1/1)¹ |
| `post_processing` | 0.00 | 0.10 | 0.80 | 1.00 | 1.00 |
| `post_processing-tightened` | 0.00 | 0.00 | 0.30 | 0.30 | 0.30 |
| `bounds_free` | 0.00 | 0.00 | 0.90 | 1.00 | 1.00 |
| `bounds_free-tightened` | 0.00 | 0.00 | 0.30 | 0.60 | 1.00 |
| `geo_free` | 0.00 | 0.00 | 0.20 | 0.40 | 1.00 |
| `geo_free-bounds_free` | 0.00 | 0.00 | 0.10 | 0.60 | 1.00 |
| `gradient` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `model_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

¹ Job cancelled at the time limit after 1 of 10 trials; that trial succeeded. **n=1, not comparable.**

### 2.2 The two failure modes are different

This is the part worth keeping. The collapse is not one thing degrading smoothly — it is two distinct regimes.

| `dpcc-c` | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|---:|
| S&C | 0.00 | 0.00 | 0.90 | 1.00 | 1.00 |
| goal reached | **0.00** | **0.70** | 1.00 | 1.00 | 1.00 |
| constraint violations | 192 | 58 | 1 | 0 | 0 |
| track err | 36.38 | 8.80 | 0.52 | 0.51 | 0.51 |
| steps to goal | — | 275 | 270 | 270 | 272 |

- **K=1 — navigation failure.** Goal reached **0.00**. Every rollout burns the full 396-step budget, racks up ~190 violations, and tracking error is 36–175 depending on variant. One Euler step from noise does not produce a trajectory; it produces a direction.
- **K=2 — precision failure.** Goal reached jumps to **0.70** (0.90 for `hardflow_new-c`) but S&C stays at **0.00**. The drone *gets there* and *clips the corridor on the way*. Violations drop 192 → 58, tracking error 36.4 → 8.8. The plan is now roughly right and locally wrong.
- **K=5 — recovered.** Violations 58 → 1, tracking error 8.8 → 0.52. This is the knee.
- **K=10, K=20 — flat.** `dpcc-c` is 1.00 at both, with identical tracking error (0.51) and identical steps-to-goal (270 vs 272). **K=10 → K=20 buys nothing on corridor and costs 1.7× the wall time.** That confirms the DA_20260819 §6.5 premise from the other direction: at K=10 the Euler discretisation error is already negligible, so there is nothing left for MeanFlow to recover.

### 2.3 HardFlow is the K-robustness story

Unplanned, and the clearest new finding in the batch:

| at K=5 | S&C | violations | track err |
|---|---:|---:|---:|
| `hardflow_new` / `-c` / `-t` | **1.00 / 1.00 / 1.00** | 0 / 0 / 0 | 0.51 / 0.50 / 0.51 |
| `dpcc-c` / `-r` / `-t` | 0.90 / 0.90 / 0.90 | 1 / — / — | 0.52 / 0.52 / 0.52 |
| `dpcc-*-tightened` | 0.30 | 73 | 17.4 |

Solving the constrained problem *inside* each ODE step degrades more gracefully than generate-then-project, which is exactly the mechanism HardFlow claims. It does **not** rescue K=2 (0.00–0.10), so it shifts the cliff by roughly one halving of K, not more.

Cost: `hardflow_new-c` at K=5 is **492.9 ms** vs `dpcc-c`'s **167.0 ms** — 3.0× more expensive for +0.10 S&C. **Non-dominated, not better:** it wins on success, loses on time.

### 2.4 Tightened variants need K=20

Every `-tightened` DPCC variant is 0.30 at K=5, 0.60–0.70 at K=10, and only reaches 1.00 at **K=20**. `post_processing-tightened` never recovers (0.30 flat across K=5/10/20 — that is a *variant* problem, not a K problem, and it also showed 127.2 tracking error at K=10; worth a separate look).

If the paper story needs the tightened threshold, the operating point is K=20, and the K-efficiency argument gets harder, not easier.

---

## 3. Real-time budget — the framing that matters

Control is 33 Hz → **30.3 ms per replan**.

| K | network only | `dpcc-c` proj | `dpcc-c` total | ×budget | `hardflow-c` total | ×budget |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.1 | 98.2 | 107.2 | 3.5× | 612.3 | 20.2× |
| 2 | 17.6 | 51.8 | 69.1 | 2.3× | 296.7 | 9.8× |
| 5 | 44.1 | 123.2 | 167.0 | 5.5× | 492.9 | 16.3× |
| 10 | 85.7 | 156.9 | 241.9 | 8.0× | 756.4 | 25.0× |
| 20 | 171.9 | 244.8 | 415.5 | 13.7× | 1376.6 | 45.4× |

(`network only` = the `diffuser` variant, which runs no projector.)

Three things fall out:

- **Per-NFE cost is ~8.6 ms and perfectly linear** (9.1 / 17.6 / 44.1 / 85.7 / 171.9 for K = 1/2/5/10/20). The 30.3 ms budget fits **K≤3 of network and nothing else**.
- **The Pareto-efficient full-success cell is `post_processing` @ K=10: 1.00 S&C at 111.9 ms.** It dominates `dpcc-c` @ K=10 (same 1.00, 241.9 ms) and `hardflow_new-c` @ K=5 (same 1.00, 492.9 ms) on time at equal success. Still 3.7× over budget.
- **Projection cost is non-monotonic in K.** `dpcc-c` proj is 98.2 ms at K=1 but only 51.8 ms at K=2 — at K=1 the plans are so bad the SLSQP solver has to work much harder to (still unsuccessfully) fix them. Bad generation makes projection *more* expensive, not less. That kills the naive "drop K and let the projector absorb it" hope.

---

## 4. `fm` vs `mf` at K=10 — unchanged from 2026-08-19, restated with caveats

No new `mf` data. This is the same 2026-08-15 snapshot re-aggregated.

| variant | fm S&C | mf S&C | fm total ms | mf total ms | fm steps | mf steps |
|---|---:|---:|---:|---:|---:|---:|
| `diffuser` | 0.00 | 0.00 | 85.7 | 88.5 | 314 | — |
| `dpcc-c` | **1.00** | 0.70 | 241.9 | 269.7 | 270 | 259 |
| `dpcc-r` | **1.00** | 0.80 | 241.8 | 271.0 | 266 | 256 |
| `dpcc-t` | **1.00** | 0.70 | 240.8 | 273.0 | 268 | 259 |
| `dpcc-c-tightened` | **0.60** | 0.10 | 279.0 | 430.3 | 269 | 251 |
| `post_processing` | **1.00** | 0.00 | 111.9 | 196.1 | 271 | 262 |
| `bounds_free` | **1.00** | 0.70 | 217.3 | 242.0 | 268 | 258 |
| `geo_free` | 0.40 | **0.50** | 164.1 | 184.8 | 267 | 258 |

Two readings that must **not** be taken from this table:

- ❌ *"MeanFlow's network is 3× slower."* The `diffuser` row settles it: **88.5 ms (mf) vs 85.7 ms (fm)** at K=10 for pure network time with no projector. Same UNet, same 4.0 M params, same cost. The inflated `mf` numbers elsewhere are the pre-Fix_1 attribution bug (§5.1) — `proj_ms` reads 0.0 for **every** `mf` variant including `dpcc-*`, even though `n_proj_steps` is 259, so `fm_ms` is carrying the projector time. `mix_uav/models/mf_diffusion.py:319-324` documents this exact failure by name.
- ❌ *"MeanFlow needs fewer steps"* from the steps-to-goal column. `mf` reaches the goal in ~256–262 steps vs `fm`'s ~266–271 — but `mf` only succeeds 70–80% of the time, so that mean is taken over the easier rollouts. Not a like-for-like comparison.

What *is* supportable: **at K=10, on corridor, `fm` dominates `mf`** — higher S&C on 6 of 8 shared variants, equal on one, lower on one (`geo_free`, 0.40 vs 0.50), at equal or lower total wall time. And DA_20260819 §6.5 already explains why K=10 is the wrong place to look.

---

## 5. Data integrity

### 5.1 The `mf` timing split is broken (pre-Fix_1 data)
`proj_ms = 0.0` for all 20 `mf` variants; `fm_ms == avg_time_ms` exactly. Cause is `eval_mix_uav.py:1064` — `step_proj_ms = getattr(policy, 'last_proj_ms', 0.0)`, and Gen3v6's sampler never set it. Fixed in **`0da86dc6` (Gen15 Fix1, 2026-08-15)**; the `mf` corridor snapshot is `20260815_050702`, i.e. **before** that commit. Totals are trustworthy, the split is not. Re-running `mf` also re-runs it on a Fix_1 build, which closes this.

### 5.2 `mf` has no HardFlow variants
`mf` ran 20 variants, `fm` ran 23 — `hardflow_new`, `-c`, `-t` are missing from the `mf` arm entirely. Any HardFlow claim is currently `fm`-only.

### 5.3 K=20 `hardflow_new-t` is n=1
Job 24718 hit the 24 h cap mid-variant. That cell is a single trial and must be excluded from any comparison. All other 114 `fm` cells are n=10, `timing_missing=0`, `cb_tripped=0`.

### 5.4 Gen11 archaeology still pollutes the auto-scan
28 of 38 candidates have unparsable eval tags (blank scene/engine/K), spread over 13 stale `plans(...)` snapshot directories under `logs/UAV_FM`. They inflate `units_loaded` to 462 and make the ranking CSVs meaningless. Unchanged from DA_20260819 §7 items 4–5; the per-K tables above were built by filtering on `engine ∈ {fm, mf}` and are unaffected.

### 5.5 Provenance is clean on the `fm` side
`run_config.csv` confirms `path_K == flow_steps == K` for all five `fm` candidates (1/2/5/10/20) — the Gen11 K-labelling bug is not present here. `epoch=latest`, `n_trials=10`, `projection=fm_only` across the board.

---

## 6. Verdict

1. **The sweep half-succeeded.** `fm`'s K-response on corridor is now fully characterised; `mf`'s is not measured at all.
2. **The cliff is between K=2 and K=5**, and it is a cliff, not a slope: 0.00 → 0.90.
3. **K=10 was, as suspected, a dead operating point for this comparison.** `fm` is saturated there (identical to K=20 on every metric), so a MeanFlow win was never geometrically possible at that K. Everything interesting is at K≤2.
4. **HardFlow's in-loop projection is measurably more K-robust than generate-then-project** — 1.00 vs 0.90 at K=5, at 3× the cost. Non-dominated. This is a real Gen15 finding and it was not what the sweep was aimed at.
5. **Nothing here is close to real-time.** 3.7× over budget at the cheapest full-success cell; the network alone eats the budget by K=4.
6. **`fm` still dominates `mf` at K=10**, and that statement remains uninformative about MeanFlow's actual claim.

---

## 7. What to do next

**0. Fix the checkpoint loading, then re-run the `mf` half.** Nothing else in this list matters as much. Two sub-steps:
   - Check on the cluster what is actually in `logs/UAV_MIX/uav-corridor/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/6/` — if numbered checkpoints are gone and only `state_best.pt` survives, that is the whole story.
   - Teach `eval_mix_uav.py` to accept `--epoch best` (one line at `:193`, plus passing it through `eval_mix_uav.sh`), and make `get_latest_epoch` say *why* it returned -1 instead of failing three frames later on a nonsense filename. **Not done — awaiting go-ahead.**

**1. Re-run the `mf` sweep at K∈{1,2,5,10,20}** on the same corridor/seed-6/10-trial grid. This is still the experiment that decides Gen15's direction, and it is still cheap (eval-only, no retraining).

**2. Consider narrowing the K list to {1,2,3,5}.** K=10 and K=20 are now known-saturated for `fm` and cost 12 h and 24 h+ respectively. The decision lives entirely at the low end, and dropping the top two K values would have let the whole sweep finish inside one day.

**3. Re-run K=20 `hardflow_new-t`** if that cell is needed — or accept 22/23 for K=20, since `hardflow_new` and `-c` are both already 1.00 there.

**4. `af` and `diffusion` arms have still never run** in Gen15. The `diffusion` arm in particular is THE baseline; a K sweep of `fm` with no diffusion-DPCC reference cannot support a headline claim.

**5. Investigate `post_processing-tightened`** — flat 0.30 across K=5/10/20 with 127.2 tracking error at K=10 while plain `post_processing` is 1.00. That looks like a variant bug, not a budget effect.

**6. Exclude the 25 stale `plans(...)` dirs from auto-scan** (DA_UAV_v1 work, belongs in `logs_in_develop/DA_Code/DA_UAV_v1/`). Carried over unaddressed from DA_20260819 §7.
