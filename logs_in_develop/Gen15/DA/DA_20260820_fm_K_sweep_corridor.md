# DA — Gen15 UAV Mix-ML: K sweep on corridor, `fm` vs `mf` (K ∈ {1,2,5,10,20})

**Date:** 2026-08-20, **updated 2026-08-21 with the `mf` arm**
**Batches:** `temp/2008/batch_uav_20260820_092522` (`fm`) + `temp/2108/batch_uav_20260821_105229` (`fm` + `mf`, re-aggregated)
**Job logs:** `temp/2008/2026-08-19/` (fm: 24714–24718) · `temp/2108/2026-08-20/` (mf: 24745–24749)
**Scope:** corridor, seed 6, 10 trials/cell, 23 projection variants, `--projection fm_only`, `--record none`
**Supersedes:** [`DA_20260819_fm_vs_mf_3scenes_K10.md`](DA_20260819_fm_vs_mf_3scenes_K10.md) — its `mf` K=10 corridor row is now overwritten by a `best`-checkpoint re-run and must not be cited.

> **Revision note (2026-08-21).** The original version of this DA reported only the `fm` half; all five `mf` jobs had crashed on `state_-1.pt`. That bug is fixed (see [`../../checkpoint_epoch_best/CHANGELOG_20260820_epoch_latest_to_best.md`](../../checkpoint_epoch_best/CHANGELOG_20260820_epoch_latest_to_best.md)) and the `mf` arm has now run. **The headline conclusion has changed.**

---

## 0. TL;DR

1. **MeanFlow wins at K=1 and K=2. Flow Matching wins at K≥5. The crossover sits between K=2 and K=5.** At K=1 `fm` scores **0.00 S&C on all 23 variants**; `mf` scores non-zero on 8 of them, topping out at **0.80** (`dpcc-c`). At K=10/20 that reverses: `fm` reaches 1.00 on **8** variants at K=10 and **14** at K=20; `mf` reaches 1.00 on two variants at K=10 and none at K=20.
2. **This is the predicted MeanFlow result, and it is the first time Gen15 has actually tested it.** Every previous comparison sat at K=10, where `fm` is saturated and there was nothing for MeanFlow to win.
3. **The two arms fail in mechanically different ways.** At low K `fm` degrades *uniformly* — at K=1, all ten `dpcc-c` rollouts log 158–231 violations, not one is clean. `mf` fails *bimodally* — eight rollouts log **exactly zero** violations and two diverge (226, 342). MeanFlow at K=1 produces either a correct plan or a broken one; it never produces a mediocre one. See §2.3.
4. **New cheapest viable cell in the whole sweep: `mf` @ K=1 `dpcc-c` — 0.80 S&C at 58.9 ms, 1.9× over the real-time budget.** The previous best was `fm` @ K=10 `post_processing` at 111.9 ms / 3.7×. Still not real-time, but it halves the gap.
5. 🔴 **The comparison is confounded on checkpoints, in `mf`'s favour.** `mf` ran on `epoch=best` (**step 99000**); the entire `fm` curve is the pre-fix data on `epoch=latest` (**step 80000**). See §5.1 — this gates claim 1 and must be closed before the result is used.
6. **`mf` K=20 is 20/23 variants** (TIME LIMIT, all three HardFlow variants missing), and a config commit landed mid-sweep that renamed and re-parameterised the HardFlow arm. See §5.3.
7. **Nothing is real-time on either arm.** Network cost is ~8.6 ms/NFE (identical on both), so the 30.3 ms budget fits K≤3 of network before any projection at all.

---

## 0.5 The core question: does MeanFlow beat FM on UAV the way it does on `avoiding`?

**Partly. Same shape, weaker claim.**

| | `avoiding-d3il` (Gen3v6) | **UAV corridor (this DA)** |
|---|---|---|
| MF wins at matched low K (1–2)? | yes | **yes** — wins 8 variants at K=1, 9 at K=2, **loses 0** |
| MF at low K vs FM at its *best* K? | **exceeds it** (L1: MF/AF K=2 ≥ FM K=20 on success, 12× less time) | **does not reach it** — mf 0.80 @ K=1 vs fm **1.00** @ K=10 |
| Verdict | **Pareto win** | **trade-off / non-dominated** |

- **At matched K, the UAV result reproduces `avoiding`.** `fm` is 0.00 on all 23 variants at K=1; `mf` reaches 0.80. The low-K advantage is a real capability on both tasks, not a discount.
- **What does not carry over is the ceiling.** On `avoiding`, MeanFlow at K=2 *beat* naive FM at K=20 outright, so it was Pareto-dominant. On UAV it does not: `mf` tops out at 0.80–0.90 while `fm` reaches 1.00 from K=10 up. `mf` @ K=1 is **1.9× cheaper** than the cheapest `fm` cell that reaches 1.00 (58.9 vs 111.9 ms) but **0.20 less reliable**. Neither dominates — say "trade-off", not "better".
- **Why the difference is plausible:** the `avoiding` gap was won on *time* against a saturated baseline. UAV corridor has a harder feasibility structure, and `mf`'s residual 20% is not spread thinly — it is two fully divergent rollouts out of ten (§2.3). The ceiling is a tail problem, not a quality problem.

🔴 Both rows of this table are subject to §5.1 — `mf` ran on step 99000, `fm` on step 80000.

---

## 1. What ran

| Job | Arm | K | Epoch | Outcome |
|---|---|---:|---|---|
| 24714–24717 | `fm` | 1, 2, 5, 10 | `latest` (80000) | ✅ 23/23 variants each |
| 24718 | `fm` | 20 | `latest` (80000) | 🟡 22/23 — 8 h TIME LIMIT in `hardflow_new-t` trial 2/10 |
| 24745–24748 | `mf` | 1, 2, 5, 10 | **`best` (99000)** | ✅ 23/23 variants each |
| 24749 | `mf` | 20 | **`best` (99000)** | 🟡 20/23 — 8 h TIME LIMIT before any HardFlow variant |

Wall times (job start → job end; the sweep submits with `--time=$((N_SEEDS*8))h`, i.e. **an 8 h cap** here):

| K | `fm` | `mf` |
|---:|---|---|
| 1 | 4 h 50 m | 5 h 56 m |
| 2 | 3 h 00 m | 6 h 23 m |
| 5 | 3 h 47 m | 5 h 23 m |
| 10 | 5 h 08 m | 7 h 13 m |
| 20 | **8 h 00 m (cap)** | **8 h 00 m (cap)** |

Both arms hit the cap at K=20 and neither finished. (The wall times printed in the pre-update version of this DA were measured from *submission* and included queue wait; these are run times.)

Two things the `mf` logs confirm that the previous run could not:

- **`Restored loss history from checkpoint at step 99000`** — `best` is 19000 steps further trained than `latest`. This is the direct measurement of what the epoch bug cost, and it is not small.
- **`proj_ms` is non-zero on every `mf` variant** (e.g. K=1 `dpcc-c`: `fm_ms=9.5 proj_ms=49.4`). Gen15 Fix_1 (`0da86dc6`) is working; the old §5.1 attribution caveat is **closed**.

---

## 2. The two K-response curves

### 2.1 S&C across every variant × K

corridor, seed 6, 10 trials/cell, mask `proj_valid`. **Bold** = the arm winning that cell.

| variant | fm K1 | mf K1 | fm K2 | mf K2 | fm K5 | mf K5 | fm K10 | mf K10 | fm K20 | mf K20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dpcc-c` | 0.00 | **0.80** | 0.00 | **0.80** | **0.90** | 0.60 | **1.00** | 0.60 | **1.00** | 0.90 |
| `dpcc-r` | 0.00 | 0.00 | 0.00 | **0.10** | **0.90** | 0.50 | **1.00** | 0.80 | **1.00** | 0.70 |
| `dpcc-t` | 0.00 | **0.30** | 0.00 | **0.20** | **0.90** | 0.50 | **1.00** | 0.80 | **1.00** | 0.80 |
| `dpcc-c-tightened` | 0.00 | **0.10** | 0.00 | **0.50** | **0.30** | 0.00 | **0.60** | 0.30 | **1.00** | 0.40 |
| `dpcc-r-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | **0.30** | 0.00 | **0.70** | 0.20 | **1.00** | 0.10 |
| `dpcc-t-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | **0.30** | 0.00 | **0.70** | 0.20 | **1.00** | 0.50 |
| `hardflow_new` | 0.00 | **0.10** | 0.10 | 0.10 | **1.00** | 0.50 | 1.00 | 1.00 | 1.00 | — |
| `hardflow_new-c` | 0.00 | **0.50** | 0.00 | **0.50** | **1.00** | 0.80 | 1.00 | 1.00 | 1.00 | — |
| `hardflow_new-t` | 0.00 | **0.40** | 0.00 | **0.20** | **1.00** | 0.80 | **1.00** | 0.90 | (n=1)¹ | — |
| `post_processing` | 0.00 | **0.10** | 0.10 | 0.10 | **0.80** | 0.30 | **1.00** | 0.00 | **1.00** | 0.30 |
| `post_processing-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | **0.30** | 0.10 | **0.30** | 0.00 | **0.30** | 0.00 |
| `bounds_free` | 0.00 | 0.00 | 0.00 | **0.20** | **0.90** | 0.70 | **1.00** | 0.50 | **1.00** | 0.60 |
| `bounds_free-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | **0.30** | 0.10 | **0.60** | 0.10 | **1.00** | 0.20 |
| `geo_free` | 0.00 | **0.10** | 0.00 | **0.20** | 0.20 | **0.30** | 0.40 | **0.50** | **1.00** | 0.60 |
| `geo_free-bounds_free` | 0.00 | 0.00 | 0.00 | **0.10** | 0.10 | **0.30** | **0.60** | 0.50 | **1.00** | 0.80 |
| `diffuser` (no projection) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.40** | 0.00 |
| `gradient` / `model_free` (+tightened) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

¹ `fm` K=20 `hardflow_new-t` is a single trial (job cancelled at the 8 h cap). n=1, not comparable.

**Tally over all 23 variants** (K=20 over the 20 both arms completed):

| | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|---:|
| `mf` wins | **8** | **9** | 2 | 1 | 0 |
| ties | 15 | 14 | 8 | 10 | 7 |
| `mf` loses | **0** | **0** | 13 | 12 | 13 |

The sign flips between K=2 and K=5. Note `mf` does not lose a single variant at K=1 or K=2.

### 2.2 The shapes are different, not just the levels

| `dpcc-c` | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|---:|
| **fm** S&C | 0.00 | 0.00 | 0.90 | 1.00 | 1.00 |
| **fm** goal reached | 0.00 | 0.70 | 1.00 | 1.00 | 1.00 |
| **fm** track err | 36.38 | 8.80 | 0.52 | 0.51 | 0.51 |
| **mf** S&C | 0.80 | 0.80 | 0.60 | 0.60 | 0.90 |
| **mf** goal reached | 0.80 | 0.80 | 0.90 | 0.90 | 1.00 |
| **mf** track err | 13.50 | 19.20 | 7.55 | 9.18 | 0.53 |

- **`fm` is a sigmoid in K.** Zero below the knee, saturated above it. K=1 is *navigation* failure (goal 0.00, full 396-step budget burned); K=2 is *precision* failure (goal 0.70, still 0.00 S&C — the drone gets there and clips the corridor on the way); K=5 is the knee; K=10 and K=20 are identical on every metric, so **K=10 → K=20 buys nothing on corridor and costs 1.7× the wall time**.
- **`mf` is roughly flat in K, with a ceiling.** 0.80 / 0.80 / 0.60 / 0.60 / 0.90 is noise around ~0.75 at n=10, not a trend. That flatness *is* the MeanFlow claim — one NFE is as good as twenty — and here it holds. What it does not do is reach 1.00 at any K.

So the two curves cross rather than one dominating: MeanFlow trades a lower ceiling for K-independence.

### 2.3 The failure modes are mechanically different — the strongest finding here

Per-rollout `n_violations` for `dpcc-c`, all ten rollouts:

| arm, K | S&C | per-rollout violations |
|---|---:|---|
| `fm` K=1 | 0/10 | 168, 177, 204, 158, 231, 184, 165, 221, 193, 224 |
| `fm` K=2 | 0/10 | 6, 24, 194, 19, 9, 15, 141, 2, 3, 167 |
| `mf` K=1 | **8/10** | **0, 0, 0, 0, 226, 0, 342, 0, 0, 0** |
| `mf` K=5 | 6/10 | 0, 0, 29, 0, 325, 0, 14, 45, 0, 0 |
| `mf` K=10 | 6/10 | 0, 47, 0, 0, 11, 0, 18, 330, 0, 0 |

**`fm` at low K has no clean rollouts at all** — every single one is degraded, and at K=2 even the best rollout still logs 2 violations. **`mf` at K=1 has eight perfectly clean rollouts and two divergences.** The mean violation count for `mf` (56.8 at K=1) is entirely an outlier artifact and should never be quoted as a typical rollout.

This matters practically: `mf`'s failure is *detectable and recoverable* — a per-replan feasibility check could fall back on the 20% of bad plans and would recover most of the gap to 1.00. `fm`'s low-K degradation is spread across every rollout and has nothing to fall back to. That asymmetry is a Gen15 result in its own right and it is not visible in aggregate S&C.

### 2.4 HardFlow, revisited

The K-robustness finding from the `fm`-only version survives and extends. On `fm` at K=5, all three HardFlow selectors hit 1.00 vs DPCC's 0.90. On `mf`, `hardflow_new-c` is the best low-K cell after `dpcc-c` (0.50 at both K=1 and K=2) and the only `mf` variant to reach 1.00 at any K (K=10).

Cost remains the objection: `hardflow_new-c` on `mf` at K=1 is **1015.7 ms** — worse than `fm` at K=20 with the DPCC projector. In-loop projection at K=1 pays a fixed NLP cost that the single NFE does nothing to amortise. **Non-dominated, not better.**

### 2.5 Tightened variants still need K=20

Every `-tightened` DPCC variant on `fm` is 0.30 at K=5, 0.60–0.70 at K=10, 1.00 only at K=20. On `mf` they never exceed 0.50. `post_processing-tightened` is flat 0.30 (fm) / 0.00–0.10 (mf) across all K — that is a *variant* defect, not a budget effect, and still needs a separate look.

---

## 3. Real-time budget

Control is 33 Hz → **30.3 ms per replan**.

| K | net (fm) | net (mf) | `dpcc-c` total (fm) | ×budget | `dpcc-c` total (mf) | ×budget |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.1 | 9.5 | 107.2 | 3.5× | **58.9** | **1.9×** |
| 2 | 17.6 | 18.2 | 69.1 | 2.3× | 64.8 | 2.1× |
| 5 | 44.1 | 44.6 | 167.0 | 5.5× | 224.7 | 7.4× |
| 10 | 85.7 | 89.4 | 241.9 | 8.0× | 339.4 | 11.2× |
| 20 | 171.9 | 181.9 | 415.5 | 13.7× | 452.8 | 14.9× |

(`net` = the `diffuser` variant, which runs no projector.)

- **Network cost is identical on the two arms** — 9.1 vs 9.5 ms at K=1, 85.7 vs 89.4 at K=10. `mf` is 3–6% slower, consistent across all K. The two-time objective costs essentially nothing at inference. *(But see §5.2 — this does not settle the parameter question.)*
- **Per-NFE cost is ~8.6 ms and perfectly linear** on both arms. The budget fits K≤3 of network and nothing else.
- **`mf` @ K=1 `dpcc-c` is the cheapest ≥0.8 S&C cell in the sweep at 58.9 ms**, and it is cheap for a compounding reason: better plans make SLSQP converge faster. `proj_ms` is 49.4 for `mf` vs 98.2 for `fm` at K=1 — the projector spends half as long because it is being handed a nearly-feasible trajectory.
- **Projection cost is non-monotonic in K on `fm`** (98.2 ms at K=1 vs 51.8 at K=2): bad generation makes projection *more* expensive, not less. That kills the naive "drop K and let the projector absorb it" hope for `fm` — and is exactly the mechanism `mf` exploits.

Ranked by cost among cells with S&C ≥ 0.80:

| rank | arm | K | variant | S&C | total ms | ×budget |
|---:|---|---:|---|---:|---:|---:|
| 1 | **mf** | 1 | `dpcc-c` | 0.80 | **58.9** | 1.9× |
| 2 | **mf** | 2 | `dpcc-c` | 0.80 | 64.8 | 2.1× |
| 3 | fm | 5 | `post_processing` | 0.80 | 70.7 | 2.3× |
| 4 | fm | 10 | `post_processing` | 1.00 | 111.9 | 3.7× |
| 5 | fm | 5 | `bounds_free` | 0.90 | 145.5 | 4.8× |

---

## 4. Head-to-head verdict at matched K

Using the Pareto rule (equal S&C → fewer steps *and* lower time to claim a win):

- **K=1: `mf` dominates.** Higher S&C on 8 variants, lower on none, and cheaper on the best cell (58.9 vs 107.2 ms). This is a clean win, subject to §5.1.
- **K=2: `mf` dominates.** Wins 9, loses **0**. Best cell 64.8 vs 69.1 ms.
- **K=5: `fm` dominates.** Wins 13, loses 2, and is cheaper on `dpcc-c` (167.0 vs 224.7 ms).
- **K=10: `fm` dominates** (wins 12, loses 1). **K=20: `fm` dominates** (wins 13, loses 0).

The old §4 table (the 2026-08-15 `mf` K=10 snapshot showing 0.70–0.80 on `dpcc-*`) is **superseded**: the re-run on `best` gives 0.60 / 0.80 / 0.80 for `dpcc-c` / `-r` / `-t`. Do not merge the two; cite only the new numbers.

---

## 5. Data integrity

### 5.1 🔴 Checkpoint asymmetry — the one thing that gates §0 item 1

`run_config.csv` is unambiguous:

| arm | all five corridor candidates | resolved step |
|---|---|---:|
| `fm` | `epoch=latest` | 80000 |
| `mf` | `epoch=best` | **99000** |

**`mf` is a 19000-step better-trained model than `fm` in every cell of this sweep.** The direction of the bias runs both ways against `fm`: it may inflate `mf`'s low-K win, and it certainly *understates* `fm`'s high-K win.

Two arguments that the low-K result survives anyway, neither conclusive:

- `fm` is at **exactly 0.00 on all 23 variants at K=1**, not 0.2 or 0.3. A 19% training-length increase lifting a uniformly-broken arm to 0.80 would be a remarkable amount of work for the last 19000 steps.
- `fm`'s K=1 failure is *structural* (goal reached 0.00, full step budget burned on every rollout — §2.3), which is what one Euler step from noise predicts geometrically, not what an undertrained checkpoint predicts.

**Neither substitutes for the measurement.** Re-run `fm` on `best` at K ∈ {1,2,5} before this DA's headline is used anywhere. That is ~3 jobs and ~20 GPU-h.

### 5.2 🔴 Parameter asymmetry — still open, and the eval does not log it

Unchanged from the changelog §2.3. The `mf` eval prints `[ MFTrajectoryModel ] backbone=unet unet_width(freq_dim)=32 params=4.0M`. **The `fm` eval prints no parameter count at all** — checked across all five `fm` job logs — so the only evidence remains the checkpoint sizes (`mf` 63,954,430 B vs `fm` 31,825,312 B ⇒ ~4.0 M vs ~2.0 M at 16 B/param).

`config/uav_mix.py:244` claims dim=32 "is what makes the three arms parameter-identical (gate G3 asserts it)", and `mix_uav_test/gates_mix_uav.py:342` fails above 25% divergence — a 2× gap should have tripped it. Either G3 has not been run against this config or the arms genuinely differ.

Note that §3's network-time parity (9.1 vs 9.5 ms) is **not** evidence of parameter parity — a 2× parameter difference in a U-Net can come from channel widths that barely move wall time at this size. Settle it directly:

```bash
python -c "
import torch
for a,p in [('fm','logs/UAV_MIX/uav-corridor/mix_uav_fm/H8_Dmodels.diffusion.FlowMatchingODE_9D/6'),
            ('mf','logs/UAV_MIX/uav-corridor/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/6')]:
    d=torch.load(f'{p}/state_best.pt',map_location='cpu')
    print(a,'step',d['step'],'params',sum(v.numel() for v in d['model'].values()))"
```

### 5.3 🟡 A config commit landed mid-sweep and changed the HardFlow arm

Jobs 24745–24748 logged `HardFlow arm: +3 variants ['hardflow_new', 'hardflow_new-c', 'hardflow_new-t']`. Job 24749 (started 22:08, ~11 h later) logged `['hardflow_new-r', ...]`. `config/uav_mix.py:212` now reads `['hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t']`, changed by **`0f1aa7fc` "introduce resolve_hf_batch_size to enforce correct arm-C candidate fan parity with DPCC arms"**.

Two consequences:
- That commit changed HardFlow's **candidate fan size**, not just a name. HardFlow numbers produced before and after it are not comparable, on either arm.
- All HardFlow rows in this DA predate the change and are therefore internally consistent — but any *future* HardFlow run will be measuring something different. Re-baseline before extending.

### 5.4 `mf` K=20 is 20/23 variants
Job 24749 hit the 8 h cap having completed the 20 non-HardFlow variants. `hardflow_new*` was never reached on `mf` at K=20. Combined with 5.3, HardFlow at K=20 on `mf` should be re-run from scratch rather than patched in.

### 5.5 Aggregate `mf` violation counts are outlier-dominated
As §2.3 shows, `mf`'s per-variant mean `n_violations` is driven by 1–2 divergent rollouts out of 10. Quote S&C and the per-rollout distribution; do **not** quote mean violations for `mf` without saying so. (`mf` K=2 `dpcc-c`: mean 71.2, median 0.)

### 5.6 Gen11 archaeology still pollutes the auto-scan
28 of 42 candidates have unparsable eval tags across 13 stale `plans(...)` snapshot directories under `logs/UAV_FM`. Unchanged from DA_20260819 §7 items 4–5. All tables above were built by filtering on `engine ∈ {fm, mf}` and `scene == corridor`, so they are unaffected — but the batch's own `candidates_ranking.csv` remains unusable.

### 5.7 Provenance is otherwise clean
`path_K == flow_steps == K` for all ten corridor candidates, `n_trials=10`, `projection=fm_only`, `seed=6`, `timing_missing=0`, `cb_tripped=0`. The Gen11 K-labelling bug is not present.

---

## 6. Verdict

1. **MeanFlow's central claim holds on UAV corridor at K≤2, and this is the first Gen15 experiment that could have shown it.** `fm` is at 0.00 across the board at K=1; `mf` reaches 0.80.
2. **MeanFlow's ceiling is lower.** It reaches 1.00 on only two variants (`hardflow_new`, `hardflow_new-c`, both at K=10), while `fm` reaches 1.00 on 8 variants at K=10 and 14 at K=20. K-independence is bought at the cost of peak reliability.
3. **The curves cross between K=2 and K=5.** Which arm is "better" is entirely a function of the NFE budget, so the question has no budget-free answer — and any headline claim must name its K.
4. **`mf`'s failure mode is bimodal and therefore addressable**; `fm`'s low-K failure is uniform and is not. §2.3 is the most actionable result in this DA.
5. **The best real-time story in Gen15 so far is `mf` @ K=1 `dpcc-c`: 0.80 S&C at 1.9× budget.** Still not real-time, but the first cell within a factor of 2.
6. 🔴 **None of items 1–5 is safe to publish until §5.1 is closed.** `mf` ran on a 99000-step checkpoint and `fm` on an 80000-step one.
7. **The diffusion-DPCC baseline has still never been run on UAV.** Per the benchmark hierarchy, everything above is measured against naive FM, not against the actual baseline.

---

## 7. What to do next

**1. Re-run `fm` on `epoch=best` at K ∈ {1, 2, 5}.** Closes §5.1, which gates the entire headline. Highest priority by a wide margin; ~20 GPU-h.
```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh fm corridor "6" "1 2 5" 10 fm_only none
```

**2. Settle the parameter counts** (§5.2, command inline). Ten seconds, no GPU, and it can invalidate the comparison independently of everything else. Then run gate G3 against the live config and find out why it did not catch this.

**3. Run the `af` arm at K ∈ {1, 2, 5}.** α-Flow anneals from flow matching to MeanFlow, so if the crossover in §2.1 is real, `af` is the arm designed to sit on top of it. It has never run on UAV, on any backbone, and it reuses an existing checkpoint.

**4. Run the `diffusion` arm** (§6 item 7). It is THE baseline; a K sweep with no diffusion-DPCC reference cannot support a headline claim. Note this arm needs a *separate training run per K* — the beta schedule is built from K at training time.

**5. Chase the bimodality (§2.3).** If eight of ten `mf` K=1 rollouts are already perfect, a per-replan feasibility gate with a K=5 fallback would plausibly reach ~1.00 S&C at an amortised cost near 58.9 ms. That is the shortest path to a real-time claim that currently exists in Gen15.

**6. Narrow future sweeps to K ∈ {1, 2, 3, 5}.** K=10 and K=20 are now measured as saturated for `fm` and flat for `mf`, and K=20 does not even fit the 8 h job cap on either arm. The decision lives entirely at the low end.

**7. Re-baseline HardFlow after `0f1aa7fc`** (§5.3), and drop the n=1 `fm` K=20 `hardflow_new-t` cell.

**8. Investigate `post_processing-tightened`** — flat 0.30 (fm) / ~0.00 (mf) across all K. Looks like a variant defect.

**9. Exclude the stale `plans(...)` dirs from auto-scan** (DA_UAV_v1 work). Carried over unaddressed from DA_20260819 §7.
