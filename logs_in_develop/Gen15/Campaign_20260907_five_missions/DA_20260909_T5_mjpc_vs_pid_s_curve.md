# DA — Mission 5: `mjpc` vs `pid_stopgo` on `s_curve` (jobs 25514 → 25554 vs 25502)

*Gen15 · campaign `Campaign_20260907_five_missions` · 2026-09-09.
Source batch: **`temp/0909/batch_uav_20260909_205118`** (`DA_UAV_v1`, 2026-09-09 20:51:50, 1288 units,
0 failed). Candidates **C94** (`controller=mjpc`) vs **C95** (`controller=pid_stopgo`).*

📄 **Paper-ready write-up:**
[`Data_Analysis/DA_Result_Curated_MD/Report_20260909_MJPC_vs_PID_s_curve/README.md`](../../../Data_Analysis/DA_Result_Curated_MD/Report_20260909_MJPC_vs_PID_s_curve/README.md)
(curated findings table F1–F6, figures + specs, LaTeX table)

The question, in the user's words: *"select the worst failing S_curve example, run the mjpc solver vs
the old pid solver … let's see if the controller is not powerful."*

## 0. TL;DR — the controller **was** the bottleneck, but it does not rescue the scene

1. **On the raw MeanFlow plan the answer is unambiguous: `pid_stopgo` reaches the goal 0/10;
   `mjpc` reaches it 3/3, on the same three initial conditions.** Goal distance collapses
   2.86/2.72/2.89 → 0.299/0.298/0.294. Paired, and Fisher-exact **p ≈ 0.0035** even at n=3.
   **The plan was flyable all along. The tracker could not fly it.** §3
2. **But S&C stays 0.000 on every cell, under both controllers.** `mjpc` converts *"never arrives"*
   into *"arrives, still violates"* — it buys `goal_reached`, not `success_and_constraints`. §6
3. **The failure is not one bug but three different ones, one per variant** — a stalling tracker
   (`diffuser`), a broken projection (`dpcc-r`), and a ground-scraping flight path (`hardflow`).
   Only the first is the controller's fault. §3–§5

---

## 1. Provenance, and why n=3 is stronger than it looks

| | C95 — **PID** | C94 — **MJPC** |
|---|---|---|
| job | 25502 | 25514 → **25554** |
| controller | `pid_stopgo` | **`mjpc`** (`FMPCC_mjx` env, `MJPCTracker`) |
| engine / model | `mf` · `MeanFlowODE_9D_dp0.5_bbunet` | *identical* |
| scene · K · tag · seed | `s_curve` · K=10 · `u7hg` · 6 | *identical* |
| eval path | `Emf_K10_mpc4_pid_stopgo_T0.5_u7hg` | `Emf_K10_mpc4_mjpc_T0.5_u7hg` |
| **n per cell** | **10** | **3** |
| shared variants | `diffuser`, `dpcc-r`, `hardflow_sls-r` | *same 3* |

Everything except the controller is matched. The trial seed is derived from `rollout_idx`, and
`phys_min_z` at a given index is identical across *different variants* within a run
(1.106 / 1.186 / 1.154 at indices 0/1/2 for both `diffuser` and `dpcc-r` under PID) — so
**`rollout_idx` names the same initial condition in both runs.** MJPC's rollouts 0–2 are therefore
*paired* against PID's rollouts 0–2, not an unpaired sample of 3.

🔴 **Standing caveat:** n=3 vs n=10 is unmatched, deliberately (mission 5 was specified with fewer
trials). Every rate on the MJPC side is out of 3 — a bare "3/3" carries a Wilson 95 % lower bound of
only 0.29. The §3 result survives this because PID's 0/10 has an *upper* bound of 0.31, so the two
intervals do not overlap; nothing else in this DA leans on the MJPC rate alone.

---

## 2. The three variants, side by side

`goal_dist` and `n_violations` are means; **`steps_to_goal = nan` means the goal was never reached.**

| variant | ctrl | S&C | success | safe | goal_reached | goal_dist | steps_to_goal | track_err | n_viol | avg_ms | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `diffuser` | PID | 0.000 | 0.000 | 0.000 | **0.000** | 2.809 | **nan** | **0.304** | 13.7 | 88.4 | 10 |
| `diffuser` | **MJPC** | 0.000 | **0.667** | 0.667 | **1.000** | **0.297** | **636** | 0.455 | 48.7 | 89.8 | 3 |
| `dpcc-r` | PID | 0.000 | 0.000 | 0.000 | 0.000 | 2.665 | nan | 0.397 | 29.1 | 2212.2 | 10 |
| `dpcc-r` | **MJPC** | 0.000 | 0.000 | 0.000 | **0.000** | 2.571 | nan | 0.585 | 59.0 | 2512.7 | 3 |
| `hardflow_sls-r` | PID | 0.000 | 0.000 | 0.000 | 0.700 | 0.834 | 639.7 | 0.662 | 166.9 | 1718.6 | 10 |
| `hardflow_sls-r` | **MJPC** | 0.000 | 0.000 | 0.000 | **1.000** | **0.297** | **618** | **0.511** | **48.0** | **895.5** | 3 |

---

## 3. `diffuser` — the controller was the whole failure

Per rollout, PID (all 10) vs MJPC (the 3 paired indices):

| idx | PID `goal_dist` | PID `n_steps` | **MJPC `goal_dist`** | **MJPC `steps_to_goal`** |
|---|---|---|---|---|
| 0 | 2.863 | 871 (budget exhausted) | **0.299** | **633** |
| 1 | 2.718 | 871 | **0.298** | **650** |
| 2 | 2.889 | 871 | **0.294** | **625** |
| 3–9 | 2.617 – 2.873 | 871 ×7 | — | — |

**All ten PID rollouts burn the full 871-step budget and stop ~2.8 m short.** Not one diverges, not
one crashes: `phys_min_z` stays at 0.96–1.24 throughout. The drone is airborne, stable, and simply
does not get there.

The diagnostic that names the mechanism is **`track_err`, which is anti-correlated with success**:

* PID: `track_err` **0.294–0.341** — the *best* tracking in the entire table — and `goal_reached` = 0.
* MJPC: `track_err` **0.422–0.490** — visibly *worse* — and `goal_reached` = 1.

A controller that follows the reference tightly while never advancing along it is not failing to
track; it is failing to make **progress**. That is `pid_stopgo`'s stop-and-go behaviour saturating on
a scene whose reference demands sustained forward velocity. MJPC accepts a looser corridor around
the reference and converts it into motion.

**Verdict: on the unprojected MeanFlow plan, the `s_curve` failure was a controller artefact,
exactly as mission 5 hypothesised.**

---

## 4. `dpcc-r` — the controller is *not* the problem; the projector is

**0/10 under PID, 0/3 under MJPC.** Goal distance barely moves (2.665 → 2.571) and violations rise
(29.1 → 59.0). MJPC, which rescued the raw plan completely, does nothing here.

PID's ten rollouts split into two failure modes:

* **8 of 10** — `goal_dist` ≈ 2.85, `n_violations` 7–11, `track_err` ≈ 0.33: the same stall as §3.
* **2 of 10** (idx 3, 4) — `goal_dist` 1.80/2.02, `n_violations` 99/120, `phys_min_z` **0.545**,
  `phys_contact_frac` 0.172: these are not stalls, they are scrapes.

**Verdict: the DPCC-projected plan on `s_curve` is not flyable by either tracker.** Whatever the
projector emits here, a stronger controller cannot fly it. This is a projector finding, not a
controller finding, and it isolates `dpcc-r` on `s_curve` as the thing to investigate next.

Note also the cost: `proj_ms` = **2122 (PID) / 2420 (MJPC)** — the projection dominates the step by
more than 20×.

---

## 5. `hardflow_sls-r` — PID "succeeds" by dragging along the floor

This is the row where the aggregate numbers most mislead. PID posts `goal_reached` = 0.700, which
looks respectable. The per-rollout altitudes say otherwise:

| ctrl | `phys_min_z` per rollout |
|---|---|
| **PID** | −0.005, 0.005, 0.014, 0.002, 0.022, −0.002, 0.020, −0.009, 0.013, 0.019 |
| **MJPC** | **1.050, 1.143, 1.112** |

**Every PID rollout descends to ground level or below** (two are negative). It reaches the goal by
sliding along the floor. MJPC flies the same plans at ~1.1 m and reaches the goal 3/3, with

* `goal_dist` 0.834 → **0.297**,
* `n_violations` 166.9 → **48.0** (3.5× fewer),
* `track_err` 0.662 → **0.511**,
* `proj_ms` 1587.5 → **761.8** (the NLP converges faster from better-conditioned states).

`phys_safe` = 0 for both, but for different reasons: PID's is an altitude failure, MJPC's is
`phys_contact_frac` (0.147–0.220).

**Verdict: MJPC strictly improves the HardFlow arm on every axis measured, including cost.**

---

## 6. 🔴 What mission 5 does *not* fix

**S&C = 0.000 in all six cells.** Under MJPC, `diffuser` reaches `success` 0.667 and `safe` 0.667 —
but S&C requires success *and* a clean constraint record, and `n_violations` runs 28–63 on the very
rollouts that arrive.

Normalising by episode length shows the trade is **real, not a length artefact** — and that it
reverses between variants:

| variant | PID viol/step | MJPC viol/step | |
|---|---|---|---|
| `diffuser` | 13.7/871 = **0.0157** | 48.7/636 = **0.0765** | MJPC **4.9× worse** |
| `dpcc-r` | 29.1/871 = 0.0334 | 59.0/871 = 0.0677 | MJPC 2.0× worse |
| `hardflow_sls-r` | 166.9/709 = **0.2354** | 48.0/618 = **0.0777** | MJPC **3.0× better** |

On the raw plan MJPC genuinely violates ~5× more per step: it buys progress with constraint
adherence. On the HardFlow arm it violates 3× *less* per step while also flying at altitude — a
strict improvement. Note too that MJPC's rate is near-constant across variants (0.068–0.077) while
PID's spans 15× (0.016–0.235): PID's violation statistics are dictated by which pathology it falls
into, whereas MJPC has a characteristic cost.

So mission 5 answers its own question — *yes, the controller was not powerful enough* — while
**failing to rescue `s_curve` as a rankable scene.** The S&C floor at 0.00 that made `s_curve`
useless for ranking is still at 0.00. Swapping the tracker moves the failure from
*"cannot reach the goal"* to *"reaches the goal through the constraints"*.

### Consequence for the campaign

`corridor` saturates at 1.000 (`DA_20260908_T2`), pre-U7 `pillars` is void, and `s_curve` floors at
0.000 under **both** controllers. **The af/mf UAV arm still has no scene with usable dynamic range**,
and mission 5 — the last hypothesis that a single knob would restore it — has now been tested and
does not deliver it. The dynamic-range problem is a **scene/constraint-set** problem, not a
controller problem.

---

## 7. What this licenses

**Supported.** `pid_stopgo` cannot fly the raw MeanFlow `s_curve` plan (0/10 vs 3/3 paired,
p ≈ 0.0035); its failure mode is a progress stall, not divergence, identified by tight `track_err`
with zero goal reach. `mjpc` strictly dominates on the HardFlow arm. The DPCC-projected plan is
unflyable by either controller. `mjpc` runs correctly on Gen15 UAV (U10 validated end-to-end).

**Not supported.** Any S&C claim — the floor is unchanged. Any multi-seed claim (seed 6). Any rate
from the MJPC side alone at n=3 beyond the paired `diffuser` contrast. Any statement about the other
five variants of C95 (`dpcc-c`, `dpcc-t`, `hardflow_sls`, `-c`, `-t`), which have no MJPC counterpart.

**Next.**
1. Re-run the MJPC arm at **n=10** and across the full 8-variant subset — the effect is large enough
   to be worth a matched-power measurement, and the run cost only 1.6 h.
2. Investigate **`dpcc-r` on `s_curve`** specifically: a projected plan that no controller can fly is
   a projector defect, and it is the one failure mission 5 cleanly isolated.
3. Do **not** expect a controller change to make `s_curve` rankable; that needs the constraint set
   revisited.
