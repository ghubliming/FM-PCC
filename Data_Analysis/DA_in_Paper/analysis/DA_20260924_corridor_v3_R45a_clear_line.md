# DA 2026-09-24 · UAV-corridor v3 — R45a: the finish line at the end of the corridor (x′ = 2.0 m), beside the original rule

**For v3 (Ch 5 success clause, the Ch 6 corridor section, the §6.4 flag) and v4.** Spec: `data_status/PENDING_20260924_uav_corridor_step_cap_R45.md`
§2 (author, 24-09). Data processing only, nothing run. Corpus = the DA of record
[`DA_20260924_corridor_v3.md`](DA_20260924_corridor_v3.md): outcomes from `batch_uav_20260924_081422`, flown paths from the npz of
all 136 cells; the first ten flights per cell (1 360 flights). Script: [`corridor_v3_r45a_clear_line.py`](corridor_v3_r45a_clear_line.py)
(`--recorded-only` for the alternative reading of §3). The v3 agent's preview (`R45_preview/`) is reproduced exactly on the
reading it used.

**Code (24-09, author's go-ahead):** `mix_uav_test/eval_mix_uav.py` now applies this rule itself
(`SCENE_CLEAR_LINE_X = {'corridor': 2.0}`). The test runs after physics, the flights are unchanged, and the goal-plane
rule is kept as `success.relaxed_goal_line`. See
[`CHANGELOG_20260924_U19_corridor_finish_line_coding2.md`](../../../logs_in_develop/Gen15/U19/CHANGELOG_20260924_U19_corridor_finish_line_coding2.md).
Corpora evaluated before this change keep the old rule in their files; for R33 this DA gives what the new code would
score.

## 1 · The rule

- **success (original):** the eval's finish latch — within 0.30 m of the route's goal point, or past the plane through it
  (x ≈ 2.8) — **and** controlled flight (contact within the scene limit, lowest altitude > 0.2 m, no divergence abort).
- **success_clear (R45a):** the original success, **or** the vehicle's centre reaches **x ≥ 2.0 m** — the end of the corridor,
  where the walls and both test-time constraints end — within the 396-step budget, in controlled flight.
- **S&C:** the success flag **and** no violating step over the whole flight, counted exactly as before.
- **Where x is read:** like the eval's own latch, on the position **after** each control step's physics. The recorded
  `obs_all` holds each step's *start* position, so the position after the last step is not in it; it is recovered from the
  stored final goal distance and final altitude (validated on all 222 flights that ran to the cap: equal to "last recorded x
  + last step's Δx" within 0.52 mm). Reading x on the recorded start positions only changes **one** flight (§3).

## 2 · Result — both rules side by side

Only the six baseline projected cells change in S&C; no flow-model cell does. Three flow-model flights gain a success without
gaining S&C: three of the six $\nfe=2$ flights (MeanFM 1, CI-MeanFM 2, rule t) that get over the hump's apex and then drift
sideways across the wall's boundary reach x′; they violate the wall and keep S&C 0.

| baseline ★, per-step projection | rule | S&C original | **S&C at x′ = 2.0** | success original → x′ | violation-free | ms/step |
| :-- | :-- | --: | --: | :-- | --: | --: |
| tilt | r | 0/10 | **3/10** | 2 → 10 | 3/10 | 679.7 |
| tilt | c | 0/10 | **8/10** | 0 → 10 | 8/10 | 666.3 |
| tilt | t | 0/10 | **3/10** | 3 → 10 | 3/10 | 650.6 |
| hump | r | 3/10 | **10/10** | 3 → 10 | 10/10 | 654.2 |
| hump | c | 3/10 | **9/10** | 3 → 9 | 10/10 | 623.1 |
| hump | t | 6/10 | **9/10** | 7 → 10 | 9/10 | 625.9 |

### 2.1 The tables

- **`tab:uav-corridor-raw`** (before projection): unchanged — every unprojected flight already succeeds.
- **`tab:uav-corridor-projection`**: the baseline's per-step row becomes tilt **3 / 8 / 3** and hump **10 / 9 / 9** (r / c / t;
  was 0 / 0 / 0 and 3 / 3 / 6); every other row unchanged.
- **`tab:uav-corridor`** (best rule per projector; ties → fewer violating steps → lower ms/step): the baseline row becomes
  tilt **8/10, c, 666.3 ms** (was 0/10, c, 666.3) and hump **10/10, r, 654.2 ms** (was 6/10, t, 625.9); every other row
  unchanged.

### 2.2 The frontier (`fig:uav-corridor-tradeoff`, DA §3.4)

S&C against ms/step, one point per model × budget × projector at its best rule, S&C 0 not eligible:

| block | original rule | x′ = 2.0 |
| :-- | :-- | :-- |
| tilt | none (0 eligible of 17) | **the baseline, per-step c** — 8/10 at 666.3 ms, the only eligible point |
| hump | FM $\nfe=20$ endpoint single (8/10, 291.0 ms); the baseline's t (6/10, 625.9 ms) dominated | **two non-dominated points: FM $\nfe=20$ endpoint single (8/10, 291.0 ms) and the baseline per-step r (10/10, 654.2 ms)** — a trade-off: two more violation-free successes for 2.2 × the time per action |

### 2.3 The budget facts (DA §3.2)

Unchanged: violating steps are counted over the whole flight as before and no flight is re-flown, so the mean violating steps,
the deepest steps, the price per action and the baseline's step counts (cell means 377–396) all stand. Only the result sentence
changes:
- *Original:* on the tilt no configuration succeeds violation-free; on the hump FM $\nfe=20$ under endpoint projection (8/10,
  291 ms) and the baseline under per-step projection (6/10, 626 ms).
- *x′ = 2.0:* on the tilt the only violation-free successes are the baseline's, under per-step projection (c 8/10 at 666 ms;
  r and t 3/10); no flow model has one at any budget under either projector. On the hump they are FM $\nfe=20$ under endpoint
  projection (single 8/10 at 291 ms; r 6, c 6, t 4) and the baseline under per-step projection (r 10/10 at 654 ms; c 9, t 9).
  Every flow model under per-step projection still scores 0/10 there: stopped on the roof at $\nfe \le 2$, a residue at the
  apex from $\nfe=3$.

### 2.4 The conclusion line (`sec:res:uav:conclusion`, DA §3.7) — facts under both rules

| | original rule (DA §3.7) | x′ = 2.0 |
| :-- | :-- | :-- |
| tilt | every flow model at every budget crosses the line, none violation-free (hand-over residue of 1.4–11 steps at the window end, setpoint clean); the baseline reaches the line on at most 3 of 10 flights | flow models unchanged (all their violating steps lie inside the corridor, x 1.71–2.00, so the moved line cannot reach them); the baseline reaches the end of the corridor on 10/10 flights under every rule and is violation-free on 3–8 of 10 — **the only configuration with S&C > 0: per-step c, 8/10 at 666 ms** |
| hump | FM $\nfe=20$ endpoint 8/10 at 291 ms, ahead of the baseline's best, 6/10 at 626 ms; per-step flow stops on the roof at $\nfe \le 2$ | **a trade-off:** FM $\nfe=20$ endpoint single 8/10 at 291 ms and the baseline per-step r 10/10 at 654 ms, neither dominates (two more violation-free successes for 2.2 × the time per action); per-step flow still stops on the roof at $\nfe \le 2$ (3 of 180 flights reach x′ after drifting into the wall, S&C 0) |
| the three flow models | do not separate: within two violating steps of each other, each at its best rule (the v3.80 wording, DA addendum) | unchanged (violation counts do not move) |

Sentences of the DA of record that hold **only under the original rule**:
- §1 "FM $\nfe=20$ endpoint is ahead of the baseline on the hump on both counts" and §3.4 "the baseline point is dominated".
- §3.7 §6.4 facts "the best corridor configuration is FM $\nfe=20$ endpoint single on the hump; on the tilt no combination
  succeeds violation-free". At x′ = 2.0 there is no single best: on the tilt only the baseline succeeds violation-free
  (per-step c 8/10, 666 ms); on the hump FM $\nfe=20$ endpoint single and the baseline per-step r are both non-dominated. The
  v2-corridor quotes of §6.4 still do not hold (the chapter's 🔒 applies — author's call).
- §3.5 "Baseline, projected … run out of steps 0.16–0.39 m before the line": 42 of the baseline's 60 projected flights used all
  396 control steps without reaching the old line; 41 of them had passed the end of the corridor and were still moving toward the
  goal (1.8–20.0 mm per step). The one that had not, hump c trial 7, ends at x = 1.23 m (§5).

## 3 · The flight that ends at x = 2.0 (the author's question)

Hump, baseline, per-step **r**, trial 7, route C:
- furthest **recorded** position (the start of its 396th step): **x = 1.998785 m — 1.2 mm short** of the line;
- position **after its 396th and last step**: **x = 2.0167 m — 16.7 mm past** the line (moving +18.8 mm per step);
- safe, no abort, **violation-free**.

The eval tests its own finish latch after each step's physics, so the reading consistent with it counts this flight: **past
the line, S&C** → hump r **10/10**. On recorded start positions only (the preview's reading) it is short → hump r 9/10, and
the baseline's best hump rule becomes c (9/10, 623.1 ms). Nothing else differs between the two readings; the hump frontier is
a trade-off under both. No other flight ends within 5 cm of the line.

## 4 · The four checks on every flight the moved line turns into a success (44 flights)

- **(1) past x′ before the cap:** all 44 (the baseline's first recorded step past x = 2.0 is step 260–382 of 396; the flight
  of §3 crosses during its last step).
- **(2) safe, no abort:** all 44 (no corridor flight aborted or was unsafe; lowest altitude over all flights 0.63 m, contact 0).
- **(3) still moving toward the goal at the cap:** all 44 — baseline flights 1.8–20.0 mm per step along x, 0.7–19.8 mm per step
  closer to the goal point (over the last 30 control steps).
- **(4) no violating step past x′:** 41 of 44 — every baseline flip; the exceptions are the three flow wall-drift flights
  (6–10 violating steps each past x = 2.0, at x 2.01–2.23 and y −0.97 to −0.73: the body clips a wall-end cap after the
  flight has crossed the wall's boundary sideways). They have 65–79 violating steps each, most of them against the wall before
  x′, and stay S&C 0 under any line.
- Of the 41 baseline flips, 30 are violation-free (→ S&C) and 11 are not (1–12 violating steps, all before x = 2.0 — the tilt's
  window-end residue).

## 5 · Corpus facts of the spec — verified

- flights read: 1360 (136 cells × 10); re-scored violating steps equal the stored ones on 1360/1360
- not successful under the old rule: 222; on the 396-step cap: 222; aborted 0; unsafe 0 (min altitude over all flights 0.63 m, max contact fraction 0.0000)
- by group: baseline hump ps-c 7; baseline hump ps-r 7; baseline hump ps-t 3; baseline tilt ps-c 10; baseline tilt ps-r 8; baseline tilt ps-t 7; flow hump ps-c 60; flow hump ps-r 60; flow hump ps-t 60
- the 42 baseline flights on the cap end 0.41–0.74 m (tilt) and 0.36–1.60 m (hump) from the goal point (last recorded position), moving along x at 1.8–4.8 mm per step on the tilt and 2.8–23.4 on the hump
- violating steps at x ≥ 2.00: 25 in 3 flight(s): hump MeanFM K2 ps-t trial 9 (9), hump CI-MeanFM K2 ps-t trial 8 (6), hump CI-MeanFM K2 ps-t trial 9 (10)
- violating steps at x ≥ 2.31: 0 in 0 flight(s)
- violating steps at x ≥ 2.36: 0 in 0 flight(s)
- violating steps at x ≥ 2.50: 0 in 0 flight(s)

**Correction to the spec's §2 sensitivity note.** Trial *i* = the i-th flight of the cell, counted from 0, seeded
`default_rng(10000 + i)`, route `LCR[i % 3]` — the same indexing as the DA of record. The hump baseline flight that ends at
x ≈ 2.26 (recorded 2.2569, after its last step 2.2751) and drops out between x′ = 2.0 and 2.31 is rule **c, trial 8, route R**,
not trial 7. Trial 7 of that cell (route C) ends at x = 1.23 m, 1.32 m high, still moving at +23 mm per step: short of every
line, and the one baseline flight that stays unsuccessful at x′ = 2.0.

## 6 · Sensitivity to the line's position

Rule of record (position after each step). On the recorded start positions the only difference is hump r at x′ = 2.00: 9
instead of 10 (§3); every other entry is equal. The preview's table is reproduced on that reading.

| block | model | nfe | variant | S&C old | x'=2.00 | x'=2.31 | x'=2.36 | x'=2.50 |
| :-- | :-- | --: | :-- | --: | --: | --: | --: | --: |
| tilt | Diffusion ★ | 20 | ps-c | 0 | 8 | 8 | 8 | 3 |
| tilt | Diffusion ★ | 20 | ps-r | 0 | 3 | 3 | 3 | 3 |
| tilt | Diffusion ★ | 20 | ps-t | 0 | 3 | 3 | 3 | 3 |
| hump | MeanFM | 2 | ps-t | 0 | 0 | 0 | 0 | 0 |
| hump | CI-MeanFM | 2 | ps-t | 0 | 0 | 0 | 0 | 0 |
| hump | Diffusion ★ | 20 | ps-c | 3 | 9 | 8 | 8 | 4 |
| hump | Diffusion ★ | 20 | ps-r | 3 | 10 | 9 | 9 | 6 |
| hump | Diffusion ★ | 20 | ps-t | 6 | 9 | 9 | 9 | 9 |

## Appendix A · Every cell, both rules

| block | model | nfe | variant | success old → clear | S&C old → clear | changed |
| :-- | :-- | --: | :-- | :-- | :-- | :-- |
| tilt | MeanFM | 1 | none | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 1 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 1 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 1 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 2 | none | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 2 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 2 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 2 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | ep-c | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | ep-r | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | ep-single | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | ep-t | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | none | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | MeanFM | 3 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 1 | none | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 1 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 1 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 1 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 2 | none | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 2 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 2 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 2 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | ep-c | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | ep-r | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | ep-single | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | ep-t | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | none | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | CI-MeanFM | 3 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 1 | none | 10 → 10 | 0 → 0 |  |
| tilt | FM | 1 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 1 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 1 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 2 | none | 10 → 10 | 0 → 0 |  |
| tilt | FM | 2 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 2 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 2 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | ep-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | ep-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | ep-single | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | ep-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | none | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 3 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | ep-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | ep-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | ep-single | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | ep-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | none | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 5 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | ep-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | ep-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | ep-single | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | ep-t | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | none | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | ps-c | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | ps-r | 10 → 10 | 0 → 0 |  |
| tilt | FM | 20 | ps-t | 10 → 10 | 0 → 0 |  |
| tilt | Diffusion ★ | 20 | none | 10 → 10 | 0 → 0 |  |
| tilt | Diffusion ★ | 20 | ps-c | 0 → 10 | 0 → 8 | **yes** |
| tilt | Diffusion ★ | 20 | ps-r | 2 → 10 | 0 → 3 | **yes** |
| tilt | Diffusion ★ | 20 | ps-t | 3 → 10 | 0 → 3 | **yes** |
| hump | MeanFM | 1 | none | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 1 | ps-c | 0 → 0 | 0 → 0 |  |
| hump | MeanFM | 1 | ps-r | 0 → 0 | 0 → 0 |  |
| hump | MeanFM | 1 | ps-t | 0 → 0 | 0 → 0 |  |
| hump | MeanFM | 2 | none | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 2 | ps-c | 0 → 0 | 0 → 0 |  |
| hump | MeanFM | 2 | ps-r | 0 → 0 | 0 → 0 |  |
| hump | MeanFM | 2 | ps-t | 0 → 1 | 0 → 0 | **yes** |
| hump | MeanFM | 3 | ep-c | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 3 | ep-r | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 3 | ep-single | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 3 | ep-t | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 3 | none | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 3 | ps-c | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 3 | ps-r | 10 → 10 | 0 → 0 |  |
| hump | MeanFM | 3 | ps-t | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 1 | none | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 1 | ps-c | 0 → 0 | 0 → 0 |  |
| hump | CI-MeanFM | 1 | ps-r | 0 → 0 | 0 → 0 |  |
| hump | CI-MeanFM | 1 | ps-t | 0 → 0 | 0 → 0 |  |
| hump | CI-MeanFM | 2 | none | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 2 | ps-c | 0 → 0 | 0 → 0 |  |
| hump | CI-MeanFM | 2 | ps-r | 0 → 0 | 0 → 0 |  |
| hump | CI-MeanFM | 2 | ps-t | 0 → 2 | 0 → 0 | **yes** |
| hump | CI-MeanFM | 3 | ep-c | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 3 | ep-r | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 3 | ep-single | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 3 | ep-t | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 3 | none | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 3 | ps-c | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 3 | ps-r | 10 → 10 | 0 → 0 |  |
| hump | CI-MeanFM | 3 | ps-t | 10 → 10 | 0 → 0 |  |
| hump | FM | 1 | none | 10 → 10 | 0 → 0 |  |
| hump | FM | 1 | ps-c | 0 → 0 | 0 → 0 |  |
| hump | FM | 1 | ps-r | 0 → 0 | 0 → 0 |  |
| hump | FM | 1 | ps-t | 0 → 0 | 0 → 0 |  |
| hump | FM | 2 | none | 10 → 10 | 0 → 0 |  |
| hump | FM | 2 | ps-c | 0 → 0 | 0 → 0 |  |
| hump | FM | 2 | ps-r | 0 → 0 | 0 → 0 |  |
| hump | FM | 2 | ps-t | 0 → 0 | 0 → 0 |  |
| hump | FM | 3 | ep-c | 10 → 10 | 0 → 0 |  |
| hump | FM | 3 | ep-r | 10 → 10 | 0 → 0 |  |
| hump | FM | 3 | ep-single | 10 → 10 | 0 → 0 |  |
| hump | FM | 3 | ep-t | 10 → 10 | 0 → 0 |  |
| hump | FM | 3 | none | 10 → 10 | 0 → 0 |  |
| hump | FM | 3 | ps-c | 10 → 10 | 0 → 0 |  |
| hump | FM | 3 | ps-r | 10 → 10 | 0 → 0 |  |
| hump | FM | 3 | ps-t | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | ep-c | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | ep-r | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | ep-single | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | ep-t | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | none | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | ps-c | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | ps-r | 10 → 10 | 0 → 0 |  |
| hump | FM | 5 | ps-t | 10 → 10 | 0 → 0 |  |
| hump | FM | 20 | ep-c | 10 → 10 | 6 → 6 |  |
| hump | FM | 20 | ep-r | 10 → 10 | 6 → 6 |  |
| hump | FM | 20 | ep-single | 10 → 10 | 8 → 8 |  |
| hump | FM | 20 | ep-t | 10 → 10 | 4 → 4 |  |
| hump | FM | 20 | none | 10 → 10 | 0 → 0 |  |
| hump | FM | 20 | ps-c | 10 → 10 | 0 → 0 |  |
| hump | FM | 20 | ps-r | 10 → 10 | 0 → 0 |  |
| hump | FM | 20 | ps-t | 10 → 10 | 0 → 0 |  |
| hump | Diffusion ★ | 20 | none | 10 → 10 | 0 → 0 |  |
| hump | Diffusion ★ | 20 | ps-c | 3 → 9 | 3 → 9 | **yes** |
| hump | Diffusion ★ | 20 | ps-r | 3 → 10 | 3 → 10 | **yes** |
| hump | Diffusion ★ | 20 | ps-t | 7 → 10 | 6 → 9 | **yes** |

cells that change: 8 of 136; in S&C: 6

## Appendix B · The 44 flights the moved line turns into a success

| block | model | nfe | variant | trial | route | x max recorded | x after the last step | first recorded step at x ≥ 2.0 / steps | safe, no abort | pace at the cap: x (mm/step) · toward goal (mm/step) | violating steps past 2.0 | violation-free → S&C |
| :-- | :-- | --: | :-- | --: | :-- | --: | --: | :-- | :-- | :-- | --: | :-- |
| hump | CI-MeanFM | 2 | ps-t | 8 | R | 2.1269 | 2.1502 | 390 / 396 | yes | +25.7 · +18.3 | 6 | no (65 steps) → no S&C |
| hump | CI-MeanFM | 2 | ps-t | 9 | L | 2.2253 | 2.2470 | 386 / 396 | yes | +24.5 · +18.1 | 10 | no (75 steps) → no S&C |
| hump | Diffusion ★ | 20 | ps-c | 0 | L | 2.4704 | 2.4728 | 330 / 396 | yes | +3.8 · +3.4 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-c | 3 | L | 2.4086 | 2.4124 | 334 / 396 | yes | +4.9 · +4.2 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-c | 5 | R | 2.3808 | 2.3978 | 375 / 396 | yes | +19.1 · +17.2 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-c | 6 | L | 2.5025 | 2.5049 | 315 / 396 | yes | +3.6 · +1.8 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-c | 8 | R | 2.2569 | 2.2751 | 382 / 396 | yes | +20.0 · +19.8 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-c | 9 | L | 2.4932 | 2.4980 | 327 / 396 | yes | +5.8 · +4.6 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-r | 0 | L | 2.5278 | 2.5305 | 307 / 396 | yes | +4.0 · +3.2 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-r | 1 | C | 2.4663 | 2.4794 | 364 / 396 | yes | +14.6 · +14.5 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-r | 3 | L | 2.4567 | 2.4608 | 328 / 396 | yes | +5.6 · +4.2 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-r | 6 | L | 2.5820 | 2.5844 | 298 / 396 | yes | +2.9 · +2.2 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-r | 7 | C | 1.9988 | 2.0167 | — (after the last step) / 396 | yes | +18.8 · +18.8 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-r | 8 | R | 2.3700 | 2.3849 | 373 / 396 | yes | +16.4 · +15.6 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-r | 9 | L | 2.5173 | 2.5206 | 316 / 396 | yes | +4.1 · +2.6 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-t | 0 | L | 2.6432 | 2.6451 | 282 / 396 | yes | +2.8 · +1.9 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-t | 6 | L | 2.7029 | 2.7045 | 299 / 396 | yes | +2.8 · +0.7 | 0 | yes → S&C |
| hump | Diffusion ★ | 20 | ps-t | 9 | L | 2.6239 | 2.6257 | 313 / 396 | yes | +3.7 · +2.3 | 0 | yes → S&C |
| hump | MeanFM | 2 | ps-t | 9 | L | 2.1650 | 2.1841 | 387 / 396 | yes | +22.5 · +13.3 | 9 | no (79 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-c | 0 | L | 2.5501 | 2.5521 | 276 / 396 | yes | +2.9 · +2.2 | 0 | no (5 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-c | 1 | C | 2.4075 | 2.4097 | 297 / 396 | yes | +2.6 · +1.5 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-c | 2 | R | 2.4671 | 2.4708 | 296 / 396 | yes | +3.3 · +2.0 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-c | 3 | L | 2.5057 | 2.5082 | 284 / 396 | yes | +3.1 · +2.0 | 0 | no (3 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-c | 4 | C | 2.5326 | 2.5356 | 301 / 396 | yes | +3.9 · +2.4 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-c | 5 | R | 2.4107 | 2.4145 | 313 / 396 | yes | +4.6 · +3.2 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-c | 6 | L | 2.5411 | 2.5437 | 279 / 396 | yes | +3.1 · +1.8 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-c | 7 | C | 2.4816 | 2.4829 | 295 / 396 | yes | +2.3 · +1.2 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-c | 8 | R | 2.4252 | 2.4291 | 302 / 396 | yes | +4.6 · +3.1 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-c | 9 | L | 2.6243 | 2.6258 | 260 / 396 | yes | +2.5 · +1.0 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-r | 0 | L | 2.5614 | 2.5641 | 268 / 396 | yes | +4.0 · +2.8 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-r | 1 | C | 2.5600 | 2.5622 | 276 / 396 | yes | +2.6 · +1.1 | 0 | no (2 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-r | 3 | L | 2.5432 | 2.5461 | 284 / 396 | yes | +3.7 · +2.6 | 0 | no (5 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-r | 4 | C | 2.5562 | 2.5604 | 297 / 396 | yes | +4.8 · +3.1 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-r | 6 | L | 2.4854 | 2.4874 | 289 / 396 | yes | +2.3 · +1.5 | 0 | no (1 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-r | 7 | C | 2.5861 | 2.5885 | 286 / 396 | yes | +3.3 · +1.8 | 0 | no (3 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-r | 8 | R | 2.5491 | 2.5513 | 285 / 396 | yes | +2.3 · +0.8 | 0 | no (4 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-r | 9 | L | 2.5152 | 2.5190 | 296 / 396 | yes | +3.8 · +2.7 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-t | 0 | L | 2.5964 | 2.5985 | 263 / 396 | yes | +3.1 · +2.6 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-t | 1 | C | 2.5124 | 2.5151 | 288 / 396 | yes | +3.0 · +2.0 | 0 | yes → S&C |
| tilt | Diffusion ★ | 20 | ps-t | 3 | L | 2.5071 | 2.5109 | 315 / 396 | yes | +3.8 · +2.6 | 0 | no (8 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-t | 4 | C | 2.6753 | 2.6770 | 284 / 396 | yes | +2.8 · +1.0 | 0 | no (12 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-t | 6 | L | 2.6931 | 2.6940 | 275 / 396 | yes | +1.8 · +1.1 | 0 | no (5 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-t | 7 | C | 2.6392 | 2.6421 | 268 / 396 | yes | +3.3 · +2.0 | 0 | no (7 steps) → no S&C |
| tilt | Diffusion ★ | 20 | ps-t | 9 | L | 2.6008 | 2.6038 | 293 / 396 | yes | +4.1 · +2.8 | 0 | yes → S&C |

- (1) past x' before the cap: all; (2) safe, no abort: all; (3) moving toward the goal at the cap: 44/44; (4) no violating step past x': 41/44 (not: [('hump', 'MeanFM', 2, 'ps-t', 9, 9), ('hump', 'CI-MeanFM', 2, 'ps-t', 8, 6), ('hump', 'CI-MeanFM', 2, 'ps-t', 9, 10)])

## Appendix C · The near-satisfaction bar: the corridor's second frontier reading (author, 24-09)

**Status.** The author asked (24-09) whether the flow models reach the frontier if a flight counts as satisfying the
constraints when at least 90–95 % of its control steps are violation-free. The author then took this bar as the corridor's
**second frontier reading, printed beside the strict one.**
- **Scope:** UAV-corridor only.
- **Unchanged:** the tables and every S&C column stay strict (success at x′ and zero violating steps).

**Definition.** A flight is *near-satisfying* when it succeeds at x′ (§1) **and** at least **95 %** of its control steps
are violation-free. A bar of 0.90 gives the same frontier; the result at 0.97 is shown as the sensitivity.

**Method.**
- Best rule per projector: most near-satisfying flights, then fewer violating steps, then lower ms.
- Frontier as in §2 (costs within 1 % count as one cost).
- Inputs: the R45a flights (`--json`: `n_viol`, `steps`, success at x′) and the batch's `avg_time_ms`, first ten flights.

**The frontier under each bar:**

| violation-free share of a flight's steps | tilt frontier | hump frontier |
| :-- | :-- | :-- |
| 1.00, strict S&C (**the rule of the tables**) | baseline per-step c 8/10, 666.3 ms | FM K20 endpoint single 8/10, 291.0 ms; baseline per-step r 10/10, 654.2 ms |
| **≥ 0.95** (the same at ≥ 0.90) | **nfe=1 per-step c, 10/10:** MeanFM 27.4 ms = FM 27.6 (CI-MeanFM 27.9) | **nfe=3 endpoint c, 10/10:** CI-MeanFM 71.9 ms = FM 71.9 = MeanFM 72.1 |
| ≥ 0.97 (sensitivity) | a staircase: FM K1 per-step 2/10 (27.6 ms) → MeanFM K1 3/10 (28.5) → MeanFM = CI-MeanFM K2 4/10 (37.6) → FM K3 endpoint 7/10 (71.6) → CI-MeanFM K3 endpoint 9/10 (72.7) → FM K3 per-step 10/10 (79.2) | CI-MeanFM K3 endpoint c = FM K3 endpoint c, 10/10, 71.9 ms |

**Every configuration** (the best rule at ≥ 0.95; the other two columns are the *same* rule):

| block | model | nfe | projector | strict S&C (x′) | **≥ 0.95** | ≥ 0.97, same rule | rule | ms/step |
| :-- | :-- | --: | :-- | --: | --: | --: | :-- | --: |
| tilt | MeanFM | 1 | per-step | 0/10 | **10/10** | 1/10 | c | 27.4 |
| tilt | MeanFM | 2 | per-step | 0/10 | **10/10** | 0/10 | c | 36.4 |
| tilt | MeanFM | 3 | per-step | 0/10 | **10/10** | 10/10 | c | 93.6 |
| tilt | MeanFM | 3 | endpoint | 0/10 | **10/10** | 7/10 | c | 73.5 |
| tilt | CI-MeanFM | 1 | per-step | 0/10 | **10/10** | 1/10 | c | 27.9 |
| tilt | CI-MeanFM | 2 | per-step | 0/10 | **10/10** | 0/10 | c | 36.1 |
| tilt | CI-MeanFM | 3 | per-step | 0/10 | **10/10** | 10/10 | c | 93.4 |
| tilt | CI-MeanFM | 3 | endpoint | 0/10 | **10/10** | 9/10 | c | 72.7 |
| tilt | FM | 1 | per-step | 0/10 | **10/10** | 2/10 | c | 27.6 |
| tilt | FM | 2 | per-step | 0/10 | **10/10** | 2/10 | c | 34.7 |
| tilt | FM | 3 | per-step | 0/10 | **10/10** | 10/10 | c | 79.2 |
| tilt | FM | 3 | endpoint | 0/10 | **10/10** | 7/10 | c | 71.6 |
| tilt | FM | 5 | per-step | 0/10 | **10/10** | 10/10 | c | 116.9 |
| tilt | FM | 5 | endpoint | 0/10 | **10/10** | 10/10 | c | 115.2 |
| tilt | FM | 20 | per-step | 0/10 | **10/10** | 10/10 | c | 360.4 |
| tilt | FM | 20 | endpoint | 0/10 | **10/10** | 10/10 | c | 416.9 |
| tilt | Diffusion ★ | 20 | per-step | 8/10 | **10/10** | 10/10 | c | 666.3 |
| hump | MeanFM | 1 | per-step | 0/10 | **0/10** | 0/10 | c | 26.4 |
| hump | MeanFM | 2 | per-step | 0/10 | **0/10** | 0/10 | r | 35.0 |
| hump | MeanFM | 3 | per-step | 0/10 | **10/10** | 10/10 | c | 91.1 |
| hump | MeanFM | 3 | endpoint | 0/10 | **10/10** | 8/10 | c | 72.1 |
| hump | CI-MeanFM | 1 | per-step | 0/10 | **0/10** | 0/10 | t | 26.6 |
| hump | CI-MeanFM | 2 | per-step | 0/10 | **0/10** | 0/10 | r | 35.4 |
| hump | CI-MeanFM | 3 | per-step | 0/10 | **10/10** | 10/10 | c | 89.2 |
| hump | CI-MeanFM | 3 | endpoint | 0/10 | **10/10** | 10/10 | c | 71.9 |
| hump | FM | 1 | per-step | 0/10 | **0/10** | 0/10 | c | 26.2 |
| hump | FM | 2 | per-step | 0/10 | **0/10** | 0/10 | r | 34.5 |
| hump | FM | 3 | per-step | 0/10 | **10/10** | 10/10 | c | 77.1 |
| hump | FM | 3 | endpoint | 0/10 | **10/10** | 10/10 | c | 71.9 |
| hump | FM | 5 | per-step | 0/10 | **10/10** | 10/10 | c | 114.1 |
| hump | FM | 5 | endpoint | 0/10 | **10/10** | 10/10 | c | 112.4 |
| hump | FM | 20 | per-step | 0/10 | **10/10** | 10/10 | c | 355.2 |
| hump | FM | 20 | endpoint | 8/10 | **10/10** | 10/10 | single | 291.0 |
| hump | Diffusion ★ | 20 | per-step | 10/10 | **10/10** | 10/10 | r | 654.2 |

**Reading.**
- At ≥ 0.95 every projected configuration reaches 10/10, except per-step projection at nfe ≤ 2 on the hump, where the
  flights stall on the roof and do not succeed.
- The frontier is therefore the cheapest configuration: nfe=1 per-step on the tilt, nfe=3 endpoint on the hump. That puts
  the flow models ahead of the baseline by 24× (tilt) and 9× (hump) in time per action.
- **The three flow models tie** (within 2 % in time per action at the same count). MeanFM and CI-MeanFM do not dominate FM.
- **It is a knife-edge.** Violating MeanFM/CI-MeanFM flights are 0.80–0.99 violation-free (median 0.971). At 0.97 the
  tilt's nfe 1–2 cells fall to at most 4/10 at their best rule (0–2/10 with the rule chosen at 0.95).
- **The floor.** Unprojected flights are about 66 % (tilt) and 87 % (hump) violation-free. A bar below about 0.87 would
  count unprojected hump flights as satisfying, so the defensible range is about 0.90–0.96.
- **What 0.95 forgives:** up to about 14 steps (≈ 0.4 s) inside the constraint per flight, at up to 10 cm (tilt,
  nfe 1–2; §2 of the DA of record).
- **A bar on the configuration instead** (S&C rate ≥ 0.9 to enter the frontier) only removes points: nothing enters on the
  tilt, and only the baseline on the hump.

---
Claude (Opus 5.5, Claude Code, U19 corridor chat) · 2026-09-24 · computed locally with python3.14 (numpy, yaml); nothing run on
the cluster; no thesis file edited.
