# DA — `corridor_ball` first results: the gate passes, and the scene overshoots

*Gen15 · U11 · 2026-09-11. Source batch: **`temp/1009/batch_uav_20260911_202307_TEMP`**
(`DA_UAV_v1`, 1350 units, 0 failed). Jobs **25599–25604 → 25605–25610**, 3 engines × 2 tiers,
seed 6, n = 10, geo `corridor_hgb`.
Changelog: [`CHANGELOG_20260910_corridor_ball_and_geo_variant_override.md`](CHANGELOG_20260910_corridor_ball_and_geo_variant_override.md).*

## 0. TL;DR

1. ✅ **The gate passes decisively.** The unprojected plan goes from **0.00 violations / S&C 1.000**
   on `corridor_hg` to **32.8–34.7 violations / S&C 0.000** on `corridor_ball`, on all six arms.
   The ball binds, `collision_free_completed` collapses 1.000 → 0.000. §2
2. 🔴 **But nothing solves it: S&C = 0.000 in all 34 cells.** No projector — DPCC or HardFlow, at
   K=2 or K=5, on any engine — gets a single rollout both successful *and* constraint-clean. §3
3. **The scene has jumped from constraint-trivial straight to constraint-infeasible.**
   `corridor_hg` could not rank because everything passed; `corridor_ball` cannot rank because
   nothing does. It skipped the discriminating middle. §5
4. ✅ **It does deliver one axis `corridor_hg` never had: projector effort.** Violation *reduction*
   is measurable and it ranks — **DPCC at K=5 ≈ −2 to −4 violations · DPCC at K=2 makes it worse on
   af/mf · HardFlow does essentially nothing (−0.4 to +0.2).** §4

---

## 1. Provenance and gates

| | |
|---|---|
| geo | `corridor_ball` → `corridor_hgb_bounds+dynamics+geo_bounds+halfspace+obstacles`, **obs=5** (4 wall-end caps + the virtual ball) |
| ball | centre `[0.0, 0.0, 0.75]`, r = 0.35, 3-D (`['x','y','z']`), **virtual** — no geom in `scene_corridor.xml` |
| tiers | **A** K=2, 4 variants, HardFlow degenerate and blocked · **B** K=5, 7 variants |
| engines | af (`AlphaFlowODE_9D_as1_ae0.2_bbunet`, 3.97 M) · mf (`MeanFlowODE_9D_dp0.5_bbunet`) · fm |
| seed · n | 6 · **10** on every one of the 33 cells |

**Gates green:** `n_cb_tripped = 0`, `cb_sentinel = 0`, `hf_degenerate = 0`, and
**`hf_n_genuine = 2`** on the Tier-B HardFlow rows — genuine and citable at K=5.

⚠️ `track_err` is **absent** from the metric set for this geo, so the tracking axis used elsewhere
in the campaign is unavailable here. Single seed, n = 10.

---

## 2. ✅ The gate — the ball binds

`diffuser`, the unprojected plan, same checkpoints, only the constraint set changed:

| arm | geo | S&C | `n_violations` | `total_violations` | `collision_free` |
|---|---|---|---|---|---|
| af K=2 | `_hg` | **1.000** | **0.00** | 0.00 | **1.000** |
| af K=2 | **ball** | **0.000** | **34.60** | 5.923 | **0.000** |
| mf K=2 | `_hg` | 1.000 | 0.00 | 0.00 | 1.000 |
| mf K=2 | **ball** | 0.000 | **34.70** | 5.940 | 0.000 |
| fm K=2 | `_hg` | 1.000 | 0.00 | 0.00 | 1.000 |
| fm K=2 | **ball** | 0.000 | **32.80** | 5.605 | 0.000 |
| af K=5 | **ball** | 0.000 | 34.40 | 5.871 | 0.000 |
| mf K=5 | **ball** | 0.000 | 34.60 | 5.876 | 0.000 |
| fm K=5 | **ball** | 0.000 | 33.40 | 5.693 | 0.000 |

The check the CHANGELOG made the whole exercise conditional on — *"the `diffuser` row must now
report `n_violations > 0`"* — **passes on all six arms.** U11's geometry does what it was built to do.

Two corroborations: the eval's Fix_12 gate now warns that the **expert route itself** penetrates the
planning set at **24/200, 30/200, 18/200** samples (homotopies L/C/R), and `goal_reached` stays
**1.000** with `phys_safe` **1.000** everywhere — the drone completes the traverse and flies straight
through the ball, which is exactly what a *virtual* obstacle should do.

---

## 3. 🔴 …and nothing avoids it

**S&C = 0.000 in every one of the 34 cells.** `collision_free_completed` is 0.000 in 32 of them.

| arm | `diffuser` | `dpcc-t-geo_free` | `dpcc-t` | `dpcc-t-tightened` | `hardflow_sls` | `hardflow_sls-t` | `hf-t-geo_free` |
|---|---|---|---|---|---|---|---|
| af K=2 | 34.60 | 35.20 | 36.30 | 34.60 | — | — | — |
| mf K=2 | 34.70 | 35.60 | 36.20 | 35.10 | — | — | — |
| fm K=2 | 32.80 | 29.40 | 29.40 | **26.50** | — | — | — |
| af K=5 | 34.40 | 36.70 | **30.60** | 31.80 | 34.60 | 34.50 | 36.10 |
| mf K=5 | 34.60 | 36.80 | **31.50** | 32.20 | 34.20 | 34.60 | 36.60 |
| fm K=5 | 33.40 | 32.90 | **31.30** | 31.90 | 33.40 | 33.00 | 33.10 |

*(`n_violations`, mean of 10. Lower is better; `diffuser` is the no-projection reference.)*

The **only** cells with any collision-free rollouts at all are **fm K=2**: `dpcc-t-geo_free` and
`dpcc-t` at `collision_free` **0.100**, `dpcc-t-tightened` at **0.200** — 2 of 10. Even there S&C is
0.000, so **no rollout was ever both successful and clean**: `n_success` falls to 0.800 on exactly
those rows, i.e. the runs that cleared the ball are not the runs that reached the goal.

---

## 4. ✅ What *is* rankable: projector effort

Δ violations against each arm's own `diffuser` (negative = the projector removed violations):

| arm | `geo_free` | `dpcc-t` | `dpcc-t-tightened` | `hardflow_sls` | `hardflow_sls-t` | `hf-t-geo_free` |
|---|---|---|---|---|---|---|
| af K=2 | +0.60 | **+1.70** | 0.00 | — | — | — |
| mf K=2 | +0.90 | **+1.50** | +0.40 | — | — | — |
| fm K=2 | −3.40 | −3.40 | **−6.30** | — | — | — |
| af K=5 | +2.30 | **−3.80** | −2.60 | +0.20 | +0.10 | +1.70 |
| mf K=5 | +2.20 | **−3.10** | −2.40 | −0.40 | 0.00 | +2.00 |
| fm K=5 | −0.50 | **−2.10** | −1.50 | 0.00 | −0.40 | −0.30 |

Four readings:

**(a) The `-geo_free` negative control behaves exactly as designed.** At K=5 it is *worse* than no
projection at all (+2.2 to +2.3) on af and mf. Dropping the geometry term makes the ball invisible
to the projector, so it perturbs the plan without accounting for the obstacle. The control validates
the instrument.

**(b) Budget is what lets the projector engage.** At K=2, DPCC *increases* violations on af (+1.70)
and mf (+1.50); at K=5 it removes 2–4 (−3.80, −3.10, −2.10). With only two ODE steps the projector
has too little trajectory to reshape.

**(c) 🔴 HardFlow does essentially nothing here.** Across all three engines its violation counts sit
within **−0.4 to +0.2** of the unprojected plan — indistinguishable. At K=5 with `n_genuine = 2` the
arm is genuine and citable, and it still does not engage the obstacle, while DPCC on the same rows
removes 2–4. This is the first scene in the campaign where **HardFlow is clearly outperformed by the
DPCC projector on constraint satisfaction**, and it inverts the `pillars` picture (DA_20260910_T3),
where HardFlow held S&C parity at 8–21× lower cost.

**(d) The effort is expensive.** `dpcc-t` costs `proj_ms` **138–146 ms at K=2** and
**1289–1637 ms at K=5** — roughly 10× more work to buy a ~10 % violation reduction. HardFlow's
`proj_ms` is 66–86 ms (plain) and 278–378 ms (`-t`), i.e. far cheaper, but it buys nothing.

---

## 5. 🔴 Verdict: the scene overshot

| scene | regime | why it cannot rank |
|---|---|---|
| `corridor_hg` | constraint-**trivial** | 0 violations with or without the projector; S&C ≡ success |
| **`corridor_ball`** | constraint-**infeasible** | S&C 0/34 cells; `collision_free` 0.000 in 32 of 34 |

`corridor_ball` jumped from one degenerate regime straight to the other without passing through the
discriminating middle. As a *ranking* scene it is no more usable than its predecessor — and for the
same reason `s_curve` is not (CLOSURE_20260910 §3).

**Why it is unsolvable, by construction.** The CHANGELOG computed the escape band as
z ∈ **[1.41, 1.80]** — 0.39 m of headroom — because a lateral bypass is impossible (walls give
\|y\| ≤ 0.14, the ball demands \|y\| ≥ 0.66). But the policy is trained at z ~ **U(0.90, 1.30)**, and
the DPCC projector works over a **horizon-8** window: it cannot execute a sustained ≥ 0.3 m climb
inside its constraint-satisfaction step. The plan is asked to leave an envelope it has never seen,
via a manoeuvre the projector has no horizon to build. Fix_12's warning that the *expert route
itself* penetrates at 18–30/200 samples said the same thing before any rollout ran.

### 5.1 The fix — implemented as U12

Keep the ball giant and **lower it**, so the escape band lands inside the trained altitude envelope
rather than above it. Escape altitude is `z_c + r + 0.31`:

| `z_c` | r | escape band | vs trained U(0.90, 1.30) |
|---|---|---|---|
| 0.75 *(current)* | 0.35 | z ≥ **1.41** | **entirely above** — unreachable |
| **0.60** | 0.35 | z ≥ **1.26** | at the top of the envelope — a ~0.16 m climb from mean cruise |
| 0.55 | 0.35 | z ≥ **1.21** | inside the envelope — demanding but routine |

**Superseded — the actual fix was the ceiling, not the radius.** See
[`../U12/CHANGELOG_20260911_corridor_ball_v2_on_trajectory.md`](../U12/CHANGELOG_20260911_corridor_ball_v2_on_trajectory.md):
`corridor_ball_v2` places a **small** ball (r = 0.12) **on the measured raw trajectory**
(z_c = 1.13, from `phys_min_z` == `phys_final_z` on 100 rollouts) and raises the synthetic ceiling
1.80 → 2.80, opening a **0.93 m** slot where U11 had 0.08 m.

---

---

## 6. 🔴 Root cause — the constraint geometry, confirmed

§5 attributed the failure to the horizon/altitude argument. That was right in direction but wrong in
magnitude: **the escape slot is 0.08 m, not the 0.39 m the U11 changelog computed**, and the scene
was never reachable by anything.

### 6.1 The corrected arithmetic

The changelog took the escape band as z ∈ [1.41, **1.80**], using the raw `workspace_bounds`. But the
same margin that inflates the obstacle **shrinks the workspace box** — `eval_mix_uav.py:1256`:

```python
lb = ws_lb + margin        ub = ws_ub - margin        # margin = r_drone = 0.31
```

and the sphere is built as `radius + margin` (`eval_mix_uav.py:1316`). So:

| quantity | value |
|---|---|
| planner keep-out radius | 0.35 + 0.31 = **0.66 m** (1.32 m across) |
| corridor width | 0.90 m — **the keep-out is wider than the corridor** |
| centre lateral band | \|y\| ≤ 0.45 − 0.31 = **0.140 m** |
| centre vertical band | z ∈ [0.30+0.31, 1.80−0.31] = **[0.61, 1.49]**, 0.88 m tall |
| ball blocks (at y=0) | \|z − 0.75\| < 0.660 → z ≥ **1.410** |
| **escape slot** | **z ∈ [1.410, 1.490] = 0.080 m** (0.095 m at the walls) |

The drone centre must thread an **8 cm** vertical slot, starting from a trained cruise of
z ~ U(0.90, 1.30), while pinned laterally to ±0.14 m.

### 6.2 Two independent confirmations

**(a) U7's own feasibility test.** U7 judged scenes by channel half-width against measured tracking
error. `corridor_ball` is worse than anything U7 declared infeasible:

| scene · route | half-width | best `track_err` | verdict |
|---|---|---|---|
| pillars · outer | 0.060 | 0.336 | 5.6× too tight |
| pillars · centre | 0.150 | 0.336 | 2.2× too tight |
| s_curve · corridor | 0.120 | 0.302 | 2.5× too tight |
| **corridor_ball** | **0.040** | ~0.30 | **7.5× too tight** |

**(b) The violation counts match a straight flight through the ball.** If the drone simply holds its
trained altitude and never climbs, it violates for the sphere's chord at that height:

| cruise z | chord at y=0 | predicted violating steps |
|---|---|---|
| 0.90 | 1.29 m | ~58 |
| 1.10 | 1.12 m | ~51 |
| 1.30 | 0.73 m | ~33 |

**Observed `diffuser` violations: 32.8 – 34.7** across all six arms — the bracket, and closest to the
top of the trained band. The aircraft flies straight through at cruise altitude. It never attempts
the climb, because there is no reachable target to climb to.

### 6.3 Why no projector could have fixed it

`dpcc-t` at K=5 removes 2–4 violations of ~34 (§4) — a nudge, not a climb. That is what a
horizon-8 projector does when the feasible set is an 8 cm slot 0.3 m above the current trajectory:
it reduces penetration locally and converges to the least-violating iterate, because the manoeuvre
that would actually satisfy the constraint does not fit in its window. The Fix_12 gate said the same
thing before any rollout ran — **the expert route itself** penetrates at 18–30/200 samples.

### 6.4 Verdict

**The geometry is the whole cause.** It is not a policy failure, not a projector failure, and not a
tracker failure — the constraint set admits a solution so thin that no closed-loop system with
~0.30 m tracking error can occupy it. S&C 0/34 was determined at config-write time.

🔴 **The U11 changelog's §2 "THE NUMBERS" block is wrong** (escape band stated as 0.39 m with 24 cm
of headroom) and should be corrected to the table in §6.1.

**Consequence for the design.** Shrinking the ball does not rescue it: the centre has only 0.88 m of
vertical and 0.28 m of lateral freedom, so any ball large enough to block the trained band leaves a
slot narrower than the tracking error. The one honest lever is the **ceiling** — `scene_corridor.xml`
has a floor and two 1.5 m walls and **no ceiling geom**, so `workspace_bounds.ub[2] = 1.80` is a
synthetic number, exactly the class U7 permits changing (it is how pillars' y-bound went ±1.5 →
±2.5). At `ub[2] = 2.60` the slot becomes 2.29 − 1.41 = **0.88 m**, half-width 0.44 m against a
0.30 m tracking error — feasible with margin, ball unchanged. If that is done, the corridor
halfspaces should also gain a `z_active` bound: they currently model 1.5 m walls as infinitely tall.

---

## 7. Discussion — can the PID parameters be tuned at all?

*The controller is **not** implicated in this scene: §6 shows the geometry alone explains S&C 0/34,
and a tracker cannot steer around a **virtual** obstacle the simulator does not contain. What
follows is a parameter audit only — what is exposed, what is hardcoded, and what would have to
change — recorded here because the question recurs.*

### 7.1 Three PID behaviours are already selectable, with no code change

`pid_stopgo` is not a gain setting. It is a choice about the velocity feedforward handed to the PID
each FM step (`eval_mix_uav.py:1587-1598`):

| `UAV_MIX_CONTROLLER` | `v_des` | behaviour |
|---|---|---|
| `pid` | `action / dt_fm` | continuous, timing-derived (E7 default) |
| **`pid_stopgo`** *(every run in this campaign)* | **`0`** | brakes to zero every FM step |
| `pid_const_v` | `unit(action) · v_des_magnitude` | constant speed, timing-free |
| `mjpc` | ignored | MJX predictive-sampling tracker |

All four pass the U10 validator and are results-path keys, so switching between them is a per-job
env var — **already tunable today, at zero cost**.

### 7.2 The gains exist and are parameterised — but hardcoded at eval time

`CascadedPID`, `uav_naive_test/flight_controller.py:61-70`:

```
Kp_pos   = [ 4.0,  4.0,  8.0]      Kd_pos   = [3.0, 3.0, 4.0]
Kp_att   = [70.0, 70.0,  4.0]      Kp_omega = [2.5, 2.5, 1.0]
u_min = 0.0     u_max = max(2·u_hover, 6.0)          # thrust clipped, line 132
```

Three presets exist (`uav_expert_data_collect/generator.py:72-77`):

| variant | `kp_scale` | `kd_scale` |
|---|---|---|
| `pid_default` | 1.0 | 1.0 |
| `pid_high_gain` | 1.2 | 1.0 |
| `pid_low_gain` | 0.8 | 0.9 |

🔴 **`eval_mix_uav.py:1482` hardcodes `gen._make_pid(model, 'pid_default')`** — no config key, no env
override. At eval time the gains are **effectively fixed**; the presets are reached only by the
expert-data collector (`generator.py:291`). Exposing them is a small change mirroring U10: an env
var, a validator, and the value echoed into the results path so arms cannot pool.

### 7.3 🔴 The presets are position-only

`_make_pid` scales **only `Kp_pos` and `Kd_pos`** (`generator.py:114-119`). The attitude loop
(`Kp_att`, `Kp_omega`) and the thrust clip are untouched by all three variants — so the presets
cannot address an attitude-domain failure, and a ±20 % position-gain change is the only tuning
currently reachable from a config.

### 7.4 Summary

| question | answer |
|---|---|
| Are the PID parameters fixed? | **No** — they are parameterised, but hardcoded to `pid_default` at eval |
| Can anything be varied today? | **Yes** — the feedforward mode, via `UAV_MIX_CONTROLLER`, no code change |
| What is *not* reachable? | the gains themselves, and the entire attitude loop + thrust clip |
| What would expose them? | one env var + validator mirroring U10, plus an attitude-gain scale that does not currently exist |

---

## 8. What this licenses

**Supported.** The `corridor_ball` geometry loads and binds (obs=5, `diffuser` 0.00 → 32.8–34.7
violations, `collision_free` 1.000 → 0.000, all six arms). No configuration achieves S&C > 0 in 34
cells. The `-geo_free` negative control is worse than no projection at K=5, as designed. DPCC
removes 2–4 violations at K=5 and *adds* violations at K=2 on af/mf. HardFlow is indistinguishable
from no projection on this obstacle while costing 66–378 ms of projection.

**Not supported.** Any engine ranking — S&C is floored, and violation counts sit within ~2 of each
other across af/mf/fm. Any multi-seed claim (seed 6, n = 10). Any tracking-quality statement
(`track_err` unavailable for this geo). Any claim that HardFlow is *generally* weaker than DPCC —
this is one scene, one obstacle type, and it contradicts `pillars` K=5.

**Next.**
1. **Decide whether to keep the scene.** As built it cannot rank. §5.1 gives the one-line config
   change (`z_c` 0.75 → 0.55–0.60) that would move it into the usable middle; it costs one re-run of
   the same six jobs, which took ~0.5–2 h each.
2. **The HardFlow result in §4(c) is worth a second look regardless** — a genuine arm
   (`n_genuine = 2`) that does not engage a 3-D sphere at all, while DPCC does, is a mechanism
   question about HardFlow's guidance on this constraint type, not a scene artefact.
3. Do **not** spend seeds on `corridor_ball` until §5.1 is resolved.
