# Gen15 U12 — `corridor_ball_v2`: a small ball *on* the trajectory, and the ceiling raised

**Supersedes U11's `corridor_ball`.** No code changes — one new geo variant in
`config/uav_projection.yaml`. The U11 machinery (`UAV_MIX_GEO_VARIANTS`) is reused unchanged.

## 0. TL;DR

U11's ball was **unreachable, not hard**: S&C 0/34 was decided at config-write time. The root cause
was not the radius — it was the **ceiling**. U12 makes the ball **small (r 0.35 → 0.12)** and places
it **on the measured raw trajectory (z_c = 1.13)**, and raises the synthetic ceiling
**1.80 → 2.80**, which opens a **0.93 m** escape slot where U11 had **0.08 m**.

## 1. What went wrong in U11

The margin that inflates an obstacle **also shrinks the workspace box**:

```
eval_mix_uav.py:1316   sphere radius := radius + margin        # margin = r_drone = 0.31
eval_mix_uav.py:1256   box          := [ws_lb + margin, ws_ub - margin]
```

U11's changelog used the raw `ws_ub = 1.80` and computed a 0.39 m escape band. The real centre band
is `[0.30+0.31, 1.80−0.31] = [0.61, 1.49]`, so the slot was **0.080 m**.

| | U11 claimed | U11 actual |
|---|---|---|
| keep-out radius | 0.66 m | 0.66 m ✓ |
| centre vertical band | [0.30, 1.80] | **[0.61, 1.49]** |
| escape slot | 0.39 m | **0.080 m** |
| vs ~0.30–0.50 m tracking error | "feasible" | **7.5× too tight** |

Confirmed independently: a drone that simply holds cruise altitude violates for the sphere's chord —
predicted 33–58 steps, **observed 32.8–34.7** on all six arms. It never attempted the climb because
there was nothing reachable to climb to. Full analysis:
[`../U11/DA_20260911_corridor_ball_first_results.md`](../U11/DA_20260911_corridor_ball_first_results.md) §6.

## 2. 🔴 Why a smaller ball alone could not have fixed it

With `ub_z = 1.80` the centre band is 0.88 m tall. Three requirements, and no `(z_c, r)` meets them:

| requirement | condition |
|---|---|
| block any of the trained band [0.90, 1.30] | `z_c + r ≥ 0.59` |
| block **all** of it | `z_c + r ≥ 0.99` |
| leave a slot ≥ 1× tracking error | `z_c + r ≤ 0.88` |
| ball centred **on** the flown path | `z_c ≈ 1.13`, so `z_c + r ≥ 1.13` |

The blocking and slot conditions do not overlap, and an on-path ball violates the slot condition at
**any** radius including zero. **The binding quantity is the ceiling, not the radius.**

## 3. The ball is on the trajectory — measured, not guessed

Over **100** corridor `diffuser` rollouts (`batch_uav_20260911_202307_TEMP`), `phys_min_z` equals
`phys_final_z` on **every** rollout — the drone holds a **constant altitude**:

| | value |
|---|---|
| flown altitude | z ∈ **[0.956, 1.236]**, mean **1.127** |
| lateral channels | `CORRIDOR_CHANNELS` L/C/R = −0.12 / **0.0** / +0.12 |

→ ball centre **[0.0, 0.0, 1.13]** — mid-corridor, on the centre channel, at the flown altitude.

## 4. The numbers

Drone planning radius 0.31; `ub_z = 2.80` → centre band z ∈ **[0.61, 2.49]**.

| quantity | value |
|---|---|
| ball | r = **0.12** → 0.24 m across, in a 0.90 m corridor |
| keep-out radius | 0.12 + 0.31 = **0.43 m** |
| blocked at y=0 | z ∈ **[0.70, 1.56]** — covers the flown band [0.96, 1.24] with 0.26 m margin each end |
| **escape slot** | z ∈ **[1.56, 2.49] = 0.93 m** (half-width 0.465) |
| required climb | 1.13 → 1.56 = **0.43 m** (= r + r_drone; irreducible for an on-path ball) |
| under the ball | [0.61, 0.70] = 0.09 m → not a route |

**The ball is deliberately small.** The drone's own 0.31 m inflation does the blocking: even r = 0
would block z ∈ [0.82, 1.44] and still cover the flown band. r = 0.12 adds margin without eating the
slot. U11's r = 0.35 bought no extra blocking and cost 0.46 m of headroom.

## 5. The ceiling change is synthetic — which is why it is allowed

`scene_corridor.xml` holds a floor plane and two walls (`pos="0 ±0.5 0.75" size="2.0 0.05 0.75"` →
walls span z ∈ [0, 1.5]) and **no ceiling geom**. `ub[2] = 1.80` was an invented number. Raising it
touches no physical geometry — the same class of edit U7 made when `pillars` went y ±1.5 → ±2.5.

**Known conservatism, left in deliberately:** the wall halfspaces carry no `z_active`, so the planner
treats the 1.5 m walls as infinitely tall and still pins \|y\| ≤ 0.14 above z = 1.5 where no wall
exists. That constrains **more** than physics, never less, so it cannot manufacture false
feasibility. Adding `z_active` is a separate change and is not made here.

## 6. The change

| file | change |
|---|---|
| `config/uav_projection.yaml` | new `geo_constraint_variants` entry **`corridor_ball_v2`**, `geo_tag_suffix: '_hgb2'`. Not in `active_geo_variants`. |

Everything else — walls, the four wall-end caps, `planning_inflation`, `constraint_types` — is
byte-identical to `corridor_hg`. Only `ub[2]` and the added ball differ. No code touched; selection
is via U11's `UAV_MIX_GEO_VARIANTS`.

`_hgb2` keeps the results in a third sibling folder, so `_hg` (trivial), `_hgb` (U11, unreachable)
and `_hgb2` can never pool in a DA.

## 7. Verification

* YAML structurally validated against the `corridor_hg` sibling — identical key names and indent
  levels, 6 list items (2 halfspaces + 4 caps + ball). *(No `yaml` module in this container; do one
  `safe_load` on the cluster.)*
* No Python or shell touched, so nothing to compile.
* 🔴 **Not run on the cluster.** First-run checks below.

### First-run checks

1. `[ U11 ] geo variants for 'corridor': ['corridor_ball_v2']`
2. `[ eval ] E9 geo 'corridor' ← variant 'corridor_ball_v2': … (bounds=True, hs=2, obs=5)`
3. a results path containing `corridor_hgb2_`
4. 🔴 **the two gates that decide the design:**
   * `diffuser` must report **`n_violations > 0`** — the ball must still bind (U11 passed this);
   * **a projected row must report `collision_free_completed > 0`** — at least one rollout routed
     around it. U11 scored 0.000 on 32 of 34 cells; if U12 does the same, the slot is still not
     reachable and the ceiling must go higher or the scene be abandoned.

---

## 8. Runs

### 8.1 Injection test — job 25654 → **25657** (2026-09-11 21:11 → 21:20, 9 min)

mf · corridor · K=2 · `corridor_ball_v2` · seed 6 · n=10 · `diffuser, dpcc-t`.
Driver: `Slurm_Codes/temp_bash/eval_20260911_u12_injection.sh`.

Provenance checks all green: `[ U11 ] geo variants for 'corridor': ['corridor_ball_v2']`,
`E9 geo … (bounds=True, hs=2, obs=5)`, results under `corridor_hgb2_`.

| gate | row | value | |
|---|---|---|---|
| **1** ball binds | `diffuser` | `n_violations` **27.10**, `collision_free_rate` 0.000 | ✅ **pass** |
| **2** something routes around | `dpcc-t` | `collision_free_rate` **0.000** | 🔴 **fail** |

`dpcc-t` moved the plan — `proj_ms` 146.6, steps 271.7 → 253.2 — but removed only 27.1 → 25.8
violations, and **`phys_min_z` == `phys_final_z` on all 10 rollouts** (0.955–1.236): the projector
never changed altitude by a centimetre. It optimised the only axis it had, arriving 18 steps earlier.

### 8.2 🔴 Suspected cause — the action bound, not the geometry

From the same log:

```
Fix_16 DEGENERATE actions[1]/[2]: constant in the expert data — no training signal
action_bounds=auto → lb=[1.24e-04 -2.20e-05 -2.20e-05] ub=[4.3886e-02 2.2000e-05 2.2000e-05]
```

The corridor expert flies a straight line at constant y **and** constant z, so Δy and Δz have zero
variance in training. `action_bounds='auto'` derives the projector's action cap from that range,
leaving ~**2.2e-05 m/step** in both axes — **8.7 mm** over a 396-step episode, against the ~0.43 m
detour the ball demands. Measured across scenes:

| scene | Δx | Δy | Δz |
|---|---|---|---|
| **corridor** | 4.4e-02 | **±2.2e-05** 🔴 | **±2.2e-05** 🔴 |
| pillars | 4.4e-02 | **±3.97e-02** ✅ | ±3.1e-05 🔴 |
| s_curve | 2.2e-02 | **±2.29e-02** ✅ | ±1.1e-05 🔴 |

**Δz is degenerate on every UAV scene** — every expert cruises at fixed altitude — so a *vertical*
detour is unreachable everywhere, and `corridor` additionally has no lateral authority. If this holds,
no obstacle requiring a detour can be solved on `corridor` at any geometry, and U11/U12's ball
placement was never the binding issue.

**Not yet confirmed.** `'dpcc-t-bounds_free'` was added to `projection_variants` to test it directly
(`bounds_free` skips only the action-magnitude family, `eval_mix_uav.py:1259`; geometry and dynamics
stay on). Probe driver: `Slurm_Codes/temp_bash/eval_20260911_u12_boundsfree_probe.sh`, ~12 min.
`collision_free > 0` confirms the cap is the blocker; `= 0` refutes the diagnosis.

### 8.3 Full wave — jobs 25663–25668 (2026-09-11)

Driver: `Slurm_Codes/temp_bash/eval_20260911_u12_full_wave.sh`. Same two-tier structure as U11's
25599–25604, geometry swapped to `corridor_ball_v2`.

| tier | K | variants | af | mf | fm |
|---|---|---|---|---|---|
| **A** PCC only | 2 | 4 | **25663** | **25664** | **25665** |
| **B** PCC + HF-SLSQP | 5 | 7 | **25666** | **25667** | **25668** |

All: seed 6, n=10, `u7hg` eval tag, `UAV_EVAL_HOURS=24`; af carries `BONE_AF=unet`,
`AF_ALPHA_END=0.2`, `EPOCH=latest`. Results land in `corridor_hgb2_`, a third sibling geo folder
alongside `_hg` and U11's `_hgb`, so a DA pairs all three on the `geo` axis.

⚠️ 25664 (mf K=2) repeats the injection test's configuration plus two variants; expect §8.1 to
reproduce there. The wave's new information is the other two engines and the Tier-B HardFlow rows.

**Awaiting results.**
