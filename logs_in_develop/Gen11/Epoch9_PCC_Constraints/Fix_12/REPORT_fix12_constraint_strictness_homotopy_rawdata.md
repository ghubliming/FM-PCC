# Fix_12 — Research report: (Q1) constraint strictness, (Q2) the LLL/LRL log label, (Q3) where the raw trajectory data lives

**Date:** 2026-07-08. Research-only (no code changed). Triggered by cluster inspection of
`.../uav-pillars/plans/flow_matching_v3_uav/H8_.../K20_mpc4_pid_stopgo_T0.5/6/pillars_bounds+dynamics+geo_bounds+halfspace+obstacles/dpcc-r/diagnostics/rollout_0_stats.json`.

Sources: `config/uav_projection.yaml`, `FM_v3_uav_test/eval_fm_uav.py`
(`setup_dpcc_projector`, `_exec_constraint_violations`, `rollout_one`, `_run_variant`),
`FM_v3_uav_test/eval_artifacts.py`, `uav_expert_data_collect/generator.py` +
`trajectories.py`, scene XMLs under `d3il/.../quadrotor/scenes/`.

---

## Q1 — Yes, the constraint set is mathematically near-infeasible for pillars (and very tight for s_curve / corridor)

### The one number that causes it

`config/uav_projection.yaml`:

```yaml
inflation: {r_drone: 0.36, margin_base: 0.05}
```

`setup_dpcc_projector` (eval_fm_uav.py:549-552) offsets **every** spatial surface by
`margin = 0.36 + 0.05 = 0.41 m` (+0.025 more for `-tightened`). 0.36 is the *worst-case
diagonal* rotor-tip radius (`sqrt(0.14²+0.18²)+0.13`); the drone's actual lateral (y)
half-width is **0.31 m** (`trajectories.py: PILLAR_ROTOR_REACH = 0.31`) and it never yaws
(`yaw=0` in all paths), so 0.41 already over-states the body by ~0.10 m, before the second
problem below.

### Pillars: the feasible set excludes every trained route

Raw geometry (scene_pillars.xml + yaml): 6 pillars r=0.12 at y=±0.6, x∈{-2,0,2};
"envelope" halfspaces at y=±1.2; workspace box x∈[-3,3], y∈[-1.5,1.5], z∈[0.30,1.60].

With margin 0.41:

| Surface | Raw | Inflated | Consequence |
|---|---|---|---|
| Pillar keep-out radius | 0.12 | **0.53** | Pillar pair at same x: free centre band is only \|y\| ≤ 0.6−0.53 = **0.07 → a 14 cm corridor** |
| Envelope halfspace | \|y\| ≤ 1.2 | \|y\| ≤ **0.79** | Outer channels: inflated pillar edge reaches 0.6+0.53 = **1.13 > 0.79 → L/R channels completely closed** |
| Workspace box x | [-3, 3] | [-2.59, 2.59] | `pillar_path` starts at **x = -3.2 and ends at x = +3.2** (`trajectories.py:67`) — start AND goal are outside the box, even outside the *raw* box |
| Workspace box z | [0.30, 1.60] | [0.71, 1.19] | Altitude is drawn `z ~ U(0.90, 1.30)` (`generator.py:124`) — **~27 % of episodes start above the inflated ceiling** |

Meanwhile the expert/training routes fly the outer channels at **y = ±1.11**
(`trajectories.py: _Y_L/_Y_R = ±(0.6+0.12+0.31+0.08)`). Under the projector, y=±1.11
violates *both* the inflated envelope (0.79) *and* the inflated pillar keep-out
(|1.11−0.6| = 0.51 < 0.53).

**Net result: not a single training-distribution route is feasible.** The only feasible
passage is a 14 cm centre channel the model was never trained on (pillars homotopy classes
are L/R combinations only — there is no centre route in the data). The projector therefore
fights the policy at every replan → "almost impossible" is literally correct. This also
matches the observation "halfspace alone is fine": the envelope halfspaces by themselves
leave |y| ≤ 0.79 (plenty of room, including the natural centre route); it is the
**obstacle inflation** (0.12 → 0.53) that closes everything.

### s_curve: the corner balls nearly close the crossover

- Corridor band (yaml walls y∈{-1.3,-0.3}): inflated feasible band = **[-0.89, -0.71] —
  0.18 m wide** for an expert route centred at -0.8 with ±0.04 jitter → **±0.05 m slack**,
  smaller than typical PID tracking error.
- The wall-end "cap" balls at (-0.5,-0.3) and (0.5,0.3) have r=0.05 but inflate to
  **0.46**. During the gap crossing (the Z-route through x=0), passing the corner at
  y=-0.3 requires x ≥ -0.04, and the corner at y=+0.3 requires x ≤ +0.04 → the crossover
  is squeezed through an **~8 cm gate** in a physically 1.0 m open gap.
- Start x=-3.2 vs inflated box lb -3.5+0.41 = -3.09 → start infeasible here too.

Note also a small inconsistency: the s_curve yaml uses wall **centrelines** (-1.3/-0.3)
while the corridor yaml uses **inner faces** (±0.45); the physical inner faces of s_curve
are at -1.25/-0.35 (box half-thickness 0.05 in the XML).

### corridor (user didn't ask, but it's the same disease, worse)

Inner faces ±0.45, inflated → feasible **|y| ≤ 0.04** (4 cm!). Expert L/R channels at
y=±0.12 are infeasible; even C with its ±0.03 jitter is marginal. Physically the drone
(y-half-width 0.31) fits with |y| ≤ 0.14 — the projector is ~3.5× stricter than reality.
Start x=-2.8 vs inflated lb -2.59 → start infeasible again.

### It also poisons the *metrics*, not just the projection

`_exec_constraint_violations` (eval_fm_uav.py:248) checks the flown path against raw
geometry ⊕ `r_drone` (0.36) and is billed as "physical collision truth". But:

- The pillars **envelope halfspaces and the workspace box are synthetic** — there is no
  physical wall at y=±1.2, no ceiling at z=1.6, nothing at x=±3 (the XML contains only the
  6 pillars and the floor). A perfectly safe expert-style flight at y=1.11 is scored as a
  0.27 m/step "violation"; the mandatory start at x=-3.2 is a guaranteed 0.56 m violation
  on step 0 of **every** pillars rollout.
- Hence `constraint.collision_free` is structurally False and
  `success.strict_and_constraints ≡ 0` for pillars regardless of flight quality. The
  `rollout_*_stats.json` numbers you saw on the cluster are dominated by these phantom
  violations, not by real crashes (real contact truth is the separate `physical.*` axis).

### Recommended fixes (for a later Fix — not applied)

1. **`r_drone: 0.36 → 0.31`** (true y-reach; yaw is held 0) or even the per-axis honest
   value; keep `margin_base` small or 0. The yaml's own comment already flags this
   ("reduce toward the ~0.16 m body radius if pillar gaps prove too tight").
2. **Do not inflate synthetic surfaces.** Inflation exists so the *body* clears *physical*
   geometry; envelope halfspaces, wall-end/corner cap balls, and workspace boxes are
   bookkeeping constructs. Either exempt them from `margin` or size them post-inflation.
3. **Pillars specifically:** drop the y=±1.2 envelope halfspaces (the y=±1.5 workspace box
   already bounds the field). Check: with envelope removed and margin 0.36, the outer
   channel becomes [0.6+0.12+0.36, 1.5−0.36] = [1.08, 1.14] — the expert channel 1.11 is
   feasible again (with margin 0.31+0: [1.03, 1.19], comfortable).
4. **Fix the workspace boxes to contain start/goal:** pillars/s_curve paths span
   x∈[-3.2, 3.2] → box x should be ≥ ±(3.2+margin); z ub ≥ 1.30+margin (altitude draw max).
5. **s_curve corner balls:** keep r small (they only mark the wall-end edge) and exempt
   from inflation, or replace with short x_active halfspaces on the actual wall faces.
6. **Add a cheap feasibility self-check at eval start:** project the expert waypoints of
   each homotopy through the built constraint set and warn loudly if any are infeasible —
   this entire class of bug would have been caught on job step 1.

---

## Q2 — "LLL / LRL" is the expert-route homotopy label, and you are right: in the pillars eval it controls nothing

**What it is.** `uav_expert_data_collect/generator.py:65`:

```python
HOMOTOPY_CLASSES = {'pillars': ['(L,L,L)', '(L,R,L)', '(R,L,R)', '(R,R,R)'], ...}
```

L/R = which side of each of the 3 pillar pairs the **expert reference path** weaves
(channel centres y=±1.11). It shows up in the eval outputs because
`episode_id = f'{scene}_{homotopy}_{trial_seed}'` → `rollout_pillars_(L,L,L)_10000.log`
filenames, the `homotopy=` field in `eval_<variant>.log` / `results.json` / the behaviour
`.log` SUMMARY block, and the per-rollout colour in `<variant>.png`.

**What it does in eval.** `_run_variant` cycles `homotopy = homotopies[i % 4]` per trial
and passes it to `rollout_one`, which calls `gen._build_traj_and_init(scene, homotopy, rng)`
— but then uses the returned expert `traj_fn` **only** to compute the goal point
(`traj_fn(dur)`), the finish-line direction, `init_pos`, and `dur`. **The drone never
tracks `traj_fn`.** The FM policy is conditioned only on `obs = [p_des | p | v]`, and
`p_des` is self-integrated from the FM's own actions (`p_des += action`). There is no
homotopy input to the model at train or eval time.

**And for pillars specifically the label is fully inert:** `pillar_path` starts and ends
on the centreline (`ys = [0.0, …, 0.0]`, `xs = [-3.2, …, 3.2]`), so `init_pos = (-3.2, 0, z)`
and `goal = (3.2, 0, z)` are **identical for all four homotopy classes**. (Contrast
corridor, where the label *does* move start/goal to the L/C/R channel y.) So in pillars,
trial *i*'s "LLL" tag changes: nothing physical. It is a vestige of mirroring the expert
data-collection API, where the same argument genuinely selects the flown route.

**Your framing is exactly right:** the model was trained on all four routes and picks its
own mode at eval; you cannot command it, and the logged label is *not* the route flown.
Worse, it's actively misleading — a reader of `results.json` will assume rollout 1 flew
(L,R,L). Recommendation for a later fix: compute the **realized** homotopy class post-hoc
from the flown path (sign of y at x=-2, 0, +2 — trivial from `obs_all[:, 4]`) and log that
(e.g. `homotopy_commanded` vs `homotopy_flown`), or drop the label for pillars eval.

---

## Q3 — Where the raw trajectories are saved

Per (scene, seed, geo_tag, **projection variant**) output folder — the one your cluster
path points into, built in `_run_variant` (eval_fm_uav.py:1090-1097):

```
<scene_root>/plans/<model>/<eval_params>/<seed>/<geo_tag>/<variant>/
├── <variant>.npz                ← ★ ALL raw arrays, all rollouts of this variant
├── results.json                 summary + per-rollout metrics (heavy arrays stripped)
├── eval_<variant>.log           one line per rollout + summary
├── <variant>.png, all.png       2-D path overview
├── rollout_<scene>_<homotopy>_<trialseed>.log    real-time behaviour text log (per rollout)
└── diagnostics/
    ├── rollout_<i>_stats.json          metrics only — the file you inspected
    ├── rollout_<i>_mpc_foresight.svg   candidate-fan plot
    └── rollout_<i>.gif                 only when --record ≠ none
```

**Yes — everything is aggregated into the single `<variant>.npz`** (`diffuser.npz`,
`dpcc-r.npz`, …), written by `eval_artifacts.save_npz`. The raw-trajectory keys are
object arrays with one entry per rollout (load with `np.load(p, allow_pickle=True)`):

| npz key | shape per rollout | content |
|---|---|---|
| `obs_all` | (T, 9) | executed per-FM-step obs `[p_des(0:3) \| p(3:6) \| v(6:9)]` — **cols 3:6 are the actual flown position**, cols 0:3 the commanded setpoint |
| `act_all` | (T, 3) | executed FM action Δp_des per step |
| `sampled_trajectories_all` | list of T arrays, each (B, H, obs_dim) | **the full MPC candidate fan of every replan** — all B candidates' H-step foresight, every FM step |
| + flat metric vectors | (n_rollouts,) | `success_*`, `phys_*`, `constraint_*`, `goal_*` (Fix_10 group-prefixed schema) |

So "each MPC foresight" is indeed all in the npz (`sampled_trajectories_all`), and the
flown trajectory is `obs_all[:, 3:6]` at DATASET_HZ (33 Hz) resolution. The
`diagnostics/rollout_<i>_stats.json` files contain **no arrays** — `HEAVY_KEYS =
('obs_traj', 'act_traj', 'plans', 'frames')` are stripped before every JSON write.

**Visual-aligning comparison — your memory is correct.** Gen7's
`eval_fm_visual_aligning.py` writes, per rollout, `diagnostics/rollout_<r>_data.pkl`
(plus `_stats.json`, `_report.png`, `_mpc_foresight.png`, mp4/gif) and a per-variant
`results_seed_<s>.pkl` — a pkl-heavy per-rollout scheme inherited from Gen6V4. The UAV
eval deliberately did **not** port that; `eval_artifacts.py` restores the older
FMv3ODE "legacy npz schema" instead (its docstring says so). So on UAV there is **no
per-rollout pkl** — the npz is the single source of raw data. (Agreed the visual-aligning
artifact logic is chaotic — three overlapping conventions now exist across generations:
Gen6V4/Gen7 pkl-per-rollout, FMv3ODE npz, UAV npz+behaviour-log. Worth unifying in a later
fix, but out of scope here.)

---

## TL;DR

1. **Q1:** Confirmed and quantified. `margin = r_drone 0.36 + 0.05 = 0.41 m` inflates the
   0.12 m pillars to 0.53 m keep-outs, which (combined with the inflated ±1.2 envelope)
   closes both trained L/R channels and leaves only an untrained 14 cm centre corridor;
   s_curve's 5 cm corner balls become 46 cm gates squeezing the crossover to ~8 cm; the
   workspace boxes exclude the start/goal of every constrained scene. The violation
   *metrics* are also contaminated because synthetic surfaces (envelope, box) are scored
   as physical. Fix directions in §Q1.
2. **Q2:** `(L,L,L)`/`(L,R,L)`… is the expert-route homotopy label cycled per trial. In
   pillars eval it changes nothing physical (identical start/goal for all four classes;
   policy is unconditioned; the expert path is never tracked) — it is only a (misleading)
   metadata tag. Log the realized class computed from the flown path instead.
3. **Q3:** All raw data for a variant is in that variant's folder: full flown trajectory +
   every replan's candidate fan aggregated in `<variant>.npz`
   (`obs_all`/`act_all`/`sampled_trajectories_all`); JSONs are metrics-only; per-rollout
   behaviour `.log` text files sit next to the npz. UAV has no per-rollout pkl — that's a
   visual-aligning (Gen6V4/Gen7) convention the UAV eval intentionally replaced with the
   FMv3ODE npz schema.
