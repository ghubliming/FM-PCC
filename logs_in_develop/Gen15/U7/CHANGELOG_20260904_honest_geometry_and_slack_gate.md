# Gen15 U7 — Honest scene geometry, a slack-aware feasibility gate, and the geo_tag collision fix

**Date:** 2026-09-04
**Motivation:** [`../DA/DA_20260903_fix16_AB_mf_pillars.md`](../DA/DA_20260903_fix16_AB_mf_pillars.md) **Part II §II.3**
**Files touched:** `config/uav_projection.yaml`, `mix_uav_test/eval_mix_uav.py`,
`Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh`
**Status:** written and statically checked here; ⚠️ **not executed — run on cluster.**

---

## 0. TL;DR

1. 🔴 **The scenes leave less room around their own expert routes than the policy's tracking
   error.** `corridor` L/R is the extreme: the trained channels sit at `y = ±0.12` and the
   planning band is **exactly** `[−0.12, +0.12]` — **0.000 m of slack**. `pillars` outer has
   0.060 m, `s_curve` 0.120 m; measured `track_err_mean` is 0.30–0.49 m. Success+constraints is
   therefore bounded near zero **before any engine runs**.
2. 🔴 **The Fix_12 feasibility gate reported all of this as "OK"** — it tests penetration > 0,
   and a route lying exactly on the boundary penetrates by 0.000. Now it reports **slack in
   metres** and warns when slack < the policy's tracking error.
3. 🔴 **`geo_tag` never encoded the geo entry name**, despite its comment claiming it did. Two
   entries for one scene with the same `constraint_types` — exactly what a geometry A/B is —
   produced the **same output folder** and the second silently overwrote the first. Opt-in
   `geo_tag_suffix` fixes it with **zero change to any existing tag**.
4. 🟢 **Three new geo entries** `pillars_hg` / `corridor_hg` / `s_curve_hg` ("honest geometry"):
   only *synthetic* quantities changed; every physical number is untouched.
5. 🟢 **Planner pad and body radius are now separate objects** (`planning_inflation` vs
   `inflation`), so loosening the projector's tube never loosens the yardstick that scores
   collisions.
6. 🟡 **Expect this to help `pillars` and not to rescue `corridor`/`s_curve`.** The remaining
   deficit there is physical (0.90 m gap, 0.62 m drone) and needs a scene + dataset change.

---

## 1. The measurement that prompted it

Free channel around the expert route vs. the policy's measured tracking error
(`track_err_mean`, best-behaved variant per scene, from `temp/0309/.../per_rollout_detail.csv`):

| scene · route | channel half-width | best `track_err` | ratio |
|---|---|---|---|
| `corridor` · L and R (`y=±0.12`) | 🔴 **0.000 m** | — | infeasible by construction |
| `pillars` · outer (expert `\|y\|=1.11`) | 🔴 0.060 m | 0.336 m | 5.6× too tight |
| `pillars` · centre (expert `y=0`) | 🔴 0.150 m | 0.336 m | 2.2× too tight |
| `s_curve` · corridor (expert `y=−0.8`) | 🔴 0.120 m | 0.302 m | 2.5× too tight |
| `corridor` · C (`y=0`) | 0.120 m | — | 2.5× too tight |

Derivations, from the scene XMLs and `config/uav_projection.yaml`, at the shipped
`margin = r_drone 0.31 + margin_base 0.02 = 0.33`:

```
corridor  walls y=±0.5, half-thickness 0.05 → inner faces ∓0.45
          band = [-0.45+0.33, 0.45-0.33] = [-0.12, +0.12]
          trajectories.CORRIDOR_CHANNELS = {L: -0.12, C: 0.0, R: +0.12}   → L/R slack = 0.000
pillars   rows y=±0.6, r=0.12 → forbidden disc 0.45; box y ±1.5 → 1.17
          outer channel = [1.05, 1.17], expert at 1.11  → 0.06 either side
          centre channel = [-0.15, +0.15]
s_curve   inner faces -0.35 / -1.25 → band [-0.92, -0.68], expert -0.80  → 0.12
```

**This is why `pillars` scores 0 / 2876 on success+constraints for every engine, every K and
both Fix_16 arms.** It is not an engine result.

---

## 2. What changed

### 2.1 `config/uav_projection.yaml` — three new entries, originals untouched

Six scene-geometry entries are now **defined** (plus `empty_no_constraint`); **three are
active**. The originals are kept verbatim for rollback and A/B — every DA number up to
2026-09-03 was produced under them.

```yaml
active_geo_variants: ['empty_no_constraint', 'corridor_hg', 'pillars_hg', 's_curve_hg']
```

| change | scene | rationale |
|---|---|---|
| workspace box `y ±1.5 → ±2.5`, `x ±3.6 → ±4.0` | `pillars_hg` | `scene_pillars.xml` has **no walls** — floor is a 10×10 plane. The box was invented, and at ±1.5 it was a *second* binding surface 0.06 m outside the expert route. At ±2.5 only the pillars bind. |
| `x_active: [-2.0, 2.0]` on both wall halfspaces | `corridor_hg` | `scene_corridor.xml`: `pos="0 ±0.5 0.75" size="2.0 0.05 0.75"` — the walls span `x ∈ [−2, 2]`, but the halfspaces were infinite lines constraining the approach and departure where **no wall exists**. The mechanism already existed (`s_curve` uses it); `corridor` never did. |
| `planning_inflation: {r_drone: 0.31, margin_base: 0.0}` | all three | drops the arbitrary 2 cm pad **from the planner only** |
| box `x ±3.6 → ±4.0`, `y ±1.6 → ±1.8` | `s_curve_hg` | path spans ±3.2; the box was never the intended constraint there |
| obstacle centres, radii, wall faces, altitude slab | — | 🔴 **unchanged — these are physics** |

Resulting slack after the change:

| scene · route | before | after | still short of 0.30 m? |
|---|---|---|---|
| `corridor` · L/R | 0.000 | **0.020** | 🔴 yes |
| `corridor` · C | 0.120 | **0.140** | 🔴 yes |
| `pillars` · outer | 0.060 | **0.080** (pillar side) / 1.080 (box side) | 🟡 pillar side capped by the dataset |
| `pillars` · centre | 0.150 | **0.170** | 🔴 yes |
| `s_curve` · corridor | 0.120 | **0.140** | 🔴 yes |

🔴 **Stated up front so the runs are not over-read.** The `pillars` outer route clears the
pillar by 0.080 m **by construction** — `trajectories.PILLAR_SAFETY = 0.08`, chosen when the
data was generated on the stated assumption that 8 cm was *"sufficient for PID tracking
error"*. It is not: the measured error is 4× that. No projector config can widen it; that needs
a scene + dataset change. Likewise `corridor`/`s_curve`: a 0.90 m physical gap and a 0.62 m
drone leave 0.140 m best case. **The point of these entries is to remove every *artificial*
contribution to the deficit**, so what remains is attributable to the policy and the scene.

### 2.2 `mix_uav_test/eval_mix_uav.py` — four changes

| # | where | what |
|---|---|---|
| 1 | `_apply_geo_entry` | carries two new **optional** per-entry keys, `geo_tag_suffix` and `planning_inflation`. Absent from every pre-U7 entry → behaviour there is byte-identical. |
| 2 | `_apply_geo_entry` (geo_tag) | 🔴 **collision fix.** The comment claimed geo_tag encoded the "resolved geo entry name"; the code never did. `corridor` and `corridor_hg` declare the same `constraint_types`, so both hashed to `corridor_bounds+dynamics+geo_bounds+halfspace+obstacles` and the second run would have silently overwritten the first. Now `f'{scene}{suffix}_{families}'`. |
| 3 | `setup_dpcc_projector`, `plot_geo_constraints` | read `planning_inflation or inflation`. `_exec_constraint_violations` (line 683) still reads `inflation` — **deliberately unchanged**, so the collision yardstick stays physical. |
| 4 | `_warn_expert_route_infeasibility` | now **slack-aware** (§2.3). |

### 2.3 The gate that should have caught this

Before: `_exec_constraint_violations(expert_route)` → pass/fail on penetration > 0. A route
lying exactly on the boundary penetrates by 0.000 and printed *"expert route OK"*.

After: bisect the extra offset at which the route first violates, and report it.

```
[ eval ] corridor feasibility check: homotopy=L expert route NEAR-ZERO SLACK under planning
         margin 0.31 m — slack 0.020 m (need >= 0.30 m to absorb the policy tracking error)
[ eval ] WARNING corridor homotopy=L: the expert route clears the PLANNING set by only
         0.020 m. Any rollout whose tracking error exceeds that violates the constraints even
         when it flies the trained route perfectly in expectation — success+constraints is
         bounded near 0 for reasons that have nothing to do with the engine. …
```

`FMPCC_GEO_SLACK_PROBE_M` sets the bar (default **0.30** = the best measured `track_err_mean`
median across scenes); `0` restores the old bare pass/fail. Print-only — it never blocks a run.

### 2.4 `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh`

`--time` is now `UAV_EVAL_HOURS`-overridable (default unchanged: `8 h × n_seeds`, clamped to the
24 h cap). Jobs 25318/25321 (`pillars` K5, 1 seed) both hit the wall with 5 of 17 variants: the
default was silently too small once Fix_16 made rollouts ~10× longer.

---

## 3. Verification done here

| check | result |
|---|---|
| `ast.parse(eval_mix_uav.py)` | 🟢 clean |
| `bash -n eval_k_sweep.sh` | 🟢 clean |
| 7 geo entries, unique `name` | 🟢 |
| 7 geo_tags, all unique (simulated against the real entry definitions) | 🟢 |
| pre-U7 geo_tags byte-identical to before | 🟢 |
| `corridor` vs `corridor_hg` would collide **without** the suffix | 🟢 confirmed — this was a live bug |
| slack arithmetic reproduced from the XMLs | 🟢 §1 |
| `_exec_constraint_violations` still reads physical `inflation` | 🟢 line 683 untouched |

⚠️ **Not verified here** — no Python packages in this container: yaml parses (`pyyaml` absent),
the projector builds with `planning_inflation`, the bisection terminates on real routes, and the
`_hg` entries resolve at eval time. **All of that is cluster-side.**

---

## 4. The runs — 3 scenes × 3 engines, matched K

**K = 2, for every arm.** Reasoning: the sbatch header's rule is *matched budget or nothing*,
and K2 is the largest budget that reliably **completes** — both K2 arms finished 10/10 variants
while both K5 arms died at the wall (`dpcc-c` `proj_ms` is 63 ms at K2 against **1751 ms** at
K5, a 28× step). K<3 also blocks the 7 `hardflow_*` variants via the degeneracy guard, which is
correct: those rows carry no HardFlow claim anyway.

```bash
# ── Gen15 U7 flagship: honest geometry, matched K=2, seed 6 ────────────────────
export FMPCC_SAFE_EPS_MODE=scaled          # Fix_16 (already the default; explicit for the log)
export FMPCC_UAV_EVAL_TAG=u7hg             # eval-folder tag, on top of the _hg geo_tag

for SCENE in pillars corridor s_curve; do
  for ENGINE in fm mf; do
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh \
        "$ENGINE" "$SCENE" "6" "2"
  done
done

# af needs its U6 knobs — see the blocker below before running these three
for SCENE in pillars corridor s_curve; do
  UAV_MIX_BONE_AF=unet UAV_MIX_EPOCH=latest \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh \
      af "$SCENE" "6" "2"
done
```

🔴 **Blocker to check first (af only).** U6 changed the default af bone from `sit` (9.4 M) to
`unet` (4.0 M, parameter-matched to fm/mf). Each bone has its own checkpoint tree, so **an af
eval fails on a missing checkpoint until the U-Net af arm has been trained**. On the cluster:

```bash
ls logs/UAV_MIX/uav-pillars/mix_uav_af/*bbunet*/6/   # exists → run as above
```
If it is missing, either train the af U-Net arm first, or fall back to
`UAV_MIX_BONE_AF=sit` and 🔴 label those rows **confounded** (objective, backbone and parameter
count all move together — see the standing benchmark rule).

**Follow-up, not part of the 9** — K5 needs the full 17 variants and ~27 h, over the cap:

```bash
UAV_EVAL_HOURS=24 ./Slurm_Codes/submit.sh \
    Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf pillars "6" "5"
```
Split the variant list rather than assuming 24 h is enough.

### What to read first when they land

1. The **feasibility banner** — `slack 0.xxx m` per scene/homotopy. It is now the headline
   number of every job and it should read `0.080` (pillars outer), `0.020` (corridor L/R),
   `0.140` (s_curve). If it does not, the `_hg` entry did not resolve.
2. `pillars` `n_violations` **per live step** (not raw — diverged rollouts stop accumulating).
   The box was the second binding surface; if removing it changes nothing, the pillars are the
   whole story.
3. `corridor` approach/departure violations at `|x| > 2`. Those were **artificial** before the
   `x_active` fix and should now be gone.
4. Success + constraints. 🔴 **Do not expect it to move on `corridor`/`s_curve`** — §2.1 says
   why. If it moves on `pillars`, the box was doing real damage.

---

## 5. Rollback

One line, no code revert:

```yaml
active_geo_variants: ['empty_no_constraint', 'corridor', 'pillars', 's_curve']
```

The `_hg` entries stay defined but inactive, and the pre-U7 output folders are byte-identical,
so old and new results coexist without overwriting.

🔴 **Never activate a scene's `<scene>` and `<scene>_hg` entries in the same job.** Fix_6
multi-match runs both, doubling the job — and K5 already hits the wall. They write to different
folders, so run them as separate submissions.
