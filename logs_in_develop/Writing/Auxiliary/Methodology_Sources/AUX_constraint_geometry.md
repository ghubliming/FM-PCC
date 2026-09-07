# AUX — constraint geometry: the constraint sets built beside the environments

**Thesis home:** `sec:method:constraints`, `sec:setup:tasks`, `sec:disc:threats` · **TARGET** §5.2, §6
**Primary dev logs:** `Gen11/Epoch9_PCC_Constraints/` (whole tree — `Plan/PLAN_E9_PCC_constraints.md`,
`Fix_5`, `Fix_12`, `U_8`, `U_13`, `notes/`) · `Gen15/U7/CHANGELOG_20260904_honest_geometry_and_slack_gate.md`
· `Gen15/DA/DA_20260906_U7_honest_geometry_first_results.md` · `config/uav_projection.yaml`

---

## 1. What the thesis must say

This is the most under-appreciated piece of apparatus in the project. **The projector is inherited
from DPCC; the constraint geometry it enforces is ours, per environment, and it had to be designed.**
A projector is only as meaningful as the feasible set it projects onto, so this section is what makes
every constraint number in the thesis interpretable.

Four claims carry it:

1. **No new solver machinery was needed.** `SafetyConstraints` (lower/upper bounds and inequalities),
   `ObstacleConstraints` (quadratic sphere terms) and `DynamicConstraints` (the Euler rows) already
   supported every form required, byte-identical to the DPCC engine. The work was **geometry design +
   config wiring + violation metrics**, not solver development. Say this — it keeps the projector a
   controlled variable across engines and embodiments.
2. **One full-stack constraint set per scene, activated by config.** Each scene carries its own
   complete geometry — box bounds, halfspace planes, sphere/cylinder obstacles, plus the dynamics
   rows — and `constraint_types` + `active_geo_variants` in the YAML choose what is switched on.
3. **`empty` is deliberately unconstrained.** It is the raw-generator denominator and the plumbing
   regression baseline, mirroring the visual-aligning `no_constraint` entry (`constraint_types: []`).
   An unconstrained control condition is a design choice, not an omission.
4. **The ablation variants are ablations, never results.** `model_free` (skip dynamics),
   `bounds_free` (skip action bounds), `geo_free` (skip `geo_bounds` + `halfspace`) and their
   compositions exist to attribute *which* constraint family costs what. Rows produced under them are
   tagged and **must never be ranked against legal arms** — they routinely produce the best raw
   distances precisely because they are not enforcing the constraints.

---

## 2. The constraint families

| family | form | where it comes from |
|---|---|---|
| `dynamics` | Euler consistency rows linking consecutive states | DPCC engine, unchanged |
| `bounds` | action box bounds (`Bounds(−5, 5)` in the SLSQP call) | DPCC engine |
| `geo_bounds` | 3-D workspace box — the volume the plan may occupy | **ours**, per scene |
| `halfspace` | plane inequalities; on `s_curve`, **switched per segment** | **ours**, per scene |
| `obstacles` | sphere/`sphere_outside` quadratic terms; cylinders for `pillars` | **ours**, per scene |

Per-scene activation (UAV):

| scene | `constraint_types` |
|---|---|
| `empty` | `[]` — **no constraints, explicitly marked** |
| `corridor` | `['dynamics','bounds','halfspace','obstacles']` |
| `pillars` | `['dynamics','bounds','halfspace','obstacles']` (+ 6 cylinders) |
| `s_curve` | `['dynamics','bounds','halfspace','obstacles']`, halfspaces switched per segment |

The design is explicitly templated on the Gen7 visual-aligning bone
(`setup_dpcc_projector`, `formulate_halfspace_constraints`, per-geometry config blocks, exec-time
violation metrics) — i.e. the **same DPCC lineage the whole repo follows**. That parallel is worth
one sentence: the two embodiments share not only the projector but the way its geometry is declared.

### The variant-toggle design (worth a footnote)

Ablations were first built as *separate geo entries per scene* (6 extra entries, each competing for
`active_geo_variants` slots and needing a multi-geo loop to run alongside the full-stack entry). That
was replaced by **variant-level toggles orthogonal to the geo entry** — a one-line name check per
family — because `model_free` had already proved the pattern. It is a small piece of design
hygiene, but it is why the variant names read the way they do in every table.

### `tightened` siblings

Every geometry-keeping variant has a `-tightened` sibling (a stricter margin). **This turns out to be
the single biggest lever in the avoiding results** — larger than the choice of projection arm; no
untightened row clears S&C 0.75 in the K∈{2,3,5} ladder. The thesis must define `tightened`
precisely and report it as an experimental factor, not as a formatting suffix.

---

## 3. 🔴 Honest geometry — the self-diagnosed benchmark defect

This is the part that belongs in **both** `sec:method` and `sec:disc:threats`, and it is a strength
if written as such: the project measured its own benchmark and found it broken.

**The finding (2026-09-04).** The UAV scenes leave less room around their own expert routes than the
policy's tracking error:

| scene | slack around the expert route | measured `track_err_mean` |
|---|---|---|
| `corridor` L/R | **0.000 m** — trained channels at `y = ±0.12`, planning band exactly `[−0.12, +0.12]` | 0.30–0.49 m |
| `pillars` outer | 0.060 m | 0.30–0.49 m |
| `s_curve` | 0.120 m | 0.30–0.49 m |

**Success-plus-constraints is therefore bounded near zero before any engine runs.** No ranking of
engines on these scenes can mean anything, which is exactly what the corpus observed (S&C 0/2876 on
`pillars`).

**Why it went unnoticed.** The existing feasibility gate tested *penetration > 0*. A route lying
exactly on the boundary penetrates by 0.000 and passed as "OK". The gate now reports **slack in
metres** and warns when slack is below the policy's tracking error. *A pass/fail gate that cannot
distinguish "safe" from "exactly on the edge" is a good methodological cautionary tale and should be
written as one.*

**Two fixes shipped with it, both worth naming:**
- **Planner pad and body radius are now separate objects** (`planning_inflation` vs `inflation`), so
  loosening the projector's tube no longer loosens the yardstick that scores collisions. Conflating
  the two is a subtle way to grade one's own homework.
- **`geo_tag` never encoded the geo entry name**, despite a comment claiming it did — so two entries
  for one scene with the same `constraint_types` (i.e. exactly a geometry A/B) wrote to the **same
  folder and silently overwrote each other**. Fixed by an opt-in `geo_tag_suffix`.

**Scope of the remedy.** The `*_hg` entries (`pillars_hg`, `corridor_hg`, `s_curve_hg`) change only
*synthetic* quantities — every physical number is untouched. This was expected to help `pillars` and
**not** to rescue `corridor`/`s_curve`, where the deficit is physical: a 0.90 m gap against a 0.62 m
drone. That honesty about what a fix can and cannot do should survive into the text.

---

## 4. Holes — fill before writing

- [ ] **The V_A constraint geometry is not documented here at all.** `combined_5`,
      `combined_5-tightened`, and the `dt*` sweep are used in every visual-aligning table but their
      actual definitions (how many halfspaces, what obstacles, what the tightening margin is) live
      only in `config/aligning-d3il-visual.py`. **This is the largest single hole in the methodology
      chapter** — the flagship environment's feasible set is undefined in prose.
- [ ] **Numeric definition of `tightened`** for both embodiments — the margin, in metres.
- [ ] **`dt0p25` / `dt0p5` / `dt2p0` / `dt4p0`** — what the `dt` sweep varies, and why
      `dpcc-c-dt4p0` reaches high safety by freezing the box (it must be named as disqualified).
- [ ] **Post-honest-geometry status.** `DA_20260906_U7_honest_geometry_first_results.md` exists; the
      `*_hg` K-sweep with all three engines has **not** been run. Until it is, UAV carries a
      methodology chapter, not a results chapter (TARGET §8 kill criterion 3).
- [ ] **The τ = 0.850 NLP non-convergence** — engine-independent, 24/24 arm-C items, same
      non-converged SLSQP solve at call #2. Suspected constraint-Jacobian conditioning. It is a
      property of *this* feasible set and belongs here once diagnosed.
