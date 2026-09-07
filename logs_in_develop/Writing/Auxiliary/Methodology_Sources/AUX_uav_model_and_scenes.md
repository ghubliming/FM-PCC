# AUX — the UAV model and the four scenes

**Thesis home:** `sec:setup:tasks`, `sec:method:uav` · **TARGET** §5.2, §5.3
**Primary dev logs:** `Gen11/Epoch1_UAV_model/` · `Gen11/Epoch2_UAV_mujoco_run/` · `Gen11/Epoch3_uav_in_env/`

---

## 1. What the thesis must say

The UAV embodiment is **not a new physics model**. It is the Skydio X2 quadrotor from
`mujoco_menagerie`, brought in verbatim, with the MJPC task patch applied, and wrapped in four
static obstacle scenes built for this work. Saying this plainly is what makes the embodiment-transfer
claim (RQ4) credible: the *dynamics* are third-party and validated; only the *scenes*, the
*controller wrapper* and the *data* are ours.

Three points carry the section:

1. **Provenance and the "copy, don't generate" policy.** Every byte of the physics XML is `cp` or
   `patch` output from a known upstream. This was an explicit decision, on the grounds that
   generated XML risks silent physics errors — wrong inertia tensors, wrong actuator gear ratios,
   wrong mesh scales — that would corrupt every downstream training run without ever raising an
   error. Worth one sentence in `app:repro`; it is a methodology strength.
2. **The scene wrapper is transparent.** Epoch 3 re-ran an Epoch 2 tracking task inside
   `scene_empty` and got **bit-for-bit identical RMS** (0.029114 m, job 21028). MuJoCo walls are
   passive rigid bodies with contact but no aerodynamic effect, so adding geometry around the drone
   does not change its dynamics. This licenses treating the scene as a backdrop and attributing every
   cross-scene difference to geometry rather than to physics.
3. **Scenes are a difficulty ladder, not a set of variations.** `empty` → `corridor` → `s_curve` →
   `pillars` increases the number of binding constraints and the number of distinct homotopy classes.
   `empty` is deliberately unconstrained and serves as the raw-generator denominator.

---

## 2. Parameters of record

| item | value | source |
|---|---|---|
| base model | Skydio X2, `mujoco_menagerie/skydio_x2/x2.xml`, verbatim copy → `quadrotor.xml` | `Epoch1/METHODOLOGY.md` |
| task variant | `quadrotor.xml` + MJPC's `quadrotor.xml.patch` → **`quadrotor_modified.xml`** (the file every later epoch includes) | `Epoch1/METHODOLOGY.md` |
| patch content | adds `quat="0 0 0 1"` (level, nose-forward spawn); removes the MJPC-only sensor block and hover keyframe | `Epoch1/METHODOLOGY.md` |
| actuation | 4 rotors (MuJoCo actuators), `data.ctrl[:4]` | `Epoch4` generator |
| scene files | `d3il/environments/.../scenes/scene_*.xml`, each `<include>`-ing `quadrotor_modified.xml` | `Epoch3/METHODOLOGY.md` |
| obstacle metadata | `uav_expert_data_collect/generator.py` → `SCENE_OBSTACLES` | `Epoch3/METHODOLOGY.md` |

### Scene geometry

| scene | contents | key dimensions | homotopy routes |
|---|---|---|---|
| `empty` | floor + skybox | — | free-space baseline; **no constraints applied** |
| `corridor` | 2 parallel box walls | y = ±0.5, inner faces y = ±0.45 → **0.9 m clear** | L / C / R |
| `s_curve` | 2 offset corridor segments + diagonal gap (two chicanes, 4 wall segments) | seg 1 y ≈ −0.8; seg 2 y ≈ +0.8 | single (`default`) |
| `pillars` | 6 cylinders, **r = 0.12 m** | x ∈ {−2, 0, +2}; column A at y = −0.6, B at y = +0.6 | (L,L,L) / (L,R,L) / (R,L,R) / (R,R,R) |

*(The corridor row is quoted as `y = ±0.5` walls in Epoch 3 and "2 flat walls at y = ±0.5" in Epoch 8
— consistent. The `s_curve` segment count differs between the two logs (2 vs 4); resolve against the
XML before it goes in a table. See Holes.)*

---

## 3. A known asset-path trap worth one footnote

MuJoCo resolves `<compiler assetdir="assets"/>` **relative to the top-level XML's directory**, not
the including file's. When `scene_empty.xml` includes `quadrotor/quadrotor_modified.xml`, MuJoCo
looked for the X2 mesh under `scenes/assets/` and crashed with
`ValueError: file not found: .../scenes/assets/X2_lowpoly.obj`. This is the kind of detail that
belongs in `app:repro`, not the main text, but it will bite anyone reproducing the setup.

---

## 4. Holes — fill before writing

- [ ] **X2 physical parameters are not tabulated anywhere in our logs** — mass, arm length, inertia
      tensor, rotor thrust coefficient, control range. They live only inside the Menagerie XML. The
      thesis needs a parameter table; read it out of `quadrotor_modified.xml` and cite Menagerie.
- [ ] **`s_curve` geometry is stated two ways** (2 segments vs 4 wall segments). Settle from the XML.
- [ ] **Licence for `mujoco_menagerie` and `mujoco_mpc`** — needed for `app:repro` (§5.3). Both are
      Apache-2.0 upstream; verify and cite the exact version/commit pulled.
- [ ] **Simulation timestep and integrator** for the UAV scenes are not recorded in the Gen11 logs
      (the 100 Hz figure is the *control* rate, see `AUX_uav_control_stack.md`).
- [ ] **Scene selection rationale** — why `pillars` became the headline scene is argued in the DA
      corpus (Headline 8), not here. Cross-reference rather than re-derive.
