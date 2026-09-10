# Future work — outlook material, held outside the thesis text

**Created:** 2026-09-10 · **Type:** outlook / idea capture · **Status:** 🟡 **none of this is built, none of it is a result**
**Thesis home (eventually):** `sec:conc:future` — currently an empty stub at `Working_Space/v2/thesis_v2.tex:2016`.
Touches `sec:disc:limitations`, `sec:method:constraints`, `sec:method:visual`.

---

## Why this folder exists

The conclusion chapter owes a Future Work section, and the material for it is not results — it is
**where this architecture could go and what it would be used for in the field**. Keeping it here
rather than in `v2/thesis_v2.tex` means the ideas are recorded while they are fresh, without any of
them leaking into the thesis as if they were done.

> 🔴 **Nothing in this folder is evidence.** No number here supports a claim. Every file separates
> **what the code does today** (cited, file-level) from **what would have to be built** (labelled as
> such). If a future reader cannot tell those two apart in a paragraph, that paragraph is wrong.

The ideas are the author's, recorded on the date in each file's header. They are written up rather
than transcribed: each one is checked against the code and against the project's own measurements,
and the checks are part of the note — including where they cut the idea down.

## The files

| file | the idea | one-line honest status |
| :-- | :-- | :-- |
| [`OUTLOOK_20260910_perception_to_constraints.md`](OUTLOOK_20260910_perception_to_constraints.md) | A perception front-end that recognises real obstacles and emits **constraint geometry** (planes, spheres, cylinders) into the DPCC/FM-PCC projector — from RGB, depth, RGB-D or LiDAR, optionally with SLAM so the map persists. | Nothing built. The projector's input format is fixed and known, and the per-step rebuild hook already exists; the perception stack does not. A **cheap ground-truth-noise ablation** is proposed as the first real step. |
| [`OUTLOOK_20260910_zero_shot_constraint_retasking.md`](OUTLOOK_20260910_zero_shot_constraint_retasking.md) | Train once (lab / CAD-derived sim), then meet **new constraints at deployment with no retraining** — a fire closes a gallery, a keep-out volume is declared, the mission changes, the same policy still flies. | The mechanism is real and verified in code: constraints never enter training. The **limit** is also already measured — the honest-geometry slack finding bounds how far re-tasking can go. The experiment that would turn it into a claim is specified. |

## The two ideas are one story

They are the two halves of the same deployment argument, and the conclusion should present them in
this order:

1. **Constraints are a runtime input, not a learned behaviour** → new restrictions need no retraining.
   *(note 2 — the property)*
2. **In the field, someone has to produce those constraints from sensors** → the perception front-end.
   *(note 1 — the missing input path)*

Note 2 is the claim the architecture already earns. Note 1 is what it would take to feed it outside
a simulator. Written the other way round, the perception idea reads as an unmotivated add-on.

## Standing rules for anything added here

- **Date and attribute each idea** in the header; ideas age, and a stale one must be visible as stale.
- **Cite the code** for every "this already exists" claim, `file:line`. An uncited claim is a guess.
- **State the kill condition.** An outlook item that cannot fail is not an experiment, it is a wish.
- **Do not renegotiate results here.** Findings live in `Data_Analysis/DA_Result_Curated_MD/`; this
  folder may cite them, never revise them.
