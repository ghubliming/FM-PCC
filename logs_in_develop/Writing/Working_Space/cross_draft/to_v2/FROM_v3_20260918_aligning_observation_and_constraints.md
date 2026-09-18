# TO v2 — §4.6.2 (alignment): two things the section implies but never states

**2026-09-18 · from v3.36.** Both surfaced while writing v3 §5.2.2, which now states them from the
setup side. Chapter 4 is where a reader meets the environment first, so the method side is yours.

## 1. The box pose is not observed, and the context is random

§4.6.2 says the state half of the observation is $(\vect{p}\sidx{des},\vect{p})\in\R^{6}$, commanded
and measured end-effector position. That is correct and it is the whole vector — but a reader who
has just read the obstacle-avoidance section will carry over the assumption that the object's pose is
in there somewhere, because in that task the scene is fixed and the goal is a line. Here it is not:

* the initial box pose **and** the target pose are drawn per episode — box in $x\in[0.4,0.6]$,
  $y\in[-0.25,-0.10]$ at any yaw in $\pm90^{\circ}$; target in $x\in[0.4,0.6]$, $y\in[0.20,0.35]$ at
  any yaw in $\pm90^{\circ}$ (`gym_aligning/envs/aligning.py:62--67`, `box_space` and `target_space`);
* so the two cameras are the **only** source of both. The policy has to find the box in the image
  before it can move toward it.

Worth one clause in §4.6.2, because it is what makes the visual conditioning load-bearing rather than
decorative. The check that this is the dataset actually trained on:
`fm_visual_aligning/datasets/sequence.py::ParityAligningDataset.OBS_DIM = 6` (the 9D visual
trajectory), selected at `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py:168--170`. The
23D `StateOnlyAligningDataset` in the same file *does* carry box and target pose — it is not used for
any result in this thesis, and naming it in passing would prevent the next reader from assuming it was.

## 2. "A workspace box on the planned end-effector position" is not the geometry evaluated

§4.6.2 describes the alignment constraint set as a workspace box plus a box on the action. The
evaluated geometry is `combined_5`, and it is a **halfspace across the far corner of the table plus a
circular keep-out region** between the box and its target, with the workspace bounds only the reachable
table area (`config/visual_aligning_eval.yaml`: `active_geo_variants: [combined_5]`). v3 §5.2.2 was
wrong about this until v3.33 and it cost a figure; the same sentence in Chapter 4 is still live. The
forms are then exactly the two D3IL-avoiding uses, which is a better story than the box anyway.

New in v3 that you may want to point at: `fig:expert-aligning` draws all 120 recorded contexts against
that set and finds the direct push from box to target crosses the keep-out region in 106 of them.
