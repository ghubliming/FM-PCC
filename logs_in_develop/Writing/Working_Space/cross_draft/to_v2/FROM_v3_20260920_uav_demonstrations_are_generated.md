# v3 → v2 · 2026-09-20 · the quadrotor demonstrations are **generated and flown**, not human

**One paragraph to add, if Ch 1–4 does not already say it.** Check first: if §4 already states that the
quadrotor data is synthetic, do nothing.

## The fact

There is no human teleoperation anywhere in the quadrotor data. For each scene a reference path is laid
out **analytically** per passage class, and every episode is then **flown in MuJoCo by the same cascaded
geometric controller the evaluation uses**, under the same physics and the same time step. What the
dataset records is the *measured state of that flight*, not the reference. An episode is kept only if the
flight stayed airborne and inside its contact budget — which is why 25 of the 500 pillar episodes were
rejected.

So each demonstration is a trajectory the plant has been **shown to track**. That is all the word
*expert* means on this benchmark, and it is worth saying once in Ch 1–4, because a reader who has just
been told D3IL supplies human demonstrations will carry that assumption into the quadrotor chapter.

Sources: `uav_expert_data_collect/generator.py` (`_build_traj_and_init`, `SCENE_MAX_CONTACT_FRACTION`,
`Z_FLOOR_MARGIN`), `trajectories.py`, `dataset_writer.py`.

## What v3 did

v3.46 added the paragraph to §5.4.3 and put the same point in the caption of `fig:expert-uav`, which
draws the **reference** paths rather than the flights that tracked them — stated in the caption, because
a constraint crossed by the reference is crossed by construction rather than by tracking error.

## Contrast worth keeping straight

D3IL-avoiding and D3IL-aligning use D3IL's **human** demonstrations. The quadrotor scenes do not. If
Ch 1–4 has a sentence that covers all three environments with one description of the data, it needs the
split.
