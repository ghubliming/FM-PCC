# Open questions — blocking or shaping the structure

Initial phase. These need an answer from the supervisor/advisor (or a decision
by the author) before prose drafting starts, because each one moves chapters.

## Blocking (decide before writing Chapter 4)

1. ~~**Title, EN + DE.**~~ **FIXED (2026-09-02):**
   **"Flow Matching Predictive Control with Constraints"** — no hyphens, no
   subtitle, do not paraphrase. Still open: the German rendering for the
   registration form (draft in the bone: *Flow-Matching-basierte prädiktive
   Regelung unter Nebenbedingungen*; German orthography forces hyphens in the
   compound, so the English "no dashes" rule cannot carry over literally).

   *Accepted limitation.* The title reads as flow-matching-only, while the work
   also covers visual conditioning, UAV embodiment, and further generative
   engines (MeanFlow, α-Flow, diffusion). This is a known, deliberate choice —
   the name stays. It is handled in the text, not by renaming: the bone adds
   `sec:intro:scope` ("Scope and Terminology"), which states once that *flow
   matching* names the deterministic ODE-transport **family** rather than a
   single engine, and that the study spans two observation modalities and two
   embodiments. Every later chapter can then rely on that sentence instead of
   re-justifying its own breadth.
2. **Scope: which generations are in the thesis?** The repo has Gen0–Gen16 plus
   a proposed unified rebuild. A Master's thesis cannot carry all of them.
   Proposed inclusion, to be confirmed:
   - **In:** state-based FM vs. diffusion DPCC (Gen0/Gen3v*), MeanFlow (Gen3v6),
     α-Flow (Gen3v7), HardFlow constraint arm (Gen12), visual aligning (Gen14),
     UAV (Gen15).
   - **Out / one-paragraph mention at most:** iMF (Gen3v4/Gen13 — refuted, but
     the *refutation* is a legitimate negative result worth a subsection in
     `sec:disc:negative`), the Gen13 HardFlow-base branch, GEN_X rebuild.
   - **Undecided:** Gen16 visual-avoiding (code complete, unverified on
     hardware) and Gen9 — in only if results land in time.
3. **Research-question set.** Four RQs are drafted in `sec:intro:questions`.
   They determine the Results sectioning one-to-one; changing them later
   reshuffles Chapter 6.
4. **Notation convention** (see the collision warning in
   `NOTES_tum_formatting_rules.md`). Needed before any method text exists.

## Shaping (decide before Chapter 6)

5. **Which baseline variant is *the* Target?** The convention used in the data
   analysis is: the best projection variant of diffusion DPCC at K=20 / aw=10.
   The thesis must state and justify this pinning once, in `sec:setup:baselines`,
   and then never renegotiate it per table.
6. **Definition of "better".** Pareto dominance at equal success and constraint
   satisfaction — strictly fewer function evaluations *and* lower wall-clock.
   Placed in `sec:setup:metrics:pareto`. Everything else is a trade-off, and
   must be written as one.
7. **Backbone confounding.** The strong, defensible claim is the
   *architecture-matched* one (our U-Net vs. the baseline U-Net at comparable
   parameter count). Transformer-backbone wins are confounded by capacity and
   must be reported as secondary, with parameters in the table. Every results
   table therefore needs backbone + parameter-count columns — build them into
   the table templates now, not after the fact.
8. **Degenerate configurations.** Some low-step HardFlow settings run no
   constraint math at all. Those rows must be marked and excluded from claims
   (`sec:res:constraints:degenerate`), not silently averaged.
   *Context worth keeping:* the **published** HardFlow config (`N=10`, activate on
   the second half) sits at `n_gen = 4` — inside the admissible regime. The
   degeneracy is a property of the convention at low budget, **not** a defect of
   their experiment. Write it that way; see `sec:method:hardflow`.
8b. **Arm C is not comparable to HardFlow's published table.** Their setup is
   **H16 / 8 actions per plan / IPOPT / fitted linear dynamics**; ours is
   **H8 / 1 action per plan / SLSQP / Euler**. Because we replan every step, the
   sampler+projector runs ~8× more often per environment step, so **their
   per-replan timing and our per-step timing must never share a table.**
   Recorded as `tab:hardflow-setup` in `Working_Space/v2`.
9. **UAV timing.** The 33 Hz / budget-ms figure is a data-rate and
   cluster-latency artefact, not a real-time target. Do not present it as a
   pass/fail criterion anywhere in Chapter 6.
10. **German abstract** — required, or is the English one sufficient? Confirm.
11. **One-sided vs. two-sided print.** Affects `BCOR` and the class options.

12. **Visual conditioning: which arm is the method?** Two mechanisms exist —
    *concatenated conditioning* (the shipped default, the source of every reported visual number)
    and *affine/FiLM conditioning* (opt-in, zero-initialised). The v2 draft presents the first as
    the method and the second as an ablation. Both are written as equations in
    `sec:method:backbone`. Changing this reshapes that section and every visual table.
    **Do not decide it from the flag names** — the default's flag is called `film_mode='v1'` and it
    is not FiLM. See `NOTES_naming_and_rebuild.md` §2.
13. **The artefact→thesis name translation table for `app:repro`.** Every log, checkpoint path and
    DA row uses the old tokens (`filmv1`, `dpcc-c`, `diffuser`). If the thesis renames in prose
    without shipping the table, no reader can match a row to a checkpoint.
    Blocking for `app:repro`, not for the chapters.

## Process

14. Register the thesis (Informatics Infopoint or RCI, programme-dependent);
    final-submission deadline is the 15th of the month.
15. 3–4 page exposé at the start — largely derivable from Chapters 1 and 4 of
    this bone.
16. Schedule the initial topic presentation (5 + 5 min) at I6 Defense Day.
