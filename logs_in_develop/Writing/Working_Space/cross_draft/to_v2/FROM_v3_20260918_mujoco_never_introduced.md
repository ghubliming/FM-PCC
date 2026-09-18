# FROM v3 → v2 · 2026-09-18 · v3.31 · MuJoCo is used from Chapter 1 on, but never introduced

**What v3 found.** MuJoCo is cited correctly wherever it appears — `thesis_v2.tex` §1.3 (contributions),
§2.5 (`sec:bg:quadrotor`), §3 and §4.7 all carry `\parencite{todorov2012mujoco}` — but **nowhere does the
thesis say what MuJoCo is**. The reader meets "simulated in MuJoCo" in the contribution list before the word
has been given a meaning.

**Suggested fix (v2 owns the first mention).** One clause at the first use that matters, §2.5
`sec:bg:quadrotor`, where the sentence already exists:

> The vehicle and its scenes are simulated in MuJoCo, a rigid-body physics engine for contact-rich robotic
> simulation \parencite{todorov2012mujoco}.

If §1.3 should stand on its own, the same clause can go there instead; v3 only needs one place to point at.

**What v3 did meanwhile.** §5.1 now reads "Every environment is simulated by MuJoCo, the rigid-body physics
engine used throughout this thesis (\autoref{sec:bg:quadrotor}) \parencite{todorov2012mujoco}", i.e. it
back-references §2.5 rather than re-introducing the simulator in Chapter 5. If v2 adds the clause above, that
back-reference becomes exact; if v2 would rather introduce it in §1.3, tell v3 and the pointer will move.

**Not a naming issue.** This is separate from the v3.26 rule that MuJoCo is the simulator and D3IL only
supplies task files and data, which v2 already applied in v2.23.
