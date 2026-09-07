# v1 — the first written draft

**Created:** 2026-09-07 · **File:** [`thesis_v1.tex`](thesis_v1.tex)
**Governed by:** [`../TARGET_20260905_thesis_claim_ladder.md`](../TARGET_20260905_thesis_claim_ladder.md)
**Apparatus source:** [`../../Auxiliary/Methodology_Sources/`](../../Auxiliary/Methodology_Sources/README.md)

---

## What is filled, and to what density

| chapter | state |
|---|---|
| 1 Introduction | ✅ **filled** — abstract density |
| 2 Background | ✅ **filled** — abstract density |
| 3 Related Work | ✅ **filled** — abstract density |
| 4 Method | ✅ **filled** — abstract density |
| Abstract | 🟡 sketched; headline numbers left as a `\hole` |
| 5 Setup · 6 Results · 7 Discussion · 8 Conclusion | ⬜ **bone only, deliberately** |

**"Abstract density"** means: every section states *what it will argue and in what order*, in the
thesis's own voice, at roughly the length of a long abstract. It is not finished prose and it is not
a bullet outline. The intent is that writing the final text becomes expansion, not invention — the
argument, its order, and its commitments are already fixed.

Chapters 5–8 are untouched because **nothing in them is settled**. Filling them now would mean
writing claims the run queue has not closed. The queue lives in
`Data_Analysis/DA_Result_Curated_MD/NOTEBOOK_20260829_key_headlines.md`.

## Two drafting macros — remove before submission

```latex
\hole{...}     % red, inline: a decision or number that is not available yet
\srcnote{...}  % grey, small: where the material for this passage lives
```

---

## Structure changes vs. `Bone/thesis_bone_broad.tex`

The bone's eight chapters are kept. Four changes inside them:

| # | change | why |
|---|---|---|
| 1 | **`Scope and Terminology` stays in Chapter 1, before Background** | `K` vs. candidate fan, arms A/B/C, genuine steps, "tightened", and the Pareto definition of *better* are used in every later chapter. The thesis is unreadable if they arrive late. |
| 2 | **Method's `Constraint Enforcement` split into three sections** — per-step projection · in-loop trajectory optimisation · *When the Guided Sampler Is Not Guided* | The degeneracy condition is a **contribution**, not a caveat. As a subsection of a shared "constraints" section it reads as an implementation note; as its own section it reads as what it is. This is the single most substantive reorganisation. |
| 3 | **`Candidate Selection` promoted to its own Method section** | Goal C is a claim of the thesis, and the fan-of-one argument is *analytic* — it belongs in Method, discharged before Results, so the results chapter only handles the fan > 1 case. |
| 4 | **`Closed-Loop Deployment` covers both embodiments in one section** | The manipulator and aerial control chains are the *same* structure one layer apart. Writing them together is the substance of RQ4; writing them in separate chapters would make the transfer claim look like a coincidence. |

Unchanged on purpose: Background and Related Work stay separate (TUM I6 convention, and the
mechanism argument in `sec:bg:fewstep` needs room); `Starting Point: DPCC` stays *before* Problem
Formalisation, because the delta-from-inherited framing needs the reader to know what is inherited.

---

## Decisions this draft takes, that the reader should be able to challenge

1. **α-Flow is written as an ablation and a negative result, not as the top rung of the ladder.**
   TARGET §8 kill criterion 1 fired: with α provably live and the architecture matched, it ties
   MeanFlow at low budget, loses at high budget, and is better nowhere across two environments
   (`CLOSURE_20260907`, notebook Headline 10). Per that criterion the **title and RQ set are
   unchanged** and the engine claim the thesis defends is the average-velocity objective. If you
   want α-Flow removed entirely rather than demoted, `sec:method:engine` is the only place to cut —
   the rest of the draft does not depend on it.
2. **The engine claim is stated as one engine separating, not as a four-rung ladder.** `fm >
   diffusion` does not hold on the visual task — a 0.0055 m gap inside a ~0.4 m band. RQ1 is phrased
   as "does it order the engine family", which the evidence can answer honestly either way.
3. **The genuine-step condition is presented as a contribution and applied to our own past
   results.** The draft says explicitly that it invalidates several of this work's earlier
   measurements. That is the honest form and it is also the stronger one.
4. **Arm B and arm C are stated as *not* evaluation-matched** in Method rather than as a footnote
   in Results — arm C spends an extra network evaluation per guided step to form the endpoint
   estimate.
5. **The honest-geometry defect is framed as a methodology strength.** A benchmark defect found by
   one's own instrumentation, reported with the correction *and* with what the correction cannot fix.

---

## Open items before v2

- [ ] **Author, supervisor, advisor, submission date** — still `TODO` in the metadata block.
- [ ] **German title** needs confirming (`\getTitleGer`).
- [ ] **Outline section** (`sec:intro:outline`) is a `\hole` — write it last, as convention.
- [ ] **Abstract headline numbers** — blocked on the run queue, not on writing.
- [ ] **Bibliography** — no citations placed yet. Harvest from
      `aux_repo/PAPERS/Recommand_Paper/HF/reference.bib` per `NOTES_paper_map.md`.
- [ ] **The system-overview figure** (`sec:method:overview`) is referenced throughout and does not
      exist. It is the highest-value figure in the thesis; draw it early.
- [ ] **Notation table** — `\vect`/`\matr`/`\trans` macros are carried from the bone but the symbol
      set is not fixed. `NOTES_notation_decisions.md` does not exist yet.
- [ ] Chapters 5–8 stay closed until the run queue clears.
