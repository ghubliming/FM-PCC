# DPCC lineage — the basis of the work

The thesis is **based on DPCC**. This is the single most important framing fact
for the writing, and it has consequences in three separate chapters.

## The chain

```
Diffuser (Janner et al.)            temporal U-Net trajectory diffusion
        │
        ▼
D3IL (ALR Karlsruhe)                Aligning / Avoiding environments + demos
        │
        ▼
DPCC (L4DC 2025, arXiv 2412.09342)  diffusion planner + constraint projection
        │                           upstream code: /workspaces/aux_repo/dpcc
        ▼
FM-PCC (this thesis)                deterministic flow-matching engine,
                                    few-step engines, alternative constraint
                                    arm, visual conditioning, UAV embodiment
```

Papers: `PAPERS/in Proposual/DPCC.pdf` (+ `DPCC.xopp` annotations),
`PAPERS/in Proposual/Diffuser_Janner.pdf`,
`PAPERS/auxiliary_papers/D3IL_relevant/D3IL.pdf`.

## DPCC plays two roles — keep them in separate chapters

| Role | Where it belongs | What to write |
| :-- | :-- | :-- |
| **Ancestor** — the code this repo is a copy-modify derivation of | `sec:method:dpcc` (Method, first section) | Summarise DPCC enough to make Ch. 4–6 readable, then **delineate** inherited vs. own. |
| **Baseline** — the system the results are measured against | `sec:setup:baselines` + Ch. 6 | Pin the exact baseline variant, then never renegotiate it per table. |

Conflating the two is the easy mistake: it reads either as under-crediting the
upstream, or as claiming credit for a comparison you inherited.

## Delineation — draw the line yourself

An examiner will look for the boundary between upstream and own work. Stating
it once, precisely and early, is far better than letting it be inferred.

- **Inherited:** temporal U-Net backbone, D3IL environments and demonstration
  data, the constraint-projection machinery, the training/eval scaffolding,
  normalisation and data pipeline.
- **Own:** the deterministic flow-matching engine replacing the diffusion
  engine; few-step engines (MeanFlow, α-Flow); the trajectory-optimization
  constraint arm ported from HardFlow; visual conditioning; the UAV embodiment
  and its closed-loop tracking; the multi-engine unified interface; the
  evaluation and analysis framework.
- **Verify before writing this list** — it is written from the generation index,
  not from a file-level audit. The reproducibility appendix needs a real
  provenance table (upstream / modified / new, plus licence notices), so build
  that table from the actual tree and let this list follow from it.

## Why it is a strength, not a caveat

Because FM-PCC and DPCC share a backbone, a data pipeline and an evaluation
harness, the comparison is unusually well controlled — the diffusion-vs-flow
substitution is close to a single-variable change. Say this explicitly in
`sec:setup:baselines`. It is the reason the architecture-matched claim (our
U-Net vs. the baseline U-Net at comparable parameter count) is the strong one,
and why transformer-backbone wins have to be reported as confounded secondary
evidence.
