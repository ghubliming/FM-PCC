# FROM v3 → v2 · 2026-09-17 · generative objectives use mechanism names

**Author's decision (v3.20):** published brands remain confined to Related Work and explicit
provenance or artefact mappings. The three thesis-facing names are:

| artefact family | thesis name |
| :-- | :-- |
| `fm` | **instantaneous-velocity matching** |
| `mf` | **analytic average-velocity matching** |
| `af` | **consistency-interpolated average-velocity matching** |

The names distinguish the targets actually trained. Instantaneous-velocity matching learns the
velocity at one transport time. Analytic average-velocity matching predicts velocity averaged over
an interval and constructs its target analytically through a Jacobian--vector product.
Consistency-interpolated average-velocity matching uses the consistency step ratio to interpolate
between the instantaneous velocity and a stop-gradient network prediction. The final ratio is
written `$\alpha_{\mathrm{end}}=0.2$` (or `0.05`), never *floor*.

This supersedes both the earlier request in this file to restore the α-Flow brand and v2.20's
*instantaneous-velocity / analytic average-velocity / intermediate finite-difference objective*
wording. The canonical translation table now records the v3.20 decision.

## For v2

- Apply the three names throughout Chapters 1--4 and the abstract, except Related Work and explicit
  provenance or artefact-token mappings.
- Rename the Chapter 4 subsection to
  `\subsection{Consistency-Interpolated Average-Velocity Matching}`.
- Keep the label `sec:method:alphaflow`; v3 references it.
- Keep `$\alpha_{\mathrm{end}}$` as the parameter name.

v3.20 applies these names to Chapters 5, 6, 8, the appendix and the affected figure labels.
