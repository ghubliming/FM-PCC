# FROM v2 → v3 · 2026-09-17 · generative models use mechanism names

**Author's correction:** MeanFlow and α-Flow are published brands, not the thesis-facing names. The
canonical translation table was updated in v2.20.

## Names to use outside Related Work

| artefact token | thesis name |
| :-- | :-- |
| `fm` | **instantaneous-velocity objective** |
| `mf` | **analytic average-velocity objective** |
| `af` | **intermediate finite-difference objective** |
| `hardflow_*` | **endpoint projection** (unchanged) |

The intermediate objective constructs its target by a finite difference from the network's own
stop-gradient prediction. If the final consistency step ratio is needed, write
`$\alpha_{\mathrm{end}} = 0.2$`, never *floor*. Published brands may be used in Related Work with
their citations; artefact tokens remain unchanged in paths, code, CSVs and provenance notes.

## For v3

Apply these names to Chapters 5, 6, 8 and the appendix, including tables, captions, result summaries
and the artefact-to-prose translation table. Do not alter labels, code tokens or stored artefacts.
The inherited abstract will receive the same terminology through the next v2 front-matter sync.

Source of decision:
`Writing/Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md` §2 and §8, v2.20.
