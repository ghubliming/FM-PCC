---
name: architecture-matched-beat-is-the-strong-claim
description: "A beat over the DPCC baseline only counts as a controlled result when the backbone matches the baseline's UNet; SiT/DiT wins are bigger but confounded by architecture+capacity and must be reported as secondary"
metadata:
  type: feedback
---

The diffusion-DPCC baseline uses a **temporal UNet** (`models.UNet1DTemporalCondModel`). So when one
of our engines beats it, **the claim's strength depends on which backbone our engine used**:

| our backbone | params | vs baseline | claim strength |
|---|---|---|---|
| **`unet`** (e.g. MeanFlow-UNet, `freq_dim=32`) | **4.0 M** | same architecture family, *smaller* | ✅ **controlled — this is the headline beat** |
| `mf_dit` (MeanFlow-DiT) | 10.1 M | different arch, 2.5× capacity | 🟡 secondary, confounded |
| `sit` (AlphaFlow-SiT) | 10.0 M | different arch, 2.5× capacity | 🟡 secondary, confounded |

**Report the architecture-matched row as the primary result even when a SiT/DiT row has better
numbers.** On `avoiding-d3il` (2026-08-13) AlphaFlow-SiT reached 41.7× the baseline's speed and
MeanFlow-UNet only 14.6× — but AF-SiT changes *two* things at once (objective **and** network), so
its margin cannot be attributed to the flow/few-step objective. MeanFlow-UNet changes only the
objective and is the smaller model, so its 14.6× is the number that actually isolates the
contribution.

**The missing control that would upgrade the SiT/DiT claims:** run the *baseline* (diffusion-DPCC)
and naive FM on a SiT/DiT backbone. Until that exists, a SiT/DiT win over a UNet baseline is
"our system beats theirs", not "flow matching beats diffusion". Say so explicitly rather than
letting the bigger number carry the abstract.

**Why:** the research question is whether the deterministic few-step engine replaces the stochastic
diffusion one — not whether a larger transformer beats a smaller convnet. A cross-architecture
comparison answers the second question while appearing to answer the first, and reviewers will find
it immediately.

**How to apply:**
- In every results table, carry a **backbone + parameter-count column**. Never present a SiT/DiT row
  next to a UNet baseline without it.
- Structure baseline comparisons as: **(1) architecture-matched beat** → **(2) best-overall beat,
  flagged as confounded** → **(3) the control that is missing**.
- The same rule applies to strict Pareto dominance: on 2026-08-13 the *only* strictly dominating row
  (AlphaFlow-SiT K10, both axes significant) was cross-architecture, so the honest statement is
  "no architecture-matched configuration strictly dominates the baseline yet".
- Backbone is set by `imf_backbone` in `config/avoiding-d3il.py` (train **and** plan blocks must
  match) and is printed at eval as `[ …TrajectoryModel ] backbone=… params=…` — quote that line as
  provenance.
- Stacks on top of [[benchmark-hierarchy-who-beats-whom]] and
  [[da-target-is-best-baseline-variant]]: the Target is still DPCC K20/`aw10`, and the win rule is
  unchanged — this memory only governs **which winning row leads the write-up**.
