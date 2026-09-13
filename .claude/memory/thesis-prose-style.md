---
name: thesis-prose-style
description: How the user wants thesis prose written — abstract short, storytelling not justification, no jargon/self-talk, no global "better", de-facto names
metadata:
  type: feedback
---

Thesis prose (FM-PCC Master's thesis, `Writing/Working_Space/`) must read as **plain scientific
storytelling**: say what is used and what it does, optionally with a short "because". The user has
corrected each of the following explicitly, some more than once (2026-09-11 → 09-13):

- **Abstract = an abstract.** Problem, approach, settings, results, contributions — under ~200
  words. Not a multi-paragraph pitch.
- **No justification / rebuttal prose.** No "why not X", "does not claim", "must not be read as",
  "worth stating", "rather than left to be discovered", "Neither is strictly better".
- **No self-talk.** No "The section argues…", "This chapter builds…", "the load-bearing section",
  "the comparison the whole thesis turns on", "Chapter summary." labels.
- **Never define "better" globally.** State the compared metrics where a claim is made (success,
  constraint satisfaction, NFE, wall-clock). The Pareto rule in [[pareto-definition-of-good]] governs
  DA reports, not thesis prose.
- **No jargon or coined words.** Not "arm A/B/C" (→ unguided / iterate projection / endpoint
  projection), not "candidate fan", "harness", "ladder", "bootstrapped target" (invented — it is
  consistency training). Use the de-facto scientific name, verified in the source paper.
- **"Backbone" is fine** — DPCC, DiT and MeanFlow all use it.

**Why:** the user reads the draft as an examiner would; jargon, meta-commentary and defensive
framing read as filler and obscure the method.

**How to apply:** before writing or editing `thesis_v*.tex`, check new text against this list and
the canonical name table `Writing/Auxiliary/Naming/NAMING_20260910_master_table.md`. See also
[[master-thesis-writing-tum]].
