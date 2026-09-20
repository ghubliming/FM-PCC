---
name: thesis-prose-style
description: How the user wants thesis prose written — abstract short, storytelling not justification, no jargon/self-talk, no global "better", de-facto names
metadata:
  type: feedback
---

Thesis prose (FM-PCC Master's thesis, `Writing/Working_Space/`) must read as **plain scientific
storytelling**: say what is used and what it does, optionally with a short "because". The user has
corrected each of the following explicitly, some more than once (2026-09-11 → 09-13):

- **Abstract: NO NUMBERS, ever** (user was furious when v2.11 put them in). Model it on the
  reference papers in `aux_repo/PAPERS` — DPCC, HardFlow, SafeFlowMPC, Diffuser all do the same: one
  paragraph, ~150–220 words, context → "However," limitation → what this work does → how → what it
  enables → one *qualitative* results sentence ("simulations show that X … as reliably as … at a
  fraction of the cost"). Never list negative results in the abstract.
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
- **Branded method names only in Related Work** (user angry, 2026-09-14; final author decision v3.20,
  2026-09-17). *MeanFlow*, *α-Flow*, *HardFlow* never outside Related Work / provenance. Write
  **instantaneous-velocity matching** (fm), **analytic average-velocity matching** (mf),
  **consistency-interpolated average-velocity matching** (af), **endpoint projection**, with in-place
  `\parencite`; tables use FM / MeanFM / CI-MeanFM. *Flow matching* stays as the family name; *DPCC* stays.
  Never "consistency training" (a different published method); write *α_end = 0.2*, never "floor". Keep
  labels and code tokens unchanged. (The earlier same-day "α-Flow stays named" exception is superseded.)
- **Describe only what this thesis has and does** (user angry again, 2026-09-14). Never compare our
  configuration with a paper's published configuration ("what differs from the published…", "as
  published", "departures from the source"), and never explain why a choice was made. Never
  describe variants that no reported run uses (e.g. the affine/FiLM conditioning — the thesis has
  one conditioning, *feature-wise conditional biasing*). No defensive remark blocks ("facts that
  are not free choices", "not cosmetic", "rather than by omission"), no "shipped". If such a
  remark hides a real defect, tell the user in chat instead of writing it into the thesis.
- **No statistics in the thesis** (user angry, 2026-09-17: "that is confidential inside job… that NOT how
  do the storytelling"). No p-values, no test names (sign/permutation/Fisher), no "significant" — in
  protocol, results or recap. Report counts (*nine of ten contexts*, *12/12 vs 0/12*) and margins. DA
  reports may keep tests; the thesis never does. Recap sections = plain "who is best" tables.
- **MuJoCo is the simulator; D3IL only supplies task files, demonstrations, camera placements**
  (2026-09-17). Never "D3IL simulator view" or "(D3IL)" on a MuJoCo render.
- **Caveats are told as story sections** (2026-09-17): e.g. "Caveat: What the Metrics of DPCC Do Not
  Measure", "Caveat: The Tracking Controller" — a weakness stated plainly, with evidence, not as a
  defensive remark.
- **The prompt is never thesis text** (user angry, 2026-09-20: *"this is my prompt, why it is inside the
  thesis? this is self talking and prompt rephasing!"*). What the user asks for enters the draft as a
  *fact about the figure/table*, never as (a) a restatement of the instruction ("both translucent **so
  that** the excluded region stays readable underneath"), (b) a "which is why …" justification tail, or
  (c) a claim that it was delivered ("in the two forms D3IL-avoiding **also uses**"). A caption says what
  is drawn, what each colour/style/panel means, the numbers not recoverable from the drawing, and a
  pointer to the table carrying the claim — nothing else. Binding rule with worked examples:
  `Writing/Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md` (also naming-table §8 rule 9).
  The user names two alerts: 🔴 **RED** = the prompt is in the thesis; 🟡 **YELLOW** = *self-reasoning*,
  a `because` the evaluation did not produce (*"does not recover the gap, **because what the budget
  buys back sits in the projector rather than in the denoiser**"*), or a paragraph answering an
  objection the reader never made.
- **Write as the person who ran the experiments** (2026-09-20): *"you just fact truth telling, and story
  telling, you are the boss ... not the self explosion suicide way."* State the measurement flatly; no
  hedging, no apologising for the sample, no anticipating objections, no telling the reader what to
  conclude. A real limit goes once into a `\guard`, plainly — a limit argued around reads as weakness.
  This is not licence to overclaim: say exactly what the data supports, then stop.
- **Tell the evaluation as two stages: prove, then extend.** Obstacle avoidance is DPCC's benchmark,
  used 1:1 to meet the baseline on its own ground. The vision-conditioned alignment task and the
  quadrotor benchmark were **built** for this thesis (the quadrotor: everything but the Skydio X2
  model). Never say they were *chosen*.

**Why:** the user reads the draft as an examiner would; jargon, meta-commentary and defensive
framing read as filler and obscure the method.

**How to apply:** before writing or editing `thesis_v*.tex`, check new text against this list and
the canonical name table `Writing/Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md`. See also
[[master-thesis-writing-tum]].
