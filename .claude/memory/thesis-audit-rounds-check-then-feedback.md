---
name: thesis-audit-rounds-check-then-feedback
description: "External (ChatGPT/Codex) audits of a thesis draft land in <draft>/audit from chatgpt/; Claude verifies each finding against code, appends a signed feedback section at the END of that MD, and applies nothing — the author debates with the auditor first"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 357d0d2d-7ccb-40ae-8f1a-5e2aba614f98
  modified: 2026-09-23T15:23:58.255Z
---

**Workflow (author, 2026-09-23).** A ChatGPT/Codex audit of a draft is dropped as
`Working_Space/<draft>/audit from chatgpt/AUDIT_<...>.md`. Claude's job on it: read it with suspicion (the
auditor "maybe doesn't have enough context"), check every finding against the code and the current sibling
draft, and write the feedback **at the end of that same MD** as a new section — verdict per finding, the file
and line checked, what to debate. **Do not apply the audit's fixes**; the author and the auditor debate until
they agree, and fixes come later as separate instructions. Only when the author asks for other edits in the
same message and those edits overlap an audit finding, apply it inside the requested rewrite and say so in
the feedback (v2.26: F02/F15/F20 inside the §4.6.3 rebuild).

**Why:** the audit is a second reviewer, not an instruction list; the author wants two independent readings
reconciled before the text moves, and wants the verification trail (what was read) to judge both.

**How to apply:**
- One verdict table (✅ / ⚠️ / ❌), evidence column with `file:lines`, a "what the audit missed" part, and a
  "what I applied anyway and why" part; sign it.
- Findings that reach another draft's chapters go to `cross_draft/to_<draft>/` as usual.
- Real defects found while checking (e.g. the FM sampler's σ = 0.5 prior vs σ = 1 training,
  `flow_matcher_v3/models/diffusion.py:164`) are told to the author in chat, never written into the thesis.

Related: [[thesis-draft-ownership]], [[thesis-prose-style]], [[no-unrequested-code-edits]]
