---
name: archived-codes-is-dead-code
description: "Archived_Codes/ (and *(legacy_based_*) folders) are DEAD/WRONG code — never use, run, or edit them"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c4045e45-890e-41a2-8f57-08a0d18f3d74
---

Everything under `Archived_Codes/` — and any sibling folder marked `(Abandoned)`, `legacy`,
`(Outdated)`, or `*(legacy_based_on_*)` — is **dead code / wrong code**. It is never used and
must never be run or changed.

**Why:** the repo uses copy-modify isolation; abandoned generations are kept only as inert
snapshots. Touching or "fixing" them wastes effort and the user is explicit that it is pointless.

**How to apply:** when patching cross-generation, operate ONLY on live/active sibling folders.
Never edit, run, or "fix consistency" in archived/legacy folders. Do not even list them as
skipped work items — just ignore them. The ONLY allowed use is, in very rare cases, reading them
to learn prior behavior — read-only, never modify. Related: [[fmpcc-dev-logs-navigation]].
