---
name: no-unrequested-urls-or-artifacts
description: Never publish artifacts or put URLs in output unless the user explicitly asks
metadata:
  type: feedback
---

Do NOT publish Artifacts, and do NOT write any URL/link into responses or files, unless the user
explicitly asks for it. Deliver work as files in the repo (MD + figures) and report results in the
terminal.

**Why:** user works in-repo on a cluster-backed research project; deliverables live in
`logs_in_develop/` and `Data_Analysis/` and are consumed via git, not the web. Unrequested links are
noise and were called out directly ("forbidden to write any URL thing unless asked").

**How to apply:** default output = repo files + a terminal summary. If a visual is needed, write
PNG/SVG into the repo next to the MD (see [[changelog-after-coding-tasks]]). Only publish/link when
the user says so. Related: [[no-unrequested-code-edits]].
