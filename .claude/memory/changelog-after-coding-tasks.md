---
name: changelog-after-coding-tasks
description: "After each coding task, write a changelog MD into logs_in_develop under the working epoch/version folder"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1acee9f0-c0f1-4ab7-8468-d4aeb6a7f718
---

After each coding task in FM-PCC, write a changelog MD into the matching `logs_in_develop/<Gen...>/<epoch/version>/` folder. The user should give the working epoch/version (e.g. Gen3v4_imf/U9); if not given, ask before creating a folder manually.

**Why:** logs_in_develop is the project's development record; every change must be traceable there (the repo's whole history navigation depends on it).

**How to apply:** Default to a concise changelog unless the user says "full" (use full for complicated jobs). Concise = clear, no code quotes needed, but MUST cover every file/change touched: what changed from what — line numbers may shift, but what the job did must be unambiguous. See also [[fmpcc-dev-logs-navigation]] and [[no-auto-commit-no-coauthor]].
