---
name: no-auto-commit-no-coauthor
description: Never auto-commit; user commits manually. Never add Claude co-author lines to commits
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1acee9f0-c0f1-4ab7-8468-d4aeb6a7f718
---

Never commit automatically in FM-PCC — the user almost always commits manually and wants to control commits themselves. Never add a `Co-Authored-By: Claude ...` line to any commit.

**Why:** The user manages git history manually and syncs code to the Slurm cluster via git; unexpected commits or Claude attribution lines pollute their workflow.

**How to apply:** After making changes, stop and let the user commit. Only run `git commit` if explicitly asked in that moment, and even then omit the co-author trailer. See also [[docker-no-python-cluster-only]].
