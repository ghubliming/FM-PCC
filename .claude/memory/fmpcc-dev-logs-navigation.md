---
name: fmpcc-dev-logs-navigation
description: "How to find FM-PCC's current state — MASTER_TEST_HISTORY.md is the master index; repo is unfinished and based on aux_repo/dpcc"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1acee9f0-c0f1-4ab7-8468-d4aeb6a7f718
---

FM-PCC is under heavy active development (unfinished, buggy in places; as of July 2026 Gen7/Gen8/Gen9/Gen11 and HardFlow+iMF are "working on"). To find the current state:

- `logs_in_develop/MASTER_TEST_HISTORY.md` — master index with a Gen0–Gen11+ trace map (model folder ↔ test folder ↔ status). Not 100% accurate; cross-check git history.
- `logs_in_develop/` — full per-generation dev logs (too many MDs to read blindly; navigate via the master file).
- Commit messages carry generation tags like `(Gen11 Fix11 & Sync to Gen7/Gen6V4 C4)`; fixes are often mirrored across sibling generations.
- Repo is majorly based on DPCC at `/workspaces/aux_repo/dpcc`, mixed with other repos in `/workspaces/aux_repo/` (d3il, diffuser, imeanflow, mujoco_mpc, SafeFlowMPC, UAV-Flow, drifting_policy).

**Why:** README.md is intentionally minimal, and user prompts can be inaccurate or forget details — verify against logs and git history.

**How to apply:** Before acting on assumptions about project state, check MASTER_TEST_HISTORY.md and recent git log. Follow the copy-modify sibling-folder pattern for new work. See also [[docker-no-python-cluster-only]].
