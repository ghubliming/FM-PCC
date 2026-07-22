---
name: fmpcc-dev-logs-navigation
description: "How to find FM-PCC's current state — MASTER_TEST_HISTORY.md is the master index; repo is unfinished and based on aux_repo/dpcc"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1acee9f0-c0f1-4ab7-8468-d4aeb6a7f718
  modified: 2026-07-22T15:34:17.447Z
---

FM-PCC is under heavy active development (unfinished, buggy in places; as of July 2026 Gen7/Gen8/Gen9/Gen11 and HardFlow+iMF are "working on"). To find the current state:

- `logs_in_develop/MASTER_TEST_HISTORY.md` — master index with a Gen0–Gen11+ trace map (model folder ↔ test folder ↔ status). Not 100% accurate; cross-check git history.
- `logs_in_develop/` — full per-generation dev logs (too many MDs to read blindly; navigate via the master file).
- Commit messages carry generation tags like `(Gen11 Fix11 & Sync to Gen7/Gen6V4 C4)`; fixes are often mirrored across sibling generations.
- Repo is majorly based on DPCC at `/workspaces/aux_repo/dpcc`, mixed with other upstream reference repos in `/workspaces/aux_repo/`. Current contents (2026-07-22, late): alphaflow, d3il, diffuser, dpcc, drifting, drifting_policy, flow_guidance, HardFlow, imeanflow, l4casadi, MeanFlow, mujoco_mpc, SafeFlowMPC, UAV-Flow — **this is a DYNAMIC, growing list; new reference repos get added over time, so always `ls /workspaces/aux_repo/` for the live set rather than trusting this snapshot.** (`MeanFlow` + `alphaflow` added 2026-07-22 — see [[meanflow-family-upstreams]].)
- `flow_guidance` and `l4casadi` were pulled in because HardFlow's README credits them: HardFlow's structure is inspired by flow_guidance, and l4casadi supplies the PyTorch↔CasADi bridges that HardFlow + the projection baselines need (the `hardflow_new` l4casadi-free variant does not). When HardFlow/iMF behavior is unclear, these are the upstreams to compare against.

**Why:** README.md is intentionally minimal, and user prompts can be inaccurate or forget details — verify against logs and git history.

**How to apply:** Before acting on assumptions about project state, check MASTER_TEST_HISTORY.md and recent git log. Follow the copy-modify sibling-folder pattern for new work. See also [[docker-no-python-cluster-only]].
