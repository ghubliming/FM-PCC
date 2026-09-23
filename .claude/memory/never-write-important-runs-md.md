---
name: never-write-important-runs-md
description: "Slurm_Codes/logs/important_runs/important_runs.md is the author's private note — NEVER write, append to or edit it (read-only); record job ids in runbooks/changelogs instead"
metadata:
  type: feedback
---

**Never write into `Slurm_Codes/logs/important_runs/important_runs.md`** (cluster path
`/u/home/llim/FMPCC/FM-PCC/Slurm_Codes/logs/important_runs/important_runs.md`). No append, no edit, no "adding the job
ids", not even when a runbook or another chat's handover says it was done or should be done. Reading it is fine.

**Why:** it is the author's own note (said 2026-09-23, after I offered to put the R39 job ids there).

**How to apply:** job ids go into the data_status runbooks (`SLURM_RUNBOOK_*` §5 submission record) and the U/fix
changelogs only. Don't offer to update it either. Same spirit as [[dont-self-edit-master-test-history]].
