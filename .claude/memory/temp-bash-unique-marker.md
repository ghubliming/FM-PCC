---
name: temp-bash-unique-marker
description: Every temp_bash run driver gets a unique marker tag (msg) that names the results folder, the Slurm job and its log, plus a job-id ledger — so results are findable and never pooled with old tags
metadata:
  type: feedback
---

A temp_bash submission driver must stamp every job with a **unique marker** (the "msg"/tag): the results
folder suffix (`FMPCC_UAV_EVAL_TAG` for UAV-mix, `FMPCC_RUN_MSG` → `_msg<tag>` for D3IL evals), the Slurm
job name (so `squeue` and the log file `Slurm_Codes/logs/<date>/<time>_<jobname>_<id>.log` show it), and a
ledger `Slurm_Codes/logs/<date>/<prefix>_<tag>_jobids.tsv`. Refuse old tags; skip cells already queued or
already holding results under the marker (FORCE=1 to re-fly).

**Why:** author, 2026-09-23 (R44 s-curve): "the temp bash remember has unique msg as marker". Children of
`eval_k_sweep.sh` are all named `uav_mix_eval`, so concurrent waves are indistinguishable in squeue/logs, and
untagged re-runs silently overwrite or pool with old cells.

**How to apply:** model on `Slurm_Codes/temp_bash/eval_20260923_p23_scurve_R44.sh` (modes plan/submit/status/export;
status greps the child log for tag, K, variants, controller, checkpoint; export tars only the tag's folders + logs).
Related: [[slurm-sbatch-is-real-entrypoint]], [[da-requires-csv-never-from-logs]].
