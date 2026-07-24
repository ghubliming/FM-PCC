---
name: docker-no-python-cluster-only
description: Python/interpreters ARE allowed here for debugging & (on request) data analysis, but NEVER run the FMPCC pipeline — that's a cluster job
metadata: 
  node_type: memory
  type: project
  originSessionId: 1acee9f0-c0f1-4ab7-8468-d4aeb6a7f718
---

A Python interpreter (and other coding interpreters) IS available in this Docker container and may **always** be used for debugging — e.g. `py_compile`/syntax checks, small stdlib-only scripts, inspecting logic. Data analysis with Python is also fine **when the user asks for it**. What's missing is the heavy FMPCC/conda stack (GPUs, MuJoCo, torch, the FMPCC conda env).

**NEVER run the pipeline locally** (training, eval, the closed-loop rollout/projection pipeline, pytest suites that import the FMPCC deps). Those are **cluster jobs** — they run on the remote Slurm cluster (i6-gpu-1), which has the full FMPCC conda env, GPUs, MuJoCo, and all deps; code is synced via git.

**Why:** The container has no conda/FMPCC packages, so anything importing torch/MuJoCo/etc. will fail; but stdlib-only Python (syntax validation, quick debugging) works and is encouraged.

**How to apply:** Freely use Python to debug/validate code (e.g. `py_compile` after edits). Do data analysis in Python only on request. For real training/eval/pipeline validation, write the code and note "run on cluster" — let the user run it. See also [[slurm-sbatch-is-real-entrypoint]] and [[no-auto-commit-no-coauthor]].
