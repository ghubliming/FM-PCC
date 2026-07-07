---
name: docker-no-python-cluster-only
description: This Docker container has no Python packages; all execution happens on the remote Slurm cluster i6-gpu-1
metadata: 
  node_type: memory
  type: project
  originSessionId: 1acee9f0-c0f1-4ab7-8468-d4aeb6a7f718
---

This Docker container is AI coding only — no Python packages installed (no conda/FMPCC env). All real execution (training, eval, tests) runs on the remote Slurm cluster (i6-gpu-1), synced via git.

**Why:** The Docker env has no conda/FMPCC packages. The cluster has the full FMPCC conda env, GPUs, MuJoCo, and all deps.

**How to apply:** Never attempt to run Python scripts, pytest, or any Python command locally. Write code and tests; the user runs them on the cluster. When a test needs validation, write it and note "run on cluster." See also [[no-auto-commit-no-coauthor]].
