---
name: project-env
description: "FM-PCC development environment — Docker for AI coding only, all runs on remote Slurm cluster via git sync"
metadata: 
  node_type: memory
  type: project
  originSessionId: c9ec8b03-7774-4a8a-a163-60517cad8227
---

This Docker container is **AI coding only** — no Python packages installed, cannot run Python scripts locally.

All real execution (training, eval, tests) runs on the **remote Slurm cluster** (`i6-gpu-1`), synced via git.

**How to apply:** Never attempt to run Python scripts, pytest, or any Python command locally. Write code and tests; the user runs them on the cluster. When a test needs to be validated, write it and note "run on cluster."

**Why:** The Docker env has no conda/FMPCC packages. The cluster has the full FMPCC conda env, GPUs, MuJoCo, and all deps.
