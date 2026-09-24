---
name: uav-eval-deterministic
description: The UAV-mix eval is deterministic for a fixed configuration — re-flying an identical cell under a new tag reproduces every flight exactly (only ms/step moves); never propose re-runs just to re-tag
metadata:
  type: project
---

Verified 2026-09-24 (R44a, s-curve): six cells re-flown under tag `p23scgrid` matched their old tags (`u7hg`,
`EPlatest_u6unet_ae02`) on **every outcome value of every flight** (0 of 600 differ: success, S&C, distance, abort
and its step, steps, violating steps, crossed, min z, tracking error). Only ms/step moved (≤ 1.2 ms, cluster latency).
Trial i is seeded by its index; the checkpoint file was the same (logs print `checkpoint = state_… (trained to step N)`).

**Why:** identical (checkpoint, seed, nfe, variant, geometry, controller, SAFE_EPS, eval code) ⇒ identical flights, so
a re-run to get a fresh tag costs GPU and adds nothing. A code change breaks the identity (u7hg vs u18sc HardFlow
rows differ after the 18-09 switched-wall fix).

**How to apply:** when planning runs, cite existing identical cells instead of re-flying them (e.g. skip Phase B's
`diffuser` row when it equals a Phase A cell); check the loaded checkpoint step and the code revision before assuming
identity. Related: [[temp-bash-unique-marker]], [[da-requires-csv-never-from-logs]].
