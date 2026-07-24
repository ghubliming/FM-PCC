---
name: meanflow-family-upstreams
description: "The four MeanFlow-family upstreams in aux_repo — which one has a trainer, which is JAX vs PyTorch, and what each is good for"
metadata: 
  node_type: memory
  type: reference
  originSessionId: b70c3905-edff-449a-9085-024b6e096154
  modified: 2026-07-22T15:34:39.788Z
---

FM-PCC's iMF work (Gen3v4_imf, Gen13/HardFlow) depends on a family of MeanFlow upstreams in `/workspaces/aux_repo/`. They are easy to confuse; the practical differences:

- **`MeanFlow`** (haidog-yaqub/MeanFlow, PyTorch, **unofficial**) — added 2026-07-22. ⭐ The only upstream with a **working PyTorch trainer for iMF**: `meanflow.mode` ∈ `{"meanflow", "i-meanflow"}`, dual-head DiT (u + v), CFG-as-input, MNIST/CIFAR-10/ImageNet-latent configs. ~830 LOC total. Its author calls `i-meanflow` "more stable and recommended". This is the natural PyTorch oracle for parity-testing our ports.
- **`imeanflow`** (Lyy-iiis/imeanflow, **official** iMF, arXiv 2512.02012) — `main` is **JAX/Flax, TPU-shaped** (`jax[tpu]==0.4.27`, TF 2.15, orbax, `pmap`); `imf.py::forward` is the authoritative loss. Its `origin/torch` branch is **inference-only** (`assert eval`, no `forward()`, no trainer) — do not plan a PyTorch training path around it, but it IS a good architecture/sampler oracle. Note: official `imfDiT` **deliberately ignores `t`, conditioning only on `h = t − r`**.
- **`alphaflow`** (snap-research, **official** PyTorch, arXiv 2510.20771, "Understanding and Improving MeanFlow models") — added 2026-07-22. ⭐ Replaces MeanFlow's JVP target with a **self-bootstrapped finite-difference target** (`_compute_mean_velocity_d`), with `alpha` annealed **1.0 → 0** on a sigmoid schedule (FM → MeanFlow curriculum), plus `clamp_utgt: 4.0` and high FM-anchor ratios (0.25–0.75). Hydra/torchrun, ImageNet-latent, DiT.
- Lineage: MeanFlow (baseline, FID 3.43) → independent 2025–26 follow-ups: **iMF** (FID 1.72), **α-Flow**, and "Understanding, Accelerating, and Improving MeanFlow Training" (FID 2.87, not yet vendored) — each attacking a different weakness of the same baseline (target instability, gradient conflict, training dynamics).

**Why:** Gen13's iMF was empirically refuted (`Gen13/fix_7/RESULTS_..._VERDICT_imf_refuted.md` — FM@K=2 dominates), and the diagnosis in `HF_iMF/Research/COMPARE_gen13_hardflow_vs_gen3v4_imf_training.md` §8.2 is that the **JVP-based residual has a blind direction whose width is `h`**. α-Flow's bootstrapped target attacks exactly that mechanism, and the `MeanFlow` repo supplies the PyTorch oracle whose absence was the main gap in our audit.

**How to apply:** When asked whether our iMF port is wrong, use `MeanFlow` (PyTorch, objective-level) or `imeanflow` (JAX, authoritative) as the oracle rather than re-reading our own code. When asked how to make iMF work on avoiding, α-Flow's α-anneal is the first thing to try. Full analysis: `logs_in_develop/imeanflow_train/AUDIT_port_vs_upstream_and_the_train_in_imeanflow_proposal.md`. See also [[fmpcc-dev-logs-navigation]].
