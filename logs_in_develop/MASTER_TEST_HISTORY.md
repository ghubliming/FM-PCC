# Test History

Purpose: concise record of what was tested across all generations/vresions. Master logging markdown.

## Gen1

Keywords: wrong code, reversed ODE trajectory.

1. Early FM code had reversed ODE trajectory direction.
2. Result interpretation from this phase is not trusted as final baseline.

## Gen2

Keywords: diffusion engine replacement, basic FM engine, uniform time, 20/20/20.

1. Replaced old diffusion engine with a basic FM engine.
2. Time handling used uniform time in [0,1].
3. Main setting used 20 train steps, 20 sampling steps, 20 ODE steps.

## Gen2 (U-Net v2)

Keywords: U-Net v2 build, TODO architecture change, no effective behavior change.

1. Built U-Net v2 path.
2. Structural U-Net-v2 upgrade remained TODO.
3. Net behavior change was not material in this phase.

## Gen3 Upgrade 1 Hyperparameter Tuning

Keywords: action_weight_a0 tuning, HP1=1, HP2=5.

1. Tuned FM action_weight_a0 from original 10.
2. HP1 set action_weight_a0 to 1.
3. HP2 set action_weight_a0 to 5.

## Gen3 Upgrade 2 FM-v2

Keywords: beta time, two de facto tests, ODE=10 eval change.

1. Implemented beta-time sampling in FM-v2.
2. De facto test #1: Beta-time only.
3. De facto test #2: Beta-time plus eval ODE changed to 10. (in logs it is mark with FMv2, ie. default name)
4. Test markings:
5. "Beta Time" marks beta-only test.
6. "ODE=10" marks beta-time plus eval ODE=10 test.

## Gen3 Upgrade 3 FM-v3

Keywords: SafeFlow-style time semantics, continuous-time query, flow_steps_v3.

1. Introduced v3 path with SafeFlow-style continuous-time model query semantics.
2. Added v3 config/script path and v3 parameter naming.
3. Kept v2 path intact for rollback and comparison.


## Gen4 Visual Model for Avoiding D3IL (Abandoned, Not Usable, Code Kept for Reference)

Keywords: visual avoiding, vendored d3il, config split, copy-modify isolation, compatibility guard.

Objective:
1. Build a visual-avoiding train/eval path while preserving the old state baseline for rollback and A/B checks.

What was done:
1. **DANGER: major code structure change.** D3IL was integrated into FM-PCC (vendored) instead of being cloned separately.
2. Created a Gen4 visual-avoiding train/eval path using copy-modify isolation.
3. Added visual-specific config and eval split for avoiding experiments.
4. Kept the old state baseline runnable for rollback and A/B comparison.

Critical error identified:
1. Avoiding task code in D3IL was modified directly.
2. This should have been implemented as an additive extension on top of the existing avoiding path.
3. Direct modification increased regression risk and code entanglement.

Correction rule carried forward:
1. Fix in Gen5 by keeping baseline avoiding stable and extending via isolated visual paths.
2. Follow the same separation style used by other D3IL visual models to avoid coupling.

## Gen5 FMv3 Aligning Vision First

Keywords: reuse-first, benchmark existing vision models, FMv3 aligning vision, avoiding extension, fake-vision guard.

Strategy reset:
1. Validate existing D3IL visual models first (aligning, sorting, stacking) before avoiding extension.
2. Rewire and reuse existing visual model contracts before any new architecture work.

Execution rules:
1. Extend into avoiding only after visual health checks pass.
2. Keep baseline avoiding path stable in vendored FM-PCC/d3il.
3. Use isolated copy-modify paths for FMv3 aligning vision work.

Non-negotiable guard:
1. Vision mode must be real image-conditioned behavior and must not silently fall back to state-only behavior.

## Gen3v2 ODE Solver Addon (U2/U3)

### [Part 1] Benchmark Evolution (Scientific Audit V1-V4)
*   **V1 (Analytic)**: Verified math scaling ($1\times, 4\times$) on synthetic fields. Identified the **1.5s cold-start delay** in `torchdiffeq`.
*   **V2 (Failed Real-VF)**: First attempt at the **Real Vector Field** (trained model). **Problem**: Results were invalid due to **Broken Loading Logic**; the runner failed to actually wire the real ODE solvers from `diffusion.py`. 
*   **V3 (Fixed Integration)**: Successfully bridged the solvers to the production `diffusion.py` paths. **Problem**: High statistical variance across trials because every trial used new random noise batches (**Inter-Trial Divergence**).
*   **V4 (Deterministic Standard)**: Final standard with a **Locked Noise Basis** (`global_x_init`). **Logic**: Fixes the V3 randomness by ensuring all solvers in the trial integrate the exact same batch for bit-identical auditing.

### [Benchmarking Conclusion (V1-V4)]
*   **Backend Reliability**: `torchdiffeq` validated as a stable and reliable backend with manageable initialization/kernel overhead on GPU.
*   **Math Proofs**: Audits confirmed that mathematical stage scaling ($1\times, 2\times, 4\times$) holds true for Euler, Midpoint, and RK4.
*   **ODE Fidelity**: Validated that at a fixed $ODE\_steps=10$, RK4 is mathematically more accurate (lower L2 drift) than Euler. Found an **Accuracy Crossing Effect** where at extremely low step counts (e.g. 2-3), Euler is comparable, but RK4's advantage scales exponentially as step resolution increases.
*   **Per-Step Drift Research**: Audited the relationship of cumulative drift at each individual integration step. This is critical for **DPCC (Differentiable Predictive Constraint Control)** as it informs the frequency and strength required for the per-step projection logic.
*   **Production Handshake**: The grid-search verified the relationship between solver complexity and accuracy; this same mapping logic is now hardened and wired into the production `FMv3` engine.

> [!NOTE]
> All findings in this benchmarking audit are derived from the **real trained Vector Field (from FMv3)**, ensuring that the documented precision and latency characteristics are representative of the actual production system.

### [Part 2] FMv3 "Ode-Selectable" Engine
*   **What it does**: Decouples the solver from the model core to allow plug-and-play integrators via configuration.
*   **Problem met**: Hardcoded 1st-order Euler prevented the use of high-precision safety methods in narrow-gap environments.
*   **The Upgrade**: Implemented the **Generic Solver API** in `diffusion.py`. Optimized the internal loops to ensure high-order methods (RK4) have minimal hardware overhead.

---

**Final Verification (20. April)**: The suite is now scientifically hardened. All future solver comparisons must use the V4 deterministic harness.

### [Final Verdict]
*   **Result**: Tested RK4 ($10$ steps) vs. Legacy Euler ($10$ steps). 
*   **Outcome**: RK4 only cost more redundant latency (~20%) with zero improvement on environment steps or success metrics. 
*   **Conclusion**: For the current trained model on the `avoiding-d3il` task, the Vector Field is stable enough that 1st-order integration is sufficient; high-order methods provide mathematical safety overhead but no macro-behavioral gain.
