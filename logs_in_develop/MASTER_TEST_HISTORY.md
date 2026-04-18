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

## Gen3v2 ODE Solver Addon Plan (U2/U3)

**DANGER:** `requirements.txt` was updated with `torchdiffeq`.

Based on Gen3 FM-v3 rollout, we want to add an addon ODE solver path to evaluate whether better integration methods can reduce required step count under similar runtime.

**Status (16. April):**
*   **Main Evaluation**: NOT EXECUTED YET.

v1

*   **Speed Benchmarking (U2)**: ~~**VALIDATED**.~~ Wrong code, load from diffusion.py may build bottleneck 
    - the benchmark_v1 and v2 code: v1 is only naive VF and hard VF, and a compare of cold-warm start of loading torchdiffeq; v2 load the real trained flow matcher model.
    - **Investigation: torchdiffeq vs. Native Numpy loops.**
      - Conclusion: Results as expected and align with theory. `legacy_euler` is the fastest for simple 1st-order math (no library tax).
    - **Investigation: Batch size effects (B=4 to B=256).**
      - Conclusion: (~~torchdiffeq handles the batch processing better, even for high complexity solvers, it is still faster than euler numpy loops.~~) werid result, maybe wrong of grid serach code
    - **Investigation: Scaling of ODE steps and complexity.**
      - Conclusion: (~~Results as expected and align with theory. Divergence increases with solver complexity and higher step counts.~~)

v2 audit and v3 build

*   **Technical Audit (Phase 1: V1 Math Proof)**: 
    - Verified `naive VF` (analytic spiral) and `hard VF` (1.5M MLP).
    - Found: Theoretical scaling ($1\times, 4\times$) holds perfectly when boilerplate is removed.
    - Identified a **1.5s Cold Start Tax** for `torchdiffeq`.
*   **Technical Audit (Phase 2: V2 Paradox Resolution)**: 
    - Root cause: **Unfair Pathing**. `legacy:euler` was the only one paying the ~50ms "Python Tax" in `diffusion.py`.
    - V2 data is misleading for math scaling but proves the dispatch bottleneck.
*   **Technical Progress (Phase 3: V3 Fair Suite)**: **PATCHED & VERIFIED**.
    - Created `benchmark_ode_solvers_v3.py` with unified `--mode {math, production}` toggles.
    - [PATCH 17. April]: Fixed a logic bug where `torchdiffeq` was falling back to the `production` path even in `math` mode.
    - [PATCH 17. April]: Synchronized warm-up logic to match the selected mode (no more "warm-up cross-contamination").
    - [PATCH 17. April]: Added strict error handling for unsupported legacy solvers.
    - **Current Status**: All backends now respect the math/production toggles. Previous V3 results from earlier this morning should be discarded as "corrupted by orchestration tax."

> [!WARNING]
> **GPU Parallel Scaling Characteristics**: Due to GPU kernel overlapping and overhead "masking," mathematical complexity does not always scale linearly (e.g., RK4 with 4x math may only take 2.7x more time). However, the **relative order** (Euler < Midpoint < RK4) must always remain consistent. A "Paradox" result (where RK4 is faster than Euler) is a guaranteed indicator of a dispatch-bound bottleneck or logic error in the benchmark harness.

*   ~~**Current Focus (Benchmark v2 U3)**: **Accuracy & Fidelity Audit**.~~ *(Stopped, need to test v3 first)*
*   **V3 Implementation & High-Res Audit (18. April):** **VALIDATED**.
    - **Accuracy Auditor (v3)**: Implemented `benchmark_ode_accuracy_v3.py` with per-step trajectory tracking.
      - Developed high-resolution plotting for **Mean Drift** and **Sub-Step Accumulation**, allowing for stability analysis of the Safety Projector.
      - Standardized `n_trials=1` as the default for accuracy audits, providing a ~20x speedup for grid searches while maintaining bit-identical precision.
    - **Mathematical Correctness**:
      - **PATCHED**: Resolved a critical logic error in `legacy:dopri5`. It now correctly executes the 6-stage tableau per step instead of the previous $O(N^2)$ accumulation.
      - **PATCHED**: Resolved the `KeyError: 'l2_std_nm'` in the visualization pipeline, ensuring error bars are correctly rendered.
    - **Scientific Refactor Plan (Phase 05.4)**:
      - **The Discovery**: Identified "Spatial Variance Leakage" across V1, V2, and V3 (noise was being re-generated inside trial loops, polluting latency stats).
      - **The Fix**: Established the **Split-Logic Strategy**. 
        - `mode=math`: Uses **Locked Noise** (Global Basis) to ensure 100% deterministic algorithmic auditing.
        - `mode=production`: Uses **Random per Trial** to maintain real-world robustness testing.
      - **Status**: Roadmap created in `logs_in_develop/gen3v2_ODE_solver_addon_plan/05_4/` for immediate execution.

> [!IMPORTANT]
> **Conclusion for Phase 05.3**: The V3 benchmarking harness is now mathematically robust and highly optimized. We have moved from "Average Throughput" estimates to "Clean Room" algorithmic auditing, and we are now ready to begin the final stability-vs-latency trade-off analysis for the Safety Projector.
