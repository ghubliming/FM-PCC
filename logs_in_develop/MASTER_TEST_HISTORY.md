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
*   **V4.1 22. April (Trajectory Visualization)**: Implemented the "Zero-Interference Logging" flag (`--datalog-for-traj`) to capture raw state tensors without affecting latency metrics.
    *   **New Tool**: Created `traj_gen_script_for_v4.py` which unnormalizes the model's latent robotic plans and overlays them on the exact environmental constraints (obstacles/halfspaces) from `projection_eval.yaml`.
    *   **Mission Goal**: Enables visual "Precision-Drift" auditing, allowing users to compare solvers like Euler and RK4 directly against the Oracle ground truth to verify robotic safety.


#### V4.1: Gen3v2: Solver Comparison Mission (Pending: 25. April)

Keywords: accuracy audit, Euler vs RK4 vs Oracle, trajectory visualization.

1. **Objective**: Run the full "Comparison Mission" as documented in the V4 Usage Guide.
2. **Target**: Quantify the physical L2 drift of Euler ($K=20$) and RK4 ($K=20$) against the Oracle ($Dopri5$ @ $1e-10$) reference.
3. **Validation**: Use `traj_gen_script_for_v4.py` to confirm that all solvers respect environmental constraints in the `avoiding-d3il` narrow-gap scenario.



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

## Gen3v2U1 FMv3 Threshold & Final-Step Snap Fix (21. April)

Keywords: final-step snap, threshold override fix, robotics-grade safety, data-end robustness.

1.  **Problem**: Identified a "Data-End" safety gap where the FMv3 integration could skip the final safety snap if the threshold was small or floating-point math rounded poorly. In contrast, the legacy DPCC code was robust due to its countdown logic.
2.  **Problem**: Discovered a "Chain of Custody" bug where the `diffusion_timestep_threshold` from the YAML config was ignored by the evaluation scripts.
3.  **Fix (Logic)**: Modified `flow_matcher_v3_ode_selectable/models/diffusion.py` to use an integer-based boundary and an explicit **force-include for the final integration step** ($idx = S-1$). This guarantees SafeFlow parity.
4.  **Fix (Override)**: Updated `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` to correctly extract and inject the threshold from the YAML config.
5.  **Outcome**: The safety window is now truthfully enforced and robotics-grade robust.

## Gen3v2U1.5 FMv3 Config Naming Alignment (22. April)

Keywords: config renaming, K20 legacy, ODE steps alignment, total synchronization.

1. **Change**: Created `args_to_watch_v3` to globally track `flow_steps_v3` instead of `n_diffusion_steps`.
2. **Change**: Updated the `exp_name` and `diffusion_loadpath` in `config/avoiding-d3il.py` for all FMv3 training AND plan models (`flow_matching_v3` & `flow_matching_v3_ode_selectable`).
3. **Outcome**: Total synchronization! The training script will now correctly save newly trained model folders as `K10` (or whatever the ODE steps are), and the evaluation scripts will look for and save results to that exact same `K10` folder. 
4. > [!NOTE] 
   > If you have an **old** trained model folder on disk named `K20` (trained before this patch), you will need to manually rename it to `K10` so the evaluation script can find it. Future training runs will name it correctly automatically.

## Gen3v2u2: RK4 Solver Validation & Loading Hotfix (23. April)

Keywords: RK4 solver, loading hotfix, benchmark auditing, solver validation.

Recap of I,II,III,IV tests:
1. **Test "I" (Wrong)**: Failed validation. The benchmark comparison was invalid because the "4x relation" in the diffuser metrics (expected for higher-order solvers) was non-existent in the actual model outputs, indicating the script was not yet running the intended RK4 code.
2. **Test "II" (Wrong)**: Tested "both-hard" constraints; output was still incorrect. Verified that legacy paths in pickled checkpoints were still overriding the current codebase.
3. **Test "III" (Success)**: Tested "both-hard" again with the dynamic override active. **Confirmed RK4 is running** correctly! The interceptor successfully pointed the model to the `flow_matcher_v3_ode_selectable` folder.
4. **Test "IV" (Correct)**: Generated high-fidelity RK4 data. This will serve as the gold standard for comparison against Euler FMv3 to quantify the precision-latency trade-off.

### IV results: 24 April Finished
FM-PCC\Results_and_Data_Analysis\Data_Analysis\Eval_Seed6_FMv3_RK4_vs_FMv3_Euler\IV



> [!IMPORTANT]
> **Dynamic Override**: Evaluation scripts now automatically detect and fix pickled module path mismatches (e.g., from `flow_matcher_v3` to `flow_matcher_v3_ode_selectable`) and sanitize outdated keyword arguments at runtime. This ensures that the configuration is always "King" and the most recent code is always used for inference.

---

## Gen3v2: DPCC Style Cost Comparison (Ongoing)

1. **Test Parameters**: `FMv3` testing is currently ongoing with `aw=10`, `ODE=10`, and the `euler` solver.
2. **Target**: Compare the DPCC style computational and performance cost directly against this configuration.

## Gen3v2: Plot Output Hotfix (24. April)

Keywords: plot output path, FM_test cleanup, dedicated plots folder.

1. **Problem**: Identified that the `load_results_flow_matching_v3_ode_selectable.py` script was hardcoding its plot outputs to the legacy `FM_test/` root folder, which contains unrelated scripts and is not the designated results directory for the v3-selectable path.
2. **Fix**: Updated the script to save comparison plots into a dedicated `plots/` subdirectory within `FM_v3_ode_selectable_test/` (relative to the script itself).
3. **Outcome**: Cleaner directory structure and proper isolation of test results. No more "weird" output in the legacy `FM_test/` folder.

---
