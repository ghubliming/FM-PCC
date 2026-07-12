# Test History

For SLURM jobs history, refer to [important_runs.md](../Slurm_Codes/logs/important_runs/important_runs.md)

Purpose: Concise record of what was tested across all generations/versions. Master logging markdown.

## 🗺️ Master Trace Map: Workspace Architecture (Gen1 - Gen11)

Below is the definitive index mapping every research generation (internal index) to its corresponding isolated sibling folders inside the workspace. This maps out how the codebase transitioned from **State-Only** models to the state-of-the-art **Visual Flow Matching** models:

| Internal Index | Model/Code Folder | Test/Eval Folder | Key Period | What is it / Status | FLAG |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen0 (Baseline)** | [diffuser/](../diffuser) | [scripts/](../scripts) | Pre-April 2026 | Original DPCC Baseline model (after functional upgrades like wandb, resume training, etc.). | |
| **Gen1** | [flow_matcher/](../flow_matcher) | [FM_test/](../FM_test) | Early April 2026 | Early Flow Matching baseline (State-Only). Crucial math bug: reversed ODE trajectory during rollout. | |
| **Gen2** | [flow_matcher/](../flow_matcher) | [FM_test/](../FM_test) | Mid April 2026 | Basic Flow Matching engine with uniform time sampling in $[0, 1]$ (State-Only). | |
| **Gen2 (U-Net v2)** | [flow_matcher_unet_v2/](../flow_matcher_unet_v2) | [FM_Unet_v2_test/](../FM_Unet_v2_test) | Mid April 2026 | Built U-Net v2 backbone shell/path structure, but no material changes to net behavior (structural upgrades remained TODO). | |
| **Gen3 Upgrade 1** | [flow_matcher/](../flow_matcher) | [FM_hp_tune_test/](../FM_hp_tune_test) | Mid April 2026 | Action loss weight ($a_0$) hyperparameter tuning sweep. | |
| **Gen3 Upgrade 2** | [flow_matcher_v2/](../flow_matcher_v2) | [FM_v2_test/](../FM_v2_test) | Mid-to-Late April 2026 | **FM-v2**: Introduced continuous Beta distribution time prior sampling ($1 - \text{Beta}(\alpha=1.5, \beta=1.0)$) (State-Only). | |
| **Gen3 Upgrade 3** | [flow_matcher_v3/](../flow_matcher_v3) | [FM_v3_test/](../FM_v3_test) | Late April 2026 (up to Apr 20) | **FM-v3**: Introduced SafeFlow-style continuous-time model query semantics (State-Only). | |
| **Gen3v2 (ODE Solver Addon)** | [flow_matcher_v3_ode_selectable/](../flow_matcher_v3_ode_selectable) | [FM_v3_ode_selectable_test/](../FM_v3_ode_selectable_test) | April 21 – May 4, 2026 | Added advanced ODE solvers (`torchdiffeq`, RK4, Euler, Dopri5) with a dynamic override mechanism (State-Only). | finished |
| **Gen3v3 (Drifting Engine)** | [flow_matcher_v3_drifting/](../flow_matcher_v3_drifting) | [FM_v3_drifting_test/](../FM_v3_drifting_test) | May 12, 2026 | Drifting baseline recovery and path reconstruction (State-Only). | working on |
| **Gen3v4 (iMeanFlow)** | [flow_matcher_v3_imeanflow/](../flow_matcher_v3_imeanflow) | [FM_v3_imeanflow_test/](../FM_v3_imeanflow_test) | May 13, 2026 | **iMeanFlow (iMF)** planning/inference infrastructure (State-Only). | working on |
| **Gen3v5 (BNS Solver)** | Pending | Pending | Pending | **BNS Solver**: Boundary-constrained Noise-guided Solver (Pending Plan). | |
| **Gen4 (Abandoned Visual)** | [(Abandoned)flow_matcher_v3_avoiding_visual/](../(Abandoned)flow_matcher_v3_avoiding_visual) | [(Abandoned)FM_v3_avoiding_visual_test/](../(Abandoned)FM_v3_avoiding_visual_test) | Late April 2026 (Apr 25–28) | **Abandoned**. Coupled code and regression risks via direct D3IL source modifications. | |
| **Gen5 (Visual Aligning)** | [ddpm_encdec_vision_Legacy/ddpm_encdec_vision/](../ddpm_encdec_vision_Legacy/ddpm_encdec_vision) | [ddpm_encdec_vision_Legacy/ddpm_encdec_vision_test/](../ddpm_encdec_vision_Legacy/ddpm_encdec_vision_test) | May 12 – May 17, 2026 | **Legacy baseline** (archived). Based on the `ddpmact d3il base` (imitation framework). Succeeded only once and never returned good results since. | |
| **Gen6 (Visual DPCC)** | [ddpm_encdec_vision/](../ddpm_encdec_vision) | [ddpm_encdec_vision_test/](../ddpm_encdec_vision_test) | May 17, 2026 | **Visual-Aligning Differentiable MPC (DPCC Upgrade)**. Reused FMv3ODE's DPCC projection logic on top of the visual baseline, enforcing 6D absolute workspace constraints. | |
| **Gen6v2 (Old Abandoned Pending)** | [ddpm_encdec_vision/](../ddpm_encdec_vision) | [ddpm_encdec_vision_test/](../ddpm_encdec_vision_test) | May 17, 2026 | **Abandoned pending**. Dual-Backbone Calibration & Pipeline Orchestration. Will do it later. | |
| **Gen6v3 (Non-Visual Aligning)** | [diffuser/](../diffuser) | [diffuser_test/](../diffuser_test) | May 18, 2026 | **State-only non-visual aligning pipeline** for Gen6. Fixed 17D vs 20D proprioceptive mismatch. | |
| **Gen6v4 (Visual DPCC 9D)** | [diffuser_visual_aligning/](../diffuser_visual_aligning) | [diffuser_visual_aligning_test/](../diffuser_visual_aligning_test) | May 18, 2026 | **New Principle**: Migrated from the `ddpmact d3il base` (imitation) to the robust physical `dpcc base` using a unified 9D joint representation `[act(3) \| des_c_pos(3) \| c_pos(3)]` to enforce safety cage constraints directly on the simulator physics. | working on |
| **Gen7 (Visual Flow Matching)** | [fm_visual_aligning/](../fm_visual_aligning) | [fm_visual_aligning_test/](../fm_visual_aligning_test) | May 20, 2026 | **Continuous-time visual Flow Matching (FMv3ODE)**. Clean copy-modify sibling transition from proofed Gen6V4 to continuous-time FM ODE engine with Beta(1.5, 1.0) time sampling and velocity target training. | working on |
| **Gen8 (iMeanFlow Visual Engine)** | ~~Pending~~ <br>[imf_visual_aligning/](../imf_visual_aligning) | ~~Pending~~ <br>[imf_visual_aligning_test/](../imf_visual_aligning_test) | ~~Planned~~ <br>June 2026 | ~~**Planned Extension**: Add the iMeanFlow (iMF) engine as an alternative to the Gen7/Gen6v4 visual aligning pipelines. **Key Milestone**: This represents a major leap in making the core ML engine completely switchable, effectively merging the architectural capabilities of Gen6v4 and Gen7, and now seamlessly integrating iMF. (Maybe too complex and causing hidden bugs, so keep current in diff folders entry. i.e. no more implement pending, low ranking)~~ <br><br> **IN PROGRESS (Partial)**: iMeanFlow (iMF) Visual Engine. Successfully added the iMeanFlow (iMF) engine. Merged architectural capabilities of Gen6v4 and Gen7 into a unified iMeanFlow ODE inference engine. Supports Official DiT backbone, MeanFlow-JVP objective, and Interval-CFG. | ~~planning~~ <br>working on |
| **Gen9 (Visual Avoiding Env)** | ~~Partial~~ <br>[fm_visual_avoiding/](../fm_visual_avoiding) | ~~Partial~~ <br>[fm_visual_avoiding_test/](../fm_visual_avoiding_test) | ~~In Progress~~ <br>June 2026 | ~~**PARTIAL COMPLETION (30 May 2026): Camera Environment Capture Confirmed**. Created a visual avoiding dataset and environment in MuJoCo by adding cameras to capture visual data from the existing avoiding expert trajectories. **COMPLETED**: Camera capture pipeline. **PENDING**: Modify and test the Gen6v4 and Gen7 visual aligning models in this new environment to learn and validate MuJoCo environment generation. **Features**: 1. Flexible 3D (xyz) and 2D (xy) tensor switch for training/learning (evaluation consistently outputs 3D plots regardless of tensor shape to maintain identical behavior). 2. Add parameters for avoiding env tensor inputs, since aligning-specific inputs (box angle/position) are not present.~~ <br><br> **IN PROGRESS (Partial)**: Visual Avoiding Pipeline. DPCC and FM visual models ported to avoiding tasks with single-camera observations and 6-D trajectories. Includes FiLM v2 true architecture updates. | ~~in progress~~ <br>working on |
| **Gen10 (DDPM ACT / Transformer)** | Pending | Pending | Planned | **Planned Upgrade**: Add a new DDPM ACT backbone (the generally best and theoretically most powerful model in D3IL). This will upgrade the Gen6v4 (Diffusion) and Gen7 (FM) U-Net backbones to a VAE + Transformer (or superior mathematical ML architecture). **Note**: If Gen8 is successful in establishing a modular engine switch, Gen10 will be directly based on Gen8 and should only focus on the ML architecture design itself (unless paradigm shifts like action chunking dictate a broader system redesign beyond the model backbone). | |
| **Gen11 (UAV Vis-Traj in MuJoCo)** | ~~Partial~~ <br>[flow_matcher_v3_uav/](../flow_matcher_v3_uav) | ~~Partial~~ <br>[FM_v3_uav_test/](../FM_v3_uav_test) | ~~In Progress~~ <br>June 2026 | ~~**PARTIAL COMPLETION (30 May 2026): UAV Model Migration (Epoch 1) Completed**. Implemented a UAV visual-trajectory planning environment manually in MuJoCo with abstract 3D geometric constraints. **COMPLETED**: Skydio X2 model assets (XML, mesh, texture) migrated from upstream `mujoco_menagerie` and patched for MJPC tasks. **PENDING**: Python environment class, residual/transition logic, and training pipeline implementation. Features a custom drone dynamic model and 0-shot evaluation on random start/end locations under geometric constraints, utilizing a visual-aligning-style backbone.~~ <br><br> **IN PROGRESS (Partial)**: UAV Flow-Matching & DPCC. Full closed-loop 33 Hz receding-horizon control for UAV trajectory planning in MuJoCo. Includes Cascaded PID trackers, MJPC thrust control, real-time logging, and DPCC safety projection on constraint spaces. | ~~in progress~~ <br>working on |
| **Gen11+ / X** | [/workspaces/HardFlow](/workspaces/HardFlow) | Pending | June 2026 | Integrating /workspaces/HardFlow into FMPCC. | |
| **Gen11+ / X  (HF + iMF)** | [/workspaces/HardFlow](/workspaces/HardFlow) | Pending | July 2026 | A new model of HardFlow + IMF, which includes HardFlow individual evaluation tests and the HF + IMF integrated framework. | |

***

## 🛠️ Auxiliary Infrastructure & Benchmark Suites

In addition to the main model training/evaluation pipelines, the repository hosts specialized auxiliary systems for ODE precision benchmarking, result aggregation (Data Analysis), and cluster deployment (SLURM orchestrators):

| Infrastructure Component | Folder / Script Path | Key Purpose | Relevant Phase / Period |
| :--- | :--- | :--- | :--- |
| **ODE Solver Benchmarks** | [flow_matcher_v3_ode_selectable/](../flow_matcher_v3_ode_selectable) (and scripts inside) | Comparative precision analysis of Euler, RK4, and Oracle (Dopri5) solvers on a locked noise basis (`global_x_init`). | Gen3v2 (Late April 2026) |
| **Trajectory Quality Visualizer** | `traj_gen_script_for_v4.py` (inside [flow_matcher_v3_ode_selectable_test/](../FM_v3_ode_selectable_test)) | Overlays unnormalized latent robotic plans on environmental half-space/obstacle constraints for visual precision-drift auditing. | Gen3v2 U4.1 (April 22, 2026) |
| **Data Analysis & Plotting** | [Data_Analysis/](../Data_Analysis) | Dynamic plotting scripts for generating thesis-ready success rate heatmaps and latency charts. <br><br> **UPDATE (June 2026):** Includes v3 cross-experiment combined analysis via comma-separated pathing. | Ongoing (April - May 2026) |
| **Colab Plotting Suites** | [Results_and_Data_Analysis_Colab_T4/](../Results_and_Data_Analysis_Colab_T4) & [ipynbs_Colab/](../ipynbs_Colab) | Plotting pipelines and Google Colab T4 GPU integration scripts. | Ongoing (April - May 2026) |
| **Cluster Job Orchestrators** | [Slurm_Codes/](../Slurm_Codes) | Pipeline runner scripts (SBATCH shell scripts) for GPU cluster node dispatch (e.g. `Visual_Aligning/` pipeline). | Gen3v2 Remote Migration & Gen5/Gen7 Visual Aligning (Ongoing) |
| **Real-Time Simulation Recording Ideas** | ~~[REALTIME_RECORDING/IDEAS.md](REALTIME_RECORDING/IDEAS.md)~~ <br>[realtime_recording/](../realtime_recording/) | ~~Need to analyze real-time recordings (not only GIFs, just ideas!)~~ <br><br> **COMPLETED**: Portable `RTRecorder` implemented across 10 evaluation pipelines logging `total_ms`, `fm_ms`, `proj_ms`, and track errors to `realtime_*.log`. | ~~Pending~~ <br>working on |
| **Remote Log Sync Pipeline** | [Slurm_Codes/download_remote_logs/](../Slurm_Codes/download_remote_logs/) | Automated bash pipeline (`export_to_laptop.sh`) using `rsync`/`scp` to cleanly package and download evaluation artifacts and cluster logs to a local machine. | Ongoing |

***

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
4. > [!CAUTION]
5. > **ODE Setup Warning**: It has been audited that FMv2 (`flow_matcher_v2`) ignores eval-time ODE step changes due to a "Pickle Lock" (it uses the value saved during training). 
6. > Thus, any previous test claiming **ODE=20** for FMv2 was actually running at **ODE=10** (the training default).
7. > This was finally resolved in **## Gen3v2u2: RK4 Solver Validation & Loading Hotfix (23. April)** via the **Dynamic Override** mechanism for FMv3-selectable models.
8. Test markings:
9. "Beta Time" marks beta-only test.
10. "ODE=10" marks beta-time plus eval ODE=10 test.

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

### Visual Pipeline Bug Fixes (Gen5 Phase 1)
1. **Hydra Setup**: Fixed Device Serialization (converted `torch.device` to primitive strings) and Recursion Logic (`_recursive_: False`) to properly interface nested parameters with D3IL's hardcoded manual instantiation.
2. **CUDA Fork Crashes**: Initialized `Dataset` strictly on `cpu` RAM to prevent PyTorch `DataLoader` workers from crashing due to unshareable CUDA contexts.
3. **Tuple Batching**: Rewrote `batch_to_device` in D3IL array tools to dynamically support standard PyTorch `list`/`tuple` batches alongside existing `namedtuples`.

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

#### V4.2: Gen3v2: Trajectory Quality Audit & Fairness Hotfix (27. April)

Keywords: production mode fairness, shared noise basis, per-batch audit, raw environment restoration.

**Code Hotfixes**:
1.  **Noise Fairness**: Refactored `benchmark_ode_solvers_v4.py` so that **both** `math` and `production` modes share the exact same noise basis across all solvers in a trial. Euler, RK4, and Oracle now solve the **identical random challenge**.
2.  **Timing Determinism**: Fixed the trial loop so all trials in a run use the same mathematical workload. Latency averages are now 100% stable.
3.  **Visual Audit Upgrades**:
    *   **Per-Batch Audits**: Added `batch_comparison_BX.png` plots to isolate 1:1 solver comparisons on specific noise vectors.
    *   **Raw Env Restoration**: Stripped "Projection" obstacles from plots to show only original dataset obstacles (Red Circles).
    *   **High-Res Quality**: Upgraded to 300 DPI, SVG output, and reserved Red for the Oracle.

**Result of Today's Test**:
*   **Status**: **Not finished, Colab time out.**
*   **Observations**: 
    *   Tested Math Mode (raw drift) vs. Production Mode (locked start point).
    *   **Drift Sensitivity**: In Math Mode (no pullback), the Euler solver often shows better alignment to Dopri5 at the "0,0" starting point, but in other random start positions, the results differ significantly; in some cases, RK4 clearly demonstrates superior precision.
    *   **Pending**: Full batch=20 audit in Math Mode to quantify the exact influence of different start-point noise on ODE solver error.

#### V4.3: Gen3v2: Safety Shield Audit & Plotter Rebuild (28. April)

Keywords: Safety Shield Audit, Corrupted Unnormalization fix, Rebuild in progress.

**Objective**: 
1. Audit the "Observation Snap" ($t=0$) logic to verify if the "Jumps" seen in plots were mathematical errors or visualization bugs.
2. Verify the 10-step internal ODE "Conditioning" loop.

**Findings (rom [06_audit_and_fixes_summary.md]**:
1. **Snap Logic**: Confirmed the code correctly anchors the initial state ($t=0$) across all 10 internal thought steps.
2. **The "Jump" Bug**: Discovered that the weird visual jumps were **NOT** in the model, but in the plotter's **corrupted unnormalization** (slicing the wrong dimensions of the 4D tensor).

**Status**: **NOT FINISHED.** 
*   The visualization code is currently being rebuilt to implement the "Corrected Dimensions" logic from the 06 audit document.
*   The final Comparison Mission is on hold until the new plotter is verified.

#### V4.4: Gen3v2: Production Anchoring & Plotter Stabilization (30. April)

Keywords: Double Anchor, Action Snapping, strict assertions, zigzag fix, SUCCESS.

**Final Rebuild & Stabilization**:
1.  **Double Anchor Safety Shield**: Re-implemented the anchoring logic to snap **both** the first Observation and the first Action (Waypoint) at $t=0$ to the physical robot position.
2.  **Persistent ODE Snapping**: Updated the integrators to re-anchor Step 0 after every internal ODE step, ensuring zero numerical "leakage" at the start of the plan.
3.  **Plotter Scaling Fix**: Corrected the visual dimension slicing (`[action_dim:]`) and ensured the use of the `observations` normalizer. This permanently resolved the "zigzag" artifacts and scaling mismatches.
4.  **Strict Safety Assertions**: Added hard runtime checks in both the benchmark and plotter scripts. The pipeline now automatically **ABORTS** and throws a `CRITICAL` error if it detects any drift (> 1e-4) in the initial state.
5.  **Visual Verification**: Confirmed that the Green Dot (Solver Start) now perfectly overlays the Yellow Star (True Start) across all solvers (Euler, RK4, Dopri5).

**Status**: **TEST PASSED (Production Grade)**
*   The V4 pipeline is now scientifically hardened, visually precise, and safe for automated large-scale benchmarking.



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

## Gen3v2: Metadata Root Leak Hotfix (24. April)

Keywords: metadata leak, root directory cleanup, Parser.savepath fix, resume indexing.

1. **Problem**: Discovered that `args_resume_N.json` files were leaking into the project root directory (reaching index 272). This was caused by the `Parser` class in `utils/setup.py` failing to synchronize its internal `self.savepath` with the experiment-specific `args.savepath`.
2. **Fix**: Updated `flow_matcher_v3_ode_selectable/utils/setup.py` to ensure `self.savepath` is updated in the `mkdir` method before saving. This forces the metadata into the correct experiment log folder.
3. **Outcome**: Future runs will no longer pollute the root directory, and run configurations will be properly encapsulated within their respective trial folders.

## Gen3v2: FMv3 aw & DPCC Step Matrix Tests (27. April)

Keywords: ODE steps (10 vs 20), action weight (aw1 vs aw10), DPCC diffusion floor, FM-VF efficiency.

1. **FMv3 ODE Step Sensitivity (10 vs 20)**:
   - **Parameters**: `aw=1`, `seed=6`.
   - **Observation**: Increasing from 10 to 20 ODE steps provided no significant improvement in environment steps or success; in some edge cases, behavior was slightly worse.
   - **Conclusion**: The FMv3 Vector Field is sufficiently smooth/accurate at 10 steps; additional integration resolution yields diminishing returns for this task.
   - **Path**: `\Results_and_Data_Analysis\Data_Analysis\Eval_Seed6_FMv3(aw1ODE10)vs_FMv3_aw1_ODE_20` 

2. **FMv3 Action Weight Ablation (aw=1 vs aw=10)**:
   - **Parameters**: `ODE=20`, `seed=6`.
   - **Observation**: Almost no measurable influence on computation time or environment steps across most criteria.
   - **Conclusion**: The inference quality is robust to these `action_weight` variations.
   - **Path**: `\Results_and_Data_Analysis\Data_Analysis\Eval_seed6_FMv3_aw1_ode_20_vs_FMv3_aw10_ode20`.

3. **DPCC Diffusion Step Floor (26. April)**:
   - **Observation**: Reducing DPCC to 10 diffusion steps caused a severe degradation in all performance criteria.
   - **Conclusion**: **FM-VF vs. Diffusion Efficiency**: We can achieve high-quality planning with lower step counts (10) in a well-trained FM Vector Field, whereas traditional Diffusion (DPCC) requires higher step resolution (20+) to maintain plan quality.
   - **Path**: `\FM-PCC\Results_and_Data_Analysis\Data_Analysis\Eval_seed_6_FMv3_aw10_ode20_vs_DPCC_vs_DPCC_Step10`.

4. **Training Status Update**:
   - **FMv3 (aw10, ODE10)**: Training is currently **in progress** (aimed at a direct 1:1 "Step-Floor" comparison with DPCC 10-step results).


---

### Midpoint5 vs ODE10 euler (same NFE test)

train FMv3 midpoint 5 compare to ODE10 euler
(after the benchmark_test, individual midpoint 5 compare to ODE10 euler, time, accuracy, traj.! (in v4 folder))

---

## Gen3v2: Remote SLURM Migration & Config-Code Alignment Hotfix (29. April)

Keywords: SLURM migration, vmknoll cluster, AttributeError hotfix, n_diffusion_steps fallback, pro-logging.

1. **Remote Migration**: Successfully migrated the development environment from Google Colab to a remote SLURM-managed Linux cluster (`vmknoll`).
2. **Environment Setup**: Configured a dedicated Conda environment (`FMPCC`, Python 3.10) and established a "Headless Rendering Standard" using EGL (`MUJOCO_GL="egl"`, `PYOPENGL_PLATFORM="egl"`) for GPU-accelerated simulation on compute nodes.
3. **Environment Stabilization**: Standardized Conda pathing and established unified `PYTHONPATH` logic across all job scripts to ensure zero-modification parity with the Colab baseline.
4. **Log Infrastructure**: Implemented a "Pro-Logging" wrapper (`submit.sh`) with date-based subdirectories and a `latest.log` symlink for high-speed job monitoring.
5. **Trainer Robustness Hotfix**: 
    - **Problem**: Identified an `UnboundLocalError` in `utils/training.py` where the script crashed if a training epoch was too short to trigger a validation phase (common in "smoke tests").
    - **Fix**: Updated the `Trainer` class to safely track and log the last known test loss, ensuring stability for short debug runs.
6. **Evaluation Plotter IndexError Hotfix**: 
    - **Problem**: Identified an `IndexError` in `eval_flow_matching_v3_ode_selectable.py` where the script crashed during 2D axes indexing if `n_trials` was set to 1 (matplotlib squeezes the array by default).
    - **Fix**: Applied `squeeze=False` to `plt.subplots` calls for multi-trial grids, ensuring the axes object is always a 2D array regardless of trial count.
7. **Validation Success**: 
    - **Status**: Verified that SLURM training and evaluation jobs are passing on the `vmknoll` cluster.
    - **W&B Integration**: Confirmed that Weights & Biases (W&B) logging is functional, syncing metrics from the remote nodes to the project dashboard.

## Gen3v2: Eval Console Logging Upgrade (29. April)

Keywords: Tee logger, eval console logging, evaluation output persistence.

1. **Problem**: Evaluation outputs (Success rates, violation metrics, etc.) printed to the console were not being saved anywhere, making it difficult to review results after a job finished.
2. **Fix**: Implemented a `Tee` logger class in `eval_flow_matching_v3_ode_selectable.py` that redirects `sys.stdout` to both the console and a variant-specific log file (`eval_{variant}.log`).
3. **Execution Safety**: Wrapped the main evaluation variant loop in a `try...finally` block to ensure the console output is always restored even if an evaluation crashes.
4. **Outcome**: Every evaluation run now automatically generates a text-based log file in the same `results/` folder as its images and `.npz` data, providing a permanent record of the console output.

## Gen3v2: FMv3-ODE Configuration & Folder Naming Cleanup (29. April)

Keywords: K-less training, folder naming logic, dead parameter safety, diffusion_loadpath.

1. **Problem**: FMv3-ODE training folders were incorrectly labeled with `_K20` or `_K10` labels, which are mathematically irrelevant for continuous-time Flow Matching training and caused confusion in model loading.
2. **Fix (Folder Naming)**: Commented out all step-related parameters (`n_diffusion_steps`, `flow_steps_v3`) in the `flow_matching_v3_ode_selectable` training block. This allows the `watch` logic to omit the `K` label entirely, resulting in cleaner `H8_D...` folders.
3. **Fix (Load Path)**: Updated `diffusion_loadpath` in the planning block to remove the `_K{...}` segment. Evaluation scripts now correctly load models from the "K-less" training folders while still saving evaluation results into `_K10` folders (where step count matters).
4. **Safety Audit (Dead Parameters)**: Verified that removing these parameters from the config is 100% safe:
    - **Training**: `train_flow_matching_v3_ode_selectable.py` uses `getattr(args, '...', default)` for all step-related keys.
    - **Model Math**: `GaussianDiffusion` (v3) uses floating-point time $t$ for training, which bypasses all discrete step-count calculations (verified in `_time_from_timestep`).
5. **Outcome**: The codebase is now "penetrated" against naming bugs. Training is streamlined, and evaluation correctly handles its own ODE step configuration while finding models reliably.

## Gen3v2: TQDM Log Pollution Hotfix (30. April)

Keywords: tqdm log pollution, SLURM stdout fix, mininterval infinity, cleaner logs.

1. **Problem**: In non-interactive SLURM logs, progress bars generated thousands of lines of redundant output (one line per step refresh), making log files nearly impossible to audit.
2. **Fix**: Injected `mininterval=1e10` into the `tqdm` constructor across all training utility files to suppress intermediate updates.
3. **Outcome**: Progress bars now stay silent during the loop and only pop a single "100%" completion line at the end of each epoch. This eliminates thousands of lines of log "shits" while ensuring all critical prints and errors remain visible.

**Affected Files & Lines:**
- `diffuser/utils/training.py`: Line 117
- `flow_matcher_v3_ode_selectable/utils/training.py`: Line 119
- `flow_matcher/utils/training.py`: Line 117
- `flow_matcher_v2/utils/training.py`: Line 117
- `flow_matcher_unet_v2/utils/training.py`: Line 117
- `flow_matcher_v3/utils/training.py`: Line 117

## Gen3v2: W&B Artifact Upload, TQDM Cleanup, & Root Leak Fix (1. May)

Keywords: W&B crash, AttributeError, storage optimization, TQDM log pollution, metadata root leak, global setup fix.

1. **Problem (W&B)**: Multi-seed training jobs crashed after the first seed due to an `AttributeError` (`run.Artifact` typo) and an `import wandb` scoping issue.
2. **Fix (W&B)**: Corrected `run.Artifact` to `wandb.Artifact`, moved imports to global scope, and commented out large weight uploads (`state_best.pt`) to save cloud storage.
3. **Problem (TQDM)**: Progress bars generated thousands of redundant lines in SLURM logs because `update(1)` was called every step on non-interactive terminals.
4. **Fix (TQDM)**: Implemented a "Refined 1-Line-Per-1,000-Steps" logic. Progress bars now only update at the end of every 1,000 steps or at the epoch's end, ensuring clean logs.
5. **Problem (Metadata Leak)**: Training scripts were "shitting" `args_resume_N.json` files into the project root instead of the experiment folder.
6. **Fix (Metadata)**: Synchronized `self.savepath` in `Parser.mkdir()` across all setup utilities (including DPCC). Metadata is now correctly encapsulated in seed-specific log folders.
7. **Outcome**: Training stability, log clarity, and filesystem hygiene are fully restored.

**Affected Files (W&B Fix):**
- `scripts/train.py`
- `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py`
- `FM_Unet_v2_test/train_FM_Unet_v2.py`
- `FM_v3_test/train_FM_v3.py`
- `FM_v2_test/train_FM_v2.py`
- `FM_test/train_FM.py`
- `FM_hp_tune_test/train_FM_hp_tune.py`

**Affected Files (TQDM Fix):**
- `diffuser/utils/training.py`
- `flow_matcher_v3_ode_selectable/utils/training.py`
- `flow_matcher/utils/training.py`
- `flow_matcher_v2/utils/training.py`
- `flow_matcher_unet_v2/utils/training.py`
- `flow_matcher_v3/utils/training.py`
- `d3il/agents/models/bet/libraries/mingpt/trainer.py`

**Affected Files (Metadata Fix):**
- `diffuser/utils/setup.py` (DPCC)
- `flow_matcher/utils/setup.py`
- `flow_matcher_v2/utils/setup.py`
- `flow_matcher_v3/utils/setup.py`
- `flow_matcher_v3_ode_selectable/utils/setup.py`

## Gen3v2: Slurm Job End Logging & Eval Time Limit Update (2. May)

Keywords: Job End logging, EXIT trap, submit.sh Job ID, evaluation time limit (8h).

1. **Job End Logging**: Standardized all sbatch scripts (`eval_dpcc`, `eval_fmv3`, `train_dpcc`, `train_fmv3`, `verify_env`, `load_results`) to use an `EXIT` trap for printing a `JOB END` timestamp. This ensures end-of-job visibility in logs even if the script aborts due to `set -e`.
2. **Evaluation Time Limit**: Increased the `#SBATCH --time` limit from **2 hours to 8 hours** for all evaluation scripts to prevent timeouts during large benchmark sweeps.
3. **Submission Wrapper Enhancement**: Updated `submit.sh` to capture the Job ID from the `sbatch --parsable` output and provide cleaner terminal feedback.
4. **Template Standardization**: Updated `2026_04_30_job_template.sh` to include the new logging standards, ensuring future scripts inherit these improvements.
5. **Pipeline Submission Fix**: Resolved a `sbatch: error: No partition specified` issue for the `fmv3_ode_pipeline.sh` script by adding mandatory SBATCH headers and standardizing it with the "Pro-Logging" architecture.
6. **Smart Unified Logging Upgrade**: Implemented a session-based logging system. `submit.sh` now exports `SUBMIT_TIME/DATE` metadata, allowing pipeline managers and their sub-jobs to share the exact same timestamp prefix. This ensures all logs from a single pipeline run are perfectly grouped and searchable in the filesystem.

## Gen3v2u3: Evaluation Persistence & Aggregation Hotfix (3. May)

Keywords: gen3v2u3 critical, all_seeds aggregation, full data persistence, obs_all saving, modular evaluation.

1.  **Full Data Persistence (CRITICAL)**: Resolved the "Ephemeral Result" bottleneck by modifying evaluation scripts to save raw trajectory coordinates (`obs_all`) and actions (`act_all`) for all trials into `.npz` files. 
2.  **Aggregation Mode**: Implemented the `--aggregate-only` flag, allowing users to regenerate `all_seeds` summary plots instantly from disk data without re-running model inference or MuJoCo.
3.  **Slurm Parallelization**: Added `--seed` command-line support to allow running individual seeds as separate Slurm jobs, which can then be retrospectively aggregated into a single summary plot.
4.  **Baseline Parity**: Applied these upgrades to both `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` and the baseline `scripts/eval.py` (Note: Tee logger fix for baseline injected on 4. May).
5.  **Audit Visibility**: Created a dedicated audit report at `logs_in_develop/gen3v2u3_hot_fix_eval_data_saving/hotfix_report.md` detailing the "Before vs. After" architectural shift.

## Gen3v2  misc hotfix: Evaluation Configuration Metadata Cleanup (4. May)

Keywords: gen3v2u4, metadata cleanup, redundant args logging, Parser architecture, evaluation noise reduction.

1.  **Redundant Logging Fix**: Eliminated the generation of hundreds of confusing `args_resume_X.json` files during evaluation runs.
2.  **Conditional Parser Save**: Re-architected the `Parser` class in `utils/setup.py` to only enable automatic configuration saving when the experiment type is explicitly set to `'train'`.
3.  **Module Standardization**: Synchronized this fix across both the core `diffuser` module and the `flow_matcher_v3_ode_selectable` module to ensure consistent logging behavior.
4.  **Audit Visibility**: Documented the problem and technical fix in `logs_in_develop/gen3v2_hotfix_arg_resume_eval/hotfix_report.md`.

## Gen3v2 misc hotfix: W&B Run Naming & Grouping Stabilization (4. May)

Keywords: wandb naming logic, path-based identity, descriptive groups, experiment tracking.

1.  **Problem**: W&B runs were cryptically named `{dataset}-seed-{seed}`, making it impossible to identify the model type or hyperparameters without deep inspection.
2.  **Fix**: Updated `scripts/train.py` and `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py` to derive run names from the relative save path (e.g., `avoiding-d3il-diffusion-H8_K20-S5`).
3.  **Grouping**: Implemented automatic W&B grouping by experiment folder, ensuring all seeds of a configuration are clustered together.
4.  **Visibility**: Created a detailed hotfix report at `logs_in_develop/gen3v2_hotfix_wandb_naming_better/hotfix_report.md`.

## Gen3v2 misc hotfix: Plot Output Path Standardization (4. May)

Keywords: gen3v2u4, plot path standardization, load_results cleanup, experiment encapsulation.

1.  **Standardized Plot Paths**: Redirected plot outputs from the script directory/CWD to a dedicated `plots/load_results_output_all_seeds` subfolder within the experiment log directory.
2.  **Dynamic Resolution**: Implemented dynamic `plot_path` logic in `load_results.py` and its FMv3 variant to ensure plots are always saved relative to the loaded data.
3.  **Audit Visibility**: Detailed the changes and rationale in `logs_in_develop/Gen3v2/gen3v2u4_load_results_path_fix/load_results_path_fix.md`.

---

## Drifting Project Integration & Evaluation (May 2026)

Keywords: drifting, motion generation, VAE latent, MAE models, visual-free baseline.

### Objective
Integrate the Drifting project (latent-space motion generation using VAE/MAE) as a baseline comparison point for FM-based planning. This complements the Flow Matching pipeline by offering an alternative generative model architecture for trajectory synthesis.

### Components
1. **MAE Model Training** (`train_mae.py`): Vision transformer-based masked autoencoder for motion encoding
2. **Generator Models** (`models/generator.py`): Generative networks for latent-space motion synthesis
3. **Inference Pipeline** (`inference.py`): End-to-end latent motion generation and decoding
4. **ConvNeXt Feature Extractor** (`models/convnext.py`): Backbone for visual feature extraction
5. **Dataset Management** (`dataset/`): VAE and latent motion dataset handling

### Integration Status
- **Code Location**: `/workspaces/drifting/`
- **Purpose**: Baseline comparison (non-FM motion generation via latent diffusion/VAE)
- **Evaluation**: To be integrated into FM-PCC evaluation pipelines for relative performance benchmarking

---

## Data Analysis (DA) Tool Implementation (May 12, 2026)

Keywords: DA tool, evaluation aggregation, Pareto frontier, thesis-focused analysis, automated reporting.

### Problem Statement
FM v3 ODE-Selectable evaluation produced **834+ .npz result files** across:
- **5 random seeds** [6, 7, 8, 9, 10]
- **18 projection variants** (dpcc-c/r/t, diffuser, gradient, post_processing, model_free, + tightened variants, + dt variants)
- **4 constraint types** (halfspace, obstacles, dynamics, bounds)
- **3 halfspace geometries** (top-right-hard, top-left-hard, both-hard)

**Challenge**: Manual visualization and comparison across all dimensions was impossible. A systematic analysis pipeline was required.

### Solution Architecture

**Core Modules** (in `/workspaces/FM-PCC/Data_Analysis/DA_Code/`):

1. **data_loader.py**: 
   - Auto-discovers directory tree structure (seed → halfspace variant → .npz files)
   - Loads all .npz result files
   - Generates detailed loading report (files found/loaded/failed)

2. **aggregator.py**:
   - Aggregates metrics across all seeds (computes mean, std, min, max)
   - Creates views by variant, constraint type, halfspace variant
   - Builds pivot tables for cross-dimensional analysis
   - Generates per-variant rankings

3. **visualizer.py**:
   - **Pareto Frontier** (`00_pareto_frontier_accuracy_vs_time.png`): Accuracy vs. Time tradeoff with color-coded variants
   - **Variant Comparisons**: Bar charts by metric
   - **Constraint Analysis**: Grouped performance by constraint type
   - **Heatmaps**: Variant × Constraint success rates
   - **Boxplots**: Seed-to-seed variability analysis
   - **Efficiency Plots**: Time vs. Accuracy scatter
   - Publication-quality output (300 DPI, matplotlib styling)

4. **reporter.py**:
   - `results_summary.txt` (human-readable rankings and statistics)
   - `results_by_variant.csv` (variant-level aggregation)
   - `results_by_constraint.csv` (constraint-type aggregation)
   - `results_by_halfspace.csv` (halfspace-geometry aggregation)
   - `detailed_results.csv` (all data points for custom analysis)

5. **config.py**:
   - Default seeds, variants, constraint types, halfspace variants
   - Plot styling constants (colors, fonts, DPI)
   - Metric definitions and labels

6. **utils.py**:
   - Logger setup (console + file output)
   - File path utilities
   - Directory discovery helpers

7. **main_da.py** (Entry Point):
   - CLI interface with argument parsing
   - Coordinates data loading → aggregation → reporting → visualization
   - Timestamp-based output folder organization
   - Error handling and summary reporting

### Key Features

- **Automatic Data Discovery**: No manual file enumeration needed; script finds all .npz files in nested structure
- **Robustness**: Missing/corrupted files logged but don't halt execution
- **Flexible Input**: CLI arguments for seeds, variants, constraint types; defaults auto-apply
- **Thesis-Focused**: Pareto frontier plot highlights main variants (dpcc-c/r/t) vs. baseline (diffuser)
- **Fast Execution**: ~1-2 minutes for full analysis (or ~30s with `--no-plots` flag)
- **Comprehensive Output**: 10+ plots, 4 CSV tables, 1 human-readable summary, detailed logs

### Usage Example

```bash
# Basic analysis
python Data_Analysis/DA_Code/main_da.py \
    --input-path FM_v3_ode_selectable_test \
    --output-path ./analysis_results

# Thesis-focused (main variants only)
python Data_Analysis/DA_Code/main_da.py \
    --input-path FM_v3_ode_selectable_test \
    --variants dpcc-c,dpcc-c-tightened,dpcc-r,dpcc-r-tightened,dpcc-t,dpcc-t-tightened,diffuser

# Quick check (no plots)
python Data_Analysis/DA_Code/main_da.py \
    --input-path FM_v3_ode_selectable_test \
    --no-plots
```

### Output Structure

```
20260512_143022_FM_V3_ODE_Analysis/
├── plots/
│   ├── 00_pareto_frontier_accuracy_vs_time.png    ← THESIS MAIN FIGURE
│   ├── 01_variants_n_success_and_constraints.png
│   ├── 02_constraints_*.png
│   ├── 03_heatmap_variant_constraint_*.png
│   ├── 04_efficiency_*.png
│   ├── 05_boxplot_seeds_*.png
│   └── [10+ plots total]
├── results_summary.txt                             ← HUMAN-READABLE
├── results_by_variant.csv
├── results_by_constraint.csv
├── results_by_halfspace.csv
├── detailed_results.csv                            ← ALL DATA POINTS
└── logs/
    ├── analysis.log
    ├── data_loading.log
    └── warnings.log
```

### Thesis Integration

**Primary Output for Results Section**:
- **Pareto Frontier Plot**: Shows accuracy (Y) vs. time (X) with dpcc-c/r/t highlighted in red/orange/yellow and diffuser (baseline) in blue
- **Variant Rankings**: Top 10 methods by goal + constraint success with ± error bars
- **Constraint Breakdown**: Performance by constraint type (halfspace, obstacles, dynamics, bounds)

**Supplementary Material**:
- All 10+ plots for publication
- CSV tables for detailed metrics
- Seed variability analysis (proving robustness across random initializations)

### Documentation

**User Guides** (in `/workspaces/FM-PCC/logs_in_develop/DA_Code/`):
- **DA_PLAN.md**: Full technical plan (objectives, architecture, phases, success criteria)
- **MISSION_BRIEFING.md**: Research context and thesis motivation
- **USAGE.md**: Step-by-step usage guide with 6+ practical examples

### Success Criteria Met

✅ Script auto-discovers and loads all 834+ .npz files  
✅ Aggregates metrics across 5 seeds with statistics  
✅ Generates 10+ publication-quality plots (300 DPI)  
✅ Produces thesis-ready figures (Pareto frontier)  
✅ Highlights main methods (dpcc-c/r/t) in color-coded comparison  
✅ Shows baseline comparison (diffuser as raw ML reference)  
✅ Execution time < 2 minutes  
✅ Detailed logging of data loading and processing  
✅ CSV export for Excel and statistical tools  

### Status

**COMPLETE** (May 12, 2026) - Ready for thesis analysis and result generation

## Gen3v3u5: FMv3-ODE Standardized Naming & Snapshot Hotfix (4. May)

Keywords: standardized naming, descriptive folder paths, Smart Config Snapshot, full traceability, hyperparameter auditing.

1.  **Standardized Folder Naming**: Refactored the naming logic for FMv3-ODE to include crucial tuneable parameters. 
    - **Training**: Paths now reflect Beta sampling (`a`, `b`) and action weights (`aw`) (e.g., `H8_D..._a1.5_b1.0_aw1`).
    - **Planning**: Paths include the solver method (`M`) (e.g., `H8_K10_Meuler_D...`), keeping the paths clean of training-only metadata while ensuring uniqueness.
2.  **Smart Config Snapshots**: Implemented an automated archiving system in `Parser.mkdir()`. Every training and evaluation run now captures a snapshot of the exact `.py` and `.yaml` configuration files used.
    - **Archive Path**: `logs/.../seed_X/config_snapshot_{name}/`
    - **Files Captured**: `avoiding-d3il.py`, `projection_eval.yaml`.
    - **Force Overwrite (Updated 4. May)**: Snapshots now overwrite on every run (matching evaluation behavior) and include a trailing timestamp file to verify copy completion.
3.  **Sync Logic**: Updated `diffusion_loadpath` to automatically resolve the new descriptive training folder names, ensuring zero-configuration loading for evaluation.
4.  **Audit Visibility**: Created detailed reports at `logs_in_develop/Gen3v2/Gen3v3u5_log_output_path_config_update/`.

## Gen3v3 hotfix: Nested Evaluation Folder Structure (6. May)

Keywords: nested paths, evaluation isolation, parent-model-attribution.

1.  **Nesting Fix**: Standardized the FMv3-ODE evaluation output to be nested under a subfolder named after the training model's hyperparameters.
    - **New Structure**: `logs/.../plans/flow_matching_v3_ode_selectable/[TRAIN_PATH]/[EVAL_PATH]/`
2.  **Implementation**: Accomplished via a single-line concatenation in `config/avoiding-d3il.py` using lazy f-strings.
3.  **Audit Visibility**: Updated documentation in `logs_in_develop/Gen3v2/Gen3v3u5_log_output_path_config_update/config_update_report.md`.

## Gen3v3 hotfix: Strict YAML Threshold Parsing Hotfix (8. May)

Keywords: strict config parsing, abort on missing, no silent defaults, diffusion_timestep_threshold.

1. **Problem**: The evaluation threshold (`diffusion_timestep_threshold`) in `avoiding-d3il.py` used a `try/except` block with a silent fallback default of `0.5`. This was identified as catastrophic because missing or misconfigured YAML settings would silently run with the wrong threshold while labeling the folder as `T0.5`.
2. **Fix**: Replaced the safe fallback with strict dictionary indexing. The code now dynamically reads `projection_eval.yaml` at import time and explicitly aborts the program (`ValueError`) if `diffusion_timestep_threshold` is missing.
3. **Outcome**: The experiment pipeline now guarantees that the threshold stamped on the output folder exactly matches a deliberately defined configuration in the YAML file.

## Gen3v3 hotfix: DPCC Baseline Config Naming Parity (9. May)

Keywords: DPCC folder naming, tracking parameters, aw in training, T in planning, loadpath backward compatibility.

1. **Problem**: The legacy DPCC baseline (`diffusion` and `plan` blocks) did not expose critical hyperparameters in their folder names, making it hard to identify models trained with different Action Weights (`aw`) or evaluated with different Thresholds (`T`).
2. **Fix (Train)**: Created a new tracking list (`args_to_watch_dpcc_train`) for the `diffusion` block to explicitly append the action weight to the training folder name (e.g., `diffusion/..._aw10`).
3. **Fix (Plan Nesting & Naming)**: 
    - *Attempt 1 (Failed)*: Tried to nest evaluation results using a lazy f-string prefix (`f:plans/diffusion/...`). This failed silently because the custom `eval_fstrings` parser in `diffuser/utils/setup.py` failed to evaluate the string correctly for the DPCC baseline, resulting in un-nested flat folders.
    - *Attempt 2 (Success)*: Completely bypassed the buggy f-string parser. Hardcoded the nested folder structure directly into the `exp_name` variable using a Python `lambda` function (`lambda args: f"plans/diffusion/H{args.horizon}.../" + watch(...)(args)`). This perfectly mirrors FMv3's nesting architecture with 100% certainty, without relying on unstable string evaluation black-magic.
4. **Outcome**: The DPCC baseline now has parity with FMv3 regarding hyperparameter visibility in its file paths.

> [!WARNING]
> **Old DPCC Folder Compatibility**: The `diffusion_loadpath` for DPCC evaluations was updated to strictly look for `_aw{action_weight}`. As a result, **old DPCC models trained before this hotfix will fail to load** because their folder names lack the `_aw10` suffix. To evaluate older DPCC models, you must manually rename their output folders to append `_aw10` to the end.

---

## Gen5: Bridging Visual Aligning Pipeline (12 May)

Keywords: visual aligning, D3IL bridge, VisualDiffusionBridge, ResNet18 encoder, image conditioning, Phase 1 Done.

### Objective
Integrate the D3IL visual aligning pipeline (multi-camera images + state) into the FM-PCC framework as a robust control baseline before migrating to Flow Matching.

### Accomplishments (Phase 1: Rewire - CODE DONE)
1.  **Engine Bridging**: Created the `ddpm_encdec_vision/` engine folder (copy-modified from `flow_matcher_v3_ode_selectable`) to host the visual pipeline without affecting state-only baselines.
2.  **Visual Bridge Implementation**: Developed `ddpm_encdec_vision/models/d3il_visual_bridge.py`. 
    - This module acts as the single integration point, directly instantiating and wrapping D3IL's `MultiImageObsEncoder` (dual ResNet18) and `Diffusion` (DDPM) model.
    - Handles the conversion of 5-tuple visual data `(bp_imgs, inhand_imgs, obs, act, mask)` into latent embeddings for the transformer-based diffusion core.
3.  **Dataset Integration**: Wired the `Aligning_Img_Dataset` from `d3il/environments/dataset/aligning_dataset.py` into the FM-PCC training loop.
4.  **Training entry point**: Created `ddpm_encdec_vision_test/train_ddpm_encdec_vision.py` which supports multi-seed training, W&B logging, and artifact management for the new visual engine.
5.  **Configuration**: Defined `config/aligning-d3il-visual.py` to manage visual-specific hyperparameters (128-dim embeddings, 3D action space, image normalization).

### Status
- **Phase 1 (Rewire)**: **COMPLETE**. Code is implemented, verified, and ready for baseline training.
- **Phase 2 (Replace)**: **Pending**. Next step is to swap the DDPM core for the FMv3ODE flow-matching core while retaining the bridged visual encoder.
- **Phase 3 (Validate)**: **Pending**. Sensitivity tests and benchmark comparisons.

### Technical Note
The implementation follows the **Copy-Modify Isolation** principle. The original state-only engines (`flow_matcher_v3_ode_selectable/`) and D3IL core files remain untouched, ensuring a safe rollback path and clear A/B comparison capability.

---

## Data Analysis Tool v2: Multi-Candidate Batch Analysis (12. May)

Keywords: DA v2, batch analysis, cross-candidate comparison, Pareto frontier, thesis-ready results.

### Problem Statement
- **v1 limitation**: Analyzes ONE experimental folder at a time (e.g., single diffusion variant)
- **Research need**: Compare 5+ experimental configurations side-by-side to identify best hyperparameter/method
- **Challenge**: 834+ .npz files across 5 seeds × 18 variants × 4 constraints = impossible manual comparison

### Solution: v2 Implementation
Implemented comprehensive multi-candidate batch analysis pipeline with 6 new modules (~1,692 lines):

1. **Phase 1 - Discovery**: `multi_candidate_discovery.py` - Auto-identifies candidate folders (A, B, C, D, E...)
2. **Phase 2 - Loading**: `batch_data_loader.py` - Loads all candidates in parallel
3. **Phase 3 - Aggregation**: `batch_aggregator.py` - Computes statistics & rankings per candidate
4. **Phase 4 - Visualization**: `batch_visualizer.py` - Generates 5 cross-candidate comparison plots
5. **Phase 5 - Reporting**: `batch_reporter.py` - Exports CSVs & human-readable summaries
6. **CLI**: `main_da_batch.py` - Orchestrates full pipeline with flexible arguments

### Key Features
✅ **Auto-discovery**: Finds candidates containing seeds [6,7,8,9,10]  
✅ **5 Comparison Plots**:
   - Pareto frontier (accuracy vs time - MAIN THESIS FIGURE)
   - Success rate comparison (bar chart)
   - Computation time comparison
   - Robustness/seed variability (boxplot)
   - Constraint × Candidate heatmap

✅ **Ranking Tables**: CSV export for thesis supplementary tables  
✅ **Custom Naming**: Support for meaningful candidate names (e.g., "aw=1", "aw=10", "dpcc-baseline")  
✅ **Flexible Filtering**: Select specific candidates, seeds, variants, constraints  
✅ **Publication-Quality**: 300 DPI, color-coded, annotated plots

### Usage (Quick Start)
```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --candidate-names "aw=1,aw=5,aw=10,dpcc" \
    --output-path ./thesis_batch_results
```

### Documentation
- **IMPLEMENTATION_ROADMAP.md**: Technical architecture & 5 phases
- **MISSION_BRIEFING_v2.md**: Research context & thesis integration
- **USAGE_v2.md**: 7 practical examples + troubleshooting guide

All available in: `logs_in_develop/DA_Code/v2/`

### Status
**✅ COMPLETE**: v2 fully implemented, documented, and ready for thesis batch analysis.

### Typical Use Cases
- **Ablation studies**: Compare aw=1 vs aw=5 vs aw=10 across all variants
- **Method comparison**: DPCC vs Diffuser vs FM-v3 head-to-head
- **Solver benchmarking**: Euler vs RK4 vs Midpoint performance
- **Constraint analysis**: Which method handles which constraint best

---

## Gen3v3: FM-D Drifting Engine Recovery & Wiring (12. May)

Keywords: FM-D recovery, drifting engine, training wiring, batch_to_device polymorphism, Slurm pipeline fix.

1.  **Pipeline Recovery**: Identified and fixed a catastrophic disconnect in the "Drifting" pipeline where the Slurm scripts were hallucinating non-existent repositories and the Python scripts were lazy copies of the standard FMv3 baseline.
2.  **Training Logic Wiring**: 
    - **Problem**: The `flow_matcher_v3_drifting` trainer was not actually performing drifting training; it was missing the `DriftTrainingWrapper` integration. 
    - **Fix**: Rewired `utils/training.py` to instantiate the `DriftLoss` memory bank and scheduler. The trainer now correctly computes the hybrid FM + λ·Drift loss and updates the distribution buffer during each epoch.
3.  **Polymorphic Batching Fix**:
    - **Problem**: `batch_to_device` in `utils/arrays.py` was hardcoded to `namedtuples`, causing crashes when using standard PyTorch `list`/`tuple` datasets.
    - **Fix**: Refactored the utility to be fully polymorphic, recursively handling all container types (matching the Gen5 standard).
4.  **Slurm Standardization**: Fully rewrote `train_drifting.sh`, `eval_drifting.sh`, and `load_results_drifting.sh` to match the project's production `fmv3_ode` standards, ensuring correct `PYTHONPATH` and conda environment activation.
5.  **Outcome**: **TRAIN WORKING**. The Drifting engine is now fully functional, wired to the `flow_matching_v3_drifting` config block, and producing drift-augmented trajectories.

---

## Gen3v4: iMeanFlow (iMF) Phase 1 Foundation Completion (13. May)

Keywords: iMeanFlow, dual-velocity decomposition, FMv3ODE foundation, Phase 1 complete, 8 core modules, 1994 LOC.

1. **Architecture Established**: Implemented Improved Mean Flows (iMeanFlow) on FMv3ODE foundation (not FM-D) using FM-D's 4-phase methodology.
2. **Core Modules Delivered** (8 files, 1,994 lines):
   - `imf_velocity.py`: Dual-velocity field (u=global, v=local) with time conditioning
   - `jvp_guidance.py`: Jacobian-Vector Product constraint guidance (collision, smoothness)
   - `imf_ode_solvers.py`: Multi-backend ODE solvers (Euler, RK4, dopri5) with NFE=1/2 flexibility
   - `imf_training.py`: Dual-loss training, u_first curriculum scheduler, training wrapper
   - `imf_metrics.py`: Comprehensive trajectory metrics (u/v error, smoothness, decomposition)
   - `imf_dit_trajectory.py`: Optional Transformer backbone (DiT) for sequence modeling
   - `imf_trajectory_sampler.py`: High-level inference API (single/dual/multi-step, goal-guided, obstacle-avoidance)
   - `test_imf_core.py`: 65+ unit tests covering all modules
3. **Examples & Configs Delivered**:
   - `example_imf_training.py`: End-to-end training on synthetic data
   - `example_imf_inference.py`: 5 inference demonstration scenarios
   - `fm_imeanflow_base.yaml`, `fm_imeanflow_d3il.yaml`, `fm_imeanflow_avoiding.yaml`: Task-specific configs
4. **Integration & DevOps**:
   - Updated `dpcc/config/avoiding-d3il.py` with iMF config block (3 locked parameters)
   - Created `Slurm_Codes/sbatch/iMF/` folder with `train_imf.sh`, `eval_imf.sh`, `load_results_imf.sh`
   - Generated `HOW_TO_RUN.md` and `Phase1_Completion.md` documentation
5. **Outcome**: **PHASE 1 COMPLETE**. All foundation infrastructure in place. Ready for Phase 2 (training integration with d3il).

---

## Gen3v4: iMeanFlow Phase 2 - Real Training Infrastructure (13. May)

Keywords: Phase 2 complete, real training/eval/load scripts, multi-seed, W&B logging, SLURM integration, production-ready.

1. **Real Training Script** (`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`, 465 lines):
   - Multi-seed loop (supports `--seeds 6 7 8 9 10` pattern matching Drifting)
   - Dual-velocity loss computation with curriculum scheduler
   - W&B logging (`--use-wandb` flag, FMPCC-iMF project)
   - Checkpoint saving (best + periodic epochs)
   - Synthetic data pipeline (easily swappable for real d3il avoiding-d3il data)
   - Config-driven hyperparameter control (batch_size, LR, epochs, device)

2. **Real Evaluation Script** (`FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`):
   - Multi-variant testing: 3 solvers (Euler, RK4, Dopri5) × 2 NFE values (1, 2) = 6 variants
   - Per-seed evaluation with metrics tracking (trajectory error, path length, smoothness)
   - Per-variant .npz result saving + aggregate JSON reporting
   - Compatible with d3il environment integration (structure ready, data synthetic)

3. **Real Results Loader** (`FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py`):
   - Loads all evaluation .npz files across seeds
   - Computes aggregate statistics (mean, std, min, max per variant)
   - Generates 3 comparison plots (trajectory error, path length, smoothness)
   - Exports CSV + JSON summary reports

---

## DA v3 & iMF Phase 3 Integration (May 13, 2026)
**Keywords**: Zero-Manifest, audit-logging, iMF-PCC real-data integration.

1. **Matrix Explorer v3**: Stabilized with Zero-Manifest HTML discovery and hybrid zoom (FigWidth + Magnify). Implemented automated `.txt` audit logs including absolute source paths for every PNG download.
   - *Ref*: [`logs_in_develop/DA_Code/v3/fix_3/fix_3.md`](./DA_Code/v3/fix_3/fix_3.md)
2. **iMeanFlow (iMF) Phase 3**: Migrated to official `iMeanFlowEngine` (dual-velocity field). Wired `iMFDiffusion` wrapper and `u_first` curriculum training for real `avoiding-d3il` dataset. Standardized multi-seed Slurm scripts and W&B logging.
   - *Ref*: [`logs_in_develop/Gen3v4/fix_3/REAL_IMF_IMPLEMENTATION.md`](./Gen3v4/fix_3/REAL_IMF_IMPLEMENTATION.md)

**Status**: **VERIFIED STABLE**. Visualizer and iMF-PCC core are production-ready for final thesis analysis.

## Gen5 Phase 1 Addendum (13. May 2026) — Today's Fixes

Keywords: Hydra instantiation, device serialization, DataLoader CUDA fork, PYTHONPATH, diffusion bounds, batch_to_device

Summary of fixes applied today (engineering integration, not algorithmic changes):

1. **Hydra instantiation & device handling**: Cast `torch.device` to primitive strings and set `"_recursive_": False` in bridge configs to prevent eager Hydra instantiation conflicts.
2. **DataLoader / CUDA fork crash**: Ensured visual datasets are initialized on CPU and moved to GPU only at batch time (`batch_to_device`) to avoid CUDA context corruption in worker forks.
3. **PYTHONPATH / simulator imports**: Added `d3il/environments/d3il` to `PYTHONPATH` and updated evaluation entrypoints to ensure `envs.*` imports resolve in SLURM jobs.
4. **Diffusion action bounds**: Initialized `min_action` / `max_action` inside `VisualDiffusionBridge` so diffusion sampling clamps do not raise AttributeError during eval.
5. **Polymorphic batch handling**: Made `batch_to_device` robust to `list`/`tuple` batches (and namedtuples) to support D3IL dataset outputs.
6. **Logging & stability**: Additional small fixes to logging and checkpoint path resolution in the visual test scripts to make baseline runs reproducible.

Outcome: The `ddpm_encdec_vision` baseline is runnable inside FM-PCC; training/eval failures observed earlier were due to integration gaps listed above and are now addressed. Next step: run full baseline training and collect W&B traces to validate learning curves.
## Gen5: Visual Aligning Diagnostic & Baseline Stabilization (May 15, 2026)

Keywords: U-Net H=2 support, diagnostic fidelity, 7-metric report, ACT-parity, capacity analysis.

1.  **Architectural Fix (U-Net H=2)**: Implemented "Auto-Padding" in `VisualUNet.py` to support small horizons. The model now dynamically pads short trajectories to a multiple of 8, resolving the 3-stage downsampling crash.
2.  **Diagnostic Fidelity Restoration**:
    - **Frozen View Fix**: Implemented deep memory copying (`.copy()`) for simulator frames to prevent pointer-shadowing.
    - **Color Fix**: Added floating-point clipping (`.clip(0, 255)`) to prevent color-inversion/overflow artifacts in GIF generation.
3.  **Scientific Reporting**: Standardized the evaluation output to match the FMv3ODE **7-metric report** (Success Rate, Constraints, Steps, Violations, and Inference Time $\pm$ std).
4.  **Baseline Synchronization**: Verified and documented the **ICLR 2024 DDPM-ACT** official hyperparameters (500 Epochs, $5\cdot10^5$ steps, $5\cdot10^{-4}$ LR, Batch 64).
5.  **Backbone Capacity Analysis**: Documented the **20x capacity difference** between Gen5 U-Net (18M+ params) and Native ACT Transformer (~0.9M params) for thesis justification.
6.  **Status**: **Visual Pipeline Stable**. High-fidelity training and evaluation are now scientifically aligned with the FMPCC standards.
---

## Gen5: Visual-Aligning Stabilization & Diagnostic Finalization (May 16, 2026)

Keywords: Masked Statistics, Zero-Variance Lock, Mixed-Loop control, Battle-Ready.

1.  **Masked Statistics Optimization**: Implemented masked mean/std calculation in the `GaussianNormalizer` to ignore zero-padding in expert trajectories, effectively eliminating the "Hypersonic Drift" caused by numerical scaling artifacts.
2.  **Zero-Variance Safety Lock**: Enforced a `1e-4` standard deviation floor in the normalizer to prevent division-by-zero crashes on constant dimensions (e.g., end-effector Z-height).
3.  **Mixed-Loop Control Logic**: Finalized the "Mixed-Loop" paradigm (Open-Loop Mental Map for proprioception + Closed-Loop Visual Correction), ensuring temporal auto-sync and hand-eye coordination.
4.  **Verification**: Successfully validated the end-to-end pipeline on a 3k-step stable model.
    - **Log**: `FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-16/23_26_39_eval_visual_aligning_20403.log` (**WORKED!** 3k train)
5.  **Status**: **BATTLE-READY**. The Gen5 visual pipeline is fully stabilized and prepared for the 500k-step benchmark suite.

## Gen5: Visual-Aligning Diagnostic & Logging Robustness (May 17, 2026)

Keywords: Tee-Stderr redirection, MuJoCo mju_openResource fix, deferred XML deletion, atexit.

1.  **Redirection Robustness (Tee-Stderr)**: Modified `eval_ddpm_encdec_vision.py` to intercept and Tee `sys.stderr` in addition to `sys.stdout` to the `eval_diffuser.log` file. This prevents tracebacks, standard library/framework warnings, and critical error messages from being lost when runs fail or end abruptly (e.g. at Context 7).
2.  **MuJoCo Resource Loading Stabilization**: Resolved the persistent `WARNING: mju_openResource: could not open resource 'panda_tmp_rb*.xml'` warnings that occurred during simulator rollouts.
    - **Root Cause**: The generated temporary robot XML assets were deleted immediately after compilation by the `cleanup()` method inside `MjSceneParser.create_scene()`, but before MuJoCo's offscreen renderer lazily initialized and loaded resources on the first camera frame.
    - **Technical Fix**: Overrode `cleanup()` in `MjIncludeTemplate` inside the active `mj_beta/MjLoadable.py` (explicitly keeping legacy `mujoco/MujocoLoadable.py` untouched and unmodified) to defer physical XML file deletion to the Python interpreter's exit using `atexit.register()`.
    - **Impact**: Completely eliminated all MuJoCo C++ resource provider warnings in the active simulator backend, ensuring flawless offscreen camera rendering and solid visual-controller stability.
3.  **Real-Time Per-Rollout Debug Statistics**: Integrated an instant audit callback `update_rollout_info(info)` inside the simulator loop.
    - **Impact**: Rollout statistics are printed to the console in real time (containing success, total steps, mean distance, env mode, max tracking error, and average inference time).
    - **Artifacts**: For every trial run, a human-readable summary file `rollout_{idx}_stats.txt` is automatically written to the `diagnostics/` directory alongside the corresponding rollout MP4/GIF file, and `rollout_{idx}_stats.json` is exported to `realtime_diagnostics/` for zero-friction debugging.
4.  **Scientific Step Reporting Decoupling**: Updated final metric averages to calculate and print both `Avg number of steps (successful trials)` and `Avg number of steps (all trials)`. This provides complete transparency into step performance regardless of the success rate, resolving misleading legacy `0.00` printouts when success is `0.0%`.
5.  **Dynamic Train vs. Test Context Toggle & Isolated Outputs**: Introduced a `--eval-on-train` flag to the evaluation script. This dynamically switches the robot and block initial positions to use those from the seen expert dataset (`train_contexts.pkl`) rather than unseen validation data (`test_contexts.pkl`). To prevent overwriting the standard generalization results, all logs, `.npz`, `.pkl`, and `.png` outputs are automatically routed to a distinct `results_train_set/` directory with `_train_set` labels, while console logs print explicit `Seen Training Context` labels in real time.
6.  **Enforce Configured Max Episode Length**: Fixed a configuration discrepancy where the `'max_episode_length'` parameter defined in `config/aligning-d3il-visual.py` was silently ignored. The evaluation engine now dynamically extracts this limit and passes it down as `max_steps_per_episode` to `Robot_Push_Env`, ensuring that custom research limits are fully respected by the physics engine.
7.  **Dynamic Planning Batch Size & Batched Candidate Trajectory Sampling**: Integrated full support for custom planning batch sizes (e.g. `batch_size: 4`) during visual closed-loop evaluations. The model wrapper automatically duplicates image and coordinate context sequences along the batch dimension in PyTorch, query the model to sample multiple parallel candidate paths in a single fast GPU pass, and executes the primary selected candidate path.
8.  **Status**: **PRODUCTION STABLE**. The entire diagnostic, logging, and audit pipeline is robustly secured and fully operational.

## Gen6: Visual-Aligning Differentiable MPC (DPCC Upgrade) (May 17, 2026)

**Keywords**: DPCC visual injection, compatibility adapter, direct FMv3ODE code-reuse, zero-code model wrapping, Euler kinematics indexing.

1.  **Pure FMv3ODE Code-Reuse**: Analyzed class inheritance and confirmed that `VisualGaussianDiffusion` directly inherits from `diffuser.models.diffusion.GaussianDiffusion`. Because the base `GaussianDiffusion` class already contains 100% of the in-denoising snapping (`projector.project`) and gradient-guidance hooks, we upgraded the pipeline to Gen6 with **zero new custom model or VAE code creation**.
2.  **Compatibility Normalizer Adapter**: Implemented a lightweight adapter wrapper (`VisualNormalizerAdapter` & `VisualNormalizerDict`) directly in the evaluation script. This extracts physical coordinate limits from D3IL's standard-deviation `Scaler` class and presents them to the `Projector` class's expected Min/Max `mins` and `maxs` dictionary interface at runtime, eliminating the need to patch core codebase libraries.
3.  **Euler Kinematics Indexing (6D Trajectory)**: Successfully mapped the $6$-dimensional visual trajectory space `[actions (3D), proprioception (3D)]` to the Projector's constraint matrices:
    - **Absolute Workspace Cage Limits**: Implemented bounds vectors of size 6 `[-inf, -inf, -inf, lb_x, lb_y, lb_z]` and `[inf, inf, inf, ub_x, ub_y, ub_z]`, restricting the absolute physical end-effector position (proprioception) to the workspace cage while letting the actions remain dynamic.
    - **Dynamics Integrator binding**: Configured dynamic Euler step transitions binding proprioceptive coordinates (indices 3, 4, 5) directly to action coordinate deltas (indices 0, 1, 2) in the SLSQP solver.
4.  **100% Parity Safety Lock**: Implemented a bypass guard in the evaluation script (`projector = None` if `variant == 'diffuser'`). This guarantees that when running the baseline diffuser mode, the model completely bypasses all projection checks, ensuring 100% numerical and computational parity with Gen5.
5.  **Status**: **IMPLEMENTATION SUCCESSFUL**. The Gen6 vision-conditioned differentiable MPC safety engine is fully configured and ready for production benchmarking.

## Gen6v2: Dual-Backbone Calibration & Pipeline Orchestration (May 17, 2026)

Keywords: Hyperparameter Calibration Blueprint, W&B GroupName Safety Lock, visual_aligning_pipeline, Chained Slurm Dependencies, K-less parity.

1.  **Dual-Backbone Hyperparameter Blueprint**: Authored a comprehensive blueprint comparing the 1D Temporal CNN U-Net vs. Transformer VAE parameters:
    - **MUST Change**: `learning_rate` (2e-4 vs 5e-4 to prevent CNN gradient explosions), `condition_dropout` (0.25 for CFG prior vs 0.10 for direct visual context), and Sequence Lengths (`horizon = 8` vs `5+4-1=8`).
    - **Invariant**: `n_diffusion_steps`, `action_dim`, `loss_type` ('l2'), `batch_size`, `ema_decay` (0.995), and scaling normalizers must remain unchanged to ensure experimental comparison parity.
2.  **W&B GroupName 128-Character Safety Lock**: Patched `train_ddpm_encdec_vision.py` to enforce a hard maximum length of 128 characters (`wandb_group = wandb_group[:128]`) right before `wandb.init()`. This permanently resolves the `CommError 400 Bad Request` where long model class names inside generated experiment log folders exceeded Weights & Biases API server limits.
3.  **Slurm Pipeline Orchestration Master**: Developed the `visual_aligning_pipeline.sh` orchestrator under `Slurm_Codes/sbatch/Visual_Aligning/` that mirrors the structure and pro-logging conventions of `fmv3_ode_pipeline.sh`.
    - **Implementation**: Sequentially dispatches training (`train_visual_aligning.sh`), extracts the Slurm `TRAIN_ID`, and schedules the evaluation (`eval_visual_aligning.sh`) with `--dependency=afterok:$TRAIN_ID` under a unified timestamp log directory for zero-friction run tracking.
4.  **Status**: **PIPELINE COMPLETED**. Dual-backbone parameter strategies, API safety measures, and chained job managers are fully standardized.

## Gen7: Visual Flow Matching (FMv3ODE) Migration (May 18, 2026)

Keywords: sibling directories, visual U-Net FiLM projection, Beta sampling noise schedule, unified Slurm suite, registry config parity.

1. **Sibling Package Decoupling**: Created a fully independent sibling package `fm_encdec_vision/` and `fm_encdec_vision_test/` by duplicating the legacy DDPM codebases. Decoupled and renamed all training, evaluation, and loading scripts to guarantee 100% parallel workspace parity without modifying original DDPM code.
2. **U-Net FiLM Parity Guard**: Swapped the temporal backbone inside [fm_encdec_vision/models/visual_unet.py](file:///workspaces/FM-PCC/fm_encdec_vision/models/visual_unet.py) to use `UNet1DTemporalCondModel` (instead of state-only `Flow_matcher_U_Net_v2`), preserving the critical FiLM projection mechanism (`use_cond_projection=True`) for spatial visual token conditioning.
3. **Continuous-Time ODE Solver Integration**: Overwrote the core diffusion engine in [fm_encdec_vision/models/visual_gaussian_diffusion.py](file:///workspaces/FM-PCC/fm_encdec_vision/models/visual_gaussian_diffusion.py) to inherit from the continuous-time `GaussianDiffusion` base class. Configured linear interpolation path training, continuous time sampling $t \sim \text{Beta}(\alpha=1.5, \beta=1.0)$, and iterative Euler integration solvers for simulator rollouts.
4. **Registry Config Parity & Comment Restoration**: Appended the new `'fm_encdec_vision'` and `'plan_fm_encdec_vision'` dictionaries directly inside [config/aligning-d3il-visual.py](file:///workspaces/FM-PCC/config/aligning-d3il-visual.py). Replicated all legacy inline comments and developer notes, while integrating the new continuous-time parameters (e.g. `time_beta_alpha_v3`, `flow_steps_v3`, `ode_solver_backend_v3`) and watch lists.
5. **Unified Slurm Manager**: Built and authorized (`chmod +x`) a complete suite of Slurm submit templates in `Slurm_Codes/sbatch/Visual_Aligning/`:
   * `train_visual_aligning_fm.sh`: Launches U-Net training.
   * `eval_visual_aligning_fm.sh`: Executes MuJoCo rollout evaluations.
   * `load_results_visual_aligning_fm.sh`: Compiles and plots success metrics.
   * `visual_aligning_pipeline_fm.sh`: Chains training and evaluation sequentially.
6. **Config Alignment (Offtopic)**: Reorganized [config/avoiding-d3il.py](file:///workspaces/FM-PCC/config/avoiding-d3il.py) to move the iMeanFlow (iMF) training and planning configurations into their correct logical sections (training under models, planning under inference).
7. **Status**: **COMPLETE & VERIFIED**. Visual Flow Matching architecture, configs, and Slurm managers are fully standardized and ready for production GPU runs.

***

## Gen6v3: Non-Visual Aligning Pipeline (May 18, 2026)

**Keywords**: 17D vs 20D compatibility, U-Net transition-dim scaling, state-only multi-seed evaluation.

1. **State Dimension Parity**: Resolved the $17\text{D} \text{ vs. } 20\text{D}$ proprioceptive state mismatch between baseline datasets and visual-aligned configurations. Rewrote preprocessing pipelines to support conditional state-only load operations.
2. **Backbone Generalization**: Updated the U-Net spatial layers to dynamically scale `transition_dim` based on evaluation targets, preventing shape crashes when loading visual-trained weights in state-only runs.
3. **Training & Evaluation**: Stabilized training workflows to bypass visual encoding matrices when running in non-visual mode, aligning standard metrics sweeps.

## Gen6v4: Unified 9D Visual-DPCC Safety Engine (May 18, 2026)

**Keywords**: 9D Joint Trajectory representation, SLSQP Euler Projection, actual proprioceptive boundaries, DPCC Base Pivot, DDPM-ACT Failure.

1. **Strategic Pivot: No more `ddpmact d3il base`**:
   Historically, the visual encoder-decoder baseline (`ddpm_encdec_vision` from Gen6, and Gen7 which was based on it) utilized the `ddpmact d3il base` (ACT imitation framework). However, this architecture proved highly unstable, **only succeeding once** (archived inside the outdated legacy folders) and failing to return any reproducible good results thereafter. 
   
   To resolve this structural deadlock, Gen6v4 introduces a **fundamental new principle**: **migrating entirely to the `dpcc base`** as the core foundation for visual-conditioned trajectory alignment.
2. **9D Trajectory Paradigm ($x_t \in \mathbb{R}^{H \times 9}$)**:
   Designed a unified state-action-observation planning representation on top of the DPCC base:
   $$x_t = \left[ \text{act}(3\text{D}) \;\mid\; \text{des\_c\_pos}(3\text{D}) \;\mid\; \text{c\_pos}(3\text{D}) \right]$$
   This shifts boundary constraints directly onto the physical, actual end-effector position ($c\_pos$) rather than the commanded position ($des\_c\_pos$), guaranteeing real-world safety cage violations are blocked by the controller.
3. **Dataset Preprocessing Alignment**:
   Implemented the `ParityAligningDataset` parser. The normalizer restricts limits fitting strictly to `valid_mask` data points to prevent zero-padded tails from pulling normalizer bounds toward $0$.
4. **Denoising Clamping Hooks**:
   Modified `p_mean_variance` inside `VisualGaussianDiffusion` to selectively clamp only the active control slots ($[..., :3]$) to $[-5.0, 5.0]$ while leaving physical $c\_pos$ dimensions unclamped. This ensures physical coordinate integrity is maintained during step integrations.

## Gen5: DDPM EncDec Legacy Restoration & Safety Auditing (May 18, 2026)

**Keywords**: Legacy code protection, Scaler normalization restoration, hyperparameter sanity locks.

1. **Legacy Recovery**: Re-added `add_Legacy_working_Good_Codes (Gen5_DDPM_EncDec)` inside the source tree to preserve baseline training stability.
2. **Scaler Stabilization**: Restored legacy normalization scale mapping inside `VisualUNet` and `Scaler` objects. This prevents statistical regression and secures reproducible baselines for the $500\text{k}$ training checkpoints.
3. **Path Fix**: Resolved file loading references in `config/aligning-d3il-visual.py` to ensure proper dataset routing inside cluster configurations.


## Gen6v4 / Gen7: Robustness Fixes, Pipeline Standardization & Evaluation Upgrades (May 19, 2026)

**Keywords**: clip_denoised=False, eval-on-train launcher, Slurm pipeline naming alignment, double-prefix importer fix, dataset buffer overflow bypass, actual simulation state tracking.

### 1. Denoising Chain Protection (`clip_denoised=False`)
* **Problem**: Setting `clip_denoised=True` in training scripts caused the ±5 action clamping to trigger at every early denoising step. Combined with the cosine noise schedule, this amplified bounds mathematically and permanently corrupted the actions by pinning them to thresholds, leading to 100% rollout failures.
* **Resolution**: Disabled denoising clipping by setting `clip_denoised=False` by default in training and forced it to `False` in evaluation routines. This allows the denoising chain to generate smooth, natural action velocity plans.

### 2. Default Visual Evaluation on Training Set (`--eval-on-train`)
* **Feature**: Enabled the `--eval-on-train` flag by default inside all three visual evaluation Slurm launcher scripts:
  * `Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh`
  * `Slurm_Codes/sbatch/Visual_Aligning/eval_visual_aligning_fm.sh`
  * `Slurm_Codes/sbatch/Visual_Aligning/eval_visual_aligning.sh`
* **Impact**: Ensures that visual evaluations run on seen expert training contexts by default to establish robust diagnostic baselines.

### 3. Slurm Pipeline & Job Naming Consistency
* **Action**: Renamed `visual_aligning_dpcc_pipeline.sh` to `visual_aligning_pipeline_dpcc.sh` to match the naming convention of other pipelines (`visual_aligning_pipeline.sh` and `visual_aligning_pipeline_fm.sh`).
* **Alignment**: Standardized the `#SBATCH --job-name` directives of all 12 visual sbatch scripts (including train, eval, load, and pipeline runners) to exactly match their `.sh` filenames, eliminating job name mismatches.

### 4. Double Prefix Class Importer Guard
* **Problem**: During evaluation weight loading, `import_class()` prepended a double `diffuser_visual_aligning.` prefix to classes already containing it, triggering a catastrophic `ModuleNotFoundError`.
* **Resolution**: Added a strict guard in class resolution to skip prefix injection if the import string already begins with the correct package prefix.

### 5. Path Length Alignment & Dataset Buffer Overflow Bypass
* **Fix**: Standardized `max_path_length: 1000` in both training and evaluation configs to prevent `FileNotFoundError` during model loading.
* **Bypass**: Solved a buffer overflow limit in D3IL dataset loaders by bypassing `Aligning_Dataset` and loading expert trajectory state data directly from raw pickle files, opening the full dataset for visual-DPCC training.

### 6. Closed-Loop Simulation State Tracking
* **Fix**: Corrected the observation construction in `VisualAgentWrapper`. The observation vectors now concatenate actual simulator commanded positions (`des_robot_pos_np`) instead of dead-reckoning initial coordinate estimates, eliminating trajectory drift under execution.

### 7. Evaluation Logging and Safety Safeguards
* **WandB Crash Fix**: Disabled WandB logging during D3IL closed-loop evaluation runs to avoid PyTorch/MuJoCo segmentation faults, and cleanly redirected run reports to offline diagnostic dumps (`diag_first_replan.txt`).
* **Visual Validation**: Implemented strict console logging of scaling normalizer parameters and added sequence length validation locks to prevent silent failures.

### 8. Manual Legacy Retrieval & D3IL Revert Parity (FIX_7.1, FIX_7.2, FIX_7.3)
* **Revert Fix 38 (FIX_7.1)**: Removed experimental `max_episode_length` plumbing in `Aligning_Sim` environment initialization to restore physics-based default steps.
* **BGR-to-RGB Image Parity (FIX_7.2)**: Reverted the color-space conversion in D3IL's image loaders to preserve byte-for-byte image alignment with the original dataset, preventing visual distribution shifts.
* **Material Simulator & Robot Parity (FIX_7.3)**: Reverted custom simulator control loops, named camera registrations, and rod-tip collisions to restore 100% behavioral parity with original D3IL benchmarks.
* **Traceability Matrix**: Created [D3IL_DIFF_AUDIT.md](Gen6_dpcc_Engine_for_visual_aligning/Gen6V4_dataset_upgrade_visual_dpcc/Manual_Legacy_retrieval_FIX_7/D3IL_DIFF_AUDIT.md) and [FIX7_LEGACY_REVERT_LOG.md](Gen6_dpcc_Engine_for_visual_aligning/Gen6V4_dataset_upgrade_visual_dpcc/Manual_Legacy_retrieval_FIX_7/FIX7_LEGACY_REVERT_LOG.md) to log all changes and verify parity.

---

## Gen6v4: Visual-DPCC Robustness & Projector Safeguards (Fix 8 & Fix 9) (May 19, 2026)

**Keywords**: BGR→RGB flip, dead assertion, LimitsNormalizer eps-guard, Projector batch-0 initial state broadcast, initial-state scaling B1, Deque temporal ordering B3, post-processing selection Fix 9.4, no-op guard Fix 9.1/9.2, SLSQP delta logging Fix 9.3, B1 unit test.

### 1. Fix 8: Projector and Normalization Robustness
* **A1: BGR→RGB Inference Correction**: Added a `[::-1].copy()` channel flip to the transposed images inside `aligning_sim.py` (both at init and per-step) to align evaluation's BGR frames with the RGB format the dataset loader (`sequence.py`) produces. *(Note: Later reverted in Fix 11 after deeper audit).*
* **A2: Dead Assertion Fix**: Corrected `assert RuntimeError()` to `raise RuntimeError(...)` inside `GaussianDiffusion.__init__()` when `clip_denoised=False`.
* **A3: LimitsNormalizer zero-variance guard**: Prevented division-by-zero crashes on constant dimensions (e.g. end-effector z-height) by adding an eps-guard (`range_ < 1e-8 → 1.0` in `normalize()`, `0.0` in `unnormalize()`).
* **A4: Batch initial-state broadcast fix**: Fixed the SLSQP projector (`projection.py`) broadcasting sample 0's initial state `s_0` to all batch elements during `project()` and `compute_gradient()`. Moved extraction inside the batch loop so that `s_0` is correctly resolved per-sample.
* **B1: Dynamics constraint scaling alignment**: Re-scaled the initial-state anchor constraint row in `mat_fix_initial` using `x_diff` (instead of `1`) and the `b` vector using `x_diff * s_0` to match the scale of the dynamics rows, ensuring the solver does not treat the initial state as proportionally weaker.
* **B3 & B3-ext: Deque temporal ordering**: Replaced `appendleft` with `append` in deques for both visual and non-visual paths in `eval_visual_aligning_dpcc.py` to store trajectories in chronological order (`[oldest, ..., newest]`) instead of inverted order.
* **C4: Closed-loop proprioceptive feedback**: Corrected observation construction in `eval_visual_aligning_dpcc.py` and `aligning_sim.py`. Previously, both commanded (`des_c_pos`) and actual (`c_pos`) halves of `obs_6d` were fed the commanded position, creating a zero-lag evaluation discrepancy. Correctly concatenated the actual `robot_pos` alongside commanded `des_robot_pos` to match the model's training distribution.
* **Cascade fixes**: Corrected video capture block in `predict()` to remove redundant `cvtColor(BGR2RGB)` since `bp_np` is already RGB after Fix A1.

### 2. Fix 9: Empty-Constraint SLSQP Safeguards & Cost Selection
* **9.1 & 9.2: No-op constraints early exit**: Added early-exit guards in `project()` and `compute_gradient()` when `constraint_types: []` (no constraints). This prevents SLSQP from needlessly searching a constrained space and saturating actions to the ±5 bounds (noise amplification), resolving the catastrophic ±94 action range explosion seen in empty-constraint runs.
* **9.3: SLSQP Delta Logging**: Added verbose logging in `project()` to capture when the solver modifies the trajectory by a norm delta `> 1e-4`.
* **9.4: Cost-based trajectory selection**: Changed trajectory selection for `post_processing` and `model_free` variants from `random` to `minimum_projection_cost` to select the best trajectory from the batch of 6 instead of a random one.
* **B1 Unit Test**: Created a new unit test suite `diffuser_visual_aligning_test/test_projector_b1.py` validating that the B1 initial-state scale changes are structurally and functionally correct.

---

## Gen6v4: Evaluation Wiring, Pipeline Alignment & Diagnostics (Fix 10 & Fix 11) (May 20, 2026)

**Keywords**: max_episode_length, Robot_Push_Env, dead parameters cleanup, BGR flip revert, rollout GIF color correction, seeding process dynamic, .copy() safety.

### 1. Fix 10: Episode Rollout Cap Wiring
* **Wiring rollout steps**: Resolved a dead-field issue where the 400-step episode rollout budget (`max_episode_length`) in `config/aligning-d3il-visual.py` was ignored, silently capping evaluations at 400 steps due to D3IL's hardcoded defaults in `Robot_Push_Env`. Forwarded `max_episode_length` directly to `Robot_Push_Env(max_steps_per_episode=...)`.
* **Dead configuration cleanup**: Cleaned up the `plan_visual_aligning_dpcc` config block by removing four dead parameters (`policy`, `test_ret`, `value_loadpath`, `dynamic_loss`).

### 2. Fix 11 & 11b: BGR Channel Pipeline Certification
* **BGR inference revert**: Re-audited the RGB/BGR pipeline channel formats. Discovered that the training dataset is stored RGB-on-disk, but loaded via `cv2.imread` (reading as BGR) and converted via `cvtColor(BGR2RGB)` (reversing back to BGR/RGB). Thus, the training pipeline produced RGB and inference produced BGR (swapped channels). Reverted the premature `[::-1]` flip in `aligning_sim.py` (which had been introduced in Fix 8) and restored the correct `cvtColor(BGR2RGB)` for rollout visualization/GIF color capture (Fix 11b) to fix blue-red swapped visual diagnostics, ensuring model inference input remains BGR.
* **Smart RNG Seeding & Defensive copies**:
  * Replaced CPU-process seeding `random.seed(pid)` in `aligning_sim.py` (which caused all eval seeds 6-10 to use the same random rollout sequence with `n_cores=1` and `pid=0`) with process-dynamic seeding `random.seed(self.seed + pid)`. This correctly restores stochastic diversity and ensures deterministic yet unique initial noise `x_T` across eval seeds.
  * Added defensive deep copies (`.copy()`) to `des_robot_pos` initialization to prevent downstream mutations.

---

## Gen7: Continuous-Time Visual Flow Matching (FMv3ODE) Migration & Baseline Parity (May 20, 2026)

**Keywords**: sibling package scaffolding, fm_visual_aligning, Beta continuous-time, velocity target, Euler ODE forward integration, args_to_watch_fm_visual, gym_aligning_env BGR Native.

### 1. Continuous-Time FM Engine Scaffolding
* **Scaffolding**: Duplicated the Gen6V4 `diffuser_visual_aligning` package and `diffuser_visual_aligning_test` directory into `fm_visual_aligning` and `fm_visual_aligning_test` (Copy-Modify isolation strategy).
* **Namespace Refactoring**: Globally refactored all package imports to use the sibling namespace `fm_visual_aligning`, guaranteeing 100% parallel workspace coexistence without regressing the DDPM baseline.

### 2. Continuous-Time Flow Matching Engine
* **FM Core Math**: Implemented the FMv3ODE mathematical core in `models/diffusion.py` and `models/visual_gaussian_diffusion.py` using linear interpolation (`(1-t)*noise + t*data`) and continuous-time Beta(1.5, 1.0) sampling.
* **Velocity-Target learning**: Modified the training objective to learn the direct velocity vector field (`v = x_data - x_noise`) instead of the DDPM discrete noise step $\epsilon$.
* **Inference forward ODE loop**: Developed the forward deterministic ODE solver (legacy Euler) integrating from $t=0 \to 1$ over `flow_steps_v3` (default 16 steps, down from DDPM's 100).
* **Projector Integration**: Ensured the SLSQP projector is hooked near the end of the forward ODE chain ($t \ge (1 - \text{threshold}) \times K$).

### 3. Configuration & CLI Synchronization
* **Config update**: Configured `config/aligning-d3il-visual.py` by adding `fm_visual_aligning` training and `plan_fm_visual_aligning` planning blocks.
* **Descriptive directory naming**: Designed custom visual-specific watch lists `args_to_watch_fm_visual_train` and `args_to_watch_fm_visual_plan` to dynamically include the `if_vision` flag, ensuring visual checkpoints are correctly isolated in the filesystem.
* **Benchmark suite registration**: Enabled `'fm_visual_aligning'` under the benchmark experiments suite in `config/visual_aligning_eval.yaml`.

### 4. Gen7 Fix 1: Native BGR Return & Comments Cleanup
* **Native BGR return**: Re-audited the RGB/BGR pipeline channel formats. In D3IL environment package (`gym_aligning/envs/aligning.py`), restored `cvtColor(RGB2BGR)` for `bp_image` and `inhand_image` to native BGR.
* **Authoritative comments restoration**: Restored factually accurate comments in `aligning_sim.py` documenting that training uses BGR and inference also receives BGR natively via `aligning.py`, resolving the incorrect comment in Phase 0 which falsely claimed training was RGB.

***

## Gen3v3: Drifting Engine Forensic Audit & Major Upgrade (May 20, 2026)
* **Logs**: [`Audit`](./Gen3v3_Drifting/Audit_fix_1/AUDIT_REPORT.md) & [`Changelog`](./Gen3v3_Drifting/Gen3v3u2_Major_Upgrade_direct/CHANGELOG.md)
* **Critical Issues & Fixes**:
  * **C-1 (Crash)**: `DriftLossScheduler.step` counter name shadowed the class method → *Crashed on first call*. **Fix**: Renamed to `_step_count`.
  * **C-2 (Dead Code)**: `DriftTrainingWrapper` not wired into Trainer → *Augmentation was unused (pure FM loss)*. **Fix**: Wired into `train_epoch()`.
  * **C-3 (Leak)**: `DriftConditioner` created fresh `nn.Linear` layers on every forward pass → *Noisy conditioning & CPU/GPU mismatches*. **Fix**: Moved to `__init__`.
  * **M-1/M-2 (Math)**: Detached reference encoder (`with torch.no_grad()`) mapped samples toward a fixed, random cluster.
  * **M-3 (Math)**: Inverted gradient sign performed gradient ascent, pushing trajectories *away* from expert distribution. **Fix**: Inverted to gradient descent.
  * **D-1 (Port)**: JAX-based force-field algorithm completely replaced by parametric MLP distance.
* **Infrastructure**: Standardized remote GPU execution via SLURM scripts (`train_drifting.sh`, `eval_drifting.sh`).

***

## Gen3v4: iMeanFlow (iMF) Adaptation Forensic Audit & Upgrade (May 21, 2026)
* **Logs**: [`Audit`](./Gen3v4_imf/Audit_Fix6/AUDIT_REPORT.md) & [`Changelog`](./Gen3v4_imf/Audit_Fix6/CHANGELOG.md)
* **Critical Issues & Fixes**:
  * **BUG-01**: high-precision `torchdiffeq` backend was silently ignored in rollout → *Euler sampler fallback*.
  * **BUG-02/BUG-03**: Missing `loss_discount` and `gradient_accumulate_every` in config → *Muted future timesteps and 5x effective learning rate*. **Fix**: Restored `loss_discount: 1.0` and `gradient_accumulate_every: 2`.
  * **BUG-04/BUG-05**: Empty projection costs and CFG parameters dropped in UNet model wrapper.
  * **MATH-01/MATH-02**: Auxiliary `v` head trained to predict zero against a zero target, with serial dependency on u-head (`aux = aux_head(velocity)`). **Fix**: Implemented real continuous Mean Flow objective and parallel independent head structure.
  * **MATH-03/MATH-04**: Standalone samplers used wrong reverse 1→0 integration and incorrect noise scale ($\sigma=1.0$ vs $\sigma=0.5$). **Fix**: Standardized to forward 0→1 integration and matching noise scales.
  * **MATH-05**: Step-size `h = t - r` parameter silently dropped in UNet. **Fix**: Integrated step-size embedding layers inside spatial blocks.

***

## Gen7: Multi-Variant State Contamination ("Frozen Problem") (May 21, 2026)
* **Logs**: [`Bug Report`](./Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/KEY_fix_6/BUG_REPORT.md) & [`Changelog`](./Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/KEY_fix_6/CHANGELOG.md)
* **Investigation & Fixes**:
  * **Symptom**: Sequentially evaluated variants (`[diffuser, post_processing, model_free]`) produced byte-for-byte identical plans.
  * **Cause A**: In-process expert video generation shifted camera views (`bp_image std = 0.1978` vs clean `0.2093`) due to dirty scene compiling.
  * **Cause B**: YAML had `constraint_types: []` → DPCC projector was a no-op, forcing pp ≡ mf ≡ raw FM.
  * **Fixes**: Moved expert gen pre-loop (AUDIT-FIX-1), re-enabled `['bounds', 'dynamics']` constraints (AUDIT-FIX-2), and set variant-specific result paths (AUDIT-FIX-3).

***

## Gen7 / Gen6v4: Process-Global Isolation & Teardown (Fix 7 & 7.2) (May 21, 2026)
* **Logs**: [`Bug 7`](./Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_7/BUG_REPORT_7_Audit.md) · [`Post-Mortem 7`](./Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_7/POSTMORTEM%26change_log.md) · [`7.2 Plan`](./Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_7/7.2/PLAN_FIX7.2.md) · [`7.2 Changelog`](./Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_7/7.2/CHANGELOG_FIX7.2.md)
* **Chronicle**:
  * **Fix 7 (Counter Leak)**: `MjRobot.GLOBAL_MJ_ROBOT_COUNTER` was process-global. Expert gen advanced it `0 -> 1`, forcing later variants to compile under `"rb1"` body names. This shifted cameras and mutated visual features. **Cure**: Reset counter to `0` pre-loop and in each variant's `finally:` teardown.
  * **Fix 7.2 (Cache Collision)**: Name parity (`"rgbd_cage"`) caused the variant renderer to hit `__RENDER_CTX_MAP` in `mj_render_singleton.py`. It returned the stale expert context, freezing trajectories. **Cure**: Injected `reset_singleton()` pre-loop and in variant teardowns. Verified perfect `bp_image std = 0.2093` restoration.

> [!IMPORTANT]
> **Forensic Post-Mortem**: For the complete commit timeline and technical breakdowns, see [`POSTMORTEM_FIX7.2.md`](Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_7/7.2/POSTMORTEM_FIX7.2.md).

***

## Gen7 / Gen6v4: MPC Logic Recovery, Config Refactoring & Codebase Organization (May 21, 2026)

**Keywords**: MPC recovery, `mpc_batch_size` alignment, `clip_denoised` config-driven, SLSQP projection restoration, Data Analysis v3, SLURM logs validation, Codebase Archival.

### 1. Master MPC Inference Loop Recovery (Fix 8 / Upgrade 8)
* **Plan & Research**: Authored [`PLAN_FIX8_MPC_RECOVERY.md`](Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/upgrade_8/PLAN_FIX8_MPC_RECOVERY.md) and [`RESEARCH_BATCH_SIZE_TRAJECTORY_SELECTION.md`](Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/upgrade_8/RESEARCH_BATCH_SIZE_TRAJECTORY_SELECTION.md) to reconstruct full MPC logic.
* **Batch Size Parametrization (`mpc_batch_size`)**: Renamed and synchronized all candidate batching parameters to `mpc_batch_size` (e.g., set to 4 in training/evaluation) across scripts (`eval_visual_aligning_dpcc.py`, `eval_fm_visual_aligning.py`) and config files to avoid collision with standard training batch sizes.
* **Trajectory Selection & Clamping (`clip_denoised`)**:
  * Made the `clip_denoised` parameter config-driven in `aligning-d3il-visual.py`, preventing unwanted hard clamps inside early denoising chains unless explicitly required.
  * Corrected DPCC selection logic to choose candidate trajectories based on `minimum_projection_cost` rather than random indices, guaranteeing MPC selects mathematically optimal plans.
* **SLSQP Projection Optimization**:
  * Reverted initial-state anchoring modifications to original projection logic configurations.
  * Enhanced visualization outputs in PNG/HTML reports to display all computed candidate trajectories for debug traceability.
  * Documented changes in [`CHANGELOG_FIX8.md`](Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/upgrade_8/CHANGELOG_FIX8.md).

### 2. Configuration & Path Standardization
* **YAML Path Resolution**: Updated parser utilities to support correct YAML config paths during visual-aligning evaluation.
* **Nested Plan Path Saving**: Fixed evaluation path generation in `config/aligning-d3il-visual.py` to ensure all parallel runs save diagnostic logs and trajectory results to correctly nested directories matching loaded weight parameters.

### 3. Data Analysis & SLURM Runs
* **Scientific Runs Validation**: Executed multi-seed evaluations for both `ODE1_FM` and `ODE1_vs_Diffusion1` configurations on SLURM cluster nodes.
* **Visual Plots Update**: Dispatched Data Analysis (DA) scripts to process loaded `.npz` files, generating heatmaps, success rate charts, and computation time metrics.
* **SLURM Tracking**: Updated [`important_runs.md`](../Slurm_Codes/logs/important_runs/important_runs.md) to serve as a high-fidelity audit trail for cluster evaluations.

### 4. Workspace Hygiene & Codebase Archival
* **Workspace Cleanliness**: Moved outdated/legacy directories (e.g. legacy `diffuser_visual_aligning(Outdated)`) into an `Archived_Codes/` directory to prevent namespace/importer pollution, ensuring 100% clean development dependencies in the primary project tree.

## Gen7 / Gen6v4: Data Analysis Updates & Baseline Evaluations (May 22, 2026)

**Keywords**: Data Analysis module, Visual Aligning DA, MPC foresight visualization, working baselines.

### 1. Data Analysis (DA) Pipeline Upgrades for Visual Aligning
* **DAv3 Visualizer Enhancements**: Implemented a rebuilt DAv3 adding a custom seed comparison mode, per-seed scatter visualizations, and missing seed detection in the DA pipeline.
* **MPC Foresight & Context Logging**: Upgraded the diagnostics to include context info logging and MPC decision-point foresight stride visualization (high-res plots with improved Z panel overlays).
* **NPZ Enhancements**: Added raw seed data tracking and updated data loading logic to support comprehensive raw data analysis.

### 2. Working Baselines Verification
* **Visual Aligning DPCC (Working Run)**: The baseline DPCC configuration located at `FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/visual_aligning_dpcc/H8_K20_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VTrue_steps900_bs64` was evaluated. Note that this is not a final run, but a working run that returns quite good results.
* **FM Visual Aligning (Working Run)**: The continuous-time FM architecture evaluation at `FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_a1.5_b1.0_aw1_VTrue_steps1000_bs64/H8_K100_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_VTrue_mpc4/6/results` was completed. Similarly, this is not a final run, but a working run that also returns quite good results.

***

## [DANGER]Gen7 / Gen3: API Rename - Flow Matching Classes (May 25, 2026)

**Keywords**: API Rename, Flow Matching, GaussianDiffusion, FlowMatchingODE, Danger Change.

* **DANGER CHANGE**: Renamed `GaussianDiffusion` to `FlowMatchingODE` across 4 active FM modules (`FMv3ODE`, `FMDrifting`, `FMiMeanFlow`, `FMVisual`).
* **Other Renames**: `iMFDiffusion` to `iMeanFlowODE`, `VisualGaussianDiffusion` to `VisualFlowMatching`.
* **Details**: For the full audit trail and breakdown of changed files, refer to [`CHANGELOG_API_RENAME.md`](./API_UPDATE/CHANGELOG_API_RENAME.md).

***

## Gen7 / Gen6v4: API Rename Completion, Checkpoint Patching & Geometric Constraint Upgrades (May 26, 2026)

**Keywords**: Checkpoint Patching, RemapUnpickler, UF-13 Non-Visual recording, UF-14 Geo Constraint Sweep, UF-15 Constraint Geometry Visualisation, MPC foresight SVG overlays.

1. **API Rename Completion & Checkpoint Patching / Reverting Tools**:
   * Refined module-specific renaming of diffusion classes: `FlowMatchingDrifting` (for drifting FM) and `FlowMatchingIMF` (for iMeanFlow base class) to perfectly isolate class types per algorithm variant, while `FlowMatchingODE` remains standard for FMv3ODE and FMVisual.
   * Built a robust, production-grade scan-and-fix checkpoint remapping utility (`patch_legacy_checkpoints.py` and `revert_legacy_checkpoints.py` under `logs_in_develop/API_UPDATE/Tools/`).
   * Implemented a custom `RemapUnpickler(pickle.Unpickler)` that intercepts outdated class namespaces (e.g., `GaussianDiffusion`, `iMFDiffusion`, `VisualGaussianDiffusion`) on load and dynamically maps them to the new class names.
   * The tools automatically scan entire checkpoint directory subtrees, patch `.pkl` binary serializations and `args.json` text configs, and rename the root folder structure safely (with high-protocol binary re-encoding and `--dry-run`/`--backup` options), or revert them cleanly back to legacy states for 100% bidirectional parity.
2. **UF-13: Non-Visual Mode GIF/Video Capture Restoration**:
   * Resolved a critical diagnostic gap where evaluations in non-visual mode (`if_vision=False` in configs) completely suppressed GIF and video generation even when explicitly requesting recording (`--record gif`/`video`/`all` flags).
   * Patched both `eval_fm_visual_aligning.py` (Gen7) and `eval_visual_aligning_dpcc.py` (Gen6v4) to automatically promote `if_vision` to `True` during simulator instantiation if a recording mode is active, cleanly generating execution GIFs with console warning transparency.
3. **UF-14: Multi-Variant Geo Constraint Evaluation Sweep**:
   * Replaced static top-level constraint configuration with a fully dynamic, named `geo_constraint_variants` sweep outer loop. Both evaluation scripts now sequentially run all active geometries (`no_constraint`, `dynamics_only`, `bounds_only_1`, `combined_2`, etc.) in a single job execution, saving trajectories to logically structured subtrees: `results/{geo_name}/{variant}/`.
   * Re-architected constraint tightening to reside at the geo-constraint level instead of duplicating projection variants. A tightened sibling `{geo_name}-tightened` is automatically instantiated for any active geometry containing bounds, halfspaces, or obstacles when a global `enlarge_constraints` is defined.
   * Integrated full 2D `halfspace` constraints parsing with the SLSQP projector and the outer loop sweeps (reorienting boundaries inward during tightening), and standardized 2D/3D workspace boundaries (`bounds_only_1` at x-y 2D plane, `bounds_only_2` / `combined_2` at full 3D xyz box limits).
   * Authored `UF14_investigation_constraint_loading.md` analyzing the brittle integer-index mapping and redundant bounds declarations in legacy DPCC architectures, ensuring our modular design is structurally safe against item indexing shifts.
4. **UF-15 & UF-15.2: Automated Constraint Visualization & MPC Foresight Overlays**:
   * **UF-15**: Integrated pre-run constraint diagnostics. The eval scripts now generate a self-contained 3-panel `constraint_overview.png` at the start of each geo entry:
     * *Panel 1 (3D View)*: Steelblue semi-transparent wireframe workspace box & tomato obstacle spheres.
     * *Panel 2 (XY Top-down View)*: Feasible bounding rectangle, obstacle circles, and darkorange halfspace lines with directional feasible-region normal arrows.
     * *Panel 3 (XZ Side View)*: Floor/ceiling box limits.
   * **UF-15.2**: Upgraded `VisualAgentWrapper` diagnostics to dynamically overlay active constraint geometries directly onto the MPC foresight SVG charts (`rollout_N_mpc_foresight.svg`). Planned and candidate end-effector trajectories are now visually mapped against the steelblue bounding box wireframes, darkorange halfspaces, and tomato obstacle zones, providing absolute validation of safety boundaries.

***

## Gen7 / Gen6v4: Constraint Metrics, Advanced Geo-Visualization & Native D3IL Aligning Pipeline (May 27, 2026)

**Keywords**: constraint metrics, violation tracking, active_geo_variants, halfspace/obstacle visualization, D3IL native pipeline, prior model evaluation.

1. **UF-16.3: Constraint Satisfaction and Violation Metrics**:
   - Implemented `check_trajectory_constraints` and `_check_planned_violations` for deep evaluation of geometric constraints.
   - Enhanced `VisualAgentWrapper` to track constraint metrics (execution vs planning violation rates) across rollouts and log them to JSON and console summaries.
   - Authored `CONSTRAINT_METRICS.md` as a comprehensive reference guide.
2. **UF-15.3 - UF-15.5: Enhanced 2D/3D Geometric Visualization**:
   - Upgraded halfspace visualization (`_hs_xy_draw`) to fill infeasible regions with semi-transparent polygons for clarity, and added 3D boundary plane rendering spanning the workspace.
   - Improved obstacle rendering in both 2D and 3D panels.
   - Fixed parametric clipping to correctly represent halfspaces.
   - Documented reading guides in `READING_CONSTRAINT_PLOTS.md`.
3. **UF-16.2: Selective Geo Entry Execution & PC Debugging**:
   - Added `active_geo_variants` support via YAML to run selected geometry variants selectively, speeding up evaluation pipelines.
   - Relaxed overly strict constraints and suppressed noisy SLSQP per-sample logging for a cleaner debugging experience.
4. **D3IL Native Visual Aligning Pipeline Integration**:
   - Authored `D3IL_Native_Visual_Aligning_Pipeline_Guide.md` to establish a standardized pipeline for training and evaluating D3IL native visual aligning agents (e.g., DDPM-MLP), matching the FMPCC Slurm submission workflow.
5. **Prior Model Evaluation Documentation & Independent Constraint Skewing**:
   - Audited the prior model script (`scripts/eval_prior_model.py`) and documented its behavior regarding environment setup and rollout performance.
   - Refined `combined_5` halfspace skewing for task difficulty and verified visualization via `plot_yaml_constraints.py`.

***

## Gen7 / Gen6v4 & D3IL Baseline: Dual Boundary Representation, Final Box Angle Tracking & Baseline Framework (May 28, 2026)

**Keywords**: dual boundary, final box angle, dashed line, D3IL baseline, epoch loop, DiffusionMLPNetwork sequence expand.

1. **UF-16.4: Dual Boundary Representation & Final Box Angle Tracking**:
   - Enhanced constraint visualization (`_hs_xy_draw`) to support dashed line representation for planning boundaries, differentiating nominal and planning constraints.
   - Upgraded `VisualAgentWrapper` to calculate and store the final box position and angle (via 3-point planar estimation) at the end of rollouts.
   - Introduced detailed logging for the final box state (distance to target, orientation angle).
   - Authored `SUCCESS_MODE_ANGLE_EXPLAINER.md` detailing success criteria and mode definitions.
2. **D3IL Visual Aligning Baseline - Core Framework Implementation**:
   - Introduced `train_d3il_visual_aligning.py` and `eval_d3il_visual_aligning.py` for training and evaluating native D3IL aligning agents with Hydra configuration.
   - Built a complete SLURM job suite (`train_d3il_baseline.sh`, `eval_d3il_baseline.sh`, `pipeline_d3il_baseline.sh`) for automated deployment.
   - Authored `PLAN.md`, `USAGE.md`, and `CHANGELOG.md` inside `logs_in_develop/D3IL_Visual_Aligning_RUN/` to document the new baseline setup.
3. **D3IL Baseline Bug Fixes & Agent Support Initialization**:
   - **Epoch Loop & Wandb Refactor**: Repaired the training script by adding an outer `_train_vision()` epoch loop to mirror state-based agents and ensure complete training cycles. Refactored WandB config initialization.
   - **DiffusionMLPNetwork Bug Fix**: Patched a tensor dimension mismatch in `d3il/agents/models/diffusion/diffusion_models.py` where 3D visual paths were missing sequence expansion (`.expand(-1, x.shape[1], -1)`).
   - **Submit Script Hotfix**: Fixed `Slurm_Codes/submit.sh` to correctly forward extra arguments (like `agent_name`, `seed`) and dispatch the correct `train_vision_agent` payload.

***

## Gen9 Epoch 1: Visual Avoiding Task Initialization & Data Collection (May 29, 2026)

**Keywords**: Gen9, visual avoiding, data collection, inhand camera, bp-cam, .gitignore.

1. **Visual Data Collection Pipeline**: Created a standalone pipeline (`collect_visual_avoiding_data.py`) to collect both `bp-cam` (third-person/cage) and `inhand-cam` (first-person/wrist) frames for the D3IL avoiding task by replaying existing state expert demonstrations in MuJoCo.
2. **In-Hand Camera Integration**: Sourced the wrist view directly from `env.robot.inhand_cam` instead of duplicating `bp-cam`.
3. **Pipeline Refactoring**: Added SLURM job wrappers (`collect_visual_avoiding.sh`) with proper EGL setup for offscreen rendering, and preflight checks to ensure camera data is not corrupted.
4. **Repository Maintenance**: Updated `.gitignore` to avoid checking in the generated visual datasets (`all_data`).
5. **Documentation**: Authored `CAMERAS_IN_D3IL_AND_VISUAL_ALIGNING.md` detailing the D3IL camera definitions and how the pipeline consumes them.

***

## Gen7 & Gen6v4: UF-17 Non-Visual Aligning Architecture Fix (May 29, 2026)

**Keywords**: UF-17, StateOnlyAligningDataset, action_dim=3, pure DPCC principle.

1. **Non-Visual Architecture Fix (UF-17)**: Fixed the structurally broken state-only aligning pipeline. Changed `action_dim=2` back to `action_dim=3` to match the visual path and restored the 23D trajectory `[act(3) | des_c_pos(3) | c_pos(3) | box_pos(3) | box_quat(4) | tgt_pos(3) | tgt_quat(4)]`. This places `c_pos` correctly at dims 6-8 for the projector.
2. **StateOnlyAligningDataset**: Introduced `StateOnlyAligningDataset` that produces 23D trajectories without image keys, explicitly built for non-visual training.
3. **Evaluation Restoration**: Rewrote the prediction branch to provide full 20D observation to the model instead of collapsing it to a fake 6D vector (which previously discarded box and target info). Restored pure DPCC architecture with proper initial state pinning.
4. **Flow Matching p_losses Fix**: Added an `if_vision` guard in `VisualFlowMatching.loss()` to route directly to base `p_losses` without requesting missing image conditions.

***

## Gen7 & Gen6v4: UF-18 Non-Visual Aligning Architecture Fix Completion (May 30, 2026)

**Keywords**: UF-18, non-visual adjustments, DPCC, FM sync.

1. **Non-Visual Adjustments**: Applied critical non-visual adjustments and fixes to DPCC and FM evaluation and training scripts. This finalizes the sync between Gen7 and Gen6v4 non-visual pipelines.

***

## Data Analysis Tool v2: U_2 Dynamic Compare Upgrade (May 30, 2026)

**Keywords**: DA v2, U_2, dynamic compare, constraint metrics, interactive plotting, final_xy_dist.

1. **Extended Metrics Extraction**: Upgraded the `data_loader.py` to recursively extract all `constraint_metrics` and extended `context_info` (e.g., `final_xy_dist_m`, `final_box_xy`) from diagnostic JSON files.
2. **Interactive Dynamic Compare Mode**: Added a "COMPARE" mode to the PyScript HTML visualizer, enabling dynamic generation of scatter, bar, and box plots comparing metrics across variants (e.g., highlighting `final_xy_dist_m` vs `mean_dist_m` or `constraint_sat_rate`).

***

## Gen11 Epoch 1: MuJoCo MPC UAV Model Initialization (May 30, 2026)

**Keywords**: Gen11, UAV, MPC, MuJoCo.

1. **UAV Model Assets Migration (Epoch 1 Completed)**: Successfully completed Epoch 1 of the Gen11 UAV pipeline. Migrated the Skydio X2 quadrotor model (XML, low-poly mesh, textures, and racing gates) from `mujoco_menagerie` and applied the MJPC drone-task modifications. These assets are now correctly placed within the `d3il` MuJoCo environment models directory, laying the physical groundwork for the UAV Visual-Trajectory environment.

***

## Data Analysis Tool v2: U_2 Hotfixes (May 31, 2026)

**Keywords**: DA v2, COMPARE fallback, DataLoader nested schemas, PyScript stderr suppression, geo-layer filters.

1. **Hotfix 1 (COMPARE Mode Fallback)**: Implemented an aggregated data fallback for COMPARE mode, allowing it to function and display bar and scatter charts even without per-rollout detail CSVs.
2. **Hotfix 2 (DataLoader Schema Auto-Retrieval)**: Upgraded `data_loader.py` to seamlessly parse nested directory structures introduced by geometric constraints (e.g., `combined_5`), resolving candidate loading failures without requiring CLI flag changes.
3. **Hotfix 3 (UI Filters & PyScript Stability)**: Added Geo-Layer UI Filters to declutter variant checkboxes. Executed a global standard error suppression (`sys.stderr` buffer redirection) to prevent PyScript from displaying fatal red-box overlays triggered by non-fatal Matplotlib warnings.

***

## Gen11 Epoch 2: UAV Naive Test Framework Closure (May 31, 2026)

**Keywords**: Gen11, Epoch 2, cascaded PID, 9D trajectory, hover instability.

1. **Architectural Hypothesis Validated**: Proven that planning and control can be safely decoupled. A hand-written cascaded PID flight controller successfully tracked a 30-second 9D `[p, v, a]` reference trajectory with 2.9 cm RMS error.
2. **Hover Instability Diagnosis**: Documented a discrete-time limit cycle failure during stationary hover, caused by overly aggressive rate damping (`Kp_omega`) for the 100 Hz simulation rate. This was classified as a known-acceptable tuning gap since the downstream FM-PCC pipelines produce continuously-moving paths, avoiding the zero-velocity instability.

***

## Gen11 Epoch 3: UAV Environment Test Framework Closure (May 31, 2026)

**Keywords**: Gen11, Epoch 3, MuJoCo scenes, obstacle traversal, asset-path resolution.

1. **Environment Integration**: Successfully loaded the Skydio X2 quadrotor into full MuJoCo scenes (`empty`, `corridor`, `pillars`, `s_curve`). Resolved a critical XML asset-path resolution bug that caused mesh loading failures by overriding compiler mesh/texture directories.
2. **Controller Scene-Agnosticism Proven**: The `empty` scene test achieved the exact same 2.9 cm RMS error as the Epoch 2 baseline, confirming the PID flight controller behaves identically with or without a surrounding scene wrapper.
3. **Obstacle Demos & Tracking Status**: Demonstrated clean end-to-end traversal in the `corridor` scene and mild-grazing traversal in the `pillars` scene. The `s_curve` trajectory resulted in lag and collisions, traceable to the same zero-velocity tuning gap identified in Epoch 2.

***

## Gen3v4u2: iMeanFlow Major Upgrade Direct (Fix 1) (May 31, 2026)

**Keywords**: iMeanFlow, training target math bug, velocity over-scaling, linear interpolant.

1. **Velocity Target Math Fix**: Identified and resolved a critical bug in `imf_diffusion.py` where the training target `(x_start - x_r)/h` over-scaled the velocity regression by a factor of roughly `N` at small timesteps. 
2. **Correction**: Updated the target formula to `(x_t - x_r)/h`, which correctly equals the instantaneous velocity `v` for linear interpolants, resolving the issue where model rollouts would shoot off in chaotic straight lines. Retraining is required to benefit from the corrected scale.

***

## Gen7 & Gen6v4: Non-Visual Aligning Code Fixes 18.3–18.6 (May 31, 2026)

**Keywords**: non-visual evaluation, normalizer guards, diagnostic block, projector dims, GIF capture hook.

1. **UF-13 Normalizer Guard (Fix 18.3)**: Fixed a broadcast crash during non-visual evaluation by guarding the UF-13 auto-visual override. The evaluation now correctly identifies when a checkpoint lacks an image encoder and proceeds with non-visual rollouts instead of forcing visual dependencies.
2. **First-Replan Diagnostic Alignment (Fix 18.4)**: Addressed an `UnboundLocalError` in the evaluation scripts by branching diagnostic variable names (`obs_6d_np` vs `obs_20d_np`) depending on whether the code path is visual or non-visual.
3. **Projector Normalizer Slicing (Fix 18.5)**: Fixed the `setup_dpcc_projector` logic to slice the observation normalizer based on the trajectory dimension minus action dimension (e.g., 20 for non-visual) instead of a hardcoded 6. This resolved a `(23,) vs (9,)` broadcast crash during DPCC evaluation constraints construction.
4. **Environment-Based GIF Capture (Fix 18.6)**: Implemented `record_sim_frame(env)`, an environment-render hook directly pulling from MuJoCo cameras during non-visual rollouts. This restores GIF generation for genuine 23-D non-visual models, matching the legacy visual behavior independently of the policy's image capabilities.
5. **Dimension Invariants Documentation**: Created comprehensive documentation (`0_READ_ME_DIM_INVARIANTS.md`) standardizing the exact dimension rules for visual (9-D) and non-visual (23-D) data handling across both Gen6v4 and Gen7 models.

***

## Gen3v4u2: iMeanFlow Major Upgrade Direct (Fix 2 Investigation) (May 31, 2026)

**Keywords**: iMeanFlow, jittery trajectory, aux head, test methodology.

1. **Jittery Trajectory Diagnosis**: Investigated why iMF rollouts exhibited step-quantized jitter despite the Fix 1 correction preventing explosions. Identified the auxiliary head (`iMFTrajectoryModel.aux_head`) as the most probable cause, as it introduces step-to-step noise when fed with drifting off-manifold observations during Euler integration.
2. **Inference Monkey-Patching**: Designed a runtime monkey-patch (`disable_aux_at_inference.py`) to silence the auxiliary head during sampling without requiring source code edits or retraining. This provides an immediate, cheap hypothesis test for the jitter issue before progressing to costlier architectural changes like modifying the training `(t, h)` joint distribution.

***

## Gen7: Non-Visual One-Shot Run & Stabilization (Fix 18.1 - 18.6.1) (June 1, 2026)

**Keywords**: Non-Visual, One-Shot, obs_dim override, normalizer, broadcast crash, color swap, STALE_CONFIG.

1. **Non-Visual Training Dimension Override (Fix 18.1)**: Fixed a training crash (model building 9-D input while expecting 23-D data) by dynamically overriding `args.obs_dim` to match the dataset normalizer for non-visual pipelines.
2. **Evaluation Slicing and Safeguards (Fix 18.2 - 18.5)**: 
   - Derived `_traj_dim` directly from saved normalizers to prevent CLI flag mismatches.
   - Guarded the UF-13 auto-visual override by checking the saved normalizer dimension, avoiding a `(1,6) vs (20,)` broadcast crash when evaluating genuine non-visual models.
   - Branched first-replan diagnostic variables (`obs_6d_np` vs `obs_20d_np`) to eliminate an `UnboundLocalError`.
   - Updated `setup_dpcc_projector` to slice the obs normalizer to `trajectory_dim - action_dim`, fixing a projector initialization crash for non-visual variant evaluation.
3. **Non-Visual GIF Hook & Color Correction (Fix 18.6 & 18.6.1)**: Added an environment-render hook (`record_sim_frame`) specifically for non-visual rollouts to safely pull from MuJoCo cameras, fully restoring GIF capabilities. Promptly fixed a cosmetic RGB/BGR color-inversion bug introduced by this hook.
4. **Stale Config Side-Patch**: Updated `utils.Config.save()` to forcefully overwrite `model_config.pkl`, preventing legacy configs from misleading eval scripts and causing shape-mismatch crashes.

***

## Gen3v4u2: iMeanFlow Forensic Audit vs. Reference Repo (June 1, 2026)

**Keywords**: iMeanFlow, reference audit, aux head, t-conditioning, deviation analysis.

1. **Mathematical Verification of Target**: Verified that the previously implemented training target formula `(x_t - x_r)/h` is mathematically correct and identical to the original MeanFlow formulation.
2. **Auxiliary Head Jitter (Deviation A)**: Conducted a deep code-level audit of the reference image-domain `imeanflow` repository and confirmed that it completely discards the auxiliary `v` head at inference. Validated that our practice of adding `sample_aux_weight * aux` was the root cause of step-to-step jitter, requiring a permanent codebase change.
3. **Time vs. Step Conditioning (Deviation B)**: Identified that our model conditions on both time `t` and step size `h`, whereas the reference architecture uses `h` alone. Proposed evaluating the impact of dropping the time dependency during inference.

***

## Conceptual Math & Architecture Notes: One-Shot vs. Horizon (June 1, 2026)

**Keywords**: Conceptual, DGM time, real-world horizon, MPC, one-shot generation.

1. **Orthogonality of Time Axes**: Documented the critical distinction between "diffusion time" (NFE) and "trajectory horizon time" (H). "One-shot" (NFE=1) collapses diffusion iterations but does not imply generating the entire real-world trajectory at once.
2. **MPC Chunking vs. Open-Loop**: Clarified that applying iMF at `H=8` is fundamentally correct. Long-horizon (H ≈ 300) open-loop generation suffers from compounding state drift and multi-modality collapse, confirming that small-H Model Predictive Control (MPC) with replanning remains the most robust design choice for contact-rich manipulation tasks.
3. **D3IL Agent Typology**: Classified D3IL's heterogeneous agent suite to emphasize that "no horizon" agents are actually `H=1` single-step reactive predictors, distinguishing them from chunk-based MPC policies like our `H=8` DPCC/FM implementations.

***

## Gen3v4u2: iMeanFlow Architectural Deviations Resolution (Fix 3) (June 1, 2026)

**Keywords**: iMeanFlow, reference audit, aux head, t-conditioning, fix 3, no retraining.

1. **Auxiliary Head Removal at Inference (Deviation A)**: Removed the auxiliary `v` head contribution (`sample_aux_weight * aux`) from the inference output in `_predict_velocity`. This aligns our implementation with the reference iMF architecture, which explicitly relies solely on the mean-velocity (`u`) head during inference, while retaining the aux head for training only.
2. **Constant Time Conditioning at Inference (Deviation B)**: Froze the continuous time input `t` to a constant (`T_CONST_INFERENCE = 0.5`) during the sampling loop. This converts the `(t,h)`-conditioned model into an effectively `h`-only conditioned model at inference, mimicking the reference iMF code and preventing the excitation of spurious time-dependencies learned during training.
3. **Outcome**: These structural corrections, implemented without requiring a model retrain, ensure the iMF inference pipeline strictly adheres to the canonical reference. Trajectories are expected to be significantly smoother, fully resolving the step-quantized jitter identified in previous diagnostic audits.

***

## Gen11 Epoch 4 & Path Preparations: UAV-Flow Replication and Expert Data Sourcing (June 1, 2026)

**Keywords**: Gen11, UAV-Flow, SafeFlowMPC, expert data collection, MuJoCo replication.

1. **UAV-Flow Sim-to-Real Analysis**: Analyzed the UAV-Flow reference framework and confirmed it lacks explicit dynamic equations, relying entirely on Unreal Engine for black-box physics and waypoint generation. Porting the logic to MuJoCo requires building a custom quadrotor environment and abstract geometry constraints.
2. **Architecture Transfer Strategy (V-A-FM-DPCC)**: Finalized the strategy to reuse existing drone models (e.g., Skydio X2 from `mujoco_menagerie`) and wire them with `mujoco_mpc` for receding-horizon control. Outlined a 7-step implementation roadmap for transferring the FlowMP/SafeFlowMPC paradigm from robotic arms to a UAV context.
3. **Epoch 4 Expert Data Sourcing Strategy**: Conducted a thorough evaluation of expert data sources for FM-PCC training. Ruled out using UAV-Flow directly due to simulation mismatches (Unreal vs. MuJoCo, different scales, waypoints instead of actions) and MuJoCo MPC due to its racing-task focus. 
4. **Manual Generation Approach**: Determined that manual data generation in the custom MuJoCo stack is the only viable path to obtain `(state, action)` trajectories in the required 9-D format (`[act(3) | p(3), v(3)]`). Planned the extraction of UAV-Flow kinematic statistics (e.g., typical altitudes, max velocities) to inform hand-designed reference trajectories for generating the expert dataset.

***

## Gen7 & Gen6v4: Non-Visual GIF Capture Color Pipeline Correction (Fix 18.6.2) (June 2, 2026)

**Keywords**: Non-Visual, capture_frame, RGB-BGR inversion, Aligning_Sim, color pipeline.

1. **Color Pipeline Re-Architecture (Fix 18.6.2)**: Addressed persistent R↔B color inversion in non-visual GIF captures. The previous assumptions in Fix 18.6 and 18.6.1 about native camera output formats were proven empirically false on the cluster.
2. **Visual Pipeline Parity**: Replaced the `record_sim_frame(env)` hook with `capture_frame(bp_np, inhand_np)`. Shifted the image acquisition and RGB→BGR conversion directly into `Aligning_Sim`'s non-visual branch, enforcing a line-for-line mirror of the proven visual capture pipeline to guarantee structurally correct colors without relying on unverified assumptions.

***

## Gen11 Epoch 4: Expert-Data Provenance and Generation Strategy (June 2, 2026)

**Keywords**: Gen11, Expert Data, Provenance, D3IL, UAV-Flow, sim-to-real.

1. **Provenance Audit**: Conducted a deep-dive audit into upstream expert data sources (`EXPERT_DATA_PROVENANCE.md`). Confirmed that D3IL relies on human teleoperators on real Franka robots (multi-modal IL), while UAV-Flow uses human expert pilots flying real drones.
2. **Generation Strategy Defense**: Formalized the defense for using "manual generation" (PID/MPC scripts) for FM-PCC drone planning validation. Since the current goal is validating a constraint-aware planner architecture rather than studying multi-modal style transfer or language conditioning, algorithmically generated constraint-respecting data provides the necessary controllability and format alignment without the prohibitive cost of real-world multi-modal collection.

***

## Gen9 Epoch 2: Single Camera Visual Avoiding Pipeline (June 3, 2026)

**Keywords**: Gen9, visual avoiding, single camera, 6-D trajectory, sphere_outside, config split.

1. **Pipeline Architecture**: Successfully scaffolded Visual-DPCC and Visual-FM pipelines for the D3IL avoiding task by porting from Gen7/Gen6v4 visual aligning pipelines. Dropped trajectory dimension from 9-D to 6-D (`[act(2) | des_xy(2) | c_xy(2)]`) and scaled vision down to a single camera (bp-only, `LATENT_DIM=64`).
2. **Dataset & Models**: Implemented `ParityAvoidingDataset` and updated `VisualUNet`, `VisualGaussianDiffusion`, and `VisualFlowMatching` to process single-camera payloads without `wrist_img`. Created `Avoiding_Img_Dataset` for D3IL-native loops. 
3. **Constraint Injection**: Configured the 6 fixed obstacles from the avoiding task as explicit `sphere_outside` projector constraints directly into the planning configurations rather than observation vector entries.
4. **Configuration Structuring**: Followed the Gen7 pattern by splitting the config into a dedicated `config/avoiding-d3il-visual.py` (mirrors `aligning-d3il-visual.py`), keeping non-visual configurations isolated from visual logic.
5. **Minor Fixes (Fix 1 & Fix 2)**: Addressed package-level import issues by fixing stale dataset class re-exports in `datasets/__init__.py`. Replaced aligning-era integer hardcodes (like `_obs_dim = 6`) with dynamic configuration properties to avoid silent dimension mismatch bugs.

***

## Gen8 Epoch 1: iMeanFlow Visual Aligning Initialization (June 3, 2026)

**Keywords**: Gen8, iMeanFlow, VisualIMF, FiLM-conditioning, visual aligning.

1. **Engine Swap**: Initiated the Gen8 extension to swap the Gen7 vanilla Flow Matching (`FlowMatchingODE`) engine with the newly audited and stabilized iMeanFlow engine (`iMeanFlowODE` from Gen3v4). This introduces mean-flow training, step-size `h`-conditioning, and dual `u/v` heads into the visual aligning architecture.
2. **VisualIMF Wrapper**: Created `visual_imf_diffusion.py` hosting `VisualIMF`, extending `iMeanFlowODE`. It unpacks multi-camera FiLM conditions (bp, in-hand) and delegates to the base iMF `p_losses` and forward methods, perfectly mirroring the Gen7 logic.
3. **Model Integration**: Integrated `VisualUNet` (FiLM-conditioned) as the primary velocity-net within `iMFTrajectoryModel`, allowing visual data to flow into the iMF core, while retaining the unconditioned `aux_head` on raw trajectory data. 
4. **Configuration Reuse**: Appended `imf_visual_aligning` and `plan_imf_visual_aligning` entries directly to `aligning-d3il-visual.py`. Unlike Gen9, Gen8 introduces no dimension changes (9-D visual trajectory preserved), allowing it to cleanly piggyback on the existing visual configuration.

***

## Gen3v4u2: iMeanFlow Forensic Audit Conclusion & Fix 3 (June 3, 2026)

**Keywords**: iMeanFlow, paper-readiness, reference audit, structural alignment.

1. **Audit Completion**: Finalized the forensic architectural audit comparing the FM-PCC `flow_matcher_v3_imeanflow` implementation with the canonical reference iMeanFlow repository.
2. **Correctness Confirmed**: Verified that all critical mathematical deviations are definitively resolved. Fix 1 correctly scales the mean-flow target as `(x_t - x_r)/h`. Fix 3 correctly mimics reference inference by silencing the auxiliary `v`-head and freezing the continuous-time parameter to `t=0.5` (relying on `h`-conditioning alone during generation).
3. **Remaining Known Behavior**: Documented acceptable domain adaptations, including the use of a 1-D U-Net backbone (instead of DiT) and a shallow MLP for the `v`-head. Acknowledged an "E4 stability spike" in training due to numerical noise amplification at extremely small `h`, recommending explicit gradient clipping and an `h_min` threshold for future retrains. The inference codebase is ruled paper-ready.

***

## Gen8 Epoch 1: iMeanFlow Architecture Fix 1 (June 3, 2026)

**Keywords**: Gen8, iMeanFlow, import collision, UNet1DTemporalCondModel, FlowMatchingODE.

1. **Two-Source Copy Collision Resolution**: Fixed catastrophic package import crashes (`ImportError`) caused by colliding class names from Gen3v4 and Gen7 branches.
2. **UNet Backbone Merge (Fix 1.1)**: Merged the iMF conditioning capabilities (`h_mlp`, additive `t` conditioning) from `Flow_matcher_U_Net_v2` with the FiLM visual capabilities (`cond_mlp`, concatenated visual conditioning) of Gen7 into a single, unified `UNet1DTemporalCondModel` in `unet1d_temporal_cond.py`. Retained backward-compatible aliases.
3. **Diffusion Base Alias (Fix 1.2)**: Added a `FlowMatchingODE` alias mapping to `FlowMatchingIMF` in `diffusion.py` to seamlessly satisfy the Gen7 scaffold's inheritance imports without requiring structural changes to the core iMF ODE solvers.

***

## Gen9 Epoch 2: Single Camera Avoiding Pipeline Stabilization (Fixes 3–7) (June 3, 2026)

**Keywords**: Gen9, single camera, avoiding simulation, dimension hardcodes, camera resize, diagnostic cleanup.

1. **Context & Dimension Independence (Fixes 3 & 5)**: Updated `eval_visual_avoiding` scripts to correctly handle the 6D trajectory shape for avoiding simulation. Stripped out aligning-task-specific 4-tuple contexts (`push-box`, `target-box`) and adapted the loops to utilize random environment resets natively.
2. **Constraint & Environment YAML (Fix 4)**: Completely redesigned `visual_avoiding_eval.yaml` to configure 2D environments and upgraded the associated dynamics constraints logic.
3. **Camera Resolution Mismatch (Fix 6)**: Identified a shape mismatch crashing `MultiImageObsEncoder` (expected 96×96, received 1024×1024). Applied `cv2.resize` with `INTER_AREA` interpolation to the `BPCageCam` output to match the expert dataset format.
4. **Export & Plotting Crash Recovery (Fix 7)**: 
   - Disabled the redundant non-visual `capture_frame` hook in the visual loop that caused `video_frames` shape mismatch (96x96x3 vs 96x192x3), restoring GIF generation.
   - Guarded per-rollout 3D and Z-axis export plotting logic behind `shape[1] > 2` checks, preventing `IndexError` when processing 2D avoiding positions.

***

## Gen8 Epoch 1: iMeanFlow Visual Aligning Validation & Config Fixes (Fix 2 Series) (June 4, 2026)

**Keywords**: Gen8, iMeanFlow, model_config.pkl, eval fallback, dim inference, validation.

1. **Config Wrapper Restoration (Fix 2)**: Resolved a `FileNotFoundError` during evaluation caused by the missing `model_config.pkl`. The `iMeanFlowEngine` in `train_imf_visual_aligning.py` was being directly instantiated without the `utils.Config` wrapper used by Gen7. Wrapped the engine to ensure `model_config.pkl` is reliably generated for future training runs.
2. **Evaluation Fallback via Checkpoint Parsing (Fix 2.2 - 2.4)**: Added fallback logic in `eval_imf_visual_aligning.py` for older checkpoints that lacked `model_config.pkl`. Initial attempts to rebuild from `args.json` failed (as it's not saved for non-'train' names). Subsquently developed path parsing to infer `horizon` and `if_vision`, and extracted the critical `dim` parameter directly from the `time_mlp.1.weight` shape in the `state_*.pt` checkpoint file, avoiding hardcoded mismatches (e.g. `dim=128` vs `dim=32`).
3. **Stale Config Eviction (Fix 2.5)**: Implemented a robust validation check in the evaluation loader that verifies the loaded `vis_config.dim` against the actual checkpoint weight shape. If a mismatch is detected (due to previous buggy fallbacks writing a stale pkl), the stale configuration is evicted and correctly rebuilt from the checkpoint path, ensuring evaluation stability without requiring model retraining.

***

## Gen11 Epoch 4: UAV Expert Data Collection Pipeline Implementation (June 4, 2026)

**Keywords**: Gen11, UAV expert data, data collection, homotopy, dataset_writer, slurm.

1. **PID Controller Stability Fix**: Addressed hover/limit-cycle instability identified in Epoch 2 by reducing the aggressive rate damping `Kp_omega` (`[10,10,2]` → `[2.5,2.5,1.0]`) in both naive and environment flight controllers. This ensures trajectory generation quality without excessive obstacle contact rates.
2. **UAV-Flow Kinematic Statistics (Phase 4-α)**: Extracted and analyzed kinematic statistics from UAV-Flow evaluation trajectories (stored in `phase4_alpha_uavflow_stats.json`) to establish targets for the generator (0.3–0.5 m/s velocity, 0.7–1.1 m altitude).
3. **Expert Data Generator Pipeline (Phase 4-β)**: 
   - Implemented `generator.py` to handle scene-specific (corridor, pillars, s_curve) rollout trials with configurable homotopy classes (e.g. L/C/R paths) and controller gains (`pid_default`, `pid_high_gain`, `pid_low_gain`).
   - Added `trajectories.py` for scene-aware trajectory formulations wrapping the baseline functions.
   - Built `dataset_writer.py` to downsample the 100 Hz physics engine to ~33 Hz, apply Gaussian noise `N(0, 0.02²)` to thicken the data manifold, and package the rollouts into schema-locked pickle episodes mapping `[dt, obs(T,6), actions(T-1,3)]` with position-delta actions.
   - Introduced `collect.py` as the CLI driver with automatic rejection limits for high-collision scenes, and `stats_validator.py` to validate generated dataset statistics against the Phase 4-α targets.
4. **SLURM Automation (Phase 4-γ)**: Created `collect.sh` in the `Slurm_Codes` directory to execute batch collection across multiple scenes in parallel using SLURM arrays. Updated `USAGE.md` providing comprehensive instructions for cluster execution and local testing.

***

## Gen11 Epoch 4: UAV Expert Data Collection Refinements (June 4-5, 2026)

**Keywords**: Gen11, UAV expert data, fixes, s_curve, trajectories, noise offset.

1. **Trajectory Generation & Noise Fix (Fix 1)**: Corrected the noise model to apply a per-episode constant offset instead of per-step noise, restoring the expected `[Δp_des]` action norm distribution. Updated s_curve and pillars trajectories.
2. **Continuous Paths & Amplitude Correction (Fix 2)**: Replaced piecewise path planning with continuous `tanh` trajectories for `s_curve_scene_path` to prevent zero-velocity stops at wall ends. Adjusted amplitudes to zero for centre-pass homotopy classes.
3. **Thresholds & Segment Tuning (Fix 3 - 5)**: Tuned trajectory parameters (e.g., reverting to `k=3.66`), raised contact thresholds (e.g., s_curve to 8%), and implemented proportional-duration segments ensuring uniform velocity (e.g., ~0.57 m/s) across gap crossings. Increased SLURM abort limits to improve trial completion rates against seed variance, yielding a final dataset of 1769 accepted state-only episodes.

***

## Gen9 Epoch 2: "U2" Avoiding Evaluation Rebuild (June 5, 2026)

**Keywords**: Gen9, U2, avoiding evaluation, evaluation scripts, dataset config, importlib.

1. **Evaluation Framework Redesign**: Discarded legacy evaluation loops and replaced them with streamlined, reduced-complexity evaluation scripts for both FM and DPCC avoiding pipelines.
2. **Workstream Strategy**: Established execution plans (`PLAN.md` and `CHANGELOG.md`) that separate tasks into independent workstreams for better code maintainability.
3. **Config & Class Loading Fixes (Fix 1 - 2)**: Updated the `Parser` class to strictly enforce the `avoiding-d3il-visual` dataset configuration in eval scripts. Upgraded `load_diffusion_with_override` to use standard Python `importlib` for module loading, resolving `ModuleNotFoundError` during checkpoint hydration.
4. **VisualAgent Return Interface Enhancement**: Updated `VisualAgent` to properly return the planned trajectory output, fixing col-5 plotting gaps in FM and DPCC evaluations.

***

## Gen11 Epoch 5: Visual Collection & Validation Pipeline (June 5, 2026)

**Keywords**: Gen11, Epoch 5, visual data collection, GIF generation, mini-FM, dataloader sanity gate.

1. **Camera Image Collection (WS-A)**: Implemented `collect_camera_images.py` to replay Epoch 4 state-only episodes, injecting absolute `qpos`/`qvel` directly into the MuJoCo engine (bypassing PID action replay). Captured 96x96 images from both bird's-eye (`bp-cam`) and FPV (`fpv-cam`) perspectives using offscreen rendering. Fixed a renderer lifecycle bug (`Fix 1`) ensuring safe EGL shutdown.
2. **Trajectory GIF/Video Generation (WS-B)**: Built `generate_trajectory_gifs.py` and `assemble_gifs_from_pngs.py` to compile rendered frames into stitched (`bp-cam` alongside `fpv-cam`) side-by-side GIFs and MP4s, allowing human visual inspection of expert dataset quality.
3. **Mini-FM Sanity Gate (WS-C)**: Authored `mini_fm_sanity.py` to train a minimal Flow Matching model (`H=8`, `D=9`, 20 ODE steps) over a subset of `empty` scene trajectories. This acts as a strict structural verification gate to confirm that the `[dt, obs(6), action(3)]` schema and position-delta action conventions are sound before scaling up to full FM-PCC training.

***

## Gen9 Epoch 2: "U2" Avoiding VisualAgent Batch Trajectory Sampling (June 6, 2026)

**Keywords**: Gen9, U2, VisualAgent, plan_batch_size, trajectory sampling, batch processing, col-5 visualization.

1. **VisualAgent Batch Trajectory Generation (Fix 3)**: Identified that `VisualAgent` generated a single trajectory sample (`B=1`) per replanning step, causing col-5 visualizer plots to show thin, noisy, and unrepresentative single-path lines compared to the rich multi-seed evaluation fan in state-based avoiding pipelines.
2. **Batch Repeats and ODE Multi-seed Initialization**: Added a `plan_batch_size=4` parameter to `VisualAgent.__init__`. The inference loops in both DPCC and FM evaluation scripts were upgraded to repeat identical observation/image contexts across the batch dimension while maintaining independent random noise seeds during the diffusion ODE solver steps.
3. **Consistent Col-5 Visualization Parity**: This ensures the visual agent generates and returns multiple diverse candidate paths per inference step `(B, H, 2)`. The resulting col-5 foresight plots now present a coherent, multi-path fan that is fully consistent with established state-based diagnostic standards, while preserving acceptable inference costs for evaluation.

***

## Gen11 Documentation and Methodology Consolidation (June 6, 2026)

**Keywords**: Gen11, methodology, documentation, organization.

1. **Architecture & Planning Documentation**: Conducted a major organizational pass over Gen11. Created and structured `METHODOLOGY.md`, `PLAN.md`, and `CHANGELOG.md` files across Epochs 1 through 5.
2. **Knowledge Persistence**: Solidified the reasoning and processes for UAV sim-to-real transfer, MuJoCo environment construction, expert data generation, and visual dataset compilation into permanent architectural records, aiding long-term project maintainability.

***

## Gen11 Epoch 4 "U2": Observation Schema and Tolerance Upgrades (June 7, 2026)

**Keywords**: Gen11, Epoch 4, U2, observation schema, 9D observation, contact thresholds, SLURM.

1. **9D Observation Expansion**: Upgraded the UAV state-observation schema from 6D (`[p, v]`) to 9D (`[p_des, p, v]`) in `dataset_writer.py`. Explicitly including the desired position `p_des` allows the Flow Matching and DPCC networks to learn stronger goal conditioning, bridging the gap between raw tracking and trajectory generation.
2. **Tightened Environment Tolerances**: Reduced acceptable wall contact thresholds in the `generator.py` scenes to minimize collision events in the expert dataset. Corridor threshold was halved from 0.02 to 0.01, and S-Curve from 0.08 to 0.04.
3. **Data Collection Job Management**: Temporarily introduced, then removed `collect_all.sh` in favor of standardizing instructions for SLURM array job submission, maintaining pipeline simplicity.

***

## Gen11 Epoch 5 "U2": Visual Pipeline Correction and Quaternion Injection (June 7, 2026)

**Keywords**: Gen11, Epoch 5, U2, visual pipeline, observation indexing, quaternion, FPV camera, mini-FM.

1. **Visual Pipeline Dimension Sync**: Propagated the new 9D observation structure (`[p_des, p, v]`) across the Epoch 5 visual validation suite (`collect_camera_images.py` and `generate_trajectory_gifs.py`).
2. **Observation Indexing Bug Fix**: Addressed a critical rendering bug in WS-A and WS-B where the drone was being visually rendered at its *commanded* position (`p_des`) instead of its *actual* physical position (`p`) due to stale 6D slicing logic. Updated column slicing `obs[t, 3:6]` guarantees the drone is drawn exactly where it physically flew.
3. **Attitude-Aware Rendering Preparation**: Wired `q(T,4)` quaternion data collection into the data pipeline. This prepares the visual generators to correctly render the drone's tilt and pitch (rather than forcing it to remain flat), greatly improving image realism for subsequent visual-policy training.
4. **FPV Camera Semantics Fix**: Modified the quadrotor XML to remove `mode="trackcom"` from the tracking camera, converting it from an orientation-locked chase camera into a true body-frame FPV camera that rotates with the drone's pitch and roll.
5. **Mini-FM Gate Sync**: Updated `mini_fm_sanity.py` (WS-C) configuration to accept `OBS_DIM=9` and the newly expanded `DATA_DIM=12` tensor chunk sizes (`[actions(3) ‖ obs(9)]`) to ensure the sanity gate validates the correct U2 data schema.

***

## Gen11 Epoch 4 & 5 "U3": Trajectory Waypoints & Physics Replay (June 8, 2026)

**Keywords**: Gen11, Epoch 4, Epoch 5, U3, physics replay, trajectory waypoints, collision resolution.

1. **Deterministic Trajectory Enhancement (Epoch 4 U3)**: Resolved rotor-obstacle collisions in the expert data generator by widening corridor margins and replacing the continuous pillar weave with a deterministic 8-waypoint scheme. This significantly improves generation reliability without collision clipping.
2. **Physics Replay GIFs (Epoch 5 U3)**: Designed the "U3" visual validation architecture to generate physics replay GIFs via `mj_step` rather than `mj_forward` state-injection. This approach serves as a "digital twin," allowing visual auditing of actual contact dynamics, physical deflections, and live quaternion attitude during the rollouts.

***

## Infrastructure Hotfix: SLURM GPU Leak Resolution (June 8, 2026)

**Keywords**: SLURM, GPU leak, MuJoCo EGL, device isolation.

1. **GPU Leak Diagnosis**: Diagnosed a critical violation on the SLURM cluster where rendering tasks silently escaped to an unallocated physical GPU. Identified that `MUJOCO_GL="egl"` unconditionally opens the lowest DRM node (GPU 0), entirely bypassing `CUDA_VISIBLE_DEVICES` limits.
2. **Global EGL Pinning Implementation**: Applied a sweeping fix across 31 evaluation and training SLURM shell scripts. Eradicated legacy `EGL_DEVICE_ID=0` hardcodes and explicitly pinned the EGL renderer to the CUDA-allocated physical GPU by exporting `MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"`.

***

## Real-Time Behavior Recording: Digital Twin Audit Framework (June 8, 2026)

**Keywords**: Digital twin, real-time logging, timing audit, behavior tracking.

1. **Framework Conceptualization**: Formalized an evaluation framework (`IDEAS.md`) proposing structured text logs to capture per-step timings (e.g., `fm_ms`, `qp_ms`) and behavioral contexts (e.g., contact events, tracking error, DPCC overrides).
2. **Real-Time Feasibility Metric**: Emphasized that real-time feasibility—such as adhering to a 33 Hz (30 ms) control loop budget on hardware—is critical for deployment and cannot be validated by standard visualization GIFs alone. The log becomes the authoritative digital twin for timing and behavioral audits.

***

## Gen9 Epoch 2 "U2": Single Camera Avoiding Denoising Fix (Fix 3) (June 9, 2026)

**Keywords**: Gen9, E2 U2, clip_denoised, denoising chain divergence, DDPM explosion.

1. **Denoising Chain Divergence Diagnosis**: Investigated a catastrophic failure in the Visual-DPCC baseline where trajectory lines exploded to extreme coordinates, while Flow Matching (FM) trajectories remained well-behaved. Traced the root cause to `clip_denoised=False`, which allowed noise amplification over 100 stochastic DDPM steps to compound without bounds.
2. **Action Clamping Fix**: Switched `clip_denoised` to `True` in `config/avoiding-d3il-visual.py` for DPCC planning. This reactivates the `[-5.0, 5.0]` clamp strictly on the action dimensions during inference, bounding the trajectories. FM configurations correctly remain `False` as their deterministic ODE steps do not inject noise.

***

## Gen11 Epoch 5 "U3": Physics-Based Trajectory GIF Generation & Camera Parity (June 9, 2026)

**Keywords**: Gen11, Epoch 5 U3, physics replay GIF, FPV camera, chase camera.

1. **Physics-Based Digital Twin Generator**: Implemented `generate_physics_gifs.py` to create authentic physics replays using actual `mj_step` simulation (instead of `mj_forward` state injection). The generator produces side-by-side GIFs equipped with real-time obstacle proximity bars and red contact overlays to precisely audit physical drone behaviour.
2. **Camera Orientation Fix (Fix 1)**: Discovered the original `track` camera was behaving as a chase camera (1 m behind, 0.5 m above) rather than a true first-person view. Added a new `fpv` body-fixed camera mounted on the drone's nose to capture authentic rotational dynamics (pitch/roll), restoring realistic visual data collection while keeping the `track` camera for backward compatibility.

***

## Gen11 Epoch 4 "U3" & "U4": Expert Data Floor Crash Purge & Hover Stabilization (June 9, 2026)

**Keywords**: Gen11, Epoch 4, U3 Fix 2, U4, floor crash contamination, s_curve hover pauses.

1. **Floor Crash Contamination Purge (U3 Fix 2)**: Detected that `_is_obstacle_contact` explicitly ignored floor collisions, allowing severely degraded trajectories (e.g., 85% of `s_curve` episodes) to pass the dataset rejection filter with `contact_fraction=0`. Implemented a strict `Z_FLOOR_MARGIN` check, necessitating re-collection of contaminated scenes.
2. **Hover Stabilization (U4)**: Resolved lag-induced altitude collapse and overshoot in the `s_curve_scene_path` by inserting 1.0-second hover pauses at segment junctions. This allows the PID controller to stabilize lateral velocity before transitions. 
3. **Data Collection Pipeline Update**: Adjusted duration ranges to accommodate hover times and updated `collect.sh` with a positional argument to allow targeted re-collection of specific homotopies (e.g., `(R,R,R)`).

***

## Infrastructure Hotfix: SLURM GL Backend Deactivation (June 9, 2026)

**Keywords**: SLURM, MuJoCo GL leak, collect.sh, disabled.

1. **GPU Leak Prevention**: Updated `collect.sh` to explicitly disable the MuJoCo GL backend (`MUJOCO_GL="disabled"`) during state-only data generation. This prevents silent rendering leaks onto unallocated physical GPUs during headless trajectory generation.

***

## Infrastructure Hotfix: SLURM GPU Leak Safeguard Check (U2) (June 9, 2026)

**Keywords**: SLURM, GPU leak, MuJoCo EGL, device isolation safeguard.

1. **GPU Leak Prevention Safeguard**: Injected a 5-line GPU-check block into 31 EGL-rendering SLURM scripts and the baseline job template. This block dynamically verifies that the MuJoCo EGL device is perfectly pinned to the allocated CUDA GPU (`MUJOCO_EGL_DEVICE_ID == CUDA_VISIBLE_DEVICES`).
2. **Fail-Fast Mechanism**: If a mismatch is detected (indicating a broken pinning block and a potential leak to GPU 0), the job now immediately aborts instead of silently leaking memory for hours. Every SLURM log now persistently records the allocated GPU and EGL device at startup for straightforward auditing.

***

## Gen9 Epoch 2 "U2": Avoiding Baseline DDPM Exploded Lines Verdict & Config Reverts (June 10, 2026)

**Keywords**: Gen9, U2, DDPM, exploded lines, clip_denoised, overfitting.

1. **Failure Diagnosis**: Conducted a deep forensic audit to determine if the DDPM "exploded lines" failure was a code bug or a model capability limit. Concluded that severe overfitting is a major contributor: the evaluated checkpoint (step 99k) had a test loss 5.19x worse than the best checkpoint (step 11k). 
2. **DDPM vs FM Robustness**: Documented how DDPM's multiplicative noise schedule severely amplifies $\epsilon$ errors (up to ~1284x at early steps), turning overfitting into trajectory explosion. By contrast, Flow Matching moves a deterministic 5% along the predicted velocity vector per step, remaining structurally immune to the same overfitting cascade.
3. **`clip_denoised` Revert**: Reverted `clip_denoised` back to `False` in the configuration and training scripts, acknowledging that the parameter only alters the visual signature of the failure (clamped thresholds vs. exploded coordinates) and does not fix the underlying model divergence.

***

## Gen11 Epoch 4 & 5 "U5 - U7": UAV Expert Data Collection Stability & S-Curve Z-Route Upgrade (June 10, 2026)

**Keywords**: Gen11, Epoch 4, Epoch 5, S-Curve, Z-route, thrust-priority allocation, accepted_clip_list.

1. **Data Collection Instrumentation & Stability (U5 & U6)**:
   - Instrumented the collection generator to accurately track `contact` vs `floor` rejection reasons and modified the `n_clip` telemetry to use pre-allocation raw saturated flags for truthful saturation reporting.
   - Added +0.20 m altitude headroom (`z ~ [0.90, 1.30]`) to resolve persistent floor crashes during initial hover.
   - Re-architected the `CascadedPID` thrust allocation from a binary search to an exact-scale method with a 50% torque floor, preserving essential attitude authority and recovering success rates for cross-channel homotopies.
2. **S-Curve Z-Route Geometric Resolution (U7)**:
   - Proved mathematically that the previous single diagonal path for `s_curve` was geometrically infeasible (passing 0.019 m inside the rotor-contact zone even on the nominal path). 
   - Replaced the single diagonal with a 3-leg Z-route (pure-x, pure-y, pure-x) traversing the gap centerline, achieving strict 0.50 m clearances and resolving deterministic collisions.
3. **Accepted Episode Telemetry (U7)**: Introduced `accepted_clip_mean` and `accepted_clip_max` logging to measure whether "healthy" episodes still suffer from chronic motor saturation, closing a critical diagnostic gap.
4. **Naive Framework Sync & Camera Correction (Gen11E5U3 Fix2)**:
   - Updated the naive `run_env.py` task evaluator with the new 5-leg Z-route waypoints, resolving its previous 40.9% contact failure rate on `s_curve`.
   - Fixed `collect_camera_images.py` to correctly utilize the forward-facing `'fpv'` camera instead of the legacy 3rd-person `'track'` camera, unifying the visual collection format with the rest of the Epoch 5 pipeline.

***

## Gen8 Epoch 1 U2 & Gen3v4 U3: iMeanFlow Engine Architectural Fixes (June 11, 2026)

**Keywords**: iMeanFlow, frozen t bug, redundant sampler, true t_i, endpoint conditioning.

1. **Frozen-t Bug Fix (C1)**: Discovered that `p_sample_loop` in both Gen3v4 (`flow_matcher_v3_imeanflow`) and Gen8 (`imf_visual_aligning`) had a hardcoded `T_CONST_INFERENCE = 0.5`. Since the UNet model adds a `time_mlp(t)` to its hidden states during training with the true time, freezing `t` to a constant value at inference created out-of-distribution conditioning, resulting in chaotic rollouts. Replaced the constant `t=0.5` with the physically correct true `t_i = loop_idx / max(flow_steps, 1)`.
2. **Endpoint Conditioning Swap (C4)**: Identified a training/inference mismatch in `p_losses`. The model was being conditioned on `x_t` (the data-side interpolant) instead of `x_r` (the noise-side current state). Swapped the conditioning endpoint from `x_t` to `x_r` to properly teach the model to predict velocity from noise-side inputs. This necessitates a fresh retrain for both iMeanFlow pipelines.
3. **Dead Sampler Cleanup (C2, C3)**: Deleted redundant and inconsistent `sample()` methods from `imf_engine.py` and `sample_trajectory()`/`sample()` from `imf_trajectory_model.py`. These unused paths disagreed with `p_sample_loop` by passing true `t_cur` and mixing auxiliary velocities, which posed a high risk of corrupting future debug harnesses.

***

## Gen9 Epoch 2 U3: Avoiding Baseline DDPM Exploded Lines & Config Trap (June 11, 2026)

**Keywords**: Visual-DPCC, clip_denoised, exploded trajectories, pkl mismatch warning.

1. **`clip_denoised` Fix Restored (C1)**: Restored `clip_denoised=True` in the `config/avoiding-d3il-visual.py` plan configuration block. An earlier revert had turned it to `False`, which allows `x_recon` errors to compound exponentially across denoising steps, leading directly to exploded trajectory evaluation while training metrics appear deceptively healthy.
2. **PKL Config Mismatch Warning System (C2)**: Diagnosed a "silent precedence" trap where changes to the `.py` eval configuration were completely ignored because the script loaded `clip_denoised`, `n_diffusion_steps`, and `horizon` strictly from the `diffusion_config.pkl` saved during training. Created and injected a `_warn_pkl_config_mismatch()` helper into `eval_visual_avoiding_dpcc.py`. It explicitly compares the live `.py` config against the frozen `.pkl` config and emits a highly visible table/warning to the console and Slurm logs if any divergence is found.

***

## Gen11 Epoch 5 Closure: UAV Expert Data Visual Validation (June 11, 2026)

**Keywords**: UAV expert data, GIF validation, stop-and-go behavior, dataset closure.

1. **Validation Complete**: Successfully concluded Gen11 Epoch 5 visual validation, securing 1,952 episodes for the E4 dataset. The generated trajectory (WS-B) and physics (WS-D) GIFs confirm that drone trajectories perfectly respect scene geometry, with zero obstacle clipping/ghosting. The nose-mounted FPV camera fix was also verified correct.
2. **Stop-and-Go Trajectory Characteristics Observed**: Visual audits flagged a distinct "stop-and-go" behavior in the data: the drone decelerates to zero at every waypoint and pauses. This is due to the 1.0s hover pauses inserted to stabilize the PID controller, and the cosine velocity profile that forces zero velocity at segment endpoints. 
3. **Policy Implications**: This intentionally conservative behavior means that any downstream FM-PCC model trained on this dataset will inherit the stop-and-go flight style. Smooth, continuous flight remains out of scope for the current dataset, moving instead as an open question for Epoch 6+ generation.

***

## Gen11 Epoch 4 "U9": Smooth Trajectories & Stop-and-Go Elimination (June 12, 2026)

**Keywords**: Gen11, Epoch 4, U9, blended_path, stop-and-go, circular fillet, peak lateral acceleration.

1. **Stop-and-Go Elimination**: Implemented a new `blended_path` primitive in `uav_env_test/trajectories.py` that replaces piecewise linear segments with a globally smoothed cosine speed profile. Interior corners are now cut using circular fillets, completely eliminating the zero-velocity hover pauses that plagued previous Epoch 4 datasets.
2. **Path Updates**: Migrated `pillar_path` and `s_curve_scene_path` to use the blended chain. Verified using a new numerical verifier (`verify_blends.py`) that strict geometric clearances (e.g. 0.43m for pillars, 0.31m for walls) are flawlessly maintained without requiring the MuJoCo runtime.
3. **PID Saturation Fix (Fix 1)**: Detected that tight corner fillets in mixed-homotopy pillar scenes (LRL/RLR) demanded up to 8.6 m/s² lateral acceleration, exceeding the cascaded PID limits and causing a >45% rejection rate. Increased `BLEND_RADIUS` from 0.30 m to 0.45 m, reducing peak acceleration to a manageable 5.7 m/s² and restoring >70% acceptance.

***

## Gen11 Epoch 5 "U2" Fix 2: Visual Rendering Dimension Extraction & Selective Inspection (June 12, 2026)

**Keywords**: Gen11, Epoch 5, U2 Fix 2, visual data collection, 12D schema, selective inspection, per-homotopy.

1. **Dimension Extraction Patch**: Fixed visual rendering dimension extraction for U9 compatibility, enabling visual data collection scripts (`collect_camera_images.py`, `generate_trajectory_gifs.py`) to gracefully run on the new 12D schema (3D act, 9D obs) without relying on brittle hardcoded slicers.
2. **Selective Inspection Strategy**: Introduced a `--per-homotopy N` flag to GIF generation scripts. This drastically reduces diagnostic overhead by rendering only a representative sample (e.g., 1 GIF per homotopy bucket) rather than thousands of episodes, cutting inspection time from hours to seconds.

***

## Gen9 Epoch 2 "U4": Single Camera Avoiding Pipeline All-Bugs Fix Pass (June 12, 2026)

**Keywords**: Gen9, Epoch 2, U4, RGB/BGR, 96x96 render, EMA weights, episode-level split.

1. **RGB Pipeline Parity (B1)**: Removed a redundant `[:, :, ::-1]` BGR-to-RGB flip in both DPCC and FM evaluation scripts, ensuring that visual evaluations receive native RGB frames perfectly matching the training distribution.
2. **Render Resolution Match (B2)**: Upgraded evaluation cameras to render directly at `96x96` natively, removing a `1024x1024` intermediate step and avoiding `INTER_AREA` downsampling artifacts.
3. **EMA Weights & Config Wiring (B3, B6, B10)**: 
   - `load_diffusion_with_override` now correctly returns EMA-smoothed weights (`ema_model`) instead of raw weights for enhanced evaluation performance.
   - Connected `args.mpc_batch_size` directly to the `VisualAgent` constructor, removing hardcoded defaults. 
   - Ported the PKL config mismatch warning from DPCC into the FM eval script.
4. **Data Leakage & Training Stability (B7, B8, B9)**: 
   - Implemented `episode_split()` in `sequence.py` to enforce strict episode-level train/test splits, preventing window-leakage where near-duplicate frames could appear in both sets.
   - Enforced an explicit `self.save(self.step)` at the end of training loops to prevent discarding the final 20k steps of progress.
   - Patched a state-leak bug where validation loops left the model in `.eval()` mode, silently degrading subsequent training steps; the model is now explicitly restored to `.train()` mode.

***

## Documentation Maintenance: Legacy Workspace Cleanup (June 12, 2026)

**Keywords**: Gen11, documentation reorganization, init_0.

1. **Workspace Decluttering**: Performed a mass reorganization of Gen11 logs and documentation. Moved numerous legacy Epoch 4 and Epoch 5 changelogs, execution plans, and methodology files into `init_0` subdirectories to declutter the active workspace. Relocated `U8_Stop_and_Go` to `U8X_Stop_and_Go_Ideas` to reflect its conceptual nature.

***

## Gen9 Epoch 2 "U3": Adversarial Codebase Audit (June 12, 2026)

**Keywords**: Gen9, adversarial audit, bug confirmation, documentation update.

1. **Bug Audits Confirmed**: Completed an exhaustive adversarial audit on the avoiding pipeline against the codebase (`U3_audit_Fable_ADVERSARIAL_RESPONSE.md`), confirming 8 out of 10 bug claims as genuine and 2 as partially correct.
2. **Key Findings Authenticated**: Validated a critical RGB/BGR channel swap in the evaluation scripts causing a domain gap, a render resolution mismatch (96x96 vs 1024x1024), and overlapping window data leakage during train/test splits. Corrective actions were formulated and subsequently implemented in the Gen9 U4 Fix Pass.

***

## Gen11 Epoch 4 "U10" & Epoch 5 "U4": Stress-Test Collection and Rendering (June 12, 2026)

**Keywords**: Gen11, Epoch 4 U10, Epoch 5 U4, stress test, edge cases, physics replay.

1. **Stress-Test Trajectory Generators (E4 U10)**: Implemented 8 stress cases (e.g., `extreme_speed`, `wall_crossing`, `floor_dive`, `gain_extreme`) via a new `stress_trajectories.py` builder to intentionally push the UAV tracking pipeline into failure modes. Created `collect_stress.py` to drive this collection, bypassing contact rejection to deliberately save failure/collision episodes along with stress metadata.
2. **Stress Rendering Dispatcher (E5 U4)**: Modified `generate_physics_gifs.py` to support physics replay of stress episodes by explicitly rebuilding trajectories from the stress case functions and recovering modified PID controller gains (`kp_scale`, `kd_scale`) from metadata. This ensures exact dynamic reproduction of the failure scenarios.

***

## Gen11 Epoch 5 "U5": 2-D Overview Plotting Tool (June 12, 2026)

**Keywords**: Gen11, Epoch 5 U5, 2D overview plots, matplotlib, headless rendering.

1. **Standalone 2D Plotter**: Added `generate_overview_plots.py`, a pure matplotlib, CPU-only tool that renders static top-down (XY-plane) trajectory paths without needing MuJoCo or GPU hardware.
2. **Visualization Modes**: The tool supports a `summary` mode mapping dense trajectory bundles into color-coded heatmaps over scene geometry, and a `per-episode` mode comparing the time-gradient commanded path (`p_des`) against the actual flown path (`p`).
3. **Slurm Execution**: Integrated `generate_overview_plots.sh` for headless cluster execution and patched a typo across E5 sbatch scripts to correctly use the `gpu-1-student` partition instead of `cpu-1-student`.

***

## Gen9 Epoch 2 "U4" Fix 1: EMA Selection / Deployment Consistency (June 13, 2026)

**Keywords**: Gen9, EMA selection, deployment consistency, DPCC divergence.

1. **Selection/Deployment Mismatch Fix**: Addressed a critical bug where `test()` was scoring the raw `self.model` weights while evaluation deployed `trainer.ema_model`. Modified `diffuser_visual_avoiding/utils/training.py` and `fm_visual_avoiding/utils/training.py` to correctly score the `ema_model` during validation, ensuring `state_best` is selected on the exact network that will be deployed in evaluation.
2. **DPCC Comparability Ledger**: Authored `DPCC_DIVERGENCE_AND_COMPARABILITY.md` to document how fixes like B6 (EMA eval), B7 (episode split), B8 (terminal save), and B9 (eval mode) diverge from the published DPCC code. Verified that the critical Gen9 comparison (visual-FM vs visual-DPCC) remains fair (common-mode fixes), and the `diffuser/` state baseline remains untouched and paper-faithful.
3. **Revert Plan Memo**: Drafted `MEMO_DPCC_REVERT_PLAN.md` to outline how to safely revert to exact DPCC-identical behavior if required by reviewers, while retaining the correctness improvements.

***

## Gen9 Epoch 2 "U5": 1e5 Training Steps Observation (June 13, 2026)

**Keywords**: Gen9, Epoch 2, U5, training extension, loss curves, state_best.

1. **Extended Training Analysis**: Documented observations from extending training from 1e4 to 1e5 steps. Both train and validation loss hit a trough around ~1e4 steps, then slightly rebounded and plateaued through the remaining steps.
2. **`state_best` Efficacy**: Concluded that `diffusion_epoch: 'best'` functions correctly by capturing the optimal checkpoint at the trough. While the overall curve looks healthier with longer training, the operative checkpoint is naturally captured early, verifying the robustness of the checkpoint selection logic.

***

## Gen3v2: State-Based Pipeline Cross-Check vs Fable Audit (June 13, 2026)

**Keywords**: Gen3v2, state-based avoiding, DPCC baseline, Fable audit, provenance.

1. **Audit Cross-Check**: Executed a comprehensive cross-check (`Gen3v2_FMv3ODE_DPCC_CrossCheck_vs_Fable.md`) mapping the 10 visual-avoiding bugs from the Fable audit back to the state-based DPCC and Gen3v2 pipelines.
2. **Visual-Only Separation**: Confirmed that the eval-side bugs (B1, B2, B3, B5) are exclusively visual artifacts. The state-based pipeline remains structurally clean and was proven to correctly implement `trajectory_selection` and constraint configurations.
3. **Provenance Validation**: Confirmed that training and serialization quirks (B6, B7, B8, B9) are inherited verbatim from the published `/workspaces/dpcc` codebase. Concluded these should be preserved in the `diffuser/` baseline for strict comparability but fixed in new Gen3v2 and Gen11 architectures as targeted ablations.

***

## Gen11 Epoch 5 "U5": Overview Plots Explainer (June 13, 2026)

**Keywords**: Gen11, Epoch 5 U5, 2D overview plots, PLOT_EXPLAINER, p_des vs p.

1. **Plot Documentation**: Authored `PLOT_EXPLAINER.md` to formalize the interpretation of the 2D overview plotting tool.
2. **Commanded vs Actual Tracking**: Detailed the visual distinction between the commanded `p_des` (blue time-gradient dashed line) and actual physical `p` (solid red line). Emphasized that the divergence between these lines serves as a direct indicator of PID tracking error, controller saturation, or contact events.

***

## Gen11 Epoch 6: FM-PCC Quadrotor Training Blueprint (June 13, 2026)

**Keywords**: Gen11, Epoch 6, UAV FM-PCC training, mini-FM sanity gate, DPCC projection.

1. **Epoch 6 Training Spine Formalization**: Drafted the `IDEAS.md` blueprint to transition from expert-data collection into fully closed-loop FM-PCC policy training. Outlined a 4-phase rollout: data finalization (Phase 0), mini-FM sanity checks (Phase 1), state-only FM training (Phase 2), and DPCC safety projection integration (Phase 3).
2. **Prerequisite Gating**: Established strict prerequisites for training initiation, including a mandatory re-collection of the pillars dataset to reduce rejection rates, and a "mini-FM" state-only check to mathematically verify the positional-delta action schema before committing full compute resources.
3. **Safety Projection Architecture**: Planned the integration of the DPCC SLSQP projector using a differential-flatness unlearned quadrotor model for exact state constraint evaluation against `SCENE_OBSTACLES`, effectively closing the loop on real-time safe motion planning.

***

## Gen3v4 "U4" & Gen8 "U3": iMeanFlow (iMF) MeanFlow-JVP Objective Implementation (June 13 - 14, 2026)

**Keywords**: iMeanFlow, imfv2, MeanFlow-JVP, low-NFE inference, JVP flag-gated, Gen3v4, Gen8 visual.

1. **iMF Objective Math Implementation**: Integrated the exact `MeanFlow-JVP` (imfv2) training target from the official JAX implementation into both the state-based Gen3v4 and visual-aligning Gen8 engines. This restores the missing total derivative term (`d/dt u`) using forward-mode AD (`torch.func.jvp`), anchoring the velocity field at `r=t` with 25% probability and using adaptive weighted MSE loss to prevent bootstrapped outliers. 
2. **Flag-Gated Surgical Swap (imfv2)**: Decided to keep the existing FM-equivalent objective as the default to ensure existing baseline runs are byte-for-byte unaffected. The true iMF objective is now opted-in via the `imf_objective: 'meanflow_jvp'` configuration flag, establishing a direct A/B testing mechanism against the FM baseline.
3. **Dual Backbone Support**: The single-flag implementation seamlessly propagates to both Gen3v4 (state) and Gen8 (visual), enabling "real iMF" behavior for both environments without code duplication. For the Gen8 visual pipeline, the JVP was explicitly routed around the image-conditioning to ensure functional purity.
4. **Train Script Crash Hotfix (Fix 1)**: Resolved a critical startup crash in both the Gen3v4 and Gen8 training scripts caused by a missing attribute access (`ode_inference_steps_v3`) and unforwarded imfv2 parameters. Replaced direct attribute checks with `getattr` safe defaults to ensure robust configuration parsing.

***

## Gen3v4 "U5" Phase 1: Real iMeanFlow (iMF) Enhancements on UNet Backbone (June 15 - 16, 2026)

**Keywords**: Gen3v4, U5, iMeanFlow, dual_head, interval-CFG, IMFBackbone, config.

1. **Shared-Backbone Dual Head**: Replaced the detached `aux_head` MLP with a genuine dual-head architecture (`v_final_conv`) that branches off the shared UNet features (`velocity_net`). This ensures that `meanflow_aux_weight` correctly regularizes the main network flow, achieving parity with the official `u_heads`/`v_heads` split.
2. **Interval Classifier-Free Guidance (CFG)**: Implemented interval-CFG by integrating sinusoidal embeddings for `omega`, `t_min`, and `t_max` into the time/`h` embedding logic. This enables guided sampling restricted strictly to the `[t_min, t_max]` interval, enhancing generation quality exactly as the canonical iMF reference prescribes.
3. **IMFBackbone Abstraction**: Established an `IMFBackbone` interface around the current UNet, incorporating a clear drop-in placeholder (`# TODO(real-iMF-NN)`) for future substitution with the official DiT-based iMF architecture. This allows continuous testing of the iMF *method* without immediately overhauling the underlying model.
4. **Configuration Upgrades**: Modified configuration structures to properly expose the new capabilities (`dual_head`, `interval_cfg`, `meanflow_cfg_*`). Configured the repository to default to enabling iMF behavior (`imf_objective='meanflow_jvp'`, `dual_head=True`, `interval_cfg=True`) to streamline upcoming evaluation runs.
5. **Validation Status**: Phase 1 is code complete and passes local syntax checks (`py_compile`), but remains untested on the cluster runtime. Forward-AD (JVP) stability across the new UNet interval-CFG embeds is pending empirical verification.

***

## Documentation Maintenance: Core Dynamics & Architecture Audits (June 16, 2026)

**Keywords**: DPCC, MjRobot, UAV math, dynamic model, code critique.

1. **MjRobot Critique**: Conducted a deep architectural and mathematical critique of the `MjRobot.py` file within the `d3il` stack. Identified critical dormant bugs in the quaternion velocity computation and Jacobian query logic, while confirming that the live avoiding task path remains safe due to bypassing these flawed methods.
2. **DPCC Dynamics Guide**: Authored a comprehensive guide bridging the 2D kinematic outputs of the DPCC policy to the complex multi-stage physical control (Cartesian Impedance, Inverse Kinematics, Computed Torque Feedforward) executed by the simulated Panda arm.
3. **UAV Mathematical Model**: Formalized the mathematical representation of the Skydio X2 UAV in MuJoCo MPC, clarifying the 13-dimensional kinematic state vector and how its "0 DoF" internal label translates into 6 spatial degrees of freedom via the unactuated free joint.

***

## Gen3v4 "U6": Config-Switchable Official-iMF DiT Backbone (June 16 - 17, 2026)

**Keywords**: Gen3v4, U6, iMeanFlow, DiT backbone, config-switchable, JVP-safe RoPE, trajectory adaptation.

1. **Official DiT Integration**: Ported the official `/workspaces/imeanflow/models/imfDiT.py` transformer architecture into the `flow_matcher_v3_imeanflow` codebase, adapting it for 1D trajectories (`[B, H, D]`) via a new `TrajPatchEmbedder`. This introduces a native dual-head structure with equal-depth `u_heads` and `v_heads`, addressing the conditioning bottleneck observed in the UNet.
2. **Config-Switchable Architecture**: Introduced a single configuration flag (`imf_backbone: 'unet' | 'dit'`) to seamlessly dispatch the `velocity_net` implementation at runtime. The default remains `unet`, ensuring byte-for-byte backwards compatibility for existing runs, while the `dit` branch fully conforms to the `IMFBackbone` contract without modifying the MeanFlow-JVP objective, interval-CFG sampler, or DPCC projector.
3. **JVP-Safe RoPE Adaptation**: Replaced the official complex-number-based Rotary Position Embedding (RoPE) with a mathematically identical, real-valued interleaved rotation. This crucial modification ensures the forward-mode automatic differentiation (`torch.func.jvp`) required by the MeanFlow objective remains functionally pure and differentiable.
4. **Configuration & State Management**: Plumbed the new `dit_*` hyperparameters through the configuration blocks and appended a `_bb{imf_backbone}` tag to the folder names and `diffusion_loadpath`. This prevents silent checkpoint mismatches and cross-loading errors between disjoint UNet and DiT parameter trees.

***

## Gen3v4 "U6" DiT Debugging & Architecture Tutorials (June 18, 2026)

**Keywords**: Gen3v4, U6, DiT, trajectory explosion, fake convergence, interval-CFG, tutorials.

1. **Trajectory Explosion Diagnosis**: Investigated a paradoxical issue where the iMeanFlow DiT backbone exhibited clean training convergence but produced "exploded" or chaotic trajectories during evaluation. Discovered that the evaluation metric logs remained superficially healthy because task-success metrics (like goal-reaching and discrete obstacle checks) are blind to continuous motion quality (e.g., jerkiness).
2. **Interval-CFG Amplification (H1)**: Identified the primary culprit: interval-CFG with `ω=4.0` was amplifying a meaningless null-class token during the final integration step. Because the DiT lacked the real ImageNet class labels used in the official implementation, the CFG extrapolation injected noise rather than sharpening the conditioning signal.
3. **Training Objective vs. Smoothness (H2 & §4B)**: Analyzed how the DiT's lack of a structural smoothness prior (unlike the UNet's convolutions) combined with a purely step-0-weighted velocity MSE loss led to compounding errors in multi-step generation. Diagnosed the "fake convergence" phenomenon where a self-referential bootstrapped objective minimizes a reweighted proxy loss without improving generated trajectory quality. Recommended a regime shift: higher effective batch size, lower LR, warmup, and disabled CFG (`ω=0.0`) during evaluation.
4. **DiT Operational Tutorials**: Authored comprehensive guides (`HOW_TO_TEST_DiT.md`, `DiT_Explained_For_Beginners.md`, `Eval_Call_Chain_Traj_to_Plot.md`) detailing how to properly toggle the Config-Switchable DiT backbone, execute a cluster smoke test for JVP/AD safety, and correctly interpret the generated plots versus the executing receding-horizon MPC path.

***

## Data Analysis Tool v3: `npz_analysis` Trajectory Quality Auditing (June 19, 2026)

**Keywords**: npz_analysis, trajectory quality, jerk, motion smoothness, CSV aggregation, replotting.

1. **Motion Quality Deficit Addressed**: Recognizing that existing discrete "success/violation" metrics could falsely validate chaotic models (as proven in the U6 DiT debug), developed the `analyze_npz.py` tool. This tool explicitly computes dynamic trajectory quality metrics—including `traj_straightness`, `traj_roughness`, and `traj_max_jerk`—from the `obs_all` executed path.
2. **CSV Aggregation Engine**: Built a schema-generic Python utility that recursively scans for `.npz` evaluation files across both the state-based (avoiding) and visual-aligning pipelines. Automatically aggregates per-trial means/stddevs and trajectory quality scores, exporting them into structured `files_summary_<ts>.csv` and `per_trial_<ts>.csv` reports without requiring pandas.
3. **Execution Path Replotting**: Implemented a `--replot` feature allowing users to retroactively visualize the true closed-loop physical execution path directly from the stored `.npz` coordinates. This completely bypasses the need for costly MuJoCo simulator reruns while validating the numerical quality scores visually.

***

## D3IL Visual-Aligning Baseline "U2": Paper-Faithful Evaluation Upgrade (June 20, 2026)

**Keywords**: D3IL baseline, behavior entropy, paper-faithful eval, evaluation scale, success rate.

1. **Behavior Entropy Implementation**: Resolved a critical metrics gap (identified in `D3IL_Metrics_SuccessRate_Entropy_Explained.md`) by implementing `compute_behavior_entropy` in the `d3il_visual_aligning_baseline_test` evaluation script. This faithfully ports the official D3IL entropy formula (success-conditioned, base-|B| normalized) to allow direct comparison against the D3IL paper's reported entropy (0.139).
2. **Paper-Faithful Evaluation Scale**: Added a `--paper` CLI flag and pipeline arguments to scale the evaluation from a fast smoke-test (3x1) to the mathematically required paper scale (60 contexts × 18 trajectories = 1080 rollouts). The SLURM pipeline now seamlessly chains training to paper-faithful evaluation.
3. **Metric Output**: The baseline now outputs `entropy` and a combined `score` (0.5 * (SR + H)) alongside `success_rate`, fully matching the published benchmarks.

***

## Data Analysis Tool v3: NPZ Capabilities and MPC Selection Documentation (June 20, 2026)

**Keywords**: npz_analysis, open-loop plans, Gen3v4 avoiding, MPC candidate selection, np.savez.

1. **Capability Gap Identified & Resolved**: Documented (`CAPABILITY_GAP_plan_not_saved.md`) that the Gen3v4 avoiding evaluation `.npz` files only saved executed trajectories, preventing offline scoring of open-loop plan explosions (blue fan plots). Fixed this by injecting `sampled_trajectories_all` into the `np.savez` call, unlocking future offline re-plotting and plan-quality measurements (e.g., plan straightness, roughness) without requiring simulator reruns.
2. **MPC Candidate Selection Explainer**: Authored `MPC_Candidate_Selection_Explained.md` to comprehensively document the receding-horizon control loop, including how a batch of plans is sampled, how the projector/temporal consistency selects exactly one candidate, and how visualization cadence (every H/2 steps) relates to execution. Also detailed a latent indexing inconsistency in `dpcc-c` temporal tracking.

***

## Gen3v4 "U6" Fix 2: iMF CFG Randomization and EMA Evaluation Switch (June 20, 2026)

**Keywords**: Gen3v4, iMeanFlow, CFG randomization, EMA evaluation, bug fix, divergence.

1. **CFG Randomization Implementation**: Addressed a critical bug where training used a fixed-constant Classifier-Free Guidance (CFG) operating point instead of the official iMeanFlow per-sample randomized CFG distribution. Added `_sample_cfg_scale` and `_sample_cfg_interval` to dynamically sample `omega` and `(t_min, t_max)` for every training sample, properly aligning the training behavior with the official paper's method.
2. **EMA-at-Eval Config Switch**: Confirmed that evaluating on raw weights (instead of Exponential Moving Average weights) is a legitimate inherited convention from the DPCC baseline, despite diverging from the official iMeanFlow repository. To allow fair A/B testing without breaking baseline compatibility, a config switch `eval_use_ema` was added to toggle between raw and EMA weights during evaluation.

***

## D3IL Visual-Aligning Baseline "U2.2 & U2.3": Eval Time Limits, Salvage Tool, & Root-Cause Audit (June 21, 2026)

**Keywords**: D3IL baseline, evaluation wall-clock, partial results salvage, zero-success audit, checkpoint selection.

1. **Evaluation Wall-Clock & Salvage Tool (U2.2)**: Discovered that the paper-faithful eval scale (1080 rollouts) exceeded the 4-hour SLURM time limit, causing job cancellations before summary JSONs were written. Extended evaluation time limits across all SLURM scripts to 24 hours. Additionally, developed `aggregate_partial_results.py` to retrospectively salvage incomplete run metrics by parsing per-rollout `diagnostics` JSONs, allowing successful recovery of interrupted evaluation sweeps.
2. **Zero-Success Root-Cause Audit (U2.3)**: Conducted a deep dive into the initial paper-scale sweeps which yielded ~0% success rates. An exhaustive cross-check against the upstream `d3il` repository verified that the evaluation loop, success termination logic, and EMA checkpoint machinery are identical and mathematically sound. The audit isolated the divergence to the **checkpoint selection criterion**: the FM-PCC baseline uses validation-loss selection, whereas `d3il` selects based on best simulation success. It is currently unproven whether this choice directly causes the 0% collapse or if the model inherently struggles with the image modality. An upstream protocol harness run is planned to decisively falsify this hypothesis.

***

## Gen11 Epoch 6: UAV State-Only Flow-Matching Build & Multi-Env Pipeline (June 21, 2026)

**Keywords**: Gen11, Epoch 6, UAV Flow-Matching, multi-env architecture, data curation, closed-loop eval.

1. **Dataset Curation Protocol**: Implemented `curate_dataset.py` to establish a strict separation between raw collection directories (which contain debug and rejected episodes) and the model training data. Accepted episodes are now explicitly manifested into `data/uav_fm/v1/`.
2. **UAV Flow-Matching Pipeline (`flow_matcher_v3_uav`)**: Forked the `flow_matcher_v3_ode_selectable` stack and retooled it for the quadrotor schema (12D transitions: 3D action `Δp_des` and 9D obs `[p_des|p|v]`). Removed legacy D3IL dependencies and integrated the `UAVSequenceDataset`.
3. **Multi-Environment Unified Training**: Upgraded the architecture to natively support multi-environment training across the four UAV scenes (`empty`, `corridor`, `s_curve`, `pillars`). Introduced a `--scene` selector that threads through the dataloader to dynamically pool scenes (e.g., `--scene all`) or isolate them, replacing the legacy flat `logs/` directory with a structured `logs/fm_uav/<run_id>/<scene-or-all>/` hierarchy.
4. **Closed-Loop MPC Evaluation (`eval_fm_uav.py`)**: Rewrote the evaluation script from scratch to mirror the UAV expert generation loop. The model now acts as a receding-horizon policy operating at 33 Hz, emitting `Δp_des` commands that are tracked by a low-level PID controller. Introduced real-time inference timing and JSON-based execution outputs.

***

## System-Wide Patch: MPC Trajectory Persistence & Temporal Consistency Fix (June 21, 2026)

**Keywords**: MPC_NPZ_PATCH, sampled_trajectories_all, np.savez, open-loop plans, temporal consistency, dpcc-c.

1. **Plan Persistence (`sampled_trajectories_all`)**: Fixed a major analytical gap where open-loop MPC trajectory forecasts were computed during evaluation but never saved to `.npz` files. Injected `sampled_trajectories_all` into the `np.savez` call across 10 evaluation scripts (including early-gen `obs_all`/`act_all` retrofits), enabling offline diagnosis of plan explosions and trajectory smoothness.
2. **Temporal Consistency Reference Fix**: Corrected a latent bug in 11 `policies.py` modules where `self.prev_observations` erroneously hardcoded its reference to the 0th trajectory candidate (`observations[0]`). Updated the assignment to use `observations[which_trajectory]`, ensuring that minimum-cost selection projectors (`dpcc-c`) accurately track the executed action for their temporal consistency window rather than defaulting to candidate 0.

***

## Gen9 Epoch 2 "U4" Fix 2: Visual Avoiding Eval YAML Fix & Seed Fix (June 22, 2026)

**Keywords**: Gen9, U4, visual avoiding, projection_eval.yaml, visual_avoiding_eval.yaml.

1. **Config Decoupling**: Discovered that the visual-avoiding evaluation was erroneously reading the globally shared `projection_eval.yaml` instead of its own dedicated config, causing only a single seed to run. Rewrote `config/visual_avoiding_eval.yaml` to faithfully mirror the avoiding schema (half-spaces, obstacles, bounds, dims) independently of the visual-aligning configuration.
2. **Script Repointing**: Reactivated the dedicated config by repointing all four active visual-avoiding scripts (`eval_visual_avoiding_dpcc.py`, `eval_fm_visual_avoiding.py`, etc.) to use `visual_avoiding_eval.yaml`. Added explicit seed list logging to end configuration ambiguity.

***

## Gen11 Epoch 6 "Aux-1": Batched "Collect → Data-Ready" SLURM Pipeline (June 22, 2026)

**Keywords**: Gen11, Epoch 6, Aux-1, data curation, mini_fm_sanity, SLURM orchestrator.

1. **Automated Pipeline Orchestration**: Wrapped the fragile manual curation steps (raw → flat dataset) into a fully unattended batched SLURM pipeline (`collect_to_ready_pipeline.sh`). This eliminates the risk of login-session timeouts disrupting the data preparation process.
2. **Data Safety Guarantees**: Enforced strict copy-only (`shutil.copy2`) and read-only verification operations within the curation script `prepare_uav_fm_data.sh`. The pipeline never moves or deletes raw expert data, ensuring a safe fallback.
3. **Mini-FM Sanity Gate**: Introduced an optional `mini_fm_gate.sh` step that trains a tiny Flow Matching model on the curated data to check held-out RMS error, catching action-convention or shape mismatches cheaply before committing full compute resources. Included fixes to `mini_fm_sanity.py` to correctly sample random parameter indices and compute the numerical gradient of `fm_loss`.

***

## Gen11 Epoch 6 "U2 & F1": Per-Scene FM Models & SLURM Submission Refactor (June 22, 2026)

**Keywords**: Gen11, Epoch 6, U2, per-scene FM models, scene-keyed run, SLURM orchestrator, sbatch-storm fix.

1. **Per-Scene FM Allocation (U2)**: Implemented an architecture where one Flow Matching model is trained per-scene (e.g., `empty`, `corridor`, `s_curve`, `pillars`). A state-only universal model was deemed underdetermined across different geometries. Created cross-scene aggregation scripts (`aggregate_scene_summaries.py`) to roll up evaluation metrics.
2. **Submission Logic Optimization (F1)**: Addressed an "sbatch-storm" risk where the `fm_uav_all_pipeline.sh` orchestrator previously spawned an unconstrained number of SLURM jobs (e.g., 25 jobs for 4 scenes × 3 seeds). Moved the seed loop *inside* the execution scripts (`train_fm_uav.sh`, `eval_fm_uav.sh`). Total job count now strictly scales as `2 * num_scenes + 1`, drastically reducing cluster impact while mathematically extending the per-job wall-clock limits appropriately.

***

## Gen11 Epoch 6 "U2" Fixes: Output Namespacing & Argument Parsing (June 23, 2026)

**Keywords**: Gen11, Epoch 6, UAV-FM, output path, namespace, argument parsing, logs/UAV_FM.

1. **Output Namespacing (Fix 3)**: Relocated UAV-FM evaluation outputs to be properly namespaced under `logs/UAV_FM/` (e.g. `logs/UAV_FM/uav-<scene>`) instead of cluttering the root `logs/` directory. Unified the base output path logic across configurations and aggregation scripts.
2. **Argument Parsing Fix**: Updated argument parsing in `eval_fm_uav.py` to prevent conflicts with re-parsed flags.

***

## Gen11 Epoch 6 "F3": Train NaN Resolution & SafeLimitsNormalizer (June 23, 2026)

**Keywords**: Gen11, Epoch 6, F3, UAV-FM, NaN, SafeLimitsNormalizer, constant feature column.

1. **Bug Diagnosis**: Investigated a training failure where all losses hit NaN at epoch 0 for the `pillars` scene (Job 21898). Identified that the `LimitsNormalizer` was performing `0/0` divisions on constant-feature columns (e.g., fixed altitude or zero-velocity components in the dataset).
2. **Resolution**: Transitioned `config/uav.py` to use `SafeLimitsNormalizer`, which intelligently maps constant dimensions to their midpoint (0.0) without corrupting the scaling of other dimensions.
3. **Verification**: Confirmed clean training on the `empty` scene up to step 1e5 without NaN occurrences.

***

## Gen11 Epoch 6 "U3": Legacy Evaluation Artifacts & EGL/GL Leak Hotfix (June 23, 2026)

**Keywords**: Gen11, Epoch 6, U3, artifacts, npz, GIF, EGL leak, renderer crash.

1. **Artifact Restoration**: Re-implemented pure-IO and matplotlib artifact writers in `eval_artifacts.py` without requiring MuJoCo or GPU access. Restored legacy `.npz` logging with DPCC placeholders for Epoch 7, 2D top-down and altitude overview PNG plots, per-rollout statistics, and opt-in GIF generation.
2. **Evaluation Pipeline Integration**: Upgraded `eval_fm_uav.py` to buffer heavy trajectory arrays and generate diagnostics under the new `logs/UAV_FM/` path.
3. **EGL/GL Leak Hotfix (Commit 6293c02)**: Resolved a critical context leak and renderer crash during GIF generation (`EGLError`). Consolidated to a single shared `mujoco.Renderer` per scene instead of per rollout, and implemented a guaranteed context release sequence (`_free_renderer()`) before interpreter teardown.
4. **W&B and GPU-Leak Guard Parity (Fix 3)**: Restored W&B logging and the EGL GPU-leak guard parity to the SLURM batch scripts `train_fm_uav.sh` and `eval_fm_uav.sh`.

***

## Gen11 Epoch 6 "U2" Close-out: Pillars Evaluation Analysis (June 23, 2026)

**Keywords**: Gen11, Epoch 6, pillars, evaluation, success rate, constant dims.

1. **Pillars Analysis**: Analyzed evaluation results for `pillars` seed 6. While the pipeline plumbing (training to eval) executed cleanly, the Flow-Matching policy failed with a 0% success rate.
2. **Failure Mode**: The drone never took off (resting height remained ~0.087m), generating extreme tracking errors (~92m) due to runaway commanded positions (`p_des`).
3. **Diagnosis**: Identified that action dimensions 0 and 2 were flagged as constant during evaluation, indicating the FM only learned 1 degree of freedom from the training data. This suggests a dataset-level deficiency rather than a pipeline fault.

***

## Gen11 Epoch 6 "U2" Trajectory Diagnosis: OOD Compounding & Missing Stabilizers (June 24, 2026)

**Keywords**: Gen11, Epoch 6, trajectory diagnosis, compounding error, OOD observation, tracking, goal-conditioning, absolute frame.

1. **Failure Mechanism Confirmed**: Analyzed the npz traces for the UAV pillars evaluation and confirmed that the model is **not** badly trained (it performs perfectly in early steps). The catastrophic failure (`p_des_z` runaway to -227m causing the drone to dive into the floor) is caused by **compounding error in the open-loop integration**.
2. **Missing Stabilizers**: Diagnosed that while the evaluation logic is identical to the working D3IL visual-aligning code, the UAV setup lacks the two features that keep working tasks in-distribution:
   - **Tight Tracking**: A position-controlled arm stays near its command, keeping the observation `[p_des, p]` on the trained diagonal. A 2nd-order quadrotor under gravity diverges from `p_des`, creating an out-of-distribution (OOD) observation the network never saw.
   - **Goal-Conditioning**: Visual-aligning models are anchored by images, allowing recovery. The current state-only UAV model has no goal reference to pull it back from a drift.
3. **Corrective Direction**: Next steps must restore these stabilizers. Proposed adding explicit goal-conditioning (the principled E7 direction) and re-grounding the prediction to the actual physical state (e.g., via a local/body-frame reformulation) so that the network input stays within the training distribution.

***

## Gen11 Epoch 6 "U3" Fix 1: Evaluation Output Migration & Diffusion Documentation (June 24, 2026)

**Keywords**: Gen11, Epoch 6, U3 Fix 1, plans directory, evaluation migration, API documentation.

1. **Evaluation Output Namespacing**: Relocated all UAV-FM evaluation outputs (e.g., `results.json`, `.npz`, `.png`) from inside the trained model's parameter directory (`<seed>/eval/`) into a dedicated sibling `plans/` hierarchy (e.g., `logs/UAV_FM/uav-<scene>/plans/...`). This establishes structural parity with legacy FM-PCC pipelines and cleanly separates model weights from inference artifacts.
2. **Cross-Scene Aggregation Robustness**: Fixed a brittle path globbing bug in `aggregate_scene_summaries.py` that failed to reach the evaluation data due to deeper internal folder structures. Aggregations now write the `SCENE_SUMMARY.json` into the `plans/` directory.
3. **Legacy API Translation Guide**: Authored `logs_in_develop/API_UPDATE/diffuser.py_Update/Table of Func Update Advice` mapping legacy DDPM function names (`p_mean_variance`, `q_sample`) to their true deterministic Flow Matching equivalents (`euler_integration_step`, `interpolate_flow_path`), clarifying the calculus-vs-probability paradigm shift without breaking inheritance.

***

## Data Analysis Tool v3: npz_analysis "JOB D" Plan-Candidate Analysis (June 24, 2026)

**Keywords**: npz_analysis, JOB D, open-loop plans, trajectory explosion, sampled_trajectories_all, replot-plans.

1. **Foresight Plan Processing**: Upgraded the `analyze_npz.py` tool to consume the `sampled_trajectories_all` arrays (the open-loop foresight plans) alongside the actual executed path (`obs_all`), unlocking analysis of the internal planner candidate fan for both DPCC avoiding and UAV tasks.
2. **Trajectory Explosion & Divergence Metrics**: Introduced new robust metrics: `plan_max_abs` and `traj_max_abs` (axis-agnostic explosion detectors that expose non-xy blowups like `p_des_z`), `plan_cand_spread` (measuring candidate diversity), and `plan_exec_div` (quantifying the deviation between the FM prediction and the PID's actual physical execution).
3. **Visual Plan Diagnostics**: Added a `--replot-plans` flag that overlays the foresight plan fan (blue) onto the executed trajectory (black) directly from the `.npz` files. Using this, confirmed that the UAV's predicted plans often remain bounded (~3m) while the executed command integrates into a runaway, isolating the fault to the action-channel scaling loop rather than pure model prediction failure.

***

## Gen11 Epoch 6 "U3" Fix 2: Evaluation Metrics Update & Honest Success (June 25, 2026)

**Keywords**: Gen11, Epoch 6, evaluation metrics, success_rate, goal_reached, safe_rate, fixed-route.

1. **Success Metric Redefinition**: Addressed a bug where the `success` metric originally only checked if the drone was airborne and contact-free, leading to misleading 100% success rates on empty scenes where the drone never reached the goal.
2. **Honest Goal-Reaching Rate**: Updated the `success` definition to strictly require reaching the goal (`goal_dist < goal_radius`) in addition to flying safely. Re-evaluation yields a 0% honest task success rate across all scenes, confirming that a goal-conditioning signal is fundamentally required (Epoch 7 focus).
3. **Scene-Aware Evaluation**: Adjusted evaluation so `success` requires reaching the goal only for fixed-route scenes (`corridor`, `s_curve`, `pillars`). For `empty` scenes with random unobservable goals, it rightly scores based only on stability (safe flight).

***

## Gen11 Epoch 6 "U3": Homotopy Ambiguity & Diffuser Baseline Coherence (June 25, 2026)

**Keywords**: Gen11, Epoch 6, diffuser baseline, homotopy ambiguity, mode oscillation, trajectory explosion.

1. **Pure-ML Baseline Verification**: Evaluated the pure-ML `diffuser` baseline (batch_size=1, no candidate selection, no projector) and found it successfully produces clean, well-tracked trajectories on single-mode scenes.
2. **Multi-Mode Trajectory Explosion**: Confirmed that on multi-mode scenes (corridor, pillars), the un-selected sample oscillates between different expert modes at each step. This open-loop drift creates an unstable, runaway command sequence that the 2nd-order quadrotor cannot absorb, leading to inevitable crashes.
3. **PCC/MPC Motivation**: Established that this multi-mode coherence gap is not a training bug but rather the expected limitation of a pure generative baseline. It perfectly motivates the introduction of candidate selection and constraint projection (the Epoch 7 PCC pipeline) to commit to a single coherent mode.

***

## Gen11 Epoch 7: Full PCC/MPC Implementation for UAV (June 25, 2026)

**Keywords**: Gen11, Epoch 7, PCC, MPC, dynamics constraint, uav_eval.yaml, p_des.

1. **PCC Bone Restoration**: Migrated and restored the full PCC/MPC architecture from the Gen7 visual-aligning pipeline into the UAV task in a single comprehensive pass, supporting `diffuser`, `dpcc-r`, `dpcc-c`, and `dpcc-t` variants.
2. **Dynamics Constraint Definition**: Configured a genuine Euler `dynamics` constraint with `dt=1.0` that strictly binds `p_des` (desired position) rather than the physical position `p`, correctly enforcing the `Δp_des` action semantic and preventing set-point drifting.
3. **Evaluation Configuration Migration**: Relocated the PCC evaluation configuration out of code defaults into a standalone `config/uav_eval.yaml` that includes empty placeholders for future geometry bounds, halfspace, and obstacle constraints. Added constraint-aware evaluation metrics.

***

## Gen11 Epoch 7 "Hotfix 1": Goal Dimension Shape Crash (June 25, 2026)

**Keywords**: Gen11, Epoch 7, Hotfix 1, projector shape mismatch, goal_dim.

1. **Projector Shape Crash Resolution**: Fixed a `mat1 and mat2 shapes cannot be multiplied (4x88 and 96x96)` error that crashed `dpcc-*` variants during the SLSQP projection on the corridor scene.
2. **Dynamic Dimension Subtraction**: Diagnosed that the `get_goal_dim()` heuristic incorrectly flagged an incidentally-constant observation column as a goal dimension, reducing the active trajectory width passed into the projector. Resolved by dynamically subtracting `model_fm.goal_dim` from `trajectory_dim` in `setup_dpcc_projector`, ensuring correct dimensionality without requiring a model retrain.

***

## Gen11 Epoch 7 "U1 & U2": MPC Foresight Visualization & UAV-Specific Panels (June 25, 2026)

**Keywords**: Gen11, Epoch 7, MPC foresight, candidate fan, XY top-down, XZ altitude.

1. **Candidate-Fan Visualization**: Replaced the placeholder foresight rendering with actual real-time candidate-fan visualization in `eval_artifacts.py` using the Gen7 dual-path convention (green fan for MPC candidates, black for commanded path, red for physical execution).
2. **UAV-Specific Panel Layout**: Transitioned the plot layout from a generic 3D view to a tailored "XY top-down + XZ altitude" setup. The dedicated XZ panel isolates the Z-axis profile, explicitly highlighting `p_des_z` trajectory explosions and altitude gates for improved diagnostic clarity.
3. **Contextual Additions**: Augmented the visualization by plotting scene obstacles and altitude silhouettes directly into the eval artifact outputs.

***

## Gen11 Epoch 7 "U1 & U2": Why PCC Works on Pillars (June 25, 2026)

**Keywords**: Gen11, Epoch 7, pillars, PCC efficacy, mode-blend, open-loop integration.

1. **PCC vs Diffuser Discrepancy**: Documented how the `diffuser` baseline fails immediately on the multi-modal `pillars` scene (drone sinks to the floor as `p_des_z` integrates to -228m), while `dpcc-*` variants fly cleanly through the maze.
2. **Dynamics Constraint Mechanism**: Explained that the 24 linear equality constraints solved by the SLSQP projector force the initially unconstrained `act[0]` output to mathematically align with the model's coherent `p_des` sequence, effectively neutralizing mode-oscillation before execution.
3. **Goal-Seeking Distiction**: Acknowledged that while PCC solves the coherence crisis, it still stops ~1m short of the endpoint because the policy is not explicitly goal-conditioned (a target for future epochs).

***

## Gen11 Epoch 7 "U3": Full DPCC Variant Suite Restoration (June 25, 2026)

**Keywords**: Gen11, Epoch 7, U3, DPCC variant suite, gradient, post_processing, model_free, tightened.

1. **Variant Restoration**: Restored the full suite of 13 DPCC paper Table 1 projection variants in `config/uav_eval.yaml` (expanding from the 4 initial `dpcc-*` variants). Scaffolded new variants including `gradient`, `post_processing`, `model_free`, and their `-tightened` counterparts.
2. **Projector Logic Updates**: Updated `setup_dpcc_projector` in `eval_fm_uav.py` to correctly parse and configure the new variant flags, ensuring exact functional parity with the original `fm_visual_aligning` reference. Dynamics constraints are bypassed for `model_free` and enforced at all steps for `post_processing`.
3. **Future Spatial Scaling**: Structured `model_free` and `tightened` variants to function cleanly as no-ops while spatial constraints (obstacles/halfspaces/bounds) remain pending.

***

## Gen11 Epoch 7: Real-Time Behaviour Text Logging (June 26, 2026)

**Keywords**: Gen11, Epoch 7, real-time logging, BehaviorLogger, inference timing, total_ms, budget_ms.

1. **Digital-Twin Timing Audit**: Implemented a comprehensive `BehaviorLogger` in `FM_v3_uav_test/behavior_logger.py` to quantify whether the FM-PCC pipeline can close the 33 Hz (30.3 ms budget) control loop. The logger emits per-step structured text logs capturing purely separated FM inference time (`fm_ms`), projector overhead (`proj_ms`), and total latency, directly isolating hardware bottlenecks.
2. **Model Internal Split**: Refactored the core `p_sample_loop` in `diffusion.py` to time the internal `projector.project()` and `projector.compute_gradient()` calls. This accurately unbundles projection time from FM inference time without breaking the integrated ODE generation loop.
3. **Data Source Consolidation**: Hardened the configuration flow by wiring `DATASET_HZ=33` as an authoritative import from `dataset_writer.py`, while exposing `behavior_log` toggle and `control_hz` as user-configurable keys in `uav_eval.yaml`.
4. **Gradient Shape Crash Fix**: Resolved a latent shape-mismatch bug in the `gradient` variant by ensuring the gradient update is only applied to the non-goal 11-D slice of the trajectory tensor (`x[:, :, :-self.goal_dim] = x[:, :, :-self.goal_dim] + grad`), preventing a crash when `goal_dim > 0`.

***

## Gen11 Epoch 7 "ANALYSIS": UAV Trajectory Explosion Diagnostics (June 26, 2026)

**Keywords**: Gen11, Epoch 7, crash diagnosis, self-referential observation, projection cost, p_des divergence, hardware timing.

1. **Real-Time Bottleneck Identification**: Evaluated the new real-time logging on the `corridor_C` task. Concluded that the cluster hardware (i6-gpu-1) is insufficient for real-time control, as FM inference alone requires ~85 ms against the 30.3 ms budget, resulting in 100% of steps exceeding the budget.
2. **Proj_Cost as Early Warning**: Discovered that `proj_cost` spikes (from ~1 to >10,000) serve as a highly accurate leading indicator of trajectory crash, firing ~0.5 seconds before the drone actually makes contact with an obstacle. This validates the DPCC projector's constraint-fighting metric as a real-time safety signal.
3. **Self-Referential Divergence Proof**: Diagnosed why the FM policy continues to plan wildly (accumulating 2.5m of track error) after the physical drone crashes and freezes. Because the model conditions on its own integrated `p_des` output rather than actual drone position `p`—and the dynamics constraint only checks internal plan consistency—the system enters an out-of-distribution positive feedback loop when tracking errors spike.
4. **Actionable Fixes Established**: Outlined mandatory structural fixes including spatial halfspace/bounds constraints, a hard evaluation episode-termination switch (`track_err > 0.5 m`), and future tracking constraints inside the projector to re-anchor `p_des` to real `p`.

***

## Documentation Maintenance: Deep Analysis and Architecture Guides (June 26, 2026)

**Keywords**: DPCC, trajectory origin, planning oscillation, architecture guide, visual-aligning expansion.

1. **Model Architecture Formalization**: Authored the `MODEL_ARCHITECTURE_GUIDE.md` to map the complex multi-stage DPCC inference pipeline (from raw images to ResNet18 embeddings, down to U-Net generation). This provides a single source of truth for the dimensionality, routing, and data structures used across the `diffuser` and Flow Matching visual pipelines.
2. **Analysis of Receding-Horizon Oscillation**: Published `WHY_FM_KEEPS_PLANNING.md` containing a rigorous audit of the internal candidate generation process. Documented why unconstrained FM and Diffuser baselines experience "mode oscillation" across consecutive plans, providing theoretical justification for the temporal consistency enforced by the PCC projector.
3. **Tensor Origin Tracing**: Authored `DPCC_TENSOR_ORIGIN.md` to forensically trace exactly where every tensor (e.g., `p`, `p_des`, images) originates inside the `d3il` simulation framework and how it reaches the generative backbone, ensuring correct coordinate scaling and referencing.
4. **Visual-Aligning Environment Expansion**: Outlined the theoretical bounds and required codebase additions in `ALIGNING_EXPANSION.md` to safely transition the aligning framework toward generalizing across novel object locations.

***

## Gen7 & Gen6v4: True-FiLM (FiLMv2) Temporal U-Net Upgrade (June 27, 2026)

**Keywords**: Gen7, Gen6v4, True-FiLM, FiLMv2, visual conditioning, U-Net, per-channel scale and shift.

1. **True-FiLM Architecture Implemented**: Replaced the legacy "Fake FiLM" (where visual embeddings were merely concatenated to time embeddings) with a genuine Feature-wise Linear Modulation (FiLM) bottleneck (`UNet1DTemporalFiLMModel`). Visual conditioning is now injected per-block via learned per-channel scale (`γ`) and shift (`β`) parameters, decoupling it from the temporal positional embedding.
2. **Zero-Initialized Opt-In Deployment**: Designed the upgrade as a strict opt-in via a new configuration key (`film_mode: 'v2'`), leaving `v1` as the default. To preserve learning stability, the new `film_proj` dense layers are zero-initialized, ensuring the network acts as an identity block for the visual signal at step 0. 
3. **Cross-Pipeline Integration & Automation**: Deployed the `FiLMv2` backbone symmetrically across both the state-of-the-art Flow Matching pipeline (`fm_visual_aligning`) and the Diffuser baseline (`diffuser_visual_aligning`). Automated the routing logic within `visual_unet.py` to seamlessly instantiate either the `v1` or `v2` backbone based on the configuration key, guaranteeing 100% backward compatibility for all existing checkpoints.
4. **Comprehensive Upgrade Documentation**: Authored a suite of documentation (`CHANGELOG_FiLM_V2.md`, `PLAN_FiLM_V2.md`, `Ideas.md`, and `MEMO_FiLM_code_to_math.md`) detailing the exact mathematical deltas, usage instructions for isolating `v2` checkpoints, and fallback plans.

***

## Gen11 Epoch 7 "U4" & "F5": Anchor-P Integration & UAV Grounding Modes (June 28, 2026)

**Keywords**: Gen11, Epoch 7, U4, F5, anchor_to_p, real-position grounding, trajectory explosion, DPCC.

1. **Trajectory Drift Diagnosis**: Identified that the dynamics constraint (`p_des[t+1] = p_des[t] + action[t]`) caused unbounded trajectory divergence because it blindly projected action feasibility from commanded setpoints (`p_des`) instead of the drone's actual lagging position (`p`).
2. **Anchor-P Integration**: Implemented the `anchor_to_p` evaluation mode to directly ground the PID setpoint and DPCC dynamics constraint into the real drone position (`p_des = p + action`). This strictly corrects the integration loop and physically re-anchors the planning observation without requiring any model retraining.
3. **Deprecated Blending**: Initially experimented with `cond_mode='real_p'` (retraining) and `reanchor_alpha` blending, but removed them in favor of the mathematically robust `anchor_to_p` logic which safely preserves existing checkpoints.

***

## Gen11 Epoch 7 Config Refactor: Projection & Eval Decoupling (June 28, 2026)

**Keywords**: Gen11, Epoch 7, config split, uav_projection.yaml, uav.py, evaluation configuration.

1. **Architecture Migration**: Restructured the UAV configuration logic to match the robust `avoiding-d3il.py` pattern. Created a dedicated `config/uav_projection.yaml` exclusively for projection parameters (variants, geometry, thresholds), deprecating the over-stuffed `config/uav_eval.yaml`.
2. **Unified Planning Block**: Consolidated evaluation-specific control parameters (`batch_size`, `control_hz`, `anchor_to_p`) into a `plan_flow_matching_v3_uav` block inside `config/uav.py`. `eval_fm_uav.py` now seamlessly merges this block with `uav_projection.yaml` at runtime.
3. **Run-Quantity Centralization**: Moved `seed` and `n_trials` settings from hardcoded magic numbers in `parse_args()` to the new `uav_projection.yaml`. This fixes a silent bug where evaluation defaulted to `seed=5` instead of the required UAV `seed=6`.

***

## Gen11 Epoch 7: CLI Overrides & SLURM Orchestration Fix (June 28, 2026)

**Keywords**: Gen11, Epoch 7, SLURM scripts, bash parameter expansion, n_trials, CLI override.

1. **Bash Override Hotfix**: Identified that four SLURM batch scripts (`eval_fm_uav.sh`, `fm_uav_pipeline.sh`, etc.) were hardcoding `NTRIALS="${3:-20}"`, forcefully injecting `--n-trials 20` to Python and preventing the `uav_projection.yaml` configuration from ever being read.
2. **Conditional Flag Passing**: Updated all orchestrator scripts to use conditional expansion (`${NTRIALS:+--n-trials "$NTRIALS"}`). If the user omits the trial argument, the bash scripts simply pass no flag, allowing Python to cleanly fall back to the YAML defaults.
3. **Pipeline Seed Correction**: Corrected `fm_uav_pipeline.sh` to default to `SEED=6` rather than the legacy D3IL `5`, properly aligning cluster workflows with the UAV training standards.

***

## Gen11 Epoch 7 Misc: Buffer Optimization & Projector Performance (June 28, 2026)

**Keywords**: Gen11, Epoch 7, max_path_length, replay buffer, projector performance, tightened variants.

1. **Per-Scene Buffer Allocation**: Implemented a `MAX_PATH_LENGTH_PER_SCENE` dictionary in `config/uav.py` to dynamically size the training replay buffer based on actual scene duration limits (e.g., `corridor=360`, `empty=450` vs the blanket `750`). This prevents massive memory overallocation and waste across shorter scenes.
2. **Redundant Projection Bypass**: Optimized evaluation performance by explicitly skipping "tightened" projection variants (`dpcc-*-tightened`) when spatial constraints are inactive, eliminating redundant SLSQP solving cycles.

***

## Cross-Gen Enhancement: Real-Time Behavior Recording Rollout (June 28, 2026)

**Keywords**: real-time recording, RTRecorder, inference timing, total_ms, bundled timing, behaviour logging, 10 evals.

1. **Global Recording Infrastructure**: Extracted the Gen11 UAV `BehaviorLogger` into a standalone, portable `realtime_recording.behavior_logger.RTRecorder` module. This module provides an observation-layout-agnostic framework to record per-step timings (e.g., `total_ms`, `fm_ms`, `proj_ms`) and behavioral contexts (e.g., `track_err`).
2. **System-Wide Rollout (10 Evals)**: Deployed the new recorder across all 10 active evaluation pipelines, including the DPCC baseline, state-only FM/DPCC, visual-aligning (via `VisualAgentWrapper`), and visual-avoiding pipelines. Each rollout now automatically generates a `realtime_*.log` file.
3. **Summary Audit Block**: Every log concludes with a summary block that answers deployment-critical questions, explicitly calculating whether the system can close the control loop within the hardware budget (e.g., 30 Hz/33 ms) and identifying the computation ratio between FM inference and DPCC projection.
4. **Bundled Timing Design**: Acknowledged that non-UAV legacy policies do not expose isolated projection times. The logger honestly bundles FM and projection times under `total_ms`, allowing architectural comparisons (e.g., Diffuser vs DPCC) to extract the projection overhead at the aggregate level.

***

## Gen3v4 "U7": iMeanFlow Logit-Normal Time Schedule (June 28, 2026)

**Keywords**: Gen3v4, U7, iMeanFlow, logit-normal, time schedule, p_mean, p_std.

1. **Canonical Time Schedule Integration**: Integrated the official iMeanFlow logit-normal time sampling schedule (`t = sigmoid(randn * p_std + p_mean)`) as the new default across both the state-based (`flow_matcher_v3_imeanflow`) and visual-aligning (`imf_visual_aligning`) pipelines. This mathematically aligns training with the reference implementation (which targets a median `t` near 0.40).
2. **Backward Compatibility**: Preserved the legacy `1 - Beta(α, β)` schedule as an explicit `t_schedule='beta'` configuration option (which seamlessly supports a uniform `Beta(1,1)` schedule as well). This maintains backward compatibility for older checkpoints and enables direct A/B schedule ablations.
3. **Configuration & Path Tracking**: Wired the `t_schedule` parameter into `args_to_watch` (appending `_tslogit_normal` or `_tsbeta` to checkpoint directories), ensuring transparent model loading without path collisions between differently scheduled models.

***

## Gen11 Epoch 8: UAV MJPC Thrust Control & Multiple Controllers Implementation (June 29, 2026)

**Keywords**: Gen11, Epoch 8, MJPC, thrust control, pid_stopgo, pid_const_v, 9D position planner, cond_mode.

1. **MJPC Thrust Control**: Implemented an optional FM→MJPC path using a strict-DPCC 9D position planner (`[action|p_des|p]`, velocity dropped) alongside an MJPC optimal-control thrust tracker. The new `MJPCTracker` class mirrors the existing `CascadedPID` interface for controller-agnostic inner loops, reusing the existing `mujoco_mpc` cartpole architecture. 
2. **PID Stop-and-Go (`pid_stopgo`)**: Added a controller option that reuses the cascaded PID but sets `v_des = 0`, forcing the UAV to actively brake to zero velocity at each timestep. 
3. **PID Constant Velocity (`pid_const_v`)**: Added a controller option that auto-derives a consistent flight speed (`v_des_magnitude = mean(|action|) × DATASET_HZ`) from the training dataset, enabling timing-free continuous flight without hardcoded velocity scalars.
4. **Configuration Robustness & Path Decoding (Fix 4)**: Resolved bugs in experiment path generation where prefixes were missing and controller suffixes polluted trained model directories. Cleaned up config resolution to auto-derive `cond_mode` directly from checkpoint metadata rather than the volatile plan block, preventing `ValueError` shape mismatches during evaluation.

***

## Gen11 Epoch 8 "Fix 5": MuJoCo MPC Package Bundling & ODE Step Correction (June 29, 2026)

**Keywords**: Gen11, Epoch 8, Fix 5, mujoco_mpc, gRPC, flow_steps_v3, ODE inference.

1. **MuJoCo MPC Bundling**: Addressed a `ModuleNotFoundError` during cluster evaluation by bundling the pure-Python `mujoco_mpc` package (including gRPC stubs) directly into `third_party/mujoco_mpc/`. Generated proto files locally to eliminate Docker-environment dependencies, providing clear instructions for compiling the C++ `agent_server` binary on cluster nodes.
2. **ODE Inference Step Correction**: Discovered a silent omission where `flow_steps_v3` was never set in the UAV config, forcing `FlowMatchingODE` to fall back to the default 1000 Euler ODE steps per inference call instead of the intended 20. Pushed the fix to the evaluation configuration to reduce inference latency by ~50× without invalidating existing mathematical results.

***

## Gen9 Epoch 2 "U5": FiLM v2 Port for Single Camera Avoiding (June 29, 2026)

**Keywords**: Gen9, Epoch 2, U5, FiLM v2, visual avoiding, UNet1DTemporalFiLMModel.

1. **True FiLM Architecture Port**: Ported the true-FiLM per-block γ/β architecture (`UNet1DTemporalFiLMModel`) from Gen7 visual-aligning directly into the Gen9 visual-avoiding pipeline.
2. **Unified Configuration Toggle**: Replaced duplicate branching logic with a clean `film_mode` parameter in `config/avoiding-d3il-visual.py`. Setting `film_mode: 'v2'` dynamically swaps the backbone to the new architecture and routes checkpoints to dedicated `_filmv2/` directories without disrupting baseline operations.

***

## Data Analysis Tool v3: Cross-Experiment Combined Analysis (June 29, 2026)

**Keywords**: DA Code v3, combined analysis, cross-experiment, comma-separated paths.

1. **Cross-Domain Path Support**: Upgraded `main_da_batch.py` to accept comma-separated directories in the `--parent-path` argument. This seamlessly extends the batch pipeline to simultaneously discover, merge, and compare candidates across both state-only avoiding (`logs/avoiding-d3il`) and visual avoiding (`logs/avoiding-d3il-visual`) experiments in a single unified run.
2. **Pipeline Wrapper Integration**: Added a dedicated `run_da_batch_avoiding_combined.sh` SLURM script to automate this cross-domain evaluation workflow.

***

## Gen11 Epoch 8 "Eval": Performance Insights on Pillars (June 29, 2026)

**Keywords**: Gen11, Epoch 8, evaluation, pid_stopgo, anchor_p, MJPC, pillars.

1. **MJPC Status**: The MJPC optimal-control tracker implementation is currently a work in progress and its evaluation runs are still ongoing.
2. **PID Stop-and-Go Success**: The `pid_stopgo` controller demonstrated exceptional performance, yielding really great, paper-ready results specifically in the complex multi-modal `pillars` scene.
3. **Anchor-P on Legacy 12D Performance**: The `anchor_to_p` feature successfully provided a measurable performance improvement on the 12D state using the legacy velocity-based PID controller. However, despite the improvements, its overall performance still remains noticeably inferior when compared directly against the robust results achieved by `pid_stopgo`.

***

## Gen11 Epoch 7 "Hotfix": Pipeline Projection & Record Arguments (June 29, 2026)

**Keywords**: Gen11, Epoch 7, fm_uav_pipeline, projection arguments, sbatch script.

1. **Pipeline Argument Support**: Updated `Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh` to pass through projection selection variants (e.g., `dpcc-r`, `diffuser`) and the real-time recording toggle directly to the underlying `eval_fm_uav.sh` scripts. This fixes an issue where the unified SLURM orchestrator lacked parameter-forwarding capabilities for recent projector features.

***

## D3IL Visual-Aligning Baseline "U3": Simulation Evaluation Fixes (June 30, 2026)

**Keywords**: D3IL baseline, simulation evaluation, ValueError, unpacking crash, U3 checkpoint.

1. **State Tuple Unpacking Bug (Fix 1)**: Addressed a crash in `d3il/agents/ddpm_encdec_vision_agent.py` caused by a strict 3-unpack receiving a 4-tuple state `(bp_image, inhand_image, des_robot_pos, robot_pos)`. This 4th element was added during Gen6V4 DPCC. Refactored the agent to safely index the required 3 dimensions without breaking backward compatibility.
2. **Simulation Result Unpacking Bug (Fix 1.2)**: Fixed a subsequent crash in `train_d3il_visual_aligning.py` where `aligning_sim.test_agent()` returned 4 values instead of the expected 2. Updated the unpacking logic to discard the unused metrics, allowing the epoch loop to run to completion and save the correct U3 best-success checkpoints.
3. **Changelog Documentation**: Published a targeted `CHANGELOG.md` inside `D3IL_Visual_Aligning_RUN/U3_train&other_d3il_direct_call/fix_1/` mapping the exact bug causes and testing commands.

***

## Gen11 Epoch 8 "Design Audit": MuJoCo Simulation vs IRL Latency (June 30, 2026)

**Keywords**: Gen11, Epoch 8, sim vs irl latency, v_real, MuJoCo frozen time, 12D obs, zero-latency problem.

1. **12D Observation Clarification**: Authored `DESIGN_sim_vs_irl_latency.md` detailing that the `v` in the 12D observation tensor is exclusively the real physics engine velocity (`v_real`), and `v_des` is strictly a post-inference derivative used by the PID.
2. **Simulation Time-Freeze Verification**: Audited the MuJoCo execution loop to confirm that simulation time halts entirely during Flow Matching inference. `mj_step` is correctly isolated, meaning the "zero-latency problem" (sensor staleness) is purely a real-robot artifact, making simulation results slightly optimistic.
3. **v_des Override Critique**: Conducted a theoretical analysis comparing expert dataset `v_des` replays versus `pid_const_v`. Concluded that `pid_const_v` effectively replicates the global dataset mean, whereas per-step expert overrides are flawed because the FM generates its own divergent waypoints, causing spatial mismatch.

***

## Data Analysis & Infrastructure Hotfixes (June 30, 2026)

**Keywords**: DA visualizer, visual avoiding, pandas DtypeWarning, mjpc compile, mpc_batch_size.

1. **DA Visualizer Support**: Upgraded `Data_Analysis/Visualizer/index.html` to support the rendering of visual avoiding results alongside state results, and resolved a pandas `DtypeWarning` by explicitly setting `low_memory=False` during CSV parsing.
2. **MuJoCo MPC Build Script**: Updated `Slurm_Codes/sbatch/uav_fm/build_mjpc_agent_server.sh` to accommodate recent dependency changes for compiling the C++ `agent_server` binary.
3. **Gen9 Epoch 2 "U5" Hotfix**: Refactored `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` to correctly map the evaluation batch size to the new `mpc_batch_size` parameter, ensuring consistent tensor sizing in visual avoiding rollouts.

***

## Gen9 Epoch 2 "U5" Hotfix: FiLM Configuration Unification (June 29, 2026)

**Keywords**: Gen9, Epoch 2, U5, FiLM v2, avoiding-d3il-visual.py, config refactor.

1. **Config Deduplication**: Refactored `config/avoiding-d3il-visual.py` to eliminate duplicate model blocks for FiLM v1 and v2. Consolidated them into a single, unified configuration definition controlled dynamically via the `film_mode` switch, streamlining the configuration surface and preventing divergence.

***

## Data Analysis Tool v3: Candidate Identification Refactor (June 30, 2026)

**Keywords**: DA Code v3, candidate identification, numeric indices, alphabetical labels.

1. **Numeric Candidate Indices**: Transitioned candidate identification in the Data Analysis (DA) code from alphabetical labels (e.g., A, B, C) to numeric indices (e.g., 1, 2, 3) across all analysis modules. This standardizes the naming convention and improves clarity in candidate tracking.

***

## Gen11 Epoch 8 "Fix 6": MJPC `agent_server` Compilation and Debugging (June 30, 2026)

**Keywords**: Gen11, Epoch 8, MJPC, agent_server, segmentation fault, subprocess streaming, build script.

1. **Build Process Optimization**: Iteratively updated the build script for the MJPC `agent_server` to configure correct environment paths and include missing dependencies, successfully achieving compilation on the cluster.
2. **Debugging Diagnostics Enhancement**: Added subprocess output streaming to `MJPCTracker` in `mjpc_tracker.py` and a startup diagnostic probe in the `eval_fm_uav.sh` script to improve visibility into the agent's initialization process.
3. **Deployment Segfault & Pipeline Revert**: Documented a critical deployment failure in `STATUS_agent_server_segfault.md`. While the compiled binary runs successfully in the golden build environment, it crashes with a segmentation fault (`si_addr=0x4`) during early initialization when deployed in the FMPCC evaluation environment. Consequently, the MJPC controller installation has been aborted and paused. The evaluation pipeline has been cleanly reverted to use the stable `pid_stopgo` controller.

***

## Gen11 Epoch 8 "U6": MJPC Tracker JAX/MJX Rebuild (July 1, 2026)

**Keywords**: Gen11, Epoch 8, U6, MJPC, MJX, JAX, predictive sampling, python solver.

1. **Architecture Pivot**: Abandoned the C++ gRPC `agent_server` architecture due to untraceable segmentation faults on the cluster. Pivoted to a pure Python implementation using MuJoCo's JAX backend (MJX).
2. **DeepMind Solver Integration**: Integrated DeepMind's `predictive_sampling.py` solver directly into `third_party/mujoco_mpc/mujoco_mpc/mjx/`. This solver performs vectorized rollouts via `jax.vmap` on the GPU, removing all subprocess and binary dependencies.
3. **MJPCTracker Rewrite**: Completely rewrote the `MJPCTracker` class in `FM_v3_uav_test/mjpc_tracker.py` to wrap the new MJX planner. Maintained the identical external `.compute()` API, ensuring zero disruption to the `eval_fm_uav.py` rollout loop. Implemented a custom JAX-compatible UAV position-tracking cost function.
4. **Configuration Transition**: Updated `config/uav.py` and `eval_fm_uav.py` to replace legacy gRPC parameters (`mjpc_planner_steps`, etc.) with MJX-native knobs (`mjx_n_samples`, `mjx_horizon`). Deleted the obsolete `build_mjpc_agent_server.sh` script.

***

## Cross-Gen "DC_FIX": Dynamics Constraint Rectification (July 1, 2026)

**Keywords**: DC_FIX, dynamics constraints, anchor_to_p, cond_on_p, hallucination.

1. **Constraint Dimensionality Bug Resolved**: Discovered and fixed a critical bug where all ported evaluation scripts were missing half of the required dynamics constraint rows. Specifically, the projectors were only constraining either real position (`p`) or desired position (`p_des`), leaving the other free to hallucinate.
2. **Global Implementation**: Updated `eval_fm_visual_aligning.py`, `eval_imf_visual_aligning.py`, `eval_visual_aligning_dpcc.py`, and `eval_fm_uav.py` to correctly construct all 6 rows for 3D tasks, anchoring both channels strictly.
3. **Deprecation of `anchor_to_p`**: Formally deprecated the `anchor_to_p` (cond_on_p) mechanism for constraint selection, identifying it as a symptom-level workaround for the missing constraint rows rather than a mathematically robust solution. 
4. **Documentation**: Published comprehensive analysis and changelogs under the `DC_FIX` flag documenting the shift from 3-row to 6-row dynamics constraints.

***

## Evaluation Logging & Configuration Hotfixes (July 1, 2026)

**Keywords**: config snapshot, wandb, logging, npz analysis, dynamics gap.

1. **Config Snapshot Scoping**: Moved the configuration snapshotting logic from the global `setup.py` directly into the evaluation script (`eval_fm_uav.py`). This prevents path collisions and ensures that `uav_projection.yaml` snapshots are strictly scoped within their respective seed directories.
2. **W&B Dependency Robustness**: Replaced broad exception handling with specific `ImportError` checks in logger stubs across the codebase, resolving a silent failure caused by a protobuf mismatch with the `wandb` package.
3. **NPZ Analysis Upgrades**: Expanded the `analyze_npz.py` utility to calculate trajectory and plan dynamics gaps. Added action data extraction to `dump_xy_rows` for more comprehensive trajectory forensic analysis.

***

## Gen11 Epoch 8 "U6" Post-Rebuild Hotfixes (July 2, 2026)

**Keywords**: Gen11, Epoch 8, U6, MJX, JAX, policy improvement, hover thrust, corridor bug.

1. **JAX 0.5+ Compatibility Shim**: Added a compatibility shim for `jax.extend.backend.backends` to support JAX 0.5+ in `mjx_tracker.py`, resolving an `AttributeError` crash on cluster nodes when attempting to check for CUDA devices.
2. **CYLINDER-BOX Collision Bypass**: Disabled geometric collisions (`geom_contype` and `geom_conaffinity`) during `mjx.put_model()` to bypass an unsupported CYLINDER-BOX collision error in MJX. This restores the rollout simulation for the `pillars` scene while upstream DPCC handles obstacle avoidance.
3. **MJX Policy Improvement & Velocity Penalty**: Enhanced `MJPCTracker` with a velocity cost penalty (`mjx_vel_weight`) to properly enforce the stop-and-go task objective (preventing high-speed overshoot) and enabled configurable multi-iteration policy improvement (`mjx_n_improve`) to fully utilize the GPU compute budget. Explicitly restored `jnp.zeros` policy initialization to avoid domain-engineered hover warm-starting.
4. **Corridor Bug & False-Positive Goal Dim**: Fixed a false-positive `goal_dim` calculation in the UAV evaluation script to prevent out-of-bounds projector index errors specifically on the `corridor` scene.
5. **Parser Ghost Directory Fix**: Updated `Parser` savepath logic to handle `plan`/`eval` experiments cleanly without creating empty ghost directories in the workspace root.

***

## Data Analysis Tool v3: NPZ Analysis Updates (July 2, 2026)

**Keywords**: npz_analysis, trajectory dynamics gap, out-of-bounds safety, coordinate mapping.

1. **Dynamics Gap Metric Context**: Disabled the invalid `plan_dyn_gap_max` calculation specifically for the visual avoiding task (where physics constraints do not neatly map to the 9D representation) and updated `README.md` documentation to clarify metric validity across different environments.
2. **UAV Coordinate Mapping & Safety**: Updated UAV coordinate columns in the NPZ analysis tool for accurate extraction and introduced out-of-bounds safety checks when processing plan snapshots.
3. **Unknown Schema Gap Fix**: Fixed a bug where an unknown coordinate schema would cause incorrect plan dynamics gap calculations by safely setting `plan_act_cols` to `None`, preventing out-of-bounds array access.

***

## HardFlow & iMeanFlow (HF·iMF) Architectural Blend Theory (July 2, 2026)

**Keywords**: HardFlow, iMeanFlow, HF-iMF, constrained sampling, average-velocity flows, terminal prediction.

1. **Architectural Blend Conceptualization**: Authored comprehensive theoretical and engineering documentation (`BLEND_HardFlow_iMeanFlow.md`, `THEORY_DeepMix_HF_iMF.md`) to evaluate blending the HardFlow (HF) hard-constrained sampling framework with the iMeanFlow (iMF) average-velocity flows architecture. 
2. **Complementary Strengths**: Established that the two methods fix each other's central weaknesses. HardFlow relies on a biased first-order Euler extrapolation (`x̂1 = z + (1−τ)·v`), which iMF's average-velocity field exactly replaces without additional NFE cost. Meanwhile, iMF lacks hard constraint capabilities, which HardFlow provides via its algebraic Prox-NLP and pull-back mechanism. 
3. **Proposed K-step iMF-HardFlow Algorithm**: Formalized a K-step (K=2-4) solver that reduces the required NLP solves from 20 (in stock HardFlow) to just a few, cutting sampling cost drastically while substituting HardFlow's structural Euler bias with a τ-uniform network error. 
4. **Implementation Roadmap**: Outlined a clear integration plan requiring the retraining of an iMF-recipe model on the avoiding trajectory domain (using HardFlow's UNet1D and terminal-anchored interval sampling) followed by a targeted call-site substitution in the HardFlow constraint solver.

***

## Infrastructure Maintenance: Legacy MJPC Cleanup (July 2, 2026)

**Keywords**: MJPC, cleanup, gRPC, cpp, archive.

1. **Archive Old MJPC C++ Code**: Moved the obsolete C++ and gRPC binary assets for the old `mujoco_mpc` agent server into an `Archived_Codes` folder. This formally completes the pivot to the pure-Python MJX predictive sampling architecture and removes unused binary dependencies from the active workspace.

***

## Gen11 Epoch 8: UAV Sim2Real Timing & IK Analysis (July 2, 2026)

**Keywords**: Gen11, Epoch 8, Sim2Real, UAV, timing analysis, IK, MJPC latency, REALTIME_RECORDING.

1. **Sim2Real Timing Budget Audit**: Authored `REALTIME_RECORDING/U1/ideas.md` to analyze the measurement gap in the `avg_time` metric, which currently excludes MuJoCo Inverse Kinematics (IK), PID/MJPC control loops, and physics simulation. 
2. **Latency Bottleneck Identification**: Established that while IK and low-level numerical controllers take a negligible amount of time (< 1-15 ms), the heavy generative inference (`fm_ms` + `proj_ms`) operates at around 1 Hz, which heavily violates the real-time budget (typically 10-50 Hz) for a dynamic UAV.
3. **Catastrophic State Drift Diagnostics**: Concluded that deploying a 1 Hz planner directly to a UAV causes "super lag" leading to catastrophic failures due to state drift (the UAV flies blind on a 1-second-old observation) and waypoint starvation.
4. **Future Mitigation Strategy**: Identified that achieving Sim2Real viability requires model compression (e.g., fewer diffusion steps), projection-free constraints, and asynchronous control loop architectures (running PID at 50Hz+ while FM guides asynchronously).

***

## Gen11 Epoch 8 "U7": MJX Conda Environment Isolation & Auto-Selection (July 2, 2026)

**Keywords**: Gen11, Epoch 8, U7, conda, environment isolation, mujoco-mjx, sbatch auto-select.

1. **Dependency Crisis Resolution**: Addressed a critical environment crisis where installing `mujoco-mjx` (which requires MuJoCo 3.x) into the main `FMPCC` conda environment broke D3IL's XML scene parsing and caused segfaults due to library collisions (D3IL relies on MuJoCo 2.3.7).
2. **Isolated MJX Environment Setup**: Created an isolated cloned environment (`FMPCC_mjx`) dedicated exclusively to the UAV `mjpc` controller, ensuring that the new dependencies do not pollute the core training and evaluation baselines. Authored `install_UAV_mjpc_mjx_env.md` for reproducible cluster deployment.
3. **Automatic SBATCH Environment Selection**: Implemented transparent environment switching in `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`. The orchestrator now reads `plan_flow_matching_v3_uav['controller']` from `config/uav.py` and automatically activates the correct conda environment (`FMPCC_mjx` for MJPC, `FMPCC` for standard PID controllers) prior to execution, eliminating manual environment management errors.
4. **PID Constant Velocity Fix (`F1`)**: Resolved an `AttributeError` in the `pid_const_v` controller by correctly accessing the dataset's action `fields` attribute, ensuring that stable dataset-derived flight velocities can be correctly simulated.

***

## Gen11 Epoch 8 "U7": `success_relaxed` Finish-Line Crossing Metric (July 3, 2026)

**Keywords**: Gen11, Epoch 8, U7, success_relaxed, finish-line crossing, evaluation metric.

1. **Evaluation Metric Redefinition**: Addressed the issue where the `success` metric scored an outright `FAIL` for rollouts that reached the goal but subsequently drifted/overshot during the remaining fixed-length physics steps.
2. **`success_relaxed` Implementation**: Implemented a new `success_relaxed` metric in `eval_fm_uav.py` that treats the goal as a race finish line. It uses a one-way `crossed_line` latch triggered when the drone crosses a vertical plane at the goal, oriented perpendicular to the final approach heading.
3. **Artifact Integration**: Added `n_success_relaxed` to the legacy npz schema in `eval_artifacts.py` and incorporated it into the per-rollout diagnostics and `results.json`. The original strict `success` metric remains untouched and strictly additive.

***

## Data Analysis Tool v3: Batch Indexing and CLI Passthrough Updates (July 3, 2026)

**Keywords**: DA Code v3, batch analysis, argument passthrough, candidate indexing, numeric sorting.

1. **Candidate Indexing Normalization**: Updated `main_da_batch` candidate indexing from alphanumeric to integer keys to correctly match downstream pipeline expectations. Modified the DA Visualizer to normalize the Candidate column to a string type, implementing smart sorting for mixed numeric/alphabetic IDs.
2. **CLI Argument Passthrough**: Enabled argument passthrough in the SLURM DA batch analysis scripts (`run_da_batch_avoiding_combined.sh`, `run_da_batch_v3.sh`, `run_da_batch_visual_aligning.sh`), supporting native DA flags like `--no-plots` to streamline workflow executions directly from the cluster orchestrator.

***

## Gen3v4 "U8": iMeanFlowODE `torchdiffeq` Solver Integration (July 3, 2026)

**Keywords**: Gen3v4, U8, iMeanFlowODE, torchdiffeq, homing missile, h_sub, interval-aware step sizing.

1. **Solver Silent-Failure Fix**: Diagnosed a bug where `iMeanFlowODE.p_sample_loop` silently ignored `ode_solver_backend_v3` configurations (like `Mrk4`) and always fell back to legacy Euler integration, invalidating prior solver-comparison data.
2. **`torchdiffeq` Integration**: Ported the `torchdiffeq` dispatch logic from the sibling `FlowMatchingIMF` to correctly support higher-order integrators (RK4, Midpoint) within `iMeanFlowODE`.
3. **"Homing Missile" `h_sub` Sizing**: Implemented a dynamic sub-stage step sizing (`h_sub = t1 - t_scalar`) mechanism to strictly preserve the iMeanFlow mathematical formulation. This ensures that every sub-stage query inside a macro-step always targets the macro-step's exact end (`t1`), keeping network queries inside their trained interval domain `[t, t+h]` regardless of the solver backend.

***

## Gen11 Epoch 9: Full PCC Constraint Geometry — Planning & Architecture Audit (July 4, 2026)

**Keywords**: Gen11, Epoch 9, E9, PCC constraints, per-scene geometry, halfspace, obstacles, bounds, geo_bounds, DPCC-faithful, UAV.

### Motivation

E7 restored the full PCC/DPCC projector skeleton (candidate fan, selection, constraint metrics) but ran **dynamics-only** — the spatial constraint slots (`halfspace_constraints`, `obstacle_constraints`, `workspace_bounds`) were wired but gated off. E6-U2 had left the `pillars` scene at **0% success** with the raw FM policy. E9's objective is to bring in real per-scene geometry and validate that DPCC projection lifts that failure, following the Gen7 visual-aligning lineage precisely.

### Forensic Audit: DPCC Constraint Dimensionality (Pre-E9)

1. **DPCC Constraint Binding Audit**: Before any code was written, authored `PLAN_E9_PCC_constraints.md` and `STUDY_DPCC_constraint_dim_binding.md` to definitively settle the question of which trajectory dimensions each constraint family binds to. Verified against the canonical avoiding-task config/eval pair: halfspace and obstacle constraints bind to **actual position `p` only** (dims 6,7,8 in the UAV 12D tensor); the `dynamics` rows couple `p_des` and `p` through the action; `bounds` in avoiding binds to **action dims** (velocity), not position. Gen7 visual-aligning's `_DIM` mapping was confirmed correct, not a bug.
2. **Identified the `bounds` Conflation Bug (Cross-Repo)**: Discovered that visual-aligning had silently **repurposed** the `bounds` constraint_types flag to mean a Cartesian workspace position box, dropping DPCC-avoiding's original action-magnitude limit. This was documented in `PROBLEM_bounds_velocity_vs_geo.md` and immediately scheduled for a cross-repo fix (Gen7 C3 patch, below).
3. **UAV Tensor Invariant**: Established that the UAV `p` slice sits at dims 6–8 in **both** the 12D (E7/PID) and 9D (E8/MJPC) layouts, meaning one geometry config serves both controllers with no branching.

### Gen11 Epoch 9 Init — Per-Scene Constraint Geometry Implementation (July 4, 2026)

1. **Per-Scene DPCC Constraint Resolution**: Restructured `config/uav_projection.yaml` from a flat global-placeholder format to a `geo_constraint_variants` named-entry list (mirroring `visual_aligning_eval.yaml`), with one full-stack geometry block per scene (`empty`, `corridor`, `pillars`, `s_curve`). An `active_geo_variants` selector controls which scenes run — editable in the YAML without any code change.
2. **`empty` Baseline**: Explicitly configured `constraint_types: []` for the `empty` scene, making it a deliberate no-op raw-FM baseline (the denominator every other scene is measured against).
3. **`corridor` Geometry**: Encoded the two corridor walls as halfspace planes (`formulate_halfspace_constraints`), since a box-exclusion in DPCC-native form is halfspace faces. Added workspace bounds on `p` and the action-magnitude bound.
4. **`pillars` Geometry**: Represented the 6 full-height pillars as `sphere_outside` constraints on `[x, y]` only — the exact cross-section of an infinite vertical cylinder in DPCC's quadratic primitive. Also added an outer envelope halfspace and workspace bounds.
5. **`s_curve` Per-Segment Constraint Switching**: Implemented a declarative `x_active: [lo, hi]` field on halfspace entries to enable per-replan active-set selection. This addresses the non-convex geometry of the S-curve (whose two wall segments have an empty intersection globally), resolving the constraint at each MPC step based on the drone's current `x` position.
6. **Halfspace Helper Robustness Fix (`constraints_helpers.py`)**: Fixed a divide-by-zero in `formulate_halfspace_constraints` where horizontal walls (slope `m = 0`, i.e., corridor walls) caused `1/m = inf`. Replaced the slope-intercept normal with the perpendicular construction (`n = (-dy, dx)`) already used by the plotting code, validated for numeric equivalence on existing avoiding-task inputs.
7. **`geo_tag` Output Axis (Fix 1)**: Added a `geo_tag` dimension to UAV eval output paths (e.g., `pillars_bounds+dynamics+halfspace+obstacles/`) to prevent result collisions when running multiple geometry configurations under the same scene in a single sweep.

***

## Gen7 C3 / Gen6V4 C3 — `bounds` Constraint Split: Restoring DPCC Action-Magnitude Guard (July 4, 2026)

**Keywords**: Gen7, Gen6V4, C3, `bounds`, `geo_bounds`, action-magnitude, DPCC-faithful, constraint_types rename.

1. **Root Cause**: The `bounds` `constraint_types` flag in `config/visual_aligning_eval.yaml` had been silently redefined to mean a Cartesian position box on actual position (`workspace_bounds`), which **dropped** the DPCC-paper's original meaning: a normalized action-magnitude limit on action dims 0,1,2 (`dx, dy, dz`). This left all `combined_*` constraint entries missing one of the Table-1 DPCC constraint families.
2. **Rename to `geo_bounds`**: All ~9 call sites per file that gated the position box on `'bounds'` were renamed to `'geo_bounds'` in both `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7) and `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4). Both files share one YAML and carry structurally identical copies of `setup_dpcc_projector` — both needed the identical fix to remain consistent.
3. **Restored `bounds` (Action-Magnitude)**: Added a new `if 'bounds' in constraint_types:` block that self-derives action limits from `act_normalizer.mins/.maxs` (the dataset's own normalized `dx,dy,dz` range) by default (`action_bounds: 'auto'`), eliminating the need to hardcode task-specific velocity limits. This mirrors the E9 Fix_3 pattern applied on the UAV.
4. **YAML Updates**: Added `action_bounds: 'auto'` top-level key; renamed `bounds_only_1/2` entries to `geo_bounds_only_1/2`; added a new `action_bounds_only` ablation entry; upgraded `combined_4`/`combined_5` (the currently active entries) to carry **both** `geo_bounds` and `bounds`, making them genuinely match the full DPCC constraint set. The `combined_5` entry — the active default — now enforces the restored action-magnitude cap on the next cluster run.
5. **Tightening Exclusion**: The restored action bound is correctly excluded from the `-tightened` enlarge margin (it is a dataset-range cap, not a spatial surface), matching the avoiding-task convention.

***

## Gen11 Epoch 9 — Constraint Geometry Refinements (Fixes 2–6, July 4, 2026)

**Keywords**: Gen11, Epoch 9, E9, selectable geo variants, ablation tiers, geo_bounds split, multi-geo loop, config snapshot.

### Fix 2: Action-Magnitude Bounds Auto-Derivation (E9 Fix 3)
1. **Self-Derived Action Bounds**: Implemented an `action_bounds: 'auto'` mode in `eval_fm_uav.py` that derives the action-magnitude bound (lb/ub on dims 0,1,2 = Δp_des) directly from the action normalizer's `mins/maxs`, exactly matching the C3 approach above. Prevents hardcoding a per-task magic number.

### Fix 3: Selectable Per-Scene Geometry Configurations (E9 Fix 4)
1. **Per-Scene Geo Variant Selection**: Extended `uav_projection.yaml` with the `active_geo_variants` selector and the ability to have multiple named entries per scene, identified by a `scene` field. Added `load_pcc_config` resolution logic that raises a `ValueError` on ambiguity (> 1 match) as a safety guard — later promoted to the multi-loop pattern in Fix 6.
2. **Selectable Ablation Tiers**: Added `<scene>_dynamics_only` and `<scene>_dynamics_bounds_only` entries alongside the `<scene>_combined_1` full-stack entries for `corridor`, `pillars`, and `s_curve`. Switching the active tier requires only editing `active_geo_variants` in the YAML, no code change — identical to the visual-aligning mechanism.

### Fix 4: Diagnostic Observability Parity (E9 U2)
1. **Constraint Geometry Schematic Plot**: Ported `plot_geo_constraints` from Gen7 visual-aligning into the UAV eval. The 3-panel figure (3D wireframe / XY top-down / XZ side) is generated once per geometry configuration before any rollout, rendering the **true enforced boundary** (including `r_drone + margin_base` inflation, and `+ enlarge_constraints` for tightened variants). The UAV-specific `x_active` halfspace segments (s_curve) are clipped and rendered over their live x-range only, not as infinite lines. Output saved as both `.png` and `.svg` for thesis/paper use.
2. **GIF Step-Counter Overlay**: Added a per-frame `sK` step counter burned into overhead-camera GIF frames (yellow text, top-left, `cv2.putText`), matching visual-aligning's diagnostic convention. Allows cross-referencing visible events in recordings against per-step structured logs without re-deriving the frame index.

### Fix 5: `geo_bounds`/`bounds` Split Propagated to UAV (E9 Fix 5)
1. **Consistent Naming**: Applied the same `geo_bounds`/`bounds` split from Gen7 C3 to the UAV eval codebase. All five call sites in `eval_fm_uav.py` that gated the workspace-box on `'bounds'` were renamed to `'geo_bounds'`; the action-magnitude block (Fix 3) now sits behind the independent `if 'bounds' in ctypes:` gate. The repo's naming convention for these two orthogonal constraint families is now fully unified.
2. **Ablation Entry Consistency**: Updated `uav_projection.yaml`'s `*_combined_1` entries to list `['dynamics', 'geo_bounds', 'halfspace', 'obstacles', 'bounds']`, making the distinction between the position box and the action cap explicit in the config.

### Fix 6: Multi-Geo-Variant Loop + Config Snapshot Wrong-YAML Bug (E9 Fix 6 / July 5, 2026)
1. **Multi-Geo-Variant Loop**: Resolved a `ValueError` triggered by the Fix 4 ambiguity guard when multiple geo entries for the same scene (e.g., `s_curve_dynamics_only`, `s_curve_dynamics_bounds_only`, `s_curve_combined_1`) were simultaneously listed in `active_geo_variants`. The fix refactors `eval_scene` to loop over all matching entries rather than raising — mirroring how Gen7's eval already loops over `_geo_specs` in a single invocation. Extracted `_load_base_cfg`, `_resolve_active_geo_matches`, and `_apply_geo_entry` as reusable helpers; `load_pcc_config` retained its single-match contract for any caller that still requires it.
2. **Config Snapshot Wrong-YAML Bug**: Discovered that `flow_matcher_v3_uav/utils/setup.py::snapshot_configs` was hardcoding `'config/projection_eval.yaml'` (the avoiding-task YAML) as the snapshot target, so every UAV run's `config_snapshot_uav/` folder silently contained avoiding's config instead of the UAV's own `uav_projection.yaml`. Fixed by correcting the hardcoded path and destination filename. Extended the same audit to `diffuser_visual_avoiding/utils/setup.py` and `fm_visual_avoiding/utils/setup.py` — both also corrected. All past UAV runs before this fix carry an invalid provenance snapshot; the git commit hash in the SLURM job header is the authoritative config reference for those runs.

***

## Gen11 Epoch 9 Fix 7: UAV Rollout Render Optimization (July 5, 2026)

**Keywords**: Gen11, Epoch 9, E9 Fix 7, UAV, gif, resolution reduction, delta-frame encoding, storage footprint.

1. **UAV Visual Asset Profiling**: Investigated a massive storage footprint discrepancy where UAV evaluation GIFs were ~7x larger than the visual-aligning arm GIFs. Established that the arm's low resolution (96x96) is structurally tied to its vision policy input, whereas the UAV's high-resolution (360x360) overhead render was a purely arbitrary debug artifact since its policy is state-only.
2. **Resolution Downscaling**: Reduced the default `_make_overhead_renderer` resolution from 360x360 to 200x200, achieving a ~3.2x reduction in per-frame pixel count without degrading visual utility or impacting the training pipeline.
3. **Delta-Frame Encoding Integration**: Optimized `save_rollout_gif` by injecting `subrectangles=True` to the `imageio.mimsave` writer, enforcing delta encoding that only stores pixels changing between frames. This is highly synergistic with MuJoCo's static backgrounds (floors/pillars). Additionally added a reduced color palette constraint (`palettesize=128`), fully enveloped in a fallback `try/except` guard against backend incompatibilities.

***

## Gen11 Epoch 9 U8: Variant-Level Constraint Ablation Toggles (July 5, 2026)

**Keywords**: Gen11, Epoch 9, U8, constraint ablation, geo_free, bounds_free, variant toggle, projection_variants.

1. **Ablation Paradigm Shift**: Refactored the constraint ablation methodology away from proliferating redundant per-scene geometry configurations (e.g., `<scene>_dynamics_only`). The new design aligns with the pre-existing `model_free` toggle by utilizing orthogonal variant-level substring gates.
2. **Implementation of `geo_free` and `bounds_free`**: Introduced `geo_free` to collectively disable spatial constraints (`geo_bounds`, `halfspace`, `obstacles`) and `bounds_free` to specifically bypass action-magnitude limits.
3. **Combinatorial Expressiveness**: Enabled pure substring composition (e.g., `geo_free-bounds_free` safely equates to `dynamics_only`, and `geo_free-model_free` equates to `action_bounds_only`) to run distinct ablations without redefining the underlying physical `geo_constraint_variants` in the YAML.
4. **YAML Simplification**: Cleaned `config/uav_projection.yaml` by pruning 6 redundant ablation geometries. `active_geo_variants` now strictly mandates exactly one definitive entry per scene, significantly reducing configuration surface area and cognitive overhead.

***

## Gen7 & Gen6V4: U8 Sync for Visual-Aligning Projection Ablations (July 5, 2026)

**Keywords**: Gen7, Gen6V4, U8 sync, visual-aligning, projection_variants, geo_free, bounds_free.

1. **Cross-Project Consistency**: Synchronized the new U8 constraint ablation architecture (variant-level toggles) backward into the original `visual-aligning` pipeline where the pattern originated with `model_free`.
2. **DPCC Projector Patching**: Updated the live `setup_dpcc_projector` dispatch logic identically across both the Flow Matching engine (`fm_visual_aligning_test/eval_fm_visual_aligning.py`) and the baseline DDPM engine (`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`).
3. **YAML Pruning**: Removed legacy and redundant ablation structures (`dynamics_only`, `dynamics_bounds_only`, `action_bounds_only`) from `config/visual_aligning_eval.yaml`, replacing them entirely with the combinatorial `geo_free` and `bounds_free` flags within `projection_variants`. Maintained non-redundant experimental entries (e.g., specific 2D vs 3D test scenarios) to preserve full benchmarking fidelity.

***

## Infrastructure Hotfix: Config Snapshot Persistence & Stale YAML Cleanup (July 5, 2026)

**Keywords**: config snapshot, provenance drift, process-level guard, stale YAML, filesystem check.

1. **Snapshot Guard Bug Discovery**: Identified that the previously implemented configuration snapshot fix was failing to execute during subsequent runs. The root cause was traced to a filesystem-based existence check (`os.path.exists(_snap_dir)`) in the evaluation script, which permanently skipped the snapshot update once an older (potentially incorrect) snapshot directory existed.
2. **Process-Level Memory Set Implementation**: Replaced the flawed filesystem check with an in-memory, process-scoped set (`_SNAPSHOTTED_DIRS`). This ensures that every new job execution (fresh process) unconditionally re-snapshots the active configuration, naturally overwriting stale contents, while still preventing redundant disk I/O within the same job's internal loops.
3. **Stale Artifact Eradication**: Added explicit cleanup logic post-snapshot to detect and delete wrongly named legacy files (e.g., `projection_eval.yaml` inside UAV snapshots or `visual_aligning_eval.yaml` inside visual-avoiding snapshots) left over from the earlier hardcoded-path bugs, completely eliminating config provenance drift.
4. **Scope of Fix**: Successfully deployed to `eval_fm_uav.py`, with corresponding cleanup logic patched into the setup utilities for both `fm_visual_avoiding` and `diffuser_visual_avoiding` packages.

***

## SLURM Infrastructure: UAV Eval Timing Logging (July 6, 2026)

**Keywords**: SLURM, eval_fm_uav.sh, timing.

1. **Job Execution Tracking**: Added evaluation job timing logs to the UAV `eval_fm_uav.sh` SLURM script to improve workflow execution observability.

***

## Gen11 Epoch 9 Fix 9: UAV Visualization GIF Aggressive Size Reduction (July 6, 2026)

**Keywords**: Gen11, Epoch 9, E9 Fix 9, UAV, gif size, resolution, frame stride, palette.

1. **Aggressive Footprint Reduction**: Iterated on previous visual asset optimizations to further reduce the storage footprint of debug UAV rollouts.
2. **Quality-for-Size Tradeoff**: Shrunk the overhead renderer resolution from 200x200 down to 140x140, increased the frame stride from 2 to 3, and halved the color palette size to 64. This explicit quality-for-size tradeoff yields GIFs roughly 10-15x smaller than the original baseline.

***

## Gen11 Epoch 9 Fix 10: Nested Metric Schema & Success Invariant Fix (July 6, 2026)

**Keywords**: Gen11, Epoch 9, E9 Fix 10, success_relaxed, crossed_line, json schema, npz metrics.

1. **Success Invariant Violation Fixed**: Addressed a bug where degraded UAV trajectories with high tracking error could trigger `success` (by physical proximity to the goal) but fail `success_relaxed` because they approached from an un-modeled angle. The `crossed_line` latch now safely triggers based on proximity as well, restoring the strict `success ⇒ success_relaxed` mathematical invariant.
2. **Artifact Schema Unification**: Entirely redesigned the evaluation metrics schema for both `.json` and `.npz` outputs. Outputs are now neatly nested under logical groupings (`physical`, `constraint`, `goal`, `success`, `timing`). 
3. **Metric Extraction Pipeline Upgrade**: Updated downstream metric processing scripts, including `analyze_npz.py` and `aggregate_scene_summaries.py`, to natively extract and report from the new structured metrics format while preserving backward compatibility.

***

## Gen7 & Gen6V4 C4: Visual Aligning Metric Schema Unification & Contact Tracking (July 6, 2026)

**Keywords**: Gen7, Gen6V4, C4, metric schema, success_relaxed, contact tracking, visual aligning.

1. **Cross-Architecture Schema Alignment**: Synchronized the visual-aligning evaluation scripts for both Gen7 (Flow Matching) and Gen6V4 (Diffuser) pipelines to adopt the exact same grouped JSON/NPZ structure introduced in UAV's Fix 10 (`success`, `outcome`, `timing`, `context`, `contact`, `constraint`).
2. **First/Last Contact Tracking**: Implemented new metrics capturing the step index and physical `XY` coordinates for both the first and last recorded contacts during an evaluation rollout. 
3. **Diagnostic Plotting Enrichment**: Automatically rendered first-contact (blue star) and last-contact (purple X) markers directly onto the MPC foresight SVG plots for instant visual debugging of collision dynamics.
4. **`success_relaxed` Integration**: Ported the `success_relaxed` concept into the visual-aligning domain, evaluating outcome solely on terminal distance thresholding.

***

## Gen11 Epoch 9 Hotfix: Timing Aggregation KeyError (July 6, 2026)

**Keywords**: Gen11, Epoch 9, hotfix, timing dictionary, KeyError.

1. **Nested Key Access Fix**: Resolved a `KeyError` within the evaluation summary generation caused by the transition to the nested timing dictionary structure. The script now correctly navigates the nested paths to extract timing data.

***

## Infrastructure Refactor: SLURM Seed Standardization (July 6, 2026)

**Keywords**: SLURM, seed, pipeline standardization, visual aligning, visual avoiding.

1. **Seed Consolidation**: Updated all major training and evaluation SLURM batch scripts across multiple pipelines (visual aligning, visual avoiding, drifting) to standardize on `seed=6` as the default execution seed.

***

## Gen11 Epoch 9 & Gen7/Gen6V4 C4: Evaluation Progress Logging & ETA (July 7, 2026)

**Keywords**: Gen11, Epoch 9, Fix 11, C4, progress tracking, ETA, breadcrumbs, SLURM timeout, job monitoring.

1. **UAV Pipeline Progress Breadcrumbs**: Addressed the issue where long-running UAV evaluation jobs provided insufficient logs before a potential 24h SLURM timeout. Added detailed nesting breadcrumbs (`scene` → `geo_entry` → `variant` → `trial`) and per-rollout timing/ETA prints inside the execution loops of `eval_fm_uav.py`, without altering any model behavior or outputs.
2. **Visual Aligning Pipeline Parity**: Synchronized identical progress-tracking logic into the visual-aligning pipelines (both Gen7 `eval_fm_visual_aligning.py` and Gen6V4 `eval_visual_aligning_dpcc.py`). The existing per-rollout debug info was preserved, with a new index/total and elapsed/ETA summary appended at the bottom.
3. **SLURM Script Visibility**: Updated the outer wrapper SLURM scripts to explicitly print the current seed being evaluated out of the multi-seed list, ensuring complete job provenance visibility directly from the `.log` files.

***

## Gen3v4 iMF Condition Analysis & AI Coding Guardrails (July 7, 2026)

**Keywords**: Gen3v4, iMF, conditioning mechanism, theory vs code audit, CLAUDE.md.

1. **Theoretical Deep-Dive (Conditioning)**: Authored `thought&theory.md` mapping exactly how the Flow Matching and iMeanFlow (iMF) architectures achieve location-awareness. Clarified that the starting state and goal `c = (s_0, s_goal)` are injected cleanly via concatenated tokens, ensuring that the integration trajectory is anchored to the true robot position without injecting location data into the pure noise tensor `z_t`.
2. **Code vs. Theory Audit**: Conducted a comprehensive audit (`AUDIT_thought&theory_vs_code.md`) verifying that the actual codebase accurately implements the mathematical derivations of iMF conditioning.
3. **AI Coding Guidelines**: Introduced `CLAUDE.md` and related `.claude` memory files to establish concrete architectural guidelines and repository guardrails for future AI-assisted development.

***

## Gen3v4 U9: Validation Loss Stability & Incremental W&B Logging (July 7, 2026)

**Keywords**: Gen3v4, U9, validation loss, W&B syncing, seeded train/test split, eval mode fix, raw_mse.

1. **Incremental W&B Flushing**: Fixed an issue where Weights & Biases logs were only uploaded after a full training run finished, leaving data permanently lost if SLURM killed the job via timeout. Training loops (`utils/training.py` and `train_flow_matching_v3_imeanflow.py`) now incrementally sync logs and validation losses at the end of every epoch.
2. **Seeded Dataset Split**: Seeded the PyTorch `random_split` generator (`split_seed=42`) used for separating training and validation data. This ensures consistent data partitioning across paused/resumed runs and multi-seed trials, preventing validation data from leaking back into the training set.
3. **Eval-Mode Restored**: Fixed a critical legacy bug inherited from DPCC where the model's `.eval()` state was never reverted back to `.train()` after the first validation pass. While current backbones were unaffected, this future-proofs the system for components like Dropout and BatchNorm.
4. **`raw_mse` Exposure**: Added a scale-invariant `val/raw_mse` metric to W&B that captures the unweighted reconstruction loss, providing a stable baseline comparison point across runs that is immune to adaptive loss reweighting schemas.

***

## Gen3v4 U9 Hotfix: Trainer Import Correction & W&B Crash Resolution (July 8, 2026)

**Keywords**: Gen3v4, U9 hotfix, TypeError, Trainer import, diffuser.utils, raw_mse.

1. **Bug Diagnosis**: Investigated a training crash at step 0 (`TypeError: Trainer.train() got an unexpected keyword argument 'on_epoch_end'`). Found that the train script was importing the legacy DPCC `diffuser.utils.Trainer` instead of the newly updated iMF-specific Trainer.
2. **Import Resolution**: Updated `train_flow_matching_v3_imeanflow.py` to import `Trainer` from `flow_matcher_v3_imeanflow.utils.training`, aligning the training script with the evaluation script and giving it access to the U9 callbacks and metrics.
3. **Loss Restoration**: Updated the `load()` function in the iMF Trainer to restore `test_raw_mse_losses` on resume, ensuring the raw MSE metric history is not reset when training is paused and restarted.

***

## Gen3v4 U9.3: Comprehensive Metric-Parity for W&B (July 8, 2026)

**Keywords**: Gen3v4, U9.3, metric-parity, W&B, a0_loss, aux_loss, raw_mse, lr_history, debugging.

1. **Objective Parity**: Closed the visibility gap between DPCC, Gen0, and iMeanFlow logging. Implemented tracking and uploading for `train/a0_loss`, `test/a0_loss`, `train/raw_mse`, `train/aux_loss`, `test/aux_loss`, and `train/lr` (learning rate).
2. **Trainer Aggregation Upgrade**: Expanded the `test()` method to return a 4-tuple including `aux_loss` when available. Tracked the learning rate history dynamically from the cosine scheduler to detect proper warmups on resume.
3. **W&B Upload Revamp**: Rewrote the upload logic in `train_flow_matching_v3_imeanflow.py` to use a declarative dictionary map (`companion_keys`), cleanly piping all metrics from `losses.pkl` into W&B while silently skipping missing keys on older checkpoints.

***

## Documentation Maintenance: U-Net & Horizon Cross-Study Audit (July 8, 2026)

**Keywords**: TemporalUnet, horizon adaptability, dim_mults, CasADi flattening.

1. **Default vs Runtime Correction**: Re-audited the `TemporalUnet` architectures across HardFlow, DPCC, and FM-PCC. Corrected previous misconceptions by confirming that all active models override defaults to run at `dim=32`, with the primary differences lying in `dim_mults` downsampling levels and conditioning mechanisms.
2. **CasADi Dimensionality Impact**: Documented how horizon length directly dictates the NLP solver complexity in `WrappedFlowUnet`. A smaller horizon (H=8 vs H=16) drastically reduces the degrees of freedom for the CasADi decision variables.

***

## Gen11 Epoch 9 Fix 12: Constraint Feasibility & Realized Homotopy (July 8, 2026)

**Keywords**: Gen11, Epoch 9, Fix 12, constraint feasibility, inflation margin, homotopy_flown, phantom violation, pillars, s_curve.

1. **Infeasible Geometry Rectification**: Investigated 0% success rates on `pillars` and discovered the inflated constraints (r=0.53m) were mathematically near-infeasible, closing off all expert training routes. Reduced `r_drone` inflation from worst-case 0.36m to accurate lateral 0.31m and `margin_base` from 0.05m to 0.02m.
2. **Synthetic Constraint Purge**: Removed synthetic envelope halfspaces from the `pillars` scene and expanded the workspace boxes across all scenes (pillars, s_curve, corridor) to physically contain the start and goal positions, eliminating structural phantom violations on step 0.
3. **Realized Homotopy Logging**: Addressed the misleading nature of the commanded `homotopy` label (e.g., LLL, LRL) in unconditioned policies by implementing `_realized_homotopy()`. The evaluation script now physically analyzes the flown path to determine the true route taken (`homotopy_flown`) and records it in the JSON artifacts.
4. **Pre-Flight Feasibility Check**: Added an upfront `_warn_expert_route_infeasibility()` diagnostic in `eval_fm_uav.py`. This checks the dataset's expert route against the instantiated DPCC projection geometry before rolling out, loudly warning if the constraint set is impossible to solve.

***

## Gen11 Epoch 9 U13: Deterministic Episode Length & DPCC-style Goal Stop (July 9, 2026)

**Keywords**: Gen11, Epoch 9, U13, deterministic episode length, fixed step budget, early stop, goal-reach latch.

1. **Deterministic Episode Budgets**: Replaced the per-trial random sampling duration with a fixed, per-scene step budget (`SCENE_MAX_EPISODE_LENGTH`). This eliminates trial-to-trial randomness and ensures uniform execution length across tests, resolving issues where trajectories were cut off prematurely or overshot the goal.
2. **DPCC-Style Early Termination**: Integrated a `goal_reached_latch` in the physics inner loop and adopted an early-stop mechanism that terminates the evaluation exactly when the goal radius is breached. Time-to-goal measurements (`n_fm_steps`) are now perfectly deterministic functions of policy quality rather than being coupled to the execution budget.
3. **CLI & Config Overrides**: Added a `--max-episode-length` CLI argument and yaml support, providing identical precedence and parity with the legacy DPCC testing infrastructure for setting global or per-scene step limits.

***

## Gen7 & Gen6V4 C5: Consolidate Visual-Aligning Artifacts & Crash-Safety (July 9, 2026)

**Keywords**: Gen7, Gen6V4, C5, npz_pkl_consolidation, raw truth, crash-safety, partial saves, MPC fan.

1. **Single Source of Raw Truth**: Consolidated the fragmented per-rollout `.pkl` arrays into a single `<variant>.npz` file for visual-aligning eval scripts, strictly mirroring the UAV pipeline schema.
2. **Schema Upgrades**: Widened `obs_all` from 3-D to 6-D to store both the commanded and executed paths seamlessly. Upgraded `sampled_trajectories_all` to store the full candidate fan matrix (`(B,H,3)`) for each replanning step, rather than only the selected trajectory.
3. **Incremental Crash-Safety Saves**: Implemented a best-effort, crash-safe sidecar mechanism (`<variant>.partial.npz`) that atomically flushes aggregated data every 5 rollouts. This safeguards partial rollout data against abrupt SLURM kills or 24h timeouts without polluting the authoritative single-write completion semantics expected by downstream Data Analysis pipelines.

***

## Gen11 E9 U8b & Gen7/Gen6V4 C3: Geometry-Alone Projection Variant (July 9, 2026)

**Keywords**: Gen11, Epoch 9, U8b, Gen7, Gen6V4, C3, model_free-bounds_free, geometry-alone ablation.

1. **Trilogy Completion**: Added the `model_free-bounds_free` projection variant to `config/uav_projection.yaml` and `config/visual_aligning_eval.yaml`. This variant cleanly projects spatial geometry without the constraints of dynamics or action limits.
2. **Ablation Clarity**: This addition fulfills the gap in the subset "X alone" ablation tests. It explicitly enables direct analysis into why pure-geometry projection worsens behavior, confirming hypotheses regarding the necessity of coupled dynamics anchoring (`diffuser` performance) versus geometry-only constraint corruption.

***

## Infrastructure Refactor: Realtime SLURM Log Streaming (July 9, 2026)

**Keywords**: SLURM, real-time logging, PYTHONUNBUFFERED, stdout streaming, Fix 11b.

1. **Unbuffered Stdout**: Exported `PYTHONUNBUFFERED=1` across the visual-aligning (`eval_fm_visual_aligning.sh`, `eval_visual_aligning_dpcc.sh`) and UAV flow-matching (`eval_fm_uav.sh`) evaluation sbatch scripts. This prevents Python's native IO buffering from retaining stdout prints, ensuring immediate visibility into real-time rollout progress and ETA logs on the cluster.

***

## Gen11 Epoch 9 U13: Investigation: s_curve Geometry Destabilization & Ordering Flip (July 10, 2026)

**Keywords**: Gen11, Epoch 9, U13, investigation, s_curve, non-convex geometry, ordering flip, bounds_free.

1. **Ablation Ordering Flip**: Investigated a dramatic reversal in ablation performance on the `s_curve` scene where geometry-keeping variants (e.g., `dpcc-*`, `post_processing`) perform significantly worse than geometry-free variants (`geo_free-bounds_free`), opposite to the findings on the `corridor` scene.
2. **Root Cause (Non-Convex Feasible Set)**: Traced the failure to the `s_curve`'s narrow (~24 cm), non-convex, per-segment switching constraints. The aggressive state corrections required to satisfy this geometry cause destabilization of the action command, particularly at the crossover corner where `x_active` switches walls.
3. **Action-Bound Importance**: Found that removing the action bound (`bounds_free`) when geometry and dynamics are active causes an immediate step-0 crash, confirming that the action bound is a critical load-bearing stability cap against large dynamics-coupled geometry corrections on tight scenes.
4. **Long-Horizon Dynamics Anchoring**: Concluded that on a long 750-step horizon, keeping dynamics active while stripping geometry (`geo_free-bounds_free`) produces the best results. The repeated re-anchoring to the measured state smooths out the raw policy drift enough to almost reach the goal safely.

***

## Gen11 Epoch 9 U8c: Missing Tightened Siblings for Geometry-Keeping Variants (July 10, 2026)

**Keywords**: Gen11, Epoch 9, U8c, tightened, bounds_free, geometry-keeping, visual-aligning sync.

1. **UAV Missing Tightened Twins**: Identified an inconsistency in `config/uav_projection.yaml` where the newly added geometry-keeping ablation variants (`bounds_free`, `model_free-bounds_free`) were missing their explicit `-tightened` counterparts. Added `bounds_free-tightened` and `model_free-bounds_free-tightened` to ensure the ablation suite is complete, following the rule that tightening is only meaningful (and should only be applied) if the variant actually retains spatial geometry.
2. **Visual-Aligning Sync Check**: Confirmed no changes were needed for the visual-aligning pipeline configurations (`config/visual_aligning_eval.yaml`) because that codebase automatically generates tightened permutations via a programmatic outer loop rather than static YAML enumeration. Both pipelines now reach the identical end state where all geometry-keeping variants evaluate a tightened ablation.

***

## Gen11 Epoch 9 Fix 14: MPC Foresight SVG Enforced Constraints Overlay (July 10, 2026)

**Keywords**: Gen11, Epoch 9, Fix 14, UAV, foresight SVG, enforced constraints, all.png.

1. **Foresight SVG Constraint Overlay**: Updated `write_mpc_foresight` in `FM_v3_uav_test/eval_artifacts.py` to draw the actual enforced geometric surfaces (workspace bounds, halfspace walls with their `x_active` clipping, obstacle balls) at their true inflated margin onto the MPC candidate fan visualization. This allows for direct validation of whether the DPCC projector successfully solved constraints like the non-convex corners on `s_curve`.
2. **Redundant `all.png` Removal**: Removed the writing of a byte-identical duplicate image (`all.png`) from `plot_overview`, as it was erroneously saving per-variant data under an aggregate alias name.

***

## NPZ Analysis Tool v4: Horizon Plan Comparison & Interactive Trajectory Visualizer (July 11, 2026)

**Keywords**: npz_analysis, compare_horizon_plans, horizon comparison, MPC fan, npz_traj_visualizer, HTML visualizer, interactive, fix_1.

1. **Horizontal Plan Comparison (`compare_horizon_plans.py`)**: Implemented a standalone tool for horizontal (across-method) H-step plan comparison. Unlike `analyze_npz.py` which is vertical (aggregates per file), this tool extracts the K-candidate MPC foresight fan and candidate-mean across all methods for a specific `[trial, snapshot]` coordinate, quantifying divergence (`div_ref`) from a baseline. Includes physical violation overlays mapping executed trajectory versus constraints.
2. **Interactive HTML Trajectory Visualizer (`npz_traj_visualizer.html`)**: Built an offline, self-contained HTML visualizer to reconstruct entire evaluation scenes from `.npz` files. Features include:
    - Interactive scrubbable timeline with playback.
    - Full representation of executed paths, receding-horizon MPC fans, mean candidate lines, and violation markers over a 2D environment (avoiding halfspaces/obstacles).
    - Synchronized quantitative analytics panels (drift, explosion, tracking error) linked directly to the scrubber.
3. **Robust Exporter (`npz_traj_export.py`)**: Implemented the backend exporter which safely parses nested/flat `.npz` schemas (avoiding and UAV pipelines), decimated sampling (`plan_every`), and metric pre-computation to inline data securely into the viewer template without a server dependency.
4. **Visualizer UX Hotfixes (Fix 1)**: 
    - **MPC Fan Decoupling**: Resolved a bug where the MPC candidate fan visualization was hidden by default due to coupled selection logic. Candidates and layers (fan/mean) are now completely decoupled.
    - **Path/URL Loader**: Improved the path loader to iteratively fallback through path suffixes when loading absolute filesystem paths, correctly resolving HTTP server roots.
    - **View Controls**: Added explicit Redraw and Recenter view controls for better user navigation.

***

## Gen11 Epoch 9 Fix 15 & Gen7/Gen6V4 C6: Projection Cost-Explosion Wall-Clock Guard (July 11, 2026)

**Keywords**: Gen11, Epoch 9, Fix 15, Gen7, Gen6V4, C6, projection cost explosion, SLSQP, wall-clock guard, bounds_free.

1. **Wall-Clock Deadline Guard**: Investigated multi-hour evaluation stalls (e.g., `bounds_free` variants on non-convex UAV `pillars` geometry or Gen7 `dpcc-r` taking hours for single runs). Identified that the DPCC upstream solver lacked a wall-clock limit, allowing SLSQP to thrash on non-converging QCQP formulations. Implemented a strict per-solve budget (`FMPCC_PROJ_SOLVE_BUDGET_S`, default 2.0s) as a callback within `Projector.project()`.
2. **Graceful Fallback**: If a solve exceeds the budget, an exception (`_SolveBudgetExceeded`) is raised, the sample falls back to the unprojected Flow-Matching trajectory, its projection cost is set to infinity (preventing selection), and a loud `COST EXPLODED` marker is logged.
3. **Cross-Pipeline Sync**: Applied this guard identically across all active projection engines: `flow_matcher_v3_uav` (Gen11), `fm_visual_aligning` (Gen7), and `diffuser_visual_aligning` (Gen6V4).

***

## Gen11 Epoch 9 Fix 15.2 & Gen7/Gen6V4 C6: Sustained-Slowness Circuit Breaker (July 12, 2026)

**Keywords**: Gen11, Epoch 9, Fix 15.2, Gen7, Gen6V4, circuit breaker, sliding window, sustained slowness.

1. **Circuit Breaker Strategy (Superseding Fix 15)**: Discovered that the per-solve 2s hard cap from Fix 15 was punishing legitimate rare spikes and silently corrupting the outputs of pathologically slow variants (falling back to unprojected trajectories 6,000+ times per trial). Replaced the rigid cap with a "sliding-window circuit breaker" that only trips on *sustained* solver slowness.
2. **Generous Backstop & Sliding Window**: Raised the per-solve backstop to 60s to allow healthy but tough solves to finish cleanly. The projector now monitors the last `N` steps (default 40); if a high fraction (e.g. 90%) of recent steps take longer than `SLOW_MS` (default 1000ms), the breaker trips OPEN.
3. **OPEN-State Skips & Half-Open Probing**: Once tripped, subsequent `project()` calls skip SLSQP entirely, saving time on a hopeless episode. A cooldown timer eventually allows a "HALF-OPEN" probe to re-test the solver. If fast, it CLOSES and resumes normal behavior, smoothly transitioning between hard and easy episodes.
4. **Cross-Pipeline Parity**: Identically deployed across the `flow_matcher_v3_uav`, `fm_visual_aligning`, and `diffuser_visual_aligning` projectors, all natively overridable via environment variables.

***

## NPZ Analysis Tool v4: Trajectory Visualizer Fix 2 & UI Refinements (July 12, 2026)

**Keywords**: npz_traj_export, npz_traj_visualizer, fix_2, recording-phase misalignment, offset detection, candidate UI.

1. **Recording-Phase Misalignment Detection**: Addressed an off-by-one visual bug where receding-horizon MPC fans did not connect cleanly to the executed path. Traced the root cause to evaluation scripts appending the `obs_buffer` post-step vs. pre-step, creating a consistent +1 offset in certain environments (e.g. avoiding/iMeanFlow) compared to offset-0 logic (e.g. UAV). 
2. **Data-Driven Offset Correction**: Implemented an automated offset detector (`_recording_offset`) inside the exporter (`npz_traj_export.py`) that aligns the `h=0` root of the MPC candidate fans perfectly with the actual executed trajectory points.
3. **Browser Hot-Patch Script**: Created a stopgap `resnap_plan_steps.js` to realign existing `scene.json` files and embedded HTML viewers without requiring costly cluster re-exports.
4. **Viewer Candidate UI Enhancement**: Refined the visualizer's candidate-selection checkboxes to automatically populate based on the loaded variant's fan size and default to checked, clarifying that "unticking all" hides the layer.
5. **Local Exporter Execution**: Documented that local `/usr/local/bin/python3` can successfully run the exporter pipeline locally without spinning up a SLURM node.

***
