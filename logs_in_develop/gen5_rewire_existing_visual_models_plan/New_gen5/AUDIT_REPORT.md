# Gen5 Visual Pipeline Audit Report

**Date**: May 13, 2026
**Auditor**: Agent Review
**Scope**: Gen5 visual rewire effort (DDPM visual baseline wired into FM-PCC, plan to replace core with FMv3ODE)

---

## Executive Summary

The Gen5 visual pipeline (rewire of D3IL visual aligning into FM-PCC) is implemented as a real, runnable baseline: a DDPM-based visual engine has been wired into FM-PCC as `ddpm_encdec_vision` with train and eval entrypoints. The rewire approach (bridge the visual encoder & dataset, then swap generative core) is architecturally sound — the first-principle idea to reuse the D3IL visual encoder and dataset inside FM-PCC and then replace the generative core with FMv3ODE is valid.

Observed issues (early training convergence, eval failures) stem from engineering integration gaps (Hydra instantiation, device serialization, PYTHONPATH, DataLoader/CUDA fork, missing action bounds) rather than a fundamental flaw in the approach. Those gaps have documented fixes applied in Phase 1.

Verdict: First-principle is OK — Gen5 baseline is real and suitable as a controlled testbed for comparing DDPM-visual vs FMv3ODE-visual.

---

## Evidence & Findings

- Code presence: `ddpm_encdec_vision/` and `ddpm_encdec_vision_test/` exist with `VisualDiffusionBridge`, `train_ddpm_encdec_vision.py`, and `eval_ddpm_encdec_vision.py` implemented.
- Integration work: A bridging module wraps D3IL's `MultiImageObsEncoder` + `DiffusionEncDec` so the visual stack can be used with FM-PCC Trainer.
- Known runtime failures and fixes (all documented):
  - Hydra recursive instantiation and non-serializable `torch.device` objects — fixed by using `_recursive_: False` and string-casting devices.
  - DataLoader CUDA fork crash — fixed by keeping dataset on CPU and moving batches to GPU in the trainer.
  - PYTHONPATH/import errors for D3IL env modules — fixed by adding `d3il/environments/d3il` to path.
  - Missing diffusion action bounds — fixed by initializing `min_action`/`max_action` in the bridge.

- Training curve note: early apparent convergence (first ~5%) was observed. This can be caused by:
  - Mismatched loss/reporting (e.g., tracking an easy-to-fit proxy such as an encoder-only loss)
  - Incorrect normalization/scaler or trivial dataset subset during debug runs
  - Very small effective learning horizon or immediate overfitting when using tiny batches

These are engineering/training-setup hypotheses (not proof of a principled failure).

---

## PCC Bone Consistency Check

### Training Infrastructure (`train_ddpm_encdec_vision.py`)
- **Status**: **Partially Consistent (Scaffolding only)**
- **Findings**:
  - **Match**: Scaffolding (W&B, Seed Management, Manifesting, Checkpointing) is perfectly replicated from `FMv3ODE`.
  - **Gap**: The **Config Modularity** is broken. While `FMv3ODE` uses a multi-stage `utils.Config` setup for Dataset/Model/Diffusion, the vision script uses a monolithic `VisualDiffusionBridge`. This prevents the "PCC Bone" feature of swapping ML engines (e.g., swapping DDPM for FMv3) via command-line arguments without modifying code.

### Evaluation Infrastructure (`eval_ddpm_encdec_vision.py`)
- **Status**: **Inconsistent (Legacy/Standalone)**
- **Findings**:
  - **Major Gap**: The vision eval is "weird" because it is a standalone script that lacks almost all FMv3ODE "PCC Bone" features:
    - No `load_diffusion_with_override` (manual checkpoint loading).
    - No `aggregate_only` mode for result processing.
    - No `Policy` or `Projector` abstraction (uses a custom `VisualAgentWrapper`).
    - No `Tee` logging or standardized result directory nesting.
  - **Verdict**: The evaluation pipeline is significantly lagging behind the FMv3ODE standard and needs to be refactored to use the unified `sampling.Policy` and `load_diffusion` patterns.


---

## First-Principle Assessment

1. Data/encoder separation: Solid. `Aligning_Img_Dataset` and `MultiImageObsEncoder` can be reused without changing generative core.
2. Generative-core swapability: Sound. FMv3ODE is a different inductive bias (flow/ODE vs DDPM), but system-level interfaces (conditioning vector → generative core → action sequence) are compatible with a thin adapter layer.
3. Feasibility: High — replacing DDPM with FMv3ODE is a controlled experiment, not an architectural rewrite.

Conclusion: The architectural principle — take the working visual pipeline, run the DDPM baseline inside FM-PCC, then swap to FMv3ODE — is valid.

---

## Recommendations (next minimal verification steps)

- Run the DDPM-visual baseline end-to-end in FM-PCC with full dataset and standard scaler to confirm stable training curves.
- Log raw losses and per-component metrics (encoder loss vs denoiser loss) to confirm the early convergence source.
- Once baseline is stable, swap the generative core to FMv3ODE (keep encoder/dataset unchanged) and run identical training to compare behavior.
- Collect evaluation metrics (success rate) from `Aligning_Sim` for both runs.

---

## Location
Report saved at: `logs_in_develop/gen5_rewire_existing_visual_models_plan/New_gen5/AUDIT_REPORT.md`


**End of report**
