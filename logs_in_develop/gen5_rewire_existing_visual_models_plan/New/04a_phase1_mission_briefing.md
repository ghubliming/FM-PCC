# Phase 1 Mission Briefing: DDPM-Visual Baseline Integration

## Objective
Establish a robust control baseline by wiring the D3IL visual aligning pipeline into the FM-PCC framework using a "copy-modify" strategy. This ensures that the visual backbone runs cleanly in our framework before we swap the generative core to FMv3-ODE in Phase 2.

## Architectural Changes Made

1. **Engine Clone & Isolation**
   - **Action:** Copied `flow_matcher_v3_ode_selectable/` to `ddpm_encdec_vision/`.
   - **Rationale:** Creating an isolated engine space ensures no original D3IL files or other FMv3 components are polluted.

2. **The Bridging Module**
   - **File:** `ddpm_encdec_vision/models/d3il_visual_bridge.py`
   - **Action:** Created `VisualDiffusionBridge` class.
   - **Rationale:** Acts as the single point of integration. It wraps `MultiImageObsEncoder` (2× ResNet18 -> 128-dim) and `DiffusionEncDec` (Transformer), passing the exact API footprint required by FM-PCC's Trainer while keeping D3IL logic intact.
   - **Modification:** Updated the `loss()` signature to natively accept `(bp_imgs, inhand_imgs, obs, act, mask)` unpacked from the dataloader.

3. **Engine Internal Modfications**
   - **Files:** `ddpm_encdec_vision/models/__init__.py`, `ddpm_encdec_vision/__init__.py`
   - **Action:** Exposed the new bridging module and stripped references to the previous flow matching components.

4. **Configuration Mapping**
   - **File:** `config/aligning-d3il-visual.py`
   - **Action:** Inserted new `ddpm_encdec_vision` (train) and `plan_ddpm_encdec_vision` (evaluation) config blocks.
   - **Rationale:** Binds the system variables (e.g., `device`, `horizon`, `learning_rate`) natively to the new model bridge.

5. **Test/Entrypoint Scripts (Train)**
   - **File:** `ddpm_encdec_vision_test/train_ddpm_encdec_vision.py`
   - **Action:** 
     - Instantiated `Aligning_Img_Dataset` manually bypassing FM-PCC's sequence loaders.
     - Hooked `VisualDiffusionBridge` directly into the standard `utils.Trainer`.
     - Ensures the loop expects 5 outputs per batch matching D3IL visual tensors.

6. **Test/Entrypoint Scripts (Evaluation)**
   - **File:** `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`
   - **Action:** Completely rewritten.
   - **Rationale:** FM-PCC's standard projection geometric logic doesn't apply to pure visual inputs. We instead built a thin `VisualAgentWrapper` to provide a `predict(obs)` method and piped it directly into D3IL’s native `Aligning_Sim.test_agent()`. This instantly retrieves D3IL's metrics (Success Rate, Entropy, Mean Distance).

## Current Status
- Code modification for Phase 1 is **100% Complete**.
- The codebase is ready for initial test runs. No verifications have been triggered yet.
