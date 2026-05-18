# Diagnostic Extension: Automated Expert Reference (Fix 9)

## Overview
As part of the Fix 9 diagnostic upgrade, the **Expert Reference Generator** has been directly embedded into the main evaluation pipeline. This ensures that every evaluation run automatically produces a "Gold Standard" benchmark for visual comparison.

## Integration Details
The capability is now part of: `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`

## How it Works
1.  **Automatic Trigger:** When you run the evaluation, the script calls `generate_expert_reference()` before starting the model rollouts.
2.  **Dataset Playback:** It selects the first 3 contexts from the test set and replays the ground-truth expert trajectories.
3.  **Video Capture:** It records the dual-camera view (Cage + In-hand) to match the diagnostic format of the model rollouts.
4.  **Synchronized Storage:** The videos are saved in the same results directory under an `expert_references/` sub-folder.

## Output Path
`logs/aligning-d3il-visual/plans/ddpm_encdec_vision/H8/<seed>/results/expert_references/`
- `expert_rollout_0.mp4`
- `expert_rollout_1.mp4`
- `expert_rollout_2.mp4`

## Value for Thesis Analysis
This automation guarantees that every experiment you run comes with a "built-in" expert comparison. You can instantly see if your model's failure in `rollout_0.mp4` is due to a divergent strategy compared to `expert_rollout_0.mp4`.

---

**Documentation updated for FM-PCC Diagnostic Phase 9.**
