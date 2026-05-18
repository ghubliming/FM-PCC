# FIX_11: Restoring Physical Interaction and Diagnostic Visuals

## Problem Description
The Gen5 Visual-Aligning pipeline suffered from two catastrophic failures after the transplant from the D3IL codebase:
1. **Physical Ghosting**: The robot end-effector passed through the target box without interaction.
2. **Frozen Diagnostics**: The captured GIFs showed a static cage camera view (left) while the in-hand camera (right) worked correctly.

## Investigation & Root Causes

### 1. Physical Ghosting
- **Root Cause**: The `panda_rod_invisible.xml` model had its `rod:tip` collision geometry disabled with `contype="0"` and `conaffinity="0"`.
- **Solution**: Enabled collisions by setting both to `"1"`.

### 2. Frozen Diagnostic GIF
- **Root Cause**: MuJoCo name-mangling vs. Python Camera IDs.
    - The `MjRobot` class injects IDs (like `_rb0_`) into all XML names during initialization to support multi-robot setups.
    - The `BPCageCam` (Cage Camera) was initialized in Python with the hardcoded name `"bp_cam"`.
    - When the XML was generated, the camera body was renamed to something like `bp_cam_rb0_`.
    - The `render()` call in Python requested `"bp_cam"`, which MuJoCo could not find.
    - MuJoCo defaulted to a **Free Camera** at the origin, which remained stationary and looked at a static part of the scene, giving the appearance of a "frozen" frame.
- **Solution**: Dynamically resolve the camera name using the robot's `add_id2model_key()` method so the Python camera object matches the injected name in the XML.

## Changes Implemented

### 1. Robot Model (`panda_rod_invisible.xml`)
- Updated `rod:tip` geom to enable contact processing.

### 2. Aligning Environment (`aligning.py`)
- Modified `BPCageCam` to accept a dynamic name.
- Updated `Robot_Push_Env` to use `robot.add_id2model_key("bp_cam")` for initializing the cage camera.

## Verification Plan
1. **Interaction Check**: Run a single rollout and verify that the box moves when the rod tip touches it.
2. **Visual Check**: Inspect the generated `rollout_0.gif` in the `eval_ddpm_encdec_vision` output directory. Verify that the left image (Cage Cam) shows the robot and box moving in sync with the right image.
