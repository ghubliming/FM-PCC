# SafeFlowMPC-Style FM in FM-PCC: Strict v2-Only Execution Plan

## Scope Lock (Non-Negotiable)
1. Only work on a copied v2 branch.
2. Do not modify existing `flow_matcher`, `flow_matcher_unet_v2`, `FM_test`, `FM_hp_tune_test`, or any DPCC/projector logic.
3. Implement SafeFlowMPC-style time sampling only in the copied v2 architecture and copied v2 test pipeline.
4. Add exactly two new v2 config parameters in `config/avoiding-d3il.py`.

---

## Objective
Create a new v2 branch that uses SafeFlowMPC-style FM training-time sampling:
- sample `u ~ Beta(alpha, beta)`
- set `t = 1 - u`
- use `t` in the v2 FM interpolation/loss path

No other feature expansion, no sampler framework, no extra modes.

---

## Folder Workflow (As Requested)

## Step 1: Copy and Rename Folders
1. Copy `flow_matcher_unet_v2` to `flow_matcher_v2`.
2. Copy `FM_Unet_v2_test` to `FM_v2_test`.

Naming rule:
1. use only `v2` naming
2. no extra suffixes/prefixes

## Step 2: Rewire Copied Test Scripts
Update imports/config references in copied test/train/eval scripts so they point to copied architecture folder only.

Constraint:
No edits in original v2 test folder.

---

## Architecture Change (Only in Copied v2)

## Step 3: Implement SafeFlowMPC-Style Time Sampling
File to edit in copied architecture:
1. `flow_matcher_v2/models/diffusion.py`

Required behavior in loss path:
1. remove uniform `torch.randint(...)` timestep sampling in training loss
2. create `Beta(alpha, beta)` distribution
3. sample per-batch time values
4. apply flip `t = 1 - t`
5. feed this `t` into interpolation and model time conditioning in copied v2 path

Constraint:
No extra sampler options or generic abstraction layer.

---

## Config Change (Exactly 2 New Parameters)

## Step 4: Add 2 New v2 Parameters in `avoiding-d3il.py`
Add only these two keys in v2 config entries used by copied v2 branch:
1. `time_beta_alpha_v2`
2. `time_beta_beta_v2`

Behavior policy:
1. `flip` (`t = 1 - t`) is fixed in code
2. no third/fourth sampling parameter
3. no global sampler mode parameter

---

## Wiring in Copied v2 Train Path

## Step 5: Pass the 2 Parameters into Copied Diffusion Config
In copied v2 training script:
1. read `time_beta_alpha_v2` and `time_beta_beta_v2` from args/config
2. pass them into copied diffusion constructor

In copied diffusion constructor:
1. store both values
2. build Beta distribution on active device during loss computation

---

## File-Level Change List (Planned)
1. Create folder `flow_matcher_v2` (copy of `flow_matcher_unet_v2`).
2. Create folder `FM_v2_test` (copy of `FM_Unet_v2_test`).
3. Edit `flow_matcher_v2/models/diffusion.py` for Beta sampling logic.
4. Edit copied scripts in `FM_v2_test` to use `flow_matcher_v2` only.
5. Edit `config/avoiding-d3il.py` to add exactly two v2 parameters and copied-v2 experiment entries.

---

## Acceptance Criteria
1. Original folders still run unchanged.
2. Copied v2 train script runs and imports only `flow_matcher_v2`.
3. Copied v2 loss path uses Beta sampling + fixed flip (`t = 1 - t`).
4. Exactly two new v2 parameters are added in `avoiding-d3il.py`.
5. No additional architecture/training/eval feature changes are introduced.

---

## Execution Order
1. Copy folders.
2. Rename and rewire copied package/test imports.
3. Implement Beta sampling in copied v2 diffusion.
4. Add 2 new v2 config parameters and wire them through copied v2 train config.
5. Run quick import/config sanity check for copied v2 script.

Strict edit boundary:
1. Allowed edit locations are only `flow_matcher_v2`, `FM_v2_test`, and `config/avoiding-d3il.py`.
2. No edits anywhere else.

---

## Out of Scope (Dropped)
1. Any modification to original v1/v2 architecture folders.
2. Any change to DPCC selection logic.
3. Any new sampler framework, mode switching, or extra config beyond the 2 requested parameters.
4. Any broad experiment matrix, wandb redesign, metadata redesign, or projector changes.
