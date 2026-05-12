# FM-D Engine Fix 1: Training Logic & Array Serialization

**Date**: 2026-05-12  
**Status**: ✅ Implemented  

This document tracks the technical fixes required to activate the "Drifting" capability in the `flow_matcher_v3_drifting` engine. Although the foundation was established in Gen3v3, several critical "wiring" bugs prevented the engine from actually utilizing the drift loss during training.

---

## 1. Disconnected Drifting Trainer

**The Objective**:  
Integrate the `DriftTrainingWrapper` into the main `Trainer` epoch loop to enable hybrid FM + Drift loss optimization.

**The Issue**:  
The `flow_matcher_v3_drifting/utils/training.py` module was a literal copy of the non-drifting version. It inherited the standard `Trainer` but never instantiated the drift loss scheduler or applied the drift loss gradients during `loss.backward()`.

**The Fix**:  
Updated the `Trainer` to:
1.  Check for `use_drift_augmentation` in the configuration.
2.  Instantiate `DriftLoss` and `DriftTrainingWrapper` if enabled.
3.  Intercept the standard FM loss and combine it with the drift loss before the optimizer step.
4.  Update the circular memory bank with expert trajectories from each training batch.

---

## 2. Hardcoded Batch Serialization (NamedTuple Bug)

**The Objective**:  
Allow the `Trainer` to push training batches to the GPU regardless of their container type (`list`, `tuple`, `dict`, or `namedtuple`).

**The Issue**:  
The `batch_to_device` utility in `utils/arrays.py` was hardcoded to search for a `._fields` attribute, which only exists on `namedtuple` objects. When using standard PyTorch datasets that return lists or tuples, the training loop would crash with an `AttributeError`.

**The Fix**:  
Refactored `batch_to_device` into a polymorphic router that recursively handles all standard Python containers.

---

## 3. Training Entry Point Re-wiring

**The Objective**:  
Ensure the training scripts in `FM_v3_drifting_test/` actually target the drifting engine and configuration.

**The Issue**:  
Scripts were still importing from the `ode_selectable` package and using the non-drifting experiment keys.

**The Fix**:  
Renamed and updated scripts to use `flow_matcher_v3_drifting` and the `flow_matching_v3_drifting` config block.
