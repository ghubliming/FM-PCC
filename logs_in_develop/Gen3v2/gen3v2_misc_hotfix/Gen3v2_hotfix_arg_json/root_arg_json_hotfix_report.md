# Hotfix Report: Metadata Leak to Root Directory

## 🔍 The Incident
An audit of the remote environment revealed a massive accumulation of configuration files (e.g., `args_resume_272.json`) residing in the project root directory. These files contain the serialized namespace of experiment arguments but are disconnected from their respective experiment log folders.

## 🛠 Root Cause Analysis
The issue resides in the `Parser` class within `flow_matcher_v3_ode_selectable/utils/setup.py`. 

1.  **Faulty Initialization**: When `Parser()` is instantiated, `self.savepath` defaults to an empty string (`''`).
2.  **Disconnected State**: During `parse_args()`, the `mkdir()` method calculates the correct experiment-specific path and assigns it to `args.savepath`. However, it **fails to update** the `Parser` instance's own `self.savepath` variable.
3.  **The Leak**: When `self.save(args)` is called, it uses the stale `self.savepath` (the empty root string) to determine the write location:
    ```python
    # flow_matcher_v3_ode_selectable/utils/setup.py
    fullpath = os.path.join(self.savepath, 'args.json') # Resolves to './args.json'
    ```
4.  **Global Conflict**: Because every experiment is trying to write to the same `args.json` in the root, the versioning logic (intended for resumes) triggers globally. This leads to the astronomical file index (`_resume_272.json`).

## ⚠️ Impact
*   **Root Pollution**: Critical project root is cluttered with hundreds of junk JSON files.
*   **Traceability Loss**: It is impossible to identify which `args_resume_N.json` belongs to which experiment without manual inspection of the `savepath` string inside the JSON.
*   **Metadata Fragmentation**: The actual experiment folders are missing their local `args.json` backups.

## 🚀 Execution Summary

### 1. Code Fixes Applied
The following files were updated to synchronize the `Parser` instance's `self.savepath` with the experiment-specific `args.savepath`:
*   [flow_matcher_v3_ode_selectable/utils/setup.py](file:///workspaces/FM-PCC/flow_matcher_v3_ode_selectable/utils/setup.py)
*   [(Abandoned)flow_matcher_v3_avoiding_visual/utils/setup.py](file:///workspaces/FM-PCC/(Abandoned)flow_matcher_v3_avoiding_visual/utils/setup.py)

**Change Detail:**
```python
# flow_matcher_v3_ode_selectable/utils/setup.py
def mkdir(self, args):
    if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
        args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name, str(args.seed))
+       self.savepath = args.savepath  # Fix: ensures save() uses the local folder, not root
        self._dict['savepath'] = args.savepath
```

### 2. Verification Results
As per user instructions, formal verification was skipped. However, the logic fix directly addresses the path mismatch identified in the root cause analysis.

### 3. Recommendation for Remote Environment
On the remote environment where `args_resume_272.json` was found, the following manual cleanup is recommended:
1.  **Delete all global config files** in the root directory:
    ```bash
    rm args.json args_resume_*.json
    ```
2.  **Verify local logging**: Run a new experiment and confirm `args.json` is created only within the `logs/` subfolder.

## 🛡️ Safety & Logic Verification

### 1. Test Integrity (Scientific Validity)
The fix is **strictly non-destructive** to experimental results. 
*   **Passive Logging**: The `save()` method is a passive operation that dumps existing parameters to disk. It has no feedback loop into the model or environment.
*   **Independence**: The numerical outputs (trajectories, success rates) were unaffected because they were calculated correctly and saved to the intended trial folders.
*   **Training Integrity**: The `Trainer` class manages model weights and checkpoints independently of the `Parser`'s JSON logging. Even when metadata was leaking to the root, model weights (`.pt` files) were saved and loaded from the correct experiment folders. Resumed training runs remained mathematically intact.
*   **Conclusion**: **Previous results and trained models remain scientifically valid.** The bug was an organizational/clutter issue, not a mathematical one.

### 2. Path Isolation
The "leak" was caused by a stale `self.savepath` variable. By synchronizing this variable with the experiment's specific folder, we have restored path isolation. Metadata will now be private to each experiment.

### 3. Resume Logic Preservation
The versioning logic (`_resume_N.json`) is fully preserved. The only difference is its **scope**:
*   **Old Behavior (Broken)**: Checked for `args.json` in the **Root**. Since *any* experiment's `args.json` existed there, it triggered a false "resume" count globally.
*   **New Behavior (Fixed)**: Checks for `args.json` in the **Experiment Folder**. It will now only create a `_resume` file if *that specific experiment* is being resumed or re-run in the same folder. 

This ensures that the resume history is now accurate and trial-specific.

## 🏁 Final Status: RESOLVED
The metadata leak is plugged. Future runs will correctly encapsulate their configuration logs within their respective experiment directories.
