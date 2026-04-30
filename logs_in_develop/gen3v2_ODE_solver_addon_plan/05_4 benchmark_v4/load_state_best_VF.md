# Audit: Model Checkpoint Loading (latest vs. best)

This document audits how different evaluation and benchmarking scripts load model weights. There is a discrepancy between the numerical benchmark scripts and the robotic evaluation pipeline regarding whether they use the "latest" numbered checkpoint or the "best" validation checkpoint.

## Summary Table

| Script | Primary Purpose | Default `epoch` | Source of Default |
| :--- | :--- | :--- | :--- |
| `benchmark_ode_solvers_v4.py` | Numerical solver audit | **`'latest'`** | `argparse` default |
| `traj_gen_script_for_v4.py` | Visualizing benchmarks | **`'latest'`** | **Hardcoded** in `main()` |
| `eval_flow_matching_v3_ode_selectable.py` | Robotic policy eval | **`'best'`** | `plan_fm_v3_ode_selectable` config |

---

## 1. Technical Root Cause: `get_latest_epoch`
In [serialization.py](file:///workspaces/FM-PCC/flow_matcher_v3_ode_selectable/utils/serialization.py), the `get_latest_epoch` function specifically filters for integer suffixes:

```python
def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        try:
            epoch = int(state.replace('state_', '').replace('.pt', ''))
        except ValueError:
            # 'best' is NOT an integer, so it is ignored here
            epoch = -1
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch
```

Because of this, if `epoch='latest'` is passed, the system will find the highest numbered step (e.g., `state_80000.pt`) and **ignore** `state_best.pt`.

---

## 2. Analysis of the 3 Key Scripts

### Case A: `benchmark_ode_solvers_v4.py`
- **Location:** [benchmark_ode_solvers_v4.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v4/benchmark_ode_solvers_v4.py)
- **Loading Logic:**
  ```python
  ap.add_argument("--diffusion-epoch", type=str, default="latest")
  # ...
  fm_exp = utils_serialization.load_diffusion(..., epoch=args.diffusion_epoch, ...)
  ```
- **Finding:** It defaults to the final training step (e.g., 80000).

### Case B: `traj_gen_script_for_v4.py`
- **Location:** [traj_gen_script_for_v4.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v4/traj_gen_script_for_v4.py)
- **Loading Logic:**
  ```python
  fm_exp = utils_serialization.load_diffusion(..., epoch="latest", ...)
  # (See line 54)
  ```
- **Finding:** This is **hardcoded** to load the latest step. This is intentional for benchmarking current training progress, but may lead to discrepancies if comparing against "best" eval results.

### Case C: `eval_flow_matching_v3_ode_selectable.py`
- **Location:** [eval_flow_matching_v3_ode_selectable.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py)
- **Loading Logic:**
  ```python
  args = Parser().parse_args(experiment='plan_fm_v3_ode_selectable', ...)
  # ...
  fm_experiment = load_diffusion_with_override(..., epoch=args.diffusion_epoch, ...)
  ```
- **Config ([avoiding-d3il.py](file:///workspaces/FM-PCC/config/avoiding-d3il.py)):**
  ```python
  'plan_fm_v3_ode_selectable': {
      ...
      'diffusion_epoch': 'best',
  }
  ```
- **Finding:** This script correctly defaults to the **best** model according to the research configuration.

---

## 3. Training Logic for `state_best.pt`

When loading with `--diffusion-epoch best`, you may see a message like:
`[ utils/training ] Restored loss history from checkpoint at step 90000`

This is the correct and expected behavior. Here is why:

1. **Saving Logic:** In [training.py](file:///workspaces/FM-PCC/flow_matcher_v3_ode_selectable/utils/training.py), the `Trainer` performs a test at regular intervals. If the `test_loss` is the lowest found so far, it triggers `save_best()`.
2. **Step Metadata:** The `save_best()` function captures the current `self.step` and stores it inside the `.pt` file.
3. **Loading Verification:** When you load the "best" model, the `Trainer.load()` method reads that step metadata and prints it to the console.

**Conclusion:** If you see "step 90000" while loading the best model, it confirms that the best validation performance occurred at that specific step, and the script is correctly ignoring any later checkpoints (e.g., step 100,000) which might have higher loss.

---

## 4. Recommended Workflow

> [!IMPORTANT]
> To ensure your **Benchmark V4** results match your **Robotic Eval** performance, you should explicitly set the epoch in the benchmark command:
> ```bash
> python .../benchmark_ode_solvers_v4.py --diffusion-epoch best
> ```

> [!WARNING]
> If `state_best.pt` does not exist in the log folder (e.g., if validation was disabled), both scripts will fail if set to `'best'`. In that case, `'latest'` is the only option.
