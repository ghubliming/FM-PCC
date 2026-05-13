# Visual Aligning Evaluation Fix 2

This document records the fix applied to the visual aligning evaluation pipeline after the SLURM job failed at runtime with an import error inside the simulator stack.

## Failure Summary

The evaluation job reached the simulator initialization step, then aborted with:

```text
Traceback (most recent call last):
  File "/data/home/llim/FMPCC/FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py", line 11, in <module>
    from d3il.simulation.aligning_sim import Aligning_Sim
  File "/u/home/llim/FMPCC/FM-PCC/d3il/simulation/aligning_sim.py", line 6, in <module>
    from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env
ModuleNotFoundError: No module named 'envs'
```

The shell warning about `libtinfo.so.6` was unrelated. The real failure was a Python module path problem.

## Root Cause

`d3il/simulation/aligning_sim.py` imports the aligning environment with an absolute package path:

```python
from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env
```

That import only works when the D3IL environment package root is present on `PYTHONPATH` or `sys.path`. The visual-aligning evaluation job originally only exposed:

- `FM-PCC`
- `FM-PCC/d3il`

That was not enough for `envs...` resolution, because the missing package root is:

- `FM-PCC/d3il/environments/d3il`

## Fix Applied

The evaluation entrypoint and the SLURM scripts were updated to add the D3IL environment root explicitly.

### 1. Python entrypoint fix

File: `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`

```diff
 sys.path.append(os.path.abspath('d3il'))
+sys.path.append(os.path.abspath('d3il/environments/d3il'))
 from d3il.simulation.aligning_sim import Aligning_Sim
```

This makes the simulator import work even if the job is launched from a different working directory.

### 2. SLURM evaluation fix

File: `Slurm_Codes/sbatch/Visual_Aligning/eval_visual_aligning.sh`

```diff
 export FMPCC="$REPO"
 export D3IL_ROOT="$FMPCC/d3il"
-export PYTHONPATH="$FMPCC:$D3IL_ROOT:$PYTHONPATH"
+export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
+export PYTHONPATH="$FMPCC:$D3IL_ROOT:$D3IL_ENV_ROOT:$PYTHONPATH"
```

### 3. SLURM training fix

File: `Slurm_Codes/sbatch/Visual_Aligning/train_visual_aligning.sh`

```diff
 export FMPCC="$REPO"
 export D3IL_ROOT="$FMPCC/d3il"
-export PYTHONPATH="$FMPCC:$D3IL_ROOT:$PYTHONPATH"
+export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
+export PYTHONPATH="$FMPCC:$D3IL_ROOT:$D3IL_ENV_ROOT:$PYTHONPATH"
```

The same fix was applied to training because it uses the same simulator stack and would fail later for the same reason.

## Result

After the update, `Aligning_Sim` can import `Robot_Push_Env` successfully, and the evaluation job can proceed past simulator initialization.

## Notes

- The `libtinfo.so.6` warning is environmental and does not block the Python fix.
- The underlying issue was not the model checkpoint or the eval loop itself; it was the missing D3IL environment import root.