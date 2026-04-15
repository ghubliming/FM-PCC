# Investigation Note: The `batch_size` Parameter in DPCC / FMPCC

This document clarifies the role, default values, and architectural impact of the `batch_size` parameter within the Flow Matching and Diffusion planning pipelines.

## 1. What is the `batch_size`?

In terms of **Inference / Deployment** (the `plan` stage), the `batch_size` does NOT refer to training data blocks. Instead, it represents the **Number of Parallel Candidate Trajectories**.

At every single environment step (the robot's "thought" cycle):
1. The **Policy** initializes $N$ parallel noise tensors (where $N = batch\_size$).
2. The **Flow Matcher / ODE Solver** integrates all $N$ candidates simultaneously on the GPU.
3. The **Selection Logic** evaluates all $N$ trajectories and chooses the "best" one based on the selection strategy (e.g., `temporal_consistency` or `minimum_projection_cost`).
4. The robot executes the first action of the **winning trajectory**.

> [!NOTE]
> Increasing the `batch_size` improves the safety and optimality of the robot (more candidates to choose from) but increases the GPU calculation load.

---

## 2. Default Configuration Logic

The default value for evaluation in the `avoiding-d3il` task is **`4`**.

### Where is it defined?
The primary source of truth is the configuration file:
*   **Location**: [config/avoiding-d3il.py](file:///workspaces/FM-PCC/config/avoiding-d3il.py)
*   **Lines**: 
    - `plan` configuration block (line ~335)
    - `plan_fm` configuration block (line ~368)

```python
    'plan': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,        # <--- The default set point for planning
        ...
    },
```

### How is it passed to the code?
The evaluation script [scripts/eval.py](file:///workspaces/FM-PCC/scripts/eval.py) parses this config and passes it directly to the policy's call method:

```python
# scripts/eval.py: Line 231
action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, ...)
```

The policy then mirrors this batch size to the model and the reward/return tensors in [diffuser/sampling/policies.py](file:///workspaces/FM-PCC/diffuser/sampling/policies.py).

---

## 3. Why is it set to 4?

The choice of `4` is a balance between **Safety (Diversity)** and **Hardware Latency**:
1. **Diversity**: Sampling multiple trajectories allows the "Minimum Projection Cost" logic to choose the safest path among several options.
2. **GPU Efficiency**: Modern GPUs handle batch sizes of 4 or 8 nearly as fast as a batch size of 1 because the overhead is dominated by Python and data transfer (the "Bridge Tax").

> [!IMPORTANT]
> If you set `batch_size` to `1`, you are effectively turning off the "Parallel Selection" feature, forcing the robot to rely on a single greedy rollout without any alternatives to fall back on if that path is unsafe.
