# UF14: Investigation on Constraint and Obstacle Loading

**Date**: 2026-05-26
**Target**: `config/projection_eval.yaml` and `scripts/eval.py`

## 1. Overview
The goal of this investigation was to determine how `avoiding_halfspace_variants` defined in the configuration file (`projection_eval.yaml`) actually maps to the obstacles and halfspaces loaded during evaluation (`scripts/eval.py`). We also identified a potential conflict regarding the `bounds` configurations.

## 2. Findings on Halfspace & Obstacle Loading Logic
In `projection_eval.yaml`, multiple constraints are defined in lists under `halfspace_constraints` and `obstacle_constraints` for the `avoiding-d3il` environment.

The evaluation script (`scripts/eval.py`) iterates through each item in the `avoiding_halfspace_variants` array (`'top-right-hard'`, `'top-left-hard'`, `'both-hard'`). Inside the execution loop, **the script maps these variants to hard-coded integer indexes** in the YAML lists.

### The Exact Code Logic in `scripts/eval.py` (Lines 55-65)

Here is the exact code showing how the variants are mapped to integer indexes:
```python
    for halfspace_variant in halfspace_variants:
        robot_name = exp.split('-')[0]
        if halfspace_variant == 'top-left-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][0]]
            obstacle_constraints = [config['obstacle_constraints'][exp][3]]
        elif halfspace_variant == 'top-right-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][1]]
            obstacle_constraints = [config['obstacle_constraints'][exp][4]]
        elif halfspace_variant == 'both-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][2], config['halfspace_constraints'][exp][3]]
            obstacle_constraints = [config['obstacle_constraints'][exp][5]]
```

### The Source Lists in `projection_eval.yaml`

To see what the `/` and `\` shapes actually are, we look at the corresponding lists in `projection_eval.yaml`. **Important:** The commented-out lines do *not* count towards the index!

```yaml
halfspace_constraints: {
  'avoiding-d3il': [
    # [[0.8, -0.3], [0.3, 0.5], 'below'],   # hard (\ shape) -- COMMENTED OUT
    # [[0.2, -0.3], [0.7, 0.5], 'below'],     # hard (/ shape) -- COMMENTED OUT
    
    # --- The active array indices begin here ---
    [[0.8, -0.5], [0.4, 0.5], 'below'],   # Index 0: hard (\ shape)
    [[0.2, -0.5], [0.6, 0.5], 'below'],   # Index 1: hard (/ shape)
    [[0.8, -0.3], [0.575, 0.5], 'below'], # Index 2: easier (\ shape)
    [[0.2, -0.3], [0.425, 0.5], 'below'], # Index 3: easier (/ shape)
    ],
}
```

### How They Combine

When we combine the logic from Python and the arrays from YAML, here is the exact result of the 3 iterations:

1. **When `halfspace_variant == 'top-left-hard'`:**
    * **Halfspace loaded:** Index `[0]` → `[[0.8, -0.5], [0.4, 0.5], 'below']` *(The hard `\` shape)*
    * **Obstacle loaded:** Index `[3]` → `center: [0.5, -0.1], radius: 0.06`

2. **When `halfspace_variant == 'top-right-hard'`:**
    * **Halfspace loaded:** Index `[1]` → `[[0.2, -0.5], [0.6, 0.5], 'below']` *(The hard `/` shape)*
    * **Obstacle loaded:** Index `[4]` → `center: [0.4, 0.08], radius: 0.08`

3. **When `halfspace_variant == 'both-hard'`:**
    * **Halfspace loaded:** Indexes `[2]` and `[3]` → `[[0.8, -0.3], [0.575, 0.5], 'below']` and `[[0.2, -0.3], [0.425, 0.5], 'below']` *(Notice these are actually the "easier" `\` and `/` shapes despite the "both-hard" name!)*
    * **Obstacle loaded:** Index `[5]` → `center: [0.6, 0.08], radius: 0.08`

### Critical Danger: Brittle Indexing
Because `eval.py` uses explicit array indices (`[0]`, `[3]`, `[4]`, etc.), **the order of the items in the `projection_eval.yaml` is strictly critical.** 
If a developer comments out one item (for example, commenting out the first halfspace to "disable" it), all subsequent items shift up by one index. This will cause `eval.py` to accidentally load completely mismatched obstacles and constraints, silently corrupting the evaluation scenario.

## 3. Findings on Bounds Redundancy
While examining the YAML file, a conflict was found in the `bounds` definition for `avoiding-d3il`:

```yaml
bounds: { 
  'avoiding-d3il': [
    {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.01, 0]},
    {'type': 'upper', 'dimensions': ['vx', 'vy'], 'values': [0.01, 0.01]},
    {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.012, 0]},
    {'type': 'upper', 'dimensions': ['vx', 'vy'], 'values': [0.012, 0.012]},
  ],
}
```

**Important Note on Bounds Loading:**
Unlike `halfspace_constraints` and `obstacle_constraints`, which are conditionally loaded using `if/elif` logic based on the specific variant (`top-left-hard`, etc.), **the bounds are loaded unconditionally for ALL 3 variants.**

In `scripts/eval.py`, the code fetches `bounds = config['bounds'][exp]` outside of the variant-specific `if/elif` blocks, and injects them into the global constraint list.

**The Redundancy Issue:**
Because multiple `lower` and `upper` bounds for the exact same dimensions (`'vx'`, `'vy'`) are defined sequentially, this creates a major issue. When parsed, this results in either overriding the previous constraints (meaning the first ones are ignored) or feeding conflicting, redundant constraints to the projection solver. **This redundancy is silently fed into the solver for every single one of the 3 variants**, which may cause unintended behavior.

## 4. Recommendations
1. **Refactor Indexing Logic:** Update `scripts/eval.py` to use dictionaries (key-value mapping) for variants rather than relying on list integer indices. This allows configuration items to be safely added, removed, or commented out without breaking the script.
2. **Fix `bounds` Array:** Remove the duplicate lower/upper bound declarations in `projection_eval.yaml` so that only a single, explicit domain limit exists for `vx` and `vy`.
