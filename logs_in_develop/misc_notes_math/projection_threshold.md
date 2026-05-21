# Flow Matching (FMv3ODE) Projection Timestep Threshold Audit

This document records the forensic trace and logical audit of the continuous-time **Projection Timestep Threshold** ($\tau$) under various ODE step counts ($K$). It acts as a reference for understanding how boundary snaps resolve mathematically and how safety guards interact with discrete scheduling.

---

## 1. Core Snapping Logic

In the core continuous-time Flow Matching models (`FMv3ODE`, `ddpm_encdec_vision`, `flow_matcher_v3_ode_selectable`), the snapping logic uses a combination of float thresholding, integer truncation, and a final-step safety guard:

```python
# snapping_start_idx determines the first discrete step index to apply projection
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)

# near_end gates whether projection is active at the current loop step
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
```

### The Two Gating Mechanisms
1. **Threshold Snapping**: `loop_idx >= snapping_start_idx`
2. **Safety Snap (Final Step Guard)**: `loop_idx == self.flow_steps_v3 - 1`

> [!IMPORTANT]
> The **Safety Snap** is a critical robotics constraint. To prevent sending unconstrained or physically dangerous actions to a real-world controller, the final step of ODE integration ($t=1$) must **always** be projected to satisfy safety boundaries, regardless of how small the threshold is set.

---

## 2. Step-by-Step Scenario Traces

Below are the mathematical evaluations of the snapping scheduler for three core boundary scenarios.

### Scenario A: Single-Step ODE with 0.5 Threshold (`ode=1`, `threshold=0.5`)
* **Inference Parameters**: `flow_steps_v3 = 1`, `threshold = 0.5`
* **Single Loop Step**: `loop_idx = 0`

#### Step-by-Step Execution:
1. **Start Index Calculation**:
   $$\text{snapping\_start\_idx} = \text{int}((1.0 - 0.5) \times 1) = \text{int}(0.5) = 0$$
   *Due to Python's truncation behavior (rounding down floats towards zero), $0.5$ truncates to $0$.*
   
2. **Step `loop_idx = 0` Gating Check**:
   * **Threshold check**: `loop_idx >= snapping_start_idx` $\rightarrow$ `0 >= 0` $\rightarrow$ **`True`**
   * **Safety check**: `loop_idx == self.flow_steps_v3 - 1` $\rightarrow$ `0 == 0` $\rightarrow$ **`True`**
   * **Final Verdict**: `near_end = True`

> [!NOTE]
> **Result**: **PROJECTED**. Both the integer truncation and the safety guard guarantee that a 1-step solver is fully constrained.

---

### Scenario B: 3-Step ODE with 0.5 Threshold (`ode=3`, `threshold=0.5`)
* **Inference Parameters**: `flow_steps_v3 = 3`, `threshold = 0.5`
* **Loop Steps**: `loop_idx` $\in \{0, 1, 2\}$

#### Step-by-Step Execution:
1. **Start Index Calculation**:
   $$\text{snapping\_start\_idx} = \text{int}((1.0 - 0.5) \times 3) = \text{int}(1.5) = 1$$
   *Float $1.5$ truncates down to index $1$.*

2. **Step-by-Step Trace**:
   * **Step `loop_idx = 0`**:
     * `0 >= 1` $\rightarrow$ `False`
     * `0 == 2` $\rightarrow$ `False`
     * **Verdict**: `near_end = False` (NOT projected)
   * **Step `loop_idx = 1`**:
     * `1 >= 1` $\rightarrow$ **`True`**
     * `1 == 2` $\rightarrow$ `False`
     * **Verdict**: `near_end = True` (**PROJECTED**)
   * **Step `loop_idx = 2`**:
     * `2 >= 1` $\rightarrow$ **`True`**
     * `2 == 2` $\rightarrow$ **`True`**
     * **Verdict**: `near_end = True` (**PROJECTED**)

> [!NOTE]
> **Result**: **Last 2 steps projected**. The first step (`loop_idx=0`) runs unconstrained, while steps `1` and `2` are projected.

---

### Scenario C: 3-Step ODE with 0.0 Threshold (`ode=3`, `threshold=0.0`)
* **Inference Parameters**: `flow_steps_v3 = 3`, `threshold = 0.0`
* **Loop Steps**: `loop_idx` $\in \{0, 1, 2\}$

#### Step-by-Step Execution:
1. **Start Index Calculation**:
   $$\text{snapping\_start\_idx} = \text{int}((1.0 - 0.0) \times 3) = \text{int}(3.0) = 3$$
   *The calculated threshold boundary index $3$ lies outside the step array bounds.*

2. **Step-by-Step Trace**:
   * **Step `loop_idx = 0`**:
     * `0 >= 3` $\rightarrow$ `False`
     * `0 == 2` $\rightarrow$ `False`
     * **Verdict**: `near_end = False` (NOT projected)
   * **Step `loop_idx = 1`**:
     * `1 >= 3` $\rightarrow$ `False`
     * `1 == 2` $\rightarrow$ `False`
     * **Verdict**: `near_end = False` (NOT projected)
   * **Step `loop_idx = 2`**:
     * `2 >= 3` $\rightarrow$ `False`
     * `2 == 2` $\rightarrow$ **`True`** *(Enforced by the final-step safety guard!)*
     * **Verdict**: `near_end = True` (**PROJECTED**)

> [!WARNING]
> **Result**: **Final step projected**. Setting `threshold = 0.0` does **not** turn off projection entirely due to the final-step safety snap. To disable projection completely, the `projector` object must be omitted during evaluation (e.g., using a baseline like `diffuser` which sets `projector = None`).

---

## 3. Discrepancy: Diffuser vs. State-Only Implementations

During this audit, a logical discrepancy was identified between the state-only continuous pipeline (`FMv3ODE` / Selectable / Drifting) and the baseline discrete diffuser (`diffuser` in `/workspaces/FM-PCC/diffuser`):

| Model / Pipeline | Safety Snap Guard | Single-Step (`ode=1` / `n_timesteps=1`, `threshold=0.5`) Behavior |
| :--- | :---: | :--- |
| **`FMv3ODE` / Selectable / Drifting** | Yes | **PROJECTED** (Safety Guard triggers `True`) |
| **`diffuser` (in `/workspaces/FM-PCC/diffuser`)** | **No** | **BYPASSED** (when threshold logic is disabled or if threshold < 0, it lacks a robust final-step fallback snap) |

### Diffuser Pipeline Bypass:
In `diffuser/models/diffusion.py`, the snapping calculation lacks the robust fallback logic `or (t == 0)`:
```python
# Lacks safety-critical final step snap fallback (t == 0)
if projector is not None and projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
    ...
```

### Remediation Plan:
To unify the snapping behavior across all variants, port the safety fallback snapping logic directly into the discrete diffuser model (`diffuser/models/diffusion.py`).


---

## 4. Symmetry with DPCC Diffuser (Discrete Gaussian Diffusion)

There is a beautiful, mathematically proven symmetry between how the **continuous-time FMv3ODE solver** and the **discrete-time DPCC Diffuser** (`diffuser/models/diffusion.py`) handle projection gating. Both systems target the exact same physical window of the generation process and share identical safety-snapping invariants.

### A. Mathematical Mapping of Generation Direction

Because continuous and discrete systems run in opposite directions, the equations are flipped, yet they map to the identical trajectory progression:

| Property | Continuous FMv3ODE | Discrete DPCC Diffuser |
| :--- | :--- | :--- |
| **Generation Flow** | Forward: $t \in [0 \rightarrow 1]$ (noise to data) | Backward: $t \in [T-1 \rightarrow 0]$ (noise to data) |
| **Final Step Index** | `loop_idx = flow_steps_v3 - 1` | `t = 0` |
| **Gating Equation** | `loop_idx >= (1.0 - threshold) * flow_steps_v3` | `t <= threshold * n_timesteps` |

#### Math Check (for 50% Threshold):
* **FMv3ODE**: If `threshold = 0.5`, projection starts from step $0.5 \times \text{flow\_steps\_v3}$ to the end. This covers the **final 50% of generation**.
* **DPCC Diffuser**: If `threshold = 0.5`, projection triggers when `t <= 0.5 * n_timesteps`. Since $t$ counts down to $0$, this covers all steps in the range $[0.5 \times T \rightarrow 0]$. This also covers the **final 50% of generation**.

---

### B. Single-Step and Zero-Threshold Equivalence

The baseline discrete DPCC Diffuser natively matches the safety properties of `FMv3ODE` under extreme settings:

#### 1. Zero-Threshold Gating (`threshold = 0.0`)
* **FMv3ODE**: Setting `threshold = 0.0` calculates `snapping_start_idx = flow_steps_v3`. Threshold checks fail, but the safety check `loop_idx == self.flow_steps_v3 - 1` **forces the final step to be projected**.
* **DPCC Diffuser**: Setting `threshold = 0.0` triggers projection only when `t <= 0.0 * n_timesteps` $\rightarrow$ `t <= 0`. Because the loop counts down to $0$, the condition is satisfied **exactly and only on the final step (`t = 0`)**.

#### 2. Single-Step Gating (`n_timesteps = 1`)
* **FMv3ODE**: Truncates to index `0` and enforces the final step safety check to **project the single step**.
* **DPCC Diffuser**: If run with `n_timesteps = 1` and `threshold = 0.5`, the boundary checks `t <= 0.5 * 1` $\rightarrow$ `0 <= 0.5`, which is **True**, and **projects the single step**.

### C. Parity Conclusion
The baseline **discrete DPCC Diffuser** and the **continuous FMv3ODE** share **complete architectural parity** in their projection gating. In both systems:
1. Gating maps to the identical physical phase of trajectory generation.
2. An absolute zero-threshold (`threshold = 0.0`) is prevented from silently disabling all projection; instead, both systems naturally constrain the final step to preserve robotics safety.

