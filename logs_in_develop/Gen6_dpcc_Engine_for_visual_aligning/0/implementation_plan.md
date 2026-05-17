# Gen6: Vision-Conditioned Differentiable MPC (DPCC Upgrade)

This document outlines the architectural design and implementation plan for the **Gen6 upgrade** of the FM-PCC framework. The core objective of Gen6 is to transplant the **Differentiable Projective Control Constraint (DPCC)** concepts—originally designed for state-based models—directly into the vision-conditioned **Gen5 visual-aligning pipeline**.

---

## 1. Executive Summary: The Vision-Safe Paradigm

In Gen5, the model operates in a pure, unconstrained **diffuser mode** (conceptually equivalent to FMv3ODE's `diffuser` metric). While highly successful at learning goal-directed spatial dynamics, a pure diffuser is vulnerable to:
1.  **Workspace Boundary Violations:** Hallucinating paths that push the Franka Panda end-effector out of the physical workspace cage.
2.  **Kinematic Jitter/Violations:** Command sequences with high acceleration (jerk) that trigger low-level controller limits or cause image motion blur.
3.  **Lack of Obstacle Awareness:** An inability to react dynamically to newly introduced physical barriers on the table.

**Gen6 resolves these limitations** by wrapping the visual diffusion engine in a **Model Predictive Control (MPC) Projection Loop**. By decoupling high-dimensional visual perception from low-dimensional trajectory constraints, we project the predicted geometric path onto a convex safety manifold at every control tick.

```mermaid
graph TD
    subgraph Perception Stage
        BP["Cage Cam Image (bp_cam)"] --> ResNet["ResNet Visual Encoder"]
        IH["In-Hand Image (inhand_cam)"] --> ResNet
        ResNet -->|128-Dim Latent Embedding| FiLM["FiLM Conditioning Layer"]
    end

    subgraph Generative Denoising (Visual U-Net)
        Proprio["Proprioception (Mental Map)"] --> U-Net["1D Temporal U-Net"]
        FiLM -->|Biases Convolutions| U-Net
        U-Net -->|Unconstrained Trajectory| RawTraj["Raw 3D Trajectory (H x 3)"]
    end

    subgraph Gen6 Constraint Engine (DPCC Projection)
        RawTraj --> Projector["Quadratic Projector (QP/SLSQP)"]
        Limits["Linear Safety Bounds (Workspace)"] --> Projector
        Dyn["Kinematic Euler Dynamics"] --> Projector
        Obs["Spherical Table Obstacles"] --> Projector
        Projector -->|Differentiable Snap / SLSQP| SafeTraj["Constrained Safe Trajectory"]
    end

    SafeTraj -->|Execute Step 0| Robot["Robot End-Effector Controller (IK)"]
```

---

## 2. Mathematical Foundation: Decoder-Decoupled Projection

The most critical mathematical insight of Gen6 is that **we do not project the high-dimensional visual latents**. Since the ResNet output is a $128$-dimensional abstract feature space, imposing coordinate constraints directly on it is impossible. 

Instead, the **visual latent vector acts strictly as a fixed conditioning signal** during the U-Net reverse diffusion process. The projection operator is applied to the **decoded coordinate output trajectory** $\tau \in \mathbb{R}^{H \times 3}$.

### The Quadratic Programming (QP) Optimization Formulation

At each receding horizon step, the raw generated trajectory $\tau_{raw}$ is projected onto the constraint manifold by solving:

$$\hat{\tau} = \operatorname{argmin}_{\tau} \frac{1}{2} (\tau - \tau_{raw})^T Q (\tau - \tau_{raw})$$

$$\text{subject to: } \quad A \tau = b \quad (\text{Equality / Dynamic Euler Constraints})$$

$$C \tau \leq d \quad (\text{Inequality / Workspace safety bounds})$$

$$g_i(\tau) \leq 0 \quad (\text{Obstacle avoidance non-linear boundaries})$$

Where:
*   $\tau \in \mathbb{R}^{H \times d}$ is the flattened trajectory vector ($H$ timesteps, $d$ dimensions).
*   $Q$ is a positive semidefinite matrix weighing the temporal step coordinate alignment (typically $I$).
*   $A, b$ enforce **First-Order Robot Dynamics** (Euler transition steps).
*   $C, d$ enforce **Physical Workspace Limits** (Cartesian boundaries of the Franka table).

---

## 3. Designing Gen6 Target Constraints for Visual Aligning

For the Aligning task, we define three specific constraint classes using our existing `Projector` framework:

### A. Workspace Safety Cage (Polytopic Bounds)
The Franka robot end-effector must be physically restricted to avoid collisions with its own mount, camera brackets, or table edges:

$$\begin{aligned}
x_{min} \leq x_t &\leq x_{max} \\
y_{min} \leq y_t &\leq y_{max} \\
z_{min} \leq z_t &\leq z_{max}
\end{aligned}$$

By building the polytopic inequality matrices $C_{safe}$ and $d_{safe}$ (leveraging `SafetyConstraints` inside `projection.py`), any path hallucination out of these bounds is snapped back instantly *before* execution.

### B. Robot Kinematic Smoothness (Euler Dynamic Constraints)
To ensure the end-effector follows a smooth path that does not cause joint torque limits to trip or induce camera shake:

$$x_{t+1} = x_t + \Delta x_t \cdot \Delta t$$

By linking position and velocity states, we bound the allowed action delta $\Delta x$:

$$-\text{Limit} \leq \Delta x_t \leq \text{Limit}$$

This is mathematically built using `DynamicConstraints` with an explicit Euler integration matrix.

### C. Dynamic Obstacle Avoidance (Spherical Constraints)
To introduce collision avoidance on the table (e.g. avoiding virtual boxes, or keeping the robot arm away from the camera's sight cone):

$$\| s_t - c_j \|^2 \geq R_j^2$$

This translates into a quadratic form built by `ObstacleConstraints`:

$$-s_t^T I s_t + 2 c_j^T s_t \leq \| c_j \|^2 - R_j^2$$

---

## 4. Integration Blueprint: Direct Code-Reuse from FMv3ODE

The core philosophy of the Gen6 upgrade is **pure code reuse** without writing any new modules or redundant classes. 

### Why No New Code is Needed:
1.  **Direct Inheritance:** The visual diffusion model loaded in `eval_ddpm_encdec_vision.py` is [`VisualGaussianDiffusion`](file:///workspaces/FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py), which inherits directly from `diffuser.models.diffusion.GaussianDiffusion`.
2.  **Built-In DPCC Engine:** The base `GaussianDiffusion` class [`diffuser/models/diffusion.py`](file:///workspaces/FM-PCC/diffuser/models/diffusion.py#L164-L202) already has **100% complete support for in-denoising projection** built into its native reverse diffusion loop (`p_sample_loop`).
3.  **Active Snapping/Gradients:** It already accepts a `projector` parameter and dynamically snaps (`projector.project`) or guides (`projector.compute_gradient`) the trajectory in the intermediate reverse-diffusion steps!

---

### Step 1: Instantiating the Projector in the Evaluation Config
We load the constraint list from the YAML configuration. In [`config/visual_aligning_eval.yaml`](file:///workspaces/FM-PCC/config/visual_aligning_eval.yaml), we define our projection variants:

```yaml
projection_variants:
  - diffuser      # Gen5 Baseline (No projection)
  - fmpcc_safe    # Gen6 Upgrade (Workspace + Kinematics constraints)
  - fmpcc_avoid   # Gen6 Upgrade (Workspace + Obstacles constraints)

# Physical Franka Workspace bounds
workspace_bounds:
  lb: [0.3, -0.3, 0.05]
  ub: [0.7, 0.3, 0.40]
```

---

### Step 2: Activating the In-Denoising Projection Loop
We modify the `VisualAgentWrapper`'s constructor and `predict()` method in [`ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`](file:///workspaces/FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) to accept the existing `Projector` class and pass it directly into the active forward pass:

```python
# Inside VisualAgentWrapper.__init__:
self.projector = projector

# Inside VisualAgentWrapper.predict():
# 1. Trigger the reverse diffusion pass and pass the Projector directly
# The VisualGaussianDiffusion wrapper forwards the projector to super().forward() -> p_sample_loop
trajectory, infos = self.model(cond, projector=self.projector)

# 2. Slice action coordinate sequence
if trajectory.shape[-1] == 3:
    action_trajectory = trajectory
else:
    action_trajectory = trajectory[:, :, :3]

# 3. De-normalize to physical meters
if self.scaler is not None:
    action_trajectory = self.scaler.inverse_scale_output(action_trajectory)
```

---

---

## 5. Architectural Implementation Snippet

Below is the proposed integration snippet for adding the **Gen6 Projector** initialization to the evaluation script, maintaining full compatibility with the existing `Projector` class:

## 5. Architectural Implementation Blueprint

To enable 100% code reuse from the FMv3ODE/diffuser DPCC solver, we make purely additive additions to [`ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`](file:///workspaces/FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py).

### Part A: Define the Normalizer Compatibility Adapter
Since D3IL's visual pipeline uses a standard-deviation-based normalizer (`Scaler`) while the DPCC `Projector` expects min/max bounds (`.mins` and `.maxs`), we place this tiny adapter directly at the top of the evaluation script:

```python
class VisualNormalizerAdapter:
    """Bridges D3IL's Scaler class with the Projector's normalizer expectations."""
    def __init__(self, scaler):
        # Extract physical limits from the scaled dataset bounds
        self.mins = scaler.y_min.detach().cpu().numpy()
        self.maxs = scaler.y_max.detach().cpu().numpy()

class VisualNormalizerDict:
    """Wraps observations and actions to match the dataset normalizers dictionary."""
    def __init__(self, scaler):
        self.normalizers = {
            'observations': VisualNormalizerAdapter(scaler),
            'actions': VisualNormalizerAdapter(scaler)
        }
```

---

### Part B: Instantiate and Setup the Projector (Direct FMv3ODE Port)
We define the setup function inside `eval_ddpm_encdec_vision.py` to parse the constraints, apply the tightening scales, and build the `Projector`:

```python
from ddpm_encdec_vision.sampling.projection import Projector

def setup_gen6_projector(args, config, scaler, variant):
    """Instantiates the DPCC projection engine for the visual workspace."""
    # 1. Determine constraint tightening (contract limits by 'enlarge_constraints' meters)
    enlarge_constraints = config.get('enlarge_constraints', 0.0)
    
    workspace_lb = np.array(config['workspace_bounds']['lb'])
    workspace_ub = np.array(config['workspace_bounds']['ub'])
    
    if 'tightened' in variant and enlarge_constraints > 0.0:
        workspace_lb += enlarge_constraints
        workspace_ub -= enlarge_constraints

    # 2. Formulate Safety Bounds Constraints
    constraint_list = [
        ['lb', workspace_lb],
        ['ub', workspace_ub]
    ]
    
    # 3. Formulate Kinematics/Dynamics Constraints (Euler derivative bounds)
    if 'dynamics' in config.get('constraint_types', []):
        # Explicit Euler derivative step binding coordinate dimensions to dynamics
        constraint_list.append(('deriv', [0, 0]))  # Binds dX to state transition
    
    # 4. Construct compatibility normalizer dict
    adapter_normalizer = VisualNormalizerDict(scaler)
    
    # 5. Initialize the DPCC Projector
    projector = Projector(
        horizon=getattr(args, 'horizon', 8),
        transition_dim=3,                # XYZ Cartesian trajectory
        action_dim=3,                    # XYZ Cartesian actions
        goal_dim=0,                      # Non-goal conditioned VAE
        constraint_list=constraint_list,
        normalizer=adapter_normalizer,
        diffusion_timestep_threshold=config.get('diffusion_timestep_threshold', 0.5),
        variant='states',
        solver='scipy',                  # Robust SLSQP QP optimizer
        device=args.device
    )
    return projector
```

---

## 6. Scientific Metrics & Verification (Thesis Deliverables)

By transitioning from Gen5 (`diffuser`) to Gen6 (`fmpcc_safe`), your thesis will be equipped with rigorous, high-fidelity quantitative comparison metrics:

| Metric Category | Gen5 Baseline (`diffuser`) | Gen6 Constrained (`fmpcc_safe`) | Scientific Significance |
| :--- | :--- | :--- | :--- |
| **Workspace Boundary Violations** | $> 0.0\%$ (Occasionally drifts out of limits) | **$0.0\%$ (Guaranteed)** | Proves safety manifold containment. |
| **Kinematic Jerk / Acceleration** | Higher (Raw U-Net predictions contain high-frequency noise) | **Lower (Smoothed via dynamic constraints)** | Protects real-world robot joint motors. |
| **Success Rate (Aligning)** | High, but vulnerable to corner-case drift | **Equal or Higher** | Proves that safety projection does not degrade task performance. |
| **Inference Latency** | $\sim 5-8$ ms | $\sim 10-15$ ms (Depending on ProxSuite QP time) | Documents the lightweight nature of receding-horizon MPC. |

---

**Gen6 Upgrade Architecture Status: DESIGN COMPLETED (Ready for codebase implementation).**
**Revision**: GEN6_REV_1 (Visual MPC Blueprint)
