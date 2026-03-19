# FM-PCC

## 1. Traditional Model Predictive Control (MPC)
**Methodology:**
This baseline utilizes a classical receding horizon control framework. At each physical time $t$, the controller solves a constrained optimization problem over a finite horizon $H$. It minimizes a predefined cost function while strictly adhering to the system's transition dynamics, $s_{t+1} = f(s_t, a_t)$, and operational boundaries.

**Key Characteristics:**
* **Explicit Modeling:** Relies on hand-crafted cost functions and accurate system dynamics.
* **Deterministic Planning:** Produces a single optimal path based on mathematical optimization.
* **Limitation:** Struggles to capture complex, multimodal behaviors from human demonstrations without expert-designed heuristics.

---

## 2. Diffusion Predictive Control with Constraints (DPCC)
**Methodology:**
DPCC replaces the hand-crafted cost function with a generative 1D U-Net trained via Denoising Diffusion Probabilistic Models (DDPM). The model learns an underlying trajectory distribution from expert data, enabling the system to "hallucinate" goal-reaching paths that reflect learned expert intuition.

**Technical Implementation:**
* **Stochastic Denoising:** During inference, a batch of trajectories is generated from Gaussian noise $\tau^K \sim \mathcal{N}(0, I)$ over $K$ reverse steps.
* **Model-Based Projections:** To ensure safety against novel obstacles, a model-based projection $\Pi_{\mathcal{Z}_f}$ is integrated into every denoising microstep $k$. This forces the AI-generated path to satisfy physical dynamics and tightened state constraints.
* **Constraint Tightening:** Incorporates a safety buffer to account for model mismatch $w_t$ between the predicted and real-world states.

---

## 3. Flow Matching Predictive Control (FM-PCC)
**Methodology:**
This module upgrades the generative engine from stochastic diffusion to deterministic Flow Matching (FM) with Optimal Transport (OT). The U-Net is retrained to predict a velocity vector field $v_\theta$ rather than noise components.

**Technical Implementation:**
* **Deterministic Inference:** Replaces the stochastic reverse steps with an Ordinary Differential Equation (ODE) solver. The trajectory evolves according to the learned velocity field:

$$
\tau_{new} = \Pi_{\mathcal{Z}_f}(\tau_{old} + v_\theta(\tau_{old}, t) \Delta t)
$$

* **Efficiency Gains:** By eliminating random noise injection ($\epsilon_k$) during the generation process, the trajectory follows a smooth, direct path. This reduces the computational load on the geometric projection operator, leading to faster inference and more stable physical execution.
* **Refined Selection:** Since paths are deterministic and smooth, selection criteria like Cumulative Projection Cost are largely replaced by simpler consistency filters.

---
# Appendix
**Methodology:**
The core architecture decouples the generative "brain" from the physical "brakes." The AI operates in an idealized mathematical space, while the Model Predictive Control (MPC) enforces strict real-world hardware limits. The success of the framework depends entirely on the handoff between these two mathematical regimes.

**The Dual-Path Dynamics:**
* **The Unconstrained Reference (The Ghost Path):** The Flow Matching ODE solver continuously integrates the learned velocity field $v_\theta$, generating a perfectly straight Optimal Transport path from the initial state to the target. This reference path, $X_{ref}$, represents pure intent. It completely ignores friction, joint torque limits, and environmental obstacles.
* **The Constrained Execution (The Robot Path):** The MPC algorithm receives $X_{ref}$ and attempts to track it by minimizing a tracking error cost function over the receding horizon. However, the optimization solver is bounded by strict physical inequalities, such as maximum actuator torque ($u_{min} \le u_t \le u_{max}$) and velocity limits. 
* **The Result:** If the AI's "ghost" trajectory demands a velocity of $0.5$ units per step, but the physical motors are mathematically capped at $0.1$ units, the MPC solver intervenes. It forces the system to output the maximum safe physical action, resulting in a slower, constrained trajectory that reliably tracks the AI's vector field without shattering the hardware.

**Mathematical Vulnerabilities:**
* **The Infeasibility Trap:** Because the Flow Matching generation is entirely unconstrained, it can theoretically generate a path that passes directly through a physical barrier. If this unconstrained reference diverges too far from physical reality, the constrained optimization solver (e.g., SLSQP) may face conflicting objectives and become mathematically infeasible, resulting in control failure.
* **Lipschitz Continuity Requirements:** For the MPC solver to reliably calculate gradients and find optimal safe actions, the AI's generated vector field must remain smooth. If the neural network learns a highly turbulent or stiff velocity field, the resulting erratic reference trajectory can induce numerical instability in the control loop, manifesting as physical joint chatter.

---

## The FM-PCC Execution Loop

**1. State Observation (The Starting Line)**
* **Input:** The current physical reality of the robot ($s_t$).
* **Action:** The system reads the exact joint angles, velocities, and target coordinates from the MuJoCo simulation or physical hardware.

**2. The Generative Brain (Flow Matching)**
* **Input:** The current state ($s_t$) and the current integration time step ($t$).
* **Action:** The trained U-Net processes the state and predicts the optimal velocity vector field ($v_\theta$). 
* **The Math:** Instead of guessing noise, it calculates the exact direction and speed needed to flow toward the target distribution.

**3. Trajectory Integration (The Ghost Path)**
* **Input:** The predicted velocity field ($v_\theta$).
* **Action:** A deterministic ODE solver (like a simple Euler integrator) steps forward through time, drawing a smooth, unconstrained path.
* **Output:** The Reference Trajectory ($X_{ref}$). This is the "hallucination"—the mathematically perfect, straightest-line path to the goal that completely ignores real-world physics like friction or torque limits.

**4. The Physical Reality Check (Model Predictive Control)**
* **Input:** The unconstrained reference trajectory ($X_{ref}$) from the AI.
* **Action:** The MPC algorithm takes the AI's "ghost" path and attempts to map it onto physical reality over a receding horizon.
* **The Constraints:** The internal optimizer (like SLSQP) rigorously checks the math against physical boundaries: Can the motors output this much torque? ($u_{min} \le u_t \le u_{max}$) Will this action cause a collision?
* **Output:** A mathematically solved sequence of physically safe actions that tracks the ghost path as closely as hardware allows.

**5. Execution and Receding Horizon (The Physical Step)**
* **Action:** The system extracts only the very first action ($a_t$) from the optimized sequence and sends it to the physical robot's motors.
* **Result:** The robot physically moves to the new state ($s_{t+1}$). The entire loop then instantly restarts, generating a fresh vector field from the new location.

***

```text
[ Current State: s ]
        │
        ▼
[ Flow Matching U-Net ] ────► Predicts Velocity Field: vθ
        │
        ▼
[ ODE Solver (Euler) ]  ────► τ_new = τ_old + vθ · Δt
        │
        ▼
[ Reference Path: τ_ref ] ──► The Unconstrained "Ghost" Trajectory
        │
        ▼
[ MPC Safety Filter ]   ────► Enforces Constraints (u_max, obstacles)
        │
        ▼
[ Optimal Action: α ]   ────► Safe Motor Command
        │
        ▼
[ Physical Environment ] ───► s moves to s' (Loop Restarts)
```
