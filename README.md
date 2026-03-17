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
