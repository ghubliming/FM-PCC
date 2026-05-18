# Implementation Plan: Unified Visual-DPCC Joint Trajectory Paradigm (Gen6v4)

This document presents a comprehensive, scientifically rigorous research and implementation plan to transition the **Gen6 (D3IL Visual Bridge)** architecture into a unified, **single-horizon joint trajectory DPCC model** (matching the standard DPCC diffuser design), with a highly innovative proposal for handling multi-modal visual data.

---

## 🗺️ Architectural Concept: The Visual-Latent Joint Space

In the standard state-only DPCC way, the trajectory is formulated as a flat joint space: $x \in \mathbb{R}^{H \times (d_a + d_o)}$. 
However, when dealing with multi-modal visual inputs (dual cameras yielding `primary` and `wrist` RGB streams of shape $3 \times 128 \times 128$), directly concatenating raw pixel matrices into the 1D temporal grid is mathematically impossible due to **dimensional explosion** (e.g. over $98,000$ dimensions per step).

To solve this, we propose **Gen6v4-Visual-DPCC**: a **Joint Spatial-Temporal Latent Trajectory Diffusion** model.

```
       [Dual Camera Streams (Primary + Wrist)]
                         │
                         ▼
           [Frozen ResNet-18 Encoder]
                         │
                         ▼
        [Low-Dimensional Visual Latents z_t] (e.g., 64D)
                         │
                         ▼
  [Joint Trajectory Vector x_t = [a_t, s_t, z_t]] (84D per step)
                         │
                         ▼
           ┌───────────────────────────┐
           │   1D Temporal U-Net Core  │  ◄── [Denoise whole trajectory H=8]
           └─────────────┬─────────────┘
                         │
                         ▼
           ┌───────────────────────────┐
           │  DPCC Boundary Projector  │  ◄── [Convex QP table/obstacle constraints]
           └─────────────┬─────────────┘
                         │
                         ▼
                 [Executed Action a_0]
```

---

## 🔬 Core Innovations for Visual-DPCC Ingestion

We propose three distinct architectural inventions to merge D3IL's high-fidelity visual data with standard DPCC diffusion/Flow Matching backbones:

### Invention 1: The Visual-Latent Joint Trajectory (Direct Diffusion)
* **Mechanism:** Rather than treating images as external conditioners, we encode each frame into a compact latent vector $z_t^{vis} \in \mathbb{R}^{64}$ using a pre-trained ResNet + Spatial Softmax.
* **Trajectory Vector Formula:** 
  $$x_t = \begin{bmatrix} a_t & s_t^{proprio} & z_t^{vis} \end{bmatrix} \in \mathbb{R}^{2 + 18 + 64} = \mathbb{R}^{84}$$
* **Mathematical Optimization:** The U-Net diffuses the entire $84D$ sequence of length $H=8$ simultaneously.
* **Key Benefit:** Complete structural unification. The U-Net treats visual representations as standard continuous trajectories, allowing temporal smoothers to run over visual changes.

### Invention 2: Cross-Attention Spatial-Temporal FiLM (Conditioned Diffusion)
* **Mechanism:** The trajectory diffused by the U-Net remains low-dimensional, consisting only of actions and proprioceptive states: $x_t = [a_t, s_t^{proprio}] \in \mathbb{R}^{20}$.
* **Image Context Processing:** Multi-step history images (steps $0 \dots 4$) are encoded into a key-value sequence $K, V$.
* **Integration:** Intermediate U-Net layers utilize Cross-Attention blocks or FiLM linear projections to guide the denoising trajectory:
  $$\hat x_{t-1} = \operatorname{UNet}(x_t, t \mid \operatorname{Attention}(x_t, z^{vis}))$$
* **Key Benefit:** Leverages the raw visual encoder's deep feature representations without burdening the joint diffusion dimensions.

### Invention 3: Self-Supervised Masked Joint Trajectory Modeling
* **Mechanism:** Train the joint U-Net model on the complete space $x = [a, s^{proprio}, z^{vis}]$ using dynamic masking.
* **Masking Strategy:** During training, randomly mask $80\%$ of either the actions, proprioceptive states, or visual latents.
* **Key Benefit:** Converts the U-Net from a simple policy planner into a **unified forward dynamics simulator**. The model can either predict future actions (imitation learning) or simulate future visual frames and physical states given active actions (forward world modeling).

---

## 📋 Phase-by-Phase Migration Plan

To implement this paradigm in your codebase, follow this structured plan:

### 🚀 Phase 1: The D3IL-to-DPCC Visual Dataloader
1. **Target File:** Create a custom dataset parser inside [diffuser/datasets/d4rl.py](file:///workspaces/FM-PCC/diffuser/datasets/d4rl.py) (similar to the Avoiding scraper).
2. **Implementation:**
   * Read the serialized `.pkl` visual logs.
   * Parse proprioception and action delta velocities.
   * Extract image paths or raw byte streams and compress them into pre-computed ResNet feature tensors.
   * Package them into sequence-based trajectory batches of length $H=8$.

---

### 🚀 Phase 2: Joint U-Net and Normalizer Adaptations
1. **Visual Normalization Dict:** Modify `DatasetNormalizer` ([diffuser/datasets/normalization.py](file:///workspaces/FM-PCC/diffuser/datasets/normalization.py)) to support z-score normalization of both flat proprioception and the 64D latent visual vectors.
2. **Transition Dimension scaling:**
   * Scale the U-Net transition dimension to $d_a + d_{proprio} + d_{latent} = 2 + 18 + 64 = 82$.
   * Lock the initial state ($t=0$) for both proprioception and visual latents inside `apply_conditioning()` in `helpers.py`.

---

### 🚀 Phase 3: The Step-by-Step Receding Horizon Controller
1. **Control Loop Migration:** Replace the chunked evaluation step in `eval_ddpm_encdec_vision.py` with standard closed-loop step replanning:
   ```python
   # Proposed step execution in visual evaluation loop
   def step(self, obs_state, image_frames):
       # 1. Compress active dual images into 64D visual latent
       z_t = self.resnet_encoder(image_frames)
       
       # 2. Package current state condition
       cond = {0: np.concatenate([obs_state, z_t])}
       
       # 3. Call joint trajectory model (denoises full H=8 trajectory)
       trajectory, infos = self.model(cond, projector=self.projector)
       
       # 4. Extract ONLY the first step action velocity (a_0)
       action = trajectory[0, :self.action_dim]
       return action
   ```

---

## 🏆 Scientific Value and Thesis Contribution
By adapting Gen6 to the Visual-DPCC joint trajectory paradigm:
1. **Real-time Obstacle Avoidance:** Eliminates the latency of chunked planning, enabling the robot to dynamically evade obstacles during visual rollouts.
2. **Safety Guarantees:** Applies QP boundary projections directly to continuous physical trajectories, ensuring tabletop and collision limits are never violated.
3. **Advanced World Modeling:** Introduces self-supervised visual-latent trajectory forecasting as a novel method for model-based imitation learning.
