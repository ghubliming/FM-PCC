# Head-to-Head Architectural Comparison: Gen6 (D3IL Bridge) vs. The DPCC Way (3D Cartesian Edition)

This document provides a concrete, code-level comparison between the two primary learning paradigms in this workspace: **Gen6 (D3IL Visual/Non-Visual Bridge)** and the **DPCC Way (Standard Trajectory Diffuser)**. It outlines the core architectural contrasts, control flows, dataset strategies, and exact code references, adapted for full 3D coordinates.

---

## 📊 Summary Matrix

| Comparative Dimension | **Gen6 Paradigm (D3IL Bridge)** | **The DPCC Way (Standard Diffuser)** |
| :--- | :--- | :--- |
| **Primary Architecture** | Hybrid VAE-ACT Compatibility Adapter wrapping a 1D U-Net | Pure Receding-Horizon 1D Convolutional Trajectory Model |
| **Temporal Parameterization** | Split: `obs_seq_len` (5), `action_seq_size` (4) | Unified: Single **`horizon = 8`** parameter |
| **Temporal Relationship** | $\text{window\_size} = \text{obs\_seq\_len} + \text{action\_seq\_size} - 1 = \mathbf{8}$ | $H = \mathbf{8}$ (No subdivisions) |
| **Inference Control Flow** | **Chunked Receding-Horizon:** Executes block of 4 actions before replanning | **Step-by-Step Receding-Horizon:** Executes $a_0$, replans every step |
| **Evaluation Script** | [eval_ddpm_encdec_vision.py](file:///workspaces/FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) | [diffuser/sampling/policies.py](file:///workspaces/FM-PCC/diffuser/sampling/policies.py) |
| **Safety Integration** | Post-hoc adapter execution (QP boundary limits applied on evaluation) | Continuous **DPCC projection** inside reverse denoising loop |
| **Dataset Ingestion** | Pre-processed D3IL pickles (`train_files.pkl` via complex loaders) | Custom in-memory sequence wrapper parsing raw log pickle lists |

---

## 🧮 Mathematical Derivations and Code Implementation

The two paradigms diverge fundamentally in how they formulate the joint probability distribution of observations ($o$) and actions ($a$) across the planning horizon, and how they execute plans.

### 1. The DPCC Way: Joint Trajectory Diffusion & Step-by-Step Execution
The DPCC framework treats trajectory generation as the estimation of a joint distribution over state-action spaces. 

In our **3D Cartesian Edition**, the trajectory vector $x \in \mathbb{R}^{H \times (d_a + d_o)} = \mathbb{R}^{8 \times 9}$ represents the sequence of joint actions and observations over the horizon $H=8$:
$$x = \begin{bmatrix} (a_0, o_0), & (a_1, o_1), & \dots, & (a_{H-1}, o_{H-1}) \end{bmatrix}$$
where:
* $a_t = [dx_t, dy_t, dz_t] \in \mathbb{R}^3$ (3D delta commands)
* $o_t = [des\_x_t, des\_y_t, des\_z_t, x_t, y_t, z_t] \in \mathbb{R}^6$ (6D stacked proprioception)

The diffusion model learns to estimate the probability density of the joint trajectory $p_\theta(x)$ by optimizing the standard variational lower bound (ELBO) over forward-noised trajectories $x_t$:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

#### 💻 Step-by-Step Planning Execution Code:
During rollout, the DPCC policy generates the plan over the entire `horizon`, but extracts and executes **only the very first action ($a_0$)**, replanning on the next simulation frame:

```python
# File: diffuser/sampling/policies.py (Lines 87-92)
    def __call__(self, conditions, batch_size=1, horizon=16, test_ret=None, constraints=None, disable_projection=False):
        # ... [Inverse diffusion loop generates samples of shape (B, H, A+O)]
        
        else:
            ## 1. Extract action trajectory [ batch_size x horizon x action_dim ]
            actions = trajectories[:, :, :self.action_dim] # self.action_dim = 3
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## 2. Mathematically extract ONLY the first action step (index 0, which is 3D XYZ)
            action = actions[which_trajectory, 0]

        trajectories = Trajectories(actions, observations)
        return action, trajectories
```

---

### 2. Gen6 (D3IL Bridge): Conditional Action Diffusion & Chunked Block Execution
Gen6 separates observations into a conditional context block $c$, and applies the diffusion process **strictly over future actions** $y \in \mathbb{R}^{\text{action\_seq\_size} \times d_a}$:

Let the observation context history $c$ of length $L_{obs}$ be:
$$c = \begin{bmatrix} o_{t-L_{obs}+1}, & \dots, & o_t \end{bmatrix}$$

Let the predicted action plan $y$ of length $L_{act}$ be:
$$y = \begin{bmatrix} a_t, & a_{t+1}, & \dots, & a_{t+L_{act}-1} \end{bmatrix}$$

The U-Net learns the conditional distribution $p_\theta(y \mid c)$ by optimizing:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, y_0, \epsilon, c} \left[ \| \epsilon - \epsilon_\theta(y_t, t \mid \psi(c)) \|^2 \right]$$

#### 💻 Chunked Block Planning Execution Code:
During online rollouts, Gen6 caches the predicted action plan and executes it over multiple environment steps before triggering the next diffusion generation pass:

```python
# File: ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py (Lines 469-552)
    def step(self, obs_state, cond):
        # ...
        
        # 1. Check if the active action chunk (size = action_seq_size = 4) has been fully executed
        if self.action_counter == self.action_seq_size:
            # Plan! Call inverse diffusion model to get action trajectories
            trajectory, infos = self.model(cond, projector=self.projector)
            
            # Slice and normalize generated actions
            action_trajectory = trajectory[:, :, :self.action_dim] # action_dim = 3
            action_trajectory = self.scaler.inverse_scale_output(action_trajectory)

            # Store the action plan block (chunk size = 4)
            self.curr_action_seq = action_trajectory[:, :self.action_seq_size, :]
            self.action_counter = 0

        # 2. Extract and execute the sequential action within the cached block
        action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        
        return action
```

---

## 🔄 Dynamic Control Flow Contrast

The structural execution loops in the simulator highlight the difference in execution frequency and closed-loop responsiveness:

### 1. The DPCC Step-by-Step Execution Loop
Operates at high closed-loop frequency, replanning on every single tick in full 3D:

```
[Get Simulator Poses des_pos, c_pos (3D)]
         │
         ▼
[Format 6D State Vector [des_pos, c_pos]]
         │
         ▼
[Denoise Horizon H=8 (9D)] ◄───[Apply 3D SLSQP QP Projector on intermediate steps]
         │ (Code: diffuser_visual_aligning/sampling/policies.py)
         ▼
[Extract only first 3D action a_0]
         │ (Code: action = actions[which_trajectory, 0])
         ▼
[Execute 3D a_0 directly in MuJoCo]
         │
         ▼
     [t = t + 1] ──► (Loop back to start)
```

---

### 2. The Gen6 Chunked Block Execution Loop
Executes actions in open-loop blocks of size `action_seq_size = 4`, reducing computation at the cost of reaction latency:

```
[Gather Context History c of length 5]
         │
         ▼
[Encode Context c via ResNet / FiLM]
         │
         ▼
[Denoise Action Block y of size 4]
         │ (Code: eval_ddpm_encdec_vision.py)
         ▼
[Execute a_0, a_1, a_2, a_3 open-loop in MuJoCo]
         │ (Code: self.action_counter += 1)
         ▼
     [t = t + 4] ──► (Loop back to start)
```

---

## 🏆 Comparative Verification

### 1. Reactivity vs. Computational Budget
* **DPCC Way:** Extremely reactive. Because it replans at every timestep (extracting `actions[which_trajectory, 0]` in `policies.py`), the robot can instantly dodge obstacles or recover from unexpected contact forces. However, this demands high computational power because diffusion inference runs at every simulation step.
* **Gen6 Paradigm:** Computations are cut by $4\times$ because inference only runs every $4$ steps (guarding model calls behind `if self.action_counter == self.action_seq_size:` in `eval_ddpm_encdec_vision.py`). However, because it runs open-loop within the block, the robot cannot react to collision forces or dynamic obstacles until the execution block finishes.

### 2. Dataset Scrapers vs. Processed Datasets
* **DPCC Way:** Uses raw `.pkl` simulation dictionary dumps parsed in-memory, extracting 3D desired position and 3D physical feedback pose, and stacking them into a unified 6D observation vector.
* **Gen6 Paradigm:** Loads from pre-compiled D3IL alignments databases ([d3il/environments/dataset/aligning_dataset.py](file:///workspaces/FM-PCC/d3il/environments/dataset/aligning_dataset.py)) specifically prepared for ACT benchmarks.
