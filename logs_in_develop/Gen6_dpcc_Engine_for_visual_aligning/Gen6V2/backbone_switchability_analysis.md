# Gen6V2: Dual-Backbone Switchability Analysis (U-Net vs. Transformer)

This document investigates the feasibility, architectural design, and code blueprints required to transition the **Gen6 Visual-Aligning Engine** into a **Dual-Backbone Switchable System**. Specifically, we analyze if we can swap the core machine learning backbone between the **1D Temporal CNN U-Net** and the **VAE Transformer** while preserving Gen6's safety safeguards (Safe Scaler, Joint 6D Trajectory, Hard Boundary Snapping, and prioritized Loss Weighting).

---

## 🎯 Feasibility Verdict: **Architectural Mismatch Detected (Requires Structural Adapters)**

While the high-level diffusion interface makes swapping models look simple at a distance, a deep code-level audit of D3IL's [act_vae.py](file:///workspaces/FM-PCC/d3il/agents/models/act/act_vae.py) reveals **fundamental trajectory coordinate and sequence mismatches**. Swapping the backbone is **not** a simple drop-in substitution; doing so requires introducing custom structural adapters or an engine-level coordinator.

---

## 🚨 The Code-Level Mismatch Conflicts (ACT VAE vs. Gen6 U-Net)

If we attempt to plug the native `ActVAE` model directly into the Gen6 joint diffusion loop, the system will crash immediately due to three hardcoded structural assumptions in [act_vae.py](file:///workspaces/FM-PCC/d3il/agents/models/act/act_vae.py#L325):

### 1. The Proprioceptive State Slicing Conflict
Inside `ActVAE.forward()`, the state history is explicitly truncated to the first frame, discarding all intermediate trajectory states:
```python
# act_vae.py (Line 391)
state = state[:, :1, :]
```
* **Gen6 Joint Parity**: Gen6 tracks and denoises proprioceptive joint states along the *entire* sliding temporal window (`[B, T, 3]`). Discarding intermediate states completely breaks joint sequence planning.

### 2. Fixed-Length Query Embeddings
The Transformer Decoder in ACT utilizes a fixed query embedding bank tailored strictly to the action sequence chunk size (`action_seq_size` = 4):
```python
# act_vae.py (Line 362 & 438)
self.query_embed = nn.Embedding(act_seq_size, hidden_dim)
...
action_seq = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
```
* **Gen6 Joint Parity**: Gen6 requires the backbone to predict noise vectors over a dynamically padded sequence window of size $8$ (`padded_horizon`). A fixed 4-query token bank cannot generate 8-step trajectories.

### 3. Coordinate Dimension Mismatch
`ActVAE`'s projection heads and output decoders are hardcoded to output 3D action delta velocities ($3\text{D}$ action pose target offsets):
```python
# act_vae.py (Line 359)
self.action_head = nn.Linear(hidden_dim, action_dim) # Outputs 3D Actions
```
* **Gen6 Joint Parity**: Gen6 performs diffusion over a concatenated 6D trajectory space $x = [act, obs]$ (Shape: `[B, T, 6]`). `ActVAE` is mathematically incapable of predicting noise vectors for the remaining 3 proprioceptive observation dimensions.

---

## 🛠️ The Dual Architectural Solutions

To achieve true switchability, we must implement one of two design options:

### ⚙️ Option A: Engine-Level Coordinator Switching (Highly Recommended)
Instead of forcing the Transformer VAE to output joint trajectories, we introduce an **Engine-Level Swappable Trainer** inside `train_ddpm_encdec_vision.py`. This trainer toggles the entire trajectory representation and loss formulation based on configuration flags:

```
                                 [Engine Toggler]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌──────────────────────────────────┐                 ┌──────────────────────────────────┐
│      Joint 6D DDPM Engine        │                 │     Action-Only VAE Engine       │
│  - Trajectory: [act, obs] (6D)   │                 │  - Trajectory: [action] (3D)     │
│  - Model: 1D Temporal U-Net      │                 │  - Model: Transformer VAE        │
│  - Snap: apply_conditioning()    │                 │  - Snap: Latent cross-attention  │
└──────────────────────────────────┘                 └──────────────────────────────────┘
```

#### Code Modifications Required for Option A:
1. **Trainer Swapping**: Modify `train_ddpm_encdec_vision.py` L260 to conditionally instantiate either `VisualGaussianDiffusion` (for joint 6D U-Net planning) or `Diffusion` (for action-only Transformer VAE planning).
2. **Evaluator Swapping**: Modify `eval_ddpm_encdec_vision.py` to map the agent's prediction outputs correctly: U-Net outputs 6D joint coordinates which are sliced at index $0$ (`pred_action[0]`), while the VAE outputs pure actions directly.

---

### 🧠 Option B: The Joint 6D VAE Transformer Adapter
If we must preserve the joint 6D trajectory formulation (`[act, obs]`), we must implement a custom wrapper model — `JointTransformer` — which adapts the attention layers to process and output 6D sequences:

```python
# ddpm_encdec_vision/models/joint_transformer.py
import torch
import torch.nn as nn
from d3il.agents.models.act.act_vae import TransformerEncoder, TransformerDecoder

class JointTransformer(nn.Module):
    """
    Structural Adapter translating ACT-style Attention blocks into
    a joint 6D trajectory model that acts as a direct drop-in replacement for VisualUNet.
    """
    def __init__(self, config):
        super().__init__()
        self.device = getattr(config, "device", "cuda")
        self.horizon = config.horizon # Arbitrary temporal sequence T (e.g. 8)
        self.hidden_dim = getattr(config, "hidden_dim", 256)
        
        # 1. Image Encoder (yields 128D visual embedding)
        # 2. Linear projection for Joint 6D sequences [act, obs]
        self.joint_input_proj = nn.Linear(config.action_dim + 3, self.hidden_dim)
        
        # 3. Transformer blocks
        self.encoder = TransformerEncoder(
            embed_dim=self.hidden_dim,
            n_heads=4,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            n_layers=2,
            block_size=self.horizon + 1
        )
        self.decoder = TransformerDecoder(
            embed_dim=self.hidden_dim,
            cross_embed=128, # Match visual encoder output channels
            n_heads=4,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            n_layers=4,
            block_size=self.horizon
        )
        
        # 4. Joint Output Projection Head (maps back to 6D joint vector)
        self.joint_output_proj = nn.Linear(self.hidden_dim, config.action_dim + 3)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.horizon, self.hidden_dim))
        
    def forward(self, x, cond, t, **kwargs):
        """
        x: [B, T, 6] (noisy joint sequence)
        cond: {'visual': (bp_imgs, inhand_imgs, state)}
        """
        B, T, D = x.shape
        bp_imgs, inhand_imgs, state = cond['visual']
        
        # 1. Encode high-dimensional camera frames
        visual_emb = self.encode_visual(bp_imgs, inhand_imgs, state=state) # [B, T, 128]
        
        # 2. Project joint sequence to embedding space and add temporal pos embeddings
        x_embed = self.joint_input_proj(x) + self.pos_emb[:, :T, :]
        
        # 3. Add diffusion time-step embedding (t) as a latent context token
        # 4. Decode via cross-attention over visual embeddings
        decoder_output = self.decoder(x_embed, visual_emb)
        
        # 5. Map back to 6D coordinates to output predicted noise epsilon
        noise_pred = self.joint_output_proj(decoder_output) # Shape: [B, T, 6]
        return noise_pred
```

---

## 📊 Summary of Switchability Strategies

| Strategy | Implementation Complexity | Architectural Cleanliness | Training Parity |
| :--- | :--- | :--- | :--- |
| **Simple Drop-In VAE** | Impossible (Crashes immediately) | Poor | None (Dimension mismatch) |
| **Option A (Engine-Level Toggling)** | Medium | High (Maintains pure VAE vs. pure U-Net concepts) | High (Direct benchmark comparison under native conditions) |
| **Option B (Joint Transformer Adapter)** | High (Requires custom query projection and adapter code) | High (All backbones conform to 6D planning) | High (Compares Transformer vs. U-Net attention capacities under identical 6D snapped loops) |

---

## 📈 Conceptual Trajectory Slice Identity: U-Net `horizon` vs. VAE `window_size`

In terms of data loading from expert demonstrations, **`horizon`** and **`window_size`** represent the **exact same physical concept**: the temporal length of the expert trajectory chunk (e.g., 8 steps) carved out of the full rollout ($512$ steps) to train the model at each training iteration.

### 🔄 Slicing Parity inside Dataset Loaders
In `train_ddpm_encdec_vision.py` L230, the dataset loader is instantiated by directly mapping the U-Net horizon to the dataset's `window_size` parameter:
```python
window_size=args.horizon,
```
Thus, from the dataset's perspective, both models receive the same $8$-frame expert trajectory slice.

### 🔀 Internal Architectural Processing Mappings
While the slice length is identical (8 frames), the backbones internally split and interpret this sequence in fundamentally different ways:

```
                            [ 8-Step Contiguous Slice ]
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  <-- U-Net Contiguous Horizon [B, 8, 6]
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

┌─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │                      <-- obs_seq_len (First 5 frames used as context)
└─────┴─────┴─────┴─────┼─────┼─────┬─────┬─────┐
                        │  5  │  6  │  7  │  8  │  <-- action_seq_size (Last 4 frames predicted as actions)
                        └─────┴─────┴─────┴─────┘
```

1. **U-Net CNN (Joint 6D Formulation)**:
   * **Sequence Shape**: `[B, 8, 6]`
   * **Processing**: Treats all 8 steps of proprioceptive states ($3\text{D}$) and actions ($3\text{D}$) as a contiguous joint vector, performing 1D temporal convolutions to denoise the entire $8$-step block.
2. **VAE Transformer (`ddpm_act`)**:
   * **Sequence Shape**: Split into `[B, 5, 128]` visual tokens and `[B, 4, 3]` action coordinates.
   * **Processing**: Splits the 8-frame slice into $5$ past observation frames (visual context) and $4$ future action planning frames (target predictions), overlapping at step 5:
     $$\text{obs\_seq\_len} + \text{action\_seq\_size} - 1 = 5 + 4 - 1 = 8$$

---

## 🎛️ Hyperparameter Calibration Blueprint (Backbone-Specific vs. Invariant Parameters)

When swapping the backbone between the **1D Convolutional U-Net** and the **Transformer VAE**, hyperparameters must be calibrated carefully. Some parameters **must** be changed to prevent training divergence, while others are **invariant** (make no sense to change) because they govern shared dataset, environment, or evaluation structures.

### 🔄 1. Parameters That MUST Change (Backbone-Specific)

These parameters directly shape the optimization landscape, network sizing, and temporal alignment of the chosen backbone:

| Parameter | U-Net Target Value | Transformer Target Value | Scientific Reason for Calibration |
| :--- | :---: | :---: | :--- |
| `learning_rate` | **`2e-4`** | **`5e-4`** | **Convolutional Gradient Limits**: Standard 1D CNNs become highly unstable and explode if training rates exceed `5e-4`. Self-normalizing attention layers (with LayerNorm) in VAE are highly stable and require `5e-4` for rapid convergence. |
| `condition_dropout` | **`0.25`** | **`0.10`** | **CFG vs. Direct Mapping**: For U-Net under PCC, a `0.25` dropout trains a strong unconditional prior necessary for stable Classifier-Free Guidance (w=1.2). VAE has direct latent conditioning and works best with a lower `0.10` dropout to maximize visual context mapping. |
| `dim` | **`32`** or **`128`** | **`N/A`** (Ignored) | **CNN Channel Width**: Controls the layer filters for the 1D temporal convolution down/up sampling blocks. Ignored by VAE. |
| `hidden_dim` | **`N/A`** (Ignored) | **`256`** (or **`64`** for D3IL) | **Attention Vector Size**: Controls the token embedding dimension inside the VAE encoder/decoder self-attention blocks. Ignored by U-Net. |
| Sequence Lengths | `horizon = 8` | `obs_seq_len = 5`<br>`action_seq_size = 4`<br>`window_size = 8` | **Structural Coordinates**: U-Net downsamples sequence blocks by divisions of 8, requiring `horizon = 8`. VAE strictly splits tokens into overlapping histories and predictions (`5 + 4 - 1 = 8`). |

---

### 🔒 2. Parameters That MAKE NO SENSE to Change (Invariant)

These parameters must remain identical across both backbones. Changing them between models ruins experimental parity and destroys comparative validity:

* **`n_diffusion_steps` (16 / 100)**: 
  * *Why Invariant*: This defines the temporal discretization of the stochastic denoising process. While a researcher can increase this to `100` for higher physical precision, this change **must** be done identically on both models. Changing it for only one model invalidates any scientific comparison.
* **`action_dim` (3)**: 
  * *Why Invariant*: Hard-locked by the physical simulator. Both backbones plan spatial trajectories for the 3D translation velocities of the MuJoCo robot hand effector.
* **`loss_type` ('l2')**: 
  * *Why Invariant*: The mathematical formulation of the DDPM reconstruction objective requires L2 loss (Mean Squared Error) to compute denoising targets correctly. Switching to L1 makes no mathematical sense for Gaussian diffusion.
* **`batch_size` (64)**: 
  * *Why Invariant*: Standardized to match the parallel image rendering buffers of the multi-camera ResNet encoders. Changing batch sizes would skew gradient step noise and invalidate learning curves.
* **`ema_decay` (0.995)**: 
  * *Why Invariant*: Standardizes parameter smoothing. Both networks use the same exponential moving average decay factor (`0.995`) to ensure physical trajectory continuity during rollouts.
* **`normalizer` ('LimitsNormalizer')**: 
  * *Why Invariant*: Trajectory scaling statistics must be identical to guarantee that both models operate on the same normalized action/state bounds.
