# Implementation Plan: Migration to D3IL DDPM-ACT Backbone (Fix 10)

## 1. Objective
Swap the current convolutional **1D U-Net** backbone for the D3IL native **Transformer-based DDPM-ACT** (`DiffusionEncDec`) while maintaining the DPCC "Bridge" structure. This allows for a scientific comparison between convolutional temporal modeling and attention-based temporal modeling within the same perception framework.

## 2. Architectural Comparison

| Feature | Current Gen5 (U-Net) | **Target Gen5 (ACT)** |
| :--- | :--- | :--- |
| **Backbone** | `UNet1DTemporalCondModel` | `DiffusionEncDec` (Transformer) |
| **Logic** | Temporal Convolutions | Self/Cross Attention |
| **Integration** | FiLM conditioning | Token-based / Cross-Attention |
| **Provenence** | DPCC (Legacy Diffuser) | **D3IL (Native Agents)** |

## 3. Step-by-Step Implementation

### Step 1: Configuration Rewiring
Modify `config/aligning-d3il-visual.py` to point the `backbone` to the `VisualDiffusionBridge`:
```python
# From:
"backbone": "ddpm_encdec_vision", # Targeted VisualUNet

# To:
"backbone": "d3il_visual_bridge", # Targets VisualDiffusionBridge
```

### Step 2: Bridge Refinement (`d3il_visual_bridge.py`)
Ensure the `VisualDiffusionBridge` correctly instantiates the D3IL `DiffusionEncDec` using Hydra.
- Verify that the `encoder` (Vision) and `decoder` (Transformer) configs match the D3IL benchmark exactly to ensure scientific parity.

### Step 3: Training Integration
Update `train_ddpm_encdec_vision.py` (or create a new entry point) to:
1.  Load the Bridge-based model.
2.  Map the 128-dim visual features to the Transformer's observation tokens.
3.  Ensure the `GaussianDiffusion` engine from DPCC remains compatible with the Transformer's output shape.

### Step 4: Comparison & Benchmarking
Utilize the Fix 9 upgraded evaluation output (`trajectories_seed_<seed>.pkl`) to analyze:
- **Trajectory Smoothness:** Does the Transformer jitter more than the U-Net?
- **Alignment Error:** Is the attention mechanism superior at matching the box to the target?
- **Inference Latency:** Benchmark the speed of the Transformer vs. the U-Net.

## 4. Risks & Mitigations
- **Instability:** Transformers often require more data/epochs than U-Nets for robot manipulation.
- **Divergence:** Ensure the "Bridge" doesn't add any hidden bias that prevents the ACT model from reaching its benchmark success rates.

---

**Plan generated for FM-PCC Diagnostic Phase 10.**
