# Training Details Analysis: FM-PCC vs. Two-Step Safety Fine-Tuning

This report provides a technical analysis of the training methodology used in the **FM-PCC** project and contrasts it with a "two-step" safety fine-tuning procedure (Safe Flow Matching).

## 1. FM-PCC Training Overview

The core training methodology in FM-PCC is designed to learn a generative **prior** over expert demonstrations, decoupling safety from the learning process.

- **What is trained?** A generative model (Diffusion or Flow Matching) is trained to predict the noise or velocity field required to recreate expert trajectories. These trajectories consist of a sequence of states $(s)$ and actions $(a)$.
- **Fine-Tuning:** There is no separate "safety fine-tuning" phase. The model is trained directly on raw expert data. Fine-tuning in this codebase refers strictly to resuming training from a previous checkpoint or adaptation to a new environment's dataset.
- **Role of the Projector:** The projector is **not used during training**. It operates exclusively during the **sampling/inference** phase (Trajectory Integration) to enforce constraints on the generated "ghost" paths.

---

## 2. Comparison: FM-PCC vs. Two-Step Safety Fine-Tuning

The following table summarizes the key differences between the **FM-PCC** approach and the **Two-Step Safety Fine-Tuning** procedure (as seen in recent "Safe Flow Matching" research).

| Feature | FM-PCC (Inference-time Safety) | Two-Step Fine-Tuning (Training-time Safety) |
| :--- | :--- | :--- |
| **Training Stages** | **One-Step:** Train the Prior on raw data. | **Two-Step:** 1) Train base model. 2) Fine-tune on safety dataset. |
| **Dataset Construction** | Uses **original expert demonstrations** ($D$) directly. | Creates a **synthetic safety dataset** by projecting each demo in $D$ onto the safe manifold ($M_{safe}$). |
| **Safety Injection** | **External (Control):** Constraints are enforced during the generation loop via projection. | **Internal (Weights):** The model "learns" to be safe by training on projected/clamped data. |
| **Computation** | Fast training. Projection overhead is shifted to **inference** steps. | Slow training. Requires **finite differences** for the projection derivative during fine-tuning. |
| **Flexibility** | **High:** Constraints can be changed at inference without retraining the model. | **Low:** Changes to obstacles or safety rules require a new fine-tuning phase. |

---

## 3. Detailed Architectural Comparison

### Feature 1: The Training Objective
In **FM-PCC**, the model's loss function (`p_losses`) minimizes the error between predicted velocity/noise and the *raw* expert target. In the **Two-Step** approach, the fine-tuning step minimizes the error between the model output and the *projected* target, effectively mapping the flow toward the safe region $M_{safe}$.

### Feature 2: Dataset Adaptation
The user-provided snippet describes $p_1(q(t) \mid O(t))$ being computed by projecting each sample. In FM-PCC, the data samples in `SequenceDataset` are **never projected**. The dataset is purely representative of expert behavior, which may or may not satisfy new constraints introduced at test time.

### Feature 3: The Role of Derivatives
A critical technical difference is the use of the **projection operator's derivative**:
- **Two-Step Fine-Tuning:** Needs the derivative of the projection with respect to the input (often via finite differences) to backpropagate safety through the network.
- **FM-PCC:** Uses the projector directly on the sampled points. If gradients are used, they are used to **guide the integration** (ODE step) during sampling, not to update the model weights.

---

> [!NOTE]
> **Conclusion:**
> FM-PCC follows a "Safety via Control" philosophy, whereas the two-step procedure follows a "Safety via Mapping" philosophy. FM-PCC is better suited for scenarios where the environment or safety rules change frequently, as it avoids the expensive safety-specific fine-tuning step.
