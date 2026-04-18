# DPCC Architecture: Safety Integrity vs. Computational Bottlenecks

This document audits the **DPCC (Diffusion Policy with Control Constraints)** architecture, specifically examining the trade-off between robot safety and real-time latency.

---

## 1. The Core Mandate: Why DPCC is Mandatory

Based on the audit of the original `diffuser` and current `flow_matcher_v3` codebases, the system overhead is not "bloat"—it is the fundamental requirement for **Physical Correctness**.

### 1.1 Per-Step Safety Projections (`projector.project`)
In standard Diffusion or Flow Matching, the solver moves from noise to data in a straight mathematical line. However, in a constrained environment (e.g., a robot arm avoiding a wall), that mathematical line might pass through a "Forbidden Zone."
*   **The DPCC Solution**: At every single step of the ODE integration, the state is projected back onto the **Safety Manifold**. 
*   **The Cost**: This adds a mandatory "Check & Fix" cycle to every solver step, preventing the math from ever detaching from physical reality.

### 1.2 Multi-Pass Conditioning (`apply_conditioning`)
The robot's current state (Observations) is the "Anchor" of the entire trajectory.
*   **The DPCC Solution**: The architecture reapplies current observations up to **twice per step** during the integration.
*   **The Cost**: This ensures that even if the ODE solver "drifts" during the math, the start of the plan remains locked to the robot's actual hand position.

---

## 2. The Bottleneck: Busted Myths and Hard Truths

Our Benchmark V3 and "Fair Mirror" tests completely rewrote our understanding of the system's latency. We previously thought "Python Slicing" was dominating the runtime. **Empirical data proved this false.**

| Level | Component | Est. Latency (B4, H8) | Character |
| :--- | :--- | :--- | :--- |
| **L1: Math** | U-Net Forward Pass | ~11ms per Euler call | **Heavy** GPU Bound |
| **L2: Safety** | Projector (SLSQP/SQP) | *Highly Variable* | **Severe** Scipy/Optimization Bound |
| **L3: Orchestration**| Python Slicing / Dicts | **~0.5ms per step** | Lightweight CPU Bound |

### The "Myth of the Python Tax"
In V3, running the Production loop *without* the Line 2 Projector (Level 2) added only **~5ms total over 10 steps** compared to raw math. The Python dictionary orchestration (Level 3) is highly efficient. The latency explosion is exclusively caused by the Sequential Handoff to the Safety Projector and the raw GPU math footprint.

---

## 3. The New Scaling Paradox: The Projection Tax

Because the DPCC architecture applies the safety projector at every step, it creates a unique paradox when choosing ODE solvers:

1.  **Euler's Hazard**: Euler uses $1 \times$ U-Net math per step. Because it drifts, the intermediate state strongly violates collision boundaries. The Safety Projector (SLSQP) takes the brunt of the hit, spending massive iteration time trying to fix both the obstacle collision AND the numerical drift.
2.  **RK4's Shield**: RK4 uses $4 \times$ U-Net math per step (which is mathematically expensive). However, the intermediate state is highly accurate and feasible. The Safety Projector converges almost instantly.
    *   **The Payoff**: It is physically faster to run **10 Steps of RK4** (expensive physics, cheap projections) than **40 Steps of Euler** (cheap physics, extremely expensive and failing projections).

> [!IMPORTANT]
> **Conclusion**: In the DPCC architecture, the **Number of Projector Calls ($S$) and the Feasibility of the Proposed State** dictatates the true latency limit. High-order solvers like RK4 allow us to reduce $S$ drastically, sparing the system from the crippling SLSQP optimization bottleneck.

---

## 4. Engineering Outlook: Achieving 20Hz

To reach a 20Hz control loop (50ms budget) while keeping the DPCC safety mandate intact, we must follow these architectural rules:

*   **Avoid "Pure Math" loops**: We cannot strip the safety boilerplate; the robot will hit constraints.
*   **Minimize $S$ (Steps)**: Every step saved is ~20ms saved. 
*   **Leverage RK4**: Use higher-order solvers to enable ultra-low step counts ($S \le 5$) without losing trajectory fidelity.

**Verdict**: The DPCC architecture is safe but "heavy." The only way to move fast is to move **smarter** using RK4, rather than removing the essential safety guardrails.
