# 05.4: Mission Briefing — Standardizing the V4 Deterministic Suite

## 🕵️ Mission Objective
The V4 suite establishes a "Clean Room" benchmarking environment by eliminating spatial variance from our performance metrics. By moving noise generation outside the integration loop, we ensure that every solver trial is solving the **exact same mathematical problem**.

---

## ⚡ Key Principles

### 1. Deterministic Algorithmic Auditing (Math Mode)
- **Protocol**: Locked Noise Basis.
- **Goal**: Zero "Path Luck." 
- **Effect**: Every trial produces bit-identical trajectories. Latency results reflect only algorithmic complexity and hardware throughput.

### 2. Robustness Validation (Production Mode)
- **Protocol**: Random per Trial.
- **Goal**: Statistical coverage of the state space.
- **Effect**: Mirrors real-world deployment where starting states are unpredictable.

---

## 🛠️ The V4 Toolset

The following files have been migrated to the `/v4` directory with these principles enforced:

1.  **`benchmark_ode_solvers_v4.py`**: Latency auditor with deterministic math support.
2.  **`benchmark_ode_accuracy_v4.py`**: Drift auditor with unified global noise for Oracle/Candidate alignment.
3.  **`grid_search_benchmark_for_v4.py`**: Orchestrator with fixed-seed subprocess propagation.
4.  **`grid_search_accuracy_v4.py`**: Multi-dimensional accuracy grid mapper.

---

## 📋 Conclusion
The V4 suite is now the official standard for the Safety Projector selection process. We have moved from "Statistical Throughput Estimation" to **"Algorithmic Auditing."**
