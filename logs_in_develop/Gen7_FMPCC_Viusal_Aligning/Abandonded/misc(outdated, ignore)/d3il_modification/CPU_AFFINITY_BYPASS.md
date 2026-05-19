# D3IL CPU Affinity Bypass for Visual Aligning

This document records the modification made to the D3IL simulation framework to resolve deadlocks and process freezing in high-fidelity visual evaluation.

---

## 🔴 The Problem

During evaluation runs with high diffusion steps (e.g., $K=100$), the evaluation script hung indefinitely at `Context 0 Rollout 0` on many-core CPU systems (e.g., 64 CPU cores).

### Root Cause Analysis:
1. **CPU Pinning Starvation**:
   - The legacy D3IL simulation pins each evaluation process to exactly one CPU core via `os.sched_setaffinity` (yielding `cpu_set = {0}` in single-core runs).
   - This works perfectly for lightweight state-only tasks. However, **visual aligning** is extremely heavy: it runs MuJoCo OpenGL camera rendering, GPU-based deep neural networks (ResNet feature extraction and 1D Temporal U-Net diffusion steps), CUDA context execution/polling, and PyTorch dataloading.
   - Forcing all of these background threads, OpenMP helpers, and OpenGL drivers to share a **single CPU core** causes critical thread starvation, OpenMP spin-lock deadlocks, and CUDA freezes.
2. **SciPy SLSQP Noise Search Complexity**:
   - Setting $K=100$ diffusion steps extends the DPCC projection activation threshold to timestep $50$.
   - In early timesteps (t=50 to t=30), the trajectory is noisy and highly disordered.
   - SciPy's SLSQP optimizer must perform hundreds of iterations to compute a mathematically feasible trajectory satisfying dynamic constraints. This highly intensive CPU computation starves all remaining worker/GPU threads on Core 0, triggering a permanent freeze.

---

## 🛠️ The Modification

We modified the core simulation wrapper `d3il/simulation/aligning_sim.py` to bypass CPU affinity pinning specifically for high-fidelity visual evaluation:

* **File**: [d3il/simulation/aligning_sim.py](../../../d3il/simulation/aligning_sim.py#L50-L60)
* **Change**:
  ```diff
  def eval_agent(self, agent, contexts, n_trajectories, mode_encoding, successes, mean_distance, pid, cpu_set):
  
          print(os.getpid(), cpu_set)
-         assign_process_to_cpu(os.getpid(), cpu_set)
+         # For visual aligning, we MUST NOT pin the process to a single CPU core.
+         # Visual rollouts utilize heavy GPU PyTorch workers, OpenMP threads, and MuJoCo rendering.
+         # Pinning everything to a single CPU core causes thread starvation, OpenMP deadlocks, and GPU freezes.
+         if not self.if_vision:
+             assign_process_to_cpu(os.getpid(), cpu_set)
+         else:
+             print(f"Process {os.getpid()} is running unpinned to utilize all available CPU threads safely!")
  
          env = Robot_Push_Env(render=self.render, if_vision=self.if_vision, max_steps_per_episode=self.max_episode_length)
  ```

---

## 📊 Summary of Benefits
* **Legacy Preservation**: Retains exact benchmark CPU affinity parity for D3IL's lightweight state-only tasks (`self.if_vision = False`).
* **Deadlock Elimination**: Allows PyTorch multi-threading, OpenMP workers, and OpenGL/MuJoCo drivers to naturally distribute their workloads across all 64 cores.
* **Massive Speedups**: Increases the execution speed of multi-candidate projection and SciPy optimization pipelines by utilizing the entire host CPU.
