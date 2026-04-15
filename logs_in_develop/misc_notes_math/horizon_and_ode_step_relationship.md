# Relationship: Planning Horizon ($H$) vs. ODE Integration Steps ($N$)

This note explains the structural and mathematical relationship between the trajectory horizon and the number of ODE integration steps in the FM-PCC framework.

## 1. Why `horizon = 8` was a Hard Limit (Structural Reason)

Previously, attempts to change the horizon (e.g., setting $H=10$ or $H=20$) resulted in shape mismatch errors during the forward pass of the U-Net.

### The $2^L$ Halving Constraint
The `models.Flow_matcher_U_Net_v2` (and its predecessors) uses a temporal U-Net architecture with downsampling layers. In the current configuration:
- **`dim_mults`**: `(1, 2, 4, 8)`
- **Downsampling Layers ($L$)**: 3 (each doubling of the `dim_mult` typically corresponds to a temporal halving via convolution with stride 2 or pooling).

Mathematically, the temporal sequence length must be divisible by $2^L$ to ensure that:
1. The sequence can be halved $L$ times without resulting in fractional dimensions.
2. The **Skip Connections** can align perfectly between the downsampling path and the upsampling path.

$$ H \pmod{2^L} = 0 \implies H \pmod{8} = 0 $$

Thus, valid horizons are $8, 16, 24, 32, \dots$. Choosing $H=8$ was the minimum viable multiple for this architecture.

## 2. Relationship between Horizon ($H$) and ODE Steps ($N$)

There is a common confusion between the **Horizon** (environmental time/sequence length) and **ODE Steps** (numerical integration resolution).

| Parameter | Symbol | Definition | Scope |
| :--- | :---: | :--- | :--- |
| **Horizon** | $H$ | Number of future states/actions predicted in one shot. | **Space Dimension**: $\mathbf{x} \in \mathbb{R}^{H \times D}$ |
| **ODE Steps** | $N$ | Number of discretization steps used to solve the flow from noise ($t=0$) to data ($t=1$). | **Time Resolution**: $\Delta t = 1/N$ |

### 2.1 Mathematical Decoupling
In Flow Matching, we solve an ODE in the space of *trajectories*. 
$$ \frac{d\mathbf{x}}{dt} = \mathbf{v}_\theta(\mathbf{x}, t, \mathbf{c}) $$
Here, $\mathbf{x}$ is the **entire trajectory** of length $H$. The ODE steps $N$ determine how accurately we follow the "probability flow" from the Gaussian distribution to the data distribution.
- Increasing $H$ makes the state space higher-dimensional.
- Increasing $N$ makes the numerical integration more precise.

### 2.2 Numerical and Conceptual Sensitivity
While mathematically decoupled, they are linked by **error propagation and signal complexity**:
1. **Dimensionality vs. Stability**: As $H$ increases, the vector field $\mathbf{v}_\theta$ lives in a much higher-dimensional space ($\mathbb{R}^{H \times D}$). High-dimensional flows can be more "brittle" or sensitive to discretization errors.
2. **Trajectory Complexity**: A longer horizon ($H$) typically covers more physical time in the environment. Trajectories over longer durations often have more "turns," intersections, or complex dynamical transitions. Resolving these "details" in the flow field may require a higher number of ODE steps ($N$) to maintain the same level of accuracy as as shorter-horizon predictions.
3. **Accuracy Baselining**: If the model was trained with $N_{\text{train}}$ steps (or continuous-time flow matching), but evaluated with a small $N_{\text{eval}}$, the discretization error will likely manifest more severely on longer horizons where small angular errors in velocity prediction have more "space" ($H$) to propagate into large positional errors.

## 3. The $O(N \cdot H)$ Computational Coupling

In the actual implementation, the total inference cost is tightly coupled across $N$ and $H$.

### 3.1 Total Complexity
The evaluation loop (`p_sample_loop`) performs $N$ iterations. In each iteration, it performs a forward pass of the 1D Temporal U-Net. The complexity of a 1D Convolution over the sequence is linear with the sequence length ($H$).
$$ \text{Total Cost} \approx N \times H \times (\text{Channel Width}) $$

### 3.2 The Projection Bottleneck
Crucially, the evaluation breaks the ODE integration into $N$ separate "chunks" to allow for **State Constraint Projection** at each step. 
- Each of these $N$ projections must process the entire sequence of length $H$.
- If you double the horizon ($H=16$), each of the $20$ integration steps becomes ~2x more expensive in both neural network evaluation AND the projection optimization.

## 4. Summary for ODE Tests
- **Don't use $H < 8$**: It will break the U-Net skip connections.
- **Use multiples of 8**: If testing beyond $H=8$, only use $16, 24, 32$.
- **Baseline for Benchmarking**: Since the model was trained with $H=8$, this is the most numerically stable "ground truth" for evaluating different ODE solvers (Euler vs. RK4, etc.). Testing a solver on $H=16$ with a model trained on $H=8$ may introduce out-of-distribution artifacts that aren't the fault of the solver itself.
