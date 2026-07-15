"""Closed-form validation of the HF-iMF deep-mix theory (1D, no training needed).

Model: linear interpolant z_tau = tau*x1 + (1-tau)*x0,
  x0 ~ N(0,1),  x1 ~ 1/2 N(+m, sig^2) + 1/2 N(-m, sig^2)   (two modes),
  constraint C = { |x1| >= c }  (obstacle slab in the middle — 1D avoiding).

Everything is analytic: the marginal velocity v(z,tau) has a closed form, the
PF-ODE can be integrated to machine precision, so the *exact* flow endpoint
F(z,tau) and the *exact* posterior mean PM(z,tau) are both available.

Validates three claims:
  (1) HardFlow's Euler extrapolation x1_hat = z + (1-tau) v(z,tau) IS the
      posterior mean E[x1|z_tau] (mode average) — identically, not approximately —
      and it diverges from the true flow endpoint F at small tau.
  (2) HardFlow's pull-back gain tau (z' = z + tau*Delta) under-delivers the
      requested endpoint change; the Newton gain 1/F'(z) delivers it exactly.
  (3) On the obstacle task, constraining the posterior mean early ('all')
      corrupts samples that were never going to violate; constraining the
      exact endpoint F corrects only actual violators, minimally.
"""
import numpy as np

rng = np.random.default_rng(0)
M, SIG, C = 1.0, 0.35, 0.5   # modes at +-1, mode width, obstacle |x1| < 0.5
N_ODE = 400                   # RK4 steps for exact integration


def v_field(z, tau):
    """Exact marginal velocity E[x1 - x0 | z_tau = z] for the GMM target."""
    tau = np.asarray(tau, dtype=float)
    s2 = (tau * SIG) ** 2 + (1.0 - tau) ** 2          # per-component var of z_tau
    out = np.zeros_like(z)
    wsum = np.zeros_like(z)
    num = np.zeros_like(z)
    for mu in (+M, -M):
        w = np.exp(-((z - tau * mu) ** 2) / (2.0 * s2))
        # E[x1|z,comp] - E[x0|z,comp] = mu + (tau*SIG^2 - (1-tau)) * (z - tau*mu)/s2
        cond = mu + (tau * SIG ** 2 - (1.0 - tau)) * (z - tau * mu) / s2
        num += w * cond
        wsum += w
    return num / wsum


def posterior_mean(z, tau):
    """Exact E[x1 | z_tau = z] (Tweedie / x-prediction)."""
    tau = np.asarray(tau, dtype=float)
    s2 = (tau * SIG) ** 2 + (1.0 - tau) ** 2
    num = np.zeros_like(z)
    wsum = np.zeros_like(z)
    for mu in (+M, -M):
        w = np.exp(-((z - tau * mu) ** 2) / (2.0 * s2))
        cond = mu + (tau * SIG ** 2) * (z - tau * mu) / s2
        num += w * cond
        wsum += w
    return num / wsum


def flow_endpoint(z, tau0, n=N_ODE):
    """Exact F(z, tau0): integrate the PF-ODE from tau0 to 1 with RK4 (batched)."""
    z = np.array(z, dtype=float, copy=True)
    taus = np.linspace(tau0, 1.0, n + 1)
    for k in range(n):
        t, dt = taus[k], taus[k + 1] - taus[k]
        k1 = v_field(z, t)
        k2 = v_field(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = v_field(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = v_field(z + dt * k3, t + dt)
        z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def flow_jac(z, tau0, h=1e-5):
    """dF/dz by central difference (F is exact, so this is accurate)."""
    return (flow_endpoint(z + h, tau0) - flow_endpoint(z - h, tau0)) / (2 * h)


def project(x1):
    """Euclidean projection onto C = {|x| >= C}. sign(0) := +1 (the ill-posed tie)."""
    s = np.where(x1 >= 0, 1.0, -1.0)
    return np.where(np.abs(x1) >= C, x1, s * C)


print("=" * 72)
print("(1) Euler extrapolation == posterior mean;  both vs true endpoint F")
print("=" * 72)
zs = np.linspace(-2, 2, 9)
for tau in (0.05, 0.3, 0.6, 0.9):
    euler = zs + (1 - tau) * v_field(zs, tau)
    pm = posterior_mean(zs, tau)
    F = flow_endpoint(zs, tau)
    print(f" tau={tau:4.2f}  max|Euler-PM| = {np.max(np.abs(euler - pm)):.2e}"
          f"   mean|PM-F| = {np.mean(np.abs(pm - F)):.3f}"
          f"   mean|F| = {np.mean(np.abs(F)):.3f}  mean|PM| = {np.mean(np.abs(pm)):.3f}")

print()
print("=" * 72)
print("(2) Pull-back gain: request endpoint change Delta, measure achieved/requested")
print("=" * 72)
delta = 0.3
z0 = rng.standard_normal(2000)
print("  tau   HardFlow gain tau -> achieved/req      Newton 1/F' -> achieved/req")
for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
    z_tau = None
    # transport a batch of noise draws to time tau (integrate 0 -> tau)
    z = np.array(z0, copy=True)
    taus = np.linspace(0.0, tau, 200 + 1)
    for k in range(200):
        t, dt = taus[k], taus[k + 1] - taus[k]
        k1 = v_field(z, t); k2 = v_field(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = v_field(z + 0.5 * dt * k2, t + 0.5 * dt); k4 = v_field(z + dt * k3, t + dt)
        z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    F0 = flow_endpoint(z, tau)
    # HardFlow update
    ach_hf = flow_endpoint(z + tau * delta, tau) - F0
    # Newton update
    J = flow_jac(z, tau)
    ach_nt = flow_endpoint(z + delta / J, tau) - F0
    print(f"  {tau:4.2f}        {np.mean(ach_hf / delta):6.3f}"
          f"                          {np.mean(ach_nt / delta):6.3f}")

print()
print("=" * 72)
print("(3) Obstacle task |x1| >= 0.5 : who corrupts the innocent?")
print("=" * 72)
NS = 4000
z0 = rng.standard_normal(NS)
x1_free = flow_endpoint(z0, 0.0)                 # unconstrained samples
innocent = np.abs(x1_free) >= C                  # would never have violated
print(f"  unconstrained violation rate: {np.mean(~innocent):.3f}"
      f"   (these are the only samples that NEED correction)")

# exact reference: true conditional distribution via rejection
x_ref = x1_free[innocent]

def run_hardflow(z0, N=20, activation="all"):
    z = np.array(z0, copy=True)
    dtau = 1.0 / N
    corrections = np.zeros_like(z)
    for k in range(N):
        t = k * dtau
        z = z + v_field(z, t) * dtau              # Euler ref step (their step 1)
        t1 = t + dtau
        if activation == "late" and k < N // 2:
            continue
        x1_hat = z + (1 - t1) * v_field(z, t1)    # Euler extrapolation == PM
        X1s = project(x1_hat)
        corr = t1 * (X1s - x1_hat)                # tau-gain pull-back
        z = z + corr
        corrections += np.abs(corr)
    return z, corrections

def run_mf_newton(z0, anchors=(0.5, 1.0)):
    z = np.array(z0, copy=True)
    t_prev = 0.0
    corrections = np.zeros_like(z)
    for t1 in anchors:
        # exact interval jump (stand-in for the learned u-head)
        z_jump = flow_endpoint(z, t_prev) if t1 == 1.0 else None
        if t1 == 1.0:
            z = z_jump
        else:
            # integrate t_prev -> t1
            taus = np.linspace(t_prev, t1, 200 + 1)
            for k in range(200):
                t, dt = taus[k], taus[k + 1] - taus[k]
                k1 = v_field(z, t); k2 = v_field(z + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = v_field(z + 0.5 * dt * k2, t + 0.5 * dt); k4 = v_field(z + dt * k3, t + dt)
                z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        F = z if t1 == 1.0 else flow_endpoint(z, t1)
        X1s = project(F)
        if t1 == 1.0:
            corr = X1s - F
            z = X1s
        else:
            J = flow_jac(z, t1)
            corr = (X1s - F) / J
            z = z + corr
        corrections += np.abs(corr)
        t_prev = t1
    return z, corrections

for name, (xs, corr) in {
    "HardFlow-all  (N=20, PM ref, tau-gain)": run_hardflow(z0, 20, "all"),
    "HardFlow-late (N=20, PM ref, tau-gain)": run_hardflow(z0, 20, "late"),
    "MF-Newton K=2 (exact F, Newton gain)  ": run_mf_newton(z0, (0.5, 1.0)),
    "MF-project K=1 (exact F at end)       ": run_mf_newton(z0, (1.0,)),
}.items():
    viol = np.mean(np.abs(xs) < C - 1e-9)
    dist_innocent = np.mean(np.abs(xs[innocent] - x1_free[innocent]))
    corr_innocent = np.mean(corr[innocent])
    # W1 distance to the exact conditional, sample-based
    a = np.sort(xs); b = np.sort(rng.choice(x_ref, size=NS))
    w1 = np.mean(np.abs(a - b))
    print(f"  {name}: viol={viol:.3f}  |move on innocent|={dist_innocent:.4f}"
          f"  |corr applied to innocent|={corr_innocent:.4f}  W1(true cond)={w1:.4f}")
