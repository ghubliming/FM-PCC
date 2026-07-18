"""Gen13 — gates G0 + G1 for the iMF package. Run BEFORE any real training.

    python run/imf_gates.py          (cluster login node is fine — CPU, <2 min)

G0 — mechanics: TemporalImfUnet builds and (x, tau, h) -> (u, v) shapes flow
     for the real avoiding configuration (H16, transition 6, dim_mults (1,4,8)).

G1 — end-to-end convention/loss/sampler validation on a 1D two-mode GMM
     (modes +/-2), the style of Research/validate_theory.py. A tiny MLP with the
     same (x, tau, h) -> (u, v) interface is trained with ImfMatcher and then:
       A. h->0 limit: u(z, tau, h~0) matches the v-head        (identity wiring)
       B. 1-NFE generation covers both modes near +/-2         (SIGN gate: a
          flipped convention walks toward noise and fails loudly here)
       C. K=1 vs K=2 endpoint distributions agree (W1)         (K-invariance)
       D. exact-jump consistency: one K=1 jump == two K=2 half-jumps (W1)

Exit code 0 = all gates passed; non-zero otherwise. No files are written.
"""

import sys

import torch
import torch.nn as nn

from hardflow.models_flow.imf import ImfMatcher, TemporalImfUnet
from hardflow.models_flow.imf.imf_sampler import imf_sample

torch.manual_seed(0)

FAILURES = []


def check(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"[ {status} ] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


# ----------------------------------------------------------------- G0: shapes
def gate_g0():
    print("\n=== G0: TemporalImfUnet mechanics (avoiding config) ===")
    net = TemporalImfUnet(
        horizon=16, transition_dim=6, cond_dim=4, dim=32, dim_mults=(1, 4, 8)
    )
    x = torch.randn(3, 16, 6)
    for tau, h in [(0.0, 0.5), (torch.rand(3), torch.rand(3) * 0.3), (1.0, 0.0)]:
        u, v = net(x, tau, h)
        check(
            "G0 shapes",
            u.shape == x.shape and v.shape == x.shape,
            f"tau={type(tau).__name__}: u{tuple(u.shape)} v{tuple(v.shape)}",
        )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[ info ] TemporalImfUnet params: {n_params/1e6:.2f}M")


# ------------------------------------------------------------ G1: 1D GMM test
class ToyImfNet(nn.Module):
    """Minimal (x, tau, h) -> (u, v) net for 1D 'trajectories' (B, 1, 1)."""

    def __init__(self, width=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, x, tau, h):
        b = x.shape[0]
        def as_b(val):
            if not torch.is_tensor(val):
                val = torch.tensor(float(val))
            val = val.to(x.device, x.dtype)
            while val.dim() > 1:
                val = val[..., 0]
            if val.dim() == 0:
                val = val.repeat(b)
            return val
        tau, h = as_b(tau), as_b(h)
        inp = torch.stack([x.reshape(b), tau, h], dim=-1)
        out = self.mlp(inp)
        u, v = out[:, :1], out[:, 1:]
        return u.reshape(b, 1, 1), v.reshape(b, 1, 1)


def gmm_sample(n):
    modes = torch.where(torch.rand(n) < 0.5, -2.0, 2.0)
    return (modes + 0.25 * torch.randn(n)).reshape(n, 1, 1)


def w1_1d(a, b):
    return (torch.sort(a.flatten()).values - torch.sort(b.flatten()).values).abs().mean()


def gate_g1():
    print("\n=== G1: 1D GMM end-to-end (convention / loss / sampler) ===")
    net = ToyImfNet()
    matcher = ImfMatcher(model=net, action_dim=1, data_proportion=0.25,
                         p_mean=-0.4, p_std=1.4)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    steps = 4000
    for i in range(steps):
        x1 = gmm_sample(256)
        loss, infos = matcher.loss(x1, {})
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 1000 == 0:
            print(
                f"[ train ] step {i}: raw_mse_u {infos['raw_mse_u']:.4f} "
                f"raw_mse_v {infos['raw_mse_v']:.4f}"
            )

    net.eval()
    n = 4000
    ref = gmm_sample(n)

    with torch.no_grad():
        # A: h->0 limit — u must approach the v-head
        z = gmm_sample(512) * 0.5
        tau = torch.full((512,), 0.5)
        u_small, v_small = net(z, tau, torch.full((512,), 1e-3))
        rel = ((u_small - v_small) ** 2).mean() / ((v_small ** 2).mean() + 1e-8)
        check("G1-A h->0 (u ~= v)", rel.item() < 0.05, f"rel err {rel.item():.4f}")

        # B: 1-NFE generation — THE sign gate
        x0 = torch.randn(n, 1, 1)
        x_gen1 = imf_sample(net, x0.clone(), {}, 1, 1)
        mean_abs = x_gen1.abs().mean().item()
        frac_near = ((x_gen1.abs() - 2.0).abs() < 0.75).float().mean().item()
        both_modes = min(
            (x_gen1 > 1.0).float().mean().item(), (x_gen1 < -1.0).float().mean().item()
        )
        w1_k1 = w1_1d(x_gen1, ref).item()
        check(
            "G1-B 1-NFE lands on data (sign)",
            abs(mean_abs - 2.0) < 0.5 and frac_near > 0.8 and both_modes > 0.2,
            f"mean|x| {mean_abs:.3f} (target ~2), near-mode frac {frac_near:.2f}, "
            f"min-mode frac {both_modes:.2f}, W1 {w1_k1:.3f}",
        )

        # C: K=1 vs K=2 distribution agreement
        x_gen2 = imf_sample(net, x0.clone(), {}, 1, 2)
        w1_k12 = w1_1d(x_gen1, x_gen2).item()
        check("G1-C K1 ~ K2 (W1)", w1_k12 < 0.25, f"W1(K1,K2) {w1_k12:.3f}")

        # D: composition — one full jump vs two half jumps from the same noise
        z0 = torch.randn(1000, 1, 1)
        u_full, _ = net(z0, 0.0, 1.0)
        one_jump = z0 + u_full
        u_h1, _ = net(z0, 0.0, 0.5)
        mid = z0 + 0.5 * u_h1
        u_h2, _ = net(mid, 0.5, 0.5)
        two_jump = mid + 0.5 * u_h2
        w1_comp = w1_1d(one_jump, two_jump).item()
        check("G1-D jump composition (W1)", w1_comp < 0.25, f"W1 {w1_comp:.3f}")


if __name__ == "__main__":
    gate_g0()
    gate_g1()
    print()
    if FAILURES:
        print(f"GATES FAILED: {FAILURES} — do NOT proceed to training/eval.")
        sys.exit(1)
    print("ALL GATES PASSED — cleared for Phase 2 (training).")
    sys.exit(0)
