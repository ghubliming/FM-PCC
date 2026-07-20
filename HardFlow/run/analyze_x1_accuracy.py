"""Gen13 fix_7.3 — T4: DIRECT test of the seam swap (x̂1 accuracy vs τ).

The Gen13 thesis is a claim about ONE quantity: how accurately the terminal
prediction x̂1 estimates where the trajectory actually ends up.

    FM  : x̂1(k) = x_k + (1-τ_k)·v(x_k, τ_k)          Euler shot, error O((1-τ)²)
    iMF : x̂1(k) = x_k + (1-τ_k)·u(x_k, τ_k, 1-τ_k)   exact endpoint map

This measures that error directly, per ODE step, using the chains already dumped
by run/eval_imf.py (`chain_full`, `x1_full`):

    err(k) = || x̂1(k) - x_final ||   where x_final = chain[-1] (what it converged to)

PREDICTION if the thesis holds: FM's err(k) is large at k=0 and decays ~(1-τ)²;
iMF's is low and roughly FLAT in τ. If instead iMF's err(0) is as large as FM's,
the exact endpoint map is NOT delivering its advantage in practice — which would
mean the u-field's training error dominates, and would undercut the Gen13 story.

Pure POST-PROCESSING: no GPU, no simulator, no model.

Usage:
    python run/analyze_x1_accuracy.py --dirs <dir1> <dir2> ... [--out plot.png]
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _episode_errors(npz_path):
    """Per-ODE-step x̂1 error, averaged over the replans in this episode."""
    d = np.load(npz_path, allow_pickle=True)
    if "chain_full" not in d:
        return None, None
    chain, x1 = d["chain_full"], d["x1_full"]   # (n_replans, n_ode+1, H, T)
    ad = int(d["action_dim"])
    px, py = ad + 2, ad + 3
    n_replans, n_ode = chain.shape[0], chain.shape[1]
    errs = np.zeros((n_replans, n_ode))
    for r in range(n_replans):
        final = chain[r, -1]                      # what the plan converged to
        for k in range(n_ode):
            d2 = (x1[r, k][:, [px, py]] - final[:, [px, py]]) ** 2
            errs[r, k] = np.sqrt(d2.sum(axis=-1)).mean()   # mean L2 over horizon
    tau = np.linspace(0.0, 1.0, n_ode)
    return tau, errs.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.dirs]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    print(f"{'run':<44}{'err(tau=0)':>12}{'err(tau=1)':>12}{'decay':>9}")
    print("-" * 78)
    for d, lab in zip(args.dirs, labels):
        curves = []
        for p in sorted(glob.glob(os.path.join(d, "*_fan.npz"))):
            tau, e = _episode_errors(p)
            if e is not None:
                curves.append(e)
        if not curves:
            print(f"{lab:<44}  (no chain_full — re-run with IMF_PLOT_FAN=1)")
            continue
        # episodes may differ in n_ode only across backbones, not within
        e = np.mean(np.stack(curves), axis=0)
        tau = np.linspace(0, 1, len(e))
        ax.plot(tau, e, "-o", markersize=4, label=lab)
        decay = e[0] / max(e[-1], 1e-12)
        print(f"{lab:<44}{e[0]:>12.4e}{e[-1]:>12.4e}{decay:>9.1f}x")

    ax.set_xlabel(r"ODE progress $\tau$  (0 = noise, 1 = data)", fontsize=12)
    ax.set_ylabel(r"mean $\|\hat{x}_1(\tau) - x_{\rm final}\|$", fontsize=12)
    ax.set_title(
        "Terminal-prediction accuracy vs ODE progress\n"
        r"(FM: Euler shot, error $O((1-\tau)^2)$   vs   iMF: exact endpoint map)",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = args.out or "x1_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved: {out}")
    print("READ: if the Gen13 thesis holds, iMF should be LOW and FLAT while FM "
          "starts HIGH at tau=0 and decays. If iMF's tau=0 error matches FM's, "
          "the exact endpoint map is not delivering in practice.")


if __name__ == "__main__":
    main()
