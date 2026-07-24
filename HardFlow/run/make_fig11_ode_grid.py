"""Gen13 u_8.2 — paper-Fig.11-style ODE-step grid (the generation process).

Reproduces the STRUCTURE of HardFlow's appendix Fig. 11 (a 2 x N grid tracking
one planning instance across ODE steps), reusing the existing plotter:

    row 1 : x_tau    — the intermediate sample at each ODE step
                       (chaotic near tau=0, collapsing to a clean path at tau=1)
    row 2 : x1_hat   — the terminal prediction decoded from that state

With --both, two runs are stacked into a 4 x N grid (iMF rows then FM rows) —
which makes the Gen13 seam swap visible: FM's x1_hat is an Euler shot
z + (1-tau)*v (the paper notes it is deformed at early steps), while iMF's is
the EXACT endpoint map z + (1-tau)*u and should be well localised from step 0.

NOT a pixel-perfect replication (deliberate): the paper also draws a dashed
ground-truth "future" reference, which HardFlow's eval does not carry. Past =
executed rollout is drawn instead.

Pure POST-PROCESSING — reads the `*_fan.npz` dumps written by run/eval_imf.py
(u_8.2 keeps the full chains), so no GPU / simulator / model is needed.

Usage:
    python run/make_fig11_ode_grid.py --dir <eval_dir> --run_id 0 --out grid.png
    python run/make_fig11_ode_grid.py --dir <imf_dir> --dir2 <fm_dir> --both ...
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hardflow.utils.rendering import AvoidingTrajectoryPlotter


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "chain_full" not in d:
        raise KeyError(
            f"{npz_path} has no 'chain_full' — it predates u_8.2. Re-run the "
            f"diagnostic with IMF_PLOT_FAN=1 to capture the full ODE chains."
        )
    return {
        "chain": d["chain_full"],   # (n_replans, n_ode+1, H, T)
        "x1": d["x1_full"],
        "real": d["real"],
        "action_dim": int(d["action_dim"]),
        "backbone": str(d["backbone"]),
    }


def _cell(ax, traj, real, plotter, ad, title, show_y):
    """One grid cell: environment + the executed rollout + one trajectory."""
    plotter._configure_axis(ax, show_x_label=False, show_y_label=show_y, compact=True)
    # past/context: the executed rollout (paper draws past+future reference here)
    ax.plot(real[:, 2], real[:, 3], "-", color="0.75", linewidth=2.0, zorder=1)
    # the trajectory for this ODE step — upstream Fig.11 styling (one plan/cell,
    # which is exactly the use case that style was designed for)
    plotter.plot_single_trajectory(traj[:, ad:], ax, style="predicted",
                                   show_labels=False)
    plotter.add_environment_elements(ax)
    ax.set_xlabel("")
    ax.set_ylabel(ax.get_ylabel(), fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)


def _rows_for(data, plan_idx, n_cols):
    """Pick the replan instance and subsample ODE steps to n_cols columns."""
    n_plans, n_ode = data["chain"].shape[0], data["chain"].shape[1]
    p = n_plans // 2 if plan_idx is None else max(0, min(plan_idx, n_plans - 1))
    cols = np.unique(np.linspace(0, n_ode - 1, min(n_cols, n_ode)).astype(int))
    return p, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="eval dir (iMF if --both)")
    ap.add_argument("--dir2", default=None, help="second eval dir (FM) for --both")
    ap.add_argument("--both", action="store_true", help="stack two runs (4 rows)")
    ap.add_argument("--run_id", type=int, default=0)
    ap.add_argument("--plan_idx", type=int, default=None,
                    help="which replan instance (default: middle)")
    ap.add_argument("--n_cols", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--constraint", default="novel")
    ap.add_argument("--obstacle_margin", type=float, default=0.02)
    args = ap.parse_args()

    sources = [("iMF", args.dir)]
    if args.both:
        if not args.dir2:
            ap.error("--both requires --dir2")
        sources.append(("FM", args.dir2))

    loaded = []
    for label, d in sources:
        p = os.path.join(d, f"{args.run_id}_fan.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p} (run the diagnostic with IMF_PLOT_FAN=1)")
        loaded.append((label, _load(p)))

    plotter = AvoidingTrajectoryPlotter(
        constraint=args.constraint, obstacle_margin=args.obstacle_margin
    )

    n_rows = 2 * len(loaded)
    _, cols0 = _rows_for(loaded[0][1], args.plan_idx, args.n_cols)
    n_cols = len(cols0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 3.0 * n_rows))
    axes = np.atleast_2d(axes)

    for s, (label, data) in enumerate(loaded):
        ad = data["action_dim"]
        p, cols = _rows_for(data, args.plan_idx, args.n_cols)
        n_ode = data["chain"].shape[1] - 1
        print(f"[ grid ] {label:<4} backbone={data['backbone']:<3} "
              f"plan {p + 1}/{data['chain'].shape[0]}  ODE steps {list(cols)} of {n_ode}")
        for c, k in enumerate(cols):
            _cell(axes[2 * s, c], data["chain"][p, k], data["real"], plotter, ad,
                  f"ODE step {k}", show_y=(c == 0))
            _cell(axes[2 * s + 1, c], data["x1"][p, k], data["real"], plotter, ad,
                  "", show_y=(c == 0))
        axes[2 * s, 0].set_ylabel(f"{label}: $x_\\tau$", fontsize=12)
        axes[2 * s + 1, 0].set_ylabel(f"{label}: $\\hat{{x}}_1$", fontsize=12)

    fig.suptitle(
        "Generation process across ODE steps — top: intermediate sample $x_\\tau$, "
        "bottom: terminal prediction $\\hat{x}_1$   (HardFlow Fig. 11 style)",
        fontsize=13,
    )
    fig.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(args.dir.rstrip("/")), f"fig11_ode_grid_run{args.run_id}.png"
    )
    plotter.save_figure(fig, out, bbox_inches="tight")


if __name__ == "__main__":
    main()
