"""Gen13 u_8 — paper-Fig.11-style side-by-side comparison: iMF vs original FM.

Reproduces the visual language of HardFlow's Fig. 11 ("Visualization of the
generation process ... one representative planning instance during execution")
as a TWO-PANEL comparison of the two backbones.

Unlike the fan diagnostic (which overlays every replan to show the envelope),
this shows **ONE representative planning instance per method** — exactly the
paper's framing — and therefore *can* use HardFlow's own unused
`style="predicted"` without the marker clutter that made it unsuitable for the
fan. See ../../logs_in_develop/HF_iMF/Research/MEMO_hardflow_fig11_predicted_style.md.

Pure POST-PROCESSING: reads the `*_fan.npz` dumps written by run/eval_imf.py
(u_8), so it needs no GPU, no simulator, and no model — just numpy+matplotlib.

Usage:
    python run/make_fig11_comparison.py \
        --imf_dir  logs/avoiding-v0/eval/diag_smooth_imf_guided_K5_n3 \
        --fm_dir   logs/avoiding-v0/eval/diag_smooth_fm_guided_K10_n3 \
        --run_id   0 \
        --out      logs/avoiding-v0/eval/fig11_imf_vs_fm.png
    # --plan_idx <i>  picks the replan instant (default: middle of the episode)
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
    return {
        "planned": d["planned"],          # (n_replans, H, transition_dim)
        "x1": d["x1"],
        "real": d["real"],                # (T, state_dim)
        "action_dim": int(d["action_dim"]),
        "backbone": str(d["backbone"]),
        "guidance": str(d["guidance"]),
    }


def _panel(ax, data, plotter, plan_idx, title, show_y_label, show_x1):
    """One Fig.11-style panel: the environment + one planning instance + rollout."""
    plotter._configure_axis(
        ax, show_x_label=True, show_y_label=show_y_label, compact=True
    )
    ad = data["action_dim"]
    n_plans = len(data["planned"])
    idx = n_plans // 2 if plan_idx is None else max(0, min(plan_idx, n_plans - 1))

    plan = data["planned"][idx]
    # `plot_single_trajectory` reads x,y from columns 2,3 (OBSERVATION layout);
    # planned trajectories are full transitions -> slice off the actions.
    plotter.plot_single_trajectory(plan[:, ad:], ax, style="predicted")

    if show_x1:
        x1 = data["x1"][idx]
        ax.plot(
            x1[:, ad + 2], x1[:, ad + 3], "--", color="tab:orange",
            linewidth=1.4, alpha=0.9, zorder=4, label="Terminal prediction x̂1",
        )

    plotter.plot_single_trajectory(data["real"], ax, style="actual")
    plotter.add_environment_elements(ax)
    ax.set_title(f"{title}   (replan {idx + 1}/{n_plans})", fontsize=13)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imf_dir", required=True, help="eval dir of the iMF run")
    ap.add_argument("--fm_dir", required=True, help="eval dir of the FM run")
    ap.add_argument("--run_id", type=int, default=0, help="which episode")
    ap.add_argument("--plan_idx", type=int, default=None,
                    help="which replan instant (default: middle)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--constraint", default="novel")
    ap.add_argument("--obstacle_margin", type=float, default=0.02)
    ap.add_argument("--no_x1", action="store_true",
                    help="hide the x̂1 terminal-prediction overlay")
    args = ap.parse_args()

    paths = {
        "iMF (average velocity u)": os.path.join(args.imf_dir, f"{args.run_id}_fan.npz"),
        "FM (instantaneous velocity v)": os.path.join(args.fm_dir, f"{args.run_id}_fan.npz"),
    }
    for label, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {p}\n  -> run the diagnostic with IMF_PLOT_FAN=1 first "
                f"(it writes *_fan.npz alongside *_fan.png)"
            )

    plotter = AvoidingTrajectoryPlotter(
        constraint=args.constraint, obstacle_margin=args.obstacle_margin
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))

    for ax, (label, p) in zip(axes, paths.items()):
        d = _load(p)
        idx = _panel(
            ax, d, plotter, args.plan_idx, label,
            show_y_label=(ax is axes[0]), show_x1=not args.no_x1,
        )
        print(f"[ fig11 ] {label:<32} {d['guidance']:<18} "
              f"plan {idx + 1}/{len(d['planned'])}  from {p}")

    plotter.apply_legend(axes[0])
    fig.suptitle(
        "Generation process: one representative planning instance "
        "(HardFlow Fig. 11 style)",
        fontsize=15,
    )
    fig.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(args.imf_dir), f"fig11_imf_vs_fm_run{args.run_id}.png"
    )
    plotter.save_figure(fig, out, bbox_inches="tight")


if __name__ == "__main__":
    main()
