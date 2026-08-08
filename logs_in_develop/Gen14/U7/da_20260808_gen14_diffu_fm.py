#!/usr/bin/env python3
"""
DA for the two NEW Gen14 arms — engine=diffusion (job 24338/24340/24341) and engine=fm
(job 24343/24345/24346), both trained from scratch and evaluated at K=100 on the 30
train contexts.

Question: does the Gen14 `diffusion` arm still behave like Gen6V4, and does the Gen14
`fm` arm still behave like Gen7?

Input : temp/2026-08-07/batch_va2_20260808_105342/per_rollout_detail.csv

Candidate -> generation map (folder paths in the batch, cross-checked against
config/aligning-d3il-visual.py and logs_in_develop/MASTER_TEST_HISTORY.md):
    1-4  fm_visual_aligning/...VisualFlowMatching        -> Gen7   (K=20)
    5    mix_visual_aligning_af/...K100                  -> Gen14 af  K=100
    6    mix_visual_aligning_af/...K2                    -> Gen14 af  K=2
    7    mix_visual_aligning_diffusion/...K100           -> Gen14 diffusion K=100  [NEW]
    8    mix_visual_aligning_fm/...K100                  -> Gen14 fm        K=100  [NEW]
    9    mix_visual_aligning_mf/...K100                  -> Gen14 mf  K=100
    10   mix_visual_aligning_mf/...K2                    -> Gen14 mf  K=2
    11   visual_aligning_dpcc/...VisualGaussianDiffusion -> Gen6V4 (K=20 @ steps400)

Output: logs_in_develop/Gen14/U7/figs/fig{1,2,3}_0808_*.png + stdout tables

This container has no project env; run with a scratchpad venv holding
pandas / numpy / matplotlib.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "temp/2026-08-07/batch_va2_20260808_105342/per_rollout_detail.csv")
FIGS = sys.argv[2] if len(sys.argv) > 2 else "logs_in_develop/Gen14/U7/figs"

LAB = {1: "Gen7-c1", 2: "Gen7-c2", 3: "Gen7-c3", 4: "Gen7-c4",
       5: "G14af-K100", 6: "G14af-K2", 7: "G14diffu-K100", 8: "G14fm-K100",
       9: "G14mf-K100", 10: "G14mf-K2", 11: "Gen6V4"}
GEN = {c: ("Gen14" if c in (5, 6, 7, 8, 9, 10) else "Gen6V4" if c == 11 else "Gen7")
       for c in LAB}
K100 = ["G14diffu-K100", "G14fm-K100", "G14mf-K100", "G14af-K100"]
# The 3 train contexts every candidate in the batch shares (Gen7/Gen6V4 have only these).
PAIRED_CTX = 3
# A rollout whose peak commanded-vs-actual gap exceeds this is one where the arm left the
# commanded trajectory entirely — the failure mode §2 of the 08-06 DA found in Gen6V4.
DIVERGE_M = 1.0


def load():
    d = pd.read_csv(CSV)
    d["lab"] = d.Candidate.map(LAB)
    d["gen"] = d.Candidate.map(GEN)
    # Box never moved: final offset to target equals the initial offset.
    d["box_move"] = (d.context_final_xy_dist - d.context_init_xy_dist).abs()
    d["unmoved"] = d.box_move < 1e-3
    d["diverged"] = d.max_phys_error_per_rollout > DIVERGE_M
    return d


def health(sub, by=("lab",)):
    """The columns that decide whether a run is alive, per group."""
    return sub.groupby(list(by)).agg(
        n=("rollout_idx", "size"),
        diverged=("diverged", "mean"),
        unmoved=("unmoved", "mean"),
        phys_med=("max_phys_error_per_rollout", "median"),
        boxmove_med=("box_move", "median"),
        dist=("mean_dist_per_rollout", "mean"),
        viol=("constraint_exec_n_violated_steps", "mean"),
        sat=("constraint_exec_sat_rate", "mean"),
        succ=("n_success", "sum"),
        relaxed=("success_relaxed", "sum"),
        ms=("avg_time_ms", "mean")).round(3)


def paired_basis(d, geo="combined_5"):
    """The only cross-generation context-paired basis: train split, contexts 0..2."""
    return d[(d.split == "train") & (d.rollout_idx < PAIRED_CTX) & (d.geo == geo)]


def cross_table(sub, metric, labs=None):
    p = sub.pivot_table(index="variant", columns="lab", values=metric, aggfunc="mean")
    return p[[c for c in labs if c in p.columns]] if labs else p


# ---------------------------------------------------------------- figures
def fig_divergence(d):
    """The headline: the fm arm loses the arm, the diffusion arm does not."""
    sub = d[(d.variant == "diffuser") & (d.geo == "combined_5")]
    k = sub[sub.lab.isin(K100)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    t = k.groupby("lab")[["diverged", "unmoved"]].mean().mul(100).loc[K100]
    x = np.arange(len(t))
    ax.bar(x - 0.19, t.diverged, 0.38, label=f"peak tracking error > {DIVERGE_M:.0f} m",
           color="tab:red")
    ax.bar(x + 0.19, t.unmoved, 0.38, label="box never moved (<1 mm)", color="tab:gray")
    for i, (a, b) in enumerate(zip(t.diverged, t.unmoved)):
        ax.text(i - 0.19, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8)
        ax.text(i + 0.19, b + 1.5, f"{b:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, fontsize=9)
    ax.set(ylabel="% of rollouts", ylim=(0, 100),
           title="Unprojected rollouts (`diffuser`), same 30 train contexts, K=100")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    for lab, col in zip(K100, ("tab:blue", "tab:red", "tab:green", "tab:olive")):
        s = k[k.lab == lab].sort_values("rollout_idx")
        ax.plot(s.rollout_idx, s.max_phys_error_per_rollout, "o-", ms=4, lw=1,
                color=col, label=lab, alpha=.85)
    ax.axhline(DIVERGE_M, color="k", ls="--", lw=1)
    ax.text(0, DIVERGE_M * 1.1, f"{DIVERGE_M:.0f} m", fontsize=8)
    ax.set_yscale("log")
    ax.set(xlabel="train context idx", ylabel="peak physical tracking error [m] (log)",
           title="Per-context, paired: fm sits in the diverged branch almost everywhere")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.suptitle("Gen14 K=100 arms — the new `fm` and `diffusion` runs against mf/af",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig1_0808_divergence.png", dpi=140)


def fig_vs_old(d):
    """Cross-generation health, each generation on the data it actually has."""
    rows = [("G14diffu-K100", "train", "Gen14 diffusion arm\n(K=100, 30 train ctx)"),
            ("Gen6V4", "test", "Gen6V4 archived\n(K=20, 30 test ctx)"),
            ("Gen6V4", "train", "Gen6V4 archived\n(K=20, 3 train ctx)"),
            ("G14fm-K100", "train", "Gen14 fm arm\n(K=100, 30 train ctx)"),
            ("Gen7-c3", "test", "Gen7-c3 filmv2\n(K=20, 30 test ctx)"),
            ("Gen7-c4", "test", "Gen7-c4 filmv1\n(K=20, 30 test ctx)")]
    sub = d[d.variant == "diffuser"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [("diverged", "% rollouts with peak tracking error > 1 m", 100),
               ("unmoved", "% rollouts where the box never moved", 100),
               ("box_move", "median box displacement [m]", 1)]
    for ax, (m, title, scale) in zip(axes, metrics):
        vals, names, cols = [], [], []
        for lab, split, name in rows:
            s = sub[(sub.lab == lab) & (sub.split == split)]
            if not len(s):
                continue
            vals.append((s[m].median() if m == "box_move" else s[m].mean()) * scale)
            names.append(f"{name}\nn={len(s)}")
            cols.append("tab:blue" if lab.startswith("G14diffu") else
                        "tab:red" if lab.startswith("G14fm") else
                        "tab:brown" if lab == "Gen6V4" else "tab:orange")
        y = np.arange(len(vals))
        ax.barh(y, vals, color=cols)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.25, axis="x")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}" if scale == 1 else f" {v:.0f}%", va="center",
                    fontsize=7.5)
    fig.suptitle("Do the new arms reproduce the old generations? "
                 "(`diffuser` variant — splits differ, see §1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig2_0808_vs_old_generations.png", dpi=140)


def fig_cost(d):
    """ms/replan — the axis that does not depend on split or contexts."""
    sub = d[(d.geo == "combined_5") & (d.variant.isin(["diffuser", "dpcc-r"]))]
    order = ["G14diffu-K100", "G14fm-K100", "G14mf-K100", "G14af-K100",
             "G14mf-K2", "G14af-K2", "Gen6V4", "Gen7-c3"]
    p = sub.pivot_table(index="variant", columns="lab", values="avg_time_ms",
                        aggfunc="mean")
    p = p[[c for c in order if c in p.columns]]
    colours = {"G14diffu-K100": "tab:blue", "G14fm-K100": "tab:red",
               "G14mf-K100": "tab:purple", "G14af-K100": "tab:pink",
               "G14mf-K2": "tab:green", "G14af-K2": "tab:olive",
               "Gen6V4": "tab:brown", "Gen7-c3": "tab:orange"}
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(p.index))
    w = 0.10
    for i, lab in enumerate(p.columns):
        ax.bar(x + i * w, p[lab].values, w, label=lab, color=colours.get(lab, "gray"))
    ax.set_yscale("log")
    ax.axhline(33, color="k", ls="--", lw=1)
    ax.text(-0.4, 36, "33 ms (30 Hz budget)", fontsize=8)
    ax.set_xticks(x + w * (len(p.columns) - 1) / 2)
    ax.set_xticklabels(p.index, fontsize=9)
    ax.set(ylabel="ms / replan (log)",
           title="Cost per replan — the two K=100 items the new runs got through")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig3_0808_cost.png", dpi=140)


def main():
    d = load()

    print("### INVENTORY (what each candidate has)")
    print(d.groupby(["gen", "lab", "split", "geo"]).agg(
        variants=("variant", "nunique"), rollouts=("rollout_idx", "size"),
        contexts=("rollout_idx", "nunique")).to_string())

    print("\n### CONTEXT PAIRING (max |Δ| in box init / target / angle vs cand 10 train)")
    key = ["context_box_init_xy_x", "context_box_init_xy_y", "context_target_xy_x",
           "context_target_xy_y", "context_box_angle_deg"]
    ref = d[(d.Candidate == 10)].drop_duplicates("rollout_idx").set_index("rollout_idx")[key]
    for (lab, split), g in d.groupby(["lab", "split"]):
        x = g.drop_duplicates("rollout_idx").set_index("rollout_idx")[key]
        common = x.index.intersection(ref.index)
        print(f"  {lab:14s} {split:5s} n_ctx={len(x):>2} overlap={len(common):>2} "
              f"maxΔ={(x.loc[common] - ref.loc[common]).abs().max().max():.3e}")

    print("\n### THE CLEAN COMPARISON — four K=100 engines, identical 30 train contexts")
    for v in ("diffuser", "dpcc-r"):
        s = d[(d.lab.isin(K100)) & (d.variant == v) & (d.geo == "combined_5")]
        print(f"\n-- variant={v} --")
        print(health(s).loc[[l for l in K100 if l in health(s).index]].to_string())

    print("\n### K EFFECT inside the flow arms (mf/af have both K=100 and K=2)")
    s = d[(d.variant == "diffuser") & (d.geo == "combined_5") &
          (d.lab.isin(K100 + ["G14mf-K2", "G14af-K2"]))]
    print(health(s).to_string())

    print("\n### CROSS-GENERATION HEALTH (each on the data it has — splits differ)")
    s = d[d.variant == "diffuser"]
    print(health(s, by=("lab", "split")).to_string())

    for geo in ("combined_5",):
        sub = paired_basis(d, geo)
        print(f"\n### PAIRED BASIS — train split, contexts 0-2, {geo}")
        order = ["G14diffu-K100", "Gen6V4", "G14fm-K100", "Gen7-c3", "Gen7-c4"]
        for m in ("mean_dist_per_rollout", "max_phys_error_per_rollout",
                  "constraint_exec_n_violated_steps", "box_move", "avg_time_ms"):
            print(f"\n-- {m} --")
            print(cross_table(sub, m, order).round(3).to_string())

    print("\n### MODE vs DIVERGENCE (train, diffuser + dpcc-r)")
    s = d[(d.split == "train") & d.variant.isin(["diffuser", "dpcc-r"])]
    print(pd.crosstab([s.lab, s.mode_encoding], s.diverged).to_string())

    fig_divergence(d)
    fig_vs_old(d)
    fig_cost(d)
    print(f"\nfigures written to {FIGS}/")


if __name__ == "__main__":
    main()
