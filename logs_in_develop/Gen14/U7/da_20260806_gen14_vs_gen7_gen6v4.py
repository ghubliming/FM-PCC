#!/usr/bin/env python3
"""
Cross-generation DA: Gen14 mf/af (massive K=2 run) vs the Gen7 (FM) and Gen6V4 (Gaussian
diffusion) runs sitting in the same DA_VA_v2 batch export.

Input : temp/0608/batch_va2_20260806_204620/per_rollout_detail.csv

Candidate -> generation map (from the folder paths in the batch, cross-checked against
logs_in_develop/MASTER_TEST_HISTORY.md and config/aligning-d3il-visual.py):
    1-4  fm_visual_aligning/...VisualFlowMatching      -> Gen7   (K=20)
    5,7  mix_visual_aligning_{af,mf}/...K100           -> Gen14  (K=100, partial)
    6,8  mix_visual_aligning_{af,mf}/...K2             -> Gen14  (K=2, the massive run)
    9    visual_aligning_dpcc/...VisualGaussianDiffusion -> Gen6V4 (K=20 @ steps400)

Output: logs_in_develop/Gen14/U7/figs/figX_gen_cmp_*.png  + stdout tables

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
    "temp/0608/batch_va2_20260806_204620/per_rollout_detail.csv")
FIGS = sys.argv[2] if len(sys.argv) > 2 else "logs_in_develop/Gen14/U7/figs"

LAB = {1: "Gen7-c1", 2: "Gen7-c2", 3: "Gen7-c3", 4: "Gen7-c4",
       5: "Gen14af-K100", 6: "Gen14af-K2", 7: "Gen14mf-K100", 8: "Gen14mf-K2",
       9: "Gen6V4"}
GEN = {c: ("Gen14" if c in (5, 6, 7, 8) else "Gen6V4" if c == 9 else "Gen7") for c in LAB}
# The 3 train contexts every candidate in the batch shares.
PAIRED_CTX = 3


def load():
    d = pd.read_csv(CSV)
    d["lab"] = d.Candidate.map(LAB)
    d["gen"] = d.Candidate.map(GEN)
    # A rollout where the box's final offset equals its initial offset never moved the box.
    d["box_move"] = (d.final_xy_dist_m - d.context_init_xy_dist).abs()
    d["unmoved"] = d.box_move < 1e-3
    return d


def inventory(d):
    """What data each candidate actually has — the thing that decides comparability."""
    return d.groupby(["gen", "lab", "split", "geo"]).apply(
        lambda x: pd.Series({"variants": x.variant.nunique(), "rollouts": len(x),
                             "n_per_variant": sorted(x.groupby("variant").size().unique()),
                             "viol_avail_%": round(
                                 x.constraint_exec_n_violated_steps.notna().mean() * 100)}),
        include_groups=False)


def paired_basis(d, geo):
    """The only context-paired basis in the batch: train split, contexts 0..2."""
    return d[(d.split == "train") & (d.rollout_idx < PAIRED_CTX) & (d.geo == geo)]


def cross_table(sub, metric, labs=None):
    p = sub.pivot_table(index="variant", columns="lab", values=metric, aggfunc="mean")
    if labs:
        p = p[[c for c in labs if c in p.columns]]
    return p


def dead_run_table(d):
    """Fraction of rollouts in which the box never moved, per candidate/split."""
    return d.groupby(["gen", "lab", "split"]).apply(
        lambda x: pd.Series({"rollouts": len(x),
                             "unmoved_%": round(x.unmoved.mean() * 100, 1),
                             "median_box_move_m": round(x.box_move.median(), 4),
                             "phys_err_median_m": round(x.phys_err_m.median(), 3),
                             "goal_successes": int(x.n_success.sum())}),
        include_groups=False)


# ---------------------------------------------------------------- figures
def fig_dead_run(d):
    """The finding that decides the whole comparison: does the policy move the box at all."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    t = d.groupby(["lab", "split"]).unmoved.mean().mul(100).sort_values()
    names = [f"{a}\n({b})" for a, b in t.index]
    cols = ["tab:green" if n.startswith("Gen14") else
            "tab:red" if n.startswith("Gen6V4") else "tab:orange" for n in names]
    ax.barh(range(len(t)), t.values, color=cols)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("% of rollouts where the box never moved (<1 mm)")
    ax.set_title("Gen6V4 barely touches the box; Gen7 and Gen14 do", fontsize=10)
    ax.grid(alpha=.25, axis="x")
    for i, v in enumerate(t.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=7)

    ax = axes[1]
    for lab, col in (("Gen14mf-K2", "tab:green"), ("Gen14af-K2", "tab:olive"),
                     ("Gen7-c4", "tab:orange"), ("Gen6V4", "tab:red")):
        s = d[(d.lab == lab)].box_move.dropna()
        ax.hist(s, bins=np.linspace(0, 1.2, 40), histtype="step", lw=1.8,
                density=True, color=col, label=f"{lab} (n={len(s)})")
    ax.set(xlabel="|final box offset − initial box offset| [m]", ylabel="density",
           title="displacement distribution")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=.25)
    fig.suptitle("Does the policy actually act? — all rollouts in the batch", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig1_gen_cmp_dead_run.png", dpi=140)


def fig_cost(d):
    """ms/replan is the one axis comparable across generations regardless of split."""
    sub = paired_basis(d, "combined_5")
    shared = ["diffuser", "dpcc-c", "dpcc-r", "dpcc-t", "model_free", "post_processing"]
    sub = sub[sub.variant.isin(shared)]
    order = ["Gen14mf-K2", "Gen14af-K2", "Gen6V4", "Gen7-c4", "Gen7-c3", "Gen7-c2",
             "Gen14mf-K100", "Gen14af-K100"]
    p = cross_table(sub, "avg_time_ms", order)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(p.index))
    w = 0.10
    for i, lab in enumerate(p.columns):
        col = ("tab:green" if lab.startswith("Gen14") and "K2" in lab else
               "tab:purple" if lab.startswith("Gen14") else
               "tab:red" if lab == "Gen6V4" else "tab:orange")
        ax.bar(x + i * w, p[lab].values, w, label=lab, color=col)
    ax.set_yscale("log")
    ax.axhline(33, color="k", ls="--", lw=1)
    ax.text(-0.4, 36, "33 ms (30 Hz budget)", fontsize=8)
    ax.set_xticks(x + w * (len(p.columns) - 1) / 2)
    ax.set_xticklabels(p.index, fontsize=8)
    ax.set(ylabel="ms / replan (log)",
           title="Cost per replan, matched variants, same 3 contexts, combined_5")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig2_gen_cmp_cost.png", dpi=140)


def fig_availability(d):
    """Why most of the cross-generation comparison cannot be made."""
    piv = d.pivot_table(index="lab", columns=["split", "geo"],
                        values="rollout_idx", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    m = piv.values.astype(float)
    im = ax.imshow(np.log10(m + 1), cmap="Blues", aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{a}\n{b}" for a, b in piv.columns], fontsize=7.5)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if np.log10(m[i, j] + 1) > 2 else "black")
    ax.set_title("Rollouts per candidate × split × geometry — Gen14 is train-only, "
                 "Gen7/Gen6V4's n=30 data is test-only", fontsize=10)
    plt.colorbar(im, ax=ax, label="log10(rollouts + 1)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig3_gen_cmp_availability.png", dpi=140)


def main():
    d = load()

    print("### INVENTORY")
    print(inventory(d).to_string())

    print("\n### DEAD-RUN TEST (did the box move at all?)")
    print(dead_run_table(d).to_string())

    for geo in ("combined_5", "combined_5-tightened"):
        sub = paired_basis(d, geo)
        print(f"\n### PAIRED BASIS — train split, contexts 0-2, {geo}")
        for m in ("mean_dist_m", "constraint_exec_n_violated_steps",
                  "phys_err_m", "avg_time_ms", "box_move"):
            print(f"\n-- {m} --")
            print(cross_table(sub, m).round(3).to_string())

    print("\n### SPEED: Gen14-K2 vs the rest, matched variants (train ctx 0-2, combined_5)")
    sub = paired_basis(d, "combined_5")
    p = cross_table(sub, "avg_time_ms")
    base = p[["Gen14mf-K2", "Gen14af-K2"]].mean(axis=1)
    for c in p.columns:
        if "K2" in c:
            continue
        r = (p[c] / base).dropna()
        if len(r):
            print(f"  {c:14s}: {r.median():7.1f}x slower than Gen14-K2 "
                  f"(over {len(r)} shared variants)")

    fig_dead_run(d)
    fig_cost(d)
    fig_availability(d)
    print(f"\nfigures written to {FIGS}/")


if __name__ == "__main__":
    main()
