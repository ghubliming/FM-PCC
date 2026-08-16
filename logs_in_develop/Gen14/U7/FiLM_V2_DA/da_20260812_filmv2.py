#!/usr/bin/env python3
"""
DA — FiLM V2 vs FiLM V1 visual conditioning, Gen14 mix arms (aligning-d3il-visual).

Source: temp/1208/Gen14_Mix_FilmV2/batch_va2_20260812_114647/  (DA_VA_v2 batch,
15 candidates, 282 units, 5634 rollouts, seed 6, train split).

Design: the V1 and V2 checkpoints of each mix engine were evaluated on the SAME
contexts under the SAME projector grid, so every V1<->V2 comparison is a paired,
within-context contrast. Tests:
  - continuous: paired sign-flip permutation, 20000 resamples, two-sided
  - binary:     exact McNemar (binomial on discordant pairs)
  - multiplicity: Holm-Bonferroni within each family

stdlib only — this container has no numpy/scipy (see CLAUDE.md).
"""

import csv
import math
import random
import collections
import os

random.seed(20260812)

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = "/workspaces/FM-PCC/temp/1208/Gen14_Mix_FilmV2/batch_va2_20260812_114647"
NPERM = 20000

# candidate -> (engine label, film version, K)
CAND = {
    "1":  ("fm_visual (legacy Bf_U8)",  "v2", 20),
    "2":  ("fm_visual (legacy bounds)", "v2", 20),
    "3":  ("fm_visual",                 "v2", 20),
    "4":  ("fm_visual",                 "v1", 20),
    "5":  ("mix af",  "v1", 100),
    "6":  ("mix af",  "v1", 2),
    "7":  ("mix af",  "v2", 2),
    "8":  ("mix diffusion", "v1", 100),
    "9":  ("mix diffusion", "v1", 20),
    "10": ("mix fm", "v1", 100),
    "11": ("mix fm", "v1", 20),
    "12": ("mix mf", "v1", 100),
    "13": ("mix mf", "v1", 2),
    "14": ("mix mf", "v2", 2),
    "15": ("dpcc diffuser (baseline)", "v2", 20),
}

CONTINUOUS = [
    ("constraint_exec_sat_rate",           "constraint sat rate",        "+"),
    ("constraint_exec_n_violated_steps",   "violated steps",             "-"),
    ("constraint_exec_margin_mean_m",      "constraint margin (m)",      "+"),
    ("constraint_exec_max_obstacle_penetration_m", "max obstacle pen (m)", "-"),
    ("mean_dist_per_rollout",              "mean distance to goal (m)",  "-"),
    ("max_phys_error_per_rollout",         "max phys tracking err (m)",  "-"),
    ("context_final_xy_dist",              "final xy dist to target (m)", "-"),
    ("avg_time_ms",                        "time per replan (ms)",       "-"),
    ("n_steps",                            "episode steps",              "-"),
]

BINARY = [
    ("n_success",                     "goal success"),
    ("success_relaxed",               "goal success (relaxed)"),
    ("collision_free_completed",      "collision-free completed"),
    ("constraint_exec_zero_violation", "zero-violation rollout"),
]


def fnum(s):
    if s is None or s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def load():
    with open(os.path.join(BATCH, "per_rollout_detail.csv")) as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- statistics
def perm_paired(diffs, nperm=NPERM):
    """Two-sided paired sign-flip permutation test on the mean difference."""
    d = [x for x in diffs if x is not None]
    n = len(d)
    if n == 0:
        return None
    obs = sum(d) / n
    if all(abs(x) < 1e-15 for x in d):
        return dict(n=n, mean=0.0, p=1.0)
    hits = 0
    a = abs(obs)
    for _ in range(nperm):
        s = 0.0
        for x in d:
            s += x if random.getrandbits(1) else -x
        if abs(s / n) >= a - 1e-15:
            hits += 1
    return dict(n=n, mean=obs, p=(hits + 1) / (nperm + 1))


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p-value (used for McNemar on discordant pairs)."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    target = pmf(k) * (1 + 1e-9)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= target))


def mcnemar(pairs):
    """pairs = [(a_bool, b_bool)] -> b_only/a_only counts + exact p."""
    a_only = sum(1 for a, b in pairs if a and not b)
    b_only = sum(1 for a, b in pairs if b and not a)
    both = sum(1 for a, b in pairs if a and b)
    neither = sum(1 for a, b in pairs if not a and not b)
    return dict(a_only=a_only, b_only=b_only, both=both, neither=neither,
                n=len(pairs), a_tot=a_only + both, b_tot=b_only + both,
                p=binom_two_sided(min(a_only, b_only), a_only + b_only))


def holm(results):
    """results = [(label, p, payload)] -> same list with adjusted p, sorted by raw p."""
    rs = sorted(results, key=lambda r: r[1])
    m = len(rs)
    out, running = [], 0.0
    for i, (label, p, payload) in enumerate(rs):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out.append((label, p, running, payload))
    return out


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ---------------------------------------------------------------- pairing
def paired_rollouts(rows, cand_v1, cand_v2, drop_frozen=True):
    """Match rollouts on (geo, variant, rollout_idx); verify the context is identical."""
    idx = {}
    for r in rows:
        idx[(r["Candidate"], r["geo"], r["variant"], r["rollout_idx"])] = r
    ctx_cols = ["context_box_init_xy_x", "context_box_init_xy_y", "context_box_angle_deg",
                "context_target_xy_x", "context_target_xy_y"]
    pairs, dropped_frozen, unmatched, ctx_mismatch = [], 0, 0, 0
    for key, r1 in idx.items():
        if key[0] != cand_v1:
            continue
        k2 = (cand_v2,) + key[1:]
        r2 = idx.get(k2)
        if r2 is None:
            unmatched += 1
            continue
        ok = True
        for c in ctx_cols:
            a, b = fnum(r1[c]), fnum(r2[c])
            if a is not None and b is not None and abs(a - b) > 1e-6:
                ok = False
        if not ok:
            ctx_mismatch += 1
            continue
        if drop_frozen and (fnum(r1["frozen"]) or fnum(r2["frozen"])):
            dropped_frozen += 1
            continue
        pairs.append((r1, r2))
    return pairs, dict(dropped_frozen=dropped_frozen, unmatched=unmatched,
                       ctx_mismatch=ctx_mismatch)


def contrast(pairs, title, subset=None):
    """Run the full metric battery on a paired set. v1 = first, v2 = second."""
    if subset:
        pairs = [p for p in pairs if subset(p[0])]
    print(f"\n{'-' * 78}\n{title}   (n = {len(pairs)} paired rollouts)\n{'-' * 78}")
    if not pairs:
        print("  (empty)")
        return {}

    cont_res, out = [], {}
    for col, label, good in CONTINUOUS:
        d, v1s, v2s = [], [], []
        for r1, r2 in pairs:
            a, b = fnum(r1[col]), fnum(r2[col])
            if a is None or b is None:
                continue
            d.append(b - a)
            v1s.append(a)
            v2s.append(b)
        if not d:
            continue
        st = perm_paired(d)
        st.update(v1=sum(v1s) / len(v1s), v2=sum(v2s) / len(v2s), label=label, good=good)
        cont_res.append((label, st["p"], st))

    print(f"  {'metric':<30} {'V1':>11} {'V2':>11} {'Δ(V2−V1)':>11} {'n':>5} "
          f"{'p_raw':>8} {'p_holm':>8}")
    for label, p, padj, st in holm(cont_res):
        flag = ""
        if padj < 0.05:
            better = (st["mean"] > 0) == (st["good"] == "+")
            flag = "  V2 better" if better else "  V1 better"
        print(f"  {label:<30} {st['v1']:>11.4f} {st['v2']:>11.4f} {st['mean']:>+11.4f} "
              f"{st['n']:>5} {p:>8.4f} {padj:>8.4f} {stars(padj):<3}{flag}")
        out[label] = dict(st, p_holm=padj)

    bin_res = []
    for col, label in BINARY:
        pr = []
        for r1, r2 in pairs:
            a, b = fnum(r1[col]), fnum(r2[col])
            if a is None or b is None:
                continue
            pr.append((a > 0.5, b > 0.5))
        if not pr:
            continue
        st = mcnemar(pr)
        st["label"] = label
        bin_res.append((label, st["p"], st))

    if bin_res:
        print(f"\n  {'binary outcome':<30} {'V1':>11} {'V2':>11} {'discordant':>12} "
              f"{'n':>5} {'p_raw':>8} {'p_holm':>8}")
        for label, p, padj, st in holm(bin_res):
            flag = ""
            if padj < 0.05:
                flag = "  V2 better" if st["b_tot"] > st["a_tot"] else "  V1 better"
            disc = f"{st['a_only']}/{st['b_only']}"
            print(f"  {label:<30} {st['a_tot']:>11d} {st['b_tot']:>11d} {disc:>12} "
                  f"{st['n']:>5} {p:>8.4f} {padj:>8.4f} {stars(padj):<3}{flag}")
            out[label] = dict(st, p_holm=padj)
    return out


def per_variant_table(pairs, col, label, good):
    """Direction of the V2−V1 effect broken out by projector variant."""
    by = collections.defaultdict(list)
    for r1, r2 in pairs:
        a, b = fnum(r1[col]), fnum(r2[col])
        if a is not None and b is not None:
            by[r1["variant"]].append((a, b))
    print(f"\n  per-variant: {label}  (V1 -> V2)")
    print(f"  {'variant':<24} {'V1':>9} {'V2':>9} {'Δ':>9} {'n':>4}  {'p':>7}")
    wins = losses = 0
    for v in sorted(by):
        vals = by[v]
        a = sum(x for x, _ in vals) / len(vals)
        b = sum(y for _, y in vals) / len(vals)
        st = perm_paired([y - x for x, y in vals], nperm=5000)
        better = (b > a) == (good == "+")
        if st["p"] < 0.05:
            wins += better
            losses += not better
        print(f"  {v:<24} {a:>9.4f} {b:>9.4f} {b - a:>+9.4f} {len(vals):>4}  "
              f"{st['p']:>7.4f} {stars(st['p'])}")
    print(f"  significant cells: V2 better in {wins}, V1 better in {losses}")


def describe(rows, cand, note=""):
    rs = [r for r in rows if r["Candidate"] == cand]
    if not rs:
        return
    eng, film, K = CAND[cand]
    frozen = sum(1 for r in rs if fnum(r["frozen"]))
    succ = sum(1 for r in rs if (fnum(r["n_success"]) or 0) > 0.5)
    cfree = sum(1 for r in rs if (fnum(r["collision_free_completed"]) or 0) > 0.5)
    sat = [fnum(r["constraint_exec_sat_rate"]) for r in rs]
    sat = [x for x in sat if x is not None]
    t = [fnum(r["avg_time_ms"]) for r in rs]
    t = [x for x in t if x is not None]
    print(f"  C{cand:<3} {eng:<26} FiLM {film}  K={K:<4} n={len(rs):>5} "
          f"frozen={frozen:>3} succ={succ:>3} collfree={cfree:>4} "
          f"sat={sum(sat) / len(sat) if sat else float('nan'):.4f} "
          f"t={sum(t) / len(t) if t else float('nan'):>8.1f}ms {note}")


def raw_model_quality(rows):
    """Variants where the projector does least work — reads the generative model itself."""
    print(f"\n\n{'=' * 78}\nDIAGNOSTIC A — unprojected plan quality (is the damage upstream of the "
          f"projector?)\n{'=' * 78}")
    print(f"  {'cand':<6}{'variant':<22}{'sat':>8}{'physerr':>10}{'dist':>8}"
          f"{'finaldist':>10}{'collfree':>9}{'n':>5}")
    for c in ("13", "14", "6", "7"):
        for v in ("diffuser", "model_free", "geo_free-model_free"):
            g = [r for r in rows if r["Candidate"] == c and r["variant"] == v
                 and not fnum(r["frozen"])]
            if not g:
                continue
            def m(col):
                xs = [fnum(r[col]) for r in g]
                xs = [x for x in xs if x is not None]
                return sum(xs) / len(xs) if xs else float("nan")
            cf = sum(1 for r in g if (fnum(r["collision_free_completed"]) or 0) > 0.5)
            eng, film, _ = CAND[c]
            print(f"  C{c:<5}{v:<22}{m('constraint_exec_sat_rate'):>8.3f}"
                  f"{m('max_phys_error_per_rollout'):>10.3f}{m('mean_dist_per_rollout'):>8.3f}"
                  f"{m('context_final_xy_dist'):>10.3f}{cf:>9}{len(g):>5}   {eng} {film}")


def collapse_check(rows):
    """Behavioural spread across contexts. A collapsed policy ignores the context."""
    print(f"\n\n{'=' * 78}\nDIAGNOSTIC B — behavioural spread across contexts (collapse check, "
          f"variant=model_free)\n{'=' * 78}")
    print(f"  {'cand':<6}{'metric':<28}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'n':>5}")
    for c in ("13", "14", "6", "7"):
        g = [r for r in rows if r["Candidate"] == c and r["variant"] == "model_free"
             and not fnum(r["frozen"])]
        eng, film, _ = CAND[c]
        print(f"  -- C{c} {eng} FiLM {film}")
        for col in ("context_final_xy_dist", "mean_dist_per_rollout",
                    "max_phys_error_per_rollout"):
            xs = [fnum(r[col]) for r in g]
            xs = [x for x in xs if x is not None]
            if not xs:
                continue
            mu = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
            print(f"  {'':<6}{col:<28}{mu:>9.4f}{sd:>9.4f}{min(xs):>9.4f}{max(xs):>9.4f}"
                  f"{len(xs):>5}")


def dt_sweep(rows):
    """Does more projector authority mask a broken plan?"""
    print(f"\n\n{'=' * 78}\nDIAGNOSTIC C — dt sweep: projector authority vs plan quality "
          f"(AlphaFlow K=2)\n{'=' * 78}")
    print(f"  {'variant':<16}{'V1 sat':>9}{'V2 sat':>9}{'V1 phys':>10}{'V2 phys':>10}"
          f"{'V1 cf':>7}{'V2 cf':>7}{'V1 succ':>9}{'V2 succ':>9}")
    for v in ("dpcc-c-dt0p25", "dpcc-c-dt0p5", "dpcc-c", "dpcc-c-dt2p0", "dpcc-c-dt4p0"):
        cells = []
        for c in ("6", "7"):
            g = [r for r in rows if r["Candidate"] == c and r["variant"] == v
                 and not fnum(r["frozen"])]
            def m(col):
                xs = [fnum(r[col]) for r in g]
                xs = [x for x in xs if x is not None]
                return sum(xs) / len(xs) if xs else float("nan")
            cells.append((m("constraint_exec_sat_rate"), m("max_phys_error_per_rollout"),
                          sum(1 for r in g if (fnum(r["collision_free_completed"]) or 0) > 0.5),
                          sum(1 for r in g if (fnum(r["n_success"]) or 0) > 0.5)))
        print(f"  {v:<16}{cells[0][0]:>9.3f}{cells[1][0]:>9.3f}{cells[0][1]:>10.3f}"
              f"{cells[1][1]:>10.3f}{cells[0][2]:>7}{cells[1][2]:>7}"
              f"{cells[0][3]:>9}{cells[1][3]:>9}")


def training_curves():
    """The V2 training runs themselves — did they converge?"""
    import pickle
    print(f"\n\n{'=' * 78}\nDIAGNOSTIC D — FiLM V2 training curves "
          f"(temp/1208/Gen14_Mix_FilmV2/*_losses.pkl)\n{'=' * 78}")
    root = "/workspaces/FM-PCC/temp/1208/Gen14_Mix_FilmV2"
    for name, tag in (("mf_losses.pkl", "MF FiLM V2 (slurm 24454)"),
                      ("af_losses.pkl", "AF FiLM V2 (slurm 24457)")):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print(f"  {tag}: not found")
            continue
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        v = d["test_raw_mse_losses"]
        tl = d["test_losses"]
        lo = min(v, key=lambda p: p[1])
        print(f"\n  {tag}")
        print(f"    test_raw_mse: min {lo[1]:.3f} @ step {int(lo[0])}  ->  "
              f"final {v[-1][1]:.3f} @ {int(v[-1][0])}   ({v[-1][1] / lo[1]:.2f}x the minimum)")
        print("    raw_mse /10k:  " + "  ".join(f"{int(s / 1000)}k:{val:.1f}"
                                                for s, val in v if s % 10000 == 0))
        print("    norm loss/10k: " + "  ".join(f"{int(s / 1000)}k:{val:.3f}"
                                                for s, val in tl if s % 10000 == 0))
        a0 = d.get("training_a0_losses") or []
        if a0:
            med = sorted(x[1] for x in a0)[len(a0) // 2]
            hi = max(a0, key=lambda p: p[1])
            print(f"    a0 anchor loss: median {med:.3f}, max {hi[1]:.3f} @ step {int(hi[0])}")
        # per-1k window around the worst degradation, where the collapse is visible
        print("    raw_mse 68-80k: " + "  ".join(f"{int(s / 1000)}k:{val:.2f}"
                                                 for s, val in v if 68000 <= s <= 80000))
        print("    a0      68-80k: " + "  ".join(f"{int(s / 1000)}k:{val:.2f}"
                                                 for s, val in a0 if 68000 <= s <= 80000))
    print("\n  NOTE: no FiLM V1 loss curves were exported with this batch — the V1 side of "
          "this\n  comparison cannot be made here.")


def main():
    rows = load()
    print("=" * 78)
    print("DA — FiLM V2 vs FiLM V1, Gen14 mix arms (aligning-d3il-visual, seed 6)")
    print("=" * 78)
    print(f"batch: {BATCH}")
    print(f"rollouts loaded: {len(rows)}")

    print("\n### candidate inventory")
    for c in sorted(CAND, key=int):
        describe(rows, c)

    # ---- primary: mix MeanFlow K=2, V1 (C13) vs V2 (C14)
    mf_pairs, mf_info = paired_rollouts(rows, "13", "14")
    print(f"\n\n{'=' * 78}\nPRIMARY 1 — mix MeanFlow K=2: FiLM V1 (C13) vs FiLM V2 (C14)\n{'=' * 78}")
    print(f"  pairing: {len(mf_pairs)} usable pairs, "
          f"{mf_info['dropped_frozen']} dropped frozen, "
          f"{mf_info['unmatched']} V1 rollouts with no V2 partner, "
          f"{mf_info['ctx_mismatch']} context mismatches")
    contrast(mf_pairs, "MeanFlow K=2 — all matched variants and geos pooled")
    contrast(mf_pairs, "MeanFlow K=2 — geo combined_5 (nominal)",
             lambda r: r["geo"] == "combined_5")
    contrast(mf_pairs, "MeanFlow K=2 — geo combined_5-tightened",
             lambda r: r["geo"] == "combined_5-tightened")
    per_variant_table(mf_pairs, "constraint_exec_sat_rate", "constraint sat rate", "+")
    per_variant_table(mf_pairs, "mean_dist_per_rollout", "mean distance to goal (m)", "-")

    # ---- primary: mix AlphaFlow K=2, V1 (C6) vs V2 (C7)
    af_pairs, af_info = paired_rollouts(rows, "6", "7")
    print(f"\n\n{'=' * 78}\nPRIMARY 2 — mix AlphaFlow K=2: FiLM V1 (C6) vs FiLM V2 (C7)\n{'=' * 78}")
    print(f"  pairing: {len(af_pairs)} usable pairs, "
          f"{af_info['dropped_frozen']} dropped frozen, "
          f"{af_info['unmatched']} V1 rollouts with no V2 partner, "
          f"{af_info['ctx_mismatch']} context mismatches")
    contrast(af_pairs, "AlphaFlow K=2 — all matched variants and geos pooled")
    contrast(af_pairs, "AlphaFlow K=2 — geo combined_5 (nominal)",
             lambda r: r["geo"] == "combined_5")
    contrast(af_pairs, "AlphaFlow K=2 — geo combined_5-tightened",
             lambda r: r["geo"] == "combined_5-tightened")
    per_variant_table(af_pairs, "constraint_exec_sat_rate", "constraint sat rate", "+")
    per_variant_table(af_pairs, "mean_dist_per_rollout", "mean distance to goal (m)", "-")

    # ---- replication: do the two engines agree on the sign of the V2 effect?
    print(f"\n\n{'=' * 78}\nREPLICATION — is the V2 effect the same in both engines?\n{'=' * 78}")
    print(f"  {'metric':<30} {'MF Δ':>12} {'AF Δ':>12}  agree?")
    for col, label, good in CONTINUOUS:
        ds = []
        for pairs in (mf_pairs, af_pairs):
            d = [fnum(r2[col]) - fnum(r1[col]) for r1, r2 in pairs
                 if fnum(r1[col]) is not None and fnum(r2[col]) is not None]
            ds.append(sum(d) / len(d) if d else None)
        if ds[0] is None or ds[1] is None:
            continue
        agree = "yes" if (ds[0] > 0) == (ds[1] > 0) else "NO"
        print(f"  {label:<30} {ds[0]:>+12.4f} {ds[1]:>+12.4f}  {agree}")

    # ---- pooled across both engines (2x power on the shared question)
    print(f"\n\n{'=' * 78}\nPOOLED — both mix engines stacked (MF + AF, K=2)\n{'=' * 78}")
    contrast(mf_pairs + af_pairs, "FiLM V1 vs V2, MeanFlow and AlphaFlow pooled")

    # ---- diagnostics: where does the V2 damage come from?
    raw_model_quality(rows)
    collapse_check(rows)
    dt_sweep(rows)
    training_curves()

    # ---- secondary: legacy fm_visual arm, V1 (C4) vs V2 (C3)
    fm_pairs, fm_info = paired_rollouts(rows, "4", "3")
    print(f"\n\n{'=' * 78}\nSECONDARY — legacy fm_visual_aligning K=20: V1 (C4) vs V2 (C3)\n{'=' * 78}")
    print(f"  pairing: {len(fm_pairs)} usable pairs, "
          f"{fm_info['dropped_frozen']} dropped frozen, "
          f"{fm_info['unmatched']} V1 rollouts with no V2 partner, "
          f"{fm_info['ctx_mismatch']} context mismatches")
    print("  NOTE: different training runs / step counts (900 vs 1000) — confounded, "
          "read as directional only.")
    contrast(fm_pairs, "fm_visual K=20 — matched variants and geos")

    # ---- target row: the DPCC baseline every variant must beat
    print(f"\n\n{'=' * 78}\nTARGET — best baseline DPCC row (C15, diffuser_visual_aligning, FiLM V2)\n{'=' * 78}")
    for cand in ("15", "13", "14", "6", "7"):
        rs = [r for r in rows if r["Candidate"] == cand and not fnum(r["frozen"])]
        by = collections.defaultdict(list)
        for r in rs:
            by[r["variant"]].append(r)
        eng, film, K = CAND[cand]
        print(f"\n  C{cand} {eng} FiLM {film} K={K}")
        print(f"    {'variant':<24} {'succ':>5} {'collfree':>9} {'sat':>7} {'dist':>7} "
              f"{'t_ms':>9} {'n':>4}")
        for v in sorted(by):
            g = by[v]
            succ = sum(1 for r in g if (fnum(r["n_success"]) or 0) > 0.5)
            cf = sum(1 for r in g if (fnum(r["collision_free_completed"]) or 0) > 0.5)
            sat = [fnum(r["constraint_exec_sat_rate"]) for r in g]
            sat = [x for x in sat if x is not None]
            dist = [fnum(r["mean_dist_per_rollout"]) for r in g]
            dist = [x for x in dist if x is not None]
            t = [fnum(r["avg_time_ms"]) for r in g]
            t = [x for x in t if x is not None]
            print(f"    {v:<24} {succ:>5} {cf:>9} "
                  f"{(sum(sat) / len(sat) if sat else float('nan')):>7.3f} "
                  f"{(sum(dist) / len(dist) if dist else float('nan')):>7.3f} "
                  f"{(sum(t) / len(t) if t else float('nan')):>9.1f} {len(g):>4}")


if __name__ == "__main__":
    main()
