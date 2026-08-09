#!/usr/bin/env python3
"""
DA — Gen14 U7 horizontal VS: engines AND projector variants, with paired statistics.

Companion script for DA_20260809_gen14_horizontal_4engines.md.

Design note (this is what makes the single seed legitimate):
    the batch is ONE seed but NOT one run. Each K=2 engine ran
    19 projector variants x 30 contexts x 2 geos = 1140 rollouts,
    every variant on the SAME contexts. So variant-vs-variant is a
    within-model PAIRED comparison with n = 112 (2 engines x 56 unfrozen
    context-cells) or n = 336 when 3 selection rules are stacked.
    Engine-vs-engine is paired on contexts too, but is confounded with
    "this particular checkpoint" — see the MD.

Tests (no scipy in this container, so both are implemented directly):
    - paired sign-flip permutation test for continuous metrics (20k resamples)
    - exact McNemar (binomial) for paired binary outcomes
    - Holm-Bonferroni correction reported per family

Stdlib only. Usage:
    python3 da_20260809_gen14_horizontal_4engines.py [BATCH_DIR]
"""

import csv
import itertools
import math
import random
import statistics as st
import sys
from collections import Counter

BATCH = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspaces/FM-PCC/temp/2026-08-07/batch_va2_20260809_103838"

RESAMPLES = 20000
SEED = 0

# Candidate id -> engine label. Batch-local; check candidates_summary.txt before reuse.
LAB = {"5": "af-K100", "6": "af-K2", "7": "diffusion-K100", "8": "fm-K100",
       "9": "mf-K100", "10": "mf-K2", "11": "Gen6V4-K20"}
K100 = ["7", "8", "9", "5"]
K2 = ["10", "6"]                      # mf-K2, af-K2 — the two with the full variant grid
SEL = ["r", "c", "t"]                 # trajectory-selection rules: random / min-cost / temporal
GEO = "combined_5"
DIVERGE_M = 1.0
STUCK_M = 0.02


# ---------------------------------------------------------------- data access

def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load():
    with open(f"{BATCH}/per_rollout_detail.csv") as fh:
        return list(csv.DictReader(fh))


def cells(rows, cand, variant, drop_frozen=True):
    """{(geo, rollout_idx): row} — the pairing key for every paired test below."""
    out = {}
    for r in rows:
        if r["Candidate"] != cand or r["variant"] != variant:
            continue
        if drop_frozen and r["frozen"] in ("1.0", "1", "True"):
            continue
        out[(r["geo"], r["rollout_idx"])] = r
    return out


def progress(r):
    return fnum(r["context_init_xy_dist"]) - fnum(r["context_final_xy_dist"])


# ------------------------------------------------------------------- testing

def perm_test(diffs, resamples=RESAMPLES):
    """Paired sign-flip permutation test on the mean difference. Returns (mean, p)."""
    d = [x for x in diffs if not math.isnan(x)]
    if not d:
        return float("nan"), float("nan")
    obs = sum(d) / len(d)
    rng = random.Random(SEED)
    hits = 0
    for _ in range(resamples):
        s = sum(x if rng.random() < 0.5 else -x for x in d) / len(d)
        if abs(s) >= abs(obs) - 1e-15:
            hits += 1
    return obs, (hits + 1) / (resamples + 1)


def mcnemar(pairs):
    """Exact two-sided McNemar on paired binary outcomes. Returns (b, c, p)."""
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, cc in pairs if cc and not a)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def star(p):
    if math.isnan(p):
        return "  "
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    prev = 0.0
    for rank, i in enumerate(idx):
        val = min(1.0, (len(pvals) - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


# ------------------------------------------------------------ paired compare

CONT = [("constraint_exec_sat_rate", "sat"), ("n_violations", "viol"),
        ("max_phys_error_per_rollout", "peErr"), ("avg_time_ms", "t_ms")]
BINY = [("collision_free_completed", "cfree"), ("n_success", "succ"),
        ("n_success_and_constraints", "s+c")]


def paired(rows, cands, vA, vB):
    """Stack `cands` and compare variant vA against vB on shared contexts."""
    res, n = {}, 0
    for col, short in CONT:
        d = []
        for c in cands:
            A, B = cells(rows, c, vA), cells(rows, c, vB)
            keys = sorted(set(A) & set(B))
            d += [fnum(A[k][col]) - fnum(B[k][col]) for k in keys]
        res[short] = perm_test(d)
        n = max(n, len(d))
    for col, short in BINY:
        pr = []
        for c in cands:
            A, B = cells(rows, c, vA), cells(rows, c, vB)
            keys = sorted(set(A) & set(B))
            pr += [(fnum(A[k][col], 0) > 0, fnum(B[k][col], 0) > 0) for k in keys]
        b, cc, p = mcnemar(pr)
        res[short] = (sum(1 for x, _ in pr if x), sum(1 for _, y in pr if y), b, cc, p)
    res["n"] = n
    return res


def row_line(label, res):
    sat, psat = res["sat"]
    vio, pvio = res["viol"]
    tms, ptms = res["t_ms"]
    a1, b1, _, _, pcf = res["cfree"]
    print(f"  {label:<34} n={res['n']:<4} sat {sat:+.4f} {star(psat):<3} | "
          f"viol {vio:+8.2f} {star(pvio):<3} | cfree {a1:>3}v{b1:<3} {star(pcf):<3} | "
          f"t {tms:+8.1f} {star(ptms)}")
    return [psat, pvio, pcf]


# ------------------------------------------------------------- descriptive

HDR = (f"  {'engine':<15}{'n':>4}{'succ%':>7}{'rel%':>7}{'s+c%':>7}{'cfree%':>8}"
       f"{'div%':>7}{'peErr':>8}{'sat%':>7}{'viol':>7}{'prog_m':>8}{'stuck%':>8}"
       f"{'final_m':>8}{'t_ms':>11}")


def med(v):
    v = [x for x in v if not math.isnan(x)]
    return round(st.median(v), 4) if v else float("nan")


def pct(f):
    return round(100 * sum(f) / len(f), 1) if f else float("nan")


def describe(rows, cands, variant, geo=GEO, idxs=None):
    print(HDR)
    for c in cands:
        s = [r for r in rows if r["Candidate"] == c and r["variant"] == variant
             and r["geo"] == geo and (idxs is None or r["rollout_idx"] in idxs)]
        if not s:
            continue
        pr = [progress(r) for r in s]
        print(f"  {LAB[c]:<15}{len(s):>4}"
              f"{pct([fnum(r['n_success']) > 0 for r in s]):>7}"
              f"{pct([fnum(r['success_relaxed']) > 0 for r in s]):>7}"
              f"{pct([fnum(r['n_success_and_constraints']) > 0 for r in s]):>7}"
              f"{pct([fnum(r['collision_free_completed']) > 0 for r in s]):>8}"
              f"{pct([fnum(r['max_phys_error_per_rollout']) > DIVERGE_M for r in s]):>7}"
              f"{med([fnum(r['max_phys_error_per_rollout']) for r in s]):>8}"
              f"{round(100 * med([fnum(r['constraint_exec_sat_rate']) for r in s]), 1):>7}"
              f"{med([fnum(r['n_violations']) for r in s]):>7}"
              f"{med(pr):>8}{pct([abs(p) < STUCK_M for p in pr if not math.isnan(p)]):>8}"
              f"{med([fnum(r['context_final_xy_dist']) for r in s]):>8}"
              f"{med([fnum(r['avg_time_ms']) for r in s]):>11}")


def variant_ranking(rows, cands=K2):
    tab = []
    for v in sorted(set(r["variant"] for r in rows if r["Candidate"] == cands[0])):
        pool = []
        for c in cands:
            pool += list(cells(rows, c, v).values())
        n = len(pool)
        tab.append((
            pct([fnum(r["collision_free_completed"], 0) > 0 for r in pool]),
            100 * sum(fnum(r["constraint_exec_sat_rate"]) for r in pool) / n,
            v, sum(fnum(r["n_violations"]) for r in pool) / n,
            sum(1 for r in pool if fnum(r["n_success"], 0) > 0),
            med([fnum(r["max_phys_error_per_rollout"]) for r in pool]),
            sum(fnum(r["avg_time_ms"]) for r in pool) / n, n))
    print(f"  {'variant':<24}{'sat%':>7}{'viol':>8}{'cfree%':>8}{'succ':>6}{'peErr':>8}{'t_ms':>9}{'n':>5}")
    for cf, sat, v, vio, su, pe, t, n in sorted(tab, reverse=True):
        print(f"  {v:<24}{sat:>7.1f}{vio:>8.1f}{cf:>8.1f}{su:>6}{pe:>8.3f}{t:>9.1f}{n:>5}")


# ------------------------------------------------------------------- report

def main():
    rows = load()
    print(f"batch: {BATCH}\nrollout rows: {len(rows)}")
    frozen = Counter(r["geo"] for r in rows
                     if r["Candidate"] in K2 and r["frozen"] in ("1.0", "1", "True"))
    print(f"frozen rollouts dropped from paired tests: {dict(frozen)}\n")

    print("#" * 84)
    print("# PART 1 — ENGINE AXIS (descriptive)")
    print("#" * 84)
    print("\n### A. K=100, unprojected (`diffuser`), same 30 contexts")
    describe(rows, K100, "diffuser")
    print("\n### B. K=100, projected (`dpcc-r`) — RAGGED, evals hit the 24 h cap")
    describe(rows, K100, "dpcc-r")
    common = set.intersection(*[set(cells(rows, c, "dpcc-r", drop_frozen=False))
                                for c in K100])
    common = set(k[1] for k in common)
    print(f"\n### C. K=100, `dpcc-r` PAIRED on the {len(common)} contexts all four finished")
    describe(rows, K100, "dpcc-r", idxs=common)
    print("\n### D. K=2 flow arms, unprojected")
    describe(rows, K2, "diffuser")
    print("\n### E. K=2 flow arms, `dpcc-r`")
    describe(rows, K2, "dpcc-r")
    print("\n### F. Gen6V4 K=20 reference")
    describe(rows, ["11"], "diffuser")
    describe(rows, ["11"], "dpcc-r")

    print("\n### G. Mode coverage (`mode_encoding`), unprojected")
    for c in K100 + K2 + ["11"]:
        s = [r for r in rows if r["Candidate"] == c and r["variant"] == "diffuser"
             and r["geo"] == GEO]
        if not s:
            continue
        cnt = Counter(r["mode_encoding"] for r in s)
        tot = sum(cnt.values())
        print(f"  {LAB[c]:<15} n={tot:<4} " +
              " ".join(f"mode{k}={v} ({100 * v / tot:.0f}%)" for k, v in sorted(cnt.items())))

    print("\n" + "#" * 84)
    print("# PART 2 — ENGINE AXIS (paired significance, K=100, 30 contexts)")
    print("#" * 84)
    ps, labels = [], []
    for a, b in itertools.combinations(K100, 2):
        A, B = cells(rows, a, "diffuser"), cells(rows, b, "diffuser")
        ks = sorted(set(A) & set(B))
        pr = [(fnum(A[k]["max_phys_error_per_rollout"]) > DIVERGE_M,
               fnum(B[k]["max_phys_error_per_rollout"]) > DIVERGE_M) for k in ks]
        _, _, pdiv = mcnemar(pr)
        m, psat = perm_test([fnum(A[k]["constraint_exec_sat_rate"])
                             - fnum(B[k]["constraint_exec_sat_rate"]) for k in ks])
        print(f"  {LAB[a]:>14} vs {LAB[b]:<14} n={len(ks)}  "
              f"div {sum(1 for u, _ in pr if u)}/{len(pr)} vs {sum(1 for _, v in pr if v)}/{len(pr)}"
              f"  p={pdiv:.4f} {star(pdiv):<3} | sat {m:+.4f} p={psat:.4f} {star(psat)}")
        ps.append(pdiv)
        labels.append(f"{LAB[a]} vs {LAB[b]}")
    print("  Holm-adjusted divergence p-values (6 comparisons):")
    for lb, p in zip(labels, holm(ps)):
        print(f"    {lb:<32} {p:.4f} {star(p)}")

    print("\n### mf vs af at K=2, paired over ALL 19 variants")
    for col, short in [("constraint_exec_sat_rate", "sat"), ("n_violations", "viol")]:
        d = []
        for v in sorted(set(r["variant"] for r in rows if r["Candidate"] == "10")):
            A, B = cells(rows, "10", v), cells(rows, "6", v)
            d += [fnum(A[k][col]) - fnum(B[k][col]) for k in sorted(set(A) & set(B))]
        m, p = perm_test(d)
        print(f"  {short:<6} n={len(d):<5} mean(mf-af)={m:+9.4f}  p={p:.4f} {star(p)}")
    for col, short in [("n_success", "succ"), ("collision_free_completed", "cfree")]:
        pr = []
        for v in sorted(set(r["variant"] for r in rows if r["Candidate"] == "10")):
            A, B = cells(rows, "10", v), cells(rows, "6", v)
            pr += [(fnum(A[k][col], 0) > 0, fnum(B[k][col], 0) > 0)
                   for k in sorted(set(A) & set(B))]
        b, cc, p = mcnemar(pr)
        print(f"  {short:<6} n={len(pr):<5} mf={sum(1 for x, _ in pr if x)} "
              f"af={sum(1 for _, y in pr if y)}  discordant {b}/{cc}  p={p:.4f} {star(p)}")

    print("\n" + "#" * 84)
    print("# PART 3 — PROJECTOR AXIS (paired, pooled over mf-K2 + af-K2)")
    print("#" * 84)

    print("\n### H. HardFlow (arm C) vs DPCC — matched selection rule, per engine")
    fam = []
    for cand in K2:
        for s in SEL:
            fam += row_line(f"{LAB[cand]}  hf-{s} - dpcc-{s}",
                            paired(rows, [cand], f"hardflow_new-{s}", f"dpcc-{s}"))
    print("\n### H2. HardFlow vs DPCC — all 6 cells stacked (n = 336)")
    stacked = {}
    for s in SEL:
        stacked[s] = paired(rows, K2, f"hardflow_new-{s}", f"dpcc-{s}")
    d_sat, d_vio, d_t, pr_cf = [], [], [], []
    for cand in K2:
        for s in SEL:
            A, B = cells(rows, cand, f"hardflow_new-{s}"), cells(rows, cand, f"dpcc-{s}")
            ks = sorted(set(A) & set(B))
            d_sat += [fnum(A[k]["constraint_exec_sat_rate"])
                      - fnum(B[k]["constraint_exec_sat_rate"]) for k in ks]
            d_vio += [fnum(A[k]["n_violations"]) - fnum(B[k]["n_violations"]) for k in ks]
            d_t += [fnum(A[k]["avg_time_ms"]) - fnum(B[k]["avg_time_ms"]) for k in ks]
            pr_cf += [(fnum(A[k]["collision_free_completed"], 0) > 0,
                       fnum(B[k]["collision_free_completed"], 0) > 0) for k in ks]
    for name, d in [("sat", d_sat), ("viol", d_vio), ("t_ms", d_t)]:
        m, p = perm_test(d)
        print(f"  {name:<6} n={len(d):<4} mean(HF-DPCC)={m:+9.4f}  p={p:.4f} {star(p)}")
    b, cc, p = mcnemar(pr_cf)
    print(f"  cfree  n={len(pr_cf):<4} HF={sum(1 for x, _ in pr_cf if x)} "
          f"DPCC={sum(1 for _, y in pr_cf if y)}  discordant {b}/{cc}  p={p:.4f} {star(p)}")

    print("\n### I. Does any projector beat no projection? (vs `diffuser`)")
    ps = []
    for v in ["dpcc-r", "dpcc-c", "dpcc-t", "hardflow_new-r", "hardflow_new-c",
              "hardflow_new-t", "gradient", "post_processing"]:
        ps.append(row_line(f"{v} - diffuser", paired(rows, K2, v, "diffuser"))[0])

    print("\n### J. Trajectory-selection rule (random / min-proj-cost / temporal)")
    for a, b in [("dpcc-c", "dpcc-r"), ("dpcc-t", "dpcc-r"), ("dpcc-t", "dpcc-c")]:
        row_line(f"{a} - {b}", paired(rows, K2, a, b))
    print("  --- same three rules inside HardFlow ---")
    for a, b in [("hardflow_new-c", "hardflow_new-r"), ("hardflow_new-t", "hardflow_new-r"),
                 ("hardflow_new-t", "hardflow_new-c")]:
        row_line(f"{a} - {b}", paired(rows, K2, a, b))

    print("\n### K. Dynamics-timestep sweep inside dpcc-c")
    dt_ps = []
    for v in ["dpcc-c-dt0p25", "dpcc-c-dt0p5", "dpcc-c-dt2p0", "dpcc-c-dt4p0"]:
        dt_ps.append(row_line(f"{v} - dpcc-c", paired(rows, K2, v, "dpcc-c"))[0])
    print("  Holm-adjusted sat p-values (4 comparisons): " +
          "  ".join(f"{p:.4f}{star(p)}" for p in holm(dt_ps)))

    print("\n### L. Constraint ablations — which constraint carries the projector?")
    for v in ["geo_free", "model_free", "bounds_free", "geo_free-bounds_free",
              "model_free-bounds_free", "geo_free-model_free"]:
        row_line(f"{v} - dpcc-c", paired(rows, K2, v, "dpcc-c"))

    print("\n### M. Full variant ranking (pooled mf-K2 + af-K2, unfrozen)")
    variant_ranking(rows)

    print("\n### N. `bounds_free` against every other variant")
    for v in sorted(set(r["variant"] for r in rows if r["Candidate"] == "10")):
        if v == "bounds_free":
            continue
        row_line(f"bounds_free - {v}", paired(rows, K2, "bounds_free", v))


if __name__ == "__main__":
    main()
