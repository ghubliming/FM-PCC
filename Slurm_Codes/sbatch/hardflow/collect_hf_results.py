#!/usr/bin/env python3
"""Gen13 — collect ALL HardFlow/iMF train+eval metrics into one timestamped zip.

Login-node friendly: pure Python **stdlib** (no numpy/pandas/torch), read-only on
the logs tree, finishes in seconds. No conda env needed, no SLURM job needed.

    python3 Slurm_Codes/sbatch/hardflow/collect_hf_results.py

What it collects from logs/hardflow/avoiding-v0/ :
    flow/<run>/metrics.csv     training curves      -> train/<run>__metrics.csv
    flow/<run>/config.yaml     training provenance  -> train/<run>__config.yaml
    eval/<exp>/trajectories.csv eval results        -> eval/<exp>__trajectories.csv
    eval/<exp>/config.yaml     eval provenance      -> eval/<exp>__config.yaml
    + SUMMARY.md               aggregate tables (the useful bit)
    + MANIFEST.txt             what was included/skipped

DELIBERATELY SKIPPED (keeps the zip small — a few MB):
    *.pth (14 MB each), *_fan.npz, *.png       [add with --with-png / --with-npz]

Options:
    --root PATH     logs dir (default: <repo>/logs/hardflow/avoiding-v0)
    --out PATH      output zip (default: <repo>/temp/hf_results_<UTC-timestamp>.zip)
    --with-png      include *_real.png / *_fan.png
    --with-npz      include *_fan.npz (can be large)
"""

import argparse
import csv
import datetime as dt
import io
import os
import statistics as st
import zipfile

SKIP_EXT = {".pth", ".pt"}


# ----------------------------------------------------------------- helpers
def _read_csv(path):
    try:
        with open(path, newline="", errors="ignore") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _nums(rows, key):
    out = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if v and v.lower() != "nan":
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _mean(v):
    return st.mean(v) if v else float("nan")


def _fmt(v, spec=".4g"):
    return "—" if v != v else format(v, spec)      # NaN check


# ------------------------------------------------------------ eval summary
def summarize_eval(name, rows):
    n = len(rows)
    if not n:
        return None
    truthy = lambda k: sum(1 for r in rows if (r.get(k) or "").strip().lower() == "true")
    d = {
        "name": name,
        "n": n,
        "succ": 100.0 * truthy("success") / n,
        "safe": 100.0 * truthy("safety") / n,
        "viol": sum(_nums(rows, "total_violations")),
        "steps": _mean(_nums(rows, "steps")),
        "s_plan": _mean(_nums(rows, "average_computation_time")),
        "nfe": _mean(_nums(rows, "nfe_per_plan")),
        "nlp": _mean(_nums(rows, "nlp_solves")),
        "nlp_fail": sum(_nums(rows, "nlp_failures")),
        "rough": _mean(_nums(rows, "plan_roughness")),
        "rough_raw": _mean(_nums(rows, "plan_roughness_raw")),
    }
    return d


# ----------------------------------------------------------- train summary
def summarize_train(name, rows):
    if not rows:
        return None
    steps = _nums(rows, "step")
    u = _nums(rows, "raw_mse_u")
    if not u:                                   # FM curve: single "loss" column
        u = _nums(rows, "loss")
        metric = "loss"
    else:
        metric = "raw_mse_u"
    a0 = _nums(rows, "a0_mse")
    q = max(1, len(u) // 4)
    first_q, last_q = u[:q], u[-q:]
    return {
        "name": name,
        "metric": metric,
        "points": len(u),
        "last_step": max(steps) if steps else float("nan"),
        "first_med": st.median(first_q),
        "last_med": st.median(last_q),
        "min": min(u),
        "drop_x": (st.median(first_q) / st.median(last_q)) if st.median(last_q) else float("nan"),
        # plateau check: last quarter vs the quarter before it
        "delta_pct": (
            100.0 * (st.median(last_q) - st.median(u[-2 * q:-q])) / st.median(u[-2 * q:-q])
            if len(u) >= 2 * q and st.median(u[-2 * q:-q]) else float("nan")
        ),
        "a0_last": st.median(a0[-q:]) if a0 else float("nan"),
    }


# ------------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(repo, "logs", "hardflow", "avoiding-v0"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-png", action="store_true")
    ap.add_argument("--with-npz", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"ERROR: root not found: {args.root}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(repo, "temp", f"hf_results_{stamp}.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    train_rows, eval_rows, manifest, skipped = [], [], [], {"pth": 0, "png": 0, "npz": 0}

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for section, sub in (("train", "flow"), ("eval", "eval")):
            base = os.path.join(args.root, sub)
            if not os.path.isdir(base):
                continue
            for run in sorted(os.listdir(base)):
                rdir = os.path.join(base, run)
                if not os.path.isdir(rdir):
                    continue
                for fn in sorted(os.listdir(rdir)):
                    fp = os.path.join(rdir, fn)
                    if not os.path.isfile(fp):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in SKIP_EXT:
                        skipped["pth"] += 1
                        continue
                    if ext == ".png" and not args.with_png:
                        skipped["png"] += 1
                        continue
                    if ext == ".npz" and not args.with_npz:
                        skipped["npz"] += 1
                        continue
                    arc = f"{section}/{run}__{fn}"
                    z.write(fp, arc)
                    manifest.append(f"{arc}  ({os.path.getsize(fp)} B)")

                # summaries
                if section == "eval":
                    s = summarize_eval(run, _read_csv(os.path.join(rdir, "trajectories.csv")))
                    if s:
                        eval_rows.append(s)
                else:
                    s = summarize_train(run, _read_csv(os.path.join(rdir, "metrics.csv")))
                    if s:
                        train_rows.append(s)

        # ---------------- SUMMARY.md ----------------
        b = io.StringIO()
        b.write(f"# HardFlow / Gen13-iMF results summary\n\n")
        b.write(f"Collected (UTC): {stamp}  ·  root: `{args.root}`\n\n")

        b.write("## Training runs\n\n")
        if train_rows:
            b.write("| run | metric | pts | last step | first-quarter med | last-quarter med | min | drop | Δ last-vs-prev quarter | a0 (last) |\n")
            b.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for t in sorted(train_rows, key=lambda x: x["name"]):
                b.write(
                    f"| `{t['name']}` | {t['metric']} | {t['points']} | {_fmt(t['last_step'],'.0f')} "
                    f"| {_fmt(t['first_med'])} | {_fmt(t['last_med'])} | {_fmt(t['min'])} "
                    f"| {_fmt(t['drop_x'],'.2f')}× | {_fmt(t['delta_pct'],'+.1f')}% | {_fmt(t['a0_last'],'.4f')} |\n"
                )
            b.write("\n*Δ near 0% ⇒ plateaued (more steps unlikely to help). "
                    "Judge iMF on `raw_mse_u`, never the adaptive `loss`.*\n")
        else:
            b.write("_none found_\n")

        b.write("\n## Eval runs\n\n")
        if eval_rows:
            b.write("| exp | n | succ% | safe% | viol | steps | s/plan | NFE | NLP | NLPfail | roughness | roughness_raw |\n")
            b.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for e in sorted(eval_rows, key=lambda x: x["name"]):
                b.write(
                    f"| `{e['name']}` | {e['n']} | {e['succ']:.0f} | {e['safe']:.0f} | {e['viol']:.0f} "
                    f"| {_fmt(e['steps'],'.1f')} | {_fmt(e['s_plan'],'.4f')} | {_fmt(e['nfe'],'.0f')} "
                    f"| {_fmt(e['nlp'],'.0f')} | {e['nlp_fail']:.0f} "
                    f"| {_fmt(e['rough'],'.3e')} | {_fmt(e['rough_raw'],'.3e')} |\n"
                )
            b.write("\n*`_from_<run>` in a name = which training produced it "
                    "(absent ⇒ the default H16_imf_100k / H16_1e6steps checkpoint).*\n")
        else:
            b.write("_none found_\n")
        z.writestr("SUMMARY.md", b.getvalue())

        # ---------------- MANIFEST.txt ----------------
        m = io.StringIO()
        m.write(f"collected UTC {stamp}\nroot {args.root}\n\n")
        m.write(f"skipped: {skipped['pth']} checkpoint(s), {skipped['png']} png, {skipped['npz']} npz\n")
        m.write(f"(re-run with --with-png / --with-npz to include them)\n\n")
        m.write("\n".join(manifest))
        z.writestr("MANIFEST.txt", m.getvalue())

    size_mb = os.path.getsize(out) / 1e6
    print(f"[ collect ] {len(train_rows)} training run(s), {len(eval_rows)} eval run(s)")
    print(f"[ collect ] skipped {skipped['pth']} checkpoints, {skipped['png']} png, {skipped['npz']} npz")
    print(f"[ collect ] wrote {out}  ({size_mb:.2f} MB)")
    print(f"[ collect ] open SUMMARY.md inside the zip for the aggregate tables")


if __name__ == "__main__":
    main()
