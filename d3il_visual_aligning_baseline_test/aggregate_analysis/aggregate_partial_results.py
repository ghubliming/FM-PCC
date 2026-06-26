#!/usr/bin/env python3
"""
Aggregate PARTIAL D3IL visual-aligning eval results from per-rollout diagnostics.

WHY THIS EXISTS
---------------
The paper-faithful eval runs 60 contexts x 18 trajectories = 1080 rollouts/seed and only
writes the per-seed summary (`results_seed_{s}.json`, with `success_rate` / `entropy`) AFTER
all 1080 finish (eval_d3il_visual_aligning.py:536). When a SLURM job hits its wall-clock limit
mid-loop, every completed rollout has *already* been exported to
    logs/d3il_visual_aligning_baseline/<agent>/seed_{s}/diagnostics/rollout_{N}_stats.json
but the per-seed summary is never written, so the headline numbers are "lost" even though the
raw data survives. This tool reconstructs `success_rate` + behavior `entropy` from those partial
diagnostics, using the EXACT paper formula, with honest coverage/bias reporting.

The eval loop is CONTEXT-MAJOR (`for ctx: for traj in range(n_trajs)`), so a time-limited run
has: low-index contexts complete, high-index contexts missing entirely, and (usually) one
context half-done at the cutoff. The aggregate is therefore biased toward low-index contexts —
this tool prints that caveat and the coverage so the number is read provisionally.

Read-only on your results. Torch-free (stdlib + numpy only) — runs anywhere the JSONs are.

USAGE
-----
    # all seeds under the default logs root, paper scale (60 x 18)
    python d3il_visual_aligning_baseline_test/aggregate_analysis/aggregate_partial_results.py

    # specific seeds, custom agent, also emit per-seed results_seed_{s}.json (marked partial)
    python .../aggregate_partial_results.py --seeds 0 1 2 3 4 42 --write-seed-json

    # if you scp'd the diagnostics elsewhere
    python .../aggregate_partial_results.py --logs-root /path/to/logs/d3il_visual_aligning_baseline
"""

import os
import sys
import glob
import json
import argparse
import datetime

import numpy as np

# Paper reference (D3IL DDPM-ACT image aligning, Table 3): success 0.278+/-0.071, entropy 0.139+/-0.054
PAPER_SUCCESS = (0.278, 0.071)
PAPER_ENTROPY = (0.139, 0.054)

DEFAULT_LOGS_ROOT = "logs/d3il_visual_aligning_baseline"
DEFAULT_AGENT = "ddpm_encdec_vision"


# ─────────────────────────────────────────────────────────────────────────────
# Core metric — faithful to eval_d3il_visual_aligning.py:compute_behavior_entropy
# ─────────────────────────────────────────────────────────────────────────────

def context_entropy(counts_by_ctx, ctx_list, n_modes):
    """Average conditional behavior entropy over the contexts in `ctx_list`.

    Faithful to the eval's `compute_behavior_entropy` (paper Eq. 2): for each context, among
    the SUCCESSFUL rollouts count how many used each mode, normalize to p(m|c), then
        entropy_c = -Sum_m p(m|c) log p(m|c) / log(n_modes)   in [0, 1].
    A context with zero successes contributes 0 (mode collapse / nothing to be diverse about),
    exactly as the original (its all-zero count row yields 0 after the +1e-12 guards).

    The original divides counts by n_trajs then row-normalizes — the n_trajs factor cancels, so
    this works unchanged on partial per-context data. The ONLY partial-data choice is WHICH
    contexts to average over (`ctx_list`): the eval averages over all n_contexts; here we pass
    only the reached (or only the fully-complete) contexts so unreached contexts don't inject
    phantom zeros and deflate the estimate.
    """
    if not ctx_list:
        return 0.0
    ents = []
    for c in ctx_list:
        counts = counts_by_ctx.get(c, np.zeros(n_modes, dtype=np.float64))
        total = counts.sum()
        if total <= 0:
            ents.append(0.0)          # no successful rollouts in this context
            continue
        p = counts / total
        ent = -(p * np.log(p + 1e-12) / np.log(n_modes)).sum()
        ents.append(float(ent))
    return float(np.mean(ents))


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

def recover_context(stats, n_trajs):
    """Context index for a rollout: prefer the recorded context_idx, else derive from the
    rollout index given the context-major loop order (ctx = ridx // n_trajs)."""
    ci = stats.get('context_info') or {}
    if isinstance(ci, dict) and 'context_idx' in ci:
        try:
            return int(ci['context_idx'])
        except (TypeError, ValueError):
            pass
    ridx = stats.get('rollout_index', None)
    if ridx is not None and n_trajs > 0:
        return int(ridx) // int(n_trajs)
    return -1


def aggregate_seed(seed_dir, n_contexts, n_trajs, n_modes):
    """Aggregate one seed's diagnostics dir. Returns a dict, or None if no stats files."""
    stat_files = sorted(glob.glob(os.path.join(seed_dir, 'diagnostics', 'rollout_*_stats.json')))
    if not stat_files:
        return None

    successes = []
    counts_by_ctx = {}          # ctx -> np.array(n_modes) of successful-rollout mode counts
    rollouts_by_ctx = {}        # ctx -> total rollouts seen (success or not)
    n_bad = 0

    for fp in stat_files:
        try:
            with open(fp) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError):
            n_bad += 1
            continue
        succ = bool(s.get('success', False))
        successes.append(1.0 if succ else 0.0)
        c = recover_context(s, n_trajs)
        if not (0 <= c < n_contexts):
            continue
        rollouts_by_ctx[c] = rollouts_by_ctx.get(c, 0) + 1
        if succ:
            m = int(s.get('mode', -1))
            if 0 <= m < n_modes:
                if c not in counts_by_ctx:
                    counts_by_ctx[c] = np.zeros(n_modes, dtype=np.float64)
                counts_by_ctx[c][m] += 1.0

    contexts_reached = sorted(rollouts_by_ctx.keys())
    contexts_complete = sorted(c for c, n in rollouts_by_ctx.items() if n >= n_trajs)

    success_rate = float(np.mean(successes)) if successes else 0.0
    entropy_reached = context_entropy(counts_by_ctx, contexts_reached, n_modes)
    entropy_complete = context_entropy(counts_by_ctx, contexts_complete, n_modes)
    score = 0.5 * (success_rate + entropy_reached)

    n_done = len(successes)
    n_expected = n_contexts * n_trajs
    complete = (n_done >= n_expected) and (len(contexts_complete) >= n_contexts)

    return {
        'n_rollouts_done': n_done,
        'n_rollouts_expected': n_expected,
        'coverage_frac': (n_done / n_expected) if n_expected else 0.0,
        'contexts_reached': len(contexts_reached),
        'contexts_complete': len(contexts_complete),
        'n_contexts': n_contexts,
        'n_trajectories_per_context': n_trajs,
        'success_rate': success_rate,
        'entropy_reached': entropy_reached,      # headline: avg over contexts with >=1 rollout
        'entropy_complete': entropy_complete,    # stricter: avg over fully 18/18 contexts only
        'score': score,
        'complete': bool(complete),
        'n_unreadable_files': n_bad,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pair(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"


def print_report(agent_name, per_seed, cross, n_contexts, n_trajs):
    line = "=" * 84
    print(line)
    print(f"  PARTIAL aggregate — {agent_name}   (paper scale = {n_contexts} ctx x {n_trajs} traj)")
    print(line)
    if not per_seed:
        print("  No seeds with diagnostics found. Nothing to aggregate.")
        print(line)
        return

    hdr = (f"  {'seed':>5} | {'done/exp':>12} | {'cov':>5} | {'ctx r/c':>9} | "
           f"{'success':>8} | {'H(reach)':>9} | {'H(compl)':>9} | {'score':>6} | done?")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for seed, r in sorted(per_seed.items(), key=lambda kv: _seed_key(kv[0])):
        done_flag = "FULL" if r['complete'] else "part"
        print(f"  {str(seed):>5} | {r['n_rollouts_done']:>5}/{r['n_rollouts_expected']:<6} | "
              f"{r['coverage_frac']*100:>4.0f}% | "
              f"{r['contexts_reached']:>3}/{r['contexts_complete']:<5} | "
              f"{r['success_rate']:>8.3f} | {r['entropy_reached']:>9.3f} | "
              f"{r['entropy_complete']:>9.3f} | {r['score']:>6.3f} | {done_flag}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  ctx r/c = contexts reached (>=1 rollout) / fully-complete (={n_trajs} rollouts)")
    print(line)

    n = cross['n_seeds_with_data']
    print(f"  CROSS-SEED  (over {n} seed{'s' if n != 1 else ''} with data)")
    print(f"    success_rate    : {fmt_pair(cross['success_rate_mean'], cross['success_rate_std'])}"
          f"     (paper {fmt_pair(*PAPER_SUCCESS)})")
    print(f"    entropy (reach) : {fmt_pair(cross['entropy_reached_mean'], cross['entropy_reached_std'])}"
          f"     (paper {fmt_pair(*PAPER_ENTROPY)})")
    print(f"    entropy (compl) : {fmt_pair(cross['entropy_complete_mean'], cross['entropy_complete_std'])}")
    print(f"    score           : {fmt_pair(cross['score_mean'], cross['score_std'])}")
    print(line)

    any_partial = any(not r['complete'] for r in per_seed.values())
    if any_partial:
        print("  [!] PARTIAL DATA — at least one seed did not finish all rollouts.")
        print("      The eval loop is context-major, so missing rollouts are the HIGH-index")
        print("      contexts: this aggregate is biased toward low-index initial states.")
        print("      Treat as a provisional read; re-run to completion (resume-capable eval +")
        print("      the bumped 24h wall-clock) for a paper-comparable number.")
        print(line)


def _seed_key(seed):
    try:
        return (0, int(seed))
    except (TypeError, ValueError):
        return (1, str(seed))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def discover_seed_dirs(agent_root, seeds):
    if seeds:
        return [(s, os.path.join(agent_root, f'seed_{s}')) for s in seeds]
    found = []
    for d in sorted(glob.glob(os.path.join(agent_root, 'seed_*'))):
        if os.path.isdir(d):
            seed = os.path.basename(d).replace('seed_', '')
            found.append((seed, d))
    return found


def main():
    p = argparse.ArgumentParser(
        description='Aggregate partial D3IL visual-aligning eval results from per-rollout diagnostics.')
    p.add_argument('--logs-root', default=DEFAULT_LOGS_ROOT,
                   help=f'Root holding <agent>/seed_*/diagnostics/ (default: {DEFAULT_LOGS_ROOT}).')
    p.add_argument('--agent-name', default=DEFAULT_AGENT,
                   help=f'Agent subdir name (default: {DEFAULT_AGENT}).')
    p.add_argument('--seeds', nargs='+', default=None,
                   help='Seeds to include (default: auto-discover all seed_* dirs).')
    p.add_argument('--n-contexts', type=int, default=60, help='Paper scale n_contexts (default: 60).')
    p.add_argument('--n-trajs', type=int, default=18,
                   help='Paper scale n_trajectories_per_context (default: 18).')
    p.add_argument('--n-modes', type=int, default=2, help='Number of behavior modes (default: 2).')
    p.add_argument('--out', default=None,
                   help='Output JSON path (default: <agent_root>/aggregate_partial.json).')
    p.add_argument('--write-seed-json', action='store_true',
                   help='Also write per-seed results_seed_{s}.json (real eval schema, marked partial).')
    args = p.parse_args()

    agent_root = os.path.join(args.logs_root, args.agent_name)
    if not os.path.isdir(agent_root):
        print(f"[ error ] agent root not found: {agent_root}", file=sys.stderr)
        print("          pass --logs-root / --agent-name, or run from the repo root.", file=sys.stderr)
        return 2

    seed_dirs = discover_seed_dirs(agent_root, args.seeds)
    per_seed = {}
    for seed, sdir in seed_dirs:
        r = aggregate_seed(sdir, args.n_contexts, args.n_trajs, args.n_modes)
        if r is None:
            print(f"[ skip ] seed {seed}: no diagnostics/rollout_*_stats.json in {sdir}")
            continue
        per_seed[seed] = r
        if args.write_seed_json:
            _write_seed_json(sdir, seed, args.agent_name, r)

    # cross-seed
    srs = [r['success_rate'] for r in per_seed.values()]
    ehr = [r['entropy_reached'] for r in per_seed.values()]
    ehc = [r['entropy_complete'] for r in per_seed.values()]
    scs = [r['score'] for r in per_seed.values()]
    cross = {
        'n_seeds_with_data': len(per_seed),
        'success_rate_mean': float(np.mean(srs)) if srs else 0.0,
        'success_rate_std':  float(np.std(srs)) if srs else 0.0,
        'entropy_reached_mean': float(np.mean(ehr)) if ehr else 0.0,
        'entropy_reached_std':  float(np.std(ehr)) if ehr else 0.0,
        'entropy_complete_mean': float(np.mean(ehc)) if ehc else 0.0,
        'entropy_complete_std':  float(np.std(ehc)) if ehc else 0.0,
        'score_mean': float(np.mean(scs)) if scs else 0.0,
        'score_std':  float(np.std(scs)) if scs else 0.0,
    }

    print_report(args.agent_name, per_seed, cross, args.n_contexts, args.n_trajs)

    out_path = args.out or os.path.join(agent_root, 'aggregate_partial.json')
    payload = {
        'agent_name': args.agent_name,
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'partial': any(not r['complete'] for r in per_seed.values()),
        'config': {
            'logs_root': args.logs_root,
            'n_contexts': args.n_contexts,
            'n_trajectories_per_context': args.n_trajs,
            'n_modes': args.n_modes,
        },
        'paper_reference': {'success_rate': list(PAPER_SUCCESS), 'entropy': list(PAPER_ENTROPY)},
        'per_seed': per_seed,
        'cross_seed': cross,
    }
    try:
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"  [ saved ] {out_path}")
    except OSError as e:
        print(f"  [ warn ] could not write {out_path}: {e}", file=sys.stderr)
    return 0


def _write_seed_json(seed_dir, seed, agent_name, r):
    """Emit a results_seed_{s}.json in the real eval schema, flagged partial, so existing
    downstream tooling (e.g. the guide's aggregation snippet) reads it without changes."""
    out = {
        'agent_name': agent_name,
        'seed': _seed_key(seed)[1] if _seed_key(seed)[0] == 0 else seed,
        'n_rollouts': r['n_rollouts_done'],
        'n_contexts': r['n_contexts'],
        'n_trajectories_per_context': r['n_trajectories_per_context'],
        'success_rate': r['success_rate'],
        'entropy': r['entropy_reached'],
        'score': r['score'],
        'partial': not r['complete'],
        'coverage_frac': r['coverage_frac'],
        'contexts_reached': r['contexts_reached'],
        'contexts_complete': r['contexts_complete'],
        '_source': 'aggregate_partial_results.py (reconstructed from diagnostics)',
    }
    fp = os.path.join(seed_dir, f'results_seed_{seed}.json')
    try:
        with open(fp, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"  [ saved ] {fp}  (partial={out['partial']})")
    except OSError as e:
        print(f"  [ warn ] could not write {fp}: {e}", file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
