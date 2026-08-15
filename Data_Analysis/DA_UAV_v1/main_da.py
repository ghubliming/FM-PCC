"""
DA_UAV_v1 — single-run analysis CLI.

Convenience wrapper for "analyse this one eval-tag folder". `--input-path`
points at the folder that holds the seed directories — the eval tag, e.g.

    logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/H8_DMeanFlowODE_9D_dp0.5_bbunet/Emf_K4_mpc4_pid_stopgo_T0.5

not at its parent.

    python main_da.py --input-path <.../Emf_K4_mpc4_pid_stopgo_T0.5>
    python main_da.py --input-path <candidate> --seeds 6 --variants dpcc-c

Everything else — filters, outputs, CSV schemas — is identical to
`main_da_batch.py`; this just points discovery one level lower so the single
folder is picked up as the only candidate. Note that a single candidate is ONE
K value, so `uav_k_sweep.csv` will have one point per curve and the K plots are
skipped: a sweep needs the batch CLI over the parent tree.
"""

import argparse
import os
import sys

import main_da_batch


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='DA_UAV_v1 — single-candidate analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--input-path', required=True,
                        help='Eval-tag folder containing the seed subdirectories')
    parser.add_argument('--output-path', default='./uav_single_analysis')
    parser.add_argument('--seeds', default=None)
    parser.add_argument('--variants', default=None)
    parser.add_argument('--geos', default=None)
    parser.add_argument('--splits', default=None)
    parser.add_argument('--source', choices=['auto', 'npz', 'json'], default='auto')
    parser.add_argument('--no-diagnostics-scan', action='store_true')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    # discover_candidates() looks *below* the parent, so hand it the parent of
    # the requested folder and keep only that folder.
    target = os.path.abspath(args.input_path.rstrip(os.sep))
    parent = os.path.dirname(target)
    name = os.path.basename(target)

    forwarded = ['--parent-path', parent, '--output-path', args.output_path]
    for flag, value in (('--seeds', args.seeds), ('--variants', args.variants),
                        ('--geos', args.geos), ('--splits', args.splits),
                        ('--source', args.source)):
        if value:
            forwarded += [flag, value]
    if args.no_diagnostics_scan:
        forwarded.append('--no-diagnostics-scan')
    if args.plots:
        forwarded.append('--plots')
    if args.verbose:
        forwarded.append('--verbose')

    # Restrict to the requested folder by index, resolved after discovery.
    from discovery import discover_candidates
    from utils import parse_int_list
    candidates = discover_candidates(parent, seed_list=parse_int_list(args.seeds))
    wanted = [str(key) for key, info in candidates.items() if info['name'] == name]
    if not wanted:
        print(f'[ DA_UAV_v1 ] No seed/result tree found under: {target}')
        return 1
    forwarded += ['--candidates', ','.join(wanted)]

    return main_da_batch.main(forwarded)


if __name__ == '__main__':
    sys.exit(main())
