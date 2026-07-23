"""Gen12 results aggregator — arms A/B/C at matched K (PLAN §5).

Reads the provenance-encoding directories written by eval_FM_v3_hardflow.py
(`results/halfspace_<variant>/K<K>_n<n>/<arm>.npz`) and prints one table per arm.
`--flow-steps` selects which matched-K bucket to report; mixing buckets in one
table is exactly the Gen13 fix_7 error and is not possible here by construction.
"""
import argparse
import os

import yaml
import numpy as np
import matplotlib.pyplot as plt
import flow_matcher_v3_hardflow.utils as utils

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('--config', default='config/hardflow_projection_eval.yaml')
cli.add_argument('--flow-steps', type=int, help='K bucket to report (default: config value).')
args_cli, _ = cli.parse_known_args()

# Load configuration
with open(args_cli.config, 'r') as file:
    config = yaml.safe_load(file)

projection_variants = config['projection_variants']
n_trials = config['n_trials']
flow_steps = args_cli.flow_steps if args_cli.flow_steps is not None else config['flow_steps']
run_tag = f'K{flow_steps}_n{n_trials}'

exp = 'avoiding-d3il'
class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

seeds = config['seeds']
avoiding_halfspace_variants = config['avoiding_halfspace_variants']

sr_goal_all = {}
sr_constraints_all = {}
timesteps_avg_all = {}
timesteps_std_all = {}

for variant in projection_variants:
    n_success_all = np.array([])
    n_success_and_constraints_all = np.array([])
    n_steps_all = np.array([])
    n_violations_all = np.array([])
    total_violations_all = np.array([])
    collision_free_completed_all = np.array([])
    avg_time_all = np.array([])
    nfe_all, nlp_solves_all, nlp_failures_all = [], [], []
    for halfspace_variant in avoiding_halfspace_variants:
        for i, seed in enumerate(seeds):
            args = Parser().parse_args(experiment='plan_fm_v3_hardflow', seed=seed)

            # Get data
            try:
                path = f'{args.savepath}/results/halfspace_{halfspace_variant}/{run_tag}/{variant}.npz'
                data = np.load(path, allow_pickle=True)
                for key, sink in (('nfe', nfe_all), ('nlp_solves', nlp_solves_all),
                                  ('nlp_failures', nlp_failures_all)):
                    if key in data.files:
                        sink.append(int(data[key]))
                n_success = data["n_success"]
                n_success_and_constraints = data["n_success_and_constraints"]
                n_steps = data["n_steps"]
                n_violations = data["n_violations"]
                total_violations = data["total_violations"]
                avg_time = data["avg_time"]
                collision_free_completed = data["collision_free_completed"]

                n_success_all = np.append(n_success_all, n_success)
                n_success_and_constraints_all = np.append(n_success_and_constraints_all, n_success_and_constraints)
                n_steps_all = np.append(n_steps_all, n_steps[n_success > 0])
                n_violations_all = np.append(n_violations_all, n_violations)
                total_violations_all = np.append(total_violations_all, total_violations)
                collision_free_completed_all = np.append(collision_free_completed_all, collision_free_completed)
                # Inherited bug: the original kept only the LAST file's avg_time and
                # printed it as the variant's mean. Aggregate it like everything else.
                avg_time_all = np.append(avg_time_all, avg_time)
            except FileNotFoundError:
                print(f"[ Error ] Could not find results at: {path}")
                continue

    if len(n_success_all) == 0:
        print(f"Skipping variant {variant} due to no data found.")
        continue

    success_rate_goal = n_success_all.mean()
    success_rate_goal_constraints = n_success_and_constraints_all.mean()
    success_rate_constraints = collision_free_completed_all.mean()
    steps_avg = n_steps_all.mean() if len(n_steps_all) > 0 else 0
    steps_std = n_steps_all.std() if len(n_steps_all) > 0 else 0
    n_violations_avg = n_violations_all.mean()
    n_violations_std = n_violations_all.std()
    total_violations_avg = total_violations_all.mean()
    total_violations_std = total_violations_all.std()

    print(f'------------------ Variant: {variant}  (K = {flow_steps}) ------------------')
    print(f'Success rate (goal): {success_rate_goal:.2f}')
    print(f'Success rate (goal + constraints): {success_rate_goal_constraints:.2f}')
    print(f'Success rate (constraints): {success_rate_constraints:.2f}')
    print(f'Average steps: {steps_avg:.2f} +- {steps_std:.2f}')
    print(f'Average violations: {n_violations_avg:.2f} +- {n_violations_std:.2f}')
    print(f'Average total violations: {total_violations_avg:.3f} +- {total_violations_std:.3f}')
    print(f'Average time (s/plan): {avg_time_all.mean():.3f} +- {avg_time_all.std():.3f}')
    # PLAN §5 requires compute to be reported next to success. Arm C costs 2 NFE per
    # ODE step plus one NLP solve; arms A/B cost 1 NFE and (arm B) one SLSQP projection.
    if nfe_all:
        print(f'NFE (sum over runs): {sum(nfe_all)}   '
              f'NLP solves: {sum(nlp_solves_all)}   NLP failures: {sum(nlp_failures_all)}')
    # PLAN §5.1: rank arms by SUCCESS, not smoothness. Post-projection roughness is
    # ~identical across models because the NLP flattens everything to one level.
    print(f'Success per second (goal+constraints / s per plan): '
          f'{success_rate_goal_constraints / max(avg_time_all.mean(), 1e-9):.2f}')
    print(rf'${steps_avg:.1f} \pm {steps_std:.1f}$ & ${success_rate_goal:.2f}$ & ${success_rate_constraints:.2f}$ & ${n_violations_avg:.1f} \pm {n_violations_std:.1f}$ \\\\')

    sr_goal_all[variant] = success_rate_goal
    sr_constraints_all[variant] = success_rate_constraints
    timesteps_avg_all[variant] = steps_avg
    timesteps_std_all[variant] = steps_std


# ── Gen12 three-arm comparison plot ───────────────────────────────────────────
# Replaces the inherited DPCC-R/T/C bar chart, which compared trajectory-selection
# rules (all arm B) and wrote its PNGs into FM_test/ — a different generation's
# folder. Gen12 compares A vs B vs C at ONE matched K and writes next to its own
# results.
variants_present = [v for v in projection_variants if v in sr_goal_all]
if not variants_present:
    print('\nNo arms had data; nothing to plot.')
else:
    out_dir = os.path.join('FM_v3_hardflow_test', 'results_plots')
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(len(variants_present))
    width = 0.35
    fig, ax = plt.subplots(figsize=(2.5 * len(variants_present) + 4, 8))
    bars1 = ax.bar(x - width / 2, [sr_goal_all[v] for v in variants_present], width,
                   label='Goal reached', color='green')
    bars2 = ax.bar(x + width / 2, [sr_constraints_all[v] for v in variants_present], width,
                   label='Constraints satisfied', color='red')
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_xticks(x)
    ax.set_xticklabels(variants_present, fontsize=11)
    ax.set_title(f'Gen12 — matched K = {flow_steps}, n = {n_trials} x {len(seeds)} seeds', fontsize=13)
    ax.legend(loc='lower left', fontsize=12)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom')
    fig.tight_layout()
    dest = os.path.join(out_dir, f'success_rates_K{flow_steps}_n{n_trials}.png')
    fig.savefig(dest)
    plt.close(fig)
    print(f'\nWrote {dest}')
