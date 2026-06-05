# Load results — FM visual avoiding.
# Copy-modified from scripts/load_results.py.
# Swap: diffuser.utils → fm_visual_avoiding.utils, exp → avoiding-d3il-visual,
#       experiment='plan' → experiment='plan_fm_visual_avoiding'
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

import fm_visual_avoiding.utils as utils

with open('config/projection_eval.yaml') as f:
    config = yaml.safe_load(f)

projection_variants       = config['projection_variants']
avoiding_halfspace_variants = config['avoiding_halfspace_variants']

exp = 'avoiding-d3il-visual'

class Parser(utils.Parser):
    dataset: str = exp
    config:  str = 'config.' + exp

seeds = config['seeds']

sr_goal_all        = {}
sr_constraints_all = {}
timesteps_avg_all  = {}
timesteps_std_all  = {}

plot_path = None

for variant in projection_variants:
    n_success_all                = np.array([])
    n_success_and_constraints_all = np.array([])
    n_steps_all                  = np.array([])
    n_violations_all             = np.array([])
    total_violations_all         = np.array([])
    collision_free_completed_all = np.array([])

    for halfspace_variant in avoiding_halfspace_variants:
        for i, seed in enumerate(seeds):
            args = Parser().parse_args(experiment='plan_fm_visual_avoiding', seed=seed)
            if plot_path is None:
                load_path = os.path.dirname(args.savepath)
                plot_path = os.path.join(load_path, 'plots', 'load_results_output_all_seeds')
                os.makedirs(plot_path, exist_ok=True)
                print(f'[ utils ] plot_path: {plot_path}')

            data = np.load(
                f'{args.savepath}/results/halfspace_{halfspace_variant}/{variant}.npz',
                allow_pickle=True)
            n_success_all                = np.append(n_success_all,                data['n_success'])
            n_success_and_constraints_all = np.append(n_success_and_constraints_all,
                                                       data['n_success_and_constraints'])
            n_steps_all                  = np.append(n_steps_all,
                                                       data['n_steps'][data['n_success'] > 0])
            n_violations_all             = np.append(n_violations_all,             data['n_violations'])
            total_violations_all         = np.append(total_violations_all,         data['total_violations'])
            collision_free_completed_all = np.append(collision_free_completed_all,
                                                       data['collision_free_completed'])

    sr_goal_all[variant]       = n_success_all.mean()
    sr_constraints_all[variant] = collision_free_completed_all.mean()
    timesteps_avg_all[variant]  = n_steps_all.mean()
    timesteps_std_all[variant]  = n_steps_all.std()

    print(f'------- {variant} -------')
    print(f'Success rate (goal):                 {sr_goal_all[variant]:.2f}')
    print(f'Success rate (goal + constraints):   {n_success_and_constraints_all.mean():.2f}')
    print(f'Success rate (constraints):          {sr_constraints_all[variant]:.2f}')
    print(f'Average steps: {timesteps_avg_all[variant]:.2f} +- {timesteps_std_all[variant]:.2f}')
    print(f'Average violations: {n_violations_all.mean():.2f} +- {n_violations_all.std():.2f}')
    print(f'Average total violations: {total_violations_all.mean():.3f}'
          f' +- {total_violations_all.std():.3f}')
