import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import flow_matcher_v3_ode_selectable.utils as utils

# Load configuration
with open('config/visual_aligning_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

projection_variants = config.get('projection_variants', ['diffuser'])
exp = 'aligning-d3il-visual'

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

seeds = config.get('seeds', [6])

sr_goal_all = {}
sr_constraints_all = {}
timesteps_avg_all = {}
timesteps_std_all = {}

plot_path = None

for variant in projection_variants:
    n_success_all = np.array([])
    n_success_and_constraints_all = np.array([])
    n_steps_all = np.array([])
    n_violations_all = np.array([])
    total_violations_all = np.array([])
    collision_free_completed_all = np.array([])
    
    for i, seed in enumerate(seeds):
        # Load from visual FM plan config key
        args = Parser().parse_args(experiment='plan_fm_encdec_vision', seed=seed)
        if plot_path is None:
            load_path = os.path.dirname(args.savepath)
            plot_path = os.path.join(load_path, 'plots', 'load_results_output_all_seeds')
            os.makedirs(plot_path, exist_ok=True)
            print(f'[ utils ] Set plot_path to: {plot_path}')

        flow_steps = getattr(args, 'flow_steps_v3', getattr(args, 'ode_inference_steps_v3', 'n/a'))
        ode_steps = getattr(args, 'ode_inference_steps_v3', flow_steps)
        beta_alpha = getattr(args, 'time_beta_alpha_v3', 'n/a')
        beta_beta = getattr(args, 'time_beta_beta_v3', 'n/a')
        print(f'Eval ODE={ode_steps}, FlowSteps={flow_steps}, Beta=({beta_alpha},{beta_beta})')

        # Get data directly from visual results folder (no halfspace obstacles)
        try:
            data_file = f'{args.savepath}/results/{variant}.npz'
            data = np.load(data_file, allow_pickle=True)
            n_success = data["n_success"]
            n_steps = data["n_steps"]
            avg_time = data["avg_time"]

            n_success_all = np.append(n_success_all, n_success)
            n_steps_all = np.append(n_steps_all, n_steps[n_success > 0])
            collision_free_completed_all = np.append(collision_free_completed_all, n_success) # Success = Safe for Aligning
        except FileNotFoundError:
            print(f"[ Error ] Could not find results at: {args.savepath}/results/{variant}.npz")
            continue

    if len(n_success_all) == 0:
        print(f"Skipping variant {variant} due to no data found.")
        continue

    success_rate_goal = n_success_all.mean()
    success_rate_constraints = collision_free_completed_all.mean()
    steps_avg = n_steps_all.mean() if len(n_steps_all) > 0 else 0
    steps_std = n_steps_all.std() if len(n_steps_all) > 0 else 0

    print(f'------------------ Variant: {variant} ------------------')
    print(f'Success rate (goal): {success_rate_goal:.2f}')
    print(f'Success rate (constraints): {success_rate_constraints:.2f}')
    print(f'Average steps: {steps_avg:.2f} +- {steps_std:.2f}')
    print(f'Average time: {avg_time.mean():.2f} +- {avg_time.std():.2f}')

    sr_goal_all[variant] = success_rate_goal
    sr_constraints_all[variant] = success_rate_constraints
    timesteps_avg_all[variant] = steps_avg
    timesteps_std_all[variant] = steps_std

# Plot results
variants_to_plot = [['diffuser']]
variants_labels = ['Diffuser']

for variants in variants_to_plot:
    if not all(variant in sr_goal_all for variant in variants):
        print(f"Skipping plot for {variants} as some data is missing.")
        continue

    sr_goal = [sr_goal_all[variant] for variant in variants]
    sr_constraints = [sr_constraints_all[variant] for variant in variants]
    timesteps_avg = [timesteps_avg_all[variant] for variant in variants]
    timesteps_std = [timesteps_std_all[variant] for variant in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 10))
    bars1 = ax.bar(x - width/2, sr_goal, width, label='Goal reached', color='green')
    bars2 = ax.bar(x + width/2, sr_constraints, width, label='Constraints satisfied', color='red')

    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants_labels, fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.legend(loc='lower left', fontsize=12) 

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels(bars1)
    add_labels(bars2)

    fig.tight_layout()
    save_name = 'success_rates_fm.png'
    plt.savefig(os.path.join(plot_path, save_name))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.bar(x, timesteps_avg, width, yerr=timesteps_std, label='Timesteps', color=[0.5, 0.5, 1], capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(variants_labels, fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.set_ylim([0, 100])
    ax.legend(loc='lower left', fontsize=12) 

    add_labels(bars)
    fig.tight_layout()
    save_name = 'timesteps_fm.png'
    plt.savefig(os.path.join(plot_path, save_name))
    plt.close(fig)

print("Load results completed.")
