"""Gen13 — iMF config dataclasses (additive siblings of the FM configs).

Inherit from hardflow.config.flow_matching so every base flag keeps working;
only iMF-specific knobs and Gen13 defaults are added/overridden here.
"""

from dataclasses import dataclass

from hardflow.config.flow_matching import (
    FlowMatchingEvaluationConfig,
    FlowMatchingTrainingConfig,
)


@dataclass
class ImfTrainingConfig(FlowMatchingTrainingConfig):
    # Gen13 identity
    exp_name: str = "H16_imf_100k"
    flow_matching_type: str = "imf"
    horizon: int = 16

    # budget (plan D6; Gen3v4: most learning happens early — 1e6 not needed)
    n_train_steps: int = 100000
    save_freq: int = 25000

    # iMF objective knobs (plan D5; official-convention p_mean/p_std)
    imf_p_mean: float = -0.4
    imf_p_std: float = 1.4
    imf_data_proportion: float = 0.25
    imf_adp_p: float = 1.0
    imf_adp_eps: float = 0.01

    # console/CSV metric cadence (steps)
    log_freq: int = 200

    # U9: W&B logging (pattern copied from FMPCC Gen3v4 iMF —
    # FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py). Same wandb
    # version (0.17.5, inherited from the FMPCC clone) and the same
    # $HOME/FMPCC/.wandb_api_key convention used by sbatch/iMF/train_imf.sh.
    use_wandb: bool = False
    wandb_project: str = "FMPCC-HF-iMF"
    wandb_entity: str = ""      # "" -> wandb default
    wandb_group: str = ""       # "" -> no group


@dataclass
class ImfEvaluationConfig(FlowMatchingEvaluationConfig):
    # Gen13 identity
    flow_exp_name: str = "H16_imf_100k"
    flow_cp: str = "4"          # final checkpoint of the 100k/25k schedule
    flow_matching_type: str = "imf"
    horizon: int = 16

    # K (sampler NFE) == ode_t_steps. K=2 is the paper regime / Gen13 headline.
    ode_t_steps: int = 2

    guidance_method: str = "original_imf"  # or hardflow_new_imf

    # fix_7: which backbone this entry drives.
    #   "imf" (default) -> TemporalImfUnet + ImfFlowPolicy (Gen13)
    #   "fm"            -> HardFlow's original TemporalUnet + FlowPolicy, so the
    #                      smoothness/fan diagnostics apply to BOTH methods through
    #                      one identical code path. run/eval.py stays untouched.
    backbone: str = "imf"

    # u_5(B): MPC foresight-fan diagnostic. DEFAULT OFF — when False, eval behaves
    # byte-identically to before (no capture, no plotting, no extra cost), so the
    # decisive paired n=200 safety run is unaffected. Enable per-run with
    # `--imf_plot_fan` / IMF_PLOT_FAN=1 for a small diagnostic run.
    imf_plot_fan: bool = False
    # u_5 fix: per-plan "Norm of Control Inputs" print — noise in batch logs, OFF.
    imf_verbose_control: bool = False
    # Plot the planned horizon only every `imf_fan_every` replans (FMPCC convention:
    # keeps the figure readable). 1 = every replan.
    imf_fan_every: int = 1
