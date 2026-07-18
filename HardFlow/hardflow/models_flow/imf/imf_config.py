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
