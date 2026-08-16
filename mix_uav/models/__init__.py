"""Gen15 (UAV Mix-ML) model exports — three ML engines over one UAV frame.

⚠️ NAME COLLISION, handled deliberately: `unet1d_temporal_cond.py` (the `fm` arm's backbone,
verbatim from Gen11) and `unet1d_twotime_cond.py` (the `mf`/`af` backbone, verbatim from
Gen3v6 = the same file plus an additive `h` embedding, optional v-head and optional
interval-CFG) BOTH define a class named `Flow_matcher_U_Net_v2`.

Only the `fm` one is exported under the bare name, because `config/uav_mix.py` resolves
`'models.Flow_matcher_U_Net_v2'` for that arm. The two-time twin is exported ALIASED and is
never selected by name from config — the two-time engines build it internally through
`MFTrajectoryModel` / `AFTrajectoryModel`. Do not "tidy" this by exporting both unaliased.
"""

# ── arm: fm (Gen11, verbatim) ─────────────────────────────────────────────────────────────
from .unet1d_temporal_cond import Flow_matcher_U_Net_v2, TemporalValue, MLPnet
from .diffusion import FlowMatchingODE

# ── shared by mf + af: the two-time backbone (Gen3v6, verbatim; aliased — see docstring) ───
from .unet1d_twotime_cond import Flow_matcher_U_Net_v2 as Flow_matcher_U_Net_v2_TwoTime

# ── arm: mf (Gen3v6) ──────────────────────────────────────────────────────────────────────
from .mf_trajectory_model import MFTrajectoryModel
from .mf_dit_trajectory import MFDiTTrajectory
from .mf_dit_official_trajectory import MFDiTOfficialTrajectory
from .mf_engine import MeanFlowEngine
from .mf_diffusion import MeanFlowODE

# ── arm: af (Gen3v7) ──────────────────────────────────────────────────────────────────────
from .af_trajectory_model import AFTrajectoryModel
from .af_dit_trajectory import AFDiTTrajectory
from .af_sit_trajectory import AFSiTTrajectory
from .af_engine import AlphaFlowEngine
from .af_diffusion import AlphaFlowODE

# ── arm: diffusion (U3) — the DPCC baseline (DDPM) ────────────────────────────────────────
# ⚠️ Another `Flow_matcher_U_Net_v2`-shaped name clash risk: `unet1d_ddpm_cond` defines
# `UNet1DTemporalCondModel`, a DIFFERENT class from the fm arm's `Flow_matcher_U_Net_v2`
# (same family, plus an optional cond_mlp branch). Distinct names, so no alias needed.
from .unet1d_ddpm_cond import UNet1DTemporalCondModel
from .ddpm_diffusion import GaussianDiffusion

# ── the dispatch table ────────────────────────────────────────────────────────────────────
from . import engine_registry
