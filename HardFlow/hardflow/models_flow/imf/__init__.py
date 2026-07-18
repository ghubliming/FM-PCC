"""Gen13 — additive iMF (improved MeanFlow) package for HardFlow.

Nothing in the pre-existing HardFlow code is modified by this package; it is
selected only via the new entry points run/train_imf.py and run/eval_imf.py.
See README_PROVENANCE.md for the port provenance and PLAN_Gen13 for design.
"""

from .convention import endpoint_from_u, jump_from_u, sample_tau_h
from .imf_config import ImfEvaluationConfig, ImfTrainingConfig
from .imf_flow_policy import ImfFlowPolicy
from .imf_matcher import ImfMatcher
from .imf_sampler import imf_sample
from .temporal_imf_unet import TemporalImfUnet

__all__ = [
    "endpoint_from_u",
    "jump_from_u",
    "sample_tau_h",
    "ImfEvaluationConfig",
    "ImfTrainingConfig",
    "ImfFlowPolicy",
    "ImfMatcher",
    "imf_sample",
    "TemporalImfUnet",
]
