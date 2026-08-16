"""Gen13 U11 — HF_Mix_ML: additive multi-MLbone package (iMF + MeanFlow + α-Flow).

Assembles the Gen3v6 (MeanFlow) and Gen3v7 (α-Flow) training objectives as sibling
addons alongside the FROZEN Gen13 iMF matcher, all sharing the one dual-head
TemporalImfUnet backbone and the one u-only sampler/policy from the `imf/` package.
Nothing in `imf/` (or any pre-existing HardFlow file) is modified; this package is
selected only via the new entry point run/train_ml.py. See README_PROVENANCE.md.
"""

from ..imf.imf_matcher import ImfMatcher  # re-export: the frozen iMF objective
from .af_matcher import AfMatcher
from .matcher_factory import build_matcher
from .mf_matcher import MfMatcher
from .ml_config import MlTrainingConfig

__all__ = [
    "ImfMatcher",
    "MfMatcher",
    "AfMatcher",
    "build_matcher",
    "MlTrainingConfig",
]
