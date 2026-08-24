from .projection import Projector

# ── Gen14 U7 ── HardFlow (`hardflow_new`) constrained sampler, ported from Gen3v7.
# Imported lazily-safe: this module only needs numpy/torch/diffuser, the same deps
# projection.py already pulls in, so a plain import is fine.
from .hardflow_projection import (
    TrajectoryLayout,
    HardFlowNLP,
    HardFlowSampler,
    resolve_activation_threshold,
    hardflow_step_budget,           # HFK1 (2026-08-24)
    hardflow_regime, HF_OK, HF_THIN, HF_DEGENERATE,   # HFK1b (2026-08-24)
    resolve_hf_batch_size,          # B4_PARITY (2026-08-20)
    resolve_engine_hf,
    encode_visual_cond,
    ENGINE_INIT_NOISE,
    ENGINE_TWO_TIME,
)

__all__ = [
    'Projector',
    'TrajectoryLayout', 'HardFlowNLP', 'HardFlowSampler', 'hardflow_step_budget',
    'hardflow_regime', 'HF_OK', 'HF_THIN', 'HF_DEGENERATE',
    'resolve_activation_threshold', 'resolve_hf_batch_size', 'resolve_engine_hf', 'encode_visual_cond',
    'ENGINE_INIT_NOISE', 'ENGINE_TWO_TIME',
]
