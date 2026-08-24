from .projection import Projector
from .policies import Policy
from .hardflow_projection import (
    TrajectoryLayout,
    HardFlowNLP,
    HardFlowSampler,
    HardFlowPolicy,
    resolve_activation_threshold,
    hardflow_step_budget,           # HFK1 (2026-08-24)
    hardflow_regime, HF_OK, HF_THIN, HF_DEGENERATE,   # HFK1b (2026-08-24)
    resolve_hf_batch_size,          # B4_PARITY (2026-08-20)
)