"""Gen14 — Visual-Mix-ML model package: four engines, one frame.

EXPLICIT re-exports (no star imports) so `import mix_visual_aligning` does not
silently pull every backbone into memory. The registry imports lazily — see
`engine_registry.import_class`.

Provenance of each arm is recorded in `engine_registry.ENGINES[...]['source']`.
"""

# ── the dispatch table (the frame) ────────────────────────────────────────────
from .engine_registry import (
    ENGINES, ENGINE_KEYS, ENGINE_INPUT_KEYS, canonical_engine, resolve, describe,
)

# ── backbones ─────────────────────────────────────────────────────────────────
from .unet1d_temporal_cond import UNet1DTemporalCondModel   # Gen7 — diffusion/fm arms
from .visual_unet import VisualUNet                          # Gen7 — diffusion/fm arms

# ── arm: diffusion (Gen6V4, verbatim) ─────────────────────────────────────────
from .diffusion import GaussianDiffusion
from .visual_gaussian_diffusion import VisualGaussianDiffusion

# ── arm: fm (Gen7, verbatim) ──────────────────────────────────────────────────
from .fm_diffusion import FlowMatchingODE
from .visual_fm_diffusion import VisualFlowMatching

__all__ = [
    'ENGINES', 'ENGINE_KEYS', 'ENGINE_INPUT_KEYS', 'canonical_engine',
    'resolve', 'describe',
    'UNet1DTemporalCondModel', 'VisualUNet',
    'GaussianDiffusion', 'VisualGaussianDiffusion',
    'FlowMatchingODE', 'VisualFlowMatching',
]

# NOTE: the two-time arms (mf/af) are deliberately NOT imported here. Their modules
# pull in the DiT/SiT backbones and torchdiffeq, and only two of the four arms ever
# need them. Reach them through the registry:
#     from mix_visual_aligning.models.engine_registry import resolve, import_class
#     cls = import_class(resolve('mf')['diffusion'])
