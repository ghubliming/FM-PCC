"""Gen13 U11 — build_matcher: ml_type -> the training matcher for that MLbone.

The single dispatch point for HF_Mix_ML. The `imf` branch constructs `ImfMatcher`
with the EXACT argument list `run/train_imf.py` uses, so `ml_type="imf"` reproduces
the frozen Gen13 iMF training path (gate G0). MF/AF build the additive U11 matchers.

All three consume the SAME TemporalImfUnet dual-head backbone and feed the SAME
u-only sampler at eval — they differ only in the training-time u-target (PLAN §2).
"""

from ..imf.imf_matcher import ImfMatcher
from .af_matcher import AfMatcher
from .mf_matcher import MfMatcher


def build_matcher(cfg, model):
    """Return the matcher for cfg.ml_type, wired to `model` (a TemporalImfUnet)."""
    ml_type = getattr(cfg, "ml_type", "imf")

    if ml_type == "imf":
        # 🔴 G0: keep byte-identical to run/train_imf.py's ImfMatcher(...) call.
        return ImfMatcher(
            model=model,
            action_dim=cfg.action_dim,
            p_mean=cfg.imf_p_mean,
            p_std=cfg.imf_p_std,
            data_proportion=cfg.imf_data_proportion,
            adp_p=cfg.imf_adp_p,
            adp_eps=cfg.imf_adp_eps,
        )

    if ml_type == "mf":
        return MfMatcher(
            model=model,
            action_dim=cfg.action_dim,
            p_mean=cfg.mf_p_mean,
            p_std=cfg.mf_p_std,
            data_proportion=cfg.mf_data_proportion,
            adp_p=cfg.mf_adp_p,
            adp_eps=cfg.mf_adp_eps,
        )

    if ml_type == "af":
        return AfMatcher(
            model=model,
            action_dim=cfg.action_dim,
            p_mean=cfg.af_p_mean,
            p_std=cfg.af_p_std,
            ratio_fm=cfg.af_ratio_fm,
            adp_eps=cfg.af_adp_eps,
            clamp_utgt=cfg.af_clamp_utgt,
            alpha_scheduler=cfg.af_alpha_scheduler,
            alpha_init=cfg.af_alpha_init,
            alpha_end=cfg.af_alpha_end,
            alpha_init_step=cfg.af_alpha_init_step,
            alpha_end_step=cfg.af_alpha_end_step,
            alpha_gamma=cfg.af_alpha_gamma,
            alpha_clamp=cfg.af_alpha_clamp,
            n_train_steps=cfg.n_train_steps,
        )

    raise ValueError(f"unknown ml_type={ml_type!r} (expected one of: imf, mf, af)")
