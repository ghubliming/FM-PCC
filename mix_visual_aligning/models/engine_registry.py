"""Gen14 — the engine dispatch table. THE only place an arm branches.

`engine` is a config key: one of 'ddpm' | 'fm' | 'mf' | 'af'. Everything that differs
between arms is declared here, so `train_mix_visual_aligning.py` and
`eval_mix_visual_aligning.py` contain no `if engine == ...` chains.

────────────────────────────────────────────────────────────────────────────────
THE STRUCTURAL RULE (PLAN §3.1)

  The 'ddpm' and 'fm' arms import ONLY verbatim copies of Gen6V4 / Gen7 files.
  Every newly-authored line lives in a module that only 'mf' and 'af' import.

That is what makes Gen14's reproduction of Gen6V4 and Gen7 a property of the file
layout rather than something a test has to establish. Do not "simplify" an mf/af
module into a path the ddpm/fm arms reach.
────────────────────────────────────────────────────────────────────────────────

`wraps_unet` is the one structural asymmetry:

  wraps_unet=False (ddpm, fm)
      VisualUNet is built directly and handed to the engine:
          engine_cls(model=VisualUNet(args), ...)
  wraps_unet=True  (mf, af)
      the two-time (u, v) surface needs an engine wrapper, which builds
      VisualUNetTwoTime internally as its velocity_net:
          engine_cls(model=MeanFlowEngine(if_vision=True, vis_config=args), ...)

Consequence: `model_config.pkl` describes the ENGINE for mf/af and the U-NET for
ddpm/fm. The eval loader reconstructs whatever it finds, so an arm mismatch between
train and eval surfaces as an opaque state_dict error — which is why
`assert_engine_matches()` exists.
"""

_P = 'mix_visual_aligning.models.'
_U = 'mix_visual_aligning.utils.'

ENGINES = {
    # ── Gen6V4 — DDPM. Verbatim copy of diffuser_visual_aligning. ─────────────
    'ddpm': dict(
        label       = 'DDPM (Gen6V4)',
        diffusion   = _P + 'visual_gaussian_diffusion.VisualGaussianDiffusion',
        model       = _P + 'visual_unet.VisualUNet',
        wraps_unet  = False,
        two_time    = False,
        trainer     = _U + 'training.Trainer',
        nfe_key     = 'n_diffusion_steps',
        source      = 'diffuser_visual_aligning/ (Gen6V4)',
    ),
    # ── Gen7 — continuous-time FM ODE. Verbatim copy of fm_visual_aligning. ───
    'fm': dict(
        label       = 'Flow Matching ODE (Gen7)',
        diffusion   = _P + 'visual_fm_diffusion.VisualFlowMatching',
        model       = _P + 'visual_unet.VisualUNet',
        wraps_unet  = False,
        two_time    = False,
        trainer     = _U + 'training.Trainer',
        nfe_key     = 'flow_steps_v3',
        source      = 'fm_visual_aligning/ (Gen7)',
    ),
    # ── Gen3v6 — MeanFlow (JVP target). ──────────────────────────────────────
    'mf': dict(
        label       = 'MeanFlow (Gen3v6)',
        diffusion   = _P + 'visual_mf_diffusion.VisualMeanFlow',
        model       = _P + 'mf_engine.MeanFlowEngine',
        wraps_unet  = True,
        two_time    = True,
        trainer     = _U + 'training_twotime.Trainer',
        nfe_key     = 'flow_steps_v3',
        source      = 'flow_matcher_v3_meanflow/ (Gen3v6)',
    ),
    # ── Gen3v7 — alpha-Flow (bootstrapped target + alpha anneal). ────────────
    'af': dict(
        label       = 'alpha-Flow (Gen3v7)',
        diffusion   = _P + 'visual_af_diffusion.VisualAlphaFlow',
        model       = _P + 'af_engine.AlphaFlowEngine',
        wraps_unet  = True,
        two_time    = True,
        trainer     = _U + 'training_twotime.Trainer',
        nfe_key     = 'flow_steps_v3',
        source      = 'flow_matcher_v3_alphaflow/ (Gen3v7)',
    ),
}

ENGINE_KEYS = tuple(ENGINES.keys())


def resolve(engine):
    """Return the registry entry for `engine`, or raise with the valid set."""
    key = str(engine).lower().strip()
    if key not in ENGINES:
        raise ValueError(
            f"[ engine_registry ] unknown engine '{engine}'. "
            f"Valid: {', '.join(ENGINE_KEYS)}."
        )
    return ENGINES[key]


def import_class(dotted_path):
    """'pkg.mod.Cls' -> the class object. Imported lazily so selecting one arm never
    pays the import cost (or the import risk) of the other three."""
    import importlib
    module_path, _, cls_name = dotted_path.rpartition('.')
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def get_trainer_cls(engine):
    """Trainer class for this arm.

    ddpm/fm -> utils.training.Trainer          (Gen7 verbatim)
    mf/af   -> utils.training_twotime.Trainer  (Gen3v7 verbatim: h-stratified metrics,
                                                real gradient_clip, seeded split, and
                                                set_train_step() for the alpha anneal)
    """
    return import_class(resolve(engine)['trainer'])


def nfe_of(engine, args, default=None):
    """Read this arm's inference-step count off `args` under ITS OWN key.

    ddpm counts denoising steps (`n_diffusion_steps`); fm/mf/af count ODE steps
    (`flow_steps_v3`). Callers that print or path-name "K" must go through here.
    """
    key = resolve(engine)['nfe_key']
    return getattr(args, key, default)


def describe(engine):
    """One-line banner string for logs: makes the active arm unmissable in a batch log."""
    spec = resolve(engine)
    return (f"engine={engine} | {spec['label']} | two_time={spec['two_time']} | "
            f"wraps_unet={spec['wraps_unet']} | nfe_key={spec['nfe_key']} | "
            f"source={spec['source']}")


def assert_engine_matches(requested, checkpoint_engine):
    """Guard the eval path against loading arm A's weights into arm B.

    Without this, a mismatch surfaces as an opaque `state_dict` key error minutes into
    a GPU allocation. `checkpoint_engine=None` means a pre-Gen14 checkpoint that never
    recorded the key — allowed, with a warning, so old runs stay loadable.
    """
    if checkpoint_engine is None:
        print(f"[ engine_registry ] WARNING: checkpoint records no 'engine' key; "
              f"assuming '{requested}'. Verify this is the arm you trained.")
        return
    if str(requested).lower() != str(checkpoint_engine).lower():
        raise ValueError(
            f"[ engine_registry ] ENGINE MISMATCH — requested '{requested}' but the "
            f"checkpoint was trained with '{checkpoint_engine}'. Loading it would fail "
            f"with a confusing state_dict error. Fix the --engine flag or the config "
            f"block so they agree."
        )
