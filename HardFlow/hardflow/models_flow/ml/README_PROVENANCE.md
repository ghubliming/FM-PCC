# Gen13 U11 — `ml/` (HF_Mix_ML) package: provenance & port notes

**Added:** 2026-07-28 (FM-PCC Gen13 U11 "Giant Upgrade"). **Additive only** — no
pre-existing HardFlow file was modified, and the `imf/` package is imported, never
edited. Selected exclusively via `run/train_ml.py`.

This package assembles the two newest Gen3 MLbones (Gen3v6 MeanFlow, Gen3v7 α-Flow)
as sibling training objectives alongside the FROZEN Gen13 iMF matcher, so one
`--ml_type` flag picks the objective. All three share ONE dual-head backbone and ONE
u-only sampler (both reused from `imf/`).

## Sources ported

| This file | Ported from | What changed in the port |
|---|---|---|
| `mf_matcher.py` | `flow_matcher_v3_meanflow/models/mf_diffusion.py::_p_losses_meanflow` (Gen3v6) | dialect swap to HardFlow: `_predict_uv`→`TemporalImfUnet(z,tau,h)`; `apply_conditioning(...,noise=)`→HardFlow `apply_conditioning(z,cond,action_dim)`; `q_sample`→inline `z = tau·x1+(1−tau)·x0`; **no CFG** (`returns`/`force_dropout` dropped). Objective kept faithful: **analytic-v JVP tangent** `(v_target,+1,−1)`, u-target form, adaptive loss, aux v-head. Structured to mirror `imf/imf_matcher.py` so iMF↔MF differ in exactly ONE line (the tangent). |
| `af_matcher.py` | `flow_matcher_v3_alphaflow/models/af_diffusion.py` (`_p_losses_alphaflow` + `compute_u_target` + `_get_ratio`, Gen3v7) | same dialect swap; **plus** the α-scheduler + step counter live on the matcher (`set_step`/`current_alpha`), and the `alpha_end_step == n_train_steps` assert is kept. Bootstrap query `z_shift` is **re-pinned with `apply_conditioning`** instead of Gen3v7's zero-v approach (equivalent — both hold the query's conditioned state fixed). `u_next` under `torch.no_grad` (gate G5). |
| `ml_config.py` | `imf/imf_config.py::ImfTrainingConfig` (subclassed) + Gen3v6/v7 knob defaults (`config/avoiding-d3il.py` `args_to_watch_fmv3_{mf,af}_train`) | one config, `ml_type` selector + separated `mf_*` / `af_*` blocks. Subclassing iMF guarantees `ml_type="imf"` inherits every iMF default byte-identically (gate G0). |
| `matcher_factory.py` | NEW | `build_matcher(cfg, model)`: `imf` branch = frozen `ImfMatcher` with the exact `train_imf.py:104` args; `mf`/`af` build the additive matchers. |

**Reused verbatim from `imf/` (imported, not copied):** `TemporalImfUnet` (dual-head
backbone), `ImfFlowPolicy` / `imf_sampler` (u-only sampler + HardFlow seam/NLP),
`convention.py` (τ mapping, `sample_tau_h`, `pad_t_like_x`, `jvp_tangents`), and
`ImfMatcher` itself. None are modified.

## The one fact that keeps the port small

Both HardFlow and Gen3v6/v7 use the **DATA-AT-1** convention (τ=0 noise, τ=1 data), so
there is **no sign flip** — unlike iMF's JAX→torch port (see `imf/convention.py`). The
port is a dialect swap (`_predict_uv`/`apply_conditioning`/`q_sample` → HardFlow idiom),
not a re-derivation. The three objectives differ ONLY in the training-time u-target:

```
iMF  u-target tangent = PREDICTED v_c          (imf_matcher.py, frozen)
MF   u-target tangent = ANALYTIC  v = x1 − x0  (mf_matcher.py)
AF   u-target = α·v + (1−α)·u_next , α:1→0      (af_matcher.py; α=0 ⇒ MF, α=1 ⇒ FM)
```

## Deviations from PLAN_Gen13_U11 (documented on purpose)

1. **No `eval_ml.py` / no `mf_config.py`+`af_config.py` split.** Eval is
   objective-agnostic — MF/AF checkpoints load into the SAME `TemporalImfUnet` and run
   through the existing `run/eval_imf.py` + `run_scripts/eval_*_imf.sh` unchanged (point
   `flow_exp_name`/`IMF_EXP_NAME` at the ML run). And the three knob-blocks are cleaner
   as namespaced fields (`imf_*`/`mf_*`/`af_*`) in one `MlTrainingConfig` for a single
   CLI. This is *less* code and a *stronger* expression of the Gen13 closure finding
   (the sampler/projection is backbone-agnostic). PLAN §4's file list is superseded here.
2. **Conditioning mirrors iMF, not Gen3v6/v7.** iMF conditions only `z` and uses the
   raw (unconditioned) velocity as target/tangent; MF/AF match that exactly so the
   A/B/C varies ONLY the objective. AF's second query `z_shift` is re-pinned because it
   is a network input (correctness, not style).

## Reading training curves

The adaptive `loss` is flat **by construction** for all three — judge convergence on
`raw_mse_u` / `raw_mse_v` / `a0_mse`. AF additionally logs `alpha` / `discrete_frac` /
`clamp_frac`: a run whose `alpha` never moved is otherwise indistinguishable from a
working one (Gen3v7 gate G4).

## Open gates (per PLAN §10 — verify on cluster, no local execution here)

G0 iMF byte-identity · G1/G2 MF/AF smoke (`raw_mse_u` drops, no NaN) · G3 α endpoints
(`alpha(0)=init`, `alpha(end)=end`) · G5 eval parity (an iMF checkpoint through
`eval_imf.py` matches the frozen numbers). MF/AF-specific unit gates
(analytic-v tangent; α=1⇒u_tgt=v bitwise) are future work.
