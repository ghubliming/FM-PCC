# Gen13 iMF package — provenance & port notes

**Added:** 2026-07-18 (FM-PCC Gen13). **Additive only** — no pre-existing HardFlow file was modified. Selected exclusively via `run/train_imf.py` / `run/eval_imf.py`.

## Sources ported

| This package | Ported from | What changed in the port |
|---|---|---|
| `imf_matcher.py` (loss) | `aux_repo/imeanflow` **JAX `main`** branch `imf.py` `forward()` (the torch branch is inference-only) | JAX→PyTorch (`torch.func.jvp`); **CFG dropped entirely** (labels/omega/t_min/t_max/null-token removed — HardFlow conditions by state-inpainting); rewritten in HardFlow's time convention (see `convention.py`); adaptive `adp` and predicted-v tangent kept faithful |
| `imf_sampler.py` | `aux_repo/imeanflow` **`origin/torch`** branch `imf.py` `sample_one_step()`/`generate()` | convention flip (`t_steps = linspace(1,0)` noise-side ↘ becomes `tau = i/K` noise→data ↗); conditioning masking matched to HardFlow's `ConditionedODESolver` |
| `temporal_imf_unet.py` | HardFlow `hardflow/models_flow/unet.py` `TemporalUnet` (blocks **imported**, not copied) | + second sinusoidal embedding for interval width `h` (summed with the τ embedding); final conv widened to `2×transition_dim`, chunked into `(u, v)` heads |
| `imf_flow_policy.py` | HardFlow `flow_policy.py` (`FlowPolicy` subclassed; `hardflow_new_forward`/`warmstart`/`x1_estimate` adapted copies) | the SEAM only: Euler `v`-shot → exact `u`-endpoint; Euler ref step → exact `u`-jump; NLP/pull-back/timing untouched; + NFE accounting |
| `convention.py` | derivation (documented inline) reconciling the two conventions | the ONLY place with sign/mapping logic |

Aux repo commit at port time: `imeanflow` main @ the checkout in `/workspaces/aux_repo` (container-only; this package is self-contained — no aux imports at runtime, no JAX anywhere).

## Design decisions (from `logs_in_develop/HF_iMF/Gen13/init/PLAN_Gen13_iMF_backbone_in_HardFlow.md`)

D2 temporal-UNet backbone (not DiT) · D3 no CFG · D4 official objective (JVP, predicted-v tangent, adaptive) · D5 `data_proportion=0.25`, `p_std=1.4` (Gen3v4 §6) · D6 100k steps · D7 K∈{1,2} primary · D8 Level-1 seam swap only.

**Reading training curves:** the adaptive `loss` is flat **by construction** — judge convergence on `raw_mse_u` / `raw_mse_v` / `a0_mse` (Gen3v4 ANALYSIS §0).
