# Gen13 U11 — Giant Upgrade: HF_iMF → **HF_Mix_ML** (iMF + MeanFlow + α-Flow)

**Status:** PLAN ONLY — no code in this document. Other agents implement from here.
**Path owner:** `logs_in_develop/Gen13/U11_Giant_UPGRADE/`
**Depends on / supersedes framing of:** `Gen13/init/PLAN_Gen13_iMF_backbone_in_HardFlow.md`,
`Gen13/9_CLOSURE_I/CLOSURE_Gen13_imf_vs_fm_in_hardflow.md`,
`HardFlow/hardflow/models_flow/imf/README_PROVENANCE.md`.

---

## 0. The rule (non-negotiable, read first)

> **Additive package assembly. Absolutely do not damage existing code.**

Concretely for U11:

1. **The `imf/` package is FROZEN.** `HardFlow/hardflow/models_flow/imf/` is not edited,
   renamed, or moved. The current iMF objective **must stay byte-identical** — it is the
   faithful Gen3v4 `imf_official` math (V-form + guided `v_g` + **predicted-`v_c`** JVP tangent),
   and Gen13's entire closure (U_9 … 9_CLOSURE_I) is calibrated against it. Touching it
   invalidates every prior Gen13 result.
2. Everything U11 adds lives in **new files** (a new `ml/` package + new run/sbatch/config
   siblings). Nothing pre-existing changes behaviour on its default path.
3. `run/train_imf.py` and `run/eval_imf.py` stay as the untouched iMF-only canonical entries.
   U11 adds *new* `train_ml.py` / `eval_ml.py` that offer all three — and whose `ml_type=imf`
   branch calls the **same** `ImfMatcher` with the **same** arguments, so it reproduces the
   canonical iMF path exactly (gate G0).

---

## 1. What U11 actually is

Today Gen13 = **one** MLbone in HardFlow: improved-MeanFlow (iMF). U11 brings the two newest
Gen3 MLbones **into the Gen13/HardFlow package as concrete addon siblings**, so a single run can
select any of three training objectives:

```
ml_type ∈ { imf , mf , af }
   │        │      │     └── α-Flow      (Gen3v7, "AlphaFlow")   ← NEW addon
   │        │      └──────── MeanFlow    (Gen3v6, faithful MF)   ← NEW addon
   │        └─────────────── improved-MeanFlow (Gen3v4/Gen13)    ← EXISTING, frozen
   └──────────────────────── top-level MLbone selector
```

Each MLbone keeps its **own, separated, adjustable parameter block** (§6). "First pick the
family, then tune that family's knobs." iMF's block is exactly Gen3v4's; MF and AF arrive with
their own knobs copied verbatim from Gen3v6/Gen3v7.

This is **not** a Hydra/config-abstraction refactor. It is a physical move: the Gen3v6 and
Gen3v7 objective code is copied into `HardFlow/…/ml/` and assembled there against HardFlow's
backbone, sampler, and convention.

---

## 2. Why this is clean — the one fact that makes U11 small

All three MLbones are the **same object with three different training targets**:

| | contract |
|---|---|
| **Backbone** | one dual-head net `f(x, τ, h) → (u, v)` — HardFlow's `TemporalImfUnet` (shared) |
| **Sampler / seam** | deploys **`u` only**: `x̂₁ = z + (1−τ)·u`; iterate `z ← z + Δτ·u` (shared) |
| **NLP / projection / policy** | consumes `u`; **backbone-agnostic** (shared) |
| **What differs** | **only the regression target for `u` at train time** (the matcher `forward`) |

The Gen13 closure proved the eval side is objective-agnostic (the projection dominates; all
backbones project to 96–100 %). So U11's real work is **three matchers, one backbone, one
sampler, one selector** — the entire generative/eval stack is reused unchanged.

### The three `u` targets (this is the whole difference)

| MLbone | `u_target` (stop-grad) | z-tangent for the JVP | source |
|---|---|---|---|
| **iMF** (Gen3v4/Gen13) | `V`-form, `u ← v_g + h·D_tot`, guided | **predicted `v_c`** (network head) | `imf/imf_matcher.py` (frozen) |
| **MF** (Gen3v6) | `u ← sg(v_inst + h·du/dr)` | **analytic `v_inst = x₁ − x₀`** | `mf_diffusion.py::_p_losses_meanflow` |
| **AF** (Gen3v7) | `u ← sg(α·v_inst + (1−α)·u_next)`, α: 1→0 anneal | bootstrap (`u_next` self-eval), no JVP at α<1 | `af_diffusion.py::_p_losses_alphaflow` |

The single line that separates iMF from MF is the z-tangent: **predicted `v_c`** (iMF) vs
**analytic `v_inst`** (MF). AF drops the JVP entirely for a self-bootstrapped target that anneals
from pure-FM (α=1) to MeanFlow (α=0). Everything else (adaptive loss, dual v-head, `h`-sampling,
FM-anchor fraction) is shared machinery.

---

## 3. Source provenance — exactly what moves in, and what porting each needs

The Gen3v6/v7 loss bodies currently live on the **FMv3ODE** base (`MeanFlowODE` / `AlphaFlowODE`
extend the Gen3v4 diffusion) and speak the FMv3ODE dialect: `apply_conditioning`, `q_sample`,
`_predict_uv`, DATA-AT-1 convention (τ=0 noise, τ=1 data), and optional CFG (`returns`,
`force_dropout`). HardFlow's iMF already crossed this bridge once; MF and AF cross it the **same
way** iMF did (see `imf/README_PROVENANCE.md`).

| Moves into `ml/` | Ported from | Port operations (identical recipe to iMF) |
|---|---|---|
| `ml/mf_matcher.py` | `flow_matcher_v3_meanflow/models/mf_diffusion.py::_p_losses_meanflow` (+ `_build_info`, `_adaptive`, `_sample_tau_pair`) | (a) convention flip via `imf/convention.py` (τ mapping + tangent signs `(v_inst,+1,−1)`); (b) **drop CFG** (`returns`/`force_dropout`/null-token removed — HardFlow inpaints state); (c) call HardFlow `TemporalImfUnet` dual head instead of `_predict_uv`; (d) HardFlow conditioning masks instead of `apply_conditioning` |
| `ml/af_matcher.py` | `flow_matcher_v3_alphaflow/models/af_diffusion.py::_p_losses_alphaflow` + `compute_u_target` + `_get_ratio` (α scheduler) + `_predict_uv`-based `u_next` self-eval | same (a)–(d), **plus** carry the α-scheduler state (step counter) into the matcher and keep the `af_alpha_end_step == n_train_steps` assert |
| `ml/mf_config.py`, `ml/af_config.py` | the MF/AF knob defaults in `config/avoiding-d3il.py` (`args_to_watch_fmv3_mf_train` / `_af_train`) and the Gen3v6/v7 engine signatures | as HardFlow `@dataclass` siblings of `ImfTrainingConfig`/`ImfEvaluationConfig` |
| `ml/matcher_factory.py` | NEW (thin) | `build_matcher(cfg) → {imf: ImfMatcher, mf: MfMatcher, af: AfMatcher}[cfg.ml_type]`; `ml_type='imf'` returns the frozen `ImfMatcher` with identical args |

**Reused verbatim from `imf/` (imported, not copied):** `TemporalImfUnet` (the dual-head backbone),
`ImfSampler`, `ImfFlowPolicy` / `InstrumentedFlowPolicy`, `convention.py`. These are already
backbone-agnostic (they only ever touch `u`), so MF and AF checkpoints run through them unchanged.

> **Provenance rule:** each new file gets a header block in the style of
> `imf/README_PROVENANCE.md` — aux/Gen3 source, commit, and the exact port deltas.

---

## 4. Proposed additive layout

```
HardFlow/hardflow/models_flow/
├── imf/                      ← FROZEN. untouched. (guarantees iMF byte-identity)
│   ├── imf_matcher.py        ← the iMF u-target (predicted-v_c). NOT edited.
│   ├── temporal_imf_unet.py  ← dual-head (u,v) backbone  ┐
│   ├── imf_sampler.py        ← u-only sampler            │ imported by ml/ (shared)
│   ├── imf_flow_policy.py    ← HardFlow seam + NLP        │
│   └── convention.py         ← τ mapping / tangent signs  ┘
│
└── ml/                       ← NEW "MLbone" assembly package = HF_Mix_ML
    ├── __init__.py           ← re-exports ImfMatcher (from ..imf) + MfMatcher + AfMatcher + factory + configs
    ├── mf_matcher.py         ← Gen3v6 MeanFlow objective, ported (analytic-v JVP)
    ├── af_matcher.py         ← Gen3v7 α-Flow objective, ported (bootstrap α-anneal)
    ├── mf_config.py          ← MFTrainingConfig / MFEvaluationConfig (own knobs)
    ├── af_config.py          ← AFTrainingConfig / AFEvaluationConfig (own knobs)
    ├── ml_config.py          ← base with `ml_type` selector; imports the three blocks
    ├── matcher_factory.py    ← ml_type → matcher (imf branch = frozen ImfMatcher)
    └── README_PROVENANCE.md  ← port notes for mf/af (mirrors imf/README_PROVENANCE.md)

HardFlow/run/
├── train_imf.py   ← untouched canonical iMF entry
├── eval_imf.py    ← untouched canonical iMF entry
├── train_ml.py    ← NEW: copy-modify of train_imf.py; builds matcher via factory(ml_type)
└── eval_ml.py     ← NEW: copy-modify of eval_imf.py; same u-only policy, checkpoint-driven

Slurm_Codes/sbatch/hardflow/
├── train_ml_hardflow.sh     ← NEW: forwards ML_TYPE + that family's knobs
├── eval_ml_hardflow.sh      ← NEW
└── ml_pipeline_hardflow.sh  ← NEW: chained train→eval (copy of imf_pipeline_hardflow.sh)
```

Only **new** files. `imf/`, `train_imf.py`, `eval_imf.py`, and all existing sbatch stay put.

---

## 5. The selector surface

- **Config key:** `ml_type: str = "imf"` on the base training/eval config (`ml/ml_config.py`).
  Default `"imf"` ⇒ the frozen path.
- **Dispatch:** `train_ml.py` calls `matcher_factory.build_matcher(cfg)`. For `ml_type="imf"` it
  constructs `ImfMatcher(model=unet, action_dim=…, p_mean=cfg.imf_p_mean, …)` with the **exact**
  argument list from `train_imf.py:104` (gate G0 diffs the two).
- **Backbone:** all three build the **same** `TemporalImfUnet(dim=32, dim_mults=(1,4,8))`
  (Gen13 D2: UNet, not DiT — holds architecture constant across MLbones so the A/B/C isolates the
  *objective*). DiT is explicitly **out of scope** for U11 (§12 non-goals).
- **Eval:** `eval_ml.py` is objective-agnostic — it loads whatever checkpoint `flow_exp_name`
  points at and runs the shared `ImfFlowPolicy`. In principle the existing `eval_imf.py` already
  works for MF/AF checkpoints; `eval_ml.py` exists only so the exp-name/knob plumbing and the
  W&B/CSV metric names stay family-correct.
- **Checkpoint isolation:** `ml_type` (and its family knobs) go into the exp-name/folder tokens
  (like Gen3's `args_to_watch`), so iMF/MF/AF checkpoints can never collide.

---

## 6. The three separated parameter blocks (each adjustable)

Each family owns its knobs. Setting one family's knobs never touches another's.

### 6.1 iMF block — **frozen, = Gen3v4 `imf_official`** (already in `imf_config.py`)
```
imf_p_mean        = -0.4
imf_p_std         =  1.4
imf_data_proportion = 0.25     # FM-anchor fraction (h=0)
imf_adp_p         =  1.0       # official adaptive p
imf_adp_eps       =  0.01
# objective identity: V-form, guided v_g, PREDICTED v_c JVP tangent. DO NOT expose a tangent switch.
```

### 6.2 MF block (Gen3v6 faithful MeanFlow) — `ml/mf_config.py`
```
mf_data_proportion = 0.25      # first-class ablation axis in Gen3v6 (folder token `dp`)
mf_adp_p           = 1.0
mf_adp_eps         = 0.01
mf_p_mean          = -0.4      # τ-pair sampling (same log-normal family as iMF)
mf_p_std           =  1.4
# objective identity: u ← sg(v_inst + h·JVP), z-tangent = ANALYTIC v_inst = x1 − x0.
#   🔴 the z-tangent is analytic v_inst — feeding a predicted v_c here turns MF into iMF
#      and destroys the A/B (mf_diffusion.py's own red-banner warning). Not a knob.
```

### 6.3 AF block (Gen3v7 α-Flow) — `ml/af_config.py`
```
af_alpha_scheduler = 'sigmoid'   # 'sigmoid'|'linear'|'exponential'|'log'|'constant'|'step'
af_alpha_init      = 1.0         # α at step 0     (1.0 ⇒ start as pure FM)
af_alpha_end       = 0.0         # α at the end    (0.0 ⇒ end as MeanFlow)
af_alpha_init_step = 0
af_alpha_end_step  = <n_train_steps>   # 🔴 MUST equal n_train_steps (anneal spans real budget)
af_alpha_gamma     = 25.0        # sigmoid sharpness
af_ratio_fm        = 0.5         # batch fraction forced to r==t (h=0, FM anchors)
af_clamp_utgt      = 4.0         # clamp on the bootstrapped u_target
af_adp_eps         = 1e-3
```

### 6.4 Shared training knobs (inherited from `FlowMatchingTrainingConfig`, per-run)
`n_train_steps`, `save_freq`, `learning_rate`, `grad_clip`, `ema_decay`, `batch_size`, `horizon`,
plus the U9.2 stability lessons (`grad_clip`, LR — see closure §6). These apply to all three; the
closure's finding that **total optimisation, not LR, drives the raw-field/guided inversion**
applies to MF and AF too and should be watched (§13).

---

## 7. What is SHARED vs NEW (audit table)

| component | file | U11 action |
|---|---|---|
| dual-head backbone `(u,v)` | `imf/temporal_imf_unet.py` | **shared, untouched** |
| u-only sampler | `imf/imf_sampler.py` | **shared, untouched** |
| HardFlow seam + NLP policy | `imf/imf_flow_policy.py` | **shared, untouched** |
| convention (τ / tangent signs) | `imf/convention.py` | **shared, untouched** |
| iMF objective | `imf/imf_matcher.py` | **frozen, untouched** |
| MeanFlow objective | `ml/mf_matcher.py` | **NEW** (ported Gen3v6) |
| α-Flow objective | `ml/af_matcher.py` | **NEW** (ported Gen3v7) |
| family configs | `ml/{mf,af,ml}_config.py` | **NEW** |
| matcher selector | `ml/matcher_factory.py` | **NEW** |
| train / eval entries | `run/train_ml.py`, `run/eval_ml.py` | **NEW** (copy-modify) |
| sbatch | `…/{train,eval,ml_pipeline}_ml_hardflow.sh` | **NEW** (copy of imf sbatch) |
| iMF entries | `run/train_imf.py`, `run/eval_imf.py` | **untouched** |

---

## 8. Traps (carried over from Gen3v6/v7 + the port)

1. **iMF byte-identity (G0).** `ml_type=imf` through `train_ml.py` MUST equal `train_imf.py`.
   Diff the `ImfMatcher(...)` construction line-for-line; identical loss curve on a fixed seed.
2. **MF z-tangent is analytic `v_inst`, not predicted `v_c`.** This is the *entire* MF↔iMF
   distinction. A future agent "unifying" the two matchers will be tempted to share the tangent —
   forbidden (mirror mf_diffusion.py's red banner into `mf_matcher.py`).
3. **AF α-anneal must span the ACTUAL budget.** `af_alpha_end_step == n_train_steps` or the
   schedule is wrong (Gen3v7 keeps a hard assert; port it). If you train AF for 550k, the anneal
   must be 550k, not 100k.
4. **Convention flip.** Gen3v6/v7 are DATA-AT-1 (τ=0 noise, τ=1 data); HardFlow's iMF flipped to
   τ noise→data ↗. MF/AF ports must route **every** τ, `q_sample`, and JVP tangent sign through
   `convention.py` — the same reconciliation iMF used. This is the #1 place a silent sign bug hides
   (cf. Gen13 fix_1 "identity sign bug").
5. **Drop CFG.** HardFlow conditions by state-inpainting, not classifier-free guidance. Strip
   `returns`, `force_dropout`, null-tokens, omega/t_min/t_max from the ported bodies (iMF did).
6. **AF `u_next` self-eval cost.** α-Flow's bootstrap does an extra no-grad forward for `u_next`;
   budget for it (train wall-time > iMF/MF). Keep it `torch.no_grad` + detached (Gen3v7 gate G5).
7. **Don't read `loss` as convergence** for any family — the adaptive loss is pinned at its
   ceiling by construction. Judge on `raw_mse_u` / `raw_mse_v` / `a0_mse` (closure §6, iMF README).
8. **Checkpoint collision.** Encode `ml_type` + family knobs in the exp-name (Gen3 `args_to_watch`
   pattern) so `H16_mf_dp0.25_100k` ≠ `H16_af_ai1.0_ae0.0_100k` ≠ `H16_imf_100k`.

---

## 9. Math appendix — the three objectives side by side

All share: draw a τ-pair → `(r = min, t = max)`, `h = t − r ≥ 0`, anchor `z_r = q_sample(x₁, r)`,
analytic FM velocity `v_inst = x₁ − x₀`; a fraction of the batch is forced to `h=0` (FM anchors,
where every target collapses to `v_inst`). Loss is the adaptive-weighted per-sample sum over the
`u` head plus a full `v` head:  `L = adp(‖u_pred − u_target‖²) + adp(‖v_pred − v_inst‖²)`.

**iMF (Gen3v4/Gen13, frozen).** Compound `V = u − h·sg(D_tot)`; JVP tangent uses the **predicted**
velocity `v_c` and guided `v_g`; regress `V → v_target`. (The faithful improved-MeanFlow.)

**MF (Gen3v6).** MeanFlow identity, START-anchored:
`u_target = sg( v_inst + h · du/dr )`, JVP tangents `(∂z=v_inst, ∂r=+1, ∂h=−1)`.
z-tangent is the **analytic** `v_inst`. At `h=0` ⇒ `u_target = v_inst`.

**AF (Gen3v7).** No JVP; self-bootstrapped, α-annealed:
`u_target = sg( α·v_inst + (1−α)·u_next )`, `u_next = f(z_r, r, h−dt)` (no-grad), with
`dt = α·h`; α: 1→0 over training. α=1 ⇒ pure FM (`u_target=v_inst`); α=0 ⇒ MeanFlow-like.
`u_target` clamped to `±af_clamp_utgt`.

At the shared sampler, **all three deploy `u`**: `x̂₁ = z + (1−τ)·u` — identical seam.

---

## 10. Gated implementation order (for the coding agent)

- **G0 — iMF regression.** Add `ml/` skeleton + `matcher_factory` + `train_ml.py`; run
  `ml_type=imf` and prove the loss curve / first-N-step metrics match `train_imf.py` on a fixed
  seed. **No MF/AF until G0 passes.**
- **G1 — MF matcher.** Port `mf_matcher.py`; unit-check `h=0 ⇒ u_target == v_inst`, and that the
  JVP z-tangent is `v_inst` (not a network head). Convention signs cross-checked vs `convention.py`.
- **G2 — MF train smoke.** 2–5k steps on cluster; `raw_mse_u` drops; no NaN; grad_norm sane.
- **G3 — AF matcher.** Port `af_matcher.py` + α scheduler; check α(0)=init, α(end_step)=end, the
  `end_step==n_train_steps` assert, and `u_next` is no-grad/detached.
- **G4 — AF train smoke.** 2–5k steps; watch the extra `u_next` cost; α logged to CSV/W&B.
- **G5 — eval parity.** `eval_ml.py` runs an iMF checkpoint and matches `eval_imf.py` numbers
  (same policy, u-only). Then eval an MF and an AF checkpoint end-to-end.
- **G6 — pipeline.** `ml_pipeline_hardflow.sh` chains train→eval via `--dependency=afterok` for
  each family (copy of `imf_pipeline_hardflow.sh` with `ML_TYPE` forwarding).

Everything is **AI-coding-here / run-on-cluster** — no local execution (no Python pipeline in the
Docker container). Note "run on cluster" at each gate.

---

## 11. Experiment matrix — the decisive Mix_ML A/B/C

U11's payoff is a **fully architecture-controlled objective comparison** the Gen13 closure asked
for: same backbone, same sampler, same NLP, same data — only the training target varies.

| arm | ml_type | key knob | trains |
|---|---|---|---|
| iMF (control) | `imf` | `imf_official` | reuse existing Gen13 checkpoints |
| MF | `mf` | `mf_data_proportion ∈ {0.25, 0.5}` | new |
| AF | `af` | `af_alpha_gamma`, `af_ratio_fm` (α:1→0) | new |

Eval each at **matched K ∈ {1, 2, 5, 10}**, n=200 paired, reporting **both**:
- **unguided** raw-field success + `rough_raw` (the honest field quality), and
- **guided** success + time (the projected planner).

**The two questions U11 answers:**
1. Does the **analytic-v (MF)** or **α-annealed (AF)** target produce a better *raw field* than
   iMF's predicted-`v_c` target? (Closure §Q1: iMF did **not** beat FM raw; is either sibling the
   first to?)
2. Does the raw-field↔guided **rank inversion** (ρ=−1.00, closure ⭐) reproduce across *objectives*,
   or was it specific to iMF's optimisation? If MF/AF invert too, the projection-dominates
   conclusion generalises; if not, one objective is genuinely better-conditioned.

Watch the closure's **total-optimisation** effect: train each family at ≥2 budgets (e.g. 100k and
a long run) since "best raw field = least-converged" may repeat.

---

## 12. Non-goals (explicitly out of scope for U11)

- **No DiT backbone.** Gen13 is UNet (D2). MF/AF use `TemporalImfUnet` to hold architecture
  constant. (Gen3v6's `mf_dit_official_trajectory` / Gen3v7's `af_sit_trajectory` are not ported.)
- **No CFG / classifier-free guidance** (HardFlow inpaints state).
- **No edits to `imf/`, `train_imf.py`, `eval_imf.py`, or any existing sbatch.**
- **No new HardFlow guidance mode** — MF/AF reuse the existing `hardflow_new_imf` / `original_imf`
  seam unchanged (they only change the checkpoint the sampler loads).
- **No refactor unifying the three matchers.** Copy-modify siblings, per repo convention.

---

## 13. Deliverables

1. New package `HardFlow/hardflow/models_flow/ml/` (7 files, §4) + `README_PROVENANCE.md`.
2. `run/train_ml.py`, `run/eval_ml.py`.
3. `Slurm_Codes/sbatch/hardflow/{train,eval,ml_pipeline}_ml_hardflow.sh`.
4. Config entries: iMF (existing) + MF/AF blocks wired through `ml_config.ml_type`.
5. A U11 changelog under `logs_in_develop/Gen13/U11_Giant_UPGRADE/` once G0–G6 pass, plus the
   A/B/C eval writeup (§11) into `HF_iMF/Research/` (link back here and to `9_CLOSURE_I`).

---

## 14. File reference index

| purpose | path |
|---|---|
| frozen iMF matcher (do not edit) | `HardFlow/hardflow/models_flow/imf/imf_matcher.py` |
| shared dual-head backbone | `HardFlow/hardflow/models_flow/imf/temporal_imf_unet.py` |
| shared sampler / policy / convention | `imf/imf_sampler.py`, `imf/imf_flow_policy.py`, `imf/convention.py` |
| iMF configs | `imf/imf_config.py` |
| iMF port notes (template) | `imf/README_PROVENANCE.md` |
| iMF train / eval entries (untouched) | `HardFlow/run/train_imf.py`, `HardFlow/run/eval_imf.py` |
| **MF source** to port | `flow_matcher_v3_meanflow/models/mf_diffusion.py::_p_losses_meanflow` |
| **AF source** to port | `flow_matcher_v3_alphaflow/models/af_diffusion.py::_p_losses_alphaflow` + `compute_u_target` + `_get_ratio` |
| Gen3v6/v7 plans | `logs_in_develop/Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md`, `logs_in_develop/Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md` |
| Gen3 config knob defaults | `config/avoiding-d3il.py` (`args_to_watch_fmv3_{mf,af}_train`) |
| Gen13 closure (why eval is objective-agnostic) | `logs_in_develop/Gen13/9_CLOSURE_I/CLOSURE_Gen13_imf_vs_fm_in_hardflow.md` |
| iMF pipeline sbatch (copy template) | `Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh` |

---

*Plan only. iMF stays frozen and faithful to Gen3v4. MF and AF arrive as additive siblings, each
with its own parameter block, selected by `ml_type`. No code written in this document.*
