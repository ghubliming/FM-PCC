# PLAN — Gen15: **UAV Mix-ML** — one UAV frame, three ML engines activated by config

**Date:** 2026-08-10 · **Type:** initialization / implementation plan · **NO CODE WRITTEN YET**
**Status:** draft for review — nothing may be implemented until the §1 decisions are confirmed.
**Goal:** bring **Gen3v6 (MeanFlow)** and **Gen3v7 (α-Flow)** into the **Gen11 (UAV Vis-Traj)**
closed-loop MuJoCo pipeline alongside the existing **FM-ODE** engine, so the few-step objectives
are finally measured where their claimed advantage is supposed to *matter*: a real-time
receding-horizon controller under continuous geometric constraints.

**Master-index row (already present, not written by this plan):**

> | **Gen15 (UAV Mix-ML)** | Planned | Planned | Planned | **Planned idea: Mix-ML for UAV**. Extending Gen11 (UAV Vis-Traj). Gen11 currently utilizes standard Flow Matching with DPCC. The idea is to initialize a Gen15 to integrate Gen3v6 (MeanFlow), Gen3v7 (α-Flow), and possibly iMF into the UAV pipeline to evaluate these advanced objectives on continuous real-time constraints. | idea |

> ## 🔑 The governing principle (inherited verbatim from Gen14)
>
> **Write the least code. Reassemble, don't rewrite.**
> Every file is a **copy** of a currently-working file, edited as little as possible.
> **Redundancy is explicitly allowed and preferred** over clever de-duplication.
> **Fidelity to the working code is non-negotiable** — if a "copy" comes out with an unexplained
> code diff, the merge assumption broke: stop and re-open this plan.
>
> Target: **< 450 newly-authored lines** across the whole generation (§4.2).

**Sources — all three are live and current**

| arm | model folder | test folder | engine class |
|---|---|---|---|
| `fm` (Gen11) | [`flow_matcher_v3_uav/`](../../../flow_matcher_v3_uav) | [`FM_v3_uav_test/`](../../../FM_v3_uav_test) | `models.diffusion.FlowMatchingODE` |
| `mf` (Gen3v6) | [`flow_matcher_v3_meanflow/`](../../../flow_matcher_v3_meanflow) | [`FM_v3_meanflow_test/`](../../../FM_v3_meanflow_test) | `MeanFlowODE` + `MeanFlowEngine` + `MFTrajectoryModel` |
| `af` (Gen3v7) | [`flow_matcher_v3_alphaflow/`](../../../flow_matcher_v3_alphaflow) | [`FM_v3_alphaflow_test/`](../../../FM_v3_alphaflow_test) | `AlphaFlowODE` + `AlphaFlowEngine` + `AFTrajectoryModel` |

**Prior plans to read first:**
[`Gen14/init/PLAN_Gen14_visual_mix_ml.md`](../../Gen14/init/PLAN_Gen14_visual_mix_ml.md) (the direct
methodological ancestor — same merge, different task) ·
[`Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md`](../../Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md) ·
[`Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md`](../../Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md) ·
[`Gen11/Epoch9_PCC_Constraints/Plan/PLAN_E9_PCC_constraints.md`](../../Gen11/Epoch9_PCC_Constraints/Plan/PLAN_E9_PCC_constraints.md)

---

## 0. Why this generation exists

Gen3v6 and Gen3v7 answer the Gen13 iMF refutation, but both live on **state-only avoiding-d3il**,
an *open-loop-ish* planning benchmark where a plan's wall-clock cost is an accounting number, not a
control constraint. Gen14 moved them to **visual aligning** — a harder observation space, still the
same benchmark family.

**UAV (Gen11) is the only task in this repo where NFE is physically binding.** `eval_fm_uav.py` runs a
receding-horizon controller at `control_hz = 33` → **~30 ms per replan**, of which the DPCC
projector already eats a measured share (`policy.last_proj_ms`, logged per step since E9). Every
Euler step of the FM sampler is spent inside that budget. A single-step (or 2-step) mean-velocity
field is not a nice-to-have here; it is the difference between meeting and missing the control
deadline.

So the thesis question of Gen3v6/Gen3v7 — *does a few-step objective buy anything once a safety
projector is in the loop?* — gets its sharpest possible form in Gen15:

> **At a matched real-time budget, does MeanFlow / α-Flow reach the goal as safely as FM-ODE
> with fewer NFE, and does the freed wall-clock actually convert into better closed-loop
> behaviour (more projector iterations, fewer circuit-breaker trips, lower tracking error)?**

Note the second clause. On avoiding-d3il, "cheaper" was an end in itself and the answer was
disappointing (Gen13). On UAV, cheaper generation **releases budget to the projector**, which is a
mechanism by which few-step objectives could win *even if their raw sample quality is worse*. That
mechanism does not exist in any other generation in this repo. It is the reason Gen15 is worth
running rather than being a fourth re-ask of the same question.

**And the reassembly is cheap.** Gen11's UAV fork and the Gen3v6/v7 folders are all copy-modify
descendants of `flow_matcher_v3/`. Verified today:

```
diff -rq flow_matcher_v3/ flow_matcher_v3_uav/          → 10 files differ, 0 files added
   datasets/{d4rl,normalization,sequence}.py   UAV data branch + SafeLimitsNormalizer
   models/diffusion.py, models/__init__.py     FM-ODE + UAV logging
   sampling/{policies,projection}.py           real-time diagnostics + UAV constraint families
   utils/{constraints_helpers,serialization,setup,training}.py

diff -rq flow_matcher_v3/ flow_matcher_v3_meanflow/     → 5 files ADDED (mf_*.py), rest cosmetic
```

The UAV divergence is **entirely in data / constraints / logging**; the MF-AF divergence is
**entirely in the engine + backbone**. The two forks touch almost-disjoint file sets. Gen15 is the
union, not a merge conflict.

---

## 1. Decisions (confirm before any code)

1. **Generation number: Gen15, name `UAV_Mix_ML`.** Sibling of Gen14 `Visual_Mix_ML` — same
   reassembly, different payload task. ⚠️ **Gen14 = D3IL visual aligning; Gen15 = UAV MuJoCo.
   Never pool their results.**
2. **Folder pair: `mix_uav/` ↔ `mix_uav_test/`.**
   (Rejected: extending `flow_matcher_v3_uav/` in place — it breaks Gen11 rollback and invalidates
   every existing Gen11 checkpoint path, all of which are keyed on
   `logs/UAV_FM/uav-<scene>/flow_matching_v3_uav/…`.)
3. **Frame base: `flow_matcher_v3_uav/` @ HEAD**, i.e. everything through E9 Fix 15.3 (deadline
   guard, sliding-window circuit breaker, skipped-projection marking) and the
   `config_override_pkl` two-tier reconciliation. The UAV frame is the *host*; the engines are the
   *graft*. Never the other way round — the engine folders have no MuJoCo, no PID/MJPC tracker, no
   scene geometry, and no real-time logging.
4. **Engines shipped: `fm | mf | af`. Three.**
   ❌ **iMF is OUT** (the master row says "possibly iMF"). Rationale: Gen3v4-iMF is `abandoned` in
   `MASTER_TEST_HISTORY.md`, its efficiency thesis was refuted by Gen13 CLOSURE I, and Gen13's
   reassembly attempt ended with *"imF not work outputs not smooth traj"* — non-smooth trajectories
   are exactly the failure mode a 33 Hz PID/MJPC tracker punishes hardest. Adding it costs a 4th
   training run per scene to re-derive a known negative.
   → If the user overrides this: the graft is mechanical (`flow_matcher_v3_imeanflow/models/imf_*.py`
   are twins of the `mf_*.py` files), so it can be added later as a 4th registry entry with no
   restructuring. **Keep the registry open to a 4th key; do not build the 4th arm now.**
5. **No `ddpm` arm.** Unlike Gen14, there is **no diffusion-DPCC UAV checkpoint anywhere in this
   repo** — Gen11 never trained one. ⚠️ **This has a direct consequence for the DA** (§7.1): the
   usual "beat diffusion-DPCC" target does not exist for the UAV task, so Gen15's claim can only be
   *"vs Gen11 naive FM + DPCC"*. That is a weaker claim and must be stated as such, never dressed up
   as beating DPCC. Porting `diffuser_visual_*`'s `GaussianDiffusion` into the UAV frame to create a
   real DPCC baseline is a **separate generation** (§10).
6. **Backbone LOCKED to the two-time `Flow_matcher_U_Net_v2`** (`imf_backbone='unet'`,
   `freq_dim = 32` = Gen11's `dim`). See §6 — this is the decision most likely to be argued with,
   and the one that makes the three-way comparison architecture-controlled.
7. **Task / data / dims frozen at Gen11's E8 defaults:** `cond_mode='pos_only'`
   (obs `[p_des|p]` 6-D, action `Δp_des` 3-D, transition **9-D**), `horizon=8`,
   `controller='pid_stopgo'`, `SafeLimitsNormalizer`, scenes
   `{empty, corridor, s_curve, pillars}`, `MAX_PATH_LENGTH_PER_SCENE` unchanged.
8. **Constraint layer untouched.** `sampling/projection.py` and `utils/constraints_helpers.py`
   copied byte-identical from the UAV fork; `config/uav_projection.yaml` reused **unchanged**
   (all 22 `projection_variants`, all 4 `geo_constraint_variants`). Constraints belong to the
   *task*, not the engine — that is what keeps the arms comparable.
9. **ZERO shared mutable files with Gen11.** Gen15 gets its **own config module,
   `config/uav_mix.py`** (§5 G5). `config/uav.py` is not appended to, not imported from, and not
   edited — Gen11's `_uav_exp_name` and its two blocks stay byte-identical. The **only** shared
   artifact is `config/uav_projection.yaml`, which is **read-only and shared on purpose** (§5 G5).
10. **Own logbase: `logs/UAV_MIX`** (Gen11 keeps `logs/UAV_FM`). Isolation at the top of the path,
    not only at the `prefix` segment — a shared root is one bad `prefix` away from a collision.
11. **HardFlow is OUT of scope** (Gen3v6/v7 both carry `sampling/hardflow_projection.py`; the UAV
    frame has no linear-dynamics `.npz` in UAV normalizer units). Deferred → §10.
12. **Zero retraining of existing work.** Gen11's checkpoints stay where they are and remain
    evaluable from `FM_v3_uav_test/`. Gen15 writes under `logs/UAV_MIX/`. The `fm` arm of
    Gen15 is retrained from scratch **only** to make the parity gate G1 meaningful; if G1 passes on
    a copied Gen11 checkpoint instead, skip the retrain and say so.

---

## 2. What the UAV frame actually is (and what must not break)

Read this section before touching `eval_fm_uav.py`. It is 1601 lines and it is the most
task-entangled file in the repo.

| Load-bearing behaviour | Where | Why it constrains Gen15 |
|---|---|---|
| **33 Hz closed loop, MuJoCo physics, PID / MJPC inner tracker** | `eval_fm_uav.py:rollout_one`, `mjpc_tracker.py` | The engine is called once per outer step; anything that raises per-call latency shows up as tracking error, not as a slower log. |
| **Real-time diagnostics on the Policy** (`last_proj_ms`, `last_proj_cost`, `last_which_trajectory`, `last_infos`) | `flow_matcher_v3_uav/sampling/policies.py` | **Gen3v6/v7's `policies.py` does NOT have these** (§5, graft G1). The eval reads them unconditionally. |
| **`goal_dim` forced to 0 + model patched** | `eval_fm_uav.py:1234-1238` | `SequenceDataset.get_goal_dim()` false-positives on constant UAV channels; a non-zero `goal_dim` slices the trajectory before the projector and throws `IndexError` in `build_matrices`. Must be applied to `mf`/`af` models too. |
| **Circuit breaker + deadline guard + skipped-projection marking** (E9 Fix 15.x) | `sampling/projection.py`, read back via `projector._cb_trips`, `_cost_exploded_count` | These are the metrics that will decide whether "cheaper sampling → more projector budget" is real. Do not copy an older projection.py. |
| **Scene-aware success** (`GOAL_PATH_SCENES`, `GOAL_RADIUS=0.30`, `empty` judged on stable flight only) | `eval_fm_uav.py:78-85` | Cross-arm comparison must use the same definition; never re-derive it. |
| **Eval-tag folder** `K{k}_mpc{B}_{controller}_T{thresh}` | `eval_fm_uav.py:_uav_eval_tag` | Has no engine token — harmless once each arm has its own `prefix` (the arms already separate one level up), but adding it makes the leaf self-describing (§5, graft G4). |
| **Path discriminator** `_uav_exp_name` = `prefix + H{h}_D{diffusion}[_9D]` | `config/uav.py:85-94` | ⚠️ **Shared helper — Gen15 must NOT edit it.** Gen15 defines its own `_uav_mix_exp_name` in its own config module. `D{diffusion}` is the raw engine class path so the three arms separate, but the MF/AF *objective knobs* (`dp`, α-schedule, backbone) appear nowhere — that is the real collision surface (§5, graft G5). |
| **`SafeLimitsNormalizer`, not `LimitsNormalizer`** | `config/uav.py:142` | Constant feature columns in some scenes (`pillars`) → `0/0 = NaN` poisons training from step 0. MF/AF blocks must inherit this. |

---

## 3. The one rule that makes fidelity structural

> **The `fm` arm imports ONLY verbatim copies of `flow_matcher_v3_uav/`.**
> Every newly-authored line lives in a file that only `mf` and `af` import — or in the
> registry/dispatch layer, which `fm` reaches through exactly one lookup.

Gen15's reproduction of Gen11 is then guaranteed **by construction**, and gate G1 (§8) becomes a
cheap confirmation rather than a load-bearing check. The cost is duplicated trainers and duplicated
backbone files. **That duplication is the point** — Gen8 (`imf_visual_aligning/`) is dead precisely
because it shared files with a generation that then moved on without it.

---

## 4. File tree with provenance

### 4.1 Kind legend

**V** verbatim (byte-identical copy) · **S** sed-only (`s/flow_matcher_v3_uav/mix_uav/g`) ·
**G** graft (copy of file A + a block pasted verbatim from file B) · **N** new code.

### 4.2 Copy ledger — the budget this plan is accountable to

| Kind | Files | Newly-authored lines |
|---|---|---|
| **V** verbatim | ~16 | 0 |
| **S** sed-only | ~12 | 0 |
| **G** graft | 6 | ~180 |
| **N** new | 3 | ~250 |
| | | **≈ 430 total** |

Anything that pushes **N** past ~450 means the plan drifted from reassembly into rewriting.
`config/uav_mix.py` (~230 lines) is counted **separately** and is not in the budget — it is
declarative config modelled line-for-line on `config/uav.py`, not authored logic.

### 4.3 Tree

```
mix_uav/
├── __init__.py  setup.py                     S  ← Gen11
├── datasets/                                 S  ← Gen11  (d4rl UAV branch, SafeLimitsNormalizer,
│                                                          sequence.py — the UAV data path is the
│                                                          host and is NEVER taken from Gen3v6/v7)
├── sampling/
│   ├── projection.py                         V  ← Gen11  (E9 Fix15.3 deadline guard + CB)
│   ├── policies.py                           G  ← Gen11 + Gen3v6 fix_5 `executed_idx`   (§5 G1)
│   └── __init__.py                           S  ← Gen11
├── utils/
│   ├── arrays.py constraints_helpers.py logger.py plot.py progress.py timer.py
│   │                                         V  ← Gen11
│   ├── config.py serialization.py setup.py   S  ← Gen11  (setup.py carries the UAV yaml-snapshot
│   │                                                      fix and the plan_-prefix ghost-dir fix —
│   │                                                      do NOT take Gen3v6's copy)
│   ├── training.py                           S  ← Gen11 VERBATIM        → used by `fm`
│   └── training_twotime.py                   G  ← Gen3v7 + Gen11's test-loss/log_freq fixes  (§5 G2)
└── models/
    ├── __init__.py                           N  ← exports the three engines (~15 lines)
    ├── engine_registry.py                    N  ← the dispatch table (~90 lines)          (§5 G3)
    ├── helpers.py                            S  ← Gen11 (apply_conditioning, Losses)
    │
    │  ── arm: fm ───────────────────────────────────────────────────────────────
    ├── diffusion.py                          S  ← Gen11 `FlowMatchingODE`
    ├── unet1d_temporal_cond.py               S  ← Gen11 `Flow_matcher_U_Net_v2`  (one-time)
    │
    │  ── shared by mf + af ──────────────────────────────────────────────────────
    ├── unet1d_twotime_cond.py                S  ← Gen3v6 `unet1d_temporal_cond.py`
    │                                              (= Gen11's file + h_mlp + dual_head +
    │                                               interval_cfg; verified additive, §6)
    │
    │  ── arm: mf ───────────────────────────────────────────────────────────────
    ├── mf_diffusion.py mf_engine.py mf_trajectory_model.py
    │                                         S  ← Gen3v6 (import path → unet1d_twotime_cond)
    │
    │  ── arm: af ───────────────────────────────────────────────────────────────
    └── af_diffusion.py af_engine.py af_trajectory_model.py
                                              S  ← Gen3v7 (same import retarget)

mix_uav_test/
├── train_mix_uav.py                          G  ← Gen11 `train_fm_uav.py` + `--engine`   (§5 G3)
├── eval_mix_uav.py                           G  ← Gen11 `eval_fm_uav.py` (1601 ln) + engine
│                                                  dispatch + engine token in the eval tag (§5 G4)
├── mjpc_tracker.py behavior_logger.py eval_artifacts.py aggregate_scene_summaries.py
│                                             V  ← Gen11 (zero engine coupling — verified: none of
│                                                  the four imports a model class)
└── gates_mix_uav.py                          N  ← G0…G6 of §8 (~140 lines)

config/uav_mix.py                             N  ← NEW FILE, modelled on `config/uav.py`.
                                                 6 blocks + own `_uav_mix_exp_name`.        (§5 G5)
config/uav.py                                 UNTOUCHED — not appended to, not imported from.
config/uav_projection.yaml                    UNCHANGED — read-only, shared on purpose (§5 G5).
Slurm_Codes/sbatch/uav_mix/                   S  ← Gen11's `uav_fm/` sextet + one gates job (§9)
```

**Not copied, deliberately:** `mf_dit_trajectory.py`, `mf_dit_official_trajectory.py`,
`af_dit_trajectory.py`, `af_sit_trajectory.py` (DiT/SiT backbones — excluded by decision §1.6),
`sampling/hardflow_projection.py` (§1.9), anything from `flow_matcher_v3_imeanflow/` (§1.4).

---

## 5. The six real grafts

Everything else is `cp` + `sed`. These six are where actual thought is required.

### G1 — `policies.py`: UAV host + one Gen3v6 fix (~15 authored lines)

The two `policies.py` copies have **diverged in both directions**, verified by diff today:

| | Gen11 UAV | Gen3v6/v7 |
|---|---|---|
| `last_proj_ms` / `last_proj_cost` / `last_which_trajectory` / `last_infos` | ✅ present | ❌ absent |
| `fix_5` `executed_idx` (temporal-consistency reorder bug: `prev_observations` was taking a *different* candidate than the one executed) | ❌ absent | ✅ present |

**Base = Gen11's file** (the eval reads the RT attributes unconditionally). **Graft = Gen3v6's
`executed_idx` block**, pasted verbatim. ⚠️ `fix_5` changes `prev_observations` under
`trajectory_selection='temporal_consistency'`, which is exactly the selection the `dpcc-t` variants
use → **this is a behaviour change for the `fm` arm too**, breaking §3's "verbatim" guarantee.
**Decision required in review:**
- (a) graft `fix_5` into all three arms and document that Gen15-`fm` ≠ Gen11-`fm` on `dpcc-t*`
  variants (recommended — it is a real bug fix, and gate G1 can be run on the non-`t` variants), or
- (b) keep Gen11's file untouched and accept the bug on all three arms (comparison stays internally
  consistent; the bug is shared).
**Recommendation: (a)**, with G1 parity asserted on `diffuser` + `dpcc-c` only.
Whichever is chosen, **it applies to all three arms identically** — never one arm with the fix and
another without.

### G2 — trainers: two files, side by side, neither merged (~40 authored lines)

`fm` uses `utils/training.py` (Gen11, 348 lines, verbatim). `mf`/`af` use
`utils/training_twotime.py`, copied from **Gen3v7** (550 lines).

| Feature | Needed by | Why |
|---|---|---|
| `EXTRA_METRIC_KEYS` passthrough of `_build_info` | mf, af | The adaptive MeanFlow loss is pinned at its ceiling by construction and says nothing about convergence. `raw_mse_u`, `h_mse_b0..b3`, `fm_frac` (+ AF's `alpha`, `clamp_frac`) are the only readable signals. |
| `gradient_clip` applied before `optimizer.step()` | mf, af | Dead config key in this lineage while Gen3v4/Gen13 logged 65–500× loss spikes. |
| `split_seed=42` on `random_split` | mf, af | Unseeded split re-splits on resume, leaking test trajectories into training. |
| `set_train_step(self.step)` before the loss call | **af (mandatory)** | α-Flow's `current_alpha()` reads it. No-op for the others. |

**Why Gen3v7's copy and not Gen3v6's:** Gen3v7's is a strict superset, and it has **LF line
endings** — Gen3v6's `training.py` is **CRLF** (verified: `file` reports CRLF for
`flow_matcher_v3_meanflow/utils/training.py`, LF for `flow_matcher_v3_alphaflow/utils/training.py`),
which makes every subsequent diff in this generation unreadable.
⚠️ **Gen11's `utils/training.py` is ALSO CRLF** — normalize the copied `training.py` to LF as the
first commit, or every future `diff` in `mix_uav/` reads as a whole-file rewrite.

**The graft:** Gen11's two trainer fixes must be pasted into `training_twotime.py`:
`current_test_loss` / `current_test_a0_loss` caching (so `loss_test` is not logged as `None` on
non-test steps) and the `(self.step + 1) % self.log_freq` off-by-one. Both are 4-line blocks.

**Known confound (accept + document):** `split_seed=42` for `mf`/`af` vs Gen11's unseeded split for
`fm` means the arms train on different train/test splits. Compare arms on **closed-loop task
metrics**, which are split-independent (eval uses MuJoCo scenes and fixed episode seeds
`10_000 + i`, not held-out trajectories) — **never** on `test_loss`.

### G3 — `train_mix_uav.py` + `engine_registry.py`: the config-shape problem (~90 + ~40 lines)

This is the one place where the two lineages genuinely disagree, and it is **not** cosmetic.
The `model_config` kwarg-sets are different objects:

```python
# fm  (Gen11): model_config describes the U-NET
utils.Config(args.model, horizon=…, transition_dim=obs+act, cond_dim=obs,
             dim_mults=…, dim=…, returns_condition=…, condition_dropout=…)

# mf/af (Gen3v6/v7): model_config describes the ENGINE, which builds its backbone internally
utils.Config(args.model, state_dim=obs+act, seq_len=horizon, freq_dim=…, depth=…,
             num_heads=…, mlp_dim=…, time_dim=…, dual_head=True, interval_cfg=False,
             imf_backbone='unet', dit_*=…)
```

Consequence, inherited from Gen8's L3 lesson: **`model_config.pkl` describes the engine, not the
U-Net, on the `mf`/`af` arms** — so `load_diffusion` reconstructs an engine there and a bare U-Net
on `fm`. The registry owns both kwarg-set builders; **no `if engine == …` chain may appear in the
train or eval script.**

```python
# mix_uav/models/engine_registry.py                                  (N, ~90 lines)
_P = 'mix_uav.models.'
ENGINES = {
 'fm': dict(diffusion=_P+'diffusion.FlowMatchingODE',
            model=_P+'unet1d_temporal_cond.Flow_matcher_U_Net_v2',
            wraps_backbone=False, trainer='utils.training',
            model_kwargs=_unet_kwargs, diffusion_kwargs=_fm_kwargs,
            supports_num_steps=False),
 'mf': dict(diffusion=_P+'mf_diffusion.MeanFlowODE',
            model=_P+'mf_engine.MeanFlowEngine',
            wraps_backbone=True,  trainer='utils.training_twotime',
            model_kwargs=_twotime_kwargs, diffusion_kwargs=_mf_kwargs,
            supports_num_steps=True),
 'af': dict(diffusion=_P+'af_diffusion.AlphaFlowODE',
            model=_P+'af_engine.AlphaFlowEngine',
            wraps_backbone=True,  trainer='utils.training_twotime',
            model_kwargs=_twotime_kwargs, diffusion_kwargs=_af_kwargs,
            supports_num_steps=True),
}
```

The train script keeps **all** of Gen11's UAV machinery unchanged: `--scene`, the seed resolver,
`MAX_PATH_LENGTH_PER_SCENE`, the seed manifest, W&B naming, auto-resume. Only the three
`utils.Config(...)` blocks route through the registry, and `--engine {fm,mf,af}` is added
(default `fm`). Two one-line retargets: `exp = 'uav_mix'` (→ `config.uav_mix`) and
`from config.uav_mix import MAX_PATH_LENGTH_PER_SCENE` (`train_fm_uav.py:203`) — the latter is easy
to miss and would silently pull Gen11's copy back in.

### G4 — `eval_mix_uav.py`: three touch points in 1601 lines (~30 lines)

1. **`build_experiment`** — the inner `Parser` takes `config = 'config.uav_mix'` (not `config.uav`)
   and `parse_args(experiment=f'mix_uav_{engine}')` / `plan_mix_uav_{engine}` instead of the
   hard-coded `'flow_matching_v3_uav'`. The `config_override_pkl` two-tier reconciliation
   (`override_args=args`) is kept as-is; it is what lets the eval's `flow_steps_v3` (K) override
   the pickled training value.
2. **`_uav_eval_tag` gains an engine token** → `E{engine}_K{k}_mpc{B}_{controller}_T{thresh}`.
   This is **defence in depth, not the primary guard** — the arms already separate two levels up
   via their own `prefix` (`mix_uav_{fm,mf,af}/`) and via `D{diffusion}` in `exp_name`, and Gen11
   is separated further up still by `logbase`. The token exists so the leaf is self-describing when
   results are copied out of the tree by hand or aggregated across arms.
   **The collision that is actually live is intra-arm** (same engine, different `dp`/α/backbone) —
   that one is closed by `_uav_mix_exp_name` (§5 G5) and is what gate G2 must target.
3. **NFE plumbing** — `MeanFlowODE.p_sample_loop` / `AlphaFlowODE.p_sample_loop` accept
   `num_steps=`; `FlowMatchingODE.p_sample_loop` does **not** (it reads `self.flow_steps_v3`).
   The registry's `supports_num_steps` decides whether K goes into `Policy.sample_kwargs` or is set
   on the model. Follow Gen3v6 U3's matched-K discipline: **K is a first-class axis, set identically
   for every arm in a given comparison, and it appears in the output path.**
   The `goal_dim → 0` patch (§2) is applied to whatever model comes back, unchanged.

### G5 — `config/uav_mix.py`: a NEW config module, not an append (~230 lines of config)

**Rule: Gen15 shares no mutable file with Gen11.** `config/uav.py` is left byte-identical —
not appended to, not imported from. Gen15 gets **`config/uav_mix.py`**, written *modelled on*
`config/uav.py` (same single-file train+plan layout, same `watch`/`exp_name` mechanics), with its
own copies of everything Gen11's file defines.

**Why a new file rather than appended blocks** (this reverses the first draft of this plan):
`config/uav.py` defines module-level state that its two existing blocks depend on —
`_uav_exp_name`, `_COND_MODE_DIM`, `MAX_PATH_LENGTH_PER_SCENE`, `args_to_watch`, `logbase`. Gen15
needs a *different* `exp_name` helper (§ below). Editing the shared one silently rewrites Gen11's
checkpoint paths; adding a second helper beside it makes one file serve two generations, which is
exactly the drift that turned Gen8 into dead code. A separate module costs ~80 duplicated lines of
constants and buys total isolation.

**Naming** — `config/` convention is `.py` = train+eval setup entries, `.yaml` = constraint /
projection configs. `uav.py` → **`uav_mix.py`**; the train/eval scripts set `exp = 'uav_mix'` →
`config: str = 'config.' + exp`, so the module name is the only wiring needed.

**Contents:**

| block | role |
|---|---|
| `mix_uav_fm` / `plan_mix_uav_fm` | Gen11's `flow_matching_v3_uav` / `plan_flow_matching_v3_uav` blocks **copied verbatim**, with only `logbase` → `logs/UAV_MIX` and `prefix` → `mix_uav_fm/`. Every training hyper-parameter identical, so gate G1 is a real parity test. |
| `mix_uav_mf` / `plan_mix_uav_mf` | the above `+` Gen3v6's objective knobs (`mf_objective`, `meanflow_data_proportion`, `mf_adp_p/eps`, `t_schedule='logit_normal'`, `p_mean=-0.4`, `p_std=1.0`, `u/v_loss_weight`, `gradient_clip`) `+` backbone keys (`imf_backbone='unet'`, `freq_dim=32`, `dual_head=True`, `interval_cfg=False`). |
| `mix_uav_af` / `plan_mix_uav_af` | same, with Gen3v7's α-schedule keys (`af_alpha_start/end`, anneal shape, target clamp 4.0). |

**Duplicated from `config/uav.py` (copy, do not import):** `MAX_PATH_LENGTH_PER_SCENE`,
`_COND_MODE_DIM`, `args_to_watch`, the `uav_projection.yaml` read + `diffusion_timestep_threshold`
assertion. ⚠️ These are now **two copies of the same constants** — if Gen11 ever re-tunes a
per-scene `max_path_length`, Gen15 does not follow. That is the intended trade (isolation over
DRY, the repo's standing convention), but note it in the changelog so a future reader does not
"fix" it by re-importing.

**The one deliberate share: `config/uav_projection.yaml`.** Read-only, never written, and shared
*because* the constraints must be identical for Gen11 and Gen15 numbers to be comparable at all
(decision §1.8). Provenance is already captured per run — `utils/setup.py:snapshot_configs` copies
the loaded yaml into each run's `config_snapshot_uav/`, so if Gen11 ever edits it, every Gen15 run
still carries the exact file it used. If stronger isolation is wanted later, snapshot-copy it to
`config/uav_mix_projection.yaml` and accept that the two generations then drift apart
scientifically.

**`_uav_mix_exp_name` — Gen15's own path discriminator.** Same shape as `_uav_exp_name`
(`prefix + H{h}_D{diffusion}[_9D]`) **plus** the engine's identity knobs, because that is the real
collision surface: two `mf` runs differing only in `meanflow_data_proportion` (a first-class
ablation axis in Gen3v6, folder-tagged there as `dp`) or in `imf_backbone` land in the **same
checkpoint directory** under Gen11's helper. Suffixes: `_dp{v}_bb{v}` for `mf`,
`_as{v}_ae{v}_bb{v}` for `af`, **empty for `fm`**. Gen3v6/v7's
`args_to_watch_fmv3_{mf,af}_train` lists are the token-for-token reference.

### G6 — `mix_uav/models/unet1d_twotime_cond.py`: verified additive, no authoring (0 lines)

Diffed today: Gen3v6's `unet1d_temporal_cond.py` = Gen11's file **plus** `h_mlp` (the step-size
embedding, summed into the time embedding), the optional `dual_head` v-head, the optional
`interval_cfg` (ω, t_min, t_max) MLPs, and the extended `forward(..., h=, omega=, t_min=, t_max=,
return_v=)` signature. With `h=None, dual_head=False, interval_cfg=False` it is **byte-equivalent in
behaviour** to Gen11's UNet. So this is an `S` copy, not a graft — the file is taken whole from
Gen3v6 and both `mf` and `af` import it. `fm` keeps its own copy and never imports this one (§3).

---

## 6. Backbone: why `unet`, not `mf_dit` / `af_sit`

Gen3v6 defaults to `imf_backbone='mf_dit'` and Gen3v7 to `af_sit`. **Gen15 locks all three arms to
`Flow_matcher_U_Net_v2` with `freq_dim=32`.** Three reasons, in order of weight:

1. **Architecture control.** With the UNet locked, `fm` vs `mf` vs `af` differ *only* in objective
   and sampler. Running MF-on-DiT against FM-on-UNet answers nothing — that is the confound Gen14
   §1.5 locked out for the same reason.
2. **`freq_dim` is the backbone size, not an embedding width.** Per Gen3v6 `FIX_8_UNET_WIDTH`:
   `freq_dim` is passed straight through as the UNet's `dim`, so `32 → 3.97 M` params and
   `256 → 253 M`. Gen11 trains at `dim=32`. **Pass `freq_dim=32` explicitly; never inherit
   Gen3v6's `mlp_dim`/`time_dim=256` into it.** A silent 253 M UNet at 33 Hz would not merely lose,
   it would miss the control deadline and the result would be read as an objective failure.
3. **Real-time cost.** DiT/SiT forward cost at H=8 is not obviously worse, but it is unmeasured in
   this repo's UAV loop, and Gen15's headline metric is wall-clock. Introducing an unmeasured
   architecture into a wall-clock claim is how a generation gets refuted for the wrong reason.

**Verified compatibility detail:** `MFTrajectoryModel`'s `unet` branch constructs
`Flow_matcher_U_Net_v2(horizon=seq_len, transition_dim=state_dim, cond_dim=state_dim, dim=freq_dim,
dim_mults=(1,2,4,8), …)` — note `cond_dim=state_dim` (9), where Gen11 passes
`cond_dim=observation_dim` (6). **This is inert**: `Flow_matcher_U_Net_v2.__init__` never reads
`cond_dim` (grep: it is consumed only by `TemporalValue` / `MLPnet`, which UAV does not use). So the
`mf`/`af` UNets are parameter-identical to `fm`'s. Confirm with gate G3 (param count).

**Optional secondary arm (defer):** once the UNet-locked comparison lands, `mf_dit`/`af_sit` can be
added as extra registry rows for a "best-effort per engine" appendix. Not in the first pass.

---

## 7. What Gen15 measures

### 7.1 The comparison target — read this before writing any DA

There is **no diffusion-DPCC baseline for the UAV task** (§1.5). The DA target is therefore:

> **Target row = the best Gen11 (`fm` + DPCC) row on the same scene, same geo variant, same
> projection variant, same K, same seed set.** A Gen15 arm "wins" when, at equal-or-better success
> and equal-or-better constraint satisfaction, it is Pareto-dominant on the cost axes (fewer NFE
> **and** lower avg per-plan time). Anything else is a **trade-off / non-dominated** result, and must
> be worded that way — never "best".

And the standing hierarchy applies: **MF/AF must beat naive FM.** Beating nothing but themselves is
not a result.

### 7.2 Metrics (all already emitted by the Gen11 eval; nothing new to instrument)

| axis | source |
|---|---|
| success rate, goal-reach, collisions | `eval_fm_uav.py` scene-aware success (§2) |
| steps-to-goal, avg step time | per-episode summary JSON |
| **per-plan wall clock** and **projection ms** | `policy.last_proj_ms`, behaviour log |
| **projector health** — `cb_trips`, `backstop_hits`, skipped-projection marks | `projector._cb_trips`, `_cost_exploded_count` |
| trajectory smoothness / tracking error | `behavior_logger.py` + physics replay |
| NFE budget K | eval tag `K{k}` (matched across arms) |

### 7.3 The K sweep is the experiment

Run every arm at **K ∈ {1, 2, 5, 10, 20}** (Gen11's default is 20). The claim Gen15 exists to test is
that `mf`/`af` hold success at K ≤ 4 where `fm` collapses, and that the released ~15–25 ms/plan shows
up as **fewer circuit-breaker trips** (metric 4 above). If success holds but projector health does
not improve, the result is "cheaper, not better" — report it that way.

---

## 8. Gates (`mix_uav_test/gates_mix_uav.py`) — run on cluster before any science run

| gate | assertion |
|---|---|
| **G0** import | all three arms build a model + diffusion + trainer on CPU/GPU with no missing config key. |
| **G1** `fm` parity | Gen15-`fm` reproduces a Gen11 rollout: same scene / seed / variant / K → identical success, steps, and (bit-for-bit if G1a chosen) trajectory. Run on `diffuser` + `dpcc-c` (see G1 caveat in §5). |
| **G2** path collision + Gen11 isolation | (a) `_uav_mix_exp_name` + `_uav_eval_tag` produce **distinct** paths across a generated cross-product of (engine, K, `dp`, α-schedule, backbone, seed, variant) — the intra-arm knobs are the live risk. (b) **No Gen15 path is a prefix of, or equal to, any Gen11 path**: assert every resolved savepath starts with `logs/UAV_MIX/`. (c) `config/uav.py` is unmodified — assert via `git diff --quiet -- config/uav.py`. |
| **G3** backbone identity | `sum(p.numel())` of the velocity net is **equal** across `fm`, `mf`, `af`. Print it (Gen3v6's FIX_8 lesson: a width defect is invisible for months without a printed param count). |
| **G4** two-time domain | during sampling, every `(t, h)` query satisfies `t, h ∈ [0,1]`, `t + h ≤ 1` — the domain the model was trained on. |
| **G5** projector wiring | with `goal_dim` forced to 0, the projector receives the full 9-D trajectory on all three arms; constraint indices 6,7,8 (`p`) are in range. |
| **G6** real-time budget | mean per-plan wall clock at the configured K is reported per arm, and the fraction of steps exceeding `1/control_hz` is asserted to be logged (not asserted to be zero — that is the experiment). |

---

## 9. Cluster wiring

`Slurm_Codes/sbatch/uav_mix/`, copied from `uav_fm/` (`train_fm_uav.sh`, `train_all_scenes.sh`,
`eval_fm_uav.sh`, `eval_all_scenes.sh`, `aggregate_summaries.sh`, `fm_uav_pipeline.sh`,
`fm_uav_all_pipeline.sh`) with `--engine` threaded through and `ENGINE` exported into the job name,
plus `gates_mix_uav.sh`. Rules that apply (unchanged): submit via `Slurm_Codes/submit.sh`; keep the
GPU/EGL isolation exactly as `uav_fm/` has it; `--time` = 2× expected, 24 h cap; **no tqdm / live
progress bars in batch logs** — the UAV eval's E9 breadcrumb/ETA prints are the intended pattern.

Nothing in this repo runs locally: this container has no Python packages. Every gate and every run
in this plan is a **cluster job (i6-gpu-1)**.

---

## 10. Explicitly out of scope

- **iMF as a 4th arm** (§1.4) — registry stays open, arm is not built.
- **A real DDPM/DPCC UAV baseline** — needs `GaussianDiffusion` ported into the UAV frame *and* a
  full training sweep. Own generation. Until it exists, Gen15 cannot claim to beat DPCC on UAV.
- **HardFlow's constrained sampler on UAV** — needs a linear-dynamics `.npz` refit in the UAV
  normalizer's units (the Gen12 refit warning); a wrong-units `.npz` silently enforces wrong physics.
- **Visual UAV** (FPV camera conditioning) — Gen11's E5 visual pipeline exists but the FM path is
  state-only; visual conditioning + two-time JVP is Gen14's problem shape, not Gen15's.
- **DiT/SiT backbones** (§6, deferred appendix).

---

## 11. Execution order

1. **Decisions §1 confirmed by the user** — especially §1.4 (iMF out), §1.6 (UNet lock), and the
   §5 G1 `fix_5` choice. Nothing is written before this.
2. `mix_uav/` skeleton: V/S copies + LF normalization. No new logic. → **gate G0**.
3. `config/uav_mix.py` (new module, §5 G5) + `engine_registry.py` + `train_mix_uav.py`.
   → **gates G2, G3**.
4. `training_twotime.py` graft. Short smoke train (1k steps, `empty` scene, 1 seed) per arm.
5. `eval_mix_uav.py` graft + `policies.py` graft. → **gates G1, G4, G5, G6**.
6. Full train: 3 arms × 4 scenes × seed set, `n_train_steps=1e5`.
7. Eval: K sweep (§7.3) × projection variants, per scene.
8. DA against the Gen11 target rows (§7.1); write up under `logs_in_develop/Gen15/DA/`.

**Changelog convention:** each coding step writes its own MD under
`logs_in_develop/Gen15/<step>/` (e.g. `init/CHANGELOG_Gen15_coding1.md`), following
`Prompt_for_auto_update_HISTORY_MD.md`. The `MASTER_TEST_HISTORY.md` Gen15 row already exists and
is **not** edited by this plan.
