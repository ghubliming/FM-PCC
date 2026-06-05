# U2 — Avoiding Eval Rebuild: Copy-Modify from fmv3ode / scripts Baselines

**Date**: 2026-06-05  
**Status**: Plan — no code yet  
**Parent**: [`../Fix_7/CHANGELOG.md`](../Fix_7/CHANGELOG.md)

---

## 0. Problem statement

The current eval scripts were copy-modified from Gen6V4/Gen7 **visual aligning** code:

| Script | Lines | Wrong origin |
|---|---|---|
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | 2204 | `fm_visual_aligning_test/eval_fm_visual_aligning.py` |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | 2214 | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` |

They carry ~1800 lines of aligning-specific dead weight: 700-line `VisualAgentWrapper`, `_export_rollout_realtime`, `diag_first_replan.txt`, expert-reference GIF generation, WandB hooks, Z/3D-XYZ panels (aligning is 3D; avoiding is 2D). Fixes 1–7 were all band-aids on inherited aligning code.

**Correct baselines** — the two production-proven state-only avoiding scripts already in the repo:

| Role | File | Lines | Package |
|---|---|---|---|
| FM eval | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | 417 | `flow_matcher_v3_ode_selectable` |
| DPCC/diffuser eval | `scripts/eval.py` | 422 | `diffuser` |
| Train (both share pattern) | `scripts/train.py` | 390 | `diffuser` |
| Load results | `scripts/load_results.py` | 154 | `diffuser` |

These already handle: YAML config, `ObstacleAvoidanceEnv`, DPCC Projector dispatch, NPZ/PNG/log output, aggregate all-seeds plot. The visual rebuild is **pure state → visual-state adjustment** — three targeted swaps per file.

---

## 1. Step 0 — Rename legacy folders (do this first)

Rename in-place — do NOT modify file contents:

| Current | New name |
|---|---|
| `fm_visual_avoiding_test/` | `fm_visual_avoiding_test (legacy_based_on_visual_aligning)/` |
| `diffuser_visual_avoiding_test/` | `diffuser_visual_avoiding_test (legacy_based_on_visual_aligning)/` |

Then create fresh empty replacements that will hold the new files.

The `fm_visual_avoiding/` **library package** (models, utils, sampling) is NOT renamed.

---

## 2. The 3 targeted swaps — applied identically to both evals

### Swap A — Package import

| | FM eval | DPCC eval |
|---|---|---|
| **Before** | `import flow_matcher_v3_ode_selectable.utils as utils` | `import diffuser.utils as utils` |
| **After** | `import fm_visual_avoiding.utils as utils` | `import fm_visual_avoiding.utils as utils` |
| Projector | `from flow_matcher_v3_ode_selectable.sampling.projection import Projector` | `from diffuser.sampling import Policy, Projector` |
| After | `from fm_visual_avoiding.sampling.projection import Projector` | `from fm_visual_avoiding.sampling.projection import Projector` |

`load_diffusion_with_override` is verbatim from the baseline — it loads four pkls and the class swap / kwarg-filter logic is identical. The loaded model is `VisualFlowMatching` (extends `FlowMatchingODE`).

### Swap B — Env: `ObstacleAvoidanceEnv` → `Avoiding_Sim` wrapper

```python
# Baseline (both scripts):
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
env = ObstacleAvoidanceEnv()
env.start()

# Visual replacement (3-line addition):
from d3il.simulation.avoiding_sim import Avoiding_Sim
_base_env = ObstacleAvoidanceEnv()
_base_env.start()
sim = Avoiding_Sim(_base_env)      # wraps env; exposes camera for bp_image
env = sim.env                      # existing reset/step calls unchanged
```

All `env.reset()`, `env.step()`, `env.robot_state()` calls in the baseline loop are unchanged — they go through `sim.env` transparently.

### Swap C — Per-step inference: state-only `policy(...)` → visual `agent.predict(...)`

```python
# Baseline (both scripts):
action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon)

# Visual replacement:
bp_img_raw = sim.env.cameras[0].get_image()[0]
bp_image   = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)).copy() / 255.   # (3,H,W) float
action     = agent.predict((bp_image, pred_xy.copy(), c_xy.copy()), if_vision=True)
# samples = None  (not needed for avoiding metrics)
```

One additional item required: a thin `VisualAgent` wrapper (~35 lines, placed at the top of the eval file) that holds the model + normalizers + optional Projector and implements `.predict()`. This replaces the legacy 700-line `VisualAgentWrapper`.

```python
class VisualAgent:
    def __init__(self, diffusion_model, obs_normalizer, act_normalizer,
                 projector=None, device='cuda:0'):
        self.model    = diffusion_model
        self.obs_norm = obs_normalizer
        self.act_norm = act_normalizer
        self.projector = projector
        self.device   = device

    def predict(self, state, if_vision=False):
        bp_image, des_xy, c_xy = state
        obs_6d    = np.concatenate([des_xy, c_xy, np.zeros(2)])   # 6D obs
        cond_norm = self.obs_norm.normalize(obs_6d)
        cond = {0: torch.tensor(cond_norm, dtype=torch.float32,
                                device=self.device).unsqueeze(0)}
        bp_t = torch.tensor(bp_image, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            samples = self.model(cond, bp_image=bp_t, if_vision=True)
        action_norm = samples.actions[0, 0].cpu().numpy()
        return self.act_norm.unnormalize(action_norm)
```

---

## 3. File-by-file copy-modify table

### `fm_visual_avoiding_test/` (new)

| File | Copy source | Swaps applied | Est. final lines |
|---|---|---|---|
| `eval_fm_visual_avoiding.py` | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` (417) | A + B + C + `VisualAgent` (~35 lines) | **~470** |
| `train_fm_visual_avoiding.py` | Legacy `fm_visual_avoiding_test (legacy)/train_fm_visual_avoiding.py` | Already uses `fm_visual_avoiding.utils` — carry forward unchanged | ~260 |
| `load_results_fm_visual_avoiding.py` | `scripts/load_results.py` (154) | Swap `diffuser` → `fm_visual_avoiding`; `exp='avoiding-d3il'` → `'avoiding-d3il-visual'`; `experiment='plan'` → `'plan_fm_visual_avoiding'` | ~154 |

### `diffuser_visual_avoiding_test/` (new)

| File | Copy source | Swaps applied | Est. final lines |
|---|---|---|---|
| `eval_visual_avoiding_dpcc.py` | `scripts/eval.py` (422) | A + B + C + `VisualAgent` (~35 lines) | **~475** |
| `train_visual_avoiding_dpcc.py` | Legacy `diffuser_visual_avoiding_test (legacy)/train_visual_avoiding_dpcc.py` | Already correct — carry forward unchanged | ~260 |
| `load_results_visual_avoiding_dpcc.py` | `scripts/load_results.py` (154) | Same 3 swaps as FM version above | ~154 |

> **Structural identity**: the two new evals are near-identical. The only line-level difference is Swap A (package name). Every other loop, metric, output path, and plot is verbatim shared.

---

## 4. What is NOT ported from the legacy scripts

| Legacy item | Reason dropped |
|---|---|
| `VisualAgentWrapper` (700 lines) | Replaced by 35-line `VisualAgent` |
| `_export_rollout_realtime` | Not in baseline; aligning-specific |
| `diag_first_replan.txt` | Not in baseline |
| `expert_references/` GIFs | Not in baseline |
| `results_seed_*.pkl` | Not in baseline |
| Per-rollout JSON / `_stats.json` | Not in baseline |
| Z-panel / 3D-XYZ panel | Never valid for 2D avoiding |
| WandB eval hooks | Not in baseline |
| `ProjectorNormalizer` class | Baseline uses normalizer directly |

---

## 5. Output format — identical to baseline scripts

```
{savepath}/results/halfspace_{variant}/
    {projection_variant}.npz       ← n_success, collision_free, n_steps, obs_all, …
    {projection_variant}.png       ← per-trial 6-panel plot
    eval_{projection_variant}.log  ← stdout capture
{savepath}/all_seeds/{halfspace_variant}/
    {projection_variant}.png / .pdf ← aggregate trajectory plot across seeds
```

`scripts/load_results.py` reads this exact NPZ structure — the visual copy works unchanged.

---

## 6. What does NOT change

| Component | Status |
|---|---|
| `fm_visual_avoiding/` library package | Unchanged |
| `d3il/simulation/avoiding_sim.py` | Unchanged |
| `config/avoiding-d3il-visual.py` | Unchanged |
| `config/projection_eval.yaml` | Unchanged — same YAML config |
| `Slurm_Codes/sbatch/fm_visual_avoiding/` | Unchanged — call scripts by path |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/` | Unchanged |

---

## 7. Effort estimate

| Task | Effort |
|---|---|
| Rename 2 legacy folders | 2 shell commands |
| `eval_fm_visual_avoiding.py` — copy + 3 swaps + `VisualAgent` | ~45 min |
| `eval_visual_avoiding_dpcc.py` — copy + 3 swaps (same `VisualAgent`) | ~10 min |
| 2× `load_results_*.py` — copy + 3 line swaps | ~5 min |
| Copy 2 train scripts from legacy | ~2 min |
| AST check all 6 files | ~5 min |
| **Total** | **~1 h** |

---

## 8. Acceptance criteria

| Check | Pass condition |
|---|---|
| Legacy folders renamed, contents intact | `ls "fm_visual_avoiding_test (legacy_based_on_visual_aligning)/"` returns original files |
| Both new eval AST-parse clean | `python3 -m py_compile eval_fm_visual_avoiding.py eval_visual_avoiding_dpcc.py` |
| FM eval line count | `wc -l eval_fm_visual_avoiding.py` ≤ 520 |
| DPCC eval line count | `wc -l eval_visual_avoiding_dpcc.py` ≤ 520 |
| Cluster: NPZ written | `results/halfspace_no_constraint/diffuser.npz` exists after job |
| Cluster: PNG written | `results/halfspace_no_constraint/diffuser.png` exists after job |

---

## 9. Cross-references

| File | Role |
|---|---|
| `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | FM eval source (copy base) |
| `scripts/eval.py` | DPCC eval source (copy base) |
| `scripts/train.py` | Train source pattern reference |
| `scripts/load_results.py` | Load-results source (copy base) |
| `d3il/simulation/avoiding_sim.py` | Swap B — camera access |
| `fm_visual_avoiding/models/visual_gaussian_diffusion.py` | `VisualFlowMatching.forward` — Swap C model call |
| `config/projection_eval.yaml` | YAML config used by all evals |
| `../Fix_7/CHANGELOG.md` | Last patch on legacy — motivation for rebuild |
