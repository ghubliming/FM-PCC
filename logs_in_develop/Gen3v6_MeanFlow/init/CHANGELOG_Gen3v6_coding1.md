# CHANGELOG — Gen3v6 (MeanFlow baseline), coding pass 1

**Date:** 2026-07-22 · **Type:** implementation changelog · **Status:** code complete, **NOTHING RUN**
**Implements:** [`PLAN_Gen3v6_meanflow_baseline.md`](PLAN_Gen3v6_meanflow_baseline.md)
**Base:** copy-modify of Gen3v4 `flow_matcher_v3_imeanflow/` ↔ `FM_v3_imeanflow_test/`
**Nothing committed.** No local execution — this container has no Python deps; all gates run on the cluster.

---

## 0. TL;DR

Gen3v6 is a new sibling generation implementing the **original MeanFlow paper** (arXiv 2505.13447)
as the missing controlled baseline for Gen3v4-iMF. The one scientific difference between the two
generations is the **JVP z-tangent** — **analytic `v = x₁ − x₀`** here vs iMF's **predicted `v_c`**
there. Everything architectural is held identical so the A/B is controlled.

Beyond the objective, this pass also lands three things the plan called out as long-standing repo
debts: the **h-stratified residual metric**, a **gradient clip that is actually applied**, and
config plumbing that **cannot silently overwrite** a sibling run's checkpoint folder.

---

## 1. Files created

| path | origin | note |
|---|---|---|
| `flow_matcher_v3_meanflow/` | copy of `flow_matcher_v3_imeanflow/` | model package; `__pycache__` purged |
| `FM_v3_meanflow_test/` | copy of `FM_v3_imeanflow_test/` | test/eval package |
| `Slurm_Codes/sbatch/MeanFlow/` | copy of `Slurm_Codes/sbatch/iMF/` | 4 sbatch files |
| `FM_v3_meanflow_test/gates_meanflow.py` | **new** | G0/G1/G3′ pre-flight harness (see §7) |
| this file | **new** | changelog |

**Untouched (hard rule, verified):** `flow_matcher_v3_imeanflow/`, `FM_v3_imeanflow_test/`,
`flow_matcher_v3_ode_selectable/`, `diffuser/`, `d3il/`, every pre-existing sbatch dir.
The only pre-existing file modified is `config/avoiding-d3il.py` (additive only — §5).

## 2. Files modified inside the new folders

### 2.1 Renames

Module path `flow_matcher_v3_imeanflow` → `flow_matcher_v3_meanflow` everywhere
(**verified: zero stale references remain**, including docstrings — the only surviving mentions
are deliberate prose pointing at the *sibling* generation).

Model modules renamed `imf_*` → `mf_*`, and the classes with them, so tracebacks are unambiguous:

| old | new |
|---|---|
| `imf_diffusion.py` / `iMeanFlowODE` | `mf_diffusion.py` / `MeanFlowODE` |
| `imf_engine.py` / `iMeanFlowEngine` | `mf_engine.py` / `MeanFlowEngine` |
| `imf_trajectory_model.py` / `iMFTrajectoryModel` | `mf_trajectory_model.py` / `MFTrajectoryModel` |
| `imf_dit_trajectory.py` / `IMFDiTTrajectory` | `mf_dit_trajectory.py` / `MFDiTTrajectory` |

Test scripts (prefixes kept — sbatch/DA tooling assumes them):
`train_/eval_/load_results_flow_matching_v3_imeanflow.py` → `..._v3_meanflow.py`.

Experiment keys switched: `flow_matching_v3_imeanflow` → `flow_matching_v3_meanflow` (train),
`plan_fm_v3_imeanflow` → `plan_fm_v3_meanflow` (eval **and** load_results).

### 2.2 Deletions (Gen3v2/Gen3v4 leftovers — copies only; originals untouched in Gen3v4)

- `FM_v3_meanflow_test/*_ode_selectable.py` (3 files) — per plan §2.2.
- `FM_v3_meanflow_test/Benchmark_ode_solver_Tests/` — ODE-solver-selectable benchmark suite,
  same Gen3v2 lineage, dead for this generation. *(Beyond the plan's literal list; flagged here
  because it is a deletion. Recoverable — it still exists in `FM_v3_imeanflow_test/`.)*
- `flow_matcher_v3_meanflow/models/mf_losses.py` (`iMFTrainingLoss`) — imported by
  `models/__init__.py` and by nothing else; the import was removed too.
- `flow_matcher_v3_meanflow/models/diffusion.py.with_calling_log` — stray debug artifact.

`models/diffusion.py` (`FlowMatchingIMF`) is **kept**: `sampling/policies.py` and the eval script's
class dispatch still reference it.

---

## 3. The objective — `flow_matcher_v3_meanflow/models/mf_diffusion.py`

Rewritten. The derivation, the DATA-AT-1 convention and the JVP tangent triple `(v_inst, +1, −1)`
were already correct in Gen3v4's `_p_losses_meanflow_jvp` and are carried over verbatim.

### 3.1 Deleted (not kept as dead arms)

`_p_losses_imf_official`, `_sample_cfg_scale`, `_sample_cfg_interval`, the `_v_head` closure,
the `fm_equivalent` finite-difference arm, `p_losses()` (its body *was* the `fm_equivalent` arm),
the `imf_objective` dispatch, and every `meanflow_cfg_*` / `meanflow_class_dropout_prob` /
`meanflow_r_equals_t_frac` / `meanflow_adaptive_*` / `meanflow_aux_weight` config key.

Kept: `_predict_uv`, `_predict_velocity`, `q_sample`, `p_sample_loop`, `conditional_sample`,
`sample`, `get_loss_weights`, the state-dict compat shims.

### 3.2 `_p_losses_meanflow` — the four fixes (D3/D5 from the U10 fidelity audit)

**FIX-1 — `(t, r)` sampling.** `r = t·U(0,1)` (which forces `h ≤ t` and starves large-`h` queries
exactly where 1–2-NFE sampling lives) replaced by **two independent logit-normals on the τ axis**:
`τᵢ ~ sigmoid(N(−p_mean, p_std))`, `t = max`, `r = min`, then `meanflow_data_proportion` of the
batch forced to `r = t`. Factored into `_sample_tau_pair()`, which carries the **`−p_mean` trap**
warning in its docstring (using `+p_mean` puts all the mass near noise and looks *almost* fine).
Consequence, as the plan requires: `loss()` no longer pre-samples a single `t` and calls
`_p_losses_meanflow` directly — the `p_losses()` hop is gone.

**FIX-2 — official adaptive loss.** `p=0.5 / c=1e-3 / mean` → `err / sg((err + 0.01)^1.0)` with
`err` the **per-sample SUM** over `(H, D)`. Factored into `_adaptive()`. The docstring records the
discrepancy G2 must account for: official `imeanflow` reduces with SUM, the unofficial
`aux_repo/MeanFlow` with MEAN — a `H·D` rescale that changes what `eps` means.

**FIX-3 — DPCC `loss_weights` no longer injected.** `* self.loss_fn.weights` removed from the
squared error. `action_weight=10` / `loss_discount` remain as config keys (folder naming and
several utils read them, and `loss_fn.weights` is part of the state_dict) but are **not applied
to this loss** — stated in a comment at the `__init__` signature *and* at the loss, so the next
auditor does not "fix" it back.

**FIX-4 — the v head is a full second loss.** `meanflow_aux_weight=0.05` stabiliser replaced by a
second `_predict_uv(x_r, cond, r, h=h)` query regressed to `sg(v_inst)` through the *same* adaptive
form; `loss = (adp(err_u) + adp(err_v)).mean()`. This matches `aux_repo/MeanFlow`'s `fm_loss`,
which likewise trains `v_p` at the same query point on equal footing.

### 3.3 The tangent — guarded

```python
u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
#                                         ^^^^^^ ANALYTIC v — this IS the Gen3v6 hypothesis
```
Wrapped in a boxed 🔴 comment: feeding a predicted `v_c` here turns Gen3v6 into Gen3v4-iMF and
destroys the ablation. `mf_objective` is validated in `__init__` and raises on any value other
than `'meanflow'`, so a stray config key cannot silently select something else.

**Sign cross-check added to the docstring** (this is new — it did not exist in Gen3v4): mapping
`aux_repo/MeanFlow`'s noise-at-1 identity `u = v − h·du/dt` (tangents `(v_c, +1, 0)`, anchor at
their `t`) onto our DATA-AT-1 convention via `u_ours = −u_theirs`, `v_ours = −v_theirs`,
`d/ds = −d/dτ` yields `u = v + h·du/dr` exactly. **The signs agree.** This is a paper-level check,
not a numerical one — G2 still owes the numerical version.

### 3.4 Sampler — untouched, minus CFG

`p_sample_loop`'s interval-jump update (`x += dt·u`, `h = dt = 1/N`) and the U8 torchdiffeq
"homing-missile" `h_sub = t1 − t_scalar` branch are byte-equivalent to Gen3v4. Removed: the
`imf_objective == 'imf_official'` special case (Gen3v6 always takes the unconditional path) and
the `cfg_on` / `omega_b` / `step_cfg` output-mix plumbing.

The DPCC **returns-CFG** branch in `_predict_velocity` is kept but inert (`condition_guidance_w = 0.0`).

### 3.5 Metrics (PLAN §3.4) — all new

`info` now carries: `diffusion_loss` (adaptive; **flagged do-not-read**), `raw_mse_u`, `raw_mse_v`,
`per_dim_rms_u = sqrt(raw_mse_u / (H·D))`, `a0_loss`, `h_mean`, `fm_frac`, and ⭐ the **h-stratified
residuals** `h_mse_b0..b3` over buckets `h==0`, `(0,0.3)`, `[0.3,0.6)`, `[0.6,1.0]`. `raw_mse` and
`aux_loss` are kept as back-compat aliases of `raw_mse_u` / `raw_mse_v` so the inherited pkl keys
and DA tooling keep working.

An **empty h-bucket emits NaN**, not 0 — with `B=32` and `dp=0.5` the high-`h` buckets legitimately
draw no samples on some steps. The trainer drops NaN points, so those curves are **sparse by
design**; a gap is not a failure.

---

## 4. `flow_matcher_v3_meanflow/utils/training.py`

- 🔴 **`gradient_clip` is actually applied.** New `Trainer(gradient_clip=...)` arg;
  `torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)` runs immediately
  before `optimizer.step()`, guarded by `> 0`. It was a **dead key** in this whole lineage
  (POST_U10_III §4.1) while Gen3v4/Gen13 logged 65–500× loss spikes.
- New `grad_norm_history` series records the **pre-clip** norm returned by `clip_grad_norm_` — the
  direct diagnostic for whether the clip is biting.
- New `EXTRA_METRIC_KEYS` mechanism: the 9 metrics above are tracked generically on both
  train and test sides (`train_extra_losses` / `test_extra_losses`), NaN-filtered, and persisted
  as `training_<key>_losses` / `test_<key>_losses` in `losses.pkl` and `losses.json`.
- `test()` now returns a 5-tuple (added `test_extra`, a NaN-safe per-key running mean).
  **Only caller is `train_epoch`** — verified.
- `save()` / `save_best()` refactored onto a shared `_checkpoint_payload()` so the two can no
  longer drift; `load()` and the `losses.pkl` fallback restore the new series.

## 5. `config/avoiding-d3il.py` — additive only

- **`args_to_watch_fmv3_mf_train`** — `H`, `D`, `aw`, `obj{mf_objective}`, `bb{imf_backbone}`,
  `ts{t_schedule}`, **`dp{meanflow_data_proportion}`**. `dp` is in the folder name deliberately:
  POST_U10_II §1.1 documents two Gen3v4 runs that wrote to a byte-identical folder because four
  swept knobs were absent from `args_to_watch`.
- **`flow_matching_v3_meanflow`** (train block) — `mf_objective='meanflow'`,
  `meanflow_data_proportion=0.5`, `mf_adp_p=1.0`, `mf_adp_eps=0.01`, `dual_head=True`,
  `interval_cfg=False`, `imf_backbone='dit'`, `dit_condition_on_t=False`, `p_mean=-0.4`,
  `p_std=1.0`, `t_schedule='logit_normal'`, 100k steps, bs 32, accum 2, lr 5e-4,
  `gradient_clip=1.0`, ema 0.995, `train_test_split=0.9`, `condition_guidance_w=0.0`.
- **`plan_fm_v3_meanflow`** (eval block) — every architecture key mirrors the train block;
  `flow_steps_v3=2` (sweep {1,2,5,10}); `eval_use_ema=True`.

**Verified mechanically** (stubbing `watch`/`yaml`, no torch needed) that the plan block's
`diffusion_loadpath` renders **token-for-token identical** to the train block's `exp_name`:

```
flow_matching_v3_meanflow/H8_Dflow_matcher_v3_meanflow.models.MeanFlowODE_aw10_objmeanflow_bbdit_tslogit_normal_dp0.5
```

The same check run against the Gen3v4 pair also matches, confirming the method (not just the result).
An architecture-key parity diff between the two Gen3v6 blocks reports **no mismatches**.

## 6. `Slurm_Codes/sbatch/MeanFlow/` — copy-modify, 4 files

`train_meanflow.sh`, `eval_meanflow.sh`, `load_results_meanflow.sh`, `meanflow_pipeline.sh`.
Changed **only**: job names (`mf_train` / `mf_eval` / `mf_load` / `mf_pipeline`), the three python
invocations, `--wandb-project FMPCC-MeanFlow`, `SBATCH_DIR`, and the two script names inside the
pipeline. A normalized diff against `sbatch/iMF/` confirms nothing else moved — the conda
prologue, `PYTHONPATH`, `MUJOCO_GL=egl`, the **GPU/EGL isolation guard**, the W&B key file, the
`trap on_exit` and the `latest.log` symlink are byte-identical. `--time` stays `24:00:00`.
No tqdm/live bars (the inherited trainer keeps `mininterval=1e10`).

Submission is unchanged:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/meanflow_pipeline.sh   # train -> eval, afterok
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh      # train only
```

## 7. `FM_v3_meanflow_test/gates_meanflow.py` — **added beyond the plan's deliverable list**

The plan's hand-off checklist requires G0–G3 to pass before the full run but ships no harness.
This script provides one; it is dataset-free and takes seconds:

- **G0** build engine+ODE from the Gen3v6 defaults, one forward `(x, τ, h) → (u, v)`, assert
  shapes, assert no `flow_matcher_v3_imeanflow` module got imported.
- **G1** `h → 0` degeneracy, `‖u − v‖/‖v‖ < 0.05`. Takes `--ckpt`; **at random init this cannot
  pass** (the u/v heads are independent) and is reported as INFO, not FAIL.
- **G3′** 200 Adam steps of the real `_p_losses_meanflow` on a fixed random batch: `raw_mse_u`
  must fall, no NaN loss, no infinite `h_mse_b*`.

**G2 is deliberately not scripted.** It needs `aux_repo/MeanFlow` importable next to this package,
and that checkout is local, not on the cluster. Run it where both exist; the docstring records the
two things that must be normalized first (their predicted `v_c` → analytic `v`; their MEAN
reduction → our SUM).

The **real** G3 (200-step smoke on the actual dataset) needs no new code:
```bash
python FM_v3_meanflow_test/train_flow_matching_v3_meanflow.py --seed 6 --n_train_steps 200 --n_steps_per_epoch 100
```

---

## 8. Incidental fix found while porting

Gen3v4's train script **never forwarded `t_schedule`, `p_mean`, `p_std`** into `diffusion_config`.
It happened to be harmless because `iMeanFlowODE.__init__`'s defaults (`'logit_normal'`, `-0.4`,
`1.0`) coincide with the config values — but any config change to those keys would have been
silently ignored. Gen3v6's train script passes all three explicitly. **Gen3v4 is not touched**;
flagging it here in case anyone ever sweeps `p_mean` there.

## 9. Deviations from the plan, stated

1. **`interval_cfg=False` does NOT change the state_dict on the DiT arm.** The plan (§3.5) says it
   does. That is true for the **UNet** arm, where the flag gates the CFG embedders — but
   `MFTrajectoryModel` never forwards `interval_cfg` to `MFDiTTrajectory`, which always builds
   `omega_embedder` / `cfg_t_*_embedder` / `t_min|max_tokens`. With `imf_backbone='dit'` (the
   configured arm) those tokens still exist and are fed a **constant default** (`ω→0 ⇒ w_arg=0`,
   guidance off) at both train and sample time, making them inert.
   **This is left as-is on purpose:** it keeps the Gen3v6 backbone parameter-identical to Gen3v4's
   DiT, which is exactly what makes the A/B architecture-controlled. Consequence to keep in mind:
   Gen3v6 and Gen3v4 DiT checkpoints are *shape*-compatible even though they are semantically
   incompatible — the folder separation (§5) is what keeps them apart, not a load error.
2. **v-head query obtained by a second forward pass, not from the JVP's `has_aux`.** Gen3v4's
   `imf_official` gets `v_aux` free from `_jvp(..., has_aux=True)`. That would be numerically
   identical here and ~1 forward cheaper, but it makes the v gradient depend on forward-mode aux
   unwrapping, which cannot be verified in this container. The plan specifies the explicit second
   call and `aux_repo/MeanFlow` also does two forwards; followed the plan.
3. **Deleted `Benchmark_ode_solver_Tests/` and `mf_losses.py`** beyond the plan's literal deletion
   list (§2.2 above) — both dead in this generation, both still present in Gen3v4.

## 10. Inherited traps NOT fixed (must be stated in the results MD)

- ⚠️ **Window-level `train_test_split=0.9` leaks.** POST_U10_III §4.2: at H=8 adjacent windows
  share 7 of 8 frames, so `loss_test` is effectively a train loss. Gen3v6 **inherits the leak** —
  an episode-level split was not implemented. Either implement it, or label every val number as
  leaking. Do not present it as generalisation. (Flagged in a comment in the config block.)
- ⚠️ **Never read `diffusion_loss` / `train/loss` as convergence** — the adaptive loss is pinned at
  its ceiling by construction. Read `raw_mse_u`.
- ⚠️ **Matched-budget or nothing** — every MeanFlow-vs-X table must be at equal K (fix_7.3 §9).

## 11. Status vs the plan's hand-off checklist

- [x] folders copied, `__pycache__` purged, zero stale package references
- [x] scripts renamed, `*_ode_selectable.py` deleted, experiment keys switched
- [x] `imf_official` / `fm_equivalent` arms and all CFG machinery deleted
- [x] `_p_losses_meanflow` rewritten with FIX-1..FIX-4
- [x] JVP z-tangent still analytic `v_inst`, with the do-not-change comment
- [x] `interval_cfg=False`, `dit_condition_on_t=False`, `dual_head=True` (see §9.1 caveat)
- [x] `h_mse_b0..b3` + `raw_mse_u/v` + `per_dim_rms_u` logged and plumbed to W&B
- [x] `gradient_clip` actually applied in `utils/training.py` (+ pre-clip norm logged)
- [x] 2 config blocks + `args_to_watch_fmv3_mf_train`; `dp` in the folder name;
      `diffusion_loadpath` verified to match token-for-token
- [x] `Slurm_Codes/sbatch/MeanFlow/` copied, only the listed lines changed, EGL guard untouched
- [ ] **G0–G3 not run** — no Python in this container. Run `gates_meanflow.py` on i6-gpu-1 first.
- [ ] **G2 (parity vs `aux_repo/MeanFlow`) not run** — see §7.
- [x] changelog written (this file)
- [ ] **MASTER_TEST_HISTORY row handed to the user, not self-applied** — see §12
- [x] nothing committed

## 12. MASTER_TEST_HISTORY row — prepared, NOT applied

Per repo convention this was **not** written into the master index. Insert directly after the
`Gen3v5 (BNS Solver)` row:

```markdown
| **Gen3v6 (MeanFlow Baseline)** | [flow_matcher_v3_meanflow/](../flow_matcher_v3_meanflow) | [FM_v3_meanflow_test/](../FM_v3_meanflow_test) | July 2026 | **MeanFlow (2505.13447) faithful baseline** — Gen3v4 sibling with the ANALYTIC-v JVP tangent (vs iMF's predicted v_c), official adaptive loss (p=1, eps=0.01, sum), two independent logit-normals, no CFG. Isolates iMF's headline contribution as a controlled A/B. Adds the h-stratified residual metric and a real gradient clip. | working on |
```

## 13. Next steps

1. Sync to the cluster, run `python FM_v3_meanflow_test/gates_meanflow.py` (G0 + G3′).
2. Run the real G3 smoke: 200 steps on the dataset (command in §7).
3. Run G2 wherever `aux_repo/MeanFlow` is importable.
4. Full train (1 seed) → eval at `flow_steps_v3 ∈ {1,2,5,10}`, matched-K against the FM comparator.
5. **Kill criterion (pre-registered, PLAN §7):** if `h_mse_b3` is flat at its step-0 value while
   `h_mse_b0` drops ~10×, the field is untrained exactly where 1–2 NFE lives. Report and **stop** —
   that is `COMPARE §7.3` confirmed and the motivation for Gen3v7 (AlphaFlow).
