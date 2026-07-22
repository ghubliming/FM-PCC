# PLAN — Gen3v6: faithful **MeanFlow** baseline as a Gen3v4 sibling

**Date:** 2026-07-22 · **Type:** implementation plan · **NO CODE WRITTEN YET**
**Status:** ready for hand-off to an implementing agent
**Order:** ⭐ **Implement Gen3v6 FIRST**, test it, then Gen3v7 (AlphaFlow). See §0.3.
**Sibling plan:** [`../../Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md`](../../Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md)
**Background:** [`../../imeanflow_train/AUDIT_port_vs_upstream_and_the_train_in_imeanflow_proposal.md`](../../imeanflow_train/AUDIT_port_vs_upstream_and_the_train_in_imeanflow_proposal.md)

---

## 0. Decisions taken (read before touching anything)

### 0.1 Generation number: **Gen3v6**, not Gen3v5

`Gen3v5` is already reserved — `MASTER_TEST_HISTORY.md` line 23 lists **"Gen3v5 (BNS Solver) — Pending"**, and `logs_in_develop/Gen3v5_BNS_FMv3ODE/` exists. Do not reuse it. This plan is **Gen3v6**; AlphaFlow is **Gen3v7**.

### 0.2 Base to copy-modify: **Gen3v4 (`flow_matcher_v3_imeanflow`)**, NOT FMv3ODE

This was the open question. The answer is unambiguous:

| requirement | Gen3v4 `flow_matcher_v3_imeanflow` | Gen3v2 `flow_matcher_v3_ode_selectable` (FMv3ODE) |
|---|---|---|
| two-time network `u(z, τ, h)` | ✅ exists (`_predict_uv`, `h=` arg through UNet **and** DiT) | ❌ only `v(z, t)` — single time |
| dual u/v heads | ✅ `dual_head=True` | ❌ none |
| JVP machinery (`torch.func.jvp`) | ✅ `_p_losses_meanflow_jvp` | ❌ none |
| logit-normal `t` schedule | ✅ `t_schedule='logit_normal'`, `p_mean`, `p_std` | ❌ Beta only |
| interval-CFG net inputs (ω, t_min, t_max) | ✅ `interval_cfg=True` | ❌ none |
| few-step `u`-sampler (`x += dt·u`) | ✅ `p_sample_loop` iMF branch | ❌ FM Euler/RK4 only |
| DPCC projection + eval harness | ✅ | ✅ |

**MeanFlow is a strict ancestor of iMF.** Every component MeanFlow needs is already in Gen3v4; starting from FMv3ODE would mean re-writing the two-time backbone, the dual head, the JVP and the u-sampler — i.e. re-doing Gen3v4. **Copy Gen3v4.**

⭐ **Bonus that makes this generation cheap:** Gen3v4 *already contains* a MeanFlow arm. `imf_diffusion.py::_p_losses_meanflow_jvp` was identified in [`../../Gen3v4_imf/U10/debug_notes/INVESTIGATION_imf_fidelity_vanilla_vs_improved_meanflow.md`](../../Gen3v4_imf/U10/debug_notes/INVESTIGATION_imf_fidelity_vanilla_vs_improved_meanflow.md) §4 as **original (vanilla) MeanFlow, not iMF** — it uses the *analytic* `v_inst` as JVP tangent. So Gen3v6 is not "write MeanFlow"; it is **"take the MeanFlow arm that already exists and make it faithful"**, per the D3/D5 fix list that audit produced. That is a small, well-specified delta.

### 0.3 Why MeanFlow first, AlphaFlow second

1. **It is the missing baseline.** Gen3v4 compared iMF against *FM*, never against *MeanFlow*. Any claim of the form "iMF/α-Flow beats MeanFlow on constrained control" is currently unsupported — there is no MeanFlow number.
2. **It is the cheapest.** ~1 new loss function (mostly editing an existing one), no new schedule plumbing, no new trainer hooks.
3. **It de-risks the shared scaffolding** — folder copy, config-block pair, `args_to_watch`, sbatch dir, eval override path. Gen3v7 reuses all of it. If the scaffold is wrong, find out on the cheap generation.
4. **α-Flow's α→0 limit *is* MeanFlow** (proved in the Gen3v7 plan §3.4). Gen3v6 therefore doubles as the **α=0 anchor** of Gen3v7's schedule — a free correctness gate for Gen3v7.

### 0.4 What "MeanFlow" means here — pin this down, it is the whole point

There are three things in the wild called MeanFlow. **Gen3v6 implements (a).**

| | JVP z-tangent | regression target | in repo? |
|---|---|---|---|
| **(a) MeanFlow paper** (2505.13447), *what we build* | **analytic** `v = x₁ − x₀` | u-form: `u_pred → sg(v + h·D_tot)` | Gen3v4 `meanflow_jvp` (unfaithful in t/r + loss) |
| (b) `aux_repo/MeanFlow` `mode='meanflow'` | **predicted** `v_c` | u-form | reference only |
| (c) iMF / `mode='i-meanflow'` | **predicted** `v_c` | V-form: `u + h·sg(D_tot) → sg(v_g)` | Gen3v4 `imf_official` ✅ |

⚠️ Note (b): the unofficial `aux_repo/MeanFlow` repo distinguishes its two modes **only by u-form vs V-form loss** — it feeds predicted `v_c` in *both*. That is *not* the paper's MeanFlow-vs-iMF distinction. Use `aux_repo/MeanFlow` as a PyTorch cross-check for **shapes, adaptive-loss form and the jvp call signature**, never as the definition of "MeanFlow".

**The scientific payload of Gen3v6:** with Gen3v4-`imf_official` as the other arm, Gen3v6 isolates the **analytic-vs-predicted JVP tangent** — the single change the iMF paper claims as its contribution — measured on constrained control instead of FID.

---

## 1. Deliverables

| # | artifact | action |
|---|---|---|
| D1 | `flow_matcher_v3_meanflow/` | full copy of `flow_matcher_v3_imeanflow/`, then modify |
| D2 | `FM_v3_meanflow_test/` | full copy of `FM_v3_imeanflow_test/`, then modify |
| D3 | `config/avoiding-d3il.py` | **+2 blocks**: `flow_matching_v3_meanflow`, `plan_fm_v3_meanflow`; **+1** `args_to_watch_fmv3_mf_train` |
| D4 | `Slurm_Codes/sbatch/MeanFlow/` | copy-modify of `Slurm_Codes/sbatch/iMF/` — 4 files |
| D5 | `logs_in_develop/Gen3v6_MeanFlow/init/CHANGELOG_Gen3v6_coding1.md` | changelog after coding (repo convention) |
| D6 | MASTER_TEST_HISTORY row | **prepared in §8 — do NOT self-edit the master file**; hand it to the user |

**Untouched (hard rule):** `flow_matcher_v3_imeanflow/`, `FM_v3_imeanflow_test/`, `flow_matcher_v3_ode_selectable/`, `diffuser/`, `d3il/`, every existing sbatch dir. Copy-modify isolation — no shared refactor.

---

## 2. Step-by-step: folder creation

```bash
cd /workspaces/FM-PCC
cp -r flow_matcher_v3_imeanflow flow_matcher_v3_meanflow
cp -r FM_v3_imeanflow_test      FM_v3_meanflow_test
find flow_matcher_v3_meanflow FM_v3_meanflow_test -name __pycache__ -type d -exec rm -rf {} +
```

Then, inside the two new folders only:

**2.1 Rename the module path everywhere.** Every `flow_matcher_v3_imeanflow` → `flow_matcher_v3_meanflow`.
```bash
grep -rl "flow_matcher_v3_imeanflow" flow_matcher_v3_meanflow FM_v3_meanflow_test \
  | xargs sed -i 's/flow_matcher_v3_imeanflow/flow_matcher_v3_meanflow/g'
```
⚠️ **Then verify by grep that zero references remain**, including in docstrings. A stale reference silently imports the *old* package and you will debug a phantom.

**2.2 Rename the test scripts** (keep the `train_/eval_/load_results_` prefixes — the sbatch and DA tooling assume them):
```
FM_v3_meanflow_test/train_flow_matching_v3_imeanflow.py       -> train_flow_matching_v3_meanflow.py
FM_v3_meanflow_test/eval_flow_matching_v3_imeanflow.py        -> eval_flow_matching_v3_meanflow.py
FM_v3_meanflow_test/load_results_flow_matching_v3_imeanflow.py-> load_results_flow_matching_v3_meanflow.py
```
Delete the `*_ode_selectable.py` copies that ride along in that folder — they are Gen3v2 leftovers and only cause confusion.

**2.3 Experiment keys.** In the three scripts, change:
- `parse_args(experiment='flow_matching_v3_imeanflow')` → `'flow_matching_v3_meanflow'` (train script, ~line 166)
- `parse_args(experiment='plan_fm_v3_imeanflow')` → `'plan_fm_v3_meanflow'` (eval script, ~line 79)

**2.4 Class renames inside `flow_matcher_v3_meanflow/models/`** — optional but strongly recommended so tracebacks are unambiguous:
`iMeanFlowODE` → `MeanFlowODE`, `iMeanFlowEngine` → `MeanFlowEngine`, `iMFTrajectoryModel` → `MFTrajectoryModel`, `IMFDiTTrajectory` → `MFDiTTrajectory`. Update `models/__init__.py` and the `'model'`/`'diffusion'` config strings to match.
⚠️ If you rename, do it **before** any training run — the class name is baked into the pickled `args`, and `config_override_pkl` will fight you afterwards. If you would rather not risk it, keep the old names; it costs nothing scientifically.

---

## 3. The code change — `flow_matcher_v3_meanflow/models/imf_diffusion.py`

This is the only file with real logic changes. Rename it `mf_diffusion.py` if you did 2.4.

### 3.1 Delete what is not MeanFlow

Remove (do not keep as dead arms — this generation must be readable):
- `_p_losses_imf_official` and everything only it uses: `_sample_cfg_scale`, `_sample_cfg_interval`, the `_v_head` closure, `meanflow_cfg_*`, `meanflow_class_dropout_prob`, `meanflow_data_proportion` handling of `cond_drop`.
- the `'fm_equivalent'` and `'imf_official'` branches of the `imf_objective` dispatch in `loss()` and `p_losses()`.
- the `imf_objective` config key itself → replaced by `mf_objective` (§3.5).

Keep: `_predict_uv`, `_predict_velocity`, `q_sample`, `p_sample_loop`, `conditional_sample`, `sample`, `get_loss_weights`, the state-dict compat shims.

### 3.2 Rewrite `_p_losses_meanflow_jvp` → `_p_losses_meanflow` (faithful)

Start from the existing body (`imf_diffusion.py:526-642`) — the derivation, the DATA-AT-1 convention comment, and the JVP tangent triple `(v_inst, +1, −1)` are **already correct and verified**; keep them verbatim. Apply exactly four fixes, which are the D3/D5 items from the U10 fidelity audit §7.

**FIX-1 (D3) — `(t, r)` sampling.** Replace
```python
r = t * torch.rand_like(t)                                   # WRONG: forces h <= t
anchor = torch.rand_like(t) < self.meanflow_r_equals_t_frac
r = torch.where(anchor, t, r)
```
with two **independent** logit-normals on the τ axis, mirroring `_p_losses_imf_official:667-673` (which is already correct — copy its pattern):
```python
# τ axis (DATA-AT-1). Official samples s ~ sigmoid(N(p_mean, p_std)) with mass near data
# (s→0); under τ = 1 − s that is τ ~ sigmoid(N(−p_mean, p_std)).
# TRAP: using +p_mean here puts the mass near NOISE. See imf_diffusion.py:666.
tau1 = torch.sigmoid(torch.randn(B, device=device) * self.p_std - self.p_mean)
tau2 = torch.sigmoid(torch.randn(B, device=device) * self.p_std - self.p_mean)
t = torch.maximum(tau1, tau2)      # data-side end
r = torch.minimum(tau1, tau2)      # noise-side anchor = the query point
anchor = torch.rand(B, device=device) < self.meanflow_data_proportion
r = torch.where(anchor, t, r)
h = t - r
```
⚠️ **Consequence:** `loss()` must stop passing a pre-sampled `t`. MeanFlow samples its *own* `(t, r)`. Route it exactly like `imf_official` does — `loss()` calls `self._p_losses_meanflow(x, cond, returns)` directly and never enters the single-`t` `p_losses()` path (`imf_diffusion.py:416-417` is the pattern).

**FIX-2 (D5) — adaptive loss to official form.** Replace the `p=0.5 / c=1e-3 / mean` variant with the official one:
```python
err = (u_pred - u_target).pow(2).sum(dim=(1, 2))     # per-sample SUM over (H, D)
loss_u = err / (err + self.mf_adp_eps).pow(self.mf_adp_p).detach()   # p=1.0, eps=0.01
```

**FIX-3 (D5) — drop the DPCC `loss_weights` injection.** Remove `* self.loss_fn.weights` from the squared error. `action_weight=10` / `loss_discount` are a DPCC idea with no counterpart in MeanFlow, and they change the gradient geometry of the identity.
⚠️ Keep the config keys `action_weight`/`loss_discount` present (the folder name and several utils read them) — just do not apply them to this loss. Say so in a comment or the next auditor will "fix" it back.

**FIX-4 (D5) — v-head is a full second loss, not a 0.05 stabiliser.** MeanFlow's dual head trains the v-head on the same footing:
```python
_u2, v_pred = self._predict_uv(x_r, cond, r, h=h, returns=returns)
err_v  = (v_pred - v_inst.detach()).pow(2).sum(dim=(1, 2))
loss_v = err_v / (err_v + self.mf_adp_eps).pow(self.mf_adp_p).detach()
loss   = (loss_u + loss_v).mean()
```

### 3.3 🔴 The one thing that MUST NOT change — the tangent

```python
u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
#                                         ^^^^^^ ANALYTIC v — this IS the Gen3v6 hypothesis
```
**Feeding predicted `v_c` here turns Gen3v6 into Gen3v4-iMF and destroys the entire ablation.** Put that sentence in the code as a comment. A future agent reading the iMF audit will be tempted to "fix" it.

### 3.4 Metrics — instrument before you run

The U9 lesson (`COMPARE ... §7.1`) is that the adaptive `loss` is **flat at its ceiling by construction** and must never be read. Emit in `info`:

| key | what |
|---|---|
| `diffusion_loss` | the adaptive loss (kept for pipeline compat — **do not read it**) |
| `raw_mse_u`, `raw_mse_v` | `err.mean()`, `err_v.mean()` — **the real convergence signals** |
| `per_dim_rms_u` | `sqrt(raw_mse_u / (H·D))` — comparable across horizons and generations |
| `a0_loss` | `(u_pred − u_target)[:, 0, :action_dim]²`.mean() — the DPCC-comparable number |
| ⭐ `h_mse_b0..b3` | **h-stratified** `err` in buckets `h==0`, `(0,0.3)`, `[0.3,0.6)`, `[0.6,1.0]` |
| `h_mean`, `fm_frac` | sampler sanity |

⭐ `h_mse_b*` is **free** (the per-sample `err` already exists before `.mean()`) and it is the single highest-value metric in this project — `COMPARE §7.4.1` called for it a month ago and **it was never implemented in any generation**. Gen3v6 is where it lands. It answers directly: *is the field bad only at large `h`, which is exactly where few-NFE sampling lives?*

Wire the new keys into `flow_matcher_v3_meanflow/utils/training.py` (the `self.train_*_losses` lists around lines 91-101) and into the W&B replay map in the train script (`companion_keys`, ~line 63).

### 3.5 Config keys owned by this generation

| new key | default | note |
|---|---|---|
| `mf_objective` | `'meanflow'` | only value for now; keeps a folder-name slot for future arms |
| `meanflow_data_proportion` | `0.5` | FM anchors (`h=0`). Official MeanFlow value |
| `mf_adp_p` | `1.0` | official |
| `mf_adp_eps` | `0.01` | official |
| `p_mean` / `p_std` | `-0.4` / `1.0` | official-convention; negated inside the sampler |

**Delete** from both new blocks: `imf_objective`, `meanflow_r_equals_t_frac`, `meanflow_adaptive_p`, `meanflow_adaptive_c`, `meanflow_aux_weight`, `meanflow_cfg_*`, `meanflow_class_dropout_prob`, `interval_cfg`, `condition_guidance_w` (leave at `0.0` in the plan block if any util reads it).

⚠️ `interval_cfg=False` means the backbone no longer takes `(ω, t_min, t_max)` tokens ⇒ **the state_dict shape changes** ⇒ Gen3v6 checkpoints are not interchangeable with Gen3v4's. That is intended and it is why the folders are siblings.

### 3.6 Sampler — do not touch

`p_sample_loop`'s iMF branch (`x += dt·u`, `h=dt=1/N`) is **already faithful** (U10 audit F1, re-verified). MeanFlow and iMF share the identical sampler. Leave it alone. If you renamed the class, make sure the `imf_objective == 'imf_official'` special-case at `imf_diffusion.py:259` is replaced by the unconditional path — MeanFlow always uses the interval-jump sampler.

---

## 4. `config/avoiding-d3il.py` — the two new blocks

**4.1** Add next to `args_to_watch_fmv3_imf_train` (~line 66):
```python
args_to_watch_fmv3_mf_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('action_weight', 'aw'),
    ('mf_objective', 'obj'),
    ('imf_backbone', 'bb'),      # keep the key name: the backbone classes are inherited
    ('t_schedule', 'ts'),
    ('meanflow_data_proportion', 'dp'),   # NEW: dp is a first-class ablation axis here
]
```
⭐ **`dp` in the folder name is a deliberate fix.** `POST_U10_II` §1.1 documents a live overwrite hazard: four knobs changed between two Gen3v4 runs and **none was in `args_to_watch`**, so both runs wrote to a byte-identical folder. Any knob you intend to sweep must appear here.

**4.2** `'flow_matching_v3_meanflow'` — copy `'flow_matching_v3_imeanflow'` (line 449) and change:
```python
'model':     'flow_matcher_v3_meanflow.models.MeanFlowEngine',
'diffusion': 'flow_matcher_v3_meanflow.models.MeanFlowODE',
'prefix':    'flow_matching_v3_meanflow/',
'exp_name':  watch(args_to_watch_fmv3_mf_train),
'mf_objective': 'meanflow',
'meanflow_data_proportion': 0.5,
'mf_adp_p': 1.0, 'mf_adp_eps': 0.01,
'dual_head': True,          # v-head is a real head here (FIX-4)
'interval_cfg': False,      # 🔴 no CFG in Gen3v6 — changes state_dict shape, intended
'imf_backbone': 'dit',      # match Gen3v4's DiT arm so the A/B is controlled
'dit_condition_on_t': False,# official: condition on h only. KEEP FALSE (see audit §2.3)
'p_mean': -0.4, 'p_std': 1.0, 't_schedule': 'logit_normal',
'n_train_steps': 100000, 'batch_size': 32, 'gradient_accumulate_every': 2,
'learning_rate': 5e-4, 'gradient_clip': 1.0, 'ema_decay': 0.995,
'train_test_split': 0.9,
```
🔴 **`gradient_clip: 1.0` is a DEAD KEY in this lineage** — `POST_U10_III` §4.1: it exists in the config and `utils/training.py` **never reads it**. Both Gen3v4 and Gen13 show 65–500× loss spikes and neither clips. **Gen3v6 must actually implement it:** add `torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)` in `utils/training.py` immediately before `self.optimizer.step()`, guarded by `if self.gradient_clip > 0`. This is a required item, not optional.

**4.3** `'plan_fm_v3_meanflow'` — copy `'plan_fm_v3_imeanflow'` (line 833) and change `prefix`, `diffusion`, `diffusion_loadpath`, `exp_name`. 🔴 **`diffusion_loadpath` must reproduce `args_to_watch_fmv3_mf_train` exactly, token for token**, or the eval silently fails to find the checkpoint:
```python
'prefix': 'f:plans/flow_matching_v3_meanflow/' +
          'H{horizon}_D{diffusion}_aw{action_weight}_obj{mf_objective}_bb{imf_backbone}_ts{t_schedule}_dp{meanflow_data_proportion}/',
'diffusion_loadpath': 'f:flow_matching_v3_meanflow/' +
          'H{horizon}_D{diffusion}_aw{action_weight}_obj{mf_objective}_bb{imf_backbone}_ts{t_schedule}_dp{meanflow_data_proportion}',
```
Every architecture key in the plan block **must equal** the training block (`dual_head`, `interval_cfg`, `imf_backbone`, `dit_*`) or the `state_dict` load fails. Keep `eval_use_ema: True` (few-step MeanFlow is EMA-sensitive; official samples with EMA). Sweep `flow_steps_v3` ∈ {1, 2, 5, 10}.

---

## 5. SLURM — copy-modify `sbatch/iMF/`, write nothing from scratch

```bash
cp -r Slurm_Codes/sbatch/iMF Slurm_Codes/sbatch/MeanFlow
cd Slurm_Codes/sbatch/MeanFlow
mv train_imf.sh train_meanflow.sh
mv eval_imf.sh  eval_meanflow.sh
mv load_results_imf.sh load_results_meanflow.sh
mv imf_pipeline.sh meanflow_pipeline.sh
```

**Edit only these lines** — the whole prologue (conda activate, `PYTHONPATH`, `MUJOCO_GL=egl`, the **GPU/EGL isolation guard**, W&B key file, `trap on_exit`, `latest.log` symlink) is battle-tested. 🔴 **Never touch the `MUJOCO_EGL_DEVICE_ID` / `GPU-LEAK` abort block.**

| file | line | from | to |
|---|---|---|---|
| `train_meanflow.sh` | `#SBATCH --job-name=` | `imf_train` | `mf_train` |
| | python invocation | `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` | `FM_v3_meanflow_test/train_flow_matching_v3_meanflow.py` |
| | `--wandb-project` | `FMPCC-iMF` | `FMPCC-MeanFlow` |
| `eval_meanflow.sh` | job-name | `imf_eval` | `mf_eval` |
| | python invocation | `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` |
| `load_results_meanflow.sh` | same pattern | | |
| `meanflow_pipeline.sh` | job-name, `SBATCH_DIR`, both script names | `Slurm_Codes/sbatch/iMF` | `Slurm_Codes/sbatch/MeanFlow` |

**`--time`:** keep `24:00:00`. Rule of thumb in this repo is 2× expected with a 24 h cap; Gen3v4's 100k-step DiT run fits well inside it.

**Submission (unchanged mechanism):**
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/meanflow_pipeline.sh   # train -> eval, afterok
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh      # train only
```
🔴 **No `tqdm`/live progress bars in batch logs** (repo rule). The inherited trainer already sets `mininterval=1e10`; keep it.

---

## 6. Gates — run in order, do not skip

| gate | what | pass criterion | cost |
|---|---|---|---|
| **G0** | import + shapes: build the model from the new config, one forward `(x, τ, h) → (u, v)` | correct shapes, no stale `flow_matcher_v3_imeanflow` import anywhere | seconds |
| **G1** | `h → 0` degeneracy | with `h=1e-4`, `‖u − v‖/‖v‖ < 0.05` | seconds |
| **G2** | ⭐ **objective parity vs `aux_repo/MeanFlow`** | wrap the **same** torch net in our `_p_losses_meanflow` and in `aux_repo/MeanFlow`'s `MeanFlow.loss(mode='meanflow')`; feed identical `(x, t, r, e)`; compare `u_pred`, `du`, and the raw MSE to `<1e-4` rel. **Expect a controlled mismatch on the tangent only** (they use predicted `v_c`, we use analytic `v`) — patch their `v_c` to the analytic `v` for the comparison and the numbers must then agree | ~half a day |
| **G3** | 200-step smoke train on 1 seed | `raw_mse_u` falls monotonically from its step-0 value; no NaN; `h_mse_b*` all finite | ~10 min |
| **G4** | full train, 1 seed, then eval at `flow_steps_v3 ∈ {1,2,5,10}` | see §7 | 1 job |

**G2 is the gate that has never existed in this project.** Every prior verification was internal self-consistency. Do not skip it — it is why the Gen3v4/Gen13 sign question stayed open for a month.

---

## 7. Success / kill criteria — pre-register these BEFORE looking at results

The bar is **not** "better than Gen3v4-iMF". It is FM, and specifically the config that `Gen13/fix_7/RESULTS_Gen13_fix7.3_VERDICT_imf_refuted.md` §2 identified: **FM @ K=2 → 100 % safe, 0.1894 s/plan.** Note that is a *HardFlow* number; the Gen3v4-lineage FM comparator is FMv3ODE on the same avoiding task — use whichever the DA aggregation already carries, and **state which one in the results MD**.

| outcome | reading |
|---|---|
| MeanFlow ≈ iMF at matched NFE | the predicted-`v` tangent buys nothing here ⇒ **iMF's headline contribution does not transfer to constrained control.** A clean, publishable negative. |
| MeanFlow < iMF | the tangent *does* help ⇒ Gen3v4's direction was right and the failure is data/conditioning, not the objective |
| MeanFlow > iMF | 🔴 the predicted-`v` tangent is actively harmful at this data scale — re-read the audit §2.4 (`v_c` evaluated at the wrong `h`) before believing it |
| both ≪ FM at every matched K | expected from fix_7.3; the h-stratified metric then tells you *where* the field fails |

⚠️ **Matched-budget or nothing.** fix_7.3 §9: the whole Gen13 claim collapsed because one hard-coded `k_steps=10` made the decisive control unrunnable. **Every MeanFlow-vs-X table must be at equal K.** Never compare MeanFlow@K=5 against FM@K=10.

**Kill criterion:** if `h_mse_b3` (`h ∈ [0.6,1.0]`) is flat at its step-0 value while `h_mse_b0` (`h=0`) drops 10×, the field is untrained exactly where 1–2 NFE lives. Report it and **stop** — that is the `COMPARE §7.3` hypothesis confirmed, and no amount of extra training on this schedule will fix it. It is also the precise motivation for Gen3v7.

---

## 8. MASTER_TEST_HISTORY row — **prepared, not applied**

Per repo convention the implementing agent must **not** self-edit `MASTER_TEST_HISTORY.md`. Hand this to the user:

```markdown
| **Gen3v6 (MeanFlow Baseline)** | [flow_matcher_v3_meanflow/](../flow_matcher_v3_meanflow) | [FM_v3_meanflow_test/](../FM_v3_meanflow_test) | July 2026 | **MeanFlow (2505.13447) faithful baseline** — Gen3v4 sibling with the ANALYTIC-v JVP tangent (vs iMF's predicted v_c), official adaptive loss (p=1, eps=0.01, sum), two independent logit-normals, no CFG. Isolates iMF's headline contribution as a controlled A/B. Adds the h-stratified residual metric and a real gradient clip. | working on |
```
Insert directly after the `Gen3v5 (BNS Solver)` row.

---

## 9. Hand-off checklist

- [ ] `flow_matcher_v3_meanflow/` + `FM_v3_meanflow_test/` copied; `__pycache__` purged; **zero** `flow_matcher_v3_imeanflow` references remain
- [ ] test scripts renamed; `*_ode_selectable.py` leftovers deleted; experiment keys switched
- [ ] `imf_official` / `fm_equivalent` arms and all CFG machinery deleted
- [ ] `_p_losses_meanflow` rewritten with FIX-1..FIX-4
- [ ] 🔴 JVP z-tangent is still **analytic `v_inst`**, with the do-not-change comment
- [ ] `interval_cfg=False`, `dit_condition_on_t=False`, `dual_head=True`
- [ ] `h_mse_b0..b3` + `raw_mse_u/v` + `per_dim_rms_u` logged, plumbed to W&B
- [ ] 🔴 `gradient_clip` actually applied in `utils/training.py` (it is a dead key upstream)
- [ ] 2 config blocks + `args_to_watch_fmv3_mf_train`; `dp` in the folder name; `diffusion_loadpath` matches token-for-token
- [ ] `Slurm_Codes/sbatch/MeanFlow/` copied from `iMF/`, 4 files, only the listed lines changed, EGL guard untouched
- [ ] G0–G3 pass locally/on a smoke job before the full run
- [ ] changelog MD written to `logs_in_develop/Gen3v6_MeanFlow/<epoch>/`
- [ ] MASTER row handed to the user, **not** self-applied
- [ ] nothing committed without explicit permission

---

## 10. Known traps (each one has already bitten this repo)

1. **Folder-name collision → silent checkpoint overwrite.** `POST_U10_II` §1.1. Any swept knob must be in `args_to_watch`.
2. **Reading the adaptive `loss` as convergence.** It is pinned at its ceiling by construction. `COMPARE §1`.
3. **`+p_mean` vs `−p_mean` on the τ axis.** `imf_diffusion.py:666` flags it. Getting it wrong puts all the mass near noise and looks *almost* fine.
4. **Window-level `train_test_split` leaks.** `POST_U10_III` §4.2 — at H=8 adjacent windows share 7 of 8 frames, so `loss_test` is effectively a train loss. Gen3v6 inherits `train_test_split: 0.9` and therefore **inherits the leak**. Either implement an **episode-level** split (hold out ~19 of 96 episodes, build indices only from the remaining 77) or **label every val number in the results MD as leaking**. Do not quietly present it as generalisation.
5. **`gradient_clip` is a dead key.** §4.2.
6. **Plan-block architecture keys drifting from the train block** ⇒ `state_dict` mismatch at eval.
7. **`dit_condition_on_t`** — official conditions on `h` only. Gen13's U-Net does *not*, and the audit flags it (§2.3). Keep `False` here so Gen3v6 stays comparable to Gen3v4.
