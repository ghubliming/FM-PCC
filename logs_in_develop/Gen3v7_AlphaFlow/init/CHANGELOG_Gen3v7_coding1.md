# CHANGELOG — Gen3v7 (α-Flow), coding pass 1

**Date:** 2026-07-23 · **Type:** implementation changelog · **Status:** code complete, **NOTHING RUN**
**Implements:** [`PLAN_Gen3v7_alphaflow.md`](PLAN_Gen3v7_alphaflow.md)
**Base:** copy-modify of Gen3v6 `flow_matcher_v3_meanflow/` ↔ `FM_v3_meanflow_test/` (the plan's ⭐ preferred base — Gen3v6 shipped, so Gen3v4 was not used)
**Upstream:** `/workspaces/aux_repo/alphaflow` @ `b0fef77` — snap-research, official PyTorch, arXiv 2510.20771
**Nothing committed.** No local execution — this container has no Python deps; all gates run on the cluster.

---

## 0. TL;DR

Gen3v7 implements **α-Flow**: MeanFlow's JVP target is replaced by a **self-bootstrapped,
no-grad, derivative-free** target

```
u_tgt = (dt·v + (h−dt)·u_next) / h  ==  α·v + (1−α)·u_next ,     dt = α·h
u_next = u(z_r + dt·v, r+dt, h−dt)      ← under torch.no_grad()
```

with **α annealed 1 → 0** on a sigmoid, making training a genuine homotopy from plain flow
matching (α=1, `u_tgt = v` exactly) to Gen3v6 MeanFlow (α=0, the JVP branch verbatim).
It targets COMPARE §8.2's *blind direction*: the MeanFlow residual only sees `δ_u − h·δ_D`,
so any error with `δ_u = h·δ_D` is invisible to the loss while the sampler uses `u` alone.
A fixed target has no blind direction.

Beyond the objective, this pass lands three things the plan called out as load-bearing:

1. **`set_train_step` plumbing** — the loss finally sees the global optimizer step (§4).
2. **The α-schedule cannot silently die** — a hard constructor assert, a pre-flight banner
   printed before step 0, three W&B curves and five W&B summary keys (§3.4, §4, §7).
3. **The K-grid is built into the eval path**, not left as a config edit (§8). fix_7.3 §9's
   post-mortem is that one hard-coded `k_steps=10` killed an entire generation's claim.

**Sampling is untouched.** α is training-only; the sampler is byte-equivalent to Gen3v4/Gen3v6,
which is exactly what makes the three-way matched-K comparison clean.

---

## 1. Files created

| path | origin | note |
|---|---|---|
| `flow_matcher_v3_alphaflow/` | copy of `flow_matcher_v3_meanflow/` | model package; `__pycache__` purged |
| `FM_v3_alphaflow_test/` | copy of `FM_v3_meanflow_test/` | test/eval package |
| `diffuser/flow_matcher_v3_alphaflow/` | copy of `diffuser/flow_matcher_v3_meanflow/` | ⭐ **the 5th location** — see §6 |
| `Slurm_Codes/sbatch/AlphaFlow/` | copy of `Slurm_Codes/sbatch/MeanFlow/` | 4 sbatch files |
| `FM_v3_alphaflow_test/gates_alphaflow.py` | **rewritten** | G0–G5 + smoke harness (§7) |
| `FM_v3_alphaflow_test/endpoint_error_alphaflow.py` | **new** | ⭐ PLAN §8.2's decisive metric (§8) |
| this file | **new** | changelog |

**Untouched (hard rule, verified):** `flow_matcher_v3_meanflow/`, `FM_v3_meanflow_test/`,
`flow_matcher_v3_imeanflow/`, `FM_v3_imeanflow_test/`, `diffuser/` (except the new sibling
folder), `d3il/`, every pre-existing sbatch dir. The only pre-existing file modified is
`config/avoiding-d3il.py` — **`git diff --stat` reports 229 insertions, 0 deletions.**

## 2. Renames inside the new folders

Module path `flow_matcher_v3_meanflow` → `flow_matcher_v3_alphaflow` everywhere.
**Verified: zero stale module/class references remain** — the only surviving `meanflow`
mentions are (a) deliberate prose pointing at the *sibling* generation, (b) `gates_alphaflow.py`
importing Gen3v6 on purpose for gate G2.

| old (Gen3v6) | new (Gen3v7) |
|---|---|
| `mf_diffusion.py` / `MeanFlowODE` | `af_diffusion.py` / `AlphaFlowODE` |
| `mf_engine.py` / `MeanFlowEngine` | `af_engine.py` / `AlphaFlowEngine` |
| `mf_trajectory_model.py` / `MFTrajectoryModel` | `af_trajectory_model.py` / `AFTrajectoryModel` |
| `mf_dit_trajectory.py` / `MFDiTTrajectory` | `af_dit_trajectory.py` / `AFDiTTrajectory` |
| `train_/eval_/load_results_..._v3_meanflow.py` | `..._v3_alphaflow.py` |

Experiment keys: `flow_matching_v3_meanflow` → `flow_matching_v3_alphaflow` (train),
`plan_fm_v3_meanflow` → `plan_fm_v3_alphaflow` (eval **and** load_results).
W&B project `FMPCC-MeanFlow` → `FMPCC-AlphaFlow`.

**No deletions this pass.** Gen3v6 already pruned the Gen3v2/Gen3v4 leftovers; nothing new
went dead.

---

## 3. The objective — `flow_matcher_v3_alphaflow/models/af_diffusion.py`

### 3.1 Convention mapping (PLAN §2) — done once, in the module docstring

α-Flow is NOISE-AT-1 and calls the query point `t`; we are DATA-AT-1 and call it `r`. The
mapping `τ = 1 − t` is stated in the file header and **only our names appear in the code**.
The `r`/`t` role swap (their `t` is our `r`) is flagged there as the most likely bug in this
generation.

| α-Flow | ours | in code |
|---|---|---|
| `x_t`, `t`, `t_next` | `z_r`, `r`, `t` | `x_r`, `r`, `t` |
| `t − t_next` | `h` | `h` |
| `dt = α·(t−t_next)` | `dt = α·h` | `dt` |
| `x_t − dt·v` | `z_r + dt·v` | `z_shift` (sign flips twice ⇒ `+`) |
| `net(·, sigma_next=t_next, sigma=t−dt)` | `model(z_shift, τ=r+dt, h=h−dt)` | ✅ |

### 3.2 `compute_u_target()` — factored out on purpose

The target lives in its own method, not inline in the loss, because **G1/G2/G3/G5 are all
statements about this one function** — a gate that re-implemented it would prove nothing.
It always returns a **detached** tensor.

Routing (PLAN §3.1). Note that the three branches are never all live at once: for a given
step, `h == 0` ⇒ FM anchor, and every `h > 0` sample goes to *either* the discrete branch
(α>0) *or* the JVP branch (α==0), never both.

| branch | when | target |
|---|---|---|
| FM anchor | `h == 0` (`r` forced to `t`, prob `af_ratio_fm`) | `u_tgt = v` |
| discrete (bootstrapped) | `h > 0` and `α > 0` | `α·v + (1−α)·u_next`, clamped to ±`af_clamp_utgt` |
| continuous (JVP) | `h > 0` and `α == 0` | `v + h·du/dr` — **Gen3v6's body, unmodified** |

- **α = 1 short-circuits** to `u_tgt = v.clone()` with no forward pass at all. Required for
  G1: `(h·v + 0·u_next)/h` is only `v` up to float round-off, and the gate demands bitwise.
  Upstream short-circuits the same way (`isclose(1 − dt/(t−t_next), 0)`, loss.py:536).
- **`u_next` is under `torch.no_grad()`** — this is the entire point of the generation (G5).
- The clamp is applied **only** to the discrete target, matching upstream (loss.py:542); FM
  anchors and JVP targets are left alone.

### 3.3 The α schedule

`_get_ratio` is a line-by-line port of `AlphaFlowLoss.get_ratio` (loss.py:390-427) —
sigmoid/linear/exponential/log/constant/step, plus the `clamp_value` snap. It is a
`@staticmethod` so the train script and the gates can evaluate the curve without a model.

The **`clamp_value: 0.005` snap is kept** and matters: without it α becomes a tiny-but-nonzero
number, every sample takes the discrete branch with `dt ≈ 0`, and `u_next` is evaluated
essentially at the query point — a degenerate near-identity target. The snap routes those
samples to the exact JVP branch instead.

Verified numerically (pure-Python, the real `_get_ratio` extracted via `ast`), on
`sigmoid 1.0→0.0, [0, 100000], γ=25, clamp=0.005`:

```
step        0   10000   20000   29000   40000   50000   60000   70000   72000  100000
alpha   1.000   1.000   1.000  0.9948  0.9241  0.5000  0.0759  0.0067   0.000   0.000
```

- **α ≡ 1.0 (pure FM) for steps < 28 900**, transition ≈ 29k→71k, **α ≡ 0.0 (MeanFlow) from 71 200**.
- 🔴 **Trap check reproduced:** with upstream's `change_end_steps: 400000` copied verbatim,
  α at steps 0 / 50k / 100k is `1.0 / 1.0 / 1.0` — *plain flow matching for the entire budget,
  wearing an α-Flow folder name.* This is PLAN §11 trap 1 and it is now **impossible to hit
  silently** (§3.4).

### 3.4 Three independent guards against a dead schedule

1. **Constructor assert.** `AlphaFlowODE.__init__` raises `ValueError` if
   `af_alpha_end_step != af_n_train_steps` (skipped for `constant`/`step` schedulers, and
   bypassable with `af_n_train_steps=None` for gates that force a constant α on purpose).
   The train script passes `args.n_train_steps`, so the assert is armed in every real run.
2. **Pre-flight banner.** `print_alpha_schedule()` prints the whole α curve at 11 points
   *before the first optimizer step*, with an explicit `*** WARNING: alpha is CONSTANT ***`
   if it never moves.
3. **Telemetry.** `alpha`, `discrete_frac`, `clamp_frac` are logged every `log_freq`, pushed
   to W&B as `train/alpha` etc., and reduced to run summaries `alpha_first/last/min/max/
   alpha_schedule_alive`.

### 3.5 Loss and weighting

```python
err_u = (u_pred - u_target).pow(2).sum(dim=(1, 2))         # per-sample SUM
w_br  = where(discrete_mask, alpha, 1.0)                   # upstream weight_d = α
loss  = (w_br * adp(err_u) + adp(err_v)).mean()            # adp(e) = e / sg(e + af_adp_eps)
```

`af_adp_eps = 1e-3` (α-Flow's value), with a boxed comment saying it is **deliberately ≠**
MeanFlow/iMF's `0.01`. The v-head loss is Gen3v6's, unweighted, at the same query point.

⚠️ **Stated fidelity gap (§9.1):** upstream reduces the squared error with `.flatten(1).mean(1)`
(MEAN); we use SUM, inherited from Gen3v6/official-imeanflow. That is an `H·D = 8·6 = 48×`
rescale, so upstream's `eps=1e-3` sits at a different point relative to the error scale than
ours does. PLAN §3.5 specifies SUM **and** 1e-3, and SUM is what keeps the Gen3v6 A/B
controlled, so that is what was implemented — with the discrepancy written into the
`_adaptive` docstring, including how to match upstream's balance (MEAN, or `eps = 0.048`) if
anyone ever wants to. Practical consequence: `err ≫ eps`, so the adaptive weight is ≈1 and
this term is near-inert. **Read `raw_mse_u`, never `diffusion_loss`.**

### 3.6 Prediction: one forward, both heads

Gen3v6 took `u_pred` from the JVP primal and paid a *second* forward for `v_pred`. α-Flow
computes the target first, then does **one** forward for both heads at the same query point
— upstream's structure (loss.py:579), and one forward *cheaper*. This is numerically
identical to reusing a primal because the DiT arm has **no stochastic dropout** (its
`dropout_rate` only feeds the deterministic null-class CFG token, and `force_dropout=False`
here) — verified by reading `af_dit_trajectory.py`. That determinism is what makes gate G2's
comparison against Gen3v6 exact rather than approximate.

### 3.7 Sampler — untouched

`p_sample_loop`'s interval-jump update (`x += dt·u`, `h = dt = 1/N`) and the U8 torchdiffeq
"homing-missile" branch are byte-equivalent to Gen3v6. ✅ **α never appears at inference.**

---

## 4. `flow_matcher_v3_alphaflow/utils/training.py`

- 🔴 **`set_train_step` plumbing (PLAN §3.7).** `train_epoch` calls
  `self.model.set_train_step(self.step)` **outside** the accumulation loop, immediately
  before the loss. Two things this gets right on purpose:
  - it uses **`self.step`, the optimizer-step counter** — the inner `for step in range(...)`
    is the accumulation loop, and with `gradient_accumulate_every=2` using it would halve
    the effective schedule length;
  - placing it outside the inner loop means every micro-batch of one optimizer step sees the
    *same* α.
- **Resume is safe by construction.** `load()` restores `self.step`, and `train_epoch` pushes
  it on every iteration, so α picks up where it left off instead of restarting at 1.0 and
  making the model unlearn (PLAN §11 trap 6). *(Note: this lineage's train script never calls
  `trainer.load()` — there is no resume path wired at all. The plumbing is correct if one is
  added.)*
- `EXTRA_METRIC_KEYS` gains `alpha`, `discrete_frac`, `clamp_frac` — persisted to
  `losses.pkl`/`losses.json`, restored on load, and pushed to W&B.
- `gradient_clip` is applied before `optimizer.step()` and the pre-clip norm is logged —
  inherited from Gen3v6, unchanged. It matters *more* here: the discrete branch has no JVP
  and should be calmer, so a surviving spike is diagnostic rather than background noise.

## 5. `config/avoiding-d3il.py` — additive only (229 insertions, 0 deletions)

- **`args_to_watch_fmv3_af_train`** — `H`, `D`, `aw`, `bb`, `ts`, **`ai`/`ae`/`ag`/`rf`**.
  Every α knob is in the folder name deliberately (POST_U10_II §1.1: four un-watched knobs
  once let two Gen3v4 runs write a byte-identical directory and clobber each other).
- **`flow_matching_v3_alphaflow`** (train) — the α schedule (`sigmoid`, 1.0→0.0, `[0,100000]`,
  γ=25, clamp=0.005), `af_ratio_fm=0.5`, `af_clamp_utgt=4.0`, `af_adp_eps=1e-3`,
  `dual_head=True`, `interval_cfg=False`, `imf_backbone='dit'`, `dit_condition_on_t=False`,
  `p_mean=-0.4`, `p_std=1.0`, `t_schedule='logit_normal'`, 100k steps, bs 32, accum 2,
  lr 5e-4, `gradient_clip=1.0`, ema 0.995, `train_test_split=0.9`, `condition_guidance_w=0.0`.
- **`plan_fm_v3_alphaflow`** (eval) — every architecture key mirrors the train block;
  `flow_steps_v3=2` (K grid is swept from the CLI, §8); `eval_use_ema=True`.

**Verified mechanically** (stubbing `watch`/`yaml`, no torch needed) that the plan block's
`diffusion_loadpath` renders **token-for-token identical** to the train block's `exp_name`:

```
flow_matching_v3_alphaflow/H8_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_aw10_bbdit_tslogit_normal_ai1.0_ae0.0_ag25.0_rf0.5
```

The same check run against the **Gen3v4 and Gen3v6** pairs also matches, confirming the
method and not just the result. An architecture-key parity diff between the two Gen3v7 blocks
reports **no mismatches**, and no `af_*` key is missing from the plan block.

## 6. `diffuser/flow_matcher_v3_alphaflow/` — the 5th location, done up front

Gen3v6 fix_1 §8 asked for exactly this: `diffuser/utils/config.py:import_class()`
hard-prefixes every config class string with `diffuser`, so the **train** path (and only the
train path — the package's own `import_class` has a `startswith` guard the shared one lacks)
resolves `flow_matcher_v3_alphaflow.models.AlphaFlowODE` as
`diffuser.flow_matcher_v3_alphaflow.models.AlphaFlowODE`. Four shim files re-export the real
classes, so the resolved class keeps its true `__module__` and the eval-time
"pickled class does not match" check stays a no-op.

**Gen3v6 lost a job to this.** The checklist is five locations, not four:

| # | location | Gen3v7 |
|---|---|---|
| 1 | `flow_matcher_v3_alphaflow/` | ✅ |
| 2 | `FM_v3_alphaflow_test/` | ✅ |
| 3 | `config/avoiding-d3il.py` (2 blocks + args_to_watch) | ✅ |
| 4 | `Slurm_Codes/sbatch/AlphaFlow/` | ✅ |
| 5 | **`diffuser/flow_matcher_v3_alphaflow/` shim** | ✅ **this pass, not a follow-up fix** |

## 7. `FM_v3_alphaflow_test/gates_alphaflow.py` — G0–G5, rewritten

| gate | what it checks |
|---|---|
| **G0** | build + one forward; **every Gen3v7 module resolves to a file inside the Gen3v7 folder**; and the `af_alpha_end_step != n_train_steps` assert actually fires |
| **G1** | α=1, `ratio_fm=0` ⇒ `torch.equal(u_tgt, v)` — **bitwise** |
| **G2** | α=0 ⇒ target matches **Gen3v6**'s `_p_losses_meanflow` target (<1e-5) **and** the scalar losses match once eps is aligned at 0.01 |
| **G3a** | discrete branch is a first-order-consistent discretisation of the JVP branch |
| **G3b** | the plan's literal wording (see §9.2) |
| **G4** | α moves across the budget, α(0)=1, α(end)=0, and `discrete_frac` shifts with it |
| **G5** | `u_tgt.requires_grad is False` **and** backprop from `u_tgt` reaches zero parameters |
| **G3'** | 200 optimizer steps walking across the middle of the anneal (so both branches run): `raw_mse_u` falls, nothing NaN/Inf |

⭐ **G2 is scriptable here, unlike Gen3v6's.** Gen3v6's G2 needed `aux_repo/MeanFlow`, which
is not on the cluster, so it was never run. Gen3v7's comparator is `flow_matcher_v3_meanflow`
— a real, importable sibling in this repo. It loads Gen3v6 with α-Flow's own weights
(`mf.load_state_dict(af.state_dict())`; the backbones are the same code under different
names) and reseeds both models identically, which works because **the RNG consumption order
is preserved**: `_sample_tau_pair` → `rand(fm_mask)` → `randn_like(noise)`. That contract is
written into the loss as a comment.

```bash
python FM_v3_alphaflow_test/gates_alphaflow.py                    # G0–G5 + smoke, dataset-free
python FM_v3_alphaflow_test/gates_alphaflow.py --ckpt .../state_best.pt   # makes G3b binding
# the real G3 smoke on the actual dataset needs no new code:
python FM_v3_alphaflow_test/train_flow_matching_v3_alphaflow.py --seed 6 --n_train_steps 200 --n_steps_per_epoch 100
```

## 8. Matched budget — built in, not aspirational

PLAN §8: *"Build the K-grid into the eval config from day one."* Implemented as a real
mechanism rather than a note:

- `eval_flow_matching_v3_alphaflow.py --flow-steps K` patches
  `config.avoiding-d3il.base['plan_fm_v3_alphaflow']['flow_steps_v3']` **before any Parser
  reads it**. This is the intended data path, not a monkey-patch: `Parser.read_config` does
  `importlib.import_module(args.config)` and copies the block key by key, and Python caches
  modules — so `exp_name`, `savepath` and the diffusion kwargs all follow automatically.
  Because `flow_steps_v3` is watched as `K` in `args_to_watch_fmv3_ode_plan`, **each K writes
  its own results directory** and no two budgets can overwrite each other.
  *(Needed because `Parser.add_extras` — the generic `--key value` override — is commented
  out in this lineage, so plain CLI overrides do nothing.)*
- `load_results_flow_matching_v3_alphaflow.py --flow-steps K` mirrors it, so aggregation
  reads the K that was actually evaluated instead of the K=2 default.
- `sbatch/AlphaFlow/eval_alphaflow.sh` and `load_results_alphaflow.sh` loop
  `FLOW_STEPS_GRID="${AF_FLOW_STEPS:-1 2 5 10}"`.

### 8.1 ⭐ `FM_v3_alphaflow_test/endpoint_error_alphaflow.py` — the decisive metric

PLAN §8 diagnostic 2 ("the single most decisive number in the generation"). Two
complementary measurements, both per K ∈ {1,2,5,10}:

- **T-A — interval endpoint error on the data coupling.** At every `(τ, h)` the K-step
  sampler queries, compare `z_τ + h·u(z_τ,τ,h)` against the true `z_{τ+h}`, alongside the
  **same model's Euler shot** `z_τ + h·v` as the intra-model control (u beating v *is* the
  claim). ⚠️ The floor is not zero — the best possible field predicts `E[z_{τ+h}|z_τ]`, not
  the paired sample — but the floor is a property of the DATA and is identical for Gen3v4 /
  Gen3v6 / Gen3v7 / FM, so the numbers are comparable across models. This is stated in the
  docstring so nobody reports it as an absolute accuracy.
- **T-B — terminal prediction error, self-consistency.** Exactly fix_7.3 §4's metric
  (`‖x̂₁(k) − x_final‖` along the model's own K-step rollout), so the output is directly
  comparable to iMF's `0.1539 / 0.1538 / 0.1595 / 0.1572`. The script computes the relative
  spread across K and prints the verdict: **flat in K ⇒ fixed field error ⇒ the objective
  change did not help**, which is the pre-registered kill reading.
  The `err(τ=1) = 0` metric caveat from fix_7.3 is printed too.

It takes `--loadpath`, so it runs unchanged against **Gen3v4 and Gen3v6 checkpoints** for the
cross-generation table (their `MeanFlowODE`/`iMeanFlowODE` expose the same `_predict_uv` /
`q_sample` surface). Writes `endpoint_error.json` next to the checkpoint.

## 9. Deviations from the plan, stated

1. **SUM vs MEAN reduction in the adaptive loss** — followed the plan (SUM + eps=1e-3), which
   differs from upstream's MEAN + 1e-3 by an `H·D = 48×` rescale of the error relative to eps.
   Documented in `_adaptive.__doc__` with the two ways to match upstream instead. See §3.5.
2. **G3 split into G3a/G3b.** The plan's literal G3 —
   `‖u_tgt(α) − u_tgt(α→0 JVP)‖/‖u_tgt‖ < 0.1` — expands to
   `(1−α)(u − v − h·D_tot) + O(dt)`, i.e. **the model's own MeanFlow residual**, which is
   O(1) at random initialisation and only small once the field sits near its fixed point. As
   written it therefore cannot pass pre-training. Implemented as: **G3a**, a
   training-independent check that the discrete branch is a first-order consistent
   discretisation of the continuous one (`u_tgt(α) ≈ α·v + (1−α)(u + α·h·D_tot)`, with
   `D_tot = (u_tgt(0) − v)/h` taken from the JVP branch itself) — this *is* what §3.4 proves
   and it binds at random init; plus **G3b**, the plan's literal form, reported as INFO
   without `--ckpt` and binding with one. Same treatment Gen3v6 gave its h→0 degeneracy gate.
3. **The α=0 JVP is NOT wrapped in `torch.no_grad()`** — the result is `.detach()`ed instead.
   Forward-mode AD shares the GradMode guard with reverse mode in several torch versions, so
   a `jvp` inside `no_grad` can return null/zero tangents, which would silently degrade the
   branch to `u_tgt = v` (plain FM) and break G2 in a way that looks like a science result.
   Gen3v6 computes the JVP with grad enabled and detaches the result; that is the path this
   lineage has actually run, so it is the path kept. **`u_next` in the discrete branch IS
   under `no_grad`** — that is ordinary reverse-mode and is the one that matters for G5.
4. **No batch subsetting.** Upstream splits the batch with boolean masks and runs each branch
   on its own sub-batch. We run the full batch and select with `torch.where`, so the `h == 0`
   FM anchors ride along in the `u_next` forward and are discarded. Cost: ≤ `af_ratio_fm` of
   one extra small-DiT forward per step. Benefit: no need to index `cond` (a dict of
   per-timestep tensors) by mask — which is precisely where a subtle conditioning bug would
   live. Both queries stay in-domain (`r+dt ≤ t ≤ 1`, `h−dt ≥ 0`).
5. **`.clone()` in the α=1 short-circuit.** `apply_conditioning` writes **in place**
   (`models/helpers.py:161`). Aliasing `v_inst` would mutate the caller's `v` and would make
   G1's `torch.equal(u_tgt, v)` tautological (comparing an object with itself).
6. **`af_ratio_fm` is a constant, not a schedule.** Upstream supports scheduling it but ships
   it constant in every published config; the plan says start at 0.5. Keeping it constant
   also keeps the knob count (and the folder-name length) down.
7. **The sbatch trains 5 seeds, not the plan's 1.** `config/projection_eval.yaml` fixes
   `seeds: [6,7,8,9,10]` for *every* generation, so a 1-seed train would leave eval unable to
   find 4 of its 5 checkpoints. Gen3v6 does the same. For a genuine 1-seed G6, run
   `train_...py --seed 6` and set the yaml's `seeds` to `[6]`.
8. **`clamp_frac` is measured over discrete rows only** — counting the discarded `h==0` rows
   would dilute it by `af_ratio_fm` and make the diagnostic read low for the wrong reason.

## 10. Inherited traps NOT fixed (must be stated in the results MD)

- ⚠️ **Window-level `train_test_split=0.9` leaks.** POST_U10_III §4.2: at H=8 adjacent windows
  share 7 of 8 frames, so `loss_test` is effectively a train loss. Gen3v7 **inherits the leak**
  (flagged in a comment in the config block, and in the endpoint-error script's output).
  Either implement an episode-level split (~19 of 96 episodes held out), or label every val
  number as leaking. Do not present it as generalisation.
- ⚠️ **Never read `diffusion_loss` / `train/loss` as convergence** — the adaptive loss is
  pinned at its ceiling by construction. Read `raw_mse_u`.
- ⚠️ **Matched budget or nothing** — every α-Flow-vs-X table must be at equal K.
  The comparator is **FM @ K=2 → 100 % safe, 0.1894 s/plan** (fix_7.3 §2).
- ⚠️ **DiT checkpoints across Gen3v4/Gen3v6/Gen3v7 are *shape*-compatible but semantically
  incompatible** (Gen3v6 §9.1). The folder separation is what keeps them apart, not a load
  error. This is why every α knob is in `args_to_watch`.

## 11. Status vs the plan's hand-off checklist (§10)

- [x] Based on **Gen3v6**; folders copied; `__pycache__` purged; zero stale module refs
- [x] Three-way batch routing implemented, with `alpha` / `discrete_frac` / `clamp_frac` logged
- [x] §3.3 discrete target in **our** convention, `u_next` under `no_grad`, clamped to `af_clamp_utgt`
- [x] α=1 short-circuits the `u_next` forward (and is bitwise-exact)
- [x] Continuous branch is Gen3v6's JVP loss, unmodified
- [x] `get_ratio` sigmoid + `clamp_value` snap ported; 🔴 `af_alpha_end_step == n_train_steps`
      enforced by a constructor assert **and** verified numerically (§3.3)
- [x] `set_train_step` plumbed from `utils/training.py`; uses the **optimizer** step;
      resume-correct; `alpha` logged to pkl/json/W&B + summaries
- [x] `af_adp_eps = 1e-3`, commented as deliberately ≠ MeanFlow's 0.01
- [x] 🔴 `gradient_clip` actually applied in `utils/training.py` (inherited from Gen3v6)
- [x] 2 config blocks + `args_to_watch_fmv3_af_train` with every α knob watched;
      `diffusion_loadpath` verified token-for-token
- [x] `Slurm_Codes/sbatch/AlphaFlow/` copied from `MeanFlow/`; **normalized diff confirms only
      the listed lines changed** (job names, python paths, W&B project, `SBATCH_DIR`, script
      names, plus the K loop in eval/load_results) — conda prologue, `PYTHONPATH`,
      `MUJOCO_GL=egl`, the **GPU/EGL isolation guard**, the W&B key file, `trap on_exit` and
      the `latest.log` symlink are byte-identical; `--time` stays `24:00:00`; no live progress bars
- [x] `diffuser/` shim created **up front** (Gen3v6 fix_1's lesson)
- [x] Endpoint-error-at-sampler-grid diagnostic implemented (§8.1)
- [ ] **G0–G5 not run** — no Python in this container. Run `gates_alphaflow.py` on i6-gpu-1 first.
- [x] changelog written (this file)
- [ ] **MASTER_TEST_HISTORY row handed to the user, not self-applied** — see §12
- [x] nothing committed

**What WAS verified locally** (no torch needed): the α schedule numerically, including the
400k trap (§3.3); `exp_name` ⇄ `diffusion_loadpath` token-for-token for Gen3v4/Gen3v6/Gen3v7
plus architecture-key parity; `py_compile` on every new/edited Python file; `bash -n` on all
four sbatch scripts; a normalized diff of the sbatch dir against `MeanFlow/`; and a full
grep sweep for stale module/class references.

## 12. MASTER_TEST_HISTORY row — prepared, NOT applied

Per repo convention this was **not** written into the master index. Insert directly after the
`Gen3v6 (MeanFlow Baseline)` row:

```markdown
| **Gen3v7 (α-Flow)** | [flow_matcher_v3_alphaflow/](../flow_matcher_v3_alphaflow) | [FM_v3_alphaflow_test/](../FM_v3_alphaflow_test) | July 2026 | **α-Flow (arXiv 2510.20771, snap-research)** — replaces the MeanFlow JVP target with a self-bootstrapped no-grad target `u_tgt = α·v + (1−α)·u_next`, with α annealed 1→0 (sigmoid) so training is a homotopy from flow matching to MeanFlow. Targets COMPARE §8.2's blind direction, the diagnosed cause of the Gen13 iMF refutation. Adds target clamping (4.0), a step-scheduled objective, a built-in matched-K eval sweep and the endpoint-error-at-sampler-grid diagnostic. | working on |
```

## 13. Next steps

1. Sync to the cluster, run `python FM_v3_alphaflow_test/gates_alphaflow.py`.
   🔴 **If G1 or G2 fails, STOP** — the homotopy is not wired at its endpoints and every
   intermediate α is meaningless (PLAN §7).
2. Run the 200-step dataset smoke (§7). Confirm the α banner shows a moving schedule.
3. Full train → the pipeline already chains eval across K ∈ {1,2,5,10}:
   ```bash
   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/alphaflow_pipeline.sh
   ```
4. Run `endpoint_error_alphaflow.py` on the α-Flow checkpoint **and** (via `--loadpath`) on
   Gen3v4-iMF and Gen3v6-MeanFlow, for the cross-generation table.
5. **Pre-registered kill criteria (PLAN §8) — read them before looking at the numbers:**
   - ⭐ α-Flow @ K=1–2 ≥ 95 % safe at ≤ 0.12 s/plan ⇒ the generation succeeded.
   - α-Flow ≫ Gen3v6 and ≫ Gen3v4 but < FM@K=2 ⇒ the blind direction was *a* cause, not the
     binding one; the remaining gap is data scale.
   - 🔴 α-Flow ≈ MeanFlow ≈ iMF ⇒ the blind direction was **never** the operative cause;
     COMPARE §8.2 is refuted as *the* explanation. **Stop the line.**
   - α-Flow worse than MeanFlow ⇒ suspect the schedule or a `u_next` gradient leak **before**
     believing the science — re-run G1/G2/G5.
   - And the endpoint metric: **if `err(τ=0)` is flat in K, the objective change did not fix
     the field** (§8.1).
