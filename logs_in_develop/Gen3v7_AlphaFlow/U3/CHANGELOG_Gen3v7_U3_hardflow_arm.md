# Gen3v7 U3 — HardFlow arm (arm C) ported into α-Flow

**Date:** 2026-07-30 · **Status:** code complete, **unvalidated** (nothing has been run — this
container has no Python packages; every gate and sweep below is a cluster job).
**Brief followed:** [`../init/PORT_GUIDE_hardflow_into_Gen3v7.md`](../init/PORT_GUIDE_hardflow_into_Gen3v7.md)
**Source of truth copied from:** Gen3v6 U3 + fix_4
(`flow_matcher_v3_meanflow/sampling/hardflow_projection.py`), *not* Gen12's.

Before U3, Gen3v7 had **zero** HardFlow wiring. It now has the full three-arm setup:

| arm | what it is | where the constraint is applied |
|---|---|---|
| **A** `diffuser` | unguided α-Flow ODE | nowhere (field-quality floor) |
| **B** `dpcc-*` | DPCC post-hoc projection | after the sample |
| **C** `hardflow_new-*` | HardFlow `hardflow_new` in-loop NLP | inside the ODE, per step |

---

## 1. Files

**New**

| file | lines | what |
|---|---|---|
| `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` | 725 | the arm: `TrajectoryLayout` + `HardFlowNLP` + `HardFlowSampler` + `HardFlowPolicy` |
| `FM_v3_alphaflow_test/gates_hardflow_alphaflow.py` | 193 | pre-flight gates H0/H1/H3 **+ new H4** |
| `config/alphaflow_projection_eval.yaml` | 182 | Gen3v7-dedicated unified eval config (DPCC arms + arm C) |
| `Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh` | 104 | submit entry (gates then eval) |

**Edited**

| file | change |
|---|---|
| `flow_matcher_v3_alphaflow/sampling/__init__.py` | export the five HardFlow symbols |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | arm-C branch, matched-K, HF metrics, config repoint |
| `config/avoiding-d3il.py` → `plan_fm_v3_alphaflow` | matched-K plumbing (PORT_GUIDE caution 4) |

**Deliberately NOT touched:** `config/projection_eval.yaml` (read by ~50 files across every
generation — adding `hardflow_new` there would inject arm C into generations with no
`HardFlowPolicy`), `flow_matcher_v3_hardflow/` (Gen12 owns its copy; its `0.5 * randn` is
correct *for Gen12*), Gen3v6's copy, and `MASTER_TEST_HISTORY.md`.

No `diffuser/` shim was needed — `diffuser/flow_matcher_v3_alphaflow/` mirrors only `models`,
never `sampling`, exactly like the MeanFlow shim.

---

## 2. The math is verbatim — and that is now *verified*, not asserted

The claim "byte-identical to Gen3v6's post-fix_4 copy" is checkable, so it was checked. Stripping
comments and blank lines from both files and diffing:

```
$ diff <(strip flow_matcher_v3_meanflow/sampling/hardflow_projection.py) \
       <(strip flow_matcher_v3_alphaflow/sampling/hardflow_projection.py)
368c368
<     """`init_noise_scale` defaults to Gen3v6's sigma=1.0 (mf_diffusion.py:204).
---
>     """`init_noise_scale` defaults to Gen3v7's sigma=1.0 (af_diffusion.py:260).
```

**One line, and it is a docstring.** Every executable statement — the dof layout, the CasADi
NLP assembly, the τ²-weighted prox cost, the reference-step / terminal-predict / project /
pull-back loop, the DPCC-parity selection rules, the NFE and NLP-failure accounting — is
unchanged. That is the whole point: arm C must differ from Gen3v6's arm C *only* in which field
it queries, or the cross-generation comparison is void.

The per-ODE-step math, unchanged:

```
1. reference step    v_k    = f(x_k, τ_k);   x_ref = x_k + v_k·dt
2. terminal predict  x1_ref = x_ref + (1 − τ_{k+1})·f(x_ref, τ_{k+1})
3. prox-NLP          min ½·reg·‖x1 − x1_ref‖²·τ_{k+1}²   s.t. constraint_list
4. pull-back         x_{k+1} = x_ref + τ_{k+1}·(x1_proj − x1_ref)
```

---

## 3. What actually changed for α-Flow (the three generation-specific things)

### 3.1 The `h = 0` grounding — α-Flow is a *better* host than Gen3v6

The NLP needs an instantaneous `v(x, τ)`; α-Flow's net emits the interval average `u(x, r, t)`.
The port queries it at `h = 0`, where `u(x, t, 0) = v(x, t)` exactly. Gen3v6 relied on that
identity holding **structurally**. α-Flow **trains** it:

```python
# af_diffusion.py:694
fm_mask = torch.rand(B, device=device) < self.af_ratio_fm   # default 0.5 (af_diffusion.py:94)
r = torch.where(fm_mask, t, r)   # FM anchors: h = 0  ⇒  u_tgt = v_inst
```

Half of every training batch is direct supervision of exactly the corner this sampler queries.
So the h=0 field here is *better supported* than in Gen3v6 — the strongest argument for porting
the arm to this generation at all, and it is now written into the module docstring rather than
left as folklore.

`h = 0` is passed **explicitly** (`h=torch.zeros_like(t)`), never relying on an `h=None`
backbone default. `gate_h1` pins it.

`_predict_velocity(self, x, cond, t, h=None, returns=None)` at `af_diffusion.py:221` is the
same signature as MeanFlow's (PORT_GUIDE caution 3, confirmed), so `_velocity_batch` copies
unchanged.

### 3.2 Initial noise σ = 1.0 — the mistake already paid for once

`init_noise_scale=1.0`, read off **`af_diffusion.py:260`** (`torch.randn(shape)`,
*"sigma=1.0 to match q_sample training noise"*).

🔴 **NOT** `flow_matcher_v3_alphaflow/models/diffusion.py:183,302`. That is the legacy `FMv3ODE`
class living in the same folder, still on `0.5 * randn`, and reading it is precisely how Gen3v6
ran arm C at half the trained noise scale against a σ=1.0 checkpoint — silently, for an entire
K-sweep (Gen3v6 fix_4). The trap is that the folder layout makes the wrong file the *obvious*
one to open.

Three independent defences, all carried over:
- `init_noise_scale` is a **required** argument on `HardFlowSampler` (no default) — a wrong
  scale is silent, so the call site must state it.
- the eval driver passes `init_noise_scale=1.0` explicitly with the warning inline.
- `gate_h3` pins it numerically (empirical std of a 4096-sample draw, 2 % tolerance — loose
  enough not to flake, tight enough to catch 0.5 vs 1.0).

### 3.3 Matched-K plumbing (PORT_GUIDE caution 4 — was missing entirely)

`plan_fm_v3_alphaflow` hardcoded `flow_steps_v3: 2` with **no `flow_steps` key at all**, so arm
C had no K to read and every K in a sweep would have overwritten the same results directory.
Now mirrors `plan_fm_v3_meanflow`:

```python
'flow_steps_v3': int(os.environ.get('HFFM_FLOW_STEPS', 2)),   # arms A/B
'flow_steps':    int(os.environ.get('HFFM_FLOW_STEPS', 2)),   # arm C
```

Both keys, because K must flow through the **config** so `args.savepath` encodes `_K{K}_`
(`flow_steps_v3` is the `'K'` token in `args_to_watch_fmv3_ode_plan`, `config/avoiding-d3il.py:61`).

Gen3v7 also has a `--flow-steps` CLI that Gen3v6 does not, and it patched only
`flow_steps_v3` / `ode_inference_steps_v3`. It now patches `flow_steps` too — otherwise
`--flow-steps 5` would move arms A/B to K=5 and leave arm C at K=2, quietly producing an
unmatched-budget table. The driver also re-forces K onto the loaded model
(`fm_model.flow_steps_v3 = fm_model.ode_inference_steps_v3 = flow_steps`), because the
`getattr` chain above it can otherwise pick up a stale checkpoint value.

---

## 4. Two real bugs found while porting

### 4.1 🐛 Plot loop crashes arm C at its own default fan size — **fixed here**

`FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:489` iterates the candidate fan with
`range(min(args.batch_size, 4))`. `args.batch_size` is the **arms A/B** fan (4). Arm C's fan is
`HFFM_BATCH`, whose YAML default is **1**. Indexing `0..3` into a length-1 fan is an
`IndexError` in the plotting block, after the rollout has already been paid for.

Gen12 had this right — `FM_v3_hardflow_test/eval_FM_v3_hardflow.py:445` uses
`min(batch_size, 4)`, the arm-aware variable. The Gen3v6 U3 copy regressed it, and it never
fired because every Gen3v6 U3 run used `HFFM_BATCH=4`.

Gen3v7 restores Gen12's form. The same fix applied to `batch_size=` on the `RTRecorder`
constructor and on the `policy(...)` call, both of which had the same `args.batch_size` leak.

**Gen3v6 still has this latent bug.** Not touched — cross-generation edits are the user's call.
Flagged in §7.

### 4.2 ⚠️ `hardflow_new-c` is expected to be degenerate — do not read it as a port bug

PORT_GUIDE caution 5, and Gen3v7's own
[`U2 investigation`](../U2/INVESTIGATION_dpcc-c_Gen3v7.md) makes it concrete. `-c` selects the
candidate needing least intervention. Its cost is **identically zero on the whole feasible set**
(`projection.py:145`), so it cannot distinguish "barely feasible" from "comfortably feasible"
and its only active gradient points *at* the constraint surface. Two failure modes follow, both
already measured on the DPCC arm:

- a **motionless** candidate is the cheapest thing to leave alone → freeze (`bbsit` @ K=2, 17 %
  of candidates collapse, and only within ≈0.01 of the start pose);
- a **boundary-grazing** candidate is the next cheapest → violations (plain `dpcc-c`, all K).

So a Gen3v7 K=2 run's `dpcc-c` / `dpcc-c-tightened` cells are *expected* to read 0.0, and
`hardflow_new-c` may do the same for its own reasons. This is written into the YAML header so
the next reader does not spend a day re-deriving it. **Check the candidate fan before calling a
`-c` collapse a port bug.**

---

## 5. Gates — H0/H1/H3 carried over, **H4 is new**

```bash
python FM_v3_alphaflow_test/gates_hardflow_alphaflow.py     # H0 H1 H3 H4
python FM_v3_alphaflow_test/gates_hardflow_alphaflow.py --checkpoint <loadpath>   # adds H2
```

| gate | pins | why it exists |
|---|---|---|
| H0 | imports resolve; casadi availability | cheap smoke test |
| H1 | `_velocity_batch` calls `_predict_velocity` with `h` a `(B,)` **all-zero tensor** | a backbone default change would silently swap the grounded field for the interval average |
| H3 | `init_noise_scale == 1.0` and the empirical draw std matches | §3.2 — the Gen3v6 fix_4 bug, numerically |
| **H4** | **`af_ratio_fm > 0`, and plan == train** | **new, Gen3v7-only** |
| H2 | numeric `u(x,t,0) ≈ v_fd` on a real checkpoint | still a **stub** (§7) |

**H4** is PORT_GUIDE caution 2 made executable. If a checkpoint was trained with
`af_ratio_fm = 0`, no batch ever saw `r == t`, the `u(x,t,0) = v` identity was never supervised,
and the entire arm queries an untrained corner of the field — with no crash and no solver error
to show for it. H4 reads both `plan_fm_v3_alphaflow` and `flow_matching_v3_alphaflow` from the
config module and fails if either is 0 or if they disagree (they disagree ⇒ the loadpath token
`_rf{af_ratio_fm}` resolves to a different or missing checkpoint).

All three silent-failure modes now have a gate. The sbatch runs the gates **before** the eval
under `set -e`, so a broken port aborts the job instead of producing plausible numbers.

---

## 6. How to run

```bash
# gates only (fast, no GPU needed beyond torch)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh   # gates + eval

# matched-K sweep — the PORT_GUIDE's "before believing any number" protocol
for K in 1 2 5 20; do
  HFFM_FLOW_STEPS=$K HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh
done
```

Knobs (env overrides the YAML `hardflow:` block):

| var | sbatch default | meaning |
|---|---|---|
| `HFFM_BATCH` | **4** | arm-C candidate fan. **1** = upstream-faithful but reproduces the Gen3v6 fix_3 confound (arm B batched+selected vs arm C batch-1), so 4 is the default here for any B-vs-C claim. |
| `HFFM_ACT_THRESHOLD` | 0.5 | fraction of late ODE steps the NLP is active; 0.5 == DPCC's `diffusion_timestep_threshold` 0.5. Final step always solved. |
| `HFFM_FLOW_STEPS` | plan block (2) | matched K for **every** arm. |

**Arm B is the control.** If `dpcc-*` numbers move relative to a DPCC-only Gen3v7 run at the
same K and seed, the port leaked into the shared path and nothing else in the run is
trustworthy.

---

## 7. What is NOT done / known gaps

1. **Nothing has been executed.** No gate has passed, no eval has run. Every number this arm
   will produce is unvalidated. → cluster.
2. **H2 is still a stub** (inherited from Gen3v6). It is the only gate that would confirm the
   `u(x,t,0) = v` identity *numerically on real weights* rather than structurally. Given §3.1
   says α-Flow trains that anchor directly, H2 is the natural place to *measure* how well —
   finite-difference the sampler endpoint and compare. Worth finishing here more than it was in
   Gen3v6.
3. **Gen3v6's plot-loop bug (§4.1) is unfixed** — it will crash any Gen3v6 arm-C run at
   `HFFM_BATCH=1`. One-line fix (`args.batch_size` → `batch_size`, three sites); not applied
   because Gen3v6 was out of scope for this task.
4. **`--config` collides with the Parser's own `config` field.** Passing `--config <yaml>`
   sets both the YAML path *and* `utils.Parser.config` (the python config module). Inherited
   verbatim from Gen3v6 for sibling parity; harmless on the default path (nothing passes
   `--config`), but it is a live footgun.
5. **`dynamics_mode: linear_fit` is unusable here.** It needs an `.npz` fitted against *this
   generation's* normalizer (`FM_v3_hardflow_test/fit_dynamics_fmv3.py`), and none exists for
   α-Flow. `deriv` (the matched-feasible-set default) is what will run.
6. **`goal_dim == 0` is asserted.** Fine for avoiding-d3il; a goal-conditioned env needs the
   goal columns carried through the dof vector.
7. **NFE accounting is 2K per plan when the NLP is active** (reference step + terminal predict),
   vs K for arms A/B. That is inherent to `hardflow_new`, not a port artefact — but it means
   "matched K" is matched *ODE steps*, not matched network evaluations. State which one any
   table means.
