# Gen14 U7 — HardFlow (`hardflow_new`) arm C, ported into the three flow arms

**Date:** 2026-08-04
**Scope:** eval / sampling side only. No training code, no model weights, no checkpoint touched.
**Hosts:** `fm` (Gen7), `mf` (Gen3v6), `af` (Gen3v7). **`diffusion` is refused by design** — see §2.
**Status:** implemented, statically verified, **never run**. H1/H3 need the cluster.

**Ask:** *"where is my HardFlow projection? they ARE in the Gen3v6/7 but lost in Gen14, also the Gen12
FM HardFlow should plug in to Gen14 (diffusion not, since no FM). Bring back the HF projection for the
3 … keep the code fidelity, try to copy the HF implementation rather than write again."*

**Outcome: possible, done, not aborted.** One thing in the port would have been silently wrong if copied
verbatim; it is fixed and gated. See §3.

---

## 1. Where HardFlow was, and what "copy, don't rewrite" meant here

| generation | file | host field |
|---|---|---|
| Gen12 | `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` (633 l) | FMv3 ODE — instantaneous `v(x,t)` |
| Gen3v6 | `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` (694 l) | MeanFlow — two-time `u(x,r,t)` |
| Gen3v7 | `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` (733 l) | α-Flow — two-time |
| **Gen14** | **`mix_visual_aligning/sampling/hardflow_projection.py` (801 l)** | **all three, visual** |

Gen14 had **zero** HardFlow wiring — `mix_visual_aligning/sampling/` held only `projection.py`.

The port follows `logs_in_develop/Gen3v7_AlphaFlow/init/PORT_GUIDE_hardflow_into_Gen3v7.md`, which is
the written procedure for exactly this move and lists the mistakes already paid for. Per that guide the
source of truth is the **latest** copy (Gen3v7), not Gen12's.

**Method: `cp`, then four documented edits.** The file was produced by copying
`flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` and editing it in four places. The NLP, the
dof layout, the prox schedule, the activation gate and the pull-back were **not** retyped.

`gates_hardflow_mix_visual.py::gate_h0` pins that, and it passes today:

```
ok   TrajectoryLayout                 src=  50 dst=  50
ok   HardFlowNLP                      src= 216 dst= 216
ok   resolve_activation_threshold     src=  28 dst=  28
H0: PASS
```

266 of the 733 source lines are byte-identical; the rest is the module docstring, `HardFlowSampler`
(4 changed lines), and the new Gen14 glue. `HardFlowSampler` itself diffs **+41 / −4** against Gen3v7,
and 3 of the 4 removed lines are the two edits below.

---

## 2. The `diffusion` arm is refused, not silently supported

`hardflow_new` integrates `v = f(x, τ)` **outside** the solver — that is the only reason it is portable
at all (HardFlow's other three modes compile the network into the NLP via l4casadi and are
architecture-locked). A DDPM reverse chain has no velocity field, so the mode is not "unsupported" on
the `diffusion` arm, it is **undefined**.

`resolve_engine_hf()` therefore raises, and the eval catches it and skips the cell with a reason instead
of crashing a sweep mid-GPU-allocation:

```
[ eval ] SKIPPING variant 'hardflow_new-r': HardFlow is not available for engine 'diffusion'.
         Hosts are ['af', 'fm', 'mf'] — the flow-matching arms. …
```

`gate_h4` asserts `'diffusion'` is absent from both engine tables.

---

## 3. 🔴 The one place a verbatim copy would have been silently wrong

Gen3v6/v7 are **state-only**. Their sampler strips string keys before calling the network:

```python
cond_net = {k: v for k, v in cond.items() if not isinstance(k, str)}
```

Correct there — those keys are diffuser bookkeeping. **In Gen14 the visual conditioning IS a string
key.** `cond['visual_latent']` (B, 128) is what `VisualUNetTwoTime._project_cond`
(`visual_unet_twotime.py:185`) reads, and that method returns **`None`** when the key is absent — no
exception, no warning, no log line.

A verbatim port would therefore have run **every HardFlow rollout with no image conditioning at all**,
produced entirely plausible trajectories from an unconditioned field, and been indistinguishable from a
working arm in the results. Given that Gen14's arms already fail the task (U5 DA), it would have been
essentially undetectable.

Two changes close it:

1. `_VISUAL_COND_KEYS = frozenset({'visual_latent', 'visual'})` is an explicit **allow-list** through the
   filter.
2. If the host has `if_vision=True` and the cond carries neither key, `sample()` **raises**. Running
   blind is not an allowed silent fallback.

Plus a trap inside the trap: `if_vision` lives on the **backbone**, not the diffusion wrapper
(`visual_fm_diffusion.py:47` reads it as `self.model.if_vision`). Probing `getattr(model, 'if_vision')`
alone returns `False` and disarms the guard, so the check walks both.

`gate_h2` pins all of it statically — including that the old blind filter is gone from *executable*
code (the docstring quotes it deliberately, so the gate strips docstrings/comments before searching).

---

## 4. The other three deltas

**(2) `two_time` flag.** Gen14 hosts two velocity signatures:

| arm | signature | call |
|---|---|---|
| `fm` | `_predict_velocity(x, cond, t, returns=None)` (`fm_diffusion.py:71`) | Gen12's, verbatim — no `h` |
| `mf` / `af` | `_predict_velocity(x, cond, t, h=None, returns=None)` | Gen3v6/v7's — `h=0` passed **explicitly** |

`h=0` is the mean-flow grounding `u(x,t,0) = v(x,t)`, and it is passed explicitly (never via an
`h=None` backbone default) so a future default change cannot swap the grounded field for the interval
average. Getting `two_time` wrong is a `TypeError` on the first call, not a silent degradation.

**(4) `init_noise_scale` re-derived from Gen14's own samplers.** fix_4's standing rule: read the scale
off the HOST generation, never inherit it. Gen3v6's original port inherited Gen12's 0.5 against a σ=1.0
checkpoint and lost an entire K-sweep to it.

| arm | source | σ |
|---|---|---:|
| `fm` | `fm_diffusion.py:164` `x = 0.5 * torch.randn(shape, …)` | **0.5** |
| `mf` | `mf_diffusion.py:204` `x = torch.randn(shape, …)` | **1.0** |
| `af` | `af_diffusion.py:260` `x = torch.randn(shape, …)` | **1.0** |

The argument stays **required** (no default). `ENGINE_INIT_NOISE` is the single source of truth;
`gate_h4` re-reads each engine's `p_sample_loop` and fails if the table and the sampler disagree;
`gate_h3` measures the realised σ numerically on GPU.

**(1) `HardFlowPolicy` dropped (126 lines not copied).** Gen3v6/v7 drive the sampler through diffuser's
`Policy`/`Trajectories`; Gen14's closed loop is `VisualAgentWrapper`. Its `__init__` body (layout →
mins/maxs → NLP → sampler) is lifted verbatim into `build_hardflow_sampler()`, minus the
`Policy.__call__` bookkeeping (`preprocess_fn`, `test_ret`, `prev_observations`) that nothing else used.
Gen14 has no `sampling/policies.py`, so keeping the class would have meant inventing one. **Deliberate
deviation, recorded here rather than left implicit.**

---

## 5. Files touched

### Created

| file | lines | what |
|---|---:|---|
| `mix_visual_aligning/sampling/hardflow_projection.py` | 801 | the port: `TrajectoryLayout`, `HardFlowNLP`, `resolve_activation_threshold` (byte-identical), `HardFlowSampler` (+41/−4), and new Gen14 glue — `_VISUAL_COND_KEYS`, `ENGINE_INIT_NOISE`, `ENGINE_TWO_TIME`, `resolve_engine_hf`, `encode_visual_cond`, `build_hardflow_sampler` |
| `mix_visual_aligning_test/gates_hardflow_mix_visual.py` | 210 | gates H0–H4 |
| `logs_in_develop/Gen14/U7/CHANGELOG_Gen14_U7_hardflow_arm_C.md` | — | this file |

### Modified

| file | Δ | what |
|---|---:|---|
| `mix_visual_aligning/sampling/__init__.py` | +21 | re-export the HardFlow symbols alongside `Projector` |
| `mix_visual_aligning_test/eval_mix_visual_aligning.py` | +141 / −3 | see below |
| `config/visual_aligning_eval.yaml` | +61 | `hardflow_variants: []` (opt-in, default OFF) + `hardflow:` knob block |

**`eval_mix_visual_aligning.py`, the six edits:**

1. import `build_hardflow_sampler` / `encode_visual_cond` / `resolve_activation_threshold`.
2. `setup_dpcc_projector(..., return_constraint_list=False)` — **additive**. Returns
   `(constraint_list, ProjectorNormalizer, dt)` instead of a `Projector` when asked. The port guide is
   explicit that arm C's NLP must be built from *FMPCC's* constraint list and never from HardFlow's own
   `avoiding_geometry.py`, "otherwise arms B and C enforce different feasible sets and the comparison is
   void". Sharing the builder makes that divergence unrepresentable. Default path byte-unchanged.
3. `hardflow_variants` / `HFFM_VARIANTS` → appended to `projection_variants` (see §6).
4. arm-C branch in the variant loop — builds the sampler, or skips the cell with a reason. `projector`
   stays `None` on these variants; the two are mutually exclusive.
5. selection-suffix grammar extended so `hardflow_new-c` / `-t` map to the same rules as `dpcc-c` / `-t`,
   and `minimum_projection_cost` reads HardFlow's own per-candidate `candidate_costs` (Σ‖x1_proj−x1_ref‖²)
   instead of re-solving through a projector that does not exist on this arm.
6. `VisualAgentWrapper`: new `hf_sampler=None` argument, an assert that it and `projector` are never both
   set, and a branch in `predict()` that calls `hf_sampler.sample(...)` — which returns the same
   `(x, infos)` contract as `p_sample_loop`, so candidate selection, metrics, npz and recording are all
   unchanged downstream.

**Not touched:** any file under `mix_visual_aligning/models/`, the training script, the sbatch scripts,
`Slurm_Codes/`, and every other generation's HardFlow copy (Gen12's still has its 0.5 — correct for it).

---

## 6. How existing runs are protected

`config/visual_aligning_eval.yaml` is **shared with the Gen6V4 and Gen7 evals**, neither of which has
HardFlow code. Putting `hardflow_new-*` into `projection_variants` would inject arm C into those
generations and crash them — the exact trap `config/meanflow_projection_eval.yaml:6` documents.

So HardFlow variants live under a **separate key**, `hardflow_variants`, read only by the Gen14 eval.
Every other consumer ignores an unknown top-level key.

| protection | status |
|---|---|
| `hardflow_variants: []` by default | HardFlow is **off** unless explicitly enabled |
| `projection_variants` unchanged | 16 entries, no `hardflow` substring — verified |
| `hf_sampler=None` on every non-HardFlow variant | the `predict()` branch cannot fire |
| `projector` / `hf_sampler` mutual exclusion | asserted in `VisualAgentWrapper.__init__` |
| `diffusion` arm | refused by `resolve_engine_hf`, cell skipped with a message |
| `return_constraint_list` | additive parameter, default path identical |
| existing `G0` gate | **PASS, 23/23** after all edits |

A run that does not set `hardflow_variants` (or `HFFM_VARIANTS`) executes the same code path it did
before U7.

---

## 7. Verification

Static only — this container has no torch and no GPU.

| check | result |
|---|---|
| `ast.parse` on all 3 modified/created `.py` | OK |
| `bash -n` on all `mix_visual_aligning` sbatch | OK |
| `yaml.safe_load` on `visual_aligning_eval.yaml` | OK |
| **`gates_mix_visual.py --gate g0`** (pre-existing) | **PASS — 23 verbatim files match** |
| **`gate_h0`** byte-identity of copied regions vs Gen3v7 | **PASS** |
| **`gate_h2`** visual-latent survival + blind-rollout raise | **PASS** |
| **`gate_h4`** engine table, `diffusion` refusal, σ cross-check | **PASS** |
| `gate_h1` (h=0 grounding / signatures) | **not run — needs torch** |
| `gate_h3` (measured init-noise σ) | **not run — needs torch** |

```
H0=PASS  H1=FAIL(no torch)  H2=PASS  H3=FAIL(no torch)  H4=PASS
```

**Never executed:** no HardFlow rollout has run in Gen14. The NLP path needs CasADi/IPOPT on the cluster
and has not been exercised here at all. Treat every number it produces as unvalidated until H1/H3 pass
and a smoke run completes.

---

## 8. Commands

```bash
# gates first — H1/H3 need the cluster env
python mix_visual_aligning_test/gates_hardflow_mix_visual.py

# smoke: one HardFlow variant, mf, K=2, no recording
HFFM_VARIANTS="hardflow_new-r" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6 none 2

# the matched arm-B/arm-C comparison (dpcc-c is already in projection_variants)
HFFM_VARIANTS="hardflow_new-r hardflow_new-c hardflow_new-t" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh af 6 none 2
```

⚠️ `HFFM_VARIANTS` must reach the compute node. `submit.sh` passes `--export=ALL`, so exporting it in
the submitting shell works; setting it inline as above also works for the same reason. Verify from the
log banner:

```
[ eval ] HardFlow variants enabled (U7): ['hardflow_new-r']
[ hardflow ] engine=mf  init_noise_scale=1.0  two_time=True  act_thr=0.5  visual=True
[ eval ] HardFlow sampler active for variant 'hardflow_new-r' (engine=mf, trajectory_dim=9)
```

**If `visual=False` appears on a visual checkpoint, stop** — that is §3's failure and the numbers are
worthless.

---

## 9. Known limitations / open

1. **Never run.** H1/H3 unexecuted; the CasADi/IPOPT path unexercised in Gen14.
2. **`activation_threshold` has almost no resolution at K=2.** `int((1−T)·K)` is 1 for `T ∈ (0, 0.5]`
   and 0 for `T > 0.5`, so 0.5 and 0.0 give the same single terminal solve (U5 DA §12.1). The HardFlow
   threshold sweep only becomes meaningful at larger K. Default is `null` = inherit
   `diffusion_timestep_threshold`, so arms B and C stay matched.
3. **`dynamics_mode: 'linear_fit'` is not wired** for the visual task — it needs a fitted `.npz` from a
   Gen12-style `fit_dynamics` run on aligning-d3il. Only `'deriv'` (the matched-feasible-set mode, and
   the recommended default) works.
4. **`-c` selection is known-degenerate** on both engines (port guide caution 5): it picks whichever
   candidate needed least intervention, which is a motionless one whenever the field produces one. Do
   not read a `-c` collapse as a port bug without checking the candidate fan.
5. **Cost is unmeasured.** Gen3v6 measured `hardflow_new-*` at ~3.3× the `dpcc-*` variants with
   716–1410 NFE of inner-solver work, and recommended dropping it from the headline table. Whether that
   holds at K=2 under vision is exactly what the first run should answer.
6. **`af_ratio_fm > 0` matters for the `af` host.** If a checkpoint was trained with `af_ratio_fm = 0`
   the `h=0` anchor was never trained and the arm rests on an untrained corner of the field. Gen14's
   config sets `af_ratio_fm: 0.5`, so the current checkpoints are fine — re-check before evaluating any
   future one.
