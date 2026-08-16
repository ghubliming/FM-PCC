# Gen3v6 U3 — PLAN: port the Gen12 HardFlow arm **into** Gen3v6 (in-folder addon)

**Status: PLAN ONLY — no coding yet.** Builds on
`logs_in_develop/Gen12/fix_7/DISCUSS_U5_gen12_loadable_FM_models.md` (esp. §6), and on the U2
`mf_dit` backbone. This plan turns the DISCUSS's "port HardFlow into v6/v7" recommendation into a
concrete, math-preserving worklist.

---

## 1. Goal & scope (what the user asked for)

Take Gen12's HardFlow codes, **copy them into Gen3v6's own folders**, and **reassemble them as an
add-on arm inside Gen3v6** (not as a new sibling folder). After this, one Gen3v6 eval run must
produce, side-by-side:

- the existing **DPCC arms** (`diffuser`, `dpcc-{r,c,t}[-tightened]`, gradient/post_processing/…), **and**
- the **HardFlow arm** (`hardflow_new-{r,c,t}[-tightened]`) with its **HF metrics**
  (`nfe`, `nlp_solves`, `nlp_failures`, `avg_time`, `candidate_costs`) and **"set entries"**
  (the `projection_variants` registry + the results collector rows).

**Bottom line (non-negotiable): the HardFlow math must be identical to Gen12's.** The NLP, the
endpoint identity `x₁ = x_τ + (1−τ)·v`, the fix_7 DPCC-parity batching, the tightening, the
selection suffixes, and the metrics are all **copied verbatim**. The *only* model-specific point is
how the instantaneous velocity `v` is obtained — resolved by the `h=0` identity in §3.

### Deviation from DISCUSS §6e (intentional, per user)

§6e recommended a **new copy-modify sibling** (`flow_matcher_v3_meanflow_hardflow/`). The user
instead wants the HF modules added **inside the existing `flow_matcher_v3_meanflow/` +
`FM_v3_meanflow_test/`**, "Gen12-mixed-Gen3v6 style" — i.e. exactly how **Gen12 itself** carries the
FMv3ODE engine and the HardFlow modules together in one folder (`flow_matcher_v3_hardflow/` holds
both `models/diffusion.py` *and* `sampling/hardflow_projection.py`). This is consistent: Gen3v6
becomes the "engine + all three arms" folder, mirroring Gen12's own layout. Trade-off vs a sibling:
lower folder proliferation and a single eval that does everything, at the cost of the DPCC-only
Gen3v6 eval gaining a new optional arm (kept **additive/off-by-default** so existing DPCC runs are
unaffected — see §7).

---

## 2. The mathematical crux — why the math stays exactly Gen12's

Gen12's HardFlow arm requires an **instantaneous** probability-flow velocity `v(x,τ)`, queried as
`_predict_velocity(traj, cond, t, returns=…)` and extrapolated by the linear-FM endpoint
`x₁ = x_τ + (1−τ)·v` (`hardflow_projection.py:390, :456`). A mean-flow net emits the **interval
average** `u(x,t,r)`. The MeanFlow training target **grounds `u` on `v` at zero interval**:

> `u(x, t, h=0) = v(x, t)` exactly, by construction of the JVP target
> (`mf_diffusion.py:443-461`; the `h==0` FM-anchor reduces `u_target` to `v_inst`).

So a mean-flow checkpoint **can** hand HardFlow a genuine instantaneous velocity — provided the HF
arm queries the u-head at **`h = 0` (r = t)**, not the native few-step interval `h = dt`. With
`h=0`, HardFlow's Euler step and endpoint identity are the field they were derived for → **no
re-derivation, math identical to Gen12**. With `h=dt` you'd reintroduce the §2 "wrong field" bug.

**Confirmed already true by default:** `MeanFlowODE._predict_velocity(x,cond,t,h=None,…)` passes
`h=None`, and every DiT/UNet backbone maps `h=None → _as_batched(h, default=0.0) → 0`. So Gen12's
existing call (no `h` arg) **already lands on `u(x,t,0)=v`**. The plan still makes this **explicit**
(pass `h=torch.zeros_like(t)`) so a future backbone default change can't silently break it — a
one-line, value-preserving guard, not a math change.

*(Scope note carried from DISCUSS §6c: running HardFlow's K-step Euler on `u(·,·,0)` uses the
mean-flow model as a plain FM field — it gives up mean-flow's few-step speed for this arm. The HF
arm is therefore a **field-quality** A/B under identical in-loop constrained sampling, NOT a
few-step-speed claim. The writeup must say so.)*

---

## 3. What gets copied — file-by-file port list

Source = `flow_matcher_v3_hardflow/*` and `FM_v3_hardflow_test/*` (Gen12). Target =
`flow_matcher_v3_meanflow/*` and `FM_v3_meanflow_test/*` (Gen3v6).

### 3a. Core HF module — **verbatim copy**

| source | → target | change |
|---|---|---|
| `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` | **verbatim**, except the single explicit-`h=0` guard on the velocity query (§2). Everything else — `TrajectoryLayout`, `HardFlowNLP` (casadi/ipopt), `HardFlowSampler` (fix_7 batched ODE), `HardFlowPolicy`, `resolve_activation_threshold` — copied unchanged. |

The velocity query, `_velocity_batch` (`:389-390`):

```python
# Gen12 (FM, instantaneous v):
v = self.model._predict_velocity(traj, cond, t, returns=returns)
# Gen3v6 (mean-flow, explicit h=0 so u(x,t,0)=v — SAME field, SAME math):
v = self.model._predict_velocity(traj, cond, t, h=torch.zeros_like(t), returns=returns)
```

This is **the only** model-specific line. `HardFlowNLP`, the endpoint construction
(`X1_ref = X_ref + (1−τ)·V_next`), the DPCC-parity batching, tightening, and metrics are untouched.

### 3b. Already present in Gen3v6 — **reuse, do not copy**

- `flow_matcher_v3_meanflow/sampling/{policies.py, projection.py}` — DPCC arms A/B already work
  here (the Gen3v6 eval uses them today).
- `flow_matcher_v3_meanflow/utils/constraints_helpers.py` — **verified byte-identical** to the
  hardflow copy (`formulate_halfspace_constraints → (C_row, d)`, the exact tuple the NLP unpacks).
  The HF NLP consumes the same constraint objects DPCC does → tightened-vs-exact semantics match by
  construction (Gen12 `VERIFY_U5_hardflow_tightening_codepath.md`).
- `flow_matcher_v3_meanflow/models/diffusion.py::_predict_velocity` — the plain FM baseline contract
  (`return self.model(x,cond,t)`), identical to Gen12's; irrelevant to the real mean-flow arm but
  keeps the folder's baseline loadable too.

### 3c. Test-side helpers — **copy + light adapt**

| source | → target | change |
|---|---|---|
| `FM_v3_hardflow_test/hf_paths.py` | `FM_v3_meanflow_test/hf_paths.py` (optional) | Only if we adopt Gen12's K/threshold/batch-encoded results dir. **Recommended simpler path:** skip it and write `hardflow_new-*.npz` into Gen3v6's **existing** `results/halfspace_*/` dir alongside `dpcc-*.npz`, so one run yields both sets in one place. (Decision D2, §9.) |
| `FM_v3_hardflow_test/gates_hardflow.py` | `FM_v3_meanflow_test/gates_hardflow_meanflow.py` | Copy; adapt imports to `flow_matcher_v3_meanflow`. The critical extra gate: **`u(x,t,0)` really equals the FM velocity** on a mean-flow checkpoint (numerically compare `_predict_velocity(...,h=0)` against a finite-difference of the sampler), i.e. verify §2 empirically before trusting the arm. |

### 3d. Eval driver — **graft arm C into the existing Gen3v6 driver**

Do **not** replace `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py`; **graft** the Gen12
arm-C logic into it (both are copy-modifies of the same DPCC eval, so the diff is small and local):

1. Imports: `from flow_matcher_v3_meanflow.sampling.hardflow_projection import HardFlowPolicy, resolve_activation_threshold`.
2. Parse the `hardflow:` config block + env overrides (`HFFM_BATCH`, `HFFM_ACT_THRESHOLD`,
   `HFFM_FLOW_STEPS`) — verbatim from `eval_FM_v3_hardflow.py:54-67`.
3. In the `for variant` loop, add `is_hardflow = variant.startswith('hardflow')` and the arm-C
   branch (`eval_FM_v3_hardflow.py:295-323`): build `HardFlowPolicy` with the DPCC-parity selection
   suffix parse (`_sel_base = variant.replace('-tightened','')` → `-t`/`-c`/random).
4. HF metric accumulation already present in the shared MPC loop
   (`nlp_solves_total`, `nlp_failures_total`, `nfe`, `avg_time`) — copy those lines
   (`:349-352, :397-399`) and the per-variant result row that records them.
5. Matched-K knob: one `--flow-steps` / `HFFM_FLOW_STEPS` override sets K for **all** arms
   (Gen12 PLAN §5 — the central lesson: never compare arms at different K).

---

## 4. Config additions (additive — existing DPCC runs unaffected)

### 4a. One unified, Gen3v6-dedicated eval YAML — `config/meanflow_projection_eval.yaml`

**Decision (user): keep ONE YAML — but a Gen3v6-OWNED one, not the shared file.**
`config/projection_eval.yaml` is read by **~50 files across every generation** (every `FM_*_test`,
every `flow_matcher_v3_*/utils/setup.py`, `scripts/`, `config/*.py`), so adding `hardflow_new` there
would inject the HF arm into Gen3/Gen7/αFlow/etc. — and crash the gens whose drivers don't import
`HardFlowPolicy`. So we do NOT touch it.

Instead: create **one** new file `config/meanflow_projection_eval.yaml` that is the single source of
truth for Gen3v6 eval — it merges the **full DPCC set + the HardFlow set + the `hardflow:` block**
into one document, and Gen3v6's driver reads only this file. This gives exactly the "one unified
yaml" the user wants (one file does dpcc **and** hardflow), while staying isolated from other gens.

Concretely: copy the current Gen3v6 DPCC variant set from `config/projection_eval.yaml`, append the
HardFlow set, and add the verbatim `hardflow:` block:

```yaml
projection_variants: [
  'diffuser',
  # ── DPCC arms (unchanged from the shared projection_eval.yaml) ──
  'dpcc-r','dpcc-r-tightened','dpcc-c','dpcc-c-tightened','dpcc-t','dpcc-t-tightened',
  'gradient','gradient-tightened','post_processing','post_processing-tightened',
  'model_free','model_free-tightened',
  # ── HardFlow arm (new; gated by variant.startswith('hardflow')) ──
  'hardflow_new-r','hardflow_new-c','hardflow_new-t',
  'hardflow_new-r-tightened','hardflow_new-c-tightened','hardflow_new-t-tightened',
]
```

plus the verbatim `hardflow:` block (batch_size, dynamics_mode=deriv, activation_threshold,
candidate_cost=prox, ipopt/casadi print switches) and the identical constraint geometry
(halfspace/obstacle/bounds/enlarge — already matches Gen3v6's current `projection_eval.yaml`).
`dynamics_mode: deriv` keeps arms B and C on an **identical feasible set** (matched comparison).

**Driver repoint:** change `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` (line 43) and
`load_results_flow_matching_v3_meanflow.py` from `config/projection_eval.yaml` →
`config/meanflow_projection_eval.yaml`. A DPCC-only run then just means listing only the `dpcc-*`
variants in this one file (the HF arm never fires without a `hardflow*` entry).
*Minor caveat:* Gen3v6 no longer auto-inherits future edits to the shared `projection_eval.yaml`
(constraint geometry etc.) — acceptable, and exactly the isolation copy-modify intends; the geometry
is already identical and stable. (Note: `flow_matcher_v3_meanflow/utils/setup.py` also references
the shared yaml for the config-snapshot copy — check whether it must be repointed too, or left as a
harmless snapshot of the shared file, during the coding pass.)

### 4b. `plan_fm_v3_meanflow` block (`config/avoiding-d3il.py`) — add HF knobs

Add the HF-arm control keys the Gen12 `plan_fm_v3_hardflow` block carries but Gen3v6's plan block
lacks:

- `'flow_steps': <K>` — the HF Euler budget (Gen12 default 10; for Gen3v6 the mean-flow deploy K is
  small — pick a K and apply it to **all** arms via the single override, §3d.5).
- keep `diffusion_loadpath` pointing at the **real mean-flow checkpoint**
  (`…_objmeanflow_bbmf_dit_…`), NOT the plain FM baseline (DISCUSS §3/§6d.4).

No new `args_to_watch` token is needed for training (this is eval-only); the results dir already
separates by K (`…_K{flow_steps}_…`).

---

## 5. Slurm sbatch

Add `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` — copy
`Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh`, retarget:

- `python FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` — reads the unified
  `config/meanflow_projection_eval.yaml` via the repointed path (§4a). *(Optionally add a `--config`
  arg like Gen12's driver has, if you want to switch configs without editing the source — a small
  convenience, not required.)*
- keep the **EGL/GPU isolation guard verbatim** (never weaken it), `MUJOCO_GL=egl`, conda `FMPCC`.
- pass through `HFFM_BATCH` / `HFFM_ACT_THRESHOLD` / `HFFM_FLOW_STEPS` env knobs.
- casadi/ipopt already available in the FMPCC env (Gen12 runs prove it) — no env change.

The existing DPCC-only `eval_meanflow.sh` stays as-is for pure-DPCC runs.

---

## 6. The "HF metrics and set entries" deliverable (explicit)

After U3, a Gen3v6 eval must record, per `(scenario, variant, seed)`:

- **DPCC + shared metrics** (already emitted): `success`, `constraints_satisfied`,
  `success_and_constraints`, `n_violations`, `total_violations`, `avg_time`.
- **HF-specific metrics** (new, only for `hardflow_new-*` variants): `nfe`, `nlp_solves`,
  `nlp_failures`, `candidate_costs` (prox) / `candidate_costs_control`, `avg_time` (the timing the
  fix_7 batching improved).
- **Set entries**: the `projection_variants` union (§4a) is the "set"; the results collector
  (`load_results_flow_matching_v3_meanflow.py`) must **learn the `hardflow_new-*` rows** — extend its
  variant list / regex so the aggregated table (and any plot) includes the HF arm next to DPCC.
  Cross-check safeguard: `hardflow_new-c-tightened` and `dpcc-c-tightened` share the same tightened
  constraint set, so their **violation** columns should both hit 0 where each reaches the goal
  (Gen12 §4a parity check) — a built-in correctness probe that the port is clean.

---

## 7. Additivity / isolation guarantees (so existing Gen3v6 work is safe)

- The shared `config/projection_eval.yaml` (read by ~50 files across all gens) is **left untouched**
  → every other generation's eval is byte-for-byte unchanged. Gen3v6 repoints to its own single
  `config/meanflow_projection_eval.yaml`.
- The eval driver's arm-C branch is gated on `variant.startswith('hardflow')`, so a `projection_variants`
  list with only `dpcc-*` entries never executes the HF path (one file, both modes).
- No training-side change (U3 is eval-only). The U2 backbone work and all checkpoints are untouched.
- `diffuser/` namespace shim: **no new file needed** — `HardFlowPolicy` is imported directly by the
  test driver (not via `import_class` config strings), and the loaded checkpoint class
  (`MeanFlowODE`) already resolves through the fix_1 shim. The HF sampler only calls the existing
  `MeanFlowODE._predict_velocity`.

---

## 8. Risks & caveats

1. **`h=0` must be honored everywhere the HF arm queries velocity** (§2). This is THE correctness
   pin. Gate G-h0 (§3c) must pass before any results are trusted.
2. **casadi/ipopt dependency** — present on the cluster (Gen12 runs), absent in this container, so
   **no local numerical validation** — py_compile only here; real validation on i6-gpu-1.
3. **Three-way sync burden** — HF modules now live in Gen12, and Gen3v6 (and later v7). fix_7-class
   changes get a third home. Accept as the normal sibling-sync tax; revisit a shared import only if
   drift bites (DISCUSS §6d.1).
4. **Checkpoint identity** — point the eval at the real `mf_diffusion` checkpoint
   (`…_objmeanflow_bbmf_dit_…`), never the plain FM baseline in the same folder, else U3 just
   re-runs Gen12 under a new name (DISCUSS §3, §6d.4).
5. **Expectation management** — Gen13 already ran HardFlow+iMF natively and the *efficiency* thesis
   was refuted; U3 answers a different, cleaner **field-quality** question. Don't expect the HF arm
   to favor mean-flow on timing (every active step is a full net eval).
6. **Field-quality vs speed labeling** — the writeup MUST state the HF arm is a field-quality A/B,
   not a few-step-speed test (§2 scope note).

---

## 9. Open decisions for the user (before coding)

- **D1 — eval K for the HF arm.** Gen12 used K=10/20. Gen3v6's mean-flow deploy is few-step (K=1/2).
  Recommend running the HF arm at a **matched K** shared with the DPCC arms via the single override,
  and sweeping K∈{2,5,10} in separate runs. *Which K set do you want first?*
- **D2 — results path scheme.** (a) **Simple (recommended):** write `hardflow_new-*.npz` into the
  existing `results/halfspace_*/` dir next to `dpcc-*.npz` (one dir, both sets). (b) **Gen12-style:**
  port `hf_paths.py` (K/threshold/batch encoded in the dir name). *Simple, or Gen12-style?*
- **D3 — batch/candidate fan.** `hardflow.batch_size` default 1 (faithful; `-r/-c/-t` collapse to
  one candidate). Set 4 to make the selection suffixes meaningful (matches DPCC's mpc=4). *Default 1,
  or 4?*
- **D4 — variant set size.** Full 6-HF-variant parity matrix (with batch>1) vs the 2 non-redundant
  ones (`hardflow_new`, `hardflow_new-c-tightened`) at batch 1. *Which set?*

---

## 10. File-change summary (for the eventual coding pass)

| action | path |
|---|---|
| **NEW (verbatim + 1-line h=0)** | `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` |
| edit (export) | `flow_matcher_v3_meanflow/sampling/__init__.py` |
| **graft arm C** | `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` |
| edit (HF rows + repoint yaml) | `FM_v3_meanflow_test/load_results_flow_matching_v3_meanflow.py` |
| edit (arm C **+ repoint yaml** to the unified file) | `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` |
| NEW (gate) | `FM_v3_meanflow_test/gates_hardflow_meanflow.py` |
| NEW (opt) | `FM_v3_meanflow_test/hf_paths.py` *(only if D2=Gen12-style)* |
| **NEW (ONE unified eval config)** | `config/meanflow_projection_eval.yaml` (DPCC + HardFlow + `hardflow:` block) |
| edit (HF knobs) | `config/avoiding-d3il.py` → `plan_fm_v3_meanflow` (`flow_steps`, …) |
| **NEW (sbatch)** | `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` |
| unchanged | training, U2 backbone, **shared `config/projection_eval.yaml`**, `diffuser/` shim, DPCC arms |

## 11. Validation plan (cluster)

1. `py_compile` all touched files locally (no casadi/torch here).
2. **Gate G-h0**: on a real mean-flow checkpoint, assert `‖_predict_velocity(x,t,h=0) − v_fd‖` is
   tiny (finite-diff of the flow) → §2 holds empirically.
3. **Parity safeguard**: in the joint run, confirm `dpcc-c-tightened` and `hardflow_new-c-tightened`
   both reach 0 violations where each reaches the goal (shared tightened set).
4. Then read HF metrics (nfe/nlp_solves/avg_time) next to DPCC and write the U3 results insight,
   explicitly framed as a **field-quality** DPCC-vs-HardFlow comparison on the mean-flow checkpoint.

---

*No code has been written. This is the plan; on approval (and D1–D4 answered) I'll execute the
file-change list in §10, verbatim-copying the HF math per §2–§3.*
