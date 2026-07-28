# Gen3v6 U3 — CHANGELOG: HardFlow arm ported into Gen3v6 (in-folder addon)

Implements `PLAN_U3_hardflow_addon.md`. The Gen12 HardFlow constrained sampler is now an add-on
**arm C** inside the existing Gen3v6 folders, so one eval run emits **diffuser + DPCC + HardFlow**
on a mean-flow checkpoint, with HF metrics and set entries — via **one unified, Gen3v6-dedicated
YAML**.

**Bottom line (as required): the HardFlow math is identical to Gen12.** The core module is a
byte-for-byte copy; the ONLY change is the single velocity-query line, which uses the MeanFlow
identity `u(x,t,0)=v(x,t)` so the field handed to the NLP/endpoint is exactly what Gen12 feeds.

## Defaults chosen for the plan's open decisions (D1–D4)

Proceeded with sensible defaults (all overridable, documented below):
- **D1 (K):** matched-K for all arms; `flow_steps` added to the plan block = `flow_steps_v3` = **2**.
  Override with `HFFM_FLOW_STEPS=<K>` (forces K onto every arm at once).
- **D2 (results path):** simple — HF `.npz` files land in the **same** `results/halfspace_*/` dir as
  the DPCC `.npz`, one run = both sets. (No `hf_paths.py` port.)
- **D3 (batch):** `hardflow.batch_size = 1` (faithful, = Gen12 default; `-r/-c/-t` collapse to one
  candidate). Override with `HFFM_BATCH=4` for the DPCC-parity candidate fan.
- **D4 (variant set):** full 6-HF-parity matrix + full DPCC set in the unified YAML.

## Files changed

### NEW — `flow_matcher_v3_meanflow/sampling/hardflow_projection.py`
**Verbatim copy** of `flow_matcher_v3_hardflow/sampling/hardflow_projection.py`
(`TrajectoryLayout`, `HardFlowNLP` casadi/ipopt, `HardFlowSampler` incl. the fix_7 DPCC-parity
batching, `HardFlowPolicy`, `resolve_activation_threshold`). `diff` vs the Gen12 source is **only**
the velocity query in `_velocity_batch` (+ its comment):

```python
# Gen12 (instantaneous-v FM model):
v = self.model._predict_velocity(traj, cond, t, returns=returns)
# Gen3v6 U3 (mean-flow model, queried at h=0 ⇒ u(x,t,0)=v EXACTLY):
v = self.model._predict_velocity(traj, cond, t, h=torch.zeros_like(t), returns=returns)
```

`h=0` is passed **explicitly** (not relying on the `h=None→0` backbone default) so a future default
change can't silently reintroduce the interval-average field (the §2 wrong-field bug). Its relative
imports (`..models.helpers.apply_conditioning`, `.policies.Trajectories`) and `diffuser.utils`
resolve unchanged in the Gen3v6 package — `constraints_helpers.py` is already byte-identical between
the two folders, so the NLP's constraint formulation drops in risk-free.

### EDIT — `flow_matcher_v3_meanflow/sampling/__init__.py`
Export `TrajectoryLayout, HardFlowNLP, HardFlowSampler, HardFlowPolicy, resolve_activation_threshold`
(matches Gen12's `sampling/__init__.py` exactly).

### EDIT — `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` (arm C graft)
- Import `HardFlowPolicy, resolve_activation_threshold`.
- **Repoint** the config from the shared `config/projection_eval.yaml` → the unified
  `config/meanflow_projection_eval.yaml` (with an optional `--config` override).
- Parse the `hardflow:` block + env knobs (`HFFM_BATCH`, `HFFM_ACT_THRESHOLD`, `HFFM_FLOW_STEPS`)
  — verbatim Gen12 semantics.
- Resolve **matched-K** `flow_steps` after model load (env > plan `flow_steps` > native
  `flow_steps_v3`); an explicit override is also applied to the native sampler so all arms share K.
- In the variant loop: `is_hardflow = variant.startswith('hardflow')` branches policy construction
  — arm C builds `HardFlowPolicy` (DPCC-parity `-r/-c/-t` selection suffix parse, `dynamics_mode=
  deriv`, `linear_dynamics=None`) at `hf_batch_size`; arms A/B keep the existing `Projector`+`Policy`
  at `args.batch_size`. Per-arm `batch_size` now flows into `policy(...)` and the RTRecorder.
- Accumulate HF metrics (`nlp_solves_total`, `nlp_failures_total`, and `nfe` from `policy.nfe`),
  print an `[hardflow] NFE=… NLP solves=…` summary line, and persist
  `is_hardflow / nfe_total / nlp_solves_total / nlp_failures_total / hf_batch_size /
  hf_act_threshold` into each variant's `.npz`. Arms A/B are untouched (all HF metrics stay 0).

### EDIT — `FM_v3_meanflow_test/load_results_flow_matching_v3_meanflow.py`
Repoint to the unified YAML; sum & print the HF compute metrics (`NFE / NLP solves / NLP failures`)
for `hardflow_new-*` rows when present in the `.npz` (older npz without them are skipped). DPCC rows
are aggregated exactly as before.

### NEW — `config/meanflow_projection_eval.yaml` (the ONE unified config)
Copied from `config/hardflow_projection_eval.yaml` (so the `hardflow:` block + constraint geometry
are verbatim), retitled for Gen3v6, `projection_variants` set to the **union**: `diffuser` +
`dpcc-{r,c,t}[-tightened]` + `hardflow_new-{r,c,t}[-tightened]`. This is the single source of truth
for Gen3v6 eval. **The shared `config/projection_eval.yaml` (read by ~50 files across every
generation) is left untouched**, so no other generation is affected. A DPCC-only run = list only
`dpcc-*`/`diffuser` variants here (the HF arm never fires without a `hardflow*` entry).

### EDIT — `config/avoiding-d3il.py` → `plan_fm_v3_meanflow`
Added `'flow_steps': 2` (the HF-arm Euler K), kept equal to `flow_steps_v3` so all three arms run at
matched K; override with `HFFM_FLOW_STEPS`.

### NEW — `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh`
Copy of `eval_meanflow.sh` (EGL/GPU isolation guard, conda `FMPCC`, `MUJOCO_GL=egl` **all
unchanged**) + the HF env knobs (`HFFM_BATCH=1`, `HFFM_ACT_THRESHOLD=0.5`, optional
`HFFM_FLOW_STEPS`). Runs the unified eval (reads `config/meanflow_projection_eval.yaml` by default).
The DPCC-only `eval_meanflow.sh` is untouched.

### NEW — `FM_v3_meanflow_test/gates_hardflow_meanflow.py`
Pre-flight gates for the one novel thing (Gen12's own gates cover the copied math):
- **H0** — imports resolve + casadi availability report.
- **H1** — the correctness pin (casadi/checkpoint-free): a recording stub asserts the ported
  `_velocity_batch` queries the model at `h==0` (a zeros tensor of shape `(B,)`), i.e. the
  `u(x,t,0)=v` identity is actually wired. Fails if anyone reverts the explicit `h=0`.
- **H2** — harness stub for the numeric `u(x,t,0)≈v` check on a real checkpoint (`--checkpoint`,
  cluster only; wire to the eval loader there).

## What was deliberately NOT changed
- The shared `config/projection_eval.yaml`, all other generations, Gen3v6 **training**, the U2
  backbone, the `diffuser/` shim (HF modules are imported directly by the test driver, not via
  `import_class`, and the checkpoint class still resolves through the fix_1 shim).
- The HardFlow **math** — NLP, endpoint identity `x₁=x_τ+(1−τ)v`, fix_7 batching, tightening,
  selection, metrics — all verbatim Gen12.

## Validation status
- **Local (this container):** `python3 -m py_compile` passes on all touched `.py`; the new HF module
  `diff`s to Gen12 as exactly the one `h=0` line; the unified YAML's variant/`hardflow:` blocks are
  structurally intact. No torch/casadi/yaml here ⇒ **no numerical run possible locally.**
  *(Pre-existing `\pm` SyntaxWarning in `load_results` is untouched LaTeX, harmless.)*
- **Cluster (must run on i6-gpu-1):**
  1. `python FM_v3_meanflow_test/gates_hardflow_meanflow.py` → H0+H1 PASS (the h=0 pin).
  2. `./submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` on a **real mean-flow
     checkpoint** (`…_objmeanflow_bbmf_dit_…`, NOT the plain FM baseline).
  3. **Parity safeguard:** in the joint run, `dpcc-c-tightened` and `hardflow_new-c-tightened` share
     the tightened set → both should reach 0 violations where each reaches the goal (a built-in
     port-correctness probe).
  4. Then read HF metrics next to DPCC and write the U3 results insight — framed as a **field-quality**
     DPCC-vs-HardFlow comparison (the HF arm runs the mean-flow model as a plain FM field at h=0, so
     it is NOT a few-step-speed test).

## How to run
```bash
# gate (h=0 correctness pin) — cluster, casadi not needed for H0/H1
python FM_v3_meanflow_test/gates_hardflow_meanflow.py

# unified eval: diffuser + DPCC + HardFlow, all arms at matched K, seed 6
./submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
#   knobs: HFFM_BATCH=4 (candidate fan) · HFFM_ACT_THRESHOLD=0.5 · HFFM_FLOW_STEPS=2

# aggregate (prints DPCC rows + [hardflow] NFE/NLP metrics)
python FM_v3_meanflow_test/load_results_flow_matching_v3_meanflow.py
```
