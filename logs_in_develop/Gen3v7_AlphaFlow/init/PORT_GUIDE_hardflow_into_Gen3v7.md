# Porting the HardFlow arm into Gen3v7 (α-Flow) — agent guide

Short brief for whoever does the port. Gen3v6 U3 already did this for MeanFlow; copy that, not
Gen12's. Read [`Gen3v6 fix_4`](../../Gen3v6_MeanFlow/fix_4/CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md)
first — it is the list of mistakes already paid for.

**Status:** Gen3v7 has **zero** HardFlow wiring today. Everything below is new.

## Files

**Source of truth (copy from):**

| file | role |
|---|---|
| `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` | the whole arm (NLP + sampler + policy), post-fix_4 |
| `FM_v3_meanflow_test/gates_hardflow_meanflow.py` | pre-flight gates H0/H1/H3 |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | arm-C branch in the eval driver (~L318-345) |
| `config/meanflow_projection_eval.yaml` | arm list + `hardflow:` knob block |
| `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` | env-knob block + submit entry |

**Targets (create/edit):**

- `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` ← copy
- `flow_matcher_v3_alphaflow/sampling/__init__.py` ← add the exports (currently only `Projector`, `Policy`)
- `FM_v3_alphaflow_test/gates_hardflow_alphaflow.py` ← copy + retarget
- `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` ← add arm C
- `config/alphaflow_projection_eval.yaml` ← new
- `config/avoiding-d3il.py` → `plan_fm_v3_alphaflow` (L1307) ← **see caution 4**
- `Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh` ← new

## Principles / math

- **Only `hardflow_new` is portable.** HardFlow's other three modes compile the network *into* the
  NLP via l4casadi and are architecture-locked. `hardflow_new` calls the net as a black box
  `v = f(x, t)` outside the solver, so any velocity field drops in.
- **Per ODE step k of K:** reference step `x_ref = x_k + v_k·dt` → terminal predict
  `x1_ref = x_ref + (1−τ_{k+1})·f(x_ref, τ_{k+1})` → prox-NLP keeping `x1` near `x1_ref` subject to
  the constraints → pull-back `x_{k+1} = x_ref + τ_{k+1}·(x1_proj − x1_ref)`. The prox weight
  carries a `τ²` factor, so early steps are nudged and late steps pulled hard onto the feasible set.
- **The `h=0` identity is what makes this legal on a two-time model.** α-Flow's net emits the
  interval average `u(x, t, h)`; the NLP needs an instantaneous velocity. At `h=0` the field is
  grounded: `u(x,t,0) = v(x,t)`. Pass `h=torch.zeros_like(t)` **explicitly** — never rely on a
  `h=None` backbone default.
- **α-Flow trains that anchor directly**, so the identity is well-supported here — arguably better
  than MeanFlow's. `af_diffusion.py:694`: `fm_mask = torch.rand(B) < self.af_ratio_fm` forces a
  fraction of the batch to `r == t` with `u_tgt = v`; default `af_ratio_fm = 0.5` (L94).
- **The NLP is built from FMPCC's `constraint_list`** — the same list DPCC's `Projector` consumes —
  never HardFlow's own `avoiding_geometry.py`. Otherwise arms B and C enforce different feasible
  sets and the comparison is void.
- **Selection rules** `-r`/`-t`/`-c` mirror DPCC's so arms B and C stay comparable; they only differ
  at `mpc > 1`.

## Cautions

1. **Initial noise scale — the one that already bit us.** α-Flow samples at **σ=1.0**
   (`af_diffusion.py:260`, *"sigma=1.0 to match q_sample training noise"*). Pass
   `init_noise_scale=1.0`.
   ⚠️ **Do NOT read the scale from `flow_matcher_v3_alphaflow/models/diffusion.py:183,302`** — that
   is the legacy FMv3ODE class living in the same folder and still on `0.5 * randn`. Reading the
   wrong one is exactly how Gen3v6 broke: arm C at σ=0.5 against a σ=1.0 checkpoint, silently, for
   an entire K-sweep. Post-fix_4 the argument is **required** on `HardFlowSampler` and
   `gate_h3` pins it numerically — keep both properties.
2. **Verify `af_ratio_fm > 0` in the config that trained the checkpoint you evaluate.** If it is 0,
   the `h=0` anchor was never trained and the whole port rests on an untrained corner of the field.
3. **`_predict_velocity` signature must match.** `af_diffusion.py:221` is
   `_predict_velocity(self, x, cond, t, h=None, returns=None)` — same as MeanFlow's, so
   `_velocity_batch` copies unchanged. Confirm before assuming.
4. **Matched-K plumbing is missing in `plan_fm_v3_alphaflow`.** It currently hardcodes
   `flow_steps_v3: 2` with no `flow_steps` key. Mirror `plan_fm_v3_meanflow` (L1255-1256):
   ```python
   'flow_steps_v3': int(os.environ.get('HFFM_FLOW_STEPS', 2)),
   'flow_steps':    int(os.environ.get('HFFM_FLOW_STEPS', 2)),
   ```
   Both keys, because K must flow through the **config** so `args.savepath` encodes `_K{K}_`. Skip
   this and every K in a sweep overwrites the same results directory.
5. **`-c` (minimum-projection-cost) is degenerate on both engines.** It selects whichever candidate
   needs least intervention, which is a motionless one whenever the field or the NLP produces one.
   Expect it to fail; do not treat a `-c` collapse as a port bug without checking the candidate fan
   first.
6. **Don't touch Gen12's copy.** `flow_matcher_v3_hardflow/` keeps `0.5 * randn` and that is
   **correct for Gen12** (its base sampler is 0.5, all its arms agree). Sibling isolation — each
   generation owns its copy.
7. **`goal_dim == 0` is asserted.** Fine for avoiding-d3il; a goal-conditioned env needs the goal
   columns carried through the dof vector.
8. **Two different `dt`s.** The NLP's environment `dt` (for the `deriv` constraint) and the ODE's
   `dt = 1/K` are separate. Keep them in separate scopes.

## Before believing any number

```bash
python FM_v3_alphaflow_test/gates_hardflow_alphaflow.py   # H0 imports, H1 h==0, H3 noise scale
```
Then a matched-K sweep (`HFFM_FLOW_STEPS` ∈ {1,2,5,20}, `HFFM_BATCH=4`, `HFFM_ACT_THRESHOLD=0.5`).
Arm B (DPCC) is the control: if it moves relative to a DPCC-only run, the port leaked into the
shared path.
