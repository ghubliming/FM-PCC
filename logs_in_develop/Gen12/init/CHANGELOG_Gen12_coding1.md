# CHANGELOG — Gen12 (HardFlow → FMv3), coding pass 1

**Date:** 2026-07-23 · **Type:** implementation changelog · **Status:** code complete, **NOTHING RUN**
**Implements:** [`PLAN_Gen12_hardflow_into_fmv3ode.md`](PLAN_Gen12_hardflow_into_fmv3ode.md)
**Base:** copy-modify of `flow_matcher_v3/` ↔ `FM_v3_test/` (the plan's §2 table)
**Nothing committed.** No local execution — this container has no Python deps; all gates run on the cluster.

---

## 0. TL;DR

Gen12 adds a **third guidance arm** to FMPCC: HardFlow's `hardflow_new` in-loop constrained
sampler, running on FMPCC's own already-trained FMv3 checkpoint. **No training, no retraining,
no new checkpoint** — that is the premise the plan verified and it survived contact with the code.

The port is ~520 lines in one new module. The three things that took the most care were **not**
the sampler loop:

1. **The NLP is built from FMPCC's `constraint_list`, not HardFlow's hard-coded geometry.**
   Upstream reads obstacle/boundary geometry from `hardflow/utils/avoiding_geometry.py`; FMPCC
   builds it from `config/*_eval.yaml`. Porting the sampler *with* upstream's geometry would have
   left arms B and C enforcing **different feasible sets**, which quietly voids §5's comparison.
2. **Matched K is structural, not a convention.** One YAML key sets K for all three arms, applied
   after checkpoint load, and the results directory is named after it.
3. **The plan's §1.2 description of the τ schedule is wrong** — see §8. The code is faithful to
   upstream; the *description* needed correcting, and a gate now asserts the corrected version.

---

## 1. Files created

| path | origin | note |
|---|---|---|
| `flow_matcher_v3_hardflow/` | copy of `flow_matcher_v3/` | model package; `__pycache__` purged |
| `FM_v3_hardflow_test/` | copy of `FM_v3_test/` | test/eval package |
| `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | **new**, 517 L | the port (§3) |
| `FM_v3_hardflow_test/gates_hardflow.py` | **new**, 365 L | G0–G3 seam assertions (§7) |
| `FM_v3_hardflow_test/fit_dynamics_fmv3.py` | port of `HardFlow/run/fit_dynamics.py`, 215 L | §5 |
| `config/hardflow_projection_eval.yaml` | **new** | Gen12 eval config (§6) |
| `Slurm_Codes/sbatch/hardflow_fmv3/` | **new**, 5 scripts | §9 |
| this file | **new** | changelog |

**Untouched (hard rule, verified with `git diff --stat`, empty output):**
`flow_matcher_v3/`, `FM_v3_test/`, `HardFlow/`.
The only pre-existing file modified is `config/avoiding-d3il.py` — **48 insertions, 0 deletions**,
verified purely additive.

## 2. Files modified inside the new folders

### 2.1 Renames and deletions

| old | new |
|---|---|
| `eval_FM_v3.py` | `eval_FM_v3_hardflow.py` |
| `load_results_FM_v3.py` | `load_results_FM_v3_hardflow.py` |
| `train_FM_v3.py` | **deleted** |

The package has **zero absolute self-imports** (everything inside `flow_matcher_v3/` is relative),
so the copy needed no internal renaming at all — only the three `import flow_matcher_v3.…` lines
in the test scripts. Verified: no stale `flow_matcher_v3.` reference remains in either new folder.

`train_FM_v3.py` is deleted rather than carried, per PLAN §2: *"No training script is needed — that
is the whole point."* Keeping a train entry point that must never be used is an invitation to use
it. Recoverable — it is untouched in `FM_v3_test/`.

---

## 3. The port — `flow_matcher_v3_hardflow/sampling/hardflow_projection.py`

Four classes:

| class | role |
|---|---|
| `TrajectoryLayout` | index bookkeeping for the s0-free dof vector |
| `HardFlowNLP` | the CasADi prox-NLP, built once, re-solved per ODE step |
| `HardFlowSampler` | the constrained ODE loop; the model enters ONLY as `v = f(x, t)` |
| `HardFlowPolicy` | drop-in for `sampling.policies.Policy` so the eval loop is shared |

### 3.1 Layout (PLAN §3.2) — verified, not assumed

The plan demanded a layout assertion "first". It exists twice: as `gate_g0`, and re-derived
independently in plain Python during this pass. Result:

```
dof = H*T − state_dim = 8*6 − 4 = 44                    (identical formula both repos)
dof = [ a_0 | a_1 s_1 | a_2 s_2 | … | a_7 s_7 ]
Gen12 state_index(t)  ==  HardFlow's 2*action_dim + i*(action_dim+state_dim),  i = t−1
   t=1 → 4      t=2 → 10     t=3 → 16 … t=7 → 40        ✅ all 7 match
transition columns   [ vx vy | x_des y_des x y ]        ✅ identical in both repos
```

HardFlow reads `x, y` at obs offsets `+2, +3`; FMPCC's `observation_indices` say `x: 2, y: 3`.
**The two layouts coincide exactly.** The transposition trap the plan warned about does not
materialise — but it is now asserted at runtime rather than believed.

### 3.2 Time direction (PLAN §3.3)

FMv3 `p_sample_loop`: `t = k/K` increasing, `x = x + v·dt`, init `0.5·randn`.
HardFlow `hardflow_new_forward`: `t_k = k·dt` increasing, `x = x + v·dt`, init `randn`.
**Same direction, τ = 0 (noise) → 1 (data).** So `x̂1 = z + (1−τ)·v` is the correct terminal
predictor in both. `gate_g1` asserts it two ways: numerically with a constant field (where the
identity is exact at every step) and by reading the convention straight out of the shipped
`p_sample_loop` / `p_mean_variance` source.

### 3.3 The constraint translation — the largest deviation from upstream

Upstream builds its NLP from `hardflow/utils/avoiding_geometry.py` (hard-coded pillar centres,
novel-obstacle radius, boundary lines, an optional quadrilateral). FMPCC builds constraints from
YAML into a `constraint_list` that the DPCC `Projector` consumes. `HardFlowNLP` consumes **that
same list**, reproducing `Projector`'s exact semantics:

| spec | steps constrained | notes |
|---|---|---|
| `('ineq', (C_row, d))` | 1 … H−1 | `skip_initial_state=True`, as DPCC |
| `('eq', (C_row, d))` | 1 … H−1 | supported, unused by avoiding |
| `['lb'/'ub', vec]` | action dims 0 … H−1, obs dims 1 … H−1 | mirrors `SafetyConstraints` |
| `['sphere_outside'/'sphere_inside', …]` | 1 … H−1 | quadratic form, as `ObstacleConstraints` |
| `('deriv', [x_idx, dx_idx])` | 0 … H−2 | `x[t+1] = x[t] + dt·dx[t]` |

All constraints are written on the **unnormalized** transition vector, with the affine
unnormalization folded into the CasADi expression — the same algebra `Projector` does numerically.

> ⭐ **Why this matters more than fidelity to upstream:** PLAN §5's question is *"does in-loop
> constrained sampling beat post-hoc projection at equal compute?"*. That question is only
> answerable if arms B and C enforce the **same feasible set**. With upstream's geometry they
> would not: HardFlow's pillar radii are `0.03/0.025`, FMPCC's yaml obstacle for `both-hard` is
> radius `0.08` at `[0.5, −0.09]`, and FMPCC additionally enforces halfspaces, velocity bounds and
> Euler kinematics that upstream has no notion of. A "faithful" port would have measured two
> different experiments.

The HardFlow-faithful path is still available: `dynamics_mode: linear_fit` swaps the `deriv`
kinematics for the fitted `s' = A s + B a + c` plus normalized input saturation (§5).

### 3.4 The sampler loop

Per ODE step `k` of `K`, exactly as PLAN §1.2:

```python
v_k     = f(x_k, τ_k);          x_ref  = x_k + v_k·dt
v_next  = f(x_ref, τ_{k+1});    x1_ref = x_ref + (1 − τ_{k+1})·v_next
x1_proj = NLP.solve(x1_ref, τ_{k+1})
x_k     = x_ref + τ_{k+1}·(x1_proj − x1_ref)
```

`activation: late` skips the projection for `k < K//2`, as upstream.

At the final step `τ_{k+1} = 1`, so `x1_ref = x_ref` and the update collapses to
`x_K = x1_proj` — **the returned plan IS the projected terminal trajectory**, hence feasible by
construction (modulo IPOPT tolerance). `gate_g3` asserts this end-to-end.

### 3.5 NFE accounting

Arm C costs **2 network evaluations per ODE step** (reference + terminal prediction) and **K NLP
solves per plan**. Arm A/B cost 1 eval per step. This is counted for real in the sampler and
computed analytically for arms A/B (`flow_steps × plan_calls × batch_size`) so the two are
reported in the same units — PLAN §5 requires NFE and NLP solves in the results table, and a 2×
NFE gap is not something to discover after the fact.

---

## 4. Deliberate deviations from upstream `hardflow_new`

| # | upstream | Gen12 | why |
|---|---|---|---|
| 1 | NLP from hard-coded `avoiding_geometry.py` | NLP from FMPCC's `constraint_list` | §3.3 — otherwise arms B and C measure different experiments |
| 2 | `warmstart()`: full K-step ODE over `warmstart_batch` candidates, best picked by a value model | **removed** | FMPCC has no value model. Upstream's warmstart only supplied (a) the `s0` parameter — which is the MPC conditioning, known exactly — and (b) a noise seed. Dropping it removes K wasted NFE and makes the compute comparison honest |
| 3 | initial noise `randn` | `0.5·randn` | matches FMv3's own `p_sample_loop`, so arms A/B/C start from the same distribution |
| 4 | `flow_model(x, t)` | `model._predict_velocity(x, cond, t, returns)` | FMv3's UNet takes `(x, cond, time)`; `_predict_velocity` is the right black-box seam and also carries the (inert) returns-CFG branch |
| 5 | `print("Norm of Control Inputs: …")` every plan | dropped | ~7 replans × n episodes of console noise (PLAN §3.5) |
| 6 | `sqrt((x−cx)² + (y−cy)²) ≥ r` | `(x−cx)² + (y−cy)² ≥ r²` | equivalent, smoother for IPOPT, and matches DPCC's own quadratic obstacle form |
| 7 | `batch_size == 1` hard assert | loop over candidates | §10.2 |

## 5. `FM_v3_hardflow_test/fit_dynamics_fmv3.py` (PLAN §3.1)

Refits `s' = A s + B a + c` on **FMv3's** `SequenceDataset` and normalizer, writing to
`logs/<env>/dynamics_gen12/linear_model_H{h}_mpl{mpl}.npz`.

Changes from `HardFlow/run/fit_dynamics.py`:

- **Episode-level held-out split** (default 90/10) with reported R²/RMSE and a printed
  `GATE (held-out R2 > 0.99)` verdict. Upstream reports **train** error only. A window-level split
  at H=8 shares 7 of 8 frames between neighbours — the leak Gen3v6's changelog §10 flagged — so
  the split is by episode.
- **The .npz stores the normalizer limits** (`obs_mins/maxs`, `action_mins/maxs`), and
  `load_linear_dynamics()` **raises** if they do not match the dataset in use. This is the runtime
  half of §3.1: a mismatched `A, B, c` produces a converging NLP enforcing wrong physics, which is
  invisible in the results, so it is made a hard error rather than a docs warning.
- `sklearn` dependency dropped (`np.linalg.lstsq` + hand-rolled R²).
- Upstream's `verify_with_env()` (reset → `set_state` → step, comparing against the simulator) is
  **not ported** — it relies on `gym.make(env).set_state(...)`, which D3IL's `ObstacleAvoidanceEnv`
  does not expose the same way. The offline held-out check is the gate; §11 lists this as an
  acknowledged gap, not a silent one.
- Refuses to overwrite an existing `.npz` without `--force`.

**Note:** with the default `dynamics_mode: deriv` this file is **not on the critical path** — it is
only needed if the YAML is switched to `linear_fit`. It ships anyway because PLAN §6 makes the
held-out check part of "the port is correct".

## 6. `config/` additions

### 6.1 `config/avoiding-d3il.py` — additive only (48 lines, 0 deletions)

New `plan_fm_v3_hardflow` **eval-only** block. Verified mechanically (stubbing `watch`/`yaml`, no
torch needed) that its `diffusion_loadpath` renders **token-for-token identical** to the
`flow_matching_v3` train block's `exp_name`, and to `plan_fm_v3`'s loadpath:

```
flow_matching_v3/H8_K10_Dmodels.diffusion.GaussianDiffusion
```

i.e. **Gen12 reads exactly the checkpoint Gen3 trained**, and writes to its own
`plans/flow_matching_v3_hardflow/…` tree. An architecture-key parity diff against `plan_fm_v3`
reports no mismatches.

> 🔴 The block carries a boxed warning that `flow_steps_v3` here is the **checkpoint's** K (it is
> part of the load path) and must not be edited to sweep the eval K — that would point at a
> directory that does not exist. The eval K is a separate knob (§6.2).

### 6.2 `config/hardflow_projection_eval.yaml` — a NEW file, not an edit

`config/projection_eval.yaml` is shared by every generation. Appending `hardflow_new` to its
`projection_variants` would silently add an arm to Gen3/Gen3v2/Gen7 runs. Gen12 gets its own copy.

Two additions beyond the copy:

- **`flow_steps: 10`** — ⭐ overrides `flow_steps_v3` on the loaded model for **every arm at once**,
  applied *after* load so it never touches `diffusion_loadpath`. This is PLAN §5's matched-K
  requirement made structural. Sweep `{2, 5, 10}`.
- **`hardflow:` block** — arm C's knobs: `batch_size`, `dynamics_mode`, `linear_dynamics_path`,
  `reg_scale`, `activation`, `ipopt_print_level`, `casadi_print_time`.

`projection_variants` is cut down to the three arms of PLAN §5, run in **one process** so K, seeds
and env resets are shared by construction:

| arm | variant | |
|---|---|---|
| A | `diffuser` | unguided — field quality floor |
| B | `dpcc-c-tightened` | the incumbent |
| C | `hardflow_new` | the contribution |

## 7. `FM_v3_hardflow_test/gates_hardflow.py` — PLAN §4 step 2

Runs with **no checkpoint and no dataset** — deliberately. These gates test seams, and a real
checkpoint would hide a seam bug behind plausible-looking trajectories.

| gate | asserts |
|---|---|
| **G0** layout | dof indexing == HardFlow's formula for all 7 constrained steps; `to_dof/from_dof` round-trips exactly; prints the index table for both repos |
| **G1** direction | constant-field Euler identity `x_K = x_0 + v` and exactness of `x + (1−τ)v` at every step; plus a source-level check that `p_sample_loop` uses increasing `t` and `x + v·dt` |
| **G2** NLP | a reference parked **on the obstacle centre** comes back feasible; solver failure count is 0; **and** the τ-invariance finding of §8 |
| **G3** end-to-end | full sampler on a stub field: shape, feasibility of the returned plan, `NFE == 2K`, `NLP solves == K` |

Exits non-zero on any failure, so the pipeline's `afterok` chain cancels the rest (PLAN §4:
*"Do not proceed past a failing step"*).

## 8. ⚠️ A correction to the plan — the τ² factor is inert

PLAN §1.2 describes step 4 as *"pull-back — blend the projected terminal back, weighted by τ²
(`oc_control_cost … * self.oc_t_param**2`)"*. Those are two different objects, and the
parenthetical does not do what the sentence says.

With `objective == ""` (the mode Gen12 ports) upstream's cost is

```python
0.5 * hardflow_reg_scale * sumsqr(X - X_ref) * t_param**2
```

which is a **positive scalar multiplying the only term in the objective**. It cannot move the
argmin. The prox solve returns the same `x1_proj` at τ = 0.05 and at τ = 1.0.

The actual τ-gating is the **linear** factor in the pull-back,
`x_next = x_ref + τ·(x1_proj − x1_ref)`. That is what makes early steps a nudge and the last step
(τ = 1) an exact landing on the projected terminal.

The τ² factor **is** live in upstream's `objective == "distance"` mode, where it trades off against
`oc_distance_objective`. Gen12 keeps it in the code — faithful, and it becomes meaningful the
moment anyone adds a goal term — but `gate_g2` asserts the invariance so the next reader is not
misled, and both the code comment and this note say so explicitly.

**Consequence for §5's write-up:** do not describe arm C as "τ²-annealed guidance". It is a
hard constraint at every step with a linearly-annealed pull-back.

## 9. `Slurm_Codes/sbatch/hardflow_fmv3/` — 5 scripts

`gates_hardflow_fmv3.sh`, `fit_dynamics_hardflow_fmv3.sh`, `eval_fmv3_hardflow_job.sh`,
`load_results_hardflow_fmv3.sh`, `hardflow_fmv3_pipeline.sh`.

The conda prologue, `PYTHONPATH`, `MUJOCO_GL=egl`, the **GPU/EGL isolation guard**, the W&B key
file, the `trap on_exit` and the `latest.log` symlink are byte-identical to the AlphaFlow set.
`--time` is ~2× expected per job (24 h for eval, 1–2 h for gates/fit). No tqdm, no live bars.

**There is no training job, by design.** The pipeline chains gates → eval → load_results with
`afterok`, so a failing gate cancels everything downstream.

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/hardflow_fmv3_pipeline.sh   # full chain
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/gates_hardflow_fmv3.sh      # gates only
HFFM_FLOW_STEPS="10" ./Slurm_Codes/submit.sh …/eval_fmv3_hardflow_job.sh             # one K
FORCE_OVERWRITE=1    ./Slurm_Codes/submit.sh …/eval_fmv3_hardflow_job.sh             # re-run
```

The eval job's K grid (`2 5 10`) is **built into the script**, not left as a config edit somebody
has to remember — the same lesson AlphaFlow's sbatch encodes after Gen13 fix_7.

## 10. Decisions the plan left open (§8), now made

### 10.1 Env target — `avoiding-d3il` only

As the plan assumed. `HardFlowPolicy` **asserts `goal_dim == 0`** rather than silently mishandling
a goal-conditioned env; extending to `aligning` means carrying the goal columns through the dof
vector, which is a real change, not a config flip.

### 10.2 Batch size (PLAN §3.4) — **faithful batch-1 is the headline**

`hardflow.batch_size: 1`. Upstream hard-asserts it, and it is the honest comparison against
Gen13's numbers. The sampler nonetheless **generalises**: `batch_size > 1` solves one independent
NLP chain per candidate (wall time scales linearly). Candidate selection then falls back to index
0, because arm C has no per-candidate projection cost to rank by — DPCC's `minimum_projection_cost`
has no analogue when the constraint is enforced *during* sampling.

> 🔴 **This is the one decision most likely to be contested.** Arm B runs at `batch_size: 4` with
> candidate selection; arm C at 1. If arm B wins, "the candidate fan explains it" is a live
> alternative explanation. The counter-experiment is one YAML line (`hardflow.batch_size: 4`) and
> should be run before any strong claim. Flagged rather than resolved — it is the user's call
> which framing is the headline.

### 10.3 `MASTER_TEST_HISTORY.md`

The Gen11+ → Gen12 relabel has **already been applied** by someone; the row exists and points at
this plan. Only the two "Planned" cells are now stale. **Not self-applied** — see §12.

## 11. Traps and gaps, stated

- ⚠️ **Nothing has been executed.** No torch, no casadi, no numpy in this container. Every claim
  above is either static analysis, a pure-Python re-derivation (the layout, §3.1), or a stubbed
  config render (§6.1). The gates exist precisely because the runtime claims are unverified.
- ⚠️ **`verify_with_env` not ported** (§5). The linear model is validated against held-out *data*,
  not against the simulator. If `linear_fit` mode is ever used for a headline number, that gap
  should be closed first.
- ⚠️ **IPOPT feasibility ≠ environment feasibility.** The returned plan is feasible for the
  *constraint model*; the MPC loop still measures real violations against the env. Arm C is not
  guaranteed 0 violations and should not be described as such before the numbers exist.
- ⚠️ **NLP failure counting is per-plan, not fatal.** Like upstream, a failed solve keeps the last
  IPOPT iterate and continues. The count is aggregated into the results `.npz`
  (`nlp_failures`) — **read it**; a run with many failures is not a run whose feasibility claims
  hold.
- ⚠️ **PLAN §5.1 stands:** rank arms by task success, not smoothness. Post-projection roughness
  measures the NLP, not the model.
- Two inherited bugs were fixed in the Gen12 copies only (originals untouched): `plt.subplots`
  without `squeeze=False` crashes on a single variant/trial, and `load_results` reported the
  **last** seed's `avg_time` as the variant mean. The Gen12 aggregator also no longer writes its
  PNGs into `FM_test/` — a different generation's folder.

## 12. MASTER_TEST_HISTORY row — prepared, NOT applied

Per repo convention this was **not** written into the master index. The existing
**Gen12 (HardFlow → FMv3ODE)** row needs only its two "Planned" cells replaced:

```markdown
| **Gen12 (HardFlow → FMv3ODE)** <br>*(was "Gen11+ / X")* | [flow_matcher_v3_hardflow/](../flow_matcher_v3_hardflow) | [FM_v3_hardflow_test/](../FM_v3_hardflow_test) | July 2026 | … (existing text) … Plan: [`Gen12/init/`](./Gen12/init/PLAN_Gen12_hardflow_into_fmv3ode.md) · Coding pass 1: [`CHANGELOG_Gen12_coding1.md`](./Gen12/init/CHANGELOG_Gen12_coding1.md) | working on |
```

## 13. Status vs the plan's implementation order (§4)

- [x] **step 1** — both sibling folders copied verbatim, `__pycache__` purged, zero stale refs
- [x] **step 2** — layout + direction assertions written (`gates_hardflow.py` G0/G1); layout also
      re-derived independently in pure Python here (§3.1). **Not run.**
- [x] **step 3** — `fit_dynamics_fmv3.py` ported with an episode-level held-out gate. **Not run.**
- [x] **step 4** — `hardflow_new_forward` + `hardflow_formulate` ported; model is a black-box
      callable. **Not run.**
- [x] **step 5** — `hardflow_new` wired alongside the `Projector` variants; console silenced at
      both switches (§3.5). **Not run.**
- [x] **step 6** — sbatch entries + provenance-safe naming (`K{K}_n{n}` dirs, `FORCE_OVERWRITE`)
- [ ] **step 7** — full eval. Cluster work.
- [x] changelog written (this file)
- [ ] **MASTER_TEST_HISTORY row handed to the user, not self-applied** — §12
- [x] nothing committed

## 14. Next steps

1. Sync to the cluster.
2. `./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/gates_hardflow_fmv3.sh` — G0–G3.
   **Do not proceed past a failure.** G2/G3 need `casadi` in the FMPCC env; if it is missing,
   that is the first thing to install (`HardFlow/requirements.txt` has it).
3. Smoke: `n_trials: 1`, one halfspace variant, `flow_steps: 5` — confirm arm C completes an
   episode and the console is quiet.
4. Full run: `hardflow_fmv3_pipeline.sh`, K ∈ {2, 5, 10}, seeds 6–10, then raise `n_trials`
   toward PLAN §5's n ≥ 100.
5. Decide §10.2 before writing up: re-run arm C at `batch_size: 4` if arm B wins.
6. **A clean negative is a result** (PLAN §6). If arm C matches arm B at higher cost, that is
   consistent with Gen13's finding that the projection dominates outcomes regardless of the field,
   and should be written up as such rather than tuned away.
