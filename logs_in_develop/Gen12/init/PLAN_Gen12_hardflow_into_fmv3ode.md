# Gen12 — PLAN: port HardFlow's constrained sampler into FMv3ODE

**Date:** 2026-07-22 · **Status:** plan only, **no code written**
**Scope:** add HardFlow's eval-time constrained sampling as a new guidance variant inside a copy-modified FMv3ODE sibling pair. **No retraining.** Reuse the existing FMv3ODE checkpoint.
**Audience:** the implementing agent. Follow §4 in order; §3 lists the traps that will silently produce wrong results if ignored.
**Related:** `MASTER_TEST_HISTORY.md` row *"Gen11+ / X — Integrating /workspaces/HardFlow into FMPCC"* (this work; the row predates the Gen12 label) · Gen13 = HardFlow+iMF **inside** the vendored HardFlow repo, the opposite direction to this plan.

---

## 1. Verdict on the premise: ✅ correct, with one refinement that defines the scope

> *"HF is just in eval not train … we can reuse the current FM model already trained."*

**Confirmed by code.**

`HardFlow/run/train.py` imports exactly five things — config, dataset, `FlowMatcher`, `TemporalUnet`, array utils. **Zero constraint / NLP / CasADi / IPOPT imports.** And the loss (`hardflow/models_flow/flow_matcher.py:39-56`) is textbook conditional flow matching:

```python
t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x, t=None)
xt   = apply_conditioning(xt, cond, self.action_dim)
vt   = self.model(xt, t)
loss = torch.nn.functional.mse_loss(vt, ut).mean()
```

**HardFlow's entire contribution is at sampling time.** A HardFlow checkpoint and an FMv3ODE checkpoint are trained on the same objective; they differ only in architecture and data pipeline.

### 1.1 ⚠️ THE REFINEMENT — only ONE of HardFlow's four guidance modes is portable

`run/eval.py:578-637` splits the guidance methods into two groups with completely different requirements:

| mode | mechanism | portable to FMv3ODE? |
|---|---|---|
| `projection` | **l4casadi** wraps `WrappedFlowUnet` — the network is compiled *into* the NLP | ❌ **NO** |
| `projection_relaxed` | same | ❌ **NO** |
| `hardflow` (old) | same | ❌ **NO** |
| **`hardflow_new`** | network called as a **black-box numpy function outside the solver** | ✅ **YES** |

The three l4casadi modes hard-code the architecture at `run/eval.py:583-589`:
```python
wrapped_flow_model = WrappedFlowUnet(
    horizon=..., transition_dim=..., cond_dim=...,
    dim=32, dim_mults=(1, 4, 8), attention=False)
```
A different backbone cannot be substituted without re-deriving the l4casadi export.

`hardflow_new_forward` (`flow_policy.py:1286-1321`) instead does:
```python
def flow_eval_np(x_np, t):        # <- pure black box: (trajectory, time) -> velocity
    x_torch = to_torch(x_np, device=device).reshape(1, self.oc_dof)
    ...
```
with NLP decision variables that are plain trajectory DOFs (`oc_X_terminal_predicted`, `oc_dof = horizon*(state_dim+action_dim) - state_dim`) and **no network inside CasADi**. Its own docstring: *"NLP solved at every ODE step is purely algebraic."*

> **⭐ Gen12 ports `hardflow_new` and nothing else.** This is the single most important scoping decision in this plan. Attempting the l4casadi modes is a different, far larger project.

### 1.2 What `hardflow_new` actually does

Per ODE step `k` of `K = ode_t_steps`:
1. **reference step** — evaluate the flow field, take an Euler step
2. **terminal prediction** — `x̂1 = z + (1−τ)·v`
3. **projection** — solve a small prox-NLP: keep `x̂1` close to the reference while satisfying constraints + linear dynamics
4. **pull-back** — blend the projected terminal back, weighted by `τ²` (`oc_control_cost … * self.oc_t_param**2`)

Early steps (`τ≈0`) are nudged gently; late steps (`τ≈1`) are pulled hard onto the feasible set. It needs from the model **only** `v = f(x, t)`.

---

## 2. Deliverables — copy-modify sibling pair

Per the repo convention (each generation gets its own model + test folder; older generations stay intact for A/B):

| new | copied from | contains |
|---|---|---|
| `flow_matcher_v3_hardflow/` | `flow_matcher_v3/` | + `sampling/hardflow_projection.py` (the new sampler), + policy wiring |
| `FM_v3_hardflow_test/` | `FM_v3_test/` | modified `eval_FM_v3.py` with a `hardflow_new` variant |
| `Slurm_Codes/sbatch/hardflow_fmv3/` | `Slurm_Codes/sbatch/` FMv3ODE entries | `eval_fmv3_hardflow_job.sh`, pipeline |
| `config/` additions | — | a `hardflow` block in the eval YAML |

**Do not modify** `flow_matcher_v3/`, `FM_v3_test/`, or the vendored `HardFlow/`. Gen12 is purely additive. **No training script is needed** — that is the whole point (§1).

---

## 3. 🔴 Traps — read before writing code

### 3.1 The dynamics model must be REFIT on FMv3ODE's data

`hardflow_new`'s NLP enforces linear dynamics `s' = A·s + B·a + c` from `logs/<env>/dynamics/linear_model.npz`, produced by `HardFlow/run/fit_dynamics.py`. That file was fit on **HardFlow's** `SequenceDataset` with **HardFlow's** normalizer.

| | HardFlow | FMv3ODE |
|---|---|---|
| env id | `avoiding-v0` | `avoiding-d3il` |
| `max_path_length` | **200** | **150** |
| normalizer | `LimitsNormalizer` | `LimitsNormalizer` |
| horizon | 8 (default) / 16 (Gen13) | **8** |
| state/action dim | 4 / 2 | 4 / 2 (verify) |

Same underlying D3IL task, **different data pipeline**. `A, B, c` are expressed in normalized units, so **reusing HardFlow's `.npz` against FMv3ODE-normalized trajectories is silently wrong** — the NLP would enforce the wrong physics and still converge.

**Required:** port `fit_dynamics.py` to run on FMv3ODE's `SequenceDataset`/normalizer, and write to a Gen12-specific path. Verify by comparing predicted vs actual next states on held-out data before trusting any eval number.

### 3.2 Trajectory layout must be verified, not assumed

HardFlow's `oc_dof = horizon*(state_dim+action_dim) − state_dim` implies a specific `[action | state]` interleaving with the first state pinned by conditioning. FMv3ODE uses diffuser's `[action, observation]` convention with `apply_conditioning` on `action_dim`.

**Action for the agent:** write a shape/layout assertion test *first*, comparing index-by-index which columns are actions vs states in each repo. A silent transposition here yields plausible-looking but meaningless constraint enforcement.

### 3.3 Time-direction convention

HardFlow: τ=0 noise → τ=1 data. Confirm FMv3ODE's ODE integrates the same direction before wiring `x̂1 = z + (1−τ)·v`. Gen13 lost real time to exactly this class of bug (see `Gen13/fix_1`). **One assertion test at the seam is cheap insurance.**

### 3.4 Batch size

HardFlow's MPC path hard-asserts `batch_size == 1` (no candidate fan). DPCC/FMv3ODE evaluates with `batch_size=4` and selects among candidates. **Decide explicitly**: either run `hardflow_new` at batch 1 (faithful, and the honest comparison to Gen13's numbers) or generalize it. Document the choice — it materially affects both success rate and s/plan.

### 3.5 Console noise

Gen13 hit this **three times** (tqdm, then CasADi `print_time` twice — one log was 91 % timing spam). The NLP solves ~K times per replan × ~7 replans × n episodes.

**Before the first full run:** set IPOPT `print_level=0` and CasADi `print_time=False`. Reference implementation: `HardFlow/hardflow/models_flow/imf/imf_flow_policy.py::silence_casadi_timing()`.

### 3.6 Output naming must encode provenance

Gen13 lost results twice to hardcoded experiment names. Every Gen12 eval directory must encode **checkpoint + guidance variant + K + n**. Refuse to overwrite an existing finished directory unless `FORCE_OVERWRITE=1`.

---

## 4. Implementation order

Each step ends in a runnable check. **Do not proceed past a failing step.**

| # | step | done when |
|---|---|---|
| **1** | Create both sibling folders as verbatim copies. Change nothing else. | Existing FMv3ODE eval runs unchanged from the new folder |
| **2** | Layout + direction assertion tests (§3.2, §3.3) — pure shape/sign checks, no NLP | Tests pass, printed layout table matches both repos |
| **3** | Port `fit_dynamics` to FMv3ODE's pipeline (§3.1) | `.npz` written; held-out one-step prediction error reported and sane |
| **4** | Port `hardflow_new_forward` + `hardflow_formulate` into `sampling/hardflow_projection.py`, model as a black-box callable | Runs on **1 episode**, produces a trajectory, no crash |
| **5** | Wire the `hardflow_new` variant into the eval script alongside the existing `Projector` variants | `n=3` smoke run completes; console is quiet (§3.5) |
| **6** | Slurm entries + provenance-safe naming (§3.6) | A submitted job produces a correctly-named output dir |
| **7** | Full eval: HardFlow vs DPCC-projection vs unguided, matched K, n≥100 | Results table |

### 4.1 Isolation of concerns

Steps 2–3 are **verification**, 4–5 are **the port**, 6–7 are **operations**. If the agent's budget is limited, steps 1–4 alone are a complete, useful increment: they establish whether the port is even coherent.

---

## 5. Experiment design (step 7)

⚠️ **Match K across arms.** Gen13's central error was comparing iMF@K=5 against FM@K=10, which invalidated a whole round of conclusions (`Gen13/fix_7`). In HardFlow, `K == ode_t_steps` controls **both** NFE **and** the number of NLP solves — it is not a free axis.

| arm | guidance | purpose |
|---|---|---|
| A | none (unguided) | field quality floor |
| B | DPCC `Projector` (existing) | the incumbent |
| C | **`hardflow_new`** (new) | the contribution |

At matched `K ∈ {2, 5, 10}`, n≥100, same seeds. Report **success, safety/violations, s/plan, NFE, NLP solves**.

**The question Gen12 answers:** *does in-loop constrained sampling beat post-hoc projection at equal compute, on FMPCC's own model?*

### 5.1 Do not reuse Gen13's roughness metric as a quality proxy

Gen13 established (`Gen13/U_9_train_curve/results_analysis/`) that **post-projection roughness is ~identical across models** — the NLP flattens everything to one level, so it measures the projection, not the model. FM's *roughest* unguided setting was also its *only* successful one. **Rank arms by task success, not smoothness.** Record roughness as a descriptor only.

---

## 6. Success criteria

**Minimum (port is correct):** arm C runs end-to-end, respects constraints at least as well as arm B, and the dynamics model passes §3.1's held-out check.

**Real result:** arm C beats arm B on success-per-second at matched K — i.e. enforcing constraints *during* sampling beats projecting *after*.

**A clean negative is also a result**, and should be written up as such: if `hardflow_new` matches DPCC at higher cost, that is worth knowing and consistent with the Gen13 finding that the projection dominates outcomes regardless of the field.

---

## 7. Out of scope

- Training anything (§1 — the premise of this plan)
- The three l4casadi guidance modes (§1.1)
- iMF / MeanFlow backbones — that is Gen13, and it runs *inside* the vendored HardFlow repo
- Any edit to `flow_matcher_v3/`, `FM_v3_test/`, or `HardFlow/`

---

## 8. Open questions for the user

1. **Env target** — `avoiding-d3il` only, or also `aligning`? This plan assumes avoiding only.
2. **Batch-size policy** (§3.4) — faithful batch-1, or generalize to FMPCC's candidate fan?
3. **`MASTER_TEST_HISTORY.md`** — there is an existing *"Gen11+ / X"* row describing exactly this integration. It should probably be relabelled Gen12 and pointed at this plan. **Not edited here** — say the word and I will.
