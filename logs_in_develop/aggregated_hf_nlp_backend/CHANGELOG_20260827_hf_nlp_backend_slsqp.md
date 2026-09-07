# HardFlow's NLP now runs on DPCC's SLSQP by default

**Aggregated fix — not a generation.** It lands identically in six sibling generations, so it is
filed here rather than under any one of them, alongside `aggregated_hardflow_lowK/` and
`aggregated_divergence_abort/`. (Do not confuse this with **Gen16**, which is Visual-Avoiding
Mix-ML — one of the six trees this touches, nothing more.)

**Date** 2026-08-27 · **Marker** grep `SolverSwap` · **Evidence** job 25121
(`logs_in_develop/Gen12/Solver_Bench/RESULTS_20260827_solver_bench_ipopt_vs_slsqp.md`)

**Touches:** `flow_matcher_v3_alphaflow` (Gen3v7) · `flow_matcher_v3_hardflow` (Gen12) ·
`flow_matcher_v3_meanflow` (Gen3v6) · `mix_uav` (Gen15) · `mix_visual_aligning` (Gen14) ·
`mix_visual_avoiding` (Gen16) — plus their eval scripts and the DA configs.

**Status: written, AST-verified, NOT run.** No Python env in this container — every claim
below about *behaviour* is a code-level claim; the numbers come from job 25121.

---

## 1 · Why

Job 25121 timed both projectors on the **identical** NLP, same `constraint_list`, 3 seeds × 50 reps:

| reference regime | IPOPT (HF) | SLSQP (DPCC) | ratio |
|---|---:|---:|---:|
| `endpoint` — what HardFlow actually solves | 47.6 ms | 11.0 ms | **4.33×** |
| `iterate` — what post-hoc projection solves | 54.2 ms | 34.0 ms | 1.63× |

and the two backends returned the **same point**: mean 3.4e-4, max 1.0e-3 over 100 solves.

Same answer, 4.3× the price. IPOPT is an interior-point code for large *sparse* NLPs; ours is
44 dense variables, so most of its cost is size-independent per-call setup — it is only 1.14×
slower on a 3× harder problem, against SLSQP's 3.09×. That is the signature of overhead, not work.

Consequence worth stating plainly: HardFlow's central optimisation is to project the
*near-feasible predicted endpoint* instead of a noisy iterate. That trick is worth **3.09× to
SLSQP and 1.14× to IPOPT**. **HardFlow ships the one solver that cannot cash in its own idea.**

---

## 2 · What changed

**Add-on, not a replacement. No code was deleted.** The IPOPT path is intact and selectable;
`_solve_ipopt` is the old `solve()` body and this was verified by AST comparison against a
pre-patch copy — **identical in all six generations**, not merely "looks the same".

### 2.1 · `*/sampling/hardflow_projection.py` (×6)

`flow_matcher_v3_alphaflow`, `flow_matcher_v3_hardflow`, `flow_matcher_v3_meanflow`,
`mix_uav`, `mix_visual_aligning`, `mix_visual_avoiding` — identical transformation to each.

| addition | what it does |
|---|---|
| `NLP_BACKENDS`, `DEFAULT_NLP_BACKEND = 'slsqp'`, `resolve_nlp_backend()` | precedence: explicit kwarg → `FMPCC_HF_NLP_BACKEND` env → default. Unknown value raises. |
| `HardFlowNLP(..., nlp_backend=None)` | new kwarg; `None` means "resolve". |
| `solve()` → dispatcher | routes to `_solve_slsqp` or `_solve_ipopt`. |
| `_solve_ipopt` | the original body, verbatim. |
| `_build_slsqp_projector()` | builds DPCC's `Projector` on the **same** `constraint_list`, with a stub normalizer made from the **same** `mins`/`maxs` the IPOPT path uses. |
| `_solve_slsqp()` | `from_dof` → `Projector.project` → `to_dof`; counts failures; accumulates `solve_ms`. |
| `set_s0()` | also stores `self._s0` (the DPCC projector takes a *full* trajectory). |
| episode info dict | new key `nlp_backend`. |
| policy / `build_hardflow_sampler` signatures | `nlp_backend=None` passed through. |

**Why the two backends solve the same problem.** HF's cost is
`0.5 · reg_scale · τ² · ‖x − x_ref‖²` — a positive *scalar* multiple of the squared distance. A
positive scalar does not move an argmin, so HF's NLP is exactly `Π_S(x_ref)`, which is what
`Projector.project` computes (`Q = I`, `r = −x_ref`). `τ` is therefore **accepted and ignored**
by the SLSQP path. That is exact, not an approximation — and job 25121's 1e-3 agreement is the
measurement of it.

The IPOPT NLP is **still built** on every construction even when idle, so the backend can be
flipped without touching anything else. CasADi therefore remains a hard dependency — unchanged
from before.

### 2.2 · `*/sampling/projection.py` (×6) — 6 added lines each, behaviour-neutral

`self.last_solve_success` records `bool(res.success)` per scipy solve. **DPCC still silently
keeps `res.x` on non-convergence — deliberately unchanged, so arm B's numbers do not move.**
The list only makes those failures *countable* by `_solve_slsqp`, which is what keeps
`nlp_failures` meaningful across the swap instead of silently reading 0.

Diff verified at exactly 6 changed lines per file. Five of the six are CRLF; endings preserved.

### 2.3 · Identification — the part that matters most

A run whose solver is not recorded is a run nobody can interpret later. Three independent places:

1. **Log, every run, from the constructor:**
   `[hardflow][NLP-BACKEND] slsqp  (scipy SLSQP via DPCC Projector — IPOPT built but idle)  dof=44  reg_scale=1.0`
2. **Per-variant summary line** — `nlp_backend={...}` appended to the existing `[hardflow]` /
   `Compute:` print in each eval.
3. **Artifact** — every result file carries **both**:
   - `nlp_backend` — the string, for humans and `grep`
   - `nlp_backend_slsqp` — float twin, `1.0` = SLSQP, `0.0` = IPOPT

   The twin exists because generic DA loaders coerce npz scalars through `_as_float`, where a
   string lands as `NaN`. Without it the field would be invisible in exactly the tables that
   need it.

Arms A/B (no NLP) record `'n/a'`, never a backend name.

| file | sink |
|---|---|
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | npz |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | npz |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | npz |
| `mix_visual_avoiding_test/eval_mix_visual_avoiding.py` | npz |
| `mix_visual_aligning_test/eval_mix_visual_aligning.py` | npz |
| `mix_uav_test/eval_mix_uav.py` | summary **json** |
| `mix_visual_avoiding/sampling/policies.py` | unguided path → `'n/a'` |

### 2.3b · 🔴 Artifact naming — the swap must not destroy the IPOPT corpus

Every eval writes `{save_path}/{variant}.npz`. Re-running `hardflow_new-c-tightened` on the new
backend would have **overwritten the exact rows chapters 1–3 of the DA are built on** — silently,
with no way back. Recording the backend *inside* the file does not help: the old file is already gone.

`artifact_variant_label(variant, backend)` (next to `resolve_nlp_backend`, all six copies) fixes it:

| backend | `hardflow_new-c-tightened` becomes | effect |
|---|---|---|
| `ipopt` | `hardflow_new-c-tightened` — **unchanged** | every pre-existing path stays exactly what it was; nothing on disk moves or is reinterpreted |
| `slsqp` | `hardflow_sls-c-tightened` | lands **beside** the IPOPT corpus |

Non-HardFlow variants (`diffuser`, `dpcc-*`) pass through untouched on both backends — they never
touch this NLP. The `-r/-c/-t` and `-tightened` suffix grammar is preserved, and the name still
starts with `hardflow`, so arm-C branching keeps working.

Applied to **every** path that could collide, not just the npz:

| eval | isolated |
|---|---|
| `FM_v3_hardflow` | npz (incl. the clobber guard), png |
| `FM_v3_meanflow`, `FM_v3_alphaflow` | npz, png, `eval_*.log`, the `aggregate_only` reader |
| `mix_visual_avoiding` | npz, png, `eval_*.log`, the `aggregate_only` reader |
| `mix_visual_aligning` | the whole per-variant **directory** (`results/{geo}/{variant}`), so npz, partial sidecar, png, logs, realtime logs follow |
| `mix_uav` | the whole per-variant **directory** (`{geo_dir}/{variant}`), so npz, eval log, plots, diagnostics follow |

Two details worth knowing:

- In `FM_v3_hardflow` the path is chosen **before** the policy exists, so the label uses the
  module-level `resolve_nlp_backend()`. An `assert` after the policy is built compares it against
  the live `policy.nlp.nlp_backend` and fails loudly rather than write an SLSQP result into an
  IPOPT filename.
- In `mix_visual_aligning` the agent receives `variant=variant_out`, because `self.variant` is used
  only for naming (episode ids, realtime logs, the partial sidecar) — those would otherwise collide too.

### 2.4 · DA registration

🔴 **DA discovery is an explicit allow-list of variant names, so an unregistered name is
INVISIBLE rather than an error.** Without this section the swap runs would produce data that
never appears in any table. Registered:

- `DA_Code_v3/config.py` — `hardflow_sls*` in `HARDFLOW_VARIANTS` and the headline
  `MAJOR_VARIANTS`; plus `nlp_backend_slsqp` in `HARDFLOW_METRICS`, label, type
- `DA_UAV_v1/config.py` — `hardflow_sls*` in both the variant list and `MAJOR_VARIANTS`;
  `data_loader.py` lifts `nlp_backend_slsqp` from the `hardflow` json block
- `DA_VA_v2/config.py` — `hardflow_sls*` in `VARIANT_ORDER` and `MAJOR_VARIANTS`; the npz
  ingestion is generic, so the float twin arrives by shape with no loader change

**Never pool `hardflow_new-*` with `hardflow_sls-*`** — different solver, and the whole point of
the rename is that the two corpora stay separable.

---

## 3 · How to run either backend

```bash
# default — nothing to pass
./Slurm_Codes/submit.sh <any hardflow sbatch>

# the old solver, for an A/B on one run
FMPCC_HF_NLP_BACKEND=ipopt ./Slurm_Codes/submit.sh <same sbatch>
```

Per call site: `HardFlowNLP(..., nlp_backend='ipopt')`, or the same kwarg on the policy /
`build_hardflow_sampler`. The explicit kwarg beats the env var.

---

## 4 · What this is expected to do — and what it cannot

Extrapolating the measured 4.33× onto the fan-matched parity run (generator 18.5 ms; DPCC
2.4 ms/step, HF 30 ms/step): arm C **48.5 → 25.4 ms/step** against DPCC's 20.9, i.e.
**2.32× → 1.22×**.

⚠️ Extrapolation until an eval actually runs. It moves **only the solve term** — solve count
(`K − floor((1−A)·K)`) and generator cost are untouched.

**It cannot improve success, constraints, or steps.** Chapters 1–3 of the companion DA doc
already decide those against HardFlow. This is a *closing* move: it removes the one objection a
reviewer could still raise — that we handicapped HardFlow with the wrong solver — and per the
audit's decision rule, cost then stops being the story.

The one way it surprises us: IPOPT fails 12.5–13.5 % of solves on visual-avoiding TL untightened
(companion 2b) and 26 % on job 25121's noisy references, each failure silently keeping a possibly
infeasible iterate. If part of HardFlow's *quality* deficit was those failures rather than the
method, SLSQP could close it. Nothing else in the corpus would reveal that.

---

## 5 · Verification done here

- All 12 model files + 6 eval scripts + 3 DA files `ast.parse` clean
- `_solve_ipopt` AST-**identical** to the pre-patch `solve()` in all six generations
- `projection.py` diff is exactly 6 lines per file; CRLF preserved in the 5 CRLF files
- `resolve_nlp_backend` executed in isolation: default `slsqp`, env override works, explicit
  kwarg beats env, `'gurobi'` rejected

## 6 · Not done

- **Never executed.** No Python env here. First cluster run is also the first real test of
  `_solve_slsqp` — the `from_dof`/`to_dof` round-trip and the `Projector` kwargs are code-correct
  but unexercised.
- **Baseline shift is real and intended:** every future arm-C run changes solver. Existing
  results predate the field, so a row with no `nlp_backend` is an **IPOPT** row, and it now also
  carries the old `hardflow_new-*` name. Do not pool the two corpora.
- `FM_v3_hardflow_test/bench_solver_hf_vs_dpcc.py` is now **pinned** to `nlp_backend='ipopt'`.
  Flipping the default silently turned it into an SLSQP-vs-SLSQP bench reporting ~1.0×;
  any script whose whole point is the old solver must pin it explicitly.
- `gates_hardflow*.py` were left on the default, so the gates now exercise SLSQP — which is
  correct (a gate should test what ships). If any gate asserts an IPOPT-specific number it
  will now fail; pin that one with `nlp_backend='ipopt'` rather than reverting the default.
- `dynamics_mode: linear_fit` (companion 4d) is untouched and still open.
- Nothing committed.
