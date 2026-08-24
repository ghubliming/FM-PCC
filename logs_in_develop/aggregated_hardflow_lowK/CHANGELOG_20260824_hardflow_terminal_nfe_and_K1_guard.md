# HFK1 (2026-08-24) — HardFlow terminal step: skip the zero-weight NFE, announce the K=1 degeneracy, unstale the gates

**Date:** 2026-08-24
**Scope:** **AGGREGATED across six live generations** (hence `aggregated_*`, not a per-Gen folder) —
every generation that ships a HardFlow projector copy, since all six carry a byte-identical ODE loop
and therefore the identical low-K defect:
**Gen12** `flow_matcher_v3_hardflow/` · **Gen3v6** `flow_matcher_v3_meanflow/` ·
**Gen3v7** `flow_matcher_v3_alphaflow/` · **Gen15 UAV** `mix_uav/` ·
**Gen15 visual aligning** `mix_visual_aligning/` · **Gen16 visual avoiding** `mix_visual_avoiding/`
(+ the Gen12 gate suite `FM_v3_hardflow_test/gates_hardflow.py` and two K-sweep sbatch scripts).
**Trigger:** "among all the live model/gens with HF projector, how do they treat K=1? is it degenerate?"
**Status:** code changed, **nothing executed** — this container has no torch/casadi. Every claim below is
static analysis + pure-arithmetic checks. **Run `gates_hardflow.py` on the cluster before trusting it.**
**Companion (the analysis this implements):**
[`../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md`](../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md)
(2026-08-16, §0.1 / §2 / §3 / §4.1 / §6 / §10).

---

## 1. The question, answered

**All six live copies treat K=1 identically, and all six degenerate.** The core ODE loop is
byte-identical across them — I diffed the four defining lines (`dt`, `tau_next`, the `active` gate,
`X1_ref`, the pull-back) and they match character for character:

| generation | file | K=1 behaviour |
|---|---|---|
| Gen12 (FMv3ODE) | `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | degenerate |
| Gen3v6 (MeanFlow) | `flow_matcher_v3_meanflow/…` | degenerate **+ D3 confound** |
| Gen3v7 (α-Flow) | `flow_matcher_v3_alphaflow/…` | degenerate **+ D3 confound** |
| Gen15 (UAV mix) | `mix_uav/…` | degenerate |
| Gen15 (visual aligning) | `mix_visual_aligning/…` | degenerate |
| Gen16 (visual avoiding) | `mix_visual_avoiding/…` | degenerate |

### What K=1 *really* is

```
K = 1  =>  dt = 1,  k = 0 is simultaneously the first AND the terminal step,  tau_next = 1.0
  X_ref  = X_0 + f(X_0, 0)·1               one Euler step across the whole interval
  X1_ref = X_ref + (1 − 1)·f(X_ref, 1) = X_ref      I1 (endpoint lookahead) dead
  X1*    = argmin ‖X1 − X_ref‖²  s.t. h ≤ 0         a plain Euclidean projection Π_S
  X_1    = X_ref + 1·(X1* − X_ref) = X1*            I2 (damped pull-back) dead
                                                     I3 (feedback) dead — no step k+1
```

**`K=1 HardFlow ≡ Π_S(one Euler sample)` — sample-then-project.** That is DPCC's algorithm, with
IPOPT instead of SLSQP. Zero HardFlow-specific arithmetic executes. Corollaries:

* `activation_threshold` is **inert** at K=1 — `int((1−A)·1) == 0` for every `A`, and the forced
  `k == K−1` solve fires regardless. `all` / `late` / `0.0` / `1.0` are bit-identical.
* `reg_scale` and the τ² prox weight are inert at K=1 (and, in our port, at **every** K — D2).
* K=2 is degenerate too under the shipped `activation_threshold = 0.5`: `int(0.5·2) = 1` floors
  step 0 out, leaving only the terminal solve. Only `A = 1.0` buys one genuine step at K=2.

This is not a bug — the terminal collapse **is** the paper's safety proposition. It just means
HardFlow-ness lives entirely in the *active, non-terminal* steps, and at K=1 there are none.

### Is it still worth running K=1?

**Yes, but not as a HardFlow row.** Two honest uses survive, one does not:

* ✅ **As a matched-NFE one-shot-projection comparison** (IPOPT vs SLSQP on the same Euler sample,
  same `constraint_list`). Change 2.1 below makes this comparison *clean* for the first time —
  arm C used to burn 2 NFE at K=1 to arm B's 1.
* ✅ **As the cheapest safe operating point.** The architecture-matched headline row
  (MeanFlow-UNet K1 `hardflow-tightened`, S&C 1.000) lives here and the *number* stands.
* ❌ **As evidence about in-loop constrained sampling.** It cannot be: the in-loop mechanism
  never runs. DEGENERACY §8.2 is blunt about the shape of this — HardFlow's only win in the
  whole `avoiding-d3il` corpus is at K=1 (zero genuine steps) and its gate score *falls* as
  genuine steps are added (1.000 → 0.933 → 0.933 → 0.833 for K = 1/2/5/10).

If the goal is to test HardFlow the algorithm, the first non-degenerate settings are
**K ≥ 3 with A = 1.0**, or **K ≥ 5 with A = 0.5** (the paper itself runs N=10 / A=0.5).

---

## 2. Code changes

### 2.1 🔴 Skip the terminal step's zero-weight network call — all six copies

At `k = K−1`, `(1 − tau_next)` is zero, so `V_next` was computed and immediately multiplied away.

```python
# before
V_next = self._velocity_batch(X_ref, tau_next, s0_all, cond_net, returns_net)
X1_ref = X_ref + (1.0 - tau_next) * V_next
# after
if k < K - 1:
    V_next = self._velocity_batch(X_ref, tau_next, s0_all, cond_net, returns_net)
    X1_ref = X_ref + (1.0 - tau_next) * V_next
else:
    X1_ref = X_ref                                    # terminal: tau_next == 1
```

* **Trajectories are unchanged.** The removed term was exactly `0 · V_next`.
* **The test is structural (`k < K-1`), deliberately not `1.0 - tau_next > 0.0`.** For K ∈
  {1,2,5,10,20} the float sum lands on exactly 1.0, but for K ∈ {6,14,24,28,29,38,54,…} it leaves
  a `+1.1e-16` residue (and at K=93, `−2.2e-16`). That residue is orders below float32 epsilon — it
  changes nothing on the tensor — but a float test would read it as "lookahead alive" and preserve
  both the waste and the hazard for precisely those K.
* **Removes a latent NaN hazard.** `0.0 * NaN = NaN` under IEEE. `t = 1.0` is the *closed* edge of
  the CFM training support (`t ~ U[0,1)`), i.e. the point the backbone was never trained on; a
  non-finite value there used to poison `X1_ref` and go straight into the NLP.

⚠️ **Reporting consequence.** `policy.nfe` now reads **`K + n_active − 1`** (was `K + n_active`).
HardFlow NFE **and wall-time** figures are **not** comparable with pre-2026-08-24 runs.
The saving is largest exactly where it matters most:

| K | A | arm A/B NFE | arm C NFE (before) | arm C NFE (after) |
|---|---|---|---|---|
| **1** | any | 1 | **2 (+100 %)** | **1 (parity)** |
| 2 | 0.5 | 2 | 3 | 2 |
| 5 | 0.5 | 5 | 8 | 7 |
| 10 | 0.5 | 10 | 15 | 14 |
| 20 | 0.5 | 20 | 30 | 29 |

At K=1 arm C and arm B now cost the *same* network budget, so the K=1 B-vs-C gap is finally a
clean solver comparison rather than a 2×-NFE confound.

### 2.2 `hardflow_step_budget(flow_steps, activation_threshold)` — new shared helper, all six copies

```python
n_active  = max(K - int((1.0 - A) * K), 1)     # steps that solve the NLP (terminal is forced)
n_genuine = n_active - 1                       # active AND non-terminal == the only real HF steps
```

Placed next to `resolve_activation_threshold`, exported from `sampling/__init__.py` in the five
packages that re-export the HF symbols (`mix_uav` imports the module directly and needs no change).
Cross-checked against the literal loop gate on an 8×6 (K, A) grid — **zero mismatches** — and
against DEGENERACY §4.1's reference row. It is now the *single* source that the sampler, the log
banner and the Gen12 gates all read, so the gate arithmetic cannot drift apart again (it already
did once — see 2.5).

### 2.3 Degeneracy banner in `HardFlowSampler.sample()` — all six copies

When `n_genuine == 0`, the sampler prints once per `(K, A)`:

```
[hardflow][DEGENERATE] K=1 A=0.5: n_active=1, n_genuine=0 — every NLP solve is the terminal
tau=1 solve, so this arm runs Pi_S(Euler sample): sample-then-project, == DPCC modulo
solver/variable-scope, NOT HardFlow. Do NOT report these rows as HardFlow results. First
non-degenerate settings: K>=3 with activation_threshold=1.0, or K>=5 with A=0.5. …
```

Warn-once state is held on `getattr(self, '_hf_degenerate_warned', None)`, so no `__init__`
signature was touched in any copy (their constructors have diverged — `init_noise_scale`,
`two_time`, `engine` — and the mix_visual_* copies build the sampler from a module-level factory
that never sees K at all; K only arrives at `sample()` time). One line per job, never per plan —
no live-progress spam in the batch logs.

`infos` also now carries `'n_active'` and `'n_genuine'`, so a DA can separate genuine-HardFlow rows
from sample-then-project rows, and check DEGENERACY §9.1 (`nlp_solves per plan == n_active`),
without re-deriving the gate arithmetic. Nothing iterates `infos` generically — verified — so the
extra keys are inert for every existing consumer.

### 2.4 Two corrected docstring claims — all six copies

* *"The prox weight carries a tau^2 factor, so early steps are nudged and late steps are pulled
  hard"* — **false in our port**, and it contradicted `gates_hardflow.py::gate_g2`, which has been
  asserting the opposite all along. We implement the prox term alone (upstream's optional `C(·)` is
  dropped on purpose, so arms B and C solve the same problem), and `argmin c‖x1 − x1_ref‖² s.t.
  h ≤ 0` is independent of `c > 0`. The NLP is exactly `Π_S(x1_ref)` for any `reg_scale`, any τ.
  What actually damps early steps is the **linear** `tau_next` factor in the pull-back. (D2.)
* *"NFE accounting clean (2K here…)"* → `K + n_active − 1`.

Also added: an explicit "step k = K−1 is always a plain projection" paragraph pointing at
`hardflow_step_budget`, so the next porter meets the degeneracy in the module header rather than
in a DA.

### 2.5 🔴 `FM_v3_hardflow_test/gates_hardflow.py` — two gates were stale, one is new

Found while wiring the helper in. **Both would fail on the cluster today**; neither has anything to
do with K=1 per se, but both are the same activation arithmetic.

**G3 has been asserting a pre-`fix_6` contract.** It builds the sampler with
`activation_threshold=0.0` and then asserts `nlp_solves == K` and `nfe == 2K`. Under `fix_6`'s DPCC
polarity, `A = 0.0` means **terminal-only** — `n_active = 1`, not `K`. So at the default K=5 the
gate builds a 1-solve sampler and demands 5 solves. Fixed by building at `A = 1.0` (which *is* G3's
subject: the "every step is solved" end-to-end path) and deriving both expectations from
`hardflow_step_budget`.

**G4 has been asserting the pre-`fix_8` rounding.** `fix_8` (2026-08-07) moved the sampler gate from
the raw-float CEIL form to DPCC's FLOOR `int((1−T)·K)` in three files, but never updated the gate,
which still computes `k >= (1.0 - thr) * K`. The two disagree in **4 of G4's 12 cells**:

| K | thr | G4 expected | code produces |
|---|---|---|---|
| 2 | 0.9 | 1 | **2** |
| 5 | 0.5 | 2 | **3** |
| 5 | 0.9 | 4 | **5** |
| 10 | 0.9 | 9 | **10** |

Fixed by reading `hardflow_step_budget`; G4 now also prints `genuine=` per cell and flags
`DEGENERATE` rows.

**New `gate_g6` — HFK1 low-K.** Three assertions:
(a) the helper matches the literal loop gate on the 8×6 (K, A) grid, plus DEGENERACY §4.1's
reference row — *this part is pure arithmetic and I ran it here: it passes*;
(b) at K=1 the output is **bit-identical** across `A ∈ {0.0, 0.5, 1.0}` and reports
`n_genuine == 0` with exactly 1 solve — the operational proof that the whole HardFlow knob surface
is frozen at K=1;
(c) `nfe == K + n_active − 1` and `nlp_solves == n_active` over K ∈ {1,2,5} × A ∈ {0.5,1.0}.
(b) and (c) need torch + casadi → **cluster**.

### 2.6 Degeneracy note in the two shipped K sweeps

`Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` and
`Slurm_Codes/sbatch/mix_visual_avoiding/eval_k_sweep.sh` both default to `KS="1 2 5 10 20"`, i.e.
they ship two degenerate K values. A comment block above the default now says what those rows are
and are not. **The defaults were left alone deliberately** — the K=1/K=2 rows are still worth
having (§1), they just must not be labelled HardFlow.

---

## 3. What was deliberately NOT changed

* **The `K=1` / `K=2` sweep defaults.** Still useful as a one-shot-projection comparison, and now
  matched-NFE. Relabel, don't delete.
* **`C(·)` is still absent from the NLP (D2).** Adding it would restore ingredient I4 but make arm C
  solve a *different problem* than arm B, breaking exactly the constraint-set parity that
  `setup_dpcc_projector(..., return_constraint_list=True)` exists to guarantee. D2 is a deliberate
  divergence for comparability; documented as one now (2.4), not undone.
* **`h = 0` in the velocity queries (D3).** Changing the `mf`/`af` arm C to `h = dt` (or the endpoint
  to `h = 1 − τ⁺`) would make arm C a *new method* rather than a ported baseline. Legitimate to
  build — DEGENERACY §8.4 says where it would actually be live (K ≥ 5, A raised) — but it is a
  different experiment and must be run and reported as one.
* **`MASTER_TEST_HISTORY.md`.** Not touched. If this warrants an index row, say so and I'll add it.
* **Past results and curated snapshots.** No renaming or retraction applied here; DEGENERACY §8.5's
  suggested relabelling of the snapshot headline row is still a proposal.

---

## 4. Files touched

| file | change |
|---|---|
| `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | 2.1, 2.2, 2.3, 2.4 |
| `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` | 2.1, 2.2, 2.3, 2.4 |
| `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` | 2.1, 2.2, 2.3, 2.4 |
| `mix_uav/sampling/hardflow_projection.py` | 2.1, 2.2, 2.3, 2.4 |
| `mix_visual_aligning/sampling/hardflow_projection.py` | 2.1, 2.2, 2.3, 2.4 |
| `mix_visual_avoiding/sampling/hardflow_projection.py` | 2.1, 2.2, 2.3, 2.4 |
| `{flow_matcher_v3_hardflow,_meanflow,_alphaflow,mix_visual_aligning,mix_visual_avoiding}/sampling/__init__.py` | export `hardflow_step_budget` |
| `FM_v3_hardflow_test/gates_hardflow.py` | G3 unstaled, G4 unstaled, new G6, header docs |
| `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` | degeneracy note above the `KS` default |
| `Slurm_Codes/sbatch/mix_visual_avoiding/eval_k_sweep.sh` | degeneracy note above the `KS` default |

All twelve Python files pass `py_compile`; both shell scripts pass `bash -n`. CRLF endings on
`gates_hardflow.py` were preserved (27-insert diff, not a whole-file rewrite).

---

## 5. To run on the cluster

```bash
# 1) gates first — G3/G4 should now PASS (they would have failed before), G6 is new
python FM_v3_hardflow_test/gates_hardflow.py
python FM_v3_hardflow_test/gates_hardflow.py --flow-steps 1     # the degenerate case, explicitly

# 2) sanity on one live arm: the banner must appear at K=1/K=2 and NOT at K=5/K=10
HFFM_FLOW_STEPS=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
# expect exactly one `[hardflow][DEGENERATE] K=1 A=0.5 …` line in the log

# 3) the two cheap reads DEGENERACY §10 asked for and nobody has done yet
#    (a) nlp_failures on CAND_39/CAND_40 — is IPOPT bailing and keeping an infeasible iterate?
#    (b) per-K nlp_solves on the bbunet ladder vs n_active (1/1/3/5) — confirms D1 operationally
#    Both are already in `infos`; no new code path, no new cluster time.
```

**Re-baseline note:** any HF NFE / avg_time comparison that crosses 2026-08-24 is invalid (2.1).
Success and constraint numbers are unaffected — trajectories are bit-identical.
