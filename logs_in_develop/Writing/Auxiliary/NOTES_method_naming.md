# Method naming — call things what they are, and claim only what is ours

**Created:** 2026-09-10 · **Type:** author's position + options + a recommendation · **Status:** 🟡 **decision open, owned by the author**
**Companion:** [`NOTES_naming_and_rebuild.md`](NOTES_naming_and_rebuild.md) — that file covers *code-flag*
traps (`film_mode='v1'` is not FiLM). **This file covers the thesis-facing names of the methods themselves.**
**Bears on:** `sec:method:proj`, `sec:method:hardflow`, `sec:method:engine`, every results table, `app:repro`.

---

> 📍 **The consolidated table now lives in [`Naming/NAMING_20260910_master_table.md`](Naming/NAMING_20260910_master_table.md) and is canonical.** This file is kept for its *rationale* — the author's position, the options weighed, and why *endpoint projection* won. The §3.1/§5.1–5.2 recommendations were **adopted and applied to `thesis_v2.tex` (v2.6)**.


## 1. The author's position (2026-09-10) — the instruction this file records

> *"The HF-SLSQP — it is HF, or find some way rather than direct mention HardFlow. Find a good name,
> if even better, not just HF, af … but if we can abandon the paper-invented fancy name, calling it
> what it is will be better. `mf` is good — it is MeanFlow. And for SLSQP, it is our code; don't say
> HF-SLSQP, just HF or another name."*

Read as two rules, which are **separate problems and must not be solved with one word**:

| # | rule | the failure it prevents |
|---|---|---|
| **N1** | **Prefer the mechanism over the brand.** If the reader cannot recover what the code does from the name, the name is wrong for a thesis — even if it is fine as a `--flag`. | a paper-invented brand smuggles a claim (`α-Flow` sounds like a *family*; it is one bootstrapped target) |
| **N2** | **Claim only what is ours, and credit only what is theirs.** Our arm C is *our* implementation of *one* of their four modes, on *our* constraint stack. | "HardFlow" over-attributes to them; "HF-SLSQP" mis-attributes the contribution to a solver choice |

**N1 is a style rule. N2 is an accuracy rule.** N2 is the one that can be wrong in a defence.

---

## 2. What arm C actually is — verified in the code, 2026-09-10

Source: `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:1–40` (port header) and `:423–434`.

Per active ODE step `k` of `K`:

```
1. reference step     v_k    = f(x_k, τ_k);      x_ref = x_k + v_k·dt
2. endpoint predict   x1_ref = x_ref + (1 − τ_{k+1})·f(x_ref, τ_{k+1})
3. projection         x1_proj = Π_S(x1_ref)          ← an NLP over the SAME feasible set as arm B
4. damped pull-back   x_{k+1} = x_ref + τ_{k+1}·(x1_proj − x1_ref)
```

Four facts that decide the name:

1. 🔑 **Step 3 is a projection, not an optimisation.** Our port implements the proximal term **alone**
   (upstream's optional cost `C(·)` is dropped on purpose, so arms B and C solve the same
   feasible-set problem). For a pure quadratic prox, `argmin c‖x₁ − x1_ref‖² s.t. h ≤ 0` is
   independent of `c > 0` — so **the NLP is exactly `Π_S(x1_ref)` at every active step**, for any
   `reg_scale` and any `τ`. The `τ²` prox weight schedules nothing; what damps early steps is the
   **linear `τ_{k+1}`** in step 4.
2. 🔑 **The only difference from arm B is the point being projected.** Arm B projects the current
   (off-manifold) **iterate**; arm C projects the predicted **clean endpoint**. Same feasible set,
   same solver, same `argmin` — the code says so directly (`:426`: DPCC's `Projector.project`
   computes the same program with `Q = I`, `r = −x_ref`).
3. **SLSQP is a backend flag, not a method.** `FMPCC_HF_NLP_BACKEND` selects SLSQP or IPOPT; the
   activation gate that decides whether the algorithm runs at all is **solver-agnostic**
   ([`Proposal_20260905` §0.1](../../../Data_Analysis/DA_Result_Curated_MD/Proposal_20260905_HF_minK_mf_af_unet/README.md)).
   Putting the solver in the method name says the solver is the contribution. It is not.
4. **What is ours:** the NLP is built from FMPCC's own `constraint_list` (the same list arm B
   consumes — without this the two arms would enforce different constraint sets and the comparison
   would be void), no value-model warm start, clean NFE accounting (`K + n_active − 1`), and the
   `n_genuine` degeneracy boundary — **which is our finding, not theirs**. What is theirs: the idea
   of projecting the predicted endpoint inside the ODE, and the terminal-step safety proposition.

> ⚠️ **Consequence for `sec:method:hardflow`'s current title.** It reads *"Constraint Enforcement II:
> **In-Loop Trajectory Optimisation**"*. By fact 1 our port performs **no optimisation of an
> objective** — it solves a feasibility/projection problem. The title over-describes the port.

---

## 3. The naming options

### 3.1 The constraint arms

| option | arm B (DPCC's) | arm C (ours) | verdict |
|---|---|---|---|
| status quo in the DAs | `dpcc-{r,c,t}` | **`hardflow_sls-*` / "HF-SLSQP"** | 🔴 violates N1 **and** N2, and hard-codes a solver |
| brand, softened | the DPCC projector | "HardFlow-style guidance" | 🟡 honest-ish, still leans on their brand and still says nothing |
| letters only (v2 today) | arm B | arm C | 🟡 fine as shorthand **after** definition; opaque on its own |
| ⭐ **mechanism, contrastive** | **iterate projection** | **endpoint projection** | 🟢 **recommended** — one adjective each, and the pair states the whole difference |

> ### Recommendation
> **arm B = *per-step iterate projection*; arm C = *in-ODE endpoint projection* (with a damped pull-back).**
> Short forms in tables: **IP** / **EP**. First mention of arm C credits the source in the same
> sentence: *"…the endpoint projection introduced by \parencite{li2025hardflow} (their `hardflow_new`
> mode), re-derived here on our constraint set."*
>
> This also fixes §2's title problem: **"Constraint Enforcement II: In-ODE Endpoint Projection"**.

Why this pair and not something cleverer: the two arms differ in **exactly one thing** — the point
handed to the same projection operator. A contrastive name pair makes the whole method section, the
degeneracy section (`n_genuine`: at the terminal step the endpoint *is* the iterate, so EP collapses
to IP) and the cost mechanism (EP projects a near-feasible point ⇒ SLSQP converges in few
iterations ⇒ flat ~28 ms/solve) readable from the names alone.

### 3.2 The engines — the α axis already solves this

The fallback's framing ([`FALLBACK_…` §2.1](../Working_Space/fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md))
is itself the answer: these are **not three methods, they are one axis with three samples on it.**
All three train the same network for the same quantity (average velocity `u`) and differ **only in
how the target is built**, indexed by `α = dt/h`:

| α | code tag | published name | **what it is — the thesis name** |
|---|---|---|---|
| **1** | `fm` | Flow Matching | the **instantaneous-velocity** target |
| **0 < α < 1** | `af` | α-Flow | the **bootstrapped** target (finite difference closed with the network's own frozen output) |
| **0** | `mf` | MeanFlow | the **analytic** target (JVP; the exact α→0 limit) |

- **`mf` / MeanFlow stays.** It is a published name *and* descriptive (mean = average velocity). Keep
  it, gloss it once as *"the average-velocity objective"*.
- **`fm` / flow matching stays.** Descriptive already.
- **`af` / α-Flow:** keep the published name at first mention **with a citation**, then use
  **"the bootstrapped target"** in prose. This is not cosmetic — the entire negative result is *about*
  the bootstrap (`(1−α)·ε_next` propagates the model's own error), so the mechanism name **is** the
  argument, and the brand name actively hides it.
- **The baseline:** never bare **"diffuser"** — that token means three things (Janner et al.'s method,
  the ancestor codebase, *and* our no-projection variant row). Write **"the diffusion engine"** /
  **"the DPCC baseline"**; the unprojected row is **"unguided"**.

### 3.3 The selection rules

`-r` / `-c` / `-t` are lettered code tags. In the thesis they are named:
**random** / **cumulative projection cost** / **temporal consistency** — never lettered in prose.
(Already the rule in [`NOTES_naming_and_rebuild.md`](NOTES_naming_and_rebuild.md) §3.)

---

## 4. Rules that fall out (binding once the author confirms §3.1)

1. **First mention = published name + citation. Every later mention = mechanism name.**
2. **Never put a solver in a method name.** SLSQP vs IPOPT is a backend row in `app:repro`, not an
   identity. (`FMPCC_HF_NLP_BACKEND`; the 3.88× swap is a performance note.)
3. **Never invent a name for someone else's method** — a renamed method cannot be found by a reader
   or matched to its paper. Rename *our instantiation*, credit *their idea*.
4. **Never let a code tag into prose** (`hardflow_sls-r`, `dpcc-c-tightened`, `filmv1`, `af`, `mf`).
   Table headers may use short forms **after** definition.
5. 🔴 **`app:repro` owes a translation table.** Every log, checkpoint path and DA row keeps the old
   tokens. Rename in prose without shipping the table and no reader can match a row to a checkpoint.
   This table does not exist yet — it is §3 here plus §3 of the companion note plus the run ledger.

---

## 5. What is still the author's call

| # | question | recommendation |
|---|---|---|
| 5.1 | Adopt **iterate projection / endpoint projection**? | ⭐ **yes** — it is the only option that satisfies both N1 and N2 |
| 5.2 | Retitle `sec:method:hardflow` away from *"In-Loop Trajectory Optimisation"*? | ⭐ **yes** — our port solves no objective (§2 fact 1); the current title is inaccurate about *our* code |
| 5.3 | Keep **arm B / arm C** as shorthand alongside the mechanism names? | keep — they are already load-bearing across the DAs and the degeneracy register; define once, then use freely |
| 5.4 | Retire "HF-SLSQP" from **new** DAs too, or only from the thesis? | thesis: yes. DAs: 🟡 the tag is in the CSV variant strings (`hardflow_sls-*`) — renaming there breaks every existing DA's cross-reference. **Recommend: keep the artefact tag, translate at the thesis boundary** |
| 5.5 | Does "α-Flow" appear in the abstract? | no — the abstract states the mechanism (*"a target bootstrapped from the network's own output"*); the brand belongs in the chapter |

---

## 6. Provenance

| claim | source |
|---|---|
| the four-step arm-C algorithm, the prox being scale-free, `Π_S(x1_ref)`, the τ pull-back, the dropped cost term, the port deltas | `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:1–40, 205–260, 423–434` (read 2026-09-10) |
| the activation gate is solver-agnostic; the `n_genuine` floor | [`Proposal_20260905_HF_minK_mf_af_unet/README.md`](../../../Data_Analysis/DA_Result_Curated_MD/Proposal_20260905_HF_minK_mf_af_unet/README.md) §0–§1 |
| arm C's published setup vs ours (H16/8-actions/IPOPT/fitted-linear vs H8/1-action/SLSQP/Euler) | `tab:hardflow-setup` in [`../Working_Space/v2/thesis_v2.tex`](../Working_Space/v2/thesis_v2.tex); [`NOTES_open_questions.md`](NOTES_open_questions.md) §8b |
| the one-axis framing of the three engines | [`FALLBACK_20260910`](../Working_Space/fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md) §2.1–2.2 |
| `diffuser` has three meanings; rules are named not lettered | [`NOTES_naming_and_rebuild.md`](NOTES_naming_and_rebuild.md) §3 |
