# PENDING — 2026-09-19 · the D3IL-avoiding tables are LOCKED; what is missing from them

**Scope: narrower than the general ledger.** This file covers `tab:avoiding-dpcc-protocol` (Table 6.1)
and `tab:state-headline` (Table 6.2) only. It supersedes
[`PENDING_20260917_dpcc_protocol_rows_table61.md`](PENDING_20260917_dpcc_protocol_rows_table61.md),
whose rows are all closed.

Author instruction, v3.42: **the avoiding tables are locked to a fixed model × budget set; anything
missing is marked pending rather than filled from a neighbouring protocol.**

Corpus of record: `Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/`
`batch_avoiding_combined_20260919_132703`.

---

## 1 · The locked set

| model | budgets | status |
| :-- | :-- | :-- |
| MeanFM | 1, 2 | ✅ complete, 5 seeds, both protocols |
| CI-MeanFM | 1, 2 | ✅ at DPCC protocol (5 seeds) · ⏳ missing at extended protocol |
| FM | 1, 2, 20 | ✅ complete, 5 seeds, both protocols |
| Diffusion | 1, 10, 20 | ✅ at DPCC protocol · ⏳ only $\nfe=20$ at extended protocol |

Budgets deliberately **excluded** from the tables, with the reason:

| cell | why it is not in the table |
| :-- | :-- |
| FM $\nfe=5$, DPCC protocol | `H8_K5_Meuler_T0.5_…FlowMatchingODE` exists for **seed 6 only**. Above the operating point; not worth four trainings' worth of evaluation. Dropped by the author's rule. |
| FM $\nfe=10$, DPCC protocol | no evaluation exists (only `T0.05` / `T0.1` solver-temperature variants, seed 6). Dropped. |
| MeanFM / CI-MeanFM $\nfe \ge 5$ | outside their operating budgets. `fig:k-ladder` carries the budget behaviour. |
| Diffusion $\nfe=5$ | **never existed** — see §2. |
| Diffusion $\nfe=2$ | exists, **seed 6 only** — see §3. |

---

## 2 · ❌ ERRATUM, closed by deletion — the published "Diffusion $\nfe=5$" row was a flow model

The row removed from `tab:avoiding-dpcc-protocol` at v3.42 ($r$ 0.933/68.8/155.8, $c$ 1.000/63.5/153.8,
$t$ 1.000/70.2/145.8) reproduces exactly from

```
logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/
    H8_Dmodels.diffusion.GaussianDiffusion_a1.5_b1.0_aw1/
    H8_K5_Mmidpoint_Dmodels.diffusion.GaussianDiffusion
```

`Dmodels.diffusion.GaussianDiffusion` under a `flow_matching_v3_*` prefix is the **flow** model under its
pre-26-May class name; commit `cac7cc6a` renamed the class to `FlowMatchingODE`. The diffusion baseline
is `Dmodels.GaussianDiffusion` under `plans/diffusion/`. The misread also carried the midpoint solver and
action weight 1, neither of which is the baseline's configuration.

**How to avoid repeating it:** select the baseline on the `Full_Path` segment `/plans/diffusion/`, never
on `Folder_Name`. Enumerating that path across the corpus gives the baseline at **$\nfe = 1$, 10, 20
only**, 5 seeds × 3 geometries × 3 rules each, reproducing the published K=1 / K=10 / K=20 cells to the
last digit.

Unaffected: `fig:k-ladder` (its `AVOIDING_T1_DIFFUSION_FOLDERS` is `{1, 10, 20}` and never held a K=5
entry) and the §6.1.1.4 prose, which already quoted 0.667 / 1.000 / 1.000 at 1 / 10 / 20.

**No diffusion $\nfe=5$ run is requested.** Nothing in the draft needs that point.

---

## 3 · ⏳ RUN — diffusion $\nfe=2$ at five seeds (author's call, not blocking)

**What exists.** Job **25965** trained a $K{=}2$ diffusion checkpoint (2 h 41 m, A5000, 2026-09-19); job
**25966** evaluated it over the full 13-variant × 3-geometry sweep, two episodes, tightened and
untightened, projected and unprojected. **Seed 6 only.**

Results, seed 6, aggregated by geometry:

| variant | S&C | steps | ms/step |
| :-- | --: | --: | --: |
| `dpcc-r-tightened` | 1.00 | 64.3 | 225 |
| `dpcc-c-tightened` | 1.00 | 61.3 | 226 |
| `dpcc-t-tightened` | 0.83 | 62.7 | 276 |
| `dpcc-r` / `dpcc-c` / `dpcc-t` | 0.33 / 0.67 / 0.33 | 68.8–77.8 | 198–224 |
| `diffuser` (unprojected) | 0.00 | 59.0 | 18 |

**Why it is not tabled.** Every other cell of `tab:avoiding-dpcc-protocol` is five seeds. A single-seed
row is not a peer and would be read as one.

**Where it is used instead.** Two places, both single-seed and labelled as such:
* §6.1.1.4 prose — ≈ 225 ms per control step at two denoising steps, against FM's 17.3 ms at one; the
  point being that cutting the budget from 20 to 2 does not recover the gap, because the cost sits in the
  projector.
* §6.1.2 — the unprojected arm scores 0.00 S&C with 20.5 violating steps per episode, which is what keeps
  the new `fig:raw-plans` panel from reading as a success.

**To close it:** four trainings (seeds 7–10) at the 25965 configuration, then one evaluation over all
five. Cluster disk was at 43 GB free on 2026-09-19 — check before submitting.

---

## 4 · ⏳ EVALUATIONS ONLY — the extended protocol (no training needed)

`tab:state-headline` is the same model × budget set as Table 6.1, restricted to what has five seeds at
twenty episodes. Three cells are missing and all three are evaluations of **checkpoints that already
exist**:

| cell | checkpoint | note |
| :-- | :-- | :-- |
| CI-MeanFM $\nfe=1$, extended | exists (U-Net, `ae0.2`) | extended protocol was run for seed 6 only |
| CI-MeanFM $\nfe=2$, extended | same checkpoint | ODE budget is an inference dial |
| Diffusion $\nfe=1$ and $\nfe=10$, extended | `plans/diffusion/H8_K1_…`, `H8_K10_…` | would show the baseline's budget collapse at 100 episodes per geometry rather than 10 |

Marked in the draft by a `\hole` under `tab:state-headline`.

---

## 5 · ✅ Closed by this pass

* `fig:raw-plans` — the eighth panel (`fig_raw_plans_diffusion_K2`) is in the draft; `todofigure` is 0.
  See [`PENDING_20260919_fig63_diffusion_K2_panel.md`](PENDING_20260919_fig63_diffusion_K2_panel.md) for
  the run record.
* `tab:avoiding-dpcc-protocol` — all twelve FM / CI-MeanFM cells landed at v3.41; the bogus diffusion
  $\nfe=5$ row removed at v3.42. **Locked.**
* `tab:seed-spread` — CI-MeanFM row added (5 seeds, DPCC protocol, marked as a different protocol).
