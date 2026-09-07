# Changelog — running BOTH NLP backends in one job, and the plot artifacts that were still unnamed

**Date:** 2026-08-28
**Scope:** aggregated fix — follow-up to `CHANGELOG_20260827_hf_nlp_backend_slsqp.md`.
Not a generation. Touches Gen12 (`FM_v3_hardflow_test`), Gen3v6 (`FM_v3_meanflow_test`),
Gen3v7 (`FM_v3_alphaflow_test`), Gen16 (`mix_visual_avoiding_test`) and the Gen12 sbatch.

---

## 1 · Why

The 2026-08-27 patch made `slsqp` the default arm-C NLP backend and renamed its **npz**
artifacts to `hardflow_sls-*` so an SLSQP run cannot overwrite the IPOPT corpus. Two gaps
were left, both found while preparing the first A/B run:

1. **No way to run both backends in one job.** The backend is resolved once per PROCESS
   (`resolve_nlp_backend` reads `FMPCC_HF_NLP_BACKEND` at policy construction), so a single
   eval invocation is single-backend by construction. Comparing IPOPT against SLSQP therefore
   meant two separate Slurm jobs — different nodes, different queue times, different machine
   load — for a comparison whose entire payload is **wall-clock**.

2. **🔴 The PLOT artifacts were still keyed on the raw variant name.** The per-variant npz
   and png inside `results/halfspace_<hv>/` went through `artifact_variant_label`, but two
   paths did not:
   - the per-seed combined grid `…/results/halfspace_<hv>/all.png`
   - the cross-seed figures `…/all_seeds/<hv>/<variant>.{png,pdf}`

   An SLSQP pass would have written `all_seeds/<hv>/hardflow_new-c-tightened.png` — the IPOPT
   plot's filename, holding SLSQP trajectories. Worse, the shared arms (`diffuser`, `dpcc-*`)
   are SKIPPED in a second pass (the `already exists` guard), so their all-seeds figures are
   **blank** in that pass and would have overwritten the good ones under their own names.
   Same class of bug as the npz overwrite, one directory up.

## 2 · What changed

### 2.1 `Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh` — `HFFM_SOLVERS`

New env knob, plus a `run_eval` helper that the existing K-sweep and single-K branches both
route through.

```
HFFM_SOLVERS=""             (default) ONE pass on the shipped default -> slsqp
HFFM_SOLVERS="ipopt slsqp"  TWO passes per K, back to back, same node, same job
```

Empty is the previous behaviour byte for byte (one invocation, no env exported). The loop is
**K-outer, solver-inner**, so the two backends for a given K run adjacent in time.

Why the two passes do not collide:

| pass | arm C writes | `diffuser` / `dpcc-*` |
|---|---|---|
| `ipopt` (first) | `hardflow_new-*` | run and measured |
| `slsqp` (second) | `hardflow_sls-*` | **skipped** — `already exists` guard |

So the baseline arms cost one run, are measured once, and BOTH hardflow arms are compared
against the identical DPCC row. An unknown backend name raises in `resolve_nlp_backend`, so a
typo fails loudly rather than silently running the default.

### 2.2 Plot artifacts now carry the backend (4 eval scripts)

Added, identically, to `eval_FM_v3_hardflow.py`, `eval_flow_matching_v3_meanflow.py`,
`eval_flow_matching_v3_alphaflow.py`, `eval_mix_visual_avoiding.py`:

- module level, before the `for exp in exps:` loop:
  ```python
  nlp_backend_run = resolve_nlp_backend()
  backend_tag = '' if nlp_backend_run == 'ipopt' else f'_{nlp_backend_run}'
  ```
- `all.png` → `all{backend_tag}.png`
- `all_seeds/<hv>/<variant>.{png,pdf}` → `artifact_variant_label(variant, nlp_backend_run)`
- `ran_variant_idx` — a per-halfspace set of the variant indices this pass actually produced.
  The all-seeds save loop skips any index not in it and closes the figure instead, so a pass
  that skipped an arm leaves that arm's existing figure alone rather than blanking it.
  Marked in the inference path, and in the `--aggregate-only` reader after its npz is found.
- `eval_FM_v3_hardflow.py` only: the manual `variant_idx = 0` / `variant_idx += 1` counter in
  the all-seeds loop became `enumerate(...)`, and the figure is now closed (it never was).

**Under `ipopt` the tag is empty and `artifact_variant_label` is the identity**, so every
legacy filename is byte-identical and nothing on disk moves.

One deliberate behaviour change that also applies to IPOPT: re-running a finished directory no
longer overwrites the all-seeds figures of the variants it skipped. Previously those were
re-saved blank.

### 2.3 Not changed

- `config/hardflow_projection_eval.yaml` — untouched. The config still asks for
  `hardflow_new-*`; only the artifact NAME depends on the backend.
- `mix_visual_aligning_test` / `mix_uav_test` — already isolated: aligning puts `variant_out`
  in the save_path itself, uav uses a per-variant `out_dir`.
- `bench_solver_hf_vs_dpcc.py` — still pins `nlp_backend='ipopt'` on purpose (see the
  2026-08-27 changelog §2.3b); without the pin it would compare SLSQP against SLSQP.

## 3 · How to run the A/B (Gen12, seed 6, K=10 and K=20)

```bash
HFFM_FLOW_STEPS="10 20" HFFM_SOLVERS="ipopt slsqp" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh --seed 6
```

Produces, per K and per halfspace geometry, in ONE directory:

```
K10_thres1_mpc4_n2/6/results/halfspace_both-hard/
    diffuser.npz                     arm A   (run once, in the ipopt pass)
    dpcc-c-tightened.npz             arm B   (run once, in the ipopt pass)
    hardflow_new-{r,c,t}[-tightened].npz     arm C, IPOPT
    hardflow_sls-{r,c,t}[-tightened].npz     arm C, SLSQP
    all.png / all_slsqp.png
```

`nlp_backend` (string) and `nlp_backend_slsqp` (float twin, for the DA `_as_float` coercion)
are in every npz; the DA variant lists already carry the `hardflow_sls*` names.

## 4 · Why the existing Gen12 data could not just be reused

The July 27 job (23890) has seed 6 at `K20_thres0.5_mpc4_n2` with the full arm-C matrix, and
that was the obvious candidate for the IPOPT half. It is not usable as one:

- **K10 does not exist at parity.** The only K10 is `K10_thres0_mpc1_n2` — mpc fan 1, bare
  `hardflow_new` only, activation threshold 0.
- **The threshold differs.** That run is `thres0.5`; the yaml default is now `1.0`.
- **Six behavioural commits landed since** (rev `18aa683` → HEAD), several of them directly on
  arm-C wall-clock or on which checkpoint loads:
  `ec0c7812` (Gen12 Fix7, batched GPU network eval), `924db516` (Fix8, activation-gate integer
  flooring), `205c494a` (Fix2, DPCC threshold wiring), `0f1aa7fc` + `83471f8d` (batch parity,
  `FMPCC_MPC_BATCH`), `96e47ac0` + `7111fb25` (terminal-step NFE skip, step budget), and
  `1ce49201` — which moved `diffusion_epoch` to `'best'`, i.e. possibly a different checkpoint.

Reusing it would have confounded the solver swap with all of the above. Both backends are
re-run instead.

## 5 · Verification done here

- All four eval scripts parse (`ast.parse`); `bash -n` on the LF form of the sbatch (the
  working tree is CRLF, the committed blob is LF via `.gitattributes text=auto`).
- `artifact_variant_label` exercised on 8 variant shapes under both backends: `ipopt` renames
  nothing, `slsqp` renames only `hardflow*`, `diffuser`/`dpcc-*`/`gradient`/`post_processing`
  pass through unchanged.
- Each of the four files verified to carry exactly one `backend_tag`, one `ran_variant_idx =
  set()`, the guard, and both renamed savefig calls.

## 6 · Not done

- **Nothing has executed.** The cluster run below is the first execution of both `_solve_slsqp`
  and this sbatch path.
- `mix_visual_meanflow`/`alphaflow`/`avoiding` have **no** clobber-skip guard in inference mode,
  so a two-pass run there re-runs the shared arms instead of skipping them. Only Gen12 gets the
  free second pass. Left alone deliberately — adding a skip guard to those evals is a separate
  change with its own blast radius.
