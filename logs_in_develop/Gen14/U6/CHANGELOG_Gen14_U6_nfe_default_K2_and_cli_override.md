# Gen14 U6 — mf/af eval default NFE → K=2, and a working `--flow-steps` CLI override

**Date:** 2026-08-04
**Scope:** eval/plan side only. No model code, no training code, no checkpoint touched.
**Trigger:** `logs_in_develop/Gen14/U5/DA_20260804_mf_af_visual_aligning_first_run.md` §7 — the first
mf/af visual eval ran at **K=100**, inherited verbatim from the FM plan template, on two models whose
entire premise is few-step sampling.

**Two changes:**

1. `flow_steps_v3: 2` added to the `mf` and `af` **plan** blocks (their new default).
2. A `--flow-steps K` CLI override that actually works, plumbed through both sbatch scripts.

---

## 1. Why K=100 was wrong, and why it was expensive twice

`plan_fm_visual_aligning` (`config/aligning-d3il-visual.py:659`) sets `flow_steps_v3: 100` with the
comment *"100 matches the visual_aligning_dpcc baseline (K=100 denoising steps) for fair comparison."*
That reasoning is sound **for the `fm` arm** — Gen7 flow matching integrates a velocity field 0→1 and
DDPM-parity is the honest comparison.

`_mix_plan_common` copies every key from that block except `prefix` / `exp_name` / `diffusion` /
`diffusion_loadpath`, so `mf` and `af` silently inherited it. Nothing in either plan block overrode it.

For a two-time model this is not a conservative choice, it is a category error. MeanFlow's `u(x, r, h)`
predicts the **average** velocity over an interval of length `h`; the method exists so that one query
can span the whole path. The Gen3v6/v7 state-only lineage evaluates at **K=2**
(`logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md`), where AF+`dpcc-r-tightened`
was the only flow-family cell perfect in all three envs while being 15–18× faster than DPCC.

### 1.1 The cost that was not obvious

K does not only set the number of backbone evaluations. It sets **the projection budget**.
`mf_diffusion.py:284`:

```python
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * flow_steps)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps - 1)
```

and the projector fires on **every** step from `snapping_start_idx` to the end. At `T = 0.5`:

| K | `snapping_start_idx` | backbone evals / replan (× `mpc_batch_size=4`) | **SLSQP solves / replan** |
|---:|---:|---:|---:|
| 100 | 50 | 400 | **50** |
| 4 | 2 | 16 | 2 |
| **2** | **1** | **8** | **1** |
| 1 | 0 | 4 | 1 |

The U5 DA measured `dpcc-r` at 15–16 s/replan against 0.89 s for generation alone. The missing ~14 s
was never the generator — it was **~50 SLSQP solves at ~0.28 s each**. Both `dpcc-r` jobs died at
rollout 11/30 against the 24 h cap, with `~114 413 s` / `~121 134 s` still to go (≈ 50 h and 53 h
projected).

So K=2 buys a ~50× cut on the NFE axis *and* a ~50× cut on the projection axis simultaneously.
Rough projection for the 30-context `dpcc-r` sweep: **~50 h → ~1 h**.

⚠️ **Caveat, stated plainly:** fewer projection steps *is* less constraint enforcement per plan. K=2
`dpcc-r` constraint numbers are **not** directly comparable to the K=100 ones already in
`temp/0408/`. They are a different operating point, not a cheaper computation of the same one. K=2 +
`dpcc-r-tightened` is the configuration the state-only DA validated, so it is an established operating
point rather than a shortcut — but any table mixing K=2 and K=100 rows must say so.

---

## 2. Change 1 — the config default

**Not achievable by editing an existing line.** The `mf`/`af` plan blocks contain no `flow_steps_v3`
entry to change; they inherit one. The only existing line is `:659` in `plan_fm_visual_aligning`, and
editing *that* would hit three consumers at once:

- Gen7's own eval, which reads `experiment='plan_fm_visual_aligning'` directly — a different
  generation's published operating point;
- Gen14's **`fm` reference arm**, which must stay at Gen7's K for the comparison the generation exists
  to make;
- (`diffusion` is unaffected either way — it `drop`s every continuous-time key at `:1053`.)

So two new keys were added instead, one per two-time arm:

```python
base['plan_mix_visual_aligning_mf'] = _mix_plan_block(
    'mf', base['mix_visual_aligning_mf'], {
        'flow_steps_v3': 2,          # ← U6
        't_schedule': 'logit_normal',
        ...
```

`af` gets the same value on purpose: NFE is an operating point, and an mf-vs-af comparison at
different K would confound the objective with the step budget.

### 2.1 Why this is safe — the clobber trap that does not fire

`_mix_plan_block` contains a loop that unconditionally mirrors every training identity key onto the
plan block, overwriting whatever the `overrides` dict put there:

```python
for key, _label in args_to_watch_mix_visual_train:
    if key == 'prefix' or key not in train_blk: continue
    blk[plan_key] = train_blk[key]          # unconditional by design
```

An override placed in `overrides` for any key in that watch list would be **silently discarded**.
`flow_steps_v3` is not: it appears in `args_to_watch_mix_visual_plan` (`:860`) and **never** in
`args_to_watch_mix_visual_train` (`:839-853`), and the mf/af *training* blocks never define it
(`:442-443` documents it as "DEAD in training"). Verified by execution, §5.

Consequences, both verified:

- `exp_name` → `H8_K2_Meuler_T0.5_…` — a **new sibling results directory**. The existing K=100 results
  under `temp/0408/` are not overwritten.
- `diffusion_loadpath` and `prefix` are built from *training* keys only and are **byte-identical**
  before and after. Same checkpoint, no retraining, no re-derivation of any path the trainer wrote.

---

## 3. Change 2 — a `--flow-steps` CLI override that works

### 3.1 Two routes that look right and are not

**(a) Pass `--flow_steps_v3 2` through to diffuser's `Parser`.** Does not work, and fails loudly.
`utils.Parser` is a plain `argparse.ArgumentParser` declaring only `--config` and `--seed`
(`diffuser/utils/setup.py:43-48`), so an unknown flag is a hard `unrecognized arguments` exit. Its
generic override hook `add_extras()` — which is exactly what this would need — **is commented out** at
`diffuser/utils/setup.py:77`. Uncommenting it is not an option either: it reads `args.extra_args`, a
field `Parser.__init__` never defines, so it would `AttributeError` on every generation in the repo
that uses this Parser. That is presumably why it is commented out.

> The comment at `eval_mix_visual_aligning.py:2364` ("Fix 5: override flow_steps_v3 from args so
> Slurm `--flow_steps_v3 N` actually changes ODE integration steps") is **stale on the CLI claim**.
> What that code actually does — and does correctly — is push the *eval config's* value into the model,
> overriding the value baked into `diffusion_config.pkl` at training time. That mechanism is untouched
> by U6 and is why the config default in §2 reaches the sampler at all. Only the "Slurm `--flow_steps_v3`"
> half of the sentence was never true.

**(b) Assign `args.flow_steps_v3` after `Parser().parse_args()`.** Silently mislabels the run.
`eval_fstrings()` and `generate_exp_name()` both execute **inside** `parse_args()`
(`diffuser/utils/setup.py:76-81`), so the sampler would honour the new K while the results folder still
carried the old one. That is precisely the failure this file already warns about where it prints the
NFE banner: *"Printing one label for both is how a run gets read as K=100 when it actually ran K=1."*

### 3.2 The route that works

Mutate the config module's plan block **before** any `Parser().parse_args()` call.
`read_config()` does `importlib.import_module(args.config)` then `getattr(module, 'base')[experiment]`,
and importlib caches modules — so one mutation is what every seed in the loop reads, and both the
sampler and the folder name derive from the same dict. `exp_name` is `watch(args_to_watch_mix_visual_plan)`,
a **callable** resolved at parse time against `args`, and `args` receives `flow_steps_v3` from this dict
via `read_config`'s setattr loop. Agreement is structural, not maintained by hand.

```python
if args_cli.flow_steps is not None:
    _plan_blk = importlib.import_module(Parser.config).base[EXPERIMENT]
    if 'flow_steps_v3' not in _plan_blk:
        raise SystemExit(f"[ eval ] ERROR: --flow-steps does not apply to engine '{ENGINE}' — "
                         f"its NFE key is '{ENGINE_SPEC['nfe_key']}', set via '{EXPERIMENT}'.")
    ...
```

The guard is real, not defensive padding: the `diffusion` plan block genuinely has no `flow_steps_v3`
(confirmed by execution, §5), because `_mix_plan_block(drop=…)` removes every continuous-time key from
that arm. Without the guard a `--flow-steps` on the diffusion arm would insert a key that arm's
constructor never reads, and the run would proceed at the wrong K with a folder name claiming otherwise.

The override also prints the projection-budget consequence, since that is the half people do not
expect:

```
[ eval ] --flow-steps: flow_steps_v3 2 -> 4  (applies to the sampler AND the results folder name)
[ eval ] --flow-steps: projection budget 1 -> 2 projector call(s) per replan at threshold T=0.5
```

---

## 4. Files changed

| file | change |
|---|---|
| `config/aligning-d3il-visual.py` | `'flow_steps_v3': 2` + rationale comment in `plan_mix_visual_aligning_mf` and `…_af`. `plan_fm_visual_aligning:659` (K=100) **untouched** — Gen7 parity and the Gen14 `fm` reference arm both depend on it. |
| `mix_visual_aligning_test/eval_mix_visual_aligning.py` | `import importlib`; `--flow-steps K` argparse flag (with the two-anti-pattern rationale inline); the pre-`parse_args` config-block mutation + engine guard + projection-budget print. |
| `Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh` | new positional `$4` = NFE. Empty → no flag → config default. Rejects `diffusion` with an explicit message before the GPU is touched. `$FLOW_ARG` is deliberately unquoted so it vanishes when empty. |
| `Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh` | new positional `$3` = NFE, forwarded to every eval in the per-seed fan-out (as `$4`, with `all` passed positionally in `$3` so the slots line up). Same `diffusion` guard, at submit time. Training stage untouched — `flow_steps_v3` is inference-only. |

Nothing in `mix_visual_aligning/` was modified. **G0 re-run: PASS, 23/23 verbatim files still match.**

---

## 5. Verification

All static — this container has no torch and no GPU.

| check | result |
|---|---|
| `ast.parse` on both modified `.py` | OK |
| `bash -n` on all four `mix_visual_aligning` sbatch scripts | OK |
| `gates_mix_visual.py --gate g0` | **PASS — 23 verbatim files match** |
| Config executes; `mf`/`af` plan blocks carry `flow_steps_v3 == 2` | ✅ |
| `mf`/`af` `exp_name` resolves to `H8_K2_Meuler_T0.5_…` | ✅ |
| `fm` plan block still `flow_steps_v3 == 100`, `exp_name` still `H8_K100_…` | ✅ (Gen7 parity intact) |
| `diffusion` plan block has **no** `flow_steps_v3` → U6 guard fires | ✅ |
| `diffusion_loadpath` byte-identical for mf/af before and after | ✅ (same checkpoint) |
| `importlib.import_module(...).base[EXPERIMENT] is base[EXPERIMENT]` | ✅ (mutation route valid) |
| Simulated `--flow-steps 4` → `exp_name` becomes `H8_K4_…`, `diffusion_loadpath` unchanged | ✅ |

The config was executed with a minimal `diffuser.utils.watch` stub injected into `sys.modules`
(the repo's own implementation, `diffuser/utils/setup.py:21-36`, copied verbatim) because `diffuser`
does not import in this container.

**Not verified here — needs the cluster:**

- That K=2 sampling is numerically sensible on these checkpoints (a two-time model trained with
  `logit_normal` t-sampling and evaluated at 2 steps is exactly the regime the U5 DA found
  under-trained in the large-`h` buckets).
- The projected `dpcc-r` wall-clock saving. The 50→1 solve count is read off the code; the ~0.28 s/solve
  is inferred by subtraction from the U5 measurements.
- `--flow-steps` end-to-end through Slurm.

---

## 6. Commands

```bash
# mf/af now default to K=2 — no flag needed
sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6

# explicit sweep; each K lands in its own H8_K<N>_... results folder
sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6 all 1
sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6 all 4
sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh af 6 all 2

# full pipeline with an NFE override on the eval stage
sbatch Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf "6 7" 2

# rejected before the GPU is allocated
sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh diffusion 6 all 2
```

Confirm from the log banner:

```
[ eval ] --flow-steps: flow_steps_v3 2 -> 4  (applies to the sampler AND the results folder name)
[ eval ] flow_steps_v3 = 4  (Euler ODE integration steps 0→1) [overridden from args]
[ eval ] NFE (flow_steps_v3) = 4   engine=mf
```

---

## 7. Still open

Carried from U5's DA §10, none of it addressed by U6:

1. **Re-evaluate `af` at step 70 000** (pre-α-cliff). U6 does not change checkpoint selection —
   `eval_mix_visual_aligning.py:2284` is still `epoch='latest'`, so `af` is still evaluated 29 k steps
   past its own optimum at 2.62× worse test `raw_mse_u`. An `--epoch` CLI flag would use the exact
   mechanism built here and is the obvious U7 candidate.
2. **The α cliff itself** — one `af` run with `af_alpha_clamp` lowered (e.g. 1e-4) or `af_alpha_end` set
   to a small positive value, to separate "the JVP target is genuinely worse here" from "the clamp is a
   target-distribution shock". Highest-value single job in the queue.
3. **`gradient_clip: 1.0`** clips 100 % of steps at a median pre-clip norm of 67–73. Inherited from
   state-only Gen3v6/v7 where the norm is ~1/70 of this.
4. **FiLM v2 has still never been trained or evaluated**, and **G7 has still never run** (U5 §7).
5. Whether `n_contexts: 30` × 12 projection variants is the right benchmark shape even at K=2.

---

## 8. Note on the master index

`logs_in_develop/MASTER_TEST_HISTORY.md` is not edited by this changelog. If the Gen14 row should
record the new default NFE and the `--flow-steps` flag, say so and I will add it.
