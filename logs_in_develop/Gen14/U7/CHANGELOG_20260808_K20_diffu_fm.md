# CHANGELOG — Gen14 U7: default K 100 → 20 for the `diffusion` and `fm` arms

**Date:** 2026-08-08
**Trigger:** visual inspection of the two new K=100 runs — the `fm` and `diffusion` arms still
look like the old Gen6V4/Gen7 failures — plus the question of why either arm was defaulting to
K=100 at all.
**Companion:** `DA_20260808_gen14_diffu_fm_arms.md` (the data these changes react to)

## What changed

| file | change | effect |
|---|---|---|
| `config/aligning-d3il-visual.py:986` | `mix_visual_aligning_diffusion` (**train** block): added `'n_diffusion_steps': 20` | **retrain required** — new checkpoint tree `mix_visual_aligning_diffusion/H8_K20_…_Ediffusion` |
| `config/aligning-d3il-visual.py:1086` | `plan_mix_visual_aligning_fm`: added `'flow_steps_v3': 20` | **eval-only** — same checkpoint, new results dir `H8_K20_Meuler_T0.5_…_Efm` |
| `Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh:81` | comment: NFE default `fm: 100` → `fm: 20`, plus a line on why `diffusion` has no override | doc only |

`mf` and `af` were **already** at `flow_steps_v3: 2` (config lines 1105 and 1117, set in U6).
No change was needed and none was made.

Resulting NFE per arm:

| arm | K | key | where |
|---|---|---|---|
| `diffusion` | **20** | `n_diffusion_steps` | train block (checkpoint identity) |
| `fm` | **20** | `flow_steps_v3` | plan block (eval only) |
| `mf` | 2 | `flow_steps_v3` | plan block (eval only) |
| `af` | 2 | `flow_steps_v3` | plan block (eval only) |

## Why K=100 was the default in the first place

Two independent inheritances, neither of them a decision:

- **`fm`** — `_mix_plan_common` copies every key from `plan_fm_visual_aligning`, whose
  `flow_steps_v3` is **100** (config line 659). The `mf` and `af` blocks each override it to 2
  (U6); the `fm` block passed `{}`, so it silently kept the inherited 100. Gen7's own archived
  eval folders are all `H8_K20_…`, so the "reference arm" was running at 5× Gen7's NFE.
- **`diffusion`** — `_mix_train_block('diffusion', 'visual_aligning_dpcc', …)` inherits
  `n_diffusion_steps: 100` from the Gen6V4 training block. That 100 was introduced in commit
  `2c87cb70`, **after** the Gen6V4 artefacts in every comparison batch were produced — those
  are `H8_K20_…` in both the checkpoint and the results folder. So this arm was a 5×-NFE
  variant of Gen6V4, not Gen6V4.

Both arms were therefore off-parity with the generation they exist to reproduce, in the same
direction, for unrelated reasons.

## The asymmetry that matters operationally

`flow_steps_v3` is inference-only: the `fm` change costs nothing, reuses the existing
checkpoint, and lands in a sibling results folder. `n_diffusion_steps` is **not** — it is the
DDPM chain length, so it sets the training noise schedule, and it is a checkpoint-path key in
`args_to_watch_mix_visual_train`. The `diffusion` arm therefore **needs a retrain** (~4.5 h at
the observed 2.4–2.7 min/epoch × 100 epochs). This is enforced, not merely documented: both
`eval_mix_visual_aligning.py:2444` and `mix_visual_aligning_pipeline.sh:92` reject an NFE
override on `engine=diffusion` with an explicit error.

The override was placed in the Gen14 arm's own block, **not** in `visual_aligning_dpcc`, so
Gen6V4's train and plan blocks are untouched (repo copy-modify convention). One consequence:
`visual_aligning_dpcc` still says 100 while its own archived artefacts are K=20 — a pre-existing
drift this change neither creates nor fixes.

## Secondary effect: projection budget

The sampler projects on every step from `int((1 - diffusion_timestep_threshold) * K)` to the
end. At T=0.5 that is **50 SLSQP solves per replan at K=100, 10 at K=20**. The projected
variants are where the 24 h cap was lost — both K=100 evals died in item 2/32 (`dpcc-r`), which
alone had a ~26–29 h ETA. K=20 should bring a `dpcc-r` item back under the cap; it does not by
itself make a 32-item sweep fit.

## To run on the cluster

```bash
# fm — eval only, reuses the existing checkpoint. Config default is now 20.
sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh fm 6 all

# diffusion — MUST retrain first (new K=20 checkpoint tree), then eval.
sbatch Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh diffusion "6"
```

Do not pass a `$3`/`$4` NFE argument for `diffusion` — it will exit 1 by design.

## Verification done here

- `python3 -m py_compile config/aligning-d3il-visual.py` — passes.
- `bash -n mix_visual_aligning_pipeline.sh` — passes.
- Confirmed `n_diffusion_steps` appears in `args_to_watch_mix_visual_train` (line 842), so the
  plan block inherits 20 through `_mix_plan_block`'s mirror loop — it is deliberately **not**
  set a second time in `plan_mix_visual_aligning_diffusion`.

**Not verified here — no Python env in this container:** that the config actually resolves and
that `diffusion_loadpath` points at the new K=20 tree. **Run on cluster.** The first `diffusion`
train job is the check: if the loadpath is wrong the eval dies immediately with a
`FileNotFoundError` on a `H8_K20_…` directory.
