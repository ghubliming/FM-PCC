# Gen14 Fix_9 — `film_mode` gets an explicit config entry on all four arms

**Date:** 2026-08-09
**Scope:** `config/aligning-d3il-visual.py`, `mix_visual_aligning_test/train_mix_visual_aligning.py`,
`mix_visual_aligning_test/eval_mix_visual_aligning.py`, and the three
`Slurm_Codes/sbatch/mix_visual_aligning/` scripts
**Type:** discoverability + a launch-time switch. **Default unchanged (`v1`) on all four arms
— no path change, no retrain, no existing run affected.**

## 1. The problem

FiLM v2 has been fully supported on all four Gen14 arms since U5 — `Flow_matcher_U_Net_v2_FiLM`
exists, `VisualUNetTwoTime` dispatches to it, gate G7 tests it. But **none of the four Gen14
arm blocks stated `film_mode`.** The value arrived by inheritance:

| block | lines | source of `film_mode` |
|---|---|---|
| `visual_aligning_dpcc` (diffusion's parent) | 344–415 | `'v1'` at line 377 |
| `fm_visual_aligning` (fm/mf/af's parent) | 416–503 | `'v1'` at line 461 |
| the four `mix_visual_aligning_*` arms | 971–1058 | **absent — inherited** |

So reading the Gen14 section gave no indication the knob existed, let alone that v2 was
reachable. Combined with the fact that there is **no `--film-mode` CLI override anywhere in
the repo**, the honest reading of the config was "Gen14 can't do v2" — which is false.

Why no CLI override exists, for the record: `diffuser`'s generic
"override any config key" mechanism (`Parser.add_extras`) is **dead** — the call is
commented out at `mix_visual_aligning/utils/setup.py:77`, identically in
`diffuser/utils/setup.py:77` and in upstream `aux_repo/dpcc/diffuser/utils/setup.py:65`.
Uncommenting it would not help either: it reads `args.extra_args`, and the base `Parser`
(`mix_visual_aligning/utils/setup.py:42`) is a plain `argparse.ArgumentParser` declaring only
`--config` and `--seed`; the Tap parser that supplied `extra_args` was dropped upstream.
`film_mode` is also a **training** key, so an eval-side flag (the U6 `--flow-steps` trick)
would be useless on its own — the checkpoint path has to move with it.

## 2. Changes

### (a) `config/aligning-d3il-visual.py` — shared comment block, env knob, four entries

New block comment before the arm definitions documenting: what v1 and v2 are, which backbone
file each arm reaches, that it is a **training** key requiring a retrain, that v1/v2
state_dicts are not interchangeable (`embed_dim` differs, `film_proj` has no v1 counterpart,
~+1.0M params), why no CLI override is possible, the zero-init caveat from U5 §2.8, and that
G7 must not be bypassed on a v2 run.

Then the knob itself. **Each of the four mix candidates gets its OWN entry and its OWN
environment variable**, resolved by a shared helper:

```python
def _film_mode(engine):
    # 1. MIX_FILM_MODE_<ENGINE>   this arm only        2. MIX_FILM_MODE   all arms
    # 3. 'v1'                     default
```

Per-arm is the primary form because the arms are **separate experiments with separate
checkpoint trees**: putting mf on v2 must not drag `fm` — the Gen7 reference arm — along with
it. The bare `MIX_FILM_MODE` survives as an all-arms convenience, and the two compose, so
"sweep everything to v2 but hold fm at v1" is expressible:

```bash
MIX_FILM_MODE=v2 MIX_FILM_MODE_FM=v1 ...
```

An unknown value **raises** rather than falling back to `v1` — a silent fallback would train
the wrong architecture into a directory whose name claims otherwise. This mirrors the
env-var pattern already used for HardFlow in `config/avoiding-d3il.py:1334`
(`HFFM_FLOW_STEPS`).

Each arm then carries its own call, with a per-arm note:

| arm | knob | note |
|---|---|---|
| `mix_visual_aligning_diffusion` | `MIX_FILM_MODE_DIFFUSION` | backbone `VisualUNet` → `unet1d_temporal_film` |
| `mix_visual_aligning_fm` | `MIX_FILM_MODE_FM` | not tuning — the default `v1` is byte-for-byte the inherited value, so G1's Gen7 training-parity check is unaffected. 🔴 Running this arm at v2 breaks that parity *by design*: it is the Gen7 reference, so a v2 fm run is a NEW arm, not a reproduction. **The per-arm knob is what keeps an mf/af v2 sweep from moving it silently.** |
| `mix_visual_aligning_mf` | `MIX_FILM_MODE_MF` | backbone `VisualUNetTwoTime` → `Flow_matcher_U_Net_v2_FiLM`, retains `h_mlp` |
| `mix_visual_aligning_af` | `MIX_FILM_MODE_AF` | independent of mf's. ⚠️ mf-vs-af is architecture-controlled only if **both** arms run the same mode — independent knobs make that the operator's responsibility, so the block says so explicitly |

**Training blocks only.** The plan blocks are untouched: `film_mode` is in
`args_to_watch_mix_visual_train` (line 850), so `_mix_plan_block`'s training-key mirror loop
copies it across. Setting it in both is the double-set that loop exists to prevent.

🔴 **Scope.** These variables are read by the four Gen14 arms *only*. The Gen6V4/Gen7 parent
blocks keep their own hardcoded `'v1'` (lines 377 / 461), so a stray env var can never move a
Gen6V4 or Gen7 run — verified in §3.

### (b) `train_mix_visual_aligning.py` — validate + print

Before backbone construction: reject an unknown mode with a `SystemExit` (seconds, instead of
after the dataset load) and print

```
[ train ] film_mode = v1 (additive-bias FiLM (default)) — architecture key; v1/v2 checkpoints are NOT interchangeable
```

alongside the existing `n_diffusion_steps` breadcrumb, so a v1/v2 mix-up is visible at the top
of the log rather than only in a directory name.

### (c) `eval_mix_visual_aligning.py` — breadcrumb + third-case guard

In `load_diffusion_with_override`, after the configs load: read `film_mode` out of the
train-time `model_config.pkl` (`vis_config` for mf/af, `config` for diffusion/fm) and print
which mode the loaded backbone actually is.

The architecture was already safe by construction — `model = model_config()` rebuilds from the
pkl, never from the eval config, and `'..._film{film_mode}'` is in `diffusion_loadpath` so a
mismatch normally fails as a missing directory. The new `WARNING` covers the one case both
guards miss: **pkl and eval config disagreeing while the path still resolves**, where
`exp_name` (built from the EVAL args, `args_to_watch_mix_visual_plan:866`) would label the
results folder with a film mode the weights do not have.

### (d) `Slurm_Codes/sbatch/mix_visual_aligning/` — resolve, validate, narrow, propagate

All three scripts accept **either** input form (`MIX_FILM_MODE_<ENGINE>` or the broadcast
`MIX_FILM_MODE`) and run the same four steps, because each of these jobs handles exactly one
arm:

```bash
ENGINE_UC=$(echo "$ENGINE" | tr '[:lower:]' '[:upper:]')
eval "FILM_MODE=\${MIX_FILM_MODE_${ENGINE_UC}:-\${MIX_FILM_MODE:-v1}}"
case "$FILM_MODE" in v1|v2) ;; *) echo ERROR; exit 1 ;; esac
unset MIX_FILM_MODE                              # drop the broadcast form
export "MIX_FILM_MODE_${ENGINE_UC}=$FILM_MODE"   # republish arm-specifically
```

`mix_visual_aligning_pipeline.sh` additionally **validates at submit time** — a typo dies
before a GPU allocation, not after — and passes
`--export=ALL,MIX_FILM_MODE_<ENGINE>=<mode>` to both the train and eval child jobs. Explicit
rather than relying on inherited `--export=ALL`: a silently-dropped env var would train v1
while the operator believes it is running v2.

**Why the `unset` matters.** A bare `MIX_FILM_MODE` on the submitting shell rides along
through `--export=ALL` regardless of what the pipeline exports. Without dropping it, the
other three arm blocks would also resolve v2 when the config module imports inside the job.
Harmless in effect — only `$ENGINE`'s block is ever consumed — but it would make the config's
own resolution disagree with what the job is doing, and latent disagreements of exactly that
shape are what this generation keeps getting bitten by.

The gates job deliberately gets nothing — G7 builds all four arms at v2 regardless, so the
gates are film-mode independent exactly as they are seed independent.

## 3. Verification (local container — no torch, no project env)

| check | result |
|---|---|
| `ast.parse` on all 3 edited Python files | pass |
| `bash -n` on all 4 sbatch scripts | pass |
| mf path (default) vs job 24124's actual savepath | **exact match** (`H8_D…VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal`) |
| K keys unchanged | diffusion `n_diffusion_steps=20`; fm `flow_steps_v3=20`; mf/af `flow_steps_v3=2` |
| plan mirror loop | asserted `train == plan == path tag` on **every** row below — plan is never set by hand |
| **scope**: Gen6V4/Gen7 parents, every row | `visual_aligning_dpcc`, `fm_visual_aligning` and both their plan blocks stay `'v1'` ✅ |

Config resolution matrix (real config, stub `watch`/`yaml`):

| env | diffusion | fm | mf | af |
|---|---|---|---|---|
| *(nothing set)* | v1 | v1 | v1 | v1 |
| `MIX_FILM_MODE=v2` | v2 | v2 | v2 | v2 |
| `MIX_FILM_MODE_MF=v2` | v1 | v1 | **v2** | v1 |
| `MIX_FILM_MODE_MF=v2 MIX_FILM_MODE_AF=v2` | v1 | v1 | **v2** | **v2** |
| `MIX_FILM_MODE=v2 MIX_FILM_MODE_FM=v1` | v2 | **v1** | v2 | v2 |
| `MIX_FILM_MODE_DIFFUSION=v2` | **v2** | v1 | v1 | v1 |
| `MIX_FILM_MODE_MF=V2` / `MIX_FILM_MODE=true` | rejected with `ValueError` — no silent v1 fallback | | | |

Sbatch narrowing logic, exercised standalone:

| input | engine | resolved | env reaching Python |
|---|---|---|---|
| `MIX_FILM_MODE=v2` | mf | v2 | `MIX_FILM_MODE_MF=v2` (broadcast dropped ✅) |
| `MIX_FILM_MODE_MF=v2` | mf | v2 | `MIX_FILM_MODE_MF=v2` |
| `MIX_FILM_MODE=v2 MIX_FILM_MODE_FM=v1` | fm | **v1** | `MIX_FILM_MODE_FM=v1` |
| *(nothing)* | af | v1 | `MIX_FILM_MODE_AF=v1` |
| `MIX_FILM_MODE_MF=V2` | mf | — | exits 1 |

Config resolution was exercised by importing the real file against a stub `diffuser.utils.watch`
and a stub `yaml` returning the one key the config reads
(`diffusion_timestep_threshold`). **Not run locally:** anything needing torch — G0–G7 are
cluster jobs.

## 4. Blast radius

**Zero with the env var unset**, which is the default and the committed state. Every resolved
value equals what was already inherited, so no checkpoint tree, no results folder and no
existing run is affected. Setting `MIX_FILM_MODE=v2` opts a run into a **parallel** `filmv2`
tree; it cannot overwrite a v1 run, because the mode is a path key.

## 5. Running mf/af at FiLM v2, K=2 — train + eval

The pipeline chains gates → train → eval per seed. v2 is a **training** key, so a re-eval of
existing v1 weights is not an option — this is a retrain.

```bash
MIX_FILM_MODE_MF=v2 ./Slurm_Codes/submit.sh \
    Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf "6"

MIX_FILM_MODE_AF=v2 ./Slurm_Codes/submit.sh \
    Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af "6"
```

- **Per-arm variable, matching the engine argument.** `MIX_FILM_MODE=v2` also works — the
  pipeline narrows it to the arm it was given — but the explicit form documents intent and
  cannot leak to another arm if the command is later copy-pasted with a different engine.
- **No `$3`.** K stays at the mf/af config default `flow_steps_v3: 2`, which is both the
  Gen3v6/v7 lineage operating point and what the existing v1 mf/af runs used — so the
  v1-vs-v2 read stays NFE-controlled. Passing `2` explicitly is equivalent.
- **Seed `"6"`** matches the existing v1 material (the 08-08/08-09 DA batches are seed 6 only).
  Omit `$2` for the `6 7 8 9 10` fan-out — one job per seed, each with its own 24 h wall.
- **Both arms must be at the same mode for the mf-vs-af comparison to mean anything** — the
  knobs are independent by design, so this is on the operator. The two commands above do
  satisfy it.
- `submit.sh` exports `ALL`, so the variable reaches the pipeline job, which re-exports it
  arm-specifically onto train and eval.

Expected output tree:

```
mix_visual_aligning_mf/H8_D…VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv2_Emf_tslogit_normal/seed6/
```

**Check three lines before trusting the run:**

1. `[ pipeline ] film_mode = v2 …` at submit
2. `[ train ] film_mode = v2  (MIX_FILM_MODE_MF; …)` from the sbatch script, then
   `[ train ] film_mode = v2 (TRUE FiLM — per-block gamma scale + beta shift) …` from Python
3. `filmv2` in the savepath (not `filmv1`)

**Do not bypass the gates job.** v2 has still never executed a tensor op on any Gen14 arm; G7
is the first thing that will, and it runs by default (`--gate all`).

**Reading the curves:** `W_f` is zero-initialised and under v2 the visual latent reaches the
network only through `W_f`, so at step 0 a v2 model is exactly v1-with-no-vision. Early-epoch
v1-vs-v2 curves are not comparable step-for-step (U5 §2.8).

### diffusion / fm at v2

Both arms accept `MIX_FILM_MODE=v2` and have the entry, but no v2 run is scheduled for them
here. For `fm`, note that a v2 run breaks G1's bit-identical-to-Gen7 training-parity check by
design — that arm is the Gen7 reference, so a v2 fm run is a new arm, not a reproduction.
