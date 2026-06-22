# Gen9 E2 U4 — Fix 2: visual-avoiding eval only runs seed 6 (yaml says 6,7,8,9,10)

**Date:** 2026-06-22
**Script:** `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py`
**Submit:** `Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh`
**Symptom:** `projection_eval.yaml` has `seeds: [6,7,8,9,10]`, all five checkpoints exist (6 pre-existing,
7-10 trained on remote), eval launched for all five — but the SLURM logs/outputs contain **only seed 6**.

---

## TL;DR — verdict

**There IS a real bug, and your mental model is correct.** With `seeds: [6,7,8,9,10]` in the yaml and no
`--seed` on the `.sh`, the eval *should* process all five. It doesn't, because the eval script has **zero
per-seed fault isolation**: the per-seed model load is outside any `try`, and the per-variant `try` block
has a `finally:` but **no `except:`**. So the **first** seed that throws *anything* (a missing artifact, an
OOM, a solver/projection error, even a matplotlib error) **kills the whole job** — and every seed after it
never runs. Seed 6 is first in the list and completes; seed 7 throws → process dies → 8/9/10 never start.

This is **not** the `.sh` forcing a single seed (it correctly passes no `--seed` when given no positional
arg — which is the pipeline path). The bug is in the Python eval's robustness.

---

## ⚠⚠⚠ UPDATE 3 (2026-06-22) — THE FIX: rewrite the dead `visual_avoiding_eval.yaml` & reactivate it

**Decision (user):** `config/visual_avoiding_eval.yaml` is a **giant mess — it was cloned from visual
*aligning*** (it carries `geo_constraint_variants`, `active_geo_variants`, `n_contexts`,
`n_trajectories_per_context`, `mpc_foresight_stride`, scalar `dt: 1` — all aligning-eval concepts). That
makes **no sense for visual avoiding**, which is just **avoiding run on pixels** — the workspace, obstacles,
half-spaces, bounds and dims are **identical to state avoiding**. The only thing that changes is the
observation source (camera vs state), and that comes from the **model** (the eval Parser hard-codes
`config.avoiding-d3il-visual`), not from the eval yaml.

**Why projection_eval.yaml already works for visual avoiding:** the active eval reads the avoiding geometry
(`halfspace_constraints`/`obstacle_constraints`/`bounds`/`ax_limits` keyed by `'avoiding-d3il'`, dims keyed
by `'avoiding'`) and pairs it with the visual model from the Parser. Same constraints, visual prior. So
"just use the projection yaml" is functionally correct — the config content is right.

**The fix (rewrite + reactivate, not delete):**
1. **Rewrite `config/visual_avoiding_eval.yaml`** to be a faithful **avoiding** eval config — i.e. mirror
   `config/projection_eval.yaml` (all 16 keys the active eval reads: `exps`, `seeds`,
   `avoiding_halfspace_variants`, `n_trials`, `dt{}`, `observation_indices`, `action_indices`,
   `projection_variants`, `constraint_types`, `enlarge_constraints`, `halfspace_constraints`,
   `obstacle_constraints`, `bounds`, `plot_how_many`, `ax_limits`, `write_to_file`, +
   `diffusion_timestep_threshold`). Keep `exps: ['avoiding-d3il']` and the `'avoiding-d3il'` constraint keys
   (the geometry is shared; visual-ness is supplied by the model). Give it its **own** `seeds: [6,7,8,9,10]`.
2. **Reactivate it** — repoint the **active** visual-avoiding evals from `projection_eval.yaml` to
   `visual_avoiding_eval.yaml`:
   - `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:101`
   - `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` (same line)
   - and their `load_results_*` siblings if they read it.
3. **Add the startup print** `[ eval ] Seed list for this run: …` to both (Fix 2).

**Why decouple instead of staying on projection_eval.yaml:** `projection_eval.yaml` is shared by **41**
active scripts — its `seeds`/`n_trials` are global. Giving visual avoiding its **own** correctly-named yaml
lets you set its seeds/trials independently without touching the other 40 evals, and the file finally means
what its name says. (Legacy `(legacy_based_on_visual_aligning)` scripts still expect the old schema; they
are deprecated and out of scope — they stay pointed at nothing useful.)

> Status: applied in this pass — see the rewritten yaml + repointed scripts. Verified yaml parses and scripts
> compile.

---

## ⚠⚠ UPDATE 2 (2026-06-22) — WRONG YAML. This is almost certainly the cause. READ FIRST.

You asked: "we're not using `projection_eval.yaml`, we're using the visual-avoiding yaml?" — **good catch.
There are THREE eval yamls, and the active visual-avoiding eval does NOT read the one named for it:**

| yaml | `seeds:` | read by |
|---|---|---|
| `config/projection_eval.yaml` | **[6,7,8,9,10]** | the **ACTIVE** `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` — i.e. what `eval_visual_avoiding_dpcc.sh:79` runs |
| `config/visual_avoiding_eval.yaml` | **[6]** | **only** the `diffuser_visual_avoiding_test (legacy_based_on_visual_aligning)/` scripts |

- The active eval hard-codes `open('config/projection_eval.yaml')` (`:101`). It **never reads**
  `visual_avoiding_eval.yaml`.
- `visual_avoiding_eval.yaml` (seeds `[6]`) — the file you *thought* drives the run — is an **orphan** for
  the active pipeline (read only by the legacy folder). **Editing it has zero effect on the active eval.**

**This explains "only seed 6, cleanly, no error" with NO crash** — your run consumed a `seeds: [6]` source.
Two routes give that:
1. your sbatch/command invoked the **legacy** script → it reads `visual_avoiding_eval.yaml` = **[6]**; or
2. the active script read `projection_eval.yaml`, but on **remote** that file was still `[6]` (your edit went
   to the wrong file, or didn't sync) — the local copy showing `[6,7,8,9,10]` may post-date the run.

**Confirm in 20 s:**
```bash
# (a) which script does your eval job actually run? (legacy folder has a space + parens in its name)
grep -n "\.py" <the-eval-sbatch-you-submitted>.sh
# (b) which yaml does THAT script open, and what seeds does it hold ON REMOTE:
grep -n "yaml.safe_load\|open(.*\.yaml" <that-script>.py
grep -n "^seeds:" config/projection_eval.yaml config/visual_avoiding_eval.yaml
# (c) was the seed edit committed/synced?
git status config/projection_eval.yaml config/visual_avoiding_eval.yaml
```

**Fix (on top of Fix 2's startup print):** decide ONE yaml as the visual-avoiding source of truth. The
active script uses `projection_eval.yaml` → **set seeds there** (it already has `[6,7,8,9,10]`), and either
delete/retire `config/visual_avoiding_eval.yaml` or point the active script at it — don't leave two
similarly-named files where the obvious one is the wrong one. Then Fix 2's `Seed list for this run:` line
proves which seeds ran.

---

## ⚠ UPDATE (2026-06-22) — confirmation results overturn the crash theory. READ THIS FIRST.

Two confirmation steps were run:
1. **Log shows only seed 6, no others, NO traceback/error.**
2. **All of 6,7,8,9,10 have the identical, complete pkl pattern — every artifact is present and ready.**

These **rule out** the "no-fault-isolation crash" (the original root cause below). Reasoning:
- Step 2 ⇒ the per-seed load **cannot** fail on a missing artifact (the Bug-#1 trigger is gone).
- Step 1 ⇒ if the loop had even *reached* seed 7, the log would contain a `--- … seed=7 ---` line (printed
  at `:318`, before any heavy work, via the `Tee` to stdout). **No seed-7 line + no error** means the loop
  **never iterated past seed 6.**

**Corrected diagnosis: the `seeds` list consumed at runtime was `[6]`, not a crash.** The yaml is
`[6,7,8,9,10]`, so exactly two things yield runtime `seeds == [6]` — and both are "the list was [6]," not a
fault:

| # | Cause | Tell-tale |
|---|---|---|
| **1** | `--seed 6` reached the script (the `.sh` positional `$1` fired, e.g. `submit.sh …eval….sh 6`). | log line `[ eval ] Overriding seeds to: [6]` at `:108` is **present** |
| **2** | The **remote** `config/projection_eval.yaml` was NOT `[6,7,8,9,10]` (local edit not git-synced). | that line is **absent** — the yaml itself produced `[6]` |

**Distinguish in 15 s:**
```bash
grep -n "Overriding seeds to" <eval-slurm-log>      # present → cause 1 ; absent → cause 2
# on the REMOTE box:
grep -n "^seeds" config/projection_eval.yaml
git log -1 --oneline -- config/projection_eval.yaml ; git status config/projection_eval.yaml
```

The "no fault isolation" finding below is **still real and worth hardening** (defense-in-depth), but it is
**NOT** what bit you this time. The primary fix is now **Fix 1 + Fix 2 + Fix 3** in the
[How to fix](#how-to-fix-prioritized) section (yaml-only `.sh`, print the resolved seed list, sync the
remote yaml). Original crash analysis retained below for the record.

---

## How seeds actually flow (so we agree on the design)

1. `eval_visual_avoiding_dpcc.py:105` → `seeds = config['seeds']` where
   `config = yaml.safe_load('config/projection_eval.yaml')`. **The yaml is the source of truth**, not
   `avoiding-d3il-visual.py` (that file's `'seed': 0` is the *train* placeholder).
2. `:106-108` → `--seed N` (if passed) **overrides** the yaml to a single seed.
3. `.sh:70-74` → passes `--seed $1` **only if** a positional arg `$1` is given. The pipeline
   (`visual_avoiding_pipeline_dpcc.sh:58`) submits eval with **no** arg → no `--seed` → all yaml seeds.
   ✅ So "yaml set ⇒ .sh sends no seed entry" is already true on the pipeline path.
4. `:221` → `for seed in seeds:` — a correct multi-seed loop, **no `break`**.

So the loop is right. The failure is what happens *inside* it.

---

## Root cause — no fault isolation per seed (the real bug)

Inside `for seed in seeds:` the structure is:

```
for seed in seeds:                                    # :221
    args = Parser().parse_args(..., seed=seed)
    if not aggregate_only:
        diff_experiment = load_diffusion_with_override(...)   # :230  ← UNGUARDED load
        ...
        open(.../obs_normalizer.pkl) ; open(.../act_normalizer.pkl)   # :238,:240  ← UNGUARDED
        ...build constraints...
    for variant in projection_variants:               # :293
        ...
        try:                                          # :317
            ... run MPC + projection + plotting + np.savez ...
        finally:                                      # :528   ← NOTE: finally only, NO except
            sys.stdout = original_stdout ; log_file.close()
```

Two unguarded fault paths, **either** of which aborts the *entire* script (not just the current seed):

- **(A) the load (`:230`, `:238-240`) is before any `try`.** `load_diffusion_with_override` reads, per seed,
  **all** of: `dataset_config.pkl`, `model_config.pkl`, `diffusion_config.pkl`, `trainer_config.pkl`
  (`:150-153`), a `state_<epoch>.pt` (`:177`), plus `obs_normalizer.pkl` / `act_normalizer.pkl`
  (`:238,:240`). If a later seed is missing **any one** of these (e.g. a seed trained in a different run
  that didn't save the normalizer pkls, or a different latest-epoch), the load throws → uncaught → crash.
- **(B) the variant `try` has `finally` but no `except` (`:317`/`:528`).** Any runtime error during a seed's
  MPC/projection/plotting (OOM, SLSQP failure, a NaN, a figure error) propagates up → crash.

**Net:** the multi-seed eval is all-or-nothing up to the first failure. Order is `[6,7,8,9,10]`; seed 6
finishes and writes its npz/png; seed 7 hits one of the above → the job dies → **only seed 6 in the logs.**
"All trained" ≠ "all eval-ready" — the load needs 6 artifacts + a checkpoint per seed, and the run needs
every seed to complete error-free.

---

## Secondary issue — train/eval seed-list disagreement (latent footgun)

- Train sbatch: `train_visual_avoiding_dpcc.sh:57` → `--seeds 5 6 7 8 9`
- Eval yaml: `projection_eval.yaml:7` → `seeds: [6,7,8,9,10]`

Not your current trigger (you trained 7-10 separately), but committed configs disagree: seed 5 trains but
is never evaled; seed 10 is evaled but isn't in the train sbatch. With the no-isolation bug, a single
out-of-band seed (e.g. an untrained 10) would silently truncate a run.

---

## Confirm the exact trigger on the cluster (30 s)

```bash
# 1) Did it crash right after seed 6? Look for the traceback:
grep -nE "seed=|eval loading|Traceback|Error|Overriding seed" <eval-slurm-log> | head -40
#    - "Overriding seed to: 6"  → a positional arg WAS passed → single-seed by design (not this bug)
#    - seed 6 lines then a Traceback before "seed=7" → the no-isolation crash (this bug)

# 2) Which seeds are actually eval-ready (all artifacts present)?
BASE=logs/avoiding-d3il-visual/visual_avoiding_dpcc/H8_K20_Ddiffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VTrue_steps200_bs64    # adjust glob to your tree
for s in 6 7 8 9 10; do
  echo "== seed $s =="
  ls $BASE/$s/{dataset,model,diffusion,trainer}_config.pkl \
        $BASE/$s/obs_normalizer.pkl $BASE/$s/act_normalizer.pkl \
        $BASE/$s/state_*.pt 2>&1 | sed 's/^/  /'
done
```
Whichever artifact is missing for seed 7 is the abort cause.

---

## How to fix (prioritized)

> Post-UPDATE ordering. **Fix 1-3 address the actual cause (runtime `seeds == [6]`).** Fix A-D are
> defense-in-depth hardening (still worth doing, but not what bit you this time).

### Fix 1 — make the `.sh` purely yaml-driven (PRIMARY; this is what you asked for)
You said: "if I set 6,7,8,9,10 in the yaml, the `.sh` should NOT have any seed entry." Agreed — make it so.
Remove the positional `--seed` override from `eval_visual_avoiding_dpcc.sh` (lines 70-74) so the eval
**always** reads the yaml and a stray `$1` can never collapse the run to one seed:
```bash
# DELETE these lines (70-74):
#   SEED_ARG=""
#   if [ -n "$1" ]; then SEED_ARG="--seed $1"; echo "[ eval ] Overriding seed to: $1"; fi
# and change the python call to drop $SEED_ARG:
python diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py --record "$RECORD_MODE" --eval-on-train
```
(If per-seed SLURM fan-out is ever wanted, do it as a separate explicit script — don't leave the footgun on
the default path.) Same for `eval_fm_visual_avoiding.sh`.

### Fix 2 — print the resolved seed list at startup (kills this ambiguity forever)
Right now the script prints the seed list **only** when `--seed` overrides (`:108`); a normal yaml run prints
nothing, so you can't see what it's about to do. Add one unconditional line after `seeds` is resolved:
```python
seeds = config['seeds']
if args_cli.seed is not None:
    seeds = [args_cli.seed]
    print(f'[ eval ] Overriding seeds to: {seeds}')
print(f'[ eval ] Seed list for this run: {seeds}')   # ← ADD: always visible in the slurm log
```
Future logs will show `Seed list for this run: [6, 7, 8, 9, 10]` (or `[6]`) up front.

### Fix 3 — verify the remote yaml is actually synced
If the "Overriding seeds to" line is **absent** in the log, the remote `projection_eval.yaml` produced `[6]`
— your local edit didn't reach the cluster. Commit/push (or whatever sync you use) and re-check on remote:
```bash
git status config/projection_eval.yaml      # is the edit committed?
# on remote after sync:
grep -n "^seeds" config/projection_eval.yaml # must show seeds: [6,7,8,9,10]
```

---

### Fix A — fault-isolate each seed (hardening; makes the yaml list robust to a bad seed)
Wrap the per-seed body so one bad seed logs + is skipped instead of killing the run. Minimal change to the
top of the loop:

```python
for seed in seeds:
    try:
        args = Parser().parse_args(experiment='plan_visual_avoiding_dpcc', seed=seed)
        ... (the entire existing per-seed body: load, constraints, variant loop) ...
    except Exception as e:
        import traceback
        print(f'[ eval ] SEED {seed} FAILED — skipping. Reason: {e}')
        traceback.print_exc()
        continue
```
Result: yaml `[6,7,8,9,10]` are all attempted; failures are visible per seed; survivors still produce
output. This is the direct answer to "if trained + in yaml, eval should read as commanded."

### Fix B — give the variant `try` an `except` (defense in depth)
Add `except Exception: traceback.print_exc()` alongside the existing `finally:` at `:528` so a single
variant's error doesn't even abort the other variants of the same seed.

### Fix C — pre-flight artifact check (fast, clear failure)
Before the loop, print a per-seed readiness table (the 6 pkls + a `state_*.pt`) and **skip-with-warning**
any incomplete seed. Turns a mid-run crash into an upfront "seed 7 missing obs_normalizer.pkl" line.

### Fix D — reconcile the seed lists
Make `train_visual_avoiding_dpcc.sh` `--seeds` and `projection_eval.yaml:seeds` agree (or document the yaml
as the single eval source of truth). Set both to `6 7 8 9 10`.

### Fix E — (optional) per-seed SLURM fan-out
The sbatch already supports `sbatch eval_visual_avoiding_dpcc.sh <seed>` for one job per seed (SLURM-level
isolation + parallelism). Use this if you want speed AND isolation; but it reintroduces a `--seed` entry,
which is the opposite of the yaml-driven path you want. Prefer Fix A for the yaml-driven design.

> Same no-isolation pattern exists in `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` (identical
> structure) — apply Fix A/B there too.

---

## Recommendation
**First run the 15 s UPDATE check** (grep `Overriding seeds to` + remote `cat`/`git status` of the yaml) to
confirm cause 1 vs cause 2. Then:
- **Apply Fix 1 + Fix 2** regardless — they make the eval yaml-only (your stated design) and make the seed
  list visible in every log, so this can't recur ambiguously. (Fix 3 if cause 2.)
- **Also apply Fix A + C + D** as cheap hardening so a genuinely bad seed later degrades gracefully instead
  of truncating the run.
All eval-side only — no retrain.

*Investigation + report only — no code changed yet. Say the word and I'll apply Fix 1/2 (+A/C/D) to both
eval scripts and the two `.sh`s.*
