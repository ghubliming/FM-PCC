# CHANGELOG — U10.1: run provenance for ENV/CLI-overridden runs (all generations)

**Date:** 2026-08-16 · **Status:** code complete, **nothing run, nothing committed**
**Trigger:** *"For the Overloading CLI will the config snapshot change, if not how do I know what is
the config at then?"* — the answer was **no**, and the gap turned out to predate U10 in most
generations.
**Related:** `../Gen3v6_MeanFlow/H8+8_U10/CHANGELOG_Gen3v6_U10_H16_replan8.md` (the change that
prompted this) · `../Gen3v6_MeanFlow/H8+8_U10/GUIDE_H16_replan8_MF_UNet.md`

---

## 1. The problem, stated precisely

Several generations resolve experiment knobs from the **environment** rather than from literals in
`config/*.py`. That was a deliberate, good decision — it makes a run configurable at submit time on
the cluster and killed a class of half-applied-edit bugs. But it moved information out of every
durable artifact:

| artifact | who writes it | records the resolved value? |
|---|---|---|
| **results-path tokens** (`H16_`, `bbunet`, `K1`, `T0.5`, `A0.5`, `B1`, `_msg…`) | `args_to_watch` | ✅ **the primary record, unchanged by this work** |
| `args.json` | `Parser.save` | ✅ values — but **train only**: `save = (experiment == 'train')` (`diffuser/utils/setup.py:85`) |
| `config_snapshot_*/<config>.py` | `Parser.snapshot_configs` | ❌ **verbatim source copy**. Since U10 it reads `'horizon': _mf_horizon` — identical bytes for an H8 and an H16 run |
| `config_snapshot_*/<yaml>` | `Parser.snapshot_configs` | ✅ in the Gen3v6/Gen3v7 copies (FIX_9 made it follow `FMPCC_PROJ_CFG`); ❌ still hardcoded in shared `diffuser/utils/setup.py:212` |
| sbatch log echo | the job scripts | ✅ — but lives in `Slurm_Codes/logs/<date>/`, detached from the results |

Three consequences:

1. **Eval runs recorded their resolved knobs nowhere** except the folder name — in every generation,
   not just Gen3v6.
2. **Knobs consumed by the eval *script*** (not the config) reach `args` at all, so they were in no
   artifact whatsoever: `MF_REPLAN_STEPS`, `HFFM_VARIANTS` (Gen7 — *adds whole arms*),
   `UAV_MIX_HF_OFF` (Gen15 — *deletes the HardFlow arm*).
3. **"Explicit" and "fallback" are indistinguishable.** A path saying `A0.5` cannot tell you whether
   the submitter asked for 0.5 or the yaml default supplied it — the ambiguity that made the Gen12
   threshold sweep hard to read.

Two generations build their output path **by hand** and never call `Parser.mkdir` at all
(Gen12 via `hf_paths`, Gen15 from the train savepath), so they got neither `args.json` nor a config
snapshot next to their results.

---

## 2. Options considered

| option | verdict |
|---|---|
| **Rewrite the resolved values back into `config/*.py`** | ❌ rejected, as the user suspected. A job mutating a git-tracked source file races concurrent jobs, leaves the cluster tree dirty, and makes the snapshot disagree with HEAD. The config being read-only at runtime is what makes the current design safe. |
| **Revert to literals so the snapshot self-describes** | ❌ reintroduces the four-independent-literals failure mode U10 removed, and does nothing for the eval side. |
| **Flip `save = (experiment == 'train')` so eval writes `args.json`** | ❌ not on its own: it changes behaviour for *every* generation at once, and still misses the script-level knobs (`MF_REPLAN_STEPS`, `HFFM_VARIANTS`, `UAV_MIX_HF_OFF`) that never enter `args`. |
| **A separate provenance file written next to the results** | ✅ **chosen.** Additive, opt-in, generation-agnostic, and able to record what `args` cannot. |

---

## 3. What was added

### 3.1 `diffuser/utils/provenance.py` (new, ~170 lines, stdlib only)

Shared so every generation can opt in with one import — and **deliberately not exported** from
`diffuser/utils/__init__.py`, nor called from `Parser.mkdir`. Adding the file changed the behaviour
of exactly zero existing runs.

`provenance.write(savepath, role, resolved, yaml_path=None, ...)` emits `run_provenance.json`:

```jsonc
{
  "schema": "fmpcc.run_provenance/1",
  "config": {                       // ← the de-duplication key
    "role": "eval",
    "resolved":   { "horizon": 16, "replan_steps": 8, "hf_act_threshold": 0.5, ... },
    "env_set":    { "MF_HORIZON": "16", "MF_BACKBONE": "unet" },   // what the submitter asked for
    "env_absent": ["HFFM_ACT_THRESHOLD", "DPCC_THRESHOLD", ...],   // what came from a fallback
    "yaml":       { "path": "config/meanflow_projection_eval.yaml", "digest": "sha256:ce28b3…" },
    "git":        { "commit": "b5ecb6ad…", "dirty": true }
  },
  "runtime": { "written_at": "…", "argv": [...], "slurm": { "SLURM_JOB_ID": "…" }, "python": "3.11" }
}
```

Four design rules, each answering a specific failure mode:

1. **Never break a run.** Every path is wrapped; a metadata bug must not cost a 6-hour eval. Failure
   prints a warning to stderr and returns `None`.
2. **`resolved` vs `env_set` vs `env_absent`.** The three together answer "what ran" *and* "who
   chose it" — the distinction the results path structurally cannot express.
3. **De-duplicate by content.** Gen3v6's eval parses args once per halfspace variant against the
   same savepath; three identical payloads become one file. A *different* configuration lands as
   `run_provenance_2.json` rather than overwriting (mirrors `args_resume_N.json`).
4. **Write beside the results**, never into `config_snapshot_*/` — `Data_Analysis/DA_UAV_v1/
   discovery.py:360` enumerates that directory by name.

`git.dirty` is worth calling out: a dirty tree means the recorded commit does **not** fully describe
the code that produced the numbers. That is exactly the state a cluster checkout is often in.

### 3.2 Call sites (5 generations, 6 scripts)

| generation | script | env knobs it resolves | file lands in |
|---|---|---|---|
| Gen3v6 MeanFlow | `FM_v3_meanflow_test/train_…meanflow.py` | `MF_HORIZON`, `MF_BACKBONE`, `TRAIN_SEEDS` | `args.savepath` |
| Gen3v6 MeanFlow | `FM_v3_meanflow_test/eval_…meanflow.py` | + `MF_REPLAN_STEPS`, `HFFM_*`, `DPCC_THRESHOLD`, `FMPCC_PROJ_CFG` | `args.savepath` |
| Gen3v7 α-Flow | `FM_v3_alphaflow_test/eval_…alphaflow.py` | `HFFM_ACT_THRESHOLD`, `HFFM_BATCH`, `HFFM_FLOW_STEPS`, `DPCC_THRESHOLD` | `args.savepath` |
| Gen12 HardFlow | `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | `HFFM_*`, `DPCC_THRESHOLD`, `FORCE_OVERWRITE` | `args.savepath` (hand-built by `hf_paths`) |
| Gen7 mix aligning | `mix_visual_aligning_test/eval_mix_visual_aligning.py` | `HFFM_VARIANTS`, `MIX_FILM_MODE[_<ENGINE>]`, `FMPCC_BOX_*` | `<savepath>/results[_train_set]` |
| Gen15 mix UAV | `mix_uav_test/eval_mix_uav.py` | `UAV_MIX_FLOW_STEPS`, `UAV_MIX_HF_OFF` | `seed_dir` (one record per eval-params folder) |

Each call site is a single `provenance.write(...)` placed at the first point where every knob is
final — after the checkpoint loads and K is matched onto the sampler, not at parse time.

`TRACKED_ENV` in the module is a **superset** across generations; a var meaningless to one simply
never appears in its `env_set`.

---

## 4. Explicitly NOT changed

- **The results-path tokens.** They remain the primary record. This is a second, machine-readable
  copy, not a replacement.
- **`Parser.save` / `Parser.mkdir` / `snapshot_configs`** — untouched in every copy. No existing
  artifact changed name, location or content.
- **The hardcoded yaml in shared `diffuser/utils/setup.py:212`.** The Gen3v6/Gen3v7 copies already
  fixed this (FIX_9); the shared one still snapshots `config/projection_eval.yaml` regardless of
  what was loaded. Fixing it would touch every generation that uses the shared Parser — a separate,
  deliberate decision. Provenance now records the real yaml path **and its sha256**, so the
  consequence is contained.
- **Aggregators** (`load_results_*.py`): read-only, produce no run.
- **`UAV_FM_DATA_ROOT`** (`*/datasets/d4rl.py`): a data *location*, not an experiment knob.
- **`config/avoiding-d3il-visual.py`**: its only env read is `FMPCC_RUN_MSG`, which is
  self-describing — its whole purpose is to appear in the path.
- **Gen3v7 α-Flow train / other trains**: no env-driven architecture knobs, so nothing to record
  that `args.json` does not already hold.

---

## 5. Verification

**Ran locally** (the module is stdlib-only, so it is actually executable in this container — unlike
anything touching torch):

| check | result |
|---|---|
| first write creates `run_provenance.json` | ✅ |
| second write, identical config | ✅ `unchanged, kept …` — no duplicate |
| third write, `MF_REPLAN_STEPS=8` | ✅ new `run_provenance_2.json`, original preserved |
| `git.commit` / `git.dirty` | ✅ real commit, `dirty: true` (correct — this tree is dirty) |
| yaml `sha256` digest | ✅ |
| `env_set` captured `MF_HORIZON`/`MF_BACKBONE`, `env_absent` listed the rest | ✅ |
| `py_compile` on all 7 touched files | ✅ |
| **AST scope check** of every name referenced in all 5 call sites | ✅ all resolve (`scene`/`config`/`parsed` in Gen15 are `_run_variant` parameters) |

**Cannot be checked here** (no torch/PyYAML): that the call sites execute. The scope check removes
the `NameError` class of failure, but the first real run is still the proof. All five sites are
non-fatal by construction, so a mistake degrades to a stderr warning rather than a lost job.

**On the cluster, first run of each generation:** confirm `[ provenance ] wrote …` appears and the
JSON sits beside the results.

---

## 6. What this buys, concretely

Before: a results folder named `…/H16_K1_Meuler_T0.5_A0.5_B1_D…MeanFlowODE_msgr8/` told you the
operating point but not (a) whether `A0.5` was chosen or inherited, (b) which yaml gated it,
(c) which commit produced it, (d) whether the tree was dirty, (e) for Gen7/Gen15, whether an arm had
been added or deleted by an env var.

After: all five are one `cat run_provenance.json` away, and a DA can diff two runs' `config` blocks
to see exactly what differed.
