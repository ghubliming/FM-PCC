# Gen13 U13 — CHANGELOG: MPC foresight-fan / smoothness diagnostic for Mix-ML (MF, AF)

**Ask:** "diag mpc fan to see the raw and projected traj if smooth on K1,2 mf and af."

Gen13's fix_7 already built this exact diagnostic (`Research/DISCUSSION_foresight_fan_and_
smoothness_paradigms.md`, `RESULTS_Gen13_fix7_smoothness_2x2.md`) — but only for backbone
`imf`/`fm`, at K5/K10, with pre-U12 flat naming. U13 extends it to the Mix-ML objectives
(MF/AF) at K1/K2, additively, reusing fix_7's instrumentation **unchanged**.

## Why no new instrumentation was needed

`run/eval_imf.py`'s roughness/fan capture is already objective-agnostic:
- `--backbone imf` loads `TemporalImfUnet` — MF and AF checkpoints ARE `TemporalImfUnet` under the
  hood (U11's design: same net, different training-time u-target), so pointing `--flow_exp_name` at
  an MF/AF checkpoint with `--backbone imf` just works, no code change.
- A single `guidance_method=hardflow_new_imf` (hfproj) run already reports **both** numbers per
  replanned horizon, straight from `eval_imf.py`:
  - `plan_roughness` — the PROJECTED (post-NLP) plan's roughness
  - `plan_roughness_raw` — the RAW warmstart plan's roughness, captured via `policy.raw_plan()`
    **before** the NLP runs (HardFlow's equivalent of DPCC's un-projected `diffuser` output)
  - `--imf_plot_fan` additionally dumps `*_fan.png` / `*_fan.npz` (the visual raw-vs-projected view)

So "raw and projected traj smooth for MF/AF at K1/K2" is answered by **one hfproj run per
(family, K)** — no new math, no new capture logic, just clean wiring + naming.

## New files (both additive)

### `HardFlow/run_scripts/eval_smoothness_diag_ml.sh`
Sibling of fix_7's `eval_smoothness_diag.sh`, with `ML_TYPE` (imf|mf|af) replacing `BACKBONE`
(imf|fm) — `--backbone` is always `imf` internally now, since the family axis is the *objective*,
not the net class. Knobs: `ML_TYPE`, `GUIDANCE` (`hfproj` default — gives the raw+projected pair;
`raw` also supported but has no pre-NLP plan to compare, so `plan_roughness_raw` stays NaN there),
`ML_K` (default 2), `N` (episodes, default 5), `ML_EXP_NAME`/`ML_CP` overrides. `flow_exp_name`
resolution matches U12.2's family-first convention: `${ml_type}/H16_ml_${ml_type}_100k`.

Output nests **inside** the U12.2 family-first tree, under a `smooth_` sub-arm so it never collides
with the decisive n=200 `raw_K*/hfproj_K*` result dirs:
```
logs/avoiding-v0/eval/<ml_type>/<run>/smooth_<guidance>_K<k>_n<n>/
  trajectories.csv (plan_roughness, plan_roughness_raw)  + *_fan.png/.npz per captured replan
```

### `Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_ml_hardflow.sh`
Loops a `CELLS` matrix, default **exactly the ask**: `mf:hfproj:1 mf:hfproj:2 af:hfproj:1 af:hfproj:2`
(format `<ml_type>:<guidance>:<K>` — extend with `imf:hfproj:2` for the frozen-iMF reference cell,
or a `:raw:` cell for the no-projection field). Prints a summary table: projected roughness, raw
(pre-NLP) roughness, the raw/projected ratio, and safety% — per cell, reading straight from the
family-first tree. Ends with a `find … -name '*_fan.png'` pointer for the visual fans.

## Backward compatibility / non-damage
- fix_7's own scripts (`eval_smoothness_diag.sh`, `eval_smoothness_diag_hardflow.sh`) and
  `run/eval_imf.py` are **untouched** (`git status --porcelain`: empty) — the existing iMF/FM 2x2
  smoothness study still runs exactly as before.
- No new columns, no new CLI flags added to `eval_imf.py` — U13 only adds two orchestration scripts
  that call the existing, frozen entry point with different arguments.

## Validation (local, syntax-only — no cluster access here)
- `bash -n` passes on both new scripts.
- Simulated the default 4-cell matrix's path derivation — confirmed it lands at:
  ```
  logs/avoiding-v0/eval/mf/H16_ml_mf_100k/smooth_hfproj_K1_n5/trajectories.csv
  logs/avoiding-v0/eval/mf/H16_ml_mf_100k/smooth_hfproj_K2_n5/trajectories.csv
  logs/avoiding-v0/eval/af/H16_ml_af_100k/smooth_hfproj_K1_n5/trajectories.csv
  logs/avoiding-v0/eval/af/H16_ml_af_100k/smooth_hfproj_K2_n5/trajectories.csv
  ```
- Ran the summary-parsing Python block against a synthetic `trajectories.csv` (3 rows, known
  `plan_roughness`/`plan_roughness_raw`/`safety` values) — output matched hand-computed mean/ratio.
- Confirmed the real CSV column names (`safety`, `plan_roughness`, `plan_roughness_raw`) against
  `eval_imf.py`'s `fieldnames`/`writer.writerow` — match.

## Fix (same day, first cluster run) — legacy-flat checkpoint auto-detect

**Symptom:** first submit failed all 4 cells with `FileNotFoundError:
logs/avoiding-v0/flow/mf/H16_ml_mf_100k/model_ema_4.pth` (and the `af/` equivalent).

**Root cause:** `H16_ml_mf_100k`/`H16_ml_af_100k` were trained **before** U12.2 and were never
physically `mv`'d into the new nested layout (U12.2's changelog flagged this as expected — the
migration `mv` commands were offered, not run). U13's default `flow_exp_name` assumed the new
nested path (`mf/H16_ml_mf_100k`) unconditionally, with no check that anything existed there.

**Fix (`eval_smoothness_diag_ml.sh` only — still no frozen-file changes):** the checkpoint LOAD
path now auto-detects: nested (`<ml_type>/H16_ml_<type>_100k`) checked first, legacy flat
(`H16_ml_<type>_100k`) as fallback, with a clear `NOTE:`/`WARNING:` log line saying which one it
picked (or that neither exists). An explicit `ML_EXP_NAME`/`IMF_EXP_NAME` override still always
wins, used verbatim, same as before. The output `exp_name` is **decoupled from the load path** —
it always uses the tidy nested form, so a legacy-flat checkpoint still produces a well-organized
`eval/<ml_type>/H16_ml_<type>_100k/smooth_.../` tree instead of writing back to a flat dir.

Verified both branches by simulation (not just read): legacy-flat-only → loads legacy, writes
nested; nested-only (post any future migration) → loads nested. The sbatch loop's `CELL_DIRS`
(used for the summary table) already computed the nested form independently, so it needed no
change — it now matches the leaf script's output path by construction.

No cluster-side `mv` was required to fix this — this is a pure wrapper-script robustness fix, not
a migration. (The U12.2 migration commands remain available if you ever want the checkpoints
physically reorganized, but are no longer necessary for this diagnostic to work.)

**Follow-up — no silent fallback.** The first version of the fallback logged a single quiet
`NOTE:` line to stdout only, which is easy to lose in a Slurm log buried under dataset/normalizer
output. Changed to a loud `_warn_fallback()` banner (`###...###` framed, `[ smooth_diag_ml ]
WARNING: ...`) piped through `tee /dev/stderr`, so it prints on **both stdout and stderr** every
time the resolved checkpoint path is not "nested + explicit request" — i.e. both the legacy-flat
fallback case and the neither-found case now bannerize identically loudly. An explicit
`ML_EXP_NAME`/`IMF_EXP_NAME` override is the caller's stated intent, not a fallback, so it still
prints quietly (just the normal `run=...` info line). Verified by simulated capture that the
banner appears in both streams.

## How to run
```bash
# the default matrix = exactly the ask (mf & af, K1 & K2, hfproj = raw+projected pair)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_ml_hardflow.sh

# add the frozen-iMF reference cell alongside:
CELLS="mf:hfproj:1 mf:hfproj:2 af:hfproj:1 af:hfproj:2 imf:hfproj:2" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_ml_hardflow.sh

# bigger n for tighter roughness estimates, or a single cell standalone:
N=10 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_ml_hardflow.sh
ML_TYPE=af GUIDANCE=hfproj ML_K=1 N=5 bash run_scripts/eval_smoothness_diag_ml.sh   # single cell
```
Fan images for visual inspection: `find logs/avoiding-v0/eval -name '*_fan.png' -path '*smooth_*'`.
