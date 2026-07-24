# MEMO — hooking `npz_traj_export.py` onto an eval so it runs automatically after eval finishes

**Goal:** after an eval job finishes writing its `.npz` results, automatically run the exporter so a
`viewer_<scene>.html` is ready without a manual step. Worked example: **imf avoiding**
(`FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` + `Slurm_Codes/sbatch/iMF/eval_imf.sh`).
The pattern generalizes to any eval — see §4.

## 0. What a "scene" is, concretely (verified in the imf eval source)

```python
# FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py
save_path = f'{args.savepath}/results/halfspace_{halfspace_variant}'   # set ONCE per (seed, halfspace_variant)
...
for variant in projection_variants:        # diffuser, dpcc-c, dpcc-r, ..., model_free, ...
    ...
    np.savez(f'{save_path}/{variant}.npz', ...)
```

So **one `save_path` directory = one scene** (every projection variant for one seed/halfspace
combo) — exactly the unit `npz_traj_export.py` expects as its `<scene_dir>` argument. `args.savepath`
is computed inside the Python `Parser`, seeded per training config — **do not try to reconstruct it in
bash**; find it with `find` after the job finishes instead (§2).

## 1. Two ways to hook it in — pick based on risk tolerance

| approach | where | risk | effort |
|---|---|---|---|
| **A — sbatch post-step (recommended)** | append a step at the end of the `.sh`, after the `python eval_*.py` line | low — zero eval.py changes | one `find` + one loop, ~6 lines |
| **B — in-script hook** | call the exporter via `subprocess` inside the eval `.py`, right after each `save_path`'s variant loop finishes | higher — touches actively-developed eval code that's sibling-synced across generations (Gen7/Gen6V4/…, per CLAUDE.md) | needs mirroring across siblings if changed |

**Default recommendation: A.** It needs no changes to fragile, frequently-edited eval scripts, and a
bug in the exporter can never break or block the (expensive, GPU-time) eval run.

## 2. Approach A — concrete sbatch addition (imf/avoiding example)

Add this to `Slurm_Codes/sbatch/iMF/eval_imf.sh`, right after the existing eval line:

```bash
# 4) Run iMF Evaluation
cd "$REPO"
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py
echo "Evaluation completed successfully."

# ------------------------------------------------------------------------------
# 5) TRAJ-VIZ EXPORT (post-step, non-fatal) — one scene per (seed, halfspace_variant) dir
# ------------------------------------------------------------------------------
echo "[traj-viz] scanning for result scenes to export..."
find logs/avoiding-d3il/plans/flow_matching_v3_imeanflow -type d -path "*/results/halfspace_*" \
  | while read -r SCENE_DIR; do
      echo "[traj-viz] exporting: $SCENE_DIR"
      python npz_analysis/npz_traj_visualizer/npz_traj_export.py "$SCENE_DIR" \
        || echo "[traj-viz] WARNING: export failed for $SCENE_DIR (continuing, eval result unaffected)"
    done
echo "[traj-viz] done."
```

Notes:
- **`find ... -path "*/results/halfspace_*"`** locates every scene dir written by *any* seed this run
  touched, without needing to know `args.savepath`'s internal templating.
- **`|| echo ...` makes the export non-fatal** — critical, since the script has `set -e` at the top;
  without this, an exporter bug would fail the whole SLURM job retroactively after a successful (and
  possibly hours-long) GPU eval. Never let a visualization nice-to-have risk the real result.
- **No new Python env needed.** The conda env already active (`FMPCC`) has `numpy` (PyTorch requires
  it) and `pyyaml` (the eval script itself does `import yaml` to read `config/projection_eval.yaml`),
  which are the exporter's only two dependencies.
- Adjust the search root (`logs/avoiding-d3il/plans/flow_matching_v3_imeanflow`) to match whichever
  model family the sbatch script actually evaluates — check the `diffusion_loadpath` / experiment name
  used by that eval's `Parser().parse_args(...)` call, or just search from `logs/` if unsure (slower
  but always correct, and this only runs on already-written small `_traj_viz/` outputs).
- Output lands in `<SCENE_DIR>/_traj_viz/` — same convention as manual runs — so nothing about how you
  open the viewer afterward changes (see `USAGE.md`).

## 3. Approach B — in-script hook (if you want it truly inline)

Inside `eval_flow_matching_v3_imeanflow.py`, after the `for halfspace_variant in halfspace_variants:`
body finishes writing every variant's `.npz` for that `save_path` (i.e. right after the
`for variant in projection_variants:` loop ends, ~line 404's block), add:

```python
import subprocess
...
# after the projection_variants loop for this (seed, halfspace_variant) has finished:
try:
    subprocess.run(
        ['python', os.path.join(os.path.dirname(__file__), '..', 'npz_analysis',
                                 'npz_traj_visualizer', 'npz_traj_export.py'), save_path],
        check=False)   # never raise -- a viz-export failure must not fail the eval run
except Exception as exc:
    print(f'[traj-viz] export failed for {save_path}: {exc} (continuing)')
```

If you go this route: **mirror the change across sibling generations** that share this eval pattern
(Gen7/Gen6V4/etc. per the repo's copy-modify convention) — do not patch only one copy. This is the
main reason Approach A is preferred: it lives in the sbatch, outside the copy-modify eval-code tree
entirely, so there's nothing to keep in sync.

## 4. Generalizing to a DIFFERENT eval (e.g. UAV, visual-aligining)

The only per-eval-specific thing is **where scene dirs land**, because save-path layout differs:
- **avoiding** (this memo's example): `<savepath>/results/halfspace_<variant>/` — one extra
  scenario-level subfolder per halfspace variant, npz **flat** inside it.
- **UAV** (`FM_v3_uav_test/eval_fm_uav.py` family): scene dir = the eval's `results/` folder itself;
  npz are **nested** one level (`results/<projection_variant>/<projection_variant>.npz`), no extra
  scenario subfolder layer (verified this session on `s_curve_bounds+dynamics+.../` — 17 sibling
  `<variant>/<variant>.npz` folders directly under the scene root).
- **visual-aligining**: not yet inspected for this tool (open item in `PLAN_npz_traj_visualizer.md`
  §13 D2) — check its eval's own `save_path`/`np.savez` call the same way this memo did (§0) before
  wiring a hook.

`npz_traj_export.py`'s own file-discovery (`find_variant_npz`, imported from `compare_horizon_plans.py`)
already handles BOTH the flat and nested layouts transparently — **you only need to get the `find`
command's search pattern right for the new eval's directory shape**; the exporter itself needs no
changes. Adjust the `-path` glob in §2's loop (or drop it entirely and just point at the eval's
`results/` folder directly, if that eval doesn't nest a scenario level).

## 5. Quick sanity check before trusting a new hook

Run the exporter once by hand against one real scene dir from the new eval (as done all session for
imf/avoiding and UAV) and confirm the printed summary + spot-check `div_ref` values look sane before
wiring it into the sbatch — this catches env/layout mismatches (wrong `--env`, wrong halfspace variant
inference, etc.) without burning a GPU job to find out.
