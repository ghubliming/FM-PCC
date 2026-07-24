# Gen13 fix-4 — kill the log noise: training tqdm bloat + IPOPT solver spam

**Date:** 2026-07-19
**Trigger:** first-run inspection (`Gen13/fix_3/INSIGHTS_Gen13_first_run.md` §6) found two log-noise problems, both in my own Gen13 code.
**Rule check:** all edits are to **Gen13-owned files only**; pre-existing HardFlow files (`run/train.py`, `run/eval.py`, `run_scripts/eval_hardflow_new.sh`, FM baselines) remain byte-identical.

---

## Problem A — training log was 98% tqdm garbage

**Evidence (job 23579):** `23_57_33_hf_imf_train_23579.log` = **4.6 MB in 586 lines**; the three longest single lines were **791,717 / 865,345 / 742,027 characters** — 98% of the file.

**Cause:** `run/train_imf.py` wrapped the 100k-step loop in `tqdm.tqdm(...)`. Under `Slurm_Codes/submit.sh` stdout is a redirected **file**, not a tty, so tqdm's in-place carriage-return update never collapses; every refresh is appended as raw text. tqdm refreshes on a **time** basis, so a 4-hour run emits ~10^5 updates — the bloat scales with *wall-clock*, not step count. **fix_2 only fixed `eval_imf.py`; training was missed.** This directly violated the SLURM memory rule written in the same session.

**Fix (`run/train_imf.py`):**
- Module-level `_IS_TTY = sys.stdout.isatty()`.
- Loop iterator: `tqdm.tqdm(range(n))` if tty, else plain `range(n)`.
- Periodic metric line: `tqdm.tqdm.write(msg)` if tty, else `print(msg, flush=True)`.
- Off-tty, prints one explanatory line up front: `non-tty (batch) mode: progress bar disabled, logging every 200 steps of 100000`.

Output is now **bounded**: `n_train_steps / log_freq` = 500 metric lines, independent of runtime. No information is lost — the `[ train_imf ]` lines already carried every real signal (`raw_mse_u/v`, `a0_mse`, adaptive loss); the bar only ever showed ETA.

**Measured (real run, redirected stdout, 3 s loop):** old 2,275 bytes / 2,274-char longest line → new **58 bytes / 57-char longest line** = **39× reduction in 3 seconds**, and the gap widens linearly with runtime (hence 4.6 MB over 4 h).

## Problem B — eval log: 70k lines, ~44% IPOPT internals

**Evidence (job 23580):** eval log was clean *per line* (fix_2 worked — max line 213 chars ✅) but **70,353 lines**, of which ~30,800 were IPOPT solver output from `solver_print_level=5` (~45 lines **per NLP solve**).

**Signal audit — what was actually in those 45 lines?** Inspected a full block:
- 17-row iteration table, function/gradient/Jacobian eval counts, `Total seconds in IPOPT` → **pure solver internals, no scientific value** once converged.
- `Constraint violation: 2.22e-16`, `Overall NLP error: ~1e-9` → **this is the only meaningful content**: proof the projection actually enforced constraints. Across the whole run these were uniformly ~1e-16/1e-9, i.e. always healthy and therefore uninformative *in aggregate* — but a solver **failure** would matter enormously (it is the prime suspect for any downstream constraint violation, e.g. E3's 3 violations).

**Fix — silence IPOPT, but preserve the failure signal in our own compact form:**
1. `run_scripts/eval_hardflow_new_imf.sh`: `solver_print_level=5` → `"${SOLVER_PRINT_LEVEL:-0}"` (env-overridable; set `SOLVER_PRINT_LEVEL=5` to restore full solver output when debugging the solver itself).
2. `hardflow/models_flow/imf/imf_flow_policy.py`: added cumulative `_nlp_solves` / `_nlp_failures` counters with `reset_nlp_stats()` / `nlp_stats()`. The existing bare `print("Solver failed...")` became a loud, greppable
   `[ eval_imf ] WARNING: NLP solve failed (step k=…) — using last available value.`
3. `run/eval_imf.py`: counters reset per episode; each episode line now ends with `nlp=<solves>` (and ` FAILED=<n>` only when non-zero); `nlp_solves` + `nlp_failures` added as **CSV columns**; the final summary prints `all NLP solves OK` or `NLP failures: N (!!)`.

**Net:** ~31k lines removed, **zero loss of actionable information** — solver health is now one field per episode plus two CSV columns, and a failure is *more* visible than before (it was previously buried in a 70k-line log).

**Deliberately NOT changed:** the FM baseline scripts (`eval_hardflow_new.sh` etc.) keep `solver_print_level=5`. They are pre-existing files under the no-edit rule, and their logs are already archived as the frozen baseline.

---

## Files touched (all Gen13-owned)

| File | Change |
|---|---|
| `HardFlow/run/train_imf.py` | tty-gated tqdm (Problem A) |
| `HardFlow/run/eval_imf.py` | per-episode NLP reset + `nlp=` in episode line + 2 CSV columns + failure-aware summary |
| `HardFlow/hardflow/models_flow/imf/imf_flow_policy.py` | NLP solve/failure counters, `reset_nlp_stats()`/`nlp_stats()`, loud failure warning |
| `HardFlow/run_scripts/eval_hardflow_new_imf.sh` | `solver_print_level` 5 → 0 (env-overridable) |

## Verification (container, minimal CPU venv)

- `py_compile` clean on all three Python files; `bash -n` clean on the run script.
- iMF package still imports; `nlp_stats()` / `reset_nlp_stats()` behave correctly (counter → 2 → reset → 0).
- tty-gating measured end-to-end with real redirected stdout: **39× smaller in a 3-second run**, output size now independent of runtime.
- Confirmed FM scripts untouched (`eval_hardflow_new.sh` still `solver_print_level=5`).

## Expected effect on the next run

| Log | Before | After (expected) |
|---|---|---|
| train | 4.6 MB, 586 lines, 865k-char lines | ~50 KB, ~510 lines, all <120 chars |
| eval | 70,353 lines | ~39k lines removed → ~4k lines; plus per-episode `nlp=` health field |

## Note

This does **not** change any numerical result — purely observability. The E1–E4 metrics from the first run remain valid; the next run's CSVs simply gain `nlp_solves`/`nlp_failures`. Recommended to fold this into the **K=4/5 experiment** (fix_3 §7 item 1) rather than re-running anything just for logging.
