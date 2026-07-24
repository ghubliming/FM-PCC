# Gen13 U9.1 — the log leak: fix_6's CasADi silencing never covered the FM backbone

**Date:** 2026-07-20 · **Trigger:** *"it again outputs thousands of lines of console log for the second eval command"* — reported mid-run, so it could be fixed before the full eval finished.
**Track back:** `grep -rn "U9 fix" HardFlow/hardflow/models_flow/imf/imf_flow_policy.py HardFlow/run/eval_imf.py`

---

## 1. Diagnosis — measured, not guessed

Job 23612's log: **21,003 lines.** Composition:

| source | lines |
|---|---|
| **CasADi timing tables** (`solver : t_proc …` + `nlp_*` rows) | **~18,000 (85%)** |
| `Saved trajectory plot to: …` | 600 |
| `[ eval_imf ] episode …` (wanted) | 300 |
| everything else | ~2,100 |

Attributing the CasADi tables **per cell type**:

| cell | timing tables |
|---|---|
| **fm_guided** | **2,534** |
| fm_unguided | 0 |
| imf_guided | **0** |
| imf_unguided | 0 |

**Every single one came from the FM backbone. iMF: zero.**

## 2. Root cause — an incomplete fix, not a new bug

fix_6 added `"print_time": False` inside **`ImfFlowPolicy.hardflow_formulate`**. But fix_7 introduced `InstrumentedFlowPolicy(FlowPolicy)` to drive the **FM** backbone through the same instrumented eval — and that class inherits the **base** `hardflow_formulate` from the frozen `flow_policy.py`, which sets only `ipopt.print_level`.

So the moment the FM backbone started running through `eval_imf.py` (fix_7 onward), the silencing silently stopped applying to half the runs. fix_6 was verified on iMF only — the exact blind spot fix_6's own "lesson recorded" section warned about (*check what remains in the log, not just that the knob was set*), repeated one level down.

## 3. Fixes

**(a) Shared CasADi silencer — the 85% fix.** `silence_casadi_timing(print_level)` moved onto `WarmstartCaptureMixin`, so **both** policies get it:

| policy | before | after |
|---|---|---|
| `ImfFlowPolicy` | inline `print_time: False` (fix_6) | calls the shared helper |
| `InstrumentedFlowPolicy` (FM) | **nothing — inherited base** | new `hardflow_formulate` override → `super()` then shared helper |

**(b) Per-figure print — 600 lines.** `hardflow/utils/rendering.py::save_figure` prints `Saved trajectory plot to: …` on every figure (2 per episode with fans on). That file is **frozen**, so it is suppressed at the **call site** with a `_quiet_stdout()` context manager in `eval_imf.py`, wrapping both `save_single_trajectory_image` and `_save_foresight_fan`. Verified working.

## 4. Expected effect

| | job 23612 | after U9.1 |
|---|---|---|
| CasADi tables | ~18,000 | **0** |
| figure prints | 600 | **0** |
| total | 21,003 | **≈2,400** (≈9×  smaller) |

For the plain eval command (no fans), the log becomes essentially the episode lines plus banners.

## 5. Files changed (Gen13-owned only)

| File | Change |
|---|---|
| `imf/imf_flow_policy.py` | `silence_casadi_timing()` on the mixin; `InstrumentedFlowPolicy.hardflow_formulate` override; `ImfFlowPolicy` now uses the shared helper instead of its inline duplicate |
| `run/eval_imf.py` | `_quiet_stdout()` + wrapped both figure-save call sites |

`run/eval.py`, `flow_policy.py`, `rendering.py` — untouched. `py_compile` clean; suppression verified by direct test.

## 6. Note on the in-flight run

This does **not** invalidate any numbers — it is purely console output. **The CSVs from the currently running job are fine.** The fix only makes the *next* run readable, so there is no need to cancel or repeat anything already in progress.

## 7. Lesson (second occurrence — worth acting on)

fix_6 recorded: *"silencing a library's verbosity means checking what remains in the log afterwards."* U9.1 is the same failure one level deeper: **a fix applied to one subclass while a sibling subclass bypasses it.**

**Rule going forward:** when a fix lands in a subclass override, check every sibling that reaches the same base method — or put the fix on a shared mixin/base-adjacent helper from the start, as done here.
