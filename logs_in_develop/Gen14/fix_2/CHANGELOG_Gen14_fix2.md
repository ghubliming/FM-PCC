# CHANGELOG — Gen14 fix_2: G6 was a no-op gate; replaced with a runtime test

**Date:** 2026-08-01 · **Follows:** [`../init/CHANGELOG_Gen14_coding1.md`](../init/CHANGELOG_Gen14_coding1.md)
**Trigger:** cluster run **24082** (`18_22_40_gates_mix_visual_24082.log`, node i6-gpu-1, git `b5846ee`)
**Scope:** 2 files, both Gen14-owned. **No engine/model/config code touched.**

---

## 1. What run 24082 showed

All 7 gates reported PASS. The load-bearing result is real:

| Gate | Result | Note |
|---|---|---|
| G0 copy fidelity | PASS | 23/23 verbatim files match sources |
| G1 reference-arm wiring | PASS | all four arms resolve correctly |
| **G2 JVP survives vision** | **PASS** | `loss=1.999845 finite=True`, no `NotImplementedError` — the single most uncertain thing in the build |
| G3 MeanFlow identity h=0 | PASS | `h_mean=0.000e+00`, `fm_frac=1.000` |
| G4 α spans the budget | PASS | `1.0, 1.0, 0.5, 0.0, 0.0` at 0/25k/50k/75k/100k |
| G5 α→0 limit | PASS | `alpha=0.000000`, `discrete_frac=0.000000` |
| **G6 projector at K=1** | **PASS — but meaningless** | see below |

**G2 is now stronger than the gate claims.** Its printed note asks for a manual peak-memory
comparison against the fm arm to prove the pre-encode is live. That comparison is
unnecessary: `VisualMeanFlow.loss()` builds `cond = {0: obs_0, 'visual_latent': ...}` with
**no image tensor in `cond` at all**, so the ResNets *cannot* be inside the JVP. The only
alternative failure mode would be an image-blind model, which a finite loss plus the
`[ VisualUNetTwoTime ] MultiImageObsEncoder initialized` line rules out.

---

## 2. The defect: G6 could never fail

G6 printed `DIFF fm: terminal-step fallback present` — the opposite of the prediction in the
init changelog §7.1. Investigating showed the **gate** was wrong, not the prediction.

The check was a whole-file substring search:

```python
has_fallback = 'flow_steps - 1' in src or 'flow_steps_v3 - 1' in src
```

That string appears in the unrelated `repeat_last` clamp, which is present in **all three**
engines:

```
fm_diffusion.py:174   loop_idx = min(i, self.flow_steps_v3 - 1)   # repeat_last clamp  <- matched
mf_diffusion.py:236   loop_idx = min(i, flow_steps - 1)           # repeat_last clamp  <- matched
mf_diffusion.py:285   near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps - 1)   <- the real fallback
```

So the predicate was true for every arm regardless of the thing being tested: **mf/af passed
for the wrong reason and fm was mislabelled.** The gate had no discriminating power at all.

Contributing to the original mistake: `fm_diffusion.py:178` ends in a `\` continuation, which
I read as leading to a fallback clause. It does not — the continuation is only the inline
`... if projector is not None else False`.

### The underlying bug is real, and it triggers at the deployed threshold

Gen7's guard is only the threshold term:

```python
near_end = loop_idx >= (1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3
```

With the deployed `diffusion_timestep_threshold: 0.5` (`config/visual_aligning_eval.yaml:48`),
the arithmetic is:

```
  K=1   loop_idx=0:  fm near_end=False   mf/af near_end=True     <-- fm NEVER projects
  K=2   loop_idx=0:  fm near_end=False   mf/af near_end=False
        loop_idx=1:  fm near_end=True    mf/af near_end=True
  K=4   loop_idx=2,3 both project on all arms
```

**Only K=1 is affected, and only on the fm arm** — exactly the low-NFE regime this research
line is about. mf/af are safe twice over: `int((1-0.5)*1) == 0` truncates to 0, *and* the
explicit terminal-step fallback fires.

---

## 3. What changed

### `mix_visual_aligning_test/gates_mix_visual.py`

G6 is now a **behavioural test**, not a source heuristic:

- **`_SpyProjector`** — mimics the surface `p_sample_loop` uses (`.gradient=False`,
  `.diffusion_timestep_threshold`, `.project()`), performs an identity projection and
  **counts calls**.
- **`_eval_threshold()`** — reads `diffusion_timestep_threshold` from
  `config/visual_aligning_eval.yaml`, the same file the config block reads, so the gate can
  never drift from what is actually deployed.
- **`gate_g6()`** — builds each arm (`mf`, `af`, `fm`) with `if_vision=False` (state-only, no
  vision encoder, **CPU-only**), runs `p_sample_loop` at `K=1`, and asserts `project()` was
  called.

There is nothing left to drift: no re-implementation of the guard predicate, no string
matching. The sampler is executed and the call is observed.

**Failure semantics, deliberately asymmetric:**
- `mf`/`af` not projecting ⇒ **G6 FAILS**. That would mean the DPCC cage is off.
- `fm` not projecting ⇒ prints a loud `KNOWN UPSTREAM DEFECT` banner but does **not** fail.
  It is a Gen7/Gen6V4 defect, not a Gen14 regression, and failing here would block the
  pipeline's `--dependency=afterok` chain indefinitely.

Also updated: the module docstring's gate/hardware map (G6 moved from "needs GPU" to
CPU-only) and the now-obsolete trailing "runtime confirmation" hint.

### `Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh`

Comment block rewritten to state that G6 is a runtime test and that the `fm` banner is
expected — with the explicit instruction to read it rather than skip past it.

---

## 4. Verification

| Check | Result |
|---|---|
| `py_compile` on the changed gate script | PASS |
| **G0 re-run locally** (no torch needed) | **PASS**, 23/23 — no regression from the edit |
| `bash -n` on the changed sbatch | PASS |
| Guard arithmetic simulated offline for K=1,2,4 at threshold 0.5 | matches §2 table exactly |
| `betas` buffer present in all three engines (used by `p_sample_loop`) | confirmed 1/1/1 |
| Files touched outside Gen14 | **none** |

**Not verified:** G6's new runtime path has **not been executed** — it needs torch, which
this container does not have. It must run on the cluster. Expected output:

```
  ok   mf: project() called 1x at K=1
  ok   af: project() called 1x at K=1
  !!   fm: project() NEVER called at K=1  <-- KNOWN UPSTREAM DEFECT
  ... banner ...
  G6 PASS (runtime)
```

If the `fm` leg instead reports `project() called`, upstream Gen7 has been fixed and the
gate's expectation should be updated.

---

## 5. Still open — NOT done in this fix

1. 🔴 **The upstream Gen7/Gen6V4 projector guard is unfixed.** The one-line change belongs in
   `fm_visual_aligning/models/diffusion.py` (add the terminal-step disjunct, mirroring
   `mf_diffusion.py:285`), then Gen14 re-copies and G0 re-verifies. **Not applied here**: it
   modifies existing working code, which needs explicit go-ahead, and it should be its own
   commit against Gen7 rather than being smuggled into a Gen14 fix.
   **Consequence until then:** any `fm`-arm K=1 result showing constraint violations is
   measuring an *unprojected* trajectory. K≥2 is unaffected.
2. **`VisualAgentWrapper` candidate-selection audit** against `ecbae16f` / `a6a7a8ad` — still
   outstanding from the init changelog.
3. **G2's peak-memory note** is now redundant (see §1) but left in place; harmless.

---

## 6. Observation for α-Flow training

G4's curve is steeper than "1→0 across the budget" implies: with `af_alpha_gamma=25.0` the
sigmoid holds α≈1.0 through the first ~25 % of training and reaches 0 by ~75 %, compressing
the transition around the midpoint. So the run is roughly *first quarter pure flow matching →
sharp transition → last quarter pure MeanFlow*. Correct and monotone, but worth knowing when
reading the α-Flow loss curve — a plateau in the first 25 k steps is the schedule, not a stall.

---

## 7. Commands

```bash
# Re-run the full battery (G6 now actually tests something)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh

# G6 alone — CPU-only, fast
python mix_visual_aligning_test/gates_mix_visual.py --gate g6

# G0 alone — runs in this container, no torch
python3 mix_visual_aligning_test/gates_mix_visual.py --gate g0
```

Training/eval commands are unchanged from
[`../init/CHANGELOG_Gen14_coding1.md`](../init/CHANGELOG_Gen14_coding1.md) §8.
