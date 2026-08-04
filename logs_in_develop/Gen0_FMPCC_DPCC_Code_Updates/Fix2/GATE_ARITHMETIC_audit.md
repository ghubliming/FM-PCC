# Gate arithmetic audit — three different `diffusion_timestep_threshold` semantics in one repo

**Date:** 2026-08-04
**Trigger:** the question "is `post_processing` really the same as `dpcc-r`? it feels like it
should be one final projection only."
**Status:** **superseded — the audit's findings were acted on.** This document is kept as the
derivation; the fix it motivated is in `CHANGELOG_Gen0_Fix2_dpcc_threshold_wiring.md` §2b/§2c.
Form B no longer exists in any live package: the five files listed in §5 were converted to the
guarded form A, and `post_processing` was restored in ten eval scripts.

---

## 1. Short answer to the question

**Both halves of the intuition are right.**

1. **`post_processing` is meant to be one final projection.** Upstream implemented it exactly
   that way: `threshold = 0` ⇒ the gate fires only on the last denoising step ⇒ project the
   final sample once. See `UPSTREAM_DPCC_same_bug_analysis.md` §4b.
2. **In the avoiding family it equalled `dpcc-r`** (before this fix) — verified, not guessed:
   `sha256(obs_all.npy)` byte-identical in all three envs × both suffixes, in the Gen0 baseline
   runs *and* in all four `temp/0408/FMv3ODE` runs. Those scripts have no `post_processing`
   branch, so the arm inherits `dpcc-r`'s threshold and its `random` selection.

And the follow-on suspicion — *"then the ones that aren't the same as DPCC post-processing are
wrong"* — turned out to be **correct, and it landed on a live arm**: in the FM visual-aligning
path the arm was wired to `threshold = 0.0` correctly, but the **gate** it ran against turned
that into **zero** projections, so `post_processing` silently became `diffuser`. §4.
Both are now fixed; §3–§4 below describe the pre-fix state that motivated the change.

---

## 2. Three gates, one config key

| form | where | expression |
|---|---|---|
| **C — DPCC** | `diffuser/`, `diffuser_visual_aligning/`, `diffuser_visual_avoiding/`, `mix_visual_aligning/models/diffusion.py` | `t <= T * K`, with `t` counting **down** from `K−1` |
| **A — int + terminal guard** | `flow_matcher_v3_ode_selectable`, `_meanflow`, `_imeanflow`, `_alphaflow`, `_drifting`, `_uav`, `imf_visual_aligning`, `mix_visual_aligning/models/{mf,af}_diffusion.py` | `idx = int((1−T)·K)`; `near_end = (loop_idx >= idx) or (loop_idx == K−1)` |
| **B — float, bare** | `flow_matcher_v3`, `flow_matcher_v3_hardflow`, `fm_visual_aligning`, `fm_visual_avoiding`, `mix_visual_aligning/models/fm_diffusion.py` | `near_end = loop_idx >= (1−T)·K` |

Closed forms:

```
C:  n_active = min(floor(T·K) + 1, K)
A:  n_active = max(K − int((1−T)·K), 1)      # guard floors it at 1
B:  n_active = K − ceil((1−T)·K)             # can be 0
```

### Number of projected steps, same K and T

| K | T | **C** DPCC | **A** int+guard | **B** float bare | |
|---|---|---|---|---|---|
| 20 | 0.0 | 1 | 1 | **0** | B: no projection at all |
| 20 | 0.05 | 2 | 1 | 1 | |
| 20 | 0.1 | 3 | 2 | 2 | |
| 20 | 0.2 | 5 | 4 | 4 | |
| 20 | 0.25 | 6 | 5 | 5 | |
| 20 | 0.5 | 11 | 10 | 10 | |
| 20 | 1.0 | 20 | 20 | 20 | all agree only here |
| 10 | 0.0 | 1 | 1 | **0** | |
| 10 | 0.05 | 1 | 1 | **0** | |
| 10 | 0.1 | 2 | 1 | 1 | |
| 10 | 0.2 | 3 | 2 | 2 | |
| 10 | 0.25 | 3 | 3 | **2** | A ≠ B: (1−T)·K non-integer |
| 10 | 0.5 | 6 | 5 | 5 | |

Three facts fall out:

- **C is one step ahead of A/B whenever `T·K` is an integer** — the finding already recorded as
  Part III §21.2, now generalised to every generation.
- **A and B agree iff `(1−T)·K` is an integer**, and diverge otherwise (K=10, T=0.25 → 3 vs 2).
- **Only A guarantees at least one projection.** B returns 0 at `T = 0` and at any `T` small
  enough that `ceil((1−T)·K) = K`.

---

## 3. Why `T = 0` matters — that is what `post_processing` *is*

Five eval scripts define the arm as "threshold zero":

```python
threshold = 0.0 if 'post_processing' in variant else config.get('diffusion_timestep_threshold', 0.5)
```

| script | line | gate form it runs against | projections at T=0 | correct? |
|---|---|---|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 269 | **C** | 1 (final step) | ✅ |
| `imf_visual_aligning_test/eval_imf_visual_aligning.py` | 157 | **A** | 1 (guard) | ✅ |
| `FM_v3_uav_test/eval_fm_uav.py` | 752 | **A** | 1 (guard) | ✅ |
| `mix_visual_aligning_test/eval_mix_visual_aligning.py` | 297 | **C**/`A`/`A`/**B** per engine | 1 / 1 / 1 / **0** | ⚠️ `fm` arm |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | 276 | **B** | **0** | ❌ |

**In forms A and C, `threshold = 0` does exactly what it should: one projection on the final
sample.** Form A gets there via the terminal guard (`loop_idx == K−1`), form C because `t <= 0`
is true only at `t = 0`. Form B has neither and simply never fires.

---

## 4. The live consequence

### 4a. Gen7 FM visual aligning

`fm_visual_aligning_test` wires `post_processing` correctly at `:276`, but
`fm_visual_aligning/models/diffusion.py:178` is form B. So the arm runs **zero** projections
and is computationally identical to `diffuser`. `post_processing` is in
`config/visual_aligning_eval.yaml`'s `projection_variants`, so this arm has been produced in
every run of that benchmark.

### 4b. Gen14 Visual-Mix — three schedules in one comparison

The whole premise of Gen14 is that only the generative engine differs. It does not:

| engine | model class | base | gate form | `post_processing` @ T=0 | `n_active` @ T=0.5 |
|---|---|---|---|---|---|
| `diffusion` | `VisualGaussianDiffusion` | `models/diffusion.py` | **C** | 1 ✅ | `floor(0.5K)+1` |
| `mf` | `VisualMeanFlow` | `models/mf_diffusion.py:284` | **A** | 1 ✅ | `K − int(0.5K)` |
| `af` | `VisualAlphaFlow` | `models/af_diffusion.py:342` | **A** | 1 ✅ | `K − int(0.5K)` |
| `fm` | `VisualFlowMatching` | `models/fm_diffusion.py:178` | **B** | **0** ❌ | `K − ceil(0.5K)` |

None of the four `Visual*` classes overrides the sampling loop, so each inherits its base's
gate. At the shared default `T = 0.5` the `diffusion` arm gets **one extra projected step** over
the other three; at `K = 2` (the Gen14 U6 default for `mf`/`af`) that is **2 of 2 steps versus
1 of 2** — a 2× difference in projection budget between arms that are supposed to differ only in
their generator.

### 4c. Gen12 HardFlow — checked, and clean for the published comparison

`flow_matcher_v3_hardflow/models/diffusion.py:178` is form B, while the FMv3ODE sibling used for
the same study is form A. **They agree at every threshold used in Part I / Part II**
(K=20, T ∈ {0.5, 0.1, 0.05} → `(1−T)·K` = 10, 18, 19, all integers), so
`DA_20260803_HardFlow_activation_threshold_0p1.md` is unaffected. Worth stating explicitly
because it easily might not have been.

Note this is a *different* gate from the one `[Gen12fix8]` repaired: fix_8 floored the
**HardFlow NLP** activation in `sampling/hardflow_projection.py`. The **DPCC** gate in the same
generation, at `models/diffusion.py:178`, was not part of that fix — it is form A now,
under `[Gen0fix2]`.

---

## 5. What was and was not done

**Converted from form B to form A (`[Gen0fix2]`), after this audit:**

| file | line |
|---|---|
| `mix_visual_aligning/models/fm_diffusion.py` | 177 |
| `fm_visual_aligning/models/diffusion.py` | 177 |
| `fm_visual_avoiding/models/diffusion.py` | 177 |
| `flow_matcher_v3_hardflow/models/diffusion.py` | 177 |
| `flow_matcher_v3/models/diffusion.py` | 177 |

Each now reads:

```python
if projector is not None:
    snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
    near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
else:
    near_end = False
```

No past result moves: A and B agree whenever `(1−T)·K` is an integer, which covers every
threshold used to date. They differ only at `T = 0` — the case this whole audit is about — and
on non-integer `(1−T)·K`.

**Still not done, deliberately:**

- **Form C's `+1` is unchanged.** DPCC's `floor(T·K)+1` is its published behaviour; aligning it
  to A would break comparison with the paper. The operating rule is instead: **match runs on
  `n_active`, never on `T`.**
- **Frozen generations untouched** — `flow_matcher/`, `flow_matcher_unet_v2/`,
  `flow_matcher_v2/` still carry form B, paired with eval scripts that never forwarded the
  threshold anyway. See changelog §3c.

---

## 6. If this is picked up later

Ordered by value:

1. **Confirm 4a/4b empirically** — one visual-aligning run, compare `post_processing.npz`
   against `diffuser.npz` on the `fm` arm. `sha256(obs_all.npy)` equal ⇒ confirmed. This is the
   same one-command check that settled the `dpcc-r` question and needs no new code.
2. **Decide a single gate for Gen14** before any cross-engine claim is published from it. §4b is
   a comparability defect in the one experiment whose entire design is "only the engine
   differs". Adopting form A everywhere is the smallest change: it already covers 3 of the 4
   arms and is the only form that cannot produce a zero-projection schedule.
3. **Match runs on `n_active`, never on `T`** — the standing rule from Part III §21.2, now with a
   third form to trip over.
4. Only then revisit whether form C's `+1` should be reconciled with A. It is DPCC's published
   behaviour and changing it breaks comparison with the paper.
