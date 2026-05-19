# Fix 7 — Post-Mortem Memo

**Date:** 2026-05-19
**Status:** Results improved after Fix 7 + K=100. Exact root cause not isolated.

---

## The Honest Situation

We did not identify a single definitive bug. We fixed multiple things simultaneously
(Fix 7.1–7.5 + switched K=16 → K=100), and results improved. This memo records the
most probable failure hypotheses — ordered by estimated severity — so future debugging
has a starting point if results regress again.

---

## What We Know Changed

| Change | Category |
|---|---|
| FIX_7.1 — removed `max_episode_length` kwarg | Env construction unblocked |
| FIX_7.2 — reverted BGR→RGB conversion in dataset | Visual pipeline parity |
| FIX_7.3-A — CPU pinning + return signature restored | Eval infrastructure |
| FIX_7.3-B — BPCageCam no-arg constructor restored | Camera rendering |
| FIX_7.3-C — rod:tip non-colliding flags restored | Physics parity |
| FIX_7.4 — `eval_on_train` restored | Eval unblocked |
| FIX_7.5 — CPU bypass for vision, video wiring, return 4 values | Eval infrastructure |
| K=16 → K=100 | Diffusion chain stability |

Any one of the items above could have been the sole cause of 0% success.
Several of them could compound independently. We cannot tell which from the
evidence we have.

---

## Candidate Root Causes (ranked by estimated severity)

### 1. K=16 Noise Amplification — Probable Major Contributor

At K=16, the DDPM reverse chain reaches t=15 (the final step) with a noise schedule
amplification factor `sqrt_recip_alphas_cumprod[15] ≈ 11×`. The predicted clean
trajectory `x_recon` has std ≈ 10.5 in normalized space — far outside the `[−1, 1]`
range the LimitsNormalizer was trained on.

**What this breaks:**
```
x_recon std ≈ 10.5
→ LimitsNormalizer clips to [min, max]
→ all denoised trajectories collapse to the same boundary value
→ SLSQP projector receives constant garbage input
→ every rollout produces the same wrong trajectory
→ 0% success, constant mean_distance regardless of context
```

K=100 does not have this problem. The amplification is spread over 100 steps;
by t=50 (the SLSQP activation threshold), the trajectory has already partially
converged and the per-step amplification per step is gradual enough to stay
within a recoverable range.

**Confidence: High.** The LimitsNormalizer clipping mechanism guarantees this
behavior mathematically for K=16. Switching to K=100 alone may have been
sufficient to unblock non-zero success.

---

### 2. BGR→RGB Channel Swap (FIX_7.2) — Probable Critical Visual Bug

`cv2.imread` returns BGR. A drift commit (`6f42a73`) introduced `cv2.cvtColor(BGR2RGB)`
in the dataset loading path at eval time. If the model was trained on BGR tensors
but evaluated with RGB tensors:

```
Training:   channel 0 = Blue, channel 2 = Red   (BGR)
Eval drift: channel 0 = Red,  channel 2 = Blue   (RGB, after cvtColor)
```

**What this breaks:**
- ResNet encodes features from the wrong channels across all visual observations
- The image latent fed into the 1D temporal U-Net is systematically corrupted
- Every conditioning signal is wrong in a structured, non-random way
- The model has no path to recover context → 0% success even with perfect action head

**Confidence: High** for any checkpoint trained before the drift commit. The failure
mode is silent — no crash, no warning, just structurally wrong embeddings.

---

### 3. BPCageCam Named Key Crash (FIX_7.3-B) — Probable Critical Camera Bug

The drift version passed a named string key to `BPCageCam()`. If MuJoCo's camera
registry did not have a camera registered under that exact name:

```
Option A: render() returns a blank/zero frame (no crash, silent failure)
Option B: render() raises a runtime exception → eval crashes mid-rollout
```

**What this breaks:**
- Blank frames → ResNet receives all-zeros or noise → uninformative latent
- Effect is equivalent to FIX_7.2 but worse: not just the wrong channel order,
  but no signal at all
- 0% success guaranteed if camera input is blank

**Confidence: Medium-High.** The exact behavior depends on MuJoCo's camera
registry implementation. Blank frames are the more likely outcome than an outright
crash (since the eval was seen to proceed for a while before other issues).

---

### 4. Rod:Tip Phantom Contacts (FIX_7.3-C) — Likely Contributor, Not Sole Cause

With `contype=1/conaffinity=1`, the invisible rod tip sphere participates in the
MuJoCo contact solver during eval, generating forces not present during data collection.

```
Training data: collected without rod:tip contacts
Eval (drift):  rod:tip generates spurious contact forces near table/objects
→ actual c_pos trajectory deviates from des_c_pos even with perfect actions
→ SLSQP constraints are calibrated to dynamics without these contacts
→ trajectory error accumulates throughout rollout
```

**Confidence: Medium.** This degrades success rate but is unlikely to cause absolute
0% on its own — the contact forces are small enough that the robot would still move
toward the goal, just less accurately.

---

### 5. CPU Pinning Starvation (FIX_7.5-A) — Wall-Clock Bug, Not Success Bug

Single-core pinning with visual eval (`cpu_set={0}`) starves OpenMP, CUDA, and
SLSQP threads, causing rollouts to hang indefinitely (observed: 15+ min at
Context 0 Rollout 0). This does not directly cause wrong predictions — it prevents
the eval from completing at all.

**Confidence: High as a hang cause, irrelevant to success rate per se.**

---

## Most Likely Failure Sequence (Pre-Fix7, K=16)

```
1. K=16 amplification → LimitsNormalizer clips → SLSQP input is garbage
   → all rollouts diverge mathematically, regardless of camera or physics

2. BGR→RGB swap → corrupted visual embeddings
   → even if K were fixed, model sees wrong colors on every step

3. BPCageCam blank frames → no visual signal
   → even if BGR were fixed, camera is silent

4. Rod:tip contacts → physics mismatch
   → even if all above were fixed, accumulated trajectory error

5. CPU starvation → eval hangs
   → infrastructure never completes even if model were correct
```

Each failure is independent. Any single one could produce 0% success.
The combination guarantees 0% from multiple angles simultaneously.

---

## Why "Better After Fix7 + K=100" Doesn't Tell Us Which Bug Was Primary

The improvement is real but ambiguous because we cannot isolate variables:
- Fix 7 corrected 5 independent issues at once
- K=16 → K=100 changed the fundamental denoising stability
- The checkpoint is also at step 50k/100k — not fully trained

**We cannot attribute the improvement to any single fix.**
The most parsimonious hypothesis is that K=16 amplification was the mathematical
floor preventing any non-zero success, and FIX_7.2 (BGR) + FIX_7.3-B (camera)
were both sufficient to cause 0% independently. Fix 7 removed all three.

---

## What Would Confirm the Primary Bug (If It Matters Later)

| Test | Isolates |
|---|---|
| Eval K=100 checkpoint with FIX_7.2 reverted (RGB conversion active) | Whether BGR swap alone causes 0% |
| Eval with blank BPCageCam frames injected manually | Whether camera blank-out alone causes 0% |
| Eval K=16 checkpoint with all Fix 7 patches active | Whether K=16 alone causes 0% |

None of these are necessary unless results regress. The current priority is completing
K=100 training and validating the full-checkpoint eval.

---

## Lessons

1. Multiple infrastructure bugs can compound to guarantee 0% success — you cannot
   tell from the metric alone which one is the cause.
2. Visual pipelines have a hidden failure mode: silent channel swaps produce
   structured wrong embeddings with no crash signal.
3. D3IL parity reverts must not erase FM-PCC extensions that cover behavior
   D3IL never defined (visual mode, CPU bypass).
4. K (diffusion steps) is not just a quality hyperparameter — below a threshold,
   the noise schedule mathematically prevents the chain from converging.
