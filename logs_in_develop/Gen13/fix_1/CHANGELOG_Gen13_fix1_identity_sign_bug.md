# Gen13 fix-1 — gate G1 caught a SIGN BUG in the iMF training identity

**Date:** 2026-07-18
**Trigger:** first gate run on the cluster (login node, CPU): `python run/imf_gates.py`
**Verdict up front:** NOT a CPU/no-GPU artifact — a **real math bug** in the matcher, caught by the gate exactly as designed, fixed in 2 files, before any GPU training was burned.

---

## 1. Symptom (the gate output)

| Gate | Result | Reading |
|---|---|---|
| G0 shapes | ✅ | backbone mechanics fine (3.69M params) |
| G1-A h→0 (u ≈ v) | ✅ rel err 0.0018 | identity wiring fine **at h=0** — where the h-term vanishes |
| G1-B 1-NFE lands on data | ❌ | samples reached the correct modes (80% near ±2, both modes covered) **but overshot**: mean\|x\| **2.499** vs target ~2.0, W1 0.677 |
| G1-C K1 ~ K2 | ❌ W1 0.460 | K-inconsistency = large-h field biased |
| G1-D jump composition | ❌ W1 0.453 | one full jump ≠ two half jumps = same large-h bias |

**The signature:** direction correct + h→0 correct + large-h systematically biased = **wrong sign on the h-term of the training identity.**

## 2. Root cause

`convention.py` derived, and `imf_matcher.py` implemented, the HF-convention MeanFlow identity as `u = v − h·D_tot` (compound `V = u + h·sg(D_tot)`). Re-derivation shows the correct identity is:

```
g(τ) = z_s − z_τ = h·u(z_τ, τ, h),  h = s − τ,  endpoint s fixed
dg/dτ = −v   and   dg/dτ = −u + h·D_tot        (D_tot = JVP tangents (v, +1, −1))
⇒  u = v + h·D_tot        ⇒  compound  V = u − h·sg(D_tot)
```

Cross-check against the official `u = v − (t−r)·du/dt` (aux `imf.py`): mapping τ = 1−t flips the derivative (`d/dt = −d/dτ`) and both u and v signs — the derivative term lands with a **“+”** in HF convention. The old “−” trained the u-field to a target biased at large h → exactly the observed 1-NFE overshoot and K1≠K2.

## 3. Fix (2 files, Gen13's own — pre-existing HardFlow still untouched)

| File | Change |
|---|---|
| `HardFlow/hardflow/models_flow/imf/imf_matcher.py` | one line: `V = u - pad_t_like_x(h, u) * du_tot.detach()` (was `+`), + docstring corrected |
| `HardFlow/hardflow/models_flow/imf/convention.py` | derivation rewritten correctly, official-convention cross-check added, gate's empirical confirmation recorded |

Sampler / policy / endpoint signs needed **no** change — G1-B's direction (samples moved *toward* data) independently confirmed those were correct. Both files pass `py_compile`; no stale `+ h` variant remains (grepped).

## 4. Why this vindicates the CPU gate (the "should I just submit SLURM?" question)

The math is bit-identical on CPU and GPU — GPU absence had nothing to do with the failure. Had this gone straight to SLURM: ~12 h of GPU training would have baked the bias into the checkpoint, E1–E4 would have produced quietly-wrong metrics, and the diagnosis would have been far harder (an under-fit-looking field, plausibly blamed on the 96-demo data ceiling). The 2-minute login-node gate converted that into a one-line fix. **Keep the gate-first workflow: the train sbatch runs `imf_gates.py` automatically and aborts on failure.**

## 5. Next step

```bash
# on the cluster, after git pull of this fix:
cd ~/FMPCC/FM-PCC/HardFlow && conda activate hardflow_clone
export PYTHONPATH="$PWD:$PWD/../Slurm_Codes/sbatch/hardflow/shims"
python run/imf_gates.py        # expect: ALL GATES PASSED
# then:
cd ~/FMPCC/FM-PCC && ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh
```

If G1-B/C/D still fail after this fix, the next suspects (in order) are: toy under-training in the gate itself (raw_mse_u was still falling at step 3000 — would show as *marginal* failures, not 0.46-level), then the (τ,h) sampling map. Report the new numbers rather than guessing.
