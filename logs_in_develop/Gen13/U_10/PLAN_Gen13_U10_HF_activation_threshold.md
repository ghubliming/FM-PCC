# Gen13 U10 — PLAN: late-activation threshold on the ORIGINAL HardFlow (FM backbone)

**Date:** 2026-07-25 · **Status:** plan only, **no code written** · **Eval-only** (reuse the FM checkpoint)
**Codebase:** the **vendored HardFlow repo** `HardFlow/` (NOT `flow_matcher_v3_hardflow/`)
**Sibling in spirit:** Gen12 [`U4/PLAN_Gen12_U4_late_activation_threshold.md`](../../Gen12/U4/PLAN_Gen12_U4_late_activation_threshold.md)
— same *idea* (threshold-gate the per-step NLP), but a **different codebase, model, env, and horizon**.
**Goal:** run the **original HardFlow** (FM backbone) with a late-activation threshold and compare it
against the same model at full-step (every-step) NLP solving.

---

## 0. The premise question, answered: is running the original HF still possible?

**Yes — fully. Nothing needs to be added back.** The iMF work in Gen13 was purely *additive*
(sibling files `run/eval_imf.py`, `run/train_imf.py`, `hardflow/models_flow/imf/`). The **original
HardFlow FM path is untouched and runnable**:

- **model:** `TemporalUnet` + `FlowMatcher('cfm')` + `FlowPolicy` (`run/eval.py`, `run/train_fm.py`),
- **checkpoint (exists):** `logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth`
  (produced by `Slurm_Codes/sbatch/hardflow/train_hardflow.sh`; bridged into FM-PCC),
- **entry point:** `HardFlow/run/eval.py` with `guidance_method="hardflow_new"`.

So U10 is **eval-only**: reuse the existing FM checkpoint, no retraining, no backbone restoration.

## 1. Why this is NOT a copy of Gen12 U4 (the differences matter)

The user is right that this is "lots different" from Gen12. Explicitly:

| | Gen12 U4 | **Gen13 U10** |
|---|---|---|
| codebase | `flow_matcher_v3_hardflow/` (our reimpl) | **vendored `HardFlow/`** (the paper's own code) |
| backbone | FMv3ODE (`FlowMatchingODE`) | **original `TemporalUnet` FM** (H16_1e6steps) |
| env | `avoiding-d3il` (D3IL) | **`avoiding-v0`** (HardFlow's own gym) |
| horizon | H8 | **H16** |
| max_path_length | 150 | **200** |
| constraint model | DPCC `Projector` geometry (yaml) | **HardFlow `avoiding_geometry`** (pillars/novel) + **l4casadi** + **fitted linear dynamics** (`logs/avoiding-v0/dynamics/linear_model.npz`) |
| comparison | arm C vs DPCC (arm B) | **HF-vs-HF: threshold ON vs OFF, same FM model** |

So U10 is a clean **ablation of the threshold on the paper's own algorithm and environment** — not a
cross-method comparison. It answers: *does late-only activation preserve HardFlow's safety/quality
while cutting cost, on HardFlow's own setup?* This is arguably the **most direct** validation of the
threshold idea, because it is the paper's algorithm tested where the paper tested it.

## 2. The math — the threshold is sound here for the same reason as U4

Gen13 runs the paper's exact algorithm, so the paper's guarantees apply verbatim:

- **Terminal safety guarantee (Prop. safety_guarantee).** `h(x_N) ≤ 0` derives from the **final step
  only**: at `t_N=1`, scheduler boundary `α₁=1, β₁=0` collapses the update to `x_N = x̂_N*`, the NLP
  solution, which is feasible. Intermediate steps do not affect the guarantee. ⇒ **skipping early-step
  NLPs preserves safety, provided the final step is always solved.**
- **Paper explicitly recommends it** (App. "Feasibility, Stability, and Efficiency"): *"it is not
  necessary to solve the constrained optimization problem at every sampling step … skip the early
  steps and activate constrained optimization only in the later stages."*

See Gen12 U4 §2 for the full quotes and proof trace — identical here (same paper, same algorithm).

## 3. What already exists in the vendored HardFlow — and its gap

The vendored `HardFlow/hardflow/models_flow/flow_policy.py` **already has a binary late switch**, in
**two** places:

- `hardflow_new_forward` (the black-box NLP-per-step, guidance_method `hardflow_new`):
  ```python
  if self.cfg.hardflow_activation == "late":
      if k < self.oc_N_steps // 2:   control_flag = False   # skip first half
  ```
- `hardflow_forward` (the l4casadi variant): the same pattern under `cfg.projection_option`.

Config `flow_matching.py`: `hardflow_activation: str = "all"`.

**Gaps (same as Gen12 U4):**
1. **Binary** (`all`/`late`), hardcoded at `oc_N_steps // 2` — no continuous threshold to sweep.
2. **No explicit final-step guard** — works today only because `k=N−1 ≥ N//2`, but a general
   threshold could skip the terminal solve and silently void the safety guarantee.
3. Never systematically evaluated (all runs used `all`).

## 4. The upgrade (in the vendored HardFlow)

### 4.1 Continuous activation threshold
Add `hardflow_activation_threshold: float = 0.0` to `hardflow/config/flow_matching.py`. Replace the
binary gate in `hardflow_new_forward` (and, for completeness, `hardflow_forward`) with:
```
solve NLP at step k  ⇔  t_{k+1} ≥ hardflow_activation_threshold   OR   k == oc_N_steps - 1
```
- `0.0` → every step (today's `all`; the full-step baseline).
- `0.5` → last half (today's `late`; DPCC-parity threshold).
- `→1.0` → terminal-only NLP (pure post-hoc projection of the final sample).

Keep `hardflow_activation: all|late` as back-compat aliases (0.0 / 0.5).

### 4.2 🔴 Final-step guard (the one invariant)
The `OR k == oc_N_steps-1` clause is mandatory — the safety guarantee (Prop.) needs the terminal NLP.
A gate (§6) must assert the final step is always solved and `x_N` is feasible for any threshold.

### 4.3 Scope of edit — additive, vendored-repo-respecting
Per repo convention the vendored `HardFlow/` should stay close to upstream. Options, least invasive
first:
- **(preferred)** extend the existing `hardflow_activation` handling (it is already a Gen13-era
  addition, not pristine upstream) to parse a threshold; keep `all`/`late` working.
- Wire `--hardflow-activation-threshold` through `run/eval.py`'s arg/config plumbing so a sweep needs
  no code edits.
- Encode the threshold in the eval output dir (`H16_fm_hardflow_new_K{K}_act{thr}`) so a sweep never
  overwrites (Gen13 lost results twice to hardcoded names — fix_1/fix_7).

## 5. Experiment design — HF threshold ON vs OFF, same FM model

Reuse `logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth`. Single backbone, single env, sweep only
the threshold (and K, matched).

| factor | values |
|---|---|
| backbone | original HF FM (H16, `hardflow_new`) — **fixed** |
| K = `ode_t_steps` | {2, 5, 10, 20} (matched per comparison) |
| activation_threshold | {0.0 (full-step baseline), 0.5 (late), 0.75, →1.0 (terminal-only)} |

Metrics (HardFlow's own, from `run/eval.py`): **Safety Rate**, **Violations**, **Total Steps (safe
trials)**, **Computation Time / plan**, plus **NLP solves** (add the counter if not present). n = 200
paired trials per cell (Gen13 uses `eval_paired_n200`; reuse it), fixed seeds.

**The questions:**
1. **Does the threshold preserve safety?** Expected yes (terminal guarantee). Safety Rate at
   threshold 0.5 should equal threshold 0.0 (both 1.00 in the paper's regime).
2. **How much compute does it save?** NLP solves drop from `K` to `≈K·(1−thr)+1` per plan; wall time
   should fall roughly proportionally. Quantify the safety-vs-time trade-off curve.
3. **Does it affect sample quality** (Total Steps / path length)? The paper claims early steps are
   "unnecessary"; test whether skipping them changes the path.
4. **At which threshold does safety finally break?** Push toward 1.0 (terminal-only). If safety holds
   even at terminal-only, that is a striking efficiency result for HardFlow on its own benchmark.

## 6. Gates (add to `run/imf_gates.py` or a new `run/hf_gates.py`)

- **G-final:** for threshold ∈ {0.0, 0.5, 0.9, 0.99} and K ∈ {2,5,10,20}, the final step is active and
  the returned `x_N` is NLP-feasible (Prop. invariant, §4.2).
- **G-count:** NLP-solve count is non-increasing in the threshold and equals `#{k : t_{k+1} ≥ thr}`
  plus the forced terminal solve.
- **G-parity:** `activation='late'` and `threshold=0.5` produce identical behaviour (alias check).

## 7. Traps (Gen13-specific)

1. **Final-step guard (§4.2)** — the safety guarantee rides on it.
2. **Two gates in the code** — `hardflow_new` uses `hardflow_activation`; the l4casadi `hardflow` uses
   `projection_option`. If both are in play, apply the threshold consistently to both, or scope U10 to
   `hardflow_new` only (recommended — it is the paper's canonical algorithm and matches Gen12).
3. **avoiding-v0 ≠ avoiding-d3il.** Do not import Gen12's yaml geometry; Gen13 uses HardFlow's
   `avoiding_geometry` + the fitted `linear_model.npz`. Keep them.
4. **Console spam** (Gen13 hit this three times — fix_6/U9): with fewer NLP solves the log shrinks, but
   still set IPOPT `print_level=0` and CasADi `print_time=False`.
5. **Provenance naming** (Gen13 lost results twice): encode `act{thr}` and K in the eval dir.
6. **Matched K.** Compare threshold on/off at equal K (Gen13 fix_7's central lesson).

## 8. Success criteria

- **Minimum:** the threshold knob works on original HF; G-final holds; at threshold 0.5, safety equals
  the full-step baseline.
- **Target:** threshold 0.5 preserves HardFlow's Safety Rate and path quality while **materially
  cutting** computation time / NLP solves — quantifying the paper's "good balance" claim on H16
  avoiding-v0.
- **Stretch:** terminal-only (threshold→1.0) still safe — HardFlow's per-step NLP is largely
  unnecessary on this benchmark, a strong efficiency finding worth writing up.

## 9. Relationship to Gen12 and to the iMF line

- **Feeds Gen12 U4:** if the threshold preserves safety + cuts cost here (paper's own setup), it
  strongly supports the same upgrade in Gen12 (FMv3ODE). Cross-check the two.
- **iMF is out of scope for U10** (the user asked for *original HF*). But the same threshold trivially
  applies to `eval_imf.py`/`ImfFlowPolicy` (which inherits the same activation gate), so a later U10.x
  could repeat the ablation for iMF and compare the compute savings across backbones.

## 10. Out of scope
- Retraining (reuse `H16_1e6steps`); the l4casadi `projection`/`projection_relaxed` modes; the iMF
  backbone; any change to avoiding-v0 geometry or the fitted dynamics.

---

### Appendix — code touch-points (vendored `HardFlow/`)
- `hardflow/config/flow_matching.py`: add `hardflow_activation_threshold` (default 0.0).
- `hardflow/models_flow/flow_policy.py`: `hardflow_new_forward` gate (~line 1327) — replace the binary
  `k < oc_N_steps//2` with the threshold + final-step guard; optionally mirror in `hardflow_forward`
  (~line 863).
- `run/eval.py`: plumb the threshold arg; encode it in the output dir name.
- Reuse `Slurm_Codes/sbatch/hardflow/eval_hardflow.sh` / `eval_paired_n200_hardflow.sh` with the FM
  checkpoint (`H16_1e6steps/model_ema_20.pth`) and a threshold sweep loop.
- Paper refs: Prop. safety_guarantee (terminal feasibility) + App. Feasibility/Stability/Efficiency
  (skip early steps) — arXiv 2511.08425v3.
