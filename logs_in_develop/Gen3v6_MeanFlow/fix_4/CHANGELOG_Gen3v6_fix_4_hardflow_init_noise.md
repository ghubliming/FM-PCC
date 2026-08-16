# Gen3v6 fix_4 — HardFlow arm (U3): initial-noise scale + three accounting defects

**Date:** 2026-07-30 · **Scope:** `flow_matcher_v3_meanflow/sampling/hardflow_projection.py`,
`FM_v3_meanflow_test/gates_hardflow_meanflow.py`, `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py`
**Trigger:** pre-port sanity audit before replicating the U3 HardFlow arm into Gen3v7 (α-Flow).
**Status:** code changed, **not yet re-run** — every gate and every number below needs a cluster run.

---

## 1. The blocker — arm C sampled at half the trained noise scale

`HardFlowSampler.sample()` opened its ODE with

```python
x_init = 0.5 * torch.randn(batch_size, L.horizon, L.transition_dim, device=self.device)
```

and the module docstring asserted *"The initial noise matches FMv3's sampler (`0.5 * randn`)"*,
with the inline comment *"so arms A, B and C start from the same distribution."*

**That was true for Gen12 and false for Gen3v6.** The U3 port is a verbatim copy of Gen12's
sampler, and the two generations do not share a noise law:

| | initial noise | source |
|---|---|---|
| Gen12 base (FMv3ODE) | `0.5 * randn` | `flow_matcher_v3/models/diffusion.py:164,227` |
| **Gen3v6 (MeanFlow)** | **`randn` (σ=1.0)** | `mf_diffusion.py:204` — *"sigma=1.0 to match q_sample training noise"* |
| Gen3v6 training noise | σ=1.0 | `mf_diffusion.py:180-183`, `noise = torch.randn_like(x_start)` |

So arm C (HardFlow) started every plan from **N(0, 0.25·I)** while the model was trained on, and
arms A/B sampled from, **N(0, I)**. Two consequences, both silent — no crash, no solver error:

1. **Off-distribution τ=0 state.** The mean-flow field was queried at a starting point of half the
   norm it was ever trained to handle.
2. **The A/B/C comparison was confounded at the root.** Arms B and C did not share a start
   distribution, which is precisely the property the sampler's own comment claimed to guarantee.

### The fix

`init_noise_scale` is now an **explicit, required** argument on `HardFlowSampler` (no default — a
wrong value is invisible, so the call site has to state it). `HardFlowPolicy` defaults it to
Gen3v6's 1.0, and the eval driver passes it explicitly anyway. The draw is isolated into
`HardFlowSampler.draw_init_noise()` so it can be gated without standing up an NLP.

A porting warning now sits at the top of the module: **the noise scale is the one thing that does
not transfer between generations.** It names where to read it for Gen3v7 (`af_diffusion.py:260`,
σ=1.0) and explicitly warns *not* to read it from `flow_matcher_v3_alphaflow/models/diffusion.py`,
which is the legacy FMv3ODE class sitting in the same folder and still on 0.5 — the exact shape of
trap that produced this bug.

### New gate

`gates_hardflow_meanflow.py::gate_h3` draws 4096 samples and asserts both the declared scale and
the empirical std against `MEANFLOW_INIT_NOISE_SCALE = 1.0` (2% tolerance — generous, but 0.5 vs
1.0 is a 50% error). Wired into the default gate run alongside H0/H1.

---

## 2. Severity assessment — what the pre-fix results are still worth

### ❌ Invalidated: every `hardflow_new-*` number produced so far

Jobs **23981** (K=2) and **24021 / 24022 / 24023** (K=1/5/20). Arm C ran at σ=0.5 throughout. All
of the following in [`../U3/INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md`](../U3/INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md)
must be treated as **unsound until re-run**:

- *"HardFlow reaches DPCC-parity at K=2"* (the original headline).
- The **crossover table** (HF ahead at K=1–2, behind at K≥5) — both sides of the crossover involve
  a mis-initialised arm C.
- The conclusion that **the in-loop NLP manufactures the collapsed trajectories.** The decisive
  "replan 0, identical inputs" comparison was **not** identical inputs: with the same
  `torch.manual_seed(i)`, arm C's `x_init` is exactly 0.5× arm B's, so candidate 2 going
  `0.097 → 5e-5` compared a σ=1.0 draw against a σ=0.5 draw. **The observation is real; the
  attribution is not established.**
- The **28.7% / 30.9% HF candidate-collapse rates** at K=5/K=20, and the **200/200 frozen-action**
  measurements — real measurements of a mis-configured arm. Both cause and magnitude are open.
- The `hardflow_new-c` recommendation to drop `-c`. Probably still right, but no longer supported.

### ⚠️ Probably survives, re-confirm

- **The compute-cost table** (HF/DPCC = 4.4× at K=1 → 1.06× at K=5 → 1.20× at K=20). Per-step cost
  is structural — NFE and NLP-solve counts per candidate depend on `K` and the activation
  threshold, not on the noise scale. But episode lengths and NLP failure counts *do* shift with
  input quality, so the arm-C rows should be re-read off the fixed run rather than carried over.

### ✅ Fully unaffected: arms A and B, and the whole `dpcc-c` investigation

Arms A (`diffuser`) and B (`dpcc-*`) never enter `hardflow_projection.py` — they run
`mf_diffusion.py::p_sample_loop`, which was already correct at σ=1.0. Therefore **everything** in
[`../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md)
stands unchanged:

- The 28.1% K=2 generative collapse and its bimodal displacement histogram.
- The K-sweep localization (0% collapse at K=1/5/20, 28.1% at K=2) and the falsification of the
  large-h hypothesis.
- The "stay put at the current pose" characterisation of the collapsed mode.
- The DPCC control results at every K, including `dpcc-c-tightened` = 1.0 across all scenarios at
  K=5 and K=20.

Only §6 item 6 and the cross-links describing the *HardFlow* mirror-image defect inherit the
caveat above.

### One honest loose end

The σ bug does **not** obviously explain the K-dependence of the HF collapse (0% at K=1, ~29–31% at
K=5/K=20). A constant mis-scaling at τ=0 applies equally at every K, yet the failure grows with the
number of in-loop NLP solves. So the **NLP-compounding hypothesis is not dead — it is merely
unproven**, and it is now confounded with the σ error. The fixed re-run separates them: if collapse
persists at K=5/20 with σ=1.0, compounding is confirmed; if it vanishes, the σ bug was the whole
story.

---

## 3. Secondary fixes (same file, no bearing on §2)

**3.1 `candidate_cost='control'` silently reproduced `'prox'`.**
`cand_ctrl` accumulated `‖(X_next − X_ref)/dt‖`, which is the NLP correction — and since
`X_next − X_ref = τ_next·(X1_proj − X1_ref)` on active steps and `0` otherwise, that is a
τ-reweighted, un-squared copy of `cand_prox`, not the documented control effort `Σ_k ‖u_k‖`.
Selecting `'control'` therefore ranked candidates by intervention magnitude, the same axis as
`'prox'`, rather than offering the independent ranking the option advertises. It now accumulates
the velocity magnitude at each ODE step, as documented. **No effect on any result to date** —
every run used the default `candidate_cost='prox'`.

**3.2 `infos['nfe']` mixed cumulative and delta semantics.**
`nfe` was the running total since sampler construction while `nlp_solves` / `nlp_failures` were
per-call deltas, so summing `infos` across an episode — which is exactly what the eval driver does
for the NLP counters — would have grown quadratically. All three are per-call deltas now, with the
running total preserved as `infos['nfe_total']`. **The logged `NFE=` figure is unchanged**, because
the eval driver reports `policy.nfe` (the sampler attribute), not the infos field.

**3.3 Misleading `'random'` comment.**
`_select` returns slot 0 for `'random'`; it does not draw. This is *correct* — the randomness is in
the noise, so slot 0 is an unbiased sample of the fan, and DPCC's `Policy` does the identical thing
(`which_trajectory = 0`). **Behaviour deliberately unchanged** (changing it would break arm B/C
parity); only the comment was fixed.

---

## 4. Files touched

| file | change |
|---|---|
| `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` | required `init_noise_scale` on `HardFlowSampler` + `draw_init_noise()`; porting warning in the module docstring; `cand_ctrl` = `Σ‖u_k‖`; `nfe` delta + `nfe_total`; `'random'` comment |
| `FM_v3_meanflow_test/gates_hardflow_meanflow.py` | `MEANFLOW_INIT_NOISE_SCALE`; new `gate_h3`; H1 constructor updated; H3 added to the default run |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | passes `init_noise_scale=1.0` explicitly to `HardFlowPolicy` |

**Not touched, deliberately:** `flow_matcher_v3_hardflow/` (Gen12). Its `0.5 * torch.randn`
(`hardflow_projection.py:419`) is **correct for Gen12** — see §4b. Per the sibling-isolation
convention, the generations keep their own copies. No other module imports the Gen3v6 file, so the
newly-required argument breaks nothing.

### 4b. Gen12 is NOT affected — its arms agree with each other

Gen12's eval builds `flow_matcher_v3.models.diffusion.GaussianDiffusion`
(`eval_FM_v3_hardflow.py:76`), whose `p_sample_loop` draws `0.5 * torch.randn`
(`flow_matcher_v3/models/diffusion.py:164,227`). Its arm C draws `0.5 * randn` too. **All three
Gen12 arms share one start distribution, so every Gen12 HardFlow-vs-DPCC comparison remains valid.**

The bug is **not** "0.5 is the wrong number." The 0.5× sampling scale is a deliberate
low-temperature convention inherited from upstream DPCC/diffuser — train at σ=1.0, sample at σ=0.5
for less diverse, higher-quality plans; the train/infer asymmetry is intentional and is documented
as such in [`Gen9/.../U3_audit_Fable.md`](../../Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/U3/U3_audit_Fable.md)
line 251. FMv3ODE follows it (`q_sample` at σ=1.0, `p_sample_loop` at σ=0.5).

The bug is that **Gen3v6 deliberately left that convention and the port did not follow.**
`mf_diffusion.py:204` reads `x = torch.randn(shape)  # sigma=1.0 to match q_sample training noise`
— an explicit, commented decision to drop the low-temperature draw for the MeanFlow generation. The
copied HardFlow sampler kept Gen12's 0.5, so **within Gen3v6** arms A/B sample at 1.0 and arm C at
0.5. Whichever value is preferable in the abstract, the arms disagreeing with each other is what
voids the comparison — the same principle MASTER_TEST_HISTORY records under MATH-03/MATH-04
("Standardized to forward 0→1 integration and matching noise scales").

This is a **sibling-sync failure**, the hazard the repo's copy-modify convention exists to manage:
a base-model change in one generation that its own copied sampler never picked up. Hence the
required-argument-plus-gate design rather than a corrected constant — the next generation is forced
to state its scale rather than inherit one.

---

## 5. To run on the cluster (nothing was executed locally — no Python env in this container)

```bash
# 1) gates first — H3 is the new pin, H0/H1 must still pass
python FM_v3_meanflow_test/gates_hardflow_meanflow.py

# 2) re-run the matched-K sweep with the fixed arm C
HFFM_FLOW_STEPS=1  HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
HFFM_FLOW_STEPS=2  HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
HFFM_FLOW_STEPS=5  HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
HFFM_FLOW_STEPS=20 HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
```

K=2 is included this time: the previous K=2 arm-C data is invalid, and the DPCC arm re-running is
harmless (it should reproduce the documented `dpcc-c` collapse exactly — a useful control that the
fix changed nothing on the B side).

**Then**, and only then, port to Gen3v7 — with `init_noise_scale` read off `af_diffusion.py:260`.

## 6. Checks that the audit cleared (no change needed)

- **`h=0` identity** — `_velocity_batch` passes `h=torch.zeros_like(t)` explicitly; the backbone
  computes `r_abs = time; t_abs = r_abs + h`, so `r == t`. Pinned by `gate_h1`.
  α-Flow supports the same identity and trains it directly (`af_ratio_fm = 0.5` forces half the
  batch to `r == t` with `u_tgt = v`), so the principle transfers to Gen3v7.
- **DOF layout / `s0` pinning** — matches HardFlow's `oc_dof`; asserted index-by-index by
  `gates_hardflow.py::gate_layout`.
- **Constraint assembly** — mirrors DPCC semantics exactly (`skip_initial_state`, action dims bound
  from step 0, observation dims from step 1), so arms B and C enforce the same feasible set.
- **`dt` scoping** — the NLP's environment `dt` (1.0, for `deriv`) and the ODE's `dt = 1/K` are
  separate variables in separate scopes. No collision.
- **Activation gate** — `k >= (1 − threshold)·K` plus the forced final step, matching DPCC's
  `diffusion_timestep_threshold` polarity (fix_6).
