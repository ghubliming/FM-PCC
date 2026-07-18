# Gen13 PLAN — iMF backbone inside HardFlow (additive package), then FM-vs-iMF showdown on avoiding

**Date:** 2026-07-18
**Generation:** Gen13 (new). Prior chain: HardFlow replication ✅ (see `../../Code_RUN_Prepare/Replication/fix_2/RUN_REPORT_original_baseline_eval.md` §7–8), theory in `../../Research/BLEND_HardFlow_iMeanFlow.md` + `../../Research/THEORY_DeepMix_HF_iMF.md`, iMF training lessons in `../../../Gen3v4_imf/U10/K2_train_eval/ANALYSIS_imf_official_K2_train_curve_and_eval.md`.
**Goal:** replace HardFlow's FM backbone (TemporalUnet + CFM, instantaneous velocity `v`) with an **iMF average-velocity field `u`**, run the same avoiding eval, and answer: **is iMF superior to FM inside HardFlow?**
**This is a PLAN for the coding agent** — steps, files, gates, decision points. No code here.

---

## 0. THE ABSOLUTE RULE — additive package assembly, zero damage

1. **NO existing file in `FM-PCC/HardFlow/` may be edited. None.** All Gen13 work is NEW files (new modules, new entry scripts, new run scripts, new sbatch). The FM path must remain byte-identical and runnable exactly as today — we *choose* iMF by invoking the new entry points, never by mutating the old ones.
2. If during coding an edit to an existing HardFlow file appears unavoidable (e.g. a dispatch hook), **STOP and surface it** — do not make the edit. The expected answer is "copy the file to a new `_imf` sibling instead" (the FMPCC copy-modify convention).
3. `/workspaces/aux_repo/*` is **read-only source material**. Never edit it.
4. Validated baseline artifacts are frozen: do not overwrite `logs/hardflow/avoiding-v0/eval/H16_1e6steps_{original,hardflow_new}_10steps/` or the FM checkpoint. New runs get new exp names.
5. After each coding session: changelog MD under `logs_in_develop/HF_iMF/Gen13/<phase>/`. The user updates MASTER_TEST_HISTORY themselves (offer, don't edit).

---

## 1. Source material map (verified locations)

| What | Where | Note |
|---|---|---|
| **iMF torch reference (the code to port)** | `/workspaces/aux_repo/imeanflow`, branch **`origin/torch`**: `imf.py` (`iMeanFlow` class — `u_fn` :44, `sample_one_step` :71, training loss w/ JVP :135), `models/imfDiT.py`, `models/torch_models.py` | pure PyTorch — **no JAX anywhere in Gen13** (`main` branch is JAX; do not touch it, do not install jax) |
| Host sampler/seam | `FM-PCC/HardFlow/hardflow/models_flow/flow_policy.py` — `hardflow_new_forward` (~:1286, seam at :1339–1340), `x1_estimate()` (:227) | read, never edit; new subclass/copy |
| Host backbone to imitate | `FM-PCC/HardFlow/hardflow/models_flow/unet.py` (`TemporalUnet`) + `flow_matcher.py` (CFM) | copy-modify into new files |
| Host data pipeline | `hardflow/datasets/` (SequenceDataset, avoiding H16, 96 demos) | **reuse unchanged via import** |
| Host entries to clone | `run/train.py`, `run/eval.py`, `run_scripts/*.sh` | copy → `_imf` siblings |
| SLURM bridge | `Slurm_Codes/sbatch/hardflow/_hardflow_common.sh` (+ d4rl shim) | reuse via `source`, unchanged |
| FMPCC iMF experience (lessons, NOT the code source per user decision) | `Gen3v4_imf` analysis MD | knobs, failure modes, "judge by raw_mse", K1/K2 regime |
| **Aux repo is container-only** | not on the cluster | ⇒ everything needed must be **vendored into FM-PCC** as part of the new package (git-pull ships it) |

## 2. The math being swapped (one paragraph, from BLEND/THEORY)

HardFlow's per-step seam: `x̂1 = z + (1−τ)·v(z,τ)` — a first-order Euler extrapolation of the terminal sample, exactly the object iMF learns *directly*: `x̂1 = z + (1−τ)·u(z, τ, h=1−τ)` (exact up to training error, 1 NFE). The prox-NLP + pull-back machinery is untouched by the substitution. **Hazard #1 is the reversed time convention** — HardFlow: τ=0 noise → τ=1 data; official iMF: t=0 data → t=1 noise. Map: `τ = 1−t`, `u_HF = −u_iMF`, `h` equal. One wrapper owns the flip; a reconstruction gate proves it before anything downstream (Phase G1).

## 3. Design decisions (fixed now so the coding agent is never blocked)

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Where iMF lives | **new subpackage `HardFlow/hardflow/models_flow/imf/`** + `_imf` sibling entry scripts | additive; runs inside HardFlow's validated env/controller stack |
| D2 | Backbone | **copy `TemporalUnet` → new `TemporalImfUnet`**: two-time conditioning `(τ, h)` (second time-embedding branch) + **dual heads (u, v)** | trajectories ≠ images; DiT is data-starved at 96 demos (Gen3v4 §2b); v-head needed for the iMF JVP loss |
| D3 | CFG | **DROP entirely** (no labels, no omega/t_min/t_max inputs, no null token) | HardFlow conditions by state-inpainting, not class labels; Gen3v4 proved CFG was dead weight at eval and the explosion source. Removing it simplifies `u_fn` to `u(x, τ, h)` |
| D4 | Loss | port the **`imf_official`-style objective** from the aux torch branch: JVP with predicted-v tangent, adaptive `adp(L)=L/(L+c)` normalization, u-head + v-head terms | it's the paper's objective; Gen3v4 knows its failure modes |
| D5 | Interval sampling | expose `data_proportion` (h=0 anchor fraction) and `(p_mean, p_std)` as config; **defaults 0.25 / (−0.4, 1.4)** | Gen3v4 §6 recommendation — trains the large-h regime K1/K2 actually uses |
| D6 | Train budget | **100k steps** first run, ckpt every 25k, cosine LR, EMA 0.995, batch 32 | 96 demos; Gen3v4 showed most learning early; FM's 1e6 is overkill to match initially |
| D7 | Sampler K | **K ∈ {1, 2} primary** (paper regime), K=10 only as an FM-parity diagnostic | Gen3v4 §7: iMF is K-invariant, high-K resolves field roughness — don't chase it |
| D8 | Guidance integration | **Level 1 (mandatory):** new guidance method `hardflow_new_imf` = copy of `hardflow_new_forward` with the seam line using `u`; **Level 2 (optional, only after L1 wins):** MF-Newton K=2 with JVP Jacobian `∇F = I + (1−τ)∇u` (THEORY §0 item 4) | L1 is the minimal fair test; L2 is the theory's full upgrade |
| D9 | tensorboard | new `train_imf.py` makes it **optional (try-import)** | kills the fix_2 trap for anyone re-running |
| D10 | Checkpoint paths | `logs/avoiding-v0/flow/H16_imf_100k/model_ema_<n>.pth` (same tree, new exp names) | reuses eval path conventions; never collides with FM's |

## 4. Build phases (each ends with a GATE; do not proceed past a failed gate)

### Phase 0 — Vendor the iMF reference into the new package
- Create `HardFlow/hardflow/models_flow/imf/` with: `__init__.py`, `imf_matcher.py` (loss/objective — ported from aux `origin/torch` `imf.py` train step), `imf_sampler.py` (multi-step composition — ported `sample_one_step` loop), `temporal_imf_unet.py` (D2 backbone), `convention.py` (the ONE place holding the τ↔t flip, D-map of §2), `README_PROVENANCE.md` (records exact aux commit + what was changed: CFG stripped, labels removed, HF convention).
- Port = rewrite in HardFlow's conventions, torch-only, no JAX, no aux-repo imports at runtime.
- **GATE G0 (container, CPU):** package imports clean; shapes flow through `TemporalImfUnet` for a dummy (B, H16, 6) batch; u-head and v-head outputs both (B, H16, 6).

### Phase 1 — Convention wrapper + sanity math
- `convention.py` exposes HF-convention API only: `u_hf(z, τ, h)`, `x1_from_u(z, τ)`. All sign/direction logic lives here with the §2 mapping table in comments.
- **GATE G1 (container, CPU, untrained net ok for mechanics + analytic check):** with a hand-built linear-Gaussian toy where `u` is known in closed form, verify (a) `h→0` limit: `u(z,τ,h≈0) ≈ v(z,τ)`; (b) 1-NFE endpoint from pure noise lands on the data mean; (c) K=2 composition equals the exact interpolant jump. A sign error fails loudly here instead of poisoning every later metric. (Mirror the style of `Research/validate_theory.py`.)

### Phase 2 — Training path
- `run/train_imf.py`: copy of `run/train.py`, swapped to build `TemporalImfUnet` + iMF objective; imports `SequenceDataset` unchanged; logs **both** the adaptive loss AND the raw per-head MSEs (`raw_mse_u`, `raw_mse_v`, and an `a0`-style first-action MSE) — Gen3v4 §0: **the adaptive loss is flat by construction; convergence is judged on raw MSEs only.**
- `run_scripts/train_imf.sh`: copy of `train.sh` with the new entry + `H16_imf_100k` exp name + D5/D6 knobs as flags.
- Sbatch: `Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh` — sources `_hardflow_common.sh`, calls the new run script. (Cluster: `pip install tensorboard` in `hardflow_clone` only if TB logging wanted — D9 makes it optional.)
- **GATE G2 (cluster):** 100k-step train completes; `raw_mse_u` drops ≥3× from ep0 and plateaus; `a0`-style MSE < ~0.15 (Gen3v4 reference magnitudes). Spikes are expected (JVP tangent variance) — a *diverging* raw_mse is the failure signal, spikes alone are not.

### Phase 3 — Unguided eval (`original`-equivalent, the raw-field test)
- `run/eval_imf.py`: copy of `run/eval.py` that constructs `TemporalImfUnet` + iMF sampler; supports `--imf_k {1,2,10}`; loads `H16_imf_100k` checkpoints; everything else (env, controller `rh`, replan 8, constraint `novel`, CSV writing) identical so CSVs are directly comparable.
- `run_scripts/eval_original_imf.sh` (K sweep via env var), sbatch `eval_imf_hardflow.sh` with a `METHODS`-style knob.
- **GATE G3 (cluster):** K1 and K2 runs complete 50 episodes and write `trajectories.csv`. Record success/safety — **no quality bar here** (96-demo ceiling may keep the raw field coarse, per Gen3v4 §2b); the gate is only "mechanically sound + plausible trajectories, not noise."

### Phase 4 — The seam swap (guided iMF — the actual Gen13 experiment)
- New `imf_flow_policy.py` in the imf package: subclass (or additive copy) of `FlowPolicy` adding guidance method **`hardflow_new_imf`** — identical prox-NLP/pull-back loop, with (a) the terminal prediction `x̂1 = z + (1−τ)·u(z, τ, 1−τ)` replacing the Euler shot, and (b) the reference step optionally taken with `u` over `Δτ` (exact jump) instead of Euler `v·Δτ`. Solver interface, value model, geometry: untouched imports.
- NFE accounting: instrument model-call counts per env step (the headline efficiency metric).
- `run/eval_imf.py` gains `--guidance_method hardflow_new_imf`; run script `eval_hardflow_new_imf.sh`; K ∈ {2} primary, {1, 10} secondary.
- **GATE G4 (cluster):** 50-episode run completes; violations counted; CSV written.

### Phase 5 — (OPTIONAL, only if G4 shows ≥ parity) MF-Newton upgrade
- Implement THEORY's Newton pull-back using `∇F = I + (1−τ)∇u` via `torch.func.jvp`, K=2 anchors. Separate guidance name `mf_newton`. Same additive rules. Skip entirely if time-boxed.

## 5. The showdown — experiment matrix & superiority criteria

All 50 episodes, seed 0, same env/controller/constraint. Baselines are FROZEN from the replication (do not rerun them):

| Run | Backbone | Guidance | K / NFE knob | Status |
|---|---|---|---|---|
| B1 | FM (1e6) | original | ode_t_steps=10 | ✅ frozen: 4% succ / 4% safe / 0.175 s/step |
| B2 | FM (1e6) | hardflow_new | ode_t_steps=10 | ✅ frozen: 100% / 100% / 0.847 s/step |
| E1 | iMF | original-equiv | K=1 | new |
| E2 | iMF | original-equiv | K=2 | new |
| E3 | iMF | hardflow_new_imf | K=2 | **new — the headline run** |
| E4 | iMF | hardflow_new_imf | K=1, K=10 | new (secondary) |
| (E5) | iMF | mf_newton | K=2 | optional Phase 5 |

**"iMF superior to FM in HardFlow" is declared iff, comparing E3 vs B2:**
1. **Safety parity:** E3 = 100% safe, 0 violations (non-negotiable — constraint satisfaction is table stakes);
2. **Efficiency win:** NFE per env step and/or wall-clock s/step strictly below B2's (theory target: ~2–4 NFE vs the FM path's per-step budget; B2 reference 0.847 s/step);
3. **Task quality not degraded:** success 100%, mean steps-to-goal within ~±20% of B2's 50.7.

Secondary (reported, not gating): E1/E2 vs B1 — raw generative quality at 2–10× fewer NFE; expected honest outcome per Gen3v4: raw iMF may be coarser (data ceiling), and that is fine — the claim being tested is *guided efficiency*, not raw-field beauty. If E3 fails (1) → check convention gate + Level-1 seam first; if it fails only (2) → iMF is "equal but not cheaper" here, report honestly.

## 6. Deliverables checklist (what "Gen13 done" means)

- [ ] `HardFlow/hardflow/models_flow/imf/` package (6 files, §4 Phase 0–1, with provenance MD)
- [ ] `run/train_imf.py`, `run/eval_imf.py` (siblings; originals untouched)
- [ ] `run_scripts/{train_imf,eval_original_imf,eval_hardflow_new_imf}.sh`
- [ ] `Slurm_Codes/sbatch/hardflow/{train_imf_hardflow,eval_imf_hardflow}.sh`
- [ ] Gates G0–G4 passed and logged
- [ ] Trained checkpoint `H16_imf_100k`
- [ ] E1–E4 CSVs + a results MD under `Gen13/` with the §5 table filled in and a verdict against the superiority criteria
- [ ] Per-phase changelog MDs; `git status` on HardFlow shows **only additions**

## 7. Known risks & pre-answered questions

| Risk | Mitigation / stance |
|---|---|
| Sign/convention slip (the #1 hazard) | G1 analytic gate before any training; all flips quarantined in `convention.py` |
| 96-demo data ceiling → coarse u-field (Gen3v4 §2b) | expected; superiority is defined on *guided* runs (E3), where the projection compensates; D5 defaults attack the interval-starvation cause |
| JVP loss spikes | known-benign (Gen3v4 §1); watch raw_mse trend, not spikes; adaptive loss is flat by design — never judge by it |
| Two-time backbone under-conditioning (h ignored by net) | G1(a) h→0 check + monitor u-vs-v head divergence during training |
| Accidental edit of existing HardFlow files | rule §0.2 (stop-and-surface) + final `git status` additions-only check in §6 |
| tensorboard / new deps on cluster | D9 try-import; no new packages required for eval path |
| JAX contamination from aux `main` branch | port only from `origin/torch`; provenance MD records the commit |
| l4casadi | not needed anywhere in Gen13 (hardflow_new path only) |

## 8. Suggested execution order for the coding agent

`Phase 0 → G0 → Phase 1 → G1` (one container session, no cluster) → `Phase 2` code + sbatch (container) → user submits train → `G2` → `Phase 3 + 4` code while training runs (they don't depend on weights) → user submits E1–E4 → results MD → verdict. Phase 5 only on a green E3.
