# HF·iMF — Evaluation Prep: HardFlow run-readiness + iMF backbone-swap plan

**Date:** 2026-07-15
**Scope (Gen11+ long-term line):** two questions.
1. Is `/workspaces/aux_repo/HardFlow` turnkey — params set, only a SLURM entry away from producing paper metrics?
2. What does the evaluation look like once iMF replaces the old ML backbone?

**Companion docs:** `../../Research/BLEND_HardFlow_iMeanFlow.md` (engineering/repo audit of the two methods), `../../Research/THEORY_DeepMix_HF_iMF.md` (the math + `validate_theory.py`), `../../HF_Study/MAP_Algorithm1_to_AvoidingCode.md`. iMF-side eval realities: `../../../Gen3v4_imf/U10/K2_train_eval/ANALYSIS_imf_official_K2_train_curve_and_eval.md`.

---

## Part 1 — Is HardFlow ready to produce paper metrics?

### TL;DR — YES, the "`train.sh` + `eval.sh` → paper metrics" mental model is correct.

**You do NOT need any pretrained package.** The repo is train+eval ready and self-contained: `run_scripts/train.sh` **produces** the checkpoint itself. The Google-Drive download in the README is only an **optional shortcut** to skip the long training run — not a required external dependency. Two equivalent paths:

| Path | Steps | Use when |
|---|---|---|
| **A — full (your mental model)** | `train.sh` → `fit_dynamics.py` → `eval_*.sh` | train it yourself; nothing to download |
| **B — shortcut** | download released `.pth` → `fit_dynamics.py` → `eval_*.sh` | skip ~1e6-step training, just reproduce eval numbers |

The **only** things that are not literally "run the .sh" (all one-time setup, not missing artifacts):
1. **Build the `hardflow` conda env once** — it's a *separate* env from FMPCC (`environment.yml`). Normal setup.
2. **Run `fit_dynamics.py` once** before eval — it's a separate one-liner, not folded into `eval.sh`.
3. **l4casadi** — only if you run the `hardflow`/`projection*` methods. `hardflow_new` needs none of it.

So: params ✅ set, code ✅ ready, data ✅ present. It just needs the env built + the one-off `fit_dynamics` + your SLURM wrapper. The detailed audit below flags where each piece lives so nothing silently degrades.

### 1.1 Readiness checklist

| Component | Status | Detail |
|---|---|---|
| Repo / package code | ✅ present | `hardflow/` package, `run/{train,eval,fit_dynamics}.py`, `run_scripts/*.sh` all here. |
| Avoiding dataset | ✅ present | `d3il/environments/dataset/data/avoiding/data/env_XXX_00.pkl` (~2.6 MB, ~96 env demos). Included despite `d3il/` being gitignored. |
| Paper hyper-params | ✅ set | Every eval script hard-codes the paper operating point (horizon 16, `ode_t_steps=10`, `random_repeat=50`, `controller=rh`, `replan_steps=8`, `constraint=novel`, cost scales, etc.). No tuning needed. |
| FM checkpoint | ⚙️ **you generate it** | `run_scripts/train.sh` writes it to `logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth`. `logs/` is gitignored so it's not shipped, but that's expected — training creates it. Optional shortcut: download the released `.pth` (README Google-Drive link) to skip the ~1e6-step run. **Not a required external package.** |
| Fitted linear dynamics | ⚙️ **one-off step** | `run/fit_dynamics.py` writes `logs/avoiding-v0/dynamics/*.npz`. Run once before any `--dynamics_constraint` eval (e.g. `hardflow_new`). Cheap (CPU, minutes). ⚠️ Eval silently proceeds without it (`eval.py:529`), so a missing model degrades results *without erroring* — run it first. |
| **`hardflow` conda env** | ❌ not built here | `environment.yml` builds a **separate** env (`python=3.9`, `d4rl` from git, `minari`, `mujoco==2.3.7`, `gym==0.20.0`) — distinct from the FMPCC env. Must be created on the cluster; the FMPCC env will not satisfy it. |
| **l4casadi (CUDA)** | ⚠️ conditional | Required for `hardflow`, `projection`, `projection_relaxed`. NOT in `environment.yml`/`requirements.txt` — needs a manual CUDA build. **Not** needed for `hardflow_new`, `original`, `oc_flow`, `gradient_guidance`. |
| SLURM entry | ❌ absent | `run_scripts/*.sh` are bare `python run/... --device cuda:0` local scripts. No sbatch wrapper, no EGL/GPU-isolation guard, no conda activation. This is the piece the user flagged. |

### 1.2 The honest verdict

- **"All parameters set?" → Yes.** The scientific configuration is complete and matches the paper; you do not need to reverse-engineer hyper-parameters.
- **"Is `train.sh` + `eval.sh` → paper metrics valid?" → Yes.** That's exactly the intended flow (Path A). No pretrained package required; training generates the checkpoint.
- **The only setup around it:** (0) build the `hardflow` conda env once, then within a run (1) `fit_dynamics.py` once, (2) `train.sh` (or download the `.pth`), (3) `eval_*.sh`. Your SLURM wrapper just orchestrates these.
- **Closest-to-turnkey method:** `hardflow_new` (the l4casadi-free HardFlow). It reproduces the same numbers as `hardflow` (per README) with no CasADi build. Recommend standing up **`hardflow_new` + `original`** first as the minimal paper-metric pair, then add the l4casadi baselines only if you need the full comparison table.

### 1.3 What "produce paper metrics" concretely requires (ordered)

```
STEP 0  conda env create -f environment.yml && conda activate hardflow
        pip install -r requirements.txt && pip install -e .
        # (only if you need hardflow/projection*): build l4casadi CUDA
STEP 1  get checkpoint → logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth
        # option A: download (README Google-Drive link)
        # option B: bash run_scripts/train.sh   (1e6 steps — long; cluster job)
STEP 2  python run/fit_dynamics.py             # → logs/avoiding-v0/dynamics/*.npz  (cheap)
STEP 3  bash run_scripts/eval_hardflow_new.sh  # + eval_original.sh, ...
        # → logs/avoiding-v0/eval/<exp>/trajectories.csv
STEP 4  notebooks/collect_results.ipynb        # aggregate the CSVs into the table
```

Each eval writes `trajectories.csv`; `notebooks/collect_results.ipynb` aggregates. **Metrics to report** (paper Sec VII.A): success rate, constraint-violation rate / obstacle hits, path cost / tracking error, and NFE budget.

### 1.35 Is it a clean 0→metric-table pipeline? (rebuild-feasibility — the real question)

**Yes.** Every stage from env-build to the final paper table is provided and unambiguously guided. Rebuilding is feasible; you do **not** need to copy the math into FMPCC merely to reproduce paper numbers.

| Stage | Provided | Clarity |
|---|---|---|
| Env build | `environment.yml` + `requirements.txt` + `pip install -e .` | exact ordered commands (README Setup) |
| Data | bundled in `d3il/.../avoiding/` | no download step |
| Train | `run_scripts/train.sh` | one command, params baked |
| Fit dynamics | `run/fit_dynamics.py` → `logs/<env>/dynamics/linear_model.npz` | one command; eval reads `eval.py:517` |
| Eval (7 methods) | `run_scripts/eval_*.sh` | one script each |
| **Final metric table** | `notebooks/collect_results.ipynb` | ✅ real & complete — reads `trajectories.csv` → safety-rate, success-rate, steps, compute-time → `tabulate` table with paper method names |

**Two friction points to prepare for (not blockers):**
1. **`torch` is unpinned** — absent from *both* `environment.yml` and `requirements.txt`, though the repo is entirely PyTorch. You must install the correct torch+CUDA build for the cluster GPUs yourself. This is the one genuinely ambiguous step.
2. **`numpy==2.0.2` vs vintage deps** — pinned against `d4rl` (2021 commit), `gym==0.20.0`, `mujoco==2.3.7`; numpy-2 + these old packages can resolve messily. Expect possible manual version nudging.
3. **l4casadi** — external CUDA build, manual; only for `hardflow`/`projection*`. `hardflow_new` avoids it.

### 1.36 Rebuild HardFlow, or copy the math into FMPCC? — split by goal

- **To reproduce the paper baseline (FM backbone): rebuild HardFlow as-is.** Re-deriving its IPOPT prox-NLP + Newton pull-back + `ProxyValueModel` + avoiding-geometry constraints + RH controller inside FMPCC is far more work than building one conda env. The pipeline is turnkey; use it.
- **For the iMF swap (the actual end goal): copy HardFlow's constrained-sampling math into FMPCC.** iMF already lives in FMPCC (debugged: K2 fix, adaptive loss, convention handling). Porting HardFlow's sampler math (`flow_policy.py` `hardflow_new_forward` + `x1_estimate` + the value/geometry pieces) *into* FMPCC is lighter than porting the whole iMF engine *out* into a separate `hardflow` env — and it puts the seam (§2.1) right where iMF is.

**Recommended sequence:** rebuild HardFlow standalone → get the FM baseline table (de-risks everything, confirms env + numbers) → then copy the (now-validated) sampler math into FMPCC for the iMF swap.

### 1.4 SLURM entry — spec (mirror the FMPCC iMF sbatch, but a *different env*)

Model it on `Slurm_Codes/sbatch/iMF/train_imf.sh`, changing three things:
- **Env:** `conda activate hardflow` (NOT `FMPCC`); repo root points at the HardFlow checkout, not `FM-PCC`.
- **Working dir:** `cd` into the HardFlow repo so relative `logs/`, `d3il/`, `run/` resolve.
- **Keep the FMPCC EGL/GPU-isolation guard** (`MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=$ALLOCATED_GPU`, the GPU-LEAK abort) — MuJoCo rendering on the headless nodes needs it, and it's the standing rule for this cluster.
- Submit through `Slurm_Codes/submit.sh` per the repo convention, not raw `sbatch`.

Pipeline shape: one sbatch that runs `fit_dynamics` (once, guard on the `.npz` existing) → the eval script(s). Checkpoint acquisition (STEP 1) is best done as a separate manual/one-off step, not inside the recurring eval job.

> **Decision needed from you before writing the sbatch:** checkpoint by **download** or **train-from-scratch**? And **which methods** — just `hardflow_new`+`original`, or the full 7 (pulls in the l4casadi build)? These change the job graph.

---

## Part 2 — iMF replacing the backbone: evaluation plan

Goal: swap HardFlow's generative brain (instantaneous velocity `v(z,τ)` from `TemporalUnet` + `ConditionalFlowMatcher`) for an **average-velocity iMF field** `u(z,r,t)` (+ co-trained v-head), then evaluate. The theory says this is a *repair*, not just a substitution — see `THEORY_DeepMix_HF_iMF.md` §0–§1.

### 2.1 Where the backbone lives (swap points)

| Concern | File / anchor | Note |
|---|---|---|
| Model class | `run/eval.py:531` `flow_model = TemporalUnet(...)`; also the `WrappedFlowUnet` clone for l4casadi (`:583`) | iMF backbone slots in here; both instantiations must swap or the l4casadi path desyncs. |
| Flow-matching engine | `hardflow/models_flow/flow_matcher.py` (`FlowMatcher`, `ConditionalFlowMatcher`, `flow_matching_type="cfm"`) | Add an `imf`/average-velocity type; the FMPCC iMF engine (`imf_diffusion.py`) is the reference implementation to port. |
| Checkpoint path | `f"model_ema_{flow_cp}.pth"` under `logs/.../flow/<flow_exp_name>/` | iMF checkpoint needs its own `flow_exp_name`. |
| The **seam** (what iMF fixes) | `hardflow/models_flow/flow_policy.py:1339-1340` (`hardflow_new_forward`) and `x1_estimate()` `:227` | HardFlow's `x̂1 = z_ref + (1−τ)·v` → replaced by iMF's **exact** endpoint `x̂1 = z + (1−τ)·u`. This is the whole point (BLEND §2, THEORY §0). |

### 2.2 The #1 hazard — reversed conventions (verify BEFORE any metric)

HardFlow: **τ=0 noise → τ=1 data**, `v_HF = x1 − x0`. Official iMF: **t=1 noise → t=0 data**, `v_iMF = e − x`. Mapping (BLEND §1.3): `τ = 1 − t`, `u_HF = −u_iMF`, interval `h = t − r = τ' − τ`.

A sign slip makes the sampler walk *toward* noise. **Gate:** fix the convention in ONE wrapper and pass a **1-NFE reconstruction check** (`u`-endpoint lands on data from pure noise) before running any HardFlow-constrained eval. Nothing downstream is trustworthy until this passes.

### 2.3 Evaluation battery

**A. Unconstrained generative fidelity (backbone in isolation), at the iMF operating regime.**
- Evaluate at **K=1 and K=2 NFE — not a K10/K50 sweep.** iMF is K-invariant (1→2 is a small refinement), not an ODE integrator; high-K "chaos" is off-paper and only surfaces an under-fit field. This is established in the K2 analysis — read it before interpreting curves: `../../../Gen3v4_imf/U10/K2_train_eval/ANALYSIS_imf_official_K2_train_curve_and_eval.md` §7.
- Watch **`val/raw_mse`** (un-normalized), **not** the adaptive `loss`/`test/loss` (flat by construction). Same doc §0.
- Data ceiling caveat: the avoiding set is ~96 demos; average-velocity is a 2-time object and is structurally data-hungry (that ANALYSIS §2b). Set expectations — the win, if any, is *fewer NFE at parity*, not beating UNet-FM outright on raw fidelity.

**B. Constrained sampling (the actual HF·iMF claim).** Run the same HardFlow eval scripts with the iMF backbone and report, vs. the FM baseline:
- success rate, constraint-violation rate, path cost / tracking error — the paper's Sec VII.A metrics;
- **NFE budget** — the headline. Target from THEORY §0: same 0% violation at **~2–4 model evals** vs HardFlow-FM's ~40–60.
- **distributional corruption of feasible samples** (how much the projection perturbs trajectories that were already feasible) — THEORY's differentiator (MF-Newton applies ~0 correction to non-violating samples; W1-to-true-conditional 0.050 vs HardFlow-'all' 0.134). This is where iMF should *win*, so measure it, not just violation rate.

**C. Ablation the theory predicts.** Compare (i) FM baseline (HardFlow's Euler `x̂1`), (ii) iMF endpoint drop-in (`u` for the seam only), (iii) full Newton–MeanFlow K=2 with the JVP Jacobian `∇F = I + (1−τ)∇u` (THEORY §1, item 4). The gain should widen (i)→(ii)→(iii).

### 2.4 Sequencing recommendation

1. Land Part-1 HardFlow-FM baseline first (get the reference numbers with the *unmodified* backbone). Without a trustworthy baseline the iMF delta is unreadable.
2. Then do the iMF swap behind the 2.2 convention gate, evaluate battery **A** (backbone alone) before touching the constrained sampler.
3. Only then run battery **B/C** through the HardFlow eval scripts.

---

## Open decisions (need your call)

1. **Checkpoint:** download the released FM model, or train from scratch on the cluster?
2. **Method set for Part 1:** minimal `hardflow_new`+`original`, or the full 7 (adds the l4casadi CUDA build)?
3. **iMF backbone source:** port the FMPCC `imf_diffusion.py` (`imf_official`) engine into HardFlow's `flow_matcher.py`, or bring the official iMF `torch` branch? The former reuses code you've already debugged (K2 fix, adaptive loss); the latter is closer to the paper.
4. **Horizon:** HardFlow uses H16; FMPCC iMF work is H8. Match HardFlow (H16) for the constrained eval, or keep H8 for cross-comparison with existing iMF checkpoints?

---

# Part 3 — Replication RUN plan: SLURM entries + CLONED-env bridge (IMPLEMENTATION SPEC)

> **This section supersedes the env assumptions in Part 1** (§1.2/§1.4) **and the earlier "run in the live FMPCC env" draft of this Part.**
>
> **DECISION (locked):** run the HardFlow repo **UNMODIFIED**, inside a **CLONE of the FMPCC conda env** — never the live/working FMPCC. Phase-1: build `conda create --name <clone> --clone FMPCC`, reconcile the clone (see §3.05), and run everything there. Phase-2 (only *after* the clone works end-to-end): optionally fold the reconciled packages back into the real FMPCC env. **In phase-1, nothing installs into the live FMPCC.** This is a plan/spec, not committed code.
>
> **Why a clone, not the live env:** there is a genuine version-level mismatch between DPCC (FMPCC) and HardFlow — **numpy 1.26 vs 2.0** and **gym 0.26 vs 0.20** (evidence in §3.1). Reconciling those *in place* could break the working FMPCC. A clone gives an isolated, disposable sandbox that starts from the known-good DPCC base (HardFlow is itself DPCC-derived), so we mutate the clone freely and the live env stays pristine.
>
> **User workflow assumed:** HardFlow is `git pull`ed onto the cluster (phase-1: sibling of `FM-PCC`; later: pulled *into* `FM-PCC`). We add **SLURM entries + a thin bridge that activates the CLONE**, save all HardFlow train/eval outputs into `FM-PCC/logs/`, and touch **no HardFlow source**.

## 3.0 Headline answers

- **Do we modify HardFlow's original code? → NO.** Everything needed is a *bridge* (env vars, PYTHONPATH order, one `logs` symlink, env reconciliation done in the clone). HardFlow stays pristine so `git pull` stays clean.
- **Which conda env? → a CLONE of FMPCC (phase-1), never the live FMPCC.** The clone is reconciled to HardFlow's needs (§3.05). All SLURM scripts target the clone via `CONDA_ENV_NAME`.
- **Do we risk the working FMPCC? → NO (phase-1).** No installs/downgrades touch it. Folding into FMPCC is a separate, later, opt-in phase-2 decision.
- **Where do outputs go? →** `FM-PCC/logs/hardflow/…` (gitignored), via a `logs` symlink so both HardFlow's config-driven paths *and* its one hardcoded path land there.
- **How thin are the SLURM entries? →** They set up the bridge and call **HardFlow's own `run_scripts/*.sh`**, so paper hyper-parameters stay baked in and nothing is duplicated or re-tuned.

## 3.05 Env strategy — clone FMPCC, then reconcile the clone

**Build once (login node, not inside an sbatch):**
```
conda create --name hardflow_clone --clone FMPCC      # isolated copy of the known-good DPCC+D3IL env
conda activate hardflow_clone
```

**Reconcile the clone to HardFlow's declared needs (all changes stay in the clone):**
1. `pip install tyro` — the one genuinely-missing package (DPCC has no tyro).
2. **gym**: HardFlow's bundled d3il/avoiding env is written against **gym 0.20.0**, the clone has **gym 0.26.2**. This is the highest-risk item (the 0.20↔0.26 step/reset API break). Try `pip install "gym==0.20.0"` in the clone and confirm `gym.make("avoiding-v0")` + a short rollout works. If the downgrade cascades badly, that's a signal to fall back to a *fresh* env from HardFlow's `environment.yml` (§3.06).
3. **numpy**: HardFlow pins **2.0.2**, the clone has **1.26.4**. Leave 1.26 first and only bump if an import/runtime error demands it (numpy-2-only APIs are unlikely in HardFlow's code). A numpy bump is invasive — treat as last resort.
4. **torch**: inherited from FMPCC (already CUDA-matched to the cluster) — do **not** touch it. HardFlow leaves torch unpinned precisely so it rides the host's build.
5. Add `l4casadi` **only** if you need the `hardflow`/`projection*` methods; the l4casadi-free set doesn't.

**Record the clone's final state** (`conda list > logs/hardflow/clone_env.txt`) so phase-2 (folding into FMPCC) has an exact diff. **Watch-item:** every `pip install` into the clone can perturb *the clone*, which is fine — but log what changed so the phase-2 FMPCC install is a known, minimal set.

## 3.06 Fallback if the clone can't be reconciled

If the gym-0.20 downgrade (or a numpy-2 requirement) breaks the clone, abandon reconciliation and build a **fresh isolated env from HardFlow's own spec** (`conda env create -f environment.yml` + `pip install -r requirements.txt` + a manually-chosen torch+CUDA). Same bridge, same SLURM scripts — only `CONDA_ENV_NAME` changes. This is strictly more isolated but re-fights the unpinned-torch / numpy-2-vs-vintage friction, so it's the fallback, not the default.

## 3.1 Env-gap audit — DPCC/FMPCC pins vs HardFlow pins (measured)

Compared `dpcc/requirements.txt` against HardFlow's `environment.yml`+`requirements.txt`. **This is NOT "DPCC + a few new packages" — two core libraries diverge at a version level, which is the whole reason we clone rather than run in-place.**

| Package | DPCC / FMPCC | HardFlow | Reconcile in the clone? |
|---|---|---|---|
| **numpy** | **1.26.4** | **2.0.2** | ⚠️ major-version boundary; can't coexist. **Leave 1.26 first**, bump only if forced (§3.05.3). |
| **gym** | **0.26.2** (+ gymnasium 0.29.1) | **0.20.0** | ⚠️ API-era break; HardFlow's d3il needs 0.20. **`pip install gym==0.20.0`** in the clone — highest-risk step. |
| mujoco | 2.3.7 | 2.3.7 | ✅ identical |
| scipy | 1.13.1 | 1.13.1 | ✅ identical |
| matplotlib / scikit-learn | 3.9.0 / 1.5.2 | 3.9.4 / 1.6.1 | patch-level, ignore |
| **tyro** | absent | required | `pip install tyro` (the clean additive one) |
| torch | CUDA-matched build | unpinned | inherit from clone; do not touch |
| `d4rl` | present (diffuser lineage) | imported | none; set `D4RL_SUPPRESS_IMPORT_ERROR=1` |
| `l4casadi` (CUDA) | absent | for `hardflow`/`projection*` only | install only if running those methods |
| `tabulate` | — | results notebook only (off-cluster) | not needed on cluster |

**Import-resolution (independent of versions):**
- `hardflow` package → put HardFlow repo on **PYTHONPATH** (see §3.3); do **not** `pip install -e .`.
- `d3il` → HardFlow ships its **own** bundled copy (registers `avoiding-v0`) that clashes with the clone's d3il → **PYTHONPATH order, HardFlow first** (§3.2.2).

## 3.2 Two verified facts that DICTATE the bridge design

1. **Mixed log paths in `eval.py` ⇒ use a symlink, not `--log_folder`.**
   `eval.py` reads/writes checkpoints, train saves, and eval CSVs via `cfg.log_folder` (`:370, :542, :593, :676, :699`), **but the fitted-dynamics path is hardcoded** `os.path.join("logs", cfg.env, "dynamics", "linear_model.npz")` (`:517`). So passing `--log_folder /somewhere` would redirect *most* outputs but **silently miss the dynamics load** → the `--dynamics_constraint` methods would run without the model and **degrade without erroring** (`eval.py:529` proceeds on missing dynamics). **A `logs` symlink neutralises both**: with cwd = HardFlow repo, default `cfg.log_folder="logs"` and the hardcoded `"logs"` both resolve to the same symlinked target.

2. **`import d3il` resolves to HardFlow's bundled package ⇒ PYTHONPATH ordering matters.**
   `d3il/__init__.py` does `from .environments.d3il.envs.gym_avoiding_env.gym_avoiding import envs`, which registers `avoiding-v0` (used at `eval.py:641` `gym.make("avoiding-v0")`). The FMPCC env also has a `d3il`. HardFlow must get **its own**. Putting the HardFlow repo root **first** on `PYTHONPATH` makes `import d3il` and `import hardflow` resolve there (PYTHONPATH wins over site-packages). **Deliberately avoid `pip install -e .`** — `setup.py`'s `find_packages()` would expose HardFlow's `d3il` into the shared env persistently and could shadow FMPCC jobs. PYTHONPATH is job-scoped and reversible.

## 3.3 The bridge — what a sourced helper must do (no HardFlow edits)

A single sourced shell helper (proposed `Slurm_Codes/sbatch/hardflow/_hardflow_common.sh`) that every hardflow sbatch sources after its `#SBATCH` header. Responsibilities, in order:

1. **Resolve paths** (all overridable by env var so phase-1 sibling / later in-FMPCC both work):
   - `FMPCC_ROOT` (default `$HOME/FMPCC`), `REPO=$FMPCC_ROOT/FM-PCC`, `CONDA_DIR`.
   - **`CONDA_ENV_NAME` — MUST default to the CLONE (e.g. `hardflow_clone`), NOT `FMPCC`.** This is the single guard that keeps the live env untouched.
   - `HARDFLOW_REPO` (default `$FMPCC_ROOT/HardFlow`; later set to `$REPO/HardFlow`).
   - `HARDFLOW_LOG_COLLECT` (default `$REPO/logs/hardflow`).
   - Abort with a clear message if `HARDFLOW_REPO` doesn't exist, **and abort if `CONDA_ENV_NAME == FMPCC`** (fail closed so a job can never run against the live env in phase-1).
2. **Pro-logging** (match FMPCC convention): `latest.log` symlink, a JOB START banner (job id/node/GPU + HardFlow git rev), and an EXIT trap printing JOB END.
3. **Activate the CLONE conda env** (`conda activate "$CONDA_ENV_NAME"` — the clone, never the live FMPCC). Assume the clone is already built + reconciled per §3.05 (the sbatch does not create/mutate the env).
4. **PYTHONPATH**: `export PYTHONPATH="$HARDFLOW_REPO${PYTHONPATH:+:$PYTHONPATH}"` (HardFlow first — see §3.2.2).
5. **Sanity check (do NOT auto-install)**: `python -c "import tyro, gym; print(gym.__version__)"` — if this fails, the clone wasn't reconciled (§3.05); abort with a clear message rather than pip-installing at job time (job-time installs into a shared clone race across concurrent jobs).
6. **Headless MuJoCo + GPU/EGL isolation** — the FMPCC standing rule (see `logs_in_develop/SLURM_GPU_IT_WARNING`): export `D4RL_SUPPRESS_IMPORT_ERROR=1`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `MPLBACKEND=agg`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, set `MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}`, and **abort if EGL device ≠ CUDA device** (the GPU-LEAK guard). Keep this even though eval passes `--no-render` (physics is fine without EGL, but rendering — if ever enabled — needs it, and the guard is mandatory here).
7. **logs symlink**: `mkdir -p $HARDFLOW_LOG_COLLECT`; if `$HARDFLOW_REPO/logs` is already a symlink, refresh it; if it's a **real** dir, abort with a message (don't clobber); else create the symlink. (HardFlow gitignores `logs/`, so a fresh pull won't have one.)
8. **`cd "$HARDFLOW_REPO"`** so all relative `./logs`, `run/`, `d3il/`, `run_scripts/` resolve.

## 3.4 SLURM entries — proposed files (thin wrappers over HardFlow's own scripts)

Create under `Slurm_Codes/sbatch/hardflow/` (sibling style to `sbatch/iMF/`). Submit via `Slurm_Codes/submit.sh` (not raw `sbatch`), per repo convention.

| File | `#SBATCH` (guide) | Body |
|---|---|---|
| `_hardflow_common.sh` | *(sourced, no header)* | the §3.3 bridge |
| `train_hardflow.sh` | 1 GPU, 32G, ~24h, `gpu-1-student` | `source _hardflow_common.sh` → `bash run_scripts/train.sh` → list resulting `logs/avoiding-v0/flow/H16_1e6steps/`. **Skip if downloading the released `.pth`.** |
| `eval_hardflow.sh` | 1 GPU, 32G, ~12h | `source _hardflow_common.sh` → **guard**: if `logs/avoiding-v0/dynamics/linear_model.npz` missing, run `python run/fit_dynamics.py` → then for each method in `METHODS` (default `"hardflow_new original"`) run `bash run_scripts/eval_${method}.sh`. |
| `hardflow_pipeline.sh` | orchestrator | chain train (or checkpoint-present check) → eval, mirroring `sbatch/iMF/imf_pipeline.sh`. Optional. |

**Why reuse `run_scripts/*.sh` verbatim:** they hardcode the paper operating point (H16, `ode_t_steps=10`, `random_repeat=50`, `controller=rh`, `replan_steps=8`, `constraint=novel`, cost scales). Calling them means the SLURM layer never re-specifies science — it only supplies the cluster bridge. Note their `--device cuda:0` is correct under `--gres=gpu:1` (CUDA_VISIBLE_DEVICES remaps the allocated GPU to index 0).

**`METHODS` knob:** default to the l4casadi-free set `hardflow_new original`. Add `oc_flow gradient_guidance` freely (no l4casadi). Only add `hardflow projection projection_relaxed` after an l4casadi CUDA build exists — otherwise those eval scripts will import-error.

## 3.5 Outputs & the metric table

- All artifacts land in `FM-PCC/logs/hardflow/avoiding-v0/{flow,dynamics,eval}/…` via the symlink; `eval/<exp>/trajectories.csv` per method.
- The paper table is produced **off-cluster** by HardFlow's `notebooks/collect_results.ipynb` (needs `pandas`+`tabulate`), pointed at the collected `eval/` dir. A cluster-side quick summary (pandas over the CSVs) is optional and can be added to `eval_hardflow.sh` later.
- Report (paper Sec VII.A): safety/success rate, constraint-violation rate, steps/path-cost, compute time, and NFE budget.

## 3.6 First-run checklist (order matters)

0. **Build + reconcile the clone** (§3.05), on the login node: `conda create --name hardflow_clone --clone FMPCC`, then `pip install tyro`, `pip install gym==0.20.0`, verify `import tyro, gym; gym.__version__ == 0.20.0`. Record `conda list`. **Live FMPCC is never touched.**
1. `git pull` HardFlow onto the cluster; set `HARDFLOW_REPO` if not the default. Ensure the sbatch's `CONDA_ENV_NAME` points at the clone.
2. Submit `eval_hardflow.sh` with `METHODS="original"` as a **smoke test** — cheapest path that exercises the bridge (clone activation, env import, `avoiding-v0` registration via HardFlow's d3il, logs symlink). Requires a checkpoint present (download, or run `train_hardflow.sh` first).
3. Confirm `logs/hardflow/...` populated and the run didn't silently skip dynamics.
4. Scale up: full `METHODS`, then `train_hardflow.sh` if training from scratch.
5. **Phase-2 (opt-in, only after the clone works):** replay the recorded clone diff (`tyro`, `gym==0.20.0`, …) into the real FMPCC env — a separate, deliberate decision, not part of this run.

**Watch-items:** (a) **gym downgraded to 0.20.0 in the clone** and the avoiding rollout actually steps (highest-risk, §3.05.2); (b) numpy stayed 1.26 and nothing demanded 2.0 (§3.05.3); (c) `import d3il` resolved to HardFlow's copy, not the clone's (print `d3il.__file__` if unsure); (d) dynamics `.npz` exists before any `--dynamics_constraint` method.

## 3.7 Explicit non-goals / guardrails for the implementing agent

- **Do NOT install/downgrade anything into the live FMPCC env in phase-1.** All env changes happen in the CLONE. The bridge must **abort if `CONDA_ENV_NAME == FMPCC`** (§3.3.1).
- **Do NOT edit any file under the HardFlow repo.** If something seems to require it, stop and surface it — the whole design is predicated on an unmodified upstream.
- **Do NOT `pip install -e .`** the HardFlow repo into the clone (d3il-shadowing risk, §3.2.2). PYTHONPATH only.
- **Do NOT auto-install packages at sbatch job time** — reconcile the clone once, up front (§3.05); job-time installs race across concurrent jobs and mask real env problems.
- **Do NOT break the GPU/EGL isolation guard** (§3.3.6). It is a hard cluster rule.
- **Do NOT redirect outputs with `--log_folder`** (misses the hardcoded dynamics path, §3.2.1). Use the symlink.
- Keep the sbatch bodies thin — orchestration only; all science stays in HardFlow's `run_scripts/*.sh`.
