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

# Part 3 — Replication RUN plan: SLURM entries + FMPCC-env bridge (IMPLEMENTATION SPEC)

> **This section supersedes the "build a separate `hardflow` conda env" assumption in Part 1** (§1.2/§1.4). The chosen approach is to run the HardFlow repo **UNMODIFIED, inside the existing FMPCC conda env** (which is a 1:1 DPCC + D3IL env). This is the spec another agent should implement — it is a plan, not committed code.
>
> **User workflow assumed:** the HardFlow repo is `git pull`ed onto the cluster (phase-1: as a sibling of `FM-PCC`; phase-2: pulled *into* `FM-PCC`). The FMPCC conda env already exists. We add **SLURM entries + a thin bridge**, save all HardFlow train/eval outputs into `FM-PCC/logs/`, and touch **no HardFlow source**.

## 3.0 Headline answers

- **Do we modify HardFlow's original code? → NO.** Everything needed is a *bridge* (env vars, PYTHONPATH order, one `logs` symlink, one pip shim). HardFlow stays pristine so `git pull` stays clean.
- **Do we build a separate `hardflow` conda env? → NO.** Run in the existing FMPCC env. Only **one** package is missing (`tyro`); `d4rl` import flakiness is tolerated by a suppress flag.
- **Where do outputs go? →** `FM-PCC/logs/hardflow/…` (gitignored), via a `logs` symlink so both HardFlow's config-driven paths *and* its one hardcoded path land there.
- **How thin are the SLURM entries? →** They just set up the bridge and then call **HardFlow's own `run_scripts/*.sh`**, so the paper hyper-parameters stay baked in and nothing is duplicated or re-tuned.

## 3.1 Env-gap audit — FMPCC (DPCC+D3IL) env vs what HardFlow imports

Verified by grepping HardFlow's `run/*.py` and `hardflow/**/*.py` imports against a DPCC-class env:

| HardFlow needs | In FMPCC env? | Action |
|---|---|---|
| `torch`, `numpy`, `gym`, `einops`, `tqdm`, `matplotlib`, `scipy`, `yaml`, `pandas` | ✅ yes (DPCC deps) | none |
| `d4rl` (`import d4rl` in `datasets/d4rl.py`) | ✅ usually (diffuser lineage) | none; also set `D4RL_SUPPRESS_IMPORT_ERROR=1` so a flaky import can't kill the job |
| **`tyro`** (arg parser in `run/eval.py`, `run/train.py`) | ❌ likely missing | **one** `pip install tyro` into the FMPCC env (benign; doesn't affect FMPCC jobs) |
| `hardflow` package | via checkout | put HardFlow repo on **PYTHONPATH** (see §3.3) — do **not** `pip install -e .` |
| `d3il` (HardFlow's **own bundled** copy, registers `avoiding-v0`) | conflicts w/ FMPCC's d3il | **PYTHONPATH order**, HardFlow first (see §3.3) |
| `l4casadi` (CUDA) | ❌ | only for `hardflow`/`projection*`; skip for `hardflow_new`+`original`+`oc_flow`+`gradient_guidance` |
| `tabulate` | for the results notebook only (runs off-cluster) | not needed on the cluster |

> **`numpy` note:** HardFlow *pins* `numpy==2.0.2`; the FMPCC env is likely numpy-1.x. HardFlow's runtime code is very unlikely to use numpy-2-only APIs, so 1.x should work — but this is the first thing to check if an import errors at first run.

## 3.2 Two verified facts that DICTATE the bridge design

1. **Mixed log paths in `eval.py` ⇒ use a symlink, not `--log_folder`.**
   `eval.py` reads/writes checkpoints, train saves, and eval CSVs via `cfg.log_folder` (`:370, :542, :593, :676, :699`), **but the fitted-dynamics path is hardcoded** `os.path.join("logs", cfg.env, "dynamics", "linear_model.npz")` (`:517`). So passing `--log_folder /somewhere` would redirect *most* outputs but **silently miss the dynamics load** → the `--dynamics_constraint` methods would run without the model and **degrade without erroring** (`eval.py:529` proceeds on missing dynamics). **A `logs` symlink neutralises both**: with cwd = HardFlow repo, default `cfg.log_folder="logs"` and the hardcoded `"logs"` both resolve to the same symlinked target.

2. **`import d3il` resolves to HardFlow's bundled package ⇒ PYTHONPATH ordering matters.**
   `d3il/__init__.py` does `from .environments.d3il.envs.gym_avoiding_env.gym_avoiding import envs`, which registers `avoiding-v0` (used at `eval.py:641` `gym.make("avoiding-v0")`). The FMPCC env also has a `d3il`. HardFlow must get **its own**. Putting the HardFlow repo root **first** on `PYTHONPATH` makes `import d3il` and `import hardflow` resolve there (PYTHONPATH wins over site-packages). **Deliberately avoid `pip install -e .`** — `setup.py`'s `find_packages()` would expose HardFlow's `d3il` into the shared env persistently and could shadow FMPCC jobs. PYTHONPATH is job-scoped and reversible.

## 3.3 The bridge — what a sourced helper must do (no HardFlow edits)

A single sourced shell helper (proposed `Slurm_Codes/sbatch/hardflow/_hardflow_common.sh`) that every hardflow sbatch sources after its `#SBATCH` header. Responsibilities, in order:

1. **Resolve paths** (all overridable by env var so phase-1 sibling / phase-2 in-FMPCC both work):
   - `FMPCC_ROOT` (default `$HOME/FMPCC`), `REPO=$FMPCC_ROOT/FM-PCC`, `CONDA_DIR`, `CONDA_ENV_NAME=FMPCC`.
   - `HARDFLOW_REPO` (default `$FMPCC_ROOT/HardFlow`; phase-2 set to `$REPO/HardFlow`).
   - `HARDFLOW_LOG_COLLECT` (default `$REPO/logs/hardflow`).
   - Abort with a clear message if `HARDFLOW_REPO` doesn't exist.
2. **Pro-logging** (match FMPCC convention): `latest.log` symlink, a JOB START banner (job id/node/GPU + HardFlow git rev), and an EXIT trap printing JOB END.
3. **Activate the FMPCC conda env** (not a hardflow env).
4. **PYTHONPATH**: `export PYTHONPATH="$HARDFLOW_REPO${PYTHONPATH:+:$PYTHONPATH}"` (HardFlow first — see §3.2.2).
5. **pip shim**: `python -c "import tyro" || pip install --quiet tyro`.
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

1. `git pull` HardFlow onto the cluster; set `HARDFLOW_REPO` if not the default.
2. Submit `eval_hardflow.sh` with `METHODS="original"` as a **smoke test** — cheapest path that exercises the bridge (env import, `avoiding-v0` registration via HardFlow's d3il, logs symlink). Requires a checkpoint present (download, or run `train_hardflow.sh` first).
3. Confirm `logs/hardflow/...` populated and the run didn't silently skip dynamics.
4. Scale up: full `METHODS`, then `train_hardflow.sh` if training from scratch.

**Watch-items:** (a) numpy 1.x vs HardFlow's pin (§3.1); (b) `tyro` install succeeded; (c) `import d3il` resolved to HardFlow's copy, not FMPCC's (print `d3il.__file__` if unsure); (d) dynamics `.npz` exists before any `--dynamics_constraint` method.

## 3.7 Explicit non-goals / guardrails for the implementing agent

- **Do NOT edit any file under the HardFlow repo.** If something seems to require it, stop and surface it — the whole design is predicated on an unmodified upstream.
- **Do NOT `pip install -e .`** the HardFlow repo into the FMPCC env (d3il-shadowing risk, §3.2.2). PYTHONPATH only.
- **Do NOT break the GPU/EGL isolation guard** (§3.3.6). It is a hard cluster rule.
- **Do NOT redirect outputs with `--log_folder`** (misses the hardcoded dynamics path, §3.2.1). Use the symlink.
- Keep the sbatch bodies thin — orchestration only; all science stays in HardFlow's `run_scripts/*.sh`.
