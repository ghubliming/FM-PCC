# Guide — Remote (cluster) setup for HardFlow replication

**Goal:** get HardFlow running on the cluster in an **isolated clone** of the FMPCC env, with zero risk to the working FMPCC env and zero edits to HardFlow's source. After this, you only submit the sbatch entries in `Slurm_Codes/sbatch/hardflow/`.

**Companion:** implementation spec `EVALUATION_hardflow_readiness_and_iMF_swap.md` §Part 3; changelog `CHANGELOG_hardflow_slurm_bridge.md` (this folder).

---

## 0. Your cluster state (from `conda env list`)

```
base        /u/home/llim/miniconda3          <- conda base
FMPCC       /u/home/llim/miniconda3/envs/FMPCC     <- the WORKING env — DO NOT touch
FMPCC_mjx   /u/home/llim/miniconda3/envs/FMPCC_mjx
```

Assumed layout (adjust if yours differs):
- `HOME` = `/u/home/llim`
- FMPCC workspace root = `/u/home/llim/FMPCC`
- FM-PCC repo = `/u/home/llim/FMPCC/FM-PCC`  ← the sbatch scripts live here
- **HardFlow = `/u/home/llim/FMPCC/FM-PCC/HardFlow`  ← VENDORED INSIDE FM-PCC** (copied code, tracked by FM-PCC git; arrives on the cluster via `git pull`)

> These match the **hardcoded** values in `Slurm_Codes/sbatch/hardflow/_hardflow_common.sh` (`FMPCC_ROOT=$HOME/FMPCC`, `REPO=$FMPCC_ROOT/FM-PCC`, `HARDFLOW_REPO=$REPO/HardFlow`, `CONDA_DIR=$HOME/miniconda3`, `CONDA_ENV_NAME=hardflow_clone`). Only `$HOME` is dynamic. **If your layout differs, edit those literal lines in `_hardflow_common.sh`** — they are intentionally not env-var overrides (so submitting from an activated shell can't perturb them).

> **📌 Reproducing this from scratch (another person / cluster)?** Read this first:
> - **Paths below are one user's** (`/u/home/llim/…`). Substitute your own `$HOME` / repo path everywhere, and if it's not `$HOME/FMPCC/FM-PCC`, edit the 5 literal lines in `_hardflow_common.sh` (§0 note above).
> - **One-time setup** (do once): §1 get code, §2 create clone env, §3 reconcile it, §4 verify. **Per-run** (each experiment): §5–§6 submit jobs. Don't rebuild the env every run.
> - **You do NOT need the separate `hardflow` conda env from HardFlow's README.** We use a reconciled *clone of FMPCC* instead (§2–§3). Ignore HardFlow's `environment.yml` unless the clone route fails (then see spec §3.06 fallback).

---

## 1. Get HardFlow onto the cluster — NO git clone needed

HardFlow (the `d3il` / avoiding branch) is **vendored into FM-PCC as plain copied code** at `HardFlow/` — only the working tree (~62 MB: `hardflow/`, `run/`, `run_scripts/`, and the `d3il/` sim + 96 avoiding demos), **no `.git`, no other branches**. It reaches the cluster with your normal FM-PCC sync:

```bash
# on the cluster, in the FM-PCC repo:
cd /u/home/llim/FMPCC/FM-PCC
git pull
ls HardFlow/run_scripts/   # sanity: train.sh, eval_hardflow_new.sh, eval_original.sh, ...
ls HardFlow/d3il/environments/dataset/data/avoiding/data | wc -l   # expect 96
```

> **Why vendored, not cloned:** the upstream `.git` history is ~469 MB (bloated); only the 62 MB working tree is real code. Vendoring keeps HardFlow versioned *with* FM-PCC and avoids a second checkout. The copy's `.gitignore` was edited so FM-PCC **tracks** the `d3il/` sim + data (upstream ignored them). Data is bundled — no download.

---

## 2. Create the isolated env — CLONE FMPCC

This copies the known-good DPCC+D3IL env; the live `FMPCC` is never modified.

```bash
conda create --name hardflow_clone --clone FMPCC
conda activate hardflow_clone
```

> **Name it exactly `hardflow_clone`** — the sbatch scripts **hardcode** `CONDA_ENV_NAME="hardflow_clone"` (mjx-style, so an FMPCC-activated submit shell can't perturb it). If you must use a different name, edit that one line in `Slurm_Codes/sbatch/hardflow/_hardflow_common.sh`.
> A clone needs a few GB of disk and a few minutes.

---

## 3. Reconcile the clone (minimal, targeted — NOT `requirements.txt`)

Only two functional changes are needed vs the DPCC base: add `tyro`, downgrade `gym` to `0.20.0` (HardFlow's d3il expects the old gym API). Old-gym installs need old build tools, so pin those first (mirrors HardFlow's `environment.yml`: `setuptools=65.*`, `pip=22.*`).

```bash
conda activate hardflow_clone

# 3a. build-tool pins so gym 0.20 installs cleanly
pip install "setuptools==65.5.0" "pip==22.3.1" "wheel<0.40"

# 3b. downgrade gym (highest-risk step)
pip install "gym==0.20.0"

# 3c. the one genuinely-missing package
pip install tyro
```

**✅ Expected output (this is SUCCESS, not failure):**
- 3a: `Successfully installed pip-22.3.1 setuptools-65.5.0 wheel-0.38.4`
- 3b: `Attempting uninstall: gym … Uninstalling gym-0.26.2 … Successfully installed gym-0.20.0` (gym builds from an sdist — the `Building wheel for gym … done` step is normal).
- 3c: `Successfully installed … tyro-…`

**⚠️ Harmless warnings you WILL see — ignore them:**
```
ERROR: pip's dependency resolver ... grpcio-tools 1.81.1 requires protobuf<7...,>=6.33.5, but you have protobuf 5.29.6 ...
grpcio-tools 1.81.1 requires setuptools>=77.0.1, but you have setuptools 65.5.0 ...
WARNING: There was an error checking the latest version of pip.
```
`grpcio-tools` is a leftover package from the FMPCC clone that **HardFlow never imports**; the setuptools/protobuf "conflicts" don't affect any HardFlow run. This is a disposable clone — these are expected and safe.

**Do NOT run `pip install -r requirements.txt`** — it would force `numpy==2.0.2`, `mujoco==2.3.7`, etc. onto the clone and may break torch. Leave **numpy at 1.26** (the clone's value); only touch it if a run actually errors demanding numpy 2 (unlikely).

**Do NOT `pip install -e .`** — the bridge puts HardFlow on `PYTHONPATH` instead (so `import d3il` resolves to HardFlow's bundled copy without polluting the env).

**l4casadi:** skip unless you plan to run the `hardflow`/`projection*` methods. The default method set (`hardflow_new original`) needs none.

Record the final state for the future phase-2 (folding into real FMPCC):
```bash
conda list > /u/home/llim/FMPCC/FM-PCC/logs/hardflow/clone_env.txt   # mkdir -p the dir first
```

---

## 4. Verify (login-node safe — imports only, no MuJoCo/GPU)

Use the **absolute** HardFlow path (not `$PWD`) so your cwd can't trip you up:

```bash
conda activate hardflow_clone
export HARDFLOW_REPO=/u/home/llim/FMPCC/FM-PCC/HardFlow
ls "$HARDFLOW_REPO/d3il/__init__.py"          # must exist (confirms the vendored copy synced)
export PYTHONPATH="$HARDFLOW_REPO:$PYTHONPATH"

python -c "import tyro, gym, torch, numpy; print('gym', gym.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"
# expect (confirmed on this cluster): gym 0.20.0 | numpy 1.26.4 | torch 2.2.2+cu121

python -c "import d3il; print('d3il from', d3il.__file__)"
# expect a path UNDER /u/home/llim/FMPCC/FM-PCC/HardFlow/d3il  (NOT the FMPCC repo's own d3il)
```

- `gym 0.20.0` and a HardFlow-path `d3il.__file__` = reconciliation worked.
- **The `$PWD` trap (we hit this):** if you `export PYTHONPATH="$PWD:…"` from your home dir instead of the HardFlow repo, `import d3il` fails with `ModuleNotFoundError`. Always use the **absolute** `$HARDFLOW_REPO`, as above.
- `ModuleNotFoundError: d3il` = a **path** problem (git pull hasn't landed `HardFlow/`, or it's not on `PYTHONPATH`) — **not** an env problem. Fix the path.
- **Do NOT** run `gym.make("avoiding-v0")` on the login node — it spins up MuJoCo/EGL and needs a GPU node. That happens inside the sbatch (which sets the EGL guard).
- **Submitting from an activated FMPCC shell is fine.** The sbatch hardcodes `hardflow_clone` and resets `PYTHONPATH` fresh, so nothing your login shell has active can leak into the job.

---

## 5. Get a checkpoint (pick one)

- **Train on the cluster** (Path A):
  ```bash
  cd /u/home/llim/FMPCC/FM-PCC
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_hardflow.sh
  ```
  → writes `logs/hardflow/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth` (~1e6 steps, long).
- **Download** (Path B): grab the released `.pth` (HardFlow README Google-Drive link) and place it at
  `/u/home/llim/FMPCC/FM-PCC/logs/hardflow/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth`.

---

## 6. Smoke test, then full eval

```bash
cd /u/home/llim/FMPCC/FM-PCC

# cheapest path first — exercises the whole bridge with one method
METHODS="original" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_hardflow.sh

# then the l4casadi-free pair (default)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_hardflow.sh

# or the whole chain (train-if-needed -> fit_dynamics -> eval)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/hardflow_pipeline.sh
```

Watch the log (`Slurm_Codes/logs/latest.log`) for: `tyro OK, gym==0.20.0`, the `GPU-CHECK` line, `logs -> …/FM-PCC/logs/hardflow`, and no `GPU-LEAK`/abort.

Results: `FM-PCC/logs/hardflow/avoiding-v0/eval/<exp>/trajectories.csv`. Aggregate off-cluster with HardFlow's `notebooks/collect_results.ipynb`.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `grpcio-tools ... protobuf/setuptools incompatible` during §3 pip installs | leftover FMPCC package, unused by HardFlow | **harmless — ignore** (see §3 note) |
| `gym==0.20.0` install fails (`extras_require`/`error in setup command`) | new setuptools/pip vs old gym | ensure step 3a ran (setuptools 65 / pip 22) **before** 3b |
| gym import error mentioning `importlib_metadata` entry points | importlib-metadata ≥5 vs old gym | `pip install "importlib-metadata<5"` in the clone |
| `python -c "import gym"` still shows 0.26 after §3 | wrong env active | `conda activate hardflow_clone` first; re-run 3b |
| `ModuleNotFoundError: d3il` on the login node | `HardFlow/` not pulled yet, or wrong/`$PWD`-based PYTHONPATH | `git pull` in FM-PCC; `export PYTHONPATH="$HARDFLOW_REPO"` with the **absolute** path (§4) |
| `d3il.__file__` points into `.../envs/hardflow_clone/...` or the FMPCC repo's own d3il | wrong d3il on path | HardFlow must be FIRST on `PYTHONPATH` (the sbatch resets it fresh; on the login node set it as in §4) |
| Job can't find the env / activates the wrong one | your clone isn't named `hardflow_clone` | rename it, or edit the hardcoded `CONDA_ENV_NAME="hardflow_clone"` line in `_hardflow_common.sh` |
| Job aborts `HardFlow/logs is a real dir` | a real `logs/` exists in the checkout | `mv HardFlow/logs HardFlow/logs.bak` so the bridge can symlink it |
| Job aborts `clone '…' is missing tyro/gym` | env not reconciled (§3 skipped) | run §3 in the clone, then resubmit |
| `hardflow_new` runs but results look off / no constraints | dynamics model missing | eval script auto-runs `fit_dynamics.py`; confirm `logs/.../dynamics/linear_model.npz` exists |
| import error demanding numpy 2 | rare HardFlow numpy-2 API use | last resort: `pip install "numpy==2.0.2"` in the clone (may pull a matching torch) |
| `eval_hardflow.sh SKIP '<method>'` | l4casadi method without l4casadi | build l4casadi, or stick to `hardflow_new original oc_flow gradient_guidance` |

---

## 8. What NOT to do (isolation guarantees)

- ❌ Don't `conda activate FMPCC` and install into it — phase-1 keeps the live env pristine. Everything goes in `hardflow_clone`.
- ❌ Don't `pip install -r requirements.txt` or `pip install -e .` into the clone (§3).
- ❌ Don't edit any file under `HardFlow/`.
- ✅ The clone is disposable — if reconciliation gets messy, `conda env remove -n hardflow_clone` and either re-clone or build a fresh env from HardFlow's `environment.yml` (spec §3.06).
