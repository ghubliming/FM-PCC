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
- HardFlow (to be cloned) = `/u/home/llim/FMPCC/HardFlow`  ← **sibling of FM-PCC**

> These match the bridge defaults (`FMPCC_ROOT=$HOME/FMPCC`, `HARDFLOW_REPO=$FMPCC_ROOT/HardFlow`, `CONDA_DIR=$HOME/miniconda3`). If your FM-PCC is elsewhere, export `FMPCC_ROOT`/`REPO`/`HARDFLOW_REPO` before submitting.

---

## 1. Clone HardFlow (sibling of FM-PCC, NOT inside it)

```bash
cd /u/home/llim/FMPCC
git clone <HARDFLOW_REMOTE_URL> HardFlow
cd HardFlow
git checkout d3il          # the branch with the avoiding/robotic-manipulation experiment
ls run_scripts/            # sanity: train.sh, eval_hardflow_new.sh, eval_original.sh, ...
```

Data is bundled (no download): `d3il/environments/dataset/data/avoiding/…`.

---

## 2. Create the isolated env — CLONE FMPCC

This copies the known-good DPCC+D3IL env; the live `FMPCC` is never modified.

```bash
conda create --name hardflow_clone --clone FMPCC
conda activate hardflow_clone
```

> Name it `hardflow_clone` to match the sbatch default `CONDA_ENV_NAME`. If you pick another name, pass `CONDA_ENV_NAME=<name>` at submit time.
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

**Do NOT run `pip install -r requirements.txt`** — it would force `numpy==2.0.2`, `mujoco==2.3.7`, etc. onto the clone and may break torch. Leave **numpy at 1.26** (the clone's value); only touch it if a run actually errors demanding numpy 2 (unlikely).

**Do NOT `pip install -e .`** — the bridge puts HardFlow on `PYTHONPATH` instead (so `import d3il` resolves to HardFlow's bundled copy without polluting the env).

**l4casadi:** skip unless you plan to run the `hardflow`/`projection*` methods. The default method set (`hardflow_new original`) needs none.

Record the final state for the future phase-2 (folding into real FMPCC):
```bash
conda list > /u/home/llim/FMPCC/FM-PCC/logs/hardflow/clone_env.txt   # mkdir -p the dir first
```

---

## 4. Verify (login-node safe — imports only, no MuJoCo/GPU)

```bash
conda activate hardflow_clone
cd /u/home/llim/FMPCC/HardFlow
export PYTHONPATH="$PWD:$PYTHONPATH"

python -c "import tyro, gym, torch, numpy; print('gym', gym.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"
# expect: gym 0.20.0 | numpy 1.26.x | torch <your FMPCC build>

python -c "import d3il; print('d3il from', d3il.__file__)"
# expect a path UNDER /u/home/llim/FMPCC/HardFlow/d3il  (NOT the FMPCC one)
```

- `gym 0.20.0` and a HardFlow-path `d3il.__file__` = reconciliation worked.
- **Do NOT** run `gym.make("avoiding-v0")` on the login node — it spins up MuJoCo/EGL and needs a GPU node. That happens inside the sbatch (which sets the EGL guard).

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
| `gym==0.20.0` install fails (`extras_require`/`error in setup command`) | new setuptools/pip vs old gym | ensure step 3a ran (setuptools 65 / pip 22) before 3b |
| gym import error mentioning `importlib_metadata` entry points | importlib-metadata ≥5 vs old gym | `pip install "importlib-metadata<5"` in the clone |
| `d3il.__file__` points into `.../envs/hardflow_clone/...` or the FMPCC repo | wrong d3il on path | ensure HardFlow repo is FIRST on `PYTHONPATH` (bridge does this; on the login node export it as in step 4) |
| Job aborts `CONDA_ENV_NAME=FMPCC is forbidden` | you targeted the live env | set `CONDA_ENV_NAME=hardflow_clone` (or leave default) |
| Job aborts `HardFlow/logs is a real dir` | a real `logs/` exists in the checkout | `mv HardFlow/logs HardFlow/logs.bak` so the bridge can symlink it |
| `hardflow_new` runs but results look off / no constraints | dynamics model missing | eval script auto-runs `fit_dynamics.py`; confirm `logs/.../dynamics/linear_model.npz` exists |
| import error demanding numpy 2 | rare HardFlow numpy-2 API use | last resort: `pip install "numpy==2.0.2"` in the clone (may pull a matching torch) |
| `eval_hardflow.sh SKIP '<method>'` | l4casadi method without l4casadi | build l4casadi, or stick to `hardflow_new original oc_flow gradient_guidance` |

---

## 8. What NOT to do (isolation guarantees)

- ❌ Don't `conda activate FMPCC` and install into it — phase-1 keeps the live env pristine. Everything goes in `hardflow_clone`.
- ❌ Don't `pip install -r requirements.txt` or `pip install -e .` into the clone (§3).
- ❌ Don't edit any file under `HardFlow/`.
- ✅ The clone is disposable — if reconciliation gets messy, `conda env remove -n hardflow_clone` and either re-clone or build a fresh env from HardFlow's `environment.yml` (spec §3.06).
