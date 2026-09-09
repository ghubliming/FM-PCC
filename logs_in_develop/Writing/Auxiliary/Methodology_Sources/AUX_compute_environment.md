# AUX — compute environment: cluster, node, GPU, software stack

**Created:** 2026-09-09 · **Updated:** 2026-09-09 (`sinfo`/`scontrol` + `lscpu` collected; hardware section complete)
**Type:** source map + fact sheet, not results
**Thesis home:** `sec:setup:implementation` (2–4 sentences), `app:repro` (the full table) ·
**TARGET** §5.3 (provenance is a deliverable), §6.5 (the `budget_ms` caveat)
**Primary evidence:** `sinfo`/`scontrol`/`lscpu` output of 2026-09-09 (§3.1–3.3, verbatim; job 25567) ·
`Slurm_Codes/logs/2026-04-29/verify_env_job_19741_153833.log` (the package check, §3.5) ·
every sbatch log header in `Slurm_Codes/logs/<date>/*.log` (`NODE:` + `GPU INFO:`) ·
`Slurm_Codes/sbatch/templates/2026_04_30_job_template.sh` · `Slurm_Codes/submit.sh` ·
`requirements.txt` · `Slurm_Codes/install_env_from_colab_ipynb_style/remote_setup_guide.md`

> 🔴 **Nothing here is a result.** Hardware is apparatus. It matters for exactly two things in the
> thesis: (a) reproducibility — a reader must be able to say *on what* the numbers were produced;
> (b) every wall-clock number (`avg_time`, `fm_ms`, `proj_ms`, the 33 Hz discussion) is meaningless
> without the machine it was measured on.

---

## 1. What the thesis must say

Three sentences in `sec:setup:implementation`, everything else in `app:repro`:

> All experiments were run on a **single shared GPU node** (`i6-gpu-1`: 2 × AMD EPYC 7282 16-core
> CPUs = 32 cores / 64 threads, 252 GiB RAM, 8 × NVIDIA RTX A5000 24 GB) of the TUM I6 chair cluster,
> scheduled with SLURM 21.08.5 on the `gpu-1-student` partition (24 h wall-clock limit). Each job
> was allocated **one** GPU — training and evaluation are single-GPU throughout; no run is
> distributed or multi-GPU. Wall-clock timings reported in Chapter *N* are therefore per-GPU on
> that device, on a node shared with other users, and are indicative rather than a hardware
> benchmark.

Two things must go with it, both already standing rules of the project:

- 🔴 **The shared-node caveat.** One node serves the whole chair's student and employee GPU queues
  (`gpu-1-student` and `gpu-1-employee` both map to `i6-gpu-1` and to the same 64 CPUs). A job sees
  one GPU but shares CPU, memory bandwidth, and I/O — the `scontrol` snapshot in §3.2 was taken with
  6 of 8 GPUs and 42 of 64 CPUs already allocated to other jobs. Timing numbers therefore carry
  contention noise. Do not present `avg_time` as a hardware benchmark, and never A/B two methods
  measured in different contention regimes without saying so.
- 🔴 **`budget_ms` / 33 Hz is a data-rate artefact, not a real-time pass/fail criterion** — see
  `NOTES_open_questions.md` and TARGET §6.5. Gen11 E7 measured ~85 ms FM inference against a
  30.3 ms nominal budget *on this hardware*; that is a statement about one A5000 under a shared
  research load, not about the deployability of the method. Report it that way or not at all.

---

## 2. Known now, from repo evidence (no cluster access needed)

| fact | value | evidence |
|---|---|---|
| scheduler | SLURM **21.08.5** | §3.2 |
| partition | `gpu-1-student` (max wall-clock **1-00:00:00**) | §3.1, §3.2, all job scripts |
| node (all runs) | `i6-gpu-1` — the **only** node in the partition | `NODE:` header — 777/777 logs that print one; §3.1 |
| GPUs on node | 8 × **NVIDIA RTX A5000**, 24564 MiB (24 GB) each, `gres=gpu:RTXA5000:8(S:0-1)` | `GPU INFO:` header block, e.g. `logs/2026-07-24/14_43_11_hffm_gates_23796.log:8-15`; §3.1 |
| NVIDIA driver | **530.41.03**, unchanged 2026-04-29 → 2026-09-02 | earliest and latest log headers |
| GPUs per job | **1** (`--gres=gpu:1`); GPU pinned via `CUDA_VISIBLE_DEVICES`, EGL pinned to the same index via `MUJOCO_EGL_DEVICE_ID` (mismatch = hard abort) | `templates/2026_04_30_job_template.sh` §3; `verify_env_job.sh` |
| node CPU | 2 × **AMD EPYC 7282** (Zen 2 "Rome"), 16 cores each → **32 physical cores / 64 threads**, 1.5–2.8 GHz, 128 MiB L3 | §3.3 |
| node RAM | **257 782 MB ≈ 251.7 GiB** | §3.1, §3.2 |
| node OS | Linux **5.15.0-94-generic**, Ubuntu SMP (kernel built 2024-01-09) | §3.2 |
| CPUs per job (typical) | 8 (training / rendering), 4, 1 (aggregation, plotting) | `--cpus-per-task=` census: 8×63, 1×32, 4×22, 2×5, 16×1 |
| RAM per job (typical) | 32 G (training), 16 G, 2–8 G (analysis) | `--mem=` census: 32G×60, 2G×30, 16G×21 |
| wall-clock cap | 24 h, partition-enforced (project rule: `--time` = 2× expected) | §3.1; job scripts |
| env manager | miniconda at `$HOME/miniconda3`, env **`FMPCC`**, **Python 3.10** | template §2; `remote_setup_guide.md:63` |
| repo path on cluster | `$HOME/FMPCC/FM-PCC`, `PYTHONPATH` += `repo`, `repo/d3il`, `gym_avoiding_env` | template §2; `verify_env_job.sh` |
| headless graphics | MuJoCo **EGL** (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `MPLBACKEND=agg`) — no X server | template §3 |
| job entry point | `./Slurm_Codes/submit.sh <script>`; logs to `Slurm_Codes/logs/<date>/HH_MM_SS_<job>_<id>.log` | `submit.sh` |
| provenance in every log | `JOB ID`, `NODE`, `GPU INFO`, `GIT REV` printed in the header | log headers |

**⚠️ Still missing:** only the confirmed **cuDNN** version and whether the conda env drifted after the
2026-04-29 snapshot. One CPU-only `srun` closes it — §5.3. **All hardware facts are now settled.**

---

## 3. Raw command output — the citation

### 3.1 `sinfo -e` — partition + node inventory  *(login node, 2026-09-09)*

```text
NODELIST             PARTITION      NODES  CPUS    MEMORY                           GRES STATE        TIMELIMIT
i6-gpu-1             gpu-1-student      1    64    257782          gpu:RTXA5000:8(S:0-1) mixed        1-00:00:00
i6-gpu-1             gpu-1-employee     1    64    257782          gpu:RTXA5000:8(S:0-1) mixed        1-00:00:00
```

**Read-out.** The chair's GPU capacity visible to this project is **one node**. `gpu-1-student` and
`gpu-1-employee` are two queues onto the *same* hardware — the student queue is not a separate
machine, so employee jobs contend directly with ours. `(S:0-1)` = the 8 GPUs are spread across both
CPU sockets.

### 3.2 `scontrol show partition gpu-1-student` + `scontrol show node i6-gpu-1`  *(login node, 2026-09-09)*

```text
PartitionName=gpu-1-student
   AllowGroups=ALL AllowAccounts=1gpu-student,2gpu-student,4gpu-student,8gpu-student AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=N/A
   DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=1-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=i6-gpu-1
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=64 TotalNodes=1 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED

NodeName=i6-gpu-1 Arch=x86_64 CoresPerSocket=16
   CPUAlloc=42 CPUTot=64 CPULoad=4.04
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:RTXA5000:8(S:0-1)
   NodeAddr=131.159.61.238 NodeHostName=i6-gpu-1 Version=21.08.5
   OS=Linux 5.15.0-94-generic #104-Ubuntu SMP Tue Jan 9 15:25:40 UTC 2024
   RealMemory=257782 AllocMem=0 FreeMem=74519 Sockets=2 Boards=1
   State=MIXED ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=gpu-1-student,gpu-1-employee
   BootTime=2026-03-23T10:31:18 SlurmdStartTime=2026-03-23T10:32:08
   LastBusyTime=2026-08-24T09:29:48
   CfgTRES=cpu=64,mem=257782M,billing=64,gres/gpu=8
   AllocTRES=cpu=42,gres/gpu=6
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
```

**Read-out for the thesis.**
- Topology: `Sockets=2 × CoresPerSocket=16 × ThreadsPerCore=2` = **32 physical cores / 64 threads**.
  A job asking `--cpus-per-task=8` gets 8 *threads*, i.e. ~4 physical cores.
- `RealMemory=257782` MB = **251.7 GiB**; `TmpDisk=0` — **no node-local scratch**, so all I/O goes to
  network/home storage. Worth one sentence if dataset-loading time is ever discussed.
- `AllocTRES=cpu=42,gres/gpu=6` at snapshot time: **6 of 8 GPUs and 42 of 64 CPUs busy with other
  users' jobs**. This is the concrete evidence behind the shared-node caveat in §1.
- Accounts are named `1gpu-student … 8gpu-student` — the account determines the GPU quota. Which one
  this project used is not recorded in the job scripts (`--account` is never set, so it is the
  default association). ➜ hole in §6.
- `PreemptMode=OFF`, `OverSubscribe=NO`: jobs are not preempted and GPUs are not oversubscribed —
  a running job keeps its GPU exclusively. Contention is on CPU/RAM/IO, **not** on the GPU itself.
  This is the strongest sentence available in defence of the timing numbers; use it.

### 3.3 `lscpu` — the CPU  ✅ *collected 2026-09-09 (job 25567, CPU-only srun)*

```text
Model name:                         AMD EPYC 7282 16-Core Processor
Core(s) per socket:                 16
Socket(s):                          2
CPU max MHz:                        2800.0000
CPU min MHz:                        1500.0000
L1d cache:                          1 MiB (32 instances)
L1i cache:                          1 MiB (32 instances)
L2 cache:                           16 MiB (32 instances)
L3 cache:                           128 MiB (8 instances)
```

**Read-out.** 2 × AMD EPYC 7282 (Zen 2 "Rome", 7 nm — vendor architecture name, not from `lscpu`) =
**32 physical cores / 64 threads**, 1.5–2.8 GHz, 32 KiB L1d + 32 KiB L1i + 512 KiB L2 per core,
128 MiB L3 total (8 × 16 MiB CCX, i.e. L3 is shared in groups of 4 cores). Two sockets = **2 NUMA
domains**; the 8 GPUs straddle both (`S:0-1` in §3.1).

> 🔑 **The ratio worth one sentence in the thesis.** 32 physical cores serve 8 GPUs — **~4 physical
> cores per GPU** when the node is full. Our eval loop is MuJoCo rollout + an SLSQP solve per step,
> both **CPU-bound** (SLSQP runs on the CPU), and a job asking `--cpus-per-task=8` receives 8 *threads* ≈ 4 physical cores.
> Evaluation wall-clock on this machine is therefore plausibly limited by the CPU side, not the
> A5000. This is a second, independent reason `avg_time` is not a hardware benchmark — and it is a
> stronger one than contention, because it holds even on an idle node.

### 3.4 `nvidia-smi` full output  *(not collected — deliberately)*

Every per-GPU fact that matters is already in every log header (model, driver, 24564 MiB). A full
`nvidia-smi` would add only the CUDA *driver* API version and the PCIe link width, and it would
require holding a GPU from a contended 8-card pool to learn them. **Not worth it** — dropped rather
than deferred.

### 3.5 Software stack — from the verified environment check

Verbatim from `Slurm_Codes/logs/2026-04-29/verify_env_job_19741_153833.log` (job 19741, git
`630dd13`), the run that certified the cluster env:

```text
--- Package Import Check ---
torch                2.2.2+cu121
numpy                1.26.4
scipy                1.13.1
gym                  0.26.2
gymnasium            0.29.1
gymnasium_robotics   1.2.4
minari               0.4.3
wandb                0.17.5
mujoco               2.3.7
diffusers            0.31.0
transformers         4.41.2

--- System Check ---
Numpy Version:   1.26.4
CUDA Available:  True
GPU Device:      NVIDIA RTX A5000

✅ VERIFICATION SUCCESSFUL: Environment is ready for training.
```

Pinned in `requirements.txt`, not printed by that check but used by the pipeline:

| package | version | role |
|---|---|---|
| `scipy` | 1.13.1 | **the default projector solver: SLSQP** (`method='SLSQP'`) — `aux_repo/dpcc/diffuser/sampling/projection.py:138`; arm C resolves to `slsqp` unless overridden |
| `casadi` | 3.6.5 | the **alternative** NLP backend, `nlp_backend='ipopt'` → IPOPT with limited-memory Hessian. Selectable, not default — `hardflow_projection.py:82-88, 245-255`. Every run announces which backend ran |
| `torchdiffeq` | 0.2.3 | ODE integration |
| `torchsde` | (unpinned) | SDE path |
| `torchvision` | (unpinned) | ResNet visual encoders (D3IL vision) |
| `hydra-core` / `omegaconf` | 1.1.1 / 2.1.1 | D3IL config machinery |
| `einops` | 0.8.0 | backbone tensor ops |
| `matplotlib` / `imageio` / `opencv-python` | 3.9.0 / 2.34.1 / 4.10.0.84 | plots, GIFs, frames |
| `jax` / `jaxlib` | 0.4.28 | present in the env (iMF upstream lineage), **not** on the reported pipeline path |
| `triton` | 2.2.0 | pulled by torch 2.2.2 |
| Python | **3.10** | `conda create -n FMPCC python=3.10` |

**Consistency check that can be stated as fact:** torch `2.2.2+cu121` (CUDA **12.1** runtime) against
driver `530.41.03` — the driver is new enough for the 12.1 runtime, and the same pair held for every
run from April to September. ⚠️ The **cuDNN** version and the exact `nvcc`/toolkit (if any) are still
unrecorded; `torch.backends.cudnn.version()` in §5.3 settles it.

> ⚠️ **Nuance to keep.** The §3.5 block is a snapshot from **2026-04-29**. The env was not frozen
> after that (later generations pulled in upstream code — MeanFlow, AlphaFlow, HardFlow, the
> vision encoders). Nothing in the logs suggests torch/mujoco were bumped, but the honest claim is
> *"the environment was verified at project start with these versions and no version bump is
> recorded"*, not *"these exact versions produced every number"*. Re-running §5.3 once and pasting
> the result here converts this from a hedge into a fact.

### 3.6 Compute budget — 🟡 **placeholder, rough estimate only**

Not yet collected with `sacct`. What follows is arithmetic over `JOB START` / `JOB END` timestamps
in the committed sbatch logs — an **order-of-magnitude figure, not an accounting record**. Do not
put it in the thesis in this form; re-derive with `sacct` (§5.4) before quoting anything.

| quantity | rough value | how obtained |
|---|---|---|
| committed log files | **1385** (777 carry the `NODE:`/`GPU INFO:` job header; the rest are per-step or per-seed side logs) | `Slurm_Codes/logs/*/*.log` |
| logs with a parseable START+END pair | **285** | the rest were killed, timed out, or run a script without the END trap |
| summed wall-clock over those 285 | **≈ 800 GPU-hours** | ⚠️ a **floor**: it covers 285 of the 777 header-bearing jobs, and the `temp/` drops hold further runs not committed here |
| median job | **≈ 0.5 h** | many short aggregation/plot/gate jobs |
| longest jobs | **24.0 h** — i.e. jobs that hit the partition wall-clock limit | several `eval_meanflow`, `eval_fmv3_ode`, `eval_dpcc`, `train_meanflow` |
| heaviest job families (total h) | `imf_train` ≈ 107, `eval_meanflow` ≈ 98, `eval_fmv3_ode` ≈ 73, `train_meanflow` ≈ 46, `eval_dpcc` ≈ 44, `af_eval` ≈ 43 | same arithmetic, grouped by `JOB NAME` |
| a known single reference run | DiT 80 k steps ≈ **12 h 50 m** (jobs 24874/24875) | `MASTER_TEST_HISTORY.md` |

**Vague statement that is safe to make today:** *"The reported results represent on the order of
10³ GPU-hours on a single RTX A5000, accumulated over roughly five months (2026-04 → 2026-09), with
individual training runs taking 3–24 h."* Tighten it with `sacct` before submission.

```text
(paste `sacct` output from §5.4 here to replace the estimate)
```

---

## 4. Parameters of record — the `app:repro` table

| item | value | source |
|---|---|---|
| Cluster | TUM I6 chair cluster | — |
| Scheduler | SLURM 21.08.5 | §3.2 |
| Partition | `gpu-1-student` (shared with `gpu-1-employee`, same node) | §3.1 |
| Wall-clock limit | 24 h | §3.1/3.2 |
| Node | `i6-gpu-1` (the partition's only node) | §3.1; log headers |
| CPU model | 2 × AMD EPYC 7282 (Zen 2 "Rome"), 16C/32T each, 1.5–2.8 GHz | §3.3 |
| CPU topology | x86-64, 2 sockets (2 NUMA domains) × 16 cores × 2 threads = 64 logical CPUs; 128 MiB L3 (8 × 16 MiB CCX) | §3.2, §3.3 |
| Node RAM | 257 782 MB ≈ 251.7 GiB | §3.2 |
| Node-local scratch | none (`TmpDisk=0`) | §3.2 |
| OS / kernel | Ubuntu, Linux 5.15.0-94-generic | §3.2 |
| GPU model | NVIDIA RTX A5000, 24 GB (24564 MiB) | log headers; §3.1 |
| GPUs on node / per job | 8 / **1**, exclusive (`OverSubscribe=NO`, `PreemptMode=OFF`) | §3.1/3.2 |
| NVIDIA driver | 530.41.03 (constant 2026-04 → 2026-09) | log headers |
| CUDA runtime (torch) | 12.1 (`torch 2.2.2+cu121`) | §3.5 |
| cuDNN | **TBD** | §5.3 |
| Python | 3.10 (conda env `FMPCC`) | `remote_setup_guide.md:63` |
| PyTorch | 2.2.2+cu121 | §3.5 |
| NumPy / SciPy | 1.26.4 / 1.13.1 | §3.5 |
| MuJoCo | 2.3.7 | §3.5 |
| Projector NLP (default) | SciPy 1.13.1 **SLSQP** | `aux_repo/dpcc/.../projection.py:138`; `hardflow_projection.py:82` |
| Projector NLP (alternative) | CasADi 3.6.5 → IPOPT, limited-memory Hessian; `nlp_backend='ipopt'` | `hardflow_projection.py:245-255` |
| ODE integration | `torchdiffeq` 0.2.3 | `requirements.txt` |
| Rendering backend | EGL, headless (`MUJOCO_GL=egl`), GPU index pinned to the SLURM allocation | template §3 |
| Typical training allocation | 1 GPU, 8 CPU threads, 32 GB, ≤24 h | `--cpus-per-task` / `--mem` census |
| Typical training wall-clock | 3–24 h (e.g. DiT 80 k steps ≈ 12 h 50 m) | §3.6 |
| Total compute | 🟡 order 10³ GPU-hours — **estimate, re-derive with `sacct`** | §3.6 |

**Citation form for the thesis:** hardware gets one table in `app:repro` and one sentence in
`sec:setup:implementation`. Do not scatter GPU names through result tables — backbone and parameter
count are what the result tables carry (see `NOTES_naming_and_rebuild.md`).

---

## 5. The commands

§5.1 and §5.2 are **done** (output in §3.1–3.3). Only §5.3 remains, and it is optional —
**CPU-only, no `--gres`**. The cluster is contended; nothing below takes a GPU.

### 5.1 Login node — cluster and partition shape ✅ *collected 2026-09-09*

```bash
sinfo -e -o "%20N %14P %.5D %.5c %.9m %.30G %12T %14l"
scontrol show partition gpu-1-student
scontrol show node i6-gpu-1
```

### 5.2 Compute node — the CPU SKU  ✅ *done (job 25567; output in §3.3)*

🟢 **No GPU needed** — `--gres` omitted, so it takes no card from the shared pool. Queued briefly,
then ran in seconds.

```bash
srun -p gpu-1-student -n1 -c1 --mem=512M -t 00:02:00 --job-name=cpuinfo \
  bash -c 'lscpu | grep -iE "model name|socket|core per|thread per|mhz|cache"'
```

### 5.3 Software stack — confirm the 2026-04 snapshot still holds

🟢 **Also no GPU needed.** `torch.backends.cudnn.version()` reads the linked library, not a device.
`torch.cuda.get_device_properties()` *would* need a card — but it is redundant here: the log headers
already establish A5000 / 24564 MiB / driver 530.41.03.

```bash
srun -p gpu-1-student -n1 -c1 --mem=2G -t 00:03:00 --job-name=swinfo bash -lc '
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate FMPCC
python - <<PYX
import torch, platform, importlib
print("python   ", platform.python_version(), "|", platform.platform())
print("torch    ", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
for m in ["numpy","scipy","mujoco","casadi","torchdiffeq","torchvision","hydra","omegaconf","einops","diffusers","transformers"]:
    try: print(f"{m:13s}", getattr(importlib.import_module(m),"__version__","?"))
    except Exception as e: print(f"{m:13s} -- {type(e).__name__}")
PYX
'
```

### 5.4 Compute budget — replaces the §3.6 estimate

```bash
# one representative TRAIN + EVAL job (pick the ones a thesis table actually cites)
sacct -j <TRAIN_JOBID>,<EVAL_JOBID> \
  --format=JobID,JobName%22,Partition,AllocCPUS,ReqTRES%40,Elapsed,TotalCPU,MaxRSS,State

# the whole project, for a defensible total-compute sentence
sacct -u $USER -S 2026-04-01 -E now -X \
  --format=JobID,JobName%22,Elapsed,AllocCPUS,ReqTRES%30,State
```

> `MaxRSS` is only populated for finished jobs with job accounting enabled; if it comes back empty,
> say so rather than guessing a memory footprint. Note `sacct` retention may not reach back to
> April — if it does not, the §3.6 log-header arithmetic is the fallback and must be labelled as an
> estimate wherever it appears.

---

## 6. Holes — fill before writing `app:repro`

- [x] ~~CPU model name~~ **2 × AMD EPYC 7282, 32 physical cores / 64 threads** — collected
      2026-09-09, job 25567. §3.3. **No hardware facts remain open.**
- [ ] **cuDNN version + whether the env drifted after 2026-04-29** — §3.5's caveat box. One CPU-only
      `srun` (no GPU).
- [ ] **Total compute is an estimate** — §3.6 is log-header arithmetic over 285 of 777 jobs and is a
      floor. Re-derive with `sacct` before any number reaches the thesis.
- [x] ~~Did the node or driver change during the project?~~ **No.** `NVIDIA RTX A5000 / 530.41.03`
      is byte-identical in the earliest (2026-04-29) and latest (2026-09-02) committed log headers,
      and all 777 header-bearing logs report `NODE: i6-gpu-1`.
- [ ] **Which SLURM account/quota the project ran under** (`1gpu-student` … `8gpu-student`). Never
      set via `--account`, so it is the default association. `sacctmgr show assoc where user=$USER`
      answers it. Only matters if the thesis claims a compute allocation.
- [ ] **Was every reported number produced on `i6-gpu-1`?** 777/777 header-bearing logs say yes, but the
      Colab notebooks (`Results_and_Data_Analysis_Colab_T4/`, `ipynbs_Colab/`) ran elsewhere.
      Analysis ≠ measurement — state once that *evaluation* ran on the cluster and
      *aggregation/plotting* ran off it.
- [ ] **Contention is unmeasured per run.** §3.2 proves contention exists (6/8 GPUs, 42/64 CPUs busy
      at snapshot) but no run records what else was on the node. Any surviving timing claim needs
      either a repeated-measurement spread or an explicit "single measurement, shared node"
      disclaimer. The mitigating fact — GPUs are **not** oversubscribed and jobs are **not**
      preempted — belongs in the same sentence.
- [ ] **Environment reproducible from a file?** The repo has a root `requirements.txt` (153 pinned
      lines) and the installer notes in `Slurm_Codes/install_env_from_colab_ipynb_style/` (plus
      `HardFlow/environment.yml` for that upstream) — but it is unverified that any of them
      reproduces the cluster `FMPCC` env as it actually stands. `app:repro` should point at exactly
      one artefact; §5.3 is the check.
