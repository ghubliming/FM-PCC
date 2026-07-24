# MJPC `agent_server` — Build & Deploy Status (PAUSED)

**Date paused:** 2026-06-30 · **Last build job:** 22272 · **Last eval job:** 22270

## Goal
Build the `mujoco_mpc` gRPC binary `agent_server` and deploy it (self-contained,
with its `.so` deps) into `third_party/mujoco_mpc/mujoco_mpc/mjpc/` so the
`mjpc` controller (`controller='mjpc'`) works during UAV FM eval. The Python
client (`agent.py`) spawns `agent_server` as a subprocess and talks to it over
local gRPC.

## CURRENT STATE: builds fine, runs in build env, **segfaults when deployed**
- ✅ Binary **compiles** successfully (conda GCC 14.3 build env).
- ✅ **SMOKE-1** (run inside the conda build env, "golden"): **WORKS** —
  prints `Server listening on [::]:19996`, stays up (exit 124 = timeout).
  → **The binary itself is NOT broken.**
- ❌ **SMOKE-2** (deployed binary, build dir + conda env deleted, eval-style env):
  **SIGSEGV** (`si_addr=0x4`, `SEGV_MAPERR`) during static/early init,
  right after `rt_sigaction(SIGRT_1, ...)` (pthread/gRPC thread setup),
  **before any application code or XML/task file access**.
- ❌ Eval (`eval_fm_uav.py` → `MJPCTracker`) therefore gets
  `grpc.FutureTimeoutError` after the 30 s connect wait (server died silently).

## What is NOT the cause (ruled out by evidence)
- **Missing shared libs** — strace shows `libmujoco.so.3.2.3`, `libstdc++.so.6`,
  `libgcc_s.so.1`, `libz`, `libm`, `libc`, `libdl` ALL resolve (job 22270/22272 ldd).
- **Missing task/XML file at startup** — strace (openat/access/stat) shows **no**
  file access before the crash; it dies during lib init, not task load.
- **`libstdc++`/`libgcc` version mismatch** — golden env (SMOKE-1) uses the same
  toolchain and runs fine; deployed deps are the same conda libs, copied.
- **RPATH pointing at deleted build dir** — was a real smell (RPATH before fix:
  `.../envs/_mjpc_build/lib:/tmp/mjpc_build_*/build/lib`), **fixed** with
  `patchelf --set-rpath '$ORIGIN'`; ldd now resolves everything next to the
  binary — **but it STILL segfaults.** So RPATH was not the root cause.
- **Build problem (bad cmake flags / regression)** — ruled out: SMOKE-1 proves
  the compiled binary runs correctly in its native environment.
- **`mujoco-mpc` PyPI wheel** — does **not exist** on PyPI; must build from source
  (confirmed: `pip index versions mujoco-mpc` → no distribution).

## What the cause LIKELY IS (next hypotheses)
The crash is environment-sensitive: identical binary + identical resolved libs,
**works in the conda env, dies under a stripped (`env -i`) / eval env.** So the
trigger is something the *process environment* provides in the golden case and
lacks when deployed. Leading candidates to test next:
1. **A missing env var the binary/gRPC/absl reads at init** (e.g. `GRPC_*`,
   `GLOG_*`, locale `LANG`/`LC_*`, `TCMALLOC_*`). `env -i` strips all of these;
   the real eval env (conda FMPCC) has many. Worth diffing `env` golden-vs-eval.
   NOTE: the *real* eval job (22270) also crashed with a full conda env, so a
   bare missing-var theory is incomplete — but a *conflicting* var (below) fits.
2. **A conflicting lib injected by the FMPCC conda env at eval time** — e.g.
   eval loads Python `mujoco==2.3.7`, EGL/OpenGL, CUDA libs into the process;
   when `agent_server` is spawned it **inherits LD_* / preloaded state** that
   clashes with its own `libmujoco.so.3.2.3` (MuJoCo 3.x). A two-MuJoCo-ABI
   clash in one process tree is a strong `si_addr=0x4` candidate.
3. **gRPC/absl global-constructor crash tied to glibc/pthread differences**
   between the conda build sysroot (glibc 2.34) and the system runtime glibc
   used when deployed (system `libc.so.6` from `/lib/x86_64-linux-gnu`).

## The ONE diagnostic still needed: a real stack trace
strace got us the *where* (early init, post-`rt_sigaction`) but not the *which
function*. To go further we MUST get a backtrace. Blocker: **no `gdb`/`catchsegv`
on the cluster.**
- Next action: `conda install -c conda-forge gdb -n FMPCC -y` (one-time), then
  re-run the build-script SMOKE-2 under `gdb -batch ... -ex run -ex "bt full"`.
- The `bt full` will name the crashing function → first non-speculative fix.

## Decision while paused
Eval reverted to the **PID stop-and-go** path (`controller='pid_stopgo'`, the
config default) which needs none of this. UAV FM eval runs cleanly without MJPC.
Return to MJPC only when the stack-trace diagnostic is available.

## Files involved
- `Slurm_Codes/sbatch/uav_fm/build_mjpc_agent_server.sh` — build + deploy +
  `patchelf $ORIGIN` + SMOKE-1/SMOKE-2 self-tests (current debugging harness).
- `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` — **reverted clean** (MJPC
  `LD_LIBRARY_PATH` + DIAG probe removed; PID path unaffected).
- `FM_v3_uav_test/mjpc_tracker.py` — `MJPCTracker`; spawns Agent w/ stdout/stderr
  capture (`subprocess_kwargs`).
- `third_party/mujoco_mpc/mujoco_mpc/` — bundled Python client (`agent.py`,
  proto stubs); `mjpc/` holds deployed `agent_server` + copied `.so` libs
  (+ `PLACE_BINARY_HERE.txt` — proof the binary is NOT shipped, must be built).
- `FM_v3_uav_test/eval_fm_uav.py` — eval; controller read from config (line ~155/543).
- `config/uav.py` — `controller` setting (`pid_stopgo` default; `mjpc` opt-in).

## Reproduce / resume quickly
```bash
# rebuild + run both self-tests (read [ SMOKE-1 ] / [ SMOKE-2 ] in build log):
sbatch Slurm_Codes/sbatch/uav_fm/build_mjpc_agent_server.sh
# clean PID eval (no MJPC needed), works today:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh s_curve 6 "" fm_only gif
```

## Key evidence snippet (job 22272)
```
[ RPATH ] before: .../envs/_mjpc_build/lib:/tmp/mjpc_build_22272/mujoco_mpc/build/lib
[ RPATH ] after : $ORIGIN
[ SMOKE-1 ] exit code = 124   # golden env: GOOD
[ SMOKE-2 ] exit code = 139   # deployed, $ORIGIN, all libs resolve: STILL SIGSEGV
#   SIGSEGV si_addr=0x4 after rt_sigaction(SIGRT_1) — early init, no file access
```
