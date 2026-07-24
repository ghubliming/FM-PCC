# Fix_5 — Bundle `mujoco_mpc` Python Package Inside FM-PCC

**Problem:** Cluster eval crashed with `ModuleNotFoundError: No module named 'mujoco_mpc'`
because `mujoco_mpc` was only available as an individual install at `/workspaces/mujoco_mpc`
(Docker dev environment), not on the Slurm cluster.

---

## What Was Done

### 1. Bundled the pure-Python package (`third_party/mujoco_mpc/`)

The `mujoco_mpc` Python package is pure Python + gRPC stubs — no compilation required.
Copied from `/workspaces/mujoco_mpc/python/mujoco_mpc/`:

```
third_party/mujoco_mpc/
└── mujoco_mpc/
    ├── __init__.py
    ├── agent.py              # gRPC client (pure Python)
    ├── mjpc_parameters.py
    ├── proto/
    │   ├── __init__.py
    │   ├── agent_pb2.py      # generated from agent.proto
    │   └── agent_pb2_grpc.py # generated + import fixed
    └── mjpc/
        └── PLACE_BINARY_HERE.txt
```

### 2. Generated proto stubs

```bash
python3.14 -m grpc_tools.protoc \
    -I/workspaces/mujoco_mpc/grpc \
    --python_out=third_party/mujoco_mpc/mujoco_mpc/proto \
    --grpc_python_out=third_party/mujoco_mpc/mujoco_mpc/proto \
    /workspaces/mujoco_mpc/grpc/agent.proto
```

Fixed the import in `agent_pb2_grpc.py`:
- Before: `import agent_pb2 as agent__pb2`
- After:  `from mujoco_mpc.proto import agent_pb2 as agent__pb2`

### 3. Updated PYTHONPATH in `eval_fm_uav.sh`

`Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`:
```bash
export PYTHONPATH="$REPO:$REPO/third_party/mujoco_mpc:$PYTHONPATH"
```

### 4. Updated error message in `mjpc_tracker.py`

Now points to the bundled location instead of the old Docker-only path.

---

## What You Still Need To Do on the Cluster

The compiled `agent_server` binary (C++) **cannot** be bundled — it must be built from source.

**Place it at:** `third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server`

**Build steps (on a build node):**
```bash
git clone https://github.com/google-deepmind/mujoco_mpc /tmp/mujoco_mpc
cd /tmp/mujoco_mpc
mkdir build && cd build
cmake .. -DMJPC_BUILD_GRPC_SERVICE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target agent_server -j$(nproc)
cp bin/agent_server $HOME/FMPCC/FM-PCC/third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server
chmod +x $HOME/FMPCC/FM-PCC/third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server
```

`agent.py` looks for the binary at `pathlib.Path(__file__).parent / "mjpc" / "agent_server"`,
which resolves to the path above — no code changes needed once the binary is placed.

---

## After the Binary Is In Place

Re-run the eval sbatch normally:
```bash
sbatch Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh pillars 6
```

The `mujoco_mpc` import will succeed from `third_party/` via PYTHONPATH,
and `MJPCTracker.__init__` will spawn the local `agent_server` automatically.
