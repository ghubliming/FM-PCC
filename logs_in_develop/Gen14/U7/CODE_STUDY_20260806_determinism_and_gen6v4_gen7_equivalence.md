# Code study — Gen14 eval: is it deterministic, and is it Gen6V4 / Gen7?

**Date:** 2026-08-06 · **Generation:** Gen14 (Visual-Mix-ML) · **Epoch:** U7
**Question:** *Under the same, correct setup, will a Gen14 run give the same result as a
Gen6V4 / Gen7 run — and will a Gen14 re-run give the same result as itself?*

**Method.** Static read of the code only: `diff` of the model/sampling packages, the config
block builders, and a line-by-line trace of the eval call chain. **No run data, no logs, no
npz.** Every claim below is a code fact with a `file:line`.

**What this method can and cannot settle.** It can settle: which code paths are shared, where
seeds are set, where randomness is drawn, where wall-clock time enters a decision, and where
control flow depends on data. It **cannot** settle whether a given CUDA kernel actually returns
bit-identical values run-to-run on the cluster's hardware — that needs an experiment (§6).

---

## §1 — The eval call chain

`action_seq_size` defaults to `1` (`eval_mix_visual_aligning.py:1059`, `:2776`), and
`:2082` replans whenever `action_counter == action_seq_size`. **One replan per env step.**

```
eval_mix_visual_aligning.py:2460   Parser().parse_args(experiment=EXPERIMENT, seed=seed)
                                     └─ utils/setup.py:157 set_seed(args.seed)
    :2602  for (geo_name, geo_config, geo_variant, is_tightened) in _run_items:   ← variant loop
    :2816    sim = Aligning_Sim(seed=seed, …, n_cores=1)
    :2832    sim.test_agent(agent)
               d3il/simulation/aligning_sim.py:221   n_cores==1 → eval_agent runs IN-PROCESS
                 :62-64   random.seed / torch.manual_seed / np.random.seed  ← RESEED, per variant
                 :66      for context in contexts:                          ← 30 rollouts
                 :101     while not done:                                   ← env decides length
                 :102       agent.predict(...)
                              eval_…:2082  replan → self.model(cond, projector=…)
                                 mf_diffusion.py:530 forward → :314 conditional_sample
                                                             → :189 p_sample_loop
                                    :204   x = torch.randn(shape, device=device)   ← THE draw
                                    :237   velocity = self._predict_velocity(...)  ← U-Net
                                    :288   projector.project(...)                  ← SLSQP
```

Same shape for the other arms: `diffusion.py:304→205→164`, `fm_diffusion.py:309→223→160`,
`af_diffusion.py`.

**Seeding is correct and per-variant.** `aligning_sim.py:62-64` reseeds `random`, `torch`
(which covers CUDA) and `numpy` at the top of every `eval_agent`, and a fresh `Aligning_Sim` is
constructed inside the variant loop. So each variant *starts* from a bit-identical RNG stream.
This is worth stating plainly because it is the opposite of what one would guess from the
single `set_seed` at `:2460`.

> Note for anyone grepping: `mf_engine.py:157`, `af_engine.py:157`,
> `mf_trajectory_model.py:248`, `af_trajectory_model.py:247` also call `torch.manual_seed(seed)`.
> Those live in the standalone `sample_trajectory()` entrypoints of the Gen3v6/v7 state-only
> lineage. **The Gen14 visual eval never reaches them** — it enters at `p_sample_loop`
> (`mf_diffusion.py:189`), which does not reseed.

---

## §2 — Every place a re-run can diverge

Enumerated from the code, classified. "Deterministic" = the code contains no mechanism for the
value to change between two identical invocations.

| # | site | what it is | verdict |
|---|---|---|---|
| 1 | `aligning_sim.py:62-64` | per-variant reseed of all three RNGs | **deterministic** |
| 2 | `mf_diffusion.py:204`, `af_diffusion.py:260`, `fm_diffusion.py:164`, `diffusion.py:168` | the initial noise draw | **deterministic given stream position** — see #6 |
| 3 | `diffusion.py:158` | per-step noise in the DDPM reverse chain (`n_diffusion_steps` extra draws per replan) | same |
| 4 | `eval_…:2148` | `trajectory_selection='random'` → `which = 0`, explicitly commented "DPCC semantics" | **deterministic** (despite the name) |
| 5 | `d3il` MuJoCo `env.step` | physics | **deterministic** |
| 6 | `aligning_sim.py:101` `while not done:` | episode length is decided by the env (success ⇒ early stop) | **🔴 stream-coupling** — see below |
| 7 | `projection.py:157, 282-301` | circuit breaker: `_slow = _call_ms > _PROJ_SLOW_MS`, from `time.perf_counter()` | **🔴 wall-clock dependent** |
| 8 | `projection.py:246-266` | per-solve 60 s deadline callback raising `_SolveBudgetExceeded` → trajectory kept unprojected, cost `inf` | **🔴 wall-clock dependent** |
| 9 | `scipy.optimize.minimize(method='SLSQP')` `projection.py:251-259` | fixed `tol`, `maxiter`, deterministic `x0` | **deterministic** given identical inputs |
| 10 | `mix_visual_aligning/utils/setup.py:15-19` | `set_seed` sets the four seeds and **nothing else** | **🔴 no determinism flags** |
| 11 | U-Net + dual ResNet-18 forward on CUDA | — | **unknown from code** — see §6 |

### #6 — why one divergence contaminates the rest of the variant

`action_seq_size = 1`, so a rollout of length *L* consumes exactly *L* initial-noise draws
(fm/mf/af) or *L·(1 + n_diffusion_steps)* (diffusion). The loop at `aligning_sim.py:101` is
`while not done:` and `done` comes from `env.step` — i.e. **episode length is data-dependent**
(early stop on success, otherwise `max_episode_length`).

All 30 rollouts of a variant share one RNG stream, seeded once at `aligning_sim.py:62-64`.
Therefore: if rollout *k* ends at a different step in two runs, every rollout *k+1 … 29* begins
at a **different stream position** and draws different initial noise. There is no per-rollout
reseed anywhere in the chain.

This is a structural amplifier, not a bug on its own — it only matters if something upstream
can perturb rollout *k*. Sites #7, #8 and #11 are the candidates.

### #7 / #8 — the projector reads the clock

`projection.py:27-31` defines the knobs:

```python
_PROJ_SOLVE_BACKSTOP_S = 60.0   # per-solve hang backstop (s)
_PROJ_SLOW_MS          = 1000.0 # a project() call slower than this = one "slow step"
_PROJ_CB_WINDOW        = 40     # steps of history to judge
_PROJ_CB_TRIP_FRAC     = 0.9    # fraction of the window that must be slow to OPEN
_PROJ_CB_COOLDOWN      = 40     # OPEN skips before a HALF-OPEN probe
```

When the breaker opens (`:297`) the projector **stops projecting** (`:149-153` returns early),
so the trajectory that reaches the robot is a different trajectory. Whether it opens depends on
`time.perf_counter()` — i.e. on cluster load, on what else is on the GPU, on page cache. Two
runs of identical code on a busy vs idle node can take different control-flow branches here.

Same for the 60 s per-solve deadline at `:248`: on a trip it keeps the **unprojected**
trajectory and sets cost `inf` (`:260-267`).

These are deliberate safety valves (Fix_15.2) and they are the right design for a 24 h wall
clock. But they mean **the projected arms are not reproducible by construction**, independently
of anything on the GPU. The unprojected `diffuser` variant does not touch this path.

### #10 — nothing opts out of nondeterministic kernels

`mix_visual_aligning/utils/setup.py:15-19`:

```python
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

No `torch.use_deterministic_algorithms`, no `cudnn.deterministic`, no `cudnn.benchmark = False`
anywhere in `mix_visual_aligning/`, `fm_visual_aligning/` or `diffuser_visual_aligning/`. The
only determinism flags in the tree are in the vendored `HardFlow/run/utils.py:24-25`, which the
Gen14 arm-C port does not call.

So the code does not *claim* bit-reproducible kernels, and does not ask for them.

---

## §3 — Is Gen14 the same code as Gen6V4 / Gen7?

For the `diffusion` and `fm` arms: **yes, the generative path is verbatim.**

| file | vs Gen6V4 | vs Gen7 |
|---|---|---|
| `models/diffusion.py` ↔ Gen14 `fm_diffusion.py` | 1 line (the import) | 1 line (the import) |
| `visual_gaussian_diffusion.py` / `visual_fm_diffusion.py` | 1 line | 1 line |
| `models/visual_unet.py` | 0 non-import | 0 non-import |
| `models/unet1d_temporal_cond.py` | **0** (byte-identical) | **0** |
| `models/unet1d_temporal_film.py` | **0** | **0** |
| `models/helpers.py` | 0 non-import | 0 non-import |
| `sampling/projection.py` | **0** | **0** |
| `datasets/normalization.py` | **0** | **0** |
| `utils/{arrays,training,constraints_helpers}.py` | **0** | **0** |
| `datasets/sequence.py` | 93 lines, *all* of them comments + the non-visual 23-D `StateOnlyAligningDataset` | **0** |
| `sampling/__init__.py` | +21, purely additive (U7 HardFlow import) | same |

This confirms the structural rule asserted in `engine_registry.py`'s docstring: the
`diffusion`/`fm` arms import only verbatim copies, and every newly-authored line sits in a
module only `mf`/`af` reach.

Reproduce with:

```bash
diff diffuser_visual_aligning/models/diffusion.py mix_visual_aligning/models/diffusion.py
diff fm_visual_aligning/models/diffusion.py       mix_visual_aligning/models/fm_diffusion.py
```

---

## §4 — …but the *setup* is not the same setup

### 4.1 Different checkpoint tree

`_mix_plan_block()` (`config/aligning-d3il-visual.py:924`) derives, at `:957-963`:

```
prefix             = plans/mix_visual_aligning_{engine}/{ckpt_id}/
diffusion_loadpath = mix_visual_aligning_{engine}/H8_…_E{engine}
```

against Gen6V4's `visual_aligning_dpcc/H8_K100_D…` and Gen7's `fm_visual_aligning/H8_D…`. Two
watch-list fragments differ on top of the folder name: `('engine','E')`
(`config/aligning-d3il-visual.py:851`) is Gen14-only, and `('diffusion','D')` carries the
package path, which reads `mix_visual_aligning.*` here.

**So Gen14 loads weights from its own training run.** Making it load Gen6V4's or Gen7's weights
requires editing `diffusion_loadpath`. And re-training to match is not available as an option
either — see #10 above.

### 4.2 The `diffusion` arm's planning knobs are Gen7's, not Gen6V4's

`_mix_plan_common` (`config/aligning-d3il-visual.py:910`) is built from
`base['plan_fm_visual_aligning']`, so **all four arms inherit Gen7's planning block.** For the
`diffusion` arm, `drop=` (`:1053-1055`) removes the ODE/Beta keys and the training-key mirror
loop (`:945-952`) restores the Gen6V4 identity values — but `mpc_batch_size` is in neither list,
because it appears in `args_to_watch_mix_visual_plan` only, never in `…_train`.

| knob | Gen6V4 `plan_visual_aligning_dpcc` | Gen14 `plan_mix_visual_aligning_diffusion` | same? |
|---|---|---|---|
| `n_diffusion_steps` | 100 (`:597`) | 100 — mirrored from the training block | ✅ |
| `action_weight` | 10 (`:603`) | 10 — mirrored (it *is* a train watch key) | ✅ |
| `horizon` | 8 | 8 | ✅ |
| `max_episode_length` | 400 | 400 | ✅ |
| `window_size` / `obs_seq_len` | 1 / 1 | 1 / 1 | ✅ |
| **`mpc_batch_size`** | **1** (`:614`) | **4** — inherited from `:694` | ❌ |

`mpc_batch_size` is the MPC candidate pool: `eval_…:2111` only enters the selection block
`if self.batch_size > 1`. So Gen6V4 always executes its single sample, while Gen14-`diffusion`
picks one of four under `trajectory_selection`. **Different controller.**

Fix, if Gen6V4 parity is the goal: add `'mpc_batch_size': 1` to the
`plan_mix_visual_aligning_diffusion` overrides (`config/aligning-d3il-visual.py:1048`).

The `fm` arm inherits `4`, which *is* Gen7's own value — no divergence there.

### 4.3 A stale comment

`config/aligning-d3il-visual.py:976-977` says gate G1 *"compares this arm against Gen7 and
expects bit-identical training."* Given §2 #10, bit-identical GPU training is not something the
code is set up to deliver. The gate should be restated as a loss-curve tolerance.

---

## §5 — Verdict

| question | answer, from the code |
|---|---|
| Is the Gen14 `fm` generative path the same code as Gen7? | **Yes** — verbatim (§3) |
| Is the Gen14 `diffusion` generative path the same code as Gen6V4? | **Yes** — verbatim (§3) |
| Does Gen14 evaluate the same *weights*? | **No** — separate checkpoint tree (§4.1) |
| Is the Gen14 `diffusion` eval config Gen6V4's? | **No** — `mpc_batch_size` 4 vs 1 (§4.2) |
| Is the Gen14 `fm` eval config Gen7's? | **Yes** |
| Is the seed handling correct? | **Yes** — reseeded per variant (§1) |
| Do the *projected* variants re-run identically? | **No** — the breaker and the solve deadline branch on wall-clock (§2 #7, #8) |
| Does the *unprojected* variant re-run identically? | **Only if the CUDA forward is bit-reproducible** — the code neither guarantees it nor asks for it (§2 #10, #11) |
| If anything does perturb one rollout, is it contained? | **No** — one shared RNG stream across all 30, no per-rollout reseed (§2 #6) |

So: **the code as written does not guarantee that a Gen14 re-run reproduces itself**, and it
positively guarantees that Gen14 ≠ Gen6V4 for the `diffusion` arm at the current config.

Two framing points worth keeping separate:

- **This is not a Gen14 defect.** Sites #6, #7, #8, #10 are all in code that Gen6V4 and Gen7
  share verbatim (`sampling/projection.py` is byte-identical across all three; `aligning_sim.py`
  and `utils/setup.py` likewise). Whatever reproducibility Gen14 has, Gen7 and Gen6V4 have the
  same.
- **It is still a limit worth knowing.** A 400-step closed loop that replans from its own output
  every step will amplify any perturbation, and #6 turns a single perturbed rollout into 30.

---

## §6 — What the code cannot answer, and the experiment that would

Unresolved: **site #11** — whether the U-Net + dual ResNet-18 forward returns bit-identical
values on two identical invocations on the cluster GPU. Nothing in the source decides this; it
depends on the cuDNN/cuBLAS kernels selected at runtime.

Minimal discriminating test (**run on cluster**):

1. Run one *unprojected* variant (`diffuser`) twice in the same job, same seed, `--record none`.
   This avoids #7/#8 entirely, so the only live candidates are #11 and, downstream of it, #6.
2. Compare the sampled trajectories per replan. If the first difference appears mid-rollout at
   ~1e-4 → nondeterministic kernels (#11) confirmed. If they are bit-identical → #11 is
   excluded and the projected arms' variance is entirely #7/#8.
3. Repeat with `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True` and
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` to confirm the fix works before adopting it.

---

## §7 — Changes this study suggests

Nothing here is implemented; no source file was modified.

1. **`mpc_batch_size: 1`** in `plan_mix_visual_aligning_diffusion` if Gen14-`diffusion` is meant
   to be the Gen6V4 baseline (`config/aligning-d3il-visual.py:1048`).
2. **Log the circuit-breaker state per rollout** so a run that took the skip branch is
   identifiable after the fact. `projection.py:297` already prints on trip; it is not recorded
   in the npz.
3. **Determinism flags** in `set_seed()` (`mix_visual_aligning/utils/setup.py:15`) — pending the
   §6 test, and at some throughput cost.
4. **Per-rollout reseed** in the eval loop would contain #6 even if #11 stays live. Cheap; it
   does not make a run correct, but it stops one perturbed rollout from resampling the other 29.
5. **Reword the G1 gate comment** (`config/aligning-d3il-visual.py:976-977`).
6. **Multi-seed protocol** — the only thing that makes a distance comparison defensible
   regardless of how #11 resolves.
