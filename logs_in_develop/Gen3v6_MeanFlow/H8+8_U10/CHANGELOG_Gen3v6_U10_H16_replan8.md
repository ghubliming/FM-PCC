# CHANGELOG — Gen3v6 U10: HardFlow-style planning structure (H16 + replan-8) for MeanFlow

**Date:** 2026-08-16 · **Scope:** `flow_matcher_v3_meanflow` ↔ `FM_v3_meanflow_test`, avoiding-d3il
**Status:** code complete, **nothing run, nothing committed**. No cluster job submitted.
**Guide:** `GUIDE_H16_replan8_MF_UNet.md` (same folder, updated in the same pass — it is the how-to;
this file is the what-changed).

---

## 0. Why

HardFlow evaluates at **horizon 16, executing the first 8 actions per plan** (`controller="rh"`,
`replan_steps=8`, `HardFlow/run/eval.py:390-397`). FM-PCC evaluates at **horizon 8, executing 1
action and replanning every env step**. Comparing arm C against HardFlow's published behaviour
therefore compares two different controllers, not two different samplers.

U10 makes both halves of that structure reachable in our own harness:

| | training-time property? | before U10 | after U10 |
|---|---|---|---|
| horizon | ✅ **yes** — dataset windows (`datasets/sequence.py:71-83`) + per-step loss weights (`models/helpers.py:295-314`) | literal `8`, two places, easy to half-apply | `MF_HORIZON`, one read, both places |
| replan cadence | ❌ no — zero hits for `replan` anywhere under `flow_matcher_v3_meanflow/` or `scripts/train.py`; `loss_discount: 1.0` ⇒ flat per-step weights | **did not exist** | `MF_REPLAN_STEPS` |

Horizon still requires its own checkpoint. Cadence does not — which is why the two are separate
knobs and why the H16/replan-1 rung (horizon effect, no code) exists between today's numbers and
the H16/replan-8 target (cadence effect).

---

## 1. The default-preservation claim

**Every pre-U10 command produces byte-identical behaviour.** This was the design constraint, not an
afterthought. Concretely:

| knob | default | why the default path is unchanged |
|---|---|---|
| `MF_HORIZON` | `8` | `_mf_horizon` resolves to the literal that was there before |
| `MF_BACKBONE` | `'mf_dit'` | same |
| `MF_REPLAN_STEPS` | `1` | the replan condition `plan_idx >= 1` is true on **every** step ⇒ `policy()` is called every step and `action` comes straight from its return value, exactly as before |
| `-t` shift | `_n = 1` | `observations[:, :-1] - prev[:, 1:]` — the original expression |
| `desired_next_pos` | `replan_steps == 1` branch | the original `samples.observations[0, 1, …]` line, untouched |
| results path | no `_msg` token | auto-tagging fires **only** when `replan_steps != 1` |

The only unconditional additions on the default path are two attribute assignments in the policies
(`last_executed_actions` / `last_executed_observations`) and the horizon guard, which is a no-op
when config and checkpoint agree.

---

## 2. Files changed

### 2.1 `config/avoiding-d3il.py` — one source for horizon + backbone

| line | change |
|---|---|
| `:59-60` | **new** `_mf_horizon = int(os.environ.get('MF_HORIZON', 8))`, `_mf_backbone = os.environ.get('MF_BACKBONE', 'mf_dit')` |
| `:739` | train block `flow_matching_v3_meanflow` `'horizon'`: `8` → `_mf_horizon` |
| `:783` | train block `'imf_backbone'`: `'mf_dit'` → `_mf_backbone` |
| `:1401` | plan block `plan_fm_v3_meanflow` `'horizon'`: `8` → `_mf_horizon` |
| `:1443` | plan block `'imf_backbone'`: `'mf_dit'` → `_mf_backbone` |

The train and plan blocks are joined by `diffusion_loadpath`, which reproduces
`args_to_watch_fmv3_mf_train` token-for-token (`:1465-1466`). Editing one and not the other resolves
to a directory that does not exist — documented as trap #6 in the file header. Reading both from one
env var makes that failure unreachable.

**Why an env var and not a CLI flag:** `utils.Parser.add_extras` is **commented out**
(`diffuser/utils/setup.py:77`), so `--horizon 16` is silently ignored. The config file is the only
input path; env-reading is how this repo already parameterises it (`HFFM_*`, `FMPCC_*`).

### 2.2 `flow_matcher_v3_meanflow/sampling/policies.py` (`Policy`, arms A/B)

- `__init__`: `self.replan_steps = 1`, `self.last_executed_actions = None`,
  `self.last_executed_observations = None`.
- temporal-consistency selection: shift is now `_n = replan_steps` instead of a hard `1`. The
  previous plan was executed for `n` steps, so **its step `n` is the new plan's step 0** — at
  `replan_steps=8` the old expression compares windows misaligned by 7 steps and the `-t` variants
  silently stop meaning what they say. Clamped to `min(_n, H-1)` so at least one step overlaps.
- publishes the executed candidate's full plan before returning.

**Why publishing is necessary and not convenience:** `which_trajectory` is local to `__call__`.
`trajectories.actions` is *never* reordered, while `observations` **is** reordered by the `-t`
branch — the fix_5 invariant. A caller replaying `samples.actions[0]` would therefore execute the
**wrong candidate** under `-c`/`-t` selection at `batch_size > 1`. The two published arrays use
`which_trajectory` and `executed_idx` respectively, matching that invariant.

### 2.3 `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` (`HardFlowPolicy`, arm C)

Same three changes, mirrored. `_select()` returns an index and never reorders `observations`, so one
index serves both published arrays.

### 2.4 `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py`

1. **Cadence resolution** (after the FIX_9 provenance block, before any `Parser().parse_args()`):
   `MF_REPLAN_STEPS` env > `replan_steps:` yaml > `1`.
2. **Path-collision guard.** The cadence is *not* a results-folder token. Promoting it to one would
   rename every historic H8 path, so instead `replan_steps != 1` auto-sets
   `FMPCC_RUN_MSG=r<N>` (the existing custom-message slot) unless the user set one. Without this an
   r1 and an r8 run at the same K/A/T write to the same directory — the exact hazard
   `args_to_watch_fmv3_hf_plan` exists to prevent.
3. **G1 horizon guard** — hard `SystemExit` when the checkpoint's `horizon` ≠ `args.horizon`,
   placed immediately after the weight source is chosen. See §3.
4. **G2 cadence guard** — `SystemExit` when `replan_steps >= horizon`. Mirrors HardFlow's own
   assertion (`run/eval.py:380-382`); a plan cannot supply more actions than it holds.
5. **`policy.replan_steps = replan_steps`** after construction — one line covering both policy
   classes.
6. **Cache-and-replay in the rollout loop**: replan when `plan_actions is None or plan_idx >=
   replan_steps`, else replay `plan_actions[plan_idx]`. Cache is reset per trial, so no plan leaks
   across an env reset.
7. **Derived-quantity fixes under replan > 1**: `proj_active` is false on non-replan steps (no plan,
   no projection ran) and `desired_next_pos` walks along the cached plan (`plan_obs[plan_idx]`)
   instead of freezing on step 1 of a plan made several steps ago.

### 2.5 `config/meanflow_projection_eval.yaml`

New documented `replan_steps: 1` key (top level, next to `diffusion_timestep_threshold`).

### 2.6 sbatch provenance

`Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh` and `eval_meanflow_hardflow.sh` each echo the
resolved `MF_HORIZON` / `MF_BACKBONE` (+ `MF_REPLAN_STEPS` on the eval), so every log states what it
ran. `submit.sh` passes `--export=ALL`, so the vars reach the job with no plumbing.

---

## 3. G1 — why the horizon guard aborts instead of warning

The existing CONFIG-OVERRIDES-PKL reconciler treats `horizon` as an architecture key: it prints a
`[WARNING]`, keeps the pkl value to protect the state_dict, and continues. That is not enough here,
because `args.horizon` independently drives the `Projector`, the `RTRecorder` and the `horizon=`
argument of every `policy()` call — so a mismatch does not merely warn, it runs a **mixed**
configuration.

Worse, whether it is even detectable depends on the backbone:

| backbone | H8 checkpoint loaded at H16 | why |
|---|---|---|
| `mf_dit` | 🟢 **crashes** | learned `pos_embed = nn.Parameter(1, num_patches, D)` is in the state_dict (`models/mf_dit_official_trajectory.py:294`) ⇒ shape mismatch |
| `dit` | 🔴 silent | RoPE tables are `register_buffer(..., persistent=False)` (`models/mf_dit_trajectory.py:301-303`) ⇒ absent from the state_dict, rebuilt at the new length, extrapolates |
| **`unet`** *(this study's choice)* | 🔴 **silent** | `ResidualTemporalBlock` accepts `horizon` and **never uses it** (`models/unet1d_temporal_cond.py:55-70`) — Conv1d + Linear only, all length-agnostic |

The configuration this study runs is the one with no natural protection. G1 supplies it.

---

## 4. Explicitly NOT done

- **No run.** No training, no eval, no cluster job. Nothing verified against real data.
- **No commit.**
- **Gen3v7 (α-Flow) not synced.** `flow_matcher_v3_alphaflow` + `plan_fm_v3_alphaflow` still carry
  literal `'horizon': 8` / `'imf_backbone': 'sit'` and their `Policy` sibling has neither knob. The
  copy-modify convention says mirror it when α-Flow needs it; doing it blind now would double the
  untested surface.
- **Cadence is not a first-class folder token.** It rides the `_msg` slot (§2.4 item 2). If replan
  becomes a standing axis rather than one study, promote it to `args_to_watch_fmv3_hf_plan` —
  and accept that this renames existing paths.
- **No H16 diffusion-DPCC baseline.** The pinned paper baseline (K20/aw10) exists only at H8/replan-1.
  An H16/8+8 MeanFlow number has no matched opponent until that is retrained too. Scope decision,
  not a code gap.
- **`n_trials` / `seeds` untouched** in the yaml (`2` / `[6]`) — deliberately, since they are the
  run's scope, not the feature's.

---

## 5. Verification

**Done locally** (this container has no Python packages — no import-level or runtime check is
possible here):

| check | result |
|---|---|
| `python3 -m py_compile` on all four edited `.py` | ✅ pass |
| `bash -n` on both edited sbatch scripts | ✅ pass |
| all four config sites land in the right blocks (`grep -n _mf_horizon\|_mf_backbone`) | ✅ 739/783 = train, 1401/1443 = plan |
| yaml re-parse | ⚠️ **not run** — no PyYAML locally. Single scalar key + comments; validated on first cluster run |

**Must be verified on the cluster, in this order:**

1. **Default regression (the important one).** Re-run an existing H8 cell with no new env vars and
   confirm the results path is unchanged (no `_msg`) and the metrics match the recorded run. This is
   what proves §1.
2. `MF_HORIZON=16 MF_BACKBONE=unet` training writes `…/H16_…_bbunet_…` and differs from the H8
   folder in the `H` token only.
3. G1 fires: point the eval at the H8 checkpoint with `MF_HORIZON=16` and confirm it aborts.
4. G2 fires: `MF_REPLAN_STEPS=16` at horizon 16 aborts.
5. `MF_REPLAN_STEPS=8` produces `nlp_solves` ≈ 1/8 of the replan-1 run at the same K — the direct
   observable that the cadence is real.

---

## 5.1 Follow-up already done — U10.1 run provenance

Making `horizon`/`imf_backbone` env-resolved cost provenance: `snapshot_configs` copies the config
verbatim, so the snapshot now reads `'horizon': _mf_horizon` — identical bytes for an H8 and an H16
run — and `MF_REPLAN_STEPS` never reaches `args` at all. Fixed in
`../../CLI_Override_Snapshot/CHANGELOG_U10.1_run_provenance.md`: a `run_provenance.json` is written
beside the results recording resolved values, which knobs were set explicitly versus inherited, the
yaml digest, and the git commit + dirty flag. Wired into Gen3v6 train+eval and four other
generations with the same env-override pattern.

## 6. Follow-ups

- Sync to Gen3v7 (α-Flow) if the H16/8+8 result is worth reproducing there.
- Decide the H16 diffusion-DPCC baseline (§4) before any H16 number enters a comparison table.
- `avg_time` is now total plan time ÷ env steps. Under replan-8 that is an **amortised** per-step
  cost and will drop by roughly the cadence factor. Correct, and interesting — the 2.8–3.3× arm-C
  wall-time penalty from the 2026-08-02 DA may largely vanish under HardFlow's own cadence — but it
  is not the same quantity as the replan-1 rows, and tables must say so.
- The foresight npz keeps one entry per env step; under replan > 1 consecutive entries repeat the
  plan currently being executed. Shapes are unchanged, downstream plotting is unaffected, but a
  reader counting distinct plans should divide by the cadence.
