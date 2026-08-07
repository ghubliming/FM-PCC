# CHANGELOG — Fix_9: config provenance — the results path must describe the run that made it

**Date:** 2026-08-07 · **Generations touched:** Gen3v6 (MeanFlow) + Gen3v7 (AlphaFlow), plus the
shared `config/avoiding-d3il.py`
**Retrieval flag:** `FIX_9_CFG_PROVENANCE` — find every site with:

```bash
grep -rn FIX_9_ --include='*.py' --include='*.yaml' . | grep -v Archived_Codes
```

**Triggered by:** job **24334**, whose results landed in `…/H8_K2_Meuler_T1_…/` while the
projector was gated at 0.5. Analysis:
[`RESULTS_Fix_8_unet_width32_rerun_24317_24334.md`](./RESULTS_Fix_8_unet_width32_rerun_24317_24334.md) §6.

---

## 1. One-line summary

Three knobs that change eval results — the projection yaml, the DPCC threshold, and the two
HardFlow overrides — did not reach the results-folder name. The name was built from
`config/projection_eval.yaml`, **a file the Gen3v6/Gen3v7 evals never open**. Every eval script now
publishes what it actually resolved, and the name (and the config snapshot) is built from that.

## 2. The three defects

### 2.1 The `T` token came from the wrong file

`config/avoiding-d3il.py:6` hard-opened `config/projection_eval.yaml` at import time and fed its
`diffusion_timestep_threshold` into the `T` token. But `eval_flow_matching_v3_meanflow.py:48`
loads `config/meanflow_projection_eval.yaml` (AlphaFlow: `alphaflow_projection_eval.yaml`), and
only *that* value ever reaches the `Projector` (`:302` → `:355`). The two files were free to
disagree — and on the cluster they did (shared file `1`, meanflow file `0.5`), which is exactly
how 24334 got named `T1`.

### 2.2 The snapshot archived the wrong file too

`utils/setup.py:214` hard-coded the same path, so the run archived a yaml it never read while the
one that actually gated it went unrecorded. This is what made the wrong name look self-consistent.

### 2.3 `HFFM_ACT_THRESHOLD` / `HFFM_BATCH` reached the name not at all

Neither appears anywhere in `config/avoiding-d3il.py`. **Two runs differing only in activation
threshold wrote to the same directory and silently overwrote each other** — Gen12 hit this and
worked around it by hand-tagging folders
(`Gen12/DA/DA_20260803_HardFlow_activation_threshold_0p1.md:32`). `HFFM_FLOW_STEPS` was the one
override that did it right (via the `K` token); this fix follows that pattern.

## 3. Mechanism — an environment handshake, one resolver

`config/avoiding-d3il.py` is imported **lazily**, by `utils.Parser.read_config`'s
`importlib.import_module` during `Parser().parse_args()` — i.e. *after* the eval script has loaded
its yaml and resolved its overrides. So the eval publishes, and the config module reads:

| env var | published by | consumed by | token |
|---|---|---|---|
| `FMPCC_PROJ_CFG` | eval, right after the yaml load | config module + `snapshot_configs` | (file choice) |
| `FMPCC_DPCC_THRESHOLD` | eval, after resolving `DPCC_THRESHOLD`/yaml | config module | `T` |
| `HFFM_ACT_THRESHOLD` | eval, **re-published resolved** | config module | `A` |
| `HFFM_BATCH` | eval, **re-published resolved** | config module | `B` |

The eval re-publishes the *resolved* HardFlow values (aliases `all`/`late` mapped, yaml fallbacks
applied) so that **`resolve_activation_threshold()` in `hardflow_projection.py` stays the single
resolver** — the config module never reimplements it, which would just recreate this bug class.

With nothing set, everything falls back to the old behaviour, so train jobs and the generations
still on the shared yaml are untouched.

## 4. Files touched — 7

| file | change |
|---|---|
| `config/avoiding-d3il.py` | `_PROJ_CFG` from env; `_yaml_threshold` prefers the resolved env value; new `_hf_act_threshold` / `_hf_batch_size`; new `_num()` token formatter; new `args_to_watch_fmv3_hf_plan`; both HardFlow plan blocks repointed at it and given the two `hf_*` keys |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | publishes `FMPCC_PROJ_CFG`; `DPCC_THRESHOLD` override; republishes resolved values; provenance print |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | same, but the whole resolve+publish block **moved above** the `--flow-steps` branch (see §5) |
| `flow_matcher_v3_meanflow/utils/setup.py` | snapshot `FMPCC_PROJ_CFG` under its own basename |
| `flow_matcher_v3_alphaflow/utils/setup.py` | same |
| `config/meanflow_projection_eval.yaml` | `hardflow.activation_threshold: 1.0 → 0.5` |
| `config/alphaflow_projection_eval.yaml` | same |

### 4.1 New knob: `DPCC_THRESHOLD`

Arm B's threshold had **no** env override — it could only be changed by editing the yaml, while
arm C had `HFFM_ACT_THRESHOLD`. Since runs are configured at submit time on the cluster, not in
git, `DPCC_THRESHOLD` now exists and is symmetric:

```bash
DPCC_THRESHOLD=0.5 HFFM_ACT_THRESHOLD=0.5 HFFM_BATCH=4 HFFM_FLOW_STEPS=2 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh
```

### 4.2 The two thresholds stay SEPARATE

Arm B (`diffusion_timestep_threshold`) and arm C (`hardflow.activation_threshold`) are two knobs
with the *same semantics* (fraction of the late trajectory over which projection is active; higher
= more projection — aligned by fix_6) and **independent values**. They are not unified: a threshold
sweep needs them independent. What was wrong was that their **defaults disagreed** (0.5 vs 1.0), so
an unset `HFFM_ACT_THRESHOLD` silently ran the two arms at different gates. The yaml default is now
0.5 on both, and the new provenance line prints them side by side so a deliberate mismatch is
visible rather than accidental.

## 5. AlphaFlow needed a code move, not just an insertion

`eval_flow_matching_v3_alphaflow.py`'s `--flow-steps` branch calls
`importlib.import_module('config.' + exp)` at `:118`, and Python caches modules — so the config
module executes *there*, not at `parse_args()`. Publishing at the knobs' original location (`~:99`)
would have been too late for the folder name on any `--flow-steps` run. The whole
threshold + HardFlow resolve block therefore moved up to immediately after the yaml load (`:80–101`,
before `:118`); a pointer comment marks the old location. MeanFlow has no such early import and
needed only insertions.

## 6. Naming impact — verified by simulation

Simulated by importing the patched config module with stubbed `yaml`/`diffuser.utils` and the
verbatim `watch()` implementation from `diffuser/utils/setup.py:21`:

| scenario | `plan_fm_v3_meanflow` exp_name |
|---|---|
| no env set (legacy / train) | `H8_K2_Meuler_T0.5_A1_B1_D…MeanFlowODE` |
| **the 24334 command** (`HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 HFFM_FLOW_STEPS=2`) | `H8_K2_Meuler_**T0.5_A0.5_B4**_D…MeanFlowODE` |
| same but `HFFM_ACT_THRESHOLD=1.0` | `H8_K2_Meuler_T0.5_**A1_B4**_D…MeanFlowODE` — **distinct, no overwrite** |

- The `T` token is now `0.5` — what the projector was gated at — instead of `1`.
- `_num()` keeps tokens stable (`1.0 → A1`, not `A1.0`), so the style matches the existing `T1` /
  `T0.5` convention.
- **Regression check passed:** `plan_fm_v3_imeanflow`, `plan_fm_v3_drifting` and
  `plan_fm_v3_ode_selectable` names are **byte-identical** before and after. The `A`/`B` tokens went
  into a *new* watch list used only by the two blocks that have a HardFlow arm — deliberately not
  into the shared `args_to_watch_fmv3_ode_plan`.

🔴 **Consequence:** Gen3v6/Gen3v7 eval results now land in new directories (`…_A…_B…`). Existing
result trees keep their old names and are **not** overwritten — they are also not re-findable by
tools that hard-code the old path. `logs_in_develop/Gen14/U7/da_20260805_hardflow_analysis.py:6`
hard-codes one such path; it points at a Gen14 tree this fix does not touch, but it is the pattern
to watch for.

## 7. Deliberately NOT changed

| | why |
|---|---|
| `config/projection_eval.yaml` itself | still the default for every generation that has no dedicated yaml; the cluster copy's stray `1` is a working-tree edit, not a repo change (§9) |
| the train-side `args_to_watch_*` lists | Fix_8 §4 rejected this: it changes `diffusion_loadpath` and makes existing checkpoints unreachable. Only *plan* lists are touched here, which name outputs |
| `args_to_watch_fmv3_ode_plan` | shared by 5 plan blocks, 3 without a HardFlow arm — see §6 |
| unifying the two thresholds | §4.2 |
| Gen12/Gen13 HardFlow evals | out of scope; they have their own config path and were not read |
| `logs_in_develop/MASTER_TEST_HISTORY.md` | never self-edited (standing convention) |

## 8. Verification

- **`ast.parse` — all 5 touched Python files pass.**
- **Line endings preserved.** 5 of 7 files are CRLF, the 2 eval scripts are LF. The patch script
  read/wrote with `newline=''`, detected each file's own terminator, and aborted on mixed endings;
  `file` reports the same terminator for every file before and after.
- **Anchor counts asserted** — each replacement declared its expected occurrence count (the
  `hf_*`-keys insertion is the only 2× one: the MeanFlow and AlphaFlow blocks share that anchor
  verbatim). The patch aborts rather than guessing.
- **Naming simulated end-to-end** (§6), including the regression check on the three non-HardFlow
  generations.
- **Import ordering checked** in both eval scripts: `os`/`yaml`/`resolve_activation_threshold` all
  precede first use, and every publish precedes the first config-module import.
- 🔴 **NOT run** — no Python environment in this container (no `pyyaml`; the simulation stubbed it).
  Every runtime claim needs a cluster job.

## 9. Do this on the cluster before the next eval

1. The cluster's `config/projection_eval.yaml` has `diffusion_timestep_threshold: 1` while git has
   `0.5` — an uncommitted working-tree edit, proven by the snapshot inside 24334's results dir.
   After this fix it no longer affects Gen3v6/Gen3v7 names, but it still drives Gen0/DPCC,
   drifting and imeanflow. Resolve it:
   ```bash
   cd ~/FMPCC/FM-PCC && git status --porcelain config/ && git diff config/projection_eval.yaml
   ```
   then `git checkout` it, or commit it if the `1` was deliberate.
2. First eval after this lands should show the new line — check it says what you passed:
   ```
   [ eval ] resolved  cfg=config/meanflow_projection_eval.yaml  dpcc_threshold=0.5  hf_act_threshold=0.5  hf_batch=4  hf_candidate_cost=prox
   ```
3. Confirm the results dir now carries `_T0.5_A0.5_B4_` and that
   `config_snapshot_avoiding-d3il/` contains **`meanflow_projection_eval.yaml`** (not
   `projection_eval.yaml`).

## 10. Follow-ups this leaves open

- **Gen12/Gen13 HardFlow evals** likely share defects 2.2/2.3; not audited.
- **`n_trials`, `seeds` and the constraint set** still do not appear in any results path — the same
  bug class, one level down. A run with `n_trials: 2` and one with `n_trials: 10` still collide.
- **A pre-flight assert** that the snapshotted yaml's threshold equals the one in the folder name
  would make this class of drift impossible rather than merely visible.
