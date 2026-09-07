# CHANGELOG — Gen15 U6: the α-Flow arm gets a U-Net, a live α, and a checkpoint selector

**Date:** 2026-09-03
**Scope:** `config/uav_mix.py` + `mix_uav/utils/` + `mix_uav_test/` + `Slurm_Codes/sbatch/uav_mix/` + `diffuser/utils/provenance.py`
**Retrain required:** ✅ **Yes for the af arm** — the default backbone changes, which is a checkpoint-path key. No other arm is touched.
**Status:** patched locally, **not committed**, **not yet run on the cluster.**
**Motivation:** [`../../Gen3v7_AlphaFlow/INVESTIGATION_20260903_af_unet_port_to_UAV_Gen15_and_VisualAligning_Gen14.md`](../../Gen3v7_AlphaFlow/INVESTIGATION_20260903_af_unet_port_to_UAV_Gen15_and_VisualAligning_Gen14.md) §1
**Sibling:** [`../../Gen14/U12/CHANGELOG_Gen14_U12_checkpoint_selector_MIX_EPOCH.md`](../../Gen14/U12/CHANGELOG_Gen14_U12_checkpoint_selector_MIX_EPOCH.md) — the same epoch fix on the visual-aligning side.

---

## 1. The problem

Before U6, `config/uav_mix.py` read **no environment variable at all**. Three consequences:

### 1.1 🔴 The af arm was on a backbone that cannot carry a headline claim

```python
# config/uav_mix.py (pre-U6), arm 'mix_uav_af'
'imf_backbone': 'sit',   # hardcoded, overriding _TWO_TIME_BACKBONE's 'unet'
```

`fm` and `mf` run the 4.0 M U-Net (`freq_dim=32`, post-Fix_8). The `af` arm ran a SiT sized from
`dit_hidden_size=256, dit_depth=8` — **≈ 9.4 M**, ~2.4× the others. So **every published Gen15
af-vs-mf row moves objective, backbone and parameter count together.** The config's own comment
already conceded this ("*NOT parameter-matched … the deferred appendix arm, never the
architecture-matched claim*"), while `_TWO_TIME_BACKBONE`'s comment says this generation is
"*LOCKED to 'unet' for the headline comparison*". The af arm was the exception to its own rule.

### 1.2 🔴 α-Flow has never actually run on the UAV

`af_alpha_end: 0.0` was hardcoded in both the train and plan blocks. The sigmoid plus
`af_alpha_clamp=0.005` snap α to **exactly 0** from ~71.2 % of the budget on, and
`mix_uav/models/af_diffusion.py:568` routes `alpha <= 0` into Gen3v6's `_p_losses_meanflow` body
**unmodified**. Therefore:

> **Every Gen15 `af` checkpoint ever trained — jobs 25135–25138 (pillars), the corridor SiT sweep,
> all of them — deployed a MeanFlow model in an α-Flow folder.** The published UAV "α-Flow" rows
> compare *MeanFlow-on-SiT (9.4 M)* against *MeanFlow-on-U-Net (4.0 M)*: an architecture ablation
> wearing an objective's name. The numbers are real; the label is not.

Gen3v7 hit the identical defect on `avoiding-d3il` and fixed it with `AF_ALPHA_END` (commit
`beb7f26c`). This is that fix in Gen15 spelling.

### 1.3 🟡 The checkpoint selector existed but was unreachable and unsafe

**Correction to the investigation's §1.4, which said Gen15 had "no epoch override".** It has had a
`--epoch` CLI flag all along (`mix_uav_test/eval_mix_uav.py`), and its help text already documented
the 80 000-of-100 000 wart. Two things were actually wrong, and the second is worse than a missing
knob:

| | |
|---|---|
| **unreachable** | `eval_mix_uav.sh` never passed it, so no Slurm run could use it |
| 🔴 **not path-safe** | it did **not** reach `_uav_eval_tag`, so a `--epoch latest` pass wrote into the **same results folder** as the `best` pass of the same weights and silently overwrote it. On the af arm those are different models: `best` is chosen on a test_loss that scales with α and therefore prefers a **mid-curriculum** checkpoint (Gen3v7 DA 2026-09-01 §3.1/§4.2, where `latest` vs `best` was the whole difference between 0/2 and 2/2 goals at K=1) |

✅ **Also a correction in Gen15's favour:** the `latest → -1 → state_-1.pt` crash that Gen14 had is
**already guarded here** (`mix_uav/utils/serialization.py:98-109`, added 2026-08-19). Nothing to do.

### 1.4 `latest` never meant the end of training

`save_freq = n_train_steps // 5` and `self.step` only reaches `n_train_steps - 1` inside
`train_epoch`, so the newest numeric checkpoint is **step 80 000 of 100 000**. No save cadence
fixes it — the last multiple of *any* frequency is below `n_train_steps`.

---

## 2. What changed

### 2.1 `config/uav_mix.py` — three env knobs (the file's first)

| knob | values | default | reaches |
|---|---|---|---|
| `UAV_MIX_BONE_AF` | `unet` \| `sit` \| `dit` | **`unet`** (was a hard `sit`) | **checkpoint** path (`_bb<val>`) |
| `UAV_MIX_AF_ALPHA_END` | `[0, 1]` | `0.0` | **checkpoint** path (`_ae<val>`) |
| `UAV_MIX_EPOCH` | `best` \| `latest` \| `<step>` | `best` | **results** path (`_EP<sel>`) |

Plus `_env_or_none` (blank == unset — Gen14 lost two eval passes to `VAR= cmd` exporting the empty
string, job 25215; Gen15 inherits the rule rather than the bug) and `import os`.

Every value is validated at config-import time, including the one that only bites later:
`UAV_MIX_AF_ALPHA_END=0.001` is **rejected** because it sits below `af_alpha_clamp=0.005`, so
`_get_ratio` would snap it to exactly 0 and the arm would train MeanFlow while the folder read
`_ae0.001`. `UAV_MIX_BONE_AF=mf_dit` is rejected too — that is the *mf* arm's class.

Both af knobs are applied to the **train block and the plan block from the same module-level
value**, so a train/plan mismatch (eval rebuilding a savepath the trainer never wrote) is now
unrepresentable rather than merely warned about in a comment.

### 2.2 🔴 The default flip — read this before submitting anything

The af checkpoint path carries `_bbsit` today and `_bbunet` from now on:

```
today   logs/UAV_MIX/uav-<scene>/mix_uav_af/H8_D…AlphaFlowODE_9D_as1_ae0_bbsit/6/
U6      logs/UAV_MIX/uav-<scene>/mix_uav_af/H8_D…AlphaFlowODE_9D_as1_ae0_bbunet/6/
```

- **`sit` is kept, not deleted.** `UAV_MIX_BONE_AF=sit` reaches the existing tree unchanged. Every
  SiT checkpoint and every SiT results folder survives byte-for-byte.
- **`bb` is an unconditional `exp_name` token**, so the two bones can never overwrite each other
  (gate G2 asserts the whole knob cross-product).
- ⚠️ **A default `af` eval will now fail on a missing checkpoint until the U-Net arm is trained.**
  That is a loud, correct failure, and it is the right trade against the silent confound it
  replaces.

### 2.3 `mix_uav_test/eval_mix_uav.py`

| change | why |
|---|---|
| `--epoch` default `'best'` → `None` | so "unset" is distinguishable from "explicitly best"; `None` means *take the plan block's value*, which the config resolves from `UAV_MIX_EPOCH` |
| module-level `EPOCH_OVERRIDE`, published in `main()` next to `ENGINE` | the file's existing pattern for a process-wide selection |
| validation delegated to `config.uav_mix._uav_epoch` | CLI and env can never disagree about what is legal |
| `_load_base_cfg` injects `cfg['diffusion_epoch']` | 🔴 exactly the same fix, for the same reason, as the `flow_steps_v3` injection five lines above: `_uav_eval_tag` reads the cfg dict, so a key that is not in it cannot reach the folder name. That is how Gen11 labelled every folder `K20` regardless of the K that ran |
| `_uav_eval_tag` emits `EP<sel>`, absent at `best`, before the free-form run tag | a `latest` pass lands **beside** the `best` one. Pre-U6 folder names are unchanged |
| `eval_scene` loads `base_cfg['diffusion_epoch']`, **not** `args.epoch` | the two differ whenever the selector came from the env; loading a different checkpoint than the folder claims is the exact failure the token prevents |
| breadcrumb: `checkpoint = state_<sel>.pt (trained to step N)` | `trainer.step` is read back out of the checkpoint file — the file's own claim, not the config's |
| **α breadcrumb + verdict, af arm only** | 🔴 the one check that separates α-Flow from MeanFlow at deployment. Prints `alpha(step N)` from the train-time pkl and says outright when α = 0 that *"these weights were trained on the MeanFlow target"*. Uses the `_get_ratio` staticmethod, which exists for exactly this "schedule questions without a training loop" purpose |
| bone breadcrumb | names the 4.0 M / 9.4 M distinction in the eval log, where the numbers are produced |
| ⚠ warning when `af` + selector is `best` | names the failure before it happens |
| provenance: `diffusion_epoch`, `checkpoint_epoch_resolved`, `checkpoint_step`, `imf_backbone`, `af_alpha_end` | `latest` is a *request*; the resolved step is the *answer*. And every pre-U6 af row in the corpus is a 9.4 M SiT — a fact the numbers themselves do not carry |

### 2.4 `mix_uav/utils/training.py` + `training_twotime.py`

A final `self.save(self.step)` at the end of `train()`, fixing §1.4. Fires **only on a completed
run** (the early `return` at the top means steps remain and the periodic saves still stand). Costs
one extra `state_100000.pt` per completed run.

*No copy-fidelity ledger exists in `mix_uav_test/gates_mix_uav.py`, so unlike Gen14 there is no
additive-graft constraint on these two files.*

> ✅ **Disk cost is zero after pruning.** `tools/clean_weights/clean_weights.py` keeps
> `state_best.pt` plus the **highest-numbered** checkpoint per directory. The new final save
> simply becomes that highest-numbered one, so a pruned tree holds the same two files it held
> before — and the one it keeps is now the end of the schedule instead of 80 % of it.

### 2.5 `Slurm_Codes/sbatch/uav_mix/`

`train_mix_uav.sh`, `eval_mix_uav.sh`, `eval_k_sweep.sh` export, **validate with the same
`best|latest|digits` and `unet|sit|dit` rules as the Python side**, and echo all three knobs into
the job banner — a typo dies at job start, not four hours in. The af-only knobs are rejected on a
non-af engine. Both pipelines (`uav_mix_pipeline.sh`, `uav_mix_ksweep_pipeline.sh`) build an
`EXPORT_OPTS` and pass it to every child `sbatch`: `--export=ALL` would carry them anyway, but two
of the three are checkpoint-path keys and one is a results-path key, and a stage that does not see
them resolves a different directory than the submitter is watching. The pipelines also warn when
α is floored while `UAV_MIX_EPOCH` is unset — the combination that trains the experiment and then
discards it.

### 2.6 `mix_uav_test/gates_mix_uav.py` — **G9**

New gate (`--gates G9`, CPU-only) asserting what each knob is *allowed* to move:

| | check |
|---|---|
| (a) | the config's bone whitelist equals `engine_registry.get('af')['backbones']` — drift is caught, not assumed away |
| (b) | U6 defaults (`unet` / `0.0` / `'best'`) and **train-block == plan-block** on bone and α in every configuration |
| (c) | bone and α each land in their **own** checkpoint tree (3 distinct names) |
| (d) | 🔴 `UAV_MIX_EPOCH` leaves the **checkpoint** path byte-identical while reaching the plan block |
| (e) | 🔴 …but **does** reach the eval-params folder: `best` is byte-identical to a pre-U6 name, `latest` strictly extends it by `_EPlatest`, and the free-form run tag stays last |
| (f) | malformed bone / α / epoch values rejected at config-import time |

### 2.7 `diffuser/utils/provenance.py`

`UAV_MIX_BONE_AF`, `UAV_MIX_AF_ALPHA_END`, `UAV_MIX_EPOCH` added to `TRACKED_ENV`.

---

## 3. Verified locally (no cluster, no GPU)

The config module was exec'd under five env settings with stub `yaml` / `diffuser.utils` /
`engine_registry`, and `_uav_eval_tag` was exec'd in isolation:

```
(a) whitelist == registry : True
(b) defaults              : bone='unet' ae=0.0 epoch='best'
(b) train == plan         : True   (default, sit, alpha0.2)
(c) 3 distinct ckpt trees : True
        default(unet)   mix_uav_af/H8_D…AlphaFlowODE_9D_as1_ae0_bbunet
        sit             mix_uav_af/H8_D…AlphaFlowODE_9D_as1_ae0_bbsit
        alpha0.2        mix_uav_af/H8_D…AlphaFlowODE_9D_as1_ae0.2_bbunet
(d) epoch leaves ckpt     : True  | plan epoch = latest
(e) pre-U6 name unchanged : True  -> Eaf_K20_mpc4_pid_stopgo_T0.5
(e) latest strict-extends : True  -> Eaf_K20_mpc4_pid_stopgo_T0.5_EPlatest
(e) run_tag stays LAST    : True  -> Eaf_K20_mpc4_pid_stopgo_T0.5_EPlatest_ab1
(f) mf_dit / unett / 1.5 / 0.001 / abc / lastest  -> all rejected
```

**The DA tooling needs no change** — checked against the real regexes in
`Data_Analysis/DA_UAV_v1/config.py`:

| folder | parses as |
|---|---|
| `Eaf_K1_mpc4_pid_stopgo_T0.5_EPlatest` | `K=1`, `engine=af`, `run_tag=EPlatest` ✅ |
| `…_T0.5_EPlatest_afu6` | `K=20`, `run_tag=EPlatest_afu6` ✅ |
| `H8_D…_as1_ae0.2_bbunet` | `backbone=unet`, `alpha_end=0.2` ✅ |

`_EP<sel>` lands in the existing `run_tag` group (added by the Fix_16 DA), which is already a real
axis in `K_SWEEP_KEYS`. **This is the failure mode Fix_16's DA §0.9 hit** — a tagged folder the
regex did not match, dropping six runs out of `candidates_detailed.csv` with no warning. It does
not recur here, but re-check it on the first tagged batch rather than trusting this table.

---

## 4. What was NOT changed

- `fm`, `mf` and `diffusion` arms: untouched. Their checkpoint paths are byte-identical
  (`mix_uav_mf/H8_D…MeanFlowODE_9D_dp0.5_bbunet`, verified).
- The α schedule shape, `af_alpha_clamp`, `af_ratio_fm`, `af_clamp_utgt`, the projector, the
  constraint YAML, the controller, the horizon — all untouched.
- The `AFTrajectoryModel` constructor default stays `'sit'`. It is a defensive fallback for a
  *missing* config key, and that rationale still holds; U6 changes what the config passes.
- Fix_16 (`FMPCC_SAFE_EPS_*`) is orthogonal and unchanged.

---

## 5. Verification — what to submit

### Stage 0 — the gate (seconds, CPU, runs on a login node)

```bash
python mix_uav_test/gates_mix_uav.py --gates G2 G9 --device cpu
```

**Pass condition:** `G9 PASS` and `G2 PASS`. G2 is included because U6 changes what the af arm puts
in its path, and G2 is the gate that owns checkpoint-path distinctness.

### Stage 1 — the free run: **no training**, and it settles §1.2 on the record

Re-evaluate the **existing SiT checkpoint** at `latest`. It reads the `_ae0_bbsit` tree that jobs
25135–25138 wrote, so nothing is retrained, and it writes to a new `_EPlatest` folder.

```bash
UAV_MIX_BONE_AF=sit UAV_MIX_EPOCH=latest FMPCC_UAV_EVAL_TAG=u6sitlatest \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af pillars "6" "1 2"
```

**What to read, in order:**

| log line | expect |
|---|---|
| `[ U6 ] af bone = sit` / `checkpoint = latest` | the knobs arrived |
| `[ eval ] checkpoint = state_80000.pt (trained to step 80000)` | **80000, not 100000** — this checkpoint predates §2.4 |
| `[ eval ] alpha(step 80000) = 0.0000 … 🔴 trained on the MeanFlow target` | ✅ **the point of this run.** Machine-readable proof, in an eval log, that every published UAV af row is MeanFlow-on-SiT |
| `[ eval ] ml bone = sit (… ~9.4 M and NOT parameter-matched)` | the second half of the same finding |

It also gives the `best`-vs-`latest` A/B against the 2026-08-30 pillars numbers for the price of one
eval. ⚠️ Those old numbers are **pre-Fix_16**; this run is post-Fix_16, so treat any difference as
*epoch + eps together* until an `--epoch best` re-run separates them.

### Stage 2 — the actual fix. **Training is required** (new `_bbunet` tree)

`uav_mix_ksweep_pipeline.sh` trains **once** and fans out one eval per K on `afterok`, so a K list
costs one training job, not N.

**Scene order is deliberate — `s_curve` first, and here is why:**

| scene | rankable? | why |
|---|---|---|
| **`s_curve`** | ✅ | a three-way `fm`/`mf`/`diffusion` DA exists (`DA_20260827`), so the **pinned DPCC target** is available — per `da-target-is-best-baseline-variant` a claim needs it |
| **`pillars`** | 🔴 **no** | **0 / 2876 success+constraint rollouts**, every engine, every K, both Fix_16 arms (`DA_20260903_fix16_AB` §0.5). And **no `diffusion` arm exists for pillars at all**. Only abort rate, goal distance and tracking error are readable there — never a ranking |

```bash
# ── s_curve, arm A: the ARCHITECTURE fix alone (alpha still off) ──────────────────────
# The control. Without it, arm B moves objective AND backbone at once and proves nothing.
UAV_MIX_BONE_AF=unet FMPCC_UAV_EVAL_TAG=u6unet_ae0 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh \
  af s_curve 6 "" fm_only none "1 2 5"

# ── s_curve, arm B: alpha ACTUALLY ON, endpoint deployed (the avoiding recipe) ────────
UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest \
  FMPCC_UAV_EVAL_TAG=u6unet_ae02 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh \
  af s_curve 6 "" fm_only none "1 2 5"
```

Then the same two on `pillars`, **K = "1 2" only** — both K5 jobs in the Fix_16 A/B died at the 24 h
wall on this scene:

```bash
UAV_MIX_BONE_AF=unet FMPCC_UAV_EVAL_TAG=u6unet_ae0 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh \
  af pillars 6 "" fm_only none "1 2"

UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest \
  FMPCC_UAV_EVAL_TAG=u6unet_ae02 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh \
  af pillars 6 "" fm_only none "1 2"
```

*(positional args: `<engine> <scene> <seed> <n_trials> <projection> <record> "<K list>"`;
`n_trials=""` takes the yaml's 20.)*

**Cost:** 4 training jobs (one per scene × arm — Gen15 trains per scene, there is no shared model),
plus 10 eval jobs. If the queue is tight, run **s_curve arms A and B first**: pillars cannot rank
anything, so it is the lower-value half.

**Expected trees:**

| arm | checkpoint | results |
|---|---|---|
| A | `…AlphaFlowODE_9D_as1_ae0_bbunet/6/` | `Eaf_K<k>_mpc4_pid_stopgo_T0.5_u6unet_ae0/` |
| B | `…AlphaFlowODE_9D_as1_ae0.2_bbunet/6/` | `Eaf_K<k>_mpc4_pid_stopgo_T0.5_EPlatest_u6unet_ae02/` |

**Train-log gates before reading a single task number** (arm B):

| signal | ❌ α off | ✅ α on |
|---|---|---|
| `[ U6 ] af_alpha_end = 0.2` | absent | present |
| `val/alpha`, final epoch | `0.0` | ≈ `0.2` |
| **`train/discrete_frac`, final epochs** | **`0.0`** | **> 0** (tracks `af_ratio_fm = 0.5`) |
| savepath | `_ae0_` | `_ae0.2_` |

**Eval-log gate:** `checkpoint = state_100000.pt` (§2.4's final save) then
`alpha(step 100000) = 0.2000 … alpha-Flow objective ACTIVE`. If it reads `state_80000.pt`, the
trainer edit did not reach the cluster — the run is still usable (α is at its floor well before
80 k) but say so in the DA.

### Stage 3 — the comparison

At **matched K**, against `mf` and `fm` on the same scene and seed, and against the pinned DPCC
target on `s_curve`. Arm A vs the existing SiT rows isolates the **backbone**; arm B vs arm A
isolates the **objective**. Only arm B may be called α-Flow.

🔴 **`fm` and `af` have never been re-run with Fix_16** (`DA_20260903_fix16_AB` §0.8). The new af
runs are post-fix; the `mf` rows in that A/B are post-fix; the existing `fm`/`af` corpus is not.
Re-run `fm` on the same scenes with `FMPCC_SAFE_EPS_MODE=scaled` before any cross-engine table, or
say plainly which rows are pre-fix.

---

## 6. Risks and open items

1. **The default flip breaks bare `af` invocations until the U-Net arm is trained.** By design
   (§2.2), but it will surprise anyone who submits `... af <scene>` from memory. The failure is a
   missing checkpoint directory naming `_bbunet`.
2. **`pillars` still cannot rank anything** — 0/2876 S&C, and no DPCC target arm exists for it.
   Fix_16 fixed the `mf` divergence; it did not make the scene solvable.
3. **Fix_16 costs 1.2–1.6× per-step projection time**, so the K5 wall-clock risk on `pillars` is
   real and is why the K list is trimmed there.
4. **Single seed (6)**, matching the existing Gen15 corpus. The Gen3v7 result this recipe comes from
   is **itself n = 1 seed** — replicating that on `avoiding` is still the higher-value job
   (investigation §4 step 0).
5. **Disk:** §2.4 adds one checkpoint per completed run, but `clean_weights.py` keeps
   `state_best.pt` + the highest-numbered file, so a pruned tree is the same size as before —
   holding the endpoint instead of step 80 000. Check free space anyway: Stage 2 is four
   training runs and `/data` was at 100 % during the Gen3v7 runs.
6. **Relabelling costs nothing and should happen regardless of these runs:** every existing Gen15
   `af` row should be annotated *"MeanFlow objective on a ~9.4 M SiT"* wherever it is cited.
7. **Not run on the cluster.** Everything in §3 is local static verification — config exec with
   stubs, `_uav_eval_tag` in isolation, the DA regexes, `bash -n`, and `py_compile`. Nothing has
   touched a GPU.
