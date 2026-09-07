# CHANGELOG 2026-08-20 — `B4_PARITY`: arm C's MPC candidate fan is now decided by the variant name

**Date:** 2026-08-20 · **Severity:** 🔴 **P0** · **Type:** correctness fix, cross-generation
**Scope:** every LIVE HardFlow (arm C) implementation — Gen3v6, Gen3v7, Gen12, Gen14, Gen15
**Trigger:** `logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260820_HF_lower_avgtime_batchsize_confound.md`
**Validation:** ⚠️ syntax + unit-checked in-container only. **Everything numeric here must be
re-run on the cluster** (this container has no Python packages / GPU / MuJoCo).

---

## 1. The bug

The HardFlow arm ran a **1-candidate** MPC fan while the DPCC and `diffuser` arms ran **4**.

Both arms loop **serially over candidates** around their CPU solve:

| arm | solver | loop |
|---|---|---|
| B — DPCC | scipy SLSQP | `sampling/projection.py::Projector.project` → `for i in range(batch_size)` |
| C — HardFlow | CasADi / IPOPT | `sampling/hardflow_projection.py::HardFlowSampler.sample` → `for b in range(batch_size)` |

The network evaluations are batched on the GPU; the NLP solves are not. So the fan multiplies
projection wall-time almost linearly and barely touches anything else. A 1-vs-4 fan is a **4×
compute discount handed to arm C**, and it landed directly on the headline metric.

**What it produced.** `avg_time` on `avoiding`, MeanFlow-UNet K10, 5 seeds × 20 trials
(C136, `both-hard`):

| arm | fan | s/step |
|---|---:|---:|
| `dpcc-c` | 4 | 0.324 |
| `hardflow_new-r` | **1** | **0.243** ← reads as "HardFlow is 25 % cheaper" |

The truth, from the same batch file at a **matched** fan (C109 / C117, same checkpoint, same
`K=10`, same `A=0.5`, `HFFM_BATCH=4`):

| arm | fan | s/step |
|---|---:|---:|
| `dpcc-c` | 4 | 0.252 – 0.259 |
| `hardflow_new-r` | **4** | **0.495 – 0.508** ← HardFlow is ~2× **slower** |

Per individual solve, HardFlow's IPOPT costs **11.5 ms → 20.4 ms**, i.e. **≈1.8–2.2× DPCC's
SLSQP**, and arm C additionally burns **15 NFE per plan against arms A/B's 10** (it needs a second
terminal velocity eval on every active step). Arm C is the more expensive arm on **both** axes.

### Where the 1 came from

Three independent places all defaulted to 1, so fixing one would not have been enough:

| # | location | was | now |
|---|---|---:|---:|
| 1 | `config/{meanflow,alphaflow,hardflow}_projection_eval.yaml` → `hardflow.batch_size` | `1` | **`4`** |
| 2 | driver fallback `hardflow_cfg.get('batch_size', 1)` (×3 drivers) | `1` | **`4`** |
| 3 | `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` → `HFFM_BATCH:-1` | `1` | **`4`** |

**#3 is what actually produced the shipped `B1` runs** — the Slurm entrypoint pinned it, so the
yaml value was never even consulted. `Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh`
already exported `HFFM_BATCH:-4`, so **Gen3v7 never emitted a B1 run.**

---

## 2. The rule now in force

**The variant NAME decides the fan.** Implemented once per generation as
`sampling/hardflow_projection.py::resolve_hf_batch_size(variant, configured_batch)`:

| variant | fan | meaning |
|---|---:|---|
| `hardflow_new` | **1** | faithful upstream batch-1 control (upstream asserts `batch == 1`) |
| `hardflow_new-r` / `-c` / `-t` | `configured_batch` (**4**) | a selection *rule* is only meaningful over a fan; asking for one asks for the fan |
| `…-tightened`, `…_train_set` | composes | bookkeeping/geometry suffixes, stripped before parsing |
| anything else | **raises `ValueError`** | arms A/B read `args.batch_size` directly and must never route through here |

Why the bare name is pinned to 1 rather than left to follow the default: **at B>1,
`hardflow_new` and `hardflow_new-r` are byte-identical** — both select index 0. Running both at
the same fan is duplicated compute under two names. Pinning the bare name is what gives it a
distinct, meaningful identity.

`HFFM_BATCH` still overrides the configured fan (for sweeps); it does **not** override the bare
arm's 1.

---

## 3. Files touched (22)

### 3.1 Samplers — the rule (5 sibling copies, per the copy-modify convention)
Added `resolve_hf_batch_size()` immediately after `resolve_activation_threshold()`; **identical
text in all five**, no shared-code refactor across generations.

- `flow_matcher_v3_meanflow/sampling/hardflow_projection.py`   (Gen3v6)
- `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py`  (Gen3v7)
- `flow_matcher_v3_hardflow/sampling/hardflow_projection.py`   (Gen12)
- `mix_uav/sampling/hardflow_projection.py`                    (Gen15)
- `mix_visual_aligning/sampling/hardflow_projection.py`        (Gen14)

Re-exported from the four `sampling/__init__.py` packages that already re-export
`resolve_activation_threshold` (`mix_uav/sampling/__init__.py` exports no HF symbols; its driver
imports from the module directly, unchanged).

### 3.2 Eval drivers — call sites

| driver | gen | change |
|---|---|---|
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | Gen3v6 | 🔴 fan was `hf_batch_size` (1) vs arms A/B's 4 |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | Gen3v7 | 🔴 same code path (sbatch saved it in practice) |
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | Gen12 | 🔴 same code path |
| `mix_uav_test/eval_mix_uav.py` | Gen15 | ✅ already matched; bare arm now resolves to 1 |
| `mix_visual_aligning_test/eval_mix_visual_aligning.py` | Gen14 | ✅ already matched (U7); bare arm now resolves to 1 |

In each: `batch_size = hf_batch_size` → `batch_size = resolve_hf_batch_size(variant, …)`, plus a
**mismatch warning** printed whenever the resolved arm-C fan differs from `args.batch_size` — so
a future mismatch announces itself in the job log instead of hiding in a timing table.

The three Gen3v6/v7/12 drivers also had their fallback `hardflow_cfg.get('batch_size', 1)` raised
to `4`, so a yaml with the key **missing** also lands on parity.

### 3.3 Configs
- `config/meanflow_projection_eval.yaml` — `hardflow.batch_size: 1 → 4`
- `config/alphaflow_projection_eval.yaml` — `1 → 4` (its stale "leaving this at 1 reproduces the fix_3 confound" note removed — that is now the thing that cannot happen)
- `config/hardflow_projection_eval.yaml` — `1 → 4`
- `config/uav_mix.py` — `hardflow_variants: ['hardflow_new', …]` → `['hardflow_new-r', …]`.
  **Zero numeric change** (at B=4 bare ≡ `-r`, both index 0) — it is a rename to an honest name.
  Bare `hardflow_new` remains available and now means B=1.
- `config/visual_aligning_eval.yaml` — **unchanged**; Gen14 has no `hardflow.batch_size` key and
  correctly reads `args.mpc_batch_size`.

### 3.4 Slurm entrypoints
- `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` — 🔴 `HFFM_BATCH:-1` → **`:-4`**. *The
  single line that produced every shipped `B1` result.*
- `Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh` — already `:-4`; rationale comment
  rewritten to the repo-wide rule.
- `Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh` — added an explicit
  `export HFFM_BATCH="${HFFM_BATCH:-4}"` + echo. It previously exported nothing and inherited the
  yaml silently; the fan now appears in the job log.
- `Slurm_Codes/sbatch/hardflow/*` — **untouched.** Those drive the vendored upstream
  `HardFlow/run/*.py`, which has its own batch semantics (upstream asserts `batch == 1`). Not an
  FMPCC port; out of scope on purpose.

### 3.5 Gate
`FM_v3_meanflow_test/gates_hardflow_meanflow.py::gate_h4` — new, wired into the default run.
Pins all 11 name→fan cases and asserts `dpcc-*` / `diffuser` are **refused**. Pure-python (no
torch, no GPU), so it is cheap to keep in the pre-flight chain.

---

## 4. 🔴 OPEN, NOT FIXED HERE — `-c` is a bad arm at B > 1

Making B=4 the default makes this reachable by default, so it must be read before trusting any
new `-c` number. Pooled over the **750 arm-C cells that already ran at B=4** in the five
2026-08-11…08-19 avoiding batches:

| selection | S&C | succ | steps | timeouts (steps > 150) |
|---|---:|---:|---:|---:|
| `-r` | 0.707 | 0.917 | 67.7 | **0 / 750** |
| `-t` | 0.707 | 0.883 | 71.2 | 5 / 750 |
| **`-c`** | **0.443** | **0.540** | **138.5** | **370 / 750 (49 %)** |

**Mechanism.** `candidate_costs` is `Σ_k ‖x1_proj − x1_ref‖²`, so `argmin` selects the candidate
the NLP *barely had to touch*. On `avoiding` that is the candidate that barely **moves** — it is
trivially dynamics-feasible and far from every halfspace — so the episode stalls. DPCC's own
`-c` does not degenerate this way: it ranks a near-final denoised sample, where all four
candidates are already plausible trajectories, whereas arm C starts accumulating at τ = 0.5 where
the iterate is still half noise and the sum is dominated by how unconverged a candidate is rather
than by constraint conflict.

**Not fixed here** because changing the ranking key is a science change that needs cluster
validation, not a config flip. What is in place instead: the drivers now print a loud
`🔴 KNOWN-BAD arm` warning whenever `-c` runs at B>1. Deciding between (a) fixing the ranking key,
(b) dropping `-c` from arm C's default variant list, or (c) reporting it with a permanent caveat
is a **call for the next session** — see §6.

---

## 5. What this does to existing data

**Nothing silently.** Old runs are identifiable and remain valid *as what they are*:

- The results-folder token **`B1`** (from `args_to_watch_fmv3_hf_plan`'s `('hf_batch_size', 'B')`)
  marks every affected Gen3v6/Gen12 run. `B4` marks the matched ones.
- Each `<variant>.npz` also records the **per-variant** local fan under the `hf_batch_size` key,
  so a DPCC row reads 4 and a HardFlow row reads 1 in the same directory. That is what made the
  confound invisible in aggregate tables — and it is also what makes every old run re-classifiable
  without re-running it.

**Consequently:** no historic result needs deleting. What needs withdrawing is the *claim*, not
the data — any arm-B-vs-arm-C **wall-clock** comparison drawn from a `B1` directory. See the DA
§7 table for the specific claims.

⚠️ **Folder-name caveat.** The `B` token is run-level (it is the configured fan), while the bare
`hardflow_new` arm is now pinned to 1 regardless. A future run containing **both** bare and
suffixed arms will therefore sit in a `B4` folder while its bare arm ran at 1. The per-variant
`hf_batch_size` inside each npz is the authoritative value — analysis code should read that, not
the path. (Gen3v6/v7/Gen12 do not currently list the bare arm in `projection_variants`, so no
present config hits this.)

---

## 6. Next steps — all "run on cluster"

1. **Pre-flight:** `python FM_v3_meanflow_test/gates_hardflow_meanflow.py` → H0/H1/H3/**H4** pass.
2. **Smoke:** one seed, `n_trials=2`, K=10 on Gen3v6. Confirm in the log:
   `[ hardflow ] HFFM_BATCH=4`, no `⚠️ arm-C fan …!=` line on the `-r/-c/-t` arms, and the
   `🔴 -c … KNOWN-BAD` line on the `-c` arms. Results dir should read `…_A0.5_B4_…`.
3. **Decide the `-c` question (§4)** before spending a full sweep on it.
4. **Re-run the headline:** C136's config at `HFFM_BATCH=4`, 5 seeds × 20 trials, K=10, A=0.5 —
   the matched-batch arm-C number at full statistics. C109/C117 already answer it at 1 seed × 2
   trials; this is the citable version.
5. **Consider also running arms A/B at `batch_size: 1`.** Reporting the whole comparison at B=1
   is the "faithful batch-1" reading, and it is far cheaper than a B=4 arm-C sweep. Having both
   lets the paper report projection cost *as a function of the candidate fan*, which is the
   strongest form of the result.
6. **Rename the npz field** `hf_batch_size` → `mpc_batch_size` (keep the old key as an alias for
   historic npz files). Storing the local fan of *whichever* arm wrote it under an `hf_`-prefixed
   name is exactly what hid this bug for weeks. **Not done here** — it touches the DA loaders
   (`Data_Analysis/DA_UAV_v1/data_loader.py`, `DA_VA_v2/config.py`, the `FM_v3_*_test/load_results_*`
   scripts) and deserves its own change.

---

## 7. Not touched, deliberately

- `Archived_Codes/`, and anything marked legacy / Abandoned / Outdated — dead code.
- `HardFlow/` (vendored upstream) and `Slurm_Codes/sbatch/hardflow/*` — upstream reference, own
  batch semantics.
- `config/visual_aligning_eval.yaml` — already correct.
- The `-c` ranking key itself — see §4.
- `logs_in_develop/MASTER_TEST_HISTORY.md` — not self-edited by convention. **This entry is
  offered for the index; add it if you want it there.**
