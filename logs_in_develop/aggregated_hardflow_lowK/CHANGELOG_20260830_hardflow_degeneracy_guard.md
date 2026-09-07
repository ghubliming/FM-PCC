# Changelog — degenerate HardFlow arms are now DISABLED, not merely warned about

**Date:** 2026-08-30 · **Tag:** `HFK1c` · **Type:** behaviour change (opt-out available), plus
warning-plumbing and DA fixes
**Scope:** AGGREGATED — all six live HardFlow ports and all six eval drivers, the UAV DA, and
four sbatch entry points.
**Driver:** [`AUDIT_20260830_lowK_warning_coverage_and_UAV_degeneracy_check.md`](./AUDIT_20260830_lowK_warning_coverage_and_UAV_degeneracy_check.md)
**Predecessors:** [`REGISTER_20260824_degenerate_HF_rows_and_warnings.md`](./REGISTER_20260824_degenerate_HF_rows_and_warnings.md) ·
[`CHANGELOG_20260824_hardflow_terminal_nfe_and_K1_guard.md`](./CHANGELOG_20260824_hardflow_terminal_nfe_and_K1_guard.md)
**Nothing was committed.** Working tree only, awaiting review.
**Nothing ran on the cluster.** Every claim below is by construction or from the stdlib self-test
in §5; the end-to-end paths need **a run on the cluster** to confirm.

---

## 0. One-line summary

> A HardFlow arm with `n_genuine == 0` no longer runs. It is dropped at config time with a
> logged reason and an `HF_DEGENERATE_SKIPPED.txt` sentinel; `FMPCC_HF_ALLOW_DEGENERATE=1`
> re-enables it for the one case that still needs it (the `A=0.0` projector control at matched K).

## 1. Why — the 2026-08-24 warning did not work

`[hardflow][DEGENERATE]` has been printed since 2026-08-24. AUDIT_20260830 checked what it
achieved:

* the Gen15 UAV K-sweep was **still** spending cluster hours on K=1/K=2 HardFlow cells;
* the flag reached cluster stdout, `results.json` and the DA loader log — but **not**
  `eval_<variant>.log`, **not** the file tree, and **not one** of the seven wide CSVs a ranking
  is actually read off;
* and the rows it was kept for do not deliver their control: 25 of 32 matched cells are
  `0.00 → 0.00` floor effects, and in the 7 cells with signal HardFlow is *worse* in 5.

A line in a 3000-line log is not a control. Two justifications for keeping the rows were examined
and both failed (AUDIT §6, including the correction of my own earlier claim that the
SLSQP-vs-IPOPT DA depended on them — it does not; that DA is K∈{10,20}, `hf_degenerate = 0`).

## 2. The threshold is `n_genuine`, not K

🔴 The shipped `A` is **not** uniform: Gen12 ships `A=1.0`, every other generation ships or
inherits `0.5`. So "K ≤ 2 is degenerate" is true for five generations and **false for Gen12**.
The guard therefore thresholds on `n_genuine = max(K − ⌊(1−A)·K⌋, 1) − 1`, which is A-aware by
construction and correct in every generation with one number.

| knob | default | effect |
|---|---|---|
| `FMPCC_HF_MIN_GENUINE` | `1` | block `DEGENERATE` (`n_genuine == 0`). `2` also blocks `THIN`. `0` disables the guard. |
| `FMPCC_HF_ALLOW_DEGENERATE` | unset | `1` runs the arm anyway. Only supported use: `A=0.0` at `K ≥ 5`. |
| `HFFM_ACT_THRESHOLD` | per-config | per-job override for `A`. **Newly wired into the UAV path** (R4). |

## 3. What changed, per file

### 3.1 The guard itself — 6 ports, byte-identical (R3)

`{flow_matcher_v3_alphaflow, flow_matcher_v3_hardflow, flow_matcher_v3_meanflow, mix_uav,
mix_visual_aligning, mix_visual_avoiding}/sampling/hardflow_projection.py`

New, inserted after `hardflow_regime` (which is unchanged):

| symbol | role |
|---|---|
| `HF_MIN_GENUINE_DEFAULT = 1` | the shipped bar |
| `HardFlowDegenerateError` | backstop exception |
| `hf_allow_degenerate()` / `resolve_hf_min_genuine()` | env resolution |
| `hardflow_guard(K, A, ...)` | `(ok, reason, tier, n_active, n_genuine, first_lookahead)` — pure arithmetic, safe to call during config assembly |
| `hardflow_skip_note(...)` | the sentinel text |

`HardFlowSampler.sample()` now calls `hardflow_guard` in place of `hardflow_regime` and
**raises `HardFlowDegenerateError`** when blocked. That raise is a *backstop*, not the primary
path — sweeps are expected to drop the arm at config time (§3.2) so the job never reaches it.
The existing `DEGENERATE` / `THIN` banners are untouched and still fire under the opt-in.

Re-exported from each `sampling/__init__.py` (`mix_uav` has no HF re-export; its driver imports
the module directly, so nothing was added there).

### 3.2 The drivers — drop the arm, don't crash (R3)

| driver | where the guard runs | on block |
|---|---|---|
| `mix_uav_test/eval_mix_uav.py` | config assembly, beside the existing `supports_hardflow` / `UAV_MIX_HF_OFF` drops | removes the HF variants from `projection_variants`; records `cfg['hardflow_skipped']`; writes `HF_DEGENERATE_SKIPPED.txt` into the geo dir |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | after `is_hardflow = ...` | `continue` + sentinel in `save_path` |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | ” | ” |
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | after `os.makedirs(save_path)` | ” |
| `mix_visual_avoiding_test/eval_mix_visual_avoiding.py` | after `is_hardflow = ...` | ” |
| `mix_visual_aligning_test/eval_mix_visual_aligning.py` | before the sampler build, beside the existing `resolve_engine_hf` refusal | ” |

The DPCC and diffuser arms at those same K **still run** — only the HardFlow variants are
dropped, so the low-K DPCC curve is unaffected.

🔴 In each driver the guard resolves `A` through the **same fallback chain** the sampler build
uses a few hundred lines later. Judging the arm with a different `A` than the one that runs was
the obvious way to get this wrong.

### 3.3 Visibility for arms that DO run degenerate (R1)

Only reachable under `FMPCC_HF_ALLOW_DEGENERATE=1`, but the opt-in is per **job** while the
artifacts are read per **variant**, weeks later — so the row carries its own warning:

* `mix_uav_test/eval_artifacts.py::write_eval_log` — a `!!!!` banner, mirroring the existing
  `PROJECTION CIRCUIT-BREAKER` / `DIVERGENCE ABORT` banners (**Gap A**, closed).
* `mix_uav_test/eval_mix_uav.py` — an `HF_DEGENERATE.txt` sentinel in the variant dir, mirroring
  `PROJECTION_CB_TRIPPED.txt`, plus a stdout line.

### 3.4 The DA — the flag now reaches the ranking tables (R2)

**Gap B**, closed. `hf_degenerate` existed only in `uav_units_long.csv`; every table used for
ranking had no degeneracy column, which is the mechanical reason a K=1 row could be promoted to
"HardFlow's best result".

* `Data_Analysis/DA_UAV_v1/aggregator.py` — new `_build_hf_flags()` (per-unit lookup) and
  `_attach_hf_flags(table, keys)`, applied to `k_sweep` (on `K_SWEEP_KEYS`) and `quality`
  (on `UNIT_KEYS`); `_candidate_hf_degenerate()` puts it on `candidate_stats`.
  **Aggregation rule: MAX over the group** — a cell pooling *any* degenerate unit is flagged,
  because a partially-degenerate cell is not citable either. Non-HardFlow rows get `0.0`, never
  NaN, so `hf_degenerate == 0` selects exactly the citable rows. Logs a warning naming the count.
* `Data_Analysis/DA_UAV_v1/reporter.py` — `hf_degenerate` column in `candidates_ranking.csv`;
  `hf_degenerate` / `hf_n_genuine` promoted into the lead columns of `data_quality.csv`, with
  the docstring explaining what the flag forbids.

No-op on batches with no HardFlow units.

### 3.5 sbatch (R3 + R4)

* `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` — **R4**: exports `HFFM_ACT_THRESHOLD`, which the
  UAV path never had (`config/uav_mix.py:247` hardcoded `0.5`, so `A` was the one HardFlow knob
  this generation could not sweep). Also exports both guard knobs and echoes all three.
  The HFK1 comment block's advice — *"keep the rows if you want the cheap one-shot-projection
  comparison"* — is explicitly **withdrawn** in place, with the reason.
  🔴 The `KS` default is deliberately left as `1 2 5 10 20`: the low-K **DPCC** points are a real
  curve, and only the HF arm is dropped.
* `Slurm_Codes/sbatch/mix_visual_avoiding/eval_k_sweep.sh` — same withdrawal + knobs.
* `Slurm_Codes/sbatch/{MeanFlow/eval_meanflow_hardflow.sh, AlphaFlow/eval_alphaflow_hardflow.sh}`
  — guard knobs added next to their existing `HFFM_ACT_THRESHOLD`.

### 3.6 R4 in the driver

`mix_uav_test/eval_mix_uav.py` reads `HFFM_ACT_THRESHOLD` during config assembly and overrides
`cfg['hardflow']['activation_threshold']`, so the guard and the policy see the same `A`. Same env
name and polarity as the sibling generations (higher = more projection; `0.0` = terminal-only).

This is what makes the replacement control runnable:

```bash
FMPCC_HF_ALLOW_DEGENERATE=1 HFFM_ACT_THRESHOLD=0.0   # projector-only arm, terminal-only at any K
#   vs the same K at A=0.5 (genuine HardFlow) and the DPCC arm
```

## 4. New gate — G7

`FM_v3_hardflow_test/gates_hardflow.py::gate_g7`, registered as `'G7 HFK1c guard'`. Pins:

(a) the guard blocks **exactly** `n_genuine == 0` over the 8×6 K×A grid — including `A=0.0`,
degenerate at every K; (b) `FMPCC_HF_ALLOW_DEGENERATE=1` re-opens; (c) `FMPCC_HF_MIN_GENUINE=2`
blocks `THIN`, `=0` disables; (d) the blocked reason names the escape hatch; (e) **all six ports
carry byte-identical guard arithmetic and the backstop raise** — a fix landing in one generation
and not its siblings is the recurring failure mode in this repo.

G7 restores the env vars it touches in a `finally`, so it cannot leak state into later gates.

## 5. Verification actually performed

Run **in this container** (stdlib only — no numpy/torch here):

```
budget table            OK  (12 cases, A in {0.0,0.5,1.0})   — matches REGISTER §1
guard default           OK  blocks 7 degenerate cells, allows the rest
A=0.0 blocked at all K  OK  (the projector control requires the explicit opt-in)
FMPCC_HF_ALLOW_DEGENERATE OK
FMPCC_HF_MIN_GENUINE    OK  (2 blocks THIN, 0 disables)
reason text             OK
all 6 ports identical   OK  (guard + backstop present in each)
G7(a) logic verified standalone over 48 cells — no mismatches
```

Plus: `ast.parse` on all 20 modified `.py` files and `bash -n` on all 4 modified `.sh` files.

### ⚠️ What has NOT been verified — **run on cluster**

* No eval, gate suite, or DA has actually executed. The full `gates_hardflow.py` needs
  numpy/torch/a GPU; only G7's arithmetic half ran here.
* The six driver call sites are correct **by inspection** — variable scope (`save_path`,
  `flow_steps`, `hf_act_threshold`, `diffusion_model`), and that `continue` inside each
  `try:`/`finally:` restores stdout. Not executed.
* The DA merge (`_attach_hf_flags`) is pandas code that has not been run against a real batch.
  Re-running the DA on `temp/3008/batch_uav_20260830_110536/` is the cheap check.
* First cluster job should be a K-sweep including K=1: expect `[hardflow][BLOCKED]`, an
  `HF_DEGENERATE_SKIPPED.txt`, no HF variant folders at K=1/2, and the DPCC arms unchanged.

## 6. Migration notes

* **Existing artifacts are untouched.** No result was edited, renamed, or deleted. The degenerate
  rows already on disk stay exactly where they are — R1/R2 make them *legible*, R3 stops new ones.
* **A sweep that previously produced HF rows at K=1/2 will now produce none.** That is the intent.
  A DA comparing old and new batches will see the HF arm vanish at low K; it has not "failed".
* **To reproduce an old degenerate row**, set `FMPCC_HF_ALLOW_DEGENERATE=1`. The output is
  numerically the same as before — the guard changes *whether* the arm runs, never *how*.
* Related: [`REGISTER_20260824`](./REGISTER_20260824_degenerate_HF_rows_and_warnings.md) still
  governs how to classify the **pre-existing** corpus; nothing in it is superseded.
