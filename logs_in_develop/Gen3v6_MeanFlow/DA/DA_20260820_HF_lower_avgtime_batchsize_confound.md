# DA 2026-08-20 — Anomaly: why is HardFlow's `avg_time` *lower* than DPCC's in CAND_136?

**Date:** 2026-08-20 · **Type:** anomaly triage / data analysis
**Status:** ✅ Explained — **not a HardFlow speedup. It is the `fix_3` batch-size confound.**
🔧 **FIXED 2026-08-20** in code/config/sbatch across all live HF generations —
`logs_in_develop/HF_Batch_Parity/CHANGELOG_20260820_HF_batch_parity.md`. The data below is
unchanged and still describes what the `B1` runs are.

**Batch:** `temp/1908/batch_avoiding_combined_20260819_214620/`
**Candidate under question:** **C136**
`H8_K10_Meuler_T0.5_A0.5_`**`B1`**`_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials`
`logs/avoiding-d3il/plans/flow_matching_v3_meanflow/H8_D…MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5/`
seeds 6–10 · `n_trials = 20` · run 2026-08-13 → 2026-08-14 20:49:32

---

## 0. TL;DR

**HardFlow is not faster. It is running ¼ of the work.**

In C136 the HardFlow arm ran with **MPC candidate batch `B = 1`**, while the DPCC and `diffuser`
arms ran with **`B = 4`** (`args.batch_size` from the `plan_fm_v3_meanflow` config block). The
projection solve is a **serial per-candidate loop in both arms**, so DPCC pays 4 NLP solves per
active step and HardFlow pays 1. That single factor more than covers the observed gap.

Per **individual** solve HardFlow's IPOPT is **~1.8–2.2× more expensive** than DPCC's scipy SLSQP.
When the batch is matched (`B = 4`, candidates **C109 / C117** in the *same batch file*, same
checkpoint, same `K = 10`, same `A = 0.5`), **HardFlow is ~2× SLOWER than DPCC**, exactly as
expected.

The answer is literally printed in the folder name: **`B1`** vs the DPCC arms' `batch_size: 4`.

---

## 1. The observation

`both-hard`, C136, 5 seeds × 20 trials (`s/ep` = `n_steps × avg_time`):

| variant | **B** | n seeds | S&C | steps | **s/step** | s/ep |
|---|---:|---:|---:|---:|---:|---:|
| `diffuser` | 4 | 5 | 0.170 | 65.55 | **0.0937** | 6.14 |
| `dpcc-r` | 4 | 5 | 0.460 | 74.96 | 0.3159 | 23.65 |
| `dpcc-c` | 4 | 5 | 0.590 | 63.12 | 0.3236 | 20.40 |
| `dpcc-t` | 4 | 5 | 0.550 | 62.69 | 0.3403 | 21.33 |
| `dpcc-r-tightened` | 4 | 5 | 0.840 | 66.15 | 0.3691 | 24.42 |
| `dpcc-c-tightened` | 4 | 5 | 0.860 | 59.51 | 0.3711 | 22.08 |
| `dpcc-t-tightened` | 4 | 5 | **0.970** | 58.12 | 0.3836 | 22.29 |
| `hardflow_new-r` | **1** | 5 | 0.500 | 66.91 | **0.2426** | 16.22 |
| `hardflow_new-c` | **1** | 4 ⚠️ | 0.450 | 66.91 | 0.2515 | 16.83 |
| `hardflow_new-t` | **1** | 4 ⚠️ | 0.450 | 66.91 | 0.2483 | 16.62 |
| `hardflow_new-r-tightened` | **1** | 4 ⚠️ | 0.637 | 65.78 | 0.2631 | 17.31 |
| `hardflow_new-c-tightened` | **1** | 4 ⚠️ | 0.637 | 65.78 | 0.2690 | 17.71 |
| `hardflow_new-t-tightened` | **1** | 4 ⚠️ | 0.637 | 65.78 | 0.2658 | 17.50 |

Two tells were visible before any code reading:

1. **`-r` / `-c` / `-t` are numerically identical** inside each geometry group (identical `n_steps`,
   `nfe_total`, `nlp_solves_total`, violations). `HardFlowPolicy._select` returns index 0 when
   `batch_size == 1`, so all three selection rules collapse. That only happens at **B = 1**.
   (`flow_matcher_v3_meanflow/sampling/hardflow_projection.py:684-686`.)
2. The `hf_batch_size` field recorded **per variant** reads **4 for the DPCC/diffuser rows and 1 for
   the HardFlow rows** — see §2.

---

## 2. Mechanism — where `B` diverges

`FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:466-497`:

```python
if is_hardflow:
    batch_size = hf_batch_size        # <- hardflow.batch_size from the yaml (default 1)
    ...
else:
    batch_size = args.batch_size      # <- 4, from plan_fm_v3_meanflow
```

and at save time (`:733`) the local batch is written out under a misleading key:

```python
hf_batch_size=int(batch_size),   # records the LOCAL batch of WHICHEVER arm
```

which is why the aggregated CSV shows `hf_batch_size = 4.0` on `dpcc-*` rows and `1.0` on
`hardflow_new-*` rows. Sources of the two values:

| | value | source |
|---|---:|---|
| arms A/B (`diffuser`, `dpcc-*`) | **4** | `config/avoiding-d3il.py:1384` — `plan_fm_v3_meanflow['batch_size'] = 4` |
| arm C (`hardflow_new-*`) | **1** | `config/meanflow_projection_eval.yaml` — `hardflow.batch_size: 1` (env `HFFM_BATCH` unset) |

**Both arms loop serially over candidates around the CPU solve:**

* DPCC — `flow_matcher_v3_meanflow/sampling/projection.py:131` → `for i in range(batch_size)` (scipy SLSQP).
* HardFlow — `hardflow_projection.py:518` → `for b in range(batch_size)` (CasADi/IPOPT).

The network evals are batched on the GPU in both arms, so **the batch multiplies the NLP cost
almost linearly and the network cost barely at all**. Hence 4× fewer solves ⇒ HardFlow looks cheap.

### 2.1 The activation gate is *not* the difference

Both arms use `threshold = 0.5` with `K = 10`, and both floor identically:

* DPCC: `snapping_start_idx = int((1-0.5)*10) = 5`; active at `loop_idx ∈ {5..9}` → **5 active steps** (`mf_diffusion.py:284-285`).
* HardFlow: `k >= int((1-0.5)*10) = 5` → `k ∈ {5..9}` → **5 active steps** (`hardflow_projection.py:511`, Gen12fix8 rounding parity).

Confirmed numerically from the counters: HardFlow's per-plan cost is `K + n_active = 15` NFE and
`n_active = 5` NLP solves, at `B = 1`:

```
nfe_total      = 20 trials × ~67.9 plans × 15 × 1  = 20 366   (measured 20 374)
nlp_solves_tot = 20 trials × ~67.9 plans ×  5 × 1  =  6 790   (measured  6 791)
nfe/nlp = 3.00 exactly  ⇒  K = 2·n_active  ⇒  K = 10, B = 1. ✔
```

So the solve *counts per candidate* match. Only the candidate count differs: **DPCC 5×4 = 20
solves/plan, HardFlow 5×1 = 5 solves/plan.**

### 2.2 HardFlow is *more* expensive on the network side too

HardFlow needs two velocity evals on every active step (`v(x_k, τ_k)` and the terminal
`v(x_ref, τ_{k+1})`), so it burns **15 NFE per plan vs the DPCC/diffuser arms' 10** — a 1.5×
generative-side penalty that the B = 1 discount is currently hiding.

---

## 3. Cost decomposition (C136, `both-hard`, per seed)

`dpccProj = t(dpcc-c) − t(diffuser)` over 20 solves · `hfProj = t(hf-r) − 1.5·t(diffuser)` over 5 solves
(the `1.5×` removes HardFlow's 15-NFE network cost, priced at the diffuser's per-call rate).

| seed | diffuser | dpcc-c | hf-r | dpcc proj (s) | hf proj (s) | **ms / SLSQP solve** | **ms / IPOPT solve** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.0960 | 0.3307 | 0.2483 | 0.2347 | 0.1043 | 11.73 | **20.86** |
| 7 | 0.0972 | 0.3353 | 0.2570 | 0.2381 | 0.1112 | 11.90 | **22.24** |
| 8 | 0.0913 | 0.3241 | 0.2419 | 0.2328 | 0.1049 | 11.64 | **20.97** |
| 9 | 0.0914 | 0.3119 | 0.2325 | 0.2205 | 0.0954 | 11.02 | **19.08** |
| 10 | 0.0924 | 0.3162 | 0.2332 | 0.2238 | 0.0946 | 11.19 | **18.92** |
| **mean** | 0.0937 | 0.3236 | 0.2426 | 0.2300 | 0.1021 | **11.50** | **20.41** |

The same computation on `top-left-hard` / `top-right-hard` gives 10.3–12.3 ms vs 18.6–25.6 ms —
the ratio is stable at **≈1.8×** across all 15 seed×geometry cells.

**Predicted matched-batch HardFlow time:** `1.5 × 0.0937 + 4 × 0.1021 = 0.549 s/step`,
i.e. **1.70× `dpcc-c`** and **1.48× `dpcc-c-tightened`**.

---

## 4. ✅ The empirical control — same batch file, HardFlow at B = 4

**C109** (`20260730_164148`) and **C117** (`20260801_004351`) are the *same* MeanFlow-UNet
checkpoint at the *same* `K = 10`, `T = 0.5`, `A = 0.5`, but with **`HFFM_BATCH = 4`**, so arm C ran
the full 4-candidate fan. Seed 6 only, `n_trials = 2`. (Their folder names predate the `A`/`B`
provenance tokens — which is precisely the problem FIX_9_CFG_PROVENANCE was added to fix.)

`both-hard`:

| variant | **B** | C109 s/step | C117 s/step | C136 s/step (B=1) |
|---|---:|---:|---:|---:|
| `diffuser` | 4 | 0.0784 | 0.0825 | 0.0937 |
| `dpcc-c` | 4 | **0.2524** | **0.2589** | 0.3236 |
| `dpcc-c-tightened` | 4 | 0.3273 | 0.3371 | 0.3711 |
| `hardflow_new-r` | **4** | **0.4948** | **0.5078** | 0.2426 (B=1) |
| `hardflow_new-t` | **4** | 0.4666 | 0.4886 | 0.2483 (B=1) |
| `hardflow_new-r-tightened` | **4** | 0.4955 | 0.5122 | 0.2631 (B=1) |
| `hardflow_new-t-tightened` | **4** | 0.4989 | 0.5122 | 0.2658 (B=1) |

**At matched batch HardFlow costs 0.47–0.51 s/step against DPCC's 0.25–0.34 — i.e. ~1.5–2.0×
SLOWER.** Cross-check on the solve rate for C109: `dpccProj = 0.2524 − 0.0784 = 0.174` over 20
solves = **8.7 ms**; `hfProj = 0.4948 − 1.5×0.0784 = 0.377` over 20 solves = **18.9 ms** →
**2.17× per solve**, consistent with §3's 1.8×.

Its counters confirm B = 4: `nfe_total = 8100`, `nlp_solves_total = 2700`
(`= 2 trials × 67.5 plans × 15 × 4` and `× 5 × 4`). ✔

> ⚠️ Side finding, out of scope here: at **B = 4** the `hardflow_new-c` / `-c-tightened` arms
> collapse (161–164 steps = timeout, S&C 0.00–0.50, C109 **and** C117 identically). The `prox`
> ranking key is selecting a pathological candidate. `-r` and `-t` at B = 4 are fine. This must be
> resolved before anyone re-runs arm C at B > 1.

---

## 5. Secondary contributors (small, but real)

1. **Tightening is nearly free for HardFlow, expensive for DPCC.** C136 `both-hard`:
   DPCC `0.3236 → 0.3711` (**+14.7 %**) vs HardFlow `0.2426 → 0.2631` (**+8.5 %**). Post-hoc SLSQP
   starts from an infeasible final sample and eats more iterations as the margin grows; HardFlow's
   in-loop iterates are already near the feasible set at the terminal step. This is a genuine
   HardFlow property and is **not** part of the anomaly, but it does widen the apparent gap.
2. **Seed-10 truncation.** 5 of the 6 HardFlow arms are missing **seed 10** (only `hardflow_new-r`
   has all 5). Same failure pattern as `DA_20260815_ntrials20_stability_MF_UNet.md` §1 — the job
   ran out of wall-clock in the arm-C block. Effect on the mean is small and in the *conservative*
   direction (seed 10 is HardFlow's cheapest seed: dropping it raises `hf-r` 0.2426 → 0.2451), so
   it does not change any conclusion — but it does mean **HF and DPCC means are over different seed
   sets** in 5 of 6 arms, which is not citable as-is.

---

## 6. This confound is documented, and it was not closed

`config/meanflow_projection_eval.yaml`, `hardflow:` block:

> ```
> # PLAN §3.4: FAITHFUL BATCH-1 is the default headline (upstream asserts batch==1;
> # honest vs Gen13). U4.2 adds a real candidate fan: batch_size>1 solves one NLP
> # chain per candidate (wall time scales linearly) and selects among them by the
> # variant suffix (-c/-r/-t) — the same MPC machinery as DPCC. Set batch_size: 4
> # to close the fix_3 confound (B ran batch-4 + selection, C ran batch-1).
> batch_size: 1
> ```

C136 ran on the default. **`fix_3` was open for every `B1` candidate in this batch** — which is
C122–C140, C142, C143, C145–C147, i.e. essentially the whole current MeanFlow HardFlow sweep,
including the `msg20trials` rows.

> 🔧 **Closed 2026-08-20.** The default is now `batch_size: 4`, the fan is resolved per variant
> (`resolve_hf_batch_size`: bare `hardflow_new` → 1, `-r/-c/-t` → 4), and
> `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` — which pinned `HFFM_BATCH:-1` and is
> what actually produced these runs — now defaults to 4. Full record:
> `logs_in_develop/HF_Batch_Parity/CHANGELOG_20260820_HF_batch_parity.md`.
> **The candidates listed above are still `B1` data and must still be read as such.**

---

## 7. Consequences for the claims

| claim | status |
|---|---|
| "HardFlow has lower `avg_time` than DPCC" | 🔴 **Confounded — do not cite.** It is a 4×-fewer-candidates artefact. |
| "HardFlow's in-loop projection is cheaper than post-hoc DPCC projection" | 🔴 **False as measured.** Per solve it is 1.8–2.2× *more* expensive, and it needs 1.5× the NFE. |
| Any `s/ep` figure comparing `hardflow_new-*` against `dpcc-*` in this batch | 🔴 Withdraw or re-annotate with `B=1 vs B=4`. |
| HardFlow **S&C** vs DPCC | ⚠️ Also affected, in HardFlow's disfavour: with no candidate fan it cannot select. `both-hard` best HF = 0.637 vs `dpcc-t-tightened` 0.970; `top-right-hard` 0.860 vs 0.950. It only ties on `top-left-hard` (1.000). |
| The earlier "**13.4× on `avg_time`**" Target result (`DA_20260811…` §2.1, K1 `hardflow-tightened`) | ⚠️ **Same confound** — that DA also states `hf_batch=1` and notes `-r/-c/-t` were identical. It should be re-checked at matched `B` before it goes in the paper. |

---

## 8. What to do

> Items 1, 3 (partly) and 5 were actioned on 2026-08-20 — see the changelog. Struck items
> are done; the rest still stand.

1. **Re-annotate**, don't delete: every HardFlow row in this batch should carry `B` next to `K`
   and `A` in any table it appears in. The token is already in the folder name (`B1`); the reader
   just has to be told the DPCC rows are `B4`.
2. **Re-run C136's config with `HFFM_BATCH=4`** (5 seeds, `n_trials=20`, `K=10`, `A=0.5`) to get
   the matched-batch HardFlow number at full statistics. C109/C117 already answer the question, but
   at 1 seed × 2 trials. → **run on cluster.**
3. **Fix the `-c` collapse at B > 1 first** (§4 footnote), or step 2 will produce three unusable
   arms out of six.
4. **Alternative, cheaper and arguably more correct:** run the DPCC arms at `batch_size: 1` as well
   and report the whole comparison at B = 1. That is the "faithful batch-1" reading the yaml calls
   the honest default, and it costs a fraction of a B = 4 HardFlow sweep. Doing **both** would let
   the paper report the projection cost as a function of the candidate fan, which is the strongest
   version of the result.
5. **Rename the saved field.** `hf_batch_size` currently stores the *local* batch of whatever arm
   wrote it, so a `dpcc-c` row reports `hf_batch_size = 4`. That is what made this anomaly hard to
   see. Suggest `mpc_batch_size` (and keep `hf_batch_size` as an alias for old npz files).

---

## 9. Reproduce

```bash
cd temp/1908/batch_avoiding_combined_20260819_214620

# the batch each arm actually ran at
awk -F, '$1==136 && $8=="both-hard" && $9=="hf_batch_size"' \
    candidates_multidimensional_aggregated.csv | awk -F, '{print $6, $10}'

# the matched-B control
awk -F, '$7=="109" && $4=="both-hard" && $5=="avg_time"' \
    candidates_multidimensional_raw.csv | awk -F, '{print $2, $6}'
```

**Code refs:** `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:466,497,733` ·
`flow_matcher_v3_meanflow/sampling/hardflow_projection.py:511,518,684` ·
`flow_matcher_v3_meanflow/sampling/projection.py:131` ·
`flow_matcher_v3_meanflow/models/mf_diffusion.py:284` ·
`config/avoiding-d3il.py:1384` · `config/meanflow_projection_eval.yaml` (`hardflow.batch_size`).
