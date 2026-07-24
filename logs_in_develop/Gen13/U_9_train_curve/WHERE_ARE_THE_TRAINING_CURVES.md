# Where the Gen13 (iMF) and HardFlow-replication (FM) training-loss data lives

**Date:** 2026-07-20
**Question:** are the trainings linked to W&B? If not, where are the local loss curves?
**Short answer:** **No W&B anywhere.** iMF **was** trained and its full curve exists in two places. **FM was never successfully trained** — the replication used the authors' downloaded checkpoint — so no FM curve exists.

---

## 1. W&B — not connected

`grep -rn "wandb"` over `HardFlow/run/`, `HardFlow/hardflow/`, `HardFlow/run_scripts/` and `Slurm_Codes/sbatch/hardflow/*.sh` returns **nothing**. Neither the vendored HardFlow code nor any Gen13 code nor the sbatch bridge touches W&B. (The FMPCC-side sbatch scripts under `sbatch/iMF/` do have a W&B login block — that convention was **not** carried into the HardFlow bridge.)

## 2. iMF (Gen13) — ✅ TRAINED, curve available in TWO places

Job **23579**, 100 000 steps, **4 h 11 m 59 s**, finished cleanly (`final checkpoints saved with cp index 4`).

| Source | Path | Contents |
|---|---|---|
| **CSV (primary)** | `/u/home/llim/FMPCC/FM-PCC/logs/hardflow/avoiding-v0/flow/H16_imf_100k/metrics.csv` | 53 KB, ~500 rows (every 200 steps): `step, loss, raw_mse_u, raw_mse_v, a0_mse, fm_frac, h_mean` |
| **Job log (already local)** | `temp/Used/HF_iMF_first_run/23_57_33_hf_imf_train_23579.log` | the same 500 points as `[ train_imf ] step … raw_mse_u … raw_mse_v … a0 …` lines — **fully parseable, no cluster access needed** |
| Cluster copy of that log | `Slurm_Codes/logs/<date>/23_57_33_hf_imf_train_23579.log` | same |
| checkpoints | same dir: `model_{0..4}.pth`, `model_ema_{0..4}.pth` | cp 4 = final (used by every eval) |

**TensorBoard: not written.** The log line `[ train_imf ] tensorboard not installed -> metrics.csv only` confirms the try-import fallback fired — `hardflow_clone` has no `tensorboard`. Nothing was lost; the CSV carries every metric. To get TB on future runs: `pip install tensorboard` in the clone (`train_imf.py` picks it up automatically).

⚠️ **Read `raw_mse_u` / `raw_mse_v` / `a0_mse` — never `loss`.** The adaptive loss is bounded and flat by construction (it sat at 1.996–1.999 for the entire run). This is the documented Gen3v4 trap.

## 3. FM (HardFlow replication) — ❌ NEVER TRAINED, no curve exists

The replication used **Path B: the authors' released checkpoint** (`H16_1e6steps/model_ema_20.pth`, ~20 MB, downloaded manually), not local training.

The one attempt to train FM (pipeline job **23559**) **crashed after 4 seconds**:
```
[ HF-PIPE ] no checkpoint -> bash run_scripts/train.sh
run/train.py:12  from torch.utils.tensorboard import SummaryWriter
ModuleNotFoundError: No module named 'tensorboard'
Total runtime: 0 hours, 0 minutes, 4 seconds
```
Root cause (documented in `../fix_2/`): pre-existing `run/train.py` imports `SummaryWriter` at **module level**, not behind a try-import, so FM training cannot start in `hardflow_clone` without `pip install tensorboard`. (My `train_imf.py` guards it — which is exactly why iMF trained fine and fell back to CSV.)

**Consequence:** there is **no FM loss curve to compare against**, and the FM baseline's training history is whatever the paper authors did — unknown to us. Any "iMF vs FM training quality" comparison is therefore impossible from local data.

The replication design that chose this path: `../../HF_iMF/Code_RUN_Prepare/Replication/EVALUATION_hardflow_readiness_and_iMF_swap.md` §Part 1 (Path A train vs Path B download) and `.../fix_2/CHANGELOG_Gen13_fix2_*.md`.

## 4. What the iMF curve actually shows (parsed from the local log, 500 points)

| step window | median `raw_mse_u` | mean | min | max | median `a0_mse` |
|---|---|---|---|---|---|
| 0–10k | 20.65 | 24.92 | 13.41 | 139.4 | 0.500 |
| 10–25k | 17.83 | 22.38 | 10.37 | 188.1 | 0.440 |
| 25–50k | 15.85 | 22.91 | 8.54 | 157.9 | 0.305 |
| 50–75k | 15.50 | 18.02 | 9.23 | 120.7 | 0.287 |
| 75–90k | 14.47 | 26.38 | 9.61 | **556.9** | 0.276 |
| 90–100k | 15.18 | 22.66 | 8.69 | 132.6 | 0.266 |

**Median `raw_mse_u` moved only −5.4 % over the last 50 k steps (15.50 → 14.67).** It is **plateaued, not still descending.** Spikes persist to the end (7 % of logged points exceed 3× the median; one reaches 557).

### Bearing on the fix_7.3 verdict

fix_7.3 localised iMF's failure to **field/training error** (its `x̂1` error is flat in K at ≈0.155, 4–6× worse than FM's Euler shot). This curve **discriminates between the two explanations**:

| Hypothesis | Prediction | Observed | Verdict |
|---|---|---|---|
| **Undertrained** — more steps would fix it | `raw_mse_u` still descending at 100 k | **−5.4 % over the last 50 k; flat** | ❌ **not supported** |
| **Data ceiling** — 96 demos cannot determine a 2-time field | early drop then plateau well above zero, persistent variance | 20.7 → ~15 by 25 k, then flat with 7 % spikes | ✅ **supported** |

`raw_mse_u ≈ 15` is summed over 96 dims ⇒ **≈0.40 per-dim residual**, versus Gen3v4's ≈0.25/dim on the easier H8 task. The field is coarse and **stopped improving at ~25 k of a 100 k-step budget**.

**Conclusion: simply training longer will not rescue iMF here.** Consistent with the pre-registered "96-demo data ceiling" risk (Gen13 plan §7) and with the fix_7.3 recommendation not to invest further at this data scale.

*Caveat:* this rules out "more steps of the same recipe". It does **not** rule out a different recipe (constant-LR tail instead of cosine-to-zero, different `data_proportion`/`p_std`, more data). The persistent spikes are the known JVP predicted-v-tangent variance, and the LR annealed to 0 while still spiky — so a *better-conditioned* schedule remains untested.

## 4b. Three follow-up questions answered

### Q1 — "Is the HF replication wrong because we never trained?"

**No.** Using the authors' released checkpoint is **Path B**, an explicitly documented and legitimate replication route — HardFlow's own README links the `.pth` precisely so the eval can be reproduced without a 1e6-step training run. The replication verdict stands: `hardflow_new` reproduced the paper's headline (0 violations, 100% safe) — see `../../HF_iMF/Code_RUN_Prepare/Replication/fix_2/RUN_REPORT_original_baseline_eval.md` §7.

**But state the scope precisely:** we replicated their **evaluation**, not their **training**. We never verified their training procedure, and we have no FM loss curve. That is a limitation of scope, not an error.

### Q2 — "Does HardFlow's training code even HAVE a loss curve feature?"

**Yes, it has one — we simply never ran it.** `run/train.py`:
```python
from torch.utils.tensorboard import SummaryWriter          # line 12 (module level, NOT guarded)
writer = SummaryWriter(log_dir=.../ "tensorboard_logs")    # line 55
writer.add_scalar("loss", loss, i)                          # line 78 — every step
```
So the capability exists, but it is **weaker than the Gen13 version** in three ways:

| | HardFlow `train.py` (FM) | Gen13 `train_imf.py` (iMF) |
|---|---|---|
| Backend | **TensorBoard only** | TensorBoard **optional** + **always-on CSV** |
| Metrics | **one scalar: `loss`** | `loss`, `raw_mse_u`, `raw_mse_v`, `a0_mse`, `fm_frac`, `h_mean` |
| Behaviour without tensorboard | **crashes at import** | falls back to CSV, keeps training |

So: **"no curve exists" is because we never ran it — not because the feature is missing.** And had we run it, it would have produced a single-scalar TensorBoard log requiring `pip install tensorboard` first.

### Q3 — ⚠️ The training-budget asymmetry (a real caveat on fix_7.3)

Checking the FM checkpoint's provenance surfaced something that must be recorded:

| model | training steps | source |
|---|---|---|
| **FM** (`H16_1e6steps`, cp 20) | **1 000 001** (`train.sh: --n_train_steps 1000001`, `save_freq 50000` ⇒ cp20 = 1e6) | authors' released checkpoint |
| **iMF** (`H16_imf_100k`, cp 4) | **100 000** | our training, Gen13 plan D6 |

**The FM baseline received 10× more training than iMF.** Every Gen13 comparison — including the fix_7.3 refutation — compares a 1e6-step FM against a 1e5-step iMF. That is a genuine confound and it was **our choice** (plan D6 set 100k on the Gen3v4 reasoning that "most learning happens early").

**How much does it undermine fix_7.3?** Partially, but probably not decisively:

| Argument | Weight |
|---|---|
| §4's curve shows iMF **plateaued from ~25 k** — median `raw_mse_u` moved −5.4 % over the last 50 k | ⇒ 10× more steps *of the same recipe* is unlikely to close a **4–6×** x̂1 accuracy gap |
| But a 1e6-step iMF run was **never performed** | ⇒ strictly, untested |
| A better-conditioned schedule (constant-LR tail rather than cosine-to-0 while spiking) is also untested | ⇒ untested |

**Honest statement of the fix_7.3 verdict, with this caveat:**
> iMF as trained here (100 k steps) is strictly dominated by the authors' FM checkpoint (1 M steps) at every matched budget. Whether a 1 M-step iMF would close the gap is **untested** — though the plateau evidence argues against it.

If Gen13 were to be defended, **the one experiment that would do it is a 1e6-step iMF run** (≈42 h at the observed 3.95 it/s — over the 24 h cap, so it would need checkpoint/resume or a reduced budget). Given the plateau, the fix_7.3 recommendation ("do not invest further at this data scale") stands — but the caveat belongs in any writeup.

## 5. Quick commands

```bash
# the CSV (cluster)
column -s, -t /u/home/llim/FMPCC/FM-PCC/logs/hardflow/avoiding-v0/flow/H16_imf_100k/metrics.csv | less

# the same curve from the log, no cluster needed
grep "^\[ train_imf \] step" temp/Used/HF_iMF_first_run/23_57_33_hf_imf_train_23579.log

# enable TensorBoard for future runs
conda activate hardflow_clone && pip install tensorboard
```

## 6. Summary

| | trained? | W&B | TensorBoard | local curve |
|---|---|---|---|---|
| **iMF (Gen13)** | ✅ 100 k steps, 4 h 12 m | ❌ | ❌ (not installed) | ✅ `metrics.csv` **+** the job log |
| **FM (replication)** | ❌ downloaded checkpoint | ❌ | ❌ | ❌ **none exists** |
