# SLRUM DATE 2026-09-18

> **⚠ 2026-09-18, later the same day: groups C, D and E are superseded — cancel them if still queued.**
> All three are `pillars_hg` work. UAV-pillars is being re-evaluated on an enlarged constraint set because
> the one it uses is the constraint the demonstration generator was built to satisfy, and its results are
> withheld from the draft meanwhile. See
> [`PENDING_20260918_pillars_geometry_redesign.md`](PENDING_20260918_pillars_geometry_redesign.md) and
> [`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](SLURM_RUNBOOK_20260919_pillars_enlarged.md).
> `scancel 25890 25891 25892 25893 25894 25895` — only those still pending or running; finished results
> are kept as the control condition. Groups A, B, F, G, H are unaffected.


This runbook covers everything [`PENDING_20260918_verified_data_audit.md`](PENDING_20260918_verified_data_audit.md)
lists as **confirmed missing** and runnable today, minus the two rows the
[2026-09-17 wave](SLURM_RUNBOOK_20260917_dpcc_protocol_rows.md) already owns (audit A1/A2 = ledger R8, R19, R6;
jobs 25878/25879/25880, still in flight at the time of writing). The driver under `Slurm_Codes/temp_bash/` is
gitignored and must be copied to the remote manually.

**Ownership boundary:** the orchestrator only submits and validates remote Slurm evaluation jobs. It does not
download, copy, export or delete results. Copying the driver to the remote and downloading completed results
are manual human tasks.

## One-command submission

```bash
bash Slurm_Codes/temp_bash/pipeline_20260918_pending_all.sh          # PLAN — prints every job, submits nothing
bash Slurm_Codes/temp_bash/pipeline_20260918_pending_all.sh submit   # submits the default groups (A C D E F)
```

Knobs: `WAVE="A C D E F G H"` (or `WAVE=all`) picks groups, `SEEDS` (default `6`), `RECORD`, `UAV_NTRIALS`,
`CORRIDOR_NTRIALS` (default 12 = four flights per route). `WAVE` is used instead of `GROUPS` on purpose —
`GROUPS` is a bash built-in and silently refuses assignment.

**Always run PLAN first.** It prints the exact variant list, tag and budget of every job without submitting.

## Run map

Every group is **evaluation only** — no training job, no new checkpoints. `eval_k_sweep.sh` fans out to one
child job per K, so the child count exceeds the driver count where a K list is given.

| group | ledger | what | entrypoint | tag | driver jobs / children | job IDs |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| A | R20 | UAV-corridor, endpoint projection under random (`-r`) and cumulative-cost (`-c`) selection at $\nfe=3,5$, for FM / MeanFM / CI-MeanFM | [`eval_k_sweep.sh`](../../../../Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh) | `u17cv2` | 3 / 6 | **25887** fm · **25888** mf · **25889** af |
| C | R13 | UAV-pillars, FM and MeanFM at $\nfe=1$, the ten per-step configurations | same | `u7hg` | 2 / 2 | **25890** fm · **25891** mf |
| D | R21 | UAV-pillars, endpoint projection at $\nfe=2$ for all three flow models, at **A = 1.0** | same | `u7hga1` | 3 / 3 | **25892** fm · **25893** mf · **25894** af |
| E | R12 | UAV-pillars, the diffusion baseline's two missing per-step cells (`dpcc-r`, `dpcc-r-tightened`) | [`eval_mix_uav.sh`](../../../../Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh) | `u7hg` | 1 / 1 | **25895** |
| F | R10 | UAV-s-curve, the endpoint rows re-run after the switched-wall fix (FM $\nfe=20$, MeanFM $\nfe=10$, CI-MeanFM $\nfe=5$) | [`eval_k_sweep.sh`](../../../../Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh) | `u18sc` | 3 / 3 | **25896** fm K20 · **25897** mf K10 · **25898** af K5 |
| G | R2 | D3IL-aligning, the diffusion baseline re-run so its tightened geometry (`combined_5-tightened`) is generated | [`eval_mix_visual_aligning.sh`](../../../../Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh) | — | 1 / 1 | `XXX` |
| H | R16 | D3IL-aligning budget ladder: FM $\nfe=2$ and $10$, CI-MeanFM $\nfe=10$ | same | — | 3 / 3 | `XXX` |
| I | R18 | D3IL-avoiding, the diffusion baseline at the extended protocol (5 seeds × 20 episodes) at $\nfe=1,10$ | [`eval_dpcc_job.sh`](../../../../Slurm_Codes/sbatch/eval_dpcc_job.sh) | `_msg20trials` | 2 / 2 | `XXX` |

Groups **G and H are gated** behind `ALIGN_OK=1`, and group **I is a separate phase** — both for reasons below.
Group **B** (corridor endpoint at $\nfe=1$) is deliberately not submitted.

## Submission record

| attempt | command / outcome | jobs created |
| :-- | :-- | :-- |
| 2026-09-18 plan | driver run without an argument (copied to the remote as `Slurm_Codes/temp_bash/18-09-PAPER.sh`); pre-flight passed, twelve jobs printed, nothing submitted | none |
| 2026-09-18 submit | same driver with `submit`; all twelve accepted in order A → C → D → E → F | **25887–25898** |

Scheduler state at submission: `/u/home` reported **5.6 GB** free, so the wave shares the disk with the
2026-09-17 CI-MeanFM training (25879, ≈430 MiB per seed). This wave writes **no checkpoints** — only rollout
npz and plots, with `RECORD=none` — but if the training job is still writing, watch the free space before
adding groups G/H or phase B. Eleven of the twelve are `eval_k_sweep.sh` drivers, which fan out to one child
job per budget: groups A's three jobs produce two children each (K=3 and K=5), so fifteen evaluation children
in total. Job 25895 (group E) is a direct `eval_mix_uav.sh` job — diffusion carries no K list.

## Four things this wave establishes, so they are not re-derived

### 1 · R22's endpoint half is not a missing run — it cannot be run
The diffusion baseline can never take endpoint projection on any quadrotor scene.
`mix_uav/models/engine_registry.py` declares `supports_hardflow=False` for the ddpm engine, and
`mix_uav_test/eval_mix_uav.py:631` drops every `hardflow_*` variant for it: endpoint projection steers a
**velocity field**, which the diffusion arm does not expose. Only R22's per-step half is a run, and that is
group E. **R22 should be split in the ledger:** the per-step row closes with this wave, the endpoint row closes
as ❌ *not applicable — model limit*, and `tab:uav-pillars` keeps *---: not evaluated* in those four cells with a
one-line note rather than a pending marker.

### 2 · Endpoint projection at $\nfe=1$ does not exist as HardFlow, at any setting
HardFlow guidance lives in active **non-terminal** ODE steps. At $\nfe=1$ the only step is the terminal step,
so no activation threshold rescues it — a "$\nfe=1$ endpoint" row is sample-then-project, i.e. DPCC's algorithm
with a different solver. That is why group B is not submitted, and why the $\nfe=1$ half of R20 and R21 cannot
be closed by a run. `ALLOW_DEGENERATE_PROBE=1` will submit it anyway, tagged `u18cv2deg`; those rows may never
carry a HardFlow claim.

At $\nfe=2$ the shipped A = 0.5 floors step 0 out, but **A = 1.0 leaves one genuine guided step**, so group D
runs at A = 1.0 and carries its own tag `u7hga1`. Its rows are A = 1.0 rows and are **not** interchangeable
with the A = 0.5 rows at $\nfe=5$; the budget paragraph of `sec:res:uav:pillars` must say so if it uses them.

### 3 · The corridor guard variant is a new cell, not a re-run
`eval_mix_uav.py` refuses a HardFlow-only variant subset — it needs a non-HardFlow row at the same budget to
compare against. Every published `u17cv2` key would be **overwritten** by such a guard, so the driver uses
`dpcc-c-bounds_free-pdes`, which does not exist in that tree. It is a legitimate new cell (the untightened
companion of the published `dpcc-c`), and nothing already downloaded is touched.

### 4 · `n_trials` is read from the working tree, so protocols cannot share a wave
`scripts/eval.py` (and the FM v3 ODE eval) read `n_trials` from `config/projection_eval.yaml` **at job start**,
with no environment override. The 2026-09-17 wave still has job 25880 queued on `afterok:25879`, and it needs
`n_trials: 2`. Flipping the file to 20 now would silently corrupt it. Group I therefore runs as **phase B**:

1. wait until 25878, 25879 and 25880 have all left the queue;
2. set `n_trials: 20` in `config/projection_eval.yaml`;
3. `WAVE=I bash Slurm_Codes/temp_bash/pipeline_20260918_pending_all.sh submit`;
4. once both jobs have **started**, put the file back to `n_trials: 2`.

The driver refuses group I unless the file says 20, the seed list is 6–10, none of the three 09-17 job IDs is
still in `squeue`, and a trained diffusion checkpoint exists for each requested budget.

## Blocked before submission — decisions for the author

| item | ledger | why it is blocked |
| :-- | :-- | :-- |
| **Alignment context count** | R2, R16 (groups G/H) | `config/visual_aligning_eval.yaml` has said `n_contexts: 3` since 2026-08-04, while the draft reports **ten** contexts for every alignment row. A run submitted today is therefore not comparable with the published rows unless 3 is what they used. Resolve, then re-run with `ALIGN_OK=1`. The driver prints the value and skips G/H without it. |
| **Held-out and 50–60 contexts** | R3 / audit B3 | `n_contexts` has no environment override (`eval_mix_visual_aligning.py:2920`), so this needs a tracked edit to the shared yaml, not a driver knob. Not in this wave. |
| **Diffusion at $\nfe=2$ on D3IL-avoiding** | R18 / audit A4 | $\nfe$ is a **training** property for DPCC diffusion — the checkpoint is literally `H8_K2_…`, and none exists. It needs a training job, and `/u/home` had 7.9 GB free on 2026-09-17. |

## Identity checks

| group | the child log must show |
| :-- | :-- |
| A | `[ U11 ] geo variants = corridor_v2_slide`; the three requested variants, no `[hardflow][BLOCKED]` at $\nfe=3,5$; results path ends `…_u17cv2`; folder names arrive as `hardflow_sls-r-…` / `hardflow_sls-c-…` (`hardflow_new` is the request name) |
| C | `pillars_hg`, `K1`, ten variants, results path ends `…_u7hg` |
| D | `[ eval ][hardflow] HFFM_ACT_THRESHOLD=1.0 → overriding`; **no** `HF_DEGENERATE_SKIPPED.txt`; results path ends `…_u7hga1` |
| E | engine `diffusion`, `Ediffusion_K20_`, `pillars_hg`, exactly `dpcc-r` and `dpcc-r-tightened`; no HardFlow names requested |
| F | `s_curve_hg`, `u18sc`, `update_constraint_list` in the sampler path, HardFlow rows non-degenerate at $\nfe=20/10/5$ |
| G | engine `diffusion`, and **both** `combined_5` and `combined_5-tightened` result subtrees |
| H | `NFE override: flow_steps_v3 = 2 / 10`, `MIX_PROJ_T` = 0.5 / 0.4; for CI-MeanFM also `MIX_EPOCH=latest` and `MIX_AF_ALPHA_END=0.2` |
| I | `n_trials` 20 in the config echo, seeds 6–10, results suffix `_msg20trials`, `H8_K1_…` / `H8_K10_…` load paths |

If a HardFlow job prints `[hardflow][BLOCKED] … DEGENERATE`, the rows are missing by design — record the budget
and the activation threshold rather than re-submitting with the guard disabled.

## Completion record and owner

| check | owner | result |
| :-- | :-- | :-- |
| A · corridor endpoint `-r`/`-c` at $\nfe=3,5$ completed, six children verified | Slurm job + human verification | ✅ 2026-09-18 — 25900/25901/25904/25905/25906/25907 all `Job completed successfully`, 3/3 variants each, **zero** `[hardflow][BLOCKED]` and zero divergence aborts |
| C · pillars FM/MeanFM $\nfe=1$ completed | Slurm job + human verification | ⚰️ **DEAD** — the runs finished (25902, 25912) but `pillars_hg` is superseded by [`PENDING_20260918_pillars_geometry_redesign.md`](PENDING_20260918_pillars_geometry_redesign.md) (Gen15 U17). Filter out of the DA |
| D · pillars endpoint $\nfe=2$ at A = 1.0 completed, no degeneracy sentinel | Slurm job + human verification | ⚰️ **DEAD** — 25899/25903/25911 finished cleanly at A = 1.0 (no degeneracy sentinel), but same `pillars_hg` supersession. Filter out |
| E · pillars diffusion `dpcc-r` / `dpcc-r-tightened` completed | Slurm job + human verification | ⚰️ **DEAD** — 25895 `scancel`-ed 2026-09-18 22:14 at trial 3/10 of `dpcc-r-tightened`, deliberately, for the same supersession. Partial output must not enter the DA |
| F · s-curve endpoint rows re-run post-fix, old rows still excluded | Slurm job + human verification | ✅ ran to completion (25908 fm K20, 25909 mf K10, 25910 af K5; 6/6 variants each) — **but see the divergence note below before using them** |
| G · aligning diffusion tightened geometry generated | Slurm job + human verification | ⏳ never submitted (gated on the `n_contexts` question) |
| H · aligning budget ladder (FM 2/10, CI-MeanFM 10) completed | Slurm job + human verification | ⏳ never submitted (same gate) |
| I · avoiding diffusion at 5 × 20, $\nfe=1,10$; yaml restored to `n_trials: 2` | Slurm job + human verification | ⏳ never submitted — phase B is now unblocked: 25878/25879/25880 all finished 2026-09-18 |
| Required result folders downloaded from the cluster | **human only** | `XXX` |
| `uav_results.py` / `va_results.py` re-run locally; new tag `u18sc` registered (`u7hga1` is dead — do NOT register) | human/local follow-up | `XXX` |
| Ledger updated: R10, R12, R13, R16, R18, R20, R21 closed; R2 closed; R22 split (per-step ✅ / endpoint ❌ model limit) | human/local follow-up | `XXX` |

## Outcome, 2026-09-19 — what landed and what is dead

Logs inspected from `temp/19-09/`.

**UAV-pillars (groups C, D, E) is dead on arrival.** The runs themselves were clean, but
[`PENDING_20260918_pillars_geometry_redesign.md`](PENDING_20260918_pillars_geometry_redesign.md) (Gen15 U17)
established that `pillars_hg` tests a constraint the demonstrations already satisfy, so every `pillars_hg` row
this wave produced — `u7hg` $\nfe=1$ and `u7hga1` $\nfe=2$ alike — measures the wrong thing. 25895 was
cancelled mid-variant on purpose. **Exclude tags `u7hg` and `u7hga1` from the new DA**, together with R12,
R13 and R21, which the redesign supersedes rather than closes.

**Corridor (group A) is clean and usable.** Six children, three variants each, no HardFlow block, no divergence
abort. One thing the DA must handle: strict success is **0.333 in every arm at every budget** — the slide
admits one of the three routes for all methods — so the corridor comparison has to be made on constraint
satisfaction, steps and wall-clock, not on success. Endpoint `-r` and `-c` now exist at $\nfe=3,5$ for all
three flow models; the $\nfe=1$ endpoint cells remain structurally impossible.

**S-curve (group F) completed but carries a physical-divergence caveat.** All three jobs finished 6/6
variants, and the rows are post-fix, so R10's exclusion can be lifted — but the drone **flips** on a large
share of flights: 34 aborts in fm $\nfe=20$, 55 in mf $\nfe=10$, 52 in af $\nfe=5$ (`inverted: body z-axis`),
and they hit the **unprojected** `diffuser` arm too (4/10, 10/10 and 8/10 respectively). The abort guard is
not new — it predates this wave — so this is a property of the scene and the controller, not a regression and
not a projection artefact. MeanFM at $\nfe=10$ scores 0.000 on every variant. Report the scene with the abort
counts beside the scores, or the numbers read as a method ranking when they are a stability failure.

**The 2026-09-17 wave closed completely** — see that runbook's own record.
