# Gen12 U5 — Results analysis: full mpc=4 DPCC-parity run (job 23890, 26–27 Jul)

**What this is:** the first *complete* Gen12 evaluation exercising the full DPCC-parity variant
matrix at **mpc=4** (candidate fan ON, so `-r/-c/-t` selection actually differs). Material:
`temp/Gen12/2707/` (downloaded from `i6-gpu-1`).

## Run identity

| | |
|---|---|
| Job | `23890` on `i6-gpu-1` (RTX A5000, CUDA dev 7) |
| Git rev | `18aa683` |
| Checkpoint | `flow_matching_v3_ode_selectable` `H8 … a1.5_b1.0_aw10`, step 98000 (FMv3 ODE) |
| Eval folder | `K20_thres0.5_mpc4_n2` |
| Grid | K=20, `activation_threshold=0.5`, **batch=4** (env overrides took), 5 seeds ×3 halfspaces ×8 arms ×2 trials = **240 trials** |
| Wall time | **8 h 04 m** (22:19:01 → 06:23:03 UTC) |
| Integrity | **0 missing npz** (120/120), `Evaluation completed successfully`, no crashes |

The three env overrides propagated correctly (log: `NFE`/`NLP solves`/`act_thr=0.5`/`sel=…`,
`batch=4` on every arm), confirming the `submit.sh --export=ALL` path works.

## Headline table (mean over 30 trials/variant = 5 seeds × 3 halfspaces × 2 trials)

| variant | succ% | **succ+con%** | collfree% | viol/tr | total_viol | steps | ms/step | nlp/run | nfe |
|---|---|---|---|---|---|---|---|---|---|
| `diffuser` (A, unguided) | 100 | **13.3** | 13.3 | 16.50 | 4.3e0 | 70 | 0.18 | 0 | 11392 |
| `dpcc-c-tightened` (B, post-hoc) | 100 | **100.0** | 100.0 | 0.000 | 3.7e-8 | 63 | 0.47 | 0 | 10277 |
| `hardflow_new-r` | 100 | 76.7 | 76.7 | 0.633 | 7.9e-3 | 74 | 1.87 | 6021 | 18064 |
| `hardflow_new-c` | 100 | 60.0 | 60.0 | 1.467 | 1.5e-2 | 101 | 1.80 | 8160 | 24480 |
| `hardflow_new-t` | 100 | 73.3 | 73.3 | 1.033 | 6.4e-3 | 67 | 1.87 | 5437 | 16312 |
| `hardflow_new-r-tightened` | 100 | 96.7 | 96.7 | 0.033 | 4.2e-5 | 73 | 1.89 | 5877 | 17632 |
| **`hardflow_new-c-tightened`** | 100 | **100.0** | 100.0 | 0.000 | 1.7e-5 | 100 | 1.82 | 8109 | 24328 |
| **`hardflow_new-t-tightened`** | 100 | **100.0** | 100.0 | 0.000 | 3.5e-5 | 67 | 1.90 | 5413 | 16240 |

(`nlp/run` = mean NLP solves per run; DPCC/diffuser arms are 0 by construction — they don't use
the in-loop solver.)

## Insights

**1. Safeguard PASSES — the DPCC engine inside Gen12 is intact.**
`dpcc-c-tightened` (arm B = the DPCC scipy projector running *inside* the dual-engine Gen12)
gives 100% succ+con, 0 violations, `total_violation = 3.7e-8` (machine precision, i.e. exact
terminal projection). This is the cross-check the run was designed for: Gen12 carries BOTH
engines and the DPCC path behaves identically to the standalone DPCC/FMv3ODE eval.
→ **Action for the user:** confirm these B numbers are byte-for-byte what the FMv3ODE eval
produced for the same seeds (I don't have those files here) to close the port-cleanliness claim.

**2. The problem is real — diffuser floor is 13.3% succ+con.**
Unguided flow reaches the goal every time (100% success) but plows through obstacles/halfspaces:
16.5 violations/trial, only 13% constraint-clean. The generative field alone is unsafe; the
brakes matter.

**3. Gen12's own in-loop HardFlow engine, tightened, MATCHES DPCC.**
`hardflow_new-c-tightened` and `hardflow_new-t-tightened` both hit **100% succ+con, 0 violations**
— the port's target result. Their `total_violation` (1.7e-5 / 3.5e-5) is ~1000× larger than
DPCC's 3.7e-8, which is expected and telling: DPCC does an *exact terminal* projection, whereas
the in-loop NLP enforces feasibility *along the chain* softly — the residual is tiny but not
machine-exact. Same feasible set, different mechanism.

**4. Untightened in-loop is NOT enough on its own (boundary-riding).**
With zero margin, `hardflow_new-{r,c,t}` land at 60–77% succ+con with small residual violations
(0.6–1.5/trial). The ODE overshoots slightly *between* NLP solves and, with no slack, clips the
exact boundary. The 0.025 `-tightened` margin is what absorbs the overshoot → jump to 96.7–100%.
**Tightening is doing the heavy lifting, not the in-loop solve.**

**5. Selection ranking (only meaningful now that mpc>1):**
- *Untightened:* `-r` (76.7) ≈ `-t` (73.3) > `-c` (60.0). **min-projection-cost is the WORST**
  untightened — by design it picks the candidate needing the *least* NLP correction, i.e. the
  one closest to the raw (unsafe) field, so with zero margin it rides the boundary hardest.
- *Tightened:* `-t` and `-c` both perfect (100); `-r` slips to 96.7 (one `both-hard` seed).
- **Temporal-consistency (`-t`) is the most robust selection** across both geometries, and it's
  also cheap (5.4k solves vs `-c`'s 8.1k).

**6. Hardest sub-task = `both-hard` (both halfspaces).** Per-halfspace succ+con for untightened
arms collapses there (`-r` 40%, `-c` 30%), while `top-right-hard`/`top-left-hard` stay 70–100%.
Tightened arms hold ≥90% everywhere.

**7. Cost.** In-loop NLP is ~1.8–1.9 ms/step ≈ **4× the DPCC post-hoc projector** (0.47) and
**~10× unguided** (0.18). `-c` variants are the most expensive (8.1–8.4k solves, ~24k NFE, and
the longest paths at ~100 steps vs ~67 for `-t`) because min-cost ranking runs the full batch-4
fan and tends to produce longer, meandering trajectories. `-t`/`-r` are ~5.4–6.0k solves.

**8. Solver robustness.** 1116 individual NLP-solve failures across the whole run (concentrated
in `-r`/`-c` tightened, 240–307 each) but **0 propagated to task failure** — every arm is 100%
goal-success, so the graceful fallback holds. Worth watching, not fatal.

## Bottom line

At **mpc=4 + tightened margin**, Gen12's in-loop HardFlow engine reproduces DPCC's perfect
constraint satisfaction (100% succ+con, 0 violations) on the avoiding task — the port goal is
met. The caveats: it costs ~4× DPCC's compute, its feasibility is *soft* (1e-5 residual vs
DPCC's 1e-8 exact terminal projection), and the win comes mostly from the eval-time tightening
margin, not from the in-loop solve replacing it — untightened in-loop still leaks (60–77%).
Best-value arm = **`hardflow_new-t-tightened`** (100%, 0 viol, cheapest of the perfect trio).

## Reproduce / extend

- This folder: `K20_thres0.5_mpc4_n2`. An mpc=1 or different-K run writes a *separate* folder.
- `n_trials=2` is thin for the per-seed spread; bump `n_trials` in
  `config/hardflow_projection_eval.yaml` for tighter CIs before drawing final conclusions.
- Aggregation was done with a pure-stdlib npz reader (no numpy in this container):
  `scratchpad/npyread.py` + `agg.py`.

---

# Deep dive: does HardFlow ever beat DPCC?

Short answer: **not on efficiency at equal safety, but YES on safety when there is no
tightening margin.** The two regimes tell opposite stories.

## 1. Same-safety efficiency — DPCC-tightened wins outright

Restrict to the arms that actually reach **100% succ+con** and compare cost (steps + time). This
is the fair "who's cheapest at full safety" question. (Time is comparable across the two runs:
`diffuser` and `dpcc-c-tightened` reproduce to the ms — see §4.)

| variant (100%-safe club) | succ+con% | steps | ms/step | source |
|---|---|---|---|---|
| **`dpcc-c-tightened`** | 100 | **63.2** | **0.477** | ODE (= 0.471 in HF run) |
| `dpcc-t-tightened` | 100 | 63.7 | 0.487 | ODE |
| `hardflow_new-t-tightened` | 100 | 66.7 | 1.902 | HF run |
| `hardflow_new-c-tightened` | 100 | 100.4 | 1.823 | HF run |

DPCC-tightened is the efficiency frontier: **fewer steps AND ~4× less time** than the best
HardFlow arm at identical (perfect) safety. `hardflow_new-t-tightened` gets *close on path length*
(66.7 vs 63.2 steps, +5%) but pays **3.9× the per-step time** for its in-loop NLP.
`hardflow_new-c-tightened` is worse on both (100 steps, min-cost selection meanders).
→ **Confirmed: no HardFlow arm beats `dpcc-c-tightened` on steps or time at equal success+constraint.**

## 2. Tightening lifts HardFlow *less* — because HardFlow starts higher, not because tightening is weaker

The margin's effect (base → `-tightened`), per selection rule:

| sel | DPCC succ+con Δ | DPCC viol/tr | HardFlow succ+con Δ | HardFlow viol/tr |
|---|---|---|---|---|
| `-r` | 30.0 → 86.7 (**+56.7**) | 11.23 → 3.77 | 76.7 → 96.7 (**+20.0**) | 0.63 → 0.03 |
| `-c` | 56.7 → 100 (**+43.3**) | 4.07 → 0.00 | 60.0 → 100 (**+40.0**) | 1.47 → 0.00 |
| `-t` | 33.3 → 100 (**+66.7**) | 6.27 → 0.00 | 73.3 → 100 (**+26.7**) | 1.03 → 0.00 |

The user's intuition ("tightening doesn't lift HardFlow as dramatically as it lifts DPCC") is
correct on the *numbers* (+20…+40 pts vs +43…+67 pts) — but the reason is the opposite of a
weakness: **HardFlow's non-tightened baseline is already 60–77%, close to the ceiling, so there's
less headroom for the margin to recover.** DPCC's non-tightened baseline is only 30–57%, so
tightening has a bigger hole to fill. Tightening is not doing *less* for HardFlow; HardFlow left
it *less to do*.

## 3. The regime where HardFlow WINS: exact constraints, no margin

Head-to-head at **non-tightened** (zero margin — constraints enforced exactly). Same checkpoint,
same seeds, same env:

| sel | DPCC succ+con | **HardFlow succ+con** | DPCC viol/tr | **HardFlow viol/tr** | DPCC goal% | HardFlow goal% |
|---|---|---|---|---|---|---|
| `-r` | 30.0 | **76.7** (+46.7) | 11.23 | **0.63** (18×) | 93.3 | **100** |
| `-c` | 56.7 | **60.0** (+3.3) | 4.07 | **1.47** (2.8×) | 96.7 | **100** |
| `-t` | 33.3 | **73.3** (+40.0) | 6.27 | **1.03** (6×) | 93.3 | **100** |

**This is the real HardFlow win.** With no slack, post-hoc DPCC projection can only fix the
*terminal* sample and cannot undo infeasibility accumulated *along* the trajectory, so it leaks
badly (4–11 violations/trial) and even sacrifices goal-reaching (drops to 93–97% success — the
projection pushes some trajectories off-goal). HardFlow's **in-loop** NLP catches violations
continuously, so at zero margin it is **3–18× safer on violation count, +40–47 pts on succ+con
(for `-r`/`-t`), and keeps 100% goal success.** `-c` is the exception (only +3.3 pts) — min-cost
selection deliberately picks the least-corrected candidate, which is the least safe one, so it
throws away most of the in-loop advantage.

## 4. Why the reversal

The two regimes reward different machinery:

- **No margin (exact):** feasibility is *tight*, the terminal-only correction DPCC applies is
  insufficient, and continuous in-loop enforcement pays off → **HardFlow wins on safety.**
- **With 0.025 margin:** the feasible set has slack, a single exact terminal projection lands
  safely and costs almost nothing, so DPCC's cheap post-hoc solve reaches 100% and HardFlow's
  expensive per-step NLP becomes overkill → **DPCC wins on cost at the same safety.**

**Takeaway:** HardFlow's value proposition is *"safe with a smaller (or zero) margin"*, not
*"cheaper at a given margin"*. If exact/zero-margin constraint satisfaction is the requirement,
HardFlow (esp. `-r`/`-t`) is materially better than post-hoc DPCC. If a small tightening margin
is acceptable, DPCC-tightened dominates on every axis and HardFlow buys nothing.

---

# Cross-comparison vs the FMv3ODE eval (CAND_75, all seeds)

Source: `temp/Gen12/2707/H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE` — the standalone
FMv3 ODE-selectable eval, **same checkpoint (`a1.5_b1.0_aw10`), same K=20, same seeds 6–10, same
3 halfspaces, n_trials=2** → 30 trials/variant, directly comparable to the HardFlow run. It
carries the full DPCC set (`dpcc-{r,c,t}[-tightened]`) plus gradient/post_processing/model_free,
which the HardFlow run did not — so it supplies the non-tightened DPCC baselines used in §3.

## 4a. Port-cleanliness cross-check (the safeguard)

The same `dpcc-c-tightened` arm ran in **both** jobs (standalone ODE, and as arm B *inside* the
dual-engine Gen12). They agree to the digit:

| metric | `dpcc-c-tightened` (FMv3ODE/CAND_75) | `dpcc-c-tightened` (Gen12 HF run, arm B) |
|---|---|---|
| succ+con% | 100 | 100 |
| viol/tr | 0.000 | 0.000 |
| **total_violation** | **3.70e-8** | **3.70e-8** |
| steps | 63.2 | 63.2 |
| ms/step | 0.477 | 0.471 |

And the `diffuser` control is **byte-identical** across the two runs (succ+con 13.3, viol/tr
16.5, total_viol 4.26e0, steps 70.2, ms/step 0.184). Two independent arms matching to machine
precision is strong evidence that **the DPCC engine embedded in Gen12 is the unmodified DPCC
projector** — the port is clean, and cross-run comparisons of steps/time in §1–3 are valid.

## 4b. Full FMv3ODE DPCC landscape (context for §3)

| variant | succ% | succ+con% | viol/tr | total_viol | steps | ms/step |
|---|---|---|---|---|---|---|
| diffuser | 100 | 13.3 | 16.50 | 4.26e0 | 70.2 | 0.184 |
| dpcc-r | 93.3 | 30.0 | 11.23 | 2.54e-1 | 82.4 | 0.459 |
| dpcc-r-tightened | 100 | 86.7 | 3.77 | 1.09e-1 | 73.3 | 0.853 |
| dpcc-c | 96.7 | 56.7 | 4.07 | 6.88e-2 | 68.9 | 0.433 |
| **dpcc-c-tightened** | 100 | **100** | 0.00 | 3.70e-8 | 63.2 | 0.477 |
| dpcc-t | 93.3 | 33.3 | 6.27 | 1.65e-1 | 69.3 | 0.444 |
| **dpcc-t-tightened** | 100 | **100** | 0.00 | 4.04e-5 | 63.7 | 0.487 |
| gradient / -tightened | 100 | 20.0 / 10.0 | 16.5 / 19.2 | ~3.6e0 | ~70 | 0.20 |
| post_processing / -tightened | 93–100 | 30.0 / 86.7 | 11.2 / 3.8 | 2.5e-1 / 1.1e-1 | 82 / 73 | 0.46 / 0.85 |
| model_free / -tightened | 100 | 16.7 / 10.0 | 17.4 / 18.6 | ~3.5e0 | ~73 | 0.25 |

Notes: only `dpcc-{c,t}-tightened` reach 100% succ+con in the whole DPCC family — the same two
arms HardFlow-tightened has to beat, and doesn't (§1). `post_processing` is numerically identical
to `dpcc-r` here (same random-selected post-hoc projection). gradient/model_free are constraint-
weak (≤20%) — not contenders. This is the landscape against which HardFlow's **non-tightened
safety win (§3)** stands out: at zero margin HardFlow-`r`/`t` (76.7 / 73.3) beats *every*
non-tightened DPCC arm (≤56.7) by a wide margin.

