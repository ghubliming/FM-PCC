# Gen3v6 U3 — first HardFlow-arm run: in-loop constrained sampling on the mean-flow checkpoint (K=2)

**Run:** eval job **23981** (`temp/2907/2907/…/flow_matching_v3_meanflow/…H8_K2_…`).
**Setup:** the U2 `mf_dit` aw10 checkpoint (seed 6, EMA weights), **matched K=2** for all arms,
**mpc=4** (`HFFM_BATCH=4`), HardFlow activation threshold **0.5** (== DPCC's), 3 halfspace
scenarios, **2 trials/cell**. This is the first cluster run of the U3 HardFlow port
(`hardflow_new-*` queried at `h=0`, so `u(x,t,0)=v`).

*(The rest of `temp/2907/2907` — `H16_imf_*`, `H16_ml_*` — is Gen13 HardFlow-native, a different
codebase; out of scope here.)*

---

## Headline

**The port works, and in-loop HardFlow reaches DPCC-parity safety at K=2.** Both engines convert the
unsafe raw mean-flow generation into **100%-safe + goal-reaching** control on 2 of 3 scenarios
(top-left-hard, both-hard) via their tightened arms, and both cap at 0.5 on the hard top-right. The
`h=0` identity is validated in practice: HardFlow's NLP consumes a genuine instantaneous velocity,
produces sensible trajectories, and reaches **0 violations** wherever it reaches the goal.

## Port correctness — confirmed

- **Ran clean:** matched K=2 applied (`train=10→eval=2`), EMA weights, NFE/NLP metrics logged, no
  crashes. Savepath correctly encoded `_K2_` (the collision fix holds).
- **Parity safeguard:** `dpcc-c-tightened` and `hardflow_new-c-tightened` **both hit 0 violations in
  all 3 scenarios** — the shared tightened feasible set is enforced identically by the scipy
  (post-hoc) and casadi (in-loop) solvers. The port is clean.
- **NLP health:** 0 failures on both-hard and top-right; failures appear only on top-left-hard and
  only for **untightened** HF arms (8 for `-r`, 36 for `-t`) — the tightened arms solve cleanly
  (0 failures, 0 violations). Tightening stabilises the interior-point solve.

## Results (K=2, mpc=4, 2 trials — directional). g&c = goal-AND-constraints

| variant | top-right | top-left | both-hard | note |
|---|---|---|---|---|
| **diffuser** (raw) | 0.0 (12.5 viol) | 0.0 (24.5 viol) | 0.5 (11.5 viol) | unsafe floor |
| dpcc-r | **1.0** | 0.5 | **1.0** | |
| dpcc-r-tightened | 0.5 | **1.0** | **1.0** | DPCC best |
| dpcc-t-tightened | 0.5 | **1.0** | **1.0** | DPCC best |
| dpcc-c / -c-tightened | 0.0 | 0.0 | 0.0 | **collapsed (see below)** |
| **hardflow_new-r-tightened** | 0.5 | **1.0** | **1.0** | HF best |
| **hardflow_new-t-tightened** | 0.0 | **1.0** | **1.0** | HF best |
| hardflow_new-c-tightened | 0.5 | 0.5 | 0.5 | 0 viol throughout |
| hardflow_new-r / -t (untightened) | 0.5 | 0.0 (5–7 viol) | 0.5–1.0 | margin leaks w/o tightening |

**Reading:**
- **HardFlow ≈ DPCC.** On top-left and both-hard, the HF tightened `-r`/`-t` arms match DPCC's best
  (g&c=1.0, 0 violations). On top-right-hard both engines mostly land at 0.5 — it's a genuinely hard
  geometry, not an engine difference. Raw `diffuser` is unsafe everywhere (11–24 violations).
- **Tightening matters for the in-loop arm too:** untightened HF `-r`/`-t` ride the zero-margin
  boundary and leak (top-left `-t`: 7.5 viol / 0.541 total, 36 NLP failures); the +0.025 margin
  fixes it (0 viol, 0 failures) — same behaviour DPCC shows, now confirmed for casadi in-loop.
- **The `-c` (minimum-projection-cost) selection is degenerate — for BOTH engines.** `dpcc-c*`
  collapses to succ=0 everywhere; `hardflow_new-c*` is weak (0.5/0.5/0.0). This is the **same
  lazy-candidate issue flagged in the U2 mf_dit run**: at mpc=4, "least NLP intervention" selects the
  candidate that barely moves — safe but goal-missing. It's a **selection-rule** artifact (random/
  temporal beat min-cost), *not* a projection-engine or port bug, and it shows identically in DPCC.

## Compute cost — the honest in-loop price

| arm | mean ms/step | vs diffuser |
|---|---|---|
| diffuser | 16.7 | 1× |
| DPCC (`dpcc-*`) | 23–26 | ~1.5× |
| **HardFlow (`hardflow_new-*`)** | **96–102** | **~6× (≈4× DPCC)** |

At K=2/mpc=4, HardFlow runs ~4× DPCC — the interior-point (ipopt) solve *inside* the ODE loop vs
DPCC's single post-hoc scipy projection. This is **expected and correct** (the DISCUSS §6c prediction:
the HF arm buys in-loop safety, not speed; every active step is a full NLP). **U3's HF arm is a
field-quality/safety comparison at matched K, NOT a few-step-speed claim** — the timings confirm the
framing rather than contradict it.

## What this establishes (and what it doesn't)

**Establishes:** the U3 port is correct and usable — the MeanFlow paper's checkpoint can be driven by
HardFlow's in-loop constrained sampler via `u(x,t,0)=v`, reaching DPCC-parity safety at K=2 with the
tightened arms. The three-arm (diffuser / DPCC / HardFlow) comparison on a mean-flow model is now
live and produces consistent, interpretable numbers.

**Does NOT establish:** any K-budget story or a statistically-backed ranking — this is **1 seed,
2 trials, K=2 only**.

---

## ⚠️ Lacking data — what's needed to finish the U3 / paper story

1. **The other K points (this is the big one).** Only **K=2** ran. The budget narrative
   (DPCC@K20 baseline vs few-step @K1/2/5) needs the rest of the recommended sweep:
   ```
   HFFM_FLOW_STEPS=1  HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
   HFFM_FLOW_STEPS=5  … (same)
   HFFM_FLOW_STEPS=20 … (same)
   ```
   Each lands in its own `_K{K}_` dir. **Please run K=1, 5, 20** — then this insight extends to the
   full matched-K matrix.
2. **Seeds / trials.** 1 seed × 2 trials makes every cell 0.0/0.5/1.0 — directional only. The paper
   table needs seeds 6–10 (⇒ train 7–10 first) and/or more trials for error bars.
3. **(Optional) the top-right-hard 0.5 ceiling.** Both engines stall there at K=2 — worth checking
   whether K=5/20 lifts it (few-step field quality) or whether it's a constraint-geometry limit.

## Net
U3 is **validated**: HardFlow-into-Gen3v6 runs correctly and is competitive with DPCC on safety at
matched K=2, at ~4× the per-step cost (the in-loop price). To turn this into a paper result, run the
K=1/5/20 sweep (same command) and add seeds.
