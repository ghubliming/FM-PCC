# DA — Gen14 · **Gate 4 closed on α-Flow: HardFlow works best on the worst engine**

**Drop** `temp/0609/II/` · **Batch** `batch_va2_20260907_141036` · **Task** `aligning-d3il-visual`
**New job** `25475` — `af` α_end=0.2, K=20, T=0.2, **arms A+B+C**, `GIT REV 963faed` — **complete**
**Protocol** seed 6 · same 10 paired contexts · train split · `mpc4` · `filmv1` · `unet` · `EPlatest`
**Parent** `DA_20260906_Gen14_four_gate_af_alpha_live_K20_T0.2.md` — this closes its Gate 4 and its Next-action #1

**The question:** the parent DA scored α-Flow on 16 variants because arm C was switched off, and
left Gate 4 open with the note *"nothing in this section may be read as 'α-Flow cannot be made
safe'."* It then recommended **against** running arm C. **That recommendation was wrong, and this
run is why.**

> 📌 **The verdict does not move — `mf` still wins every axis — but the reason to run this was
> real.** Arm C on α-Flow produces the single largest constraint improvement anywhere in the V_A
> corpus, and it produces it *for free*. That is a **HardFlow** result, not an α-Flow result, and it
> would not exist if the run had been skipped.

## Ground rules

Inherited from the parent DA (§Ground rules 1–8) — `context_final_xy_dist` in metres, initial mean
**0.4530 m**; tightened geometry only for constraint claims; 🚫 = constraint-ablated, unquotable;
MIN is n = 1. Two additions:

9. **Arm C is genuine here.** At `A=0.5` the HardFlow guard disables the arm when `n_genuine=0`,
   which is the case for K≤2. At **K=20** the arm runs real HardFlow arithmetic, so — unlike the
   K=2 operating point α-Flow actually prefers — these rows are attributable.
10. 🔴 **Single-run numbers on projected arms are not reproducible to better than ~0.4 m.** This run
    re-ran arms A+B under identical settings and they did **not** all reproduce. See §5. Every MIN
    in this DA, and in every prior V_A DA, must be read with that noise floor.

---

# 1 · Did arm C run? — ✅ **PASS**

```
JOB ID: 25475        GIT REV: 963faed
[ eval ] alpha schedule: MIX_AF_ALPHA_END=0.2   [ eval loading ] alpha(step 100000) = 0.2000
[ eval ] arm C ENABLED: HFFM_VARIANTS='hardflow_new-r hardflow_new-c hardflow_new-t'
[ eval ]   NLP backend = slsqp  ->  artifacts written as hardflow_sls-*  (IPOPT corpus untouched)
[ eval ]   arm C threshold inherits arm B's T (arms B and C matched)
[ eval ] NFE override: flow_steps_v3 = 20   ->  4 projector call(s)/replan at T=0.2
Job completed successfully.
```

Six `[hardflow][NLP-BACKEND] slsqp … dof=66 reg_scale=1.0` banners — three variants × two
geometries. The cell now holds **380 rows** (19 variants × 10 contexts × 2 geometries), exactly
matching `mf` and `fm` at the same K/T. **First arm-C execution on `engine=af` in the project.**

⚠️ **Path note.** The results landed in `…_Eaf_EPlatest/6`, *without* the `msgafon02_s6` run-tag
suffix carried by job `25417` — the run-tag env knob was not set on the resubmit. Harmless: the
checkpoint path (`AFAFend0p2`), K, T, film, bone and epoch are all identical, so this is the same
cell written to an untagged directory. The aggregator treats it as its own group, which is why it
carries a self-consistent 19 variants. **All numbers below come from the untagged 380-row group.**

---

# 2 · Gate 4 · Does projection rescue α-Flow? — 🔴 **NO. Closed.**

Tightened geometry, arm C rows ⭐, `mf` at the identical operating point for comparison:

| engine | variant | MIN | med | untouched | **zero-viol** | **viol** | sat | ms |
|---|---|---|---|---|---|---|---|---|
| `af` | `hardflow_sls-r` ⭐ | 0.0485 | 0.4247 | 3/10 | **0.80** | 2.70 | 0.993 | 294.2 |
| `af` | `hardflow_sls-t` ⭐ | 0.1291 | 0.4339 | 4/10 | 0.50 | 8.60 | 0.978 | 344.8 |
| `af` | `hardflow_sls-c` ⭐ | 0.2936 | 0.4557 | **6/10** | 0.60 | 10.10 | 0.975 | 314.8 |
| `af` | `diffuser` | **0.0110** | 0.4232 | 3/10 | 0.20 | 127.40 | 0.645 | 171.5 |
| `mf` | `hardflow_sls-r` ⭐ | **0.0220** | **0.1967** | 2/10 | **1.00** | **0.00** | **1.000** | 275.3 |
| `mf` | `hardflow_sls-c` ⭐ | 0.0241 | 0.3227 | 4/10 | **1.00** | **0.00** | **1.000** | 282.3 |
| `mf` | `hardflow_sls-t` ⭐ | 0.0307 | 0.2297 | 3/10 | 0.90 | 0.60 | 0.999 | 301.5 |
| `mf` | `diffuser` | 0.0278 | **0.0902** | 1/10 | 0.20 | 31.60 | 0.921 | 172.5 |

**No α-Flow arm reaches zero-violation 1.000. The ceiling is 0.80** (`hardflow_sls-r`), against
MeanFlow's **two** arms at a clean 1.000 with 0.00 violations, neither frozen. Paired on the shared
contexts, α-Flow is worse under every arm-C variant:

| pair (`af` − `mf`) | Δ distance | wins/losses | sign p | perm p |
|---|---|---|---|---|
| `hardflow_sls-r` | +0.1220 | 7/1 | 0.0703 | 0.0625 |
| `hardflow_sls-c` | **+0.1534** | **6/0** | **0.0312** | **0.0312** |
| `hardflow_sls-t` | +0.1129 | 7/1 | 0.0703 | 0.1094 |

And arm C does not repair α-Flow's own MIN: `hardflow_sls-r` at 0.0485 is **4.4× worse than its own
unguided 0.0110**. `hardflow_sls-c` freezes 6 of 10 rollouts — the worst freeze rate of any legal
arm in the cell.

**Gate 4 verdict: FAIL.** The parent DA's four-gate table now reads ✅ / ✅ / 🔴 / 🔴, and the
overall verdict is unchanged: **α-Flow is dominated by MeanFlow in V_A on every axis.**

---

# 3 · The result worth keeping — **HardFlow's effect scales with how bad the engine is**

Arm C versus the unguided arm, *within* each engine, paired on the same 10 contexts:

| engine | arm | Δ violations | Δ distance | Δ sat rate | Δ ms |
|---|---|---|---|---|---|
| **`af`** | `hardflow_sls-r` | **−124.70** (0/8, p=0.0078) | **−0.011 (4/3, p=1.0000)** | **+0.348** (8/0, p=0.0078) | +122.7 |
| `af` | `hardflow_sls-c` | −117.30 (2/7, perm p=0.0508) | +0.096 (6/1, p=0.1250) | +0.329 (8/1, p=0.0391) | +143.3 |
| `af` | `hardflow_sls-t` | −118.80 (2/7, perm p=0.0430) | +0.025 (4/3, p=1.0000) | +0.333 (8/1, p=0.0391) | +173.3 |
| **`mf`** | `hardflow_sls-r` | −31.60 (0/8, p=0.0078) | +0.068 (6/3, p=0.5078) | +0.079 (8/0, p=0.0078) | +102.8 |
| `mf` | `hardflow_sls-c` | −31.60 (0/8, p=0.0078) | +0.143 (7/2, p=0.1797) | +0.079 (8/0, p=0.0078) | +109.8 |
| `mf` | `hardflow_sls-t` | −31.00 (0/8, p=0.0078) | +0.112 (7/2, p=0.1797) | +0.078 (8/0, p=0.0078) | +128.9 |

> **HardFlow removes 124.70 violations per rollout from α-Flow and 31.60 from MeanFlow — 3.9× as
> much work — and it does so on α-Flow at a distance cost that is statistically zero
> (−0.011 m, 4/3, p = 1.0000), while on MeanFlow it costs +0.068 m.** α-Flow's satisfaction rate
> rises **+0.348**; MeanFlow's rises +0.079.

This is the cleanest demonstration in the corpus of what the in-loop NLP is *for*. On a well-behaved
generative field there is little left to fix, so HardFlow mostly buys latency. On a field that emits
127 violations per rollout it removes 98 % of them and gives up nothing measurable in return. The
claim belongs in the **HardFlow** ledger, not the α-Flow one:

**"The in-loop projector's benefit scales with the constraint error of the plan it is given, and on
a high-violation field it is free."**

It also reframes the ceiling. α-Flow does not fail Gate 4 because HardFlow fails on it — HardFlow
works *harder* here than anywhere. It fails because 127.40 violations is too deep a hole for four
projector calls per replan to fully climb out of: 2.70 residual violations is a 98 % reduction that
still is not 0.00.

---

# 4 · τ = 0.850 on a third engine — the NLP failure is settled

```
25475 (af, K20/T0.2):  6/6 items — first non-converged SLSQP solve at tau=0.850
25312 (fm, K20/T0.2):  6/6 items — tau=0.850, call #2
25247 (mf, K20/T0.2):  6/6 items — tau=0.850, call #2
25273 (mf, K10/T0.4):  6/6 items — tau=0.700, call #2
```

Three engines with three different generative objectives — MeanFlow, Flow Matching, α-Flow — fail
on the **same solve at the same τ, in every single item**. The generative model is ruled out. The
cause is the NLP's conditioning at that point of the schedule, and the τ shift from 0.850 to 0.700
when K drops 20 → 10 says it tracks the *call index*, not the physical time. Worth chasing on the
projector side; nothing further will come from the engine side.

---

# 5 · 🔴 The corpus has a reproducibility floor, and it is ~0.4 m

Job `25475` re-ran arms A+B at settings identical to job `25417` — same seed, same 10 contexts, same
checkpoint, same K/T. **They did not all reproduce.** Per-rollout, tightened:

| variant | identical rollouts | max abs Δ |
|---|---|---|
| `gradient`, `dpcc-c`, `dpcc-c-dt2p0`, `dpcc-c-dt4p0`, `bounds_free` | **10/10** | 0.0000 |
| `diffuser` (unguided) | 8/10 | **0.0003** |
| `dpcc-r` | 9/10 | 0.0012 |
| `dpcc-t` | 8/10 | 0.0058 |
| `dpcc-c-dt0p5` | 8/10 | 0.0463 |
| `geo_free-bounds_free` 🚫 | 8/10 | 0.1325 |
| `dpcc-c-dt0p25` | 8/10 | **0.2601** |
| `model_free` 🚫 | 5/10 | 0.2961 |
| `geo_free` 🚫 | 8/10 | **0.3501** |
| `model_free-bounds_free` 🚫 | 4/10 | **0.3979** |

**The mechanism is amplification, not randomness.** The unguided arm — no projector, no NLP — is
reproducible on 8 of 10 rollouts and diverges by **0.3 mm** on the other two. That is GPU float
non-determinism (non-associative reductions), not a seeding fault. But a 0.3 mm difference in the
first plan changes which side of a constraint boundary the SLSQP solve starts on, which changes the
solve, which changes the next observation — and 400 replan steps later the two runs are up to
**0.40 m apart**, on a task whose whole dynamic range is 0.45 m.

Consequences, and they are not small:

- **MIN on a projected arm is not a stable statistic.** `dpcc-c-dt0p25` reported MIN 0.0223 in job
  `25417` and **0.2116** in job `25475` — a 9.5× swing from re-running the same command. Any
  MIN-ranked table, including the ones in the parent DA and the funnel DA, is ranking noise at the
  top when the gaps are under ~0.1 m.
- **The engine-level conclusions survive.** Every claim carried forward rests on paired 10-context
  comparisons with sign/permutation tests, or on effects far above this floor: α-Flow − MeanFlow on
  distance is +0.20 m, HardFlow's violation reduction is −124.70, both an order of magnitude clear
  of the noise.
- **The arms that reproduce perfectly are the deterministic ones** (`gradient`, plain `dpcc-c`, the
  frozen `dt2p0`/`dt4p0`). The ones that diverge are exactly those where the SLSQP solve is doing
  real, contested work. That is a consistency check on the mechanism, not a coincidence.

---

# 6 · Why α-Flow loses here but won in avoiding — and why "AF ⊇ MF" is not a guarantee

Two objections, one answer.

**6.1 The V_A loss is a K=20 phenomenon, not a blanket defeat.** Paired at **K=2** — α-Flow's own
preferred operating point, `filmv1` both sides, tightened, unguided, MeanFlow taken as the
per-context median of its three runs:

```
af (α=0.2) − mf   at K=2 :  mean +0.0240 m   4/4   sign p = 1.0000   perm p = 0.8750
af (α=0.2) − mf   at K=20:  mean +0.2004 m   6/3   sign p = 0.5078   perm p = 0.0469
```

**At K=2 the two engines are indistinguishable.** The gap opens as K grows, because MeanFlow
improves monotonically (0.2157 → 0.1421 → 0.0902) and α-Flow does not (0.4247 → 0.4232).

This reconciles the avoiding result rather than contradicting it. `DA_20260903_AF_UNet_alphaflow_ENABLED_seed6_diffuser.md`
§4 reports the avoiding head-to-head as **W 15 / L 14 / T 6 on goal-reached, "on the aggregate this
arm is a wash"** — the AF win there is concentrated in one arm (`dpcc-t-tightened`) at **K=1**, and
that DA carries its own ⛔ citation block (single seed, `n_steps` defined two ways). So both
environments say the same thing: **α-Flow ties MeanFlow at low K and falls behind as K rises.**
V_A measured at K=20 — MeanFlow's best regime and α-Flow's worst — which is why the contrast looks
starker here.

⚠️ This qualifies the parent DA's phrase "dominated on every axis": that is true **at K=20**, which
is where the V_A flagship sits, but it is *not* true at K=2, where the honest verdict is a tie.

**6.2 α-Flow does not contain MeanFlow continuously — so it has no free lunch.** From
`flow_matcher_v3_alphaflow/models/af_diffusion.py:653-665`, with **dt = α·h**:

| branch | when | target |
|---|---|---|
| FM anchor | `h == 0` | `u_tgt = v` |
| **discrete / bootstrap** | `h > 0 and α > 0` | `u_tgt = α·v + (1−α)·u_next`, `u_next = u(z_r+dt, r+dt, h−dt)` under `no_grad` |
| **continuous (JVP)** | `h > 0 and α == 0` | `u_tgt = v + h·du/dr` — Gen3v6 MeanFlow, analytic |

Three consequences, and together they answer "isn't α-Flow guaranteed at least as good?":

1. **α is a fixed schedule, not an optimised parameter.** The *family* contains MeanFlow; training
   picks one point in it. Containment gives expressiveness, never dominance of a trained checkpoint.
2. **The α > 0 target is a bootstrap of the model's own output.** MeanFlow's target is an analytic
   directional derivative of the true instantaneous field. α-Flow's is the network evaluated at a
   shifted point and frozen. A bootstrap has a fixed point that need not be the true field — the
   learned average-velocity field can be self-consistent and wrong.
3. 🔴 **α → 0⁺ does not converge to MeanFlow — the code changes branch at α == 0.** And since
   `dt = α·h`, small α means the bootstrap is a finite difference over a *vanishing* step, which is
   the worst-conditioned place to be: neither analytic nor well-separated.

**The mechanism this predicts is exactly what the ladder shows.** A biased field is *hidden* at
low K, where one or two huge Euler steps make discretisation error dominant and both engines are
equally coarse; it is *exposed* at high K, where discretisation error vanishes and the sampler
converges faithfully to the field's own — biased — fixed point. More NFE integrates the wrong field
more accurately. MeanFlow's analytic target has no such bias, so its error keeps shrinking.

**And it explains the α ordering.** α=0.05 carries 95 % bootstrap weight at `dt = 0.05h`; α=0.2
carries 80 % at `dt = 0.2h`. Less bootstrap and a better-conditioned difference ⇒ α=0.2 should beat
α=0.05, which it does on distance, on push-aways (2/10 vs 4/10) and on success. **The prediction is
that larger α is better in V_A**, and it is cheap to test — see Next-action #4.

### 6.3 The cross-environment evidence, side by side

The avoiding DA's own §4 table (`top-right-hard`, seed 6, n=20, mean S&C over its 7 DPCC arms) is
the cleanest independent test of the same K story:

| K | AF `α→0.2` | AF `α→0.05` | MF-UNet | who leads |
|---|---|---|---|---|
| 1 | 0.493 | 0.500 | **0.557** | MF |
| 2 | 0.557 | **0.621** | 0.543 | AF |
| 5 | **0.543** | 0.536 | 0.500 | AF |
| 10 | **0.586** | 0.536 | 0.500 | AF |
| 20 | **0.564** | 0.514 | 0.493 | AF |

⚠️ **Read this carefully — it does not simply agree with V_A.** In *avoiding*, MeanFlow **degrades**
with K (0.557 → 0.493) and α-Flow holds roughly flat, so α-Flow's relative position *improves* with
K. In *V_A*, MeanFlow **improves** with K (median 0.2157 → 0.0902) and α-Flow holds flat, so
α-Flow's relative position *worsens*. **The engine that changes behaviour between the two
environments is MeanFlow, not α-Flow.** α-Flow is flat in K in both.

That is a sharper statement than "α-Flow ties at low K". The invariant across both tasks is:

> **α-Flow's plan quality is insensitive to NFE. Whether it wins therefore depends entirely on
> whether MeanFlow's own NFE curve rises or falls in that environment.**

Why MeanFlow's curve has opposite sign in the two tasks is **not** established here. The obvious
structural differences — avoiding is state-based with a halfspace constraint and an S&C/steps
metric; V_A is visual (dual-cam ResNet + FiLM) with a placement-distance metric — are candidates,
not answers. It is the most interesting open question this DA raises and it is *not* an α-Flow
question.

Two further cautions on the avoiding side, both from that DA itself: the aggregate head-to-head is
**W 15 / L 14 / T 6** ("*on the aggregate this arm is a wash*"), and the citable single-arm win is
`dpcc-t-tightened` at **K=1** — a cell where, per the table above, the *mean* over arms has MF
ahead. The avoiding result is therefore an **arm-selected** win, and it carries its own ⛔ block
(single seed, `n_steps` defined two ways across the toolchain).

### 6.4 What would falsify the bootstrap-bias hypothesis

§6.2 is a mechanism, not a measurement. It makes three predictions that this corpus has not yet
tested, listed so a later drop can kill it cleanly:

| # | prediction | refuted if |
|---|---|---|
| P1 | Larger α_end improves V_A plan quality at fixed K, monotonically over α ∈ {0.05, 0.2, 0.4, 0.6} | α=0.4/0.6 are no better than α=0.2, or non-monotone |
| P2 | α-Flow's error is **flat in K** because it is bias-dominated; MeanFlow's falls because it is discretisation-dominated | α-Flow's V_A error falls appreciably somewhere in K ∈ {5, 10} |
| P3 | The effect is objective-side, not backbone-side — it should reproduce on any bone | the same α ladder on `bbsit` shows the opposite ordering |

**P1 is the cheap one and it is Next-action #4.** P2 needs `af` K=5 and K=10 cells in V_A, which do
not exist — the ladder currently jumps 2 → 20 and cannot distinguish "flat" from "falls then
rebounds". P3 is already partly answered by the SiT-vs-U-Net history and should not need new runs.

🪤 **Do not treat §6.2 as established.** The branch structure and `dt = α·h` are read directly from
`af_diffusion.py:653-665` and are facts. The claim that bootstrap bias is *the* reason α-Flow is
flat in K is an inference consistent with every number in this DA and the avoiding one — and
untested.

---

# 🏆 Verdict

| | |
|---|---|
| **Gate 4 on α-Flow** | 🔴 **FAIL** — ceiling zv 0.80 vs MeanFlow's 1.00; MIN 4.4× worse than its own unguided; `-c` freezes 6/10 |
| **Parent DA verdict** | **unchanged at K=20** — α-Flow is dominated there. ⚠️ **At K=2 the two tie** (4/4, p=1.0000); see §6.1 |
| **What this run bought** | the **HardFlow** claim in §3, the third-engine confirmation in §4, and the reproducibility floor in §5 |
| **On my parent-DA recommendation** | **withdrawn.** "Do not run arm C on `af`" was wrong. The α-Flow verdict was already safe, but three findings only existed on the other side of the run. |

**Load-bearing weakness:** one seed, ten contexts, one α — and now a measured ~0.4 m
run-to-run floor on the projected arms that no prior DA accounted for. §3's headline effects clear
it comfortably; nothing in §2 that rests on a MIN gap under 0.1 m should be quoted without a repeat.

---

# Next actions

1. **Promote §3 into the HardFlow ledger.** "The projector's benefit scales with the plan's
   constraint error, and on a high-violation field it is free" is a claim about the flagship arm,
   supported by a 3.9× contrast at p = 0.0078 with a statistically zero distance cost. It is the
   most transferable thing in this drop.
2. **Repeat one projected cell 3× to pin the noise floor properly.** §5 is n = 2 runs. `mf` K=20/T=0.2
   with arm C is the cell to repeat, since it carries the flagship claim. Until then, treat MIN gaps
   below ~0.1 m on projected arms as unresolved.
3. **Chase τ = 0.850 on the projector side.** Three engines, 24/24 items, same solve. Extract
   `nlp_failures` totals from the run summaries and check the constraint Jacobian conditioning at
   call #2.
4. **Test α_end = 0.4 and 0.6 at K=2 in V_A — one cheap experiment, and §6.2 predicts it wins.**
   `dt = α·h`, so larger α means less bootstrap weight and a better-conditioned finite difference.
   The two α values run so far, 0.05 and 0.2, sit at 95 % and 80 % bootstrap; the ordering between
   them already points the right way. K=2 costs ~24 ms/step, so this is hours, not days. **If the
   trend continues, the V_A α-Flow verdict is not final.**
5. 🛑 **Do not spend more on α ≤ 0.2 or on K ≥ 20 for `af`.** Those regions are decided.
6. *(carried)* Tightened geometry for `diffusion` K=20/K=100 — the funnel's remaining 2 cells.

---

## Provenance

| item | value |
|---|---|
| drop | `temp/0609/II/` |
| batch | `batch_va2_20260907_141036` — 673 run-config rows, 14,102 rollout rows |
| new job | `25475` — `af`, arm C, `Job completed successfully` |
| git rev | `963faed` |
| α at eval | **0.2000**, `state_100000.pt`, sigmoid 1.0 → 0.2, clamp 0.005 |
| backbone / film | `unet` (VisualUNetTwoTime, 4.0 M) / `v1` — architecture-matched to `mf` |
| K / T | 20 / 0.2 → 4 projector calls per replan; arm C inherits T (arms B and C matched) |
| arm C | `hardflow_new-{r,c,t}` → written as `hardflow_sls-*`, SLSQP, `dof=66`, `reg_scale=1.0` |
| results dir | `…_Eaf_EPlatest/6` — **no run-tag suffix**, see §1 |
| cell size | 380 rows = 19 variants × 10 contexts × 2 geometries — matches `mf` and `fm` |
| comparator | `mf` K=20 / T=0.2, same 10 contexts by geometry fingerprint |
| statistics | exact two-sided sign test; exact paired sign-flip permutation test (stdlib only) |
