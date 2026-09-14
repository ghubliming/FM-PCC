# DA — `corridor_v2` paper evaluation (U16): does FM-PCC work, and who beats whom?

*Gen15 · U16 · 2026-09-14. Source: **`temp/1409/batch_uav_20260914_091148`** (`uav_aggregated_long.csv` mask `all`,
`per_rollout_detail.csv`, `data_quality.csv`, `candidate_axes.csv`). Child logs: `temp/1409/2026-09-13/`.
Setup: [`CHANGELOG_20260913_corridor_v2_paper_run.md`](CHANGELOG_20260913_corridor_v2_paper_run.md) ·
fixes: [`CHANGELOG_20260913_u16_fix1_fix2_FULL_REVIEW.md`](CHANGELOG_20260913_u16_fix1_fix2_FULL_REVIEW.md).*

---

## 0. TL;DR

1. **The projector works, on every engine.**
   - Unprojected, every engine hits the slide on every flight: collision-free **0/12**, 42–56 violation steps.
   - With DPCC (`-bounds_free-pdes-tightened`) at K = 3/5, **mf, af and diffusion are collision-free 12/12** with 0 violations (p = 7 × 10⁻⁷ vs unprojected).
2. **Projected flow engines complete the course; projected diffusion does not.** Every projected mf/fm/af cell
   crosses the finish line **12/12**. **Projected diffusion crosses it 0/36**: safe and collision-free, but never at
   the line within the 396-step budget.
3. **Crossed-line S&C** (finish line + zero violations): **mf 1.00 and af 1.00** at K = 3 and 5 · fm ≤ 0.17 · **diffusion
   0.00**. mf/af vs fm and vs diffusion: p = 7 × 10⁻⁷.
4. **Strict S&C** (goal point + zero violations, the pillars/s_curve metric): af K3 **0.33**, mf K3 **0.25**, fm 0.00,
   diffusion 0.00. **Capped by design:** after the forced detour only route L can reach its goal point (predicted in U16
   §8/§10), and af K3 gets L **4/4**. Engine differences on strict S&C are **not significant** at n = 12 (p 0.09–1.0).
5. **Ranking on this scene: {af ≈ mf} > fm > diffusion.**
   - The top two are separated only by strict S&C, 4/12 vs 3/12, one rollout.
   - mf > fm and {mf, af} > diffusion are solid on collision-free / crossed-line S&C.
   - This is nominally consistent with the thesis ladder af > mf > fm > diffusion. The af-over-mf step is inside noise.
6. **Cost:** projected flow cells 29 ms (K1), 86–102 ms (K3), 122–144 ms (K5) per step, vs diffusion 626–651 ms.
   HardFlow is cheaper still (45–125 ms) but not collision-free (0.00–0.58).

---

## 1. Provenance and gates

| | |
|---|---|
| jobs | `eval_k_sweep` 25750–25758 → 15 eval children 25760–25772, 25775, 25776 · diffusion train **25773** → eval **25774** |
| candidates (tag `u17cv2`) | mf C52/C56/C60 (K1/3/5) · fm C41/C45/C47 · af C29/C32/C33 · diffusion C38 (K20) |
| models | mf `MeanFlowODE_9D_dp0.5_bbunet` **3.97 M** · af `AlphaFlowODE_9D_as1_ae0.2_bbunet` (EP latest) **3.97 M** · fm `FlowMatchingODE_9D` **3.96 M** · diffusion `GaussianDiffusion_9D_K20` (trained by 25773, `action_weight 1`) **3.96 M**. All U-Net |
| geometry | `corridor_cv2s_bounds+dynamics+geo_bounds+halfspace+obstacles` = `corridor_v2_slide` |
| projector | every projected arm `-bounds_free-pdes-tightened`; HardFlow B = 1 (`hf*`) and B = 4 temporal (`hf-t*`) |
| run | seed 6 · **n = 12 per cell** (4 per route L/C/R) · `pid_stopgo` · mpc4 · T = 0.5 |

**Settings verified in all 16 eval child logs (not assumed):**
- MuJoCo walls `y = -1.00 / +1.00` (the corridor_v2 scene);
- `SAFE_EPS_FRAC=1.0` with `Constant data in actions[1] … eps=2.188e-02`;
- `n_trials=12`;
- the af checkpoint loaded is the U-Net / α-end 0.2 one.

**Gates:**
- **52/52 cells**, n = 12 each;
- 0 circuit-breaker trips, 0 missing timing, no degenerate HardFlow cell;
- divergence aborts 0.00 and MuJoCo contacts 0.000 on every cell.

## 2. Metrics used

- **collision-free (`cfree`):** zero violating steps, scored on the **real** drone.
- **crossed:** the drone was ever on the goal side of the finish line.
- **strict success:** ever within 0.30 m of the route's goal point.
- **S&C strict / S&C crossed:** the respective success **and** collision-free.
- **steps:** 396 = budget exhausted without latching the strict goal.
- **ms:** `fm_ms` + `proj_ms` per control step.

⚠️ **Strict success is geometry-capped on this slide.** It forces the drone below y ≈ −0.37 at the exit, and the corridor
model never steers back sideways. So routes C (goal y 0) and R (goal y +0.12) cannot reach their goal points under any
projector; L (goal y −0.12) is borderline. Stated before the run in `CHANGELOG_20260913_u16fix_pdes_binding.md` §8/§10.

---

## 3. The projector's effect (unprojected `diffuser` vs projected DPCC)

| engine (params) | K | unprojected: cfree / viol / strict succ | projected `dpcc-r*` | `dpcc-c*` | `dpcc-t*` |
|---|---|---|---|---|---|
| **diffusion** (3.96 M) | 20 | 0.00 / 49.8 / 1.00 | **1.00 / 0.0** / 0.00 | **1.00 / 0.0** / 0.00 | **1.00 / 0.0** / 0.00 |
| **mf** (3.97 M) | 1 | 0.00 / 54.2 / 1.00 | 0.50 / 1.2 / 0.75 | 0.17 / 1.4 / 0.58 | 0.17 / 1.8 / 0.67 |
| mf | 3 | 0.00 / 54.8 / 1.00 | **1.00 / 0.0** / 0.25 | **1.00 / 0.0** / 0.17 | **1.00 / 0.0** / 0.25 |
| mf | 5 | 0.00 / 54.6 / 1.00 | **1.00 / 0.0** / 0.00 | **1.00 / 0.0** / 0.00 | **1.00 / 0.0** / 0.00 |
| **fm** (3.96 M) | 1 | 0.00 / 42.6 / 1.00 | 0.00 / 5.2 / 0.58 | 0.00 / 5.0 / 0.67 | 0.00 / 5.2 / 0.67 |
| fm | 3 | 0.00 / 46.1 / 1.00 | 0.00 / 2.3 / 0.33 | 0.00 / 1.9 / 0.33 | 0.00 / 2.1 / 0.33 |
| fm | 5 | 0.00 / 47.6 / 1.00 | 0.00 / 1.6 / 0.33 | 0.08 / 1.7 / 0.33 | 0.17 / 1.5 / 0.33 |
| **af** (3.97 M) | 1 | 0.00 / 54.8 / 1.00 | 0.25 / 1.8 / 0.75 | 0.17 / 1.7 / 0.67 | 0.33 / 1.2 / 0.92 |
| af | 3 | 0.00 / 56.1 / 1.00 | **1.00 / 0.0** / 0.33 | **1.00 / 0.0** / 0.33 | **1.00 / 0.0** / 0.33 |
| af | 5 | 0.00 / 56.2 / 1.00 | **1.00 / 0.0** / 0.17 | **1.00 / 0.0** / 0.17 | **1.00 / 0.0** / 0.17 |

**Reading:**
- **Unprojected:** every engine flies straight through the slide and reaches every goal (strict success 1.00, goal distance 0.29 m).
- **Projected, K ≥ 3:** mf, af and diffusion drop to **zero violations**. fm keeps **1.5–2.3** violation steps per flight: shallow, but never collision-free.
- **Projected, K = 1:** violations fall 8–45× (mf/af 30–45×, fm 8×) but not to zero (a single ODE step to project into). The push is shallower, so more
  goal points stay reachable (mf `dpcc-r*` reaches C 4/4).

## 4. Engine comparison at matched K

| K | engine | cell | **S&C strict** | **S&C crossed** | strict succ | crossed | cfree | viol | steps | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | **diffusion** | `dpcc-r*` | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 | 0.0 | 396 | 650.5 |
| 20 | **diffusion** | `dpcc-c*` | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 | 0.0 | 396 | 650.9 |
| 20 | **diffusion** | `dpcc-t*` | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 | 0.0 | 396 | 625.5 |
| 3 | mf | `dpcc-r*` | 0.25 | **1.00** | 0.25 | 1.00 | 1.00 | 0.0 | 373 | 98.9 |
| 3 | mf | `dpcc-c*` | 0.17 | **1.00** | 0.17 | 1.00 | 1.00 | 0.0 | 381 | 99.4 |
| 3 | mf | `dpcc-t*` | 0.25 | **1.00** | 0.25 | 1.00 | 1.00 | 0.0 | 374 | 99.1 |
| 3 | mf | `hf*` | 0.08 | 0.42 | 0.33 | 1.00 | 0.42 | 0.8 | 368 | 47.2 |
| 3 | mf | `hf-t*` | 0.08 | 0.42 | 0.33 | 1.00 | 0.42 | 0.8 | 365 | 76.0 |
| 3 | fm | `dpcc-r*` | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 2.3 | 359 | 86.2 |
| 3 | fm | `dpcc-c*` | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 1.9 | 360 | 86.1 |
| 3 | fm | `dpcc-t*` | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 2.1 | 357 | 86.4 |
| 3 | fm | `hf*` | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 3.2 | 360 | 45.1 |
| 3 | fm | `hf-t*` | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 3.0 | 357 | 79.3 |
| 3 | af | `dpcc-r*` | **0.33** | **1.00** | 0.33 | 1.00 | 1.00 | 0.0 | 368 | 97.7 |
| 3 | af | `dpcc-c*` | **0.33** | **1.00** | 0.33 | 1.00 | 1.00 | 0.0 | 365 | 97.8 |
| 3 | af | `dpcc-t*` | **0.33** | **1.00** | 0.33 | 1.00 | 1.00 | 0.0 | 368 | 101.9 |
| 3 | af | `hf*` | 0.00 | 0.25 | 0.33 | 1.00 | 0.25 | 1.2 | 362 | 46.6 |
| 3 | af | `hf-t*` | 0.08 | 0.58 | 0.33 | 1.00 | 0.58 | 0.6 | 362 | 74.7 |
| 5 | mf | `dpcc-r*` / `-c*` / `-t*` | 0.00 | **1.00** | 0.00 | 1.00 | 1.00 | 0.0 | 396 | 142–144 |
| 5 | mf | `hf*` / `hf-t*` | 0.00 | 0.00 / 0.17 | 0.33 | 1.00 | 0.00 / 0.17 | 1.6 / 1.2 | 364–367 | 79.0 / 124.7 |
| 5 | fm | `dpcc-r*` / `-c*` / `-t*` | 0.00 | 0.00 / 0.08 / 0.17 | 0.33 | 1.00 | 0.00 / 0.08 / 0.17 | 1.5–1.7 | 358–361 | 122–134 |
| 5 | fm | `hf*` / `hf-t*` | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 2.6 / 2.1 | 360–362 | 73.2 / 113.4 |
| 5 | af | `dpcc-r*` / `-c*` / `-t*` | 0.17 | **1.00** | 0.17 | 1.00 | 1.00 | 0.0 | 381 | 139–142 |
| 5 | af | `hf*` / `hf-t*` | 0.00 | 0.00 / 0.17 | 0.33 | 1.00 | 0.00 / 0.17 | 1.9 / 1.7 | 362–364 | 83.5 / 120.7 |
| 1 | mf | `dpcc-r*` | **0.33** | 0.50 | 0.75 | 1.00 | 0.50 | 1.2 | 337 | 29.2 |
| 1 | fm | `dpcc-r*` / `-c*` / `-t*` | 0.00 | 0.00 | 0.58–0.67 | 1.00 | 0.00 | 5.0–5.2 | 323–335 | 29 |
| 1 | af | `dpcc-t*` | **0.33** | 0.33 | 0.92 | 1.00 | 0.33 | 1.2 | 312 | 29.0 |

`*` = `-bounds_free-pdes-tightened`; `hf` = `hardflow_sls`. HardFlow at K = 3/5 runs genuine NLP steps ✅.
mf K1 `dpcc-c*`/`-t*` and af K1 `dpcc-r*`/`-c*` are in §3 (S&C strict 0.08–0.17).

### 4.1 Per route: where strict success is won and lost (`dpcc-t*`, S&C strict L / C / R, 4 flights each)

| engine | K | L | C | R | reading |
|---|---|---|---|---|---|
| diffusion | 20 | 0/4 | 0/4 | 0/4 | never reaches the finish line (goal distance L 0.40, C 0.50, R 0.63) |
| mf | 3 | **3/4** | 0/4 | 0/4 | C, R cross the line collision-free; goal point geometrically out of reach |
| mf | 5 | 0/4 | 0/4 | 0/4 | same, and L also ends just outside 0.30 m (goal distance 0.42) |
| fm | 3 | 0/4 | 0/4 | 0/4 | L reaches its goal 4/4, but with 2.5 violation steps |
| af | 3 | **4/4** | 0/4 | 0/4 | L perfect on all three selections |
| af | 5 | 2/4 | 0/4 | 0/4 | |
| mf | 1 | 0/4 | 1/4 | 0/4 | (`dpcc-r*`: L 1/4, **C 3/4**, R 0/4) |
| af | 1 | 0/4 | 1/4 | **3/4** | shallow push at K = 1, so R can still reach its goal |

## 5. DA Target — the baseline's top projection cell

Candidate cells: diffusion K = 20 projected. All tie on S&C (0.00 / 0.00), collision-free (1.00) and steps (396),
so the tie is broken by time. **Target = `dpcc-t*`:** S&C 0.00 / 0.00 · cfree 1.00 · crossed **0.00** · 396 steps · **625.5 ms/step**.

| engine | cell | S&C strict / crossed | cfree | steps | ms | vs Target |
|---|---|---|---|---|---|---|
| af | K3 `dpcc-r*` | **0.33 / 1.00** | 1.00 | 368 | 97.7 | **Pareto-dominant** (higher S&C both, equal cfree, fewer steps, 6.4× cheaper) |
| mf | K3 `dpcc-r*` | **0.25 / 1.00** | 1.00 | 373 | 98.9 | **Pareto-dominant** |
| mf | K5 `dpcc-t*` | 0.00 / **1.00** | 1.00 | 396 | 143.9 | dominant on crossed S&C + time; equal steps and strict S&C |
| af | K1 `dpcc-t*` / mf K1 `dpcc-r*` | 0.33 / 0.33–0.50 | 0.33–0.50 | 312–337 | 29 | **trade-off:** higher S&C, fewer steps, 21× cheaper, but **fewer collision-free flights** |
| fm | K3 `dpcc-t*` | 0.00 / 0.00 | **0.00** | 357 | 86.4 | **trade-off:** fewer steps, 7× cheaper, crosses the line 12/12, but **not collision-free** |

## 6. HardFlow vs DPCC (same engine, same K, constraint-matched)

At K = 3 and 5, HardFlow is **1.15–2.1× cheaper** than the DPCC cell of the same engine:
- mf K3: `hf*` 47.2 ms and `hf-t*` 76.0 ms vs `dpcc-t*` 99.1 ms;
- about half its cost is the generator.

It is **clearly less collision-free**: 0.00–0.58 vs 1.00 for mf/af DPCC (mf K3 `hf-t*` 5/12 vs `dpcc-t*` 12/12, p = 0.005).
Violations are shallow (0.6–3.2 steps). HardFlow does not beat the DPCC projector on this scene. It is a speed-for-safety **trade-off**.

## 7. Who beats whom (benchmark hierarchy)

| comparison | result | evidence |
|---|---|---|
| **FM-PCC works** (projector vs none) | ✅ **yes** | cfree 0/12 → 12/12 for mf/af (K ≥ 3), p = 7 × 10⁻⁷; violations 42–56 → 0 |
| **flow family > diffusion baseline** | ✅ **yes** (mf, af); fm only on completion | crossed S&C mf/af 1.00 vs 0.00, p = 7 × 10⁻⁷; diffusion never crosses the line when projected; 4.3–22× cheaper |
| **mf > fm** | ✅ **yes** | cfree mf 12/12 vs fm 0/12 (K3 `dpcc-t*`, p = 7 × 10⁻⁷); crossed S&C 1.00 vs 0.00 |
| **af > mf** | 🟡 **nominal only** | strict S&C af 0.33 vs mf 0.25 (K3), 0.17 vs 0.00 (K5): one or two rollouts, p = 1.0; identical cfree / crossed S&C |
| **fm > diffusion** | 🟡 **partial** | fm completes the course 12/12, diffusion 0/36, but fm is not collision-free, so S&C crossed is 0.00 vs 0.00 at K3 and 0.17 vs 0.00 at K5 |
| **HardFlow > DPCC** | ❌ **no** | cheaper, less collision-free (§6) |

**Nominal ranking: af ≈ mf > fm > diffusion**, the order of the thesis ladder. Only the af-over-mf step is not supported
at this n.

## 8. Observations to follow up (not claims)

- **Projected diffusion stalls short of the line.** It is collision-free and at goal distance 0.40–0.63 on L/C and
  `dpcc-t*` R, with 1.87–2.15 m on `dpcc-r*`/`-c*` R. It uses all 396 steps and never crosses. The unprojected diffusion
  finishes normally (264 steps). The likely mechanism is slow forward progress under projection at K = 20 (10 projected
  denoising steps per plan), but **rollout traces are not in this drop**, so it is not verified.
- **mf K5 misses L by ~0.12 m** where mf K3 does not. The deeper tightened ride line puts the L goal point
  (y −0.12) just outside 0.30 m.
- **K = 1 trades violations for goal reach.** The projector moves the plan less at one step: goals are more
  reachable, but 1–5 violation steps remain.

## 9. What this licenses

**Supported (corridor_v2, seed 6, n = 12, architecture-matched U-Nets 3.96–3.97 M):**
- FM-PCC's projector turns every engine's slide-crossing flight into a collision-free one at K ≥ 3 (mf, af, diffusion), and nearly so for fm.
- Projected mf and af **complete the course collision-free on 12/12 flights**; the projected DDPM baseline completes it **0/36**.
- mf and af beat naive FM on collision-freeness; all flow engines are 4.3–22× cheaper per step than the projected baseline.

**Not supported:**
- **af > mf.** Inside noise.
- **Strict-success rankings.** Geometry-capped; only L can succeed.
- **HardFlow > DPCC.** Reversed on constraints.
- **Multi-seed or CI claims.**
- **A reproduction of DPCC's `action_weight = 10` baseline.**
- **Pooling with pillars/s_curve.** Different projector configuration (`-pdes-tightened`, `FRAC = 1.0`).
- **Anything about the original `corridor` scene.** This is `corridor_v2`, widened walls and a virtual slide.

**Next (optional):**
- Seeds 7–8 for a confidence interval.
- The rollout folders for projected diffusion, to verify the stall (§8).
