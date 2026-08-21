# PLAN — Gen16: Visual-Avoiding Mix-ML

**Date opened:** 2026-08-21 · **Status:** code complete, unverified on hardware
**Changelog:** [`CHANGELOG_Gen16_coding1.md`](./CHANGELOG_Gen16_coding1.md)

---

## 1. The question

> **How do MeanFlow (Gen3v6) and α-Flow (Gen3v7) behave on visual AVOIDING, on the
> architecture-matched U-Net bone, against diffusion-DPCC and naive visual FM?**

Gen14 answered the analogous question for visual *aligning*. Gen9 built visual avoiding but
only ever ran two engines on it (`fm`, `diffusion`), each in its own folder, on a 643-line
eval that predates the DPCC-variant harness, the HardFlow arm and the K sweep.

Gen16 is the missing cell: **four engines × three guidance arms × the avoiding task.**

---

## 2. Scope

| axis | Gen16 |
|---|---|
| **ML engine** | `diffusion` (Gen6V4) · `fm` (Gen7) · `mf` (Gen3v6) · `af` (Gen3v7) |
| **Guidance arm** | A `diffuser` (unguided) · B `dpcc-*` (DPCC projector) · C `hardflow_new*` |
| **Environment** | D3IL visual avoiding — 6-D, single bp-cam, `ObstacleAvoidanceEnv` |
| **Backbone** | `VisualUNet` (the baseline). DiT/SiT reachable via `MIX_BONE=*`, secondary |
| **Seeds** | 6 7 8 9 10 |

**Out of scope, deliberately:**

- A state-only avoiding arm. Those engines already exist as their own siblings
  (`FM_v3_test`, `FM_v3_meanflow_test`, `FM_v3_alphaflow_test`). The train script **refuses**
  `if_vision=False` with that explanation rather than creating a duplicate lineage.
- iMF. Refuted by Gen13 CLOSURE I; Gen14 excluded it and so does Gen16.
- Any edit to Gen7 / Gen9 / Gen14. Isolation is total (changelog §3).

---

## 3. Structural principle

> **Gen16 = Gen14 @ HEAD, package renamed, with the task swapped — and the task lives in
> exactly two files.**

Those two files are `models/visual_spec.py` (cameras + dims) and `datasets/sequence.py` (the
data). Everything else that differs from Gen14 is either a Gen9 fix being re-applied, or a
consequence of the avoiding harness being a gym loop rather than a D3IL sim.

This is enforced, not asserted: **gate A0** walks both packages and requires every file not
on an explicit 18-entry ledger to be byte-identical to Gen14's after reversing the rename,
and **gate A2** requires no module outside `visual_spec.py` to name a camera or a dimension.

Consequence: Gen16's reproduction of Gen6V4/Gen7/Gen3v6/Gen3v7 is a property of the file
layout, not something a numerical test has to establish — the same guarantee Gen14 §3.1 made.

---

## 4. Why a new sibling rather than upgrading Gen9 or merging into Gen14

Measured before deciding (changelog §2): the aligning→avoiding delta is 8 files / ~640 lines.
The Gen9→Gen14 eval delta is ~2500 lines. Porting the small delta onto the big frame is
strictly cheaper than porting the big frame onto the small one, and it leaves both parents
untouched.

Precedent: Gen15 did exactly this for UAV (`mix_uav/` = Gen11 + the registry, 383 new lines).

---

## 5. Risks and how each is handled

| risk | handling |
|---|---|
| A camera-count literal survives in one backbone → that bone plans half-blind | gate A2 (AST-level, comments exempt) |
| Gen16 silently diverges from Gen14's engines | gate A0, 18-entry ledger |
| `diffusion_loadpath` does not reproduce `exp_name` → eval dies after GPU alloc | one watch list drives both; gate A4 |
| Arm C fan ≠ arm B fan → timing comparison void (B4_PARITY) | default 4 both sides; gate A9, runs offline |
| Unmatched K across arms → the comparison is not a comparison | `flow_steps_v3 == flow_steps` per block; gate A4; `eval_k_sweep.sh` takes ONE K list |
| mf/af JVP breaks on a single-camera payload | pre-encoded latent; gate A7 takes a real loss step per arm |
| Window leakage inflates `test_loss`, corrupts `state_best` | episode-level split, both trainers |
| Arms select `state_best` under different criteria | EMA-consistent `test()` in both trainers |
| Cross-generation geometry drift makes tables invalid | the yaml's geometry block is byte-identical to Gen9's and Gen3v6's |

---

## 6. Order of work

1. ✅ Measure the delta.
2. ✅ Build the sibling; gates A0/A1/A2/A9 green offline.
3. ⬜ **Confirm the Gen9 avoiding image dataset still exists on i6-gpu-1.**
4. ⬜ Run the full gate battery on the cluster (A3–A8).
5. ⬜ Train the `diffusion` arm, seed 6. **Parity-check against Gen9's
   `diffuser_visual_avoiding` June results — this is the gate on the generation.**
6. ⬜ Train `fm`, `mf`, `af` at seed 6. Compare on unguided task success.
7. ⬜ K sweep {1, 2, 5, 10, 20}, same list for every arm.
8. ⬜ Widen to seeds 6–10 only for the arms that survived step 6.
9. ⬜ DA pass; then decide whether a DiT/SiT bone is worth the confound.

---

## 7. Success criteria

- **Necessary:** the `diffusion` arm reproduces Gen9's visual-avoiding DPCC baseline on
  rollout metrics (not on loss curves — the split and the `state_best` criterion changed).
- **The claim:** at matched K and matched success+constraints, `mf` or `af` is
  Pareto-dominant over `fm` and over `diffusion` on steps and `avg_time`, **on the `unet`
  bone**. Anything less is a trade-off, not a win.
- **The stretch:** arm C (HardFlow) beats the DPCC projector at a *lower* projection
  threshold.
