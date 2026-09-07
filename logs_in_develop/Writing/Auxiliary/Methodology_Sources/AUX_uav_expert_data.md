# AUX — UAV expert data: the two-layer pipeline and the D3IL-shaped schema

**Thesis home:** `sec:setup:data` · **TARGET** §5.2
**Primary dev logs:** `Gen11/Epoch4_expert_data/` (`init_0/METHODOLOGY.md`, `U9_Smooth_Trajectories/`,
`U10_Stress_Tests/`, `U11_MJPC_Controller_Dataset/`, `U8X_Stop_and_Go_Ideas/`) ·
`Gen11/Epoch8_.../DESIGN_dataset_pid_vs_mjpc.md` §1

---

## 1. What the thesis must say

There is no public expert dataset for this task, so one was generated. The design decision that
matters — and that a referee will probe — is that the demonstrations are **synthesised in two
decoupled layers**, geometry first and physics second:

**Layer 1 — geometric reference.** A pure-mathematics function
`traj_fn(t) → (p_des, v_des, a_des, yaw_des)`. No MuJoCo, no physics. It is a `blended_path`: cubic
fillets through per-scene hard-coded waypoints, constructed so the path provably stays inside the
intended homotopy channel with **≥ 8 cm clearance**.

**Layer 2 — physics execution.** The cascaded PID tracks `p_des(t)` inside the MuJoCo scene at
100 Hz, and the resulting rollout is recorded.

**Why this matters for the thesis, in one sentence:** the expert is *not* an optimal controller and
does not claim to be — it is a geometrically-safe reference tracked by a classical controller. That
sets the ceiling the learned policies are measured against, and it is an honest ceiling to state.

### 🔑 The subtlety that must not be skipped

```
actions = Δp_des = np.diff(targets, axis=0)
```

**The recorded action is the geometry layer differenced — not the PID output, and not `Δp`.** The
policy is therefore trained to imitate the *reference generator*, not the *controller*. This is what
makes the learned model a planner rather than a controller, and it is what allows the same
plan → projector → PID chain to be swapped at evaluation time. If the thesis states only one
implementation detail about the UAV data, it should be this one.

---

## 2. The schema — deliberately D3IL-shaped

| field | shape | content |
|---|---|---|
| `obs` | `(T, 9)` | `[p_des(3) \| p(3) \| v(3)]` |
| `actions` | `(T−1, 3)` | `Δp_des`, the differenced reference |
| `targets` | `(T, 3)` | the `p_des` sequence (+ a constant per-episode noise offset) |

Written by `uav_expert_data_collect/dataset_writer.py:rollout_to_episode`. The shape mirrors the
D3IL pickle convention so the existing dataset/normaliser/training stack is reused unchanged — that
reuse is itself a methodological point (the embodiment changes, the harness does not).

---

## 3. Episode generation and rejection

| aspect | setting | source |
|---|---|---|
| randomisation | random start position; altitude `z ∈ [0.90, 1.30]`; random duration within scene bounds | `DESIGN_dataset_pid_vs_mjpc.md` §1.1 |
| homotopy label | chosen **deterministically and cycled** for class balance | ibid. |
| rejection | contact fraction too high, **or** drone crashes (floor < 0.5 m) | ibid. §1.2 |
| clearance guarantee | ≥ 8 cm from obstacle surfaces, by construction of the fillets | ibid. §1.2 (U9) |
| corpus size | **1769 episodes** replayed in Epoch 5 | `Epoch5/init_0/METHODOLOGY.md` §0 |

**Per-episode constant noise offset on `targets`** is worth a sentence — it is a deliberate
augmentation, and a reader who finds it in the code without explanation will assume a bug.

---

## 4. Holes — fill before writing

- [ ] **Per-scene episode counts and the train/val/test split are not recorded here.** 1769 is the
      total replayed; the breakdown by scene and the split protocol must be established.
- [ ] **The rejection *rate* is not recorded** — how many trials were thrown away, per scene. This is
      a one-line statistic that materially affects how the dataset should be read.
- [ ] **Waypoint geometry is quoted only by example** (`_Y_L = −1.11`, `_Y_R = +1.11` for pillars).
      The full per-scene waypoint table should come from `trajectories.py`.
- [ ] **Magnitude of the `targets` noise offset** is not stated.
- [ ] **`U11_MJPC_Controller_Dataset`** exists — establish whether an MJPC-collected dataset was
      actually produced and whether any reported result uses it (see `AUX_uav_control_stack.md`).
- [ ] **Expert performance baseline.** The thesis needs the expert's own tracking error and
      constraint-satisfaction on each scene as the reference row of every UAV table. It is
      not tabulated anywhere in the Gen11 logs.
