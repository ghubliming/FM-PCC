# What The Old FM-v2 Test Means (Clear Statement)

Date: 2026-04-06

---

## 1) One-line meaning

The old reported results are an **Eval ODE=20** test with **Beta-time FM training**, not an Eval ODE=10 test.

---

## 2) Exact setup that produced the old results

During the mismatch period:
1. train block (`flow_matching_v2`) had `ode_inference_steps_v2 = 10`,
2. eval block (`plan_fm_v2`) had `ode_inference_steps_v2 = 20`,
3. `vf_time_bins_v2 = 20`,
4. Beta time sampling was active (`time_beta_alpha_v2=1.5`, `time_beta_beta_v2=1.0`).

---

## 3) What this means for interpretation

1. Label those old numbers as: **Eval ODE=20**.
2. Do not label them as: **Eval ODE=10**.
3. They reflect Beta-time FM training + 20-step eval rollout.

---

## 4) Why train-side ODE=10 did not redefine the old metric label

In current FM-v2 implementation:
1. training loss is pointwise FM velocity MSE,
2. training does not backprop through an ODE rollout loop,
3. ODE step count mainly controls sampling/eval rollout behavior.

So the reported test identity is determined by eval rollout setting, which was 20.

---

## 5) Correct short sentence to reuse

Use this exact sentence:

"Our old FM-v2 results should be interpreted as Beta-time FM training evaluated with ODE=20 rollout; they are not an ODE=10 evaluation benchmark."

---

## 6) Current state now

Current config is aligned:
1. `flow_matching_v2.ode_inference_steps_v2 = 10`
2. `plan_fm_v2.ode_inference_steps_v2 = 10`
