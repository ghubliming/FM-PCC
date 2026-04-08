# 04 Expected Results After Gen4 Visual-Avoiding Upgrade (Theory)

Date: 2026-04-08
Status: Pre-test expectation note
Depends on: 03_gen4_coding_execution_record.md

---

## 1) Baseline for Comparison

Use this baseline for first comparison:
1. Existing FM-v3 avoiding state path.
2. Same environment seed list and trial count.
3. Same projection/eval protocol except the new Gen4 visual config path.

Reason:
1. This Gen4 upgrade mainly introduces visual-input path and isolation wiring.
2. Fair comparison requires matched seeds and matched evaluation budget.

---

## 2) Train Results We Should Expect

Keywords: stable start, no regression, meaningful visual learning signal.

Expected:
1. Gen4 train starts normally with no import/config dispatch errors.
2. Loss decreases without NaN/Inf collapse.
3. Early train loss can be noisier than state-only baseline due to image input.
4. With enough steps, train trend should become smooth and consistent.

Not expected:
1. Immediate large gain at very early epochs.
2. Perfect monotonic curve every epoch.

Red flags:
1. Persistent shape or channel mismatch errors in image tensors.
2. Loss explosion that does not recover.
3. Training runs but policy outputs degenerate near-constant actions.

---

## 3) Eval Results We Should Expect

Keywords: parity-first, safe rollout, visual stability.

Expected:
1. Gen4 eval runs end-to-end from the new YAML/config binding.
2. Success rate is at least near baseline in first smoke tests.
3. Constraint behavior is not worse than baseline by a large margin.
4. Rollouts are stable and reproducible across matched seeds.

Not expected:
1. Guaranteed immediate large improvement over state baseline.
2. Zero variance across all random seeds.

Red flags:
1. Systematic drop in success across all seeds.
2. Frequent constraint violations with no compensating gain.
3. Inference path inconsistency between train and eval experiment keys.

---

## 4) Config-Binding Consistency Expectations

Required consistency for valid interpretation:
1. Python config file: config/avoiding-d3il-visual.py.
2. Eval YAML file: config/projection_eval_visual.yaml.
3. Experiment id in train/eval/load scripts: avoiding-d3il-visual.
4. Plan key for eval path: plan_fm_v3_avoiding_visual.

If any one is mismatched:
1. Results are not trustworthy,
2. comparison to baseline is invalid.

---

## 5) Decision Rule After First Test Round

Interpretation shortcut:
1. If Gen4 visual is within noise band of baseline and runs stably, proceed to larger training budget.
2. If Gen4 visual improves success/constraint metrics at similar compute, promote it to default visual path.
3. If Gen4 visual regresses consistently, inspect dataset/simulation visual alignment before retuning model hyperparameters.

---

## 6) First Sweep Recommendation

Start with small, controlled sweep:
1. Keep all current Gen4 defaults fixed.
2. Sweep only one variable first (for example flow steps: 5, 10, 20).
3. Keep seeds and trial count matched.

Why:
1. Single-variable sweeps isolate root causes faster.
2. Multi-parameter sweeps are harder to interpret in early bring-up.

---

## 7) Acceptance Gate for Moving Beyond Smoke

Move to larger-scale runs only if all checks pass:
1. No runtime config-resolution errors.
2. No persistent tensor shape/channel failures.
3. At least baseline-near success in short eval tests.
4. No severe constraint regression in matched setting.

If all pass:
1. run extended training,
2. run wider eval seeds,
3. collect final comparison table for promotion decision.
