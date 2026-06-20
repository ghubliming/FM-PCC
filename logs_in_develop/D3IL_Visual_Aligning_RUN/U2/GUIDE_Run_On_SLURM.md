# Guide — Get the two paper numbers (success_rate / entropy) on SLURM

Target: DDPM-ACT image aligning, paper Table 3 = **success 0.278 ± 0.071 / entropy 0.139 ± 0.054**.
Uses the U2 upgrade (see [CHANGELOG.md](CHANGELOG.md)). No code edits needed — just `git pull` + submit
via the repo's standard `Slurm_Codes/submit.sh` wrapper (gives unified dated logs in `Slurm_Codes/logs/`).

> **U2.1 fix:** `pipeline_d3il_baseline.sh` did not forward a "paper" scale to its eval step (it always
> evaled at smoke 3×1). Added a `$5=eval_scale` arg so the one-shot pipeline can now request paper scale.

---

## 0. Pull the code
```bash
cd ~/FMPCC/FM-PCC
git pull
```

## 1. Train (one job per seed)
Paper uses **6 seeds**, 200 epochs for the vision agent. Run all 6, or start with 1 to sanity-check first.
```bash
# single seed, paper defaults (epoch=200, eval_every=20)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh ddpm_encdec_vision 42

# all 6 seeds (paper count)
for s in 0 1 2 3 4 42; do
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh ddpm_encdec_vision $s
done
```
- Checkpoints land in `logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_{s}/weights/`.
- Checkpoint selection is currently **val-loss best**, not the paper's best-task-performance (G3, not yet
  fixed — see CHANGELOG "Not done"). Treat results as an approximation of the paper protocol until G3 lands.
- Training is the long pole (hours per seed). Wait for `eval_best_*.pth` to appear before evaling that seed.

## 2. Eval — use `paper` as the scale arg (this is the only required change vs. before)
The smoke default (3×1) is too small for entropy. Always pass `paper` as eval arg `$4`:
```bash
# single seed
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh ddpm_encdec_vision 42 none paper

# all 6 seeds
for s in 0 1 2 3 4 42; do
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh ddpm_encdec_vision $s none paper
done
```
This sets `n_contexts=60, n_trajectories_per_context=18` (1080 rollouts/seed) — required for the entropy
number to be meaningful (anything below ~8 trajs/context triggers a warning in the eval log).

## 3. Or do both in one submission (recommended) — now fixed for paper scale
`pipeline_d3il_baseline.sh` chains train → eval with a SLURM dependency. As of this guide it forwards a
5th arg (`eval_scale`) straight through to the eval job's `paper` flag:
```bash
# single seed, one submission, paper-faithful eval
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh \
    ddpm_encdec_vision 42 200 all paper

# all 6 seeds
for s in 0 1 2 3 4 42; do
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh \
        ddpm_encdec_vision $s 200 all paper
done
```
This is the recommended path: one `submit.sh` call per seed, train and eval auto-chained, eval always at
paper scale. (Arg order: `agent_name seed epoch record_mode eval_scale`.)

## 4. Read the numbers
Per-seed: `logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_{s}/results_seed_{s}.json`
→ keys `success_rate`, `entropy`, `score`.

Cross-seed mean ± std: `aggregate_results.json` in the same eval output dir (or compute manually across
the per-seed JSONs if only some seeds are done).

```bash
python - <<'EOF'
import json, glob, numpy as np
sr, ent = [], []
for f in glob.glob("logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_*/results_seed_*.json"):
    d = json.load(open(f))
    sr.append(d["success_rate"]); ent.append(d["entropy"])
print(f"success_rate: {np.mean(sr):.3f} ± {np.std(sr):.3f}  (paper: 0.278 ± 0.071)")
print(f"entropy:      {np.mean(ent):.3f} ± {np.std(ent):.3f}  (paper: 0.139 ± 0.054)")
EOF
```

## 5. Sanity checks before trusting the number
- `eval_d3il_baseline.sh` log should print `--paper preset: 60 contexts x 18 trajectories` — confirm in
  the SLURM `.log` file, not just assume the flag took.
- If `n_trajs < 8` warning fires anyway, the `paper` arg wasn't passed (check `$4` position — it's the
  4th positional arg, `$3` = record_mode must not be skipped).
- One seed isn't enough to compare to a ± std paper number — need several seeds for a fair read, ideally
  all 6.

## What's NOT covered by this guide
- G3 (best-task-perf checkpoint selection) — still val-loss based, may bias results vs. paper protocol.
- Multi-seed orchestration is manual (loop above); no single "run all 6 seeds + aggregate" script exists.
