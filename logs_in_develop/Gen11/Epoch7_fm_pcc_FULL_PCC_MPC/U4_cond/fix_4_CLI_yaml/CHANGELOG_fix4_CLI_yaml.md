# CHANGELOG — fix_4_CLI_yaml: n_trials yaml-default broken in all 4 shell scripts

**Date**: 2026-06-28
**Parent**: [../fix_3/CHANGELOG_fix3_seed_ntrials.md](../fix_3/CHANGELOG_fix3_seed_ntrials.md)

---

## Root cause

fix_3 correctly made `eval_fm_uav.py` read `n_trials` from `config/uav_projection.yaml`
when `--n-trials` is absent. But **four SLURM shell scripts** all had a hardcoded bash
default that silently forced the CLI value before python could see the yaml:

```bash
# The offending pattern in all 4 scripts:
NTRIALS="${3:-20}"   # ← always 20 when $3 not given
...
python eval_fm_uav.py ... --n-trials "$NTRIALS"   # ← always passes 20
```

Because `--n-trials` was always present on the python command line, `args.n_trials` was
never `None`, so the yaml branch in `main()` was never reached.

**Observed symptom**: user set `n_trials: 10` in `config/uav_projection.yaml` but always
got 20 rollouts.

---

## Scripts fixed (all 4)

| Script | Bug | Fix |
|---|---|---|
| `eval_fm_uav.sh` | `NTRIALS="${3:-20}"`, always passes `--n-trials $NTRIALS` | `NTRIALS="${3:-}"`, conditional `${NTRIALS:+--n-trials "$NTRIALS"}` |
| `fm_uav_pipeline.sh` | Same; also had wrong seed default `SEED="${2:-5}"` | Same n_trials fix; seed corrected to `SEED="${2:-6}"` |
| `eval_all_scenes.sh` | `NTRIALS="${3:-20}"`, passes to eval_fm_uav.sh | `NTRIALS="${3:-}"` |
| `fm_uav_all_pipeline.sh` | `NTRIALS="${3:-20}"`, passes to eval_fm_uav.sh | `NTRIALS="${3:-}"` |

### The bash pattern (eval_fm_uav.sh only — the actual python caller)

```bash
# Before
NTRIALS="${3:-20}"
python eval_fm_uav.py ... --n-trials "$NTRIALS" ...

# After
NTRIALS="${3:-}"   # empty when $3 not given
python eval_fm_uav.py ... ${NTRIALS:+--n-trials "$NTRIALS"} ...
# ${VAR:+word} expands to 'word' only when VAR is non-empty → flag absent when empty
```

The three orchestrator scripts (`fm_uav_pipeline.sh`, `eval_all_scenes.sh`,
`fm_uav_all_pipeline.sh`) pass `"$NTRIALS"` to `eval_fm_uav.sh` as its `$3`. When
`NTRIALS=""`, `eval_fm_uav.sh` receives `$3=""`, sets its own `NTRIALS=""`, and the
conditional expansion again produces no flag → python reads yaml. ✓

---

## Complete CLI flow after fix

```
uav_projection.yaml: n_trials: 10
                         ↓
eval_fm_uav.sh: NTRIALS="${3:-}" = ""
                         ↓
python call: no --n-trials flag
                         ↓
parse_args(): args.n_trials = None
                         ↓
main(): _trials_from_cli = False → args.n_trials = yaml value = 10
        prints: [ eval ] n_trials=10  (source: config/uav_projection.yaml)
```

CLI override still works:
```bash
./submit.sh eval_fm_uav.sh corridor "6" 5
# → NTRIALS=5 → --n-trials 5 → args.n_trials=5 → _trials_from_cli=True → uses 5
# prints: [ eval ] n_trials=5  (source: --n-trials CLI)
```

---

## Also fixed: wrong seed default in fm_uav_pipeline.sh

`SEED="${2:-5}"` → `SEED="${2:-6}"` (UAV trains at seed 6; 5 was a leftover D3IL default).

---

## Verification

Traced all 4 scripts end-to-end. Key cases tested (simulated in python):
- No `$3` → NTRIALS="" → no `--n-trials` flag → `args.n_trials=None` → yaml read ✓
- `$3=5`  → NTRIALS=5  → `--n-trials 5`       → `args.n_trials=5`    → CLI wins ✓
- `$3=""`  → NTRIALS="" → same as no $3 ✓

---

## How to revert

```bash
git checkout -- Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh \
                Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh \
                Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh \
                Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh
```
