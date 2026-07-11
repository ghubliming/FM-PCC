# Memory Index

- [Docker has no Python; run everything on Slurm cluster](docker-no-python-cluster-only.md) — never run Python locally, note "run on cluster"
- [Never auto-commit, no Claude co-author](no-auto-commit-no-coauthor.md) — user commits manually; omit Co-Authored-By trailer
- [FM-PCC dev-logs navigation](fmpcc-dev-logs-navigation.md) — MASTER_TEST_HISTORY.md is the index; repo based on aux_repo/dpcc; unfinished project
- [Changelog after coding tasks](changelog-after-coding-tasks.md) — write changelog MD into logs_in_develop/<gen>/<epoch>; ask for epoch if unknown; concise by default, cover all changes
- [Slurm_Codes/sbatch is the real cluster entrypoint](slurm-sbatch-is-real-entrypoint.md) — submit via Slurm_Codes/submit.sh (not raw sbatch); update sbatch scripts alongside code changes; NEVER break GPU/EGL isolation (see logs_in_develop/SLURM_GPU_IT_WARNING)
- [config/ folder convention](config-folder-convention.md) — .py = train+eval setup entries (per-experiment blocks); .yaml = constraint-projection configs (DPCC convention)
- [Don't self-edit MASTER_TEST_HISTORY.md](dont-self-edit-master-test-history.md) — never touch the master index unless explicitly told; offer, don't add
