# Memory Index

- [Python OK for debug/analysis here; pipeline runs on cluster](docker-no-python-cluster-only.md) — interpreter allowed for debugging always & data analysis on request; NEVER run training/eval/pipeline locally (cluster job)
- [Never auto-commit, no Claude co-author](no-auto-commit-no-coauthor.md) — user commits manually; omit Co-Authored-By trailer
- [No unrequested CODE edits — but always update MDs](no-unrequested-code-edits.md) — code/config needs a go-ahead; report/analysis MDs: write the update immediately, never ask
- [FM-PCC dev-logs navigation](fmpcc-dev-logs-navigation.md) — MASTER_TEST_HISTORY.md is the index; repo based on aux_repo/dpcc; unfinished project
- [Changelog after coding tasks](changelog-after-coding-tasks.md) — write changelog MD into logs_in_develop/<gen>/<epoch>; ask for epoch if unknown; concise by default, cover all changes
- [Slurm_Codes/sbatch is the real cluster entrypoint](slurm-sbatch-is-real-entrypoint.md) — submit via submit.sh; update scripts w/ code changes; never break GPU/EGL isolation; --time = 2x expected (24h cap); never tqdm/live bars in batch logs
- [config/ folder convention](config-folder-convention.md) — .py = train+eval setup entries (per-experiment blocks); .yaml = constraint-projection configs (DPCC convention)
- [Don't self-edit MASTER_TEST_HISTORY.md](dont-self-edit-master-test-history.md) — never touch the master index unless explicitly told; offer, don't add
- [Archived_Codes = dead code](archived-codes-is-dead-code.md) — Archived_Codes/ & *(legacy/Abandoned/Outdated) folders are dead/wrong; never run/edit/list-as-work; read-only to learn in rare cases
- ["Good" = Pareto-dominant](pareto-definition-of-good.md) — at equal success+constraints, fewer steps AND lower avg_time; else say "trade-off"/"non-dominated", never "best"
- [Architecture-matched beat = the strong claim](architecture-matched-beat-is-the-strong-claim.md) — baseline is a UNet; lead with our `unet` row (4.0M), report SiT/DiT wins as confounded secondary; carry backbone+params in every table
- [Benchmark hierarchy: who must beat whom](benchmark-hierarchy-who-beats-whom.md) — diffusion-DPCC is THE baseline; MF/AF must also beat naive FM; HardFlow must beat the DPCC projector via lower proj threshold
- [DA target = best variant of DPCC K20/aw10](da-target-is-best-baseline-variant.md) — paper baseline is pinned to K20+aw10+GaussianDiffusion; pick its best projection variant as Target (other K = additional/conservative check); beating it on ANY axis (S&C held) = win; run in every DA
- [MeanFlow-family upstreams in aux_repo](meanflow-family-upstreams.md) — MeanFlow (PyTorch, only real iMF trainer) vs imeanflow (JAX official, torch branch inference-only) vs alphaflow (bootstrapped target, α-anneal)
- [Visual-transformer refs in aux_repo](visual-transformer-refs-auxrepo.md) — diffusion_policy (= true upstream of D3IL's vision encoder) + act, pulled 2026-08-20 for Gen14 U8
- [UAV budget_ms/33 Hz is not a target](uav-budget-ms-not-a-goal.md) — data-rate artefact + cluster latency; never report as real-time pass/fail
- [HardFlow low-K degeneracy](hardflow-low-K-degeneracy.md) — K1/K2 at A=0.5 run NO HardFlow math; tag rows ✅/❌ and build claims on ✅ only
- [Never write URLs / publish artifacts unless asked](no-unrequested-urls-or-artifacts.md) — deliver repo files + terminal summary; no links by default
