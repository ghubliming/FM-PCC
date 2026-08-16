"""Gen13 U11 — Mix-ML config: one training config, three separated MLbone blocks.

`MlTrainingConfig` SUBCLASSES `ImfTrainingConfig`, so every iMF default/knob is
inherited byte-identically (this is what makes `ml_type="imf"` reproduce the frozen
Gen13 iMF path — gate G0). It then ADDS:
  * `ml_type` — the top-level MLbone selector (imf | mf | af)
  * `mf_*`    — MeanFlow (Gen3v6) knobs, adjustable independently
  * `af_*`    — α-Flow  (Gen3v7) knobs, adjustable independently

Each family's knobs live in their own namespace, so tuning one never touches another
("first pick the family, then tune that family's block" — PLAN §6).

EVAL is objective-agnostic (it loads the shared TemporalImfUnet + ImfFlowPolicy from
any checkpoint), so there is NO separate ML eval config: MF/AF checkpoints are
evaluated with the existing `ImfEvaluationConfig` / `run/eval_imf.py`, pointing
`flow_exp_name` at the MF/AF run. See PLAN §5 / README_PROVENANCE.md.
"""

from dataclasses import dataclass

from ..imf.imf_config import ImfTrainingConfig


@dataclass
class MlTrainingConfig(ImfTrainingConfig):
    # ── top-level MLbone selector ──────────────────────────────────────────────
    ml_type: str = "imf"          # imf | mf | af

    # Default identity carries an `ml_` prefix so U11 runs can NEVER collide with the
    # frozen Gen13 iMF checkpoints (H16_imf_100k / _300k / _lrfix). train_ml.sh derives
    # the real per-run name as H16_ml_<ml_type>_<steps>k.
    exp_name: str = "H16_ml_imf_100k"

    # ── MF (Gen3v6 faithful MeanFlow) — analytic-v JVP tangent ─────────────────
    mf_p_mean: float = -0.4
    mf_p_std: float = 1.4
    mf_data_proportion: float = 0.25   # Gen3v6 first-class ablation axis (`dp`)
    mf_adp_p: float = 1.0
    mf_adp_eps: float = 0.01

    # ── AF (Gen3v7 α-Flow) — bootstrapped, α: 1 → 0 anneal ─────────────────────
    af_p_mean: float = -0.4
    af_p_std: float = 1.4
    af_ratio_fm: float = 0.5           # α-Flow's FM-anchor fraction
    af_adp_eps: float = 1e-3
    af_clamp_utgt: float = 4.0
    af_alpha_scheduler: str = "sigmoid"   # sigmoid|linear|exponential|log|constant|step
    af_alpha_init: float = 1.0            # α at step 0  (1.0 ⇒ start as pure FM)
    af_alpha_end: float = 0.0             # α at the end (0.0 ⇒ end as MeanFlow)
    af_alpha_init_step: int = 0
    af_alpha_end_step: int = 100000       # 🔴 MUST equal n_train_steps (train_ml.sh sets it)
    af_alpha_gamma: float = 25.0          # sigmoid sharpness
    af_alpha_clamp: float = 0.005         # snap α to exactly 0/1 near the ends

    # U11 W&B lineage (override per run)
    wandb_project: str = "FMPCC-HF-Mix-ML"
