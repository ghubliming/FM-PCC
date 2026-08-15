"""DA_UAV_v1 — Gen15 UAV Mix-ML evaluation analysis.

Sibling of `DA_VA_v2` (visual aligning) and `DA_Code_v3` (state-only avoiding),
built on the same template and modifying neither. Reads the closed-loop UAV
MuJoCo trees written by `mix_uav_test/eval_mix_uav.py`.

Modules are imported flat (`from config import ...`), so the package directory
must be on PYTHONPATH — the sbatch script does that. `discovery.py` is stdlib
only and runs anywhere; everything else needs numpy/pandas.
"""

__version__ = '1.0.0'
