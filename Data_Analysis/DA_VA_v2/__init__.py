"""
DA_VA_v2 — data analysis for visual-aligning evaluations (Gen7 + Gen14).

Modules are imported flat (`from config import ...`), matching DA_Code_v3, so the
package directory must be on PYTHONPATH — which is what the sbatch wrapper and
the `python Data_Analysis/DA_VA_v2/main_da_batch.py` invocation both arrange.

Entry points:
    main_da_batch.py   multi-candidate batch analysis (primary)
    main_da.py         single model folder
    discovery.py       standalone: print what the discovery layer sees
"""

__version__ = '2.0.0'
