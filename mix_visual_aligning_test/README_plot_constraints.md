# YAML Constraint Visualization Script

The script `plot_yaml_constraints.py` is a standalone utility designed to parse and visualize all geometric constraint configurations defined in `../config/visual_aligning_eval.yaml`. 

It imports and utilizes the `plot_geo_constraints` function from `eval_mix_visual_aligning.py` to generate a 3-panel constraint overview figure (3D wireframe, XY top-down, and XZ side) for each configuration without needing to run the full D3IL evaluation pipeline.

### Features
- Parses all `geo_constraint_variants` in your `visual_aligning_eval.yaml`.
- Iterates over each geometry entry (like `combined_4`, `combined_5`, `obstacle_only_1`, etc.).
- Invokes `plot_geo_constraints` to draw the corresponding bounds, halfspaces, and obstacles.
- Automatically generates the `-tightened` versions of the plots if `enlarge_constraints` is defined in the global configuration and applicable constraints are active.

### How to Use
Navigate to the `mix_visual_aligning_test` directory and run the script:

```bash
cd /workspaces/FM-PCC/mix_visual_aligning_test
python plot_yaml_constraints.py
```

Outputs will be saved in a newly created directory `constraint_plots/`, neatly organized into subdirectories based on the constraint name (e.g., `constraint_plots/combined_5/constraint_overview.png`).
