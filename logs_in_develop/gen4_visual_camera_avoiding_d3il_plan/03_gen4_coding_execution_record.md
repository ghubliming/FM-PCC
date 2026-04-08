# 03 Gen4 Coding Execution Record

## Scope
This log records concrete implementation steps executed for the approved Gen4 visual-avoiding plan.

## Executed Changes

### 1) Vendor d3il into FM-PCC
- Copied source tree from /workspaces/d3il to /workspaces/FM-PCC/d3il.
- Removed nested git metadata at /workspaces/FM-PCC/d3il/.git.
- Added provenance file: d3il/VENDORED_FROM.md.

### 2) Create isolated Gen4 copy-modify folders
- Copied FM_v3_test -> FM_v3_avoiding_visual_test.
- Copied flow_matcher_v3 -> flow_matcher_v3_avoiding_visual.

### 3) Add two Gen4 config files in existing config folder
- Created config/avoiding-d3il-visual.py from baseline and added:
  - flow_matching_v3_avoiding_visual
  - plan_fm_v3_avoiding_visual
  - Gen4-specific prefixes/loadpaths.
- Created config/projection_eval_visual.yaml and switched to:
  - exps: avoiding-d3il-visual
  - matching constraint keys for avoiding-d3il-visual.

### 4) Rewire Gen4 FM scripts to isolated paths
- Added scripts:
  - FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py
  - FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py
  - FM_v3_avoiding_visual_test/load_results_FM_v3_avoiding_visual.py
- Updated script wiring to use:
  - flow_matcher_v3_avoiding_visual package
  - config/projection_eval_visual.yaml
  - v3 avoiding visual experiment names and dataset id.

### 5) Implement additive d3il visual-avoiding path
- Added dataset class:
  - d3il/environments/dataset/avoiding_dataset.py: Avoiding_Img_Dataset
- Extended simulator for vision mode:
  - d3il/simulation/avoiding_sim.py: added if_vision support.
- Extended avoiding env for vision observations:
  - d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py
  - returns (robot_state, bp_image, inhand_image) when if_vision=True.
- Added new vision config:
  - d3il/configs/avoiding_vision_config.yaml
- Added new launcher script:
  - d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh

### 6) Colab/local dataset protection for vendored d3il data path
- Issue observed:
  - In Colab/local pull-based test environments, large datasets are often pre-downloaded locally.
  - If repository-side placeholders or tracked files appear under d3il dataset data path, sync/pull behavior can overwrite or disturb local GB-scale dataset state.
- Mitigation applied:
  - Added targeted ignore rule in .gitignore:
    - d3il/environments/dataset/data/
  - This keeps d3il code trackable while preventing local dataset artifacts (for example dataset.zip and extracted data) from entering git history.
- Operational note:
  - Download/extract scripts should write dataset artifacts only inside d3il/environments/dataset/data/ so they remain safely local-only.

## Validation
- Static editor diagnostics were checked on all edited files.
- No syntax/lint errors were reported for changed files.

## Runtime Notes
- Runtime training/evaluation commands are user-run required due to environment/runtime dependency.
- No verification-assert code was added; all changes are additive and isolated to Gen4 paths.
- For Colab workflows, ensure data directory creation before download to avoid path errors in fresh mounts.
