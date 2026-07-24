<h1 align="center">HardFlow</h1>

<p align="center">
  <b><a href="https://arxiv.org/abs/2511.08425">Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization</a></b>
  <br><br>
  Zeyang Li &nbsp;·&nbsp; Kaveh Alim &nbsp;·&nbsp; Navid Azizan
  <br>
  <i>Massachusetts Institute of Technology</i>
  <br><br>
  <b>IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026</b>
</p>

---

## Robotic Manipulation (Section VII.A)

This branch (`d3il`) reproduces the robotic manipulation experiments from
the paper. The task is goal-reaching with obstacle avoidance on the
"avoiding" benchmark. A flow-matching model is trained on expert
demonstrations to generate (state, action) trajectories. At inference,
HardFlow steers the sampler so the planned trajectory avoids all obstacles
and heads to the target.

## Branches in this repo

The four experiments from the paper each live on a separate branch. Switch
with `git checkout <branch>` and follow that branch's README.

| Branch    | Section in paper | Task                                |
|-----------|------------------|-------------------------------------|
| `d3il`    | VII.A            | Robotic Manipulation (this branch)  |
| `maze2d`  | VII.B            | Maze Navigation                     |
| `burgers` | VII.C            | PDE Control                         |
| `image`   | VII.D            | Text-Guided Image Editing           |

## Algorithms (`--guidance_method`)

| Flag                 | Name in the paper            |
|----------------------|------------------------------|
| `original`           | Original                     |
| `gradient_guidance`  | Gradient Guidance            |
| `oc_flow`            | OC-Flow                      |
| `projection`         | Projection-All/Late*         |
| `projection_relaxed` | Projection-Relaxed           |
| `hardflow`           | **HardFlow**                 |
| `hardflow_new`       | **HardFlow (l4casadi-free)** |

\*Use `--projection_option {all,late}`. For
"+ Gradient Guidance" combinations, also set `--projection_gradient_steps 5`.

`hardflow` and `hardflow_new` solve the same surrogate problem and produce
matching numerical results; the only difference is how the reference flow is
evaluated between IPOPT solves. The
[`l4casadi`](https://github.com/Tim-Salzmann/l4casadi) bridge was initially
introduced to solve the full-horizon optimal control problem with neural
dynamics. For HardFlow, however, this dependency is unnecessary. Instead,
`hardflow_new` evaluates the flow directly in PyTorch, removing the
`l4casadi` dependency and avoiding the associated overhead.

## Setup

```bash
conda env create -f environment.yml
conda activate hardflow
pip install -r requirements.txt
pip install -e .
# l4casadi (CUDA) is required for hardflow and the projection baselines;
# the hardflow_new version does not need it. Follow
# https://github.com/Tim-Salzmann/l4casadi for the installation.
```

The simulator and dataset are included in the `d3il/` subfolder. The
pretrained flow-matching checkpoint is expected at
`logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth`. An example model is available
[here](https://drive.google.com/file/d/1BAQvUTB7muOZJ7Cy7Bf1mtFjKnv3ExUh/view?usp=sharing).
To train it from scratch, run:

```bash
bash run_scripts/train.sh
```

A linear dynamics model fitted from the training data is used as a
physical-fidelity constraint at inference time. To fit it, run:

```bash
python run/fit_dynamics.py
```

## Run

```bash
bash run_scripts/eval_original.sh           # Original
bash run_scripts/eval_gradient_guidance.sh  # Gradient Guidance
bash run_scripts/eval_oc_flow.sh            # OC-Flow
bash run_scripts/eval_projection.sh         # Projection-All / Projection-Late
bash run_scripts/eval_projection_relaxed.sh # Projection-Relaxed
bash run_scripts/eval_hardflow.sh           # HardFlow
bash run_scripts/eval_hardflow_new.sh       # HardFlow (l4casadi-free)
```

Each script writes `trajectories.csv` under `logs/avoiding-v0/eval/<exp_name>/`.
Use `notebooks/collect_results.ipynb` to aggregate results across runs.

## Acknowledgement

The "avoiding" benchmark setup on this branch is adapted from
[dpcc](https://github.com/ralfroemer99/dpcc).
Structure is also inspired by
[flow_guidance](https://github.com/AI4Science-WestlakeU/flow_guidance).
Some PyTorch–CasADi bridges are provided by
[l4casadi](https://github.com/Tim-Salzmann/l4casadi).

## Citation

```bibtex
@article{li2025hardflow,
  title={HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization},
  author={Li, Zeyang and Alim, Kaveh and Azizan, Navid},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```
