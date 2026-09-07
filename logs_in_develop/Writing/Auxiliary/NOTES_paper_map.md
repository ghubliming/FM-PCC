# Paper map — which reference goes in which chapter

Maps `/workspaces/aux_repo/PAPERS/` onto the skeleton in
`../Working_Space/Bone/thesis_bone.tex`. Not a reading list — a placement plan,
so that no paper gets cited in three chapters for three different reasons.

## Chapter 2 — Background

| Section | Papers |
| :-- | :-- |
| `sec:bg:il` — imitation learning, multi-modality | `auxiliary_papers/D3IL_relevant/D3IL.pdf` (+ `X-IL(D3IL V2)`, `D3IL-ACT`) |
| `sec:bg:diffusion` | `in Proposual/Diffuser_Janner.pdf`, `in Proposual/DPCC.pdf` |
| `sec:bg:fm` | `in Proposual/FM.pdf` (Lipman et al.), `in Proposual/FM_Guide.pdf` |
| `sec:bg:fm:ode` | `auxiliary_papers/ODE/*` — Bespoke Solvers, Bespoke Non-Stationary, GENIE, high-order FM, training-free fast solvers |
| `sec:bg:fewstep:meanflow` | `auxiliary_papers/DGM/MeanFlow.pdf`, `Improved Mean Flows…` |
| `sec:bg:fewstep:alphaflow` | `auxiliary_papers/DGM/AlphaFlow.pdf` |
| `sec:bg:mpc` | `auxiliary_papers/Mujuco/Mujuco_MPC.pdf`, `in Proposual/DPCC.pdf` |
| `sec:bg:mpc:trajopt` | `Recommand_Paper/hardflow.pdf` / `HardFlow_Pub.pdf` |

## Chapter 3 — Related Work

| Section | Papers |
| :-- | :-- |
| `sec:rel:diffusion` | Diffuser, DPCC, `auxiliary_papers/Drone/CGD…`, `Safe Offline RL using…` |
| `sec:rel:flow` | `FlowPolicy_U_e_cn`, `UWA_ManiFlow`, `Streaming Flow Policy_MIT`, `Flow Policy Gradients for Robot Control`, `XFlowMP`, `VFP`, `unicon` |
| `sec:rel:safety` | HardFlow, `SafeFlowMatcher`, `Physics-Constrained Flow Matching Sampling…`, `safe_flow_mpc`, `SAD_flower`, `GraphModel/ProjNet…` |
| `sec:rel:speed` | MeanFlow, Improved MeanFlow, AlphaFlow, `OneStep_Diffusion`, `Generative Modeling via Drifting`, `self_flow`, `Is Noise Conditioning Necessary…`, `Real-Time Execution of Action Chunking Flow Policies` |
| UAV / embodiment | `Drone/UAV-Flow Colosse.pdf`, `Drone/PID_Control_UAV.pdf` |
| Backbones (Ch. 4) | `VLA/DiT.pdf`, `VLA/FiLM.pdf`, `VLA/Pi_0.pdf`, `VITA` |
| Optimal-control framing | `Recommand_Paper/HF_related_OC/` — Reversed FM, Direct, DGM_FM_OC |
| Datasets/benchmarks | `in Proposual/D4RL.pdf` |

## Bibliography bootstrap

`Recommand_Paper/HF/reference.bib` is a curated `.bib` from the HardFlow paper
covering diffusion, flow matching, constrained sampling and robot planning —
i.e. most of Chapters 2 and 3. Start the thesis `bibliography.bib` by copying
the entries actually needed from it, then normalise keys to one scheme
(`author_shorttitle_year` recommended) before the file grows.

## Annotations

Several PDFs have a sibling `.xopp` (Xournal++) file: `DPCC`, `hardflow`,
`HardFlow_Pub`, `D3IL`, `FiLM`, `Improved Mean Flows`, `safe_flow_mpc`,
`XFlowMP`. Those are the user's own highlights — read them before writing the
corresponding section; they show which parts were judged load-bearing.
