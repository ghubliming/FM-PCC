
- DPCC full train, default para
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-01/15_39_45_train_dpcc_job_19784.log

- DPCC full eval, default para
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-04/18_00_14_eval_dpcc_job_19869.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-05/11_09_57_load_results_dpcc_job_19884.log

- FMv3ODE full train, default para
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-02/18_09_44_fmv3_train_19819.log

- FMv3ODE full eval, default para
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-03/20_49_27_eval_fmv3_ode_job_19840.log


FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-04/14_45_27_load_results_fmv3_job_19859.log

- FMv3ODE full eval, midpoint ODE5
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-04/15_36_20_eval_fmv3_ode_job_19862.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-05/11_10_47_load_results_fmv3_job_19885.log

- dpcc 10 steps
run as pipeline
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-05/14_18_01_dpcc_train_19888.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-05/14_18_01_dpcc_eval_19889.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-05/14_18_01_dpcc_load_results_19890.log

- fmv3_ode aw 10
run as pipeline, with new plan sub folder struct

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-06/14_47_57_fmv3_train_19921.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-06/14_47_57_fmv3_eval_19922.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-06/14_47_57_fmv3_load_results_19923.log

- fmv3ode aw1 ode20
eval

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-07/13_11_03_eval_fmv3_ode_job_19965.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-07/20_52_35_load_results_fmv3_job_19981.log

- fmv3ode full proj
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-08/11_35_01_eval_fmv3_ode_job_20010.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-08/21_44_30_load_results_fmv3_job_20031.log

- dpcc full proj 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-09/11_25_06_eval_dpcc_job_20038.log

(modify name into aw10 "FMPCC/FM-PCC/logs/avoiding-d3il/plans/diffusion/H8_K20_T1_Dmodels.GaussianDiffusion")

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-09/17_32_35_load_results_dpcc_job_20048.log

- dpcc aw1 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-09/17_36_21_dpcc_train_20050.log
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-09/17_36_21_dpcc_eval_20051.log
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-09/17_36_21_dpcc_load_results_20052.log

- Drifting
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-12/23_19_20_train_drifting_20135.log

Finished

Eval 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-13/12_39_47_eval_drifting_20150.log


- Visual 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-12/23_19_40_train_visual_aligning_20136.log

Interrupt, the loss curve looks wrong in WandB

Eval seed 6
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-13/12_49_01_eval_visual_aligning_20153.log

Stop, stucked, no error warning

---

Again
Visual Seed 6 train
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-14/13_39_19_train_visual_aligning_20242.log

4 epoch seems loss curve good enough for eval test (remember to change the setting when train rest seeds)

Seed 6 Eval 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-14/17_08_06_eval_visual_aligning_20279.log (BAD results)
(Archived)

Fix 7
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-14/22_08_29_train_visual_aligning_20291.log

and rest seeds 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/10_58_33_train_visual_aligning_20308.log

+ Diagnositic Eval
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/10_10_04_eval_visual_aligning_20304.log

+ Video 
(Fix 9)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/15_56_15_eval_visual_aligning_20324.log

eval(6,7,8(half))
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/15_56_15_eval_visual_aligning_20324.log

- reudce H to 2
(and change trainign steps to 1k)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/16_58_00_train_visual_aligning_20333.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/17_18_45_eval_visual_aligning_20336.log

(to 10k, overwrite 1k)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/21_32_38_train_visual_aligning_20346.log

total failure eval

- H10 (use the ddpm act styple setup)

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-15/22_49_20_train_visual_aligning_20352.log

idinetify the physical interation error (fix11)

-> fix12 fix the physical and robot. Add the max episode length to 1e5
... misc fix, rebuild

*train + eval*
    FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-16/22_45_18_train_visual_aligning_20397.log

    FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-16/23_26_39_eval_visual_aligning_20403.log
    （**worked!** 3k train）(FMPCC/FM-PCC/logs/archive/aligning-d3il-visual_256_length/plans/ddpm_encdec_vision_3k_train/H10/6/results/diagnostics/rollout_0.gif)

    parameters "FMPCC/FM-PCC/logs/archive/aligning-d3il-visual_256_length/plans/ddpm_encdec_vision_3k_train/H10/6/config_snapshot_aligning-d3il-visual/aligning-d3il-visual.py"

rerun the 100 diffusion steps trian
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-17/23_41_46_train_visual_aligning_20455.log

...

- Gen7 FM Visual Aligning
(10k train)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-18/12_02_46_train_visual_aligning_fm_20473.log

---

... ALL Failed Gen6/7v1

- to Gen6V4 Rebuild to visual algining dpcc

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-19/10_49_21_train_visual_aligning_dpcc_20508.log

---

- iMF
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-13/22_24_54_train_imf_20216.log

finished, by loss curve is bad
abandoned

update, re train
seed 6
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-14/11_42_47_train_imf_20229.log

+ seed 789 10
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-14/13_45_58_train_imf_20245.log

killed at Epoch 38, seed 8

eval
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-14/15_55_29_eval_imf_20263.log

look the `diffusor` metric, it is very bad, very bad

reset parameter correctly in d3il.py

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-14/23_26_00_eval_imf_20298.log