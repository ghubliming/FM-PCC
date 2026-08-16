
- DPCC full train, default para
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-01/15_39_45_train_dpcc_job_19784.log

- DPCC full eval, default para
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-04/18_00_14_eval_dpcc_job_19869.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-05/11_09_57_load_results_dpcc_job_19884.log

- DPCC Diffusion = 1
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/23_49_25_dpcc_train_20616.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/23_49_25_dpcc_eval_20617.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/23_49_25_dpcc_load_results_20618.log

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

-> aw10 but K20 full seeds!
02/07
22989 fix log path
22990 debug
22991 fix the mujuco pip bug, cause by mujoco mjx
22992 the numpy also destroed by mujoco mjx. RUN, fixed

- fmv3ode aw1 ode20
eval

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-07/13_11_03_eval_fmv3_ode_job_19965.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-07/20_52_35_load_results_fmv3_job_19981.log

- fmv3ode aw1 ode1 legacy euler (Incredible)

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/21_59_15_eval_fmv3_ode_job_20604.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/23_09_43_load_results_fmv3_job_20606.log

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


FMv3ODE with mpc traj npz
K5
23462 failed aw set into 10 -> 23482 mistake -> 23490

---

##  Drifting
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-12/23_19_20_train_drifting_20135.log

Finished

Eval 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-13/12_39_47_eval_drifting_20150.log

---
incorrect runs bofore! 
- u2
train + eval 21 may

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/15_33_37_train_drifting_20649.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/17_55_00_eval_drifting_20662.log

---




# iMF Gen3v4
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

---
incorrect runs bofore! 
- u2
train + eval 21 may

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/18_13_51_train_imf_20666.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/22_08_00_eval_imf_20667.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/23_45_48_load_results_imf_20672.log

---
imf ode = 1 run 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/00_13_05_eval_imf_20963.log

ode = 2 run

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/11_43_28_eval_imf_20984.log

!!! INCORRECT, STILL
change the folder name
FMPCC/FM-PCC/logs/avoiding-d3il/plans/flow_matching_v3_imeanflow(incorrect)

- FIX1
retrain
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/13_39_41_train_imf_21035.log

+
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/17_02_27_eval_imf_21047.log

- FIX2
 (K2)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/17_02_27_eval_imf_21047.log

(K20)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-01/16_19_15_eval_imf_21088.log
(Still feels TERRIBLE)
FMPCC/FM-PCC/logs/avoiding-d3il/plans/flow_matching_v3_imeanflow/H8_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE_a1.5_b1.0_aw10/H8_K20_Meuler_T0.5_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE/6/results/halfspace_top-left-hard/diffuser.png

- FIX3 
K2
21091

FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-01/17_18_45_eval_imf_21091.log

K10
21125
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-02/11_57_45_eval_imf_21125.log
(Still Exploded)

- U3
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-11/11_53_09_eval_imf_21448.log

U3.2/final

FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-11/13_34_25_train_imf_21451.log
+
21455

- U4
1e4
21547 train
21553 eval

1e5
21572/3 train + eval

21575 try 2 NFE from 10 NFE

21576/7/8 - Uniform Scheduling + turn off aux head

still bad

- U5
21634/5 1e4 trian/eval
21645 -> K10

1e5 train -> K10
21658
21659/21660

- U6 DiT 
21665 (1e4)
21666/67

(1e5)
21673/4 K10
21706 K2

21724 kill cfg in eval reeval K1, chaotic
21726 K20
21727 K40

21736 rk4 K20
21740 rk4 K2 

21744 - TRAIN MAX & other enhanced parameters
21753 eval(submit as dependency)

- U7
22154

- U9
action_w = 1, 1e4 train, dit
23163/4 -> FAILED 
retry 23172

---

Run again 5e4 train (23190)
'learning_rate': 1e-4 (lowered from 5e-4 to prevent early divergence/loss spikes); 'ema_decay': 0.9999 (increased from 0.995 to give you smooth, jitter-free evaluations); 'gradient_accumulate_every': 8 (increased from 2 so your model gets a much cleaner, stable gradient from an effective batch size of 256, without blowing up your VRAM)

-> TOTAL CHAOTIC, EXPLODED

after config_override_pkl and turn off CFG rerun
23364
-> it is no more exploding, but still not smooth, ie beat the old Unet FM or even DPCC

- U10
23391 (prvious run failed due to drain SSD place)
23420 (K2)

-> K10 23455
-> K50 23468
-> K100 23489
->K1 23508

---

from setup 1 to setup 2
see ((Post_U10 Results Analysis default K2 --coding --U10 --(iMF, Gen3v4 --(Gen3 --Works - Develop Iterations--Plan & Works (Replace, Update to FM--DPCC Code & Replace Code Works)))) (SS26-Thesis-Flow_matching))
23551(K2)

23572 K1
23574 K10
23581 K50

---

setup3
23650 (5e4 train)

-> 2e5 train
23680

---
# Visual 
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
FMPCC/FM-PCC/Slurm_Codes/logs/202FI6-05-15/15_56_15_eval_visual_aligning_20324.log

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

## one shot run
(non visual) BUG
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/00_15_06_eval_fm_visual_aligning_20967.log
rerun after fix 18
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/22_13_23_train_visual_aligning_dpcc_21007.log

FIX18.2

(visual)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/12_24_54_visual_aligning_pipeline_dpcc_20986.log
(Finished)

FIX18.4
(non visual success)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/18_03_20_eval_visual_aligning_dpcc_21050.log
...
misc fix, run 21080
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-01/16_58_59_eval_visual_aligning_dpcc_21089.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-02/11_57_45_eval_imf_21125.log
NOT good, closure task

- Gen7(Legacy, new is later) FM Visual Aligning 
(10k train)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-18/12_02_46_train_visual_aligning_fm_20473.log

---

... ALL Failed Gen6/7v1

- try last time with max_len_data=256     

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-19/14_48_06_train_visual_aligning_20523.log

- to Gen6V4 Rebuild to visual algining dpcc

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-19/10_49_21_train_visual_aligning_dpcc_20508.log

Eval Terrible

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-19/11_50_36_eval_visual_aligning_dpcc_20514.log

-> Move to Archive as Name Initail Run

KEY FIX7 Revert some D3IL changes 
Train
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-19/19_01_04_train_visual_aligning_dpcc_20543.log

Eval
Part succcess ! 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-19/22_41_46_eval_visual_aligning_dpcc_20551.log

- FIX 8 + 9
...
- FIX11
1e4 train + multi evals 
20 May Afernoon

-> try the post-processing etc. 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/20_27_15_eval_visual_aligning_dpcc_20599.log

---
*offtopic* tests to check d3il integrity
ODE=1 + RK4
(d3il looks fine)

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/21_59_15_eval_fmv3_ode_job_20604.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/23_09_43_load_results_fmv3_job_20606.log


---

-> Fix7.2 **CORRECTED FIXED** it is the expert video gen destroy the mojoco!

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/15_49_29_eval_visual_aligning_dpcc_20651.log

note: 100K 1e4 train. looks bad.

---

1e5 train & Eval Diffusion 20
(Mark as 900 STEPS)

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/00_00_51_visual_aligning_pipeline_dpcc_20675.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/00_00_51_train_visual_aligning_dpcc_20676.log

- Massive Reeval on it (killed by SLURM time limit) u9
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/00_10_16_eval_visual_aligning_dpcc_20681.log


## Turn off visual
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/21_36_27_visual_aligning_pipeline_dpcc_20697.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/21_36_27_train_visual_aligning_dpcc_20698.log


# FM Gen7 New fm_visual_alinging

First FM Gen7 New fm_visual_alinging run
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/17_41_20_train_fm_visual_aligning_20585.log

eval
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/17_41_20_eval_fm_visual_aligning_20586.log 

not on train eval
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/18_39_01_eval_fm_visual_aligning_20592.log

BOTH wrong crushed eval for `diffuser` 
BUT some success on `post processing`!!! others fail
(direct look: FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_a1.5_b1.0_aw1_VTrue_steps1000/H8_K20_Mrk4_T0.1_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_VTrue/6/results/diagnostics/post_processing)

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/19_41_48_eval_fm_visual_aligning_20595.log

-> Fix Gen7F4 done

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-20/23_51_44_eval_fm_visual_aligning_20619.log

& 

(FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/00_14_17_eval_fm_visual_aligning_20620.log) mistake.

(FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/10_13_04_eval_fm_visual_aligning_20623.log) All TASH results, maybe eval on train flag. remove, reeval


---

THIS REVEAL PROBLEM "FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_b1_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_a1.5_b1.0_aw1_VTrue_steps900"
(FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/11_54_36_eval_fm_visual_aligning_20632.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/11_54_36_eval_fm_visual_aligning_20632.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/11_35_56_eval_fm_visual_aligning_20628.log
)

-> *KEY FIX 6(Gen7)*
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/12_43_00_eval_fm_visual_aligning_20634.log (FAILED AGAIN)

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/12_58_29_eval_fm_visual_aligning_20637.log (SUCCESS)

still wrong
-> Fix7.2 **CORRECTED FIXED** it is the expert video gen destroy the mojoco!

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/15_37_02_eval_fm_visual_aligning_20650.log

note: Clamp 0.01 

turn off
->FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/16_27_58_eval_fm_visual_aligning_20655.log
&
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/17_21_49_eval_fm_visual_aligning_20660.log

u8
MPC4 batch. slightly path name wrong, will fix.
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/18_02_28_eval_fm_visual_aligning_20665.log

check if less training step is enough. (1e4 train, use trick of 1000 steps to distinguish last 900 step(1e5 train))
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-21/22_37_14_eval_fm_visual_aligning_20669.log

---
1000(1e4 train) ODE100
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/12_49_23_eval_fm_visual_aligning_20688.log

900(1e5 train) ODE20
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/17_08_32_eval_fm_visual_aligning_20696.log
(latest diag 22 May, fix10.2(after 12))


- WORK!
FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/(PCC_T1)H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64

-> SEE THE Plot "FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/(PCC_T1)H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/combined_4/dpcc-t/diagnostics/rollout_0_mpc_foresight.svg"

UF15.3 Update the Plots
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-27/11_12_49_eval_fm_visual_aligning_20837.log

---

one shot run
(non visual)BUG
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/00_15_06_eval_fm_visual_aligning_20967.log (Failed with logs)

rerun
21004
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/18_29_07_eval_fm_visual_aligning_21004.log

(Visual)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-30/12_16_37_eval_fm_visual_aligning_20985.log


(Good Compare)
diffuser
FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/combined_5/diffuser/diagnostics/rollout_1_mpc_foresight.svg

vs 
dpcc-c
FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/combined_5/dpcc-c/diagnostics/rollout_1_mpc_foresight.svg


## Turn off visual
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/21_36_49_fm_visual_aligning_pipeline_20700.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-22/21_36_49_train_fm_visual_aligning_20701.log

- [X] NOT UNDERSTAND THE CODE RESULTS, MAYBE WRONG

in FM (FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VFalse_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VFalse_mpc4/6/results/diffuser/diagnostics)
the roll out 0,1 cannot go inside the box, it is the evidence that worse than viusal?

---


**Bring Back Constraints/PCC**
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-27/11_19_40_eval_visual_aligning_dpcc_20843.log

inital run good no bugs, but looks weird results. maybe PCC 0.5 threshold wrong, maybe the Dynamic Projection wrong, need check.

(FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/visual_aligning_dpcc/H8_K20_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VTrue_steps900_bs64/H8_K20_T0.5_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_VTrue_steps400_mpc4/6/results/combined_4_uf15)

UF16.1
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-27/13_25_38_eval_visual_aligning_dpcc_20846.log
(Good enough, like the roolout 1 FM is way better than Diffusion. also the PCC obstcle can observed some hint.)
->
FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/(Good Run_27_05)combined_4

---
logging update to with metric of constratints violation
+ 
Combined_5 Yaml

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-27/16_44_22_eval_visual_aligning_dpcc_20849.log



UF17 fix
rerun, mark old as b_uf17, before uf17 update

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-29/12_12_40_fm_visual_aligning_pipeline_20927.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-29/12_12_40_eval_fm_visual_aligning_20929.log

---

**Bring Back Constraints/PCC**
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-26/16_21_22_eval_fm_visual_aligning_20817.log(old run, legacy)

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-26/17_24_08_eval_fm_visual_aligning_20824.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-27/11_12_49_eval_fm_visual_aligning_20837.log (LOGGING STOPED WHY?)
- same weird results like dpcc VA
(FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/combined_4_uf15)

UF16.1
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-27/13_02_56_eval_fm_visual_aligning_20845.log
GOOD RUN CAN VS DIFFUSION AND PROJECTIONs
->FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/(Good Run_27_05)combined_4
!THIS IS very good -> "FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/(Good Run_27_05)combined_4/dpcc-r/diagnostics/rollout_1_mpc_foresight.svg"

---
logging update to with metric of constratints violation
+ 
Combined_5 Yaml

-> FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-27/16_44_27_eval_fm_visual_aligning_20850.log

NOTE: ALSO COULD COMPARE THE TIMES OF CONSTRAINTS Violated
see model_free (FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/combined_5-tightened/model_free/constraint_metrics.json) vs the dpcc-r (FMPCC/FM-PCC/logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VTrue_steps900_bs64/H8_K20_Meuler_T0.5_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/combined_5-tightened/dpcc-t/diagnostics/rollout_1_stats.json)

- Minor Update the Plot
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/12_21_25_eval_fm_visual_aligning_20874.log

---

C4 -> test on Gen6V4(Gen7 same)
23134 visual_aligning_dpcc film v2 train + Eval (also test the C4)


# API Patching. 26. May
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-26/12_54_49_run_patch_legacy_checkpoints_20810.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-26/14_19_27_run_patch_legacy_checkpoints_20814.log
... untracked lots

---

# 28 May D3IL V_A Baseline
- ddpm vision smoking run
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/13_20_44_train_d3il_baseline_20881.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/14_21_55_eval_d3il_baseline_20885.log

---

- 200 epoch, as paper said 
(transformer)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/15_09_24_train_d3il_baseline_20888.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/15_17_17_eval_d3il_baseline_20889.log

(ddpm vision)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/15_22_59_train_d3il_baseline_20890.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/15_30_03_eval_d3il_baseline_20891.log

- increase to 1k 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/15_39_16_train_d3il_baseline_20892.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-28/16_32_56_eval_d3il_baseline_20894.log

---

U2 20/06/2026
21760-21771

---

FiLM v2 real FiLM 28/06
22118 (only model free and post proccessing)
+
22126

---
- U3 baseline train update

22210 piepline - train work
22212
debug
->
22231
32 train stil bug

-> 22248 try


---

**DC-FIX**
22396 (film v2!) TIME LIMIT KILLED

22485 (film v1)

---

**boudns fix**
23094
fail, reanchor 

23097 killed - too many redundant tests -> change into just the a new projection `bounds`
rerun in 23109

---

FM V_A Gen7 eavl not on train_set, (random init position +  the C5 Updates/Gen11 Fix14 state, 11/07)
23293 (SLSQP solver guad too strict only diffuser valid, projection all dead) -> KILLED

-> Try FilM_v1 
23314 -> Killed 
try new Fix15.2/C6 sync for no `diffuser` projections
23317

---

dpcc V_A run on random init(observe if init overlap the obstacle) 
23514

# Gen8 iMF Visual Aligning

Fix1.2
Train
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-03/17_48_36_train_imf_visual_aligning_21162.log
(Work, Killed, Run later)
21166
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-03/18_00_01_train_imf_visual_aligning_21166.log

fix2.5 eval run
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-04/12_50_31_eval_imf_visual_aligning_21196.log

- U2
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-11/12_02_36_eval_imf_visual_aligning_21450.log

U2.2/final
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-11/13_38_13_train_imf_visual_aligning_21452.log

# Gen 9 Visual Avoiding (Camera Data Collection)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-29/16_14_33_collect_visual_avoiding_20940.log

---

E2 Train 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-03/12_43_49_train_fm_visual_avoiding_21145.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-03/12_52_35_train_visual_avoiding_dpcc_21146.log

E2 eval
after Fix_6.3
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-03/17_58_38_eval_fm_visual_avoiding_21165.log
(plot bug) -> fix7

21167
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-03/18_09_54_eval_fm_visual_avoiding_21167.log
(feels ok)

- Diffusion
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-03/12_52_35_train_visual_avoiding_dpcc_21146.log
(K100 , 56789 trained not eval)

eval 6
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-04/12_16_52_eval_visual_avoiding_dpcc_21188.log

(K20)
21234

- E2U2
21277 fm (killied at end "--- avoiding-d3il both-hard dpcc-c-tightened-dt2p0 seed=6 ---")
21279 diffu.
(seems results correct, but the plot are wrong, mpc plots:
FMPCC/FM-PCC/logs/avoiding-d3il-visual/plans/fm_visual_avoiding/H8_K100_Meuler_T0.5_Dfm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching/H8_K100_Meuler_T0.5_Dfm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching_VTrue_mpc4/6/results/halfspace_both-hard/diffuser.png)

Fix 2 & FM K20
21290

Fix 3
21311 (*Been Canceled Time limit* but the run results seems better)
(seems fine, run the diffu. pipeline)
21317,18 
- [ ] Pending to read the reuslts

fix again Fix3(.2)
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-09/18_02_22_eval_visual_avoiding_dpcc_21376.log
+
21457 Eval
+
21462 Eval
+
21472

---

## U3
FM
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-12/12_40_24_train_fm_visual_avoiding_21480.log
+
21481

Diffu.
21492

from 1e4 to 1e5 train
21514/5
+
diffu.
21517/8

fix1 test revert if affect?
21530/531 same setup as 21517

---

20/06
Try redo full FM eval
21782

& Diffu Piepline
21783

bug is .sh time 4h limit... reeval FM and Diffu
FM 21822/3

Diffu. 21824/5

---

seed 78910
21969 FM
21970 Diffu

---
- 30/06 U5 film v2
->
22234/6 run eval cancel rerun 22238

---

Recollect the mpj traj data, rerun eval film v1 visual avoiding 23450

# Gen11
# E2 
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/11_57_52_run_naive_21022.log

FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/11_59_58_run_naive_21023.log

# E3
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/12_57_00_run_env_21029.log
FMPCC/FM-PCC/Slurm_Codes/logs/2026-05-31/12_57_57_run_env_21031.log

# E4
smoke
FMPCC/FM-PCC/Slurm_Codes/logs/2026-06-04/15_42_03_collect_21205.log

full run
21206 to 21209

fix1
21212 to 21215

fix2 Curve + Pillar
21220, 21221

fix3 Curve
21222

fix4 curve
21226

fix5 Curve
21227

---

all runs
21231 -> 21223

## U2
21319-21322
Fix1 

21324 -> 21327

## U3
21339 -> 21342

## U3F1 & GPU Leak Test
21356

F2
21368-21371

## U4
21398 - 21401

## U5
21408 - 21411

## U6
21415 - 21418

## U7
21419 - 21422

## U8,9
21474 - 21477

fix1
21482

-> Recollect Gif!  21484 + 21485

## U10 stress Test
21494 + GIF 21495 + 21496

# E5
21289

21291(Fail)
Fix1
21294 - Good

## U2
21329 (too long!)
21330 (./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" "" "" 3)

## U3
21358
werid behaviors
FMPCC/FM-PCC/logs/uav_expert_data/gifs_physics/pillars/L_L_L
21367

Fix 2 (on Data E4 U7)
21436
21437

## U5 2D plot
21497

# misc
touched files during rebase(already fixed)
~~Slurm_Codes/sbatch/Drifting/train_drifting.sh~~

Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh

Slurm_Codes/sbatch/diffuser_visual_aligning/train_visual_aligning_dpcc.sh

Slurm_Codes/sbatch/diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh

Slurm_Codes/sbatch/fm_visual_aligning/eval_fm_visual_aligning.sh

Slurm_Codes/sbatch/iMF/train_imf.sh


# E6
check prepare UAV FM data 21873
fm gate 21874/ fix -> 21875 -> hotfix again 21876
21877 keep testing
21878 still bug, but go full tests

21879 - TRAIN 
21880 to 21904 all our train & eval
serveal seveve bugs

Debug
21925-6,7,8

U3
21952/3

other sence tests
21988/89

21993,4,5,6
test with Gif the scurve 22016 (killed, resutls to bad)

# E7
22019 first run test on empty
22033 test the corridor behaviour
22036 rerun, fix bug
22037 pillar run with MPC. -> exploede MPC lines
- U2
pillar
22039

pillar
22041 U2.2



- U3 metrics like model_free bring back
& the real time eval logging
22093/4

---

- U4
Fix3
Test 22128
Still buggy math -> 

Fix5
22131
22137 scurve

22139 empty
22150 pillars

## U8
pillars
22176 + 7 (9D train)
7 - eval Failed!

22188 (success pid stop and go, good results!)
-> genenrate some gif 22195


before is ODE 100

---
22207 FM Scurve train + eval (ODE20)
eval fail/debug fixed
22247 scurves tracking error acculation! try the anchorP!
anchorP -> 22287

**DC-FIX**
22295 / Disk full rerun
22380 sucess, with dynamic correction is good. scurve

try the pillars (the empty and the corridor need retrain 9D)
22245 (Good)

trian + eval on corridor 22296

- try the pid_const_v
22980 debug -> 984

- train the 9D 22983
22983

empty trained, -> eval
pid_stopgo
23031

---
## MJPC
try the mjpc
22194 - need debug 
22266 / 7/  8 / 9 / 22270(with debug)
SIG error from nowher, fail


---
- build the cpp bin for mjpc solver
22209
fail, debug
22235
-> debug 
22237
->
22239
->
22240
22241
22242
22243
22244
22245
PASS
lib fix
22246
+
fix .so
22265

---
add smoke test
22271 smoke 2 fail
22272

---

OK, run eval mjpc in scurve 22257
lack pkg 
install pip install grpcio grpcio-tools
22261 + 2 + 3 

---

U6 JAX mjpc Solver
22958 pillars
22962
22963
fail, drifting badly
22964 killed, same as 63, and maybe cheating
22966 (set up mjpc faithfully)

******
---

## E9 (Gen11)
23087 (init, no `bounds`)
23088 (with bounds pre fix)
Killed

-> fix_4 
23091 pillrs 

23093 scurves with the single dynamics + boudns tests
(yaml fail)
23096 killed - too many redundant tests -> change into just the a new projection `bounds`

23110 rerun after U8 new design of prjection variants (s_curve)
23126 - pillars
failed-debug
23131

- fix12
23189 - corridor

- U13 
23233 incl. Geo free - Bounds free

- Fix14
23265 - pillars -- KILLED due to timelimit

# REAL TIME EVAL test
tried on the Gen11 E7U3
then implent to all codes!

# DA run 
22208 -> all avoiding incl visual
fail, debug

23032, 03/07 DA combined
debug -> 23033

# Hardflow Replicaion
init 15/07 
"METHODS="original" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_hardflow.sh"
23486
fix1 -> 23491 succes test 
run pipeline -> 23559 
fail, fix 2
23565 (METHODS="original" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_hardflow.sh)
success
RUN constraint 23566


# Gen13
23575 test
23578
23586/7 (K45 follow the guide)
U5 23602
Smooth test (fix7) 23608 / fix.2 23610
U8 fig11 but imf -> 23609 / fix.2 23611
23612 same NFE test

U9 Incrase train step to 300k for imf and wandb logging 
23613
EVAL on it -> 23624

23634 - check smooth of new 300k trian of imf
23636 U9.2 tune the learning rate, since loss curve so bad
23668 inspect the smooth
23669 + 23672 u9.2 follow up next step run, arm A&B
(IV)

23683 80k train
(Stop as 55k)
Eval -> 23733/4 
(
    SKIP_TRAIN=1 IMF_EXP_NAME=H16_imf_lrfix_800k IMF_CP=22 \
    > IMF_KS="1 2" RANDOM_REPEAT=200 \
    >   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
)

- U10
23832

- U11
mf 23966
+ 
af 23978 

- Fix/U12 Orgize and better naming

(disk full) 23991 to finish it (ML_EXP_NAME=H16_ml_af_100k ML_CP=4 ML_METHODS="hfproj" ML_KS="2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_ml_hardflow.sh)

---

# Gen3v6 mean flow
23734 FAIL 
23744 trian seed 6
EVAL in 23777

try unet 23812

- U2 
23925

- U3
23981

24021 K1 
24022 K5
24023 K20

- Fix4 
24034-8
K1,2,5,10,20

- Fix5 
24074 - 24078


- Fix6
resume train, 24100

---
Mean flow train the seed 78910 -> 24069
Eval 24121 BUG -> Fix7
24126

- Fix6
24316 + Eval 24334

seed 78910 for UNET 24396 train -> 24415 seed 10
24416 eval
& rerun seed 6 24470

- Sweep of HF thres  (for K in 5 10; do
  for A in 0.0 0.1 0.25; do
    HFFM_ACT_THRESHOLD=$A HFFM_FLOW_STEPS=$K \
      ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
  done
done)

24507 - 24511

---

23561 rerun the mf_dit

---
20 trails
24559 - 24560

# Gen3v7 alpha flow 
1st run 23758 seed 6 done, seed 7 kill at 80!!!!!(could resume later)
EVAL 23786

try unet
23809

- U2
23928

- U3 
24044 - 24048

- U4 (similar v6 U5 fix)
24101

---
alpha flow train the seed 78910 -> 24070

eval 24104 s78910+ 25399 s6

---

unet validation 24348 KILLED try full seeds train
-> 24385

---
24515 rerun ensure safety of all data

24594/5 ntrail 20 run

# Gen12 
init pipeline run 23767 -> All killed, not run
fix1(.3) -> debug_chain_23782 
FAIL Gate fix2
23795

fix3
sweep run 23815

U4
23826 to smoke test
23827/8 : thres 0(baseline) and 0.5 test FAIL (quick patch 23830/1 0+0.3 retest)
23829 mpc batch=4 test

- U5
23890

- Fix7 23903

- Test NFE 
HF 24179 0.1thres
FMv3ODE equiv 24180 DEAD -> Fix in 24196, write wrong aw (K10 run, K20 run is 24210)

24187 0.05thres
vs 24198 0.05thres (K10, K20 run -> 24207)

Ablation DPCC baseline 0.1thres 24215 / 0.05thres 24226
- Claude Claim the diffusion_timestep_threshold: 1 for DPCC not working, lets see if ture, set it to 1, 24254, feels run it beofre but we rerun it again. --> Gen0F2

# Gen14 
Test -> 24082
faile fix2 24907

mf run 24110 DEAD disk full -> 24122

af ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 6 (24154)

- U6 K2 run 10rollout
af/mf 24224/5 

- U7 
24255

---

24281/82 rerun massive run 3->30

- test diffu+FM vs the old run
24638 diffu
24343 fm
(K100! too big!)

20 run 24405 Diffu
24409 FM

- FiLM v2 
24417 mf
14418 af

# Gen0 
- Fix2
thres test 24279 T


# Gen15
test 24578
mf_unet corridor 24579, 24583( mf eval at K=10 (chained to the train))

scurve 24588 + 89

24612 mf pillars
