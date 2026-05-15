import sys
import os
import cv2
import numpy as np

# Let's test the env output directly to see what the camera sees
sys.path.insert(0, os.path.abspath('d3il/environments/d3il'))
os.environ['D3IL_DIR'] = os.path.abspath('d3il/environments/d3il')

from d3il.simulation.aligning_sim import Aligning_Sim
from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env

env = Robot_Push_Env(render=False, if_vision=True)
env.start()
obs = env.reset(random=False)

robot_pos, bp_image, inhand_image = obs

# bp_image is returned as BGR [H, W, C] and converted in Aligning_Sim
# wait, get_observation returns BGR image in [H, W, C].
cv2.imwrite('scratch/visual_debug/bp_cam_test.png', bp_image)
cv2.imwrite('scratch/visual_debug/inhand_cam_test.png', inhand_image)
print("Saved raw images to scratch/visual_debug/")
