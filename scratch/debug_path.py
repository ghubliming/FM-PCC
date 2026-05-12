import os
import sys
# Add d3il to path
sys.path.append(os.path.abspath('d3il'))
from agents.utils.sim_path import sim_framework_path

path = 'environments/dataset/data/aligning/train_files.pkl'
abs_path = sim_framework_path(path)
print(f"Rel path: {path}")
print(f"Abs path: {abs_path}")
print(f"Exists: {os.path.exists(abs_path)}")
