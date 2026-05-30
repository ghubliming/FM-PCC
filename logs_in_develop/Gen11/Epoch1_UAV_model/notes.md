I'll help you find the UAV/Drone model files in this MuJoCo MPC repository and provide guidance on how to use them. Let me search for relevant model files.Let me look for model files in the mjpc directory:Perfect! I found the UAV/Drone model. Let me get the quadrotor task files:Now let me check if there's a reference to the original model file:Now let me search for any documentation or README about the quadrotor model:Now I have a comprehensive understanding. Let me provide you with a detailed analysis of the UAV/Drone model in this repository.

## MuJoCo MPC UAV/Drone Model Overview

### **What It Is**

The UAV model in `mujoco_mpc` is a **Quadrotor (Skydio X2 drone)** model used for a **racing/trajectory-following task**. It's not a standalone model but rather a **task-specific implementation** integrated into the MuJoCo MPC framework for predictive control demonstrations.

### **Relevant Model Files**

Here are the key files in the quadrotor task directory:

```
mjpc/tasks/quadrotor/
├── task.xml                    # Main task definition (includes gates and goal waypoints)
├── gates.xml                   # Racing gates/obstacles for the task
├── quadrotor.xml.patch         # Patch file to modify the base Skydio X2 model
├── quadrotor.cc                # C++ controller/residual functions
└── quadrotor.h                 # Header file
```

### **How It Works**

**Source Model:** The base model is the **Skydio X2** drone from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2)

**Key Characteristics:**
- **4 Propeller Motors**: Controlled via thrust commands (4 degrees of freedom)
- **Sensors**: Position, orientation (quaternion), linear velocity, angular velocity
- **Task**: Quadrotor racing - navigate through 8 gates with 12 waypoints
- **Control**: 4 motor thrust commands for vertical/rotational control

**Model Structure** (from task.xml):
- Includes common simulation assets (`common.xml`)
- Includes the modified Skydio X2 model (`quadrotor_modified.xml`)
- Includes racing gates (`gates.xml`)
- Custom parameters for MuJoCo MPC predictive sampling (horizon, timestep, trajectories)

### **Can We Use It in Plain MuJoCo?**

**Short Answer: YES**, but with modifications.

**Why modifications are needed:**
1. The `quadrotor.xml.patch` applies MuJoCo MPC-specific changes
2. The `quadrotor.cc` code implements task-specific residuals (loss functions) for the MPC framework
3. Plain MuJoCo doesn't have these MPC-specific features

### **How to Reuse for Plain MuJoCo**

#### **Option 1: Use the Base Skydio X2 Model Directly**
The cleanest approach is to get the original Skydio X2 model from MuJoCo Menagerie:

```bash
# Get the base model from:
# https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2
```

This gives you a ready-to-use drone model for plain MuJoCo.

#### **Option 2: Extract and Adapt the Modified Model**

Here's what the `.patch` file changes:

```xml
<!-- PATCH CHANGES:
1. Adds quaternion initialization: quat="0 0 0 1"
2. Keeps 4 motor thrust actuators (unchanged)
3. Removes redundant sensors (gyro, accelerometer, quat - already in task.xml)
4. Removes keyframe definitions (handled by task.xml)
-->
```

**Steps to use in plain MuJoCo:**

```python
import mujoco
import mujoco.viewer

# 1. Load the model (use the base Skydio X2 or the modified version)
model = mujoco.MjModel.from_xml_path('quadrotor_modified.xml')
data = mujoco.MjData(model)

# 2. Create a simple control loop (no MPC needed)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Set thrust commands directly
        data.ctrl[0] = 3.25  # motor 1
        data.ctrl[1] = 3.25  # motor 2
        data.ctrl[2] = 3.25  # motor 3
        data.ctrl[3] = 3.25  # motor 4 (hover thrust from quadrotor.cc)
        
        mujoco.mj_step(model, data)
        viewer.sync()
```

#### **Option 3: Use Task XML as Template**

Create a simplified MuJoCo scene:

```xml
<mujoco model="Quadrotor Simple">
  <!-- Include the base quadrotor model -->
  <include file="quadrotor_modified.xml"/>
  
  <!-- Add your own environment -->
  <worldbody>
    <geom name="floor" type="plane" size="100 100 .2"/>
    <!-- Add goal or obstacles as needed -->
  </worldbody>
  
  <!-- Define your own sensors for control -->
  <sensor>
    <framepos name="position" objtype="body" objname="x2"/>
    <framelinvel name="linear_velocity" objtype="body" objname="x2"/>
    <framequat name="orientation" objtype="body" objname="x2"/>
    <frameangvel name="angular_velocity" objtype="body" objname="x2"/>
  </sensor>
</mujoco>
```

### **Key Technical Details for Reuse**

| Aspect | Details |
|--------|---------|
| **Actuators** | 4 thrust motors on each propeller (`thrust1`, `thrust2`, `thrust3`, `thrust4`) |
| **Control Input** | 4D thrust vector (0-5+ N per motor, typically ~3.25 N for hover) |
| **State** | Position (x, y, z), Quaternion (w, x, y, z), Linear vel (3D), Angular vel (3D) |
| **Mass** | Body mass ~0.645 kg (Skydio X2) |
| **Gravity** | Standard 9.81 m/s² |
| **Assets** | 3D mesh file: `X2_lowpoly.obj` + textures |

### **Recommended Workflow**

1. **Get the base model** from MuJoCo Menagerie (Skydio X2)
2. **Create a simple XML wrapper** that includes it with your own environment
3. **Write a plain control loop** in Python using `mujoco` library (no MPC needed)
4. **Implement your own controller** (PID, reinforcement learning, etc.)

**Example**: To create a simple hover controller:

```python
# Proportional controller for height
def hover_control(data, target_z=0.5):
    z = data.body('x2').pos[2]
    error = target_z - z
    thrust = 3.25 + error * 2.0  # Proportional gain
    data.ctrl[:] = thrust  # Apply same thrust to all motors
```

Would you like me to provide a complete standalone MuJoCo example that uses this quadrotor model, or do you need help with a specific control algorithm?



