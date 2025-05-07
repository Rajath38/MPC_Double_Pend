import casadi
from pinocchio import casadi as cpin
import pinocchio as pin
import numpy as np

# === Load the URDF model ===
urdf_model_path = "model/double_pend.urdf"  # Set your URDF path
mesh_dir = ""  # Optional, if you have meshes

# Build the fixed-base model
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)

cmodel = cpin.Model(model)

# === Display model info ===
print("Model has DOFs:", model.nv)
for i in range(model.nv):
    print(f"Joint {i}: {model.names[i+1]}")

# === Set initial configuration and velocity ===
q0 = q0 = np.array([0.0, 0.0, 0.5]) # Initial joint positions (e.g., slider, hinge1, hinge2)
v0 = np.array([0.0, 0.0, 0.0])  # Initial joint velocities

# === Set gravity (optional — defaults to [0, 0, -9.81]) ===
model.gravity.linear = np.array([0.0, 0.0, -9.81])

# === Create data container ===
data = model.createData()

# === Simulation parameters ===
dt = 0.01
nsteps = 500
tau0 = np.zeros(model.nv)  # No actuation (passive motion)

# === Buffers to log the simulation ===
qs = [q0]
vs = [v0]