# This example shows how to load and move a fixed-base robot in Meshcat.
# Install Meshcat with: pip install --user meshcat

import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt

# === Load the URDF model ===
urdf_model_path = "model/double_pend.urdf"  # Set your URDF path
mesh_dir = ""

# Build the fixed-base model
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)

# === Initialize Meshcat Visualizer ===
try:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
except ImportError as err:
    print("Error initializing viewer. Install Python Meshcat.")
    print(err)
    sys.exit(0)

viz.loadViewerModel()

print("Model has DOFs:", model.nv)
for i in range(model.nv):
    print(f"Joint {i}: {model.names[model.joints[i+1].id]}")

# === Display initial configuration ===
print("nq =", model.nq)
print("nv =", model.nv)
q0 = np.array([0,0,0.5])
viz.display(q0)
viz.displayVisuals(True)

# === OPTIONAL: Set arbitrary base pose in world ===
placement = pin.SE3.Identity()
placement.translation = np.array([0.0, 0.0, 1.0])  # Global X, Y, Z
#theta = 0  # 45° rotation around Z
#placement.rotation = pin.AngleAxis(theta, np.array([0, 0, 1])).toRotationMatrix()
viz.viewer["pinocchio"].set_transform(placement.homogeneous)

# === Optional: Build convex hull if using mesh geometry ===
mesh = visual_model.geometryObjects[0].geometry
if hasattr(mesh, 'buildConvexRepresentation'):
    mesh.buildConvexRepresentation(True)
    convex = mesh.convex
    if convex is not None:
        geometry = pin.GeometryObject("convex", 0, pin.SE3.Identity(), convex)
        geometry.meshColor = np.ones(4)
        geometry.overrideMaterial = True
        geometry.meshMaterial = pin.GeometryPhongMaterial()
        geometry.meshMaterial.meshEmissionColor = np.array([1.0, 0.1, 0.1, 1.0])
        geometry.meshMaterial.meshSpecularColor = np.array([0.1, 1.0, 0.1, 1.0])
        geometry.meshMaterial.meshShininess = 0.8
        visual_model.addGeometryObject(geometry)
        viz.rebuildData()
else:
    print("Geometry is not a mesh — convex hull not built.")

# === Set initial velocity and configuration ===
v0 = np.array([0, 0, 0])
data = viz.data
pin.forwardKinematics(model, data, q0, v0)
viz.display(q0)

# === Dynamics simulation loop ===
#model.gravity.linear = np.array([0, 0, -9.81])
dt = 0.01
f = 0.01

def sim_loop():
    #tau0 = np.array([f, 0, 0]) #np.zeros(model.nv)
    #tau0 = np.zeros(model.nv)  # No actuation
    tau0 = np.array([0, 0, 0])  # Actuate all joints

    print(f"tau0:{tau0}")
    qs = [q0]
    vs = [v0]
    print(f"vs:{vs}")
    nsteps = 500
    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pin.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q, dt * vnext)
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
    return qs, vs

qs, vs = sim_loop()


# === Optional: Convert to NumPy arrays for analysis ===
qs = np.array(qs)
vs = np.array(vs)

# === Example: Print final state ===
print(f"Final joint positions (q): {qs[-1]}")
print(f"Final joint velocities (v): {vs[-1]}")

# === Optional: Save to file for analysis ===
np.save("qs_log.npy", qs)
np.save("vs_log.npy", vs)

# === Optional: Plot results ===
nsteps = 500

time = np.arange(nsteps + 1) * dt

plt.figure()
for i in range(model.nv):
    plt.plot(time, qs[:, i], label=f"q[{i}]")
plt.xlabel("Time [s]")
plt.ylabel("Joint position")
plt.legend()
plt.grid()
plt.title("Joint Positions Over Time")
plt.show()

