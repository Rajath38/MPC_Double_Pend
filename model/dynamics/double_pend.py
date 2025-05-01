import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

# === Load the URDF model ===
urdf_model_path = "model/double_pend.urdf"  # Set your URDF path
mesh_dir = ""  # Optional, if you have meshes

# Build the fixed-base model
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)

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

# === Run simulation loop ===
for i in range(nsteps):
    q = qs[i]
    v = vs[i]

    # Compute dynamics: acceleration
    a = pin.aba(model, data, q, v, tau0)

    # Integrate forward
    vnext = v + dt * a
    qnext = pin.integrate(model, q, dt * vnext)

    # Store results
    qs.append(qnext)
    vs.append(vnext)

#uncomment to plot

"""# === Optional: Convert to NumPy arrays for analysis ===
qs = np.array(qs)
vs = np.array(vs)

# === Example: Print final state ===
print(f"Final joint positions (q): {qs[-1]}")
print(f"Final joint velocities (v): {vs[-1]}")

# === Optional: Save to file for analysis ===
np.save("qs_log.npy", qs)
np.save("vs_log.npy", vs)

# === Optional: Plot results ===

time = np.arange(nsteps + 1) * dt

plt.figure()
for i in range(model.nv):
    plt.plot(time, qs[:, i], label=f"q[{i}]")
plt.xlabel("Time [s]")
plt.ylabel("Joint position")
plt.legend()
plt.grid()
plt.title("Joint Positions Over Time")
plt.show()"""
