import numpy as np
import matplotlib.pyplot as plt
from model.dynamics.double_pend_casadi_dynamics import CasadiDoublePendCart  # or whatever filename you used
import time

# === Load simulation model ===
sim = CasadiDoublePendCart(config_file='config.yaml')  # replace with actual path

# === Initial state ===
q0 = np.array([0.0, 0.2, -0.2])
v0 = np.array([0.0, 0.0, 0.0])
x0 = np.concatenate([q0, v0])

# === Simulation parameters ===
dt = 0.01
nsteps = 500
torque_input = 0.0  # constant torque to joint 0

qs = [q0]
vs = [v0]
xs = [x0]

for i in range(nsteps):
    x_next = sim.step(x0, torque_input)
    x0 = x_next.full().flatten()
    
    qs.append(x0[:sim.nq])
    vs.append(x0[sim.nq:])
    xs.append(x0)

    # Optional: real-time pacing
    time.sleep(dt)

qs = np.array(qs)
vs = np.array(vs)

# === Plot joint positions ===
time_array = np.arange(nsteps + 1) * dt
plt.figure()
for i in range(qs.shape[1]):
    plt.plot(time_array, qs[:, i], label=f"q[{i}]")
plt.xlabel("Time [s]")
plt.ylabel("Joint position [rad or m]")
plt.title("Joint Positions Over Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
