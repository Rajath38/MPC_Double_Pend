import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import casadi
from casadi import Opti
from model.dynamics.double_pend_casadi_dynamics import CasadiDoublePendCart 
import time

# === MPC Parameters ===
T = 200                          # Horizon length
TORQUE_LIMIT = 1             # Torque bound for first joint
WARMSTART = ""
pend_length  = 1

# === Load Cart-Pendulum Model ===
config_path = 'config.yaml'  # <-- Adjust this path
cart_model = CasadiDoublePendCart(config_path)

# === Action Model Class (Direct Dynamics) ===
class CasadiActionModelDirect:
    def __init__(self, cart_model):
        self.cart_model = cart_model
        self.nx = cart_model.nq + cart_model.nv
        self.nu = cart_model.nv

    def calc(self, x, u):
        xnext = self.cart_model.Phi(x, u)
        ke = self.cart_model.kinetic_energy_fun(x)
        cost = u.T @ u + ke
        return xnext, cost

# === Setup Action Models ===
runningModels = [CasadiActionModelDirect(cart_model) for _ in range(T)]
terminalModel = CasadiActionModelDirect(cart_model)

# === Optimization Problem ===
opti = Opti()
xs = [opti.variable(cart_model.nq + cart_model.nv) for _ in range(T + 1)]
us = [opti.variable(cart_model.nv) for _ in range(T)]

x0 = np.concatenate([np.array([0.0, 0.2, -0.21]), np.zeros(cart_model.nv)])
opti.subject_to(xs[0] == x0)

# Cost and dynamics constraints
totalcost = 0
for t in range(T):
    xnext, cost = runningModels[t].calc(xs[t], us[t])
    opti.subject_to(xs[t + 1] == xnext)
    totalcost += cost
    opti.subject_to(opti.bounded(-TORQUE_LIMIT, us[t][0], TORQUE_LIMIT))  # Limit 1st torque

# Terminal constraint: upright and zero velocity
'''opti.subject_to(xs[-1][cart_model.nq:] == 0)  # zero velocity
print(cart_model.tip(xs[T]))
opti.subject_to(cart_model.tip(xs[T]) == [0, pend_length])'''

# Parameters for penalizing terminal state deviation
w_tip_pos = 1e2  # weight for tip position error
w_vel = 1e1      # weight for terminal velocity

# Terminal velocity penalty
terminal_vel = xs[-1][cart_model.nq:]
totalcost += w_vel * casadi.sumsqr(terminal_vel)

# Terminal tip position penalty (e.g., tip at [0, pend_length])
tip_terminal = cart_model.tip(xs[T])
target_tip = casadi.DM([0.2, pend_length])
totalcost += w_tip_pos * casadi.sumsqr(tip_terminal - target_tip)


# Extract q2 and q3 (pendulum angles)
q2_final = xs[-1][1]  # index 1 = q2
q3_final = xs[-1][2]  # index 2 = q3

# Define upright targets
w_upright = 1e3
totalcost += w_upright * ((q2_final - np.pi)**2 + (q3_final - 0.0)**2)




opti.minimize(totalcost)



# === Warm Start (optional) ===
try:
    xs0, us0 = np.load(WARMSTART, allow_pickle=True)
    for x, xg in zip(xs, xs0): opti.set_initial(x, xg)
    for u, ug in zip(us, us0): opti.set_initial(u, ug)
except:
    print("Cold start (no warmstart file)")

# === Solve ===
opti.solver("ipopt")
try:
    sol = opti.solve()
    xs_sol = np.array([opti.value(x) for x in xs])
    us_sol = np.array([opti.value(u) for u in us])
except:
    print("Solver failed. Using debug values.")
    xs_sol = np.array([opti.debug.value(x) for x in xs])
    us_sol = np.array([opti.debug.value(u) for u in us])

# === Plot ===
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
ax0.plot(xs_sol[:, :cart_model.nq])
ax0.set_ylabel("q"); ax0.legend(["q1", "q2", "q3"])
ax1.plot(xs_sol[:, cart_model.nq:])
ax1.set_ylabel("v"); ax1.legend(["v1", "v2", "v3"])
ax2.plot(us_sol)
ax2.set_ylabel("u"); ax2.legend(["tau1", "tau2", "tau3"])

# === Meshcat Visualization of Final Trajectory ===
model = cart_model.model
data = model.createData()

# Start Meshcat viewer
visual_data = pin.GeometryData(cart_model.visual_model)
viz = MeshcatVisualizer(model, cart_model.collision_model, cart_model.visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# Helper function to display one configuration
def display_frame(q):
    pin.framesForwardKinematics(model, data, q)
    viz.display(q)

# Visualize each configuration in trajectory
print("Visualizing trajectory in Meshcat...")
for x in xs_sol:
    q = x[:cart_model.nq]
    display_frame(q)
    time.sleep(0.05)  # Adjust speed of playback
