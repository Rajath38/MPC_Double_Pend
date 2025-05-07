#this optimal controller is based on casADi optimizer and its modelling from the pinoccio
import casadi
import numpy as np
from model.dynamics.double_pend_casadi_dynamics import CasadiDoublePendCart 
import matplotlib.pyplot as plt

# === Load simulation model ===
model = CasadiDoublePendCart(config_file='config.yaml')  # replace with actual path

# === Initial state ===
q0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([0.0, 0.0, 0.0])
x0 = np.concatenate([q0, v0])

# === termial state ===
qt = np.array([0.0, -3.14, 0.0])
vt = np.array([0.0, 0.0, 0.0])

# === Simulation parameters ===
dt = 0.01
horizon_len = 10
torque_input = 0.0  # constant torque to joint 0
force_limit = 3  
pend_length = 1 # 0.5 + 0.5 as per our fedined urdf


#optimization problem formulation 
opti = casadi.Opti()

#optimization variable x
x = model.u_c, model.x_c
n_x = model.nv + model.nq

var_xs = [opti.variable(n_x) for t in range(horizon_len + 1)]
var_us = [opti.variable(model.nv) for t in range(horizon_len)]

#cost function

totalcost = 0
opti.subject_to(var_xs[0] == x0)  #initial constraint 

for t in range(horizon_len):

    xnext = model.Phi(var_xs[t], var_us[t])  # u is full vector (size = nv)
    ke = model.kinetic_energy_fun(var_xs[t])
    pe = model.potential_energy_fun(var_xs[t])

    totalcost += ke
    if t == horizon_len - 1:
        totalcost += -pe

    opti.subject_to(var_xs[t + 1] == xnext)
    opti.subject_to(opti.bounded(-force_limit, var_us[t][0], force_limit)) # control is limited

#Additional terminal constraint
opti.subject_to(var_xs[horizon_len][model.nq:] == 0)  # 0 terminal velocities
opti.subject_to(model.tip(var_xs[horizon_len][1])==pend_length) # tip of pendulum at max altitude
#opti.subject_to(model.tip(var_xs[horizon_len])[1] >= 0.95 * pend_length)
#tip_z = model.tip(var_xs[horizon_len])[1]
#totalcost += 100 * (tip_z - pend_length)**2

### SOLVE
opti.minimize(totalcost)


opti.solver("ipopt") # set numerical backend
# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    xs_sol = np.array([ opti.value(x) for x in var_xs ])
    us_sol = np.array([ opti.value(u) for u in var_us ])
except:
    print('ERROR in convergence, plotting debug info.')
    xs_sol = np.array([ opti.debug.value(x) for x in var_xs ])
    us_sol = np.array([ opti.debug.value(u) for u in var_us ])

### PLOT AND VIZ
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3) #, constrained_layout=True)
ax0.plot(xs_sol[:,:model.nq])
ax0.set_ylabel('q')
ax0.legend(['1','2'])
ax1.plot(xs_sol[:,model.nq:])
ax1.set_ylabel('v')
ax1.legend(['1','2'])
ax2.plot(us_sol)
ax2.set_ylabel('u')
ax2.legend(['1','2'])







