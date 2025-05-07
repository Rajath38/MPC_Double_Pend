from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from scripts.CasADi_examples.simulation_code import simulate


step_horizon = 0.1  # time between steps in seconds
N = 10              # number of look ahead steps
rob_diam = 1      # diameter of the robot
wheel_radius = 0.5    # wheel radius
Lx = 0.3            # L in J Matrix (half robot x-axis length)
Ly = 0.3            # l in J Matrix (half robot y-axis length)
sim_time = 200      # simulation time

v_max = 1
v_min = -1

omega_max = 1
omega_min = -1

#in single-shot we optimize only the state (x)

# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
state = ca.vertcat(x,y,theta)
n_state = state.numel()
print(f"states:{state}")

# control symbolic variables
v = ca.SX.sym('v')
omega = ca.SX.sym('omega')
control = ca.vertcat(v, omega)
n_control = control.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_state, N + 1)
print(f"X = {X}")

U = ca.SX.sym('U', n_control, N)
print(f"U = {U}")

P = ca.SX.sym('P', n_state*2)
print(f"P = {P}")


#state dynamics
RHS =  ca.vertcat(v*cos(theta), v*sin(theta), omega)
print(f"RHS:{RHS}")

f = ca.Function('f', [state, control], [RHS])
print(f"func:{f}")

"""state_val = [1.0, 2.0, np.pi/4]
control_val = [1.0, 0.5]

result = f(state_val, control_val)
print(result)"""


def DM2Arr(dm):
    return np.array(dm.full())

def next_state_model(state, control):

    state = DM2Arr(state)
    theta = state[2]
    print(f"theta:{theta}")
    v = control[0]
    omega = control[1]

    #rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
    rhs = np.array([v * np.cos(theta), v * np.sin(theta), omega])
    
    print(f"rhs:{rhs}")
    next_state = state + step_horizon * rhs

    print(f"ns:{next_state}")

    return next_state

def model_timestep(step_horizon, t0, state_init, u, f):

    next_state = next_state_model(state_init, u[:, 0])
    t0 = t0 + step_horizon
    u0 = ca.horzcat(u[:, 1:],ca.reshape(u[:, -1], -1, 1))

    return t0, next_state, u0

dyn = ca.Function('dyn', [state, control], [next_state_model(state, control)])


#objective 
# setting matrix_weights' variables
Q_x = 2
Q_y = 2
Q_theta = 0.5
R1 = 1
R2 = 1
cost_fn = 0

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)

# controls weights matrix
R = ca.diagcat(R1, R2)

#compute objective function symbolically, remember its just a scalar and other constraints

cost_fn = 0  # cost function
g = X[:, 0] - P[:n_state]  # constraint to initial state

for i in range(N):
    st = X[:,i]
    input = U[:,i]
    cost_fn = cost_fn + (st - P[n_state:]).T @ Q @ (st - P[n_state:]) + input.T @ R @ input

    st_next = X[:, i+1]
    st_next_pred = dyn(st, input)

    g = ca.vertcat(g, X[0,i])  #state constraint in x
    g = ca.vertcat(g, X[1,i])  #state constraint in y
    g = ca.vertcat(g, st_next - st_next_pred)   #dynamic constraint on the state

#OPT_variables = X.reshape((-1, 1))
OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)

nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)



#bounds on the optimal state variable (x,y,theta)

lbg = []
ubg = []
# initial condition constraint (X[:,0] - P[:3] == 0)
lbg += [0]*n_state
ubg += [0]*n_state

for i in range(N):
    lbg += [-2, -2]  # x, y lower bound
    lbg += [0]*n_state # dynamic(equals 0) lower bound

    ubg += [2, 2]     # x, y  upper bound
    ubg += [0]*n_state  # dynamic(equals 0) lower bound

lbg = ca.DM(lbg)
ubg = ca.DM(ubg)


# Total variables: state + control 
#x0 = (x0, y0, theta0, x1, y1, theta1, x2, y2, theta2,.... (N+1) and  v0, omega0, v1, omega1, v2, omega2 ..... (N)  
num = n_state*(N+1) + n_control*N

# First the state bounds (0 to n_state*(N+1))
lbx_state = [-ca.inf]*(n_state*(N+1))
ubx_state = [ca.inf]*(n_state*(N+1))

# Then control bounds
lbx_control = [v_min, omega_min] * N
ubx_control = [v_max, omega_max] * N

lbx = ca.DM(lbx_state + lbx_control)
ubx = ca.DM(ubx_state + ubx_control)

args = {
    'lbg': lbg,  # constraints lower bound
    'ubg': ubg,  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}



t0 = 0
t = ca.DM(t0)

x_init = ca.DM([0, 0, 0])           # initial state
x_terminal = ca.DM([2, 1, 2])     # target state

Xrecord = ca.repmat(x_init, 1, n_state)         # initial state ful
X0 = ca.repmat(x_init, 1, N+1)         # initial state full

cat_states = DM2Arr(Xrecord)         #record all states

Z0 = ca.DM.zeros((1, N+1))         # initial state ful
print(f"Z:{Z0}")

u0 = ca.DM.zeros((n_control, N))    # initial control
cat_controls = DM2Arr(u0[:, 0])     #record all controls

mpc_iter = 0
times = np.array([[0]])


if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(x_init - x_terminal) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            x_init,    # current state
            x_terminal   # target state
        )
        
        # optimization variable current state and control input
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_state*(N+1), 1),
            ca.reshape(u0, n_control*N, 1)
        )


        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_state * (N + 1):], n_control, N)
        X0 = ca.reshape(sol['x'][: n_state * (N+1)], n_state, N+1)

        print(sol['x'])

        x_now = ca.reshape(X0[:,0], 1, n_state)

        cat_states = np.vstack((cat_states, DM2Arr(x_now)))
        cat_controls = np.vstack((cat_controls, DM2Arr(u[:, 0])))
        t = np.vstack((t, t0))

        t0, x_init, u0 = model_timestep(step_horizon, t0, x_init, u, f)

        # xx ...
        t2 = time()
        print(mpc_iter)
        #print(t2-t1)
        times = np.vstack((times,t2-t))

        mpc_iter = mpc_iter + 1


    main_loop_time = time()
    ss_error = ca.norm_2(x_init - x_terminal)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(cat_states, cat_controls, times, step_horizon, N,
                np.array([0, 0, 0, 2, 1, 2]), save=False)
