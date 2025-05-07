import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from pinocchio import casadi as cpin
import casadi
from pinocchio.visualize import MeshcatVisualizer
import time  # For real-time visualization pacing
import yaml


class CasadiDoublePendCart():

    def __init__(self, config_file):
        config = self.load_config(config_file)
        model_path = config.get('model_path_urdf', '')   #model_path = '/user/rajathch/Documents/LearningHumanoidWalking_new_CCR_6/models/bruce_mj_description/xml/bruce.xml'
        mesh_dir = ""  # Optional if you have no mesh files
        # Build models
        model, collision_model, visual_model = pin.buildModelsFromUrdf(model_path, mesh_dir)
        # === Set initial state ===
        q0 = np.array([0.0, 0.2, -0.2])
        v0 = np.array([0.0, 0.0, 0.0])

        model.gravity.linear = np.array([0.0, 0.0, -9.81])

        # === Create CasADi model ===
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        dt = 0.01

        #define all symbold

        # Symbolic torque input: apply to joint-0 only
        u_c = casadi.SX.sym("tau", cmodel.nv)

        # Define state and split
        n_x = cmodel.nq + cmodel.nv
        x_c = casadi.SX.sym('x', n_x, 1)
        self.nq = cmodel.nq
        q_c = x_c[:self.nq]
        v_c = x_c[self.nq:]

        # Dynamics via ABA
        a = cpin.aba(cmodel, cdata, q_c, v_c, u_c)
        v_next = v_c + a * dt
        q_next = cpin.integrate(cmodel, q_c, v_next * dt)
        x_next = casadi.vertcat(q_next, v_next)

        # Define integrator function
        self.Phi = casadi.Function('Phi', [x_c, u_c], [x_next], ['x', 'u'], ['x_next'])

        # === iniitalize the model
        cpin.forwardKinematics(cmodel, cdata, q_c, v_c)


    def step(self, x0, f):
        tau_val = [f, 0, 0]
        x1 = self.Phi(x0, tau_val)
        return x1

    def load_config(self, file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)


