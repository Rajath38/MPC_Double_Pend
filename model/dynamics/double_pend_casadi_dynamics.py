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
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(model_path, mesh_dir)
        # === Set initial state ===
        q0 = np.array([0.0, 0.2, -0.2])
        v0 = np.array([0.0, 0.0, 0.0])

        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])

        # === Create CasADi model ===
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        dt = 0.01

        #define all symbold

        # Symbolic torque input: apply to joint-0 only
        self.u_c = casadi.SX.sym("tau", self.cmodel.nv)

        # Define state and split
        n_x = self.cmodel.nq + self.cmodel.nv
        self.x_c = casadi.SX.sym('x', n_x, 1)
        self.nq = self.cmodel.nq
        self.nv = self.cmodel.nv
        q_c = self.x_c[:self.nq]
        v_c = self.x_c[self.nq:]

      
    
        # === iniitalize the model
        cpin.forwardKinematics(self.cmodel, self.cdata, q_c, v_c)

        cpin.framesForwardKinematics(self.cmodel, self.cdata, q_c)


        #====Energy calculations functions

        KE = cpin.computeKineticEnergy(self.cmodel, self.cdata, q_c, v_c)
        PE = cpin.computePotentialEnergy(self.cmodel, self.cdata, q_c)

        # Define Energy functions
        self.kinetic_energy_fun = casadi.Function("KE", [self.x_c], [KE])
        self.potential_energy_fun = casadi.Function("PE", [self.x_c], [PE])

        #====Dynamics calculations functions
        # Dynamics via ABA
        a = cpin.aba(self.cmodel, self.cdata, q_c, v_c, self.u_c)
        v_next = v_c + a * dt
        q_next = cpin.integrate(self.cmodel, q_c, v_next * dt)
        x_next = casadi.vertcat(q_next, v_next)
        # Define integrator function
        self.Phi = casadi.Function('Phi', [self.x_c, self.u_c], [x_next], ['x', 'u'], ['x_next'])

        self.tip = casadi.Function('tip', [self.x_c], [ self.cdata.oMf[-1].translation[[0,2]] ])

    def step(self, x0, f):
        tau_val = [f, 0, 0]
        x1 = self.Phi(x0, tau_val)
        ke = self.kinetic_energy_fun(x0).full().item()
        pe = self.potential_energy_fun(x0).full().item()
        return x1, ke, pe

    def load_config(self, file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)


