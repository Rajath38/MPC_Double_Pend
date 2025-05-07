import numpy as np
from envs.common import mujoco_env
from envs.common import robot_interface
import yaml
import os

class Env(mujoco_env.MujocoEnv):

    def __init__(self, config_file):

        config = self.load_config(config_file)
        self.sim_dt = config.get('sim_dt', 0.004)
        self.control_dt = config.get('control_dt', 0.02)
        model_path = config.get('model_path_xml', '') 
        
        self.frame_skip = int(round(self.control_dt / self.sim_dt))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        mujoco_env.MujocoEnv.__init__(self, model_path, self.sim_dt, self.control_dt)
        
        # set up interface which talks to mujoco
        self.interface = robot_interface.RobotInterface(self.model, self.data)

        # define init qpos and qvel
        self.init_qpos_ = [0] * self.interface.nq()
        self.init_qvel_ = [0] * self.interface.nv()

        print(f"nu = {self.interface.nu()}")
    
        self.reset_model()

    def load_config(self, file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
        
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
    
    def do_simulation(self, target, n_frames):
        print(f"nu:{self.interface.nu()}")
        for _ in range(5):
            self.interface.set_motor_torque(target)
            self.interface.step()

    def reset_model(self):
        self.init_qpos = list(self.init_qpos_)
        self.init_qvel = list(self.init_qvel_)
        self.set_state(
            np.asarray(self.init_qpos),
            np.asarray(self.init_qvel)
        )

