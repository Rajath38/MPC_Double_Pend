import time
import numpy as np

import mujoco
from envs.pend.DPC import Env
from numpy import cos, sin


config_path = "config.yaml"

env = Env(config_path)

env.render()
viewer = env.viewer
viewer._paused = True

action_space_size = 1
#action = np.zeros(action_space_size)
tau = np.array([1])*-1
#action = np.zeros()

env.reset_model()

# CONTROL FREQUENCY
loop_freq  = 40  # run at 500 Hz
print(tau.shape)

while True:
    time_ = np.array(time.time())    

    env.render()
    env.step(tau)

    time.sleep(1/loop_freq)
    obs_action = env.interface.get_act_joint_positions()




    



