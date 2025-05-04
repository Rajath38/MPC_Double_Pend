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
done = False

ts, end_ts = 0, 40000
ep_rewards = []

action_space_size = 1
#action = np.zeros(action_space_size)
tau = np.array([1])
#action = np.zeros()
PI = 3.1416
const_hand = [ -0.7, 1.3, 2.0, 0.7, -1.3, -2.0]

nominal_pose = np.array([0.1, 0.9, -1.8,     -0.1, 0.9, -1.8, 
                         0.1, 0.9, -1.8,     -0.1, 0.9, -1.8]  )


env.reset_model()
data = np.zeros(12)
# CONTROL FREQUENCY
loop_freq  = 40  # run at 500 Hz
print(tau.shape)

while True:
    time_ = np.array(time.time())    

    env.render()
    env.step(tau)

    time.sleep(1/loop_freq)
    obs_action = env.interface.get_act_joint_positions()




    



