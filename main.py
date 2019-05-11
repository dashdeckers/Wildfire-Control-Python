import gym, gym_forestfire
from DQN import DQN
from Misc import run_random, run_human, time_simulation_run

# Suppress the many unnecessary TensorFlow warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Create the simulation
forestfire = gym.make('gym-forestfire-v0')

# Create the DQN
DQN = DQN(forestfire)

# Just a sanity check, for testing
import numpy as np
m = DQN.model
s = DQN.sim.reset()
s = np.reshape(s, [1] + list(s.shape))
p = m.predict(s)
