import gym, gym_forestfire
from QTable import QT_Learner
from DQN import DQN_Learner
from Misc import run_random, run_human, time_simulation_run

# suppress unecessary warnings from tensorflow
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

"""
- Subgoals:
A* or some NN that does small tasks like move/dig to location (x, y), and
then train the main algorithm to have this new action space (eg. moving to
any location instead of the standard 6 actions)

- Hierarchical Structure:
Subgoals is an example of a hierarchical structure, but this can also be done
in different ways maybe

- Reward tweaking:
This is important, the equation that gives the reward has to really represent
the problem. There can be no way for the agent to "game the system", getting
high scores without doing what is intended. This equation has to perfectly
express what we want the agent to do.
"""

forestfire = gym.make('gym-forestfire-v0')

QT = QT_Learner(forestfire)

DQN = DQN_Learner(forestfire)

# testing
import numpy as np
m = DQN.model
s = DQN.sim.reset()
s = np.reshape(s, [1] + list(s.shape))
p = m.predict(s)
