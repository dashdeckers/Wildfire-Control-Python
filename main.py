from Simulation.forest_fire import ForestFire
from DQN import DQN
from misc import run_human, time_simulation_run

# Suppress the many unnecessary TensorFlow warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Create the simulation
forestfire = ForestFire()

# Create the DQN
DQN = DQN(forestfire)

DQN.collect_memories(10000)
DQN.learn(20000)