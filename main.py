from Simulation.forest_fire import ForestFire
from DQN import DQN
from misc import run_human, time_simulation_run

# Suppress the many unnecessary TensorFlow warnings
import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Create the simulation
forestfire = ForestFire()

# Create the DQN
DQN = DQN(forestfire)

if len(sys.argv) > 1 and sys.argv[1] == "-run":
	DQN.collect_memories(100)
	DQN.learn(20000)
