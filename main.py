from Simulation.forest_fire import ForestFire
#from DQN import DQN
from DQN_SARSA import DQN_SARSA as DQN
#from DQN_DUEL import DQN_DUEL as DQN
from misc import run_human, time_simulation_run

# Suppress the many unnecessary TensorFlow warnings
import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Create the simulation
forestfire = ForestFire()

if len(sys.argv) > 1 and sys.argv[1] == "-run":
    # Run with a specified name
    if len(sys.argv) > 2:
        DQN = DQN(forestfire, sys.argv[2])
        DQN.collect_memories(100)
        DQN.learn(10000)
    # Run without a name
    else:
        DQN = DQN(forestfire)
        DQN.collect_memories(100)
        DQN.learn(10000)
# Don't run, just create the DQN
else:
    DQN = DQN(forestfire)
