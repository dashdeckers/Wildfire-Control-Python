# Argument handling
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-r", "--run", action="store_true",
                    help="Start the learning process")

parser.add_argument("-m", "--memories", type=int, default=100,
                    help="Number of runs of demonstration data to initialize with")

parser.add_argument("-e", "--episodes", type=int, default=10000,
                    help="Number of episodes / runs to learn for")

parser.add_argument("-t", "--type", type=str, default="DQN",
                    choices=["DQN", "SARSA", "DDQN", "Human"],
                    help="The algorithm to use")

parser.add_argument("-n", "--name", type=str, default="no_name",
                    help="A custom name to give the saved log and model files")

args = parser.parse_args()

if args.run and args.name == "no_name":
    parser.error("You should provide a name when running a learning session")


# Suppress the many unnecessary TensorFlow warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


# Create the simulation
from Simulation.forest_fire import ForestFire
forestfire = ForestFire()


# Start learning straight away
if args.run:
    print(f"Running {args.type} with {args.memories} "
          f"memories and {args.episodes} episodes")
    
    if args.type == "DQN":
        from DQN import DQN
    if args.type == "SARSA":
        from DQN_SARSA import DQN_SARSA as DQN
    if args.type == "DDQN":
        from DQN_DUEL import DQN_DUEL as DQN

    Agent = DQN(forestfire, args.name)
    Agent.collect_memories(args.memories)
    Agent.learn(args.episodes)

# Just import everything for interactive mode
else:
    if args.type == "Human":
        from misc import run_human
        run_human(forestfire)
    else:
        from misc import run_human, time_simulation_run
        from DQN import DQN
        from DQN_SARSA import DQN_SARSA
        from DQN_DUEL import DQN_DUEL
        print("Imported algorithms for interactive mode")
