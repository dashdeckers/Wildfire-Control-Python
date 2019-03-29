import gym, gym_forestfire
from QTable import QT_Learner
from DQN import DQN_Learner
from Misc import run_random

# suppress unecessary warnings from tensorflow
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

"""
Just keeping track of ideas and knowledge:

- Subgoals: 
A* or maybe some NN that does small tasks like move/dig to location (x, y), and
then train the main algorithm to have this new action space (eg. of moving to
any location instead of the standard 6 actions)

- Hierarchical Structure:
Subgoals is a tiny example of a hierarchical structure, but this can also be done
in different ways maybe

- DQN:
The neural net will take the features as input and will output the Q-Values for each
possible action. This means we can even take pixel input (the entire map), which makes
feature engineering obsolete, but that will slow things down considerably and is less
flexible.
The NN should have 1 layer, of about 50-100 neurons according to Marco. Maybe in the
future we can increase to 2 layers but don't do that yet.

- DQN extensions:
Look into what DDQN and Duelling-DQN are

- Reward tweaking:
This is very important, the equation that gives the reward value has to really represent
the problem. There can be no way for the agent to "game the system", by achieving high
scores without doing what we intended. This one equation has to perfectly express what
we want the agent to do.

- Feature engineering:
What to pass to our neural net (DQN) from which it will learn what the ideal actions to
take are in each state. We want to feed it everything it needs to know about the world to
make an informed decision about what to do now, and in the future. Imagine yourself from
the perspective of the agent, only being able to see the features and having to contain
the fire.

> Clean code:
Make a super class from which all controllers should inherit. This will
contain methods such as average_reward_per_k, show_rewards etc. If this
is possible in python: make subclasses HAVE to override methods such as
learn, choose_action etc

Make a method get_state_from_pos() to test the features in more detail
Maybe even a set_fire_to() method to be even more precise

Implement the percent_burnt() to balance reward and to stop the simulation
early when all hope is lost (and give a large negative reward when that
happens)
"""

forestfire = gym.make('gym-forestfire-v0')
frozenlake = gym.make('FrozenLake-v0')
cartpole = gym.make('CartPole-v0')
Q1 = QT_Learner(forestfire)
Q2 = QT_Learner(frozenlake)

DQN1 = DQN_Learner(forestfire)
DQN2 = DQN_Learner(frozenlake)
DQN3 = DQN_Learner(cartpole)
