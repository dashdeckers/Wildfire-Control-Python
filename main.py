import gym, gym_forestfire
from QTable import Q_Learner
from Misc import run_random
from NN_Testbed import DQN_Example

forestfire = gym.make('gym-forestfire-v0')
frozenlake = gym.make('FrozenLake-v0')
Q1 = Q_Learner(forestfire)
Q2 = Q_Learner(frozenlake)

DQN = DQN_Example(forestfire)