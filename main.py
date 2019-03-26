import gym, gym_forestfire
from QTable import Q_Learner
from misc import run_random


forestfire = gym.make('gym-forestfire-v0')
frozenlake = gym.make('FrozenLake-v0')
Q1 = Q_Learner(forestfire)
Q2 = Q_Learner(frozenlake)
