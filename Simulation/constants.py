'''
Simulation constants:

n_actions:
Number of allowed actions: 6 to allow "wait", 5 disable it, 4 to also disable dig

A_SPEED:
Number of steps agent can execute before the environment updates: Both equal

debug:
Allow collection of logging info etc: 0 = off, 1 = some, 2 = all
'''

SIZE = 10
A_SPEED = 1

# Metadata and DQN parameters
METADATA = {
    # For the reward measure
    "death_penalty"   : -1000 * A_SPEED,
    "contained_bonus" :  1000 * A_SPEED,
    "default_reward"  : -1, # -0.5

    # Simulation constants
    "width" : SIZE,
    "height": SIZE,
    "wind"  : [1, (1, 1)], # "random"
    "debug" : 1,
    "n_actions"     : 4,
    "a_speed"       : A_SPEED,
    "a_speed_iter"  : A_SPEED,

    # DQN parameters
    "memory_size"    : 20000,
    "max_eps"        : 1.0,
    "min_eps"        : 0.01,
    "eps_decay_rate" : 0.001, # 0.001
    "gamma"          : 0.999, # 0.99
    "alpha"          : 0.005, # 0.0001
    "target_update"  : 20, # 20
    "batch_size"     : 32, # 32
}