from colour import Color

# Map Dimensions
WIDTH = 10
HEIGHT = 10
# (fire_x, fire_y)
FIRE_LOC = (5, 5)
# (agent_x, agent_y)
AGENT_LOC = (4, 4)
# "Random" or [wind_speed, (wind_x, wind_y)]
WIND_PARAMS = [1, (1, 1)]
# 6 actions to allow "do nothing" action, 5 to not allow it. 4 to disable dig
NUM_ACTIONS = 4
# "Ignitions_Percentage", "A-Star" or "Toy"
FITNESS_MEASURE = "A-Star"
# number of steps agent can do before the environment updates
AGENT_SPEED_ITER = AGENT_SPEED = 1
# whether the agent can commit suicide (move into fire)
AGENT_SUICIDE = True
# use small (only 1 hidden dense layer) network (otherwise original architecture)
SMALL_NETWORK = True
# whether to collect logging information for tensorboard
LOGGING = False
# print information on fitness etc
VERBOSE = False

# generate a unique name for TensorBoard
def get_name():
    import time
    NAME = (
        f"""Size:{(WIDTH, HEIGHT)}-"""
        f"""Reward:{FITNESS_MEASURE}-"""
        f"""A.Speed:{AGENT_SPEED}"""
        f"""Network:{"S" if SMALL_NETWORK else "L"}"""
#        f"""Waiting:{True if NUM_ACTIONS == 6 else False}-"""
        f"""Time:{time.asctime( time.localtime(time.time()) ).split()[3]}"""
    )
    return NAME

# convert a color to grayscale with formula from wikipedia
def grayscale(color):
    r, g, b = color.red, color.green, color.blue
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# the parameters for grass
grass = {
    "gray"          : grayscale(Color("Green")),
    "gray_burning"  : grayscale(Color("Red")),
    "gray_burnt"    : grayscale(Color("Black")),
    "heat"      : 0.3,
    "fuel"      : 20,
    "threshold" : 3,
    "radius"    : 1,
}

# the parameters for dirt
dirt = {
    "gray" : grayscale(Color("Brown")),
    "heat" : -1,
    "fuel" : -1,
    "threshold" : -1,
}

# which (depth) layer of the map corresponds to which attribute
layer = {
    "gray" : 0,
    "temp" : 1,
    "heat" : 2,
    "fuel" : 3,
    "threshold"     : 4,
    "fire_mobility" : 5,
    "agent_pos" : 6,
    "fire_pos"  : 7,
}

# convert grayscale to ascii for rendering
color2ascii = {
    grayscale(Color("Green")) : '+',
    grayscale(Color("Red"))   : '@',
    grayscale(Color("Black")) : '#',
    grayscale(Color("Brown")) : '0',
}

METADATA = {
# for reward stuff
    "death_penalty"   : 1000 * AGENT_SPEED,
    "contained_bonus" : 1000 * AGENT_SPEED,
    "total_reward"    : 0,
    "path_to_border"  : None,

# book-keeping
    "max_iteration" : 200,
    "iteration"     : 0,
    "new_ignitions" : 0,
    "burnt_cells"   : 0,
    "dug_cells"     : 0,
    "burning_cells" : 0,

# DQN parameters
    "memory_size"    : 20000,
    "max_eps"        : 1.0,
    "min_eps"        : 0.01,
    "eps_decay_rate" : 0.005,
    "gamma"          : 0.99,
    "alpha"          : 0.0001,  # 0.001
    "target_update"  : 20,
    "batch_size"     : 32,     # 32
}
