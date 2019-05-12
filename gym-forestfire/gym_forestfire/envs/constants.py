from colour import Color

# Map Dimensions
WIDTH = 10
HEIGHT = 10
# Starting point for the fire: (fire_x, fire_y)
FIRE_LOC = (5, 5)
# Starting point for the agent: (agent_x, agent_y)
AGENT_LOC = (4, 4)
# Wind speed and direction: "Random" or [wind_speed, (wind_x, wind_y)]
WIND_PARAMS = [1, (1, 1)]
# Number of allowed actions: 6 to allow "wait", 5 disable it, 4 to also disable dig
NUM_ACTIONS = 4
# Reward / Fitness measure to use: "A-Star" or "Toy"
FITNESS_MEASURE = "A-Star"
# Number of steps agent can execute before the environment updates
AGENT_SPEED_ITER = AGENT_SPEED = 1
# Allow or prohibit the agent to commit suicide (move into fire): Boolean
AGENT_SUICIDE = True

# Generate a unique name for each run, based on constants and the current time
def get_name():
    import time
    NAME = (
        f"""Size:{(WIDTH, HEIGHT)}-"""
        f"""Reward:{FITNESS_MEASURE}-"""
        f"""A.Speed:{AGENT_SPEED}-"""
        f"""NumActions:{NUM_ACTIONS}-"""
        f"""A.Suicide:{AGENT_SUICIDE}-"""
        f"""Time:{time.asctime( time.localtime(time.time()) ).split()[3]}"""
    )
    return NAME

# Convert a color to grayscale with the grayscale formula from Wikipedia
def grayscale(color):
    r, g, b = color.red, color.green, color.blue
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# The parameters for grass
grass = {
    "gray"         : grayscale(Color("Green")),
    "gray_burning" : grayscale(Color("Red")),
    "gray_burnt"   : grayscale(Color("Black")),
    "heat"      : 0.3,
    "fuel"      : 20,
    "threshold" : 3,
    "radius"    : 1,
}

# The parameters for dirt
dirt = {
    "gray" : grayscale(Color("Brown")),
    "heat" : -1,
    "fuel" : -1,
    "threshold" : -1,
}

# The (depth) layers of the map, which corresponds to cell attributes
layer = {
    "type" : 0,
    "gray" : 1,
    "temp" : 2,
    "heat" : 3,
    "fuel" : 4,
    "threshold" : 5,
    "agent_pos" : 6,
    "fire_mobility" : 7,
}

# Which cell type (from the type layer) corresponds to which value
types = {
    0 : "grass",
    1 : "fire",
    2 : "burnt",
    3 : "road",

    "grass" : 0,
    "fire"  : 1,
    "burnt" : 2,
    "road"  : 3,
}

# Convert grayscale to ascii for rendering
color2ascii = {
    grayscale(Color("Green")) : '+',
    grayscale(Color("Red"))   : '@',
    grayscale(Color("Black")) : '#',
    grayscale(Color("Brown")) : '0',
}

METADATA = {
# For the reward measures
    "death_penalty"   : 1000 * AGENT_SPEED,
    "contained_bonus" : 1000 * AGENT_SPEED,
    "total_reward"    : 0,
    "path_to_border"  : None,

# General book-keeping
    "max_iteration" : 200,
    "iteration"     : 0,

# DQN parameters
    "memory_size"    : 20000,
    "max_eps"        : 1.0,
    "min_eps"        : 0.01,
    "eps_decay_rate" : 0.005,
    "gamma"          : 0.999,
    "alpha"          : 0.005, # 0.001
    "target_update"  : 20,
    "batch_size"     : 32, # 32
}
