from colour import Color

# Environment / Simulation constants
ENV_CONS = {
    # Map Dimensions
    "width" : 10,
    "height": 10,

    # Starting point for the fire and the agent: (x, y)
    "f_loc" : (5, 5),
    "a_loc" : (4, 4),

    # Wind speed and direction: "Random" or [wind_speed, (wind_x, wind_y)]
    "wind"  : [1, (1, 1)],

    # Number of allowed actions: 6 to allow "wait", 5 disable it, 4 to also disable dig
    "n_actions" : 4,

    # Reward measure to use: "A-Star" or "Toy"
    "reward" : "A-Star",

    # Number of steps agent can execute before the environment updates: Both equal
    "a_speed"       : 1,
    "a_speed_iter"  : 1,

    # Allow or prohibit the agent to commit suicide (move into fire): Boolean
    "a_suicide" : True,

    # Allow collection of logging info etc: 0 = off, 1 = some, 2 = all
    "debug" : 0,
}

# Metadata and DQN parameters
METADATA = {
    # For the reward measures
    "death_penalty"   : 1000 * ENV_CONS['a_speed'],
    "contained_bonus" : 1000 * ENV_CONS['a_speed'],

    # General book-keeping
    "max_iteration" : 200,
    "iteration"     : 0,
    "constants"     : ENV_CONS,

    # DQN parameters
    "memory_size"    : 20000,
    "max_eps"        : 1.0,
    "min_eps"        : 0.01,
    "eps_decay_rate" : 0.0005, # 0.005
    "gamma"          : 0.999, # 0.99
    "alpha"          : 0.002, # 0.001
    "target_update"  : 20,
    "batch_size"     : 32, # 32
}


# Generate a unique name for each run, based on constants and the current time
def get_name():
    import time
    NAME = (
        f"""Time:{"-".join(time.asctime( time.localtime(time.time()) ).split()[1:4])}"""
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
