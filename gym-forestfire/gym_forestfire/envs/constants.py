from colour import Color
import time

# Map Dimensions
WIDTH = 10
HEIGHT = 10
# (agent_x, agent_y)
AGENT_LOC = (4, 4)
# 6 actions to allow "do nothing" action, 5 to not allow it
NUM_ACTIONS = 5
# "Random" or [wind_speed, (wind_x, wind_y)]
WIND_PARAMS = [1, (1, 1)]
# currently the only option
FITNESS_MEASURE = "Ignitions_Percentage"
# print information on fitness etc
VERBOSE = False

# generate a unique name for TensorBoard
NAME = (
#   f"""Size:{(WIDTH, HEIGHT)}-"""
    f"""Reward:{FITNESS_MEASURE}-"""
    f"""Waiting:{True if NUM_ACTIONS == 6 else False}-"""
    f"""Time:{time.asctime( time.localtime(time.time()) ).split()[3]}"""
)

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
    "radius"    : 2
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
    "threshold" : 4,
}

# convert grayscale to ascii for rendering
color2ascii = {
    grayscale(Color("Green")) : '+',
    grayscale(Color("Red"))   : '@',
    grayscale(Color("Black")) : '#',
    grayscale(Color("Brown")) : '0',
}
