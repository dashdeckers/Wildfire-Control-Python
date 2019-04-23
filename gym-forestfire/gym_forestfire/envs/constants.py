# Map Dimensions
WIDTH = 10
HEIGHT = 10
# "Random" or [wind_speed, (wind_x, wind_y)]
WIND_PARAMS = [1, (1, 1)]
# Num of decimal points to round the features to (False = no )
FEAT_ROUNDING = 1
# "Spread Blocked", "Fuel Burnt", "Burning Cells", or "Ignitions and Percentage"
FITNESS_MEASURE = "Ignitions and Percentage"
# (agent_x, agent_y)
AGENT_LOC = (4, 4)
# "center_block", "center_point", or (x, y)
FIRE_LOC = "center_point"
# 6 actions to allow "do nothing" action, 5 to not allow it
NUM_ACTIONS = 5
# Slow spread: high fuel (~60), low heat (<0.3), medium threshold (~3)
GRASS_PARAMS = {
    "heat" : 0.3,
    "threshold" : 3,
    "fuel" : 20
}
# use full pixel input instead of features
USE_FULL_STATE = True
# print information on fitness etc
VERBOSE = False
