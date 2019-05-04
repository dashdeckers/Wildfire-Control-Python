import math
import random as r
import numpy as np
from .constants import (
    FITNESS_MEASURE,
    color2ascii,
    WIND_PARAMS,
    AGENT_LOC,
    HEIGHT,
    WIDTH,
    dirt,
    grass,
    layer,
)

# the entire environment consists of a WIDTHxHEIGH matrix with a
# depth layer for each important attribute of a cell:
# grayscale color (this layer is the input to the DQN)
# temperature (how much heat it has gotten from burning neighbours)
# heat (how much heat the cell can give off)
# fuel (how much longer it can burn)
# threshold (the temperature it can have before igniting)
def create_map():
    gray = np.empty((WIDTH, HEIGHT))
    gray.fill(grass['gray'])

    temp = np.zeros((WIDTH, HEIGHT))

    heat = np.empty((WIDTH, HEIGHT))
    heat.fill(grass['heat'])

    fuel = np.empty((WIDTH, HEIGHT))
    fuel.fill(grass['fuel'])

    threshold = np.empty((WIDTH, HEIGHT))
    threshold.fill(grass['threshold'])

    return np.dstack((gray, temp, heat, fuel, threshold))

def reset_map(env):
    env[:, :, layer['gray']].fill(grass['gray'])
    env[:, :, layer['temp']].fill(0)
    env[:, :, layer['heat']].fill(grass['heat'])
    env[:, :, layer['fuel']].fill(grass['fuel'])

class Agent:
    def __init__(self, W, position):
        self.x, self.y = position
        self.W = W

    def is_dead(self):
        # dead if the cell at position is burning
        return self.W.is_burning((self.x, self.y))

    def dig(self):
        # change the color
        self.W.env[self.x, self.y, layer['gray']] = dirt['gray']
        # set the heat to -1 (identifying property of non-burnables)
        self.W.env[self.x, self.y, layer['heat']] = dirt['heat']

    def move(self, direction):
        (nx, ny) = self._direction_to_coords(direction)
        if self.W.inbounds(nx, ny):
            (self.x, self.y) = (nx, ny)

    def _direction_to_coords(self, direction):
        if direction in ["N", 0]:
            return (self.x, self.y - 1)
        if direction in ["S", 1]:
            return (self.x, self.y + 1)
        if direction in ["E", 2]:
            return (self.x + 1, self.y)
        if direction in ["W", 3]:
            return (self.x - 1, self.y)

class World:
    def __init__(self):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.RUNNING = True

        self.env = create_map()
        if WIND_PARAMS == "Random":
            self.wind_speed = r.randint(0, 3)
            self.wind_vector = (r.randint(-1, 1), r.randint(-1, 1))
        else:
            self.wind_speed = WIND_PARAMS[0]
            self.wind_vector = WIND_PARAMS[1]

        # a cell is a tuple (x, y)
        self.burning_cells = set()
        self.set_fire_to((5, 5))

        self.agents = [
            Agent(self, AGENT_LOC),
        ]

        self.METADATA = {
            "death_penalty" : 100,
            "contained_bonus" : 100,
            "new_ignitions" : 0,
            "burnt_cells" : 0,
        }

    def reset(self):
        self.RUNNING = True
        reset_map(self.env)
        self.burning_cells = set()
        self.set_fire_to((5, 5))
        self.agents = [
            Agent(self, AGENT_LOC)
        ]
        self.METADATA['new_ignitions'] = 0
        self.METADATA['burnt_cells'] = 0

    def inbounds(self, x, y):
        return 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT

    def set_fire_to(self, cell):
        x, y = cell
        # set the temperature to be higher than the ignition threshold
        self.env[x, y, layer['temp']] = self.env[x, y, layer['threshold']] + 1
        # update the color
        self.env[x, y, layer['gray']] = grass['gray_burning']
        self.burning_cells.add(cell)

    def is_burning(self, cell):
        x, y = cell
        # a non-burnable or a burnt out cell cannot be burning
        if self.env[x, y, layer['heat']] == -1 or self.env[x, y, layer['fuel']] <= 0:
            return False
        return self.env[x, y, layer['temp']] > self.env[x, y, layer['threshold']]

    def is_burnable(self, cell):
        x, y = cell
        # a burning cell is not burnable
        if self.env[x, y, layer['temp']] > self.env[x, y, layer['threshold']]:
            return False
        # a burnt out cell is not burnable
        if self.env[x, y, layer['fuel']] <= 0:
            return False
        # if heat is set to -1, it is not a burnable cell
        return self.env[x, y, layer['heat']] > 0

    # returns the distance and angle (relative to the wind) between two cells
    def get_distance_and_angle(self, cell, other_cell, distance_metric):
        x, y = cell
        ox, oy = other_cell
        cx, cy = ox - x, oy - y # vector between cell and other_cell
        wx, wy = self.wind_vector

        # manhattan distance: abs(x1 - x2) + abs(y1 - y2)
        if distance_metric == "manhattan":
            distance = abs(x - ox) + abs(y - oy)
        # euclidean distance: sqrt( (x1 - x2)^2 + (y1 - y2)^2 )
        elif distance_metric == "euclidean":
            distance = ((x - ox)**2 + (y - oy)**2)**(0.5)
        # angle between (wind_w, windy) and the vector between cell and other cell
        angle = abs(math.atan2(wx*cy - wy*cx, wx*cx + wy*cy))

        return distance, angle

    # applies heat from one burning cell to any other cell
    def apply_heat_from_to(self, cell, other_cell):
        x, y = cell
        ox, oy = other_cell

        distance, angle = self.get_distance_and_angle(cell, other_cell, "manhattan")

        # Formula:
        # Heat = Cell_Heat * (Wind_Speed * Angle_to_wind + Distance_to_cell)^-1
        env_factor = (self.wind_speed * angle + distance)**(-1)
        calculated_heat = self.env[x, y, layer['heat']] * env_factor

        self.env[ox, oy, layer['temp']] += calculated_heat

    # returns every cell that can be reached by taking grass["radius"] manhattan 
    # distance steps from the origin
    def get_neighbours(self, cell):
        cx, cy = cell
        neighbours = list()
        for x in range(grass['radius'] + 1):
            for y in range(grass['radius'] + 1 - x):
                if (x, y) == (0, 0):
                    continue
                # get the cell in each quadrant
                cells = [(cx + x, cy + y),
                         (cx - x, cy + y),
                         (cx + x, cy - y),
                         (cx - x, cy - y)]
                for cell in cells:
                    if self.inbounds(*cell) and self.is_burnable(cell):
                        neighbours.append(cell)
        return neighbours

    def get_reward(self):
        reward = 0
        if FITNESS_MEASURE == "Ignitions_Percentage":
            # -1 for every new ignited field
            # +100 * (1 - percent_burnt) when fire dies out
            # -100 when the agent dies
            reward -= self.METADATA['new_ignitions']
            if not self.agents:
                reward -= self.METADATA['death_penalty']
            if not self.burning_cells:
                perc_burnt = self.METADATA['burnt_cells'] / (WIDTH * HEIGHT)
                reward += self.METADATA['contained_bonus'] * (1 - perc_burnt)
        else:
            raise Exception(f"{FITNESS_MEASURE} is not a valid fitness measure")
        return reward

    def get_state(self):
        return self.env[:, :, layer['gray']]

    # pass a cell (x, y) to print information on it
    def inspect(self, cell):
        x, y = cell
        cell_info = self.env[x, y, :]
        gray = cell_info[layer['gray']]
        print("\n[Color] Grayscale:", gray, "Ascii:", color2ascii[gray])
        print("[Temperature] ", cell_info[layer['temp']])
        print("[Heat Power] ", cell_info[layer['heat']])
        print("[Fuel Level] ", cell_info[layer['fuel']])
        print("[Ignition Threshold], ", cell_info[layer['threshold']])
        if self.agents and (self.agents[0].x, self.agents[0].y) == (x, y):
            agent_at_loc = True
        else:
            agent_at_loc = False
        print("[Cell contains an Agent] ", agent_at_loc, "\n")


    # print various info about the world
    def print_info(self):
        print("[New Ignitions] ", self.METADATA['new_ignitions'])
        print("[Total Burnt Cells] ", self.METADATA['burnt_cells'])
        print("[Percent Burnt] ", self.METADATA['burnt_cells'] / (WIDTH * HEIGHT))
        print("[Reward] ", self.get_reward(), "\n")

