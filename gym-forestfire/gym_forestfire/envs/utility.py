import math
import random as r
import numpy as np
import pyastar
from .constants import (
    FITNESS_MEASURE,
    color2ascii,
    WIND_PARAMS,
    AGENT_LOC,
    FIRE_LOC,
    METADATA,
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
# fire_mobility (can fire spread over these cells? for the A* algorithm)
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

    f_mobility = np.ones((WIDTH, HEIGHT))

    return np.dstack((gray, temp, heat, fuel, threshold, f_mobility))

def reset_map(env):
    env[:, :, layer['gray']].fill(grass['gray'])
    env[:, :, layer['temp']].fill(0)
    env[:, :, layer['heat']].fill(grass['heat'])
    env[:, :, layer['fuel']].fill(grass['fuel'])
    env[:, :, layer['fire_mobility']].fill(1)

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
        # set the heat to -1 (= identifying property of non-burnables)
        self.W.env[self.x, self.y, layer['heat']] = dirt['heat']
        # set the fire mobility to inf for the A* algorithm
        self.W.env[self.x, self.y, layer['fire_mobility']] = np.inf

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
        # value to subtract from calculated reward when using A* measure
        self.default_reward = 0
        # so we dont have to recompute reward sometimes, and to not have to give 
        # the same reward sometimes even with no change
        self.saved_reward = 0

        self.env = create_map()
        if WIND_PARAMS == "Random":
            self.wind_speed = r.randint(0, 3)
            self.wind_vector = (r.randint(-1, 1), r.randint(-1, 1))
        else:
            self.wind_speed = WIND_PARAMS[0]
            self.wind_vector = WIND_PARAMS[1]

        # a cell is a tuple (x, y)
        self.burning_cells = set()
        self.set_fire_to(FIRE_LOC)

        self.agents = [
            Agent(self, AGENT_LOC),
        ]

        METADATA['iteration'] = 0

    def reset(self):
        if WIND_PARAMS == "Random":
            self.wind_speed = r.randint(0, 3)
            self.wind_vector = (r.randint(-1, 1), r.randint(-1, 1))
        self.RUNNING = True
        reset_map(self.env)
        self.burning_cells = set()
        self.set_fire_to(FIRE_LOC)
        self.agents = [
            Agent(self, AGENT_LOC)
        ]
        METADATA['new_ignitions'] = 0
        METADATA['burnt_cells'] = 0
        METADATA['iteration'] = 0

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
            # +1000 * (1 - percent_burnt) when fire dies out
            # -1000 when the agent dies
            reward -= METADATA['new_ignitions']
            if not self.agents:
                reward -= METADATA['death_penalty']
            if not self.burning_cells:
                perc_burnt = METADATA['burnt_cells'] / (WIDTH * HEIGHT)
                reward += METADATA['contained_bonus'] * (1 - perc_burnt)
            self.saved_reward = reward

        elif FITNESS_MEASURE == "A-Star":
            # get average distance between "center of fire" and two corner points.
            # "center of fire" is the starting point of the fire, but should be a
            # point on the frontier of the fire somehow
            # subtract the starting reward each time so that we start with 0 reward
            # when the fire has been contained, give +1000 * (1 - perc_burnt)
            start = np.array([FIRE_LOC[0], FIRE_LOC[1]])
            end1 = np.array([0, 0])
            end2 = np.array([WIDTH - 1, HEIGHT - 1])
            grid = self.env[:, :, layer['fire_mobility']].astype(np.float32)

            path1 = pyastar.astar_path(grid, start, end1, allow_diagonal=False)
            path2 = pyastar.astar_path(grid, start, end2, allow_diagonal=False)

            if not self.default_reward:
                self.default_reward = (path1.shape[0] + path2.shape[0]) / 2

            if not self.burning_cells or (path1.shape[0] == 0 and path2.shape[0] == 0):
                perc_burnt = METADATA['burnt_cells'] / (WIDTH * HEIGHT)
                reward += METADATA['contained_bonus'] * (1 - perc_burnt)
                self.RUNNING = False
            elif not self.agents:
                reward -= METADATA['death_penalty']
            else:
                # if only one path is blocked, it was most likely due to digging the cell
                # exactly in the corner, in that case just take the other value
                if path1.shape[0] == 0 or path2.shape[0] == 0:
                    reward = path1.shape[0] if path1.shape[0] != 0 else path2.shape[0]
                else:
                    reward = (path1.shape[0] + path2.shape[0]) / 2

                #print(f"Path1 length: {path1.shape[0]}, Path2 length: {path2.shape[0]}")
                #print(f"Reward: {reward} - {self.default_reward} = " + \
                #      f"{reward - self.default_reward}")

            reward -= self.default_reward
            self.saved_reward = reward

        elif FITNESS_MEASURE == "Toy":
            # simple gradient reward: the closer to the far corner the higher the reward
            # when it reaches the far corner, the agent wins and gets a big reward.
            # otherwise the reward is always the negative A* distance
            if not self.agents:
                reward = (-1) * METADATA['death_penalty']
            else:
                start = np.array([self.agents[0].y, self.agents[0].x])
                end = np.array([WIDTH - 1, HEIGHT - 1])
                grid = self.env[:, :, layer['fire_mobility']].astype(np.float32)
                path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

                reward = (-1) * path.shape[0]

                if path.shape[0] == 0:
                    reward = METADATA['contained_bonus']
                    self.RUNNING = False

            print("Reward: ", reward)
            self.saved_reward = reward


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
        print("[Percent Burnt] ", METADATA['burnt_cells'] / (WIDTH * HEIGHT))
        print("[Reward] ", self.saved_reward, "\n")

    # print all metadata info
    def print_metadata(self):
        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(METADATA)
        print("")
