import math
import random as r
import numpy as np
import pyastar
from collections import deque
from .constants import (
    FITNESS_MEASURE,
    AGENT_SUICIDE,
    color2ascii,
    WIND_PARAMS,
    AGENT_LOC,
    FIRE_LOC,
    METADATA,
    VERBOSE,
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
# agent and fire positions (extra inputs to the DQN)
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

    agent_pos = np.zeros((WIDTH, HEIGHT))

    fire_pos = np.zeros((WIDTH, HEIGHT))

    return np.dstack((gray, temp, heat, fuel, threshold, 
                      f_mobility, agent_pos, fire_pos))

def reset_map(env):
    env[:, :, layer['gray']].fill(grass['gray'])
    env[:, :, layer['temp']].fill(0)
    env[:, :, layer['heat']].fill(grass['heat'])
    env[:, :, layer['fuel']].fill(grass['fuel'])
    env[:, :, layer['fire_mobility']].fill(1)
    env[:, :, layer['agent_pos']].fill(0)
    env[:, :, layer['fire_pos']].fill(0)

class Agent:
    def __init__(self, W, position):
        self.x, self.y = position
        self.W = W
        self.W.env[self.x, self.y, layer['agent_pos']] = 1
        self.dead = False
        self.digging = True
        if FITNESS_MEASURE != "Toy":
            self.dig()

    def is_dead(self):
        # dead if the cell at position is burning
        return self.dead or self.W.is_burning((self.x, self.y))

    def dig(self):
        # change the color
        self.W.env[self.x, self.y, layer['gray']] = dirt['gray']
        # set the heat to -1 (= identifying property of non-burnables)
        self.W.env[self.x, self.y, layer['heat']] = dirt['heat']
        # set the fire mobility to inf for the A* algorithm
        # and increase the dug cell count (but only if a change occurs)
        if self.W.env[self.x, self.y, layer['fire_mobility']] != np.inf:
            self.W.env[self.x, self.y, layer['fire_mobility']] = np.inf
            METADATA['dug_cells'] += 1

    def toggle_digging(self):
        self.dig()
        self.digging = not self.digging

    def move(self, direction):
        self.W.env[self.x, self.y, layer['agent_pos']] = 0
        (nx, ny) = self._direction_to_coords(direction)
        # if the new position is inbounds
        if self.W.inbounds(nx, ny):
            # if the new position is not burning (or if agent can commit suicide)
            if not self.W.is_burning((nx, ny)) or AGENT_SUICIDE:
                # go to new position
                (self.x, self.y) = (nx, ny)
                self.W.env[self.x, self.y, layer['agent_pos']] = 1
                # if that cell was burning, the agent dies
                if self.W.is_burning((nx, ny)):
                    self.dead = True
            # if the agent is in digging mode, dig at the new position (provided
            # the new position is not burning and the fitness measure allows it)
            if self.digging and not self.W.is_burning((nx, ny)) \
                            and not FITNESS_MEASURE == "Toy":
                self.dig()

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
        self.set_fire_to(FIRE_LOC)

        self.agents = [
            Agent(self, AGENT_LOC),
        ]

        METADATA['iteration'] = 0

        # add all the points along the border to a deque object
        self.border_points = deque()
        for x in range(WIDTH):
            self.border_points.append([x, 0])
            self.border_points.append([x, HEIGHT - 1])
        for y in range(HEIGHT):
            self.border_points.append([0, y])
            self.border_points.append([HEIGHT - 1, y])

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
        # update the fire_pos layer
        self.env[x, y, layer['fire_pos']] = 1
        self.burning_cells.add(cell)
        METADATA['burning_cells'] = len(self.burning_cells)

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

        elif FITNESS_MEASURE == "A-Star":
            '''
            if the fire is contained (no path to any border point):
                give a large reward
            if the agent has died:
                give a large penalty
                stop simulation
            if the fire has died out (no burning cells left):
                give a large reward * percent of the map untouched
                stop simulation
            otherwise:
                give a penalty (-1) for each burning cell
            '''
            if len(self.border_points):
                grid = self.env[:, :, layer['fire_mobility']].astype(np.float32)
                start = np.array([FIRE_LOC[0], FIRE_LOC[1]])
                end = self.border_points.pop()
                path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

                # if that route was blocked, try other routes
                while path.shape[0] == 0:
                    # if there are no more routes to try (all paths to the border are blocked)
                    if len(self.border_points) == 0:
                        # the fire is contained! give a big reward
                        return METADATA['contained_bonus']
                    # try other routes
                    end = self.border_points.pop()
                    path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

                # always put back the last working end point_point for the route
                self.border_points.append(end)
                METADATA['path_to_border'] = path

            # if agent is dead, give a big penalty
            if not self.agents:
                return (-1) * METADATA['death_penalty']

            # if the fire has burnt out, give a reward based on surviving cells
            if not self.burning_cells:
                num_damaged_cells = METADATA['burning_cells'] + METADATA['dug_cells']
                perc_damaged = num_damaged_cells / (WIDTH * HEIGHT)
                return METADATA['contained_bonus'] * (1 - perc_damaged)

            # otherwise (normally), give a penalty based on the number of burning cells
            return (-1) * METADATA['burning_cells']

        elif FITNESS_MEASURE == "Toy":
            # simple gradient reward: the closer to the far corner the higher the reward
            # when it reaches the far corner, the agent wins and gets a big reward.
            # otherwise the reward is always the negative A* distance
            if not self.agents:
                reward = (-1) * METADATA['death_penalty']
            else:
                start = np.array([self.agents[0].x, self.agents[0].y])
                end = np.array([WIDTH - 1, HEIGHT - 1])
                grid = self.env[:, :, layer['fire_mobility']].astype(np.float32)
                path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

                reward = (-1) * path.shape[0]

                if path.shape[0] == 0:
                    reward = METADATA['contained_bonus']
                    self.RUNNING = False

            if VERBOSE:
                print("Reward: ", reward)

        else:
            raise Exception(f"{FITNESS_MEASURE} is not a valid fitness measure")
        return reward

    def get_state(self):
        if FITNESS_MEASURE == "Toy":
            return self.env[:, :, layer['gray']]
        else:
            return np.dstack((self.env[:, :, layer['agent_pos']],
                              self.env[:, :, layer['fire_pos']],
                              self.env[:, :, layer['fire_mobility']]))

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
        num_damaged_cells = METADATA['burning_cells'] + METADATA['dug_cells']
        print("[Percent Burnt] ", METADATA['burnt_cells'] / (WIDTH * HEIGHT))
        print("[Percent Damaged] ", num_damaged_cells / (WIDTH * HEIGHT))
        print("[Total Reward] ", METADATA['total_reward'])
        print("[Current Reward] ", self.get_reward(), "\n")

    # print all metadata info
    def print_metadata(self):
        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(METADATA)
        print("")

    def show_layer(self, layer_num=None):
        if layer_num is None:
            import pprint
            pp = pprint.PrettyPrinter()
            pp.pprint(layer)
            print("")
        else:
            print(self.env[:, :, layer[layer_num]].T)
