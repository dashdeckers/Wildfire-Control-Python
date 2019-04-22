import gym, math
import random as r
import numpy as np
from gym import error, spaces, utils, spaces
from gym.utils import seeding

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
FIRE_LOC = "center_block"
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


class ForestFire(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        # set the random seed for reproducability
        r.seed(0)
        # the environment is a 2D array of elements, plus an agent
        self.env = Environment(WIDTH, HEIGHT)
        # the action space consists of 6 discrete actions, including waiting
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # useful to have direct access to
        self.width = WIDTH
        self.height = HEIGHT
        # the observation space consists of 14 continuous and discrete values
        # see features for more information
        # if we are using the full state, then the obs. space is of size:
        # (WIDTH, HEIGHT, 5)
        if USE_FULL_STATE:
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(WIDTH, HEIGHT, 5),
                                                dtype=np.bool)
        else:
            (max_ob, min_ob) = self.get_max_min_obs()
            self.observation_space = spaces.Box(low=min_ob,
                                                high=max_ob,
                                                dtype=np.float32)

    """
    Take an action and update the environment.

    This returns:

    The features in a list,
    The reward/fitness as a value,
    A boolean for whether the simultion is still running,
    Some debugging info.
    """
    def step(self, action):
        if action in ["N", "S", "E", "W"] or action in range(4):
            self.env.agents[0].move(action)
        if action in ["D", 4]:
            self.env.agents[0].dig()
        # If the action is not handled, the agent does nothing
        self.env.update()
        return [self.env.get_features(),
                self.env.get_fitness(FITNESS_MEASURE),
                not self.env.running, # NOT: to be consistent with conventions
                {}]

    # resets environment to default values
    def reset(self):
        self.env.reset_env()
        return self.env.get_features()

    # prints an ascii map of the environment
    def render(self, mode='human', close=False):
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.agents and self.env.agents[0].get_pos() == (x, y):
                    print("A", end="")
                    continue
                element = self.env.world[x][y]
                if element.burning:
                    print("@", end="")
                    continue
                if element.type in ["Grass", "Dirt"] and element.fuel == 0:
                    print("#", end="")
                    continue
                if element.type == "Grass":
                    print("+", end="")
                if element.type == "Dirt":
                    print("O", end="")
            print("")
        print("")

    # prints information on windspeed and direction
    def wind_info(self):
        if (self.env.wind_vector[0] == 0 and self.env.wind_vector[1] == 0) \
                or self.env.wind_speed == 0:
            print("No wind!")
        else:
            wind_direction = ""
            if self.env.wind_vector[1] == 1:
                wind_direction += "S"
            elif self.env.wind_vector[1] == -1:
                wind_direction += "N"
            if self.env.wind_vector[0] == 1:
                wind_direction += "E"
            elif self.env.wind_vector[0] == -1:
                wind_direction += "W"
            print("Wind direction: " + wind_direction)
            print("Wind speed: ", self.env.wind_speed)

    """
    Returns the minimum and maximum possible observation values.

    TODO:
    This does not generalize to n_agents, only works for 1 agent!
    """
    def get_max_min_obs(self):
        min_ob = np.zeros(10)#14)
        max_ob = np.zeros(10)#14)
        # x coordinate
        min_ob[0] = 0
        max_ob[0] = WIDTH
        # y coordinate
        min_ob[1] = 0
        max_ob[1] = HEIGHT
        # distance and angle for each of the 4 directions
        for idx in range(2, 10):
            if idx % 2 == 0:
                min_ob[idx] = 0
                max_ob[idx] = WIDTH + HEIGHT
            else:
                min_ob[idx] = math.pi * (-1)
                max_ob[idx] = math.pi
        """
        # num burning cells
        min_ob[10] = 0
        max_ob[10] = WIDTH * HEIGHT
        # windspeed
        min_ob[11] = 0
        max_ob[11] = 3
        # wind x
        min_ob[12] = 0
        max_ob[12] = 1
        # wind y
        min_ob[13] = 0
        max_ob[13] = 1
        """
        return min_ob, max_ob


class Environment:
    def __init__(self, width, height):
        self.running = True
        self.verbose = VERBOSE

        # environment variables
        self.width = width
        self.height = height
        self.world = self.create_world(width, height)

        # wind variables
        if WIND_PARAMS == "Random":
            self.wind_speed = r.randint(0, 3)
            self.wind_vector = (r.randint(-1, 1), r.randint(-1, 1))
        else:
            self.wind_speed = WIND_PARAMS[0]
            self.wind_vector = WIND_PARAMS[1]

        # book-keeping variables
        self.burning_cells = list()
        self.burnt_cells = list()
        self.new_ignited_cells = 0
        self.agents = list()
        self.fuel_burnt = 0
        self.barriers = set()

        # create an agent and set the fire
        self.add_agent_at(AGENT_LOC[0], AGENT_LOC[1])
        self.set_fire_at(FIRE_LOC)

    # resets the environment by calling the initializer again
    def reset_env(self):
        self.__init__(self.width, self.height)

    # creates a simple map of grass
    def create_world(self, width, height):
        world = list()
        for x in range(width):
            line = list()
            for y in range(height):
                line.append(Grass(x, y))
            world.append(line)
        return world

    # puts an agent at (x, y)
    def add_agent_at(self, x, y):
        self.agents.append(Agent(x, y, self))

    # takes an (x, y) tuple or a string and sets that element on fire
    def set_fire_at(self, pos):
        if pos == "center_block":
            (x, y) = (int(self.width/2), int(self.height/2))
            burning = [self.world[x+1][y],
                       self.world[x][y+1],
                       self.world[x][y],
                       self.world[x+1][y+1]]
            for burning_cell in burning:
                burning_cell.burning = True
                self.burning_cells.append(burning_cell)
            return
        elif pos == "center_point":
            (x, y) = (int(self.width/2), int(self.height/2))
        else:
            (x, y) = pos
        burning_cell = self.world[x][y]
        burning_cell.burning = True
        self.burning_cells.append(burning_cell)

    # returns the element at (x, y)
    def get_at(self, x, y):
        return self.world[x][y]

    # sets the element at (x, y) to a new element
    def set_at(self, x, y, cell):
        self.world[x][y] = cell

    # checks if (x, y) is within bounds
    def inbounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    # checks if the cell at (x, y) is traversable
    def traversable(self, x, y):
        cell = self.get_at(x, y)
        return cell.type in ["Grass", "Dirt"]

    """
    Main update loop:

    Checks if any agents are on burning cells and should die

    Keeps count of the total amount of fuel burnt up
    Keeps track of the number of burnt out cells

    Spreads the fire:
        Reduces fuel of burning cells, removing them if burnt out
        Applies heat from burning cells to neighbouring cells
        If any neighbouring cells ignite, add them to burning cells

    If there are no more agents or burning cells, set running to false

    TODO: Optimize this, when we are sure of the fitness function we want
    to use, we can remove some unnecessary bookkeeping
    """
    def update(self):
        agents_to_remove = set()
        for agent in self.agents:
            status = agent.update()
            if status == "Dead":
                agents_to_remove.add(agent)
        self.agents = list(set(self.agents) - agents_to_remove)

        self.fuel_burnt += len(self.burning_cells)

        cells_to_remove = set()
        cells_to_add = set()
        for cell in self.burning_cells:
            status = cell.time_step()
            if status == "Burnt Out":
                cells_to_remove.add(cell)
            if status == "No Change":
                neighbours, barrier_list = cell.get_neighbours(self)
                self.barriers.update(barrier_list)
                for n_cell in neighbours:
                    if n_cell.burnable:
                        status = n_cell.get_heat_from(cell, self)
                        if status == "Ignited":
                            cells_to_add.add(n_cell)
        self.burning_cells = list(set(self.burning_cells) - cells_to_remove)
        self.burning_cells += list(cells_to_add)

        self.burnt_cells += cells_to_remove
        self.new_ignited_cells += len(cells_to_add)

        if not self.agents or not self.burning_cells: # or fire_out_of_control
            self.running = False

    # returns the total amount of fuel available on the map
    def get_total_fuel(self):
        total_fuel = 0
        for x in range(self.width):
            for y in range(self.height):
                total_fuel += self.get_at(x, y).fuel
        return total_fuel

    # returns the percentage of the map that is burnt
    def get_percent_burnt(self):
        return len(self.burnt_cells) / (self.width * self.height)

    """
    Returns the fitness of the current state

    Burning Cells:
    Negatively counts the number of burning cells and also counts
    the number of burnt out cells but with a factor of 1/10

    Fuel Burnt:
    Negatively counts the total amount of fuel already burnt up

    Spread Blocked:
    Positively counts the number of times a fire wanted to spread to
    a cell which was not burnable.

    Ignitions and Percentage:
    -1 for every new ignited field
    +1000 * (1 - percent_burnt) when the fire dies out
    -1000 if the agent dies
    Not cumulative, only new ignitions not past ignitions are counted
           \
            \__ this seems good except there is not much positive
                feedback at the beginning to get the agent on the
                right path.
    """
    def get_fitness(self, version="Burning Cells"):
        # TODO: it should also be game-over (same as when the agent dies) when
        # the fire reaches a border
        death_penalty = 0
        contained_bonus = 0
        if not self.agents:
            death_penalty = 100
        if not self.burning_cells:
            contained_bonus = 100

        if version == "Burning Cells":
            fire_spread = len(self.burning_cells) + len(self.burnt_cells) / 10
            return int((-1) * (fire_spread + death_penalty))

        if version == "Fuel Burnt":
            return int((-1) * (self.fuel_burnt + death_penalty))

        if version == "Spread Blocked":
            return int(len(self.barriers) - death_penalty)

        if version == "Ignitions and Percentage":
            ignitions = (-1) * self.new_ignited_cells
            contained = contained_bonus * (1 - self.get_percent_burnt())
            reward = ignitions + contained - death_penalty
            # reset the new ignited cells to zero
            self.new_ignited_cells = 0
            if self.verbose:
                print(f"New ignitions reward: {ignitions}")
                # this only counts burnt and not burning but thats fine,
                # the percentage is only used when the fire has died out
                print(f"Num burnt cells: {len(self.burnt_cells)}")
                print(f"Percent Burnt: {self.get_percent_burnt()}")
                print(f"Contained reward: {contained}")
                print(f"Death penalty {death_penalty}")
                print(f"Total reward at current step: {reward}\n")
            return reward

    """
    Returns the most N, S, E, and W burning cells

    Incase the fire has reached the edge of the map and burnt out,
    we take the center point (=average) of the burnt out cells as
    our point instead.
    """
    def get_corner_points_of_fire(self):
        N = S = E = W = self.burning_cells[0]
        for fire in self.burning_cells:
            if fire.x < W.x:
                W = fire
            if fire.x > E.x:
                E = fire
            if fire.y < N.y:
                N = fire
            if fire.y > S.y:
                S = fire
        return (N, S, E, W)

    """
    > Features:

    For each agent:
        The position (x, y) of the agent
        For the furthest fire element in each direction (N, S, E, W):
            The distance between the fire and the agent
            The angle to the fire from the agent
    The total number of burning cells
    The windspeed and direction

    Resulting in a list of length: (# agents) * 10 + 4
    [a_x, a_y, DN, AN, DS, AS, DE, AE, DW, AW, #fires, w_speed, w_x, w_y]

    These still need to be improved though!!!

    > Ranges:

    - Discrete Values:
    a_x             --> [0, WIDTH)
    a_y             --> [0, HEIGHT)
    #fires          --> [0, WIDTH * HEIGHT]
    w_speed         --> [0, 3]
    w_x, w_y        --> [0, 1]

    - Continuous Values:
    DN, DS, DE, DW  --> [0, WIDTH + HEIGHT] or [0, sqrt(WIDTH^2 + HEIGHT^2)]
    AN, AS, AE, AW  --> [-pi, pi]

    TODO: Normalize them maybe? Make it an option though
    """
    def get_features(self):
        if USE_FULL_STATE:
            return self.get_full_state()

        if not self.burning_cells:
            return [-1] * (len(self.agents) * 10 + 4)

        rounding = FEAT_ROUNDING
        features = list()
        for agent in self.agents:
            features += [agent.x, agent.y]
            corner_points = self.get_corner_points_of_fire()
            for fire in corner_points:
                if rounding:
                    distance = round(fire.distance_to(agent, "euclidean"), rounding)
                    features.append(distance)
                    angle = round(fire.angle_to(agent), rounding)
                    features.append(angle)
                else:
                    features.append(fire.distance_to(agent, "euclidean"))
                    features.append(fire.angle_to(agent))
        """
        features.append(len(self.burning_cells))
        features.append(self.wind_speed)
        features += list(self.wind_vector)
        """
        return features

    '''
    Get the raw pixel input of the map as a 3D numpy matrix of size:
    (HEIGHT, WIDTH, 5). Height and Width are switched because we are
    using x, y indexing.

    The depth is 5 because we are using one-hot encoded lists of possible
    values: [grass, dirt, burning, burnt, agent]

    TODO: Could be optimized, do we need one-hot encoding? Cant we just use
    categorical data such as 1=grass, 2=dirt? I dont think so but dont know
    why not.
    '''
    def get_full_state(self):
        full_state = np.zeros((WIDTH, HEIGHT, 5))
        for y in range(self.height):
            for x in range(self.width):
                if self.agents and self.agents[0].get_pos() == (x, y):
                    full_state[x][y][4] = 1
                    continue
                element = self.world[x][y]
                if element.burning:
                    full_state[x][y][2] = 1
                    continue
                if element.type in ["Grass", "Dirt"] and element.fuel == 0:
                    full_state[x][y][3] = 1
                    continue
                if element.type == "Grass":
                    full_state[x][y][0] = 1
                if element.type == "Dirt":
                    full_state[x][y][1] = 1
        return full_state

class Element:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # define hashvalue for set compatibility
    def __hash__(self):
        return hash((self.x, self.y))

    # define equality for set compatibility
    def __eq__(self, other):
        try:
            return (self.x, self.y) == (other.x, other.y)
        except AttributeError as e:
            return NotImplemented

    # get position of element
    def get_pos(self):
        return (self.x, self.y)

    # reduces fuel by 1 if it is burning, returns "Burnt out" if it reaches 0
    def time_step(self):
        if self.burning:
            self.fuel -= 1
            if self.fuel == 0:
                self.burning = False
                self.burnable = False
                return "Burnt Out"
        return "No Change"

    """
    Gets heat from other cell if the other cell is burning

    Formula: cell.heat * (wind_speed * angle_with_wind + distance_to_cell)^-1

    The closer and the more in wind direction to the burning cell, the more
    heat is transfered (maximum heat is equal to cell.heat)

    Returns "Ignited" if the ignition threshold is reached
    """
    def get_heat_from(self, cell, env):
        if not cell.burnable:
            return "No Change"

        distance = self.distance_to(cell, "manhattan")
        angle = self.wind_angle_to(cell, env)

        env_factor = math.pow(env.wind_speed * angle + distance, -1)
        self.temp += cell.heat * env_factor

        if (self.temp > self.threshold and not self.burning):
            self.burning = True
            return "Ignited"
        return "No Change"

    # returns the distance to the other cell according to the metric specified
    def distance_to(self, cell, metric):
        if metric == "manhattan":
            return abs(self.x - cell.x) + abs(self.y - cell.y)
        if metric == "euclidean":
            return math.sqrt(math.pow(self.x - cell.x, 2)
                            + math.pow(self.y - cell.y, 2))

    # returns the angle between vector (cell, other_cell) and the wind vector
    def wind_angle_to(self, cell, env):
        (cx, cy) = (self.x - cell.x, self.y - cell.y)
        (wx, wy) = env.wind_vector
        return abs(math.atan2(wx*cy - wy*cx, wx*cx + wy*cy))

    # returns the angle between the vector (cell, other_cell) and (0, 1)
    def angle_to(self, cell):
        (cx, cy) = (self.x - cell.x, self.y - cell.y)
        (rx, ry) = (0, 1)
        return abs(math.atan2(rx*cy - ry*cx, rx*cx + ry*cy))

    # returns the neighbours: every cell that can be reached by taking
    # cell.r cityblock/manhattan steps from the origin cell is a neighbour
    def get_neighbours(self, env):
        neighbours = set()
        barriers = set()
        for x in range(self.r + 1):
            for y in range(self.r + 1 - x):
                if (x == 0 and y == 0):
                    continue
                if env.inbounds(self.x + x, self.y + y):
                    cell = env.get_at(self.x + x, self.y + y)
                    if cell.burnable:
                        neighbours.add(cell)
                    else:
                        barriers.add(cell)
                if env.inbounds(self.x - x, self.y + y):
                    cell = env.get_at(self.x - x, self.y + y)
                    if cell.burnable:
                        neighbours.add(cell)
                    else:
                        barriers.add(cell)
                if env.inbounds(self.x + x, self.y - y):
                    cell = env.get_at(self.x + x, self.y - y)
                    if cell.burnable:
                        neighbours.add(cell)
                    else:
                        barriers.add(cell)
                if env.inbounds(self.x - x, self.y - y):
                    cell = env.get_at(self.x - x, self.y - y)
                    if cell.burnable:
                        neighbours.add(cell)
                    else:
                        barriers.add(cell)
        return neighbours, barriers


class Grass(Element):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.r = 2
        self.color = "Light Green"
        self.type = "Grass"
        self.burnable = True
        self.burning = False
        self.temp = 0

        self.heat = GRASS_PARAMS["heat"]
        self.threshold = GRASS_PARAMS["threshold"]
        self.fuel = GRASS_PARAMS["fuel"]


class Dirt(Element):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.r = 0
        self.color = "Brown"
        self.type = "Dirt"
        self.burnable = False
        self.burning = False
        self.temp = 0

        self.heat = 3
        self.threshold = 1
        self.fuel = 1


class Agent:
    def __init__(self, x, y, env):
        self.x = x
        self.y = y
        self.env = env

    # define hashvalue for set compatibility
    def __hash__(self):
        return hash((self.x, self.y))

    # define equality for set compatibility
    def __eq__(self, other):
        try:
            return (self.x, self.y) == (other.x, other.y)
        except AttributeError as e:
            return NotImplemented

    # get the position of the agent
    def get_pos(self):
        return (self.x, self.y)

    # translates direction to new coordinates
    def get_new_coords(self, direction):
        if direction in ["N", 0]:
            return (self.x, self.y - 1)
        if direction in ["S", 1]:
            return (self.x, self.y + 1)
        if direction in ["E", 2]:
            return (self.x + 1, self.y)
        if direction in ["W", 3]:
            return (self.x - 1, self.y)

    # if the agent is on a burning cell, the agent dies
    def update(self):
        if self.env.world[self.x][self.y].burning:
            if VERBOSE:
                print("Agent has died")
            return "Dead"

    # moves in direction if the cell at the new coordinates is 
    # traversable, inbounds, and not occupied by another agent
    def move(self, direction):
        (newX, newY) = self.get_new_coords(direction)
        if (newX, newY) not in [a.get_pos() for a in self.env.agents]:
            if self.env.inbounds(newX, newY) and self.env.traversable(newX, newY):
                (self.x, self.y) = (newX, newY)

    # convert cell under the agent to dirt
    def dig(self):
        self.env.set_at(self.x, self.y, Dirt(self.x, self.y))
