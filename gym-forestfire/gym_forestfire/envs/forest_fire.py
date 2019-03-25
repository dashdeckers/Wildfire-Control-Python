import gym, math
import random as r
import numpy as np
from gym import error, spaces, utils, spaces
from gym.utils import seeding

WIDTH = 30
HEIGHT = 30


class ForestFire(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        # set the random seed for reproducability
        r.seed(0)
        # the action space consists of 6 discrete actions
        self.action_space = spaces.Discrete(6)
        # the observation space consists of 14 continuous and discrete values
        # see features for more information
        (max_ob, min_ob) = self.get_max_min_obs()
        self.observation_space = spaces.Box(low=min_ob, high=max_ob, dtype=np.float32)
        # the environment is a 2D array of elements, plus an agent
        self.env = Environment(WIDTH, HEIGHT)
        # how many decimal point to round the features to (False = no rounding)
        self.rounding = 1 # False

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
        return [self.env.get_features(self.rounding), self.env.get_fitness(), self.env.running, {}]

    # resets environment to default values
    def reset(self):
        self.env.reset_env()
        return self.env.get_features(self.rounding)

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

    # prints information on windspeed and direction
    def wind_info(self):
        if (self.env.wind_vector[0] == 0 and self.env.wind_vector[1] == 0) or self.env.wind_speed == 0:
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

        # environment variables
        self.width = width
        self.height = height
        self.world = self.create_world(width, height)

        # wind variables
        self.wind_speed = 3# r.randint(0, 3)
        self.wind_vector = (1, 1) #(r.randint(-1, 1), r.randint(-1, 1))

        # book-keeping variables
        self.burning_cells = list()
        self.agents = list()
        self.fuel_burnt = 0
        self.borders_reached = ""

        # create an agent and set the fire
        self.add_agent_at(1, 1)
        self.set_fire_at("center")

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
        if pos == "center":
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

    # checks if (x, y) is at the border, if yes it remembers which one
    def atbounds(self, x, y):
        if y == 0 and "N" not in self.borders_reached:
            self.borders_reached += "N"
        if y == self.height - 1 and "S" not in self.borders_reached:
            self.borders_reached += "S"
        if x == self.width - 1 and "E" not in self.borders_reached:
            self.borders_reached += "E"
        if x == 0 and "W" not in self.borders_reached:
            self.borders_reached += "W"

    # checks if the cell at (x, y) is traversable
    def traversable(self, x, y):
        cell = self.get_at(x, y)
        return cell.type in ["Grass", "Dirt"] and not cell.burning

    """
    Main update loop:

    Checks if any agents are on burning cells and should die

    Keeps count of the total amount of fuel burnt up
    Keeps track of the borders reached by the fire

    Spreads the fire:
        Reduces fuel of burning cells, removing them if burnt out
        Applies heat from burning cells to neighbouring cells
        If any neighbouring cells ignite, add them to burning cells

    If there are no more agents or burning cells, set running to false
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
            self.atbounds(cell.x, cell.y)
            status = cell.time_step()
            if status == "Burnt Out":
                cells_to_remove.add(cell)
            if status == "No Change":
                neighbours = cell.get_neighbours(self)
                for n_cell in neighbours:
                    if n_cell.burnable:
                        status = n_cell.get_heat_from(cell, self)
                        if status == "Ignited":
                            cells_to_add.add(n_cell)
        self.burning_cells = list(set(self.burning_cells) - cells_to_remove)
        self.burning_cells += list(cells_to_add)

        if not self.agents or not self.burning_cells:
            self.running = False

    # returns the total amount of fuel available on the map
    def get_total_fuel(self):
        total_fuel = 0
        for x in range(self.width):
            for y in range(self.height):
                total_fuel += self.get_at(x, y).fuel
        return total_fuel

    # returns the amount of fuel already burnt as a negative number
    def get_fitness(self):
        extra_penalty = 0
        if not self.agents:
            extra_penalty = int(self.fuel_burnt / 2)
        return (-1) * self.fuel_burnt - extra_penalty

    # returns the middle burnt-out cell along a border if there are no
    # burning cells on that border
    # TODO: has some index errors, does not work yet
    def get_border_point(self, border):
        c_vals = list()
        if border in ["N", "S"]:
            axis_length = self.width
        elif border in ["E", "W"]:
            axis_length = self.height
        for i in range(axis_length-1):
            print(i)
            if border == "N":
                e = self.world[0][i]
            if border == "S":
                e = self.world[axis_length - 1][i]
            if border == "E":
                e = self.world[i][0]
            if border == "W":
                e = self.world[i][axis_length - 1]

            if e.burning:
                return False
            if e.fuel == 0 and border in ["N", "S"]:
                c_vals.append(e.x)
            if e.fuel == 0 and border in ["E", "W"]:
                c_vals.append(e.y)

        if c_vals:
            avg_c = sum(c_vals) / len(c_vals)
            if border == "N":
                return self.world[int(avg_c)][0]
            if border == "S":
                return self.world[int(avg_c)][axis_length - 1]
            if border == "E":
                return self.world[0][int(avg_c)]
            if border == "W":
                return self.world[axis_length - 1][int(avg_c)]

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
        """
        for direction in self.borders_reached:
            point = self.get_border_point(direction)
            if point:
                exec(f"{direction} = point")
                print(f"{direction} = {point}")
        """
        return (N, S, E, W)

    """
    Returns:
    For each agent:
        The position (x, y) of the agent
        For the furthest fire element in each direction (N, S, E, W):
            The distance between the fire and the agent
            The angle to the fire from the agent
    The total number of burning cells
    The windspeed and direction

    Resulting in a list of length: (# agents) * 10 + 4

    By default, do not round values. If an integer is passed to it, it
    will round the features to that many decimal points
    """
    def get_features(self, rounding=False):
        if not self.burning_cells:
            return [-1] * (len(self.agents) * 10 + 4)

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

    # reduces fuel by 1 if it is burning, returns "Burnt out" if fuel reaches 0
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

    # returns the angle between the vector (cell, other_cell) and the wind vector
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
        for x in range(self.r + 1):
            for y in range(self.r + 1 - x):
                if (x == 0 and y == 0):
                    continue
                if env.inbounds(self.x + x, self.y + y):
                    cell = env.get_at(self.x + x, self.y + y)
                    if cell.burnable:
                        neighbours.add(cell)
                if env.inbounds(self.x - x, self.y + y):
                    cell = env.get_at(self.x - x, self.y + y)
                    if cell.burnable:
                        neighbours.add(cell)
                if env.inbounds(self.x + x, self.y - y):
                    cell = env.get_at(self.x + x, self.y - y)
                    if cell.burnable:
                        neighbours.add(cell)
                if env.inbounds(self.x - x, self.y - y):
                    cell = env.get_at(self.x - x, self.y - y)
                    if cell.burnable:
                        neighbours.add(cell)
        return neighbours


class Grass(Element):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.r = 2
        self.color = "Light Green"
        self.type = "Grass"
        self.burnable = True
        self.burning = False
        self.temp = 0

        self.heat = 2
        self.threshold = 3
        self.fuel = 6


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
