import random as r
import numpy as np
from skimage.transform import resize

from .elements import Grass, Dirt
from .agent import Agent
from .constants import (
    WIDTH,
    HEIGHT,
    VERBOSE,
    FIRE_LOC,
    AGENT_LOC,
    WIND_PARAMS,
    FEAT_ROUNDING,
    USE_FULL_STATE,
)

class Environment:
    def __init__(self, width, height):
        # set the random seed for reproducability
        r.seed(0)
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
    Get the raw pixel input of the map as a numpy matrix of size:
    (HEIGHT, WIDTH). Height and Width are switched because we are
    using x, y indexing.

    This returns a matrix containing the grayscale values of the
    original colors (and rescales that matrix to size 84x84)
    '''
    def get_full_state(self):
        full_state = np.zeros((WIDTH, HEIGHT, 1))
        for y in range(self.height):
            for x in range(self.width):
                if self.agents and self.agents[0].get_pos() == (x, y):
                    color = self.agents[0].get_color()
                    full_state[x][y] = self.grayscale(color)
                    continue
                element = self.world[x][y]
                full_state[x][y] = self.grayscale(element.get_color())
        return resize(full_state, (84, 84), anti_aliasing=True)

    def grayscale(self, color):
        r, g, b = color.red, color.green, color.blue
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

