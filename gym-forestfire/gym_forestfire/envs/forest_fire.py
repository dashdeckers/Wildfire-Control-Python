import gym, math
import random as r
from gym import error, spaces, utils, spaces
from gym.utils import seeding

class ForestFire(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        r.seed(0)
        self.action_space = spaces.Discrete(6)
        self.env = Environment(20, 10)

    # If the action is not handled, the agent does nothing
    def step(self, action):
        if action in ["N", "S", "E", "W"] or action in range(4):
            self.env.agents[0].move(action)
        if action in ["D", 4]:
            self.env.agents[0].dig()
        self.env.update()
        return [self.env.get_features(), self.env.get_fitness(), self.env.running, {}]

    def reset(self):
        self.env.reset_env()

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


class Environment:
    def __init__(self, width, height):
        self.running = True

        self.wind_speed = r.randint(0, 3)
        self.wind_vector = (r.randint(-1, 1), r.randint(-1, 1))

        self.width = width
        self.height = height
        self.world = self.create_world(width, height)

        self.burning_cells = set()
        self.barriers = set()
        self.agents = list()

        self.total_fuel = self.get_total_fuel()
        self.fuel_burnt = 0

        self.agents.append(Agent(1, 1, self))
        burning_cell = self.world[int(width/2)][int(height/2)]
        burning_cell.burning = True
        self.burning_cells.add(burning_cell)

    def reset_env(self):
        self.__init__(self.width, self.height)

    def create_world(self, width, height):
        world = list()
        for x in range(width):
            line = list()
            for y in range(height):
                line.append(Grass(x, y))
            world.append(line)
        return world

    def get_at(self, x, y):
        return self.world[x][y]

    def set_at(self, x, y, cell):
        self.world[x][y] = cell

    def inbounds(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def traversable(self, x, y):
        cell = self.get_at(x, y)
        return cell.type in ["Grass", "Dirt"] and not cell.burning

    def update(self):
        print("Updating Environment")

        agents_to_remove = list()
        for agent in self.agents:
            status = agent.update()
            if status == "Dead":
                agents_to_remove.append(agent)
        self.agents = [a for a in self.agents if not a in agents_to_remove]

        self.fuel_burnt += len(self.burning_cells)

        cells_to_remove = set()
        cells_to_add = set()
        for cell in self.burning_cells:
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
        self.burning_cells.difference_update(cells_to_remove)
        self.burning_cells.update(cells_to_add)

        if not self.agents or not self.burning_cells:
            self.running = False

    def get_total_fuel(self):
        total_fuel = 0
        for x in range(self.width):
            for y in range(self.height):
                total_fuel += self.get_at(x, y).fuel
        return total_fuel

    def get_fitness(self):
        return self.fuel_burnt

    # returns: [distance_to, angle_to, x, y] of closest fire + [#burning cells]
    def get_features(self):
        if not self.burning_cells:
            return [-1, -1, -1, -1, 0]
        features = list()

        for agent in self.agents:
            min_distance = 1000
            closest_fire = None

            for cell in self.burning_cells:
                distance = cell.distance_to(agent, "euclidean")
                if distance < min_distance:
                    min_distance = distance
                    closest_fire = cell

            (ax, ay) = agent.get_pos()
            features.append(distance)
            features.append(closest_fire.angle_to(agent, self))
            features.append(ax)
            features.append(ay)
            features.append(len(self.burning_cells))
        return features


class Element:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_pos(self):
        return (self.x, self.y)

    def get_color(self):
        if self.fuel <= 0:
            return "Black"
        if self.burning:
            return "Red"
        if self.temp > 0.75 * self.threshold:
            return "Orange"
        if self.temp > 0.5 * self.threshold:
            return "Yellow"
        return self.color

    def time_step(self):
        if self.burning:
            self.fuel -= 1
            if self.fuel == 0:
                self.burning = False
                self.burnable = False
                return "Burnt Out"
        return "No Change"

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

    def distance_to(self, cell, metric):
        if metric == "manhattan":
            return abs(self.x - cell.x) + abs(self.y - cell.y)
        if metric == "euclidean":
            return math.sqrt(math.pow(self.x - cell.x, 2)
                            + math.pow(self.y - cell.y, 2))

    def wind_angle_to(self, cell, env):
        (cx, cy) = (self.x - cell.x, self.y - cell.y)
        (wx, wy) = env.wind_vector
        return abs(math.atan2(wx*cy - wy*cx, wx*cx + wy*cy))

    def angle_to(self, cell, env):
        (cx, cy) = (self.x - cell.x, self.y - cell.y)
        (rx, ry) = (0, 1)
        return abs(math.atan2(rx*cy - ry*cx, rx*cx + ry*cy))

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
        self.heat = 3
        self.threshold = 5
        self.fuel = 5

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
        self.threshold = 10
        self.fuel = 5


class Agent:
    def __init__(self, x, y, env):
        self.x = x
        self.y = y
        self.env = env

    def get_pos(self):
        return (self.x, self.y)

    def get_new_coords(self, direction):
        if direction in ["N", 0]:
            return (self.x, self.y - 1)
        if direction in ["S", 1]:
            return (self.x, self.y + 1)
        if direction in ["E", 2]:
            return (self.x + 1, self.y)
        if direction in ["W", 3]:
            return (self.x - 1, self.y)

    def update(self):
        if self.env.world[self.x][self.y].burning:
            print("Agent has died")
            return "Dead"

    def move(self, direction):
        (newX, newY) = self.get_new_coords(direction)
        if self.env.inbounds(newX, newY) and self.env.traversable(newX, newY):
            (self.x, self.y) = (newX, newY)

    def dig(self):
        self.env.set_at(self.x, self.y, Dirt(self.x, self.y))
