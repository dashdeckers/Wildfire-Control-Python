import math
from .constants import GRASS_PARAMS
from colour import Color

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

    # gets the color of the element
    def get_color(self):
        # grass can be a different color when burning or burnt
        if self.type in ["Grass"]:
            if self.fuel == 0:
                return Color("Black")
            if self.burning:
                return Color("Red")
        return Color(self.color)

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
        self.color = "Green"
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
