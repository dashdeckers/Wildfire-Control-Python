import math, time
import numpy as np
from colour import Color

WIDTH = HEIGHT = 10
RUNNING = True

def grayscale(color):
    r, g, b = color.red, color.green, color.blue
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def inbounds(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT

grass = {
    "gray"          : grayscale(Color("Green")),
    "gray_burning"  : grayscale(Color("Red")),
    "gray_burnt"    : grayscale(Color("Black")),
    "heat"      : 0.3,
    "fuel"      : 20,
    "threshold" : 3,
    "radius"    : 2
}

dirt = {
    "gray" : grayscale(Color("Brown")),
    "heat" : -1,
    "fuel" : -1,
    "threshold" : -1,
}

layer = {
    "gray" : 0,
    "temp" : 1,
    "heat" : 2,
    "fuel" : 3,
    "threshold" : 4,
}

color2ascii = {
    grayscale(Color("Green")) : '+',
    grayscale(Color("Red"))   : '@',
    grayscale(Color("Black")) : '#',
    grayscale(Color("Brown")) : '0',
}

# the entire environment consists of a WIDTHxHEIGH matrix with a
# depth layer for each important attribute of a cell:
# grayscale color (this layer is the input to the DQN)
# temperature (how much heat it has gotten from burning neighbours)
# heat (how much heat the cell can give off)
# fuel (how much longer it can burn)
# threshold (the temperature it can have before igniting)
def create_map():
    gray = np.empty((WIDTH, HEIGHT))
    gray.fill(grass["gray"])

    temp = np.zeros((WIDTH, HEIGHT))

    heat = np.empty((WIDTH, HEIGHT))
    heat.fill(grass["heat"])

    fuel = np.empty((WIDTH, HEIGHT))
    fuel.fill(grass["fuel"])

    threshold = np.empty((WIDTH, HEIGHT))
    threshold.fill(grass["threshold"])

    return np.dstack((gray, temp, heat, fuel, threshold))

env = create_map()
wind_speed = 1
wind_vector = (1, 1)

# a cell is a tuple (x, y)
burning_cells = set()

def set_fire_to(cell):
    x, y = cell
    # set the temperature to be higher than the ignition threshold
    env[x, y, layer['temp']] = env[x, y, layer['threshold']] + 1
    # update the color
    env[x, y, layer['gray']] = grass['gray_burning']
    burning_cells.add(cell)

def is_burning(cell):
    x, y = cell
    # a non-burnable or a burnt out cell cannot be burning
    if env[x, y, layer['heat']] == -1 or env[x, y, layer['fuel']] <= 0:
        return False
    return env[x, y, layer['temp']] > env[x, y, layer['threshold']]

def is_burnable(cell):
    x, y = cell
    # a burnt out cell is not burnable
    if env[x, y, layer['fuel']] <= 0:
        return False
    # if heat is set to -1, it is not a burnable cell
    return env[x, y, layer['heat']] > 0

# returns the distance and angle (relative to the wind) between two cells
def get_distance_and_angle(cell, other_cell, distance_metric):
    x, y = cell
    ox, oy = other_cell
    cx, cy = ox - x, oy - y # vector between cell and other_cell
    wx, wy = wind_vector

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
def apply_heat_from_to(cell, other_cell):
    x, y = cell
    ox, oy = other_cell

    distance, angle = get_distance_and_angle(cell, other_cell, "manhattan")

    # Formula:
    # Heat = Cell_Heat * (Wind_Speed * Angle_to_wind + Distance_to_cell)^-1
    env_factor = (wind_speed * angle + distance)**(-1)
    calculated_heat = env[x, y, layer['heat']] * env_factor

    env[ox, oy, layer['temp']] += calculated_heat

# returns every cell that can be reached by taking grass["radius"] manhattan 
# distance steps from the origin
def get_neighbours(cell):
    cx, cy = cell
    neighbours = list()
    for x in range(grass["radius"] + 1):
        for y in range(grass["radius"] + 1 - x):
            if (x, y) == (0, 0):
                continue
            # get the cell in each quadrant
            cells = [(cx + x, cy + y),
                     (cx - x, cy + y),
                     (cx + x, cy - y),
                     (cx - x, cy - y)]
            for cell in cells:
                if inbounds(*cell) and is_burnable(cell):
                    neighbours.append(cell)
    return neighbours

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_dead(self):
        # dead if the cell at position is burning
        return is_burning((self.x, self.y))

    def dig(self):
        # change the color
        env[self.x, self.y, layer['gray']] = dirt['gray']
        # set the heat to -1 (identifying property of non-burnables)
        env[self.x, self.y, layer['heat']] = dirt['heat']

    def move(self, direction):
        (nx, ny) = self._direction_to_coords(direction)
        if inbounds(nx, ny):
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

agents = [
    Agent(1, 1),
]

def update():
    global agents, burning_cells, RUNNING

    agents = [a for a in agents if not a.is_dead()]

    burnt_out_cells = set()
    ignited_cells = set()

    for cell in burning_cells:
        x, y = cell
        # reduce fuel of burning cell
        env[x, y, layer['fuel']] -= 1
        # if burnt out, remove cell and update color
        if env[x, y, layer['fuel']] <= 0:
            env[x, y, layer['gray']] = grass['gray_burnt']
            burnt_out_cells.add(cell)
        else:
            # else get neighbours of cell
            neighbours = get_neighbours(cell)
            for n_cell in neighbours:
                # if neighbour is burnable
                if is_burnable(n_cell):
                    # apply heat to it from burning cell
                    apply_heat_from_to(cell, n_cell)
                    # if neighbour is now burning, 
                    # add it to burning cells and update color
                    if is_burning(n_cell):
                        nx, ny = n_cell
                        env[nx, ny, layer['gray']] = grass['gray_burning']
                        ignited_cells.add(n_cell)

    burning_cells = burning_cells - burnt_out_cells
    burning_cells.update(ignited_cells)

    print(len(burning_cells))
    print(burning_cells)

    if not agents or not burning_cells:
        RUNNING = False

def render():
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if agents and (agents[0].x, agents[0].y) == (x, y):
                print("A", end="")
                continue
            print(color2ascii[env[x, y, layer['gray']]], end="")
        print("")
    print("")


from Misc import getch
def run_human():
    key_map = {'w':'N', 's':'S', 'd':'E', 'a':'W', ' ':'D', 'n':' '}
    set_fire_to((5, 5))
    render()
    while RUNNING:
        print("WASD to move, Space to dig, 'n' to wait, 'q' to quit.\n")
        char = getch()
        if char == 'q':
            break
        elif char in key_map:
            if key_map[char] in ["N", "S", "E", "W"]:
                agents[0].move(key_map[char])
            if key_map[char] == "D":
                agents[0].dig()
        update()
        render()
