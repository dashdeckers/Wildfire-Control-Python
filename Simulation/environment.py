import math
import random as r
import numpy as np
import pyastar
from collections import deque

from .utility import (
    get_agent_location,
    get_fire_location,
    color2ascii,
    dirt,
    grass,
    water,
    layer,
    types,
)

from .constants import (
    METADATA,
)

WIDTH = METADATA['width']
HEIGHT = METADATA['height']

'''
The entire environment consists of a WIDTH x HEIGH matrix with a
depth layer for each important attribute of a cell:

- Grayscale color
- Temperature (how much heat it has gotten from burning neighbours)
- Heat (how much potential heat the cell can give off when burning)
- Fuel (for how many iterations it can burn)
- Threshold (the temperature it can maximally have before igniting)
- Fire mobility (whether the fire can spread over these cells)
- Agent position
- Cell type
'''
def create_map():
    gray = np.full((WIDTH, HEIGHT), grass['gray'])
    temp = np.zeros((WIDTH, HEIGHT))
    heat = np.full((WIDTH, HEIGHT), grass['heat'])
    fuel = np.full((WIDTH, HEIGHT), grass['fuel'])
    threshold = np.full((WIDTH, HEIGHT), grass['threshold'])
    f_mobility = np.ones((WIDTH, HEIGHT))
    a_mobility = np.ones((WIDTH, HEIGHT))
    agent_pos = np.zeros((WIDTH, HEIGHT))
    type_map = np.zeros((WIDTH, HEIGHT))
    return np.dstack((type_map, gray, temp, heat, fuel, 
                      threshold, agent_pos, f_mobility,
                      a_mobility))

'''
Reset the layers of the map that have changed to default values
This function is also called after initialization

It also makes a simple river from a starting position near the top
of the map randomly downwards to almost the bottom of the map
'''
def reset_map(env, make_river=True):
    env[:, :, layer['gray']].fill(grass['gray'])
    env[:, :, layer['temp']].fill(0)
    env[:, :, layer['heat']].fill(grass['heat'])
    env[:, :, layer['fuel']].fill(grass['fuel'])
    env[:, :, layer['fire_mobility']].fill(1)
    env[:, :, layer['agent_mobility']].fill(1)
    env[:, :, layer['agent_pos']].fill(0)
    env[:, :, layer['type']].fill(0)

    if make_river:
        # Range of distances to keep from borders etc
        d = [1, 2, 3]
        # Remember where the fire is
        (fx, fy) = get_fire_location(WIDTH, HEIGHT)
        # Start anywhere on the x-axis
        river_x = np.random.choice(list(range(WIDTH)))
        # Start near the top on the y-axis but don't touch the border
        river_y = np.random.choice(d)
        # While we are not close to hitting the bottom of the y-axis
        while river_y < (HEIGHT - np.random.choice(d)):
            # Change the type
            env[river_x, river_y, layer['type']] = types['water']
            # Change the color
            env[river_x, river_y, layer['gray']] = water['gray']
            # Set the fire mobility to inf
            env[river_x, river_y, layer['fire_mobility']] = np.inf
            # Set the agent mobility to inf
            env[river_x, river_y, layer['agent_mobility']] = np.inf
            # Increment the position stochastically
            new_y = river_y + 1
            new_x = river_x + np.random.choice([1, -1])
            while not np.random.choice(d) <= new_x < (WIDTH - np.random.choice(d)) \
                    and not (new_x, new_y) == (fx, fy):
                new_x = river_x + np.random.choice([1, -1])

            (river_x, river_y) = (new_x, new_y)


# Agents can move, die and dig a cell to turn it into dirt
class Agent:
    def __init__(self, W, position):
        self.W = W
        # If we have already set a position for the agent on the map, use that instead
        if self.W.env[:, :, layer['agent_pos']].any():
            xarray, yarray = np.where(self.W.env[:, :, layer['agent_pos']] == 1)
            assert len(xarray) == 1 and len(yarray) == 1
            self.x, self.y = xarray[0], yarray[0]
        else:
            self.x, self.y = position
            self.W.env[self.x, self.y, layer['agent_pos']] = 1

        self.dead = False
        self.digging = True
        self.dig()

    # Agent is dead if the cell at current position is burning
    def is_dead(self):
        if self.dead or self.W.is_burning((self.x, self.y)):
            self.W.env[self.x, self.y, layer['agent_pos']] = 0
            return True
        return False

    # Turn the cell at agent position into a dirt cell
    def dig(self):
        # Ignore dig command if we are not in dig mode
        if self.digging:
            # Ignore dig command if the cell is already a dirt cell
            if not types[self.W.env[self.x, self.y, layer['type']]] == 'dirt':
                # Change the type
                self.W.env[self.x, self.y, layer['type']] = types['dirt']
                # Change the color
                self.W.env[self.x, self.y, layer['gray']] = dirt['gray']
                # Set the fire mobility to inf
                self.W.env[self.x, self.y, layer['fire_mobility']] = np.inf

    # Toggle whether the agent digs after every move or not
    def toggle_digging(self):
        self.digging = not self.digging
        self.dig()

    # Move the agent to a new position
    def move(self, direction):
        self.W.env[self.x, self.y, layer['agent_pos']] = 0
        # Get the new position
        (nx, ny) = self._direction_to_coords(direction)
        # If the new position is inbounds
        if self.W.inbounds(nx, ny) and self.W.traversable(nx, ny):
            # Go to new position
            (self.x, self.y) = (nx, ny)
            self.W.env[self.x, self.y, layer['agent_pos']] = 1
            # If the agent is in digging mode, and the new position isn't burning, dig
            if self.digging and not self.W.is_burning((nx, ny)):
                self.dig()
            # If that cell was burning, the agent dies
            if self.W.is_burning((nx, ny)):
                self.dead = True

    # Checks if the cell in the given direction is burning
    def fire_in_direction(self, direction):
        (nx, ny) = self._direction_to_coords(direction)
        return self.W.inbounds(nx, ny) and self.W.is_burning((nx, ny))

    # Convert (N, S, E, W) or (0, 1, 2, 3) into coordinates
    def _direction_to_coords(self, direction):
        if direction in ["N", 0]:
            return (self.x, self.y - 1)
        if direction in ["S", 1]:
            return (self.x, self.y + 1)
        if direction in ["E", 2]:
            return (self.x + 1, self.y)
        if direction in ["W", 3]:
            return (self.x - 1, self.y)

# The world contains the map and everything else
class World:
    def __init__(self):
        # Create the map
        self.env = create_map()
        # Set dimensions
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.DEPTH = self.get_state().shape[2]
        # Set the rest of the params
        self.reset()

    # Reset any changed parameters to their default values
    def reset(self):
        # Set wind parameters
        if METADATA['wind'] == "random":
            self.wind_speed = np.random.choice([0, 0.7, 0.85])
            self.wind_vector = (r.randint(-1, 1), r.randint(-1, 1))
        else:
            self.wind_speed = METADATA['wind'][0]
            self.wind_vector = METADATA['wind'][1]

        # Simulation is running
        self.RUNNING = True

        # Reset the map
        reset_map(self.env, METADATA['make_rivers'])

        # Keep track of burning cells. A cell is represented by a tuple (x, y)
        self.burning_cells = set()
        self.set_fire_to(get_fire_location(WIDTH, HEIGHT))

        # Create the agent(s)
        self.agents = [
            Agent(self, get_agent_location(WIDTH, HEIGHT))
        ]

        # Keep track of all the points along the border (For the A* algorithm)
        self.reset_border_points()
        self.fire_at_border = False

    # Reset the queue containing the border points of the map
    def reset_border_points(self):
        self.border_points = deque()
        for x in range(WIDTH):
            self.border_points.append([x, 0])
            self.border_points.append([x, HEIGHT - 1])
        for y in range(HEIGHT):
            self.border_points.append([0, y])
            self.border_points.append([HEIGHT - 1, y])

    # Determine whether a coordinate is within bounds of the map
    def inbounds(self, x, y):
        return 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT

    # Determine whether a coordinate is traversable by the agent
    def traversable(self, x, y):
        return self.env[x, y, layer['type']] != types['water']

    # Set a cell to be burning
    def set_fire_to(self, cell):
        x, y = cell
        # Set the temperature to be higher than the ignition threshold, if needed
        if self.env[x, y, layer['temp']] < self.env[x, y, layer['threshold']]:
            self.env[x, y, layer['temp']] = self.env[x, y, layer['threshold']] + 1
        # Change the color
        self.env[x, y, layer['gray']] = grass['gray_burning']
        # Change the type
        self.env[x, y, layer['type']] = types['fire']
        # Update burning cells
        self.burning_cells.add(cell)
        # If this is a point at the border, the fire can never be contained
        if (x == 0) or (x == WIDTH-1) or (y == 0) or (y == HEIGHT-1):
            self.fire_at_border = True

    # Determine whether the cell is on fire
    def is_burning(self, cell):
        x, y = cell
        return types[self.env[x, y, layer['type']]] == 'fire'

    # Determine whether the cell can catch fire
    def is_burnable(self, cell):
        x, y = cell
        return types[self.env[x, y, layer['type']]] not in \
                            ['fire', 'burnt', 'dirt', 'water']

    # Returns the distance and angle (relative to the wind direction) between two cells
    def get_distance_and_angle(self, cell, other_cell, distance_metric):
        x, y = cell
        ox, oy = other_cell
        cx, cy = ox - x, oy - y # Vector between cell and other_cell
        wx, wy = self.wind_vector

        # Manhattan distance: abs(x1 - x2) + abs(y1 - y2)
        if distance_metric == "manhattan":
            distance = abs(x - ox) + abs(y - oy)
        # Euclidean distance: sqrt( (x1 - x2)^2 + (y1 - y2)^2 )
        elif distance_metric == "euclidean":
            distance = ((x - ox)**2 + (y - oy)**2)**(0.5)
        # Angle between (wind_w, windy) and the vector between cell and other cell
        angle = abs(math.atan2(wx*cy - wy*cx, wx*cx + wy*cy))

        return distance, angle

    # Applies heat from a burning cell to another cell (can ignite the other cell)
    def apply_heat_from_to(self, cell, other_cell):
        x, y = cell
        ox, oy = other_cell

        distance, angle = self.get_distance_and_angle(cell, other_cell, "manhattan")

        # Formula:
        # Heat = Cell_Heat * Wind_Speed * (Angle_to_wind + Distance_to_cell)^-1
        env_factor = (angle + distance)**(-1)
        calculated_heat = self.wind_speed * self.env[x, y, layer['heat']] * env_factor

        # Heat up the cell with the calculated heat
        self.env[ox, oy, layer['temp']] += calculated_heat

        # If the cell heat exceeds the threshold, set fire to it
        if self.env[ox, oy, layer['temp']] > self.env[ox, oy, layer['threshold']]:
            self.set_fire_to(other_cell)

    # Reduce the fuel of a cell, return true if successful and false if burnt out
    def reduce_fuel(self, cell):
        x, y = cell
        # Reduce fuel of burning cell
        self.env[x, y, layer['fuel']] -= 1
        # If burnt out, remove cell from burning cells and update color and type
        if self.env[x, y, layer['fuel']] <= 0:
            self.env[x, y, layer['gray']] = grass['gray_burnt']
            self.env[x, y, layer['type']] = types['burnt']
            self.burning_cells.remove(cell)
            return False
        return True

    # Returns every cell that can be reached by taking grass["radius"] manhattan
    # distance steps from the origin and is burnable
    def get_neighbours(self, cell):
        cx, cy = cell
        neighbours = set()
        for x in range(grass['radius'] + 1):
            for y in range(grass['radius'] + 1 - x):
                if (x, y) == (0, 0):
                    continue
                # Get the cell in each quadrant
                cells = [(cx + x, cy + y),
                         (cx - x, cy + y),
                         (cx + x, cy - y),
                         (cx - x, cy - y)]
                for cell in cells:
                    if self.inbounds(*cell) and self.is_burnable(cell):
                        neighbours.add(cell)
        return neighbours

    '''
    Return the reward for the current state:

    if the fire is contained (no path from any fire to any border point):
        give a large reward
    if the agent has died:
        give a large penalty
        stop simulation
    if the fire has burnt out (no burning cells left):
        give a large reward * percent of the map untouched
        stop simulation
    otherwise:
        give a small penalty
    '''
    def get_reward(self):
        # Don't check paths if fire has reached border or if fire contained or 
        # the fire has burnt out
        if not self.fire_at_border and len(self.border_points) \
                                   and len(self.burning_cells):

            # Make a copy of the burning cells set, and pop one out
            burning = set(self.burning_cells)
            b_cell = burning.pop()

            # Try finding a route from that burning cell to a border point
            grid = self.env[:, :, layer['fire_mobility']].astype(np.float32)
            start = np.array([b_cell[0], b_cell[1]])
            end = self.border_points.pop()
            path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

            # If that route is blocked, try other routes (other border points)
            while path.shape[0] == 0:
                # If there are no more routes from that burning cell to any border
                if len(self.border_points) == 0:
                    # If there are no more routes from any burning cell to any border
                    if len(burning) == 0:
                        # The fire is contained! Give a big reward
                        return METADATA['contained_bonus']
                    # No more border points: try a different burning cell
                    self.reset_border_points()
                    b_cell = burning.pop()
                    start = np.array([b_cell[0], b_cell[1]])
                # Still border points left, try a route to a different border point
                end = self.border_points.pop()
                path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

            # Always put back the last working(!) end point for the route
            self.border_points.append(end)

        # If the agent is dead, give a big penalty
        if not self.agents:
            return METADATA['death_penalty']

        # If the fire has burnt out, give a reward based on surviving cells
        if not self.burning_cells:
            num_healthy_cells = np.count_nonzero(self.env[:, :, layer['type']] == 0)
            perc_healthy = num_healthy_cells / (WIDTH * HEIGHT)
            return METADATA['contained_bonus'] * perc_healthy

        # Otherwise (normally), give a small penalty
        return METADATA['default_reward']

    '''
    Return the state of the simulation:
    State consists of three layers with each containing only boolean entries
    - The agents position
    - The fire positions
    - The dirt positions
    '''
    def get_state(self):
        return np.dstack((self.env[:, :, layer['agent_pos']],
                          self.env[:, :, layer['type']] == types['fire'],
                          self.env[:, :, layer['fire_mobility']] != np.inf))

    # Print the map from the state (as the agent sees it)
    def print_state(self, state):
        for y in range(state.shape[1]):
            for x in range(state.shape[1]):
                # Agent layer
                if state[0, x, y, 0]:
                    print("A", end="")
                # Fire layer
                elif state[0, x, y, 1]:
                    print("@", end="")
                # Dirt layer (fire mobility is 0 if dirt)
                elif not state[0, x, y, 2]:
                    print("0", end="")
                else:
                    print("+", end="")
            print("")
        print("")

    # Print information about a cell (x, y)
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

    # Print various info about the world
    def print_info(self, total_reward):
        type_map = self.env[:, :, layer['type']]
        num_burnt = np.count_nonzero(type_map == types['burnt'])
        num_burning = np.count_nonzero(type_map == types['fire'])
        num_dug = np.count_nonzero(type_map == types['dirt'])
        num_healthy = np.count_nonzero(type_map == types['grass'])
        print("[# of Burnt Cells] ", num_burnt)
        print("[# of Burning Cells] ", num_burning)
        print("[# of Dug Cells] ", num_dug)
        print("[# of Healthy Cells] ", num_healthy)
        print("[# of Damaged Cells] ", num_dug + num_burnt + num_burning)
        print("[Percent Burnt] ", num_burnt / (WIDTH * HEIGHT))
        print("[Percent Damaged] ", (num_burnt+num_dug+num_burning) / (WIDTH * HEIGHT))
        print("[Total Reward] ", total_reward)
        print("[Current Reward] ", self.get_reward(), "\n")

    # Print all the METADATA contents
    def print_metadata(self):
        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(METADATA)
        print("")

    # Print a specific layer of the map
    def show_layer(self, layer_num=None):
        if layer_num is None:
            import pprint
            pp = pprint.PrettyPrinter()
            pp.pprint(layer)
            print("")
        else:
            print(self.env[:, :, layer[layer_num]].T)

