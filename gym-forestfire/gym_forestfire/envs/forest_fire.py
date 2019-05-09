import gym
from gym import spaces

from .constants import (
    grass,
    dirt,
    layer,
    get_name,
    color2ascii,
    AGENT_SPEED_ITER,
    FITNESS_MEASURE,
    SMALL_NETWORK,
    NUM_ACTIONS,
    AGENT_SPEED,
    METADATA,
    VERBOSE,
    LOGGING,
    HEIGHT,
    WIDTH,
)
from .utility import (
    World,
    Agent,
)

class ForestFire(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        self.W = World()
        self.layer = layer
        self.get_name = get_name
        self.LOGGING = LOGGING
        self.METADATA = METADATA
        self.small_network = SMALL_NETWORK
        self.FITNESS_MEASURE = FITNESS_MEASURE

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(WIDTH, HEIGHT),
                                            dtype=int)

    def step(self, action):
        global AGENT_SPEED_ITER
        if action in ["N", "S", "E", "W"] or action in range(4):
            self.W.agents[0].move(action)
        # dont let the agent dig in the toy problem
        if action in ["D", 4] and not FITNESS_MEASURE == "Toy":
            self.W.agents[0].toggle_digging()
        # if the action is not handled, the agent does nothing

        # update environment only every AGENT_SPEED steps
        AGENT_SPEED_ITER -= 1
        if AGENT_SPEED_ITER == 0:
            self.update()
            AGENT_SPEED_ITER = AGENT_SPEED

        if VERBOSE:
            self.W.print_info()
        # return the layer of the map that is the grayscaled colors,
        # the reward calculated for this state, and whether the sim is done
        return [self.W.get_state(),
                self.W.get_reward(),
                not self.W.RUNNING, # NOT: to be consistent with conventions
                {}]

    def reset(self):
        self.W.reset()
        return self.W.get_state()

    def render(self):
        print(" ", end="")
        for x in range(WIDTH):
            print(x % 10, end="")
        print("")
        for y in range(HEIGHT):
            print(y % 10, end="")
            for x in range(WIDTH):
                # if the agent is at this location, print A
                if self.W.agents and (self.W.agents[0].x, self.W.agents[0].y) == (x, y):
                    print("A", end="")
                    continue
                # otherwise use the ascii mapping
                print(color2ascii[self.W.env[x, y, layer['gray']]], end="")
            print("")
        print("")

    def update(self):
        self.W.agents = [a for a in self.W.agents if not a.is_dead()]

        METADATA['iteration'] += 1
        METADATA['new_ignitions'] = 0

        burnt_out_cells = set()
        ignited_cells = set()

        for cell in self.W.burning_cells:
            x, y = cell
            # reduce fuel of burning cell
            self.W.env[x, y, layer['fuel']] -= 1
            # if burnt out, remove cell and update color
            if self.W.env[x, y, layer['fuel']] <= 0:
                self.W.env[x, y, layer['gray']] = grass['gray_burnt']
                self.W.env[x, y, layer['fire_pos']] = 0
                burnt_out_cells.add(cell)
            else:
                # else get neighbours of cell
                neighbours = self.W.get_neighbours(cell)
                for n_cell in neighbours:
                    # if neighbour is burnable
                    if self.W.is_burnable(n_cell):
                        # apply heat to it from burning cell
                        self.W.apply_heat_from_to(cell, n_cell)
                        # if neighbour is now burning, 
                        # add it to burning cells and update color
                        if self.W.is_burning(n_cell):
                            nx, ny = n_cell
                            self.W.env[nx, ny, layer['gray']] = grass['gray_burning']
                            self.W.env[nx, ny, layer['fire_pos']] = 1
                            ignited_cells.add(n_cell)

        self.W.burning_cells = self.W.burning_cells - burnt_out_cells
        self.W.burning_cells.update(ignited_cells)

        METADATA['new_ignitions'] += len(ignited_cells)
        METADATA['burnt_cells'] += len(burnt_out_cells)
        METADATA['burning_cells'] = len(self.W.burning_cells)

        # in the toy problem, the simulation is not terminated when the fire dies out
        if not self.W.agents or (not self.W.burning_cells and not FITNESS_MEASURE == "Toy"):
            self.W.RUNNING = False

        # but it is terminated when a certain number of iterations have passed
        if FITNESS_MEASURE == "Toy" and METADATA['iteration'] == METADATA['max_iteration']:
            self.W.RUNNING = False
