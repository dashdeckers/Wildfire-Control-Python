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

        burning = list(self.W.burning_cells)
        for cell in burning:
            # reduce fuel. if not burnt out, continue
            if self.W.reduce_fuel(cell):
                # for each neighbour of the burning cell
                for n_cell in self.W.get_neighbours(cell):
                    # if neighbour is burnable
                    if self.W.is_burnable(n_cell):
                        # apply heat to it from burning cell, ignite if needed
                        self.W.apply_heat_from_to(cell, n_cell)

        # in the toy problem, the simulation is not terminated when the fire dies out
        if not self.W.agents or (not self.W.burning_cells and not FITNESS_MEASURE == "Toy"):
            self.W.RUNNING = False

        # but it is terminated when a certain number of iterations have passed
        if FITNESS_MEASURE == "Toy" and METADATA['iteration'] == METADATA['max_iteration']:
            self.W.RUNNING = False
