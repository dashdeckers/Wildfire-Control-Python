import gym
from gym import spaces

from .constants import (
    METADATA,
)

from .utility import (
    grass,
    dirt,
    layer,
    get_name,
    color2ascii,
)

from .environment import (
    World,
    Agent,
)

class ForestFire(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        self.W = World()
        self.METADATA = METADATA
        self.DEBUG = METADATA['debug']
        self.layer = layer
        self.get_name = get_name
        self.width = METADATA['width']
        self.height = METADATA['height']

        self.action_space = spaces.Discrete(METADATA['n_actions'])
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(self.width, self.height),
                                            dtype=int)

    # Execute an action in the environment
    def step(self, action):
        # Handle basic movement actions
        if action in ["N", "S", "E", "W"] or action in range(4):
            self.W.agents[0].move(action)
        # Handle the dig action
        if action in ["D", 4]:
            self.W.agents[0].toggle_digging()
        # If the action is not handled, the agent does nothing

        # Update environment only every AGENT_SPEED steps
        METADATA['a_speed_iter'] -= 1
        if METADATA['a_speed_iter'] == 0:
            self.update()
            METADATA['a_speed_iter'] = METADATA['a_speed']

        # Return the state, reward and whether the simulation is done
        return [self.W.get_state(),
                self.W.get_reward(),
                not self.W.RUNNING,
                {}]

    # Reset the simulation to its initial state
    def reset(self, circle=None):
        self.W.reset(circle)
        return self.W.get_state()

    # Print an ascii rendering of the simulation
    def render(self):
        # Print index markers along the top
        print(" ", end="")
        for x in range(self.width):
            print(x % 10, end="")
        print("")

        return_map = "\n"
        for y in range(self.height):
            # Print index markers along the left side
            print(y % 10, end="")
            for x in range(self.width):
                # If the agent is at this location, print A
                if self.W.agents and (self.W.agents[0].x, self.W.agents[0].y) == (x, y):
                    return_map += 'A'
                    print("A", end="")
                # Otherwise use the ascii mapping to print the correct symbol
                else:
                    symbol = color2ascii[self.W.env[x, y, layer['gray']]]
                    return_map += symbol
                    print(symbol, end="")
            return_map += '\n'
            print("")
        print("")
        # Return a string representation of the map incase we want to save it
        return return_map

    # Updates the simulations internal state
    def update(self):
        # Remove dead agents
        self.W.agents = [a for a in self.W.agents if not a.is_dead()]

        # Iterate over a copy of the set, to avoid ConcurrentModificationException
        burning = list(self.W.burning_cells)
        # For each burning cell
        for cell in burning:
            # Reduce it's fuel. If it has not burnt out, continue
            # Burnt out cells are removed automatically by this function
            if self.W.reduce_fuel(cell):
                # For each neighbour of the (still) burning cell
                for n_cell in self.W.get_neighbours(cell):
                    # If that neighbour is burnable
                    if self.W.is_burnable(n_cell):
                        # Apply heat to it from the burning cell
                        # This function adds the n_cell to burning cells if it ignited
                        self.W.apply_heat_from_to(cell, n_cell)

        # Simulation is terminated when there are no more burning cells or agents
        if not self.W.agents or not self.W.burning_cells:
            self.W.RUNNING = False
