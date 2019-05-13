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
    NUM_ACTIONS,
    AGENT_SPEED,
    METADATA,
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
        self.METADATA = METADATA
        self.FITNESS_MEASURE = FITNESS_MEASURE

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(WIDTH, HEIGHT),
                                            dtype=int)

    def step(self, action):
        # Handle basic movement actions
        if action in ["N", "S", "E", "W"] or action in range(4):
            self.W.agents[0].move(action)
        # Handle the dig action
        if action in ["D", 4]:
            self.W.agents[0].toggle_digging()
        # If the action is not handled, the agent does nothing

        # Update environment only every AGENT_SPEED steps
        global AGENT_SPEED_ITER
        AGENT_SPEED_ITER -= 1
        if AGENT_SPEED_ITER == 0:
            self.update()
            AGENT_SPEED_ITER = AGENT_SPEED

        # Return the state, reward and whether the simulation is done
        return [self.W.get_state(),
                self.W.get_reward(),
                not self.W.RUNNING,
                {}]

    def reset(self):
        self.W.reset()
        return self.W.get_state()

    def render(self):
        # Print index markers along the top
        print(" ", end="")
        for x in range(WIDTH):
            print(x % 10, end="")
        print("")

        return_map = "\n"
        for y in range(HEIGHT):
            # Print index markers along the left side
            print(y % 10, end="")
            for x in range(WIDTH):
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

    def update(self):
        METADATA['iteration'] += 1

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

        # In the toy reward, the simulation is not terminated when the fire dies out
        if not self.W.agents or (not self.W.burning_cells and not FITNESS_MEASURE == "Toy"):
            self.W.RUNNING = False

        # But it is terminated when a certain number of iterations have passed
        if FITNESS_MEASURE == "Toy" and METADATA['iteration'] == METADATA['max_iteration']:
            self.W.RUNNING = False
