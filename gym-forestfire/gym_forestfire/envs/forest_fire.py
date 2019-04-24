import gym
import numpy as np
from gym import spaces, spaces

from .environment import Environment
from .elements import Grass, Dirt
from .agent import Agent
from .constants import (
    get_name,
    WIDTH,
    HEIGHT,
    NUM_ACTIONS,
    USE_FULL_STATE,
    FITNESS_MEASURE,
)

'''
Reward:

A* distance from center of fire to some point?
SPE measure?

Implement A* anyway because of subgoals


'''

class ForestFire(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        # the environment is a 2D array of elements, plus an agent
        self.env = Environment(WIDTH, HEIGHT)
        # the action space consists of 6 discrete actions, including waiting
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # useful to have direct access to
        self.width = WIDTH
        self.height = HEIGHT
        self.get_name = get_name
        # the observation space consists of 14 continuous and discrete values
        # see features for more information
        # if we are using the full state, then the obs. space is of size:
        # (WIDTH, HEIGHT, 5)
        if USE_FULL_STATE:
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(WIDTH, HEIGHT, 5),
                                                dtype=np.bool)
        else:
            (max_ob, min_ob) = self.get_max_min_obs()
            self.observation_space = spaces.Box(low=min_ob,
                                                high=max_ob,
                                                dtype=np.float32)

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
        return [self.env.get_features(),
                self.env.get_fitness(FITNESS_MEASURE),
                not self.env.running, # NOT: to be consistent with conventions
                {}]

    # resets environment to default values
    def reset(self):
        self.env.reset_env()
        return self.env.get_features()

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
        print("")

    # prints information on windspeed and direction
    def wind_info(self):
        if (self.env.wind_vector[0] == 0 and self.env.wind_vector[1] == 0) \
                or self.env.wind_speed == 0:
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
