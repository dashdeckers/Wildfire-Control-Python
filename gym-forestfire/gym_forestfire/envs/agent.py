from .constants import VERBOSE

class Agent:
    def __init__(self, x, y, env):
        self.x = x
        self.y = y
        self.env = env

    # define hashvalue for set compatibility
    def __hash__(self):
        return hash((self.x, self.y))

    # define equality for set compatibility
    def __eq__(self, other):
        try:
            return (self.x, self.y) == (other.x, other.y)
        except AttributeError as e:
            return NotImplemented

    # get the position of the agent
    def get_pos(self):
        return (self.x, self.y)

    # translates direction to new coordinates
    def get_new_coords(self, direction):
        if direction in ["N", 0]:
            return (self.x, self.y - 1)
        if direction in ["S", 1]:
            return (self.x, self.y + 1)
        if direction in ["E", 2]:
            return (self.x + 1, self.y)
        if direction in ["W", 3]:
            return (self.x - 1, self.y)

    # if the agent is on a burning cell, the agent dies
    def update(self):
        if self.env.world[self.x][self.y].burning:
            if VERBOSE:
                print("Agent has died")
            return "Dead"

    # moves in direction if the cell at the new coordinates is 
    # traversable, inbounds, and not occupied by another agent
    def move(self, direction):
        (newX, newY) = self.get_new_coords(direction)
        if (newX, newY) not in [a.get_pos() for a in self.env.agents]:
            if self.env.inbounds(newX, newY) and self.env.traversable(newX, newY):
                (self.x, self.y) = (newX, newY)

    # convert cell under the agent to dirt
    def dig(self):
        self.env.set_at(self.x, self.y, Dirt(self.x, self.y))
