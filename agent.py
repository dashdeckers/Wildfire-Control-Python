from elements import Dirt


class Agent:
	def __init__(self, x, y, env):
		self.x = x
		self.y = y
		self.env = env

	def get_pos(self):
		return self.x, self.y

	def get_new_coords(self, direction):
		if direction == "N":
			return self.x, self.y - 1
		if direction == "S":
			return self.x, self.y + 1
		if direction == "E":
			return self.x + 1, self.y
		if direction == "W":
			return self.x - 1, self.y

	def update(self):
		if self.env.world[self.x][self.y].burning:
			return "Dead"

	def move(self, direction):
		(newX, newY) = self.get_new_coords(direction)
		if self.env.inbounds(newX, newY) and self.env.traversable(newX, newY):
			(self.x, self.y) = (newX, newY)

	def dig(self):
		self.env.set_at(self.x, self.y, Dirt(self.x, self.y))

