import math

class Element:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def get_pos(self):
		return (self.x, self.y)

	def get_color(self):
		if self.fuel <= 0:
			return "Black"
		if self.burning:
			return "Red"
		if self.temp > 0.75 * self.threshold:
			return "Orange"
		if self.temp > 0.5 * self.threshold:
			return "Yellow"
		return self.color

	def time_step(self):
		if self.burning:
			self.fuel -= 1
			if self.fuel == 0:
				self.burning = False
				self.burnable = False
				return "Burnt Out"
		return "No Change"

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

	def distance_to(self, cell, metric):
		if metric == "manhattan":
			return abs(self.x - cell.x) + abs(self.y - cell.y)
		if metric == "euclidean":
			return math.sqrt(math.pow(self.x - cell.x, 2) 
							+ math.pow(self.y - cell.y, 2))

	def wind_angle_to(self, cell, env):
		(cx, cy) = (self.x - cell.x, self.y - cell.y)
		(wx, wy) = env.wind_vector
		return abs(math.atan2(wx*cy - wy*cx, wx*cx + wy*cy))

	def angle_to(self, cell, env):
		(cx, cy) = (self.x - cell.x, self.y - cell.y)
		(rx, ry) = (0, 1)
		return abs(math.atan2(rx*cy - ry*cx, rx*cx + ry*cy))

	def get_neighbours(self, env):
		neighbours = set()
		for x in range(self.r + 1):
			for y in range(self.r + 1 - x):
				if (x == 0 and y == 0):
					continue
				if env.inbounds(self.x + x, self.y + y):
					cell = env.get_at(self.x + x, self.y + y)
					if cell.burnable:
						neighbours.add(cell)
				if env.inbounds(self.x - x, self.y + y):
					cell = env.get_at(self.x - x, self.y + y)
					if cell.burnable:
						neighbours.add(cell)
				if env.inbounds(self.x + x, self.y - y):
					cell = env.get_at(self.x + x, self.y - y)
					if cell.burnable:
						neighbours.add(cell)
				if env.inbounds(self.x - x, self.y - y):
					cell = env.get_at(self.x - x, self.y - y)
					if cell.burnable:
						neighbours.add(cell)
		return neighbours


class Grass(Element):
	def __init__(self, x, y):
		super().__init__(x, y)
		self.r = 2
		self.color = "Light Green"
		self.type = "Grass"
		
		self.burnable = True
		self.burning = False
		self.temp = 0
		self.heat = 3
		self.threshold = 5
		self.fuel = 5

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
		self.threshold = 10
		self.fuel = 5
