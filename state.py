import elements
import agent

class Environment:
	def __init__(self, width, height):
		self.running = True

		self.wind_speed = 3
		self.wind_vector = (1, 1)

		self.width = width
		self.height = height
		self.world = self.create_world(width, height)
		
		self.burning_cells = set()
		self.barriers = set()
		self.agents = set()
		
		self.agents.add(agent.Agent(5, 5, self))
		burning_cell = self.world[int(width/2)][int(height/2)]
		burning_cell.burning = True
		self.burning_cells.add(burning_cell)
		
	def create_world(self, width, height):
		world = list()
		for x in range(width):
			line = list()
			for y in range(height):
				line.append(elements.Grass(x, y))
			world.append(line)
		return world

	def get_at(self, x, y):
		return self.world[x][y]

	def set_at(self, x, y, cell):
		self.world[x][y] = cell

	def get_agent_coords(self):
		coords = set()
		for agent in self.agents:
			coords.add((agent.x, agent.y))
		return coords

	def inbounds(self, x, y):
		return x >= 0 and x < self.width and y >= 0 and y < self.height

	def traversable(self, x, y):
		cell = self.get_at(x, y)
		return cell.type in ["Grass", "Dirt"] and not cell.burning

	# Take the still running check out of the update() method,
	# make an isTerminal() method which deals with this situation
	def update(self):
		print("Updating Environment")
		if not self.agents or not self.burning_cells:
			self.running = False

		agents_to_remove = set()
		for agent in self.agents:
			status = agent.update()
			if status == "Dead":
				agents_to_remove.add(agent)
		self.agents.difference_update(agents_to_remove)

		cells_to_remove = set()
		cells_to_add = set()
		for cell in self.burning_cells:
			status = cell.time_step()
			if status == "Burnt Out":
				cells_to_remove.add(cell)
			if status == "No Change":
				neighbours = cell.get_neighbours(self)
				for n_cell in neighbours:
					if n_cell.burnable:
						status = n_cell.get_heat_from(cell, self)
						if status == "Ignited":
							cells_to_add.add(n_cell)
		self.burning_cells.difference_update(cells_to_remove)
		self.burning_cells.update(cells_to_add)

	def get_fitness(self):
		total_fuel = 0
		for x in range(self.world):
			for y in range(self.world[0]):
				total_fuel += self.get_at(x, y).fuel
		return total_fuel

	def get_features(self):
		features = list()

		for agent in self.agents:
			min_distance = 1000
			closest_fire = None

			for cell in self.burning_cells:
				distance = cell.distance_to(agent, "euclidean")
				if distance < min_distance:
					min_distance = distance
					closest_fire = cell

			(ax, ay) = agent.get_pos()
			features.add(distance)
			features.add(closest_fire.angle_to(agent))
			features.add(ax)
			features.add(ay)
			features.add(len(self.burning_cells))
